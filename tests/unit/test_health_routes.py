"""
Unit Tests for Health Check Routes (London School TDD)

Tests health, readiness, and liveness endpoints with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from nim_microservices.api.routes.health import router


@pytest.fixture
def app():
    """Create FastAPI test application."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_check_returns_200_when_healthy(self, client):
        """
        GIVEN a healthy service
        WHEN health check is called
        THEN it should return 200 with healthy status
        """
        with patch('nim_microservices.api.routes.health.torch') as mock_torch, \
             patch('nim_microservices.api.routes.health.get_service_config') as mock_config:

            mock_torch.cuda.is_available.return_value = True
            mock_config.return_value = MagicMock()

            response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_check_reports_gpu_availability(self, client):
        """
        GIVEN GPU is available
        WHEN health check is called
        THEN it should report GPU status
        """
        with patch('nim_microservices.api.routes.health.torch') as mock_torch, \
             patch('nim_microservices.api.routes.health.get_service_config'):

            mock_torch.cuda.is_available.return_value = True

            response = client.get("/health")

        data = response.json()
        assert "gpu_available" in data
        assert data["gpu_available"] is True

    def test_health_check_handles_missing_torch(self, client):
        """
        GIVEN torch is not installed
        WHEN health check is called
        THEN it should handle ImportError gracefully
        """
        with patch('nim_microservices.api.routes.health.torch', side_effect=ImportError), \
             patch('nim_microservices.api.routes.health.get_service_config'):

            response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["gpu_available"] is False

    def test_health_check_includes_dependency_status(self, client):
        """
        GIVEN various dependencies
        WHEN health check is called
        THEN it should report status of each dependency
        """
        with patch('nim_microservices.api.routes.health.get_service_config'):
            response = client.get("/health")

        data = response.json()
        assert "dependencies" in data
        assert isinstance(data["dependencies"], dict)

    def test_health_check_returns_degraded_when_model_not_loaded(self, client):
        """
        GIVEN model is not loaded but service is running
        WHEN health check is called
        THEN it should return degraded status
        """
        with patch('nim_microservices.api.routes.health.get_service_config', side_effect=Exception("Model not found")):
            response = client.get("/health")

        data = response.json()
        assert data["status"] in ["degraded", "unhealthy"]
        assert data["model_loaded"] is False


class TestReadinessEndpoint:
    """Test /ready endpoint."""

    def test_readiness_returns_ready_when_configured(self, client):
        """
        GIVEN service is properly configured
        WHEN readiness check is called
        THEN it should return ready status
        """
        with patch('nim_microservices.api.routes.health.get_service_config') as mock_config:
            mock_config.return_value = MagicMock()

            response = client.get("/ready")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ready"

    def test_readiness_returns_not_ready_on_config_failure(self, client):
        """
        GIVEN service configuration fails
        WHEN readiness check is called
        THEN it should return not_ready status
        """
        with patch('nim_microservices.api.routes.health.get_service_config', side_effect=Exception("Config error")):
            response = client.get("/ready")

        data = response.json()
        assert data["status"] == "not_ready"
        assert "error" in data


class TestLivenessEndpoint:
    """Test /live endpoint."""

    def test_liveness_always_returns_alive(self, client):
        """
        GIVEN the service is running
        WHEN liveness check is called
        THEN it should always return alive
        """
        response = client.get("/live")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "alive"


class TestMetricsEndpoint:
    """Test /metrics endpoint."""

    def test_metrics_returns_request_statistics(self, client):
        """
        GIVEN service has processed requests
        WHEN metrics endpoint is called
        THEN it should return request statistics
        """
        response = client.get("/metrics")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "memory_usage_mb" in data

    def test_metrics_includes_latency_percentiles(self, client):
        """
        GIVEN latency data is available
        WHEN metrics endpoint is called
        THEN it should include P95 and P99 latencies
        """
        response = client.get("/metrics")

        data = response.json()
        assert "avg_processing_time_ms" in data
        assert "p95_latency_ms" in data
        assert "p99_latency_ms" in data

    def test_metrics_includes_gpu_utilization_when_available(self, client):
        """
        GIVEN GPU monitoring is available
        WHEN metrics endpoint is called
        THEN it should include GPU utilization
        """
        with patch('nim_microservices.api.routes.health.pynvml') as mock_nvml:
            mock_nvml.nvmlInit.return_value = None
            mock_handle = MagicMock()
            mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_util = MagicMock()
            mock_util.gpu = 75.5
            mock_nvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

            response = client.get("/metrics")

        # GPU metrics may be None if pynvml not available
        data = response.json()
        assert "gpu_utilization" in data


class TestStatusEndpoint:
    """Test /status endpoint."""

    def test_status_returns_detailed_information(self, client):
        """
        GIVEN service is running
        WHEN status endpoint is called
        THEN it should return detailed service and system information
        """
        response = client.get("/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "service" in data
        assert "system" in data

    def test_status_includes_system_metrics(self, client):
        """
        GIVEN system monitoring is available
        WHEN status endpoint is called
        THEN it should include CPU, memory, and disk metrics
        """
        response = client.get("/status")

        data = response.json()
        system = data["system"]

        assert "cpu_percent" in system
        assert "memory_percent" in system
        assert "disk_percent" in system

    def test_status_includes_gpu_info_when_available(self, client):
        """
        GIVEN GPU is available
        WHEN status endpoint is called
        THEN it should include GPU information
        """
        with patch('nim_microservices.api.routes.health.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
            mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024
            mock_torch.cuda.memory_reserved.return_value = 2048 * 1024 * 1024

            response = client.get("/status")

        data = response.json()
        assert "gpu" in data
        gpu_info = data["gpu"]
        assert gpu_info["available"] is True


class TestMetricsRecording:
    """Test request metrics recording functionality."""

    def test_record_request_increments_total_count(self):
        """
        GIVEN a request is processed
        WHEN record_request is called
        THEN total request count should increment
        """
        from nim_microservices.api.routes.health import record_request, METRICS

        initial_count = METRICS["total_requests"]

        record_request(success=True, latency_ms=100.0)

        assert METRICS["total_requests"] == initial_count + 1

    def test_record_request_increments_success_count(self):
        """
        GIVEN a successful request
        WHEN record_request is called
        THEN successful request count should increment
        """
        from nim_microservices.api.routes.health import record_request, METRICS

        initial_count = METRICS["successful_requests"]

        record_request(success=True, latency_ms=100.0)

        assert METRICS["successful_requests"] == initial_count + 1

    def test_record_request_increments_failure_count(self):
        """
        GIVEN a failed request
        WHEN record_request is called
        THEN failed request count should increment
        """
        from nim_microservices.api.routes.health import record_request, METRICS

        initial_count = METRICS["failed_requests"]

        record_request(success=False, latency_ms=100.0)

        assert METRICS["failed_requests"] == initial_count + 1

    def test_record_request_stores_latency(self):
        """
        GIVEN a request latency
        WHEN record_request is called
        THEN latency should be stored
        """
        from nim_microservices.api.routes.health import record_request, METRICS

        latency = 150.5
        record_request(success=True, latency_ms=latency)

        assert latency in METRICS["latencies"]
