"""
Health Check and System Status Routes

Provides health, readiness, and liveness endpoints for Kubernetes.
"""

import time
import psutil
from fastapi import APIRouter, status
import logging
from typing import Dict, Any

from ..models import HealthCheckResponse, MetricsResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Service start time
SERVICE_START_TIME = time.time()

# Metrics storage (in production, use Prometheus)
METRICS = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "latencies": []
}


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check service health status",
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        HealthCheckResponse with service status
    """
    try:
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

        # Check model status
        model_loaded = False
        try:
            from ...config.service_config import get_service_config
            config = get_service_config()
            model_loaded = True
        except Exception:
            pass

        # Check dependencies
        dependencies = {}

        # Check NeMo
        try:
            import nemo
            dependencies["nemo"] = "available"
        except ImportError:
            dependencies["nemo"] = "unavailable"

        # Check Triton
        try:
            import tritonclient
            dependencies["triton"] = "available"
        except ImportError:
            dependencies["triton"] = "unavailable"

        # Check CUDA
        if gpu_available:
            dependencies["cuda"] = "available"
        else:
            dependencies["cuda"] = "unavailable"

        # Determine overall status
        if model_loaded and (gpu_available or True):  # Allow CPU mode
            service_status = "healthy"
        elif model_loaded:
            service_status = "degraded"
        else:
            service_status = "unhealthy"

        uptime = time.time() - SERVICE_START_TIME

        return HealthCheckResponse(
            status=service_status,
            version="1.0.0",
            uptime_seconds=uptime,
            gpu_available=gpu_available,
            model_loaded=model_loaded,
            dependencies=dependencies
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version="1.0.0",
            uptime_seconds=time.time() - SERVICE_START_TIME,
            gpu_available=False,
            model_loaded=False,
            dependencies={}
        )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Check if service is ready to accept requests",
)
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check for Kubernetes.

    Returns:
        Ready status
    """
    try:
        # Check if model is loaded
        from ...config.service_config import get_service_config
        config = get_service_config()

        return {"status": "ready"}

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not_ready", "error": str(e)}


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Check if service is alive",
)
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check for Kubernetes.

    Returns:
        Alive status
    """
    return {"status": "alive"}


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Service metrics",
    description="Get service performance metrics",
)
async def get_metrics() -> MetricsResponse:
    """
    Get service metrics.

    Returns:
        MetricsResponse with performance data
    """
    try:
        # Get GPU metrics if available
        gpu_utilization = None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = float(utilization.gpu)
        except Exception:
            pass

        # Get memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Calculate latency percentiles
        latencies = METRICS["latencies"][-1000:]  # Last 1000 requests
        if latencies:
            latencies_sorted = sorted(latencies)
            p95_idx = int(len(latencies_sorted) * 0.95)
            p99_idx = int(len(latencies_sorted) * 0.99)
            p95_latency = latencies_sorted[p95_idx] if p95_idx < len(latencies_sorted) else 0
            p99_latency = latencies_sorted[p99_idx] if p99_idx < len(latencies_sorted) else 0
            avg_latency = sum(latencies) / len(latencies)
        else:
            p95_latency = 0
            p99_latency = 0
            avg_latency = 0

        return MetricsResponse(
            total_requests=METRICS["total_requests"],
            successful_requests=METRICS["successful_requests"],
            failed_requests=METRICS["failed_requests"],
            avg_processing_time_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            gpu_utilization=gpu_utilization,
            memory_usage_mb=memory_mb
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise


@router.get(
    "/status",
    summary="Detailed status",
    description="Get detailed service status information",
)
async def get_status() -> Dict[str, Any]:
    """
    Get detailed service status.

    Returns:
        Detailed status information
    """
    try:
        # System info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # GPU info
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated": torch.cuda.memory_allocated(0),
                    "memory_reserved": torch.cuda.memory_reserved(0),
                }
        except Exception:
            gpu_info = {"available": False}

        return {
            "service": {
                "name": "portalis-nim",
                "version": "1.0.0",
                "uptime_seconds": time.time() - SERVICE_START_TIME,
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / 1024 / 1024,
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            },
            "gpu": gpu_info,
            "metrics": {
                "total_requests": METRICS["total_requests"],
                "success_rate": (
                    METRICS["successful_requests"] / METRICS["total_requests"]
                    if METRICS["total_requests"] > 0 else 0
                ),
            }
        }

    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise


def record_request(success: bool, latency_ms: float) -> None:
    """
    Record request metrics.

    Args:
        success: Whether request succeeded
        latency_ms: Request latency in milliseconds
    """
    METRICS["total_requests"] += 1
    if success:
        METRICS["successful_requests"] += 1
    else:
        METRICS["failed_requests"] += 1
    METRICS["latencies"].append(latency_ms)

    # Keep only last 10000 latencies
    if len(METRICS["latencies"]) > 10000:
        METRICS["latencies"] = METRICS["latencies"][-10000:]
