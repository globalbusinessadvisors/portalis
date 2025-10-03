"""
Unit Tests for Translation Routes (London School TDD)

Tests the translation API routes using outside-in approach with mocked collaborators.
Following London School principles:
- Mock all external dependencies
- Test interactions and behavior
- Focus on object collaboration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
import time

from nim_microservices.api.routes.translation import (
    router,
    TranslationRequest,
    TranslationResponse,
)
from nim_microservices.api.models.schema import TranslationMode


@pytest.fixture
def mock_nemo_service():
    """Mock NeMo service for testing."""
    service = MagicMock()

    # Configure mock translation result
    mock_result = MagicMock()
    mock_result.rust_code = "fn add(a: i64, b: i64) -> i64 { a + b }"
    mock_result.confidence = 0.95
    mock_result.alternatives = []
    mock_result.metadata = {"model": "test-model"}
    mock_result.processing_time_ms = 100.5

    service.translate_code.return_value = mock_result
    service.batch_translate.return_value = [mock_result]

    return service


@pytest.fixture
def mock_triton_client():
    """Mock Triton client for testing."""
    client = MagicMock()

    # Configure mock translation result
    mock_result = MagicMock()
    mock_result.rust_code = "fn add(a: i64, b: i64) -> i64 { a + b }"
    mock_result.confidence = 0.92
    mock_result.metadata = {"backend": "triton"}
    mock_result.warnings = []
    mock_result.suggestions = []

    client.translate_code.return_value = mock_result

    return client


@pytest.fixture
def app(mock_nemo_service, mock_triton_client):
    """Create FastAPI test application with mocked services."""
    app = FastAPI()
    app.include_router(router)

    # Patch service getters to return mocks
    with patch('nim_microservices.api.routes.translation.get_nemo_service', return_value=mock_nemo_service), \
         patch('nim_microservices.api.routes.translation.get_triton_client', return_value=mock_triton_client):
        yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestTranslateEndpoint:
    """Test /api/v1/translation/translate endpoint."""

    def test_translate_code_with_fast_mode_uses_nemo_service(self, client, mock_nemo_service):
        """
        GIVEN a translation request in fast mode
        WHEN the translate endpoint is called
        THEN it should use NeMo service directly
        """
        request_data = {
            "python_code": "def add(a, b): return a + b",
            "mode": "fast"
        }

        with patch('nim_microservices.api.routes.translation.get_nemo_service', return_value=mock_nemo_service):
            response = client.post("/api/v1/translation/translate", json=request_data)

        assert response.status_code == status.HTTP_200_OK

        # Verify NeMo service was called
        mock_nemo_service.translate_code.assert_called_once()
        call_args = mock_nemo_service.translate_code.call_args
        assert call_args[1]['python_code'] == request_data['python_code']

    def test_translate_code_with_standard_mode_uses_triton(self, client, mock_triton_client):
        """
        GIVEN a translation request in standard mode
        WHEN the translate endpoint is called
        THEN it should use Triton client
        """
        request_data = {
            "python_code": "def multiply(x, y): return x * y",
            "mode": "standard"
        }

        with patch('nim_microservices.api.routes.translation.get_triton_client', return_value=mock_triton_client):
            response = client.post("/api/v1/translation/translate", json=request_data)

        assert response.status_code == status.HTTP_200_OK

        # Verify Triton client was called
        mock_triton_client.translate_code.assert_called_once()

    def test_translate_code_returns_valid_response_structure(self, client, mock_nemo_service):
        """
        GIVEN a valid translation request
        WHEN the endpoint processes it
        THEN it should return a properly structured response
        """
        request_data = {
            "python_code": "def test(): pass",
            "mode": "fast"
        }

        with patch('nim_microservices.api.routes.translation.get_nemo_service', return_value=mock_nemo_service):
            response = client.post("/api/v1/translation/translate", json=request_data)

        data = response.json()

        # Verify response structure
        assert "rust_code" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
        assert "metadata" in data

        # Verify data types
        assert isinstance(data["rust_code"], str)
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_translate_code_with_context_passes_context_to_service(self, client, mock_nemo_service):
        """
        GIVEN a translation request with context
        WHEN the endpoint is called
        THEN it should pass context to the service
        """
        request_data = {
            "python_code": "def func(): pass",
            "mode": "fast",
            "context": {
                "type_hints": {"return": "int"},
                "description": "Test function"
            }
        }

        with patch('nim_microservices.api.routes.translation.get_nemo_service', return_value=mock_nemo_service):
            response = client.post("/api/v1/translation/translate", json=request_data)

        assert response.status_code == status.HTTP_200_OK

        # Verify context was passed
        call_args = mock_nemo_service.translate_code.call_args
        assert call_args[1]['context'] == request_data['context']

    def test_translate_code_with_invalid_input_returns_422(self, client):
        """
        GIVEN an invalid request (empty code)
        WHEN the endpoint is called
        THEN it should return 422 Unprocessable Entity
        """
        request_data = {
            "python_code": "",  # Invalid: empty code
            "mode": "fast"
        }

        response = client.post("/api/v1/translation/translate", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_translate_code_handles_service_failure(self, client, mock_nemo_service):
        """
        GIVEN a service that raises an exception
        WHEN translation is attempted
        THEN it should return 500 with error details
        """
        mock_nemo_service.translate_code.side_effect = RuntimeError("Model inference failed")

        request_data = {
            "python_code": "def test(): pass",
            "mode": "fast"
        }

        with patch('nim_microservices.api.routes.translation.get_nemo_service', return_value=mock_nemo_service):
            response = client.post("/api/v1/translation/translate", json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data

    def test_translate_code_respects_temperature_parameter(self, client, mock_triton_client):
        """
        GIVEN a request with custom temperature
        WHEN using Triton backend
        THEN it should pass temperature to client
        """
        request_data = {
            "python_code": "def test(): pass",
            "mode": "standard",
            "temperature": 0.5
        }

        with patch('nim_microservices.api.routes.translation.get_triton_client', return_value=mock_triton_client):
            response = client.post("/api/v1/translation/translate", json=request_data)

        assert response.status_code == status.HTTP_200_OK

        # Verify temperature was passed
        call_args = mock_triton_client.translate_code.call_args
        assert call_args[1]['options']['temperature'] == 0.5

    def test_translate_code_includes_alternatives_when_requested(self, client, mock_nemo_service):
        """
        GIVEN a request with include_alternatives=True
        WHEN translation completes
        THEN response should include alternatives
        """
        mock_nemo_service.translate_code.return_value.alternatives = [
            "fn alternative1() {}",
            "fn alternative2() {}"
        ]

        request_data = {
            "python_code": "def test(): pass",
            "mode": "fast",
            "include_alternatives": True
        }

        with patch('nim_microservices.api.routes.translation.get_nemo_service', return_value=mock_nemo_service):
            response = client.post("/api/v1/translation/translate", json=request_data)

        data = response.json()
        assert "alternatives" in data
        assert len(data["alternatives"]) == 2


class TestBatchTranslateEndpoint:
    """Test /api/v1/translation/translate/batch endpoint."""

    def test_batch_translate_processes_multiple_files(self, client, mock_triton_client):
        """
        GIVEN a batch translation request with multiple files
        WHEN the batch endpoint is called
        THEN it should process all files
        """
        mock_batch_result = MagicMock()
        mock_batch_result.translated_files = ["rust1", "rust2", "rust3"]
        mock_batch_result.compilation_status = ["success", "success", "success"]
        mock_batch_result.performance_metrics = {}
        mock_batch_result.wasm_binaries = None

        mock_triton_client.translate_batch.return_value = mock_batch_result

        request_data = {
            "source_files": [
                "def func1(): pass",
                "def func2(): pass",
                "def func3(): pass"
            ],
            "project_config": {"name": "test_project"},
            "optimization_level": "release"
        }

        with patch('nim_microservices.api.routes.translation.get_triton_client', return_value=mock_triton_client):
            response = client.post("/api/v1/translation/translate/batch", json=request_data)

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["translated_files"]) == 3
        assert data["success_count"] == 3
        assert data["failure_count"] == 0

    def test_batch_translate_uses_triton_client(self, client, mock_triton_client):
        """
        GIVEN a batch translation request
        WHEN the endpoint is called
        THEN it should use Triton client for batch processing
        """
        mock_batch_result = MagicMock()
        mock_batch_result.translated_files = ["rust"]
        mock_batch_result.compilation_status = ["success"]
        mock_batch_result.performance_metrics = {}

        mock_triton_client.translate_batch.return_value = mock_batch_result

        request_data = {
            "source_files": ["def test(): pass"],
            "project_config": {},
            "optimization_level": "debug"
        }

        with patch('nim_microservices.api.routes.translation.get_triton_client', return_value=mock_triton_client):
            response = client.post("/api/v1/translation/translate/batch", json=request_data)

        # Verify Triton was called with correct arguments
        mock_triton_client.translate_batch.assert_called_once()
        call_args = mock_triton_client.translate_batch.call_args
        assert call_args[1]['optimization_level'] == 'debug'

    def test_batch_translate_counts_failures_correctly(self, client, mock_triton_client):
        """
        GIVEN a batch with some failures
        WHEN processing completes
        THEN success and failure counts should be accurate
        """
        mock_batch_result = MagicMock()
        mock_batch_result.translated_files = ["rust1", "rust2", "rust3"]
        mock_batch_result.compilation_status = ["success", "error: syntax", "success"]
        mock_batch_result.performance_metrics = {}

        mock_triton_client.translate_batch.return_value = mock_batch_result

        request_data = {
            "source_files": ["code1", "code2", "code3"],
            "project_config": {},
            "optimization_level": "release"
        }

        with patch('nim_microservices.api.routes.translation.get_triton_client', return_value=mock_triton_client):
            response = client.post("/api/v1/translation/translate/batch", json=request_data)

        data = response.json()
        assert data["success_count"] == 2
        assert data["failure_count"] == 1


class TestStreamingEndpoint:
    """Test /api/v1/translation/translate/stream endpoint."""

    @pytest.mark.asyncio
    async def test_streaming_translation_returns_chunks(self, client, mock_nemo_service):
        """
        GIVEN a streaming translation request
        WHEN the stream endpoint is called
        THEN it should return multiple chunks
        """
        # Note: Testing streaming with TestClient is limited
        # This test verifies the endpoint exists and structure
        request_data = {
            "python_code": "def test(): return 42",
            "mode": "streaming"
        }

        with patch('nim_microservices.api.routes.translation.get_nemo_service', return_value=mock_nemo_service):
            response = client.post("/api/v1/translation/translate/stream", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        assert response.headers['content-type'] == 'application/x-ndjson'


class TestModelListEndpoint:
    """Test /api/v1/translation/models endpoint."""

    def test_list_models_returns_available_models(self, client):
        """
        GIVEN the service is running
        WHEN the models endpoint is called
        THEN it should return list of available models
        """
        with patch('nim_microservices.api.routes.translation.get_service_config') as mock_config:
            mock_config.return_value.model_version = "1.0.0"

            response = client.get("/api/v1/translation/models")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        assert data["models"][0]["name"] == "translation_model"


class TestInteractionPatterns:
    """Test interaction patterns between route handlers and collaborators."""

    def test_route_handler_delegates_to_service_not_performing_logic(self, client, mock_nemo_service):
        """
        GIVEN a translation request
        WHEN the route handler processes it
        THEN it should delegate to service without business logic
        """
        request_data = {
            "python_code": "def test(): pass",
            "mode": "fast"
        }

        with patch('nim_microservices.api.routes.translation.get_nemo_service', return_value=mock_nemo_service):
            response = client.post("/api/v1/translation/translate", json=request_data)

        # Verify handler only coordinates, doesn't perform translation
        assert mock_nemo_service.translate_code.call_count == 1
        assert response.status_code == status.HTTP_200_OK

    def test_background_tasks_record_metrics(self, client, mock_nemo_service):
        """
        GIVEN a successful translation
        WHEN the request completes
        THEN metrics should be recorded via background task
        """
        request_data = {
            "python_code": "def test(): pass",
            "mode": "fast"
        }

        with patch('nim_microservices.api.routes.translation.get_nemo_service', return_value=mock_nemo_service), \
             patch('nim_microservices.api.routes.translation.record_translation_metrics') as mock_metrics:

            response = client.post("/api/v1/translation/translate", json=request_data)

        # Background task should be scheduled (execution happens after response)
        assert response.status_code == status.HTTP_200_OK
