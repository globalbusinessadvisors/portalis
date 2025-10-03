"""
Acceptance Tests for Translation Workflows (London School TDD)

Outside-in acceptance tests that verify the complete translation workflow.
These tests describe the system from the user's perspective.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status


@pytest.fixture
def mock_complete_stack():
    """Mock the complete NVIDIA stack for acceptance testing."""
    stack = {
        'nemo_service': MagicMock(),
        'triton_client': MagicMock(),
        'cuda_available': True,
    }

    # Configure NeMo service mock
    nemo_result = MagicMock()
    nemo_result.rust_code = """
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}
"""
    nemo_result.confidence = 0.92
    nemo_result.alternatives = []
    nemo_result.metadata = {'model': 'nemo-coder'}
    nemo_result.processing_time_ms = 245.5

    stack['nemo_service'].translate_code.return_value = nemo_result

    # Configure Triton client mock
    triton_result = MagicMock()
    triton_result.rust_code = nemo_result.rust_code
    triton_result.confidence = 0.95
    triton_result.metadata = {'backend': 'triton', 'model_version': '1'}
    triton_result.warnings = []
    triton_result.suggestions = ['Consider using iterative approach for better performance']

    stack['triton_client'].translate_code.return_value = triton_result

    return stack


@pytest.fixture
def app_with_mocks(mock_complete_stack):
    """Create application with mocked services."""
    from nim_microservices.api.main import create_app

    with patch('nim_microservices.api.routes.translation.get_nemo_service',
               return_value=mock_complete_stack['nemo_service']), \
         patch('nim_microservices.api.routes.translation.get_triton_client',
               return_value=mock_complete_stack['triton_client']):
        app = create_app()
        yield app


@pytest.fixture
def client(app_with_mocks):
    """Create test client with mocked services."""
    return TestClient(app_with_mocks)


class TestSimpleFunctionTranslation:
    """
    Feature: Simple function translation
    As a developer
    I want to translate simple Python functions to Rust
    So that I can use them in performance-critical applications
    """

    def test_user_translates_simple_function_successfully(self, client):
        """
        Scenario: User translates a simple function
        Given I have a simple Python function
        When I submit it for translation
        Then I should receive valid Rust code
        And the confidence score should be high
        """
        # Given: A simple Python function
        python_code = """
def add(a: int, b: int) -> int:
    return a + b
"""

        # When: I submit it for translation
        response = client.post(
            "/api/v1/translation/translate",
            json={
                "python_code": python_code,
                "mode": "fast"
            }
        )

        # Then: I should receive valid Rust code
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "rust_code" in data
        assert len(data["rust_code"]) > 0

        # And: The confidence score should be high
        assert data["confidence"] >= 0.7

    def test_user_receives_processing_time_information(self, client):
        """
        Scenario: User wants to know processing time
        Given I have translated some code
        When I check the response
        Then I should see processing time information
        """
        # Given: I have translated some code
        response = client.post(
            "/api/v1/translation/translate",
            json={
                "python_code": "def test(): return 42",
                "mode": "fast"
            }
        )

        # Then: I should see processing time information
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0


class TestRecursiveFunctionTranslation:
    """
    Feature: Recursive function translation
    As a developer
    I want to translate recursive Python functions
    So that the recursion semantics are preserved in Rust
    """

    def test_user_translates_fibonacci_function(self, client):
        """
        Scenario: User translates Fibonacci function
        Given I have a recursive Fibonacci function
        When I submit it for translation
        Then I should receive Rust code with recursion preserved
        """
        # Given: A recursive Fibonacci function
        python_code = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

        # When: I submit it for translation
        response = client.post(
            "/api/v1/translation/translate",
            json={
                "python_code": python_code,
                "mode": "quality"
            }
        )

        # Then: I should receive Rust code with recursion preserved
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        rust_code = data["rust_code"]

        # Rust code should contain function definition
        assert "fn" in rust_code


class TestQualityModeTranslation:
    """
    Feature: High-quality translation mode
    As a developer
    I want to use quality mode for important code
    So that I get the best possible translation with optimizations
    """

    def test_user_requests_quality_mode_translation(self, client):
        """
        Scenario: User requests high-quality translation
        Given I have code that needs careful translation
        When I request quality mode
        Then I should receive optimized Rust code with suggestions
        """
        # Given: Code that needs careful translation
        python_code = """
def calculate_sum(numbers: list) -> int:
    total = 0
    for num in numbers:
        total += num
    return total
"""

        # When: I request quality mode
        response = client.post(
            "/api/v1/translation/translate",
            json={
                "python_code": python_code,
                "mode": "quality"
            }
        )

        # Then: I should receive optimized Rust code
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["confidence"] > 0.5

        # And: Potentially receive suggestions
        assert "suggestions" in data


class TestBatchTranslation:
    """
    Feature: Batch file translation
    As a developer
    I want to translate multiple Python files at once
    So that I can convert entire modules efficiently
    """

    def test_user_translates_multiple_files_in_batch(self, client):
        """
        Scenario: User translates multiple related files
        Given I have multiple Python source files
        When I submit them as a batch
        Then all files should be translated successfully
        """
        with patch('nim_microservices.api.routes.translation.get_triton_client') as mock_triton:
            # Setup batch result
            batch_result = MagicMock()
            batch_result.translated_files = [
                "// File 1 Rust code",
                "// File 2 Rust code",
                "// File 3 Rust code"
            ]
            batch_result.compilation_status = ["success", "success", "success"]
            batch_result.performance_metrics = {"total_time": 500}
            batch_result.wasm_binaries = None

            mock_triton.return_value.translate_batch.return_value = batch_result

            # Given: Multiple Python source files
            files = [
                "def func1(): return 1",
                "def func2(): return 2",
                "def func3(): return 3"
            ]

            # When: I submit them as a batch
            response = client.post(
                "/api/v1/translation/translate/batch",
                json={
                    "source_files": files,
                    "project_config": {"name": "my_project"},
                    "optimization_level": "release",
                    "compile_wasm": False
                }
            )

            # Then: All files should be translated
            assert response.status_code == status.HTTP_200_OK

            data = response.json()
            assert len(data["translated_files"]) == 3
            assert data["success_count"] == 3


class TestErrorHandlingWorkflow:
    """
    Feature: Graceful error handling
    As a developer
    I want clear error messages when translation fails
    So that I can understand and fix the problems
    """

    def test_user_submits_invalid_python_code(self, client):
        """
        Scenario: User submits invalid Python code
        Given I have syntactically invalid Python code
        When I submit it for translation
        Then I should receive a clear error message
        """
        # Given: Invalid Python code
        invalid_code = ""  # Empty code

        # When: I submit it for translation
        response = client.post(
            "/api/v1/translation/translate",
            json={
                "python_code": invalid_code,
                "mode": "fast"
            }
        )

        # Then: I should receive an error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_user_handles_service_unavailable_gracefully(self, client):
        """
        Scenario: Translation service is temporarily unavailable
        Given the translation service encounters an error
        When I try to translate code
        Then I should receive a helpful error message
        """
        with patch('nim_microservices.api.routes.translation.get_nemo_service') as mock_service:
            # Given: Service encounters an error
            mock_service.return_value.translate_code.side_effect = \
                RuntimeError("GPU memory exhausted")

            # When: I try to translate
            response = client.post(
                "/api/v1/translation/translate",
                json={
                    "python_code": "def test(): pass",
                    "mode": "fast"
                }
            )

            # Then: I should receive a helpful error
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "detail" in data


class TestHealthCheckWorkflow:
    """
    Feature: Service health monitoring
    As an operator
    I want to monitor service health
    So that I can ensure the system is running properly
    """

    def test_operator_checks_service_health(self, client):
        """
        Scenario: Operator checks if service is healthy
        Given the service is running
        When I check the health endpoint
        Then I should see the current health status
        """
        # When: I check the health endpoint
        response = client.get("/health")

        # Then: I should see health status
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_operator_checks_service_readiness(self, client):
        """
        Scenario: Operator checks if service is ready
        Given the service has started
        When I check the readiness endpoint
        Then I should know if it can accept requests
        """
        # When: I check the readiness endpoint
        response = client.get("/ready")

        # Then: I should know readiness status
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "status" in data


class TestPerformanceMonitoring:
    """
    Feature: Performance metrics monitoring
    As an operator
    I want to see performance metrics
    So that I can optimize the service
    """

    def test_operator_views_service_metrics(self, client):
        """
        Scenario: Operator views service metrics
        Given the service has processed requests
        When I check the metrics endpoint
        Then I should see request statistics and latencies
        """
        # When: I check the metrics endpoint
        response = client.get("/metrics")

        # Then: I should see request statistics
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "p95_latency_ms" in data


class TestContextualTranslation:
    """
    Feature: Translation with context
    As a developer
    I want to provide additional context for translation
    So that the generated Rust code is more accurate
    """

    def test_user_provides_type_hints_as_context(self, client):
        """
        Scenario: User provides type hints
        Given I have Python code with inferred types
        When I provide explicit type hints in context
        Then the Rust code should use appropriate types
        """
        # Given: Python code with inferred types
        python_code = "def multiply(x, y): return x * y"

        # When: I provide type hints in context
        response = client.post(
            "/api/v1/translation/translate",
            json={
                "python_code": python_code,
                "mode": "standard",
                "context": {
                    "type_hints": {
                        "x": "float",
                        "y": "float",
                        "return": "float"
                    }
                }
            }
        )

        # Then: Translation should complete successfully
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "rust_code" in data
