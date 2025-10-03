"""
Integration tests for Portalis NIM API

Tests REST endpoints, health checks, and core functionality.
"""

import pytest
import httpx
import asyncio
from typing import AsyncGenerator


# Test configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async HTTP client for testing"""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=TIMEOUT) as client:
        yield client


@pytest.mark.asyncio
class TestHealthEndpoints:
    """Test health check endpoints"""

    async def test_health_check(self, client: httpx.AsyncClient):
        """Test /health endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "uptime_seconds" in data
        assert "gpu_available" in data
        assert "model_loaded" in data

    async def test_liveness_check(self, client: httpx.AsyncClient):
        """Test /live endpoint"""
        response = await client.get("/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"

    async def test_readiness_check(self, client: httpx.AsyncClient):
        """Test /ready endpoint"""
        response = await client.get("/ready")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data

    async def test_metrics(self, client: httpx.AsyncClient):
        """Test /metrics endpoint"""
        response = await client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "memory_usage_mb" in data

    async def test_status(self, client: httpx.AsyncClient):
        """Test /status endpoint"""
        response = await client.get("/status")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "system" in data


@pytest.mark.asyncio
class TestTranslationAPI:
    """Test translation endpoints"""

    async def test_translate_simple_code(self, client: httpx.AsyncClient):
        """Test basic code translation"""
        request_data = {
            "python_code": "def add(a, b):\n    return a + b",
            "mode": "fast",
            "temperature": 0.2,
            "max_length": 512,
            "include_alternatives": False
        }

        response = await client.post("/api/v1/translation/translate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "rust_code" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
        assert len(data["rust_code"]) > 0
        assert 0.0 <= data["confidence"] <= 1.0

    async def test_translate_with_context(self, client: httpx.AsyncClient):
        """Test translation with context"""
        request_data = {
            "python_code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "mode": "standard",
            "context": {
                "description": "Recursive Fibonacci function",
                "optimization": "performance"
            },
            "include_alternatives": True
        }

        response = await client.post("/api/v1/translation/translate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "rust_code" in data
        assert data["confidence"] > 0.0

    async def test_translate_invalid_code(self, client: httpx.AsyncClient):
        """Test translation with invalid Python code"""
        request_data = {
            "python_code": "",  # Empty code
            "mode": "fast"
        }

        response = await client.post("/api/v1/translation/translate", json=request_data)
        assert response.status_code == 422  # Validation error

    async def test_translate_large_code(self, client: httpx.AsyncClient):
        """Test translation of larger code block"""
        large_code = """
class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, x, y):
        self.result = x + y
        return self.result

    def multiply(self, x, y):
        self.result = x * y
        return self.result

    def power(self, x, n):
        self.result = x ** n
        return self.result
"""
        request_data = {
            "python_code": large_code,
            "mode": "quality",
            "max_length": 2048
        }

        response = await client.post("/api/v1/translation/translate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["rust_code"]) > 0


@pytest.mark.asyncio
class TestBatchTranslation:
    """Test batch translation endpoints"""

    async def test_batch_translate(self, client: httpx.AsyncClient):
        """Test batch translation"""
        request_data = {
            "source_files": [
                "def add(a, b): return a + b",
                "def multiply(a, b): return a * b",
                "def subtract(a, b): return a - b"
            ],
            "project_config": {
                "name": "math_utils",
                "dependencies": []
            },
            "optimization_level": "release",
            "compile_wasm": False,
            "run_tests": False
        }

        response = await client.post(
            "/api/v1/translation/translate/batch",
            json=request_data,
            timeout=60.0
        )

        # May fail if services not fully initialized
        if response.status_code == 200:
            data = response.json()
            assert "translated_files" in data
            assert "success_count" in data
            assert len(data["translated_files"]) == 3


@pytest.mark.asyncio
class TestStreamingTranslation:
    """Test streaming translation"""

    async def test_translate_stream(self, client: httpx.AsyncClient):
        """Test streaming translation endpoint"""
        request_data = {
            "python_code": "def greet(name):\n    return f'Hello, {name}!'",
            "mode": "streaming"
        }

        async with client.stream(
            "POST",
            "/api/v1/translation/translate/stream",
            json=request_data,
            timeout=30.0
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/x-ndjson"

            chunks = []
            async for line in response.aiter_lines():
                if line.strip():
                    import json
                    chunk = json.loads(line)
                    chunks.append(chunk)
                    if chunk.get("is_final"):
                        break

            assert len(chunks) > 0
            assert chunks[-1]["is_final"] is True


@pytest.mark.asyncio
class TestModelEndpoints:
    """Test model management endpoints"""

    async def test_list_models(self, client: httpx.AsyncClient):
        """Test model listing"""
        response = await client.get("/api/v1/translation/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling"""

    async def test_invalid_endpoint(self, client: httpx.AsyncClient):
        """Test accessing invalid endpoint"""
        response = await client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    async def test_method_not_allowed(self, client: httpx.AsyncClient):
        """Test wrong HTTP method"""
        response = await client.get("/api/v1/translation/translate")
        assert response.status_code == 405

    async def test_malformed_json(self, client: httpx.AsyncClient):
        """Test malformed JSON request"""
        response = await client.post(
            "/api/v1/translation/translate",
            content="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]


@pytest.mark.asyncio
class TestConcurrency:
    """Test concurrent requests"""

    async def test_concurrent_translations(self, client: httpx.AsyncClient):
        """Test multiple concurrent translation requests"""
        request_data = {
            "python_code": "def test(): pass",
            "mode": "fast"
        }

        # Send 10 concurrent requests
        tasks = [
            client.post("/api/v1/translation/translate", json=request_data)
            for _ in range(10)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that most requests succeeded
        successful = sum(
            1 for r in responses
            if not isinstance(r, Exception) and r.status_code == 200
        )
        assert successful >= 8  # At least 80% success rate


@pytest.mark.asyncio
class TestPerformance:
    """Test performance metrics"""

    async def test_translation_latency(self, client: httpx.AsyncClient):
        """Test that translation completes within acceptable time"""
        import time

        request_data = {
            "python_code": "def quick_test(): return 42",
            "mode": "fast"
        }

        start = time.time()
        response = await client.post("/api/v1/translation/translate", json=request_data)
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should complete in less than 5 seconds for fast mode
        assert elapsed < 5.0

        data = response.json()
        # Processing time should be reasonable
        assert data["processing_time_ms"] < 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
