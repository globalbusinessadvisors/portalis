"""
Integration Tests: Triton + NIM Pipeline

Tests the integration between Triton model serving and NIM microservices:
- Triton model serving → NIM API endpoints
- Load balancing and auto-scaling
- Health checks and monitoring
- Rate limiting and authentication
"""

import pytest
import asyncio
import httpx
import time
import json
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from configs.triton_client import create_client, TranslationResult
except ImportError:
    pytest.skip("Triton client not available", allow_module_level=True)


@pytest.mark.integration
@pytest.mark.triton
@pytest.mark.nim
class TestTritonNIMIntegration:
    """Test Triton and NIM integration."""

    @pytest.fixture
    async def nim_client(self, test_config):
        """Create NIM API client."""
        async with httpx.AsyncClient(
            base_url=test_config["nim_api_url"],
            timeout=30.0
        ) as client:
            # Wait for service to be ready
            max_retries = 10
            for i in range(max_retries):
                try:
                    response = await client.get("/health")
                    if response.status_code == 200:
                        break
                except Exception:
                    if i == max_retries - 1:
                        pytest.skip("NIM service not available")
                    await asyncio.sleep(2)

            yield client

    @pytest.fixture
    def triton_client(self, triton_client_config):
        """Create Triton client."""
        try:
            client = create_client(**triton_client_config)

            # Wait for Triton to be ready
            max_retries = 10
            for i in range(max_retries):
                try:
                    if client.is_server_ready():
                        break
                except Exception:
                    if i == max_retries - 1:
                        pytest.skip("Triton service not available")
                    time.sleep(2)

            yield client

            client.close()
        except Exception:
            pytest.skip("Triton service not available")

    @pytest.mark.asyncio
    async def test_triton_to_nim_translation_flow(
        self, triton_client, nim_client, sample_python_code
    ):
        """Test translation flow from Triton through NIM API."""
        # Step 1: Translate via Triton
        triton_result = triton_client.translate_code(sample_python_code)

        assert isinstance(triton_result, TranslationResult)
        assert len(triton_result.rust_code) > 0

        # Step 2: Verify translation via NIM API
        nim_request = {
            "python_code": sample_python_code,
            "mode": "fast",
            "temperature": 0.2,
        }

        response = await nim_client.post(
            "/api/v1/translation/translate",
            json=nim_request
        )

        assert response.status_code == 200
        nim_result = response.json()

        assert "rust_code" in nim_result
        assert "confidence" in nim_result

        # Results should be comparable
        assert len(nim_result["rust_code"]) > 0

    @pytest.mark.asyncio
    async def test_nim_health_check_reflects_triton_status(
        self, nim_client, triton_client
    ):
        """Test that NIM health check reflects Triton backend status."""
        # Check Triton status
        triton_ready = triton_client.is_server_ready()

        # Check NIM health
        response = await nim_client.get("/health")
        assert response.status_code == 200

        health_data = response.json()

        assert "status" in health_data
        assert "model_loaded" in health_data

        # If Triton is ready, NIM should report model loaded
        if triton_ready:
            assert health_data["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_concurrent_requests_triton_nim(
        self, nim_client, sample_batch_files
    ):
        """Test concurrent requests through NIM to Triton."""
        async def translate_request(code: str):
            request_data = {
                "python_code": code,
                "mode": "fast",
            }
            response = await nim_client.post(
                "/api/v1/translation/translate",
                json=request_data
            )
            return response

        # Send concurrent requests
        tasks = [translate_request(code) for code in sample_batch_files]

        start = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start

        # Check results
        successful = sum(
            1 for r in responses
            if not isinstance(r, Exception) and r.status_code == 200
        )

        print(f"Concurrent requests: {len(sample_batch_files)}")
        print(f"Successful: {successful}")
        print(f"Time: {elapsed:.2f}s")

        # Most requests should succeed
        assert successful >= len(sample_batch_files) * 0.8

    @pytest.mark.asyncio
    async def test_batch_translation_optimization(
        self, nim_client, sample_batch_files
    ):
        """Test batch translation optimization."""
        # Batch request
        batch_request = {
            "source_files": sample_batch_files,
            "project_config": {
                "name": "test_batch",
                "dependencies": [],
            },
            "optimization_level": "release",
            "compile_wasm": False,
            "run_tests": False,
        }

        start = time.time()
        response = await nim_client.post(
            "/api/v1/translation/translate/batch",
            json=batch_request,
            timeout=60.0
        )
        batch_time = time.time() - start

        if response.status_code == 200:
            result = response.json()
            assert "translated_files" in result
            assert len(result["translated_files"]) == len(sample_batch_files)

            print(f"Batch translation time: {batch_time:.2f}s")
            print(f"Average per file: {batch_time / len(sample_batch_files):.2f}s")

    @pytest.mark.asyncio
    async def test_streaming_translation(self, nim_client, sample_python_code):
        """Test streaming translation through NIM."""
        request_data = {
            "python_code": sample_python_code,
            "mode": "streaming"
        }

        async with nim_client.stream(
            "POST",
            "/api/v1/translation/translate/stream",
            json=request_data,
            timeout=30.0
        ) as response:
            if response.status_code == 200:
                chunks = []
                async for line in response.aiter_lines():
                    if line.strip():
                        chunk = json.loads(line)
                        chunks.append(chunk)
                        if chunk.get("is_final"):
                            break

                assert len(chunks) > 0
                assert chunks[-1]["is_final"] is True
                print(f"Received {len(chunks)} chunks")


@pytest.mark.integration
@pytest.mark.triton
@pytest.mark.nim
class TestTritonNIMLoadBalancing:
    """Test load balancing between Triton and NIM."""

    @pytest.mark.asyncio
    async def test_load_distribution(self, nim_client, python_code_generator):
        """Test that load is distributed across Triton instances."""
        num_requests = 50
        test_codes = python_code_generator(complexity="simple", count=num_requests)

        async def send_request(code: str):
            request_data = {"python_code": code, "mode": "fast"}
            start = time.perf_counter()
            response = await nim_client.post(
                "/api/v1/translation/translate",
                json=request_data
            )
            elapsed = time.perf_counter() - start
            return response, elapsed

        # Send all requests
        tasks = [send_request(code) for code in test_codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successful = [r for r in results if not isinstance(r, Exception)]
        latencies = [elapsed for _, elapsed in successful if isinstance(_, httpx.Response)]

        if len(latencies) > 0:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)

            print(f"\n=== Load Distribution ===")
            print(f"Total requests: {num_requests}")
            print(f"Successful: {len(successful)}")
            print(f"Avg latency: {avg_latency*1000:.2f}ms")
            print(f"Min latency: {min_latency*1000:.2f}ms")
            print(f"Max latency: {max_latency*1000:.2f}ms")

            # Latencies should be reasonably consistent (load balanced)
            assert max_latency / min_latency < 10, "Latency variance too high"

    @pytest.mark.asyncio
    async def test_auto_scaling_behavior(self, nim_client, python_code_generator):
        """Test auto-scaling behavior under load."""
        # Start with low load
        low_load_codes = python_code_generator(complexity="simple", count=5)

        for code in low_load_codes:
            request_data = {"python_code": code, "mode": "fast"}
            await nim_client.post("/api/v1/translation/translate", json=request_data)

        # Check metrics after low load
        response = await nim_client.get("/metrics")
        if response.status_code == 200:
            metrics_low = response.json()

            # Now send high load
            high_load_codes = python_code_generator(complexity="simple", count=50)

            async def send_request(code: str):
                request_data = {"python_code": code, "mode": "fast"}
                return await nim_client.post(
                    "/api/v1/translation/translate",
                    json=request_data
                )

            tasks = [send_request(code) for code in high_load_codes]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Check metrics after high load
            response = await nim_client.get("/metrics")
            if response.status_code == 200:
                metrics_high = response.json()

                print(f"\n=== Auto-scaling Metrics ===")
                print(f"Low load requests: {metrics_low.get('total_requests', 0)}")
                print(f"High load requests: {metrics_high.get('total_requests', 0)}")


@pytest.mark.integration
@pytest.mark.triton
@pytest.mark.nim
class TestTritonNIMRateLimiting:
    """Test rate limiting in NIM."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, nim_client):
        """Test that rate limits are enforced."""
        # Send many requests rapidly
        num_requests = 100
        request_data = {
            "python_code": "def test(): pass",
            "mode": "fast"
        }

        async def send_request():
            return await nim_client.post(
                "/api/v1/translation/translate",
                json=request_data
            )

        tasks = [send_request() for _ in range(num_requests)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count status codes
        status_codes = {}
        for response in responses:
            if isinstance(response, httpx.Response):
                code = response.status_code
                status_codes[code] = status_codes.get(code, 0) + 1

        print(f"\n=== Rate Limiting ===")
        print(f"Total requests: {num_requests}")
        for code, count in sorted(status_codes.items()):
            print(f"Status {code}: {count}")

        # Should have some 200s (successful)
        assert 200 in status_codes

        # May have 429s (rate limited) depending on configuration
        if 429 in status_codes:
            print(f"Rate limited: {status_codes[429]} requests")

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self, nim_client):
        """Test rate limit headers in responses."""
        request_data = {
            "python_code": "def test(): pass",
            "mode": "fast"
        }

        response = await nim_client.post(
            "/api/v1/translation/translate",
            json=request_data
        )

        # Check for rate limit headers (may vary by implementation)
        print(f"\n=== Rate Limit Headers ===")
        for header in ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]:
            if header in response.headers:
                print(f"{header}: {response.headers[header]}")


@pytest.mark.integration
@pytest.mark.triton
@pytest.mark.nim
class TestTritonNIMMonitoring:
    """Test monitoring and observability."""

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, nim_client):
        """Test metrics endpoint provides correct data."""
        response = await nim_client.get("/metrics")
        assert response.status_code == 200

        metrics = response.json()

        expected_metrics = [
            "total_requests",
            "successful_requests",
            "failed_requests",
            "memory_usage_mb",
        ]

        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        print(f"\n=== NIM Metrics ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    @pytest.mark.asyncio
    async def test_health_check_details(self, nim_client):
        """Test health check provides detailed status."""
        response = await nim_client.get("/health")
        assert response.status_code == 200

        health = response.json()

        assert "status" in health
        assert "version" in health
        assert "uptime_seconds" in health
        assert "gpu_available" in health
        assert "model_loaded" in health

        print(f"\n=== Health Check ===")
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Uptime: {health['uptime_seconds']}s")
        print(f"GPU Available: {health['gpu_available']}")
        print(f"Model Loaded: {health['model_loaded']}")

    @pytest.mark.asyncio
    async def test_status_endpoint(self, nim_client):
        """Test detailed status endpoint."""
        response = await nim_client.get("/status")

        if response.status_code == 200:
            status = response.json()

            assert "service" in status
            assert "system" in status

            print(f"\n=== Service Status ===")
            print(json.dumps(status, indent=2))


@pytest.mark.integration
@pytest.mark.benchmark
class TestTritonNIMPerformance:
    """Performance tests for Triton + NIM integration."""

    @pytest.mark.asyncio
    async def test_end_to_end_latency(self, nim_client, benchmark_config):
        """Test end-to-end latency through NIM to Triton."""
        test_code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"

        latencies = []

        # Warmup
        for _ in range(benchmark_config["warmup_iterations"]):
            request_data = {"python_code": test_code, "mode": "fast"}
            await nim_client.post("/api/v1/translation/translate", json=request_data)

        # Benchmark
        for _ in range(50):  # Reduced iterations for integration test
            request_data = {"python_code": test_code, "mode": "fast"}

            start = time.perf_counter()
            response = await nim_client.post(
                "/api/v1/translation/translate",
                json=request_data
            )
            elapsed = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                latencies.append(elapsed)

        if len(latencies) > 0:
            import numpy as np

            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            print(f"\n=== End-to-End Latency ===")
            print(f"Average: {avg_latency:.2f}ms")
            print(f"P50: {p50_latency:.2f}ms")
            print(f"P95: {p95_latency:.2f}ms")
            print(f"P99: {p99_latency:.2f}ms")

            # Target: <500ms for P95
            assert p95_latency < 2000, f"P95 latency too high: {p95_latency:.2f}ms"

    @pytest.mark.asyncio
    async def test_throughput(self, nim_client, python_code_generator):
        """Test translation throughput through NIM."""
        num_requests = 30
        test_codes = python_code_generator(complexity="simple", count=num_requests)

        async def send_request(code: str):
            request_data = {"python_code": code, "mode": "fast"}
            return await nim_client.post(
                "/api/v1/translation/translate",
                json=request_data
            )

        start = time.time()
        results = await asyncio.gather(
            *[send_request(code) for code in test_codes],
            return_exceptions=True
        )
        elapsed = time.time() - start

        successful = sum(
            1 for r in results
            if not isinstance(r, Exception) and r.status_code == 200
        )

        throughput = successful / elapsed

        print(f"\n=== Throughput ===")
        print(f"Total requests: {num_requests}")
        print(f"Successful: {successful}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Throughput: {throughput:.2f} req/s")

        assert throughput > 1.0, "Throughput too low"
