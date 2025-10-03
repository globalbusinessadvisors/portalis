"""
Performance Benchmark Tests

Comprehensive performance testing for all NVIDIA stack components:
- Translation latency and throughput
- GPU utilization and memory efficiency
- Concurrent request handling
- System resource usage
- Performance regression detection
"""

import pytest
import time
import asyncio
import psutil
import numpy as np
from typing import List, Dict
import torch

@pytest.mark.benchmark
@pytest.mark.performance
class TestTranslationPerformance:
    """Benchmark translation performance."""

    @pytest.mark.asyncio
    async def test_translation_latency_distribution(
        self, nim_api_client, python_code_generator, benchmark_config
    ):
        """Measure translation latency distribution."""
        iterations = 100
        test_codes = python_code_generator(complexity="simple", count=iterations)

        latencies = []

        for code in test_codes:
            request = {"python_code": code, "mode": "fast"}

            start = time.perf_counter()
            try:
                response = await nim_api_client.post(
                    "/api/v1/translation/translate",
                    json=request
                )
                if response.status_code == 200:
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed)
            except:
                pass

        if len(latencies) > 10:
            metrics = {
                "count": len(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p90": np.percentile(latencies, 90),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "stddev": np.std(latencies),
            }

            print("\n=== Translation Latency Distribution ===")
            for key, value in metrics.items():
                print(f"{key:10s}: {value:8.2f}ms")

            # Performance assertions
            assert metrics["p95"] < 2000, "P95 latency exceeds 2s"
            assert metrics["p99"] < 5000, "P99 latency exceeds 5s"

    @pytest.mark.asyncio
    async def test_throughput_scaling(
        self, nim_api_client, python_code_generator
    ):
        """Test throughput with increasing concurrency."""
        test_code = python_code_generator(complexity="simple", count=1)[0]
        concurrency_levels = [1, 5, 10, 20]

        results = []

        for concurrency in concurrency_levels:
            async def send_request():
                request = {"python_code": test_code, "mode": "fast"}
                return await nim_api_client.post(
                    "/api/v1/translation/translate",
                    json=request
                )

            start = time.time()
            responses = await asyncio.gather(
                *[send_request() for _ in range(concurrency)],
                return_exceptions=True
            )
            elapsed = time.time() - start

            successful = sum(
                1 for r in responses
                if not isinstance(r, Exception) and r.status_code == 200
            )

            throughput = successful / elapsed

            results.append({
                "concurrency": concurrency,
                "successful": successful,
                "elapsed": elapsed,
                "throughput": throughput,
            })

        print("\n=== Throughput Scaling ===")
        for r in results:
            print(f"Concurrency {r['concurrency']:3d}: "
                  f"{r['throughput']:6.2f} req/s "
                  f"({r['successful']}/{r['concurrency']} success)")


@pytest.mark.benchmark
@pytest.mark.gpu
class TestGPUPerformance:
    """Benchmark GPU performance."""

    def test_gpu_utilization(self, cuda_context, benchmark_config):
        """Test GPU utilization during translation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Allocate GPU tensors
        tensors = []
        for _ in range(10):
            tensor = torch.randn(1000, 1000, device=cuda_context)
            tensors.append(tensor)

        # Perform operations
        start = time.perf_counter()

        for _ in range(100):
            result = torch.matmul(tensors[0], tensors[1])

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Check memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB

        print("\n=== GPU Utilization ===")
        print(f"Time: {elapsed:.3f}s")
        print(f"Memory allocated: {memory_allocated:.2f} MB")
        print(f"Memory reserved: {memory_reserved:.2f} MB")

        # Cleanup
        del tensors
        torch.cuda.empty_cache()

    def test_memory_efficiency(self, cuda_context):
        """Test GPU memory efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.cuda.reset_peak_memory_stats()

        # Allocate and deallocate multiple times
        for iteration in range(5):
            tensors = [
                torch.randn(100, 100, device=cuda_context)
                for _ in range(100)
            ]

            # Use tensors
            result = sum(torch.sum(t) for t in tensors)

            # Free
            del tensors
            torch.cuda.empty_cache()

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        print(f"\n=== Memory Efficiency ===")
        print(f"Peak memory: {peak_memory:.2f} MB")

        # Should not accumulate memory
        assert peak_memory < 500, "Excessive memory usage"


@pytest.mark.benchmark
class TestSystemResourceUsage:
    """Test system resource usage."""

    def test_cpu_usage(self, python_code_generator):
        """Monitor CPU usage during operations."""
        process = psutil.Process()

        # Baseline
        cpu_before = process.cpu_percent(interval=1.0)

        # Perform work
        test_codes = python_code_generator(complexity="complex", count=100)

        # Monitor during work
        cpu_during = process.cpu_percent(interval=1.0)

        # After work
        time.sleep(2)
        cpu_after = process.cpu_percent(interval=1.0)

        print("\n=== CPU Usage ===")
        print(f"Before: {cpu_before:.1f}%")
        print(f"During: {cpu_during:.1f}%")
        print(f"After: {cpu_after:.1f}%")

    def test_memory_usage(self, python_code_generator):
        """Monitor memory usage."""
        process = psutil.Process()

        # Baseline
        mem_before = process.memory_info().rss / 1024**2  # MB

        # Generate load
        data = []
        for _ in range(100):
            codes = python_code_generator(complexity="medium", count=10)
            data.extend(codes)

        # Check memory
        mem_during = process.memory_info().rss / 1024**2  # MB

        # Cleanup
        del data

        # After cleanup
        time.sleep(1)
        mem_after = process.memory_info().rss / 1024**2  # MB

        print("\n=== Memory Usage ===")
        print(f"Before: {mem_before:.2f} MB")
        print(f"During: {mem_during:.2f} MB")
        print(f"After: {mem_after:.2f} MB")

        # Memory should not grow excessively
        growth = mem_after - mem_before
        assert growth < 500, f"Memory growth too high: {growth:.2f} MB"


@pytest.mark.benchmark
@pytest.mark.slow
class TestPerformanceRegression:
    """Detect performance regressions."""

    BASELINE_METRICS = {
        "translation_p95_ms": 2000,
        "throughput_req_per_sec": 5.0,
        "gpu_memory_mb": 1000,
    }

    @pytest.mark.asyncio
    async def test_no_latency_regression(self, nim_api_client):
        """Ensure latency hasn't regressed."""
        latencies = []

        for _ in range(50):
            request = {"python_code": "def test(): return 42", "mode": "fast"}

            start = time.perf_counter()
            try:
                response = await nim_api_client.post(
                    "/api/v1/translation/translate",
                    json=request
                )
                if response.status_code == 200:
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed)
            except:
                pass

        if len(latencies) > 10:
            p95 = np.percentile(latencies, 95)

            print(f"\n=== Regression Check: Latency ===")
            print(f"Current P95: {p95:.2f}ms")
            print(f"Baseline P95: {self.BASELINE_METRICS['translation_p95_ms']:.2f}ms")

            # Allow 20% regression
            threshold = self.BASELINE_METRICS['translation_p95_ms'] * 1.2
            assert p95 < threshold, f"Latency regression detected: {p95:.2f}ms > {threshold:.2f}ms"


@pytest.mark.benchmark
class TestConcurrentLoad:
    """Test performance under concurrent load."""

    @pytest.mark.asyncio
    async def test_high_concurrency(self, nim_api_client, python_code_generator):
        """Test with high concurrent load."""
        num_concurrent = 50
        test_codes = python_code_generator(complexity="simple", count=num_concurrent)

        async def send_request(code: str):
            request = {"python_code": code, "mode": "fast"}
            start = time.perf_counter()
            try:
                response = await nim_api_client.post(
                    "/api/v1/translation/translate",
                    json=request,
                    timeout=30.0
                )
                elapsed = time.perf_counter() - start
                return response.status_code == 200, elapsed
            except:
                return False, 0

        start = time.time()
        results = await asyncio.gather(*[send_request(code) for code in test_codes])
        total_time = time.time() - start

        successful = sum(1 for success, _ in results if success)
        latencies = [elapsed * 1000 for success, elapsed in results if success]

        print(f"\n=== High Concurrency Test ===")
        print(f"Concurrent requests: {num_concurrent}")
        print(f"Successful: {successful}")
        print(f"Success rate: {successful/num_concurrent*100:.1f}%")
        print(f"Total time: {total_time:.2f}s")

        if len(latencies) > 0:
            print(f"Avg latency: {np.mean(latencies):.2f}ms")
            print(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")

        # Should handle high concurrency
        assert successful >= num_concurrent * 0.8, "Too many failures under load"
