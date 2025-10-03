"""
Integration Tests: NeMo + CUDA Pipeline

Tests the integration between NeMo translation and CUDA acceleration:
- Python → NeMo translation → CUDA kernel acceleration
- End-to-end translation with GPU optimization
- Performance validation
"""

import pytest
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any

# Import NeMo components
try:
    from translation.nemo_service import NeMoService, InferenceConfig, TranslationResult
except ImportError:
    pytest.skip("NeMo integration not available", allow_module_level=True)

# Import CUDA components
try:
    import portalis_cuda
except ImportError:
    portalis_cuda = None


@pytest.mark.integration
@pytest.mark.nemo
class TestNeMoCUDAIntegration:
    """Test NeMo and CUDA integration."""

    @pytest.fixture
    def nemo_service(self, nemo_inference_config, temp_dir, cuda_available):
        """Create NeMo service with CUDA if available."""
        model_path = temp_dir / "mock_model.nemo"
        model_path.touch()

        config = InferenceConfig(**nemo_inference_config)
        service = NeMoService(
            model_path=model_path,
            config=config,
            enable_cuda=cuda_available
        )
        service.initialize()

        yield service

        service.cleanup()

    def test_nemo_cuda_device_compatibility(self, nemo_service, cuda_available):
        """Test that NeMo service uses correct device."""
        if cuda_available:
            assert nemo_service.device.type == "cuda"
        else:
            assert nemo_service.device.type == "cpu"

    def test_translate_and_accelerate(self, nemo_service, sample_python_code):
        """Test translation followed by CUDA acceleration."""
        # Step 1: Translate with NeMo
        result = nemo_service.translate_code(sample_python_code)

        assert isinstance(result, TranslationResult)
        assert len(result.rust_code) > 0
        assert result.confidence > 0.0
        assert result.processing_time_ms > 0

        # Step 2: Verify GPU acceleration was considered
        if torch.cuda.is_available() and portalis_cuda:
            # Check if CUDA kernels could be applied
            assert "metadata" in result.__dict__ or hasattr(result, "metadata")

    @pytest.mark.cuda
    def test_batch_translation_gpu_acceleration(self, nemo_service, sample_batch_files):
        """Test batch translation with GPU acceleration."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        start_time = time.time()
        results = nemo_service.batch_translate(sample_batch_files)
        elapsed = time.time() - start_time

        assert len(results) == len(sample_batch_files)
        assert all(isinstance(r, TranslationResult) for r in results)
        assert all(r.confidence > 0.0 for r in results)

        # Batch should be faster than sequential (assuming batching optimization)
        avg_time_per_file = elapsed / len(sample_batch_files)
        assert avg_time_per_file < 5.0, "Batch processing too slow"

    @pytest.mark.cuda
    def test_embedding_generation_cuda(self, nemo_service, sample_batch_files):
        """Test embedding generation on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        embeddings = nemo_service.generate_embeddings(sample_batch_files)

        assert embeddings.shape[0] == len(sample_batch_files)
        assert embeddings.shape[1] > 0  # Embedding dimension

        # Embeddings should be on GPU if available
        if isinstance(embeddings, torch.Tensor):
            if torch.cuda.is_available():
                assert embeddings.device.type == "cuda"

    def test_translation_quality_with_cuda(self, nemo_service):
        """Test that CUDA doesn't degrade translation quality."""
        test_cases = [
            ("def add(a, b): return a + b", "simple function"),
            ("class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y", "class"),
            ("x = [i**2 for i in range(10)]", "list comprehension"),
        ]

        for python_code, description in test_cases:
            result = nemo_service.translate_code(python_code)

            assert result.confidence > 0.3, f"Low confidence for {description}"
            assert len(result.rust_code) > 0, f"Empty translation for {description}"
            assert "fn" in result.rust_code or "struct" in result.rust_code, \
                f"No Rust syntax in {description}"

    @pytest.mark.benchmark
    def test_nemo_cuda_performance(self, nemo_service, benchmark_config):
        """Benchmark NeMo translation performance with CUDA."""
        test_code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"

        latencies = []

        # Warmup
        for _ in range(benchmark_config["warmup_iterations"]):
            nemo_service.translate_code(test_code)

        # Benchmark
        for _ in range(benchmark_config["test_iterations"]):
            start = time.perf_counter()
            result = nemo_service.translate_code(test_code)
            elapsed = (time.perf_counter() - start) * 1000  # ms

            latencies.append(elapsed)
            assert result.confidence > 0.0

        # Compute statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"\n=== NeMo CUDA Performance ===")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"P50 latency: {p50_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")
        print(f"P99 latency: {p99_latency:.2f}ms")

        # Performance assertions
        assert avg_latency < 1000, "Average latency too high (>1s)"
        assert p95_latency < 2000, "P95 latency too high (>2s)"


@pytest.mark.integration
@pytest.mark.cuda
class TestCUDAKernelOptimization:
    """Test CUDA kernel optimization for translated code."""

    @pytest.mark.cuda
    def test_cuda_kernel_availability(self):
        """Test that CUDA kernels are available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if portalis_cuda is None:
            pytest.skip("Portalis CUDA module not available")

        assert hasattr(portalis_cuda, 'accelerate_translation')
        assert hasattr(portalis_cuda, 'optimize_memory')

    @pytest.mark.cuda
    def test_matrix_operation_acceleration(self, sample_cuda_tensors):
        """Test matrix operation acceleration."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        input_tensor = sample_cuda_tensors["input"]
        weights = sample_cuda_tensors["weights"]

        # CPU baseline
        cpu_input = input_tensor.cpu()
        cpu_weights = weights.cpu()

        start_cpu = time.perf_counter()
        cpu_result = torch.matmul(cpu_input, cpu_weights)
        cpu_time = time.perf_counter() - start_cpu

        # GPU accelerated
        start_gpu = time.perf_counter()
        gpu_result = torch.matmul(input_tensor, weights)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_gpu

        # Verify correctness
        assert torch.allclose(cpu_result, gpu_result.cpu(), rtol=1e-5, atol=1e-5)

        # GPU should be faster for large matrices
        if input_tensor.shape[0] >= 100:
            print(f"CPU time: {cpu_time*1000:.2f}ms, GPU time: {gpu_time*1000:.2f}ms")
            print(f"Speedup: {cpu_time/gpu_time:.2f}x")

    @pytest.mark.cuda
    def test_memory_optimization(self, cuda_context):
        """Test CUDA memory optimization."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Check initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        # Allocate tensors
        tensors = [torch.randn(1000, 1000, device=cuda_context) for _ in range(10)]

        peak_memory = torch.cuda.max_memory_allocated()

        # Cleanup
        del tensors
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        # Memory should be freed
        assert final_memory <= initial_memory + 1024 * 1024, "Memory leak detected"

        print(f"Peak memory usage: {peak_memory / 1024**2:.2f} MB")

    @pytest.mark.slow
    @pytest.mark.cuda
    def test_large_batch_cuda_acceleration(self, cuda_context):
        """Test CUDA acceleration with large batches."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 100
        seq_length = 512
        hidden_dim = 768

        # Create large batch
        batch = torch.randn(batch_size, seq_length, hidden_dim, device=cuda_context)
        weights = torch.randn(hidden_dim, hidden_dim, device=cuda_context)

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = torch.matmul(batch, weights)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        assert result.shape == (batch_size, seq_length, hidden_dim)
        print(f"Large batch processing time: {elapsed*1000:.2f}ms")

        # Should complete in reasonable time
        assert elapsed < 1.0, "Large batch processing too slow"


@pytest.mark.integration
class TestNeMoCUDAErrorHandling:
    """Test error handling in NeMo + CUDA integration."""

    def test_cuda_fallback_on_oom(self, nemo_service):
        """Test fallback to CPU on CUDA OOM."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Try to translate very large code that might cause OOM
        large_code = "\n".join([f"def func_{i}(): return {i}" for i in range(1000)])

        # Should handle gracefully
        result = nemo_service.translate_code(large_code)
        assert result is not None
        assert result.confidence >= 0.0

    def test_invalid_cuda_context(self, nemo_inference_config, temp_dir):
        """Test handling of invalid CUDA context."""
        model_path = temp_dir / "mock_model.nemo"
        model_path.touch()

        # Force CUDA even if not available
        config = InferenceConfig(**nemo_inference_config)

        # Should handle gracefully or raise appropriate error
        try:
            service = NeMoService(
                model_path=model_path,
                config=config,
                enable_cuda=True  # Force CUDA
            )
            service.initialize()

            if not torch.cuda.is_available():
                # Should fall back to CPU
                assert service.device.type == "cpu"
            else:
                assert service.device.type == "cuda"

            service.cleanup()
        except RuntimeError as e:
            # Expected if CUDA initialization fails
            assert "CUDA" in str(e) or "cuda" in str(e)

    def test_concurrent_cuda_operations(self, nemo_service, sample_batch_files):
        """Test concurrent CUDA operations don't interfere."""
        import concurrent.futures

        def translate_code(code: str):
            return nemo_service.translate_code(code)

        # Run concurrent translations
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(translate_code, code)
                for code in sample_batch_files
            ]

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == len(sample_batch_files)
        assert all(r.confidence > 0.0 for r in results)


@pytest.mark.integration
@pytest.mark.benchmark
class TestNeMoCUDABenchmarks:
    """Comprehensive benchmarks for NeMo + CUDA integration."""

    def test_throughput_benchmark(self, nemo_service, python_code_generator, benchmark_config):
        """Benchmark translation throughput."""
        num_samples = 50
        test_codes = python_code_generator(complexity="simple", count=num_samples)

        # Warmup
        for code in test_codes[:5]:
            nemo_service.translate_code(code)

        # Benchmark
        start = time.time()
        results = [nemo_service.translate_code(code) for code in test_codes]
        elapsed = time.time() - start

        throughput = num_samples / elapsed

        print(f"\n=== Throughput Benchmark ===")
        print(f"Total samples: {num_samples}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {throughput:.2f} translations/sec")

        assert throughput > 1.0, "Throughput too low (<1 translation/sec)"
        assert all(r.confidence > 0.0 for r in results)

    def test_latency_distribution(self, nemo_service, python_code_generator, benchmark_config):
        """Test latency distribution across different code complexities."""
        complexities = ["simple", "medium", "complex"]
        results = {}

        for complexity in complexities:
            test_codes = python_code_generator(complexity=complexity, count=20)
            latencies = []

            for code in test_codes:
                start = time.perf_counter()
                result = nemo_service.translate_code(code)
                elapsed = (time.perf_counter() - start) * 1000

                latencies.append(elapsed)
                assert result.confidence > 0.0

            results[complexity] = {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
            }

        print(f"\n=== Latency Distribution by Complexity ===")
        for complexity, stats in results.items():
            print(f"{complexity.capitalize()}:")
            print(f"  Mean: {stats['mean']:.2f}ms")
            print(f"  Median: {stats['median']:.2f}ms")
            print(f"  P95: {stats['p95']:.2f}ms")
            print(f"  P99: {stats['p99']:.2f}ms")

        # Complex code should take longer but not excessively
        assert results["complex"]["mean"] > results["simple"]["mean"]
        assert results["complex"]["p95"] < 5000, "Complex code P95 too high"
