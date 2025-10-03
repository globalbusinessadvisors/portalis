"""
Performance Benchmarking Suite for Portalis WASM in Omniverse

Measures and validates performance metrics:
- Frame rate (>30 FPS target)
- Execution latency (<10ms target)
- Memory footprint (<100MB per module target)
- Throughput (operations per second)
"""

import time
import psutil
import statistics
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    name: str
    description: str

    # Timing metrics
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    std_dev_ms: float

    # Throughput metrics
    operations_count: int
    operations_per_second: float

    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float

    # Quality metrics
    success_rate: float
    error_count: int

    # Target metrics
    meets_fps_target: bool  # >30 FPS
    meets_latency_target: bool  # <10ms
    meets_memory_target: bool  # <100MB

    # Raw data
    execution_times_ms: List[float] = field(default_factory=list)
    memory_samples_mb: List[float] = field(default_factory=list)

    @property
    def fps_equivalent(self) -> float:
        """Calculate equivalent FPS if run continuously"""
        if self.avg_time_ms > 0:
            return 1000.0 / self.avg_time_ms
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'timing': {
                'total_time_ms': self.total_time_ms,
                'avg_time_ms': self.avg_time_ms,
                'min_time_ms': self.min_time_ms,
                'max_time_ms': self.max_time_ms,
                'median_time_ms': self.median_time_ms,
                'std_dev_ms': self.std_dev_ms,
            },
            'throughput': {
                'operations_count': self.operations_count,
                'operations_per_second': self.operations_per_second,
                'fps_equivalent': self.fps_equivalent,
            },
            'memory': {
                'peak_memory_mb': self.peak_memory_mb,
                'avg_memory_mb': self.avg_memory_mb,
            },
            'quality': {
                'success_rate': self.success_rate,
                'error_count': self.error_count,
            },
            'targets': {
                'meets_fps_target': self.meets_fps_target,
                'meets_latency_target': self.meets_latency_target,
                'meets_memory_target': self.meets_memory_target,
                'all_targets_met': (
                    self.meets_fps_target and
                    self.meets_latency_target and
                    self.meets_memory_target
                ),
            }
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    warmup_iterations: int = 10
    benchmark_iterations: int = 1000
    fps_target: float = 30.0
    latency_target_ms: float = 10.0
    memory_target_mb: float = 100.0
    collect_memory_samples: bool = True
    memory_sample_interval: int = 100  # Sample every N iterations


class PerformanceBenchmark:
    """
    Performance benchmarking system for WASM modules

    Provides comprehensive performance analysis:
    - Timing measurements
    - Memory profiling
    - Throughput analysis
    - Target validation
    """

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process()

    def benchmark_function(
        self,
        name: str,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        description: str = ""
    ) -> BenchmarkResult:
        """
        Benchmark a function call

        Args:
            name: Benchmark name
            function: Function to benchmark
            args: Function arguments
            kwargs: Function keyword arguments
            description: Benchmark description

        Returns:
            BenchmarkResult with all metrics
        """
        kwargs = kwargs or {}

        # Warmup
        for _ in range(self.config.warmup_iterations):
            try:
                function(*args, **kwargs)
            except Exception:
                pass

        # Benchmark
        execution_times = []
        memory_samples = []
        error_count = 0

        start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        peak_memory = start_memory

        start_time = time.perf_counter()

        for i in range(self.config.benchmark_iterations):
            iter_start = time.perf_counter()

            try:
                function(*args, **kwargs)
                success = True
            except Exception:
                success = False
                error_count += 1

            iter_end = time.perf_counter()
            execution_times.append((iter_end - iter_start) * 1000)  # ms

            # Sample memory
            if self.config.collect_memory_samples and i % self.config.memory_sample_interval == 0:
                current_memory = self.process.memory_info().rss / (1024 * 1024)
                memory_samples.append(current_memory)
                peak_memory = max(peak_memory, current_memory)

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # ms

        # Calculate statistics
        avg_time = statistics.mean(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        median_time = statistics.median(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0

        # Memory statistics
        avg_memory = statistics.mean(memory_samples) if memory_samples else start_memory
        peak_memory_mb = peak_memory - start_memory
        avg_memory_mb = avg_memory - start_memory

        # Throughput
        operations_per_second = (self.config.benchmark_iterations / total_time) * 1000

        # Quality
        success_rate = (self.config.benchmark_iterations - error_count) / self.config.benchmark_iterations

        # Target validation
        fps_equivalent = 1000.0 / avg_time if avg_time > 0 else 0.0
        meets_fps = fps_equivalent >= self.config.fps_target
        meets_latency = avg_time <= self.config.latency_target_ms
        meets_memory = peak_memory_mb <= self.config.memory_target_mb

        result = BenchmarkResult(
            name=name,
            description=description,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            median_time_ms=median_time,
            std_dev_ms=std_dev,
            operations_count=self.config.benchmark_iterations,
            operations_per_second=operations_per_second,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            success_rate=success_rate,
            error_count=error_count,
            meets_fps_target=meets_fps,
            meets_latency_target=meets_latency,
            meets_memory_target=meets_memory,
            execution_times_ms=execution_times,
            memory_samples_mb=memory_samples,
        )

        self.results.append(result)
        return result

    def benchmark_wasm_module(
        self,
        wasm_bridge,
        module_id: str,
        function_name: str,
        args: tuple = (),
        description: str = ""
    ) -> BenchmarkResult:
        """
        Benchmark WASM module function

        Args:
            wasm_bridge: WasmtimeBridge instance
            module_id: Module identifier
            function_name: Function to call
            args: Function arguments
            description: Benchmark description

        Returns:
            BenchmarkResult
        """
        def wasm_call():
            return wasm_bridge.call_function(module_id, function_name, *args)

        name = f"{module_id}::{function_name}"
        return self.benchmark_function(name, wasm_call, description=description)

    def print_result(self, result: BenchmarkResult):
        """Print formatted benchmark result"""
        print(f"\n{'='*70}")
        print(f"Benchmark: {result.name}")
        print(f"{'='*70}")

        if result.description:
            print(f"Description: {result.description}")
            print()

        print("TIMING METRICS:")
        print(f"  Average:      {result.avg_time_ms:.3f} ms")
        print(f"  Median:       {result.median_time_ms:.3f} ms")
        print(f"  Min:          {result.min_time_ms:.3f} ms")
        print(f"  Max:          {result.max_time_ms:.3f} ms")
        print(f"  Std Dev:      {result.std_dev_ms:.3f} ms")
        print(f"  Total:        {result.total_time_ms:.2f} ms")

        print("\nTHROUGHPUT:")
        print(f"  Operations:   {result.operations_count}")
        print(f"  Ops/sec:      {result.operations_per_second:.2f}")
        print(f"  FPS equiv:    {result.fps_equivalent:.1f} FPS")

        print("\nMEMORY:")
        print(f"  Peak:         {result.peak_memory_mb:.2f} MB")
        print(f"  Average:      {result.avg_memory_mb:.2f} MB")

        print("\nQUALITY:")
        print(f"  Success rate: {result.success_rate*100:.1f}%")
        print(f"  Errors:       {result.error_count}")

        print("\nTARGET VALIDATION:")
        fps_status = "✓" if result.meets_fps_target else "✗"
        latency_status = "✓" if result.meets_latency_target else "✗"
        memory_status = "✓" if result.meets_memory_target else "✗"

        print(f"  {fps_status} FPS Target (>30 FPS): {result.fps_equivalent:.1f} FPS")
        print(f"  {latency_status} Latency Target (<10ms): {result.avg_time_ms:.3f} ms")
        print(f"  {memory_status} Memory Target (<100MB): {result.peak_memory_mb:.2f} MB")

        if result.meets_fps_target and result.meets_latency_target and result.meets_memory_target:
            print("\n  ✓ ALL TARGETS MET")
        else:
            print("\n  ✗ Some targets not met")

    def print_summary(self):
        """Print summary of all benchmarks"""
        if not self.results:
            print("No benchmark results available")
            return

        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Total benchmarks: {len(self.results)}")
        print()

        # Count targets met
        fps_met = sum(1 for r in self.results if r.meets_fps_target)
        latency_met = sum(1 for r in self.results if r.meets_latency_target)
        memory_met = sum(1 for r in self.results if r.meets_memory_target)
        all_met = sum(
            1 for r in self.results
            if r.meets_fps_target and r.meets_latency_target and r.meets_memory_target
        )

        print(f"FPS Target Met:      {fps_met}/{len(self.results)} ({fps_met/len(self.results)*100:.1f}%)")
        print(f"Latency Target Met:  {latency_met}/{len(self.results)} ({latency_met/len(self.results)*100:.1f}%)")
        print(f"Memory Target Met:   {memory_met}/{len(self.results)} ({memory_met/len(self.results)*100:.1f}%)")
        print(f"All Targets Met:     {all_met}/{len(self.results)} ({all_met/len(self.results)*100:.1f}%)")

        print("\nINDIVIDUAL RESULTS:")
        print(f"{'Name':<30} {'FPS':>8} {'Latency':>10} {'Memory':>10} {'Status':>8}")
        print("-" * 70)

        for result in self.results:
            status = "PASS" if (
                result.meets_fps_target and
                result.meets_latency_target and
                result.meets_memory_target
            ) else "FAIL"

            print(
                f"{result.name:<30} "
                f"{result.fps_equivalent:>7.1f}F "
                f"{result.avg_time_ms:>9.3f}ms "
                f"{result.peak_memory_mb:>9.2f}MB "
                f"{status:>8}"
            )

    def export_results(self, output_path: Path):
        """Export results to JSON file"""
        data = {
            'config': {
                'warmup_iterations': self.config.warmup_iterations,
                'benchmark_iterations': self.config.benchmark_iterations,
                'fps_target': self.config.fps_target,
                'latency_target_ms': self.config.latency_target_ms,
                'memory_target_mb': self.config.memory_target_mb,
            },
            'results': [r.to_dict() for r in self.results],
            'summary': {
                'total_benchmarks': len(self.results),
                'fps_target_met': sum(1 for r in self.results if r.meets_fps_target),
                'latency_target_met': sum(1 for r in self.results if r.meets_latency_target),
                'memory_target_met': sum(1 for r in self.results if r.meets_memory_target),
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults exported to: {output_path}")


# Example benchmark functions
def benchmark_projectile_physics():
    """Benchmark projectile physics calculations"""
    import math

    def calculate_trajectory(v, angle, t):
        angle_rad = math.radians(angle)
        vx = v * math.cos(angle_rad)
        vy = v * math.sin(angle_rad)
        x = vx * t
        y = vy * t - 0.5 * 9.81 * t * t
        return (x, y, 0.0)

    config = BenchmarkConfig(
        warmup_iterations=100,
        benchmark_iterations=10000,
    )

    benchmark = PerformanceBenchmark(config)

    result = benchmark.benchmark_function(
        "projectile_trajectory",
        calculate_trajectory,
        args=(20.0, 45.0, 1.0),
        description="Calculate projectile trajectory at t=1.0s"
    )

    benchmark.print_result(result)
    benchmark.print_summary()

    return benchmark


if __name__ == "__main__":
    print("Portalis WASM Performance Benchmark Suite")
    print("=" * 70)

    # Run example benchmark
    benchmark = benchmark_projectile_physics()

    # Export results
    output_path = Path(__file__).parent / "results" / "benchmark_results.json"
    benchmark.export_results(output_path)
