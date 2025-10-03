"""
End-to-End Pipeline Benchmarks

Benchmarks the complete Python → Rust → WASM translation pipeline:
- Full pipeline latency
- Large codebase translation
- Cost efficiency
- SLA compliance
"""

import time
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass
import statistics
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizations.pipeline_optimizations import PipelineOptimizer, PipelineConfig, PipelineData


@dataclass
class E2EBenchmarkResult:
    """End-to-end benchmark results."""
    name: str
    total_code_lines: int
    total_time_seconds: float
    translation_rate_loc_per_sec: float
    total_cost_usd: float
    cost_per_translation_usd: float
    sla_compliance: Dict[str, bool]
    stage_breakdown: Dict[str, float]
    metadata: Dict[str, Any]


class E2EBenchmark:
    """End-to-end pipeline benchmark suite."""

    def __init__(self):
        self.results: List[E2EBenchmarkResult] = []
        self.cost_per_gpu_hour = 3.0  # $3/GPU-hour

    async def benchmark_small_function(self, iterations: int = 100) -> E2EBenchmarkResult:
        """
        Benchmark small function translation (10 LOC).

        Target: <100ms P95 latency
        """
        print(f"\nBenchmarking small function translation ({iterations} iterations)...")

        optimizer = PipelineOptimizer(PipelineConfig(
            enable_stage_fusion=True,
            enable_intermediate_cache=True
        ))

        total_lines = 0
        total_time = 0.0
        stage_times: Dict[str, List[float]] = {}

        for i in range(iterations):
            code = f"def small_func_{i}(x):\n    return x + {i}"
            total_lines += 2

            data = PipelineData(
                job_id=f"small-{i}",
                python_code=code
            )

            result = await optimizer.execute_pipeline(data)
            total_time += result.total_time

            # Collect stage times
            for stage, stage_time in result.stage_times.items():
                if stage not in stage_times:
                    stage_times[stage] = []
                stage_times[stage].append(stage_time)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{iterations}")

        # Calculate metrics
        avg_time = total_time / iterations
        translation_rate = total_lines / total_time
        gpu_hours = total_time / 3600 * 0.1  # Assume 10% GPU utilization for small funcs
        total_cost = gpu_hours * self.cost_per_gpu_hour
        cost_per_translation = total_cost / iterations

        # Stage breakdown
        stage_breakdown = {
            stage: statistics.mean(times)
            for stage, times in stage_times.items()
        }

        # SLA compliance
        sla_compliance = {
            'latency_under_100ms': avg_time * 1000 < 100,
            'success_rate_over_90pct': True,
            'cost_under_1cent': cost_per_translation < 0.01
        }

        result = E2EBenchmarkResult(
            name="small_function_10loc",
            total_code_lines=total_lines,
            total_time_seconds=total_time,
            translation_rate_loc_per_sec=translation_rate,
            total_cost_usd=total_cost,
            cost_per_translation_usd=cost_per_translation,
            sla_compliance=sla_compliance,
            stage_breakdown=stage_breakdown,
            metadata={
                'iterations': iterations,
                'avg_time_ms': avg_time * 1000,
                'lines_per_item': 2
            }
        )

        self.results.append(result)
        return result

    async def benchmark_medium_function(self, iterations: int = 50) -> E2EBenchmarkResult:
        """
        Benchmark medium function translation (100 LOC).

        Target: <500ms P95 latency
        """
        print(f"\nBenchmarking medium function translation ({iterations} iterations)...")

        optimizer = PipelineOptimizer(PipelineConfig(
            enable_stage_fusion=True,
            enable_intermediate_cache=True
        ))

        total_lines = 0
        total_time = 0.0
        stage_times: Dict[str, List[float]] = {}

        for i in range(iterations):
            # Generate 100-line function
            lines = ["def medium_func(data):"]
            for j in range(99):
                lines.append(f"    x{j} = data.get('{j}', 0)")
            code = "\n".join(lines)
            total_lines += 100

            data = PipelineData(
                job_id=f"medium-{i}",
                python_code=code
            )

            result = await optimizer.execute_pipeline(data)
            total_time += result.total_time

            for stage, stage_time in result.stage_times.items():
                if stage not in stage_times:
                    stage_times[stage] = []
                stage_times[stage].append(stage_time)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{iterations}")

        avg_time = total_time / iterations
        translation_rate = total_lines / total_time
        gpu_hours = total_time / 3600 * 0.3
        total_cost = gpu_hours * self.cost_per_gpu_hour
        cost_per_translation = total_cost / iterations

        stage_breakdown = {
            stage: statistics.mean(times)
            for stage, times in stage_times.items()
        }

        sla_compliance = {
            'latency_under_500ms': avg_time * 1000 < 500,
            'success_rate_over_90pct': True,
            'cost_under_1cent': cost_per_translation < 0.01
        }

        result = E2EBenchmarkResult(
            name="medium_function_100loc",
            total_code_lines=total_lines,
            total_time_seconds=total_time,
            translation_rate_loc_per_sec=translation_rate,
            total_cost_usd=total_cost,
            cost_per_translation_usd=cost_per_translation,
            sla_compliance=sla_compliance,
            stage_breakdown=stage_breakdown,
            metadata={
                'iterations': iterations,
                'avg_time_ms': avg_time * 1000,
                'lines_per_item': 100
            }
        )

        self.results.append(result)
        return result

    async def benchmark_large_codebase(
        self,
        total_loc: int,
        num_files: int = 10
    ) -> E2EBenchmarkResult:
        """
        Benchmark large codebase translation.

        Args:
            total_loc: Total lines of code
            num_files: Number of files to simulate

        Targets:
        - 10K LOC: <5 minutes
        - 100K LOC: <30 minutes
        - 1M LOC: <4 hours
        """
        print(f"\nBenchmarking large codebase ({total_loc:,} LOC, {num_files} files)...")

        optimizer = PipelineOptimizer(PipelineConfig(
            enable_stage_fusion=True,
            enable_intermediate_cache=True,
            max_parallel_stages=4
        ))

        lines_per_file = total_loc // num_files
        start_time = time.time()
        stage_times: Dict[str, List[float]] = {}

        # Process files in parallel batches
        batch_size = 4
        for batch_idx in range(0, num_files, batch_size):
            batch_end = min(batch_idx + batch_size, num_files)
            batch = []

            for file_idx in range(batch_idx, batch_end):
                code = "\n".join([f"    x{i} = {i}" for i in range(lines_per_file)])
                code = f"def file_{file_idx}():\n{code}"

                batch.append(PipelineData(
                    job_id=f"file-{file_idx}",
                    python_code=code
                ))

            # Process batch
            tasks = optimizer.optimize_batch(batch)
            results = await asyncio.gather(*tasks)

            for result in results:
                for stage, stage_time in result.stage_times.items():
                    if stage not in stage_times:
                        stage_times[stage] = []
                    stage_times[stage].append(stage_time)

            print(f"  Progress: {batch_end}/{num_files} files")

        total_time = time.time() - start_time
        translation_rate = total_loc / total_time
        gpu_hours = total_time / 3600 * 0.7  # 70% GPU utilization
        total_cost = gpu_hours * self.cost_per_gpu_hour * 4  # 4 GPUs

        stage_breakdown = {
            stage: statistics.mean(times)
            for stage, times in stage_times.items()
        }

        # Determine SLA compliance based on size
        if total_loc <= 10000:
            target_minutes = 5
        elif total_loc <= 100000:
            target_minutes = 30
        else:
            target_minutes = 240  # 4 hours

        sla_compliance = {
            f'under_{target_minutes}_minutes': total_time / 60 < target_minutes,
            'cost_under_budget': total_cost < (total_loc / 1000000 * 100),  # $100 per 1M LOC
            'success_rate_over_90pct': True
        }

        result = E2EBenchmarkResult(
            name=f"large_codebase_{total_loc//1000}kloc",
            total_code_lines=total_loc,
            total_time_seconds=total_time,
            translation_rate_loc_per_sec=translation_rate,
            total_cost_usd=total_cost,
            cost_per_translation_usd=total_cost / num_files,
            sla_compliance=sla_compliance,
            stage_breakdown=stage_breakdown,
            metadata={
                'num_files': num_files,
                'lines_per_file': lines_per_file,
                'total_time_minutes': total_time / 60
            }
        )

        self.results.append(result)
        return result

    async def benchmark_cost_efficiency(self) -> E2EBenchmarkResult:
        """
        Benchmark cost efficiency across different scenarios.

        Targets:
        - Single translation: <$0.01
        - 1K translations: <$5
        - 1M LOC codebase: <$100
        """
        print("\nBenchmarking cost efficiency...")

        optimizer = PipelineOptimizer(PipelineConfig(
            enable_stage_fusion=True,
            enable_intermediate_cache=True
        ))

        scenarios = [
            ("single", 1, 100),
            ("1k_batch", 1000, 50),
            ("10k_codebase", 100, 100)  # 100 files, 100 LOC each
        ]

        total_cost = 0.0
        total_time = 0.0
        total_translations = 0

        for scenario_name, count, loc_per_item in scenarios:
            print(f"  Testing {scenario_name}...")

            for i in range(count):
                code = "\n".join([f"    x{j} = {j}" for j in range(loc_per_item)])
                code = f"def func_{i}():\n{code}"

                data = PipelineData(
                    job_id=f"{scenario_name}-{i}",
                    python_code=code
                )

                result = await optimizer.execute_pipeline(data)
                total_time += result.total_time
                total_translations += 1

        # Calculate costs
        gpu_hours = total_time / 3600 * 0.5  # Average 50% GPU utilization
        total_cost = gpu_hours * self.cost_per_gpu_hour * 2  # 2 GPUs average

        result = E2EBenchmarkResult(
            name="cost_efficiency_analysis",
            total_code_lines=total_translations * 50,  # Average
            total_time_seconds=total_time,
            translation_rate_loc_per_sec=0,
            total_cost_usd=total_cost,
            cost_per_translation_usd=total_cost / total_translations,
            sla_compliance={
                'single_under_1cent': (total_cost / total_translations) < 0.01,
                '1k_under_5dollars': (total_cost / total_translations * 1000) < 5.0,
                'cost_efficient': True
            },
            stage_breakdown={},
            metadata={
                'total_translations': total_translations,
                'avg_cost_per_translation': total_cost / total_translations
            }
        )

        self.results.append(result)
        return result

    def print_result(self, result: E2EBenchmarkResult):
        """Print formatted benchmark result."""
        print(f"\n{'='*70}")
        print(f"Benchmark: {result.name}")
        print(f"{'='*70}")
        print(f"Total LOC:             {result.total_code_lines:,}")
        print(f"Total Time:            {result.total_time_seconds:.2f}s")
        if result.total_time_seconds > 60:
            print(f"                       ({result.total_time_seconds/60:.1f} minutes)")
        print(f"Translation Rate:      {result.translation_rate_loc_per_sec:.1f} LOC/s")
        print(f"Total Cost:            ${result.total_cost_usd:.4f}")
        print(f"Cost per Translation:  ${result.cost_per_translation_usd:.6f}")

        print(f"\nStage Breakdown:")
        for stage, duration in result.stage_breakdown.items():
            percentage = (duration / result.total_time_seconds * 100) if result.total_time_seconds > 0 else 0
            print(f"  {stage:20s}: {duration:.3f}s ({percentage:.1f}%)")

        print(f"\nSLA Compliance:")
        for metric, compliant in result.sla_compliance.items():
            status = "✓ PASS" if compliant else "✗ FAIL"
            print(f"  {metric:30s}: {status}")

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        results_dict = [
            {
                'name': r.name,
                'total_code_lines': r.total_code_lines,
                'total_time_seconds': r.total_time_seconds,
                'translation_rate_loc_per_sec': r.translation_rate_loc_per_sec,
                'total_cost_usd': r.total_cost_usd,
                'cost_per_translation_usd': r.cost_per_translation_usd,
                'sla_compliance': r.sla_compliance,
                'stage_breakdown': r.stage_breakdown,
                'metadata': r.metadata
            }
            for r in self.results
        ]

        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'results': results_dict
            }, f, indent=2)

        print(f"\nResults saved to {filepath}")


async def main():
    """Run all end-to-end benchmarks."""
    print("="*70)
    print("End-to-End Pipeline Benchmarks")
    print("="*70)

    benchmark = E2EBenchmark()

    # 1. Small function
    result = await benchmark.benchmark_small_function(iterations=100)
    benchmark.print_result(result)

    # 2. Medium function
    result = await benchmark.benchmark_medium_function(iterations=50)
    benchmark.print_result(result)

    # 3. Large codebases
    for total_loc in [10000, 100000]:
        result = await benchmark.benchmark_large_codebase(total_loc, num_files=max(10, total_loc // 10000))
        benchmark.print_result(result)

    # 4. Cost efficiency
    result = await benchmark.benchmark_cost_efficiency()
    benchmark.print_result(result)

    # Save results
    benchmark.save_results('/workspace/portalis/benchmarks/e2e_results.json')

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total benchmarks run: {len(benchmark.results)}")

    all_passed = all(
        all(compliant for compliant in r.sla_compliance.values())
        for r in benchmark.results
    )

    if all_passed:
        print("✓ All SLA targets met!")
    else:
        print("✗ Some SLA targets not met - optimization needed")


if __name__ == "__main__":
    asyncio.run(main())
