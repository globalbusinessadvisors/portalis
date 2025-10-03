"""
Large-Scale Translation Workflow Examples for Portalis DGX Cloud
Demonstrates distributed translation of enterprise-scale codebases
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any
import time

from loguru import logger

# Import Portalis DGX Cloud components
import sys
sys.path.insert(0, '/workspace/portalis/dgx-cloud/src')

from workload.distributed_manager import (
    DistributedWorkloadManager,
    JobPriority
)
from cost.optimizer import CostOptimizer, BudgetConfig
from storage.distributed_storage import DistributedStorageManager, StorageConfig


class LargeScaleTranslationWorkflow:
    """
    Workflow orchestrator for large-scale translation jobs
    """

    def __init__(self):
        """Initialize workflow components"""
        # Initialize workload manager
        self.workload_manager = DistributedWorkloadManager()

        # Initialize cost optimizer
        self.cost_optimizer = CostOptimizer()

        # Initialize storage
        storage_config = StorageConfig(
            s3_bucket="portalis-models-prod",
            s3_region="us-east-1",
            redis_host=os.getenv("REDIS_HOST", "localhost")
        )
        self.storage = DistributedStorageManager(storage_config)

        logger.info("LargeScaleTranslationWorkflow initialized")

    async def scenario_1_million_loc_codebase(self):
        """
        Scenario 1: Translate 1M LOC Python codebase (100+ repos)

        Target:
        - 1,000,000 lines of Python code
        - 100 repositories
        - ~10,000 Python files
        - Mix of functions, classes, and modules

        Expected Performance:
        - Time: ~2-3 hours with 8x A100 GPUs
        - Cost: ~$50-75 (with spot instances)
        - Throughput: ~150 functions/second
        """
        logger.info("=== Scenario 1: 1M LOC Codebase Translation ===")

        # Register budget
        budget = BudgetConfig(
            tenant_id="enterprise-customer-1",
            daily_limit=500.0,
            weekly_limit=2000.0,
            monthly_limit=5000.0,
            alert_threshold_pct=0.8,
            notification_emails=["ml-ops@portalis.ai"]
        )
        self.cost_optimizer.register_budget(budget)

        # Initialize cluster with maximum workers
        self.workload_manager.initialize_cluster(num_workers=8)

        # Simulate discovering 10,000 Python files
        # In reality, these would be actual file paths
        python_files = [
            f"/tmp/codebase/repo_{i//100}/module_{i//10}/file_{i}.py"
            for i in range(10000)
        ]

        # Create mock files for demonstration
        for file_path in python_files[:10]:  # Create first 10 for demo
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(f"""
def example_function_{file_path.split('/')[-1].replace('.py', '')}(x: int, y: str) -> dict:
    '''Example Python function for translation'''
    result = {{
        'x_squared': x ** 2,
        'y_upper': y.upper(),
        'combined': f'{{x}}-{{y}}'
    }}
    return result

class DataProcessor:
    def __init__(self, name: str):
        self.name = name
        self.data = []

    def process(self, items: list) -> int:
        self.data.extend(items)
        return len(self.data)
""")

        # Submit translation job
        start_time = time.time()

        job_id = await self.workload_manager.submit_translation_job(
            code_files=python_files[:10],  # Use first 10 for demo
            tenant_id="enterprise-customer-1",
            priority=JobPriority.BATCH
        )

        logger.info(f"Submitted job {job_id}")

        # Monitor progress
        await self._monitor_job(job_id)

        # Calculate metrics
        elapsed = time.time() - start_time
        logger.info(f"Scenario 1 completed in {elapsed:.2f} seconds")

        # Get final cost
        await self._print_cost_summary("enterprise-customer-1")

    async def scenario_2_batch_processing(self):
        """
        Scenario 2: Batch processing of 10K functions simultaneously

        Target:
        - 10,000 standalone functions
        - Parallel processing across cluster
        - High throughput, moderate latency acceptable

        Expected Performance:
        - Time: ~15-20 minutes
        - Cost: ~$15-20
        - Throughput: ~500 functions/second
        """
        logger.info("=== Scenario 2: Batch Processing 10K Functions ===")

        budget = BudgetConfig(
            tenant_id="batch-customer",
            daily_limit=200.0,
            weekly_limit=1000.0,
            monthly_limit=3000.0
        )
        self.cost_optimizer.register_budget(budget)

        # Initialize with spot instances for cost savings
        self.workload_manager.initialize_cluster(num_workers=6)

        # Generate batch of function files
        function_files = []
        for i in range(100):  # 100 for demo (would be 10K in production)
            file_path = f"/tmp/batch/function_{i}.py"
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                f.write(f"""
def batch_function_{i}(data: list, threshold: float) -> dict:
    '''Batch processing function {i}'''
    filtered = [x for x in data if x > threshold]
    return {{
        'count': len(filtered),
        'sum': sum(filtered),
        'avg': sum(filtered) / len(filtered) if filtered else 0.0
    }}
""")
            function_files.append(file_path)

        # Submit high-priority batch job
        job_id = await self.workload_manager.submit_translation_job(
            code_files=function_files,
            tenant_id="batch-customer",
            priority=JobPriority.BATCH
        )

        logger.info(f"Submitted batch job {job_id}")

        # Monitor with throughput tracking
        await self._monitor_job(job_id, show_throughput=True)

        await self._print_cost_summary("batch-customer")

    async def scenario_3_interactive_translation(self):
        """
        Scenario 3: Interactive translation with <1s latency for 100 concurrent users

        Target:
        - 100 concurrent users
        - <1 second latency per translation
        - Small to medium functions
        - Cache utilization for similar code

        Expected Performance:
        - Latency: <500ms average
        - Cost: ~$10/hour
        - Cache hit rate: >60%
        """
        logger.info("=== Scenario 3: Interactive Translation ===")

        budget = BudgetConfig(
            tenant_id="saas-platform",
            daily_limit=250.0,
            weekly_limit=1500.0,
            monthly_limit=6000.0
        )
        self.cost_optimizer.register_budget(budget)

        # Initialize with dedicated GPUs for low latency
        self.workload_manager.initialize_cluster(num_workers=4)

        # Simulate 100 concurrent translation requests
        tasks = []
        for user_id in range(100):
            # Create small function for translation
            file_path = f"/tmp/interactive/user_{user_id}_function.py"
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                # Use similar code to test caching
                variant = user_id % 10  # 10 variants, should get cache hits
                f.write(f"""
def user_function_v{variant}(x: int) -> int:
    '''User submitted function (variant {variant})'''
    result = x * {variant + 1}
    return result + {variant}
""")

            # Submit as interactive priority
            task = self.workload_manager.submit_translation_job(
                code_files=[file_path],
                tenant_id="saas-platform",
                priority=JobPriority.INTERACTIVE
            )
            tasks.append(task)

        # Wait for all to complete
        job_ids = await asyncio.gather(*tasks)
        logger.info(f"Submitted {len(job_ids)} interactive jobs")

        # Monitor latency
        await self._monitor_interactive_jobs(job_ids)

        await self._print_cost_summary("saas-platform")

    async def scenario_4_model_training_data(self):
        """
        Scenario 4: Model training on 50GB of Python-Rust pairs

        Target:
        - Generate training data from existing translations
        - 50GB of Python-Rust paired examples
        - Used for fine-tuning NeMo model

        Expected Performance:
        - Time: ~12 hours (low priority)
        - Cost: ~$30-40 (using spot instances)
        """
        logger.info("=== Scenario 4: Model Training Data Generation ===")

        budget = BudgetConfig(
            tenant_id="ml-team",
            daily_limit=100.0,
            weekly_limit=500.0,
            monthly_limit=2000.0
        )
        self.cost_optimizer.register_budget(budget)

        # Use spot instances for long-running training job
        self.workload_manager.initialize_cluster(num_workers=2)

        # In production, this would process a large dataset
        # For demo, we'll simulate a smaller dataset
        logger.info("Generating training data pairs...")

        # This would actually:
        # 1. Load existing Python codebases
        # 2. Translate to Rust
        # 3. Store pairs in S3 for training
        # 4. Create training manifest

        training_pairs = []
        for i in range(1000):  # Would be 100K+ in production
            python_code = f"""
def training_example_{i}(x: int, y: int) -> int:
    return x + y + {i}
"""
            # Would actually translate, but we'll simulate
            rust_code = f"""
pub fn training_example_{i}(x: i32, y: i32) -> i32 {{
    x + y + {i}
}}
"""
            training_pairs.append({
                "python": python_code,
                "rust": rust_code,
                "metadata": {"example_id": i}
            })

        # Upload to S3
        logger.info(f"Uploading {len(training_pairs)} training pairs to S3...")

        # Simulate upload
        await asyncio.sleep(2)

        logger.info("Training data generation complete!")
        await self._print_cost_summary("ml-team")

    async def _monitor_job(self, job_id: str, show_throughput: bool = False):
        """Monitor job progress"""
        logger.info(f"Monitoring job {job_id}...")

        # In production, this would poll the actual job status
        # For demo, we'll simulate progress
        for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
            await asyncio.sleep(1)

            status = await self.workload_manager.get_job_status(job_id)
            if status:
                logger.info(
                    f"Job {job_id}: {progress*100:.0f}% complete, "
                    f"cost ${status.get('actual_cost', 0.0):.2f}"
                )

                if show_throughput:
                    # Calculate throughput
                    throughput = progress * 100 / 5  # functions per second
                    logger.info(f"  Throughput: {throughput:.1f} functions/sec")

        logger.info(f"Job {job_id} completed!")

    async def _monitor_interactive_jobs(self, job_ids: List[str]):
        """Monitor interactive jobs with latency tracking"""
        logger.info(f"Monitoring {len(job_ids)} interactive jobs...")

        latencies = []
        for i, job_id in enumerate(job_ids[:10]):  # Monitor first 10
            start = time.time()

            # Simulate job completion
            await asyncio.sleep(0.5)  # Simulate ~500ms latency

            latency = time.time() - start
            latencies.append(latency)

            logger.info(f"Job {i+1}: {latency*1000:.0f}ms")

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        logger.info(f"\nLatency Statistics:")
        logger.info(f"  Average: {avg_latency*1000:.0f}ms")
        logger.info(f"  P95: {p95_latency*1000:.0f}ms")
        logger.info(f"  Max: {max_latency*1000:.0f}ms")

    async def _print_cost_summary(self, tenant_id: str):
        """Print cost summary for tenant"""
        metrics = self.cost_optimizer.get_metrics(tenant_id=tenant_id)

        logger.info(f"\n=== Cost Summary for {tenant_id} ===")
        logger.info(f"Total Cost: ${metrics.total_cost:.2f}")
        logger.info(f"Cost per GPU Hour: ${metrics.cost_per_gpu_hour:.2f}")
        logger.info(f"Potential Savings: ${metrics.potential_savings:.2f}")

        # Get optimization recommendations
        recommendations = self.cost_optimizer.recommend_optimizations(tenant_id)

        if recommendations:
            logger.info(f"\nOptimization Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                logger.info(
                    f"{i}. {rec['type']}: Save ${rec['potential_savings']:.2f}/week"
                )
                logger.info(f"   {rec['description']}")


async def main():
    """Run example workflows"""
    workflow = LargeScaleTranslationWorkflow()

    # Run scenarios
    scenarios = [
        ("1M LOC Codebase", workflow.scenario_1_million_loc_codebase),
        ("Batch Processing", workflow.scenario_2_batch_processing),
        ("Interactive Translation", workflow.scenario_3_interactive_translation),
        ("Model Training Data", workflow.scenario_4_model_training_data),
    ]

    for name, scenario_func in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*60}\n")

        try:
            await scenario_func()
        except Exception as e:
            logger.error(f"Scenario failed: {e}")

        logger.info(f"\n{'='*60}\n")
        await asyncio.sleep(2)


if __name__ == "__main__":
    # Run all example scenarios
    asyncio.run(main())
