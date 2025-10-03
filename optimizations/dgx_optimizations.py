"""
DGX Cloud Optimization Module

Implements advanced scheduling and resource optimization for DGX Cloud:
- Intelligent job scheduling
- Spot instance strategy
- Cost optimization
- GPU utilization maximization
- Multi-node coordination
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class InstanceType(Enum):
    """Instance type options."""
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"


@dataclass
class Job:
    """Represents a translation job."""
    id: str
    code_size_loc: int
    estimated_gpu_hours: float
    priority: JobPriority
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    gpu_memory_gb: float = 16.0
    min_gpus: int = 1
    max_gpus: int = 8
    preemptible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterNode:
    """Represents a DGX node."""
    node_id: str
    gpu_count: int
    gpu_memory_gb: float
    instance_type: InstanceType
    cost_per_hour: float
    available_gpus: int
    current_jobs: List[str] = field(default_factory=list)


@dataclass
class DGXOptimizationConfig:
    """Configuration for DGX Cloud optimization."""

    # Cost optimization
    target_cost_per_translation: float = 0.10
    max_hourly_cost: float = 100.0
    spot_instance_ratio: float = 0.7  # 70% spot, 30% on-demand

    # Resource utilization
    target_gpu_utilization: float = 0.80
    min_gpu_utilization: float = 0.60
    max_pending_jobs: int = 100

    # Scheduling
    scheduling_algorithm: str = "priority_aware"  # "fifo", "priority_aware", "deadline_aware"
    enable_preemption: bool = True
    enable_checkpointing: bool = True

    # Scaling
    enable_autoscaling: bool = True
    scale_up_threshold: float = 0.9
    scale_down_threshold: float = 0.3
    min_nodes: int = 1
    max_nodes: int = 10

    # Spot instance strategy
    enable_spot_instances: bool = True
    spot_fallback_on_demand: bool = True
    max_spot_interruptions: int = 3


class JobScheduler:
    """
    Advanced job scheduler for DGX Cloud.

    Implements priority-aware scheduling with cost optimization.
    """

    def __init__(self, config: DGXOptimizationConfig):
        self.config = config
        self.pending_jobs: List[Job] = []
        self.running_jobs: List[Job] = []
        self.completed_jobs: List[Job] = []
        self.nodes: List[ClusterNode] = []

        self.stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'total_gpu_hours': 0.0,
            'total_cost': 0.0,
            'avg_utilization': 0.0
        }

    def submit_job(self, job: Job):
        """Submit job for scheduling."""
        self.pending_jobs.append(job)
        self.stats['total_jobs'] += 1
        logger.info(f"Job {job.id} submitted (priority: {job.priority.name})")

        # Sort by priority and deadline
        self._sort_pending_jobs()

    def _sort_pending_jobs(self):
        """Sort pending jobs by priority and deadline."""
        def job_score(job: Job) -> Tuple[int, float]:
            priority_score = job.priority.value
            deadline_score = 0.0

            if job.deadline:
                time_until_deadline = (job.deadline - datetime.now()).total_seconds()
                deadline_score = -time_until_deadline  # Urgent jobs first

            return (priority_score, deadline_score)

        self.pending_jobs.sort(key=job_score)

    def schedule_jobs(self) -> List[Tuple[Job, ClusterNode]]:
        """
        Schedule pending jobs to available nodes.

        Returns:
            List of (job, node) assignments
        """
        assignments = []

        for job in self.pending_jobs[:]:
            # Check dependencies
            if not self._dependencies_satisfied(job):
                continue

            # Find suitable node
            node = self._find_best_node(job)
            if node:
                assignments.append((job, node))
                self.pending_jobs.remove(job)
                self.running_jobs.append(job)
                node.available_gpus -= job.min_gpus
                node.current_jobs.append(job.id)
                logger.info(f"Scheduled job {job.id} on node {node.node_id}")

        return assignments

    def _dependencies_satisfied(self, job: Job) -> bool:
        """Check if job dependencies are satisfied."""
        for dep_id in job.dependencies:
            if not any(j.id == dep_id for j in self.completed_jobs):
                return False
        return True

    def _find_best_node(self, job: Job) -> Optional[ClusterNode]:
        """
        Find best node for job.

        Strategy:
        1. Prefer nodes with exact GPU count match
        2. Prefer spot instances for preemptible jobs
        3. Prefer nodes with lower cost
        4. Prefer nodes with higher utilization
        """
        candidates = []

        for node in self.nodes:
            # Check if node has enough resources
            if node.available_gpus < job.min_gpus:
                continue

            if node.gpu_memory_gb < job.gpu_memory_gb:
                continue

            # Calculate score
            gpu_match_score = 1.0 / (abs(node.available_gpus - job.min_gpus) + 1)
            cost_score = 1.0 / (node.cost_per_hour + 1)

            # Prefer spot for preemptible jobs
            instance_score = 1.0
            if job.preemptible and node.instance_type == InstanceType.SPOT:
                instance_score = 1.5

            total_score = gpu_match_score + cost_score + instance_score
            candidates.append((total_score, node))

        if not candidates:
            return None

        # Return best candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def complete_job(self, job_id: str, success: bool = True):
        """Mark job as completed."""
        job = next((j for j in self.running_jobs if j.id == job_id), None)
        if not job:
            logger.warning(f"Job {job_id} not found in running jobs")
            return

        self.running_jobs.remove(job)

        if success:
            self.completed_jobs.append(job)
            self.stats['completed_jobs'] += 1
            logger.info(f"Job {job_id} completed successfully")
        else:
            self.stats['failed_jobs'] += 1
            logger.error(f"Job {job_id} failed")

        # Release node resources
        for node in self.nodes:
            if job_id in node.current_jobs:
                node.current_jobs.remove(job_id)
                node.available_gpus += job.min_gpus

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            **self.stats,
            'pending_jobs': len(self.pending_jobs),
            'running_jobs': len(self.running_jobs),
            'avg_cost_per_job': self.stats['total_cost'] / max(1, self.stats['completed_jobs'])
        }


class SpotInstanceManager:
    """
    Manages spot instances with fallback strategy.

    Handles interruptions and cost optimization.
    """

    def __init__(self, config: DGXOptimizationConfig):
        self.config = config
        self.spot_price_history: List[float] = []
        self.interruption_count = 0

        self.stats = {
            'spot_launches': 0,
            'spot_interruptions': 0,
            'fallback_to_on_demand': 0,
            'total_cost_saved': 0.0
        }

    def request_instance(
        self,
        gpu_count: int,
        gpu_memory_gb: float,
        job: Job
    ) -> Optional[ClusterNode]:
        """
        Request instance with spot-first strategy.

        Args:
            gpu_count: Number of GPUs needed
            gpu_memory_gb: Memory per GPU
            job: Job to run

        Returns:
            Allocated node or None
        """
        # Try spot instance first
        if self.config.enable_spot_instances and job.preemptible:
            spot_node = self._request_spot_instance(gpu_count, gpu_memory_gb)
            if spot_node:
                self.stats['spot_launches'] += 1
                return spot_node

        # Fallback to on-demand
        if self.config.spot_fallback_on_demand:
            on_demand_node = self._request_on_demand_instance(gpu_count, gpu_memory_gb)
            if on_demand_node:
                self.stats['fallback_to_on_demand'] += 1
                return on_demand_node

        return None

    def _request_spot_instance(
        self,
        gpu_count: int,
        gpu_memory_gb: float
    ) -> Optional[ClusterNode]:
        """Request spot instance."""
        # Simulate spot instance pricing (30% cheaper than on-demand)
        base_cost = 3.0 * gpu_count  # $3/GPU/hour
        spot_cost = base_cost * 0.7

        node = ClusterNode(
            node_id=f"spot-{time.time()}",
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            instance_type=InstanceType.SPOT,
            cost_per_hour=spot_cost,
            available_gpus=gpu_count
        )

        logger.info(f"Launched spot instance {node.node_id} at ${spot_cost:.2f}/hr")
        return node

    def _request_on_demand_instance(
        self,
        gpu_count: int,
        gpu_memory_gb: float
    ) -> Optional[ClusterNode]:
        """Request on-demand instance."""
        cost = 3.0 * gpu_count  # $3/GPU/hour

        node = ClusterNode(
            node_id=f"on-demand-{time.time()}",
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            instance_type=InstanceType.ON_DEMAND,
            cost_per_hour=cost,
            available_gpus=gpu_count
        )

        logger.info(f"Launched on-demand instance {node.node_id} at ${cost:.2f}/hr")
        return node

    def handle_spot_interruption(self, node_id: str) -> bool:
        """
        Handle spot instance interruption.

        Returns:
            True if successfully handled
        """
        self.interruption_count += 1
        self.stats['spot_interruptions'] += 1

        logger.warning(f"Spot instance {node_id} interrupted")

        # Check if we've exceeded max interruptions
        if self.interruption_count >= self.config.max_spot_interruptions:
            logger.info("Max spot interruptions reached, switching to on-demand")
            return False

        return True

    def get_cost_savings(self) -> Dict[str, float]:
        """Calculate cost savings from spot instances."""
        spot_savings = self.stats['spot_launches'] * 0.3 * 3.0  # 30% savings
        return {
            'total_saved': spot_savings,
            'spot_launches': self.stats['spot_launches'],
            'interruptions': self.stats['spot_interruptions'],
            'savings_rate': 0.30
        }


class AutoScaler:
    """
    Automatic cluster scaling based on workload.

    Scales up/down based on utilization.
    """

    def __init__(self, config: DGXOptimizationConfig):
        self.config = config
        self.current_nodes = config.min_nodes

        self.stats = {
            'scale_ups': 0,
            'scale_downs': 0,
            'total_node_hours': 0.0
        }

    def should_scale_up(self, pending_jobs: int, current_utilization: float) -> bool:
        """Determine if cluster should scale up."""
        if current_utilization > self.config.scale_up_threshold:
            return True

        if pending_jobs > self.config.max_pending_jobs:
            return True

        return False

    def should_scale_down(self, current_utilization: float) -> bool:
        """Determine if cluster should scale down."""
        if self.current_nodes <= self.config.min_nodes:
            return False

        if current_utilization < self.config.scale_down_threshold:
            return True

        return False

    def scale_up(self) -> int:
        """Scale up cluster."""
        if self.current_nodes >= self.config.max_nodes:
            logger.warning("Already at max nodes")
            return 0

        new_nodes = min(2, self.config.max_nodes - self.current_nodes)
        self.current_nodes += new_nodes
        self.stats['scale_ups'] += 1

        logger.info(f"Scaled up cluster by {new_nodes} nodes (total: {self.current_nodes})")
        return new_nodes

    def scale_down(self) -> int:
        """Scale down cluster."""
        if self.current_nodes <= self.config.min_nodes:
            return 0

        removed_nodes = min(1, self.current_nodes - self.config.min_nodes)
        self.current_nodes -= removed_nodes
        self.stats['scale_downs'] += 1

        logger.info(f"Scaled down cluster by {removed_nodes} nodes (total: {self.current_nodes})")
        return removed_nodes


class DGXOptimizer:
    """
    Complete DGX Cloud optimization suite.

    Combines scheduling, spot instances, and autoscaling.
    """

    def __init__(self, config: Optional[DGXOptimizationConfig] = None):
        self.config = config or DGXOptimizationConfig()

        self.scheduler = JobScheduler(self.config)
        self.spot_manager = SpotInstanceManager(self.config)
        self.autoscaler = AutoScaler(self.config)

    def submit_translation_job(
        self,
        code_size_loc: int,
        priority: JobPriority = JobPriority.NORMAL,
        deadline: Optional[datetime] = None
    ) -> str:
        """
        Submit translation job.

        Args:
            code_size_loc: Lines of code to translate
            priority: Job priority
            deadline: Optional deadline

        Returns:
            Job ID
        """
        # Estimate resource requirements
        estimated_gpu_hours = code_size_loc / 10000.0  # 10K LOC per GPU-hour
        gpu_memory_gb = min(80, max(16, code_size_loc / 1000))  # Scale with size
        min_gpus = min(8, max(1, code_size_loc // 50000))  # 1 GPU per 50K LOC

        job = Job(
            id=f"job-{time.time()}",
            code_size_loc=code_size_loc,
            estimated_gpu_hours=estimated_gpu_hours,
            priority=priority,
            deadline=deadline,
            gpu_memory_gb=gpu_memory_gb,
            min_gpus=min_gpus
        )

        self.scheduler.submit_job(job)
        return job.id

    def optimize_cluster(self):
        """Run one optimization cycle."""
        # Check if we need to scale
        pending = len(self.scheduler.pending_jobs)
        utilization = self._calculate_utilization()

        if self.autoscaler.should_scale_up(pending, utilization):
            nodes_added = self.autoscaler.scale_up()
            for _ in range(nodes_added):
                node = self.spot_manager.request_instance(8, 80, Job(
                    id="temp", code_size_loc=0, estimated_gpu_hours=0,
                    priority=JobPriority.NORMAL
                ))
                if node:
                    self.scheduler.nodes.append(node)

        elif self.autoscaler.should_scale_down(utilization):
            self.autoscaler.scale_down()

        # Schedule pending jobs
        assignments = self.scheduler.schedule_jobs()
        logger.info(f"Scheduled {len(assignments)} jobs")

    def _calculate_utilization(self) -> float:
        """Calculate current GPU utilization."""
        if not self.scheduler.nodes:
            return 0.0

        total_gpus = sum(n.gpu_count for n in self.scheduler.nodes)
        used_gpus = sum(n.gpu_count - n.available_gpus for n in self.scheduler.nodes)

        return used_gpus / max(1, total_gpus)

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        utilization = self._calculate_utilization()
        spot_savings = self.spot_manager.get_cost_savings()

        return {
            'cluster': {
                'nodes': len(self.scheduler.nodes),
                'total_gpus': sum(n.gpu_count for n in self.scheduler.nodes),
                'utilization': utilization,
                'target_utilization': self.config.target_gpu_utilization
            },
            'scheduler': self.scheduler.get_stats(),
            'spot_instances': spot_savings,
            'autoscaler': self.autoscaler.stats,
            'cost_efficiency': {
                'target_cost_per_translation': self.config.target_cost_per_translation,
                'actual_cost_per_translation': self.scheduler.stats.get('avg_cost_per_job', 0),
                'total_cost_saved': spot_savings['total_saved']
            }
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = DGXOptimizationConfig(
        target_cost_per_translation=0.08,
        spot_instance_ratio=0.75,
        target_gpu_utilization=0.85
    )

    optimizer = DGXOptimizer(config)

    # Submit some jobs
    job1 = optimizer.submit_translation_job(10000, JobPriority.HIGH)
    job2 = optimizer.submit_translation_job(50000, JobPriority.NORMAL)
    job3 = optimizer.submit_translation_job(100000, JobPriority.LOW)

    # Run optimization
    optimizer.optimize_cluster()

    # Get report
    report = optimizer.get_optimization_report()
    print("\nDGX Optimization Report:")
    import json
    print(json.dumps(report, indent=2))
