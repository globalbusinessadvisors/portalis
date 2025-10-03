"""
Distributed Workload Manager for Portalis DGX Cloud
Ray-based distributed processing with task scheduling, load balancing, and fault tolerance
"""

import os
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import json

import ray
from ray.util.queue import Queue as RayQueue
from ray.util.actor_pool import ActorPool
import redis
from loguru import logger
from pydantic import BaseModel, Field


class JobPriority(str, Enum):
    """Job priority levels"""
    INTERACTIVE = "interactive"
    BATCH = "batch"
    TRAINING = "training"
    LOW_PRIORITY = "low_priority"


class JobStatus(str, Enum):
    """Job status states"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobSize(str, Enum):
    """Job size classification"""
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


@dataclass
class ResourceRequirements:
    """Resource requirements for a job"""
    gpus: int = 1
    cpu_cores: int = 4
    memory_gb: float = 16.0
    gpu_memory_gb: float = 8.0
    timeout_seconds: int = 3600
    allow_spot: bool = True


@dataclass
class JobMetadata:
    """Metadata for a translation job"""
    job_id: str
    tenant_id: str
    priority: JobPriority
    size: JobSize
    resources: ResourceRequirements
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    result: Optional[Any] = None

    # Translation-specific
    num_functions: int = 0
    num_classes: int = 0
    lines_of_code: int = 0

    # Cost tracking
    estimated_cost: float = 0.0
    actual_cost: float = 0.0

    # Worker assignment
    worker_id: Optional[str] = None
    node_id: Optional[str] = None


@dataclass
class TranslationTask:
    """Individual translation task"""
    task_id: str
    job_id: str
    code: str
    language: str = "python"
    target_language: str = "rust"
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


class WorkStealingStrategy(str, Enum):
    """Work stealing strategies"""
    RANDOM = "random"
    LEAST_LOADED = "least_loaded"
    LOCALITY_AWARE = "locality_aware"


@ray.remote(num_gpus=1)
class TranslationWorker:
    """
    Ray actor for distributed translation work
    GPU-accelerated translation worker with fault tolerance
    """

    def __init__(self, worker_id: str, gpu_id: int):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.active_tasks: Dict[str, TranslationTask] = {}
        self.completed_tasks: int = 0
        self.failed_tasks: int = 0
        self.total_processing_time: float = 0.0

        # Initialize NeMo translator (lazy loading)
        self._translator = None

        logger.info(f"Worker {worker_id} initialized on GPU {gpu_id}")

    def _get_translator(self):
        """Lazy load translator to avoid serialization issues"""
        if self._translator is None:
            from nemo_integration.src.translation.translator import NeMoTranslator
            from nemo_integration.src.translation.nemo_service import NeMoService

            service = NeMoService(
                model_path="/mnt/portalis/models/translation_model.nemo",
                gpu_id=self.gpu_id
            )
            self._translator = NeMoTranslator(service)
            logger.info(f"Translator loaded on worker {self.worker_id}")

        return self._translator

    async def translate(self, task: TranslationTask) -> Dict[str, Any]:
        """
        Translate code task

        Args:
            task: Translation task to execute

        Returns:
            Translation result with metadata
        """
        start_time = time.time()
        self.active_tasks[task.task_id] = task

        try:
            translator = self._get_translator()

            # Perform translation
            result = translator.translate_function(
                python_code=task.code,
                context=task.context
            )

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.completed_tasks += 1

            return {
                "task_id": task.task_id,
                "success": True,
                "rust_code": result.rust_code,
                "imports": result.imports,
                "validation": result.validation_result,
                "processing_time": processing_time,
                "worker_id": self.worker_id,
                "gpu_id": self.gpu_id
            }

        except Exception as e:
            logger.error(f"Translation failed on worker {self.worker_id}: {e}")
            self.failed_tasks += 1

            return {
                "task_id": task.task_id,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "worker_id": self.worker_id
            }

        finally:
            self.active_tasks.pop(task.task_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "avg_processing_time": (
                self.total_processing_time / self.completed_tasks
                if self.completed_tasks > 0 else 0.0
            ),
            "success_rate": (
                self.completed_tasks / (self.completed_tasks + self.failed_tasks)
                if (self.completed_tasks + self.failed_tasks) > 0 else 1.0
            )
        }

    def is_idle(self) -> bool:
        """Check if worker is idle"""
        return len(self.active_tasks) == 0

    def get_load(self) -> float:
        """Get current load (0.0 to 1.0)"""
        max_concurrent = 4  # Max tasks per GPU
        return len(self.active_tasks) / max_concurrent


@ray.remote
class TaskScheduler:
    """
    Distributed task scheduler with priority queues and load balancing
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.jobs: Dict[str, JobMetadata] = {}
        self.tasks: Dict[str, TranslationTask] = {}

        # Priority queues
        self.queues = {
            JobPriority.INTERACTIVE: [],
            JobPriority.BATCH: [],
            JobPriority.TRAINING: [],
            JobPriority.LOW_PRIORITY: []
        }

        # Worker pool
        self.workers: Dict[str, ray.ObjectRef] = {}
        self.worker_stats: Dict[str, Dict] = {}

        # Redis for state persistence
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )

        logger.info("TaskScheduler initialized")

    async def submit_job(self, job: JobMetadata, tasks: List[TranslationTask]) -> str:
        """
        Submit a translation job

        Args:
            job: Job metadata
            tasks: List of translation tasks

        Returns:
            Job ID
        """
        job_id = job.job_id
        self.jobs[job_id] = job

        # Store tasks
        for task in tasks:
            self.tasks[task.task_id] = task

        # Add to appropriate queue
        self.queues[job.priority].append(job_id)

        # Update status
        job.status = JobStatus.QUEUED

        # Persist to Redis
        self._persist_job(job)

        logger.info(f"Job {job_id} submitted with {len(tasks)} tasks")
        return job_id

    async def schedule_next_task(self) -> Optional[Tuple[TranslationTask, str]]:
        """
        Get next task to schedule based on priority

        Returns:
            Tuple of (task, worker_id) or None
        """
        # Find available worker
        worker_id = await self._find_available_worker()
        if not worker_id:
            return None

        # Get highest priority job with pending tasks
        for priority in [JobPriority.INTERACTIVE, JobPriority.BATCH,
                        JobPriority.TRAINING, JobPriority.LOW_PRIORITY]:
            if self.queues[priority]:
                job_id = self.queues[priority][0]
                job = self.jobs[job_id]

                # Find pending task for this job
                for task_id, task in self.tasks.items():
                    if task.job_id == job_id and task_id not in self._get_active_tasks():
                        return task, worker_id

                # No more tasks for this job, remove from queue
                self.queues[priority].pop(0)

        return None

    async def _find_available_worker(self) -> Optional[str]:
        """Find least loaded worker"""
        if not self.workers:
            return None

        # Get load for all workers
        loads = {}
        for worker_id, worker_ref in self.workers.items():
            try:
                load = await worker_ref.get_load.remote()
                loads[worker_id] = load
            except Exception as e:
                logger.warning(f"Failed to get load for worker {worker_id}: {e}")
                loads[worker_id] = 1.0  # Assume full load on error

        # Find least loaded worker with capacity
        available = [(wid, load) for wid, load in loads.items() if load < 1.0]
        if not available:
            return None

        return min(available, key=lambda x: x[1])[0]

    def _get_active_tasks(self) -> set:
        """Get set of active task IDs"""
        active = set()
        for worker_stats in self.worker_stats.values():
            active.update(worker_stats.get("active_tasks", []))
        return active

    def _persist_job(self, job: JobMetadata):
        """Persist job state to Redis"""
        key = f"job:{job.job_id}"
        self.redis_client.set(key, json.dumps({
            "job_id": job.job_id,
            "tenant_id": job.tenant_id,
            "status": job.status.value,
            "priority": job.priority.value,
            "created_at": job.created_at.isoformat(),
            "num_functions": job.num_functions,
            "estimated_cost": job.estimated_cost
        }), ex=86400)  # 24 hour expiry

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current job status"""
        if job_id not in self.jobs:
            return None

        job = self.jobs[job_id]
        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": self._calculate_progress(job_id),
            "estimated_cost": job.estimated_cost,
            "actual_cost": job.actual_cost,
            "worker_id": job.worker_id
        }

    def _calculate_progress(self, job_id: str) -> float:
        """Calculate job completion progress"""
        job_tasks = [t for t in self.tasks.values() if t.job_id == job_id]
        if not job_tasks:
            return 0.0

        completed = len([t for t in job_tasks if t.task_id not in self._get_active_tasks()])
        return completed / len(job_tasks)


class DistributedWorkloadManager:
    """
    Main distributed workload manager
    Orchestrates Ray cluster, task scheduling, and resource management
    """

    def __init__(self, config_path: str = None):
        """
        Initialize distributed workload manager

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.ray_initialized = False
        self.scheduler = None
        self.workers = []
        self.worker_pool = None

        logger.info("DistributedWorkloadManager initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file"""
        if not config_path:
            config_path = "/workspace/portalis/dgx-cloud/config/resource_allocation.yaml"

        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize_cluster(self, num_workers: int = 4):
        """
        Initialize Ray cluster and workers

        Args:
            num_workers: Number of workers to spawn
        """
        if not self.ray_initialized:
            # Initialize Ray
            ray.init(
                address=os.getenv("RAY_ADDRESS", "auto"),
                runtime_env={
                    "working_dir": "/workspace/portalis",
                    "pip": [
                        "nemo-toolkit[all]",
                        "nvidia-pytriton",
                        "transformers",
                        "pydantic",
                        "loguru"
                    ]
                }
            )
            self.ray_initialized = True
            logger.info("Ray cluster initialized")

        # Initialize scheduler
        self.scheduler = TaskScheduler.remote(self.config)

        # Spawn workers
        for i in range(num_workers):
            worker_id = f"worker-{i}"
            worker = TranslationWorker.remote(worker_id, gpu_id=i % 8)
            self.workers.append(worker)

        logger.info(f"Spawned {num_workers} translation workers")

    async def submit_translation_job(
        self,
        code_files: List[str],
        tenant_id: str = "default",
        priority: JobPriority = JobPriority.BATCH
    ) -> str:
        """
        Submit a translation job

        Args:
            code_files: List of Python source files to translate
            tenant_id: Tenant identifier for multi-tenancy
            priority: Job priority level

        Returns:
            Job ID for tracking
        """
        # Generate job ID
        job_id = f"job-{uuid.uuid4().hex[:8]}"

        # Classify job size
        total_loc = sum(len(open(f).readlines()) for f in code_files)
        size = self._classify_job_size(total_loc, len(code_files))

        # Determine resource requirements
        resources = self._determine_resources(size, priority)

        # Create job metadata
        job = JobMetadata(
            job_id=job_id,
            tenant_id=tenant_id,
            priority=priority,
            size=size,
            resources=resources,
            num_functions=len(code_files),  # Simplified
            lines_of_code=total_loc,
            estimated_cost=self._estimate_cost(resources, total_loc)
        )

        # Create translation tasks
        tasks = []
        for file_path in code_files:
            with open(file_path, 'r') as f:
                code = f.read()

            task = TranslationTask(
                task_id=f"task-{uuid.uuid4().hex[:8]}",
                job_id=job_id,
                code=code,
                context={"file_path": file_path}
            )
            tasks.append(task)

        # Submit to scheduler
        await self.scheduler.submit_job.remote(job, tasks)

        logger.info(f"Submitted job {job_id} with {len(tasks)} tasks (size: {size})")
        return job_id

    def _classify_job_size(self, lines_of_code: int, num_files: int) -> JobSize:
        """Classify job size based on LOC"""
        thresholds = self.config["job_size_thresholds"]

        if lines_of_code <= thresholds["tiny"]["max_loc"]:
            return JobSize.TINY
        elif lines_of_code <= thresholds["small"]["max_loc"]:
            return JobSize.SMALL
        elif lines_of_code <= thresholds["medium"]["max_loc"]:
            return JobSize.MEDIUM
        elif lines_of_code <= thresholds["large"]["max_loc"]:
            return JobSize.LARGE
        else:
            return JobSize.XLARGE

    def _determine_resources(
        self,
        size: JobSize,
        priority: JobPriority
    ) -> ResourceRequirements:
        """Determine resource requirements for job"""
        thresholds = self.config["job_size_thresholds"][size.value]

        return ResourceRequirements(
            gpus=thresholds["gpu_allocation"],
            cpu_cores=thresholds["cpu_cores"],
            memory_gb=thresholds["memory_gb"],
            timeout_seconds=thresholds.get("max_latency_ms", 60000) // 1000,
            allow_spot=(priority == JobPriority.LOW_PRIORITY)
        )

    def _estimate_cost(self, resources: ResourceRequirements, loc: int) -> float:
        """
        Estimate job cost based on resources and LOC

        Args:
            resources: Resource requirements
            loc: Lines of code

        Returns:
            Estimated cost in USD
        """
        # GPU cost: $3/hour per A100
        gpu_cost_per_hour = 3.0

        # Estimate processing time (100 LOC per second per GPU)
        processing_time_hours = loc / (100 * 60 * 60 * resources.gpus)

        # Calculate cost
        cost = gpu_cost_per_hour * resources.gpus * processing_time_hours

        # Add overhead (20%)
        cost *= 1.2

        # Spot discount (70% cheaper)
        if resources.allow_spot:
            cost *= 0.3

        return round(cost, 2)

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job"""
        return await self.scheduler.get_job_status.remote(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        # Implementation would cancel all tasks and cleanup
        logger.info(f"Cancelling job {job_id}")
        return True

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics"""
        return {
            "num_workers": len(self.workers),
            "ray_nodes": ray.nodes(),
            "available_resources": ray.available_resources(),
            "cluster_resources": ray.cluster_resources()
        }

    def shutdown(self):
        """Shutdown cluster gracefully"""
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False
            logger.info("Ray cluster shutdown")
