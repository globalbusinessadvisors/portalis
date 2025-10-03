"""
Monitoring and Metrics Collection for Portalis DGX Cloud
Real-time metrics, Prometheus export, and Grafana dashboards
"""

import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from loguru import logger
import psutil
import ray


class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class GPUMetrics:
    """GPU metrics snapshot"""
    gpu_id: int
    utilization: float  # 0.0-1.0
    memory_used_mb: float
    memory_total_mb: float
    memory_utilization: float  # 0.0-1.0
    temperature_c: float
    power_draw_w: float
    compute_mode: str
    processes: int


@dataclass
class ClusterMetrics:
    """Cluster-wide metrics"""
    timestamp: datetime
    num_nodes: int
    num_workers: int
    active_jobs: int
    queued_jobs: int
    completed_jobs: int
    failed_jobs: int

    # Resource metrics
    total_gpus: int
    active_gpus: int
    gpu_utilization_avg: float
    cpu_utilization_avg: float
    memory_utilization_avg: float

    # Performance metrics
    avg_job_latency_ms: float
    throughput_jobs_per_hour: float
    error_rate: float

    # Cost metrics
    hourly_cost: float
    cost_per_job: float


class PrometheusMetrics:
    """
    Prometheus metrics exporter
    Exposes metrics for scraping by Prometheus
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus metrics

        Args:
            registry: Optional custom registry
        """
        self.registry = registry or CollectorRegistry()

        # Job metrics
        self.jobs_submitted = Counter(
            "portalis_jobs_submitted_total",
            "Total number of jobs submitted",
            ["tenant_id", "priority"],
            registry=self.registry
        )

        self.jobs_completed = Counter(
            "portalis_jobs_completed_total",
            "Total number of jobs completed",
            ["tenant_id", "status"],
            registry=self.registry
        )

        self.active_jobs = Gauge(
            "portalis_jobs_active",
            "Number of currently active jobs",
            ["priority"],
            registry=self.registry
        )

        self.queued_jobs = Gauge(
            "portalis_jobs_queued",
            "Number of jobs in queue",
            ["priority"],
            registry=self.registry
        )

        # Latency metrics
        self.job_latency = Histogram(
            "portalis_job_latency_seconds",
            "Job processing latency",
            ["job_size", "priority"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0],
            registry=self.registry
        )

        self.translation_latency = Histogram(
            "portalis_translation_latency_seconds",
            "Single translation latency",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry
        )

        # Resource metrics
        self.gpu_utilization = Gauge(
            "portalis_gpu_utilization",
            "GPU utilization percentage",
            ["gpu_id", "node_id"],
            registry=self.registry
        )

        self.gpu_memory_used = Gauge(
            "portalis_gpu_memory_used_bytes",
            "GPU memory used in bytes",
            ["gpu_id", "node_id"],
            registry=self.registry
        )

        self.gpu_memory_total = Gauge(
            "portalis_gpu_memory_total_bytes",
            "Total GPU memory in bytes",
            ["gpu_id", "node_id"],
            registry=self.registry
        )

        self.gpu_temperature = Gauge(
            "portalis_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["gpu_id", "node_id"],
            registry=self.registry
        )

        self.cpu_utilization = Gauge(
            "portalis_cpu_utilization",
            "CPU utilization percentage",
            ["node_id"],
            registry=self.registry
        )

        self.memory_utilization = Gauge(
            "portalis_memory_utilization",
            "System memory utilization percentage",
            ["node_id"],
            registry=self.registry
        )

        # Worker metrics
        self.worker_count = Gauge(
            "portalis_workers_total",
            "Total number of workers",
            ["status"],
            registry=self.registry
        )

        self.worker_tasks = Gauge(
            "portalis_worker_tasks_active",
            "Active tasks per worker",
            ["worker_id"],
            registry=self.registry
        )

        # Cost metrics
        self.cost_total = Counter(
            "portalis_cost_usd_total",
            "Total cost in USD",
            ["tenant_id", "instance_type"],
            registry=self.registry
        )

        self.cost_hourly = Gauge(
            "portalis_cost_hourly_usd",
            "Current hourly cost in USD",
            registry=self.registry
        )

        # Error metrics
        self.errors_total = Counter(
            "portalis_errors_total",
            "Total number of errors",
            ["error_type", "component"],
            registry=self.registry
        )

        self.error_rate = Gauge(
            "portalis_error_rate",
            "Error rate (errors per minute)",
            registry=self.registry
        )

        # Cache metrics
        self.cache_hits = Counter(
            "portalis_cache_hits_total",
            "Total cache hits",
            ["cache_type"],
            registry=self.registry
        )

        self.cache_misses = Counter(
            "portalis_cache_misses_total",
            "Total cache misses",
            ["cache_type"],
            registry=self.registry
        )

        logger.info("PrometheusMetrics initialized")

    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(self.registry)


class GPUMonitor:
    """
    GPU monitoring using NVIDIA Management Library (NVML)
    """

    def __init__(self):
        """Initialize GPU monitor"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.enabled = True
            logger.info(f"GPU monitoring enabled ({self.gpu_count} GPUs)")

        except Exception as e:
            logger.warning(f"GPU monitoring disabled: {e}")
            self.nvml = None
            self.enabled = False
            self.gpu_count = 0

    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """
        Get current GPU metrics

        Returns:
            List of GPU metrics
        """
        if not self.enabled:
            return []

        metrics = []
        for i in range(self.gpu_count):
            try:
                handle = self.nvml.nvmlDeviceGetHandleByIndex(i)

                # Get utilization
                util = self.nvml.nvmlDeviceGetUtilizationRates(handle)

                # Get memory info
                mem = self.nvml.nvmlDeviceGetMemoryInfo(handle)

                # Get temperature
                temp = self.nvml.nvmlDeviceGetTemperature(
                    handle,
                    self.nvml.NVML_TEMPERATURE_GPU
                )

                # Get power
                power = self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W

                # Get compute mode
                mode = self.nvml.nvmlDeviceGetComputeMode(handle)
                mode_str = {
                    0: "Default",
                    1: "Exclusive_Thread",
                    2: "Prohibited",
                    3: "Exclusive_Process"
                }.get(mode, "Unknown")

                # Get running processes
                processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)

                metrics.append(GPUMetrics(
                    gpu_id=i,
                    utilization=util.gpu / 100.0,
                    memory_used_mb=mem.used / (1024 * 1024),
                    memory_total_mb=mem.total / (1024 * 1024),
                    memory_utilization=mem.used / mem.total,
                    temperature_c=temp,
                    power_draw_w=power,
                    compute_mode=mode_str,
                    processes=len(processes)
                ))

            except Exception as e:
                logger.error(f"Failed to get metrics for GPU {i}: {e}")

        return metrics


class MetricsCollector:
    """
    Main metrics collector
    Aggregates metrics from multiple sources and exports to Prometheus
    """

    def __init__(self, collection_interval: int = 15):
        """
        Initialize metrics collector

        Args:
            collection_interval: Metrics collection interval in seconds
        """
        self.collection_interval = collection_interval
        self.prometheus = PrometheusMetrics()
        self.gpu_monitor = GPUMonitor()

        # State tracking
        self.start_time = datetime.utcnow()
        self.jobs_processed = 0
        self.errors_count = 0
        self.last_error_time = None

        # Running flag
        self.running = False

        logger.info("MetricsCollector initialized")

    async def start_collection(self):
        """Start metrics collection loop"""
        self.running = True
        logger.info("Starting metrics collection")

        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)

    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        logger.info("Stopped metrics collection")

    async def _collect_metrics(self):
        """Collect all metrics"""
        # GPU metrics
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        for gpu in gpu_metrics:
            node_id = os.getenv("NODE_ID", "local")

            self.prometheus.gpu_utilization.labels(
                gpu_id=str(gpu.gpu_id),
                node_id=node_id
            ).set(gpu.utilization)

            self.prometheus.gpu_memory_used.labels(
                gpu_id=str(gpu.gpu_id),
                node_id=node_id
            ).set(gpu.memory_used_mb * 1024 * 1024)

            self.prometheus.gpu_memory_total.labels(
                gpu_id=str(gpu.gpu_id),
                node_id=node_id
            ).set(gpu.memory_total_mb * 1024 * 1024)

            self.prometheus.gpu_temperature.labels(
                gpu_id=str(gpu.gpu_id),
                node_id=node_id
            ).set(gpu.temperature_c)

        # CPU and memory metrics
        node_id = os.getenv("NODE_ID", "local")
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        self.prometheus.cpu_utilization.labels(node_id=node_id).set(cpu_percent)
        self.prometheus.memory_utilization.labels(node_id=node_id).set(memory_percent)

        # Ray cluster metrics (if available)
        try:
            if ray.is_initialized():
                resources = ray.available_resources()
                cluster_resources = ray.cluster_resources()

                # Update worker count
                num_cpus = cluster_resources.get("CPU", 0)
                num_gpus = cluster_resources.get("GPU", 0)

                logger.debug(
                    f"Cluster resources: {num_cpus} CPUs, {num_gpus} GPUs"
                )

        except Exception as e:
            logger.debug(f"Failed to get Ray metrics: {e}")

    def record_job_submitted(self, tenant_id: str, priority: str):
        """Record job submission"""
        self.prometheus.jobs_submitted.labels(
            tenant_id=tenant_id,
            priority=priority
        ).inc()

    def record_job_completed(self, tenant_id: str, status: str, latency: float):
        """Record job completion"""
        self.prometheus.jobs_completed.labels(
            tenant_id=tenant_id,
            status=status
        ).inc()

        self.jobs_processed += 1

    def record_translation_latency(self, latency: float):
        """Record translation latency"""
        self.prometheus.translation_latency.observe(latency)

    def record_error(self, error_type: str, component: str):
        """Record error"""
        self.prometheus.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()

        self.errors_count += 1
        self.last_error_time = datetime.utcnow()

        # Update error rate
        if self.last_error_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds() / 60
            if duration > 0:
                self.prometheus.error_rate.set(self.errors_count / duration)

    def record_cache_access(self, cache_type: str, hit: bool):
        """Record cache access"""
        if hit:
            self.prometheus.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.prometheus.cache_misses.labels(cache_type=cache_type).inc()

    def record_cost(self, tenant_id: str, instance_type: str, cost: float):
        """Record cost"""
        self.prometheus.cost_total.labels(
            tenant_id=tenant_id,
            instance_type=instance_type
        ).add(cost)

    def get_cluster_metrics(self) -> ClusterMetrics:
        """
        Get current cluster metrics

        Returns:
            Cluster metrics snapshot
        """
        # Get GPU metrics
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        avg_gpu_util = (
            sum(g.utilization for g in gpu_metrics) / len(gpu_metrics)
            if gpu_metrics else 0.0
        )

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0)
        memory_percent = psutil.virtual_memory().percent

        # Calculate throughput
        uptime_hours = (
            (datetime.utcnow() - self.start_time).total_seconds() / 3600
        )
        throughput = self.jobs_processed / uptime_hours if uptime_hours > 0 else 0.0

        return ClusterMetrics(
            timestamp=datetime.utcnow(),
            num_nodes=1,  # Would be fetched from Ray
            num_workers=0,  # Would be fetched from Ray
            active_jobs=0,  # Would be tracked
            queued_jobs=0,  # Would be tracked
            completed_jobs=self.jobs_processed,
            failed_jobs=self.errors_count,
            total_gpus=self.gpu_monitor.gpu_count,
            active_gpus=len([g for g in gpu_metrics if g.utilization > 0.1]),
            gpu_utilization_avg=avg_gpu_util,
            cpu_utilization_avg=cpu_percent / 100.0,
            memory_utilization_avg=memory_percent / 100.0,
            avg_job_latency_ms=0.0,  # Would be calculated
            throughput_jobs_per_hour=throughput,
            error_rate=0.0,  # Would be calculated
            hourly_cost=0.0,  # Would be fetched from cost tracker
            cost_per_job=0.0  # Would be calculated
        )
