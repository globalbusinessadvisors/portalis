//! Strategy execution and graceful GPU → CPU fallback
//!
//! This module provides intelligent execution strategy management that combines:
//! - Hardware capability detection
//! - Workload profiling
//! - System load monitoring
//! - Graceful degradation from GPU to CPU
//! - Hybrid execution across multiple compute resources

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Execution strategy for transpilation workloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Use GPU exclusively (CUDA available, large workload)
    GpuOnly,

    /// Use CPU exclusively (no GPU, or small workload)
    CpuOnly,

    /// Use both GPU and CPU (hybrid workload distribution)
    Hybrid {
        /// Percentage of work allocated to GPU (0-100)
        gpu_allocation: u8,
        /// Percentage of work allocated to CPU (0-100)
        cpu_allocation: u8,
    },

    /// Automatic selection based on runtime conditions
    Auto,
}

impl Default for ExecutionStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

/// Hardware capabilities detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// Number of CPU cores available
    pub cpu_cores: usize,

    /// GPU availability
    pub gpu_available: bool,

    /// GPU memory in bytes (if available)
    pub gpu_memory_bytes: Option<usize>,

    /// GPU compute capability (if available)
    pub gpu_compute_capability: Option<String>,

    /// SIMD support on CPU
    pub cpu_simd_support: bool,

    /// System total memory in bytes
    pub system_memory_bytes: usize,
}

impl HardwareCapabilities {
    /// Detect current hardware capabilities
    pub fn detect() -> Self {
        let cpu_cores = num_cpus::get();
        let cpu_simd_support = Self::detect_simd();
        let system_memory_bytes = Self::detect_system_memory();

        // Check GPU availability
        let (gpu_available, gpu_memory_bytes, gpu_compute_capability) = Self::detect_gpu();

        Self {
            cpu_cores,
            gpu_available,
            gpu_memory_bytes,
            gpu_compute_capability,
            cpu_simd_support,
            system_memory_bytes,
        }
    }

    fn detect_simd() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(target_arch = "aarch64")]
        {
            true // NEON is standard on ARM64
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    fn detect_system_memory() -> usize {
        // Simplified - in production would use sys-info or similar
        #[cfg(target_os = "linux")]
        {
            // Read from /proc/meminfo
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    content.lines().find(|line| line.starts_with("MemTotal:")).and_then(|line| {
                        line.split_whitespace()
                            .nth(1)
                            .and_then(|s| s.parse::<usize>().ok())
                            .map(|kb| kb * 1024)
                    })
                })
                .unwrap_or(8 * 1024 * 1024 * 1024) // Default 8GB
        }
        #[cfg(not(target_os = "linux"))]
        {
            8 * 1024 * 1024 * 1024 // Default 8GB
        }
    }

    fn detect_gpu() -> (bool, Option<usize>, Option<String>) {
        // In production, this would query CUDA runtime
        #[cfg(feature = "cuda")]
        {
            // Would use cuda-sys or similar
            (false, None, None) // Placeholder
        }
        #[cfg(not(feature = "cuda"))]
        {
            (false, None, None)
        }
    }

    /// Check if system has sufficient resources for operation
    pub fn has_sufficient_resources(&self, required_memory_bytes: usize) -> bool {
        self.system_memory_bytes >= required_memory_bytes
    }

    /// Get GPU memory utilization (0.0-1.0)
    pub fn gpu_memory_utilization(&self) -> f64 {
        // Simplified - would query actual GPU in production
        0.0
    }
}

/// Workload profile for strategy selection
#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    /// Number of tasks to execute
    pub task_count: usize,

    /// Estimated memory per task (bytes)
    pub memory_per_task: usize,

    /// Task complexity (0.0-1.0, higher = more complex)
    pub complexity: f64,

    /// Parallelization potential (0.0-1.0, higher = more parallel)
    pub parallelization: f64,
}

impl WorkloadProfile {
    /// Create workload profile from task analysis
    pub fn from_task_count(task_count: usize) -> Self {
        Self {
            task_count,
            memory_per_task: 50 * 1024 * 1024, // 50MB default
            complexity: 0.5,
            parallelization: 0.9,
        }
    }

    /// Estimate total memory requirements
    pub fn estimated_memory_bytes(&self) -> usize {
        self.task_count * self.memory_per_task
    }

    /// Check if workload is small enough for CPU-only
    pub fn is_small_workload(&self) -> bool {
        self.task_count < 10
    }

    /// Check if workload is large enough to benefit from GPU
    pub fn benefits_from_gpu(&self) -> bool {
        self.task_count >= 50 && self.parallelization > 0.7
    }
}

/// System load monitoring
#[derive(Debug, Clone)]
pub struct SystemLoad {
    /// Current CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,

    /// Current GPU utilization (0.0-1.0)
    pub gpu_utilization: f64,

    /// Available memory percentage (0.0-1.0)
    pub available_memory: f64,
}

impl SystemLoad {
    /// Get current system load
    pub fn current() -> Self {
        // Simplified - would use system monitoring in production
        Self {
            cpu_utilization: 0.3,
            gpu_utilization: 0.0,
            available_memory: 0.7,
        }
    }

    /// Check if system is under high load
    pub fn is_high_load(&self) -> bool {
        self.cpu_utilization > 0.8 || self.available_memory < 0.2
    }

    /// Check if GPU is under pressure
    pub fn gpu_memory_pressure(&self) -> f64 {
        1.0 - self.available_memory
    }
}

/// Execution result with metrics
#[derive(Debug, Clone)]
pub struct ExecutionResult<T> {
    /// Execution outputs
    pub outputs: Vec<T>,

    /// Strategy used
    pub strategy_used: ExecutionStrategy,

    /// Total execution time
    pub execution_time: Duration,

    /// Whether fallback occurred
    pub fallback_occurred: bool,

    /// Error messages (if any)
    pub errors: Vec<String>,
}

/// Strategy manager for intelligent execution
pub struct StrategyManager<C, G = NoGpu>
where
    C: CpuExecutor,
    G: GpuExecutor,
{
    /// Hardware capabilities
    capabilities: HardwareCapabilities,

    /// Configured strategy
    strategy: ExecutionStrategy,

    /// CPU bridge (always available)
    cpu_bridge: Arc<C>,

    /// GPU bridge (optional)
    gpu_bridge: Option<Arc<G>>,
}

impl<C> StrategyManager<C, NoGpu>
where
    C: CpuExecutor,
{
    /// Create new strategy manager without GPU support
    pub fn cpu_only(cpu_bridge: Arc<C>) -> Self {
        let capabilities = HardwareCapabilities::detect();
        info!(
            "Hardware detection (CPU-only): {} CPU cores",
            capabilities.cpu_cores
        );

        Self {
            capabilities,
            strategy: ExecutionStrategy::CpuOnly,
            cpu_bridge,
            gpu_bridge: None,
        }
    }
}

impl<C, G> StrategyManager<C, G>
where
    C: CpuExecutor + 'static,
    G: GpuExecutor + 'static,
{
    /// Create new strategy manager with auto-detection
    pub fn new(cpu_bridge: Arc<C>, gpu_bridge: Arc<G>) -> Self {
        let capabilities = HardwareCapabilities::detect();
        info!(
            "Hardware detection: {} CPU cores, GPU available: {}",
            capabilities.cpu_cores, capabilities.gpu_available
        );

        Self {
            capabilities,
            strategy: ExecutionStrategy::Auto,
            cpu_bridge,
            gpu_bridge: Some(gpu_bridge),
        }
    }

    /// Create with explicit strategy
    pub fn with_strategy(
        cpu_bridge: Arc<C>,
        gpu_bridge: Option<Arc<G>>,
        strategy: ExecutionStrategy,
    ) -> Self {
        let capabilities = HardwareCapabilities::detect();

        Self {
            capabilities,
            strategy,
            cpu_bridge,
            gpu_bridge,
        }
    }

    /// Detect optimal execution strategy
    pub fn detect_strategy(&self, workload: &WorkloadProfile) -> ExecutionStrategy {
        match self.strategy {
            ExecutionStrategy::Auto => self.auto_select_strategy(workload),
            other => other,
        }
    }

    /// Auto-select optimal strategy based on conditions
    fn auto_select_strategy(&self, workload: &WorkloadProfile) -> ExecutionStrategy {
        // 1. Check GPU availability
        if !self.capabilities.gpu_available || self.gpu_bridge.is_none() {
            debug!("GPU not available, selecting CPU-only strategy");
            return ExecutionStrategy::CpuOnly;
        }

        // 2. Check workload size - CPU is faster for small tasks (no GPU overhead)
        if workload.is_small_workload() {
            debug!("Small workload detected, CPU-only will be faster");
            return ExecutionStrategy::CpuOnly;
        }

        // 3. Check system load
        let load = SystemLoad::current();
        if load.gpu_memory_pressure() > 0.8 {
            warn!("GPU memory pressure high, using hybrid strategy");
            return ExecutionStrategy::Hybrid {
                gpu_allocation: 60,
                cpu_allocation: 40,
            };
        }

        // 4. Check if workload benefits from GPU
        if workload.benefits_from_gpu() {
            debug!("Large parallel workload, using GPU-only strategy");
            return ExecutionStrategy::GpuOnly;
        }

        // 5. Default to hybrid for balanced workloads
        debug!("Balanced workload, using hybrid strategy");
        ExecutionStrategy::Hybrid {
            gpu_allocation: 70,
            cpu_allocation: 30,
        }
    }

    /// Execute tasks with selected strategy and graceful fallback
    pub fn execute<T, I, F>(
        &self,
        tasks: Vec<I>,
        process_fn: F,
    ) -> Result<ExecutionResult<T>>
    where
        T: Send + Clone + 'static,
        I: Send + Sync + Clone + 'static,
        F: Fn(&I) -> Result<T> + Send + Sync + Clone + 'static,
    {
        let workload = WorkloadProfile::from_task_count(tasks.len());
        let strategy = self.detect_strategy(&workload);

        info!(
            "Executing {} tasks with strategy: {:?}",
            tasks.len(),
            strategy
        );

        let start = Instant::now();
        let mut fallback_occurred = false;
        let mut errors = Vec::new();

        let outputs = match strategy {
            ExecutionStrategy::GpuOnly => {
                // Try GPU first, fallback to CPU on error
                match self.execute_gpu_only(&tasks, &process_fn) {
                    Ok(results) => results,
                    Err(e) => {
                        warn!("GPU execution failed: {}, falling back to CPU", e);
                        fallback_occurred = true;
                        errors.push(format!("GPU error: {}", e));
                        self.execute_cpu_only(&tasks, &process_fn)?
                    }
                }
            }

            ExecutionStrategy::CpuOnly => self.execute_cpu_only(&tasks, &process_fn)?,

            ExecutionStrategy::Hybrid {
                gpu_allocation,
                cpu_allocation,
            } => {
                // Try hybrid, fallback to CPU-only if GPU portion fails
                match self.execute_hybrid(&tasks, &process_fn, gpu_allocation, cpu_allocation) {
                    Ok(results) => results,
                    Err(e) => {
                        warn!("Hybrid execution failed: {}, falling back to CPU-only", e);
                        fallback_occurred = true;
                        errors.push(format!("Hybrid error: {}", e));
                        self.execute_cpu_only(&tasks, &process_fn)?
                    }
                }
            }

            ExecutionStrategy::Auto => {
                // This shouldn't happen as detect_strategy resolves Auto
                unreachable!("Auto strategy should be resolved before execution")
            }
        };

        let execution_time = start.elapsed();

        info!(
            "Execution completed in {:?} (fallback: {})",
            execution_time, fallback_occurred
        );

        Ok(ExecutionResult {
            outputs,
            strategy_used: strategy,
            execution_time,
            fallback_occurred,
            errors,
        })
    }

    /// Execute on GPU only
    fn execute_gpu_only<T, I, F>(
        &self,
        tasks: &[I],
        process_fn: &F,
    ) -> Result<Vec<T>>
    where
        T: Send + 'static,
        I: Send + Sync + 'static,
        F: Fn(&I) -> Result<T> + Send + Sync + 'static,
    {
        let gpu = self
            .gpu_bridge
            .as_ref()
            .ok_or_else(|| anyhow!("GPU bridge not available"))?;

        gpu.execute_batch(tasks, process_fn)
            .context("GPU execution failed")
    }

    /// Execute on CPU only
    fn execute_cpu_only<T, I, F>(
        &self,
        tasks: &[I],
        process_fn: &F,
    ) -> Result<Vec<T>>
    where
        T: Send + 'static,
        I: Send + Sync + 'static,
        F: Fn(&I) -> Result<T> + Send + Sync + 'static,
    {
        self.cpu_bridge
            .execute_batch(tasks, process_fn)
            .context("CPU execution failed")
    }

    /// Execute hybrid (split work between GPU and CPU)
    fn execute_hybrid<T, I, F>(
        &self,
        tasks: &[I],
        process_fn: &F,
        gpu_allocation: u8,
        _cpu_allocation: u8,
    ) -> Result<Vec<T>>
    where
        T: Send + Clone + 'static,
        I: Send + Sync + Clone + 'static,
        F: Fn(&I) -> Result<T> + Send + Sync + Clone + 'static,
    {
        let gpu = self
            .gpu_bridge
            .as_ref()
            .ok_or_else(|| anyhow!("GPU bridge not available for hybrid execution"))?;

        let total = tasks.len();
        let gpu_count = (total as f64 * (gpu_allocation as f64 / 100.0)).ceil() as usize;
        let gpu_count = gpu_count.min(total);

        debug!(
            "Hybrid split: {} tasks to GPU, {} tasks to CPU",
            gpu_count,
            total - gpu_count
        );

        // Split tasks
        let (gpu_tasks, cpu_tasks) = tasks.split_at(gpu_count);

        // Execute in parallel using threads
        use std::thread;

        let gpu_clone = Arc::clone(gpu);
        let cpu_clone = Arc::clone(&self.cpu_bridge);

        let process_fn_gpu = process_fn.clone();
        let process_fn_cpu = process_fn.clone();

        // Convert slices to owned vectors
        let gpu_tasks_vec: Vec<I> = gpu_tasks.to_vec();
        let cpu_tasks_vec: Vec<I> = cpu_tasks.to_vec();

        let gpu_handle = thread::spawn(move || {
            gpu_clone.execute_batch(&gpu_tasks_vec, &process_fn_gpu)
        });

        let cpu_handle = thread::spawn(move || {
            cpu_clone.execute_batch(&cpu_tasks_vec, &process_fn_cpu)
        });

        // Wait for both to complete
        let gpu_results = gpu_handle
            .join()
            .map_err(|_| anyhow!("GPU thread panicked"))??;
        let cpu_results = cpu_handle
            .join()
            .map_err(|_| anyhow!("CPU thread panicked"))??;

        // Combine results
        let mut combined = gpu_results;
        combined.extend(cpu_results);

        Ok(combined)
    }

    /// Get hardware capabilities
    pub fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    /// Get current strategy
    pub fn strategy(&self) -> ExecutionStrategy {
        self.strategy
    }

    /// Set execution strategy
    pub fn set_strategy(&mut self, strategy: ExecutionStrategy) {
        self.strategy = strategy;
    }
}

/// Trait for CPU execution
pub trait CpuExecutor: Send + Sync {
    fn execute_batch<T, I, F>(
        &self,
        tasks: &[I],
        process_fn: &F,
    ) -> Result<Vec<T>>
    where
        T: Send + 'static,
        I: Send + Sync + 'static,
        F: Fn(&I) -> Result<T> + Send + Sync + 'static;
}

/// Trait for GPU execution
pub trait GpuExecutor: Send + Sync {
    fn execute_batch<T, I, F>(
        &self,
        tasks: &[I],
        process_fn: &F,
    ) -> Result<Vec<T>>
    where
        T: Send + 'static,
        I: Send + Sync + 'static,
        F: Fn(&I) -> Result<T> + Send + Sync + 'static;

    fn is_available(&self) -> bool;

    fn memory_available(&self) -> usize;
}

/// Placeholder type for no GPU support
pub struct NoGpu;

impl GpuExecutor for NoGpu {
    fn execute_batch<T, I, F>(
        &self,
        _tasks: &[I],
        _process_fn: &F,
    ) -> Result<Vec<T>>
    where
        T: Send + 'static,
        I: Send + Sync + 'static,
        F: Fn(&I) -> Result<T> + Send + Sync + 'static,
    {
        Err(anyhow!("No GPU available"))
    }

    fn is_available(&self) -> bool {
        false
    }

    fn memory_available(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockCpuExecutor;

    impl CpuExecutor for MockCpuExecutor {
        fn execute_batch<T, I, F>(
            &self,
            tasks: &[I],
            process_fn: &F,
        ) -> Result<Vec<T>>
        where
            T: Send + 'static,
            I: Send + Sync + 'static,
            F: Fn(&I) -> Result<T> + Send + Sync + 'static,
        {
            tasks.iter().map(|task| process_fn(task)).collect()
        }
    }

    struct MockGpuExecutor {
        should_fail: bool,
    }

    impl GpuExecutor for MockGpuExecutor {
        fn execute_batch<T, I, F>(
            &self,
            tasks: &[I],
            process_fn: &F,
        ) -> Result<Vec<T>>
        where
            T: Send + 'static,
            I: Send + Sync + 'static,
            F: Fn(&I) -> Result<T> + Send + Sync + 'static,
        {
            if self.should_fail {
                anyhow::bail!("GPU out of memory");
            }
            tasks.iter().map(|task| process_fn(task)).collect()
        }

        fn is_available(&self) -> bool {
            !self.should_fail
        }

        fn memory_available(&self) -> usize {
            if self.should_fail {
                0
            } else {
                8 * 1024 * 1024 * 1024
            }
        }
    }

    #[test]
    fn test_hardware_detection() {
        let caps = HardwareCapabilities::detect();
        assert!(caps.cpu_cores > 0);
        assert!(caps.system_memory_bytes > 0);
    }

    #[test]
    fn test_workload_profile() {
        let profile = WorkloadProfile::from_task_count(100);
        assert_eq!(profile.task_count, 100);
        assert!(!profile.is_small_workload());
    }

    #[test]
    fn test_cpu_only_execution() {
        let cpu = Arc::new(MockCpuExecutor);
        let manager = StrategyManager::cpu_only(cpu);

        let tasks: Vec<i32> = vec![1, 2, 3, 4, 5];
        let result = manager
            .execute(tasks, |&x| Ok(x * 2))
            .expect("Execution failed");

        assert_eq!(result.outputs, vec![2, 4, 6, 8, 10]);
        assert_eq!(result.strategy_used, ExecutionStrategy::CpuOnly);
        assert!(!result.fallback_occurred);
    }

    #[test]
    fn test_gpu_fallback_to_cpu() {
        let cpu = Arc::new(MockCpuExecutor);
        let gpu = Arc::new(MockGpuExecutor { should_fail: true });

        let manager = StrategyManager::with_strategy(cpu, Some(gpu), ExecutionStrategy::GpuOnly);

        let tasks: Vec<i32> = vec![1, 2, 3];
        let result = manager
            .execute(tasks, |&x| Ok(x * 2))
            .expect("Execution should fallback to CPU");

        assert_eq!(result.outputs, vec![2, 4, 6]);
        assert!(result.fallback_occurred);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_strategy_auto_selection() {
        let cpu = Arc::new(MockCpuExecutor);
        let manager = StrategyManager::cpu_only(cpu);

        // Small workload -> CPU only
        let small_workload = WorkloadProfile::from_task_count(5);
        assert_eq!(
            manager.detect_strategy(&small_workload),
            ExecutionStrategy::CpuOnly
        );
    }
}
