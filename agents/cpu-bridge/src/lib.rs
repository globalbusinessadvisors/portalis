//! # Portalis CPU Bridge
//!
//! CPU-based parallel processing bridge for the Portalis transpilation platform.
//!
//! This crate provides efficient multi-core CPU parallelization as an alternative
//! or complement to GPU acceleration. It leverages Rayon's work-stealing thread pool,
//! optimized data structures, and hardware-aware configuration to maximize throughput
//! across all CPU architectures.
//!
//! ## Features
//!
//! - **Automatic CPU Detection**: Auto-configures thread count and batch sizes
//! - **Thread Pool Management**: Efficient work-stealing scheduler via Rayon
//! - **Performance Monitoring**: Real-time metrics tracking and profiling
//! - **Thread Safety**: Lock-free designs with minimal synchronization overhead
//! - **Platform Agnostic**: Works on x86_64, ARM64, and other architectures
//!
//! ## Example
//!
//! ```rust,no_run
//! use portalis_cpu_bridge::{CpuBridge, CpuConfig};
//!
//! // Auto-detect optimal configuration
//! let bridge = CpuBridge::new();
//!
//! // Or use custom configuration
//! let config = CpuConfig::builder()
//!     .num_threads(8)
//!     .batch_size(32)
//!     .build();
//! let bridge = CpuBridge::with_config(config);
//!
//! // Execute parallel translation tasks
//! // let results = bridge.parallel_translate(tasks)?;
//! ```

pub mod config;
pub mod metrics;
pub mod simd;
pub mod thread_pool;

#[cfg(feature = "memory-opt")]
pub mod arena;

use anyhow::Result;
use parking_lot::RwLock;
use std::sync::Arc;

pub use config::{CpuConfig, CpuConfigBuilder, ThreadPoolConfig};
pub use metrics::{CpuMetrics, ThreadPoolMetrics};
pub use simd::{
    CpuCapabilities, batch_string_contains, detect_cpu_capabilities, parallel_string_match,
    vectorized_char_count,
};
pub use thread_pool::{ManagedThreadPool, create_thread_pool, parallel_execute};

#[cfg(feature = "memory-opt")]
pub use arena::{Arena, ArenaPool, ArenaStats, PooledArena};

#[cfg(feature = "memory-opt")]
pub use metrics::MemoryMetrics;

/// Core CPU bridge for parallel processing operations.
///
/// The `CpuBridge` manages a thread pool and coordinates parallel execution
/// of translation tasks across multiple CPU cores. It provides both batch
/// and single-task execution modes with comprehensive performance tracking.
///
/// # Thread Safety
///
/// This struct is thread-safe and can be shared across threads using `Arc`.
/// Internal state is protected by `RwLock` for concurrent access.
///
/// # Performance
///
/// The bridge is designed for high throughput with minimal overhead:
/// - Work-stealing scheduler for optimal load balancing
/// - Cache-friendly data structures to minimize memory latency
/// - Lock-free operations where possible
/// - Adaptive batch sizing based on workload characteristics
pub struct CpuBridge {
    /// Thread pool for parallel execution
    thread_pool: rayon::ThreadPool,

    /// Configuration settings
    config: CpuConfig,

    /// Performance metrics (thread-safe)
    metrics: Arc<RwLock<CpuMetrics>>,
}

impl CpuBridge {
    /// Creates a new CPU bridge with auto-detected optimal settings.
    ///
    /// This constructor automatically detects the number of available CPU cores
    /// and configures the thread pool for optimal performance on the current hardware.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use portalis_cpu_bridge::CpuBridge;
    ///
    /// let bridge = CpuBridge::new();
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Thread count is set to the number of logical CPU cores
    /// - Batch size is optimized for L2/L3 cache sizes
    /// - Stack size is set to 2MB per thread for deep recursion support
    pub fn new() -> Self {
        let config = CpuConfig::auto_detect();
        Self::with_config(config)
    }

    /// Creates a new CPU bridge with explicit configuration.
    ///
    /// Use this constructor when you need fine-grained control over thread pool
    /// settings, batch sizes, or other performance parameters.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom CPU configuration
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use portalis_cpu_bridge::{CpuBridge, CpuConfig};
    ///
    /// let config = CpuConfig::builder()
    ///     .num_threads(16)
    ///     .batch_size(64)
    ///     .enable_simd(true)
    ///     .build();
    ///
    /// let bridge = CpuBridge::with_config(config);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if thread pool creation fails (e.g., invalid configuration).
    pub fn with_config(config: CpuConfig) -> Self {
        use rayon::ThreadPoolBuilder;

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(config.num_threads())
            .stack_size(config.stack_size())
            .thread_name(|i| format!("portalis-cpu-{}", i))
            .build()
            .expect("Failed to create thread pool");

        Self {
            thread_pool,
            config,
            metrics: Arc::new(RwLock::new(CpuMetrics::new())),
        }
    }

    /// Executes parallel translation tasks across multiple CPU cores.
    ///
    /// This method distributes tasks across the thread pool using a work-stealing
    /// scheduler for optimal load balancing. Tasks are executed in parallel with
    /// automatic batching for cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `tasks` - Vector of translation tasks to execute
    ///
    /// # Returns
    ///
    /// A `Result` containing the vector of translation outputs, or an error if
    /// any task fails.
    ///
    /// # Performance
    ///
    /// - Scales linearly up to the number of configured threads
    /// - Optimal for batch sizes >= number of CPU cores
    /// - Minimal overhead for task distribution (~microseconds)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use portalis_cpu_bridge::CpuBridge;
    ///
    /// let bridge = CpuBridge::new();
    /// // let tasks = vec![task1, task2, task3];
    /// // let results = bridge.parallel_translate(tasks)?;
    /// ```
    pub fn parallel_translate<T, O, F>(
        &self,
        tasks: Vec<T>,
        translate_fn: F,
    ) -> Result<Vec<O>>
    where
        T: Send + Sync,
        O: Send,
        F: Fn(&T) -> Result<O> + Send + Sync,
    {
        use rayon::prelude::*;
        use std::time::Instant;

        let start = Instant::now();
        let num_tasks = tasks.len();

        // Execute tasks in parallel using Rayon
        let results: Result<Vec<O>> = self.thread_pool.install(|| {
            tasks
                .par_iter()
                .map(|task| translate_fn(task))
                .collect()
        });

        // Update metrics
        let elapsed = start.elapsed();
        let mut metrics = self.metrics.write();
        metrics.record_batch(num_tasks, elapsed);

        results
    }

    /// Executes a single translation task (optimized for latency).
    ///
    /// This method is optimized for low-latency execution of individual tasks.
    /// Unlike `parallel_translate`, it avoids thread pool overhead for single
    /// operations.
    ///
    /// # Arguments
    ///
    /// * `task` - Single translation task to execute
    ///
    /// # Returns
    ///
    /// A `Result` containing the translation output, or an error if the task fails.
    ///
    /// # Performance
    ///
    /// - Sub-millisecond overhead for small tasks
    /// - No thread pool synchronization required
    /// - Suitable for interactive/real-time use cases
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use portalis_cpu_bridge::CpuBridge;
    ///
    /// let bridge = CpuBridge::new();
    /// // let task = create_task();
    /// // let result = bridge.translate_single(task, |t| process(t))?;
    /// ```
    pub fn translate_single<T, O, F>(
        &self,
        task: T,
        translate_fn: F,
    ) -> Result<O>
    where
        F: FnOnce(&T) -> Result<O>,
    {
        use std::time::Instant;

        let start = Instant::now();
        let result = translate_fn(&task);
        let elapsed = start.elapsed();

        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.record_single(elapsed);

        result
    }

    /// Returns a snapshot of current performance metrics.
    ///
    /// This method provides real-time visibility into CPU bridge performance,
    /// including task counts, execution times, CPU utilization, and memory usage.
    ///
    /// # Returns
    ///
    /// A clone of the current `CpuMetrics` snapshot.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use portalis_cpu_bridge::CpuBridge;
    ///
    /// let bridge = CpuBridge::new();
    /// // ... perform operations ...
    /// let metrics = bridge.metrics();
    /// println!("Tasks completed: {}", metrics.tasks_completed());
    /// println!("Avg task time: {:.2}ms", metrics.avg_task_time_ms());
    /// ```
    pub fn metrics(&self) -> CpuMetrics {
        self.metrics.read().clone()
    }

    /// Returns a reference to the CPU configuration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use portalis_cpu_bridge::CpuBridge;
    ///
    /// let bridge = CpuBridge::new();
    /// let config = bridge.config();
    /// println!("Using {} threads", config.num_threads());
    /// ```
    pub fn config(&self) -> &CpuConfig {
        &self.config
    }

    /// Returns the number of threads in the thread pool.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use portalis_cpu_bridge::CpuBridge;
    ///
    /// let bridge = CpuBridge::new();
    /// println!("Thread pool size: {}", bridge.num_threads());
    /// ```
    pub fn num_threads(&self) -> usize {
        self.thread_pool.current_num_threads()
    }
}

impl Default for CpuBridge {
    fn default() -> Self {
        Self::new()
    }
}

// Ensure CpuBridge is Send + Sync for concurrent usage
unsafe impl Send for CpuBridge {}
unsafe impl Sync for CpuBridge {}

// Implement CpuExecutor trait from portalis-core for StrategyManager integration
#[cfg(feature = "acceleration")]
impl portalis_core::acceleration::CpuExecutor for CpuBridge {
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
        use rayon::prelude::*;
        use std::time::Instant;

        let start = Instant::now();
        let num_tasks = tasks.len();

        // Execute tasks in parallel using Rayon
        let results: Result<Vec<T>> = self.thread_pool.install(|| {
            tasks
                .par_iter()
                .map(|task| process_fn(task))
                .collect()
        });

        // Update metrics
        let elapsed = start.elapsed();
        let mut metrics = self.metrics.write();
        metrics.record_batch(num_tasks, elapsed);

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_bridge_creation() {
        let bridge = CpuBridge::new();
        assert!(bridge.num_threads() > 0);
    }

    #[test]
    fn test_cpu_bridge_with_config() {
        let config = CpuConfig::builder()
            .num_threads(4)
            .batch_size(16)
            .build();

        let bridge = CpuBridge::with_config(config);
        assert_eq!(bridge.num_threads(), 4);
    }

    #[test]
    fn test_parallel_translate() {
        let bridge = CpuBridge::new();

        let tasks: Vec<i32> = (0..100).collect();
        let results = bridge
            .parallel_translate(tasks, |&x| Ok(x * 2))
            .expect("Translation failed");

        assert_eq!(results.len(), 100);
        assert_eq!(results[0], 0);
        assert_eq!(results[99], 198);
    }

    #[test]
    fn test_translate_single() {
        let bridge = CpuBridge::new();

        let result = bridge
            .translate_single(42, |&x| Ok(x * 2))
            .expect("Translation failed");

        assert_eq!(result, 84);
    }

    #[test]
    fn test_metrics_tracking() {
        let bridge = CpuBridge::new();

        let tasks: Vec<i32> = (0..10).collect();
        let _ = bridge.parallel_translate(tasks, |&x| Ok(x * 2));

        let metrics = bridge.metrics();
        assert!(metrics.tasks_completed() > 0);
    }
}
