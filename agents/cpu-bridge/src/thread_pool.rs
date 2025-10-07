//! Thread pool management using Rayon
//!
//! This module provides a high-performance thread pool wrapper around Rayon,
//! optimized for translation workloads with work-stealing scheduler integration.

use crate::config::{CpuConfig, ThreadPoolConfig};
use crate::metrics::ThreadPoolMetrics;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;

/// Create a Rayon thread pool with the given configuration
pub fn create_thread_pool(config: &CpuConfig) -> Result<rayon::ThreadPool> {
    ThreadPoolBuilder::new()
        .num_threads(config.num_threads())
        .stack_size(config.stack_size())
        .thread_name(|idx| format!("portalis-cpu-{}", idx))
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create thread pool: {}", e))
}

/// A managed thread pool for parallel translation tasks
///
/// This wrapper provides:
/// - Rayon work-stealing scheduler for optimal load balancing
/// - Configurable thread count (defaults to num_cpus)
/// - Performance metrics and monitoring
/// - Graceful shutdown and resource cleanup
/// - Error propagation from worker threads
pub struct ManagedThreadPool {
    /// Rayon thread pool for work-stealing parallelism
    pool: rayon::ThreadPool,

    /// Configuration used to create this pool
    config: ThreadPoolConfig,

    /// Performance metrics (thread-safe)
    metrics: Arc<RwLock<ThreadPoolMetrics>>,
}

impl ManagedThreadPool {
    /// Create a new thread pool with the given configuration
    ///
    /// # Arguments
    /// * `config` - Thread pool configuration settings
    ///
    /// # Returns
    /// Result containing the managed thread pool or an error
    pub fn new(config: ThreadPoolConfig) -> Result<Self> {
        let num_threads = config.num_threads;
        let stack_size = config.stack_size;

        // Build Rayon thread pool with custom configuration
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .stack_size(stack_size)
            .thread_name(|i| format!("portalis-cpu-worker-{}", i))
            .build()
            .context("Failed to create Rayon thread pool")?;

        log::info!(
            "Created CPU thread pool with {} threads (stack size: {} bytes)",
            num_threads,
            stack_size
        );

        Ok(Self {
            pool,
            config,
            metrics: Arc::new(RwLock::new(ThreadPoolMetrics::default())),
        })
    }

    /// Create a thread pool with default configuration
    ///
    /// Uses auto-detected optimal settings:
    /// - Thread count: number of logical CPU cores
    /// - Stack size: 2MB per thread
    pub fn with_defaults() -> Result<Self> {
        Self::new(ThreadPoolConfig::default())
    }

    /// Execute a batch of tasks in parallel using work-stealing
    ///
    /// This method distributes work across all available threads using
    /// Rayon's work-stealing scheduler for optimal load balancing.
    ///
    /// # Arguments
    /// * `tasks` - Vector of tasks to execute in parallel
    /// * `f` - Function to execute for each task
    ///
    /// # Returns
    /// Result containing vector of outputs or first error encountered
    ///
    /// # Type Parameters
    /// * `T` - Input task type (must be Send + Sync)
    /// * `R` - Output result type (must be Send)
    /// * `F` - Task function type
    pub fn execute_batch<T, R, F>(&self, tasks: Vec<T>, f: F) -> Result<Vec<R>>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(&T) -> Result<R> + Send + Sync,
    {
        use rayon::prelude::*;

        let start_time = std::time::Instant::now();
        let task_count = tasks.len();

        // Execute tasks in parallel with work-stealing
        let results: Result<Vec<R>> = self.pool.install(|| {
            tasks
                .par_iter()
                .map(|task| f(task))
                .collect()
        });

        // Update metrics
        let elapsed = start_time.elapsed();
        let mut metrics = self.metrics.write();
        metrics.record_batch(task_count, elapsed);

        results.context("Error occurred during parallel batch execution")
    }

    /// Execute parallel translation batch following the plan pattern (lines 196-207)
    ///
    /// This is a convenience method that implements the pattern from the architecture plan:
    /// ```rust,ignore
    /// files
    ///     .par_iter()
    ///     .map(|file| self.translate_file(file))
    ///     .collect()
    /// ```
    ///
    /// # Arguments
    /// * `items` - Items to process in parallel
    /// * `processor` - Function to process each item
    ///
    /// # Returns
    /// Result containing processed items or first error
    pub fn parallel_translate_batch<T, R, F>(&self, items: Vec<T>, processor: F) -> Result<Vec<R>>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(&T) -> Result<R> + Send + Sync,
    {
        use rayon::prelude::*;

        let start_time = std::time::Instant::now();
        let item_count = items.len();

        let results = self.pool.install(|| {
            items
                .par_iter()
                .map(|item| processor(item))
                .collect()
        });

        // Update metrics
        let elapsed = start_time.elapsed();
        let mut metrics = self.metrics.write();
        metrics.record_batch(item_count, elapsed);

        results
    }

    /// Execute a single task on the thread pool
    ///
    /// For single tasks, this has minimal overhead but still benefits
    /// from the initialized thread pool.
    ///
    /// # Arguments
    /// * `f` - Function to execute
    ///
    /// # Returns
    /// Result from the task execution
    pub fn execute_single<R, F>(&self, f: F) -> Result<R>
    where
        R: Send,
        F: FnOnce() -> Result<R> + Send,
    {
        let start_time = std::time::Instant::now();

        let result = self.pool.install(|| f());

        // Update metrics
        let elapsed = start_time.elapsed();
        let mut metrics = self.metrics.write();
        metrics.record_single(elapsed);

        result.context("Error occurred during single task execution")
    }

    /// Execute parallel work using Rayon's parallel iterator
    ///
    /// This is the lowest-level API that gives direct access to Rayon's
    /// parallel iterators within the managed thread pool context.
    ///
    /// # Arguments
    /// * `f` - Closure that receives access to the thread pool scope
    pub fn scope<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&rayon::Scope) -> R + Send,
        R: Send,
    {
        self.pool.scope(op)
    }

    /// Get a reference to the underlying Rayon thread pool
    ///
    /// This allows advanced users to directly interact with Rayon's APIs
    /// while still using the managed pool.
    pub fn rayon_pool(&self) -> &rayon::ThreadPool {
        &self.pool
    }

    /// Get the current configuration
    pub fn config(&self) -> &ThreadPoolConfig {
        &self.config
    }

    /// Get current performance metrics
    ///
    /// Returns a snapshot of the thread pool's performance metrics.
    pub fn metrics(&self) -> ThreadPoolMetrics {
        self.metrics.read().clone()
    }

    /// Get the number of threads in the pool
    pub fn num_threads(&self) -> usize {
        self.config.num_threads
    }

    /// Check if the thread pool is currently executing tasks
    ///
    /// Note: This is a best-effort check and may not be 100% accurate
    /// in highly concurrent scenarios.
    pub fn is_active(&self) -> bool {
        self.metrics.read().active_tasks > 0
    }

    /// Reset performance metrics
    ///
    /// Useful for benchmarking or when starting a new workload phase.
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.write();
        *metrics = ThreadPoolMetrics::default();
    }

    /// Gracefully shutdown the thread pool
    ///
    /// This method ensures all running tasks complete before shutdown.
    /// After calling this, the thread pool cannot be used again.
    ///
    /// Note: Rayon thread pools automatically clean up when dropped,
    /// so explicit shutdown is optional. This method is provided for
    /// logging and metrics finalization.
    pub fn shutdown(self) -> ThreadPoolMetrics {
        let metrics = self.metrics.read().clone();

        log::info!(
            "Shutting down CPU thread pool: {} tasks completed, avg time: {:.2}ms",
            metrics.total_tasks,
            metrics.avg_task_time_ms()
        );

        // Pool will be dropped here, triggering cleanup
        metrics
    }
}

impl Default for ManagedThreadPool {
    fn default() -> Self {
        Self::with_defaults()
            .expect("Failed to create default thread pool")
    }
}

/// Convenience function for parallel batch processing without managing a pool
///
/// This creates a temporary thread pool, executes the batch, and cleans up.
/// For repeated operations, create a persistent `ManagedThreadPool` instead.
///
/// # Arguments
/// * `tasks` - Vector of tasks to execute
/// * `f` - Function to execute for each task
///
/// # Returns
/// Result containing vector of outputs or first error encountered
pub fn parallel_execute<T, R, F>(tasks: Vec<T>, f: F) -> Result<Vec<R>>
where
    T: Send + Sync,
    R: Send,
    F: Fn(&T) -> Result<R> + Send + Sync,
{
    let pool = ManagedThreadPool::with_defaults()?;
    pool.execute_batch(tasks, f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CpuConfig;

    #[test]
    fn test_create_thread_pool() {
        let config = CpuConfig::builder().num_threads(4).build();
        let pool = create_thread_pool(&config);
        assert!(pool.is_ok());
    }

    #[test]
    fn test_managed_thread_pool_creation() {
        let config = ThreadPoolConfig::default();
        let pool = ManagedThreadPool::new(config);
        assert!(pool.is_ok());
    }

    #[test]
    fn test_parallel_batch_execution() {
        let pool = ManagedThreadPool::with_defaults().unwrap();
        let tasks = vec![1, 2, 3, 4, 5];

        let results = pool.execute_batch(tasks, |&x| Ok(x * 2));
        assert!(results.is_ok());
        assert_eq!(results.unwrap(), vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_translate_batch() {
        let pool = ManagedThreadPool::with_defaults().unwrap();
        let items = vec!["file1", "file2", "file3"];

        let results = pool.parallel_translate_batch(items, |&item| {
            Ok(format!("processed_{}", item))
        });

        assert!(results.is_ok());
        let processed = results.unwrap();
        assert_eq!(processed.len(), 3);
        assert!(processed.contains(&"processed_file1".to_string()));
    }

    #[test]
    fn test_error_propagation() {
        let pool = ManagedThreadPool::with_defaults().unwrap();
        let tasks = vec![1, 2, 3, 4, 5];

        let results = pool.execute_batch(tasks, |&x| {
            if x == 3 {
                Err(anyhow::anyhow!("Test error"))
            } else {
                Ok(x * 2)
            }
        });

        assert!(results.is_err());
    }

    #[test]
    fn test_single_task_execution() {
        let pool = ManagedThreadPool::with_defaults().unwrap();
        let result = pool.execute_single(|| Ok(42));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_metrics_tracking() {
        let pool = ManagedThreadPool::with_defaults().unwrap();
        let tasks = vec![1, 2, 3, 4, 5];

        pool.execute_batch(tasks, |&x| Ok(x * 2)).unwrap();

        let metrics = pool.metrics();
        assert_eq!(metrics.total_batches, 1);
        assert_eq!(metrics.total_tasks, 5);
    }

    #[test]
    fn test_metrics_reset() {
        let pool = ManagedThreadPool::with_defaults().unwrap();
        let tasks = vec![1, 2, 3];

        pool.execute_batch(tasks, |&x| Ok(x * 2)).unwrap();
        assert_eq!(pool.metrics().total_tasks, 3);

        pool.reset_metrics();
        assert_eq!(pool.metrics().total_tasks, 0);
    }

    #[test]
    fn test_convenience_function() {
        let tasks = vec![10, 20, 30];
        let results = parallel_execute(tasks, |&x| Ok(x / 10));
        assert_eq!(results.unwrap(), vec![1, 2, 3]);
    }
}
