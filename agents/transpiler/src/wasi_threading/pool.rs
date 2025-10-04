//! Thread Pool Implementation
//!
//! Provides fixed and dynamic thread pools with work stealing and task execution.

use anyhow::{Result, Context, anyhow};
use std::sync::Arc;
use std::time::Duration;
use super::{WasiQueue, ThreadingError, WasiThread, ThreadHandle};

#[cfg(not(target_arch = "wasm32"))]
use rayon;

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads (None = number of CPUs)
    pub num_threads: Option<usize>,
    /// Thread name prefix
    pub thread_name_prefix: Option<String>,
    /// Stack size per thread
    pub stack_size: Option<usize>,
    /// Maximum pending tasks (None = unbounded)
    pub max_pending_tasks: Option<usize>,
    /// Enable work stealing (rayon-based)
    pub enable_work_stealing: bool,
}

impl ThreadPoolConfig {
    /// Create a new thread pool configuration
    pub fn new() -> Self {
        Self {
            num_threads: None,
            thread_name_prefix: Some("worker".to_string()),
            stack_size: None,
            max_pending_tasks: None,
            enable_work_stealing: true,
        }
    }

    /// Set the number of worker threads
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = Some(n);
        self
    }

    /// Set the thread name prefix
    pub fn thread_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.thread_name_prefix = Some(prefix.into());
        self
    }

    /// Set the stack size per thread
    pub fn stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }

    /// Set the maximum pending tasks
    pub fn max_pending_tasks(mut self, max: usize) -> Self {
        self.max_pending_tasks = Some(max);
        self
    }

    /// Enable or disable work stealing
    pub fn enable_work_stealing(mut self, enabled: bool) -> Self {
        self.enable_work_stealing = enabled;
        self
    }
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread pool builder
pub struct ThreadPoolBuilder {
    config: ThreadPoolConfig,
}

impl ThreadPoolBuilder {
    /// Create a new thread pool builder
    pub fn new() -> Self {
        Self {
            config: ThreadPoolConfig::default(),
        }
    }

    /// Set the number of worker threads
    pub fn num_threads(mut self, n: usize) -> Self {
        self.config.num_threads = Some(n);
        self
    }

    /// Set the thread name prefix
    pub fn thread_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.thread_name_prefix = Some(prefix.into());
        self
    }

    /// Set the stack size per thread
    pub fn stack_size(mut self, size: usize) -> Self {
        self.config.stack_size = Some(size);
        self
    }

    /// Set the maximum pending tasks
    pub fn max_pending_tasks(mut self, max: usize) -> Self {
        self.config.max_pending_tasks = Some(max);
        self
    }

    /// Enable or disable work stealing
    pub fn enable_work_stealing(mut self, enabled: bool) -> Self {
        self.config.enable_work_stealing = enabled;
        self
    }

    /// Build the thread pool
    pub fn build(self) -> Result<ThreadPool> {
        ThreadPool::with_config(self.config)
    }
}

impl Default for ThreadPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Work item type
type WorkItem = Box<dyn FnOnce() + Send + 'static>;

/// Result of work execution
pub struct WorkResult<T> {
    handle: ThreadHandle<T>,
}

impl<T> WorkResult<T> {
    /// Wait for the work to complete and get the result
    pub fn wait(self) -> Result<T> {
        self.handle.join()
    }

    /// Wait for the work with a timeout
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_timeout(self, timeout: Duration) -> Result<T>
    where
        T: Send + 'static,
    {
        self.handle.join_timeout(timeout)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn wait_timeout(self, _timeout: Duration) -> Result<T> {
        self.wait()
    }
}

/// Thread pool for executing tasks
pub struct ThreadPool {
    #[cfg(not(target_arch = "wasm32"))]
    task_queue: Option<Arc<WasiQueue<WorkItem>>>,

    #[cfg(not(target_arch = "wasm32"))]
    workers: Vec<ThreadHandle<()>>,

    #[cfg(not(target_arch = "wasm32"))]
    rayon_pool: Option<Arc<rayon::ThreadPool>>,

    #[cfg(target_arch = "wasm32")]
    task_queue: Arc<WasiQueue<WorkItem>>,

    config: ThreadPoolConfig,
    is_shutdown: Arc<parking_lot::Mutex<bool>>,
}

impl ThreadPool {
    /// Create a new thread pool with default configuration
    pub fn new(num_threads: usize) -> Result<Self> {
        let config = ThreadPoolConfig::new().num_threads(num_threads);
        Self::with_config(config)
    }

    /// Create a thread pool with custom configuration
    pub fn with_config(config: ThreadPoolConfig) -> Result<Self> {
        let num_threads = config.num_threads.unwrap_or_else(|| {
            #[cfg(not(target_arch = "wasm32"))]
            {
                WasiThread::available_parallelism()
            }
            #[cfg(target_arch = "wasm32")]
            {
                1
            }
        });

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Use rayon for work stealing if enabled
            if config.enable_work_stealing {
                let thread_name_prefix = config.thread_name_prefix.clone().unwrap_or_else(|| "worker".to_string());

                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .thread_name(move |i| {
                        format!("{}-{}", thread_name_prefix, i)
                    })
                    .build()
                    .context("Failed to create rayon thread pool")?;

                Ok(Self {
                    task_queue: None,
                    workers: Vec::new(),
                    rayon_pool: Some(Arc::new(pool)),
                    config,
                    is_shutdown: Arc::new(parking_lot::Mutex::new(false)),
                })
            } else {
                // Use manual thread pool
                let task_queue: Arc<WasiQueue<WorkItem>> = if let Some(max) = config.max_pending_tasks {
                    Arc::new(WasiQueue::with_capacity(max))
                } else {
                    Arc::new(WasiQueue::new())
                };

                let mut workers = Vec::with_capacity(num_threads);
                let is_shutdown = Arc::new(parking_lot::Mutex::new(false));

                for i in 0..num_threads {
                    let queue_clone = task_queue.clone();
                    let shutdown_clone = is_shutdown.clone();
                    let name = format!("{}-{}", config.thread_name_prefix.as_deref().unwrap_or("worker"), i);

                    let thread_config = super::ThreadConfig::new().with_name(name);

                    let handle = WasiThread::spawn_with_config(move || {
                        loop {
                            // Check for shutdown
                            if *shutdown_clone.lock() {
                                break;
                            }

                            // Try to get a task
                            if let Some(task) = queue_clone.try_pop() {
                                task();
                            } else {
                                // No task available, sleep briefly
                                super::thread_sleep(Duration::from_millis(10));
                            }
                        }
                    }, thread_config)?;

                    workers.push(handle);
                }

                Ok(Self {
                    task_queue: Some(task_queue),
                    workers,
                    rayon_pool: None,
                    config,
                    is_shutdown,
                })
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            // WASM: Simple queue-based pool (no true parallelism)
            let task_queue = if let Some(max) = config.max_pending_tasks {
                Arc::new(WasiQueue::with_capacity(max))
            } else {
                Arc::new(WasiQueue::new())
            };

            Ok(Self {
                task_queue,
                config,
                is_shutdown: Arc::new(parking_lot::Mutex::new(false)),
            })
        }
    }

    /// Execute a task in the thread pool (fire-and-forget)
    pub fn execute<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if *self.is_shutdown.lock() {
            return Err(anyhow!(ThreadingError::InvalidOperation("Thread pool is shut down".to_string())));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ref rayon_pool) = self.rayon_pool {
                rayon_pool.spawn(f);
                Ok(())
            } else if let Some(ref queue) = self.task_queue {
                queue.push(Box::new(f))
            } else {
                Err(anyhow!(ThreadingError::InvalidOperation("Thread pool not properly initialized".to_string())))
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.task_queue.push(Box::new(f))
        }
    }

    /// Submit a task that returns a value
    pub fn submit<F, T>(&self, f: F) -> Result<WorkResult<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        if *self.is_shutdown.lock() {
            return Err(anyhow!(ThreadingError::InvalidOperation("Thread pool is shut down".to_string())));
        }

        // Spawn in a separate thread for now
        // In a production implementation, this would use the pool's threads
        let handle = WasiThread::spawn(f)?;
        Ok(WorkResult { handle })
    }

    /// Execute work in parallel using rayon (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn parallel_for_each<T, F>(&self, items: Vec<T>, f: F) -> Result<()>
    where
        T: Send,
        F: Fn(T) + Send + Sync,
    {
        if let Some(ref pool) = self.rayon_pool {
            pool.install(|| {
                rayon::scope(|s| {
                    for item in items {
                        s.spawn(|_| f(item));
                    }
                });
            });
            Ok(())
        } else {
            Err(anyhow!(ThreadingError::InvalidOperation("Parallel operations require work stealing enabled".to_string())))
        }
    }

    /// Map operation in parallel (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn parallel_map<T, R, F>(&self, items: Vec<T>, f: F) -> Result<Vec<R>>
    where
        T: Send,
        R: Send,
        F: Fn(T) -> R + Send + Sync,
    {
        if let Some(ref pool) = self.rayon_pool {
            Ok(pool.install(|| {
                use rayon::prelude::*;
                items.into_par_iter().map(f).collect()
            }))
        } else {
            Err(anyhow!(ThreadingError::InvalidOperation("Parallel operations require work stealing enabled".to_string())))
        }
    }

    /// Get the number of worker threads
    pub fn num_threads(&self) -> usize {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ref pool) = self.rayon_pool {
                pool.current_num_threads()
            } else {
                self.workers.len()
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.config.num_threads.unwrap_or(1)
        }
    }

    /// Get pending task count
    pub fn pending_tasks(&self) -> usize {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ref queue) = self.task_queue {
                queue.len()
            } else {
                0
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.task_queue.len()
        }
    }

    /// Shutdown the thread pool gracefully
    pub fn shutdown(self) -> Result<()> {
        *self.is_shutdown.lock() = true;

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Wait for all workers to finish
            for worker in self.workers {
                worker.join()?;
            }
        }

        Ok(())
    }

    /// Shutdown and wait with timeout
    #[cfg(not(target_arch = "wasm32"))]
    pub fn shutdown_timeout(self, timeout: Duration) -> Result<()> {
        *self.is_shutdown.lock() = true;

        let start = std::time::Instant::now();
        for worker in self.workers {
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return Err(anyhow!(ThreadingError::Timeout("Thread pool shutdown timed out".to_string())));
            }
            worker.join_timeout(remaining)?;
        }

        Ok(())
    }
}

/// Global thread pool for convenience
static GLOBAL_POOL: once_cell::sync::Lazy<Result<ThreadPool>> = once_cell::sync::Lazy::new(|| {
    ThreadPool::new(WasiThread::available_parallelism())
});

/// Execute a task on the global thread pool
pub fn spawn<F>(f: F) -> Result<()>
where
    F: FnOnce() + Send + 'static,
{
    GLOBAL_POOL.as_ref()
        .map_err(|e| anyhow!("Global thread pool not available: {}", e))?
        .execute(f)
}

/// Submit a task on the global thread pool
pub fn submit<F, T>(f: F) -> Result<WorkResult<T>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    GLOBAL_POOL.as_ref()
        .map_err(|e| anyhow!("Global thread pool not available: {}", e))?
        .submit(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_thread_pool_basic() {
        let pool = ThreadPool::new(4).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..10 {
            let counter_clone = counter.clone();
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }).unwrap();
        }

        // Give threads time to execute
        std::thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_thread_pool_submit() {
        let pool = ThreadPool::new(2).unwrap();

        let result = pool.submit(|| {
            42
        }).unwrap();

        assert_eq!(result.wait().unwrap(), 42);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_thread_pool_rayon() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(4)
            .enable_work_stealing(true)
            .build()
            .unwrap();

        let items = vec![1, 2, 3, 4, 5];
        let results = pool.parallel_map(items, |x| x * 2).unwrap();

        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_thread_pool_config() {
        let config = ThreadPoolConfig::new()
            .num_threads(8)
            .thread_name_prefix("test")
            .max_pending_tasks(100);

        assert_eq!(config.num_threads, Some(8));
        assert_eq!(config.thread_name_prefix.as_deref(), Some("test"));
        assert_eq!(config.max_pending_tasks, Some(100));
    }
}
