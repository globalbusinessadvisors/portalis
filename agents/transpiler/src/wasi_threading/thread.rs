//! Thread Primitives
//!
//! Provides thread creation, joining, and management across platforms.

use anyhow::{Result, Context, anyhow};
use std::sync::Arc;
use std::time::Duration;
use super::ThreadingError;

/// Thread priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadPriority {
    /// Low priority (background tasks)
    Low,
    /// Normal priority (default)
    Normal,
    /// High priority (time-critical tasks)
    High,
}

/// Thread configuration
#[derive(Debug, Clone)]
pub struct ThreadConfig {
    /// Optional thread name for debugging
    pub name: Option<String>,
    /// Stack size in bytes (None = platform default)
    pub stack_size: Option<usize>,
    /// Thread priority
    pub priority: ThreadPriority,
}

impl ThreadConfig {
    /// Create a new thread configuration with default settings
    pub fn new() -> Self {
        Self {
            name: None,
            stack_size: None,
            priority: ThreadPriority::Normal,
        }
    }

    /// Set the thread name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the stack size
    pub fn with_stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }

    /// Set the thread priority
    pub fn with_priority(mut self, priority: ThreadPriority) -> Self {
        self.priority = priority;
        self
    }
}

impl Default for ThreadConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique thread identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThreadId(u64);

impl ThreadId {
    /// Create a new thread ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the underlying ID value
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ThreadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ThreadId({})", self.0)
    }
}

/// Thread handle for joining and managing spawned threads
pub struct ThreadHandle<T> {
    #[cfg(not(target_arch = "wasm32"))]
    inner: Option<std::thread::JoinHandle<T>>,

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    inner: Arc<parking_lot::Mutex<Option<T>>>,

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    inner: Option<std::thread::JoinHandle<T>>,

    thread_id: ThreadId,
}

impl<T> ThreadHandle<T> {
    /// Get the thread ID
    pub fn thread_id(&self) -> ThreadId {
        self.thread_id
    }

    /// Check if the thread has finished
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_finished(&self) -> bool {
        self.inner.as_ref().map(|h| h.is_finished()).unwrap_or(true)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn is_finished(&self) -> bool {
        // WASM doesn't support checking thread status
        false
    }

    /// Wait for the thread to finish and return its result
    pub fn join(mut self) -> Result<T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let handle = self.inner.take()
                .ok_or_else(|| anyhow!(ThreadingError::Join("Thread already joined".to_string())))?;

            handle.join()
                .map_err(|e| anyhow!(ThreadingError::Panic(format!("{:?}", e))))
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            // Web Workers: poll for result
            let result = self.inner.lock().take()
                .ok_or_else(|| anyhow!(ThreadingError::Join("Thread result not available".to_string())))?;
            Ok(result)
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            let handle = self.inner.take()
                .ok_or_else(|| anyhow!(ThreadingError::Join("Thread already joined".to_string())))?;

            handle.join()
                .map_err(|e| anyhow!(ThreadingError::Panic(format!("{:?}", e))))
        }
    }

    /// Wait for the thread to finish with a timeout
    #[cfg(not(target_arch = "wasm32"))]
    pub fn join_timeout(self, timeout: Duration) -> Result<T>
    where
        T: Send + 'static,
    {
        use std::sync::mpsc;

        let (tx, rx) = mpsc::channel();
        let thread_id = self.thread_id();

        // Spawn a monitoring thread
        std::thread::spawn(move || {
            let result = self.join();
            let _ = tx.send(result);
        });

        rx.recv_timeout(timeout)
            .map_err(|_| anyhow!(ThreadingError::Timeout(format!("Thread {:?} did not finish in time", thread_id))))?
    }

    #[cfg(target_arch = "wasm32")]
    pub fn join_timeout(self, _timeout: Duration) -> Result<T> {
        // Fallback to regular join on WASM
        self.join()
    }
}

/// Main thread API
pub struct WasiThread;

impl WasiThread {
    /// Spawn a new thread with default configuration
    pub fn spawn<F, T>(f: F) -> Result<ThreadHandle<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        Self::spawn_with_config(f, ThreadConfig::default())
    }

    /// Spawn a new thread with custom configuration
    pub fn spawn_with_config<F, T>(f: F, config: ThreadConfig) -> Result<ThreadHandle<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut builder = std::thread::Builder::new();

            if let Some(name) = config.name {
                builder = builder.name(name);
            }

            if let Some(size) = config.stack_size {
                builder = builder.stack_size(size);
            }

            let handle = builder.spawn(f)
                .context("Failed to spawn thread")?;

            let thread_id = ThreadId::new(hash_thread_id(&handle.thread().id()));

            Ok(ThreadHandle {
                inner: Some(handle),
                thread_id,
            })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            // Browser: Use Web Workers (simplified for now)
            let result = Arc::new(parking_lot::Mutex::new(None));

            // In a real implementation, this would use Web Workers API
            // For now, we execute immediately (no true parallelism in browser without Workers)
            *result.lock() = Some(f());

            // Generate a random thread ID
            use std::collections::hash_map::RandomState;
            use std::hash::{BuildHasher, Hash, Hasher};
            let random_state = RandomState::new();
            let mut hasher = random_state.build_hasher();
            std::time::SystemTime::now().hash(&mut hasher);
            let thread_id = ThreadId::new(hasher.finish());

            Ok(ThreadHandle {
                inner: result,
                thread_id,
            })
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // WASI: May support wasi-threads in the future
            let mut builder = std::thread::Builder::new();

            if let Some(name) = config.name {
                builder = builder.name(name);
            }

            if let Some(size) = config.stack_size {
                builder = builder.stack_size(size);
            }

            let handle = builder.spawn(f)
                .context("Failed to spawn thread")?;

            let thread_id = ThreadId::new(hash_thread_id(&handle.thread().id()));

            Ok(ThreadHandle {
                inner: Some(handle),
                thread_id,
            })
        }
    }

    /// Get the current thread's ID
    pub fn current_id() -> ThreadId {
        current_thread_id()
    }

    /// Get the current thread's name
    #[cfg(not(target_arch = "wasm32"))]
    pub fn current_name() -> Option<String> {
        std::thread::current()
            .name()
            .map(|s| s.to_string())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn current_name() -> Option<String> {
        None
    }

    /// Get available parallelism (number of CPUs)
    pub fn available_parallelism() -> usize {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }

        #[cfg(target_arch = "wasm32")]
        {
            1 // WASM is single-threaded by default
        }
    }
}

/// Get the current thread ID
pub fn current_thread_id() -> ThreadId {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let id = std::thread::current().id();
        ThreadId::new(hash_thread_id(&id))
    }

    #[cfg(target_arch = "wasm32")]
    {
        ThreadId::new(1) // Main thread in WASM
    }
}

/// Sleep for the specified duration
pub fn thread_sleep(duration: Duration) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::sleep(duration);
    }

    #[cfg(target_arch = "wasm32")]
    {
        // WASM: Blocking sleep not available, but we can yield
        // In a real implementation, this would use async sleep
        let start = instant::Instant::now();
        while start.elapsed() < duration {
            thread_yield();
        }
    }
}

/// Yield the current thread
pub fn thread_yield() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::yield_now();
    }

    #[cfg(target_arch = "wasm32")]
    {
        // WASM: No true yield, but we can do nothing
    }
}

/// Park the current thread
#[cfg(not(target_arch = "wasm32"))]
pub fn thread_park() {
    std::thread::park();
}

#[cfg(target_arch = "wasm32")]
pub fn thread_park() {
    // WASM: Not supported
}

/// Unpark a thread by its handle
#[cfg(not(target_arch = "wasm32"))]
pub fn thread_unpark<T>(handle: &ThreadHandle<T>) {
    // Note: This is a simplified API
    // In real implementation, we'd need to store Thread objects
}

#[cfg(target_arch = "wasm32")]
pub fn thread_unpark<T>(_handle: &ThreadHandle<T>) {
    // WASM: Not supported
}

/// Hash a thread ID to u64
#[cfg(not(target_arch = "wasm32"))]
fn hash_thread_id(id: &std::thread::ThreadId) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
}

/// Thread-local storage (native only)
#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! thread_local {
    ($name:ident: $ty:ty = $init:expr) => {
        thread_local!(static $name: std::cell::RefCell<$ty> = std::cell::RefCell::new($init));
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_thread_spawn() {
        let handle = WasiThread::spawn(|| {
            42
        }).unwrap();

        let result = handle.join().unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_thread_config() {
        let config = ThreadConfig::new()
            .with_name("test-thread")
            .with_stack_size(2 * 1024 * 1024)
            .with_priority(ThreadPriority::High);

        let handle = WasiThread::spawn_with_config(|| {
            WasiThread::current_name()
        }, config).unwrap();

        let name = handle.join().unwrap();
        assert_eq!(name.as_deref(), Some("test-thread"));
    }

    #[test]
    fn test_thread_id() {
        let id1 = current_thread_id();
        let id2 = current_thread_id();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_sleep() {
        let start = std::time::Instant::now();
        thread_sleep(Duration::from_millis(50));
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(50));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_available_parallelism() {
        let count = WasiThread::available_parallelism();
        assert!(count >= 1);
    }
}
