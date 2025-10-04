//! WASI Threading and Synchronization Primitives
//!
//! Provides a unified threading API that works across:
//! - Native Rust (std::thread + rayon + parking_lot)
//! - Browser WASM (Web Workers via wasm-bindgen)
//! - WASI WASM (wasi-threads or compatibility layer)
//!
//! This module bridges Python's threading operations to WASM-compatible implementations.
//!
//! # Examples
//!
//! ## Basic Threading
//! ```rust,no_run
//! use wasi_threading::{WasiThread, ThreadConfig};
//!
//! // Spawn a new thread
//! let handle = WasiThread::spawn(|| {
//!     println!("Hello from thread!");
//!     42
//! })?;
//!
//! // Wait for completion and get result
//! let result = handle.join()?;
//! assert_eq!(result, 42);
//! ```
//!
//! ## Synchronization
//! ```rust,no_run
//! use wasi_threading::{WasiMutex, WasiRwLock};
//!
//! // Mutex for exclusive access
//! let mutex = WasiMutex::new(0);
//! {
//!     let mut guard = mutex.lock();
//!     *guard += 1;
//! }
//!
//! // RwLock for concurrent reads
//! let rwlock = WasiRwLock::new(vec![1, 2, 3]);
//! {
//!     let read_guard = rwlock.read();
//!     println!("Length: {}", read_guard.len());
//! }
//! ```
//!
//! ## Thread Pool
//! ```rust,no_run
//! use wasi_threading::ThreadPool;
//!
//! let pool = ThreadPool::new(4)?;
//! pool.execute(|| {
//!     println!("Task running in pool");
//! });
//! ```

use anyhow::{Result, Context, anyhow};
use std::sync::Arc;
use std::time::Duration;

// Re-export platform-specific modules
pub mod thread;
pub mod sync;
pub mod collections;
pub mod pool;

// Platform-specific implementations
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub(crate) mod browser;

#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
pub(crate) mod wasi_impl;

// Re-export main types
pub use thread::{
    WasiThread, ThreadConfig, ThreadHandle, ThreadId, ThreadPriority,
    thread_sleep, thread_yield, thread_park, thread_unpark,
};

pub use sync::{
    WasiMutex, WasiRwLock, WasiSemaphore, WasiCondvar, WasiBarrier, WasiEvent,
    MutexGuard, RwLockReadGuard, RwLockWriteGuard,
};

pub use collections::{
    WasiQueue, WasiStack, WasiPriorityQueue, WasiDeque,
};

pub use pool::{
    ThreadPool, ThreadPoolConfig, ThreadPoolBuilder, WorkResult,
};

/// Threading error types
#[derive(Debug, thiserror::Error)]
pub enum ThreadingError {
    #[error("Thread spawn error: {0}")]
    Spawn(String),

    #[error("Thread join error: {0}")]
    Join(String),

    #[error("Lock poisoned: {0}")]
    Poisoned(String),

    #[error("Deadlock detected: {0}")]
    Deadlock(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Channel closed: {0}")]
    ChannelClosed(String),

    #[error("Thread panic: {0}")]
    Panic(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Platform not supported: {0}")]
    PlatformNotSupported(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}

/// Thread-local storage key
#[cfg(not(target_arch = "wasm32"))]
pub struct ThreadLocal<T: Send + 'static> {
    inner: std::thread::LocalKey<std::cell::RefCell<Option<T>>>,
}

/// Thread builder for advanced configuration
pub struct ThreadBuilder {
    name: Option<String>,
    stack_size: Option<usize>,
    priority: ThreadPriority,
}

impl ThreadBuilder {
    /// Create a new thread builder
    pub fn new() -> Self {
        Self {
            name: None,
            stack_size: None,
            priority: ThreadPriority::Normal,
        }
    }

    /// Set the thread name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the stack size in bytes
    pub fn stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }

    /// Set the thread priority
    pub fn priority(mut self, priority: ThreadPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Spawn a thread with the configured settings
    pub fn spawn<F, T>(self, f: F) -> Result<ThreadHandle<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let config = ThreadConfig {
            name: self.name,
            stack_size: self.stack_size,
            priority: self.priority,
        };
        WasiThread::spawn_with_config(f, config)
    }
}

impl Default for ThreadBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Global thread count (for debugging)
#[cfg(not(target_arch = "wasm32"))]
pub fn active_thread_count() -> usize {
    // Platform-specific implementation
    #[cfg(target_os = "linux")]
    {
        // On Linux, count threads via /proc
        std::fs::read_dir(format!("/proc/{}/task", std::process::id()))
            .ok()
            .and_then(|entries| Some(entries.count()))
            .unwrap_or(1)
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Fallback: not available on all platforms
        1
    }
}

/// Get the current thread ID
pub fn current_thread_id() -> ThreadId {
    thread::current_thread_id()
}

/// Sleep for the specified duration
pub fn sleep(duration: Duration) {
    thread_sleep(duration)
}

/// Yield the current thread
pub fn yield_now() {
    thread_yield()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_builder() {
        let builder = ThreadBuilder::new()
            .name("test-thread")
            .stack_size(1024 * 1024)
            .priority(ThreadPriority::Normal);

        assert_eq!(builder.name.as_ref().unwrap(), "test-thread");
        assert_eq!(builder.stack_size, Some(1024 * 1024));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_current_thread_id() {
        let id1 = current_thread_id();
        let id2 = current_thread_id();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_sleep() {
        let start = std::time::Instant::now();
        sleep(Duration::from_millis(10));
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }
}
