//! WASI Async Runtime Wrapper
//!
//! Provides a unified async runtime API that works across:
//! - Native Rust (tokio runtime)
//! - Browser WASM (wasm-bindgen-futures)
//! - WASI WASM (tokio on wasi or async-std)
//!
//! This module bridges Python's asyncio operations to WASM-compatible async Rust implementations.
//!
//! # Examples
//!
//! ## Running async code
//! ```rust,no_run
//! use wasi_async_runtime::{AsyncRuntime, sleep};
//! use std::time::Duration;
//!
//! // Initialize runtime (done automatically on first use)
//! AsyncRuntime::block_on(async {
//!     println!("Starting async operation");
//!     sleep(Duration::from_secs(1)).await;
//!     println!("Done!");
//! });
//! ```
//!
//! ## Spawning tasks
//! ```rust,no_run
//! use wasi_async_runtime::{spawn, TaskHandle};
//!
//! async fn example() {
//!     let handle = spawn(async {
//!         // Do some async work
//!         42
//!     });
//!
//!     let result = handle.await.unwrap();
//!     assert_eq!(result, 42);
//! }
//! ```
//!
//! ## Timeout operations
//! ```rust,no_run
//! use wasi_async_runtime::{timeout, sleep};
//! use std::time::Duration;
//!
//! async fn example() {
//!     match timeout(Duration::from_secs(5), sleep(Duration::from_secs(10))).await {
//!         Ok(_) => println!("Completed in time"),
//!         Err(_) => println!("Timed out!"),
//!     }
//! }
//! ```

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

// Platform-specific implementations
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub(crate) mod browser;

#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
pub(crate) mod wasi_impl;

/// Async runtime error types
#[derive(Debug, thiserror::Error)]
pub enum AsyncError {
    #[error("Task join error: {0}")]
    JoinError(String),

    #[error("Task cancelled: {0}")]
    Cancelled(String),

    #[error("Timeout error: operation took longer than {0:?}")]
    Timeout(Duration),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Spawn error: {0}")]
    Spawn(String),

    #[error("Task panic: {0}")]
    Panic(String),

    #[error("Platform not supported: {0}")]
    PlatformNotSupported(String),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<String> for AsyncError {
    fn from(s: String) -> Self {
        AsyncError::Other(s)
    }
}

impl From<&str> for AsyncError {
    fn from(s: &str) -> Self {
        AsyncError::Other(s.to_string())
    }
}

/// Task handle for spawned async tasks
///
/// This is a cross-platform wrapper around platform-specific join handles.
pub struct TaskHandle<T> {
    #[cfg(not(target_arch = "wasm32"))]
    inner: tokio::task::JoinHandle<T>,

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    inner: Pin<Box<dyn Future<Output = T> + 'static>>,

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    inner: tokio::task::JoinHandle<T>,
}

impl<T> TaskHandle<T> {
    /// Check if the task is finished
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn is_finished(&self) -> bool {
        // WASM doesn't support checking task state
        false
    }

    /// Abort the task
    #[cfg(not(target_arch = "wasm32"))]
    pub fn abort(&self) {
        self.inner.abort()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn abort(&self) {
        // WASM tasks cannot be aborted after spawning
    }
}

impl<T> Future for TaskHandle<T> {
    type Output = Result<T, AsyncError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            match Pin::new(&mut self.inner).poll(cx) {
                std::task::Poll::Ready(Ok(result)) => std::task::Poll::Ready(Ok(result)),
                std::task::Poll::Ready(Err(e)) => {
                    if e.is_panic() {
                        std::task::Poll::Ready(Err(AsyncError::Panic(format!("{:?}", e))))
                    } else if e.is_cancelled() {
                        std::task::Poll::Ready(Err(AsyncError::Cancelled("Task was cancelled".to_string())))
                    } else {
                        std::task::Poll::Ready(Err(AsyncError::JoinError(e.to_string())))
                    }
                }
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            match Pin::new(&mut self.inner).poll(cx) {
                std::task::Poll::Ready(result) => std::task::Poll::Ready(Ok(result)),
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            match Pin::new(&mut self.inner).poll(cx) {
                std::task::Poll::Ready(Ok(result)) => std::task::Poll::Ready(Ok(result)),
                std::task::Poll::Ready(Err(e)) => {
                    if e.is_panic() {
                        std::task::Poll::Ready(Err(AsyncError::Panic(format!("{:?}", e))))
                    } else {
                        std::task::Poll::Ready(Err(AsyncError::JoinError(e.to_string())))
                    }
                }
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        }
    }
}

/// Main async runtime interface
pub struct AsyncRuntime;

impl AsyncRuntime {
    /// Block on a future and run it to completion
    ///
    /// This is the main entry point for running async code.
    /// On native platforms, this uses tokio::runtime::Runtime.
    /// On WASM, this spawns the future into the browser's event loop.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        native::block_on(future)
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T> + 'static,
        T: 'static,
    {
        browser::block_on(future)
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    pub fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        wasi_impl::block_on(future)
    }

    /// Initialize the runtime with custom configuration
    ///
    /// On native platforms, this creates a tokio runtime with specific settings.
    /// On WASM, this is a no-op as the runtime is provided by the environment.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn init() -> Result<()> {
        native::init()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn init() -> Result<()> {
        // WASM runtime is initialized by the environment
        Ok(())
    }
}

/// Spawn an async task
///
/// # Platform-specific behavior
/// - Native: Uses tokio::spawn
/// - Browser: Uses wasm-bindgen-futures::spawn_local
/// - WASI: Uses tokio::spawn (if available) or falls back to spawn_local
pub fn spawn<F, T>(future: F) -> TaskHandle<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        TaskHandle {
            inner: tokio::spawn(future),
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    {
        TaskHandle {
            inner: Box::pin(browser::spawn(future)),
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    {
        TaskHandle {
            inner: wasi_impl::spawn(future),
        }
    }
}

/// Spawn a blocking task (runs on a dedicated thread pool)
///
/// This is useful for CPU-intensive operations that would block the async runtime.
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_blocking<F, T>(f: F) -> TaskHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    TaskHandle {
        inner: tokio::task::spawn_blocking(f),
    }
}

#[cfg(target_arch = "wasm32")]
pub fn spawn_blocking<F, T>(f: F) -> TaskHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // WASM is single-threaded, so just run it directly
    let result = f();
    spawn(async move { result })
}

/// Spawn a local async task (does not require Send)
///
/// This is useful on WASM where tasks don't need to be Send.
pub fn spawn_local<F, T>(future: F) -> TaskHandle<T>
where
    F: Future<Output = T> + 'static,
    T: 'static,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        TaskHandle {
            inner: tokio::task::spawn_local(future),
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    {
        TaskHandle {
            inner: Box::pin(browser::spawn(future)),
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    {
        TaskHandle {
            inner: wasi_impl::spawn_local(future),
        }
    }
}

/// Sleep for the specified duration
///
/// This is a platform-aware async sleep implementation.
pub async fn sleep(duration: Duration) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        tokio::time::sleep(duration).await;
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    {
        browser::sleep(duration).await;
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    {
        wasi_impl::sleep(duration).await;
    }
}

/// Run a future with a timeout
///
/// Returns an error if the future doesn't complete within the specified duration.
pub async fn timeout<F, T>(duration: Duration, future: F) -> Result<T, AsyncError>
where
    F: Future<Output = T>,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        tokio::time::timeout(duration, future)
            .await
            .map_err(|_| AsyncError::Timeout(duration))
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    {
        browser::timeout(duration, future).await
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    {
        wasi_impl::timeout(duration, future).await
    }
}

/// Yield control back to the async runtime
///
/// This allows other tasks to run.
pub async fn yield_now() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        tokio::task::yield_now().await;
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    {
        browser::yield_now().await;
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    {
        wasi_impl::yield_now().await;
    }
}

/// Helper to convert Python asyncio.run() patterns
///
/// This is the main entry point for Python code that uses asyncio.run()
pub fn asyncio_run<F, T>(future: F) -> T
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    AsyncRuntime::block_on(future)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_error_types() {
        let err = AsyncError::Timeout(Duration::from_secs(5));
        assert!(err.to_string().contains("5s"));

        let err = AsyncError::JoinError("task failed".to_string());
        assert!(err.to_string().contains("task failed"));
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_spawn_task() {
        let handle = spawn(async {
            42
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_sleep() {
        let start = std::time::Instant::now();
        sleep(Duration::from_millis(100)).await;
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(100));
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_timeout_success() {
        let result = timeout(Duration::from_secs(1), async {
            sleep(Duration::from_millis(100)).await;
            42
        }).await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_timeout_failure() {
        let result = timeout(Duration::from_millis(100), async {
            sleep(Duration::from_secs(1)).await;
            42
        }).await;

        assert!(result.is_err());
        match result {
            Err(AsyncError::Timeout(_)) => (),
            _ => panic!("Expected timeout error"),
        }
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_yield_now() {
        yield_now().await;
        // If we get here, yield worked
        assert!(true);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_block_on() {
        let result = AsyncRuntime::block_on(async {
            sleep(Duration::from_millis(10)).await;
            42
        });

        assert_eq!(result, 42);
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_spawn_blocking() {
        let handle = spawn_blocking(|| {
            // Simulate blocking work
            std::thread::sleep(Duration::from_millis(10));
            42
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_multiple_tasks() {
        let handle1 = spawn(async {
            sleep(Duration::from_millis(50)).await;
            1
        });

        let handle2 = spawn(async {
            sleep(Duration::from_millis(50)).await;
            2
        });

        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();

        assert_eq!(result1 + result2, 3);
    }
}
