//! WASI platform async runtime implementation
//!
//! Uses tokio runtime for WASI WASM execution (wasm32-wasi target).

use anyhow::Result;
use std::future::Future;
use std::time::Duration;
use tokio::task::JoinHandle;
use crate::wasi_async_runtime::AsyncError;

/// Block on a future using the tokio runtime
///
/// WASI supports tokio, so we can use the same approach as native
pub fn block_on<F, T>(future: F) -> T
where
    F: Future<Output = T>,
{
    // Create a new current-thread runtime for WASI
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to create WASI tokio runtime");

    rt.block_on(future)
}

/// Spawn an async task in WASI
pub fn spawn<F, T>(future: F) -> JoinHandle<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    tokio::spawn(future)
}

/// Spawn a local async task in WASI
pub fn spawn_local<F, T>(future: F) -> JoinHandle<T>
where
    F: Future<Output = T> + 'static,
    T: 'static,
{
    tokio::task::spawn_local(future)
}

/// WASI-compatible sleep implementation
pub async fn sleep(duration: Duration) {
    tokio::time::sleep(duration).await
}

/// WASI-compatible timeout implementation
pub async fn timeout<F, T>(duration: Duration, future: F) -> Result<T, AsyncError>
where
    F: Future<Output = T>,
{
    tokio::time::timeout(duration, future)
        .await
        .map_err(|_| AsyncError::Timeout(duration))
}

/// WASI-compatible yield implementation
pub async fn yield_now() {
    tokio::task::yield_now().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[cfg(target_arch = "wasm32")]
    async fn test_sleep() {
        let start = instant::Instant::now();
        sleep(Duration::from_millis(100)).await;
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(100));
    }

    #[tokio::test]
    #[cfg(target_arch = "wasm32")]
    async fn test_spawn() {
        let handle = spawn(async {
            42
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_block_on() {
        let result = block_on(async {
            sleep(Duration::from_millis(10)).await;
            42
        });

        assert_eq!(result, 42);
    }

    #[tokio::test]
    #[cfg(target_arch = "wasm32")]
    async fn test_timeout_success() {
        let result = timeout(Duration::from_secs(1), async {
            sleep(Duration::from_millis(100)).await;
            42
        }).await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    #[cfg(target_arch = "wasm32")]
    async fn test_timeout_failure() {
        let result = timeout(Duration::from_millis(100), async {
            sleep(Duration::from_secs(1)).await;
            42
        }).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    #[cfg(target_arch = "wasm32")]
    async fn test_yield_now() {
        yield_now().await;
        assert!(true);
    }
}
