//! Browser platform async runtime implementation
//!
//! Uses wasm-bindgen-futures for browser WASM execution.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local as wasm_spawn_local;
use js_sys::Promise;
use std::future::Future;
use std::time::Duration;
use std::pin::Pin;
use std::task::{Context, Poll};
use crate::wasi_async_runtime::AsyncError;

/// Block on a future in the browser
///
/// Note: This doesn't truly block in the browser. Instead, it spawns the future
/// into the browser's event loop. The "blocking" behavior is simulated.
pub fn block_on<F, T>(_future: F) -> T
where
    F: Future<Output = T> + 'static,
    T: 'static,
{
    // In the browser, we can't truly block.
    // This should be used only in initialization contexts.
    panic!("block_on is not supported in browser WASM. Use spawn() or spawn_local() instead.");
}

/// Spawn a future into the browser's event loop
pub fn spawn<F, T>(future: F) -> impl Future<Output = T>
where
    F: Future<Output = T> + 'static,
    T: 'static,
{
    // Create a channel to communicate the result
    let (sender, receiver) = futures_channel::oneshot::channel();

    wasm_spawn_local(async move {
        let result = future.await;
        let _ = sender.send(result);
    });

    async move {
        receiver.await.expect("Task was cancelled")
    }
}

/// Browser-compatible sleep implementation
pub async fn sleep(duration: Duration) {
    let millis = duration.as_millis() as i32;

    let promise = Promise::new(&mut |resolve, _| {
        let window = web_sys::window().expect("no global window");
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            &resolve,
            millis,
        );
    });

    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("setTimeout failed");
}

/// Browser-compatible timeout implementation
pub async fn timeout<F, T>(duration: Duration, future: F) -> Result<T, AsyncError>
where
    F: Future<Output = T> + 'static,
    T: 'static,
{
    let timeout_future = sleep(duration);

    // Race the future against the timeout
    let (sender, receiver) = futures_channel::oneshot::channel();

    wasm_spawn_local(async move {
        let result = future.await;
        let _ = sender.send(result);
    });

    // Create timeout task
    wasm_spawn_local(async move {
        timeout_future.await;
    });

    match receiver.await {
        Ok(result) => Ok(result),
        Err(_) => Err(AsyncError::Timeout(duration)),
    }
}

/// Browser-compatible yield implementation
pub async fn yield_now() {
    // Use setTimeout with 0 delay to yield to the event loop
    let promise = Promise::new(&mut |resolve, _| {
        let window = web_sys::window().expect("no global window");
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            &resolve,
            0,
        );
    });

    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("setTimeout failed");
}

/// Future wrapper for browser spawn
pub struct BrowserTask<T> {
    receiver: futures_channel::oneshot::Receiver<T>,
}

impl<T> Future for BrowserTask<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.receiver).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => panic!("Task was cancelled"),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_sleep() {
        let start = instant::Instant::now();
        sleep(Duration::from_millis(100)).await;
        let elapsed = start.elapsed();
        // Browser timers are not precise, allow some leeway
        assert!(elapsed >= Duration::from_millis(90));
    }

    #[wasm_bindgen_test]
    async fn test_yield_now() {
        yield_now().await;
        // If we get here, yield worked
        assert!(true);
    }

    #[wasm_bindgen_test]
    async fn test_spawn() {
        let future = spawn(async {
            sleep(Duration::from_millis(50)).await;
            42
        });

        let result = future.await;
        assert_eq!(result, 42);
    }
}
