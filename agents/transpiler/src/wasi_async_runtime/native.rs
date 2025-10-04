//! Native platform async runtime implementation
//!
//! Uses tokio runtime for native Rust execution.

use anyhow::Result;
use std::future::Future;
use once_cell::sync::OnceCell;
use tokio::runtime::Runtime;

/// Global tokio runtime instance
static RUNTIME: OnceCell<Runtime> = OnceCell::new();

/// Initialize the tokio runtime with default configuration
pub fn init() -> Result<()> {
    RUNTIME.get_or_try_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(num_cpus::get())
            .thread_name("portalis-async-worker")
            .build()
    })?;
    Ok(())
}

/// Get or create the runtime
fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(num_cpus::get())
            .thread_name("portalis-async-worker")
            .build()
            .expect("Failed to create tokio runtime")
    })
}

/// Block on a future using the tokio runtime
pub fn block_on<F, T>(future: F) -> T
where
    F: Future<Output = T>,
{
    runtime().block_on(future)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_init() {
        assert!(init().is_ok());
        assert!(init().is_ok()); // Should be idempotent
    }

    #[test]
    fn test_block_on() {
        let result = block_on(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            42
        });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_runtime_reuse() {
        let runtime1 = runtime();
        let runtime2 = runtime();
        assert!(std::ptr::eq(runtime1, runtime2));
    }
}
