//! Browser Platform Implementation
//!
//! Browser implementation using Web Workers for parallelism.
//! Note: True threading in browsers requires Web Workers, which have limitations.

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use web_sys::{Worker, MessageEvent, ErrorEvent};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use std::sync::Arc;

/// Browser-based worker pool (simplified)
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub struct BrowserWorkerPool {
    workers: Vec<Worker>,
    size: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl BrowserWorkerPool {
    /// Create a new worker pool
    pub fn new(size: usize) -> Result<Self, JsValue> {
        let mut workers = Vec::with_capacity(size);

        for i in 0..size {
            // Note: In a real implementation, you'd need to:
            // 1. Bundle a separate worker.js file
            // 2. Create Worker from that URL
            // 3. Set up message passing

            // Placeholder - actual worker creation would be:
            // let worker = Worker::new(&format!("worker-{}.js", i))?;
            // workers.push(worker);
        }

        Ok(Self { workers, size })
    }

    /// Submit work to a worker
    pub fn submit(&self, _worker_id: usize, _data: &JsValue) -> Result<(), JsValue> {
        // In a real implementation:
        // self.workers[worker_id].post_message(data)?;
        Ok(())
    }

    /// Get the number of workers
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Browser threading utilities
pub struct BrowserThreadUtils;

impl BrowserThreadUtils {
    /// Check if Web Workers are supported
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn supports_workers() -> bool {
        use js_sys::Reflect;
        use wasm_bindgen::JsCast;

        if let Some(window) = web_sys::window() {
            let global = window.unchecked_ref::<js_sys::Object>();
            Reflect::has(global, &JsValue::from_str("Worker")).unwrap_or(false)
        } else {
            false
        }
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    pub fn supports_workers() -> bool {
        false
    }

    /// Get the number of logical processors (always 1 in browser main thread)
    pub fn hardware_concurrency() -> usize {
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            if let Some(window) = web_sys::window() {
                if let Some(navigator) = window.navigator() {
                    return navigator.hardware_concurrency() as usize;
                }
            }
            1
        }

        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            1
        }
    }

    /// Execute work on main thread (async)
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub async fn spawn_local<F, T>(f: F) -> T
    where
        F: FnOnce() -> T + 'static,
    {
        // In browser, we can use setTimeout to yield to event loop
        f()
    }
}

/// Note on browser threading:
///
/// Browsers have strict limitations on threading:
/// 1. The main thread cannot be blocked
/// 2. Web Workers run in isolated contexts
/// 3. Only serializable data can be passed between workers
/// 4. SharedArrayBuffer is required for shared memory (and has security restrictions)
///
/// For most transpiled Python code, we recommend:
/// - Use async/await for I/O operations
/// - Use Web Workers for CPU-intensive tasks
/// - Use wasm-bindgen-rayon for parallel computations (requires SharedArrayBuffer)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_concurrency() {
        let concurrency = BrowserThreadUtils::hardware_concurrency();
        assert!(concurrency >= 1);
    }

    #[test]
    fn test_workers_support() {
        // Just test that it doesn't panic
        let _ = BrowserThreadUtils::supports_workers();
    }
}
