//! WASI Platform Implementation
//!
//! WASI implementation with support for wasi-threads when available.
//! Falls back to single-threaded execution if threads are not supported.

use std::sync::Arc;

/// WASI threading capabilities
pub struct WasiThreadingCapabilities {
    /// Whether WASI threads are supported in this runtime
    pub threads_supported: bool,
    /// Maximum number of threads
    pub max_threads: usize,
    /// Whether shared memory is available
    pub shared_memory: bool,
}

impl WasiThreadingCapabilities {
    /// Detect WASI threading capabilities
    pub fn detect() -> Self {
        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            // Check for wasi-threads support
            // This is a simplified check - actual implementation would
            // query the WASI runtime capabilities
            Self {
                threads_supported: std::thread::available_parallelism()
                    .map(|n| n.get() > 1)
                    .unwrap_or(false),
                max_threads: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
                shared_memory: false, // Most WASI runtimes don't support this yet
            }
        }

        #[cfg(not(all(target_arch = "wasm32", feature = "wasi")))]
        {
            Self {
                threads_supported: false,
                max_threads: 1,
                shared_memory: false,
            }
        }
    }

    /// Check if threading is available
    pub fn has_threads(&self) -> bool {
        self.threads_supported
    }

    /// Get recommended number of threads
    pub fn recommended_threads(&self) -> usize {
        if self.threads_supported {
            self.max_threads
        } else {
            1
        }
    }
}

/// WASI thread utilities
pub struct WasiThreadUtils;

impl WasiThreadUtils {
    /// Get WASI capabilities
    pub fn capabilities() -> WasiThreadingCapabilities {
        WasiThreadingCapabilities::detect()
    }

    /// Check if we're running in a WASI environment
    pub fn is_wasi() -> bool {
        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            true
        }

        #[cfg(not(all(target_arch = "wasm32", feature = "wasi")))]
        {
            false
        }
    }

    /// Get available parallelism
    pub fn available_parallelism() -> usize {
        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }

        #[cfg(not(all(target_arch = "wasm32", feature = "wasi")))]
        {
            1
        }
    }
}

/// WASI-specific thread pool (compatibility wrapper)
pub struct WasiThreadPool {
    capabilities: WasiThreadingCapabilities,
    size: usize,
}

impl WasiThreadPool {
    /// Create a new WASI thread pool
    pub fn new(size: usize) -> Self {
        let capabilities = WasiThreadingCapabilities::detect();
        let actual_size = if capabilities.has_threads() {
            size.min(capabilities.max_threads)
        } else {
            1
        };

        Self {
            capabilities,
            size: actual_size,
        }
    }

    /// Execute a task
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if self.capabilities.has_threads() {
            #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
            {
                // Use std::thread if available
                let _ = std::thread::spawn(f);
            }

            #[cfg(not(all(target_arch = "wasm32", feature = "wasi")))]
            {
                // Fallback: execute immediately
                f();
            }
        } else {
            // Single-threaded: execute immediately
            f();
        }
    }

    /// Get the pool size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if threading is enabled
    pub fn has_threading(&self) -> bool {
        self.capabilities.has_threads()
    }
}

/// Note on WASI threading:
///
/// WASI threading support is evolving:
/// - wasi-threads proposal adds thread support
/// - Not all WASI runtimes support threads yet
/// - Wasmtime has experimental thread support
/// - Wasmer is working on thread support
///
/// This implementation provides:
/// 1. Capability detection
/// 2. Graceful fallback to single-threaded execution
/// 3. Compatible API with native threading
///
/// For production use:
/// - Check capabilities before using threads
/// - Design code to work both with and without threads
/// - Test on your target WASI runtime

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capabilities_detection() {
        let caps = WasiThreadingCapabilities::detect();
        assert!(caps.max_threads >= 1);
        assert!(caps.recommended_threads() >= 1);
    }

    #[test]
    fn test_wasi_utils() {
        let _is_wasi = WasiThreadUtils::is_wasi();
        let parallelism = WasiThreadUtils::available_parallelism();
        assert!(parallelism >= 1);
    }

    #[test]
    fn test_thread_pool() {
        let pool = WasiThreadPool::new(4);
        assert!(pool.size() >= 1);
        assert!(pool.size() <= 4);

        // Test that execute doesn't panic
        pool.execute(|| {
            // Do nothing
        });
    }
}
