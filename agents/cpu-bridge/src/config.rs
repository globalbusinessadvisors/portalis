//! CPU configuration and hardware detection

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    num_threads: usize,
    batch_size: usize,
    enable_simd: bool,
    stack_size: usize,
}

impl CpuConfig {
    pub fn auto_detect() -> Self {
        let num_threads = num_cpus::get();
        let batch_size = Self::optimal_batch_size(num_threads);

        Self {
            num_threads,
            batch_size,
            enable_simd: Self::detect_simd_support(),
            stack_size: 2 * 1024 * 1024,
        }
    }

    pub fn builder() -> CpuConfigBuilder {
        CpuConfigBuilder::new()
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn simd_enabled(&self) -> bool {
        self.enable_simd
    }

    pub fn stack_size(&self) -> usize {
        self.stack_size
    }

    fn optimal_batch_size(num_threads: usize) -> usize {
        let base_batch_size = 32;
        (base_batch_size * num_threads).max(32).min(256)
    }

    fn detect_simd_support() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(target_arch = "aarch64")]
        {
            true
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

#[derive(Debug, Clone)]
pub struct CpuConfigBuilder {
    num_threads: Option<usize>,
    batch_size: Option<usize>,
    enable_simd: Option<bool>,
    stack_size: Option<usize>,
}

impl CpuConfigBuilder {
    pub fn new() -> Self {
        Self {
            num_threads: None,
            batch_size: None,
            enable_simd: None,
            stack_size: None,
        }
    }

    pub fn num_threads(mut self, threads: usize) -> Self {
        assert!(threads > 0, "Thread count must be greater than 0");
        self.num_threads = Some(threads);
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        assert!(size > 0, "Batch size must be greater than 0");
        self.batch_size = Some(size);
        self
    }

    pub fn enable_simd(mut self, enable: bool) -> Self {
        self.enable_simd = Some(enable);
        self
    }

    pub fn stack_size(mut self, size: usize) -> Self {
        assert!(size >= 1024 * 1024, "Stack size must be at least 1MB");
        self.stack_size = Some(size);
        self
    }

    pub fn build(self) -> CpuConfig {
        let defaults = CpuConfig::auto_detect();
        CpuConfig {
            num_threads: self.num_threads.unwrap_or(defaults.num_threads),
            batch_size: self.batch_size.unwrap_or(defaults.batch_size),
            enable_simd: self.enable_simd.unwrap_or(defaults.enable_simd),
            stack_size: self.stack_size.unwrap_or(defaults.stack_size),
        }
    }
}

impl Default for CpuConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread pool specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Number of worker threads
    pub num_threads: usize,

    /// Stack size per thread
    pub stack_size: usize,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            stack_size: 2 * 1024 * 1024, // 2MB
        }
    }
}

impl ThreadPoolConfig {
    /// Create config with specific thread count
    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            num_threads,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect_config() {
        let config = CpuConfig::auto_detect();
        assert!(config.num_threads() > 0);
    }

    #[test]
    fn test_cpu_config_builder() {
        let config = CpuConfig::builder()
            .num_threads(4)
            .batch_size(16)
            .enable_simd(false)
            .stack_size(2 * 1024 * 1024)
            .build();

        assert_eq!(config.num_threads(), 4);
        assert_eq!(config.batch_size(), 16);
        assert!(!config.simd_enabled());
    }

    #[test]
    fn test_thread_pool_config() {
        let config = ThreadPoolConfig::with_threads(8);
        assert_eq!(config.num_threads, 8);
    }
}
