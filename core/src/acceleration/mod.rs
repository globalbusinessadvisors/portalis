//! Acceleration strategy management and execution
//!
//! This module provides intelligent acceleration strategy selection and execution
//! with graceful degradation from GPU to CPU when needed.

pub mod executor;

#[cfg(feature = "memory-opt")]
pub mod memory;

pub use executor::{
    CpuExecutor, ExecutionResult, ExecutionStrategy, GpuExecutor, HardwareCapabilities,
    NoGpu, StrategyManager, SystemLoad, WorkloadProfile,
};

#[cfg(feature = "memory-opt")]
pub use memory::{
    global_interner, intern, AlignedBuffer, BatchData, InternerStats, ObjectPool, PooledObject,
    StringInterner,
};

use serde::{Deserialize, Serialize};

/// Configuration for acceleration strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerationConfig {
    /// Execution strategy to use
    pub strategy: ExecutionStrategy,

    /// Number of CPU threads (None = auto-detect)
    pub cpu_threads: Option<usize>,

    /// Batch size for parallel execution (None = auto-detect)
    pub batch_size: Option<usize>,

    /// Enable GPU acceleration if available
    pub enable_gpu: bool,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            strategy: ExecutionStrategy::Auto,
            cpu_threads: None,
            batch_size: None,
            enable_gpu: false,
        }
    }
}

impl AccelerationConfig {
    /// Create new config with auto-detection
    pub fn auto() -> Self {
        Self::default()
    }

    /// Create CPU-only config
    pub fn cpu_only() -> Self {
        Self {
            strategy: ExecutionStrategy::CpuOnly,
            cpu_threads: None,
            batch_size: None,
            enable_gpu: false,
        }
    }

    /// Create config with specific thread count
    pub fn with_threads(threads: usize) -> Self {
        Self {
            strategy: ExecutionStrategy::CpuOnly,
            cpu_threads: Some(threads),
            batch_size: None,
            enable_gpu: false,
        }
    }

    /// Create config with GPU enabled
    pub fn with_gpu() -> Self {
        Self {
            strategy: ExecutionStrategy::Auto,
            cpu_threads: None,
            batch_size: None,
            enable_gpu: true,
        }
    }
}
