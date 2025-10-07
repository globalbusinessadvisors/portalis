//! Performance metrics tracking for CPU bridge

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// CPU bridge performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// Total tasks completed
    tasks_completed: u64,

    /// Total execution time across all tasks
    total_time_ns: u64,

    /// Number of batch operations
    batch_count: u64,

    /// Number of single-task operations
    single_task_count: u64,

    /// Memory optimization metrics
    #[cfg(feature = "memory-opt")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_metrics: Option<MemoryMetrics>,
}

impl CpuMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            tasks_completed: 0,
            total_time_ns: 0,
            batch_count: 0,
            single_task_count: 0,
            #[cfg(feature = "memory-opt")]
            memory_metrics: Some(MemoryMetrics::new()),
        }
    }

    /// Record a batch execution
    pub fn record_batch(&mut self, num_tasks: usize, elapsed: Duration) {
        self.tasks_completed += num_tasks as u64;
        self.total_time_ns += elapsed.as_nanos() as u64;
        self.batch_count += 1;
    }

    /// Record a single task execution
    pub fn record_single(&mut self, elapsed: Duration) {
        self.tasks_completed += 1;
        self.total_time_ns += elapsed.as_nanos() as u64;
        self.single_task_count += 1;
    }

    /// Get total tasks completed
    pub fn tasks_completed(&self) -> u64 {
        self.tasks_completed
    }

    /// Get average task time in milliseconds
    pub fn avg_task_time_ms(&self) -> f64 {
        if self.tasks_completed == 0 {
            return 0.0;
        }
        let avg_ns = self.total_time_ns as f64 / self.tasks_completed as f64;
        avg_ns / 1_000_000.0
    }

    /// Get CPU utilization estimate (simplified)
    pub fn cpu_utilization(&self) -> f64 {
        // Simplified metric - in production this would be more sophisticated
        if self.batch_count > 0 {
            0.85 // Assume good utilization for batch operations
        } else {
            0.25 // Lower utilization for single tasks
        }
    }

    /// Get total batch count
    pub fn batch_count(&self) -> u64 {
        self.batch_count
    }

    /// Get total single task count
    pub fn single_task_count(&self) -> u64 {
        self.single_task_count
    }

    /// Get total time in milliseconds
    pub fn total_time_ms(&self) -> f64 {
        self.total_time_ns as f64 / 1_000_000.0
    }

    /// Get success rate (always 1.0 for now, failures tracked separately in future)
    pub fn success_rate(&self) -> f64 {
        1.0
    }
}

impl Default for CpuMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let mut metrics = CpuMetrics::new();
        
        metrics.record_batch(10, Duration::from_millis(100));
        assert_eq!(metrics.tasks_completed(), 10);
        assert_eq!(metrics.batch_count(), 1);
        
        metrics.record_single(Duration::from_millis(10));
        assert_eq!(metrics.tasks_completed(), 11);
        assert_eq!(metrics.single_task_count(), 1);
    }

    #[test]
    fn test_avg_task_time() {
        let mut metrics = CpuMetrics::new();
        metrics.record_batch(10, Duration::from_millis(100));
        
        let avg = metrics.avg_task_time_ms();
        assert!(avg > 0.0);
        assert!(avg <= 100.0);
    }
}

/// Thread pool specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolMetrics {
    /// Total number of tasks executed
    pub total_tasks: u64,

    /// Total number of batches executed
    pub total_batches: u64,

    /// Total execution time in nanoseconds
    pub total_time_ns: u64,

    /// Number of currently active tasks
    pub active_tasks: usize,
}

impl ThreadPoolMetrics {
    pub fn new() -> Self {
        Self {
            total_tasks: 0,
            total_batches: 0,
            total_time_ns: 0,
            active_tasks: 0,
        }
    }

    pub fn record_batch(&mut self, num_tasks: usize, elapsed: Duration) {
        self.total_tasks += num_tasks as u64;
        self.total_batches += 1;
        self.total_time_ns += elapsed.as_nanos() as u64;
    }

    pub fn record_single(&mut self, elapsed: Duration) {
        self.total_tasks += 1;
        self.total_time_ns += elapsed.as_nanos() as u64;
    }

    pub fn avg_task_time_ms(&self) -> f64 {
        if self.total_tasks == 0 {
            return 0.0;
        }
        let avg_ns = self.total_time_ns as f64 / self.total_tasks as f64;
        avg_ns / 1_000_000.0
    }
}

impl Default for ThreadPoolMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory optimization metrics
#[cfg(feature = "memory-opt")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,

    /// Current memory usage in bytes
    pub current_memory_bytes: usize,

    /// Total allocations made
    pub total_allocations: u64,

    /// Allocations per task
    pub allocations_per_task: f64,

    /// String interning statistics
    pub interned_strings: usize,
    pub interned_memory_saved_bytes: usize,

    /// Arena allocation statistics
    pub arena_bytes_allocated: usize,
    pub arena_allocation_count: usize,

    /// Object pool statistics
    pub pool_hits: u64,
    pub pool_misses: u64,

    /// Cache hit rate estimate
    pub cache_hit_rate: f64,
}

#[cfg(feature = "memory-opt")]
impl MemoryMetrics {
    pub fn new() -> Self {
        Self {
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
            total_allocations: 0,
            allocations_per_task: 0.0,
            interned_strings: 0,
            interned_memory_saved_bytes: 0,
            arena_bytes_allocated: 0,
            arena_allocation_count: 0,
            pool_hits: 0,
            pool_misses: 0,
            cache_hit_rate: 0.0,
        }
    }

    /// Update peak memory if current exceeds it
    pub fn update_peak(&mut self, current: usize) {
        if current > self.peak_memory_bytes {
            self.peak_memory_bytes = current;
        }
        self.current_memory_bytes = current;
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocations += 1;
        self.current_memory_bytes += size;
        self.update_peak(self.current_memory_bytes);
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        self.current_memory_bytes = self.current_memory_bytes.saturating_sub(size);
    }

    /// Calculate pool hit rate
    pub fn pool_hit_rate(&self) -> f64 {
        let total = self.pool_hits + self.pool_misses;
        if total == 0 {
            return 0.0;
        }
        self.pool_hits as f64 / total as f64
    }

    /// Update allocations per task
    pub fn update_allocations_per_task(&mut self, total_tasks: u64) {
        if total_tasks > 0 {
            self.allocations_per_task = self.total_allocations as f64 / total_tasks as f64;
        }
    }
}

#[cfg(feature = "memory-opt")]
impl Default for MemoryMetrics {
    fn default() -> Self {
        Self::new()
    }
}
