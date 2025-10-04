//! Web Workers API Integration
//!
//! Provides a unified Web Workers API that works across:
//! - Native Rust (thread pool via rayon)
//! - Browser WASM (Web Workers API via wasm-bindgen)
//! - WASI (thread-based fallback)
//!
//! This module bridges Python's multiprocessing/threading to WASM-compatible implementations.
//!
//! # Architecture
//!
//! The Web Workers API provides:
//! 1. **Worker**: Individual worker thread/process abstraction
//! 2. **WorkerPool**: Managed pool of workers with load balancing
//! 3. **Message Passing**: Type-safe message serialization/deserialization
//! 4. **Error Handling**: Comprehensive error types for all failure modes
//!
//! # Examples
//!
//! ```rust,no_run
//! use web_workers::{WorkerPool, WorkerMessage, WorkerPoolConfig};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct ComputeTask {
//!     data: Vec<f64>,
//! }
//!
//! #[derive(Serialize, Deserialize)]
//! struct ComputeResult {
//!     result: f64,
//! }
//!
//! // Create a worker pool
//! let config = WorkerPoolConfig::default().with_workers(4);
//! let pool = WorkerPool::new(config).await?;
//!
//! // Submit work
//! let task = ComputeTask { data: vec![1.0, 2.0, 3.0] };
//! let message = WorkerMessage::new("compute", &task)?;
//! let response = pool.execute(message).await?;
//!
//! // Parse response
//! let result: ComputeResult = response.parse()?;
//! println!("Result: {}", result.result);
//! ```

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

// Platform-specific backends
#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
mod browser;
#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
use browser::BrowserWorker as PlatformWorker;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
use native::NativeWorker as PlatformWorker;

#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
mod wasi_impl;
#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
use wasi_impl::WasiWorker as PlatformWorker;

/// Worker error types
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Worker creation error: {0}")]
    Creation(String),

    #[error("Message send error: {0}")]
    SendError(String),

    #[error("Message receive error: {0}")]
    ReceiveError(String),

    #[error("Worker termination error: {0}")]
    Termination(String),

    #[error("Timeout error: operation took longer than {0:?}")]
    Timeout(Duration),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Worker busy: no available workers")]
    WorkerBusy,

    #[error("Worker not found: {0}")]
    WorkerNotFound(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Worker message for communication between main thread and workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMessage {
    /// Unique message ID for request/response tracking
    pub id: String,
    /// Message type/command
    pub msg_type: String,
    /// Serialized payload
    pub payload: Vec<u8>,
    /// Optional metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl WorkerMessage {
    /// Create a new worker message with serialized payload
    pub fn new<T: Serialize>(msg_type: impl Into<String>, data: &T) -> Result<Self> {
        let payload = serde_json::to_vec(data)
            .map_err(|e| anyhow!("Failed to serialize message: {}", e))?;

        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            msg_type: msg_type.into(),
            payload,
            metadata: HashMap::new(),
        })
    }

    /// Create a message from raw bytes
    pub fn from_bytes(msg_type: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            msg_type: msg_type.into(),
            payload,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the message
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Parse the payload as a specific type
    pub fn parse<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        serde_json::from_slice(&self.payload)
            .map_err(|e| anyhow!("Failed to deserialize payload: {}", e))
    }

    /// Get the payload as raw bytes
    pub fn payload_bytes(&self) -> &[u8] {
        &self.payload
    }
}

/// Worker response message
pub type WorkerResponse = WorkerMessage;

/// Worker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker is idle and ready to accept work
    Idle,
    /// Worker is currently executing a task
    Busy,
    /// Worker is terminating
    Terminating,
    /// Worker has terminated
    Terminated,
}

/// Worker trait - defines the interface for all worker implementations
pub trait Worker: Send + Sync {
    /// Post a message to the worker
    fn post_message(&self, message: WorkerMessage) -> Result<(), WorkerError>;

    /// Receive a message from the worker (non-blocking)
    fn try_receive(&self) -> Result<Option<WorkerResponse>, WorkerError>;

    /// Receive a message from the worker (blocking with timeout)
    fn receive_timeout(&self, timeout: Duration) -> Result<WorkerResponse, WorkerError>;

    /// Terminate the worker
    fn terminate(&self) -> Result<(), WorkerError>;

    /// Get the current worker state
    fn state(&self) -> WorkerState;

    /// Get the worker ID
    fn id(&self) -> &str;
}

/// Worker pool configuration
#[derive(Debug, Clone)]
pub struct WorkerPoolConfig {
    /// Minimum number of workers to maintain
    pub min_workers: usize,
    /// Maximum number of workers
    pub max_workers: usize,
    /// Whether to use dynamic scaling
    pub dynamic_scaling: bool,
    /// Timeout for worker operations
    pub timeout: Duration,
    /// Maximum queue size
    pub max_queue_size: usize,
}

impl WorkerPoolConfig {
    /// Create a new config with fixed worker count
    pub fn fixed(workers: usize) -> Self {
        Self {
            min_workers: workers,
            max_workers: workers,
            dynamic_scaling: false,
            timeout: Duration::from_secs(30),
            max_queue_size: 1000,
        }
    }

    /// Create a new config with dynamic scaling
    pub fn dynamic(min: usize, max: usize) -> Self {
        Self {
            min_workers: min,
            max_workers: max,
            dynamic_scaling: true,
            timeout: Duration::from_secs(30),
            max_queue_size: 1000,
        }
    }

    /// Set the number of workers (for fixed pool)
    pub fn with_workers(mut self, count: usize) -> Self {
        self.min_workers = count;
        self.max_workers = count;
        self.dynamic_scaling = false;
        self
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the maximum queue size
    pub fn with_queue_size(mut self, size: usize) -> Self {
        self.max_queue_size = size;
        self
    }
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self::fixed(num_cpus::get())
    }
}

/// Worker pool for managing multiple workers
pub struct WorkerPool {
    config: WorkerPoolConfig,
    workers: Arc<parking_lot::RwLock<Vec<Box<dyn Worker>>>>,
    pending_responses: Arc<parking_lot::RwLock<HashMap<String, WorkerResponse>>>,
}

impl WorkerPool {
    /// Create a new worker pool
    pub async fn new(config: WorkerPoolConfig) -> Result<Self> {
        let pool = Self {
            config,
            workers: Arc::new(parking_lot::RwLock::new(Vec::new())),
            pending_responses: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        };

        // Create initial workers
        pool.scale_to(pool.config.min_workers).await?;

        Ok(pool)
    }

    /// Scale the pool to a specific number of workers
    async fn scale_to(&self, count: usize) -> Result<()> {
        let mut workers = self.workers.write();

        // Remove excess workers
        while workers.len() > count {
            if let Some(worker) = workers.pop() {
                let _ = worker.terminate();
            }
        }

        // Add new workers
        while workers.len() < count {
            let worker = self.create_worker(workers.len()).await?;
            workers.push(worker);
        }

        Ok(())
    }

    /// Create a new worker
    async fn create_worker(&self, index: usize) -> Result<Box<dyn Worker>> {
        let worker = PlatformWorker::new(format!("worker-{}", index))?;
        Ok(Box::new(worker))
    }

    /// Execute a message on any available worker
    pub async fn execute(&self, message: WorkerMessage) -> Result<WorkerResponse, WorkerError> {
        let message_id = message.id.clone();

        // Find an idle worker
        let worker = {
            let workers = self.workers.read();
            workers.iter()
                .find(|w| w.state() == WorkerState::Idle)
                .ok_or(WorkerError::WorkerBusy)?
                .id()
                .to_string()
        };

        // Send the message
        {
            let workers = self.workers.read();
            if let Some(w) = workers.iter().find(|w| w.id() == worker) {
                w.post_message(message)?;
            } else {
                return Err(WorkerError::WorkerNotFound(worker));
            }
        }

        // Wait for response with timeout
        let timeout = self.config.timeout;
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                return Err(WorkerError::Timeout(timeout));
            }

            // Check for response
            {
                let mut responses = self.pending_responses.write();
                if let Some(response) = responses.remove(&message_id) {
                    return Ok(response);
                }
            }

            // Poll workers for messages
            {
                let workers = self.workers.read();
                for worker in workers.iter() {
                    if let Ok(Some(response)) = worker.try_receive() {
                        let mut responses = self.pending_responses.write();
                        responses.insert(response.id.clone(), response);
                    }
                }
            }

            // Small sleep to avoid busy-waiting
            #[cfg(not(target_arch = "wasm32"))]
            tokio::time::sleep(Duration::from_millis(10)).await;

            #[cfg(target_arch = "wasm32")]
            {
                // In browser, yield to event loop
                wasm_bindgen_futures::JsFuture::from(
                    js_sys::Promise::new(&mut |resolve, _reject| {
                        web_sys::window()
                            .unwrap()
                            .set_timeout_with_callback_and_timeout_and_arguments_0(
                                &resolve,
                                10,
                            )
                            .unwrap();
                    })
                ).await.ok();
            }
        }
    }

    /// Broadcast a message to all workers
    pub fn broadcast(&self, message: WorkerMessage) -> Result<(), WorkerError> {
        let workers = self.workers.read();

        for worker in workers.iter() {
            // Clone message for each worker (new ID)
            let mut worker_msg = message.clone();
            worker_msg.id = uuid::Uuid::new_v4().to_string();
            worker.post_message(worker_msg)?;
        }

        Ok(())
    }

    /// Get the number of workers
    pub fn worker_count(&self) -> usize {
        self.workers.read().len()
    }

    /// Get the number of idle workers
    pub fn idle_worker_count(&self) -> usize {
        self.workers.read()
            .iter()
            .filter(|w| w.state() == WorkerState::Idle)
            .count()
    }

    /// Terminate all workers and shut down the pool
    pub fn shutdown(&self) -> Result<(), WorkerError> {
        let mut workers = self.workers.write();

        for worker in workers.drain(..) {
            worker.terminate()?;
        }

        Ok(())
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_message_creation() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct TestData {
            value: i32,
        }

        let data = TestData { value: 42 };
        let message = WorkerMessage::new("test", &data).unwrap();

        assert_eq!(message.msg_type, "test");
        assert!(!message.id.is_empty());

        let parsed: TestData = message.parse().unwrap();
        assert_eq!(parsed, data);
    }

    #[test]
    fn test_worker_message_metadata() {
        let message = WorkerMessage::from_bytes("test", vec![1, 2, 3])
            .with_metadata("priority", "high")
            .with_metadata("source", "main");

        assert_eq!(message.metadata.get("priority").unwrap(), "high");
        assert_eq!(message.metadata.get("source").unwrap(), "main");
    }

    #[test]
    fn test_worker_pool_config() {
        let config = WorkerPoolConfig::fixed(4)
            .with_timeout(Duration::from_secs(10))
            .with_queue_size(500);

        assert_eq!(config.min_workers, 4);
        assert_eq!(config.max_workers, 4);
        assert!(!config.dynamic_scaling);
        assert_eq!(config.timeout, Duration::from_secs(10));
        assert_eq!(config.max_queue_size, 500);
    }

    #[test]
    fn test_worker_pool_config_dynamic() {
        let config = WorkerPoolConfig::dynamic(2, 8);

        assert_eq!(config.min_workers, 2);
        assert_eq!(config.max_workers, 8);
        assert!(config.dynamic_scaling);
    }
}
