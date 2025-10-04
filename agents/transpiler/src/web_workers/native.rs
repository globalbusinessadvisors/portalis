//! Native Thread Pool Implementation
//!
//! Implements the Worker trait using rayon's thread pool for native Rust.

use super::{Worker, WorkerError, WorkerMessage, WorkerResponse, WorkerState};
use std::sync::Arc;
use std::time::Duration;
use parking_lot::RwLock;
use crossbeam::channel::{Sender, Receiver, bounded, TryRecvError, RecvTimeoutError};

/// Native worker implementation using threads
pub struct NativeWorker {
    id: String,
    state: Arc<RwLock<WorkerState>>,
    tx: Sender<WorkerMessage>,
    rx: Receiver<WorkerResponse>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl NativeWorker {
    /// Create a new native worker
    pub fn new(id: String) -> Result<Self, WorkerError> {
        let state = Arc::new(RwLock::new(WorkerState::Idle));

        // Create channels for bidirectional communication
        let (msg_tx, msg_rx) = bounded::<WorkerMessage>(100);
        let (resp_tx, resp_rx) = bounded::<WorkerResponse>(100);

        let state_clone = state.clone();
        let worker_id = id.clone();

        // Spawn worker thread
        let handle = std::thread::Builder::new()
            .name(format!("worker-{}", id))
            .spawn(move || {
                Self::worker_loop(worker_id, msg_rx, resp_tx, state_clone);
            })
            .map_err(|e| WorkerError::Creation(format!("Failed to spawn thread: {}", e)))?;

        Ok(Self {
            id,
            state,
            tx: msg_tx,
            rx: resp_rx,
            handle: Some(handle),
        })
    }

    /// Worker thread main loop
    fn worker_loop(
        _id: String,
        msg_rx: Receiver<WorkerMessage>,
        resp_tx: Sender<WorkerResponse>,
        state: Arc<RwLock<WorkerState>>,
    ) {
        loop {
            // Wait for incoming message
            match msg_rx.recv() {
                Ok(message) => {
                    // Mark as busy
                    *state.write() = WorkerState::Busy;

                    // Process message
                    let response = Self::process_message(message);

                    // Send response
                    if resp_tx.send(response).is_err() {
                        // Channel closed, terminate
                        break;
                    }

                    // Mark as idle
                    *state.write() = WorkerState::Idle;
                }
                Err(_) => {
                    // Channel closed, terminate
                    break;
                }
            }
        }

        *state.write() = WorkerState::Terminated;
    }

    /// Process an incoming message and generate a response
    fn process_message(message: WorkerMessage) -> WorkerResponse {
        // For now, this is a simple echo
        // In a real implementation, this would:
        // 1. Deserialize the message
        // 2. Execute the requested computation
        // 3. Serialize the result
        // 4. Return a response

        WorkerResponse {
            id: message.id.clone(),
            msg_type: format!("{}_response", message.msg_type),
            payload: message.payload,
            metadata: message.metadata,
        }
    }
}

impl Worker for NativeWorker {
    fn post_message(&self, message: WorkerMessage) -> Result<(), WorkerError> {
        self.tx.send(message)
            .map_err(|e| WorkerError::SendError(format!("Failed to send message: {}", e)))?;

        Ok(())
    }

    fn try_receive(&self) -> Result<Option<WorkerResponse>, WorkerError> {
        match self.rx.try_recv() {
            Ok(response) => Ok(Some(response)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err(WorkerError::ReceiveError("Worker disconnected".to_string()))
            }
        }
    }

    fn receive_timeout(&self, timeout: Duration) -> Result<WorkerResponse, WorkerError> {
        match self.rx.recv_timeout(timeout) {
            Ok(response) => Ok(response),
            Err(RecvTimeoutError::Timeout) => Err(WorkerError::Timeout(timeout)),
            Err(RecvTimeoutError::Disconnected) => {
                Err(WorkerError::ReceiveError("Worker disconnected".to_string()))
            }
        }
    }

    fn terminate(&self) -> Result<(), WorkerError> {
        *self.state.write() = WorkerState::Terminating;

        // Note: We can't actually drop the sender here since we only have &self
        // The worker thread will exit when the channel is dropped in Drop impl

        Ok(())
    }

    fn state(&self) -> WorkerState {
        *self.state.read()
    }

    fn id(&self) -> &str {
        &self.id
    }
}

impl Drop for NativeWorker {
    fn drop(&mut self) {
        let _ = self.terminate();

        // Join the worker thread
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Native worker pool using rayon
#[allow(dead_code)]
pub struct RayonWorkerPool {
    pool: rayon::ThreadPool,
}

impl RayonWorkerPool {
    /// Create a new rayon worker pool
    pub fn new(num_threads: usize) -> Result<Self, WorkerError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| WorkerError::Creation(format!("Failed to create thread pool: {}", e)))?;

        Ok(Self { pool })
    }

    /// Execute a task on the pool
    pub fn execute<F, R>(&self, task: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(task)
    }

    /// Execute a task asynchronously
    pub fn spawn<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.pool.spawn(task);
    }

    /// Get the number of threads
    pub fn thread_count(&self) -> usize {
        self.pool.current_num_threads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_native_worker_creation() {
        let worker = NativeWorker::new("test-worker".to_string()).unwrap();
        assert_eq!(worker.id(), "test-worker");
        assert_eq!(worker.state(), WorkerState::Idle);
    }

    #[test]
    fn test_native_worker_message_passing() {
        let worker = NativeWorker::new("test-worker".to_string()).unwrap();

        // Create a test message
        let message = WorkerMessage::from_bytes("test", vec![1, 2, 3, 4]);

        // Send message
        worker.post_message(message.clone()).unwrap();

        // Receive response with timeout
        let response = worker.receive_timeout(Duration::from_secs(1)).unwrap();

        assert_eq!(response.id, message.id);
        assert_eq!(response.payload, message.payload);
    }

    #[test]
    fn test_native_worker_try_receive_empty() {
        let worker = NativeWorker::new("test-worker".to_string()).unwrap();

        // Try to receive without sending
        let result = worker.try_receive().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_native_worker_timeout() {
        let worker = NativeWorker::new("test-worker".to_string()).unwrap();

        // Try to receive with short timeout
        let result = worker.receive_timeout(Duration::from_millis(100));
        assert!(matches!(result, Err(WorkerError::Timeout(_))));
    }

    #[test]
    fn test_rayon_pool_creation() {
        let pool = RayonWorkerPool::new(4).unwrap();
        assert_eq!(pool.thread_count(), 4);
    }

    #[test]
    fn test_rayon_pool_execution() {
        let pool = RayonWorkerPool::new(2).unwrap();

        let result = pool.execute(|| {
            42 + 8
        });

        assert_eq!(result, 50);
    }

    #[test]
    fn test_rayon_pool_spawn() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let pool = RayonWorkerPool::new(2).unwrap();
        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = executed.clone();

        pool.spawn(move || {
            executed_clone.store(true, Ordering::SeqCst);
        });

        // Give the task time to execute
        std::thread::sleep(Duration::from_millis(100));
        assert!(executed.load(Ordering::SeqCst));
    }
}
