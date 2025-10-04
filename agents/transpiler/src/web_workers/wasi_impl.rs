//! WASI Thread-Based Worker Implementation
//!
//! Implements the Worker trait using thread-based workers for WASI.
//! WASI doesn't have native Web Workers, so we emulate using threads.

use super::{Worker, WorkerError, WorkerMessage, WorkerResponse, WorkerState};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::time::Duration;
use parking_lot::RwLock;
use crossbeam::channel::{Sender, Receiver, bounded, TryRecvError, RecvTimeoutError};

/// WASI worker implementation (thread-based)
///
/// Since WASI doesn't support Web Workers natively, we use threads
/// to provide similar functionality.
pub struct WasiWorker {
    id: String,
    state: Arc<RwLock<WorkerState>>,
    tx: Sender<WorkerMessage>,
    rx: Receiver<WorkerResponse>,
}

impl WasiWorker {
    /// Create a new WASI worker
    pub fn new(id: String) -> Result<Self, WorkerError> {
        let state = Arc::new(RwLock::new(WorkerState::Idle));

        // Create channels for communication
        let (msg_tx, msg_rx) = bounded::<WorkerMessage>(100);
        let (resp_tx, resp_rx) = bounded::<WorkerResponse>(100);

        let state_clone = state.clone();

        // Spawn worker thread
        // Note: WASI thread support is limited, this may not work in all WASI runtimes
        std::thread::spawn(move || {
            Self::worker_loop(msg_rx, resp_tx, state_clone);
        });

        Ok(Self {
            id,
            state,
            tx: msg_tx,
            rx: resp_rx,
        })
    }

    /// Worker thread main loop
    fn worker_loop(
        msg_rx: Receiver<WorkerMessage>,
        resp_tx: Sender<WorkerResponse>,
        state: Arc<RwLock<WorkerState>>,
    ) {
        loop {
            match msg_rx.recv() {
                Ok(message) => {
                    *state.write() = WorkerState::Busy;

                    // Process message (simple echo for now)
                    let response = WorkerResponse {
                        id: message.id.clone(),
                        msg_type: format!("{}_response", message.msg_type),
                        payload: message.payload,
                        metadata: message.metadata,
                    };

                    if resp_tx.send(response).is_err() {
                        break;
                    }

                    *state.write() = WorkerState::Idle;
                }
                Err(_) => break,
            }
        }

        *state.write() = WorkerState::Terminated;
    }
}

impl Worker for WasiWorker {
    fn post_message(&self, message: WorkerMessage) -> Result<(), WorkerError> {
        self.tx.send(message)
            .map_err(|e| WorkerError::SendError(e.to_string()))?;
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
        drop(&self.tx);
        Ok(())
    }

    fn state(&self) -> WorkerState {
        *self.state.read()
    }

    fn id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasi_worker_creation() {
        let worker = WasiWorker::new("test-wasi-worker".to_string()).unwrap();
        assert_eq!(worker.id(), "test-wasi-worker");
        assert_eq!(worker.state(), WorkerState::Idle);
    }
}
