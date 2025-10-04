//! Browser Web Workers Implementation
//!
//! Implements the Worker trait using the browser's Web Workers API
//! via wasm-bindgen.

use super::{Worker, WorkerError, WorkerMessage, WorkerResponse, WorkerState};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::time::Duration;
use parking_lot::RwLock;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys;

/// Browser-based Web Worker implementation
pub struct BrowserWorker {
    id: String,
    worker: web_sys::Worker,
    state: Arc<RwLock<WorkerState>>,
    message_queue: Arc<RwLock<Vec<WorkerResponse>>>,
}

impl BrowserWorker {
    /// Create a new browser worker
    pub fn new(id: String) -> Result<Self, WorkerError> {
        // Create a worker from a script URL or blob
        // For now, we'll use a simple inline worker script
        let worker_script = Self::create_worker_script();
        let blob = Self::create_blob(&worker_script)
            .map_err(|e| WorkerError::Creation(format!("Failed to create blob: {:?}", e)))?;

        let worker_url = Self::create_blob_url(&blob)
            .map_err(|e| WorkerError::Creation(format!("Failed to create blob URL: {:?}", e)))?;

        let worker = web_sys::Worker::new(&worker_url)
            .map_err(|e| WorkerError::Creation(format!("Failed to create worker: {:?}", e)))?;

        let state = Arc::new(RwLock::new(WorkerState::Idle));
        let message_queue = Arc::new(RwLock::new(Vec::new()));

        // Set up message handler
        let message_queue_clone = message_queue.clone();
        let state_clone = state.clone();

        let onmessage_callback = Closure::wrap(Box::new(move |event: web_sys::MessageEvent| {
            if let Ok(data) = event.data().dyn_into::<js_sys::Uint8Array>() {
                let bytes = data.to_vec();

                // Try to deserialize as WorkerResponse
                if let Ok(response) = serde_json::from_slice::<WorkerResponse>(&bytes) {
                    message_queue_clone.write().push(response);
                    *state_clone.write() = WorkerState::Idle;
                }
            }
        }) as Box<dyn FnMut(web_sys::MessageEvent)>);

        worker.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
        onmessage_callback.forget(); // Keep callback alive

        // Set up error handler
        let state_clone = state.clone();
        let onerror_callback = Closure::wrap(Box::new(move |event: web_sys::ErrorEvent| {
            web_sys::console::error_1(&format!("Worker error: {:?}", event.message()).into());
            *state_clone.write() = WorkerState::Idle; // Reset to idle on error
        }) as Box<dyn FnMut(web_sys::ErrorEvent)>);

        worker.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
        onerror_callback.forget(); // Keep callback alive

        Ok(Self {
            id,
            worker,
            state,
            message_queue,
        })
    }

    /// Create a simple worker script
    fn create_worker_script() -> String {
        r#"
// Web Worker script
self.onmessage = function(e) {
    // Echo the message back for now
    // In a real implementation, this would process the message
    const message = e.data;

    // Simple echo response
    self.postMessage(message);
};
        "#.to_string()
    }

    /// Create a blob from script content
    fn create_blob(content: &str) -> Result<web_sys::Blob, JsValue> {
        let array = js_sys::Array::new();
        array.push(&JsValue::from_str(content));

        let mut options = web_sys::BlobPropertyBag::new();
        options.type_("application/javascript");

        web_sys::Blob::new_with_str_sequence_and_options(&array, &options)
    }

    /// Create a blob URL
    fn create_blob_url(blob: &web_sys::Blob) -> Result<String, JsValue> {
        web_sys::Url::create_object_url_with_blob(blob)
    }
}

impl Worker for BrowserWorker {
    fn post_message(&self, message: WorkerMessage) -> Result<(), WorkerError> {
        // Update state to busy
        *self.state.write() = WorkerState::Busy;

        // Serialize message
        let bytes = serde_json::to_vec(&message)
            .map_err(|e| WorkerError::Serialization(e.to_string()))?;

        // Convert to Uint8Array
        let array = js_sys::Uint8Array::from(bytes.as_slice());

        // Post message to worker
        self.worker.post_message(&array.into())
            .map_err(|e| WorkerError::SendError(format!("Failed to post message: {:?}", e)))?;

        Ok(())
    }

    fn try_receive(&self) -> Result<Option<WorkerResponse>, WorkerError> {
        let mut queue = self.message_queue.write();

        if queue.is_empty() {
            Ok(None)
        } else {
            Ok(Some(queue.remove(0)))
        }
    }

    fn receive_timeout(&self, timeout: Duration) -> Result<WorkerResponse, WorkerError> {
        let start = std::time::Instant::now();

        loop {
            if let Some(message) = self.try_receive()? {
                return Ok(message);
            }

            if start.elapsed() > timeout {
                return Err(WorkerError::Timeout(timeout));
            }

            // In browser, we can't really sleep synchronously
            // This would need to be used with async/await
        }
    }

    fn terminate(&self) -> Result<(), WorkerError> {
        *self.state.write() = WorkerState::Terminating;

        self.worker.terminate();

        *self.state.write() = WorkerState::Terminated;

        Ok(())
    }

    fn state(&self) -> WorkerState {
        *self.state.read()
    }

    fn id(&self) -> &str {
        &self.id
    }
}

impl Drop for BrowserWorker {
    fn drop(&mut self) {
        let _ = self.terminate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_script_creation() {
        let script = BrowserWorker::create_worker_script();
        assert!(script.contains("onmessage"));
        assert!(script.contains("postMessage"));
    }

    // Note: Other tests require a browser environment
    // These would typically be run with wasm-pack test --headless --chrome
}
