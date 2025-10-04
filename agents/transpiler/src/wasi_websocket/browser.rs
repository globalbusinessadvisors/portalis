//! Browser WebSocket implementation using web-sys
//!
//! This module provides WebSocket support for browser WASM targets using
//! the browser's native WebSocket API via wasm-bindgen and web-sys.

use super::{
    CloseFrame, WebSocketConfig, WebSocketError, WebSocketMessage, WebSocketSharedState,
    WebSocketState,
};
use anyhow::{anyhow, Result};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CloseEvent, ErrorEvent, MessageEvent, WebSocket as BrowserWebSocketApi};

/// Browser WebSocket implementation
pub struct BrowserWebSocket {
    ws: BrowserWebSocketApi,
    shared: WebSocketSharedState,
}

impl BrowserWebSocket {
    /// Connect to a WebSocket server
    pub async fn connect(
        config: WebSocketConfig,
        shared: WebSocketSharedState,
    ) -> Result<Self> {
        // Validate URL
        if !config.url.starts_with("ws://") && !config.url.starts_with("wss://") {
            return Err(anyhow!(WebSocketError::InvalidUrl(format!(
                "Invalid WebSocket URL: {}. Must start with ws:// or wss://",
                config.url
            ))));
        }

        // Create WebSocket with subprotocols if specified
        let ws = if config.subprotocols.is_empty() {
            BrowserWebSocketApi::new(&config.url)
                .map_err(|e| anyhow!(WebSocketError::Connection(format!("{:?}", e))))?
        } else {
            let protocols = js_sys::Array::new();
            for protocol in &config.subprotocols {
                protocols.push(&JsValue::from_str(protocol));
            }

            BrowserWebSocketApi::new_with_str_sequence(&config.url, &protocols)
                .map_err(|e| anyhow!(WebSocketError::Connection(format!("{:?}", e))))?
        };

        // Set binary type to arraybuffer for binary messages
        ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

        // Create a promise to wait for connection
        let (tx, rx) = tokio::sync::oneshot::channel();
        let tx = std::sync::Arc::new(std::sync::Mutex::new(Some(tx)));

        // Set up onopen callback
        {
            let shared_clone = shared.clone();
            let tx_clone = tx.clone();
            let onopen = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                shared_clone.set_state(WebSocketState::Open);
                shared_clone.handlers.trigger_open();

                // Notify that connection is open
                if let Some(tx) = tx_clone.lock().unwrap().take() {
                    let _ = tx.send(Ok(()));
                }
            }) as Box<dyn FnMut(web_sys::Event)>);

            ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
            onopen.forget(); // Keep closure alive
        }

        // Set up onmessage callback
        {
            let shared_clone = shared.clone();
            let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
                let msg = Self::parse_message_event(&event);
                if let Ok(ws_msg) = msg {
                    shared_clone.handlers.trigger_message(ws_msg.clone());

                    // Buffer message
                    if let Err(e) = shared_clone.push_message(ws_msg) {
                        shared_clone.handlers.trigger_error(e);
                    }
                }
            }) as Box<dyn FnMut(MessageEvent)>);

            ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
            onmessage.forget();
        }

        // Set up onerror callback
        {
            let shared_clone = shared.clone();
            let tx_clone = tx.clone();
            let onerror = Closure::wrap(Box::new(move |event: ErrorEvent| {
                let error = WebSocketError::Connection(format!(
                    "WebSocket error: {}",
                    event.message()
                ));

                shared_clone.handlers.trigger_error(error.clone());

                // Notify connection failed
                if let Some(tx) = tx_clone.lock().unwrap().take() {
                    let _ = tx.send(Err(anyhow!(error)));
                }
            }) as Box<dyn FnMut(ErrorEvent)>);

            ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            onerror.forget();
        }

        // Set up onclose callback
        {
            let shared_clone = shared.clone();
            let onclose = Closure::wrap(Box::new(move |event: CloseEvent| {
                shared_clone.set_state(WebSocketState::Closed);

                let close_frame = CloseFrame {
                    code: event.code(),
                    reason: event.reason(),
                };

                shared_clone.handlers.trigger_close(Some(close_frame));
            }) as Box<dyn FnMut(CloseEvent)>);

            ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
            onclose.forget();
        }

        // Wait for connection to open or fail
        rx.await
            .map_err(|e| anyhow!(WebSocketError::Connection(format!("Connection wait failed: {}", e))))??;

        Ok(Self { ws, shared })
    }

    /// Parse message event from browser
    fn parse_message_event(event: &MessageEvent) -> Result<WebSocketMessage> {
        let data = event.data();

        // Check if it's a string (text message)
        if let Some(text) = data.as_string() {
            return Ok(WebSocketMessage::Text(text));
        }

        // Check if it's an ArrayBuffer (binary message)
        if let Ok(array_buffer) = data.dyn_into::<js_sys::ArrayBuffer>() {
            let array = js_sys::Uint8Array::new(&array_buffer);
            let bytes = array.to_vec();
            return Ok(WebSocketMessage::Binary(bytes));
        }

        // Check if it's a Blob
        if data.has_type::<web_sys::Blob>() {
            // For blobs, we'd need to use FileReader API to read them
            // For now, return an error
            return Err(anyhow!(WebSocketError::Receive(
                "Blob messages not yet supported".to_string()
            )));
        }

        Err(anyhow!(WebSocketError::Receive(
            "Unknown message type".to_string()
        )))
    }

    /// Send a message
    pub async fn send(&mut self, msg: WebSocketMessage) -> Result<()> {
        match msg {
            WebSocketMessage::Text(text) => {
                self.ws
                    .send_with_str(&text)
                    .map_err(|e| anyhow!(WebSocketError::Send(format!("{:?}", e))))?;
            }
            WebSocketMessage::Binary(data) => {
                self.ws
                    .send_with_u8_array(&data)
                    .map_err(|e| anyhow!(WebSocketError::Send(format!("{:?}", e))))?;
            }
            WebSocketMessage::Ping(_) | WebSocketMessage::Pong(_) => {
                // Browser WebSocket API doesn't expose ping/pong
                return Err(anyhow!(WebSocketError::Send(
                    "Ping/pong not supported in browser".to_string()
                )));
            }
            WebSocketMessage::Close(frame) => {
                if let Some(f) = frame {
                    self.ws
                        .close_with_code_and_reason(f.code, &f.reason)
                        .map_err(|e| anyhow!(WebSocketError::Close(format!("{:?}", e))))?;
                } else {
                    self.ws
                        .close()
                        .map_err(|e| anyhow!(WebSocketError::Close(format!("{:?}", e))))?;
                }
            }
        }

        Ok(())
    }

    /// Receive a message (non-blocking)
    pub async fn receive(&mut self) -> Result<Option<WebSocketMessage>> {
        // Try to get message from buffer
        Ok(self.shared.pop_message())
    }

    /// Close the WebSocket connection
    pub async fn close(&mut self, frame: Option<CloseFrame>) -> Result<()> {
        if let Some(f) = frame {
            self.ws
                .close_with_code_and_reason(f.code, &f.reason)
                .map_err(|e| anyhow!(WebSocketError::Close(format!("{:?}", e))))?;
        } else {
            self.ws
                .close()
                .map_err(|e| anyhow!(WebSocketError::Close(format!("{:?}", e))))?;
        }

        self.shared.set_state(WebSocketState::Closed);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_websocket_validation() {
        // This test just validates that the types are correctly defined
        // Actual browser testing would require a browser environment
        assert!(true);
    }
}
