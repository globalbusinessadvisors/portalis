//! Native WebSocket implementation using tokio-tungstenite
//!
//! This module provides WebSocket support for native Rust targets using
//! the tokio-tungstenite library, which is a robust, async WebSocket
//! implementation built on top of tokio.

use super::{
    CloseFrame, WebSocketConfig, WebSocketError, WebSocketMessage, WebSocketSharedState,
    WebSocketState,
};
use anyhow::{anyhow, Context, Result};
use std::sync::Arc;
use tokio::sync::Mutex;

// Conditionally compile tokio-tungstenite imports
#[cfg(not(target_arch = "wasm32"))]
use tokio_tungstenite::{
    connect_async, tungstenite::Message as TungsteniteMessage, MaybeTlsStream, WebSocketStream,
};

#[cfg(not(target_arch = "wasm32"))]
use futures_util::{SinkExt, StreamExt};

#[cfg(not(target_arch = "wasm32"))]
use tokio::net::TcpStream;

/// Native WebSocket implementation
pub struct NativeWebSocket {
    stream: Arc<Mutex<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    shared: WebSocketSharedState,
}

impl NativeWebSocket {
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

        // Build connection request
        // Note: tokio-tungstenite doesn't directly support subprotocols in the URL
        // They need to be added via headers in the handshake request
        // For now, we'll connect without subprotocols and add support later

        // Connect to WebSocket server
        let (ws_stream, response) = connect_async(&config.url)
            .await
            .context("Failed to connect to WebSocket server")
            .map_err(|e| anyhow!(WebSocketError::Connection(e.to_string())))?;

        tracing::info!(
            "WebSocket connected to {}: status={}",
            config.url,
            response.status()
        );

        // Update state to Open
        shared.set_state(WebSocketState::Open);

        // Trigger on_open callback
        shared.handlers.trigger_open();

        let ws = Self {
            stream: Arc::new(Mutex::new(Some(ws_stream))),
            shared: shared.clone(),
        };

        // Start message receiving task
        if config.enable_heartbeat {
            ws.start_heartbeat_task(config.heartbeat_interval_secs);
        }

        ws.start_receive_task();

        Ok(ws)
    }

    /// Start background task to receive messages
    fn start_receive_task(&self) {
        let stream = Arc::clone(&self.stream);
        let shared = self.shared.clone();

        tokio::spawn(async move {
            loop {
                let mut stream_guard = stream.lock().await;

                if let Some(ws_stream) = stream_guard.as_mut() {
                    match ws_stream.next().await {
                        Some(Ok(msg)) => {
                            drop(stream_guard); // Release lock before processing

                            // Convert tungstenite message to our message type
                            let ws_msg = Self::convert_message(msg);

                            // Trigger message callback
                            shared.handlers.trigger_message(ws_msg.clone());

                            // Buffer message for receive() calls
                            if let Err(e) = shared.push_message(ws_msg) {
                                shared.handlers.trigger_error(e);
                            }
                        }
                        Some(Err(e)) => {
                            drop(stream_guard);
                            let error = WebSocketError::Receive(e.to_string());
                            shared.handlers.trigger_error(error);
                            break;
                        }
                        None => {
                            drop(stream_guard);
                            // Connection closed
                            shared.set_state(WebSocketState::Closed);
                            shared.handlers.trigger_close(None);
                            break;
                        }
                    }
                } else {
                    break;
                }
            }
        });
    }

    /// Start heartbeat task to send periodic pings
    fn start_heartbeat_task(&self, interval_secs: u64) {
        let stream = Arc::clone(&self.stream);
        let shared = self.shared.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));

            loop {
                interval.tick().await;

                // Check if connection is still open
                if shared.get_state() != WebSocketState::Open {
                    break;
                }

                let mut stream_guard = stream.lock().await;
                if let Some(ws_stream) = stream_guard.as_mut() {
                    let ping_msg = TungsteniteMessage::Ping(vec![]);
                    if let Err(e) = ws_stream.send(ping_msg).await {
                        drop(stream_guard);
                        let error = WebSocketError::Send(format!("Heartbeat ping failed: {}", e));
                        shared.handlers.trigger_error(error);
                        break;
                    }
                } else {
                    break;
                }
            }
        });
    }

    /// Convert tungstenite message to our message type
    fn convert_message(msg: TungsteniteMessage) -> WebSocketMessage {
        match msg {
            TungsteniteMessage::Text(text) => WebSocketMessage::Text(text),
            TungsteniteMessage::Binary(data) => WebSocketMessage::Binary(data),
            TungsteniteMessage::Ping(data) => WebSocketMessage::Ping(data),
            TungsteniteMessage::Pong(data) => WebSocketMessage::Pong(data),
            TungsteniteMessage::Close(frame) => {
                let close_frame = frame.map(|f| CloseFrame {
                    code: f.code.into(),
                    reason: f.reason.to_string(),
                });
                WebSocketMessage::Close(close_frame)
            }
            TungsteniteMessage::Frame(_) => {
                // Raw frames are not exposed in our API
                WebSocketMessage::Binary(vec![])
            }
        }
    }

    /// Convert our message type to tungstenite message
    fn convert_to_tungstenite(msg: WebSocketMessage) -> TungsteniteMessage {
        match msg {
            WebSocketMessage::Text(text) => TungsteniteMessage::Text(text),
            WebSocketMessage::Binary(data) => TungsteniteMessage::Binary(data),
            WebSocketMessage::Ping(data) => TungsteniteMessage::Ping(data),
            WebSocketMessage::Pong(data) => TungsteniteMessage::Pong(data),
            WebSocketMessage::Close(frame) => {
                let close_frame = frame.map(|f| {
                    tokio_tungstenite::tungstenite::protocol::CloseFrame {
                        code: f.code.into(),
                        reason: f.reason.into(),
                    }
                });
                TungsteniteMessage::Close(close_frame)
            }
        }
    }

    /// Send a message
    pub async fn send(&mut self, msg: WebSocketMessage) -> Result<()> {
        let mut stream_guard = self.stream.lock().await;

        if let Some(ws_stream) = stream_guard.as_mut() {
            let tungstenite_msg = Self::convert_to_tungstenite(msg);
            ws_stream
                .send(tungstenite_msg)
                .await
                .context("Failed to send WebSocket message")
                .map_err(|e| anyhow!(WebSocketError::Send(e.to_string())))?;
            Ok(())
        } else {
            Err(anyhow!(WebSocketError::InvalidState(
                "WebSocket stream is not available".to_string()
            )))
        }
    }

    /// Receive a message (non-blocking)
    pub async fn receive(&mut self) -> Result<Option<WebSocketMessage>> {
        // Try to get message from buffer first
        if let Some(msg) = self.shared.pop_message() {
            return Ok(Some(msg));
        }

        // No buffered messages
        Ok(None)
    }

    /// Close the WebSocket connection
    pub async fn close(&mut self, frame: Option<CloseFrame>) -> Result<()> {
        let mut stream_guard = self.stream.lock().await;

        if let Some(mut ws_stream) = stream_guard.take() {
            let close_msg = WebSocketMessage::Close(frame.clone());
            let tungstenite_msg = Self::convert_to_tungstenite(close_msg);

            ws_stream
                .send(tungstenite_msg)
                .await
                .context("Failed to send close frame")
                .map_err(|e| anyhow!(WebSocketError::Close(e.to_string())))?;

            ws_stream
                .close(None)
                .await
                .context("Failed to close WebSocket stream")
                .map_err(|e| anyhow!(WebSocketError::Close(e.to_string())))?;

            self.shared.set_state(WebSocketState::Closed);
            self.shared.handlers.trigger_close(frame);

            Ok(())
        } else {
            // Already closed
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_conversion() {
        let text = TungsteniteMessage::Text("hello".to_string());
        let converted = NativeWebSocket::convert_message(text);
        assert!(matches!(converted, WebSocketMessage::Text(_)));

        let binary = TungsteniteMessage::Binary(vec![1, 2, 3]);
        let converted = NativeWebSocket::convert_message(binary);
        assert!(matches!(converted, WebSocketMessage::Binary(_)));
    }

    #[test]
    fn test_invalid_url() {
        let config = WebSocketConfig::new("http://example.com");
        let shared = WebSocketSharedState::new(
            config.clone(),
            super::super::WebSocketHandlers::new(),
        );

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            NativeWebSocket::connect(config, shared).await
        });

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid WebSocket URL"));
        }
    }
}
