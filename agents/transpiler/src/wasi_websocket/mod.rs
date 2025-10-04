//! WASI WebSocket Integration
//!
//! Provides a unified WebSocket API that works across:
//! - Native Rust (tokio-tungstenite)
//! - Browser WASM (WebSocket API via wasm-bindgen)
//! - WASI WASM (tokio-tungstenite or fallback)
//!
//! This module bridges Python's WebSocket operations to WASM-compatible implementations.

use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

/// WebSocket connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSocketState {
    /// Connection is being established
    Connecting,
    /// Connection is open and ready
    Open,
    /// Connection is closing
    Closing,
    /// Connection is closed
    Closed,
}

/// WebSocket message type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebSocketMessage {
    /// Text message (UTF-8 encoded)
    Text(String),
    /// Binary message
    Binary(Vec<u8>),
    /// Ping frame
    Ping(Vec<u8>),
    /// Pong frame
    Pong(Vec<u8>),
    /// Close frame
    Close(Option<CloseFrame>),
}

/// WebSocket close frame
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloseFrame {
    /// Close code (1000 = normal closure)
    pub code: u16,
    /// Close reason
    pub reason: String,
}

impl CloseFrame {
    /// Create a normal close frame (code 1000)
    pub fn normal() -> Self {
        Self {
            code: 1000,
            reason: "Normal closure".to_string(),
        }
    }

    /// Create a close frame with custom code and reason
    pub fn new(code: u16, reason: impl Into<String>) -> Self {
        Self {
            code,
            reason: reason.into(),
        }
    }
}

/// WebSocket configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// WebSocket URL (ws:// or wss://)
    pub url: String,
    /// Optional subprotocols
    pub subprotocols: Vec<String>,
    /// Custom headers for handshake (native only)
    pub headers: Vec<(String, String)>,
    /// Enable automatic reconnection
    pub auto_reconnect: bool,
    /// Reconnection delay in milliseconds
    pub reconnect_delay_ms: u64,
    /// Maximum reconnection attempts (0 = unlimited)
    pub max_reconnect_attempts: u32,
    /// Enable ping/pong heartbeat
    pub enable_heartbeat: bool,
    /// Heartbeat interval in seconds
    pub heartbeat_interval_secs: u64,
    /// Message buffer size
    pub buffer_size: usize,
}

impl WebSocketConfig {
    /// Create a new WebSocket configuration with default settings
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            subprotocols: Vec::new(),
            headers: Vec::new(),
            auto_reconnect: false,
            reconnect_delay_ms: 1000,
            max_reconnect_attempts: 5,
            enable_heartbeat: true,
            heartbeat_interval_secs: 30,
            buffer_size: 100,
        }
    }

    /// Add a subprotocol
    pub fn with_subprotocol(mut self, subprotocol: impl Into<String>) -> Self {
        self.subprotocols.push(subprotocol.into());
        self
    }

    /// Add a custom header (native only)
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }

    /// Enable automatic reconnection
    pub fn with_auto_reconnect(mut self, enabled: bool) -> Self {
        self.auto_reconnect = enabled;
        self
    }

    /// Set heartbeat configuration
    pub fn with_heartbeat(mut self, enabled: bool, interval_secs: u64) -> Self {
        self.enable_heartbeat = enabled;
        self.heartbeat_interval_secs = interval_secs;
        self
    }

    /// Set message buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
}

/// WebSocket error types
#[derive(Debug, thiserror::Error)]
pub enum WebSocketError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Handshake error: {0}")]
    Handshake(String),

    #[error("Send error: {0}")]
    Send(String),

    #[error("Receive error: {0}")]
    Receive(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Close error: {0}")]
    Close(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Buffer overflow")]
    BufferOverflow,
}

/// Event handler callback types
pub type OnOpenCallback = Arc<dyn Fn() + Send + Sync>;
pub type OnMessageCallback = Arc<dyn Fn(WebSocketMessage) + Send + Sync>;
pub type OnErrorCallback = Arc<dyn Fn(WebSocketError) + Send + Sync>;
pub type OnCloseCallback = Arc<dyn Fn(Option<CloseFrame>) + Send + Sync>;

/// Event handlers for WebSocket
#[derive(Clone)]
pub struct WebSocketHandlers {
    on_open: Option<OnOpenCallback>,
    on_message: Option<OnMessageCallback>,
    on_error: Option<OnErrorCallback>,
    on_close: Option<OnCloseCallback>,
}

impl WebSocketHandlers {
    /// Create empty handlers
    pub fn new() -> Self {
        Self {
            on_open: None,
            on_message: None,
            on_error: None,
            on_close: None,
        }
    }

    /// Set connection opened callback
    pub fn on_open<F>(mut self, callback: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.on_open = Some(Arc::new(callback));
        self
    }

    /// Set message received callback
    pub fn on_message<F>(mut self, callback: F) -> Self
    where
        F: Fn(WebSocketMessage) + Send + Sync + 'static,
    {
        self.on_message = Some(Arc::new(callback));
        self
    }

    /// Set error callback
    pub fn on_error<F>(mut self, callback: F) -> Self
    where
        F: Fn(WebSocketError) + Send + Sync + 'static,
    {
        self.on_error = Some(Arc::new(callback));
        self
    }

    /// Set connection closed callback
    pub fn on_close<F>(mut self, callback: F) -> Self
    where
        F: Fn(Option<CloseFrame>) + Send + Sync + 'static,
    {
        self.on_close = Some(Arc::new(callback));
        self
    }

    /// Trigger on_open callback
    #[cfg(not(test))]
    pub(crate) fn trigger_open(&self) {
        if let Some(ref callback) = self.on_open {
            callback();
        }
    }

    /// Trigger on_open callback (pub for testing)
    #[cfg(test)]
    pub fn trigger_open(&self) {
        if let Some(ref callback) = self.on_open {
            callback();
        }
    }

    /// Trigger on_message callback
    #[cfg(not(test))]
    pub(crate) fn trigger_message(&self, msg: WebSocketMessage) {
        if let Some(ref callback) = self.on_message {
            callback(msg);
        }
    }

    /// Trigger on_message callback (pub for testing)
    #[cfg(test)]
    pub fn trigger_message(&self, msg: WebSocketMessage) {
        if let Some(ref callback) = self.on_message {
            callback(msg);
        }
    }

    /// Trigger on_error callback
    #[cfg(not(test))]
    pub(crate) fn trigger_error(&self, err: WebSocketError) {
        if let Some(ref callback) = self.on_error {
            callback(err);
        }
    }

    /// Trigger on_error callback (pub for testing)
    #[cfg(test)]
    pub fn trigger_error(&self, err: WebSocketError) {
        if let Some(ref callback) = self.on_error {
            callback(err);
        }
    }

    /// Trigger on_close callback
    #[cfg(not(test))]
    pub(crate) fn trigger_close(&self, frame: Option<CloseFrame>) {
        if let Some(ref callback) = self.on_close {
            callback(frame);
        }
    }

    /// Trigger on_close callback (pub for testing)
    #[cfg(test)]
    pub fn trigger_close(&self, frame: Option<CloseFrame>) {
        if let Some(ref callback) = self.on_close {
            callback(frame);
        }
    }
}

impl Default for WebSocketHandlers {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared WebSocket state
#[derive(Clone)]
pub(crate) struct WebSocketSharedState {
    state: Arc<Mutex<WebSocketState>>,
    message_buffer: Arc<Mutex<VecDeque<WebSocketMessage>>>,
    pub(crate) handlers: Arc<WebSocketHandlers>,
    pub(crate) config: Arc<WebSocketConfig>,
}

impl WebSocketSharedState {
    pub(crate) fn new(config: WebSocketConfig, handlers: WebSocketHandlers) -> Self {
        Self {
            state: Arc::new(Mutex::new(WebSocketState::Connecting)),
            message_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_size))),
            handlers: Arc::new(handlers),
            config: Arc::new(config),
        }
    }

    pub(crate) fn get_state(&self) -> WebSocketState {
        *self.state.lock().unwrap()
    }

    pub(crate) fn set_state(&self, new_state: WebSocketState) {
        *self.state.lock().unwrap() = new_state;
    }

    pub(crate) fn push_message(&self, msg: WebSocketMessage) -> Result<(), WebSocketError> {
        let mut buffer = self.message_buffer.lock().unwrap();
        if buffer.len() >= self.config.buffer_size {
            return Err(WebSocketError::BufferOverflow);
        }
        buffer.push_back(msg);
        Ok(())
    }

    pub(crate) fn pop_message(&self) -> Option<WebSocketMessage> {
        self.message_buffer.lock().unwrap().pop_front()
    }
}

// Platform-specific implementations
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub(crate) mod browser;

#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
pub(crate) mod wasi_impl;

// Re-export platform-specific implementation
#[cfg(not(target_arch = "wasm32"))]
use native::NativeWebSocket as PlatformWebSocket;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use browser::BrowserWebSocket as PlatformWebSocket;

#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
use wasi_impl::WasiWebSocketImpl as PlatformWebSocket;

/// Unified WebSocket handle that works across platforms
pub struct WasiWebSocket {
    inner: PlatformWebSocket,
    shared: WebSocketSharedState,
}

impl WasiWebSocket {
    /// Connect to a WebSocket server with default configuration
    pub async fn connect_url(url: impl Into<String>) -> Result<Self> {
        let config = WebSocketConfig::new(url);
        Self::connect(config).await
    }

    /// Connect to a WebSocket server with custom configuration
    pub async fn connect(config: WebSocketConfig) -> Result<Self> {
        Self::connect_with_handlers(config, WebSocketHandlers::new()).await
    }

    /// Connect to a WebSocket server with configuration and event handlers
    pub async fn connect_with_handlers(
        config: WebSocketConfig,
        handlers: WebSocketHandlers,
    ) -> Result<Self> {
        let shared = WebSocketSharedState::new(config.clone(), handlers);

        #[cfg(not(target_arch = "wasm32"))]
        let inner = native::NativeWebSocket::connect(config, shared.clone()).await?;

        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        let inner = browser::BrowserWebSocket::connect(config, shared.clone()).await?;

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        let inner = wasi_impl::WasiWebSocketImpl::connect(config, shared.clone()).await?;

        Ok(Self { inner, shared })
    }

    /// Get current connection state
    pub fn state(&self) -> WebSocketState {
        self.shared.get_state()
    }

    /// Check if connection is open
    pub fn is_open(&self) -> bool {
        self.state() == WebSocketState::Open
    }

    /// Send a text message
    pub async fn send_text(&mut self, text: impl Into<String>) -> Result<()> {
        if !self.is_open() {
            return Err(anyhow!(WebSocketError::InvalidState(
                "WebSocket is not open".to_string()
            )));
        }
        self.inner.send(WebSocketMessage::Text(text.into())).await
    }

    /// Send a binary message
    pub async fn send_binary(&mut self, data: Vec<u8>) -> Result<()> {
        if !self.is_open() {
            return Err(anyhow!(WebSocketError::InvalidState(
                "WebSocket is not open".to_string()
            )));
        }
        self.inner.send(WebSocketMessage::Binary(data)).await
    }

    /// Send a ping frame
    pub async fn send_ping(&mut self, data: Vec<u8>) -> Result<()> {
        if !self.is_open() {
            return Err(anyhow!(WebSocketError::InvalidState(
                "WebSocket is not open".to_string()
            )));
        }
        self.inner.send(WebSocketMessage::Ping(data)).await
    }

    /// Send a pong frame
    pub async fn send_pong(&mut self, data: Vec<u8>) -> Result<()> {
        if !self.is_open() {
            return Err(anyhow!(WebSocketError::InvalidState(
                "WebSocket is not open".to_string()
            )));
        }
        self.inner.send(WebSocketMessage::Pong(data)).await
    }

    /// Receive the next message (non-blocking, returns None if no message available)
    pub async fn receive(&mut self) -> Result<Option<WebSocketMessage>> {
        self.inner.receive().await
    }

    /// Close the WebSocket connection
    pub async fn close(&mut self) -> Result<()> {
        self.close_with_frame(CloseFrame::normal()).await
    }

    /// Close the WebSocket connection with a custom close frame
    pub async fn close_with_frame(&mut self, frame: CloseFrame) -> Result<()> {
        if self.shared.get_state() == WebSocketState::Closed {
            return Ok(());
        }

        self.shared.set_state(WebSocketState::Closing);
        self.inner.close(Some(frame)).await?;
        self.shared.set_state(WebSocketState::Closed);

        Ok(())
    }

    /// Get a stream of incoming messages
    #[cfg(not(target_arch = "wasm32"))]
    pub fn into_stream(self) -> WebSocketStream {
        WebSocketStream {
            inner: self.inner,
            shared: self.shared,
        }
    }
}

/// Stream-based API for receiving messages (native only)
#[cfg(not(target_arch = "wasm32"))]
pub struct WebSocketStream {
    inner: PlatformWebSocket,
    shared: WebSocketSharedState,
}

#[cfg(not(target_arch = "wasm32"))]
impl WebSocketStream {
    /// Receive the next message from the stream
    pub async fn next(&mut self) -> Result<Option<WebSocketMessage>> {
        self.inner.receive().await
    }

    /// Close the stream
    pub async fn close(mut self) -> Result<()> {
        self.shared.set_state(WebSocketState::Closing);
        self.inner.close(Some(CloseFrame::normal())).await?;
        self.shared.set_state(WebSocketState::Closed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_state() {
        assert_eq!(WebSocketState::Connecting, WebSocketState::Connecting);
        assert_ne!(WebSocketState::Open, WebSocketState::Closed);
    }

    #[test]
    fn test_close_frame() {
        let frame = CloseFrame::normal();
        assert_eq!(frame.code, 1000);
        assert_eq!(frame.reason, "Normal closure");

        let custom = CloseFrame::new(1001, "Going away");
        assert_eq!(custom.code, 1001);
        assert_eq!(custom.reason, "Going away");
    }

    #[test]
    fn test_websocket_config() {
        let config = WebSocketConfig::new("wss://example.com")
            .with_subprotocol("chat")
            .with_header("Authorization", "Bearer token")
            .with_auto_reconnect(true)
            .with_heartbeat(true, 60)
            .with_buffer_size(200);

        assert_eq!(config.url, "wss://example.com");
        assert_eq!(config.subprotocols.len(), 1);
        assert_eq!(config.headers.len(), 1);
        assert!(config.auto_reconnect);
        assert!(config.enable_heartbeat);
        assert_eq!(config.heartbeat_interval_secs, 60);
        assert_eq!(config.buffer_size, 200);
    }

    #[test]
    fn test_websocket_message() {
        let text_msg = WebSocketMessage::Text("Hello".to_string());
        assert!(matches!(text_msg, WebSocketMessage::Text(_)));

        let binary_msg = WebSocketMessage::Binary(vec![1, 2, 3]);
        assert!(matches!(binary_msg, WebSocketMessage::Binary(_)));

        let ping_msg = WebSocketMessage::Ping(vec![]);
        assert!(matches!(ping_msg, WebSocketMessage::Ping(_)));

        let close_msg = WebSocketMessage::Close(Some(CloseFrame::normal()));
        assert!(matches!(close_msg, WebSocketMessage::Close(_)));
    }

    #[test]
    fn test_shared_state() {
        let config = WebSocketConfig::new("ws://localhost");
        let handlers = WebSocketHandlers::new();
        let shared = WebSocketSharedState::new(config, handlers);

        assert_eq!(shared.get_state(), WebSocketState::Connecting);

        shared.set_state(WebSocketState::Open);
        assert_eq!(shared.get_state(), WebSocketState::Open);

        let msg = WebSocketMessage::Text("test".to_string());
        assert!(shared.push_message(msg.clone()).is_ok());

        let popped = shared.pop_message();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap(), msg);
    }

    #[test]
    fn test_buffer_overflow() {
        let config = WebSocketConfig::new("ws://localhost").with_buffer_size(2);
        let handlers = WebSocketHandlers::new();
        let shared = WebSocketSharedState::new(config, handlers);

        let msg1 = WebSocketMessage::Text("msg1".to_string());
        let msg2 = WebSocketMessage::Text("msg2".to_string());
        let msg3 = WebSocketMessage::Text("msg3".to_string());

        assert!(shared.push_message(msg1).is_ok());
        assert!(shared.push_message(msg2).is_ok());
        assert!(shared.push_message(msg3).is_err());
    }
}
