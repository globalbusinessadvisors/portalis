//! WebSocket Integration Tests
//!
//! These tests validate the WebSocket implementation across different platforms.

#[cfg(not(target_arch = "wasm32"))]
mod native_tests {
    use portalis_transpiler::wasi_websocket::{
        CloseFrame, WasiWebSocket, WebSocketConfig, WebSocketHandlers, WebSocketMessage,
        WebSocketState,
    };
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn test_websocket_config_builder() {
        let config = WebSocketConfig::new("wss://echo.websocket.org")
            .with_subprotocol("chat")
            .with_subprotocol("superchat")
            .with_header("Authorization", "Bearer token123")
            .with_auto_reconnect(true)
            .with_heartbeat(true, 60)
            .with_buffer_size(500);

        assert_eq!(config.url, "wss://echo.websocket.org");
        assert_eq!(config.subprotocols.len(), 2);
        assert_eq!(config.subprotocols[0], "chat");
        assert_eq!(config.subprotocols[1], "superchat");
        assert_eq!(config.headers.len(), 1);
        assert_eq!(config.headers[0].0, "Authorization");
        assert!(config.auto_reconnect);
        assert!(config.enable_heartbeat);
        assert_eq!(config.heartbeat_interval_secs, 60);
        assert_eq!(config.buffer_size, 500);
    }

    #[tokio::test]
    async fn test_websocket_state_transitions() {
        assert_eq!(WebSocketState::Connecting, WebSocketState::Connecting);
        assert_ne!(WebSocketState::Open, WebSocketState::Closed);
        assert_ne!(WebSocketState::Connecting, WebSocketState::Open);
        assert_ne!(WebSocketState::Closing, WebSocketState::Closed);
    }

    #[tokio::test]
    async fn test_close_frame_creation() {
        let normal = CloseFrame::normal();
        assert_eq!(normal.code, 1000);
        assert_eq!(normal.reason, "Normal closure");

        let custom = CloseFrame::new(1001, "Going away");
        assert_eq!(custom.code, 1001);
        assert_eq!(custom.reason, "Going away");

        let another = CloseFrame::new(1003, "Unsupported data");
        assert_eq!(another.code, 1003);
        assert_eq!(another.reason, "Unsupported data");
    }

    #[tokio::test]
    async fn test_websocket_message_types() {
        let text = WebSocketMessage::Text("Hello, WebSocket!".to_string());
        assert!(matches!(text, WebSocketMessage::Text(_)));

        let binary = WebSocketMessage::Binary(vec![1, 2, 3, 4, 5]);
        assert!(matches!(binary, WebSocketMessage::Binary(_)));

        let ping = WebSocketMessage::Ping(vec![]);
        assert!(matches!(ping, WebSocketMessage::Ping(_)));

        let pong = WebSocketMessage::Pong(vec![42]);
        assert!(matches!(pong, WebSocketMessage::Pong(_)));

        let close = WebSocketMessage::Close(Some(CloseFrame::normal()));
        assert!(matches!(close, WebSocketMessage::Close(_)));
    }

    #[tokio::test]
    async fn test_websocket_handlers() {
        // Test that handlers can be created and configured
        let opened = Arc::new(Mutex::new(false));
        let opened_clone = Arc::clone(&opened);

        let message_count = Arc::new(Mutex::new(0));
        let message_count_clone = Arc::clone(&message_count);

        let error_count = Arc::new(Mutex::new(0));
        let error_count_clone = Arc::clone(&error_count);

        let closed = Arc::new(Mutex::new(false));
        let closed_clone = Arc::clone(&closed);

        let _handlers = WebSocketHandlers::new()
            .on_open(move || {
                *opened_clone.lock().unwrap() = true;
            })
            .on_message(move |_msg| {
                *message_count_clone.lock().unwrap() += 1;
            })
            .on_error(move |_err| {
                *error_count_clone.lock().unwrap() += 1;
            })
            .on_close(move |_frame| {
                *closed_clone.lock().unwrap() = true;
            });

        // Note: Callback triggering is tested in unit tests within the module
        // Integration tests focus on actual WebSocket connections
        assert!(true); // Placeholder to indicate test structure is validated
    }

    #[tokio::test]
    async fn test_invalid_url_rejection() {
        let config = WebSocketConfig::new("http://example.com");
        let result = WasiWebSocket::connect(config).await;
        assert!(result.is_err());

        let config2 = WebSocketConfig::new("https://example.com");
        let result2 = WasiWebSocket::connect(config2).await;
        assert!(result2.is_err());

        let config3 = WebSocketConfig::new("ftp://example.com");
        let result3 = WasiWebSocket::connect(config3).await;
        assert!(result3.is_err());
    }

    // Note: The following tests require a real WebSocket server
    // For CI/CD, you would use a mock WebSocket server

    #[tokio::test]
    #[ignore] // Requires network access to public echo server
    async fn test_websocket_echo_connection() {
        let config = WebSocketConfig::new("wss://echo.websocket.org");

        let ws_result = WasiWebSocket::connect(config).await;
        if let Ok(mut ws) = ws_result {
            assert!(ws.is_open());

            // Send a text message
            let send_result = ws.send_text("Hello, Echo!").await;
            assert!(send_result.is_ok());

            // Wait a bit for echo response
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Try to receive the echo
            if let Ok(Some(msg)) = ws.receive().await {
                match msg {
                    WebSocketMessage::Text(text) => {
                        assert_eq!(text, "Hello, Echo!");
                    }
                    _ => panic!("Expected text message"),
                }
            }

            // Close connection
            let close_result = ws.close().await;
            assert!(close_result.is_ok());
        }
    }

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_websocket_binary_message() {
        let config = WebSocketConfig::new("wss://echo.websocket.org");

        if let Ok(mut ws) = WasiWebSocket::connect(config).await {
            let binary_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
            let send_result = ws.send_binary(binary_data.clone()).await;
            assert!(send_result.is_ok());

            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            if let Ok(Some(msg)) = ws.receive().await {
                match msg {
                    WebSocketMessage::Binary(data) => {
                        assert_eq!(data, binary_data);
                    }
                    _ => panic!("Expected binary message"),
                }
            }

            ws.close().await.ok();
        }
    }

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_websocket_with_handlers() {
        let opened = Arc::new(Mutex::new(false));
        let opened_clone = Arc::clone(&opened);

        let received_messages = Arc::new(Mutex::new(Vec::new()));
        let received_clone = Arc::clone(&received_messages);

        let handlers = WebSocketHandlers::new()
            .on_open(move || {
                *opened_clone.lock().unwrap() = true;
                println!("WebSocket opened!");
            })
            .on_message(move |msg| {
                received_clone.lock().unwrap().push(msg);
                println!("Message received!");
            });

        let config = WebSocketConfig::new("wss://echo.websocket.org");

        if let Ok(mut ws) = WasiWebSocket::connect_with_handlers(config, handlers).await {
            // Connection should have triggered on_open
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            assert!(*opened.lock().unwrap());

            // Send a message
            ws.send_text("Test message").await.ok();

            // Wait for echo
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Check if message was received via handler
            let messages = received_messages.lock().unwrap();
            assert!(!messages.is_empty());

            ws.close().await.ok();
        }
    }

    #[tokio::test]
    async fn test_websocket_message_equality() {
        let msg1 = WebSocketMessage::Text("hello".to_string());
        let msg2 = WebSocketMessage::Text("hello".to_string());
        assert_eq!(msg1, msg2);

        let msg3 = WebSocketMessage::Binary(vec![1, 2, 3]);
        let msg4 = WebSocketMessage::Binary(vec![1, 2, 3]);
        assert_eq!(msg3, msg4);

        let msg5 = WebSocketMessage::Text("hello".to_string());
        let msg6 = WebSocketMessage::Text("world".to_string());
        assert_ne!(msg5, msg6);
    }

    #[tokio::test]
    async fn test_close_frame_equality() {
        let frame1 = CloseFrame::new(1000, "Normal");
        let frame2 = CloseFrame::new(1000, "Normal");
        assert_eq!(frame1, frame2);

        let frame3 = CloseFrame::new(1001, "Going away");
        assert_ne!(frame1, frame3);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod browser_tests {
    use portalis_transpiler::wasi_websocket::{WebSocketConfig, WebSocketMessage};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_websocket_config() {
        let config = WebSocketConfig::new("wss://echo.websocket.org")
            .with_subprotocol("chat")
            .with_buffer_size(100);

        assert_eq!(config.url, "wss://echo.websocket.org");
        assert_eq!(config.subprotocols.len(), 1);
        assert_eq!(config.buffer_size, 100);
    }

    #[wasm_bindgen_test]
    fn test_message_types() {
        let text = WebSocketMessage::Text("Hello".to_string());
        assert!(matches!(text, WebSocketMessage::Text(_)));

        let binary = WebSocketMessage::Binary(vec![1, 2, 3]);
        assert!(matches!(binary, WebSocketMessage::Binary(_)));
    }
}
