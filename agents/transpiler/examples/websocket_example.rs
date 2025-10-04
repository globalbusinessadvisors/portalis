//! WebSocket Example
//!
//! This example demonstrates how to use the WebSocket API to connect to a server,
//! send messages, and receive responses.
//!
//! Usage:
//! ```bash
//! cargo run --example websocket_example
//! ```

use portalis_transpiler::wasi_websocket::{
    CloseFrame, WasiWebSocket, WebSocketConfig, WebSocketHandlers, WebSocketMessage,
};
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("WebSocket Example - Connecting to echo.websocket.org");
    println!("=====================================================\n");

    // Example 1: Simple connection and echo
    simple_echo_example().await?;

    // Example 2: Using event handlers
    event_handlers_example().await?;

    // Example 3: Binary messages
    binary_message_example().await?;

    // Example 4: Advanced configuration
    advanced_config_example().await?;

    Ok(())
}

/// Example 1: Simple connection and echo
async fn simple_echo_example() -> anyhow::Result<()> {
    println!("Example 1: Simple Echo");
    println!("----------------------");

    // Create a WebSocket configuration
    let config = WebSocketConfig::new("wss://echo.websocket.org");

    // Connect to the WebSocket server
    let mut ws = WasiWebSocket::connect(config).await?;

    println!("Connected to WebSocket server");
    println!("Connection state: {:?}", ws.state());

    // Send a text message
    let message = "Hello, WebSocket!";
    println!("Sending: {}", message);
    ws.send_text(message).await?;

    // Wait a bit for the echo response
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Receive the echo
    if let Some(msg) = ws.receive().await? {
        match msg {
            WebSocketMessage::Text(text) => {
                println!("Received: {}", text);
            }
            _ => println!("Received non-text message"),
        }
    }

    // Close the connection
    ws.close().await?;
    println!("Connection closed\n");

    Ok(())
}

/// Example 2: Using event handlers
async fn event_handlers_example() -> anyhow::Result<()> {
    println!("Example 2: Event Handlers");
    println!("-------------------------");

    let message_count = Arc::new(Mutex::new(0));
    let message_count_clone = Arc::clone(&message_count);

    // Create handlers
    let handlers = WebSocketHandlers::new()
        .on_open(|| {
            println!("✓ WebSocket opened!");
        })
        .on_message(move |msg| {
            *message_count_clone.lock().unwrap() += 1;
            match msg {
                WebSocketMessage::Text(text) => {
                    println!("✓ Message received: {}", text);
                }
                WebSocketMessage::Binary(data) => {
                    println!("✓ Binary message received: {} bytes", data.len());
                }
                _ => println!("✓ Other message type received"),
            }
        })
        .on_error(|err| {
            println!("✗ Error: {:?}", err);
        })
        .on_close(|frame| {
            if let Some(f) = frame {
                println!("✓ Connection closed: code={}, reason={}", f.code, f.reason);
            } else {
                println!("✓ Connection closed");
            }
        });

    // Create configuration
    let config = WebSocketConfig::new("wss://echo.websocket.org");

    // Connect with handlers
    let mut ws = WasiWebSocket::connect_with_handlers(config, handlers).await?;

    // Send multiple messages
    for i in 1..=3 {
        let msg = format!("Message {}", i);
        println!("Sending: {}", msg);
        ws.send_text(&msg).await?;
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    }

    // Wait for all echoes
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    println!("Total messages received: {}", *message_count.lock().unwrap());

    // Close with custom frame
    ws.close_with_frame(CloseFrame::new(1000, "Example complete"))
        .await?;

    println!();

    Ok(())
}

/// Example 3: Binary messages
async fn binary_message_example() -> anyhow::Result<()> {
    println!("Example 3: Binary Messages");
    println!("--------------------------");

    let config = WebSocketConfig::new("wss://echo.websocket.org");
    let mut ws = WasiWebSocket::connect(config).await?;

    // Send binary data
    let binary_data = vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]; // "Hello" in ASCII
    println!("Sending binary data: {:?}", binary_data);
    ws.send_binary(binary_data.clone()).await?;

    // Wait for echo
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Receive echo
    if let Some(msg) = ws.receive().await? {
        match msg {
            WebSocketMessage::Binary(data) => {
                println!("Received binary data: {:?}", data);
                println!("As string: {}", String::from_utf8_lossy(&data));
            }
            _ => println!("Received non-binary message"),
        }
    }

    ws.close().await?;
    println!();

    Ok(())
}

/// Example 4: Advanced configuration
async fn advanced_config_example() -> anyhow::Result<()> {
    println!("Example 4: Advanced Configuration");
    println!("----------------------------------");

    // Create advanced configuration
    let config = WebSocketConfig::new("wss://echo.websocket.org")
        .with_subprotocol("chat")
        .with_auto_reconnect(true)
        .with_heartbeat(true, 30)
        .with_buffer_size(200);

    println!("Configuration:");
    println!("  URL: {}", config.url);
    println!("  Subprotocols: {:?}", config.subprotocols);
    println!("  Auto-reconnect: {}", config.auto_reconnect);
    println!("  Heartbeat: {} (interval: {}s)", config.enable_heartbeat, config.heartbeat_interval_secs);
    println!("  Buffer size: {}", config.buffer_size);

    let mut ws = WasiWebSocket::connect(config).await?;

    println!("\nConnected with advanced configuration");

    // Send a message
    ws.send_text("Testing advanced config").await?;

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    if let Some(msg) = ws.receive().await? {
        match msg {
            WebSocketMessage::Text(text) => {
                println!("Echo received: {}", text);
            }
            _ => {}
        }
    }

    ws.close().await?;
    println!();

    Ok(())
}
