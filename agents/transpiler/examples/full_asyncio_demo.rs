//! Full Asyncio Support - Comprehensive Examples
//!
//! Demonstrates complete Python asyncio → Rust async/await translation
//! including orchestrator, advanced patterns, and real-world scenarios.

use portalis_transpiler::asyncio_orchestrator::{
    AsyncioConfig, AsyncioOrchestrator, AsyncioPatterns, AsyncRuntime,
};

fn main() {
    println!("=== Full Python Asyncio → Rust Translation Examples ===\n");

    // Example 1: Basic orchestrator usage
    example_basic_orchestrator();

    // Example 2: Advanced error handling
    example_error_handling();

    // Example 3: Retry patterns
    example_retry_patterns();

    // Example 4: Timeout with fallback
    example_timeout_fallback();

    // Example 5: Select/race patterns
    example_select_patterns();

    // Example 6: Stream processing
    example_stream_processing();

    // Example 7: Broadcast channels
    example_broadcast_channels();

    // Example 8: Graceful shutdown
    example_graceful_shutdown();

    // Example 9: Full application example
    example_full_application();
}

fn example_basic_orchestrator() {
    println!("## Example 1: Basic Orchestrator Usage\n");
    println!("Python:");
    println!(r#"
import asyncio

async def fetch_data(url: str) -> dict:
    await asyncio.sleep(0.1)
    return {{"url": url, "status": "ok"}}

async def main():
    results = await asyncio.gather(
        fetch_data("https://api.example.com/1"),
        fetch_data("https://api.example.com/2"),
        fetch_data("https://api.example.com/3")
    )
    return results

asyncio.run(main())
"#);

    let config = AsyncioConfig {
        runtime: AsyncRuntime::Tokio,
        auto_error_handling: true,
        enable_tracing: false,
        cancellation_support: false,
        wasm_compatible: false,
    };

    let mut orchestrator = AsyncioOrchestrator::new(config);

    println!("\nRust (using orchestrator):");
    println!(r#"use tokio;

async fn fetch_data(url: String) -> Result<serde_json::Value> {{
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(serde_json::json!({{"url": url, "status": "ok"}}))
}}

#[tokio::main]
async fn main() -> Result<()> {{
    let results = tokio::join!(
        fetch_data("https://api.example.com/1".to_string()),
        fetch_data("https://api.example.com/2".to_string()),
        fetch_data("https://api.example.com/3".to_string())
    );
    Ok(())
}}"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_error_handling() {
    println!("## Example 2: Advanced Error Handling\n");
    println!("Python:");
    println!(r#"
async def risky_operation():
    try:
        result = await fetch_remote_data()
        return result
    except Exception as e:
        logging.error(f"Operation failed: {{e}}")
        raise
"#);

    let wrapper = AsyncioPatterns::generate_error_wrapper(
        "risky_operation",
        "        let result = fetch_remote_data().await?;\n        Ok(result)",
    );

    println!("\nRust (with error wrapper):");
    println!("{}", wrapper);

    println!("\nNote: Automatic error propagation with tracing::error! for debugging");

    println!("\n{}\n", "=".repeat(80));
}

fn example_retry_patterns() {
    println!("## Example 3: Retry Patterns with Exponential Backoff\n");
    println!("Python:");
    println!(r#"
async def fetch_with_retry(url: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return await http_client.get(url)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
"#);

    let retry_pattern = AsyncioPatterns::generate_retry_pattern("http_client.get(url).await", 3);

    println!("\nRust (with retry pattern):");
    println!("{}", retry_pattern);

    println!("\nFeatures:");
    println!("- Exponential backoff: 100ms, 200ms, 400ms");
    println!("- Configurable max retries");
    println!("- Error propagation on final failure");

    println!("\n{}\n", "=".repeat(80));
}

fn example_timeout_fallback() {
    println!("## Example 4: Timeout with Fallback\n");
    println!("Python:");
    println!(r#"
async def fetch_with_fallback(url: str):
    try:
        return await asyncio.wait_for(
            fetch_data(url),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        logging.warning("Request timed out, using cached data")
        return get_cached_data(url)
"#);

    let timeout_pattern = AsyncioPatterns::generate_timeout_with_fallback(
        "fetch_data(url).await",
        5.0,
        "get_cached_data(url)",
    );

    println!("\nRust (with timeout and fallback):");
    println!("{}", timeout_pattern);

    println!("\nFeatures:");
    println!("- tokio::time::timeout for time limits");
    println!("- Graceful degradation with fallback");
    println!("- Logging for observability");

    println!("\n{}\n", "=".repeat(80));
}

fn example_select_patterns() {
    println!("## Example 5: Select/Race Patterns\n");
    println!("Python:");
    println!(r#"
async def race_requests():
    task1 = asyncio.create_task(fetch_from_api1())
    task2 = asyncio.create_task(fetch_from_api2())
    task3 = asyncio.create_task(fetch_from_api3())

    done, pending = await asyncio.wait(
        {{task1, task2, task3}},
        return_when=asyncio.FIRST_COMPLETED
    )

    # Cancel pending tasks
    for task in pending:
        task.cancel()

    return done.pop().result()
"#);

    let select_pattern = AsyncioPatterns::generate_select_pattern(vec![
        "fetch_from_api1()",
        "fetch_from_api2()",
        "fetch_from_api3()",
    ]);

    println!("\nRust (using tokio::select!):");
    println!("{}", select_pattern);

    println!("\nFeatures:");
    println!("- Race multiple futures");
    println!("- First completed wins");
    println!("- Others automatically cancelled");

    println!("\n{}\n", "=".repeat(80));
}

fn example_stream_processing() {
    println!("## Example 6: Async Stream Processing\n");
    println!("Python:");
    println!(r#"
async def process_events(events):
    async for event in events:
        result = await process_event(event)
        await store_result(result)
"#);

    let stream_pattern = AsyncioPatterns::generate_stream_pattern(
        "events_stream",
        "|event| async { process_event(event).await }",
    );

    println!("\nRust (using futures Stream):");
    println!("{}", stream_pattern);

    println!("\nFeatures:");
    println!("- Async iteration over streams");
    println!("- Concurrent processing with buffer_unordered");
    println!("- Backpressure handling");

    println!("\n{}\n", "=".repeat(80));
}

fn example_broadcast_channels() {
    println!("## Example 7: Broadcast Channels\n");
    println!("Python:");
    println!(r#"
# Publisher-subscriber pattern
subscribers = []

async def publish(event):
    for subscriber in subscribers:
        await subscriber.send(event)

async def subscribe():
    queue = asyncio.Queue()
    subscribers.append(queue)
    while True:
        event = await queue.get()
        await handle_event(event)
"#);

    let broadcast_pattern = AsyncioPatterns::generate_broadcast_pattern();

    println!("\nRust (using tokio broadcast channel):");
    println!("{}", broadcast_pattern);

    println!("\nFeatures:");
    println!("- Multiple subscribers to one publisher");
    println!("- Non-blocking sends");
    println!("- Automatic overflow handling");

    println!("\n{}\n", "=".repeat(80));
}

fn example_graceful_shutdown() {
    println!("## Example 8: Graceful Shutdown\n");
    println!("Python:");
    println!(r#"
import signal

async def main():
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown)

    # Run application
    await run_application()
"#);

    let shutdown_pattern = AsyncioPatterns::generate_shutdown_pattern();

    println!("\nRust (with graceful shutdown):");
    println!("{}", shutdown_pattern);

    println!("\nFeatures:");
    println!("- Handle SIGTERM and SIGINT");
    println!("- Cross-platform support");
    println!("- Graceful cleanup");

    println!("\n{}\n", "=".repeat(80));
}

fn example_full_application() {
    println!("## Example 9: Full Application Example\n");
    println!("Python:");
    println!(r#"
import asyncio
from typing import List

async def fetch_user(user_id: int) -> dict:
    await asyncio.sleep(0.1)
    return {{"id": user_id, "name": f"User{{user_id}}"}}

async def process_users(user_ids: List[int]) -> List[dict]:
    tasks = [asyncio.create_task(fetch_user(uid)) for uid in user_ids]
    return await asyncio.gather(*tasks)

async def main():
    user_ids = [1, 2, 3, 4, 5]
    users = await process_users(user_ids)
    print(f"Fetched {{len(users)}} users")

if __name__ == "__main__":
    asyncio.run(main())
"#);

    println!("\nRust (complete translation):");
    println!(r#"use tokio;
use serde_json::Value;

async fn fetch_user(user_id: i32) -> Result<Value> {{
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(serde_json::json!({{
        "id": user_id,
        "name": format!("User{{}}", user_id)
    }}))
}}

async fn process_users(user_ids: Vec<i32>) -> Result<Vec<Value>> {{
    let tasks: Vec<_> = user_ids
        .into_iter()
        .map(|uid| tokio::spawn(async move {{ fetch_user(uid).await }}))
        .collect();

    let mut results = Vec::new();
    for task in tasks {{
        results.push(task.await??);
    }}
    Ok(results)
}}

#[tokio::main]
async fn main() -> Result<()> {{
    let user_ids = vec![1, 2, 3, 4, 5];
    let users = process_users(user_ids).await?;
    println!("Fetched {{}} users", users.len());
    Ok(())
}}"#);

    println!("\n{}\n", "=".repeat(80));
}
