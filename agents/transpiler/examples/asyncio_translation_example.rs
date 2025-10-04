//! Python asyncio → Rust async/await Translation Examples
//!
//! This example demonstrates how to use the asyncio translation layer
//! to convert Python asynchronous code to idiomatic Rust async/await.

use portalis_transpiler::py_to_rust_asyncio::{
    AsyncioMapper, AsyncSyncMapper, AsyncContextMapper, AsyncIteratorMapper,
    AsyncioPatternDetector, AsyncImportGenerator, AsyncFunctionGenerator, AsyncFeatures,
};

fn main() {
    println!("=== Python asyncio → Rust async/await Translation Examples ===\n");

    // Example 1: Basic async function
    println!("Example 1: Basic Async Function");
    println!("Python:");
    println!("async def fetch_user(user_id: int) -> dict:");
    println!("    response = await http_client.get(f'/users/{{user_id}}')");
    println!("    return await response.json()");
    println!("\nRust:");
    let rust_func = AsyncioMapper::translate_async_function(
        "fetch_user",
        vec![("user_id", "i32")],
        "serde_json::Value",
        true,
    );
    println!("{}\n", rust_func);

    // Example 2: Concurrent tasks with gather
    println!("Example 2: Concurrent Tasks with gather()");
    println!("Python:");
    println!("results = await asyncio.gather(");
    println!("    fetch_user(1),");
    println!("    fetch_user(2),");
    println!("    fetch_user(3)");
    println!(")");
    println!("\nRust:");
    let gather = AsyncioMapper::translate_gather(vec![
        "fetch_user(1)",
        "fetch_user(2)",
        "fetch_user(3)",
    ]);
    println!("{}\n", gather);

    // Example 3: Async sleep
    println!("Example 3: Async Sleep");
    println!("Python:");
    println!("await asyncio.sleep(1.5)");
    println!("\nRust:");
    let sleep = AsyncioMapper::translate_sleep("1.5");
    println!("{}\n", sleep);

    // Example 4: Timeout with wait_for
    println!("Example 4: Timeout with wait_for()");
    println!("Python:");
    println!("result = await asyncio.wait_for(slow_operation(), timeout=10.0)");
    println!("\nRust:");
    let timeout = AsyncioMapper::translate_wait_for("slow_operation()", "10.0");
    println!("{}\n", timeout);

    // Example 5: Task creation
    println!("Example 5: Task Creation");
    println!("Python:");
    println!("task = asyncio.create_task(background_worker())");
    println!("\nRust:");
    let task = AsyncioMapper::translate_create_task("background_worker()");
    println!("{}\n", task);

    // Example 6: Async Lock
    println!("Example 6: Async Lock");
    println!("Python:");
    println!("lock = asyncio.Lock()");
    println!("async with lock:");
    println!("    # critical section");
    println!("    await process_data()");
    println!("\nRust:");
    let lock_creation = AsyncSyncMapper::translate_lock_creation();
    println!("let lock = {};", lock_creation);
    let lock_acquire = AsyncSyncMapper::translate_lock_acquire("lock");
    println!("{}", lock_acquire);
    println!("// critical section");
    println!("process_data().await?;\n");

    // Example 7: Async Event
    println!("Example 7: Async Event");
    println!("Python:");
    println!("event = asyncio.Event()");
    println!("await event.wait()");
    println!("event.set()");
    println!("\nRust:");
    let event_creation = AsyncSyncMapper::translate_event_creation();
    println!("let event = {};", event_creation);
    let event_wait = AsyncSyncMapper::translate_event_wait("event");
    println!("{}", event_wait);
    let event_set = AsyncSyncMapper::translate_event_set("event");
    println!("{}\n", event_set);

    // Example 8: Async Semaphore
    println!("Example 8: Async Semaphore");
    println!("Python:");
    println!("sem = asyncio.Semaphore(5)");
    println!("async with sem:");
    println!("    # at most 5 concurrent operations");
    println!("    await process()");
    println!("\nRust:");
    let sem_creation = AsyncSyncMapper::translate_semaphore_creation("5");
    println!("let sem = {};", sem_creation);
    let sem_acquire = AsyncSyncMapper::translate_semaphore_acquire("sem");
    println!("{}", sem_acquire);
    println!("// at most 5 concurrent operations");
    println!("process().await?;\n");

    // Example 9: Async Queue (Producer-Consumer)
    println!("Example 9: Async Queue (Producer-Consumer)");
    println!("Python:");
    println!("queue = asyncio.Queue(maxsize=100)");
    println!("await queue.put(item)");
    println!("item = await queue.get()");
    println!("\nRust:");
    let queue = AsyncSyncMapper::translate_queue_creation(Some("100"));
    println!("{}", queue);
    let put = AsyncSyncMapper::translate_queue_put("tx", "item");
    println!("{}", put);
    let get = AsyncSyncMapper::translate_queue_get("rx");
    println!("let item = {};\n", get);

    // Example 10: Async context manager
    println!("Example 10: Async Context Manager");
    println!("Python:");
    println!("async with database.connection() as conn:");
    println!("    await conn.execute(query)");
    println!("\nRust:");
    let async_with = AsyncContextMapper::translate_async_with(
        "database.connection()",
        Some("conn"),
        "    conn.execute(query).await?;",
    );
    println!("{}\n", async_with);

    // Example 11: Async for loop
    println!("Example 11: Async For Loop");
    println!("Python:");
    println!("async for item in stream:");
    println!("    process(item)");
    println!("\nRust:");
    let async_for = AsyncIteratorMapper::translate_async_for(
        "item",
        "stream",
        "    process(item);",
    );
    println!("{}\n", async_for);

    // Example 12: Pattern detection and feature analysis
    println!("Example 12: Feature Detection");
    let python_code = r#"
import asyncio

async def main():
    lock = asyncio.Lock()
    await asyncio.sleep(1.0)
    task = asyncio.create_task(worker())
    results = await asyncio.gather(task1(), task2())

    async for item in stream:
        process(item)
"#;
    println!("Analyzing Python code...");
    let features = AsyncioPatternDetector::analyze_features(python_code);
    println!("Detected features:");
    println!("  - Async functions: {}", features.uses_async_functions);
    println!("  - Sync primitives: {}", features.uses_sync_primitives);
    println!("  - Time operations: {}", features.uses_time_operations);
    println!("  - Futures combinators: {}", features.uses_futures_combinators);
    println!("  - Streams: {}\n", features.uses_streams);

    // Example 13: Import generation
    println!("Example 13: Import Generation");
    let imports = AsyncImportGenerator::get_imports_for_features(&features);
    println!("Required imports for detected features:");
    for import in &imports {
        println!("  {}", import);
    }
    println!();

    // Example 14: Complete function generation
    println!("Example 14: Complete Async Function with Tokio Main");
    let main_func = AsyncFunctionGenerator::generate_tokio_main(
        "    let users = fetch_users().await?;\n    println!(\"Fetched {} users\", users.len());"
    );
    println!("{}\n", main_func);

    // Example 15: Complex workflow
    println!("Example 15: Complex Async Workflow");
    println!("Python:");
    println!("async def process_pipeline(data):");
    println!("    async with rate_limiter.acquire():");
    println!("        validated = await validate(data)");
    println!("        transformed = await transform(validated)");
    println!("        await store(transformed)");
    println!("\nRust (combining multiple translations):");

    let function = AsyncFunctionGenerator::generate_async_function(
        "process_pipeline",
        vec![("data", "Data")],
        "()",
        r#"    {
        let _permit = rate_limiter.acquire().await?;
        let validated = validate(data).await?;
        let transformed = transform(validated).await?;
        store(transformed).await?;
    }"#,
        true,
    );
    println!("{}\n", function);

    // Example 16: Wait patterns
    println!("Example 16: Wait Patterns");
    println!("Python (FIRST_COMPLETED):");
    println!("done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)");
    println!("\nRust:");
    let wait_first = AsyncioMapper::translate_wait("tasks", Some("FIRST_COMPLETED"));
    println!("{}\n", wait_first);

    println!("Python (ALL_COMPLETED):");
    println!("results = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)");
    println!("\nRust:");
    let wait_all = AsyncioMapper::translate_wait("tasks", Some("ALL_COMPLETED"));
    println!("{}\n", wait_all);

    // Example 17: as_completed pattern
    println!("Example 17: as_completed Pattern");
    println!("Python:");
    println!("for future in asyncio.as_completed(tasks):");
    println!("    result = await future");
    println!("    print(f'Completed: {{result}}')");
    println!("\nRust:");
    let as_completed = AsyncioMapper::translate_as_completed(
        "tasks",
        "    println!(\"Completed: {:?}\", result);",
    );
    println!("{}\n", as_completed);

    // Example 18: Cargo dependencies
    println!("Example 18: Required Cargo Dependencies");
    let deps = AsyncImportGenerator::get_cargo_dependencies();
    println!("Add to Cargo.toml:");
    println!("[dependencies]");
    for (name, version) in deps {
        println!("{} = {}", name, version);
    }
    println!();

    // Example 19: WASM support
    println!("Example 19: WASM Async Support");
    let mut wasm_features = AsyncFeatures::default();
    wasm_features.uses_async_functions = true;
    wasm_features.target_wasm = true;
    let wasm_imports = AsyncImportGenerator::get_imports_for_features(&wasm_features);
    println!("WASM-specific imports:");
    for import in &wasm_imports {
        if import.contains("wasm") {
            println!("  {}", import);
        }
    }
    println!();

    println!("=== Translation Complete ===");
    println!("\nThese examples show how Python asyncio patterns map to Rust async/await.");
    println!("The translation layer handles:");
    println!("  - Async function definitions");
    println!("  - Await expressions");
    println!("  - Task management (spawn, join, gather)");
    println!("  - Synchronization primitives (Lock, Event, Semaphore, Queue)");
    println!("  - Timeouts and delays");
    println!("  - Context managers");
    println!("  - Async iterators and streams");
    println!("  - Import and dependency management");
}
