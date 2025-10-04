//! Integration tests for Python asyncio → Rust async/await translation

use portalis_transpiler::py_to_rust_asyncio::{
    AsyncioMapper, AsyncSyncMapper, AsyncContextMapper, AsyncIteratorMapper,
    AsyncioPatternDetector, AsyncImportGenerator, AsyncFunctionGenerator, AsyncFeatures,
};

#[test]
fn test_basic_async_function_translation() {
    let result = AsyncioMapper::translate_async_function(
        "fetch_data",
        vec![("url", "String"), ("timeout", "f64")],
        "Response",
        true,
    );

    assert!(result.contains("async fn fetch_data"));
    assert!(result.contains("url: String"));
    assert!(result.contains("timeout: f64"));
    assert!(result.contains("Result<Response>"));
}

#[test]
fn test_await_expression_translation() {
    let cases = vec![
        ("get_user()", "get_user().await?"),
        ("client.fetch(url)", "client.fetch(url).await?"),
        ("db.query(sql)", "db.query(sql).await?"),
    ];

    for (input, expected) in cases {
        let result = AsyncioMapper::translate_await(input);
        assert_eq!(result, expected);
    }
}

#[test]
fn test_asyncio_run_translation() {
    let result = AsyncioMapper::translate_asyncio_run("main()");

    assert!(result.contains("#[tokio::main]"));
    assert!(result.contains("async fn main()"));
    assert!(result.contains("Result<()>"));
    assert!(result.contains("Ok(())"));
}

#[test]
fn test_create_task_translation() {
    let result = AsyncioMapper::translate_create_task("process_data(item)");

    assert!(result.contains("tokio::spawn"));
    assert!(result.contains("async move"));
    assert!(result.contains("process_data(item).await"));
}

#[test]
fn test_gather_translation() {
    let tasks = vec!["fetch_user()", "fetch_posts()", "fetch_comments()"];
    let result = AsyncioMapper::translate_gather(tasks);

    assert!(result.contains("tokio::join!"));
    assert!(result.contains("result1, result2, result3"));
    assert!(result.contains("fetch_user()"));
    assert!(result.contains("fetch_posts()"));
    assert!(result.contains("fetch_comments()"));
}

#[test]
fn test_sleep_translation() {
    // Test float seconds
    let result_float = AsyncioMapper::translate_sleep("1.5");
    assert!(result_float.contains("Duration::from_secs_f64(1.5)"));
    assert!(result_float.contains(".await"));

    // Test integer seconds
    let result_int = AsyncioMapper::translate_sleep("5");
    assert!(result_int.contains("Duration::from_secs(5)"));
    assert!(result_int.contains(".await"));
}

#[test]
fn test_wait_for_translation() {
    let result = AsyncioMapper::translate_wait_for("fetch_data()", "10.0");

    assert!(result.contains("tokio::time::timeout"));
    assert!(result.contains("Duration::from_secs_f64(10.0)"));
    assert!(result.contains("fetch_data()"));
    assert!(result.contains(".await??"));
}

#[test]
fn test_wait_translation() {
    // Test FIRST_COMPLETED
    let result = AsyncioMapper::translate_wait("tasks", Some("FIRST_COMPLETED"));
    assert!(result.contains("select_all"));
    assert!(result.contains("result, _index, remaining"));

    // Test ALL_COMPLETED
    let result_all = AsyncioMapper::translate_wait("tasks", Some("ALL_COMPLETED"));
    assert!(result_all.contains("join_all"));

    // Test default (FIRST_COMPLETED)
    let result_default = AsyncioMapper::translate_wait("tasks", None);
    assert!(result_default.contains("select_all"));
}

#[test]
fn test_as_completed_translation() {
    let body = "    process(result);";
    let result = AsyncioMapper::translate_as_completed("tasks", body);

    assert!(result.contains("FuturesUnordered"));
    assert!(result.contains("stream.next().await"));
    assert!(result.contains("process(result)"));
}

#[test]
fn test_lock_translation() {
    let lock_creation = AsyncSyncMapper::translate_lock_creation();
    assert!(lock_creation.contains("tokio::sync::Mutex::new(())"));

    let lock_acquire = AsyncSyncMapper::translate_lock_acquire("my_lock");
    assert!(lock_acquire.contains("my_lock.lock().await"));
    assert!(lock_acquire.contains("_guard"));
}

#[test]
fn test_event_translation() {
    let event_creation = AsyncSyncMapper::translate_event_creation();
    assert!(event_creation.contains("tokio::sync::Notify::new"));
    assert!(event_creation.contains("Arc::new"));

    let event_wait = AsyncSyncMapper::translate_event_wait("event");
    assert!(event_wait.contains("event.notified().await"));

    let event_set = AsyncSyncMapper::translate_event_set("event");
    assert!(event_set.contains("event.notify_one()"));

    let event_set_all = AsyncSyncMapper::translate_event_set_all("event");
    assert!(event_set_all.contains("event.notify_waiters()"));
}

#[test]
fn test_semaphore_translation() {
    let sem_creation = AsyncSyncMapper::translate_semaphore_creation("10");
    assert!(sem_creation.contains("tokio::sync::Semaphore::new(10)"));
    assert!(sem_creation.contains("Arc::new"));

    let sem_acquire = AsyncSyncMapper::translate_semaphore_acquire("semaphore");
    assert!(sem_acquire.contains("semaphore.acquire().await"));
    assert!(sem_acquire.contains("_permit"));
}

#[test]
fn test_queue_translation() {
    let queue_creation = AsyncSyncMapper::translate_queue_creation(Some("50"));
    assert!(queue_creation.contains("tokio::sync::mpsc::channel(50)"));
    assert!(queue_creation.contains("let (tx, mut rx)"));

    let queue_put = AsyncSyncMapper::translate_queue_put("sender", "message");
    assert!(queue_put.contains("sender.send(message).await"));

    let queue_get = AsyncSyncMapper::translate_queue_get("receiver");
    assert!(queue_get.contains("receiver.recv().await"));
}

#[test]
fn test_async_with_translation() {
    let result = AsyncContextMapper::translate_async_with(
        "database.connection()",
        Some("conn"),
        "    conn.execute(query).await?;\n    conn.commit().await?;",
    );

    assert!(result.contains("let conn = database.connection().await?"));
    assert!(result.contains("conn.execute(query)"));
    assert!(result.contains("conn.commit()"));
    assert!(result.contains("automatically dropped"));
}

#[test]
fn test_async_with_no_variable() {
    let result = AsyncContextMapper::translate_async_with(
        "lock.acquire()",
        None,
        "    // critical section",
    );

    assert!(result.contains("let _context = lock.acquire().await?"));
    assert!(result.contains("critical section"));
}

#[test]
fn test_async_for_translation() {
    let result = AsyncIteratorMapper::translate_async_for(
        "item",
        "stream_data()",
        "    println!(\"Got: {}\", item);",
    );

    assert!(result.contains("use futures::stream::StreamExt"));
    assert!(result.contains("let mut stream = stream_data()"));
    assert!(result.contains("while let Some(item) = stream.next().await"));
    assert!(result.contains("println!"));
}

#[test]
fn test_async_generator_translation() {
    let result = AsyncIteratorMapper::translate_async_generator(
        "number_stream",
        "i32",
        "    stream::iter(0..100)",
    );

    assert!(result.contains("fn number_stream()"));
    assert!(result.contains("impl Stream<Item = i32>"));
    assert!(result.contains("stream::iter(0..100)"));
}

#[test]
fn test_pattern_detection_async_def() {
    let code = "async def my_function():\n    pass";
    assert!(AsyncioPatternDetector::uses_async_def(code));
}

#[test]
fn test_pattern_detection_await() {
    let code = "result = await some_coro()";
    assert!(AsyncioPatternDetector::uses_await(code));
}

#[test]
fn test_pattern_detection_asyncio_run() {
    let code = "asyncio.run(main())";
    assert!(AsyncioPatternDetector::uses_asyncio_run(code));
}

#[test]
fn test_pattern_detection_create_task() {
    let code = "task = asyncio.create_task(worker())";
    assert!(AsyncioPatternDetector::uses_create_task(code));
}

#[test]
fn test_pattern_detection_gather() {
    let code = "results = await asyncio.gather(t1(), t2())";
    assert!(AsyncioPatternDetector::uses_gather(code));
}

#[test]
fn test_pattern_detection_sleep() {
    let code = "await asyncio.sleep(1.0)";
    assert!(AsyncioPatternDetector::uses_async_sleep(code));
}

#[test]
fn test_pattern_detection_lock() {
    let code = "lock = asyncio.Lock()";
    assert!(AsyncioPatternDetector::uses_async_lock(code));
}

#[test]
fn test_pattern_detection_event() {
    let code = "event = asyncio.Event()";
    assert!(AsyncioPatternDetector::uses_async_event(code));
}

#[test]
fn test_pattern_detection_semaphore() {
    let code = "sem = asyncio.Semaphore(5)";
    assert!(AsyncioPatternDetector::uses_async_semaphore(code));
}

#[test]
fn test_pattern_detection_queue() {
    let code = "q = asyncio.Queue(maxsize=10)";
    assert!(AsyncioPatternDetector::uses_async_queue(code));
}

#[test]
fn test_pattern_detection_async_with() {
    let code = "async with resource:\n    pass";
    assert!(AsyncioPatternDetector::uses_async_with(code));
}

#[test]
fn test_pattern_detection_async_for() {
    let code = "async for item in stream:\n    pass";
    assert!(AsyncioPatternDetector::uses_async_for(code));
}

#[test]
fn test_feature_analysis_complete() {
    let code = r#"
import asyncio

async def main():
    # Async functions
    lock = asyncio.Lock()
    event = asyncio.Event()
    sem = asyncio.Semaphore(5)
    queue = asyncio.Queue()

    # Time operations
    await asyncio.sleep(1.0)

    # Task management
    task = asyncio.create_task(worker())
    results = await asyncio.gather(task1(), task2())

    # Async iteration
    async for item in stream:
        process(item)

    # Async context manager
    async with lock:
        await do_critical_work()
"#;

    let features = AsyncioPatternDetector::analyze_features(code);

    assert!(features.uses_async_functions);
    assert!(features.uses_sync_primitives);
    assert!(features.uses_time_operations);
    assert!(features.uses_futures_combinators);
    assert!(features.uses_streams);
}

#[test]
fn test_import_generation_tokio_runtime() {
    let imports = AsyncImportGenerator::get_tokio_runtime_imports();
    assert!(!imports.is_empty());
    assert!(imports.iter().any(|s| s.contains("tokio")));
}

#[test]
fn test_import_generation_tokio_sync() {
    let imports = AsyncImportGenerator::get_tokio_sync_imports();
    assert!(!imports.is_empty());
    assert!(imports.iter().any(|s| s.contains("Mutex")));
    assert!(imports.iter().any(|s| s.contains("Semaphore")));
}

#[test]
fn test_import_generation_tokio_time() {
    let imports = AsyncImportGenerator::get_tokio_time_imports();
    assert!(!imports.is_empty());
    assert!(imports.iter().any(|s| s.contains("sleep")));
    assert!(imports.iter().any(|s| s.contains("timeout")));
}

#[test]
fn test_import_generation_futures() {
    let imports = AsyncImportGenerator::get_futures_imports();
    assert!(!imports.is_empty());
    assert!(imports.iter().any(|s| s.contains("futures")));
    assert!(imports.iter().any(|s| s.contains("Stream")));
}

#[test]
fn test_import_generation_for_features() {
    let features = AsyncFeatures {
        uses_async_functions: true,
        uses_sync_primitives: true,
        uses_time_operations: true,
        uses_futures_combinators: true,
        uses_streams: true,
        target_wasm: false,
    };

    let imports = AsyncImportGenerator::get_imports_for_features(&features);

    assert!(!imports.is_empty());
    assert!(imports.iter().any(|s| s.contains("tokio")));
    assert!(imports.iter().any(|s| s.contains("futures")));
    assert!(imports.iter().any(|s| s.contains("Result")));
}

#[test]
fn test_import_generation_wasm() {
    let mut features = AsyncFeatures::default();
    features.uses_async_functions = true;
    features.target_wasm = true;

    let imports = AsyncImportGenerator::get_imports_for_features(&features);
    assert!(imports.iter().any(|s| s.contains("wasm_bindgen_futures")));
}

#[test]
fn test_cargo_dependencies() {
    let deps = AsyncImportGenerator::get_cargo_dependencies();
    assert!(!deps.is_empty());

    let tokio_dep = deps.iter().find(|(name, _)| *name == "tokio");
    assert!(tokio_dep.is_some());

    let futures_dep = deps.iter().find(|(name, _)| *name == "futures");
    assert!(futures_dep.is_some());
}

#[test]
fn test_wasm_cargo_dependencies() {
    let deps = AsyncImportGenerator::get_wasm_cargo_dependencies();
    assert!(!deps.is_empty());

    let wasm_futures = deps.iter().find(|(name, _)| *name == "wasm-bindgen-futures");
    assert!(wasm_futures.is_some());
}

#[test]
fn test_generate_async_function_with_result() {
    let result = AsyncFunctionGenerator::generate_async_function(
        "process_data",
        vec![("data", "Vec<u8>"), ("timeout", "Duration")],
        "ProcessResult",
        "    // Processing logic",
        true,
    );

    assert!(result.contains("async fn process_data"));
    assert!(result.contains("data: Vec<u8>"));
    assert!(result.contains("timeout: Duration"));
    assert!(result.contains("Result<ProcessResult>"));
}

#[test]
fn test_generate_async_function_without_result() {
    let result = AsyncFunctionGenerator::generate_async_function(
        "notify",
        vec![("message", "String")],
        "()",
        "    println!(\"{}\", message);",
        false,
    );

    assert!(result.contains("async fn notify"));
    assert!(result.contains("message: String"));
    assert!(result.contains("-> ()"));
    assert!(!result.contains("Result"));
}

#[test]
fn test_generate_tokio_main() {
    let result = AsyncFunctionGenerator::generate_tokio_main(
        "    start_server().await?;\n    println!(\"Server running\");"
    );

    assert!(result.contains("#[tokio::main]"));
    assert!(result.contains("async fn main() -> Result<()>"));
    assert!(result.contains("start_server().await?"));
    assert!(result.contains("Ok(())"));
}

#[test]
fn test_generate_async_block() {
    let result = AsyncFunctionGenerator::generate_async_block(
        "    let data = fetch().await?;\n    process(data).await?"
    );

    assert!(result.contains("async {"));
    assert!(result.contains("fetch().await?"));
    assert!(result.contains("process(data).await?"));
}

#[test]
fn test_complete_async_workflow() {
    // Simulate translating a complete Python async function
    let python_code = r#"
async def fetch_and_process(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        response = await session.get(url)
        data = await response.json()
        await asyncio.sleep(0.1)
        return data
"#;

    // Detect features
    let features = AsyncioPatternDetector::analyze_features(python_code);
    assert!(features.uses_async_functions);
    assert!(features.uses_time_operations);

    // Generate imports
    let imports = AsyncImportGenerator::get_imports_for_features(&features);
    assert!(!imports.is_empty());

    // Generate function
    let func = AsyncioMapper::translate_async_function(
        "fetch_and_process",
        vec![("url", "String")],
        "serde_json::Value",
        true,
    );
    assert!(func.contains("async fn"));

    // Generate sleep call
    let sleep = AsyncioMapper::translate_sleep("0.1");
    assert!(sleep.contains("Duration"));
}

#[test]
fn test_concurrent_tasks_translation() {
    // Test translating concurrent task pattern
    let tasks = vec!["fetch_user(id)", "fetch_posts(id)", "fetch_likes(id)"];
    let gather = AsyncioMapper::translate_gather(tasks);

    assert!(gather.contains("tokio::join!"));
    assert!(gather.contains("result1, result2, result3"));
}

#[test]
fn test_timeout_pattern_translation() {
    // Test timeout pattern
    let timeout = AsyncioMapper::translate_wait_for("slow_operation()", "30");
    assert!(timeout.contains("tokio::time::timeout"));
    assert!(timeout.contains("Duration::from_secs(30)"));
}

#[test]
fn test_producer_consumer_pattern() {
    // Test queue-based producer-consumer pattern
    let queue_setup = AsyncSyncMapper::translate_queue_creation(Some("100"));
    assert!(queue_setup.contains("mpsc::channel(100)"));

    let produce = AsyncSyncMapper::translate_queue_put("tx", "work_item");
    assert!(produce.contains("tx.send(work_item)"));

    let consume = AsyncSyncMapper::translate_queue_get("rx");
    assert!(consume.contains("rx.recv()"));
}
