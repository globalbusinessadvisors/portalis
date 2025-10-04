//! Python asyncio → Rust async/await Translation Layer
//!
//! Translates Python asynchronous programming patterns to Rust async/await:
//! - asyncio fundamentals → tokio runtime
//! - async/await syntax → Rust async/await
//! - asyncio synchronization → tokio::sync
//! - asyncio utilities → tokio/futures utilities
//!
//! This module provides comprehensive translation for Python's asyncio ecosystem.


/// Main asyncio translation mapper
pub struct AsyncioMapper;

impl AsyncioMapper {
    /// Translate Python async def to Rust async fn
    ///
    /// Python:
    /// ```python
    /// async def fetch_data(url: str) -> dict:
    ///     response = await client.get(url)
    ///     return response.json()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// async fn fetch_data(url: String) -> Result<serde_json::Value> {
    ///     let response = client.get(&url).await?;
    ///     Ok(response.json()?)
    /// }
    /// ```
    pub fn translate_async_function(
        name: &str,
        params: Vec<(&str, &str)>,
        return_type: &str,
        is_async: bool,
    ) -> String {
        let async_keyword = if is_async { "async " } else { "" };

        let param_str = params.iter()
            .map(|(name, typ)| format!("{}: {}", name, typ))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "{}fn {}({}) -> Result<{}> {{\n    // Function body\n}}",
            async_keyword, name, param_str, return_type
        )
    }

    /// Translate Python await expression to Rust .await
    ///
    /// Python:
    /// ```python
    /// result = await some_async_function()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let result = some_async_function().await?;
    /// ```
    pub fn translate_await(expression: &str) -> String {
        format!("{}.await?", expression)
    }

    /// Translate asyncio.run() to tokio runtime
    ///
    /// Python:
    /// ```python
    /// asyncio.run(main())
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     // main logic
    ///     Ok(())
    /// }
    /// ```
    pub fn translate_asyncio_run(func_call: &str) -> String {
        format!(
            r#"#[tokio::main]
async fn main() -> Result<()> {{
    {}
    Ok(())
}}"#,
            func_call
        )
    }

    /// Translate asyncio.create_task() to tokio::spawn()
    ///
    /// Python:
    /// ```python
    /// task = asyncio.create_task(do_something())
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let task = tokio::spawn(async move {
    ///     do_something().await
    /// });
    /// ```
    pub fn translate_create_task(coro: &str) -> String {
        format!(
            "let task = tokio::spawn(async move {{\n    {}.await\n}});",
            coro
        )
    }

    /// Translate asyncio.gather() to tokio::join!()
    ///
    /// Python:
    /// ```python
    /// results = await asyncio.gather(task1(), task2(), task3())
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let (result1, result2, result3) = tokio::join!(
    ///     task1(),
    ///     task2(),
    ///     task3()
    /// );
    /// ```
    pub fn translate_gather(tasks: Vec<&str>) -> String {
        let task_count = tasks.len();
        let result_vars = (1..=task_count)
            .map(|i| format!("result{}", i))
            .collect::<Vec<_>>()
            .join(", ");

        let tasks_str = tasks.join(",\n    ");

        format!(
            "let ({}) = tokio::join!(\n    {}\n);",
            result_vars, tasks_str
        )
    }

    /// Translate asyncio.sleep() to tokio::time::sleep()
    ///
    /// Python:
    /// ```python
    /// await asyncio.sleep(1.0)
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// tokio::time::sleep(Duration::from_secs_f64(1.0)).await;
    /// ```
    pub fn translate_sleep(seconds: &str) -> String {
        // Handle both integer and float seconds
        if seconds.contains('.') {
            format!("tokio::time::sleep(Duration::from_secs_f64({})).await;", seconds)
        } else {
            format!("tokio::time::sleep(Duration::from_secs({})).await;", seconds)
        }
    }

    /// Translate asyncio.wait_for() to tokio::time::timeout()
    ///
    /// Python:
    /// ```python
    /// result = await asyncio.wait_for(operation(), timeout=5.0)
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let result = tokio::time::timeout(
    ///     Duration::from_secs_f64(5.0),
    ///     operation()
    /// ).await??;
    /// ```
    pub fn translate_wait_for(coro: &str, timeout: &str) -> String {
        let timeout_expr = if timeout.contains('.') {
            format!("Duration::from_secs_f64({})", timeout)
        } else {
            format!("Duration::from_secs({})", timeout)
        };

        format!(
            "tokio::time::timeout(\n    {},\n    {}\n).await??",
            timeout_expr, coro
        )
    }

    /// Translate asyncio.wait() to futures::select_all()
    ///
    /// Python:
    /// ```python
    /// done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let (result, _index, remaining) = futures::future::select_all(tasks).await;
    /// ```
    pub fn translate_wait(tasks_var: &str, return_when: Option<&str>) -> String {
        match return_when {
            Some("FIRST_COMPLETED") | Some("asyncio.FIRST_COMPLETED") => {
                format!("let (result, _index, remaining) = futures::future::select_all({}).await;", tasks_var)
            }
            Some("ALL_COMPLETED") | Some("asyncio.ALL_COMPLETED") => {
                format!("let results = futures::future::join_all({}).await;", tasks_var)
            }
            _ => {
                // Default to FIRST_COMPLETED
                format!("let (result, _index, remaining) = futures::future::select_all({}).await;", tasks_var)
            }
        }
    }

    /// Translate asyncio.as_completed() to futures::stream::FuturesUnordered
    ///
    /// Python:
    /// ```python
    /// for future in asyncio.as_completed(tasks):
    ///     result = await future
    ///     process(result)
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// use futures::stream::StreamExt;
    /// let mut stream = futures::stream::FuturesUnordered::from_iter(tasks);
    /// while let Some(result) = stream.next().await {
    ///     process(result);
    /// }
    /// ```
    pub fn translate_as_completed(tasks_var: &str, body: &str) -> String {
        format!(
            r#"use futures::stream::StreamExt;
let mut stream = futures::stream::FuturesUnordered::from_iter({});
while let Some(result) = stream.next().await {{
    {}
}}"#,
            tasks_var, body
        )
    }
}

/// Async synchronization primitives translation
pub struct AsyncSyncMapper;

impl AsyncSyncMapper {
    /// Translate asyncio.Lock to tokio::sync::Mutex
    ///
    /// Python:
    /// ```python
    /// lock = asyncio.Lock()
    /// async with lock:
    ///     # critical section
    ///     pass
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let lock = tokio::sync::Mutex::new(());
    /// {
    ///     let _guard = lock.lock().await;
    ///     // critical section
    /// }
    /// ```
    pub fn translate_lock_creation() -> String {
        "tokio::sync::Mutex::new(())".to_string()
    }

    /// Translate asyncio.Lock acquire/release to Rust RAII pattern
    pub fn translate_lock_acquire(lock_var: &str) -> String {
        format!("let _guard = {}.lock().await;", lock_var)
    }

    /// Translate asyncio.Event to tokio::sync::Notify
    ///
    /// Python:
    /// ```python
    /// event = asyncio.Event()
    /// await event.wait()
    /// event.set()
    /// event.clear()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let event = Arc::new(tokio::sync::Notify::new());
    /// event.notified().await;
    /// event.notify_one();
    /// // Note: Notify doesn't have clear(), it's single-use per notification
    /// ```
    pub fn translate_event_creation() -> String {
        "Arc::new(tokio::sync::Notify::new())".to_string()
    }

    pub fn translate_event_wait(event_var: &str) -> String {
        format!("{}.notified().await;", event_var)
    }

    pub fn translate_event_set(event_var: &str) -> String {
        format!("{}.notify_one();", event_var)
    }

    pub fn translate_event_set_all(event_var: &str) -> String {
        format!("{}.notify_waiters();", event_var)
    }

    /// Translate asyncio.Semaphore to tokio::sync::Semaphore
    ///
    /// Python:
    /// ```python
    /// sem = asyncio.Semaphore(3)
    /// async with sem:
    ///     # at most 3 concurrent operations
    ///     pass
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let sem = Arc::new(tokio::sync::Semaphore::new(3));
    /// {
    ///     let _permit = sem.acquire().await.unwrap();
    ///     // at most 3 concurrent operations
    /// }
    /// ```
    pub fn translate_semaphore_creation(value: &str) -> String {
        format!("Arc::new(tokio::sync::Semaphore::new({}))", value)
    }

    pub fn translate_semaphore_acquire(sem_var: &str) -> String {
        format!("let _permit = {}.acquire().await.unwrap();", sem_var)
    }

    /// Translate asyncio.Queue to tokio::sync::mpsc
    ///
    /// Python:
    /// ```python
    /// queue = asyncio.Queue(maxsize=10)
    /// await queue.put(item)
    /// item = await queue.get()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let (tx, mut rx) = tokio::sync::mpsc::channel(10);
    /// tx.send(item).await.unwrap();
    /// let item = rx.recv().await.unwrap();
    /// ```
    pub fn translate_queue_creation(maxsize: Option<&str>) -> String {
        let size = maxsize.unwrap_or("100");
        format!("let (tx, mut rx) = tokio::sync::mpsc::channel({});", size)
    }

    pub fn translate_queue_put(queue_var: &str, item: &str) -> String {
        format!("{}.send({}).await.unwrap();", queue_var, item)
    }

    pub fn translate_queue_get(queue_var: &str) -> String {
        format!("{}.recv().await.unwrap()", queue_var)
    }

    /// Translate asyncio.Condition to tokio::sync::Notify (simplified)
    ///
    /// Python:
    /// ```python
    /// condition = asyncio.Condition()
    /// async with condition:
    ///     await condition.wait()
    ///     condition.notify()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let condition = Arc::new(tokio::sync::Notify::new());
    /// condition.notified().await;
    /// condition.notify_one();
    /// ```
    pub fn translate_condition_creation() -> String {
        "Arc::new(tokio::sync::Notify::new())".to_string()
    }
}

/// Async context manager translation
pub struct AsyncContextMapper;

impl AsyncContextMapper {
    /// Translate async with statement to RAII pattern
    ///
    /// Python:
    /// ```python
    /// async with resource_manager() as resource:
    ///     await resource.process()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// {
    ///     let resource = resource_manager().await?;
    ///     resource.process().await?;
    ///     // resource automatically dropped (async Drop if implemented)
    /// }
    /// ```
    pub fn translate_async_with(
        context_expr: &str,
        var_name: Option<&str>,
        body: &str,
    ) -> String {
        if let Some(var) = var_name {
            format!(
                r#"{{
    let {} = {}.await?;
    {}
    // {} automatically dropped
}}"#,
                var, context_expr, body, var
            )
        } else {
            format!(
                r#"{{
    let _context = {}.await?;
    {}
    // _context automatically dropped
}}"#,
                context_expr, body
            )
        }
    }

    /// Generate async context manager entry code
    pub fn translate_aenter(obj_expr: &str) -> String {
        format!("{}.aenter().await?", obj_expr)
    }

    /// Generate async context manager exit code (automatic via Drop)
    pub fn translate_aexit(obj_expr: &str) -> String {
        format!("{}.aexit().await?", obj_expr)
    }
}

/// Async iterator/stream translation
pub struct AsyncIteratorMapper;

impl AsyncIteratorMapper {
    /// Translate async for to Stream iteration
    ///
    /// Python:
    /// ```python
    /// async for item in async_iterator:
    ///     process(item)
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// use futures::stream::StreamExt;
    /// let mut stream = async_iterator;
    /// while let Some(item) = stream.next().await {
    ///     process(item);
    /// }
    /// ```
    pub fn translate_async_for(
        var_name: &str,
        iterator_expr: &str,
        body: &str,
    ) -> String {
        format!(
            r#"use futures::stream::StreamExt;
let mut stream = {};
while let Some({}) = stream.next().await {{
    {}
}}"#,
            iterator_expr, var_name, body
        )
    }

    /// Translate async generator/comprehension
    ///
    /// Python:
    /// ```python
    /// async def async_generator():
    ///     for i in range(10):
    ///         await asyncio.sleep(0.1)
    ///         yield i
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// use futures::stream::{self, Stream};
    ///
    /// fn async_generator() -> impl Stream<Item = i32> {
    ///     stream::iter(0..10).then(|i| async move {
    ///         tokio::time::sleep(Duration::from_millis(100)).await;
    ///         i
    ///     })
    /// }
    /// ```
    pub fn translate_async_generator(
        name: &str,
        yield_type: &str,
        body: &str,
    ) -> String {
        format!(
            r#"use futures::stream::{{self, Stream}};

fn {}() -> impl Stream<Item = {}> {{
    {}
}}"#,
            name, yield_type, body
        )
    }

    /// Translate AsyncIterator protocol
    pub fn translate_async_iter(expr: &str) -> String {
        format!("futures::stream::iter({}).boxed()", expr)
    }
}

/// Import generation for async code
pub struct AsyncImportGenerator;

impl AsyncImportGenerator {
    /// Get core tokio imports for async runtime
    pub fn get_tokio_runtime_imports() -> Vec<&'static str> {
        vec![
            "use tokio;",
            "use tokio::runtime::Runtime;",
            "use std::time::Duration;",
        ]
    }

    /// Get tokio::sync imports for synchronization primitives
    pub fn get_tokio_sync_imports() -> Vec<&'static str> {
        vec![
            "use tokio::sync::{Mutex, RwLock, Semaphore, Notify};",
            "use tokio::sync::mpsc;",
            "use std::sync::Arc;",
        ]
    }

    /// Get tokio::time imports for time-based operations
    pub fn get_tokio_time_imports() -> Vec<&'static str> {
        vec![
            "use tokio::time::{sleep, timeout, interval, Duration};",
        ]
    }

    /// Get futures imports for combinators and streams
    pub fn get_futures_imports() -> Vec<&'static str> {
        vec![
            "use futures::future::{join, join_all, select_all};",
            "use futures::stream::{Stream, StreamExt, FuturesUnordered};",
        ]
    }

    /// Get wasm-bindgen-futures for browser async
    pub fn get_wasm_async_imports() -> Vec<&'static str> {
        vec![
            "#[cfg(target_arch = \"wasm32\")]",
            "use wasm_bindgen_futures;",
        ]
    }

    /// Get all async imports based on features used
    pub fn get_imports_for_features(features: &AsyncFeatures) -> Vec<String> {
        let mut imports = Vec::new();

        if features.uses_async_functions {
            imports.extend(Self::get_tokio_runtime_imports().iter().map(|s| s.to_string()));
        }

        if features.uses_sync_primitives {
            imports.extend(Self::get_tokio_sync_imports().iter().map(|s| s.to_string()));
        }

        if features.uses_time_operations {
            imports.extend(Self::get_tokio_time_imports().iter().map(|s| s.to_string()));
        }

        if features.uses_futures_combinators || features.uses_streams {
            imports.extend(Self::get_futures_imports().iter().map(|s| s.to_string()));
        }

        if features.target_wasm {
            imports.extend(Self::get_wasm_async_imports().iter().map(|s| s.to_string()));
        }

        // Add Result and error handling
        if features.uses_async_functions {
            imports.push("use anyhow::Result;".to_string());
        }

        imports
    }

    /// Get Cargo dependencies for async features
    pub fn get_cargo_dependencies() -> Vec<(&'static str, &'static str)> {
        vec![
            ("tokio", "{ version = \"1.35\", features = [\"full\"] }"),
            ("futures", "0.3"),
            ("anyhow", "1.0"),
        ]
    }

    /// Get Cargo dependencies for WASM async
    pub fn get_wasm_cargo_dependencies() -> Vec<(&'static str, &'static str)> {
        vec![
            ("wasm-bindgen-futures", "0.4"),
            ("tokio", "{ version = \"1.35\", features = [\"sync\", \"macros\", \"time\"] }"),
        ]
    }
}

/// Feature detection for async code
#[derive(Debug, Default)]
pub struct AsyncFeatures {
    pub uses_async_functions: bool,
    pub uses_sync_primitives: bool,
    pub uses_time_operations: bool,
    pub uses_futures_combinators: bool,
    pub uses_streams: bool,
    pub target_wasm: bool,
}

/// Pattern detector for Python asyncio code
pub struct AsyncioPatternDetector;

impl AsyncioPatternDetector {
    /// Detect if code uses async def
    pub fn uses_async_def(python_code: &str) -> bool {
        python_code.contains("async def")
    }

    /// Detect if code uses await
    pub fn uses_await(python_code: &str) -> bool {
        python_code.contains("await ")
    }

    /// Detect if code uses asyncio.run()
    pub fn uses_asyncio_run(python_code: &str) -> bool {
        python_code.contains("asyncio.run(")
    }

    /// Detect if code uses asyncio.create_task()
    pub fn uses_create_task(python_code: &str) -> bool {
        python_code.contains("asyncio.create_task(")
            || python_code.contains("create_task(")
    }

    /// Detect if code uses asyncio.gather()
    pub fn uses_gather(python_code: &str) -> bool {
        python_code.contains("asyncio.gather(")
            || python_code.contains("gather(")
    }

    /// Detect if code uses asyncio.sleep()
    pub fn uses_async_sleep(python_code: &str) -> bool {
        python_code.contains("asyncio.sleep(")
    }

    /// Detect if code uses asyncio.Lock
    pub fn uses_async_lock(python_code: &str) -> bool {
        python_code.contains("asyncio.Lock(")
            || python_code.contains("Lock()")
    }

    /// Detect if code uses asyncio.Event
    pub fn uses_async_event(python_code: &str) -> bool {
        python_code.contains("asyncio.Event(")
            || python_code.contains("Event()")
    }

    /// Detect if code uses asyncio.Semaphore
    pub fn uses_async_semaphore(python_code: &str) -> bool {
        python_code.contains("asyncio.Semaphore(")
            || python_code.contains("Semaphore(")
    }

    /// Detect if code uses asyncio.Queue
    pub fn uses_async_queue(python_code: &str) -> bool {
        python_code.contains("asyncio.Queue(")
            || python_code.contains("Queue()")
    }

    /// Detect if code uses async with
    pub fn uses_async_with(python_code: &str) -> bool {
        python_code.contains("async with ")
    }

    /// Detect if code uses async for
    pub fn uses_async_for(python_code: &str) -> bool {
        python_code.contains("async for ")
    }

    /// Analyze Python code and return detected async features
    pub fn analyze_features(python_code: &str) -> AsyncFeatures {
        AsyncFeatures {
            uses_async_functions: Self::uses_async_def(python_code) || Self::uses_await(python_code),
            uses_sync_primitives: Self::uses_async_lock(python_code)
                || Self::uses_async_event(python_code)
                || Self::uses_async_semaphore(python_code)
                || Self::uses_async_queue(python_code),
            uses_time_operations: Self::uses_async_sleep(python_code),
            uses_futures_combinators: Self::uses_gather(python_code)
                || Self::uses_create_task(python_code),
            uses_streams: Self::uses_async_for(python_code),
            target_wasm: false, // Set externally based on build target
        }
    }
}

/// Helper to generate complete async Rust functions
pub struct AsyncFunctionGenerator;

impl AsyncFunctionGenerator {
    /// Generate a complete async function with proper error handling
    pub fn generate_async_function(
        name: &str,
        params: Vec<(&str, &str)>,
        return_type: &str,
        body: &str,
        use_result: bool,
    ) -> String {
        let param_str = params.iter()
            .map(|(name, typ)| format!("{}: {}", name, typ))
            .collect::<Vec<_>>()
            .join(", ");

        let return_str = if use_result {
            format!("Result<{}>", return_type)
        } else {
            return_type.to_string()
        };

        format!(
            r#"async fn {}({}) -> {} {{
    {}
}}"#,
            name, param_str, return_str, body
        )
    }

    /// Generate main function with tokio runtime
    pub fn generate_tokio_main(body: &str) -> String {
        format!(
            r#"#[tokio::main]
async fn main() -> Result<()> {{
    {}
    Ok(())
}}"#,
            body
        )
    }

    /// Generate async block
    pub fn generate_async_block(body: &str) -> String {
        format!(
            r#"async {{
    {}
}}"#,
            body
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_async_function() {
        let result = AsyncioMapper::translate_async_function(
            "fetch_data",
            vec![("url", "String")],
            "Data",
            true,
        );
        assert!(result.contains("async fn fetch_data"));
        assert!(result.contains("url: String"));
        assert!(result.contains("Result<Data>"));
    }

    #[test]
    fn test_translate_await() {
        let result = AsyncioMapper::translate_await("fetch()");
        assert_eq!(result, "fetch().await?");
    }

    #[test]
    fn test_translate_sleep() {
        let result = AsyncioMapper::translate_sleep("1.5");
        assert!(result.contains("Duration::from_secs_f64(1.5)"));

        let result_int = AsyncioMapper::translate_sleep("2");
        assert!(result_int.contains("Duration::from_secs(2)"));
    }

    #[test]
    fn test_translate_create_task() {
        let result = AsyncioMapper::translate_create_task("do_work()");
        assert!(result.contains("tokio::spawn"));
        assert!(result.contains("async move"));
    }

    #[test]
    fn test_translate_gather() {
        let result = AsyncioMapper::translate_gather(vec!["task1()", "task2()", "task3()"]);
        assert!(result.contains("tokio::join!"));
        assert!(result.contains("result1, result2, result3"));
    }

    #[test]
    fn test_translate_wait_for() {
        let result = AsyncioMapper::translate_wait_for("operation()", "5.0");
        assert!(result.contains("tokio::time::timeout"));
        assert!(result.contains("Duration::from_secs_f64(5.0)"));
    }

    #[test]
    fn test_translate_lock() {
        let result = AsyncSyncMapper::translate_lock_creation();
        assert!(result.contains("tokio::sync::Mutex::new"));

        let acquire = AsyncSyncMapper::translate_lock_acquire("lock");
        assert!(acquire.contains("lock.lock().await"));
    }

    #[test]
    fn test_translate_event() {
        let result = AsyncSyncMapper::translate_event_creation();
        assert!(result.contains("tokio::sync::Notify::new"));

        let wait = AsyncSyncMapper::translate_event_wait("event");
        assert!(wait.contains("event.notified().await"));

        let set = AsyncSyncMapper::translate_event_set("event");
        assert!(set.contains("event.notify_one"));
    }

    #[test]
    fn test_translate_semaphore() {
        let result = AsyncSyncMapper::translate_semaphore_creation("5");
        assert!(result.contains("tokio::sync::Semaphore::new(5)"));

        let acquire = AsyncSyncMapper::translate_semaphore_acquire("sem");
        assert!(acquire.contains("sem.acquire().await"));
    }

    #[test]
    fn test_translate_queue() {
        let result = AsyncSyncMapper::translate_queue_creation(Some("10"));
        assert!(result.contains("tokio::sync::mpsc::channel(10)"));

        let put = AsyncSyncMapper::translate_queue_put("tx", "item");
        assert!(put.contains("tx.send(item)"));

        let get = AsyncSyncMapper::translate_queue_get("rx");
        assert!(get.contains("rx.recv().await"));
    }

    #[test]
    fn test_translate_async_with() {
        let result = AsyncContextMapper::translate_async_with(
            "open_connection()",
            Some("conn"),
            "    conn.execute(query).await?;",
        );
        assert!(result.contains("let conn = open_connection().await?"));
        assert!(result.contains("conn.execute(query)"));
        assert!(result.contains("automatically dropped"));
    }

    #[test]
    fn test_translate_async_for() {
        let result = AsyncIteratorMapper::translate_async_for(
            "item",
            "stream",
            "    process(item);",
        );
        assert!(result.contains("while let Some(item) = stream.next().await"));
        assert!(result.contains("use futures::stream::StreamExt"));
    }

    #[test]
    fn test_pattern_detection() {
        let code = r#"
async def main():
    await asyncio.sleep(1)
    task = asyncio.create_task(worker())
    results = await asyncio.gather(task1(), task2())
"#;

        assert!(AsyncioPatternDetector::uses_async_def(code));
        assert!(AsyncioPatternDetector::uses_await(code));
        assert!(AsyncioPatternDetector::uses_async_sleep(code));
        assert!(AsyncioPatternDetector::uses_create_task(code));
        assert!(AsyncioPatternDetector::uses_gather(code));
    }

    #[test]
    fn test_analyze_features() {
        let code = r#"
async def process_data():
    lock = asyncio.Lock()
    async with lock:
        await asyncio.sleep(0.1)
        results = await asyncio.gather(fetch1(), fetch2())
"#;

        let features = AsyncioPatternDetector::analyze_features(code);
        assert!(features.uses_async_functions);
        assert!(features.uses_sync_primitives);
        assert!(features.uses_time_operations);
        assert!(features.uses_futures_combinators);
    }

    #[test]
    fn test_import_generation() {
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
    }

    #[test]
    fn test_generate_async_function() {
        let result = AsyncFunctionGenerator::generate_async_function(
            "process",
            vec![("data", "String")],
            "i32",
            "    // body",
            true,
        );
        assert!(result.contains("async fn process(data: String) -> Result<i32>"));
    }

    #[test]
    fn test_generate_tokio_main() {
        let result = AsyncFunctionGenerator::generate_tokio_main("    run().await?;");
        assert!(result.contains("#[tokio::main]"));
        assert!(result.contains("async fn main()"));
        assert!(result.contains("Result<()>"));
    }
}
