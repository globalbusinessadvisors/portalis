# WASI Async Runtime - Quick Reference

## Import Statement

```rust
use portalis_transpiler::wasi_async_runtime::{
    AsyncRuntime,          // Main runtime
    spawn,                 // Spawn async task
    spawn_local,          // Spawn local task (!Send)
    spawn_blocking,       // Spawn blocking task
    sleep,                // Async sleep
    timeout,              // Timeout wrapper
    yield_now,            // Cooperative yield
    TaskHandle,           // Task handle type
    AsyncError,           // Error type
};
```

## Common Patterns

### Run Async Code
```rust
// Native & WASI
AsyncRuntime::block_on(async {
    // your async code
});

// Browser: use spawn instead
```

### Spawn Tasks
```rust
// Send task (all platforms)
let handle = spawn(async {
    sleep(Duration::from_secs(1)).await;
    42
});

// Local task (!Send types)
let handle = spawn_local(async {
    // Can use !Send types
    42
});

// Blocking task (CPU work)
let handle = spawn_blocking(|| {
    // Blocking computation
    expensive_work()
});
```

### Sleep
```rust
use std::time::Duration;

sleep(Duration::from_secs(1)).await;
sleep(Duration::from_millis(100)).await;
sleep(Duration::from_secs_f64(1.5)).await;
```

### Timeout
```rust
match timeout(Duration::from_secs(5), future).await {
    Ok(result) => println!("Success: {:?}", result),
    Err(AsyncError::Timeout(_)) => println!("Timed out!"),
    Err(e) => println!("Error: {}", e),
}
```

### Yield
```rust
// Let other tasks run
yield_now().await;
```

### Await Task Result
```rust
let handle = spawn(async { 42 });
let result = handle.await.unwrap(); // Result<i32, AsyncError>
```

## Python asyncio → Rust Translation

| Python | Rust |
|--------|------|
| `asyncio.run(main())` | `AsyncRuntime::block_on(async { main().await })` |
| `asyncio.create_task(f())` | `spawn(async { f().await })` |
| `await asyncio.sleep(1)` | `sleep(Duration::from_secs(1)).await` |
| `asyncio.gather(t1, t2)` | `futures::join!(t1, t2)` |
| `asyncio.wait_for(f, timeout=5)` | `timeout(Duration::from_secs(5), f).await?` |
| `asyncio.Queue()` | `tokio::sync::mpsc::unbounded_channel()` |
| `asyncio.Lock()` | `tokio::sync::Mutex::new(())` |
| `asyncio.Event()` | `Arc::new(tokio::sync::Notify::new())` |

## Error Handling

```rust
pub enum AsyncError {
    JoinError(String),     // Task join failed
    Cancelled(String),     // Task was cancelled
    Timeout(Duration),     // Timed out
    Runtime(String),       // Runtime error
    Spawn(String),         // Spawn failed
    Panic(String),         // Task panicked
    PlatformNotSupported(String),
    Other(String),
}
```

## Platform Differences

### Native (tokio)
- ✅ Full task cancellation: `handle.abort()`
- ✅ Task state check: `handle.is_finished()`
- ✅ Blocking task pool: `spawn_blocking()`
- ✅ Multi-threaded executor

### Browser (wasm-bindgen-futures)
- ❌ No `block_on()` - use `spawn()` instead
- ❌ No task cancellation
- ❌ Single-threaded only
- ✅ Browser event loop integration

### WASI (tokio on wasm32-wasi)
- ✅ Full tokio API
- ✅ Current-thread runtime
- ✅ Similar to native behavior

## Examples

### Concurrent Tasks
```rust
let task1 = spawn(async { fetch_data(1).await });
let task2 = spawn(async { fetch_data(2).await });

let r1 = task1.await.unwrap();
let r2 = task2.await.unwrap();
```

### Fan-out/Fan-in
```rust
let tasks: Vec<_> = items.iter()
    .map(|item| spawn(async move { process(*item).await }))
    .collect();

let results: Vec<_> = futures::future::join_all(tasks)
    .await.into_iter()
    .map(|r| r.unwrap())
    .collect();
```

### Pipeline
```rust
let result = stage1(input).await;
let result = stage2(result).await;
let result = stage3(result).await;
```

### Error Propagation
```rust
let handle = spawn(async {
    let data = fetch().await?;
    let processed = process(data).await?;
    Ok::<_, MyError>(processed)
});

match handle.await {
    Ok(Ok(result)) => println!("Success: {:?}", result),
    Ok(Err(e)) => println!("Task error: {}", e),
    Err(e) => println!("Task failed: {}", e),
}
```

## Testing

```bash
# Unit tests
cargo test --lib wasi_async_runtime

# Integration tests
cargo test --test async_runtime_test

# Demo
cargo run --example async_runtime_demo
```

## Performance Tips

1. **Use spawn_blocking for CPU work** (native only)
   ```rust
   spawn_blocking(|| expensive_computation())
   ```

2. **Yield in long loops**
   ```rust
   for item in large_list {
       process(item);
       yield_now().await; // Let other tasks run
   }
   ```

3. **Batch spawns for many tasks**
   ```rust
   let handles: Vec<_> = items.iter()
       .map(|item| spawn(process_item(*item)))
       .collect();
   ```

4. **Use timeout for external operations**
   ```rust
   timeout(Duration::from_secs(30), network_call()).await?
   ```

## Common Patterns

### Retry with Timeout
```rust
for attempt in 1..=3 {
    match timeout(Duration::from_secs(5), operation()).await {
        Ok(result) => return Ok(result),
        Err(AsyncError::Timeout(_)) if attempt < 3 => {
            sleep(Duration::from_secs(1)).await;
            continue;
        }
        Err(e) => return Err(e),
    }
}
```

### Rate Limiting
```rust
let semaphore = Arc::new(tokio::sync::Semaphore::new(5));

for item in items {
    let permit = semaphore.clone().acquire_owned().await.unwrap();
    spawn(async move {
        let _permit = permit; // Drop when done
        process(item).await
    });
}
```

### Cancellation (Native only)
```rust
let handle = spawn(async {
    loop {
        work().await;
        sleep(Duration::from_secs(1)).await;
    }
});

// Later...
handle.abort();
```

## Translation Helpers

```rust
use portalis_transpiler::py_to_rust_asyncio::*;

// Generate Rust code
let code = translate_asyncio_run("main()");
// => "AsyncRuntime::block_on(async { main().await })"

let code = translate_asyncio_sleep(1.5);
// => "sleep(Duration::from_secs_f64(1.5))"

let imports = AsyncioMapping::get_imports(&["asyncio.run(...)", "asyncio.sleep(...)"]);
// => ["use crate::wasi_async_runtime::AsyncRuntime;", ...]
```

## Module Path

```
portalis_transpiler::wasi_async_runtime
├── AsyncRuntime        - Main runtime
├── TaskHandle<T>       - Task handle
├── AsyncError          - Error type
├── spawn              - Functions
├── spawn_local
├── spawn_blocking
├── sleep
├── timeout
└── yield_now
```

---

For detailed documentation, see `ASYNC_RUNTIME_IMPLEMENTATION.md`
