# Async Runtime Requirements for Portalis Transpiler

**Document Type**: Requirements Analysis
**Feature**: Python asyncio → Rust Async Runtime Translation
**Target**: Portalis Python-to-Rust Transpiler
**Date**: October 4, 2025
**Status**: Requirements Analysis Complete

---

## Executive Summary

This document provides comprehensive requirements for implementing Python `asyncio` support in the Portalis transpiler. The implementation will translate Python async/await code to Rust async equivalents using:

1. **Native Rust** - tokio runtime with full multi-threaded executor
2. **Browser WASM** - wasm-bindgen-futures with browser event loop integration
3. **WASI WASM** - tokio with WASI support or async-std fallback

The implementation follows the existing cross-platform architecture patterns demonstrated by `wasi_fetch`, `wasi_websocket`, and `web_workers` modules.

**Key Statistics**:
- **Total Python Async Features**: 32 (from Python 3.12 language spec)
- **Test Files Available**: 32 test cases in `/tests/python-features/async-await/`
- **Current Status**: All marked as `not_implemented`
- **Existing Foundation**: tokio, wasm-bindgen-futures already in Cargo.toml

---

## Table of Contents

1. [Python asyncio APIs to Support](#1-python-asyncio-apis-to-support)
2. [Tokio Runtime Requirements](#2-tokio-runtime-requirements)
3. [wasm-bindgen-futures Requirements](#3-wasm-bindgen-futures-requirements)
4. [Platform-Specific APIs](#4-platform-specific-apis)
5. [Async Translation Patterns](#5-async-translation-patterns)
6. [Error Handling Requirements](#6-error-handling-requirements)
7. [Integration with Existing Async Code](#7-integration-with-existing-async-code)
8. [Implementation Architecture](#8-implementation-architecture)
9. [Dependencies and Version Requirements](#9-dependencies-and-version-requirements)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Python asyncio APIs to Support

### 1.1 Async Functions (8 features - Category 11.1)

#### 1.1.1 Async Function Definition
**Complexity**: High
**Python**:
```python
async def fetch_data():
    return data
```
**Rust**:
```rust
async fn fetch_data() -> Data {
    data
}
```
**Requirements**:
- Translate `async def` to `async fn`
- Preserve function signature (parameters, return types)
- Handle type inference for return types

#### 1.1.2 Await Expression
**Complexity**: High
**Python**:
```python
result = await async_operation()
```
**Rust**:
```rust
let result = async_operation().await;
```
**Requirements**:
- Translate `await expr` to `expr.await`
- Ensure proper error propagation with `?` operator
- Handle chained awaits

#### 1.1.3 Async Function with Parameters
**Complexity**: High
**Python**:
```python
async def fetch(url: str):
    return await http_get(url)
```
**Rust**:
```rust
async fn fetch(url: &str) -> Result<Response> {
    http_get(url).await
}
```
**Requirements**:
- Translate parameter types appropriately
- Handle string → &str conversions
- Support default parameters

#### 1.1.4 Async Function Return Type
**Complexity**: High
**Python**:
```python
async def get_user(id: int) -> User:
    return await db.fetch_user(id)
```
**Rust**:
```rust
async fn get_user(id: i32) -> User {
    db.fetch_user(id).await
}
```
**Requirements**:
- Translate Python type hints to Rust types
- Wrap in `Result<T>` when appropriate
- Handle Optional types

#### 1.1.5 Multiple Awaits
**Complexity**: High
**Python**:
```python
async def process():
    data = await fetch()
    result = await process(data)
    await save(result)
```
**Rust**:
```rust
async fn process() -> Result<()> {
    let data = fetch().await?;
    let result = process(data).await?;
    save(result).await?;
    Ok(())
}
```
**Requirements**:
- Sequential await execution
- Proper error propagation
- Variable lifetime management

#### 1.1.6 Async Lambda
**Complexity**: Very High
**Python**:
```python
# Not directly supported in Python
# But can be approximated with lambda returning coroutine
```
**Rust**:
```rust
// Async closures are unstable in Rust
// Use async blocks: || async move { ... }
```
**Requirements**:
- Async blocks as workaround
- Proper capture semantics
- Future trait bounds

#### 1.1.7 Async Generator
**Complexity**: Very High
**Python**:
```python
async def async_gen():
    for i in range(10):
        await asyncio.sleep(0.1)
        yield i
```
**Rust**:
```rust
fn async_gen() -> impl Stream<Item = i32> {
    stream! {
        for i in 0..10 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            yield i;
        }
    }
}
```
**Requirements**:
- Use `async-stream` crate
- Implement `Stream` trait
- Handle backpressure

#### 1.1.8 Async Comprehension
**Complexity**: Very High
**Python**:
```python
results = [await fetch(url) async for url in urls]
```
**Rust**:
```rust
use futures::stream::{self, StreamExt};
let results: Vec<_> = stream::iter(urls)
    .then(|url| fetch(url))
    .collect()
    .await;
```
**Requirements**:
- Use `futures::stream`
- `StreamExt` trait methods
- Collect results

---

### 1.2 Asyncio Primitives (12 features - Category 11.2)

#### 1.2.1 asyncio.run()
**Complexity**: High
**Python**:
```python
asyncio.run(main())
```
**Rust Native**:
```rust
#[tokio::main]
async fn main() {
    // or
    tokio::runtime::Runtime::new().unwrap().block_on(main())
}
```
**Rust Browser**:
```rust
use wasm_bindgen_futures::spawn_local;
spawn_local(async {
    main().await;
});
```
**Requirements**:
- Native: Create tokio runtime, call `block_on`
- Browser: Use `spawn_local` from wasm-bindgen-futures
- WASI: Use tokio with WASI executor

#### 1.2.2 asyncio.create_task()
**Complexity**: High
**Python**:
```python
task = asyncio.create_task(coro())
```
**Rust Native**:
```rust
let task = tokio::spawn(coro());
```
**Rust Browser**:
```rust
use wasm_bindgen_futures::spawn_local;
spawn_local(coro());
```
**Requirements**:
- Native: `tokio::spawn` returns `JoinHandle<T>`
- Browser: `spawn_local` for non-Send futures
- Task cancellation support
- JoinHandle error handling

#### 1.2.3 asyncio.gather()
**Complexity**: High
**Python**:
```python
results = await asyncio.gather(coro1(), coro2(), coro3())
```
**Rust**:
```rust
use tokio::join;
let (r1, r2, r3) = join!(coro1(), coro2(), coro3());
// or
use futures::future::join_all;
let results = join_all(vec![coro1(), coro2(), coro3()]).await;
```
**Requirements**:
- Fixed number of tasks: use `tokio::join!` macro
- Dynamic tasks: use `futures::future::join_all`
- Collect all results (even if some fail)
- Return tuple or Vec based on usage

#### 1.2.4 asyncio.wait()
**Complexity**: High
**Python**:
```python
done, pending = await asyncio.wait(tasks, timeout=5.0, return_when=FIRST_COMPLETED)
```
**Rust**:
```rust
use tokio::select;
use futures::future::FutureExt;

tokio::select! {
    result = task1 => { /* handle result */ }
    result = task2 => { /* handle result */ }
    _ = tokio::time::sleep(Duration::from_secs(5)) => { /* timeout */ }
}
```
**Requirements**:
- Use `tokio::select!` for multiple futures
- Support timeout parameter
- Return done and pending sets
- FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED modes

#### 1.2.5 asyncio.sleep()
**Complexity**: Medium
**Python**:
```python
await asyncio.sleep(1.0)
```
**Rust Native**:
```rust
tokio::time::sleep(Duration::from_secs_f64(1.0)).await
```
**Rust Browser**:
```rust
use wasm_timer::Delay;
Delay::new(Duration::from_secs_f64(1.0)).await.ok();
```
**Requirements**:
- Native: `tokio::time::sleep`
- Browser: `wasm_timer::Delay` or `gloo_timers::future::sleep`
- WASI: tokio sleep with WASI runtime
- Support fractional seconds

#### 1.2.6 asyncio.Queue
**Complexity**: High
**Python**:
```python
queue = asyncio.Queue(maxsize=10)
await queue.put(item)
item = await queue.get()
queue.task_done()
await queue.join()
```
**Rust Native**:
```rust
use tokio::sync::mpsc;
let (tx, mut rx) = mpsc::channel(10);
tx.send(item).await?;
let item = rx.recv().await;
```
**Requirements**:
- Use `tokio::sync::mpsc` for MPSC queue
- Support bounded/unbounded queues
- Implement `put()`, `get()`, `put_nowait()`, `get_nowait()`
- Implement `task_done()` and `join()` semantics
- Browser: Use async channels from futures crate

#### 1.2.7 asyncio.Lock
**Complexity**: High
**Python**:
```python
lock = asyncio.Lock()
async with lock:
    critical_section()
# or
await lock.acquire()
try:
    critical_section()
finally:
    lock.release()
```
**Rust**:
```rust
use tokio::sync::Mutex;
let lock = Mutex::new(());
let guard = lock.lock().await;
// critical section
drop(guard);
```
**Requirements**:
- Use `tokio::sync::Mutex` (async mutex)
- Not `std::sync::Mutex` (blocking)
- Translate `async with lock:` to lock guard pattern
- Support `acquire()` and `release()` methods

#### 1.2.8 asyncio.Semaphore
**Complexity**: High
**Python**:
```python
sem = asyncio.Semaphore(10)
async with sem:
    limited_operation()
```
**Rust**:
```rust
use tokio::sync::Semaphore;
let sem = Semaphore::new(10);
let permit = sem.acquire().await.unwrap();
limited_operation();
drop(permit);
```
**Requirements**:
- Use `tokio::sync::Semaphore`
- Support async with context manager
- `acquire()` and `release()` methods
- Track permit counts

#### 1.2.9 asyncio.Event
**Complexity**: High
**Python**:
```python
event = asyncio.Event()
event.set()
event.clear()
await event.wait()
is_set = event.is_set()
```
**Rust**:
```rust
use tokio::sync::Notify;
let event = Notify::new();
event.notify_one();  // or notify_waiters()
event.notified().await;
```
**Requirements**:
- Use `tokio::sync::Notify` (closest equivalent)
- Or implement custom Event with `Mutex<bool>` + `Condvar`
- Support `set()`, `clear()`, `wait()`, `is_set()`
- Multiple waiters support

#### 1.2.10 asyncio.timeout()
**Complexity**: High
**Python**:
```python
async with asyncio.timeout(10):
    await operation()
```
**Rust**:
```rust
use tokio::time::{timeout, Duration};
match timeout(Duration::from_secs(10), operation()).await {
    Ok(result) => result,
    Err(_) => panic!("Timeout"),
}
```
**Requirements**:
- Use `tokio::time::timeout`
- Return `Result<T, Elapsed>`
- Context manager translation to match pattern
- Nested timeout support

#### 1.2.11 asyncio.shield()
**Complexity**: Very High
**Python**:
```python
task = asyncio.shield(coro())
# Task continues even if outer scope is cancelled
```
**Rust**:
```rust
// Custom implementation needed
// Spawn task and keep JoinHandle alive
let handle = tokio::spawn(coro());
// Don't cancel even if current future is dropped
```
**Requirements**:
- Spawn independent task
- Prevent cancellation propagation
- Return wrapper that doesn't cancel on drop
- Complex cancellation semantics

#### 1.2.12 asyncio.wait_for()
**Complexity**: High
**Python**:
```python
result = await asyncio.wait_for(coro(), timeout=5.0)
```
**Rust**:
```rust
use tokio::time::{timeout, Duration};
let result = timeout(Duration::from_secs_f64(5.0), coro()).await??;
```
**Requirements**:
- Wrapper around `tokio::time::timeout`
- Raise TimeoutError on timeout
- Cancel task on timeout
- Support fractional seconds

---

### 1.3 Async Iteration (6 features - Category 11.3)

#### 1.3.1 Async For Loop
**Complexity**: Very High
**Python**:
```python
async for item in async_iterable:
    process(item)
```
**Rust**:
```rust
use futures::StreamExt;
let mut stream = async_iterable();
while let Some(item) = stream.next().await {
    process(item);
}
```
**Requirements**:
- Translate to `Stream` iteration
- Use `StreamExt::next()`
- Handle `StopAsyncIteration`

#### 1.3.2 __aiter__ Method
**Complexity**: Very High
**Python**:
```python
def __aiter__(self):
    return self
```
**Rust**:
```rust
// Implement Stream trait
impl Stream for MyType {
    type Item = T;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<T>> {
        // ...
    }
}
```
**Requirements**:
- Implement `Stream` trait
- Pin projection for self-referential types
- Proper waker registration

#### 1.3.3 __anext__ Method
**Complexity**: Very High
**Python**:
```python
async def __anext__(self):
    if done:
        raise StopAsyncIteration
    return value
```
**Rust**:
```rust
// Part of Stream trait implementation
fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<T>> {
    if done {
        Poll::Ready(None)
    } else {
        Poll::Ready(Some(value))
    }
}
```
**Requirements**:
- Return `Poll::Ready(None)` for end
- Return `Poll::Ready(Some(value))` for value
- Return `Poll::Pending` when not ready

#### 1.3.4 Async Generator
**Complexity**: Very High
**Python**:
```python
async def async_range(n):
    for i in range(n):
        await asyncio.sleep(0)
        yield i
```
**Rust**:
```rust
use async_stream::stream;
fn async_range(n: i32) -> impl Stream<Item = i32> {
    stream! {
        for i in 0..n {
            tokio::time::sleep(Duration::ZERO).await;
            yield i;
        }
    }
}
```
**Requirements**:
- Use `async-stream` crate
- `stream!` macro for generator syntax
- Yield values asynchronously

#### 1.3.5 Async Comprehension
**Complexity**: Very High
**Python**:
```python
result = [x async for x in async_gen()]
```
**Rust**:
```rust
use futures::StreamExt;
let result: Vec<_> = async_gen().collect().await;
```
**Requirements**:
- Stream collect operations
- Filter, map, filter_map support
- Proper type inference

#### 1.3.6 Async Generator Expression
**Complexity**: Very High
**Python**:
```python
gen = (x async for x in async_source())
```
**Rust**:
```rust
let gen = async_source(); // Already a Stream
```
**Requirements**:
- Return Stream directly
- Chain stream operations
- Lazy evaluation

---

### 1.4 Async Context Managers (6 features - Category 11.4)

#### 1.4.1 Async With
**Complexity**: Very High
**Python**:
```python
async with async_context_manager() as resource:
    await use(resource)
```
**Rust**:
```rust
let resource = async_context_manager().await?;
// Use resource
drop(resource); // Calls async drop when implemented
```
**Requirements**:
- Call async constructor
- Hold resource in scope
- Call async cleanup on drop (future Rust feature)
- Current workaround: manual cleanup calls

#### 1.4.2 __aenter__ Method
**Complexity**: Very High
**Python**:
```python
async def __aenter__(self):
    await self.connect()
    return self
```
**Rust**:
```rust
async fn enter(&mut self) -> Result<&mut Self> {
    self.connect().await?;
    Ok(self)
}
```
**Requirements**:
- Async initialization method
- Return self or resource
- Error propagation

#### 1.4.3 __aexit__ Method
**Complexity**: Very High
**Python**:
```python
async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()
```
**Rust**:
```rust
async fn exit(&mut self, error: Option<Error>) -> Result<()> {
    self.close().await
}
```
**Requirements**:
- Async cleanup method
- Receive exception info
- Return whether to suppress exception

#### 1.4.4 contextlib.asynccontextmanager
**Complexity**: Very High
**Python**:
```python
@asynccontextmanager
async def transaction():
    await begin()
    yield
    await commit()
```
**Rust**:
```rust
// Custom guard struct
struct TransactionGuard;
impl Drop for TransactionGuard {
    fn drop(&mut self) {
        // Spawn cleanup task
    }
}
async fn transaction() -> TransactionGuard {
    begin().await;
    TransactionGuard
}
```
**Requirements**:
- Custom guard types
- Async setup and teardown
- Yield semantics

#### 1.4.5 Multiple Async Context Managers
**Complexity**: Very High
**Python**:
```python
async with cm1() as r1, cm2() as r2:
    await use(r1, r2)
```
**Rust**:
```rust
let r1 = cm1().await?;
let r2 = cm2().await?;
use_both(r1, r2).await?;
drop(r2);
drop(r1);
```
**Requirements**:
- Nested async initialization
- Proper cleanup order (reverse)
- Error handling at each stage

#### 1.4.6 Async Context Manager Exception Handling
**Complexity**: Very High
**Python**:
```python
async def __aexit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None:
        await handle_error(exc_val)
    await cleanup()
    return True  # Suppress exception
```
**Rust**:
```rust
async fn exit(&mut self, error: Option<Error>) -> Result<bool> {
    if let Some(e) = error {
        handle_error(e).await?;
    }
    cleanup().await?;
    Ok(true) // Suppress
}
```
**Requirements**:
- Receive exception information
- Handle or propagate errors
- Control exception suppression

---

## 2. Tokio Runtime Requirements

### 2.1 Runtime Creation and Configuration

**Native Platform**:
```rust
// Option 1: Macro (preferred)
#[tokio::main]
async fn main() {
    // Runtime automatically created
}

// Option 2: Manual runtime
use tokio::runtime::Runtime;
let rt = Runtime::new().unwrap();
rt.block_on(async {
    // async code
});

// Option 3: Custom configuration
use tokio::runtime::Builder;
let rt = Builder::new_multi_thread()
    .worker_threads(4)
    .thread_name("my-worker")
    .enable_all()
    .build()
    .unwrap();
```

**Requirements**:
- Default: Multi-threaded runtime
- Configurable worker thread count
- Enable IO and time drivers
- Thread naming for debugging

### 2.2 Task Spawning

**tokio::spawn**:
```rust
use tokio::task::JoinHandle;

// Spawn task (requires Send)
let handle: JoinHandle<i32> = tokio::spawn(async {
    42
});

let result = handle.await.unwrap();
```

**tokio::spawn_blocking**:
```rust
// For blocking operations
let result = tokio::task::spawn_blocking(|| {
    // CPU-intensive or blocking operation
    expensive_computation()
}).await.unwrap();
```

**Requirements**:
- `spawn` for async tasks (Send bound)
- `spawn_blocking` for sync/blocking code
- JoinHandle for task results
- Panic handling and propagation

### 2.3 Task Joining

```rust
use tokio::task::JoinHandle;

let handle1 = tokio::spawn(task1());
let handle2 = tokio::spawn(task2());

// Wait for single task
let result1 = handle1.await.unwrap();

// Join multiple tasks
let (r1, r2) = tokio::join!(task1(), task2());

// Join all in vector
use futures::future::join_all;
let handles = vec![tokio::spawn(task1()), tokio::spawn(task2())];
let results = join_all(handles).await;
```

**Requirements**:
- Single task await
- Multiple task join (macro)
- Dynamic task collection
- Error handling for panics

### 2.4 Sleep and Timeout

**Sleep**:
```rust
use tokio::time::{sleep, Duration};
sleep(Duration::from_secs(1)).await;
sleep(Duration::from_millis(500)).await;
sleep(Duration::from_secs_f64(1.5)).await;
```

**Timeout**:
```rust
use tokio::time::{timeout, Duration};

match timeout(Duration::from_secs(5), operation()).await {
    Ok(result) => println!("Got result: {:?}", result),
    Err(_) => println!("Timeout!"),
}
```

**Requirements**:
- Precise timing with async sleep
- Timeout wrapper for any future
- Fractional second support
- Cancellation on timeout

### 2.5 Select Operations

**tokio::select!**:
```rust
use tokio::time::{sleep, Duration};

tokio::select! {
    val = async_operation() => {
        println!("Operation completed with {}", val);
    }
    _ = sleep(Duration::from_secs(5)) => {
        println!("Timeout!");
    }
}
```

**Requirements**:
- Multiple future racing
- First completed wins
- Biased/unbiased selection
- Branch syntax with =>

### 2.6 Channels

**MPSC (Multi-Producer, Single-Consumer)**:
```rust
use tokio::sync::mpsc;

// Bounded channel
let (tx, mut rx) = mpsc::channel(100);

// Send
tx.send(42).await.unwrap();

// Receive
let value = rx.recv().await.unwrap();

// Unbounded
let (tx, mut rx) = mpsc::unbounded_channel();
tx.send(42).unwrap(); // Non-async
```

**Oneshot**:
```rust
use tokio::sync::oneshot;

let (tx, rx) = oneshot::channel();
tx.send(42).unwrap();
let value = rx.await.unwrap();
```

**Broadcast**:
```rust
use tokio::sync::broadcast;

let (tx, mut rx1) = broadcast::channel(16);
let mut rx2 = tx.subscribe();

tx.send(42).unwrap();
let val1 = rx1.recv().await.unwrap();
let val2 = rx2.recv().await.unwrap();
```

**Watch**:
```rust
use tokio::sync::watch;

let (tx, mut rx) = watch::channel("hello");
tx.send("world").unwrap();
let value = rx.borrow();
```

**Requirements**:
- MPSC for queue semantics
- Oneshot for single-value futures
- Broadcast for multiple consumers
- Watch for latest-value semantics

### 2.7 Async Synchronization Primitives

**Mutex**:
```rust
use tokio::sync::Mutex;

let mutex = Mutex::new(0);
let mut guard = mutex.lock().await;
*guard += 1;
```

**RwLock**:
```rust
use tokio::sync::RwLock;

let lock = RwLock::new(0);
let read_guard = lock.read().await;
let write_guard = lock.write().await;
```

**Semaphore**:
```rust
use tokio::sync::Semaphore;

let sem = Semaphore::new(3);
let permit = sem.acquire().await.unwrap();
// Do work
drop(permit);
```

**Notify**:
```rust
use tokio::sync::Notify;

let notify = Notify::new();
notify.notify_one();
notify.notified().await;
```

**Barrier**:
```rust
use tokio::sync::Barrier;

let barrier = Barrier::new(3);
barrier.wait().await;
```

**Requirements**:
- Async-aware synchronization (no blocking)
- Mutex for exclusive access
- RwLock for reader-writer pattern
- Semaphore for resource limiting
- Notify for event signaling
- Barrier for synchronization points

---

## 3. wasm-bindgen-futures Requirements

### 3.1 JsFuture for Promise Integration

**Basic Usage**:
```rust
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use web_sys::window;

let promise = window().unwrap().fetch_with_str("https://api.example.com/data");
let js_value = JsFuture::from(promise).await.unwrap();
```

**Requirements**:
- Convert JavaScript Promise to Rust Future
- Error handling for rejected promises
- Type conversion from JsValue
- Integration with web-sys APIs

### 3.2 spawn_local for Browser Task Spawning

**Usage**:
```rust
use wasm_bindgen_futures::spawn_local;

spawn_local(async {
    let data = fetch_data().await;
    process_data(data);
});
```

**Requirements**:
- Spawn non-Send futures in browser
- Fire-and-forget task execution
- Integration with browser event loop
- No JoinHandle (detached tasks)

### 3.3 Async Function Exports to JavaScript

**Export async functions**:
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub async fn fetch_user(id: u32) -> Result<JsValue, JsValue> {
    let user = fetch_user_from_api(id).await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(serde_wasm_bindgen::to_value(&user)?)
}
```

**Requirements**:
- `#[wasm_bindgen]` on async fn
- Return Result<JsValue, JsValue>
- Automatic Promise return type in JS
- Error conversion to JS exceptions

### 3.4 Promise Return Types

**JavaScript sees**:
```javascript
// Rust async fn becomes Promise in JS
const user = await fetch_user(42);
```

**Requirements**:
- Automatic Promise wrapping
- Proper error propagation
- Type serialization/deserialization
- Integration with JS async/await

### 3.5 Browser Event Loop Integration

**Considerations**:
```rust
// Browser event loop is single-threaded
// No std::thread::spawn equivalent
// Use spawn_local or Web Workers

use wasm_bindgen_futures::spawn_local;

// This works
spawn_local(async {
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        update_ui();
    }
});

// This doesn't work - no Send
// tokio::spawn(async { ... }); // ERROR: future is not Send
```

**Requirements**:
- Single-threaded executor integration
- No Send/Sync requirements
- Yield to browser event loop
- Cooperative multitasking

---

## 4. Platform-Specific APIs

### 4.1 Native Platform (tokio)

**Runtime Configuration**:
```rust
use tokio::runtime::Builder;

let runtime = Builder::new_multi_thread()
    .worker_threads(num_cpus::get())
    .thread_name("portalis-worker")
    .thread_stack_size(3 * 1024 * 1024)
    .enable_all() // Enable IO and time
    .build()
    .unwrap();
```

**Features**:
- Multi-threaded work-stealing scheduler
- IO driver (TCP, UDP, Unix sockets)
- Time driver (sleep, interval, timeout)
- Blocking thread pool
- Send + Sync futures

**Requirements**:
- Default to multi-threaded runtime
- Configurable thread count
- Enable all drivers by default
- Proper shutdown handling

### 4.2 Browser Platform (wasm-bindgen-futures)

**Runtime Configuration**:
```rust
// No explicit runtime - uses browser event loop
use wasm_bindgen_futures::spawn_local;
use wasm_timer::Delay;
```

**Features**:
- Single-threaded event loop
- Promise integration via JsFuture
- setTimeout/setInterval for timing
- No Send/Sync requirements
- Web Workers for parallelism

**Requirements**:
- Use `spawn_local` instead of `tokio::spawn`
- Use `wasm_timer` or `gloo_timers` for sleep
- No blocking operations
- Integration with web-sys APIs

**Dependencies**:
```toml
wasm-bindgen-futures = "0.4"
wasm-timer = "0.2"
gloo-timers = { version = "0.3", features = ["futures"] }
futures = "0.3"
```

### 4.3 WASI Platform (tokio with WASI or async-std)

**Option 1: tokio with WASI**:
```rust
// Use tokio with wasi feature
[dependencies]
tokio = { version = "1", features = ["rt", "time", "io-util", "sync"] }
```

**Option 2: async-std**:
```rust
use async_std::task;

task::block_on(async {
    // async code
});
```

**Requirements**:
- Limited runtime features (no threads in some WASI runtimes)
- IO operations via WASI preview 2
- Time operations supported
- Sync primitives available
- Fallback to single-threaded executor

---

## 5. Async Translation Patterns

### 5.1 Function Signatures

| Python | Rust |
|--------|------|
| `async def func():` | `async fn func()` |
| `async def func() -> int:` | `async fn func() -> i32` |
| `async def func() -> Optional[int]:` | `async fn func() -> Option<i32>` |
| `async def func() -> int:` (can raise) | `async fn func() -> Result<i32>` |

### 5.2 Await Expressions

| Python | Rust |
|--------|------|
| `result = await coro()` | `let result = coro().await;` |
| `await coro()` (discard) | `coro().await;` |
| `x = await coro()?` (Python) | `let x = coro().await?;` |

### 5.3 Task Creation

| Python | Rust Native | Rust Browser |
|--------|-------------|--------------|
| `asyncio.create_task(coro())` | `tokio::spawn(coro())` | `spawn_local(coro())` |
| `asyncio.run(main())` | `tokio::runtime::Runtime::new().unwrap().block_on(main())` | `spawn_local(main())` |

### 5.4 Concurrent Execution

| Python | Rust |
|--------|------|
| `await asyncio.gather(a(), b())` | `tokio::join!(a(), b())` |
| `results = await asyncio.gather(*tasks)` | `futures::future::join_all(tasks).await` |
| `done, pending = await asyncio.wait(tasks)` | `tokio::select!` with multiple branches |

### 5.5 Synchronization

| Python | Rust |
|--------|------|
| `asyncio.sleep(1.0)` | `tokio::time::sleep(Duration::from_secs_f64(1.0)).await` |
| `asyncio.Queue()` | `tokio::sync::mpsc::channel()` |
| `asyncio.Lock()` | `tokio::sync::Mutex::new(())` |
| `asyncio.Semaphore(n)` | `tokio::sync::Semaphore::new(n)` |
| `asyncio.Event()` | `tokio::sync::Notify::new()` |

### 5.6 Timeout and Cancellation

| Python | Rust |
|--------|------|
| `await asyncio.wait_for(coro(), 5.0)` | `tokio::time::timeout(Duration::from_secs(5), coro()).await??` |
| `async with asyncio.timeout(10):` | `match tokio::time::timeout(Duration::from_secs(10), op).await { ... }` |
| `task.cancel()` | `handle.abort()` (tokio) |

### 5.7 Context Managers

| Python | Rust |
|--------|------|
| `async with lock:` | `let _guard = lock.lock().await;` |
| `async with sem:` | `let _permit = sem.acquire().await.unwrap();` |
| `async with timeout(10):` | `tokio::time::timeout(Duration::from_secs(10), ...).await` |

---

## 6. Error Handling Requirements

### 6.1 JoinError for Task Panics

**Handling panicked tasks**:
```rust
use tokio::task::JoinError;

let handle = tokio::spawn(async {
    panic!("Task panicked!");
});

match handle.await {
    Ok(result) => println!("Task succeeded: {:?}", result),
    Err(e) if e.is_panic() => {
        println!("Task panicked: {:?}", e);
    }
    Err(e) if e.is_cancelled() => {
        println!("Task was cancelled");
    }
    Err(e) => println!("Other error: {:?}", e),
}
```

**Requirements**:
- Catch panics in spawned tasks
- Distinguish panic from cancellation
- Propagate or handle panics
- Log task failures

### 6.2 Timeout Errors

**tokio::time::error::Elapsed**:
```rust
use tokio::time::{timeout, Duration, error::Elapsed};

match timeout(Duration::from_secs(5), slow_operation()).await {
    Ok(result) => handle_result(result),
    Err(e: Elapsed) => {
        eprintln!("Operation timed out after 5 seconds");
        handle_timeout()
    }
}
```

**Requirements**:
- Timeout error type
- Distinguish from other errors
- Configurable timeout values
- Proper error messages

### 6.3 Cancellation Handling

**Task cancellation**:
```rust
use tokio::task::JoinHandle;
use tokio::sync::oneshot;

let (cancel_tx, cancel_rx) = oneshot::channel();
let handle = tokio::spawn(async move {
    tokio::select! {
        result = long_running_task() => result,
        _ = cancel_rx => {
            println!("Task cancelled");
            return Err("Cancelled");
        }
    }
});

// Cancel the task
cancel_tx.send(()).unwrap();
```

**Requirements**:
- Graceful cancellation via select
- Cleanup on cancellation
- Cancellation token pattern
- Abort for forceful termination

### 6.4 Error Propagation with ?

**Async functions with Result**:
```rust
async fn fetch_and_process(url: &str) -> Result<Data, Error> {
    let response = fetch(url).await?; // Propagate fetch error
    let data = parse(response).await?; // Propagate parse error
    validate(data).await?; // Propagate validation error
    Ok(data)
}
```

**Requirements**:
- Use `?` operator for error propagation
- Wrap non-Result futures in Result
- Custom error types with thiserror
- Error context with anyhow

---

## 7. Integration with Existing Async Code

### 7.1 wasi_fetch.rs Integration

**Current Usage**:
```rust
// From wasi_fetch.rs
use wasm_bindgen_futures::JsFuture;

pub async fn fetch(request: Request) -> Result<Response> {
    #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
    {
        Self::fetch_browser(request).await
    }
}

async fn fetch_browser(request: Request) -> Result<Response> {
    let resp_value = JsFuture::from(window.fetch_with_request(&js_request))
        .await
        .map_err(|e| FetchError::Connection(format!("Fetch failed: {:?}", e)))?;
    // ...
}
```

**Integration Pattern**:
- Already uses async/await
- Already uses JsFuture for browser
- Already uses tokio for native (reqwest)
- Pattern to follow for asyncio translation

### 7.2 wasi_websocket Integration

**Current Usage**:
```rust
// From wasi_websocket/native.rs
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, WebSocketStream};

pub async fn connect(config: WebSocketConfig, shared: WebSocketSharedState) -> Result<Self> {
    let (ws_stream, response) = connect_async(&config.url).await?;
    // ...
}

fn start_receive_task(&self) {
    let stream = Arc::clone(&self.stream);
    tokio::spawn(async move {
        loop {
            let mut stream_guard = stream.lock().await;
            // ...
        }
    });
}
```

**Integration Pattern**:
- Uses tokio::spawn for background tasks
- Uses tokio::sync::Mutex for async synchronization
- Pattern for long-running async tasks
- Message passing between tasks

### 7.3 Common Patterns from Existing Code

**Platform Dispatch**:
```rust
pub async fn operation() -> Result<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Self::operation_native().await
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    {
        Self::operation_wasi().await
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
    {
        Self::operation_browser().await
    }
}
```

**Error Handling**:
```rust
use anyhow::{Result, Context, anyhow};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MyError {
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Timeout: {0:?}")]
    Timeout(Duration),
}

pub async fn operation() -> Result<T> {
    something().await
        .context("Failed to do something")?
}
```

---

## 8. Implementation Architecture

### 8.1 Module Structure

Proposed structure following existing patterns:

```
agents/transpiler/src/
├── asyncio_runtime/
│   ├── mod.rs              # Public API and platform dispatch
│   ├── native.rs           # Tokio runtime for native
│   ├── browser.rs          # wasm-bindgen-futures for browser
│   ├── wasi_impl.rs        # WASI implementation
│   ├── primitives.rs       # Queue, Lock, Semaphore, Event wrappers
│   ├── task.rs             # Task spawning and joining
│   ├── timeout.rs          # Timeout utilities
│   ├── stream.rs           # Async iteration support
│   └── context.rs          # Async context manager support
├── stdlib_mappings_comprehensive.rs  # Add asyncio mappings
└── python_to_rust.rs       # Add async/await translation
```

### 8.2 Core Types

**RuntimeHandle**:
```rust
pub struct RuntimeHandle {
    #[cfg(not(target_arch = "wasm32"))]
    inner: tokio::runtime::Handle,

    #[cfg(target_arch = "wasm32")]
    _phantom: std::marker::PhantomData<()>,
}

impl RuntimeHandle {
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static;
}
```

**AsyncQueue (Queue wrapper)**:
```rust
pub struct AsyncQueue<T> {
    #[cfg(not(target_arch = "wasm32"))]
    inner: tokio::sync::mpsc::UnboundedSender<T>,

    #[cfg(target_arch = "wasm32")]
    inner: futures::channel::mpsc::UnboundedSender<T>,
}

impl<T> AsyncQueue<T> {
    pub fn new(maxsize: usize) -> Self;
    pub async fn put(&self, item: T) -> Result<()>;
    pub async fn get(&mut self) -> Option<T>;
}
```

**AsyncLock**:
```rust
pub struct AsyncLock {
    #[cfg(not(target_arch = "wasm32"))]
    inner: tokio::sync::Mutex<()>,

    #[cfg(target_arch = "wasm32")]
    inner: futures::lock::Mutex<()>,
}

impl AsyncLock {
    pub fn new() -> Self;
    pub async fn acquire(&self) -> AsyncLockGuard;
}
```

### 8.3 Translation Strategy

**In python_to_rust.rs**:

1. **Detect async functions**:
```rust
fn translate_function(&mut self, func: &PyFunction) -> String {
    if func.is_async {
        format!("async fn {}({}) -> {} {{\n{}\n}}",
            func.name,
            self.translate_params(&func.params),
            self.translate_return_type(&func.return_type),
            self.translate_body(&func.body)
        )
    } else {
        // Regular function
    }
}
```

2. **Translate await expressions**:
```rust
fn translate_expr(&mut self, expr: &PyExpr) -> String {
    match expr {
        PyExpr::Await(inner) => {
            format!("{}.await", self.translate_expr(inner))
        }
        // Other expressions...
    }
}
```

3. **Map asyncio calls**:
```rust
fn translate_call(&mut self, call: &PyCall) -> String {
    if call.func == "asyncio.sleep" {
        let duration = self.translate_expr(&call.args[0]);
        format!("tokio::time::sleep(Duration::from_secs_f64({})).await", duration)
    } else if call.func == "asyncio.create_task" {
        let coro = self.translate_expr(&call.args[0]);
        format!("tokio::spawn({})", coro)
    }
    // More mappings...
}
```

---

## 9. Dependencies and Version Requirements

### 9.1 Core Dependencies

**Already in Cargo.toml**:
```toml
[dependencies]
tokio = { version = "1.35", features = ["full"] }
wasm-bindgen-futures = { version = "0.4", optional = true }
```

**Additional Required**:
```toml
# Async utilities
futures = "0.3"
async-trait = "0.1"

# Async streams
async-stream = "0.3"
futures-util = { version = "0.3", features = ["stream"] }

# WASM timers
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-timer = "0.2"
gloo-timers = { version = "0.3", features = ["futures"] }
```

### 9.2 Platform-Specific Features

**Tokio Features**:
```toml
tokio = { version = "1.35", features = [
    "rt-multi-thread",  # Multi-threaded runtime
    "sync",             # Async synchronization primitives
    "time",             # Sleep, timeout, interval
    "macros",           # #[tokio::main], join!, select!
    "io-util",          # Async IO utilities
    "fs",               # Async file operations
    "net",              # Async networking
    "signal",           # Unix signal handling
] }
```

**wasm-bindgen Features**:
```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2", features = ["serde-serialize"] }
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = { version = "0.3", features = [
    "Window",
    "Performance",
    "console",
] }
```

### 9.3 Version Compatibility Matrix

| Crate | Version | Native | Browser | WASI |
|-------|---------|--------|---------|------|
| tokio | 1.35+ | ✅ | ❌ | ✅ |
| wasm-bindgen | 0.2 | ❌ | ✅ | ❌ |
| wasm-bindgen-futures | 0.4 | ❌ | ✅ | ❌ |
| futures | 0.3 | ✅ | ✅ | ✅ |
| async-stream | 0.3 | ✅ | ✅ | ✅ |
| wasm-timer | 0.2 | ❌ | ✅ | ❌ |

---

## 10. Testing Strategy

### 10.1 Test Coverage

**32 Test Files Available**:
- `/workspace/portalis/tests/python-features/async-await/test_11*.py`
- All currently marked as `not_implemented`
- Comprehensive coverage of async/await features

**Test Categories**:
1. **Basic async functions** (8 tests) - Category 11.1
2. **Asyncio primitives** (12 tests) - Category 11.2
3. **Async iteration** (6 tests) - Category 11.3
4. **Async context managers** (6 tests) - Category 11.4

### 10.2 Unit Tests

**Per-module tests**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_sleep() {
        let start = Instant::now();
        asyncio_sleep(1.0).await;
        assert!(start.elapsed() >= Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_queue_operations() {
        let queue = AsyncQueue::new(10);
        queue.put(42).await.unwrap();
        let value = queue.get().await.unwrap();
        assert_eq!(value, 42);
    }
}
```

### 10.3 Integration Tests

**Platform-specific tests**:
```rust
#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_spawn_task_native() {
    let handle = tokio::spawn(async { 42 });
    let result = handle.await.unwrap();
    assert_eq!(result, 42);
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
#[wasm_bindgen_test]
async fn test_spawn_task_browser() {
    use wasm_bindgen_futures::spawn_local;
    spawn_local(async {
        // Test browser-specific behavior
    });
}
```

### 10.4 WASM Test Setup

**wasm-bindgen-test**:
```toml
[dev-dependencies]
wasm-bindgen-test = "0.3"

[target.'cfg(target_arch = "wasm32")'.dev-dependencies]
wasm-bindgen-test = "0.3"
```

**Run tests**:
```bash
# Native tests
cargo test

# WASM tests (browser)
wasm-pack test --headless --firefox

# WASM tests (node)
wasm-pack test --node
```

---

## Summary of Key Requirements

### Must-Have Features (Priority 1)

1. ✅ **async def → async fn** translation
2. ✅ **await → .await** translation
3. ✅ **asyncio.run()** → tokio runtime creation
4. ✅ **asyncio.create_task()** → tokio::spawn
5. ✅ **asyncio.sleep()** → tokio::time::sleep
6. ✅ **asyncio.gather()** → tokio::join! or join_all
7. ✅ **asyncio.Queue** → tokio::sync::mpsc
8. ✅ **asyncio.Lock** → tokio::sync::Mutex
9. ✅ **asyncio.timeout()** → tokio::time::timeout
10. ✅ Platform-specific runtime handling (native/browser/WASI)

### Should-Have Features (Priority 2)

11. ✅ **asyncio.Semaphore** → tokio::sync::Semaphore
12. ✅ **asyncio.Event** → tokio::sync::Notify
13. ✅ **asyncio.wait()** → tokio::select!
14. ✅ **asyncio.wait_for()** → tokio::time::timeout
15. ✅ **async for** → Stream iteration
16. ✅ **async with** → async context managers

### Nice-to-Have Features (Priority 3)

17. ⚠️ **asyncio.shield()** - Complex cancellation semantics
18. ⚠️ **Async generators** - Requires async-stream
19. ⚠️ **Async comprehensions** - Stream collect
20. ⚠️ **__aenter__/__aexit__** - Full async context protocol

---

## Next Steps

1. **Create asyncio_runtime module** following the architecture in Section 8
2. **Implement platform dispatch layer** (native/browser/WASI)
3. **Add basic async translation** (async def, await) to python_to_rust.rs
4. **Implement Priority 1 features** (runtime, spawn, sleep, gather)
5. **Add stdlib mappings** for asyncio to stdlib_mappings_comprehensive.rs
6. **Write comprehensive tests** for each platform
7. **Document usage patterns** for transpiled code
8. **Update cargo_generator** to include async dependencies

---

## References

- **Existing Code Patterns**:
  - `/workspace/portalis/agents/transpiler/src/wasi_fetch.rs`
  - `/workspace/portalis/agents/transpiler/src/wasi_websocket/`
  - `/workspace/portalis/agents/transpiler/src/web_workers/`
- **Python Language Features**: `/workspace/portalis/plans/PYTHON_LANGUAGE_FEATURES.md`
- **Test Files**: `/workspace/portalis/tests/python-features/async-await/`
- **Tokio Documentation**: https://docs.rs/tokio
- **wasm-bindgen-futures**: https://docs.rs/wasm-bindgen-futures
- **Python asyncio**: https://docs.python.org/3/library/asyncio.html
