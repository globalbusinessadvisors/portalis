# Threading and Web Workers Requirements Analysis

**Document Type**: Requirements Analysis
**Feature**: Threading and Web Workers Support for Python Parallel Code
**Target**: Portalis Python-to-Rust Transpiler
**Date**: October 4, 2025
**Status**: Requirements Analysis Phase

---

## Executive Summary

This document provides a comprehensive requirements analysis for implementing threading and Web Workers support in the Portalis transpiler. The goal is to enable Python parallel and concurrent code (threading, multiprocessing, concurrent.futures, asyncio) to run across three target platforms:

1. **Native Rust** - Using std::thread, rayon, and crossbeam
2. **Browser WASM** - Using Web Workers and wasm-bindgen-rayon
3. **WASI WASM** - Using wasi-threads (experimental)

The implementation must maintain the transpiler's existing cross-platform architecture pattern, as demonstrated by the wasi_fetch, wasi_websocket, and wasi_fs modules.

---

## Table of Contents

1. [Python Threading APIs to Support](#1-python-threading-apis-to-support)
2. [Web Workers Requirements](#2-web-workers-requirements)
3. [Native Threading Requirements](#3-native-threading-requirements)
4. [Platform-Specific APIs](#4-platform-specific-apis)
5. [Synchronization Requirements](#5-synchronization-requirements)
6. [Message Passing Architecture](#6-message-passing-architecture)
7. [Architecture Patterns from Existing Code](#7-architecture-patterns-from-existing-code)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Python Threading APIs to Support

### 1.1 threading Module (Priority: HIGH)

The Python `threading` module provides high-level thread-based parallelism. Based on official Python 3.13 documentation.

#### 1.1.1 Thread Class
**Python API**:
```python
import threading

# Basic thread creation
thread = threading.Thread(target=function, args=(arg1, arg2))
thread.start()
thread.join()

# Thread with name
thread = threading.Thread(target=func, name="WorkerThread")

# Daemon threads
thread = threading.Thread(target=func, daemon=True)
```

**Required Methods**:
- `Thread.__init__(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)`
- `Thread.start()` - Start thread execution
- `Thread.join(timeout=None)` - Wait for thread to complete
- `Thread.is_alive()` - Check if thread is running
- `Thread.name` - Get/set thread name
- `Thread.ident` - Get thread identifier
- `Thread.daemon` - Get/set daemon status
- `Thread.run()` - Method representing thread activity (can be overridden)

**Rust Translation Strategy**:
- Native: `std::thread::spawn`
- Browser: Web Worker creation via `wasm-bindgen`
- WASI: wasi-threads or fallback to async

#### 1.1.2 Lock (Mutex)
**Python API**:
```python
lock = threading.Lock()
lock.acquire()
try:
    # critical section
finally:
    lock.release()

# Or with context manager
with lock:
    # critical section
```

**Required Methods**:
- `Lock.acquire(blocking=True, timeout=-1)`
- `Lock.release()`
- `Lock.locked()` - Check if locked
- Context manager support (`__enter__`, `__exit__`)

**Rust Translation**:
- Native: `std::sync::Mutex` or `parking_lot::Mutex`
- Browser: Atomics with SharedArrayBuffer or message passing
- WASI: `std::sync::Mutex` (if threads supported)

#### 1.1.3 RLock (Reentrant Lock)
**Python API**:
```python
rlock = threading.RLock()
rlock.acquire()
rlock.acquire()  # Same thread can acquire multiple times
rlock.release()
rlock.release()
```

**Rust Translation**:
- Native: `parking_lot::RwLock` or custom implementation
- Browser: Track owner thread ID with atomics
- WASI: Custom implementation

#### 1.1.4 Semaphore
**Python API**:
```python
sem = threading.Semaphore(value=3)
sem.acquire()
# critical section
sem.release()
```

**Required Methods**:
- `Semaphore.__init__(value=1)`
- `Semaphore.acquire(blocking=True, timeout=None)`
- `Semaphore.release()`

**Rust Translation**:
- Native: `std::sync::Semaphore` or `tokio::sync::Semaphore`
- Browser: Atomics-based counter
- WASI: `std::sync` types

#### 1.1.5 Event
**Python API**:
```python
event = threading.Event()
event.set()      # Set event
event.clear()    # Clear event
event.is_set()   # Check if set
event.wait(timeout=None)  # Wait for event
```

**Rust Translation**:
- Native: `std::sync::Condvar` + `Mutex<bool>`
- Browser: Promise-based waiting with SharedArrayBuffer
- WASI: `std::sync::Condvar`

#### 1.1.6 Condition Variable
**Python API**:
```python
cv = threading.Condition()
with cv:
    cv.wait()    # Wait for notification
    cv.notify()  # Wake one waiting thread
    cv.notify_all()  # Wake all waiting threads
```

**Rust Translation**:
- Native: `std::sync::Condvar`
- Browser: Atomics.wait/notify
- WASI: `std::sync::Condvar`

#### 1.1.7 Barrier
**Python API**:
```python
barrier = threading.Barrier(parties=3)
barrier.wait()  # Wait for all parties
```

**Rust Translation**:
- Native: `std::sync::Barrier`
- Browser: Atomics-based counter with wait/notify
- WASI: `std::sync::Barrier`

#### 1.1.8 Thread-Local Data
**Python API**:
```python
thread_local = threading.local()
thread_local.value = 42
```

**Rust Translation**:
- Native: `thread_local!` macro
- Browser: WeakMap keyed by worker ID
- WASI: `thread_local!` macro

---

### 1.2 queue Module (Priority: HIGH)

Thread-safe queues for producer-consumer patterns.

#### 1.2.1 Queue (FIFO)
**Python API**:
```python
from queue import Queue

q = Queue(maxsize=0)
q.put(item, block=True, timeout=None)
item = q.get(block=True, timeout=None)
q.task_done()
q.join()  # Wait for all tasks
q.qsize()
q.empty()
q.full()
```

**Rust Translation**:
- Native: `crossbeam_channel::unbounded()` or `tokio::sync::mpsc`
- Browser: SharedArrayBuffer circular buffer or MessageChannel
- WASI: `crossbeam_channel`

#### 1.2.2 LifoQueue (LIFO/Stack)
**Python API**:
```python
from queue import LifoQueue
q = LifoQueue()
```

**Rust Translation**:
- Native: `crossbeam_channel` with deque
- Browser: Custom implementation
- WASI: `crossbeam_channel`

#### 1.2.3 PriorityQueue
**Python API**:
```python
from queue import PriorityQueue
q = PriorityQueue()
q.put((priority, item))
```

**Rust Translation**:
- Native: `std::collections::BinaryHeap` with `Mutex`
- Browser: Custom heap implementation
- WASI: `std::collections::BinaryHeap`

---

### 1.3 multiprocessing Module (Priority: MEDIUM)

Process-based parallelism. Note: True multi-process may not be available in WASM; map to threads where possible.

#### 1.3.1 Process Class
**Python API**:
```python
from multiprocessing import Process

p = Process(target=func, args=(arg1,))
p.start()
p.join()
p.terminate()
```

**Translation Strategy**:
- Native: `std::process::Command` for true processes, or threads
- Browser: **Web Workers** (closest equivalent)
- WASI: Threads (no true multi-process)

**Note**: Worker processes in Python can share state via multiprocessing.Queue, Pipe, Manager - these need special handling.

#### 1.3.2 Pool
**Python API**:
```python
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(func, iterable)
    result = pool.apply_async(func, args)
```

**Required Methods**:
- `Pool.__init__(processes=None, initializer=None, initargs=())`
- `Pool.map(func, iterable, chunksize=None)`
- `Pool.apply(func, args=(), kwds={})`
- `Pool.apply_async(func, args=(), kwds={}, callback=None)`
- `Pool.close()`
- `Pool.join()`

**Rust Translation**:
- Native: `rayon::ThreadPool`
- Browser: `wasm-bindgen-rayon` with Web Worker pool
- WASI: `rayon::ThreadPool`

#### 1.3.3 Manager (Shared State)
**Python API**:
```python
from multiprocessing import Manager

manager = Manager()
shared_dict = manager.dict()
shared_list = manager.list()
```

**Translation Strategy**:
- Native: `Arc<Mutex<HashMap>>`, `Arc<Mutex<Vec>>`
- Browser: SharedArrayBuffer or shared object store
- WASI: `Arc<Mutex<T>>`

---

### 1.4 concurrent.futures Module (Priority: HIGH)

High-level interface for async execution.

#### 1.4.1 ThreadPoolExecutor
**Python API**:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    future = executor.submit(func, arg1, arg2)
    result = future.result()

    # Or map interface
    results = executor.map(func, iterable)
```

**Required Methods**:
- `ThreadPoolExecutor.__init__(max_workers=None, thread_name_prefix='', initializer=None, initargs=())`
- `Executor.submit(fn, *args, **kwargs)` → Future
- `Executor.map(fn, *iterables, timeout=None, chunksize=1)` → Iterator
- `Executor.shutdown(wait=True, cancel_futures=False)`

**Future Methods**:
- `Future.result(timeout=None)` - Block until result available
- `Future.done()` - Check if future completed
- `Future.cancelled()` - Check if cancelled
- `Future.cancel()` - Attempt to cancel
- `Future.add_done_callback(fn)`

**Rust Translation**:
- Native: `rayon::ThreadPool` + custom Future wrapper
- Browser: Promise-based with Web Workers
- WASI: `rayon::ThreadPool`

#### 1.4.2 ProcessPoolExecutor
**Python API**:
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    future = executor.submit(func, arg)
```

**Translation**:
- Map to ThreadPoolExecutor in most cases
- Native: Could use true processes
- Browser: Web Workers
- WASI: Thread pool

---

### 1.5 asyncio Threading Integration (Priority: HIGH)

Python's async/await can interact with threads via `run_in_executor`.

#### 1.5.1 run_in_executor
**Python API**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def main():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, blocking_function, arg)

    # Or with custom executor
    executor = ThreadPoolExecutor(max_workers=3)
    result = await loop.run_in_executor(executor, func, arg)
```

**Rust Translation**:
- Native: `tokio::task::spawn_blocking`
- Browser: `wasm_bindgen_futures::spawn_local` with Worker
- WASI: `tokio::task::spawn_blocking`

#### 1.5.2 to_thread (Python 3.9+)
**Python API**:
```python
result = await asyncio.to_thread(blocking_func, arg)
```

**Rust Translation**:
- Same as `run_in_executor` with default executor

---

## 2. Web Workers Requirements

### 2.1 Web Worker Lifecycle

#### 2.1.1 Worker Creation
**JavaScript API**:
```javascript
const worker = new Worker('worker.js');
const worker = new Worker('worker.js', { type: 'module' }); // ES6 modules
```

**Requirements**:
- Support for both classic scripts and module workers
- URL resolution for worker scripts
- Inline worker creation (blob URLs)

**Rust/WASM Implementation**:
```rust
use web_sys::{Worker, WorkerOptions};

let mut options = WorkerOptions::new();
options.type_(WorkerType::Module);

let worker = Worker::new_with_options("worker.js", &options)?;
```

#### 2.1.2 Worker Termination
**JavaScript API**:
```javascript
worker.terminate();  // From main thread
self.close();        // From worker thread
```

**Requirements**:
- Graceful shutdown
- Resource cleanup
- Message queue draining

#### 2.1.3 Worker Error Handling
**JavaScript API**:
```javascript
worker.onerror = (event) => {
    console.error('Worker error:', event.message, event.filename, event.lineno);
};
```

---

### 2.2 Message Passing

#### 2.2.1 postMessage/onmessage
**JavaScript API**:
```javascript
// Main thread
worker.postMessage({ type: 'task', data: payload });
worker.onmessage = (event) => {
    console.log('Result:', event.data);
};

// Worker thread
self.onmessage = (event) => {
    const result = processData(event.data);
    self.postMessage(result);
};
```

**Requirements**:
- Structured clone algorithm for data serialization
- Support for transferable objects (ArrayBuffer, MessagePort)
- Error handling for clone failures

**Rust Implementation**:
```rust
use wasm_bindgen::prelude::*;
use web_sys::{Worker, MessageEvent};

// Send message
let data = JsValue::from_serde(&payload)?;
worker.post_message(&data)?;

// Receive message
let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
    let data = event.data();
    // Process data
}) as Box<dyn FnMut(MessageEvent)>);

worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
onmessage.forget();
```

#### 2.2.2 Transferable Objects
**JavaScript API**:
```javascript
const buffer = new ArrayBuffer(1024);
worker.postMessage(buffer, [buffer]);  // Transfer ownership
// buffer is now neutered in main thread
```

**Requirements**:
- Zero-copy transfer for ArrayBuffer, MessagePort, ImageBitmap
- Automatic neutering of transferred objects
- Validation of transferable types

---

### 2.3 Shared Memory (SharedArrayBuffer)

#### 2.3.1 SharedArrayBuffer Creation
**JavaScript API**:
```javascript
const sab = new SharedArrayBuffer(1024);
const view = new Int32Array(sab);

worker.postMessage(sab);  // Shared, not transferred
```

**Requirements**:
- Cross-origin isolation headers (COOP/COEP)
- Security policy enforcement
- Memory alignment for atomics

**Security Headers Required**:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

#### 2.3.2 Atomics Operations
**JavaScript API**:
```javascript
const sab = new SharedArrayBuffer(4);
const view = new Int32Array(sab);

// Atomic operations
Atomics.add(view, 0, 5);
Atomics.store(view, 0, 42);
const value = Atomics.load(view, 0);

// Wait/notify
Atomics.wait(view, 0, 0);  // Wait until value != 0
Atomics.notify(view, 0, 1); // Wake one waiter
```

**Required Operations**:
- `Atomics.add/sub/and/or/xor`
- `Atomics.load/store`
- `Atomics.exchange`
- `Atomics.compareExchange`
- `Atomics.wait/notify` (blocking operations)
- `Atomics.isLockFree(size)`

**Rust Implementation**:
```rust
use std::sync::atomic::{AtomicI32, Ordering};

// In WASM with shared memory
let atomic = AtomicI32::new(0);
atomic.fetch_add(5, Ordering::SeqCst);
atomic.store(42, Ordering::SeqCst);
let value = atomic.load(Ordering::SeqCst);
```

---

### 2.4 Worker Pools

#### 2.4.1 Pool Management
**Requirements**:
- Dynamic worker creation/destruction
- Load balancing across workers
- Task queue management
- Worker health monitoring

**JavaScript Pattern**:
```javascript
class WorkerPool {
    constructor(size, workerScript) {
        this.workers = Array(size).fill(null)
            .map(() => new Worker(workerScript));
        this.taskQueue = [];
        this.availableWorkers = new Set(this.workers);
    }

    async execute(task) {
        if (this.availableWorkers.size === 0) {
            return new Promise(resolve => {
                this.taskQueue.push({ task, resolve });
            });
        }

        const worker = this.availableWorkers.values().next().value;
        this.availableWorkers.delete(worker);

        return new Promise(resolve => {
            worker.onmessage = (e) => {
                resolve(e.data);
                this.availableWorkers.add(worker);
                this.processQueue();
            };
            worker.postMessage(task);
        });
    }
}
```

#### 2.4.2 Work Stealing
**Requirements**:
- Deque-based task storage per worker
- Steal from idle workers
- LIFO for own tasks, FIFO for stolen tasks

---

## 3. Native Threading Requirements

### 3.1 std::thread

#### 3.1.1 Thread Spawning
**Rust API**:
```rust
use std::thread;

let handle = thread::spawn(|| {
    // Thread code
    42
});

let result = handle.join().unwrap();
```

**Features Needed**:
- Thread creation with closure
- Thread naming: `thread::Builder::new().name(String)`
- Stack size configuration: `thread::Builder::new().stack_size(usize)`
- Join handles for result retrieval
- Panic propagation

#### 3.1.2 Thread-Local Storage
**Rust API**:
```rust
use std::thread;

thread_local! {
    static COUNTER: RefCell<u32> = RefCell::new(0);
}

COUNTER.with(|c| {
    *c.borrow_mut() += 1;
});
```

---

### 3.2 rayon - Parallel Iterators

#### 3.2.1 Parallel Iterator Basics
**Rust API**:
```rust
use rayon::prelude::*;

let result: Vec<_> = (0..100)
    .into_par_iter()
    .map(|x| x * 2)
    .collect();

let sum: i32 = vec.par_iter()
    .map(|x| x * x)
    .sum();
```

**Features Needed**:
- `par_iter()`, `into_par_iter()`, `par_iter_mut()`
- Parallel map, filter, fold, reduce
- Parallel sorting
- Custom thread pool configuration

#### 3.2.2 ThreadPool Configuration
**Rust API**:
```rust
use rayon::{ThreadPoolBuilder, ThreadPool};

let pool = ThreadPoolBuilder::new()
    .num_threads(8)
    .build()
    .unwrap();

pool.install(|| {
    // Work runs in this pool
    vec.par_iter().for_each(|x| process(x));
});
```

#### 3.2.3 Work Splitting
**Rust API**:
```rust
use rayon::prelude::*;

// Automatically splits work
data.par_chunks(1000)
    .for_each(|chunk| process_chunk(chunk));
```

---

### 3.3 crossbeam - Channels and Synchronization

#### 3.3.1 Channels
**Rust API**:
```rust
use crossbeam_channel::{unbounded, bounded};

// Unbounded channel
let (tx, rx) = unbounded();
tx.send(42).unwrap();
let val = rx.recv().unwrap();

// Bounded channel
let (tx, rx) = bounded(100);

// Select over multiple channels
select! {
    recv(rx1) -> msg => process(msg),
    recv(rx2) -> msg => process(msg),
}
```

**Features Needed**:
- Bounded and unbounded channels
- Multi-producer, multi-consumer (MPMC)
- Select operation
- Timeout support

#### 3.3.2 Scoped Threads
**Rust API**:
```rust
use crossbeam::thread;

let mut vec = vec![1, 2, 3];

thread::scope(|s| {
    s.spawn(|_| {
        vec.push(4);  // Can borrow from outer scope
    });
}).unwrap();
```

#### 3.3.3 Atomic Cell
**Rust API**:
```rust
use crossbeam::atomic::AtomicCell;

let cell = AtomicCell::new(42);
cell.store(100);
let value = cell.load();
cell.compare_exchange(100, 200);
```

---

### 3.4 Thread-Safe Data Structures

#### 3.4.1 Mutex and RwLock
**Rust API**:
```rust
use std::sync::{Mutex, RwLock, Arc};

// Mutex
let data = Arc::new(Mutex::new(Vec::new()));
{
    let mut vec = data.lock().unwrap();
    vec.push(1);
}

// RwLock
let lock = Arc::new(RwLock::new(5));
{
    let r1 = lock.read().unwrap();
    let r2 = lock.read().unwrap();  // Multiple readers OK
}
{
    let mut w = lock.write().unwrap();
    *w += 1;
}
```

#### 3.4.2 Atomic Types
**Rust API**:
```rust
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

let flag = AtomicBool::new(false);
flag.store(true, Ordering::SeqCst);
let value = flag.load(Ordering::SeqCst);

let counter = AtomicUsize::new(0);
counter.fetch_add(1, Ordering::SeqCst);
```

#### 3.4.3 Arc (Atomic Reference Counting)
**Rust API**:
```rust
use std::sync::Arc;

let data = Arc::new(vec![1, 2, 3]);
let data_clone = Arc::clone(&data);

thread::spawn(move || {
    println!("{:?}", data_clone);
});
```

---

## 4. Platform-Specific APIs

### 4.1 Native Rust (std::thread, rayon, crossbeam)

**Platform**: Linux, macOS, Windows
**Threading Model**: OS threads (pthread, Windows threads)

**Capabilities**:
- ✅ True parallel execution
- ✅ Shared memory with Mutex/RwLock
- ✅ Thread-local storage
- ✅ Condition variables
- ✅ Barriers
- ✅ Thread pools (rayon)
- ✅ Work stealing
- ✅ MPSC/MPMC channels

**Limitations**:
- OS thread overhead
- Context switching costs

**Dependencies**:
```toml
[dependencies]
rayon = "1.8"
crossbeam = "0.8"
parking_lot = "0.12"  # Faster Mutex/RwLock
```

---

### 4.2 Browser WASM (Web Workers)

**Platform**: Chrome, Firefox, Safari, Edge
**Threading Model**: Web Workers + SharedArrayBuffer

**Capabilities**:
- ✅ Parallel execution via Web Workers
- ✅ Message passing (structured clone)
- ✅ Transferable objects (zero-copy)
- ✅ SharedArrayBuffer (with headers)
- ✅ Atomics operations
- ⚠️ No shared linear memory by default (requires threads proposal)
- ⚠️ No blocking operations in main thread

**Limitations**:
- ❌ No `std::thread` (WASM is single-threaded by default)
- ❌ No OS-level threads
- ❌ Requires cross-origin isolation
- ❌ No synchronous I/O in workers
- ⚠️ Limited to ~4-8 workers in practice

**Required Features**:
```javascript
// Check for SharedArrayBuffer support
if (typeof SharedArrayBuffer === 'undefined') {
    console.warn('SharedArrayBuffer not available');
}

// Check cross-origin isolation
if (!crossOriginIsolated) {
    console.warn('Not cross-origin isolated - SharedArrayBuffer disabled');
}
```

**Dependencies**:
```toml
[dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-rayon = "1.0"  # For rayon on WASM
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = { version = "0.3", features = [
    "Worker",
    "WorkerOptions",
    "WorkerType",
    "MessageEvent",
    "SharedArrayBuffer",
] }
```

**Build Configuration**:
```bash
# Nightly Rust required for threads
rustup override set nightly

# Build with atomics and bulk memory
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
    cargo build --target wasm32-unknown-unknown -Z build-std=panic_abort,std
```

---

### 4.3 WASI (wasi-threads)

**Platform**: Wasmtime, WasmEdge, Wasmer
**Threading Model**: wasi-threads (experimental)

**Current Status (2025)**:
- ⚠️ wasi-threads is **legacy** (WASI v0.1)
- 🔄 Future: shared-everything-threads proposal (WASI v0.2+)
- ⚠️ Limited runtime support
- ✅ Wasmtime has experimental support
- ⚠️ Not suitable for production

**Capabilities**:
- ⚠️ Thread spawning (wasi_thread_spawn)
- ⚠️ Shared linear memory
- ⚠️ Atomics operations
- ✅ Can use std::thread (with wasi-threads target)

**Limitations**:
- ❌ Not standardized (legacy proposal)
- ❌ Limited runtime support
- ❌ Breaking changes expected
- ⚠️ Performance varies by runtime

**Target**:
```bash
# Build for wasi-threads
cargo build --target wasm32-wasi-threads
```

**Alternative Approach**:
For WASI without thread support, fall back to:
- Async/await (tokio with wasm-compat)
- Single-threaded event loop
- Message passing via pipes/sockets

---

## 5. Synchronization Requirements

### 5.1 Locks

#### 5.1.1 Mutex (Mutual Exclusion)

**Semantics**:
- One owner at a time
- Blocking acquire
- Panic on poisoning (Rust)

**Platform Implementations**:

| Platform | Implementation | Notes |
|----------|----------------|-------|
| Native | `std::sync::Mutex<T>` | OS-level mutex |
| Native (alt) | `parking_lot::Mutex<T>` | Faster, no poisoning |
| Browser | `Atomics.wait/notify` | On SharedArrayBuffer |
| Browser (alt) | Promise-based lock | No blocking in main thread |
| WASI | `std::sync::Mutex<T>` | If threads supported |

**Example Implementation (Browser)**:
```rust
// Browser: Atomics-based mutex
pub struct WasmMutex {
    locked: AtomicU32,  // 0 = unlocked, 1 = locked
    shared_buffer: SharedArrayBuffer,
}

impl WasmMutex {
    pub fn lock(&self) -> MutexGuard {
        loop {
            let prev = self.locked.compare_exchange(
                0, 1,
                Ordering::Acquire,
                Ordering::Relaxed
            );

            if prev == Ok(0) {
                return MutexGuard { mutex: self };
            }

            // Wait using Atomics.wait
            Atomics::wait(&self.shared_buffer, 0, 1);
        }
    }
}

impl Drop for MutexGuard<'_> {
    fn drop(&mut self) {
        self.mutex.locked.store(0, Ordering::Release);
        Atomics::notify(&self.mutex.shared_buffer, 0, 1);
    }
}
```

#### 5.1.2 RwLock (Reader-Writer Lock)

**Semantics**:
- Multiple readers OR one writer
- Readers don't block readers
- Writer blocks all

**Platform Implementations**:

| Platform | Implementation |
|----------|----------------|
| Native | `std::sync::RwLock<T>` |
| Native (alt) | `parking_lot::RwLock<T>` |
| Browser | Two-counter atomics |
| WASI | `std::sync::RwLock<T>` |

---

### 5.2 Atomic Operations

#### 5.2.1 Atomic Types

**Available Types**:
- `AtomicBool`
- `AtomicI8`, `AtomicI16`, `AtomicI32`, `AtomicI64`, `AtomicIsize`
- `AtomicU8`, `AtomicU16`, `AtomicU32`, `AtomicU64`, `AtomicUsize`
- `AtomicPtr<T>`

**Operations**:
- `load(Ordering)` - Read value
- `store(val, Ordering)` - Write value
- `swap(val, Ordering)` - Exchange value
- `compare_exchange(current, new, success_order, failure_order)` - CAS
- `fetch_add/sub/and/or/xor(val, Ordering)` - RMW operations

**Memory Ordering**:
```rust
use std::sync::atomic::Ordering;

Ordering::Relaxed   // No ordering guarantees
Ordering::Acquire   // Loads can't be reordered before
Ordering::Release   // Stores can't be reordered after
Ordering::AcqRel    // Both acquire and release
Ordering::SeqCst    // Sequentially consistent (strictest)
```

**Platform Support**:

| Platform | Support | Notes |
|----------|---------|-------|
| Native | ✅ Full | All atomic types |
| Browser | ✅ With SAB | Requires SharedArrayBuffer |
| Browser | ⚠️ Limited | Without SAB, only main thread |
| WASI | ✅ Full | With threads feature |

---

### 5.3 Condition Variables

**Purpose**: Wait for condition to be met, signaled by another thread.

**Rust API**:
```rust
use std::sync::{Mutex, Condvar};

let pair = Arc::new((Mutex::new(false), Condvar::new()));
let (lock, cvar) = &*pair;

// Wait for condition
let mut started = lock.lock().unwrap();
while !*started {
    started = cvar.wait(started).unwrap();
}

// Signal condition
let mut started = lock.lock().unwrap();
*started = true;
cvar.notify_one();  // or notify_all()
```

**Platform Implementations**:

| Platform | Implementation |
|----------|----------------|
| Native | `std::sync::Condvar` |
| Browser | `Atomics.wait/notify` |
| WASI | `std::sync::Condvar` |

---

### 5.4 Barriers

**Purpose**: Synchronization point for multiple threads.

**Rust API**:
```rust
use std::sync::{Arc, Barrier};

let barrier = Arc::new(Barrier::new(10));

for _ in 0..10 {
    let c = Arc::clone(&barrier);
    thread::spawn(move || {
        // Do work
        c.wait();  // Wait for all threads
        // Continue after all arrived
    });
}
```

**Platform Implementations**:

| Platform | Implementation |
|----------|----------------|
| Native | `std::sync::Barrier` |
| Browser | Atomic counter + wait/notify |
| WASI | `std::sync::Barrier` |

---

### 5.5 Channels

#### 5.5.1 MPSC (Multi-Producer, Single-Consumer)

**Rust API**:
```rust
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();
let tx2 = tx.clone();

thread::spawn(move || tx.send(42).unwrap());
thread::spawn(move || tx2.send(100).unwrap());

let val1 = rx.recv().unwrap();
let val2 = rx.recv().unwrap();
```

**Platform Implementations**:

| Platform | Implementation |
|----------|----------------|
| Native | `std::sync::mpsc` |
| Native (alt) | `crossbeam_channel` (MPMC) |
| Browser | `MessageChannel` |
| WASI | `std::sync::mpsc` or `crossbeam` |

#### 5.5.2 MPMC (Multi-Producer, Multi-Consumer)

**Rust API**:
```rust
use crossbeam_channel::{unbounded, bounded};

let (tx, rx) = unbounded();

for i in 0..10 {
    let tx = tx.clone();
    thread::spawn(move || tx.send(i).unwrap());
}

for _ in 0..10 {
    let rx = rx.clone();
    thread::spawn(move || {
        let val = rx.recv().unwrap();
        println!("{}", val);
    });
}
```

---

## 6. Message Passing Architecture

### 6.1 Message Passing Patterns

#### 6.1.1 Request/Response Pattern

**Use Case**: RPC-style communication

**Python Example**:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    future = executor.submit(expensive_function, arg)
    result = future.result()  # Block until response
```

**Implementation**:

```rust
// Native
pub struct RequestResponse<Req, Resp> {
    tx: mpsc::Sender<(Req, oneshot::Sender<Resp>)>,
    rx: mpsc::Receiver<(Req, oneshot::Sender<Resp>)>,
}

impl RequestResponse {
    pub async fn request(&self, req: Req) -> Resp {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx.send((req, resp_tx)).unwrap();
        resp_rx.await.unwrap()
    }
}

// Browser (Web Worker)
pub struct WorkerRpc {
    worker: Worker,
    pending: Arc<Mutex<HashMap<u64, Promise>>>,
    next_id: Arc<AtomicU64>,
}

impl WorkerRpc {
    pub fn call(&self, method: &str, args: JsValue) -> Promise {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let (promise, resolve, reject) = Promise::new();
        self.pending.lock().unwrap().insert(id, (resolve, reject));

        let msg = json!({ "id": id, "method": method, "args": args });
        self.worker.post_message(&msg).unwrap();

        promise
    }
}
```

#### 6.1.2 Fire-and-Forget Pattern

**Use Case**: Background tasks, logging

**Python Example**:
```python
thread = threading.Thread(target=log_event, args=(event,))
thread.daemon = True
thread.start()
# Don't wait for result
```

**Implementation**:

```rust
// Native
thread::spawn(move || {
    process_event(event);
    // No return value expected
});

// Browser
worker.postMessage({ type: 'background_task', data: event });
// No response handler
```

#### 6.1.3 Streaming Pattern

**Use Case**: Progressive results, real-time data

**Python Example**:
```python
def stream_results():
    for item in large_dataset:
        yield process(item)

for result in stream_results():
    handle(result)
```

**Implementation**:

```rust
// Native - Channel-based stream
pub fn stream_results() -> mpsc::Receiver<Result> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        for item in large_dataset {
            tx.send(process(item)).unwrap();
        }
    });

    rx
}

// Browser - Streaming responses
worker.onmessage = (event) => {
    if (event.data.type === 'stream_item') {
        handle(event.data.item);
    } else if (event.data.type === 'stream_end') {
        complete();
    }
};
```

#### 6.1.4 Broadcast Pattern

**Use Case**: Event notification to multiple subscribers

**Python Example**:
```python
subscribers = [worker1, worker2, worker3]
for worker in subscribers:
    worker.notify(event)
```

**Implementation**:

```rust
// Native - Multiple receivers
pub struct Broadcaster<T> {
    subscribers: Vec<mpsc::Sender<T>>,
}

impl<T: Clone> Broadcaster<T> {
    pub fn broadcast(&self, msg: T) {
        for tx in &self.subscribers {
            tx.send(msg.clone()).ok();
        }
    }
}

// Browser - BroadcastChannel API
const bc = new BroadcastChannel('events');
bc.postMessage({ type: 'event', data: event });

// In workers:
const bc = new BroadcastChannel('events');
bc.onmessage = (event) => handle(event.data);
```

#### 6.1.5 Shared State Coordination

**Use Case**: Multiple workers accessing shared data

**Python Example**:
```python
from multiprocessing import Manager

manager = Manager()
shared_dict = manager.dict()
shared_dict['counter'] = 0

def worker():
    with lock:
        shared_dict['counter'] += 1
```

**Implementation**:

```rust
// Native - Arc<Mutex>
let shared = Arc::new(Mutex::new(HashMap::new()));

for _ in 0..10 {
    let shared = Arc::clone(&shared);
    thread::spawn(move || {
        let mut map = shared.lock().unwrap();
        map.insert("key", "value");
    });
}

// Browser - SharedArrayBuffer
const sab = new SharedArrayBuffer(1024);
const view = new Int32Array(sab);

// Worker 1
Atomics.add(view, 0, 1);

// Worker 2
Atomics.add(view, 0, 1);

// Main thread
const counter = Atomics.load(view, 0);
```

---

### 6.2 Serialization

#### 6.2.1 Structured Clone (Browser)

**Supported Types**:
- Primitive types (number, string, boolean, null, undefined)
- Objects and Arrays
- Date, RegExp, Map, Set
- Blob, File, FileList
- ArrayBuffer, TypedArrays
- ImageData, ImageBitmap

**Not Supported**:
- Functions
- DOM nodes
- Symbols
- Error objects (clones without stack)

#### 6.2.2 Serde JSON (Rust)

**Rust Implementation**:
```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Message {
    id: u64,
    data: String,
}

// Send
let msg = Message { id: 1, data: "hello".into() };
let json = serde_json::to_string(&msg)?;
worker.post_message(&JsValue::from_str(&json))?;

// Receive
let json: String = event.data().as_string().unwrap();
let msg: Message = serde_json::from_str(&json)?;
```

#### 6.2.3 Binary Serialization

**Options**:
- **bincode**: Fast binary serialization
- **postcard**: No-std friendly, compact
- **rkyv**: Zero-copy deserialization

**Example (bincode)**:
```rust
use bincode;

let msg = Message { /* ... */ };
let bytes = bincode::serialize(&msg)?;

// Send as Uint8Array
let array = js_sys::Uint8Array::from(&bytes[..]);
worker.post_message(&array)?;

// Receive
let array: Vec<u8> = js_sys::Uint8Array::new(&event.data()).to_vec();
let msg: Message = bincode::deserialize(&array)?;
```

---

## 7. Architecture Patterns from Existing Code

### 7.1 Cross-Platform Abstraction Pattern

**Learning from wasi_fetch.rs and wasi_websocket/mod.rs:**

The existing codebase demonstrates a consistent pattern:

1. **Common Types** - Platform-agnostic types in main module
2. **Platform Modules** - Separate implementations per platform
3. **Conditional Compilation** - `#[cfg(...)]` to select implementation
4. **Unified Interface** - Public API that works everywhere

**Pattern Structure**:
```rust
// src/wasi_threading/mod.rs

// Common types (works on all platforms)
pub struct Thread { /* ... */ }
pub struct Mutex<T> { /* ... */ }
pub enum ThreadMessage { /* ... */ }

// Platform-specific implementations
#[cfg(not(target_arch = "wasm32"))]
mod native;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod browser;

#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
mod wasi_impl;

// Re-export platform-specific implementation
#[cfg(not(target_arch = "wasm32"))]
use native::NativeThread as PlatformThread;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use browser::BrowserThread as PlatformThread;

#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
use wasi_impl::WasiThread as PlatformThread;

// Unified public API
impl Thread {
    pub fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static
    {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::spawn_native(f)
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            Self::spawn_browser(f)
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            Self::spawn_wasi(f)
        }
    }

    pub fn join(self) -> Result<()> {
        // Platform-specific implementation
    }
}
```

### 7.2 Async/Await Pattern

**Learning from wasi_fetch.rs:**

All platform implementations use async/await:

```rust
impl WasiFetch {
    pub async fn fetch(request: Request) -> Result<Response> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::fetch_native(request).await
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            Self::fetch_browser(request).await
        }
    }
}
```

**Apply to Threading**:
```rust
impl ThreadPool {
    pub async fn execute<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Return a future that resolves when task completes
    }
}
```

### 7.3 Shared State Pattern

**Learning from wasi_websocket/mod.rs:**

Shared state using `Arc<Mutex<T>>`:

```rust
pub(crate) struct WebSocketSharedState {
    state: Arc<Mutex<WebSocketState>>,
    message_buffer: Arc<Mutex<VecDeque<WebSocketMessage>>>,
    handlers: Arc<WebSocketHandlers>,
    config: Arc<WebSocketConfig>,
}

impl WebSocketSharedState {
    pub(crate) fn get_state(&self) -> WebSocketState {
        *self.state.lock().unwrap()
    }

    pub(crate) fn set_state(&self, new_state: WebSocketState) {
        *self.state.lock().unwrap() = new_state;
    }
}
```

**Apply to Threading**:
```rust
pub(crate) struct ThreadPoolSharedState {
    workers: Arc<Mutex<Vec<Worker>>>,
    task_queue: Arc<Mutex<VecDeque<Task>>>,
    config: Arc<ThreadPoolConfig>,
}
```

### 7.4 Event Handler Pattern

**Learning from wasi_websocket/mod.rs:**

Callback-based event handlers:

```rust
pub type OnMessageCallback = Arc<dyn Fn(WebSocketMessage) + Send + Sync>;

pub struct WebSocketHandlers {
    on_message: Option<OnMessageCallback>,
}

impl WebSocketHandlers {
    pub fn on_message<F>(mut self, callback: F) -> Self
    where
        F: Fn(WebSocketMessage) + Send + Sync + 'static,
    {
        self.on_message = Some(Arc::new(callback));
        self
    }

    pub(crate) fn trigger_message(&self, msg: WebSocketMessage) {
        if let Some(ref callback) = self.on_message {
            callback(msg);
        }
    }
}
```

**Apply to Threading**:
```rust
pub type OnTaskComplete = Arc<dyn Fn(TaskResult) + Send + Sync>;
pub type OnWorkerError = Arc<dyn Fn(WorkerError) + Send + Sync>;

pub struct ThreadPoolHandlers {
    on_task_complete: Option<OnTaskComplete>,
    on_worker_error: Option<OnWorkerError>,
}
```

---

## 8. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)

**Deliverables**:
1. Module structure
   - `wasi_threading/mod.rs` - Common types and public API
   - `wasi_threading/native.rs` - Native Rust implementation
   - `wasi_threading/browser.rs` - Web Worker implementation
   - `wasi_threading/wasi_impl.rs` - WASI threads implementation

2. Basic Thread API
   - `WasiThread::spawn()`
   - `WasiThread::join()`
   - Thread naming
   - Panic handling

3. Basic Mutex
   - `WasiMutex<T>` with lock/unlock
   - Platform-specific implementations
   - Poisoning support (native only)

**Tests**:
- Spawn thread and join
- Multiple threads
- Mutex shared between threads

---

### Phase 2: Synchronization Primitives (Week 3-4)

**Deliverables**:
1. Advanced Locks
   - `WasiRwLock<T>` - Reader-writer lock
   - `WasiSemaphore` - Counting semaphore

2. Condition Variables
   - `WasiCondvar` with wait/notify
   - Timeout support

3. Barriers
   - `WasiBarrier` for synchronization points

4. Atomic Operations
   - Wrapper types for all platforms
   - Memory ordering abstractions

**Tests**:
- RwLock with multiple readers
- Producer-consumer with Condvar
- Barrier synchronization
- Atomic counter increments

---

### Phase 3: Channels (Week 5)

**Deliverables**:
1. MPSC Channel
   - `wasi_channel::channel()` - Unbounded
   - `wasi_channel::bounded()` - Bounded
   - Send/Recv operations

2. Platform Implementations
   - Native: `crossbeam_channel`
   - Browser: MessageChannel wrapper
   - WASI: crossbeam or std::sync::mpsc

**Tests**:
- Simple send/receive
- Multiple producers
- Channel close/drop
- Timeout operations

---

### Phase 4: Thread Pools (Week 6-7)

**Deliverables**:
1. ThreadPoolExecutor
   - Creation with worker count
   - Submit tasks
   - Future-based result retrieval

2. Parallel Iterators
   - `par_iter()` support
   - Map/filter/reduce operations
   - Chunk processing

3. Platform Implementations
   - Native: rayon wrapper
   - Browser: wasm-bindgen-rayon
   - WASI: rayon or fallback

**Tests**:
- Pool with multiple tasks
- Parallel map operation
- Load balancing verification

---

### Phase 5: Python API Translation (Week 8-9)

**Deliverables**:
1. threading Module Translation
   - Map `threading.Thread` to `WasiThread`
   - Map `threading.Lock` to `WasiMutex`
   - Map `threading.Event` to event primitive

2. queue Module Translation
   - Map `queue.Queue` to channel
   - FIFO/LIFO/Priority variants

3. concurrent.futures Translation
   - Map `ThreadPoolExecutor` to pool
   - Future/Promise API

4. Code Generator Updates
   - Recognize threading imports
   - Generate appropriate Rust code
   - Add platform features to Cargo.toml

**Tests**:
- Translate Python threading examples
- Verify generated Rust compiles
- Cross-platform test execution

---

### Phase 6: Web Worker Integration (Week 10-11)

**Deliverables**:
1. Worker Pool Manager
   - Dynamic worker creation
   - Load balancing
   - Task queue

2. Message Protocol
   - Serialization layer
   - RPC pattern
   - Transferable objects

3. SharedArrayBuffer Support
   - Memory allocation
   - Atomic operations
   - Synchronization primitives

4. Build Configuration
   - Webpack/Rollup/Vite setup
   - Worker script generation
   - Cross-origin headers

**Tests**:
- Worker creation and termination
- Message passing
- SharedArrayBuffer synchronization
- Browser compatibility

---

### Phase 7: Documentation and Examples (Week 12)

**Deliverables**:
1. API Documentation
   - Rustdoc for all public APIs
   - Platform compatibility matrix
   - Migration guide from Python

2. Examples
   - Thread spawning
   - Producer-consumer
   - Parallel computation
   - Web Worker demo

3. Integration Tests
   - End-to-end scenarios
   - Cross-platform verification
   - Performance benchmarks

---

## Summary of Key Requirements

### Python APIs (Priority Order)

1. **HIGH PRIORITY**:
   - `threading.Thread` - Basic thread spawning
   - `threading.Lock` - Mutual exclusion
   - `queue.Queue` - Thread-safe queue
   - `concurrent.futures.ThreadPoolExecutor` - Thread pool
   - `asyncio.run_in_executor` - Async/thread integration

2. **MEDIUM PRIORITY**:
   - `threading.RLock` - Reentrant lock
   - `threading.Semaphore` - Counting semaphore
   - `threading.Event` - Thread event
   - `threading.Condition` - Condition variable
   - `multiprocessing.Pool` - Process pool (map to threads)

3. **LOW PRIORITY**:
   - `threading.Barrier` - Synchronization barrier
   - `threading.local` - Thread-local storage
   - `multiprocessing.Manager` - Shared state manager
   - `queue.PriorityQueue` - Priority queue

### Platform Support Matrix

| Feature | Native | Browser | WASI |
|---------|--------|---------|------|
| Thread Spawning | ✅ std::thread | ✅ Web Workers | ⚠️ wasi-threads |
| Mutex | ✅ std::sync::Mutex | ✅ Atomics | ✅ std::sync::Mutex |
| Channels | ✅ crossbeam | ✅ MessageChannel | ✅ crossbeam |
| Thread Pool | ✅ rayon | ✅ wasm-bindgen-rayon | ⚠️ rayon |
| Atomics | ✅ Full | ✅ SharedArrayBuffer | ✅ Full |
| Shared Memory | ✅ Arc<Mutex<T>> | ✅ SharedArrayBuffer | ✅ Arc<Mutex<T>> |

### Dependencies Required

**Native**:
```toml
rayon = "1.8"
crossbeam = "0.8"
parking_lot = "0.12"
tokio = { version = "1", features = ["rt-multi-thread"] }
```

**Browser**:
```toml
wasm-bindgen = "0.2"
wasm-bindgen-rayon = "1.0"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["Worker", "SharedArrayBuffer"] }
```

**WASI**:
```toml
# For wasi-threads (experimental)
# Target: wasm32-wasi-threads
```

### Security Considerations

1. **Browser**:
   - Requires cross-origin isolation (COOP/COEP headers)
   - SharedArrayBuffer restrictions
   - Worker script CSP policies

2. **Native**:
   - Thread safety (no data races)
   - Avoid deadlocks
   - Proper cleanup on panic

3. **WASI**:
   - Capability-based security
   - Limited syscalls
   - Sandboxed execution

---

## Next Steps

1. **Create Module Structure**: Set up `wasi_threading/` directory with platform modules
2. **Implement Basic Thread API**: Start with `WasiThread::spawn()` and `join()`
3. **Add Mutex**: Implement `WasiMutex<T>` for all platforms
4. **Write Tests**: Comprehensive test suite for each primitive
5. **Iterate**: Add more primitives based on test results and Python API coverage

---

## References

### Documentation
- [Python threading module](https://docs.python.org/3/library/threading.html)
- [Python queue module](https://docs.python.org/3/library/queue.html)
- [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [Python concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)
- [Web Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)
- [SharedArrayBuffer](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer)
- [Atomics API](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Atomics)
- [Rust std::thread](https://doc.rust-lang.org/std/thread/)
- [Rayon documentation](https://docs.rs/rayon/latest/rayon/)
- [Crossbeam documentation](https://docs.rs/crossbeam/latest/crossbeam/)
- [wasm-bindgen-rayon](https://github.com/GoogleChromeLabs/wasm-bindgen-rayon)
- [wasi-threads proposal](https://github.com/WebAssembly/wasi-threads)

### Existing Code Patterns
- `/workspace/portalis/agents/transpiler/src/wasi_fetch.rs` - Cross-platform async HTTP
- `/workspace/portalis/agents/transpiler/src/wasi_websocket/mod.rs` - WebSocket abstraction
- `/workspace/portalis/agents/transpiler/src/wasi_fs.rs` - Filesystem abstraction
- `/workspace/portalis/agents/transpiler/Cargo.toml` - Dependency configuration

---

**End of Requirements Analysis**
