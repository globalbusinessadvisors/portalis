# WASI Threading Primitives

Comprehensive threading and synchronization primitives for the Portalis transpiler, providing unified APIs that work across native Rust, browser WASM, and WASI environments.

## Overview

This module provides Python-compatible threading primitives that can be transpiled to Rust and run on multiple platforms:

- **Native Rust**: Full threading support using `std::thread`, `rayon`, and `parking_lot`
- **Browser WASM**: Limited support via Web Workers (with fallback to single-threaded execution)
- **WASI**: Thread support when available, with graceful degradation

## Modules

### `thread` - Thread Primitives
- Thread creation and joining
- Thread configuration (name, stack size, priority)
- Thread IDs and management
- Sleep and yield operations
- Thread-local storage

### `sync` - Synchronization Primitives
- **WasiMutex**: Mutual exclusion lock (wraps `parking_lot::Mutex`)
- **WasiRwLock**: Read-write lock for concurrent reads
- **WasiSemaphore**: Semaphore for resource counting
- **WasiCondvar**: Condition variable for thread coordination
- **WasiBarrier**: Barrier for synchronizing multiple threads
- **WasiEvent**: One-time signaling event

### `collections` - Thread-Safe Collections
- **WasiQueue**: FIFO queue (uses `crossbeam::channel` on native)
- **WasiStack**: LIFO stack
- **WasiPriorityQueue**: Priority queue (max-heap)
- **WasiDeque**: Double-ended queue

### `pool` - Thread Pool
- Fixed thread pool with configurable worker count
- Rayon-based work stealing pool for data parallelism
- Task submission with futures
- Parallel map and foreach operations
- Global thread pool for convenience

### Platform Implementations

#### `native` (Native Rust)
- Full threading support
- CPU affinity controls (Linux)
- Thread priority management
- Hardware concurrency detection

#### `browser` (Browser WASM)
- Web Workers integration (when available)
- Hardware concurrency detection via `navigator.hardwareConcurrency`
- Fallback to single-threaded execution

#### `wasi_impl` (WASI)
- Capability detection for wasi-threads
- Graceful degradation when threads unavailable
- Compatible API surface

## Usage Examples

### Basic Threading

```rust
use portalis_transpiler::wasi_threading::*;

// Spawn a thread
let handle = WasiThread::spawn(|| {
    println!("Hello from thread!");
    42
}).unwrap();

// Wait for result
let result = handle.join().unwrap();
assert_eq!(result, 42);
```

### Thread Configuration

```rust
use portalis_transpiler::wasi_threading::*;

let config = ThreadConfig::new()
    .with_name("worker-1")
    .with_stack_size(2 * 1024 * 1024)
    .with_priority(ThreadPriority::High);

let handle = WasiThread::spawn_with_config(|| {
    // Thread work here
}, config).unwrap();
```

### Mutex Synchronization

```rust
use portalis_transpiler::wasi_threading::*;
use std::sync::Arc;

let counter = Arc::new(WasiMutex::new(0));

let mut handles = vec![];
for _ in 0..10 {
    let counter_clone = counter.clone();
    let handle = WasiThread::spawn(move || {
        *counter_clone.lock() += 1;
    }).unwrap();
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}

assert_eq!(*counter.lock(), 10);
```

### Read-Write Lock

```rust
use portalis_transpiler::wasi_threading::*;

let data = WasiRwLock::new(vec![1, 2, 3]);

// Multiple readers can access simultaneously
{
    let read1 = data.read();
    let read2 = data.read();
    println!("Data: {:?}", *read1);
}

// Writers get exclusive access
{
    let mut write = data.write();
    write.push(4);
}
```

### Thread-Safe Queue

```rust
use portalis_transpiler::wasi_threading::*;

let queue = WasiQueue::new();

// Producer
queue.push(1).unwrap();
queue.push(2).unwrap();

// Consumer
let value = queue.pop().unwrap();
assert_eq!(value, 1);
```

### Thread Pool

```rust
use portalis_transpiler::wasi_threading::*;

// Create a thread pool
let pool = ThreadPool::new(4).unwrap();

// Execute tasks
for i in 0..10 {
    pool.execute(move || {
        println!("Task {}", i);
    }).unwrap();
}

// Submit work with result
let result = pool.submit(|| {
    expensive_computation()
}).unwrap();

let value = result.wait().unwrap();
```

### Rayon-Based Parallel Map

```rust
use portalis_transpiler::wasi_threading::*;

let pool = ThreadPoolBuilder::new()
    .num_threads(4)
    .enable_work_stealing(true)
    .build()
    .unwrap();

let numbers = vec![1, 2, 3, 4, 5];
let squares = pool.parallel_map(numbers, |x| x * x).unwrap();

assert_eq!(squares, vec![1, 4, 9, 16, 25]);
```

### Condition Variables

```rust
use portalis_transpiler::wasi_threading::*;
use std::sync::Arc;

let mutex = Arc::new(WasiMutex::new(false));
let condvar = Arc::new(WasiCondvar::new());

let mutex_clone = mutex.clone();
let condvar_clone = condvar.clone();

let handle = WasiThread::spawn(move || {
    let mut guard = mutex_clone.lock();
    *guard = true;
    condvar_clone.notify_one();
}).unwrap();

let mut guard = mutex.lock();
while !*guard {
    guard = condvar.wait(guard);
}

handle.join().unwrap();
```

### Barriers

```rust
use portalis_transpiler::wasi_threading::*;
use std::sync::Arc;

let barrier = Arc::new(WasiBarrier::new(5));

let handles: Vec<_> = (0..5).map(|i| {
    let barrier_clone = barrier.clone();
    WasiThread::spawn(move || {
        println!("Thread {} before barrier", i);
        barrier_clone.wait();
        println!("Thread {} after barrier", i);
    }).unwrap()
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

## Python to Rust Translation

This module enables transpilation of Python threading code:

### Python
```python
import threading

# Thread creation
def worker():
    return 42

t = threading.Thread(target=worker)
t.start()
result = t.join()

# Mutex
lock = threading.Lock()
with lock:
    # Critical section
    pass

# Queue
from queue import Queue
q = Queue()
q.put(1)
item = q.get()
```

### Transpiled Rust
```rust
use portalis_transpiler::wasi_threading::*;

// Thread creation
fn worker() -> i32 {
    42
}

let t = WasiThread::spawn(|| worker()).unwrap();
let result = t.join().unwrap();

// Mutex
let lock = WasiMutex::new(());
{
    let _guard = lock.lock();
    // Critical section
}

// Queue
let q = WasiQueue::new();
q.push(1).unwrap();
let item = q.pop().unwrap();
```

## Platform-Specific Behavior

### Native (std::thread)
- Full threading support
- Uses `parking_lot` for faster mutex/rwlock
- Uses `rayon` for work stealing
- Uses `crossbeam` for efficient channels
- Supports CPU affinity and priority (Linux)

### Browser WASM
- Limited to single thread or Web Workers
- No blocking operations on main thread
- SharedArrayBuffer required for shared memory
- Hardware concurrency detection available

### WASI
- Threads available only with wasi-threads support
- Wasmtime and Wasmer have experimental support
- Falls back to single-threaded execution
- Compatible API for portable code

## Performance Characteristics

### Mutex (parking_lot)
- Lock: ~20ns (uncontended)
- Lock: ~100ns (contended)
- Much faster than std::sync::Mutex

### RwLock (parking_lot)
- Read lock: ~20ns (uncontended)
- Write lock: ~20ns (uncontended)
- Supports reader-writer fairness

### Queue (crossbeam)
- Push: ~50ns
- Pop: ~50ns
- Lock-free on most operations

### Thread Pool (rayon)
- Work stealing for load balancing
- Scales to available CPU cores
- Minimal overhead for task dispatch

## Dependencies

- **parking_lot**: Fast mutex and rwlock implementations
- **crossbeam**: Lock-free data structures and channels
- **rayon**: Data parallelism and work stealing
- **tokio**: Async semaphores and notifications
- **libc**: Platform-specific thread utilities (Linux)

## Testing

Comprehensive test suite covering:
- Thread creation and joining
- All synchronization primitives
- Thread-safe collections
- Thread pools
- Concurrent scenarios
- Platform capabilities

Run tests:
```bash
cargo test wasi_threading
```

## Error Handling

All operations return `Result<T>` with `ThreadingError` for:
- Thread spawn failures
- Lock poisoning
- Deadlock detection (where possible)
- Timeout errors
- Resource exhaustion
- Platform limitations

## Future Enhancements

- [ ] Async/await integration
- [ ] More sophisticated deadlock detection
- [ ] Thread priority support on more platforms
- [ ] Web Workers full implementation
- [ ] WASI threads integration as spec matures
- [ ] Performance profiling tools
- [ ] Memory ordering guarantees documentation

## License

Part of the Portalis project - MIT License
