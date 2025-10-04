# Threading/Web Workers Architecture - Executive Summary

## Overview

Complete architecture for threading and parallel execution in Portalis WASM runtime, following established patterns from WASI filesystem and WebSocket implementations.

## Architecture Comparison

### Consistency with Existing Modules

| Aspect | WASI Filesystem | WASI WebSocket | **WASI Threading** (New) |
|--------|----------------|----------------|--------------------------|
| **Module Organization** | wasi_fs.rs + native/browser/wasi | wasi_websocket/mod.rs + native/browser/wasi | wasi_threading/mod.rs + native/browser/wasi |
| **Platform Selection** | `#[cfg(target_arch = "wasm32")]` | `#[cfg(target_arch = "wasm32")]` | `#[cfg(target_arch = "wasm32")]` |
| **Native Backend** | std::fs | tokio-tungstenite | std::thread + rayon |
| **WASI Backend** | WASI syscalls | tokio-tungstenite | wasi-threads or rayon |
| **Browser Backend** | IndexedDB (virtual) | Web Workers (native) | Web Workers |
| **Error Handling** | WasiFsError enum | WebSocketError enum | ThreadingError enum |
| **Translation Layer** | py_to_rust_fs.rs | (implicit) | py_to_rust_threading.rs |

## Core Components

### 1. Threading API (`wasi_threading.rs`)

```rust
// Unified threading interface
WasiThreading::spawn(|| { /* work */ })     // Like Python threading.Thread
WasiMutex::new(value)                        // Like Python threading.Lock
channel::<T>()                               // Like Python queue.Queue
WasiThreadPool::new()                        // Like ThreadPoolExecutor
```

### 2. Synchronization Primitives

- **WasiMutex** - Mutual exclusion lock (Arc<Mutex> native, emulated browser)
- **WasiRwLock** - Reader-writer lock
- **WasiCondvar** - Condition variable
- **Atomics** - AtomicBool, AtomicU32, etc. (native support across platforms)

### 3. Message Passing

- **Channels** - MPSC/SPSC channels using crossbeam (native) or custom (browser)
- **Serialization** - bincode for binary, serde_json for structured
- **Zero-Copy** - Arc-based sharing for read-only data

### 4. Thread Pool

- **Native**: rayon (work-stealing, production-ready)
- **WASI**: rayon or custom work-stealing pool
- **Browser**: Web Worker pool with round-robin scheduling

## Python Translation Examples

### threading.Thread

```python
# Python
import threading

def worker(name):
    print(f"Worker {name}")

t = threading.Thread(target=worker, args=("Alice",))
t.start()
t.join()
```

```rust
// Rust (generated)
use portalis_transpiler::wasi_threading::WasiThreading;

fn worker(name: String) {
    println!("Worker {}", name);
}

let thread = WasiThreading::spawn(move || {
    worker("Alice".to_string());
})?;

thread.join()?;
```

### queue.Queue

```python
# Python
import queue

q = queue.Queue(maxsize=10)
q.put(42)
item = q.get()
```

```rust
// Rust (generated)
use portalis_transpiler::wasi_threading::{channel_with_config, ChannelConfig};

let (tx, rx) = channel_with_config(ChannelConfig {
    capacity: Some(10),
    ..Default::default()
});

tx.send(42)?;
let item = rx.recv()?;
```

### ThreadPoolExecutor

```python
# Python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    future = executor.submit(pow, 2, 10)
    result = future.result()
```

```rust
// Rust (generated)
use portalis_transpiler::wasi_threading::{WasiThreadPool, ThreadPoolConfig};

let pool = WasiThreadPool::with_config(ThreadPoolConfig {
    num_threads: 4,
    ..Default::default()
})?;

let future = pool.submit(|| i32::pow(2, 10));
let result = future.wait()?;
```

## Platform Implementation Strategy

### Native Platform

```rust
// Uses standard library + rayon
use std::thread;
use rayon::ThreadPool;
use crossbeam::channel;

pub fn spawn_thread(f: impl FnOnce() + Send + 'static) {
    thread::spawn(f);
}
```

### WASI Platform

```rust
// Option 1: wasi-threads (if available)
use std::thread;  // WASI supports std::thread

// Option 2: rayon fallback
use rayon;  // Works in WASI
```

### Browser Platform (Web Workers)

```javascript
// Worker script: thread-worker.js
self.addEventListener('message', async (event) => {
    const { taskId, payload } = event.data;

    // Initialize WASM module in worker
    if (!self.wasmModule) {
        const { default: init } = await import('./portalis.js');
        self.wasmModule = await init();
    }

    // Execute task
    const result = await self.wasmModule.execute_task(payload);

    // Return result
    self.postMessage({ taskId, result });
});
```

```rust
// Rust side
use web_sys::Worker;

pub fn spawn_worker(f: Task) -> Result<WasiThread> {
    let worker = Worker::new("/workers/thread-worker.js")?;

    // Serialize and send task
    let payload = serialize_task(f)?;
    worker.post_message(&payload)?;

    Ok(WasiThread { inner: worker })
}
```

## Performance Characteristics

| Platform | Thread Creation | Context Switch | Message Passing | Notes |
|----------|----------------|----------------|-----------------|-------|
| **Native** | ~50μs | ~2μs | ~100ns (crossbeam) | Best performance |
| **WASI** | ~100μs | ~5μs | ~200ns | Good performance |
| **Browser** | ~5ms (Worker) | N/A (event loop) | ~1ms (postMessage) | Slower but functional |

## Integration with Existing Codebase

### File Structure (Following Established Pattern)

```
agents/transpiler/src/
├── wasi_threading/
│   ├── mod.rs          # Public API (like wasi_websocket/mod.rs)
│   ├── native.rs       # std::thread + rayon
│   ├── browser.rs      # Web Workers
│   └── wasi_impl.rs    # wasi-threads or rayon
├── py_to_rust_threading.rs  # Translation layer (like py_to_rust_fs.rs)
└── lib.rs              # Add: pub mod wasi_threading;
```

### Dependency Pattern (Following WASI Filesystem)

```toml
# Cargo.toml
[dependencies]
rayon = { version = "1.8", optional = true }
crossbeam = { version = "0.8", optional = true }

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2", optional = true }
web-sys = { version = "0.3", features = ["Worker"], optional = true }

[features]
default = ["native-threads"]
native-threads = ["rayon", "crossbeam"]
wasi-threads = []
wasm = ["wasm-bindgen", "web-sys"]
```

## Key Design Decisions

### 1. Module Organization
**Decision**: Use `wasi_threading/` directory structure
**Rationale**: Matches wasi_websocket pattern, clear separation of concerns

### 2. Thread Pool Backend
**Decision**: Use rayon for native/WASI, custom pool for browser
**Rationale**: rayon is production-ready and well-tested, browser needs custom solution

### 3. Message Passing
**Decision**: Use crossbeam channels (native) + custom channels (browser)
**Rationale**: crossbeam is fast and reliable, browser needs postMessage-based channels

### 4. Serialization
**Decision**: bincode for binary, serde_json for structured data
**Rationale**: bincode is fast and compact, JSON is debuggable and universal

### 5. Shared State
**Decision**: Arc<Mutex<T>> for mutable, Arc<T> for immutable
**Rationale**: Standard Rust pattern, works across platforms

## Browser Limitations and Workarounds

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **No SharedArrayBuffer** | Cannot share memory directly | Use postMessage for all communication |
| **Worker startup cost** | ~5ms per worker | Pre-spawn worker pool, reuse workers |
| **Serialization overhead** | ~1ms per message | Batch messages, use binary serialization |
| **No true threading** | Event-loop based concurrency | Accept limitation, focus on I/O parallelism |

## Testing Strategy

### Unit Tests (Per Platform)

```rust
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_native_thread_spawn() {
    let thread = WasiThreading::spawn(|| {
        assert_eq!(2 + 2, 4);
    }).unwrap();

    thread.join().unwrap();
}

#[test]
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn test_browser_worker_spawn() {
    // wasm-bindgen-test
}
```

### Integration Tests

```rust
#[test]
fn test_producer_consumer() {
    let (tx, rx) = channel::<i32>();

    // Spawn producers
    for i in 0..4 {
        let tx = tx.clone();
        WasiThreading::spawn(move || {
            for j in 0..10 {
                tx.send(i * 10 + j).unwrap();
            }
        }).unwrap();
    }

    // Consume
    let mut items = Vec::new();
    for _ in 0..40 {
        items.push(rx.recv().unwrap());
    }

    assert_eq!(items.len(), 40);
}
```

## Implementation Phases

### Phase 1: Core Threading API (Week 1)
- [ ] Implement WasiThread, WasiThreading
- [ ] Native backend (std::thread)
- [ ] Basic tests

### Phase 2: Synchronization Primitives (Week 1)
- [ ] WasiMutex, WasiRwLock
- [ ] Atomics integration
- [ ] Condition variables

### Phase 3: Message Passing (Week 2)
- [ ] Channel API
- [ ] Native backend (crossbeam)
- [ ] Serialization layer

### Phase 4: Thread Pool (Week 2)
- [ ] WasiThreadPool API
- [ ] Native backend (rayon)
- [ ] WASI backend (rayon or custom)

### Phase 5: Browser Support (Week 3)
- [ ] Web Worker pool
- [ ] Browser channels (postMessage)
- [ ] Worker script

### Phase 6: Python Translation (Week 3)
- [ ] py_to_rust_threading.rs
- [ ] threading.Thread translation
- [ ] queue.Queue translation
- [ ] ThreadPoolExecutor translation

### Phase 7: Integration & Testing (Week 4)
- [ ] Feature translator integration
- [ ] End-to-end tests
- [ ] Documentation
- [ ] Performance benchmarks

## Success Criteria

- ✅ All Python threading patterns translate correctly
- ✅ Tests pass on all platforms (native, WASI, browser)
- ✅ Performance: native <10μs overhead, browser <10ms overhead
- ✅ Thread-safe: no data races (verified by Rust type system)
- ✅ Production-ready: comprehensive error handling

## Related Documents

- [Full Architecture](/workspace/portalis/plans/WASI_THREADING_ARCHITECTURE.md) - Complete technical specification
- [WASI Filesystem Architecture](/workspace/portalis/plans/WASI_FILESYSTEM_ARCHITECTURE.md) - Pattern reference
- [WebSocket Implementation](/workspace/portalis/agents/transpiler/WEBSOCKET_IMPLEMENTATION.md) - Similar pattern

---

**Status**: Architecture complete and validated against existing patterns. Ready for implementation.

**Estimated Effort**: 4 weeks (1 developer)

**Risk Assessment**: Low - follows proven patterns, well-understood technologies
