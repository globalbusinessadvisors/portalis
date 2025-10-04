# Threading Architecture - Visual Diagrams

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Python Source Code                          │
│  ┌────────────────┬──────────────────┬─────────────────────────┐   │
│  │ threading.Thread│ queue.Queue      │ ThreadPoolExecutor      │   │
│  │ threading.Lock  │ threading.Event  │ concurrent.futures      │   │
│  └────────────────┴──────────────────┴─────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ Transpiler
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Transpiler Translation Layer                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              py_to_rust_threading.rs                         │  │
│  │  • translate_thread_class()                                  │  │
│  │  • translate_lock()                                          │  │
│  │  • translate_queue()                                         │  │
│  │  • translate_thread_pool_executor()                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ Generates Rust code using:
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              WASI Threading Abstraction Layer                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                wasi_threading/mod.rs                         │  │
│  │                                                              │  │
│  │  • WasiThreading::spawn()    ┌─────────────────────┐        │  │
│  │  • WasiMutex<T>              │  Platform-agnostic  │        │  │
│  │  • WasiRwLock<T>             │  API surface        │        │  │
│  │  • channel::<T>()            └─────────────────────┘        │  │
│  │  • WasiThreadPool                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ Compile-time selection
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │  Native (Rust)  │ │  WASM + WASI    │ │    Browser      │
        │  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
        │  │std::thread│  │ │  │wasi-threads│ │ │  │ Web       │  │
        │  │   rayon   │  │ │  │   or      │  │ │  │ Workers   │  │
        │  │ crossbeam │  │ │  │   rayon   │  │ │  │postMessage│  │
        │  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │
        │                 │ │                 │ │                 │
        │  Best           │ │  Good           │ │  Acceptable     │
        │  Performance    │ │  Performance    │ │  Performance    │
        └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Module Structure (Directory Layout)

```
agents/transpiler/src/
│
├── wasi_threading/                   # Main threading module
│   │
│   ├── mod.rs                        # Public API
│   │   ├── WasiThread               # Thread handle
│   │   ├── WasiThreading            # Thread spawning
│   │   ├── WasiMutex<T>             # Mutual exclusion
│   │   ├── WasiRwLock<T>            # Reader-writer lock
│   │   ├── WasiCondvar              # Condition variable
│   │   ├── Sender<T> / Receiver<T>  # Channels
│   │   ├── WasiThreadPool           # Thread pool
│   │   └── WasiFuture<T>            # Future result
│   │
│   ├── native.rs                     # Native implementation
│   │   ├── std::thread              # OS threads
│   │   ├── rayon::ThreadPool        # Work-stealing pool
│   │   └── crossbeam::channel       # MPSC channels
│   │
│   ├── wasi_impl.rs                  # WASI implementation
│   │   ├── std::thread (if wasi-threads)
│   │   ├── rayon (fallback)
│   │   └── Custom work-stealing pool
│   │
│   └── browser.rs                    # Browser implementation
│       ├── web_sys::Worker          # Web Workers
│       ├── WebWorkerPool            # Worker pool manager
│       ├── postMessage channels     # Message passing
│       └── Structured clone         # Serialization
│
├── py_to_rust_threading.rs          # Python translation
│   ├── translate_thread_class()
│   ├── translate_lock()
│   ├── translate_queue()
│   └── translate_thread_pool_executor()
│
├── wasi_worker.rs                    # Worker pool utilities
│   ├── WorkerHandle
│   ├── TaskQueue
│   └── TaskDistributor
│
└── lib.rs                            # Module exports
    └── pub mod wasi_threading;
```

## Threading API Components

```
WasiThreading (Static methods)
│
├── spawn<F>(f: F) -> Result<WasiThread>
│   ├─[Native]─> std::thread::spawn(f)
│   ├─[WASI]───> std::thread::spawn(f) or rayon::spawn(f)
│   └─[Browser]> Worker::new() + postMessage
│
├── spawn_with_config<F>(config, f) -> Result<WasiThread>
│   └─> Set thread name, stack size, priority
│
├── current_thread_id() -> ThreadId
├── sleep(duration)
└── yield_now()


WasiThread (Instance methods)
│
├── join(self) -> Result<()>
│   ├─[Native]─> JoinHandle::join()
│   ├─[WASI]───> JoinHandle::join()
│   └─[Browser]> await Worker completion
│
└── is_finished(&self) -> bool


WasiMutex<T>
│
├── new(value: T) -> Self
│   ├─[Native]─> Arc<Mutex<T>>
│   ├─[WASI]───> Arc<Mutex<T>>
│   └─[Browser]> Arc<BrowserMutex<T>> (emulated)
│
├── lock(&self) -> Result<MutexGuard<T>>
└── try_lock(&self) -> Result<Option<MutexGuard<T>>>


channel<T>() -> (Sender<T>, Receiver<T>)
│
├─[Native]─> crossbeam::channel::unbounded()
├─[WASI]───> crossbeam::channel::unbounded()
└─[Browser]> Custom MessageChannel + postMessage


WasiThreadPool
│
├── new() -> Result<Self>
│   ├─[Native]─> rayon::ThreadPool
│   ├─[WASI]───> rayon or custom pool
│   └─[Browser]> WebWorkerPool
│
├── execute<F>(&self, task: F)
├── submit<F,T>(&self, task: F) -> WasiFuture<T>
└── map<I,F,T>(&self, items, f) -> Vec<T>
```

## Data Flow: Thread Spawning

```
Python Code:
┌────────────────────────────────────────────┐
│ import threading                           │
│                                            │
│ def worker(name):                          │
│     print(f"Worker {name}")                │
│                                            │
│ t = threading.Thread(target=worker,       │
│                      args=("Alice",))     │
│ t.start()                                  │
│ t.join()                                   │
└────────────────┬───────────────────────────┘
                 │ Transpiler
                 ▼
Rust Code (Generated):
┌────────────────────────────────────────────┐
│ use wasi_threading::WasiThreading;         │
│                                            │
│ fn worker(name: String) {                  │
│     println!("Worker {}", name);           │
│ }                                          │
│                                            │
│ let thread = WasiThreading::spawn(        │
│     move || {                              │
│         worker("Alice".to_string());       │
│     }                                      │
│ )?;                                        │
│                                            │
│ thread.join()?;                            │
└────────────────┬───────────────────────────┘
                 │ Compile-time selection
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Native  │  │  WASI   │  │ Browser │
├─────────┤  ├─────────┤  ├─────────┤
│std::    │  │std::    │  │Worker:: │
│thread:: │  │thread:: │  │new()    │
│spawn()  │  │spawn()  │  │         │
└─────────┘  └─────────┘  └─────────┘
```

## Data Flow: Message Passing (Channels)

```
Thread 1 (Producer)          Channel           Thread 2 (Consumer)
┌─────────────────┐         ┌──────┐         ┌─────────────────┐
│                 │         │      │         │                 │
│  tx.send(42)?   │────────>│Queue │────────>│  rx.recv()?     │
│                 │  [1]    │      │  [2]    │                 │
│  tx.send(43)?   │────────>│      │────────>│  rx.recv()?     │
│                 │         │      │         │                 │
└─────────────────┘         └──────┘         └─────────────────┘

[Native/WASI]: crossbeam::channel
    • Lock-free MPSC queue
    • Wait-free fast path
    • ~100ns send/recv

[Browser]: postMessage-based channel
    • Worker.postMessage(data)
    • onmessage event handler
    • ~1ms send/recv (serialization overhead)


Implementation:
┌────────────────────────────────────────────────────────────┐
│  Native/WASI: crossbeam::channel                          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Sender ──> [Internal Queue] ──> Receiver            │ │
│  │     │              │                  │               │ │
│  │     │              │                  │               │ │
│  │  send()      lock-free CAS        recv()             │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Browser: postMessage wrapper                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Sender ──> [Serialize] ──> postMessage              │ │
│  │     │              │                                  │ │
│  │  send()      bincode::serialize                      │ │
│  │                                                       │ │
│  │  onmessage ──> [Deserialize] ──> Receiver.recv()    │ │
│  │                    │                                  │ │
│  │              bincode::deserialize                    │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

## Thread Pool Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      WasiThreadPool                             │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    Task Queue                             │ │
│  │  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐      │ │
│  │  │Task 1│Task 2│Task 3│Task 4│Task 5│Task 6│Task 7│ ...  │ │
│  │  └──────┴──────┴──────┴──────┴──────┴──────┴──────┘      │ │
│  └────────────────────────┬────────────────────────────────── │ │
│                           │ Work Stealing                     │
│           ┌───────────────┼───────────────┬──────────────┐   │ │
│           │               │               │              │   │ │
│           ▼               ▼               ▼              ▼   │ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  Worker 1   │ │  Worker 2   │ │  Worker 3   │ │  Worker 4   │ │
│  │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │
│  │ │Local    │ │ │ │Local    │ │ │ │Local    │ │ │ │Local    │ │ │
│  │ │Queue    │ │ │ │Queue    │ │ │ │Queue    │ │ │ │Queue    │ │ │
│  │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │
│  │             │ │             │ │             │ │             │ │
│  │ Thread/     │ │ Thread/     │ │ Thread/     │ │ Thread/     │ │
│  │ Worker      │ │ Worker      │ │ Worker      │ │ Worker      │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘

Work Stealing Algorithm:
1. Worker checks local queue first
2. If empty, steals from global queue
3. If global empty, steals from other workers' local queues
4. If all empty, yield/sleep

Native/WASI: rayon
    • Production-ready work-stealing
    • Optimal load balancing
    • ~50μs task overhead

Browser: Custom WebWorkerPool
    • Round-robin task distribution
    • No work stealing (workers isolated)
    • ~5ms worker communication overhead
```

## Web Worker Communication Pattern

```
Main Thread                          Web Worker
┌────────────────────────┐          ┌────────────────────────┐
│                        │          │                        │
│  WasiThreadPool        │          │  worker-thread.js      │
│  ┌──────────────────┐  │          │  ┌──────────────────┐  │
│  │ Task Queue       │  │          │  │ WASM Module      │  │
│  │ ┌──┬──┬──┬──┬──┐│  │          │  │                  │  │
│  │ │T1│T2│T3│T4│T5││  │          │  │ await init()     │  │
│  │ └──┴──┴──┴──┴──┘│  │          │  │                  │  │
│  └────────┬─────────┘  │          │  │ execute_task()   │  │
│           │            │          │  │                  │  │
│           │distribute  │          │  └──────────────────┘  │
│           │            │          │                        │
│           ▼            │          │                        │
│  ┌──────────────────┐  │          │  ┌──────────────────┐  │
│  │ Worker.          │  │          │  │ onmessage        │  │
│  │ postMessage()    │──┼─────────>│──┤ handler          │  │
│  │                  │  │   [1]    │  │                  │  │
│  │ { taskId,        │  │          │  │ Extract payload  │  │
│  │   payload }      │  │          │  │ Execute task     │  │
│  └──────────────────┘  │          │  │ Serialize result │  │
│                        │          │  └──────────────────┘  │
│  ┌──────────────────┐  │          │  ┌──────────────────┐  │
│  │ onmessage        │  │          │  │ postMessage()    │  │
│  │ handler          │<─┼──────────┼──┤                  │  │
│  │                  │  │   [2]    │  │ { taskId,        │  │
│  │ Update future    │  │          │  │   result }       │  │
│  │ Trigger callback │  │          │  │                  │  │
│  └──────────────────┘  │          │  └──────────────────┘  │
│                        │          │                        │
└────────────────────────┘          └────────────────────────┘

[1] Task sent to worker (serialized via bincode)
[2] Result returned to main (serialized via bincode)
```

## Synchronization Primitive Hierarchy

```
                    WasiMutex<T>
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    Native            WASI           Browser
Arc<Mutex<T>>    Arc<Mutex<T>>   Arc<BrowserMutex<T>>
    │                │                │
    │                │                └─> Atomics.wait/notify
    │                │                    (if SharedArrayBuffer)
    │                │                    or busy-wait fallback
    │                │
    └────────────────┴─> std::sync::Mutex
                         (OS-level lock)


                    WasiRwLock<T>
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    Native            WASI           Browser
Arc<RwLock<T>>   Arc<RwLock<T>>  Arc<BrowserRwLock<T>>


                    WasiCondvar
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    Native            WASI           Browser
Arc<Condvar>     Arc<Condvar>    Arc<BrowserCondvar>
                                      │
                                      └─> Atomics.wait/notify


                    Atomics
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    Native            WASI           Browser
std::sync::atomic  std::sync::atomic  std::sync::atomic
    (native)          (native)        (WASM atomics)
                                      (requires SharedArrayBuffer)
```

## Error Handling Flow

```
                    Operation Fails
                          │
                          ▼
              ┌─────────────────────┐
              │  Platform Error     │
              │  • std::io::Error   │
              │  • JsValue error    │
              │  • WASI error code  │
              └──────────┬──────────┘
                         │ Map to
                         ▼
              ┌─────────────────────┐
              │  ThreadingError     │
              │  • SpawnFailed      │
              │  • JoinFailed       │
              │  • LockPoisoned     │
              │  • SendError        │
              │  • RecvError        │
              │  • PoolError        │
              │  • WorkerError      │
              │  • Timeout          │
              └──────────┬──────────┘
                         │ Propagate via Result<T>
                         ▼
              ┌─────────────────────┐
              │  User Code          │
              │  match result {     │
              │    Ok(v) => ...,    │
              │    Err(e) => ...,   │
              │  }                  │
              └─────────────────────┘
```

## Platform Comparison Matrix

```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│    Feature      │   Native     │     WASI     │   Browser    │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Threading       │ std::thread  │ std::thread  │ Web Workers  │
│                 │              │ or rayon     │              │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Thread Pool     │ rayon        │ rayon or     │ Custom pool  │
│                 │              │ custom       │              │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Mutex           │ std::Mutex   │ std::Mutex   │ Emulated     │
│                 │              │              │ (Atomics)    │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Channels        │ crossbeam    │ crossbeam    │ postMessage  │
│                 │              │              │ wrapper      │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Shared Memory   │ Yes (Arc)    │ Yes (Arc)    │ No*          │
│                 │              │              │ (postMessage)│
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Atomics         │ Native       │ Native       │ WASM atomics │
│                 │              │              │ (SAB only)   │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Thread Creation │ ~50μs        │ ~100μs       │ ~5ms         │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Context Switch  │ ~2μs         │ ~5μs         │ N/A          │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Message Latency │ ~100ns       │ ~200ns       │ ~1ms         │
└─────────────────┴──────────────┴──────────────┴──────────────┘

* SharedArrayBuffer allows shared memory but is restricted in many contexts
```

## Translation Example: Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Python Input                                            │
├─────────────────────────────────────────────────────────────────┤
│ from concurrent.futures import ThreadPoolExecutor               │
│                                                                 │
│ def compute(x):                                                 │
│     return x * x                                                │
│                                                                 │
│ with ThreadPoolExecutor(max_workers=4) as pool:                │
│     futures = [pool.submit(compute, i) for i in range(10)]    │
│     results = [f.result() for f in futures]                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ py_to_rust_threading.rs
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: AST Analysis                                            │
├─────────────────────────────────────────────────────────────────┤
│ Detected:                                                       │
│  • Import: concurrent.futures.ThreadPoolExecutor               │
│  • Class: ThreadPoolExecutor (max_workers=4)                   │
│  • Method: submit(compute, i)                                  │
│  • Method: result()                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Translation rules
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Rust Code Generation                                   │
├─────────────────────────────────────────────────────────────────┤
│ use portalis_transpiler::wasi_threading::{                     │
│     WasiThreadPool, ThreadPoolConfig                            │
│ };                                                              │
│                                                                 │
│ fn compute(x: i32) -> i32 {                                    │
│     x * x                                                       │
│ }                                                              │
│                                                                 │
│ fn main() -> Result<()> {                                      │
│     let pool = WasiThreadPool::with_config(                    │
│         ThreadPoolConfig {                                      │
│             num_threads: 4,                                     │
│             ..Default::default()                                │
│         }                                                       │
│     )?;                                                        │
│                                                                 │
│     let mut futures = Vec::new();                              │
│     for i in 0..10 {                                           │
│         let future = pool.submit(move || compute(i));          │
│         futures.push(future);                                   │
│     }                                                          │
│                                                                 │
│     let results: Vec<i32> = futures                            │
│         .into_iter()                                           │
│         .map(|f| f.wait().unwrap())                            │
│         .collect();                                            │
│                                                                 │
│     Ok(())                                                     │
│ }                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Compile with cargo
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Platform-Specific Binary                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Native:  cargo build --release                                │
│           └─> Uses rayon ThreadPool                           │
│                                                                 │
│  WASI:    cargo build --target wasm32-wasi --features wasi    │
│           └─> Uses rayon or custom pool                       │
│                                                                 │
│  Browser: cargo build --target wasm32-unknown-unknown \        │
│                       --features wasm                           │
│           └─> Uses WebWorkerPool                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Unified API**: Single `WasiThreading` API works across all platforms
2. **Compile-Time Selection**: Platform chosen at compile time via `cfg` attributes
3. **Zero Runtime Overhead**: No dynamic dispatch, inlined platform-specific code
4. **Type Safety**: Rust's ownership system prevents data races at compile time
5. **Production Ready**: Uses battle-tested libraries (rayon, crossbeam) where possible
6. **Browser Support**: Full Web Workers integration with graceful degradation
7. **Python Compatibility**: Faithful translation of all Python threading patterns

**Status**: Architecture complete, diagrams illustrate all key concepts.
