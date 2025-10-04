# Threading Implementation - Task Breakdown

**Project:** Portalis WASM Runtime Threading
**Created:** 2025-10-04
**Purpose:** Detailed task breakdown for GitHub issue creation

---

## Epic: Threading via Web Workers Implementation

**Description:** Implement cross-platform threading capabilities supporting native threads (std::thread + rayon), Web Workers (browser), and WASI threads (WebAssembly System Interface).

**Acceptance Criteria:**
- All 37 Python threading APIs translated to Rust
- Cross-platform support: native, wasm32-wasi, wasm32-unknown-unknown
- Performance: <10μs native thread spawn, <100ms worker spawn
- Testing: 95%+ coverage, zero race conditions (loom/miri verified)
- Documentation: Complete rustdoc + examples

---

## Phase 1: Architecture & Foundation (Days 1-3)

### Task 1.1: Create Threading Module Structure
**Assignee:** Architecture Agent
**Priority:** P0 (blocking all others)
**Estimated Time:** 4 hours

**Description:**
Create the base module structure for the threading implementation.

**Subtasks:**
- [ ] Create `/workspace/portalis/agents/transpiler/src/threading/` directory
- [ ] Create `mod.rs` with module documentation
- [ ] Create stub files: `backend.rs`, `native.rs`, `webworker.rs`, `wasi_threads.rs`
- [ ] Create `error.rs` with base error types
- [ ] Update `lib.rs` to include `pub mod threading;`

**Acceptance Criteria:**
- Module compiles with `cargo build --package portalis-transpiler`
- Module exports empty public API
- Documentation comments present

---

### Task 1.2: Design ThreadingBackend Trait
**Assignee:** Architecture Agent
**Priority:** P0
**Estimated Time:** 6 hours

**Description:**
Design the core trait that all threading backends implement.

**Implementation:**
```rust
// threading/backend.rs
pub trait ThreadingBackend: Send + Sync {
    type ThreadHandle: Send;
    type Error: std::error::Error + Send + Sync + 'static;

    fn spawn<F>(&self, f: F) -> Result<Self::ThreadHandle, Self::Error>
    where
        F: FnOnce() + Send + 'static;

    fn join(&self, handle: Self::ThreadHandle) -> Result<(), Self::Error>;

    fn available_parallelism(&self) -> usize;

    fn sleep(&self, duration: Duration);
}
```

**Acceptance Criteria:**
- Trait compiles on all targets (native, wasm32-wasi, wasm32-unknown)
- Design document created explaining trait choices
- Example stub implementations for each backend

---

### Task 1.3: Define Message Types
**Assignee:** Architecture Agent
**Priority:** P0
**Estimated Time:** 4 hours

**Description:**
Define serializable message types for cross-thread/worker communication.

**Implementation:**
```rust
// threading/message.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message<T> {
    Data(T),
    Terminate,
    Error(String),
}

pub type ThreadResult<T> = Result<T, ThreadError>;
```

**Acceptance Criteria:**
- Types are `Send + Sync`
- Serde serialization works
- Unit tests for serialization round-trip

---

### Task 1.4: Write Architecture Document
**Assignee:** Architecture Agent
**Priority:** P0
**Estimated Time:** 8 hours

**Description:**
Create comprehensive architecture document with diagrams.

**Contents:**
- System architecture diagram (native vs WASM backends)
- Message flow diagrams
- State machine diagrams for thread lifecycle
- API design rationale
- Performance considerations
- Security model

**Deliverable:**
`/workspace/portalis/plans/threading/ARCHITECTURE.md`

**Acceptance Criteria:**
- Document reviewed by coordinator
- All agents understand the design
- No unresolved questions

---

## Phase 2: Native Backend (Days 3-7)

### Task 2.1: Implement NativeThreadPool
**Assignee:** Native Threading Agent
**Priority:** P1
**Dependencies:** Task 1.2
**Estimated Time:** 12 hours

**Description:**
Implement thread pool using rayon for native platforms.

**Implementation:**
```rust
// threading/native.rs
#[cfg(not(target_arch = "wasm32"))]
pub struct NativeThreadPool {
    pool: rayon::ThreadPool,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl ThreadingBackend for NativeThreadPool {
    // Implementation
}
```

**Acceptance Criteria:**
- Spawns threads via rayon
- Work-stealing queue operational
- Thread count configurable
- 20+ unit tests pass

---

### Task 2.2: Implement Thread Lifecycle Management
**Assignee:** Native Threading Agent
**Priority:** P1
**Dependencies:** Task 2.1
**Estimated Time:** 6 hours

**Description:**
Handle thread spawn, join, detach, and panic recovery.

**Features:**
- Spawn with custom stack size
- Join with timeout
- Detach for fire-and-forget
- Panic recovery with `catch_unwind`

**Acceptance Criteria:**
- All lifecycle operations tested
- Panics don't crash thread pool
- Resource cleanup verified

---

### Task 2.3: Implement Native Synchronization Primitives
**Assignee:** Native Threading Agent
**Priority:** P1
**Dependencies:** Task 2.1
**Estimated Time:** 8 hours

**Description:**
Implement Mutex, RwLock, Semaphore, Barrier, Condition.

**Implementation:**
```rust
// threading/native_sync.rs
#[cfg(not(target_arch = "wasm32"))]
pub struct NativeMutex<T>(parking_lot::Mutex<T>);

#[cfg(not(target_arch = "wasm32"))]
pub struct NativeRwLock<T>(parking_lot::RwLock<T>);
```

**Acceptance Criteria:**
- All primitives tested for correctness
- loom tests pass (exhaustive interleaving)
- Performance benchmarks meet targets

---

### Task 2.4: Native Backend Benchmarks
**Assignee:** Native Threading Agent
**Priority:** P2
**Dependencies:** Tasks 2.1-2.3
**Estimated Time:** 4 hours

**Description:**
Create criterion benchmarks for native backend.

**Benchmarks:**
- Thread spawn latency (<10μs)
- Message send latency (<10ns)
- Lock acquire latency (<20ns)
- Context switch overhead (<5μs)

**Deliverable:**
`benches/threading_native.rs`

**Acceptance Criteria:**
- All benchmarks meet targets
- Regression detection configured

---

## Phase 3: Web Worker Backend (Days 4-10)

### Task 3.1: Web Worker Pool Implementation
**Assignee:** Web Worker Bridge Agent
**Priority:** P1
**Dependencies:** Task 1.2
**Estimated Time:** 16 hours

**Description:**
Implement worker pool using wasm-bindgen and Web Workers API.

**Implementation:**
```rust
// threading/webworker.rs
#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
pub struct WebWorkerPool {
    workers: Vec<Worker>,
    available: Arc<Mutex<VecDeque<usize>>>,
    message_handlers: HashMap<WorkerId, Receiver<Message>>,
}
```

**Acceptance Criteria:**
- Workers spawn dynamically
- Worker pool size configurable
- Pre-warming optimization implemented
- 15+ unit tests pass

---

### Task 3.2: JavaScript Interop Layer
**Assignee:** Web Worker Bridge Agent
**Priority:** P1
**Dependencies:** Task 3.1
**Estimated Time:** 12 hours

**Description:**
Implement wasm-bindgen bindings for Worker API.

**Features:**
- PostMessage serialization via serde
- MessageEvent handling
- Error event propagation
- SharedArrayBuffer support (when available)

**Implementation:**
```rust
// threading/webworker_bindings.rs
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Worker)]
    type Worker;

    #[wasm_bindgen(constructor)]
    fn new(script_url: &str) -> Worker;

    #[wasm_bindgen(method)]
    fn postMessage(this: &Worker, msg: &JsValue);
}
```

**Acceptance Criteria:**
- Messages serialize correctly
- Round-trip test passes (send → receive)
- Error events captured

---

### Task 3.3: Web Worker Lifecycle Management
**Assignee:** Web Worker Bridge Agent
**Priority:** P1
**Dependencies:** Task 3.1
**Estimated Time:** 8 hours

**Description:**
Manage worker creation, termination, and cleanup.

**Features:**
- Dynamic worker creation
- Worker termination on pool shutdown
- Resource leak prevention
- Error handling across JS boundary

**Acceptance Criteria:**
- Workers terminate cleanly
- No memory leaks (browser dev tools verification)
- Error propagation tested

---

### Task 3.4: Browser Integration Testing
**Assignee:** Web Worker Bridge Agent
**Priority:** P2
**Dependencies:** Tasks 3.1-3.3
**Estimated Time:** 6 hours

**Description:**
Create browser-based integration tests using wasm-pack test.

**Setup:**
```bash
wasm-pack test --headless --chrome
wasm-pack test --headless --firefox
```

**Tests:**
- Worker spawn in browser
- Message passing latency
- Multiple workers concurrently
- Error handling in browser

**Acceptance Criteria:**
- Tests pass in Chrome and Firefox
- CI/CD pipeline runs browser tests

---

## Phase 4: WASI Threading Backend (Days 5-9)

### Task 4.1: WASI Thread Pool Implementation
**Assignee:** WASI Threading Agent
**Priority:** P1
**Dependencies:** Task 1.2
**Estimated Time:** 14 hours

**Description:**
Implement thread pool using wasi-threads proposal.

**Implementation:**
```rust
// threading/wasi_threads.rs
#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
pub struct WasiThreadPool {
    threads: Vec<WasiThreadHandle>,
    // wasi-threads API bindings
}
```

**Acceptance Criteria:**
- Conforms to wasi-threads specification
- Works with wasmtime runtime
- 10+ unit tests pass

---

### Task 4.2: WASI Atomic Operations
**Assignee:** WASI Threading Agent
**Priority:** P1
**Dependencies:** Task 4.1
**Estimated Time:** 6 hours

**Description:**
Implement atomic operations using wasm atomics.

**Features:**
- AtomicUsize, AtomicBool wrappers
- Memory ordering (Acquire, Release, SeqCst)
- Compare-and-swap (CAS)

**Acceptance Criteria:**
- Atomics work correctly
- Memory ordering tests pass
- loom verification (where applicable)

---

### Task 4.3: WASI Integration Testing
**Assignee:** WASI Threading Agent
**Priority:** P2
**Dependencies:** Tasks 4.1-4.2
**Estimated Time:** 4 hours

**Description:**
Test WASI threading with wasmtime runtime.

**Setup:**
```bash
cargo build --target wasm32-wasi --package portalis-transpiler
wasmtime run --wasi-modules=wasi-threads target/wasm32-wasi/debug/portalis_transpiler.wasm
```

**Acceptance Criteria:**
- Threads spawn in wasmtime
- Message passing works
- No runtime errors

---

## Phase 5: Message Passing & Channels (Days 7-11)

### Task 5.1: Unified Channel API
**Assignee:** Message Passing Agent
**Priority:** P1
**Dependencies:** Tasks 2.1, 3.1, 4.1
**Estimated Time:** 10 hours

**Description:**
Create unified channel abstraction across all backends.

**Implementation:**
```rust
// threading/channels.rs
pub enum Channel<T> {
    Native(crossbeam::channel::Sender<T>, crossbeam::channel::Receiver<T>),
    WebWorker(WorkerMessageSender<T>, WorkerMessageReceiver<T>),
    Wasi(WasiChannelSender<T>, WasiChannelReceiver<T>),
}
```

**Acceptance Criteria:**
- Same API for all backends
- Type-safe message passing
- Zero-copy optimization where possible

---

### Task 5.2: Implement Channel Patterns
**Assignee:** Message Passing Agent
**Priority:** P1
**Dependencies:** Task 5.1
**Estimated Time:** 8 hours

**Description:**
Implement MPSC, oneshot, broadcast, rendezvous channels.

**Patterns:**
- **MPSC:** Multiple producers, single consumer
- **Oneshot:** Request/response
- **Broadcast:** Event notifications
- **Rendezvous:** Synchronization barrier

**Acceptance Criteria:**
- Each pattern has 10+ tests
- Performance benchmarks pass
- Documentation with examples

---

### Task 5.3: Message Serialization
**Assignee:** Message Passing Agent
**Priority:** P1
**Dependencies:** Task 5.1
**Estimated Time:** 6 hours

**Description:**
Implement efficient serialization for cross-boundary messages.

**Features:**
- Serde-based serialization
- bincode for compact encoding
- Zero-copy via SharedArrayBuffer (when available)

**Acceptance Criteria:**
- Serialization round-trip tests
- Performance: <1ms for 1KB message
- No data loss

---

## Phase 6: Shared State Management (Days 8-13)

### Task 6.1: SharedState Abstraction
**Assignee:** Shared State Agent
**Priority:** P1
**Dependencies:** Tasks 2.1, 3.1
**Estimated Time:** 8 hours

**Description:**
Create cross-platform shared state wrapper.

**Implementation:**
```rust
// threading/shared.rs
pub struct SharedState<T> {
    #[cfg(not(target_arch = "wasm32"))]
    inner: Arc<parking_lot::Mutex<T>>,

    #[cfg(target_arch = "wasm32")]
    inner: Rc<RefCell<T>>,
}
```

**Acceptance Criteria:**
- Works on all platforms
- Type-safe API
- No data races (loom verified)

---

### Task 6.2: Lock-Free Structures
**Assignee:** Shared State Agent
**Priority:** P2
**Dependencies:** Task 6.1
**Estimated Time:** 10 hours

**Description:**
Implement lock-free data structures using atomics.

**Structures:**
- AtomicCounter
- Lock-free queue (MPSC)
- SeqLock (for read-heavy workloads)

**Acceptance Criteria:**
- loom tests pass
- Performance benchmarks better than Mutex
- Linearizability tests

---

### Task 6.3: Synchronization Primitives
**Assignee:** Shared State Agent
**Priority:** P1
**Dependencies:** Task 6.1
**Estimated Time:** 12 hours

**Description:**
Implement RwLock, Barrier, Semaphore, Condition.

**Primitives:**
- **RwLock:** Read-write lock
- **Barrier:** Wait for N threads
- **Semaphore:** Resource counting
- **Condition:** Wait/notify

**Acceptance Criteria:**
- Each primitive has 10+ tests
- Correctness verified with loom
- Performance benchmarks

---

## Phase 7: Python API Translation (Days 6-12)

### Task 7.1: Threading Module Mappings
**Assignee:** Python API Translation Agent
**Priority:** P1
**Dependencies:** Tasks 2.1, 5.1, 6.1
**Estimated Time:** 12 hours

**Description:**
Create stdlib mappings for Python threading module.

**Implementation:**
```rust
// stdlib_mappings_threading.rs
pub fn init_threading_mappings() -> Vec<(ModuleMapping, Vec<FunctionMapping>)> {
    vec![
        (
            ModuleMapping {
                python_module: "threading".to_string(),
                rust_crate: Some("portalis_threading".to_string()),
                // ...
            },
            vec![
                FunctionMapping {
                    python_name: "Thread".to_string(),
                    rust_equiv: "Thread::spawn".to_string(),
                    // ...
                },
                // ... 18 threading APIs
            ]
        )
    ]
}
```

**Acceptance Criteria:**
- All 18 threading APIs mapped
- Documentation for each mapping
- Translation examples

---

### Task 7.2: Multiprocessing & Queue Mappings
**Assignee:** Python API Translation Agent
**Priority:** P1
**Dependencies:** Task 7.1
**Estimated Time:** 10 hours

**Description:**
Map multiprocessing and queue modules to Rust equivalents.

**APIs:**
- multiprocessing.Process → Thread
- multiprocessing.Pool → ThreadPool
- queue.Queue → mpsc::channel

**Acceptance Criteria:**
- All 15 APIs mapped
- Semantic differences documented
- Examples provided

---

### Task 7.3: concurrent.futures Mappings
**Assignee:** Python API Translation Agent
**Priority:** P1
**Dependencies:** Task 7.1
**Estimated Time:** 6 hours

**Description:**
Map concurrent.futures module to rayon/tokio.

**APIs:**
- ThreadPoolExecutor → rayon::ThreadPool
- Future → tokio::sync::oneshot
- wait/as_completed → Iterator patterns

**Acceptance Criteria:**
- All 4 APIs mapped
- Examples with code generation
- Integration tests

---

### Task 7.4: Code Generation Templates
**Assignee:** Python API Translation Agent
**Priority:** P1
**Dependencies:** Tasks 7.1-7.3
**Estimated Time:** 10 hours

**Description:**
Integrate threading translations into python_to_rust.rs code generator.

**Templates:**
- Thread creation
- Lock acquisition (with statement)
- Queue operations
- Pool executor patterns

**Acceptance Criteria:**
- Code generation tests (100+ cases)
- Generated code compiles
- Semantics preserved

---

## Phase 8: Error Handling (Days 9-14)

### Task 8.1: Define ThreadError Types
**Assignee:** Error Handling Agent
**Priority:** P1
**Dependencies:** All implementation agents
**Estimated Time:** 4 hours

**Description:**
Create comprehensive error type hierarchy.

**Implementation:**
```rust
// threading/error.rs
#[derive(Debug, thiserror::Error)]
pub enum ThreadError {
    #[error("Thread panicked: {0}")]
    Panic(String),

    #[error("Join failed: {0}")]
    JoinError(String),

    #[error("Worker spawn failed: {0}")]
    SpawnError(String),

    #[error("Message send failed: {0}")]
    SendError(String),

    #[error("Message receive failed: {0}")]
    RecvError(String),
}
```

**Acceptance Criteria:**
- All error cases covered
- Error conversion traits (From<>)
- Error messages helpful

---

### Task 8.2: Panic Recovery Implementation
**Assignee:** Error Handling Agent
**Priority:** P1
**Dependencies:** Task 8.1
**Estimated Time:** 8 hours

**Description:**
Implement panic recovery for all backends.

**Features:**
- catch_unwind for native threads
- Worker error event handling
- Thread pool resilience (respawn on panic)

**Acceptance Criteria:**
- Panics don't crash pool
- Errors propagate correctly
- Tests for all panic scenarios

---

### Task 8.3: Graceful Shutdown
**Assignee:** Error Handling Agent
**Priority:** P1
**Dependencies:** Task 8.1
**Estimated Time:** 6 hours

**Description:**
Implement coordinated shutdown for all backends.

**Features:**
- Signal all threads/workers to stop
- Wait for completion (with timeout)
- Resource cleanup
- Drop trait implementation

**Acceptance Criteria:**
- Shutdown completes within 5 seconds
- No resource leaks
- Tests with 100+ threads

---

## Phase 9: Testing & Validation (Days 10-15)

### Task 9.1: Unit Test Suite
**Assignee:** Testing & Validation Agent
**Priority:** P0 (continuous)
**Estimated Time:** 20 hours

**Description:**
Create comprehensive unit test suite for all modules.

**Coverage:**
- Native backend: 50+ tests
- WebWorker backend: 40+ tests
- WASI backend: 30+ tests
- Channels: 40+ tests
- Synchronization: 50+ tests
- Error handling: 30+ tests

**Target:** 95%+ code coverage

**Acceptance Criteria:**
- All tests pass on all platforms
- Coverage report generated
- No flaky tests

---

### Task 9.2: Integration Tests
**Assignee:** Testing & Validation Agent
**Priority:** P1
**Dependencies:** All implementation tasks
**Estimated Time:** 12 hours

**Description:**
Create end-to-end integration tests.

**Scenarios:**
- Thread spawn → message pass → join
- Multiple threads with shared state
- Error propagation across boundaries
- Graceful shutdown with active threads
- Cross-platform compatibility

**Deliverable:**
`/workspace/portalis/tests/threading_integration.rs`

**Acceptance Criteria:**
- 100+ integration test cases
- Tests pass on all platforms
- CI/CD pipeline runs tests

---

### Task 9.3: Stress Testing
**Assignee:** Testing & Validation Agent
**Priority:** P2
**Dependencies:** All implementation tasks
**Estimated Time:** 8 hours

**Description:**
Create stress tests for concurrency and performance.

**Tests:**
- 10,000+ concurrent threads
- 1M+ messages/second throughput
- Resource leak detection
- Memory usage under load

**Acceptance Criteria:**
- No crashes under stress
- Performance targets met
- Memory usage stable

---

### Task 9.4: Concurrency Verification (loom/miri)
**Assignee:** Testing & Validation Agent
**Priority:** P1
**Dependencies:** All implementation tasks
**Estimated Time:** 10 hours

**Description:**
Use loom and miri to verify concurrency correctness.

**Tools:**
- **loom:** Exhaustive interleaving exploration
- **miri:** Undefined behavior detection

**Acceptance Criteria:**
- All loom tests pass (100k+ interleavings)
- miri reports zero UB
- No data races detected

---

### Task 9.5: Performance Benchmarking
**Assignee:** Testing & Validation Agent
**Priority:** P1
**Dependencies:** All implementation tasks
**Estimated Time:** 6 hours

**Description:**
Create comprehensive benchmark suite.

**Benchmarks:**
- Thread spawn latency
- Message send/recv latency
- Lock acquire/release latency
- Context switch overhead
- Throughput (threads/sec, messages/sec)

**Deliverable:**
`/workspace/portalis/agents/transpiler/benches/threading.rs`

**Acceptance Criteria:**
- All benchmarks meet targets
- Regression detection configured
- Results documented

---

## Phase 10: Documentation (Days 14-15)

### Task 10.1: API Documentation (rustdoc)
**Assignee:** All agents (own code)
**Priority:** P1
**Estimated Time:** 8 hours

**Description:**
Write comprehensive rustdoc for all public APIs.

**Requirements:**
- Module-level documentation
- Function documentation with examples
- Type documentation
- Safety notes (for unsafe code)

**Acceptance Criteria:**
- `cargo doc --open` shows complete docs
- All examples compile via `cargo test --doc`
- No missing docs warnings

---

### Task 10.2: User Guide
**Assignee:** Coordinator
**Priority:** P2
**Estimated Time:** 6 hours

**Description:**
Create user guide for threading module.

**Contents:**
- Quick start guide
- API reference
- Python → Rust translation guide
- Best practices
- Troubleshooting
- FAQ

**Deliverable:**
`/workspace/portalis/plans/threading/USER_GUIDE.md`

**Acceptance Criteria:**
- Clear examples
- Covers common use cases
- Reviewed by agents

---

### Task 10.3: Examples
**Assignee:** All agents (own domain)
**Priority:** P2
**Estimated Time:** 4 hours

**Description:**
Create runnable examples for common patterns.

**Examples:**
- Basic thread spawn
- Message passing
- Shared state with mutex
- Thread pool with work queue
- Error handling

**Deliverable:**
`/workspace/portalis/agents/transpiler/examples/threading/`

**Acceptance Criteria:**
- All examples compile and run
- Output demonstrated
- Comments explain code

---

## Summary Statistics

**Total Tasks:** 47
**Total Estimated Time:** 348 hours
**Agents:** 9
**Average Time per Agent:** 38.7 hours (~5 days @ 8hrs/day)

**Critical Path:**
1. Architecture (Days 1-3)
2. Native Backend (Days 3-7)
3. Python API Translation (Days 6-12)
4. Integration Testing (Days 14-15)

**Parallelization Opportunities:**
- Native, WebWorker, WASI backends can be developed in parallel (Days 4-10)
- Message passing and shared state can overlap (Days 7-13)
- Testing is continuous throughout

---

## Task Priority Legend

- **P0:** Blocking (must complete before others can proceed)
- **P1:** Critical (core functionality)
- **P2:** Important (enhances quality, not blocking)
- **P3:** Nice-to-have (future work)

---

**Document Status:** READY FOR GITHUB ISSUE CREATION
**Next Action:** Coordinator to create GitHub project board and issues
