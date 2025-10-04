# Threading Implementation - Quick Reference Guide

**Project:** Portalis WASM Runtime Threading
**Coordinator:** SWARM COORDINATOR
**Last Updated:** 2025-10-04

---

## Quick Links

- [Full Coordination Report](./THREADING_WEB_WORKERS_COORDINATION_REPORT.md)
- Architecture Document: TBD (Day 3)
- GitHub Project Board: TBD (setup in progress)

---

## Agent Quick Reference

### Agent 1: Architecture & Design
**Timeline:** Days 1-3
**Key Files:**
- `/workspace/portalis/agents/transpiler/src/threading/mod.rs`
- `/workspace/portalis/agents/transpiler/src/threading/backend.rs`
- `/workspace/portalis/plans/threading/ARCHITECTURE.md`

**Next Actions:**
1. Create threading module directory
2. Design trait `ThreadingBackend`
3. Define message types
4. Write architecture doc

---

### Agent 2: Native Threading
**Timeline:** Days 3-7
**Dependencies:** Architecture Agent
**Key Files:**
- `/workspace/portalis/agents/transpiler/src/threading/native.rs`
- `/workspace/portalis/agents/transpiler/src/threading/native_sync.rs`

**Implementation Checklist:**
- [ ] NativeThreadPool with rayon
- [ ] Thread spawn/join/detach
- [ ] Panic handling
- [ ] Graceful shutdown
- [ ] 50+ unit tests
- [ ] Benchmarks (<10μs spawn)

---

### Agent 3: Web Worker Bridge
**Timeline:** Days 4-10
**Dependencies:** Architecture Agent
**Key Files:**
- `/workspace/portalis/agents/transpiler/src/threading/webworker.rs`
- `/workspace/portalis/agents/transpiler/src/threading/webworker_bindings.rs`

**Implementation Checklist:**
- [ ] WebWorkerPool
- [ ] wasm-bindgen Worker API
- [ ] PostMessage protocol
- [ ] Worker lifecycle
- [ ] Browser integration tests

---

### Agent 4: WASI Threading
**Timeline:** Days 5-9
**Dependencies:** Architecture Agent
**Key Files:**
- `/workspace/portalis/agents/transpiler/src/threading/wasi_threads.rs`

**Implementation Checklist:**
- [ ] WasiThreadPool
- [ ] wasi-threads bindings
- [ ] Atomic operations
- [ ] wasmtime integration tests

---

### Agent 5: Python API Translation
**Timeline:** Days 6-12
**Dependencies:** All implementation agents
**Key Files:**
- `/workspace/portalis/agents/transpiler/src/stdlib_mappings_threading.rs`
- `/workspace/portalis/agents/transpiler/src/stdlib_mapper.rs` (modify)
- `/workspace/portalis/agents/transpiler/src/python_to_rust.rs` (modify)

**API Coverage:**
- [ ] threading.Thread → PortalisThread::spawn
- [ ] threading.Lock → Mutex<()>
- [ ] threading.RLock → RwLock<()>
- [ ] threading.Semaphore
- [ ] threading.Event
- [ ] queue.Queue → mpsc::channel
- [ ] multiprocessing.Pool → ThreadPool
- [ ] concurrent.futures.ThreadPoolExecutor

---

### Agent 6: Message Passing
**Timeline:** Days 7-11
**Dependencies:** Implementation agents
**Key Files:**
- `/workspace/portalis/agents/transpiler/src/threading/channels.rs`

**Channel Types:**
- [ ] MPSC (unbounded/bounded)
- [ ] Oneshot
- [ ] Broadcast
- [ ] Rendezvous

---

### Agent 7: Shared State
**Timeline:** Days 8-13
**Dependencies:** Native + WebWorker agents
**Key Files:**
- `/workspace/portalis/agents/transpiler/src/threading/shared.rs`
- `/workspace/portalis/agents/transpiler/src/threading/sync.rs`

**Primitives:**
- [ ] SharedState<T> (Arc<Mutex<T>> wrapper)
- [ ] RwLock abstraction
- [ ] Barrier
- [ ] Semaphore
- [ ] Condition variables

---

### Agent 8: Error Handling
**Timeline:** Days 9-14
**Dependencies:** All implementation agents
**Key Files:**
- `/workspace/portalis/agents/transpiler/src/threading/error.rs`

**Error Types:**
- [ ] ThreadError enum
- [ ] Panic recovery (catch_unwind)
- [ ] Worker error propagation
- [ ] Graceful shutdown

---

### Agent 9: Testing & Validation
**Timeline:** Days 10-15 (continuous)
**Dependencies:** All agents
**Key Files:**
- `/workspace/portalis/agents/transpiler/src/threading/tests/`
- `/workspace/portalis/tests/threading_integration.rs`

**Test Categories:**
- [ ] Unit tests (500+ tests, 95% coverage)
- [ ] Integration tests (100+ scenarios)
- [ ] Stress tests (10k+ threads)
- [ ] loom (exhaustive interleaving)
- [ ] miri (undefined behavior)
- [ ] Performance benchmarks

---

## Architecture Patterns (From Existing Code)

### Pattern 1: Thread-Safe State (from wasi_core.rs)
```rust
pub struct ThreadPool {
    threads: Arc<RwLock<Vec<ThreadHandle>>>,
    task_queue: Arc<Mutex<VecDeque<Task>>>,
}
```

### Pattern 2: Cross-Platform (from wasi_fetch.rs)
```rust
pub fn spawn<F>(&self, f: F) -> Result<ThreadHandle>
where F: FnOnce() + Send + 'static
{
    #[cfg(not(target_arch = "wasm32"))]
    { self.spawn_native(f) }

    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    { self.spawn_wasi(f) }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
    { self.spawn_webworker(f) }
}
```

### Pattern 3: Module Extension (from stdlib_mapper.rs)
```rust
// In stdlib_mapper.rs init_mappings()
let threading_mappings = crate::stdlib_mappings_threading::init_threading_mappings();

for (module_mapping, function_mappings) in threading_mappings {
    self.add_module(module_mapping.clone());
    // ...
}
```

---

## Performance Targets

| Operation | Native | Web Worker | WASI Thread |
|-----------|--------|------------|-------------|
| Thread spawn | <10μs | <100ms | <50μs |
| Message send | <10ns | <1ms | <100ns |
| Lock acquire | <20ns | N/A | <50ns |

---

## Dependencies (Add to Cargo.toml)

```toml
[dependencies]
# Existing (already in transpiler)
tokio.workspace = true
rayon = "1.8"
crossbeam = "0.8"

# NEW
flume = "0.11"           # Fast channels
parking_lot = "0.12"     # Better locks
thread_local = "1.1"     # TLS

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["Worker", "MessageEvent"] }

[dev-dependencies]
loom = "0.7"             # Concurrency testing
criterion = "0.5"        # Benchmarking
```

---

## Milestones Checklist

- [ ] **Day 3:** Architecture Review (API contracts finalized)
- [ ] **Day 7:** Native Backend Complete (thread pool operational)
- [ ] **Day 10:** WebWorker Backend Alpha (worker spawn working)
- [ ] **Day 12:** Python API Mappings (all threading APIs mapped)
- [ ] **Day 14:** Integration Testing (end-to-end tests passing)
- [ ] **Day 15:** Performance Validation (benchmarks meet targets)

---

## File Structure

```
agents/transpiler/src/
├── threading/                      # NEW MODULE
│   ├── mod.rs                      # Public API + cross-platform dispatch
│   ├── backend.rs                  # ThreadingBackend trait
│   ├── native.rs                   # std::thread + rayon backend
│   ├── webworker.rs                # Web Workers backend
│   ├── wasi_threads.rs             # WASI threads backend
│   ├── channels.rs                 # Message passing (MPSC, oneshot, etc.)
│   ├── sync.rs                     # Synchronization primitives
│   ├── shared.rs                   # Shared state (Arc/Mutex wrappers)
│   ├── error.rs                    # Error types + panic recovery
│   └── tests/
│       ├── native_tests.rs
│       ├── webworker_tests.rs
│       ├── wasi_tests.rs
│       ├── channels_tests.rs
│       └── stress_tests.rs
├── stdlib_mappings_threading.rs    # NEW - threading/multiprocessing/queue mappings
├── stdlib_mapper.rs                # MODIFY - import threading mappings
├── python_to_rust.rs               # MODIFY - threading code generation
└── lib.rs                          # MODIFY - pub mod threading;
```

---

## Common Commands

### Run Tests
```bash
# All tests
cargo test --package portalis-transpiler

# Threading tests only
cargo test --package portalis-transpiler threading

# With loom (exhaustive concurrency)
cargo test --package portalis-transpiler --features loom

# With miri (UB detection)
cargo +nightly miri test
```

### Run Benchmarks
```bash
cargo bench --package portalis-transpiler -- threading
```

### Build for WASM
```bash
# WASI target
cargo build --target wasm32-wasi --package portalis-transpiler

# Browser target
cargo build --target wasm32-unknown-unknown --package portalis-transpiler
```

---

## Python API Translation Examples

### threading.Thread
```python
# Python
import threading
t = threading.Thread(target=worker)
t.start()
t.join()
```
```rust
// Rust
let t = Thread::spawn(worker);
t.join().unwrap();
```

### threading.Lock
```python
# Python
lock = threading.Lock()
with lock:
    # critical section
    pass
```
```rust
// Rust
let lock = Mutex::new(());
let _guard = lock.lock().unwrap();
// critical section
// _guard dropped automatically
```

### queue.Queue
```python
# Python
import queue
q = queue.Queue()
q.put(42)
item = q.get()
```
```rust
// Rust
let (tx, rx) = channel::unbounded();
tx.send(42).unwrap();
let item = rx.recv().unwrap();
```

---

## Daily Standup Template

```markdown
### Agent N: [Agent Name]
**Date:** YYYY-MM-DD

#### Progress
- [x] Completed: X
- [ ] In Progress: Y

#### Blockers
- Waiting on Z from Agent M

#### Next Steps
- Tomorrow: Plan to work on A

#### Questions/Concerns
- Need clarification on B
```

---

## Code Review Checklist

- [ ] Compiles without warnings
- [ ] All tests pass
- [ ] Benchmarks meet targets
- [ ] rustdoc for public APIs
- [ ] Examples provided
- [ ] Cross-platform tested (native + wasm)
- [ ] No unsafe code (or justified)
- [ ] No clippy warnings (pedantic)
- [ ] Error handling comprehensive
- [ ] Panic recovery implemented

---

## Resources

- **Rust Threading Book:** https://doc.rust-lang.org/book/ch16-00-concurrency.html
- **rayon Documentation:** https://docs.rs/rayon/latest/rayon/
- **crossbeam Guide:** https://docs.rs/crossbeam/latest/crossbeam/
- **Web Workers API:** https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API
- **wasi-threads Proposal:** https://github.com/WebAssembly/wasi-threads
- **loom Tutorial:** https://github.com/tokio-rs/loom

---

## Contact

**Coordinator:** SWARM COORDINATOR
**Report Location:** `/workspace/portalis/plans/THREADING_WEB_WORKERS_COORDINATION_REPORT.md`
**Issues:** GitHub Project Board (TBD)
