# Threading/Web Workers Implementation Checklist

## Architecture Documents

- [x] Main Architecture (/workspace/portalis/plans/WASI_THREADING_ARCHITECTURE.md) - 2247 lines
- [x] Executive Summary (/workspace/portalis/plans/THREADING_ARCHITECTURE_SUMMARY.md) - 11KB
- [x] Visual Diagrams (/workspace/portalis/plans/THREADING_ARCHITECTURE_DIAGRAM.md) - 37KB

## Pre-Implementation Validation

### Architecture Consistency Check

| Aspect | WASI FS | WASI WebSocket | WASI Threading | Status |
|--------|---------|----------------|----------------|--------|
| Module structure | wasi_fs.rs | wasi_websocket/mod.rs | wasi_threading/mod.rs | ✅ Consistent |
| Platform backends | native/wasi/browser | native/wasi/browser | native/wasi/browser | ✅ Consistent |
| Translation layer | py_to_rust_fs.rs | (implicit) | py_to_rust_threading.rs | ✅ Consistent |
| Error handling | WasiFsError | WebSocketError | ThreadingError | ✅ Consistent |
| Test organization | tests/wasi_integration_test.rs | tests/websocket_tests.rs | tests/threading_integration_test.rs | ✅ Consistent |

### Dependencies Verification

```toml
[dependencies]
# Existing (verified working)
rayon = "1.8"          # ✅ Already used in codebase
crossbeam = "0.8"      # ✅ Standard threading library
tokio = "1"            # ✅ Already used in WebSocket
serde = "1.0"          # ✅ Already used everywhere
bincode = "1.3"        # ✅ Standard serialization
anyhow = "1.0"         # ✅ Already used everywhere
thiserror = "1.0"      # ✅ Already used everywhere
num_cpus = "1.16"      # ✅ Standard library

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"   # ✅ Already used
web-sys = "0.3"        # ✅ Already used (WebSocket)
js-sys = "0.3"         # ✅ Already used
```

## Implementation Phases

### Phase 1: Core Threading API (Week 1)

#### Task 1.1: Module Structure
- [ ] Create `agents/transpiler/src/wasi_threading/` directory
- [ ] Create `mod.rs` with public API
- [ ] Create `native.rs` skeleton
- [ ] Create `wasi_impl.rs` skeleton
- [ ] Create `browser.rs` skeleton
- [ ] Update `lib.rs` with module export

**Files to create**:
- `/workspace/portalis/agents/transpiler/src/wasi_threading/mod.rs` (~400 lines)
- `/workspace/portalis/agents/transpiler/src/wasi_threading/native.rs` (~200 lines)
- `/workspace/portalis/agents/transpiler/src/wasi_threading/wasi_impl.rs` (~200 lines)
- `/workspace/portalis/agents/transpiler/src/wasi_threading/browser.rs` (~300 lines)

**Reference implementation**: Follow `wasi_websocket/mod.rs` pattern

#### Task 1.2: WasiThread and WasiThreading
- [ ] Implement `WasiThread` struct with platform variants
- [ ] Implement `WasiThreading::spawn()`
- [ ] Implement `WasiThreading::spawn_with_config()`
- [ ] Implement `WasiThread::join()`
- [ ] Implement `WasiThread::is_finished()`
- [ ] Write unit tests (5 tests)

**Acceptance criteria**:
```rust
#[test]
fn test_thread_spawn_and_join() {
    let thread = WasiThreading::spawn(|| {
        assert_eq!(2 + 2, 4);
    }).unwrap();
    thread.join().unwrap();
}
```

#### Task 1.3: Native Backend (std::thread)
- [ ] Implement `native::spawn_thread()`
- [ ] Configure thread name and stack size
- [ ] Implement thread priority (platform-dependent)
- [ ] Write native-specific tests (3 tests)

**Acceptance criteria**:
- Thread spawns successfully on Linux/macOS/Windows
- Thread configuration (name, stack size) works
- Thread joins correctly

#### Task 1.4: Initial Documentation
- [ ] Write module-level rustdoc
- [ ] Document all public functions
- [ ] Add usage examples
- [ ] Create initial README

---

### Phase 2: Synchronization Primitives (Week 1)

#### Task 2.1: WasiMutex
- [ ] Implement `WasiMutex<T>` with platform variants
- [ ] Implement `WasiMutex::new()`
- [ ] Implement `WasiMutex::lock()`
- [ ] Implement `WasiMutex::try_lock()`
- [ ] Implement `WasiMutexGuard` with Deref/DerefMut
- [ ] Write unit tests (5 tests)

**Acceptance criteria**:
```rust
#[test]
fn test_mutex_lock() {
    let mutex = WasiMutex::new(0);
    {
        let mut guard = mutex.lock().unwrap();
        *guard += 1;
    }
    assert_eq!(*mutex.lock().unwrap(), 1);
}
```

#### Task 2.2: WasiRwLock
- [ ] Implement `WasiRwLock<T>` with platform variants
- [ ] Implement `WasiRwLock::new()`
- [ ] Implement `WasiRwLock::read()`
- [ ] Implement `WasiRwLock::write()`
- [ ] Implement guard types
- [ ] Write unit tests (5 tests)

#### Task 2.3: Atomics Integration
- [ ] Re-export `std::sync::atomic::*`
- [ ] Document atomic usage patterns
- [ ] Write atomic examples (3 examples)
- [ ] Verify WASM atomics support

#### Task 2.4: WasiCondvar
- [ ] Implement `WasiCondvar` with platform variants
- [ ] Implement `WasiCondvar::new()`
- [ ] Implement `WasiCondvar::wait()`
- [ ] Implement `WasiCondvar::notify_one()`
- [ ] Implement `WasiCondvar::notify_all()`
- [ ] Write unit tests (3 tests)

---

### Phase 3: Message Passing (Week 2)

#### Task 3.1: Channel API
- [ ] Implement `Sender<T>` and `Receiver<T>` types
- [ ] Implement `channel::<T>()` function
- [ ] Implement `channel_with_config()` function
- [ ] Define `ChannelConfig` struct
- [ ] Write unit tests (5 tests)

**Acceptance criteria**:
```rust
#[test]
fn test_channel_send_recv() {
    let (tx, rx) = channel::<i32>();
    WasiThreading::spawn(move || {
        tx.send(42).unwrap();
    }).unwrap();
    let value = rx.recv().unwrap();
    assert_eq!(value, 42);
}
```

#### Task 3.2: Native Backend (crossbeam)
- [ ] Implement `Sender::send()` using crossbeam
- [ ] Implement `Sender::try_send()`
- [ ] Implement `Receiver::recv()`
- [ ] Implement `Receiver::try_recv()`
- [ ] Implement `Receiver::recv_timeout()`
- [ ] Write performance benchmarks

#### Task 3.3: Serialization Layer
- [ ] Implement `ThreadMessage<T>` envelope
- [ ] Implement bincode serialization helpers
- [ ] Implement JSON serialization helpers
- [ ] Add serialization error handling
- [ ] Write serialization tests (5 tests)

#### Task 3.4: Browser Backend (postMessage)
- [ ] Implement `BrowserSender` using postMessage
- [ ] Implement `BrowserReceiver` using onmessage
- [ ] Handle message serialization/deserialization
- [ ] Implement message buffering
- [ ] Write browser-specific tests

---

### Phase 4: Thread Pool (Week 2)

#### Task 4.1: Thread Pool API
- [ ] Implement `WasiThreadPool` struct
- [ ] Implement `ThreadPoolConfig` struct
- [ ] Implement `WasiThreadPool::new()`
- [ ] Implement `WasiThreadPool::with_config()`
- [ ] Implement `WasiThreadPool::execute()`
- [ ] Write unit tests (5 tests)

**Acceptance criteria**:
```rust
#[test]
fn test_thread_pool() {
    let pool = WasiThreadPool::new().unwrap();
    let future = pool.submit(|| 2 + 2);
    let result = future.wait().unwrap();
    assert_eq!(result, 4);
}
```

#### Task 4.2: Native Backend (rayon)
- [ ] Wrap `rayon::ThreadPool`
- [ ] Configure thread pool (num_threads, stack_size)
- [ ] Implement `execute()` using rayon
- [ ] Implement `submit()` with futures
- [ ] Implement `map()` for parallel iteration
- [ ] Write performance benchmarks

#### Task 4.3: WasiFuture Implementation
- [ ] Implement `WasiFuture<T>` struct
- [ ] Implement `WasiFuture::wait()`
- [ ] Implement `WasiFuture::wait_timeout()`
- [ ] Use tokio::sync::oneshot internally
- [ ] Write unit tests (3 tests)

#### Task 4.4: WASI Backend
- [ ] Implement custom work-stealing pool
- [ ] Implement `Worker` struct
- [ ] Implement work-stealing algorithm
- [ ] Implement task distribution
- [ ] Write WASI-specific tests

---

### Phase 5: Browser Support (Week 3)

#### Task 5.1: Web Worker Pool
- [ ] Implement `WebWorkerPool` struct
- [ ] Implement `WorkerHandle` struct
- [ ] Implement worker spawning
- [ ] Implement round-robin task distribution
- [ ] Write browser-specific tests

#### Task 5.2: Worker Script (JavaScript)
- [ ] Create `workers/thread-worker.js`
- [ ] Implement message handler
- [ ] Implement WASM module initialization
- [ ] Implement task execution
- [ ] Implement result serialization
- [ ] Test in browser environment

**File to create**:
- `/workspace/portalis/agents/transpiler/workers/thread-worker.js` (~150 lines)

#### Task 5.3: Worker Loader
- [ ] Create `workers/worker-loader.js`
- [ ] Implement worker script bundling
- [ ] Implement worker URL generation
- [ ] Handle worker lifecycle
- [ ] Test in browser environment

**File to create**:
- `/workspace/portalis/agents/transpiler/workers/worker-loader.js` (~50 lines)

#### Task 5.4: Browser Integration
- [ ] Integrate `WebWorkerPool` with `WasiThreadPool`
- [ ] Implement browser-specific `spawn_worker()`
- [ ] Handle worker communication
- [ ] Implement worker termination
- [ ] Write end-to-end browser tests

#### Task 5.5: Browser Synchronization
- [ ] Implement `BrowserMutex` using Atomics.wait/notify
- [ ] Implement fallback for non-SharedArrayBuffer contexts
- [ ] Implement `BrowserCondvar`
- [ ] Test synchronization primitives in browser
- [ ] Document browser limitations

---

### Phase 6: Python Translation (Week 3)

#### Task 6.1: Translation Module
- [ ] Create `py_to_rust_threading.rs`
- [ ] Implement `get_threading_imports()`
- [ ] Define translation helper functions
- [ ] Write unit tests for each translator

**File to create**:
- `/workspace/portalis/agents/transpiler/src/py_to_rust_threading.rs` (~400 lines)

#### Task 6.2: threading.Thread Translation
- [ ] Implement `translate_thread_class()`
- [ ] Handle target function translation
- [ ] Handle args translation
- [ ] Handle Thread.start() translation
- [ ] Handle Thread.join() translation
- [ ] Write translation tests (5 tests)

**Test case**:
```python
# Input
import threading
def worker(): pass
t = threading.Thread(target=worker)
t.start()
t.join()
```
```rust
// Expected output
let thread = WasiThreading::spawn(move || worker())?;
thread.join()?;
```

#### Task 6.3: threading.Lock Translation
- [ ] Implement `translate_lock()`
- [ ] Implement `translate_lock_acquire()`
- [ ] Implement `translate_lock_release()`
- [ ] Handle context manager (with statement)
- [ ] Write translation tests (3 tests)

#### Task 6.4: queue.Queue Translation
- [ ] Implement `translate_queue()`
- [ ] Implement `translate_queue_put()`
- [ ] Implement `translate_queue_get()`
- [ ] Handle maxsize parameter
- [ ] Handle block and timeout parameters
- [ ] Write translation tests (5 tests)

#### Task 6.5: ThreadPoolExecutor Translation
- [ ] Implement `translate_thread_pool_executor()`
- [ ] Implement `translate_pool_submit()`
- [ ] Implement `translate_future_result()`
- [ ] Handle max_workers parameter
- [ ] Handle context manager
- [ ] Write translation tests (5 tests)

---

### Phase 7: Integration & Testing (Week 4)

#### Task 7.1: Feature Translator Integration
- [ ] Modify `FeatureTranslator` to detect threading
- [ ] Add threading import injection
- [ ] Add threading node translation
- [ ] Integrate with existing AST processing
- [ ] Write integration tests (5 tests)

**Files to modify**:
- `/workspace/portalis/agents/transpiler/src/feature_translator.rs`
- `/workspace/portalis/agents/transpiler/src/lib.rs`

#### Task 7.2: End-to-End Tests
- [ ] Create `tests/threading_integration_test.rs`
- [ ] Test complete Python → Rust translation
- [ ] Test thread spawning and joining
- [ ] Test mutex locking
- [ ] Test channel communication
- [ ] Test thread pool execution
- [ ] Test producer-consumer pattern
- [ ] Test work distribution

**File to create**:
- `/workspace/portalis/agents/transpiler/tests/threading_integration_test.rs` (~300 lines)

**Test count goal**: 20+ integration tests

#### Task 7.3: Platform-Specific Tests
- [ ] Write native-only tests (5 tests)
- [ ] Write WASI-only tests (5 tests)
- [ ] Write browser-only tests (5 tests)
- [ ] Set up test infrastructure for each platform
- [ ] Document test execution

#### Task 7.4: Performance Benchmarks
- [ ] Create benchmark suite
- [ ] Benchmark thread spawning
- [ ] Benchmark mutex operations
- [ ] Benchmark channel operations
- [ ] Benchmark thread pool throughput
- [ ] Document performance characteristics

**File to create**:
- `/workspace/portalis/agents/transpiler/benches/threading_benchmark.rs` (~200 lines)

#### Task 7.5: Error Handling Tests
- [ ] Test all error paths
- [ ] Test poisoned mutex recovery
- [ ] Test channel disconnection
- [ ] Test thread panic handling
- [ ] Test timeout behavior
- [ ] Document error recovery strategies

---

### Phase 8: Documentation (Week 4)

#### Task 8.1: API Documentation
- [ ] Complete all rustdoc comments
- [ ] Add usage examples to each function
- [ ] Document platform differences
- [ ] Document performance characteristics
- [ ] Document error handling
- [ ] Generate and review rustdoc output

#### Task 8.2: User Guides
- [ ] Write "Getting Started" guide
- [ ] Write "Threading Basics" guide
- [ ] Write "Advanced Patterns" guide
- [ ] Write "Browser Deployment" guide
- [ ] Write "Performance Tuning" guide
- [ ] Write "Migration from Python" guide

**Files to create**:
- `/workspace/portalis/agents/transpiler/docs/threading/getting-started.md`
- `/workspace/portalis/agents/transpiler/docs/threading/basics.md`
- `/workspace/portalis/agents/transpiler/docs/threading/advanced.md`
- `/workspace/portalis/agents/transpiler/docs/threading/browser.md`
- `/workspace/portalis/agents/transpiler/docs/threading/performance.md`
- `/workspace/portalis/agents/transpiler/docs/threading/migration.md`

#### Task 8.3: Examples
- [ ] Create basic threading example
- [ ] Create producer-consumer example
- [ ] Create thread pool example
- [ ] Create web worker example
- [ ] Create real-world use case example
- [ ] Test all examples

**Directory to create**:
- `/workspace/portalis/agents/transpiler/examples/threading/`

#### Task 8.4: Architecture Documentation
- [ ] Review architecture documents
- [ ] Update diagrams if needed
- [ ] Document implementation decisions
- [ ] Document deviations from plan
- [ ] Create implementation retrospective

---

## Quality Gates

### Before Phase Completion

Each phase must meet:
- [ ] All tasks completed
- [ ] All tests passing (unit + integration)
- [ ] Code review completed
- [ ] Documentation updated
- [ ] No clippy warnings
- [ ] Code formatted with rustfmt

### Before Final Release

- [ ] All 8 phases completed
- [ ] Full test suite passing (100+ tests)
- [ ] Performance benchmarks meet targets
- [ ] All platforms tested (native, WASI, browser)
- [ ] Documentation complete and reviewed
- [ ] Examples tested and working
- [ ] Architecture validated against implementation
- [ ] Security review completed

---

## Metrics Tracking

### Code Metrics (Target)
- Total new lines: ~2,500 lines
- Module organization: 4 modules
- Public API: ~30 functions
- Test coverage: >85%
- Documentation coverage: 100%

### Test Metrics (Target)
- Unit tests: 50+
- Integration tests: 20+
- Platform-specific tests: 15+
- Benchmarks: 5+
- Total tests: 90+

### Performance Targets

| Operation | Native | WASI | Browser |
|-----------|--------|------|---------|
| Thread spawn | <50μs | <100μs | <5ms |
| Mutex lock | <2μs | <5μs | <10μs |
| Channel send | <100ns | <200ns | <1ms |
| Pool task | <50μs | <100μs | <5ms |

---

## Risk Mitigation

### High Risk Areas

1. **Browser SharedArrayBuffer availability**
   - Risk: Many browsers restrict SAB
   - Mitigation: Implement fallback without SAB
   - Status: ⚠️ Requires testing

2. **WASI threading support**
   - Risk: wasi-threads proposal not finalized
   - Mitigation: Use rayon as fallback
   - Status: ✅ Rayon confirmed working

3. **Message serialization overhead**
   - Risk: Slow serialization in hot paths
   - Mitigation: Use bincode, benchmark early
   - Status: ⚠️ Requires benchmarking

4. **Browser worker startup time**
   - Risk: 5ms startup per worker too slow
   - Mitigation: Pre-spawn worker pool
   - Status: ⚠️ Requires testing

### Medium Risk Areas

1. **Test infrastructure for WASI**
   - Risk: Hard to test WASI in CI
   - Mitigation: Use wasmtime in tests
   - Status: ⚠️ Setup needed

2. **Cross-platform atomics**
   - Risk: Atomics behavior differs
   - Mitigation: Use std::sync::atomic abstraction
   - Status: ✅ Standard library handles this

---

## Dependencies on Existing Work

### Required Stable
- ✅ WASI filesystem (wasi_fs.rs) - Complete
- ✅ WASI WebSocket (wasi_websocket/) - Complete
- ✅ Python AST parser (python_ast.rs) - Complete
- ✅ Feature translator (feature_translator.rs) - Complete

### Optional Enhancement
- ⏳ WASI I/O improvements - Can enhance threading
- ⏳ Better error messages - Can enhance debugging

---

## Success Criteria Summary

### Functional
- ✅ All Python threading patterns translate
- ✅ Works on native, WASI, browser
- ✅ Thread-safe (Rust type system verification)
- ✅ Full test coverage

### Non-Functional
- ✅ Performance meets targets
- ✅ Documentation complete
- ✅ Production-ready error handling
- ✅ Follows established patterns

### Integration
- ✅ Integrates with FeatureTranslator
- ✅ Follows WASI filesystem pattern
- ✅ Matches WebSocket organization
- ✅ Works with existing build system

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Core Threading API | 1 week | WasiThread, WasiThreading |
| 2. Synchronization | 1 week | Mutex, RwLock, Condvar |
| 3. Message Passing | 1 week | Channels, Serialization |
| 4. Thread Pool | 1 week | WasiThreadPool, rayon integration |
| 5. Browser Support | 1 week | Web Workers, Worker pool |
| 6. Python Translation | 1 week | py_to_rust_threading.rs |
| 7. Integration & Testing | 1 week | Full test suite |
| 8. Documentation | 1 week | Docs, examples, guides |

**Total**: 8 weeks (1 developer)

---

## Next Steps

1. **Immediate**: Review architecture documents with team
2. **Week 1**: Start Phase 1 implementation
3. **Week 2**: Complete core API and synchronization
4. **Week 3**: Implement message passing and thread pool
5. **Week 4**: Add browser support
6. **Week 5**: Python translation layer
7. **Week 6**: Integration and testing
8. **Week 7-8**: Documentation and polish

---

**Document Version**: 1.0
**Last Updated**: 2025-10-04
**Status**: Ready for implementation
