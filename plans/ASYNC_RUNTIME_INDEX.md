# Async Runtime Architecture - Document Index

**Project**: PORTALIS Async Runtime Integration
**Status**: Architecture Complete - Ready for Implementation
**Date**: 2025-10-04
**System Architect**: AI Assistant (Claude)
**Complexity**: Medium (follows proven patterns)
**Implementation Time**: 3 weeks

---

## Quick Navigation

### For Implementers
👉 **START HERE**: [Implementation Summary](ASYNC_RUNTIME_IMPLEMENTATION_SUMMARY.md)

### For Architects/Reviewers
👉 **FULL DESIGN**: [Complete Architecture](ASYNC_RUNTIME_ARCHITECTURE.md)

### For Visual Learners
👉 **DIAGRAMS**: [Architecture Diagrams](ASYNC_RUNTIME_ARCHITECTURE_DIAGRAM.md)

---

## Document Catalog

### 1. Complete Architecture Document
**File**: `ASYNC_RUNTIME_ARCHITECTURE.md`
**Size**: 18,000+ words
**Purpose**: Comprehensive technical specification

**Contents**:
- Context & Requirements
- Architecture Overview
- Module Structure
- Platform Abstraction Strategy
- Runtime Management
- Async Primitives Design
- Synchronization Primitives
- Python asyncio Translation Layer
- Integration with Existing Code
- Error Handling Strategy
- Performance Considerations
- Implementation Roadmap (3 weeks)
- Complete API Reference

**Target Audience**: System architects, senior engineers, technical reviewers

**Key Sections**:
- [Architecture Overview](ASYNC_RUNTIME_ARCHITECTURE.md#architecture-overview)
- [Runtime Management](ASYNC_RUNTIME_ARCHITECTURE.md#runtime-management)
- [Python Translation Layer](ASYNC_RUNTIME_ARCHITECTURE.md#python-asyncio-translation-layer)
- [API Reference](ASYNC_RUNTIME_ARCHITECTURE.md#api-reference)

---

### 2. Visual Architecture Diagrams
**File**: `ASYNC_RUNTIME_ARCHITECTURE_DIAGRAM.md`
**Size**: ASCII diagrams and charts
**Purpose**: Visual representation of architecture

**Contents**:
- Component Hierarchy Diagram
- Module Structure Diagram
- Data Flow Diagrams (Native, Browser, WASI)
- Integration Points Diagram
- Python → Rust Translation Map
- Error Handling Flow
- Task Lifecycle Diagram
- Synchronization Primitive Relationships
- Compilation Flow
- Performance Characteristics Chart

**Target Audience**: Visual learners, presentation materials, onboarding

**Key Diagrams**:
- [Component Hierarchy](ASYNC_RUNTIME_ARCHITECTURE_DIAGRAM.md#component-hierarchy)
- [Module Structure](ASYNC_RUNTIME_ARCHITECTURE_DIAGRAM.md#module-structure-diagram)
- [Data Flow (Native)](ASYNC_RUNTIME_ARCHITECTURE_DIAGRAM.md#native-platform-tokio)
- [Python Translation Map](ASYNC_RUNTIME_ARCHITECTURE_DIAGRAM.md#python-asyncio--rust-translation-map)

---

### 3. Implementation Summary
**File**: `ASYNC_RUNTIME_IMPLEMENTATION_SUMMARY.md`
**Size**: Concise executive summary
**Purpose**: Quick reference for implementers

**Contents**:
- Executive Overview
- What Was Delivered
- Architecture Highlights
- Key Architectural Decisions
- Implementation Roadmap (detailed)
- API Surface Summary
- Testing Strategy
- Risk Assessment
- Success Criteria
- Next Steps

**Target Audience**: Implementation teams, project managers, stakeholders

**Key Sections**:
- [Implementation Roadmap](ASYNC_RUNTIME_IMPLEMENTATION_SUMMARY.md#implementation-roadmap)
- [API Surface Summary](ASYNC_RUNTIME_IMPLEMENTATION_SUMMARY.md#api-surface-summary)
- [Risk Assessment](ASYNC_RUNTIME_IMPLEMENTATION_SUMMARY.md#risk-assessment)
- [Success Criteria](ASYNC_RUNTIME_IMPLEMENTATION_SUMMARY.md#success-criteria)

---

## Architecture at a Glance

### Module Structure
```
wasi_async/
├── mod.rs              # Public API, platform selection
├── runtime.rs          # Runtime management
├── task.rs             # Task spawning and handles
├── primitives.rs       # Sleep, timeout, select, interval
├── sync.rs             # AsyncMutex, channels, Notify
├── native.rs           # Tokio implementation
├── browser.rs          # wasm-bindgen-futures implementation
├── wasi_impl.rs        # WASI stub/limited implementation
├── error.rs            # AsyncError types
└── README.md           # Documentation
```

### Platform Support
- ✅ **Native**: Full tokio runtime with multi-threading
- ✅ **Browser**: wasm-bindgen-futures with event loop integration
- ⚠️ **WASI**: Stubs with graceful degradation

### Key Features
- ✅ Python asyncio parity
- ✅ Zero-cost abstractions
- ✅ Seamless integration with existing code
- ✅ Comprehensive error handling
- ✅ Platform-aware compilation

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1, Days 1-5)
**Files**: `runtime.rs`, `task.rs`, `primitives.rs` (basic), `error.rs`
**Focus**: Basic runtime and task spawning
**Deliverable**: Native async runtime working

### Phase 2: Synchronization Primitives (Week 1-2, Days 5-10)
**Files**: `sync.rs` (complete)
**Focus**: AsyncMutex, channels, Notify, etc.
**Deliverable**: All sync primitives implemented

### Phase 3: Python Translation Layer (Week 2, Days 8-12)
**Files**: `python_to_rust.rs`, `stdlib_mapper.rs` extensions
**Focus**: asyncio → Rust translation
**Deliverable**: Python async code translates correctly

### Phase 4: Integration & Testing (Week 2-3, Days 11-15)
**Files**: Integration tests, benchmarks
**Focus**: Integration with wasi_fetch, wasi_websocket
**Deliverable**: End-to-end tests passing

### Phase 5: WASM Optimization (Week 3, Days 13-15)
**Files**: `browser.rs` optimization, WASM examples
**Focus**: Browser and WASI refinement
**Deliverable**: WASM builds working

---

## Quick Reference

### API Patterns

**Runtime**:
```rust
WasiRuntime::run(main_async()).expect("Runtime error");
```

**Task Spawning**:
```rust
let handle = spawn(async { 42 });
let result = handle.join().await?;
```

**Sleep**:
```rust
WasiAsync::sleep(Duration::from_secs(1)).await;
```

**Timeout**:
```rust
let result = WasiAsync::timeout(Duration::from_secs(5), slow()).await?;
```

**AsyncMutex**:
```rust
let mutex = AsyncMutex::new(0);
let mut guard = mutex.lock().await;
*guard += 1;
```

**Channels**:
```rust
let (tx, rx) = AsyncMpscChannel::unbounded();
tx.send(42).await?;
let value = rx.recv().await;
```

### Python → Rust Translation

| Python | Rust |
|--------|------|
| `asyncio.run(main())` | `WasiRuntime::run(main_async())` |
| `asyncio.create_task(worker())` | `spawn(worker())` |
| `await asyncio.sleep(1)` | `WasiAsync::sleep(Duration::from_secs(1)).await` |
| `await asyncio.gather(*tasks)` | `join_all(tasks).await` |
| `asyncio.Queue()` | `AsyncMpscChannel::unbounded()` |

---

## Integration with Existing Code

### Existing Modules (No Changes Required)

**wasi_fetch.rs**:
```rust
// Already async - works with new runtime!
pub async fn get(url: impl Into<String>) -> Result<Response> {
    // ... existing code unchanged
}
```

**wasi_websocket/mod.rs**:
```rust
// Already async - works with new runtime!
pub async fn connect(config: WebSocketConfig) -> Result<Self> {
    // ... existing code unchanged
}
```

### New Modules (To Be Created)

**python_to_rust.rs** (extend):
```rust
fn translate_async_function(&mut self, func: &PyFunctionDef) -> Result<String>
fn translate_await(&mut self, expr: &PyExpr) -> Result<String>
fn translate_asyncio_call(&mut self, call: &PyCall) -> Result<String>
```

**stdlib_mapper.rs** (extend):
```rust
"asyncio" => map_asyncio_module(),
```

---

## Decision Log

### Key Architectural Decisions

1. **Runtime Strategy**: Hybrid (global singleton + explicit runtime)
   - Rationale: Balance convenience for generated code with flexibility for advanced use

2. **Platform Abstraction**: Compile-time selection via `cfg`
   - Rationale: Zero-cost abstractions, type safety, dead code elimination

3. **Error Handling**: New AsyncError type
   - Rationale: Clear async-specific errors, easy conversion to existing types

4. **Cancellation**: AbortHandle for native, no-op for WASM
   - Rationale: Platform limitations, graceful degradation

5. **WASI Support**: Stubs initially, expand as platform matures
   - Rationale: WASI async support is evolving, avoid blocking on uncertain timeline

---

## Success Metrics

### Implementation Success
- ✅ All 5 phases completed on schedule (3 weeks)
- ✅ 80%+ test coverage across all modules
- ✅ Builds succeed on all platforms
- ✅ All existing async code works unchanged

### Translation Success
- ✅ Python asyncio code translates correctly
- ✅ 20+ async test cases passing
- ✅ asyncio.* calls map to Rust equivalents

### Performance Success
- ✅ Native: Task spawn < 100ns
- ✅ Browser: Builds and runs in browser
- ✅ Memory overhead < 100 bytes per task

---

## Resources

### Related Documentation
- [WASI Threading Implementation](../agents/transpiler/src/wasi_threading/README.md) - Similar pattern
- [WASI Fetch Implementation](../agents/transpiler/src/wasi_fetch.rs) - Existing async code
- [WASI WebSocket Implementation](../agents/transpiler/src/wasi_websocket/mod.rs) - Existing async code

### External References
- [Tokio Documentation](https://tokio.rs/tokio/tutorial)
- [wasm-bindgen-futures](https://rustwasm.github.io/wasm-bindgen/api/wasm_bindgen_futures/)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

### Internal Context
- Current codebase: `/workspace/portalis/agents/transpiler/`
- Existing async usage: 411 occurrences across 57 files
- Cargo.toml: tokio 1.35 with "full" features

---

## FAQ

### Q: Will this break existing code?
**A**: No. All existing async code (wasi_fetch, wasi_websocket) will work unchanged. The new runtime provides the underlying infrastructure they already depend on.

### Q: Why not use async-std instead of tokio?
**A**: tokio is already in the workspace dependencies (version 1.35 with "full" features). It's the de facto standard for Rust async, has better WASM support via tokio-wasi, and provides all needed features.

### Q: What if WASI async support improves?
**A**: The architecture is designed for easy extension. Simply update `wasi_impl.rs` with real implementations as WASI matures. The API remains unchanged.

### Q: Can I use multiple runtimes?
**A**: Yes, with explicit runtime mode. The global singleton is for convenience in generated code, but you can create and manage multiple WasiRuntime instances if needed.

### Q: How does this affect WASM bundle size?
**A**: Minimal impact. Platform-specific code is eliminated at compile time. Browser builds only include wasm-bindgen-futures, not tokio.

---

## Contact / Support

For questions or clarifications during implementation:

1. **Architecture questions**: Review [ASYNC_RUNTIME_ARCHITECTURE.md](ASYNC_RUNTIME_ARCHITECTURE.md)
2. **Visual understanding**: See [ASYNC_RUNTIME_ARCHITECTURE_DIAGRAM.md](ASYNC_RUNTIME_ARCHITECTURE_DIAGRAM.md)
3. **Implementation guidance**: See [ASYNC_RUNTIME_IMPLEMENTATION_SUMMARY.md](ASYNC_RUNTIME_IMPLEMENTATION_SUMMARY.md)
4. **Pattern examples**: Review existing code in `wasi_threading/` and `wasi_websocket/`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-04 | Initial architecture complete |

---

## Approval Status

- ✅ Architecture Design: COMPLETE
- ✅ Documentation: COMPLETE
- ✅ Diagrams: COMPLETE
- ⏳ Implementation: READY TO START
- ⏳ Testing: PENDING
- ⏳ Deployment: PENDING

---

**Status**: ✅ ARCHITECTURE COMPLETE - READY FOR IMPLEMENTATION
**Confidence Level**: HIGH
**Risk Level**: LOW
**Estimated Effort**: 3 weeks (15 working days)

**Next Action**: Begin Phase 1 Implementation (Core Infrastructure)
