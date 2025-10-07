# Memory Optimization Quick Reference

**Quick access guide for developers implementing memory optimizations**

---

## 📋 Key Documents

| Document | Purpose | Audience |
|----------|---------|----------|
| [Executive Summary](MEMORY_OPTIMIZATION_EXECUTIVE_SUMMARY.md) | High-level overview, ROI, timeline | Leadership, PMs |
| [Full Architecture](MEMORY_OPTIMIZATION_ARCHITECTURE.md) | Complete technical specification | Engineers |
| This Document | Quick reference for implementation | Developers |

---

## 🎯 Performance Targets at a Glance

| Metric | Before | After | Goal |
|--------|--------|-------|------|
| Allocations/task | 500 | <150 | **70% reduction** |
| Cache hit rate | 60% | >85% | **+42%** |
| Memory bandwidth | 40% | >70% | **+75%** |
| Copies/task | 20 | <5 | **75% reduction** |
| Peak memory | 100% | 70% | **30% reduction** |

**Overall: 2-5x speedup on memory-bound workloads**

---

## 🏗️ Architecture Overview

```
Memory Layer
├── Arena Allocation (AST nodes)
├── Object Pools (String/Vec/HashMap)
├── String Interning (identifiers)
├── SoA Data Structures (cache-friendly)
└── Zero-Copy Operations (Cow<>)

Integration Points
├── CPU Bridge (pooling + arenas)
├── Transpiler (arena ASTs)
├── SIMD (aligned buffers)
└── Wassette (WASM memory pools)
```

---

## 🚀 Quick Start: Using Memory Optimizations

### 1. Object Pools

```rust
use portalis_core::memory::StandardPools;

// Create pools
let pools = Arc::new(StandardPools::new());

// Use pooled objects
let mut buffer = pools.strings.acquire();
buffer.push_str("data");
// Automatically returned on drop
```

### 2. Arena Allocation

```rust
use portalis_core::memory::AstArena;

// Create arena
let mut arena = AstArena::new();

// Allocate AST nodes
let node = arena.alloc(AstNode {
    kind: NodeKind::Function,
    children: arena.alloc_slice(&child_nodes),
    // ...
});

// Reset when done
arena.reset();
```

### 3. String Interning

```rust
use portalis_core::memory::StringInterner;

// Create interner with Python keywords
let interner = StringInterner::with_python_keywords();

// Intern strings
let id1 = interner.intern("data");
let id2 = interner.intern("data");  // No allocation

// Fast equality (pointer comparison)
assert!(Arc::ptr_eq(&id1, &id2));
```

### 4. Structure-of-Arrays

```rust
use portalis_core::memory::TranslationBatch;

// Create batch
let mut batch = TranslationBatch::with_capacity(100);

// Add tasks
batch.add_task(source, path, config);

// Process with excellent cache locality
batch.process_sources(|source| {
    // ... process source
});
```

### 5. Zero-Copy Operations

```rust
use std::borrow::Cow;

// Borrow when possible
let input = TranslationInput::Borrowed(source);

// Copy only when modifying
let mut config = TranslationConfig::borrowed(&shared);
config.modify(|c| c.option = value);  // Copies here
```

---

## 📊 Implementation Priority

### Phase 1: High ROI (Weeks 1-4)

✅ **IMPLEMENT FIRST** - Biggest impact, lowest effort

```rust
// 1. Object Pools (Week 1) - 10x faster allocation
let pools = StandardPools::new();

// 2. Arena Allocation (Weeks 2-3) - 99.95% less overhead
let arena = AstArena::new();

// 3. String Interning (Week 4) - 62% memory savings
let interner = StringInterner::with_python_keywords();
```

### Phase 2: Cache Optimization (Weeks 5-8)

```rust
// 4. SoA Data Structures (Weeks 5-6) - 4x faster access
let batch = TranslationBatch::with_capacity(100);

// 5. Zero-Copy (Weeks 7-8) - 75% fewer copies
let input = TranslationInput::Borrowed(source);
```

### Phase 3: SIMD Integration (Weeks 9-12)

```rust
// 6. Aligned Buffers (Week 9-10) - SIMD optimization
let buffer = AlignedBuffer::with_capacity(1024);

// 7. Memory Prefetching (Week 11) - Hide latency
prefetch_read(future_task as *const T);

// 8. WASM Pooling (Week 12) - 1ms saved per instance
let pool = WasmMemoryPool::new(config, 100);
```

---

## 🧪 Testing Checklist

### Unit Tests

```bash
# Test object pools
cargo test --lib object_pool

# Test string interning
cargo test --lib string_interner

# Test arena allocation
cargo test --lib ast_arena

# Memory leak detection
valgrind --leak-check=full target/debug/tests
```

### Benchmarks

```bash
# Compare with/without pools
cargo bench --bench pool_comparison

# Cache locality test
cargo bench --bench cache_locality

# Memory usage profiling
heaptrack target/release/portalis convert test.py
```

### Integration Tests

```bash
# Full CPU bridge integration
cargo test --test integration_memory

# Cross-platform validation
cargo test --all-features --target x86_64-unknown-linux-gnu
cargo test --all-features --target x86_64-apple-darwin
cargo test --all-features --target x86_64-pc-windows-msvc
```

---

## 🔧 Configuration Reference

### CLI

```bash
# Enable all optimizations (default)
portalis convert script.py --optimize-memory

# Profile memory usage
portalis convert script.py --memory-profile

# Disable pooling (debugging)
portalis convert script.py --no-memory-pools

# Set pool sizes
portalis convert script.py --string-pool-size 1000
```

### Config File

```toml
# portalis.toml

[memory]
enable_pools = true
enable_string_interning = true
enable_arena_allocation = true

[memory.pools]
string_pool_size = 1000
vec_pool_size = 500
hashmap_pool_size = 100
```

### Programmatic

```rust
use portalis_core::memory::MemoryConfig;

let config = MemoryConfig {
    enable_pools: true,
    enable_string_interning: true,
    pool_sizes: PoolSizes {
        strings: 1000,
        vecs: 500,
        hashmaps: 100,
    },
    ..Default::default()
};

let bridge = CpuBridge::with_memory_config(
    CpuConfig::auto_detect(),
    config,
);
```

---

## 📈 Monitoring & Metrics

### Get Metrics

```rust
// Get memory metrics
let metrics = bridge.memory_metrics();

println!("Pool hit rate: {:.2}%", metrics.pool_hit_rate * 100.0);
println!("Strings interned: {}", metrics.strings_interned);
println!("Bytes saved: {}", metrics.bytes_saved);
println!("Cache hit rate: {:.2}%", metrics.cache_hit_rate * 100.0);
```

### Prometheus Export

```rust
// Export for monitoring
let prometheus_metrics = bridge.export_memory_metrics();

// Example metrics:
// portalis_memory_pool_hit_rate 0.92
// portalis_memory_cache_hit_rate 0.87
// portalis_memory_bytes_saved 1048576
```

### Profiling Tools

```bash
# Cache analysis with perf
perf stat -e cache-references,cache-misses \
    target/release/portalis convert test.py

# Memory profiling with Valgrind
valgrind --tool=massif \
    --massif-out-file=massif.out \
    target/release/portalis convert test.py

# Allocation tracking with heaptrack
heaptrack target/release/portalis convert test.py
heaptrack --analyze heaptrack.*.gz
```

---

## 🐛 Common Issues & Solutions

### Issue: Pool Exhaustion

**Symptom:** Warnings about pool size exceeded

**Solution:**
```rust
// Increase pool size in config
[memory.pools]
string_pool_size = 2000  // Increase from default 1000
```

### Issue: Arena Memory Leaks

**Symptom:** Memory usage grows over time

**Solution:**
```rust
// Ensure arena is reset after each translation
impl TranslationContext {
    pub fn finish_translation(&mut self) {
        if let Some(mut arena) = self.current_arena.take() {
            arena.reset();  // ← CRITICAL
        }
    }
}
```

### Issue: Cache Misses Still High

**Symptom:** Cache hit rate <80% after SoA implementation

**Solution:**
```bash
# Profile access patterns
perf record -e cache-misses target/release/portalis convert test.py
perf report

# Check alignment
assert_eq!(data.as_ptr() as usize % 64, 0);
```

### Issue: Zero-Copy Not Working

**Symptom:** Unnecessary clones in profiler

**Solution:**
```rust
// Use borrowing instead of cloning
let input = TranslationInput::Borrowed(source);  // ✅ Good
let input = TranslationInput::Owned(source.to_string());  // ❌ Bad
```

---

## 📚 Code Examples

### Complete Translation Pipeline with Memory Optimizations

```rust
use portalis_core::memory::{
    AstArena, StandardPools, StringInterner, TranslationBatch,
};

pub struct OptimizedTranslator {
    arena: AstArena,
    pools: Arc<StandardPools>,
    interner: Arc<StringInterner>,
}

impl OptimizedTranslator {
    pub fn new() -> Self {
        Self {
            arena: AstArena::new(),
            pools: Arc::new(StandardPools::new()),
            interner: Arc::new(StringInterner::with_python_keywords()),
        }
    }

    pub fn translate_batch(
        &mut self,
        sources: Vec<String>,
    ) -> Result<Vec<TranslationOutput>> {
        // 1. Create SoA batch (cache-friendly)
        let mut batch = TranslationBatch::with_capacity(sources.len());
        for source in sources {
            batch.add_task(source, PathBuf::default(), Config::default());
        }

        // 2. Parse with arena allocation
        let asts: Vec<_> = batch
            .sources
            .iter()
            .map(|source| {
                self.parse_with_arena(source, &self.arena)
            })
            .collect::<Result<_>>()?;

        // 3. Translate with pooled objects and interning
        let results: Vec<_> = asts
            .par_iter()
            .map(|ast| {
                // Use pooled String for output
                let mut output = self.pools.strings.acquire();

                // Intern identifiers
                for ident in ast.identifiers() {
                    let interned = self.interner.intern(ident);
                    self.process_identifier(&interned);
                }

                // Generate code into pooled buffer
                self.codegen(ast, &mut output)?;

                Ok(output.clone())
            })
            .collect::<Result<_>>()?;

        // 4. Reset arena for next batch
        self.arena.reset();

        Ok(results)
    }

    fn parse_with_arena(
        &self,
        source: &str,
        arena: &AstArena,
    ) -> Result<&AstNode> {
        // Parse AST with zero-copy views
        let root = arena.alloc(AstNode {
            kind: NodeKind::Module,
            children: arena.alloc_slice(&[]),
            source: source,  // Borrow, don't copy
        });

        Ok(root)
    }

    fn codegen(
        &self,
        ast: &AstNode,
        output: &mut String,
    ) -> Result<()> {
        // Generate code without unnecessary allocations
        // Use output buffer from pool
        output.push_str("// Generated code\n");
        // ...
        Ok(())
    }
}
```

### CPU Bridge Integration

```rust
impl CpuBridge {
    pub fn parallel_translate_optimized<T, O, F>(
        &self,
        tasks: Vec<T>,
        translate_fn: F,
    ) -> Result<Vec<O>>
    where
        T: Send + Sync,
        O: Send,
        F: Fn(&T, &StandardPools, &StringInterner) -> Result<O>
            + Send + Sync,
    {
        let pools = self.pools.clone();
        let interner = self.interner.clone();

        self.thread_pool.install(|| {
            tasks
                .par_iter()
                .map(|task| {
                    // Each thread gets access to shared pools/interner
                    translate_fn(task, &pools, &interner)
                })
                .collect()
        })
    }
}
```

---

## 🎓 Best Practices

### DO ✅

```rust
// Use pooled objects
let mut buffer = pools.strings.acquire();

// Use arena for AST
let node = arena.alloc(AstNode { /* ... */ });

// Intern common strings
let ident = interner.intern("data");

// Use SoA for batches
let batch = TranslationBatch::with_capacity(100);

// Borrow when possible
let input = TranslationInput::Borrowed(source);
```

### DON'T ❌

```rust
// Allocate in hot paths
let buffer = String::new();  // Use pool instead

// Individual AST node allocations
let node = Box::new(AstNode { /* ... */ });  // Use arena

// Repeated string allocations
let id = "data".to_string();  // Intern instead

// Array-of-Structs for large batches
struct Task { source: String, /* ... */ }  // Use SoA

// Unnecessary clones
let copy = source.clone();  // Use Cow<> instead
```

---

## 📞 Getting Help

### Documentation
- [Full Architecture](MEMORY_OPTIMIZATION_ARCHITECTURE.md)
- [Executive Summary](MEMORY_OPTIMIZATION_EXECUTIVE_SUMMARY.md)
- API Docs: `cargo doc --open`

### Team Contacts
- **Memory Optimization Team:** team@portalis.ai
- **CPU Bridge Maintainers:** See `agents/cpu-bridge/README.md`
- **Core Team:** See `core/README.md`

### Resources
- Rust Performance Book: https://nnethercote.github.io/perf-book/
- Bumpalo Docs: https://docs.rs/bumpalo/
- DashMap Docs: https://docs.rs/dashmap/

---

## 🔄 Quick Reference Card

**Print this section for your desk!**

### Memory Optimization Toolkit

| Tool | When to Use | Performance Gain |
|------|-------------|------------------|
| `ObjectPool<T>` | Frequent allocations | 10x faster |
| `AstArena` | AST construction | 99.95% less overhead |
| `StringInterner` | Repeated identifiers | 62% memory savings |
| `TranslationBatch` | Large batches | 4x faster access |
| `Cow<>` | Read-heavy operations | 75% fewer copies |
| `AlignedBuffer` | SIMD operations | 10-20% faster |

### Performance Targets

- Allocations/task: <150 (was 500)
- Cache hit rate: >85% (was 60%)
- Memory bandwidth: >70% (was 40%)
- Memory copies: <5 (was 20)

### Testing Commands

```bash
# Run all tests
cargo test --all-features

# Run benchmarks
cargo bench --bench memory_optimization_suite

# Check for leaks
valgrind --leak-check=full target/debug/tests

# Profile cache
perf stat -e cache-misses target/release/portalis convert test.py
```

---

**Last Updated:** 2025-10-07
**Version:** 1.0
**Status:** Ready for Use
