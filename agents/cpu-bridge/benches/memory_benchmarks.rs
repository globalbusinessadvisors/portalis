//! Memory optimization benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "memory-opt")]
use portalis_cpu_bridge::{Arena, ArenaPool};

#[cfg(feature = "memory-opt")]
use portalis_core::acceleration::memory::{intern, ObjectPool};

#[cfg(feature = "memory-opt")]
fn bench_arena_vs_heap(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_vs_heap");

    // Benchmark heap allocation
    group.bench_function("heap_allocation_1000", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                values.push(Box::new(i));
            }
            black_box(values);
        });
    });

    // Benchmark arena allocation
    group.bench_function("arena_allocation_1000", |b| {
        b.iter(|| {
            let arena = Arena::with_capacity(8192);
            for i in 0..1000 {
                let _val = arena.alloc(i);
            }
            black_box(&arena);
        });
    });

    group.finish();
}

#[cfg(feature = "memory-opt")]
fn bench_string_interning(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_interning");

    let keywords = vec!["def", "class", "if", "else", "for", "while", "import", "from"];

    // Without interning
    group.bench_function("without_interning", |b| {
        b.iter(|| {
            let mut strings = Vec::new();
            for _i in 0..1000 {
                for &kw in &keywords {
                    strings.push(kw.to_string());
                }
            }
            black_box(strings);
        });
    });

    // With interning
    group.bench_function("with_interning", |b| {
        b.iter(|| {
            let mut strings = Vec::new();
            for _i in 0..1000 {
                for &kw in &keywords {
                    strings.push(intern(kw));
                }
            }
            black_box(strings);
        });
    });

    group.finish();
}

#[cfg(feature = "memory-opt")]
fn bench_object_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("object_pool");

    // Without pool
    group.bench_function("without_pool", |b| {
        b.iter(|| {
            for _i in 0..1000 {
                let mut vec = Vec::<i32>::with_capacity(100);
                vec.push(42);
                black_box(&vec);
            }
        });
    });

    // With pool
    let pool = ObjectPool::new(|| Vec::<i32>::with_capacity(100), 50);

    group.bench_function("with_pool", |b| {
        b.iter(|| {
            for _i in 0..1000 {
                let mut vec = pool.acquire();
                vec.push(42);
                black_box(&vec);
            }
        });
    });

    group.finish();
}

#[cfg(feature = "memory-opt")]
fn bench_arena_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_pool");

    // Create new arena each time
    group.bench_function("create_arena_each_time", |b| {
        b.iter(|| {
            let arena = Arena::with_capacity(4096);
            for i in 0..100 {
                let _val = arena.alloc(i);
            }
            black_box(&arena);
        });
    });

    // Use arena pool
    let pool = ArenaPool::new(4096, 10);

    group.bench_function("arena_pool_reuse", |b| {
        b.iter(|| {
            let arena = pool.acquire();
            for i in 0..100 {
                let _val = arena.alloc(i);
            }
            black_box(&arena);
        });
    });

    group.finish();
}

#[cfg(feature = "memory-opt")]
fn bench_batch_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_allocation");
    group.throughput(Throughput::Elements(1000));

    // Individual allocations
    group.bench_function("individual_strings", |b| {
        b.iter(|| {
            let mut strings = Vec::new();
            for i in 0..1000 {
                strings.push(format!("string_{}", i));
            }
            black_box(strings);
        });
    });

    // Arena allocation
    group.bench_function("arena_strings", |b| {
        b.iter(|| {
            let arena = Arena::with_capacity(64 * 1024);
            for i in 0..1000 {
                let s = format!("string_{}", i);
                let _allocated = arena.alloc_str(&s);
            }
            black_box(&arena);
        });
    });

    group.finish();
}

#[cfg(feature = "memory-opt")]
fn bench_cache_friendly_batch(c: &mut Criterion) {
    use portalis_core::acceleration::memory::BatchData;

    let mut group = c.benchmark_group("cache_friendly_batch");

    // Array of Structures (AoS)
    #[derive(Clone)]
    struct Item {
        source: String,
        path: String,
        result: Option<String>,
    }

    group.bench_function("array_of_structures", |b| {
        b.iter(|| {
            let mut items = Vec::new();
            for i in 0..1000 {
                items.push(Item {
                    source: format!("source_{}", i),
                    path: format!("path_{}", i),
                    result: None,
                });
            }
            // Process
            for item in &mut items {
                item.result = Some(format!("result for {}", item.source));
            }
            black_box(items);
        });
    });

    // Structure of Arrays (SoA)
    group.bench_function("structure_of_arrays", |b| {
        b.iter(|| {
            let mut batch = BatchData::with_capacity(1000);
            for i in 0..1000 {
                batch.push(format!("source_{}", i), format!("path_{}", i));
            }
            // Process
            for i in 0..batch.len() {
                batch.set_result(i, format!("result for {}", batch.sources[i]));
            }
            black_box(batch);
        });
    });

    group.finish();
}

#[cfg(feature = "memory-opt")]
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("heap", size), size, |b, &size| {
            b.iter(|| {
                let mut values = Vec::new();
                for i in 0..size {
                    values.push(Box::new(i));
                }
                black_box(values);
            });
        });

        group.bench_with_input(BenchmarkId::new("arena", size), size, |b, &size| {
            b.iter(|| {
                let arena = Arena::with_capacity(size * 8);
                for i in 0..size {
                    let _val = arena.alloc(i);
                }
                black_box(&arena);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "memory-opt")]
criterion_group!(
    memory_benches,
    bench_arena_vs_heap,
    bench_string_interning,
    bench_object_pool,
    bench_arena_pool,
    bench_batch_allocation,
    bench_cache_friendly_batch,
    bench_memory_scaling
);

#[cfg(not(feature = "memory-opt"))]
criterion_group!(memory_benches,);

criterion_main!(memory_benches);
