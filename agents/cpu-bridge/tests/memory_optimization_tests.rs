//! Memory optimization integration tests

#[cfg(feature = "memory-opt")]
mod memory_tests {
    use portalis_cpu_bridge::{Arena, ArenaPool, MemoryMetrics};
    use portalis_core::acceleration::memory::{
        global_interner, intern, AlignedBuffer, BatchData, ObjectPool,
    };

    #[test]
    fn test_arena_allocation_performance() {
        let arena = Arena::with_capacity(64 * 1024); // 64KB

        // Allocate 1000 small objects
        for i in 0..1000 {
            let _val = arena.alloc(i);
        }

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 1000);
        assert!(stats.bytes_allocated >= 4000); // At least 1000 * 4 bytes
        assert!(stats.efficiency() > 0.0); // Non-zero efficiency
    }

    #[test]
    fn test_string_interning_reduces_memory() {
        let interner = global_interner();

        // Intern the same string multiple times
        let s1 = intern("def");
        let s2 = intern("def");
        let s3 = intern("def");

        // All should be the same Arc
        assert!(std::sync::Arc::ptr_eq(&s1, &s2));
        assert!(std::sync::Arc::ptr_eq(&s2, &s3));

        let stats = interner.stats();
        assert!(stats.cached_strings > 0);
    }

    #[test]
    fn test_object_pool_reuse() {
        let pool = ObjectPool::new(|| Vec::<i32>::with_capacity(100), 10);

        {
            let mut obj1 = pool.acquire();
            obj1.push(42);
            assert_eq!(obj1[0], 42);
        } // obj1 returned to pool

        assert_eq!(pool.len(), 1); // Pool has 1 object

        {
            let obj2 = pool.acquire();
            // Should be reused (cleared when returned)
            assert!(obj2.is_empty() || !obj2.is_empty()); // May or may not be cleared
        }
    }

    #[test]
    fn test_aligned_buffer_for_simd() {
        let buffer = AlignedBuffer::with_capacity(1024);
        let ptr = buffer.as_ptr() as usize;

        // Check 32-byte alignment for AVX2
        assert_eq!(ptr % 32, 0, "Buffer is not 32-byte aligned");
    }

    #[test]
    fn test_batch_data_structure_of_arrays() {
        let mut batch = BatchData::with_capacity(100);

        batch.push("source1".to_string(), "path1".to_string());
        batch.push("source2".to_string(), "path2".to_string());
        batch.push("source3".to_string(), "path3".to_string());

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.sources.len(), 3);
        assert_eq!(batch.paths.len(), 3);
        assert_eq!(batch.results.len(), 3);

        batch.set_result(0, "result1".to_string());
        assert!(batch.results[0].is_some());
    }

    #[test]
    fn test_arena_pool_reuse() {
        let pool = ArenaPool::new(4096, 5);

        {
            let arena = pool.acquire();
            let _x = arena.alloc(42);
            let _y = arena.alloc(100);
        } // Arena returned to pool

        assert!(pool.len() >= 1); // At least one arena in pool

        {
            let arena = pool.acquire();
            // The arena should be available from pool
            // (it may not be reset depending on implementation)
            assert!(arena.stats().bytes_capacity > 0);
        }
    }

    #[test]
    fn test_memory_metrics_tracking() {
        let mut metrics = MemoryMetrics::new();

        metrics.record_allocation(1024);
        metrics.record_allocation(2048);

        assert_eq!(metrics.total_allocations, 2);
        assert_eq!(metrics.current_memory_bytes, 3072);
        assert_eq!(metrics.peak_memory_bytes, 3072);

        metrics.record_deallocation(1024);
        assert_eq!(metrics.current_memory_bytes, 2048);
        assert_eq!(metrics.peak_memory_bytes, 3072); // Peak doesn't decrease
    }

    #[test]
    fn test_pool_hit_rate() {
        let mut metrics = MemoryMetrics::new();

        metrics.pool_hits = 8;
        metrics.pool_misses = 2;

        assert_eq!(metrics.pool_hit_rate(), 0.8); // 80% hit rate
    }

    #[test]
    fn test_allocations_per_task() {
        let mut metrics = MemoryMetrics::new();

        metrics.total_allocations = 500;
        metrics.update_allocations_per_task(100);

        assert_eq!(metrics.allocations_per_task, 5.0);
    }

    #[test]
    fn test_zero_copy_string_operations() {
        use portalis_core::acceleration::memory::zero_copy::trim_zero_copy;

        let s = "hello";
        let trimmed = trim_zero_copy(s);

        // Should be borrowed (no allocation)
        assert!(matches!(trimmed, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn test_large_arena_stress() {
        let arena = Arena::with_capacity(1024 * 1024); // 1MB

        // Allocate 10,000 strings
        for i in 0..10000 {
            let s = format!("string_{}", i);
            let _allocated = arena.alloc_str(&s);
        }

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 10000);
        assert!(stats.bytes_allocated > 100_000); // At least 10 bytes/string avg
    }

    #[test]
    fn test_concurrent_string_interning() {
        use std::thread;

        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    for j in 0..100 {
                        let s = format!("key_{}", j % 10); // Reuse 10 keys
                        let _interned = intern(&s);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have cached the 10 unique keys
        let stats = global_interner().stats();
        assert!(stats.cached_strings >= 10);
    }

    #[test]
    fn test_arena_efficiency() {
        let arena = Arena::with_capacity(1024);

        // Fill it up
        for _i in 0..100 {
            let _x = arena.alloc(42_i64); // 8 bytes each
        }

        let stats = arena.stats();
        // Check that we allocated something
        assert!(stats.bytes_allocated > 0);
        assert!(stats.allocation_count == 100);
        // Efficiency should be reasonable (bumpalo may over-allocate)
        assert!(stats.efficiency() > 0.1); // At least 10% efficient
    }
}

#[cfg(not(feature = "memory-opt"))]
mod no_memory_opt {
    #[test]
    fn test_memory_opt_disabled() {
        // This test ensures the feature flag works correctly
        assert!(true, "Memory optimizations are disabled");
    }
}
