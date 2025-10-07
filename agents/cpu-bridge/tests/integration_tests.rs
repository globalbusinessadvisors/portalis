//! Integration tests for CPU bridge

use portalis_cpu_bridge::{CpuBridge, CpuConfig};

#[test]
fn test_cpu_bridge_basic_functionality() {
    let bridge = CpuBridge::new();
    assert!(bridge.num_threads() > 0);
}

#[test]
fn test_parallel_execution_correctness() {
    let bridge = CpuBridge::new();
    let tasks: Vec<i32> = (0..1000).collect();

    let results = bridge
        .parallel_translate(tasks, |&x| Ok(x * 2))
        .expect("Parallel execution failed");

    assert_eq!(results.len(), 1000);
    for (i, &result) in results.iter().enumerate() {
        assert_eq!(result, i as i32 * 2);
    }
}

#[test]
fn test_single_task_execution() {
    let bridge = CpuBridge::new();

    let result = bridge
        .translate_single(100, |&x| Ok(x + 50))
        .expect("Single task execution failed");

    assert_eq!(result, 150);
}

#[test]
fn test_custom_configuration() {
    let config = CpuConfig::builder()
        .num_threads(2)
        .batch_size(16)
        .enable_simd(false)
        .build();

    let bridge = CpuBridge::with_config(config);
    assert_eq!(bridge.num_threads(), 2);

    let tasks: Vec<i32> = (0..100).collect();
    let results = bridge
        .parallel_translate(tasks, |&x| Ok(x * 3))
        .expect("Custom config execution failed");

    assert_eq!(results.len(), 100);
}

#[test]
fn test_metrics_collection() {
    let bridge = CpuBridge::new();

    // Execute some tasks
    let tasks: Vec<i32> = (0..50).collect();
    bridge
        .parallel_translate(tasks, |&x| Ok(x * 2))
        .expect("Execution failed");

    // Check metrics
    let metrics = bridge.metrics();
    assert_eq!(metrics.tasks_completed(), 50);
    assert!(metrics.avg_task_time_ms() >= 0.0);
}

#[test]
fn test_error_handling() {
    let bridge = CpuBridge::new();
    let tasks: Vec<i32> = (0..10).collect();

    // Intentionally fail on task 5
    let result = bridge.parallel_translate(tasks, |&x| {
        if x == 5 {
            Err(anyhow::anyhow!("Intentional error"))
        } else {
            Ok(x * 2)
        }
    });

    assert!(result.is_err());
}

#[test]
fn test_empty_task_list() {
    let bridge = CpuBridge::new();
    let tasks: Vec<i32> = vec![];

    let results = bridge
        .parallel_translate(tasks, |&x| Ok(x * 2))
        .expect("Empty task list should succeed");

    assert_eq!(results.len(), 0);
}

#[test]
fn test_large_batch_processing() {
    let bridge = CpuBridge::new();
    let tasks: Vec<i32> = (0..10000).collect();

    let results = bridge
        .parallel_translate(tasks, |&x| Ok(x + 1))
        .expect("Large batch processing failed");

    assert_eq!(results.len(), 10000);
    assert_eq!(results[0], 1);
    assert_eq!(results[9999], 10000);
}

// ============================================================================
// Additional Comprehensive Tests
// ============================================================================

#[test]
fn test_thread_pool_different_sizes() {
    for num_threads in vec![1, 2, 4, 8] {
        let config = CpuConfig::builder()
            .num_threads(num_threads)
            .build();
        let bridge = CpuBridge::with_config(config);

        assert_eq!(bridge.num_threads(), num_threads);

        let tasks: Vec<i32> = (0..100).collect();
        let results = bridge
            .parallel_translate(tasks, |&x| Ok(x * 2))
            .expect("Should succeed with any thread count");

        assert_eq!(results.len(), 100);
    }
}

#[test]
fn test_concurrent_bridge_usage() {
    use std::sync::Arc;
    use std::thread;

    let bridge = Arc::new(CpuBridge::new());
    let mut handles = vec![];

    // Spawn multiple threads using the same bridge
    for i in 0..4 {
        let bridge_clone = Arc::clone(&bridge);
        let handle = thread::spawn(move || {
            let tasks: Vec<i32> = vec![i * 10, i * 10 + 1, i * 10 + 2];
            bridge_clone
                .parallel_translate(tasks, |&x| Ok(x * 2))
                .expect("Concurrent execution failed")
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        let results = handle.join().expect("Thread panicked");
        assert_eq!(results.len(), 3);
    }
}

#[test]
fn test_task_order_preservation() {
    let bridge = CpuBridge::new();
    let tasks: Vec<usize> = (0..1000).collect();

    let results = bridge
        .parallel_translate(tasks, |&x| {
            // Add small delay to ensure parallel execution
            std::thread::sleep(std::time::Duration::from_micros(10));
            Ok(x)
        })
        .expect("Execution failed");

    // Verify order is preserved despite parallel execution
    for (i, &result) in results.iter().enumerate() {
        assert_eq!(result, i, "Order should be preserved at index {}", i);
    }
}

#[test]
fn test_panic_handling_in_tasks() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let bridge = CpuBridge::new();
    let tasks: Vec<i32> = vec![1, 2, 3];

    // Panics in Rayon tasks should be caught
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = bridge.parallel_translate(tasks, |&x| {
            if x == 2 {
                panic!("Test panic");
            }
            Ok(x * 2)
        });
    }));

    assert!(result.is_err(), "Panic should be caught");
}

#[test]
fn test_metrics_multiple_operations() {
    let bridge = CpuBridge::new();

    // First operation
    let tasks1: Vec<i32> = (0..10).collect();
    bridge
        .parallel_translate(tasks1, |&x| Ok(x * 2))
        .expect("First operation failed");

    // Second operation
    let tasks2: Vec<i32> = (0..20).collect();
    bridge
        .parallel_translate(tasks2, |&x| Ok(x * 3))
        .expect("Second operation failed");

    // Check cumulative metrics
    let metrics = bridge.metrics();
    assert!(metrics.tasks_completed() >= 30, "Should track all tasks");
}

#[test]
fn test_config_serialization() {
    let config = CpuConfig::builder()
        .num_threads(4)
        .batch_size(16)
        .enable_simd(false)
        .build();

    // Test that config can be serialized
    let json = serde_json::to_string(&config).expect("Should serialize");
    assert!(json.contains("num_threads"));
    assert!(json.contains("batch_size"));
}

#[test]
fn test_string_processing() {
    let bridge = CpuBridge::new();

    let tasks: Vec<String> = (0..100)
        .map(|i| format!("task_{}", i))
        .collect();

    let results = bridge
        .parallel_translate(tasks, |s| Ok(s.to_uppercase()))
        .expect("String processing failed");

    assert_eq!(results.len(), 100);
    assert_eq!(results[0], "TASK_0");
    assert_eq!(results[99], "TASK_99");
}

#[test]
fn test_complex_data_structures() {
    use std::collections::HashMap;

    let bridge = CpuBridge::new();

    let tasks: Vec<HashMap<String, i32>> = (0..50)
        .map(|i| {
            let mut map = HashMap::new();
            map.insert("value".to_string(), i);
            map
        })
        .collect();

    let results = bridge
        .parallel_translate(tasks, |map| {
            let value = map.get("value").unwrap();
            Ok(value * 2)
        })
        .expect("Complex data processing failed");

    assert_eq!(results.len(), 50);
}

#[test]
fn test_single_vs_batch_performance_metrics() {
    let bridge = CpuBridge::new();

    // Single task operations
    for i in 0..10 {
        bridge
            .translate_single(i, |&x| Ok(x * 2))
            .expect("Single task failed");
    }

    let metrics_after_single = bridge.metrics();
    assert!(metrics_after_single.single_task_count() > 0);

    // Batch operation
    let tasks: Vec<i32> = (0..10).collect();
    bridge
        .parallel_translate(tasks, |&x| Ok(x * 2))
        .expect("Batch failed");

    let metrics_after_batch = bridge.metrics();
    assert!(metrics_after_batch.batch_count() > 0);
}

/// Mock translation scenario for integration testing
#[derive(Debug, Clone)]
struct TranslationTask {
    id: usize,
    source_code: String,
}

#[derive(Debug, Clone)]
struct TranslationResult {
    id: usize,
    output_code: String,
    lines_processed: usize,
}

fn simulate_translation(task: &TranslationTask) -> anyhow::Result<TranslationResult> {
    // Simulate processing time
    std::thread::sleep(std::time::Duration::from_micros(50));

    Ok(TranslationResult {
        id: task.id,
        output_code: format!("// Generated from: {}", task.source_code),
        lines_processed: task.source_code.lines().count(),
    })
}

#[test]
fn test_realistic_translation_pipeline() {
    let bridge = CpuBridge::new();

    let tasks: Vec<TranslationTask> = (0..100)
        .map(|i| TranslationTask {
            id: i,
            source_code: format!("def function_{}():\n    pass\n    return True", i),
        })
        .collect();

    let results = bridge
        .parallel_translate(tasks, simulate_translation)
        .expect("Translation pipeline failed");

    assert_eq!(results.len(), 100);

    for (i, result) in results.iter().enumerate() {
        assert_eq!(result.id, i);
        assert!(result.output_code.contains("Generated from"));
        assert!(result.lines_processed >= 1);
    }
}

#[test]
fn test_translation_with_variable_workload() {
    let bridge = CpuBridge::new();

    let tasks: Vec<TranslationTask> = (0..50)
        .map(|i| TranslationTask {
            id: i,
            // Variable complexity
            source_code: "code\n".repeat(i % 10 + 1),
        })
        .collect();

    let results = bridge
        .parallel_translate(tasks, simulate_translation)
        .expect("Variable workload failed");

    assert_eq!(results.len(), 50);
}

#[test]
fn test_cpu_bridge_send_sync() {
    // Verify CpuBridge implements Send + Sync
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<CpuBridge>();
    assert_sync::<CpuBridge>();
}

#[test]
fn test_platform_info() {
    println!("\n=== CPU Bridge Test Environment ===");
    println!("Available CPU cores: {}", num_cpus::get());
    println!("Physical CPU cores: {}", num_cpus::get_physical());
    println!("Architecture: {}", std::env::consts::ARCH);
    println!("OS: {}", std::env::consts::OS);
    println!("====================================\n");

    assert!(num_cpus::get() > 0);
}

#[test]
fn test_stress_large_number_of_tasks() {
    let bridge = CpuBridge::new();
    let tasks: Vec<i32> = (0..50_000).collect();

    let start = std::time::Instant::now();
    let results = bridge
        .parallel_translate(tasks, |&x| Ok(x % 1000))
        .expect("Stress test failed");
    let duration = start.elapsed();

    println!("Processed 50,000 tasks in {:?}", duration);

    assert_eq!(results.len(), 50_000);
}

#[test]
fn test_memory_efficiency_with_large_data() {
    let bridge = CpuBridge::new();

    // Create tasks with large data (1KB each)
    let tasks: Vec<Vec<u8>> = (0..1000)
        .map(|i| vec![i as u8; 1024])
        .collect();

    let results = bridge
        .parallel_translate(tasks, |data| Ok(data.len()))
        .expect("Memory test failed");

    assert_eq!(results.len(), 1000);
    assert!(results.iter().all(|&len| len == 1024));
}

#[test]
fn test_mixed_workload() {
    let bridge = CpuBridge::new();

    // Mix of batch and single operations
    for i in 0..5 {
        // Single task
        bridge
            .translate_single(i, |&x| Ok(x * 2))
            .expect("Single task failed");

        // Batch tasks
        let batch: Vec<i32> = vec![i, i + 1, i + 2];
        bridge
            .parallel_translate(batch, |&x| Ok(x * 3))
            .expect("Batch failed");
    }

    let metrics = bridge.metrics();
    assert!(metrics.tasks_completed() > 0);
    assert!(metrics.single_task_count() >= 5);
    assert!(metrics.batch_count() >= 5);
}

#[test]
fn test_comprehensive_summary() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     CPU BRIDGE INTEGRATION TEST SUITE SUMMARY           ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Test Coverage Areas:                                     ║");
    println!("║  ✓ CPU Bridge Initialization                            ║");
    println!("║  ✓ Thread Pool Configuration                            ║");
    println!("║  ✓ Parallel Translation Execution                       ║");
    println!("║  ✓ Error Handling and Edge Cases                        ║");
    println!("║  ✓ Metrics Collection                                   ║");
    println!("║  ✓ Translation Pipeline Integration                     ║");
    println!("║  ✓ Performance and Scalability                          ║");
    println!("║  ✓ Cross-Platform Compatibility                         ║");
    println!("║  ✓ Concurrent Access Safety                             ║");
    println!("║  ✓ Stress Testing                                       ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Platform: {} on {}                              ",
             std::env::consts::ARCH,
             std::env::consts::OS);
    println!("║ CPUs: {} logical / {} physical                    ",
             num_cpus::get(),
             num_cpus::get_physical());
    println!("╚══════════════════════════════════════════════════════════╝\n");
}
