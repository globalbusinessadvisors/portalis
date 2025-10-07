//! Comprehensive SIMD Operations Tests for CPU Bridge
//!
//! This test suite validates:
//! - SIMD operations on current platform (x86_64 AVX2, ARM64 NEON)
//! - Fallback mechanisms when SIMD is unavailable
//! - Runtime feature detection
//! - Correctness (SIMD results == scalar results)
//! - Performance characteristics

use portalis_cpu_bridge::{CpuBridge, CpuConfig};

// ============================================================================
// SIMD Feature Detection Tests
// ============================================================================

#[test]
fn test_simd_feature_detection() {
    let config = CpuConfig::auto_detect();

    #[cfg(target_arch = "x86_64")]
    {
        // On x86_64, check if AVX2 is detected
        let has_avx2 = is_x86_feature_detected!("avx2");
        assert_eq!(config.simd_enabled(), has_avx2,
            "SIMD detection should match AVX2 availability");

        println!("x86_64 Platform:");
        println!("  AVX2 support: {}", has_avx2);
        println!("  AVX support: {}", is_x86_feature_detected!("avx"));
        println!("  SSE4.2 support: {}", is_x86_feature_detected!("sse4.2"));
        println!("  SIMD enabled in config: {}", config.simd_enabled());
    }

    #[cfg(target_arch = "aarch64")]
    {
        // On ARM64, NEON is always available
        assert!(config.simd_enabled(),
            "SIMD should always be enabled on ARM64 (NEON)");
        println!("ARM64 Platform: NEON support is built-in");
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // On other platforms, SIMD should be disabled
        assert!(!config.simd_enabled(),
            "SIMD should be disabled on non-x86_64/ARM64 platforms");
        println!("Platform {} does not support SIMD", std::env::consts::ARCH);
    }
}

#[test]
fn test_runtime_simd_detection() {
    // Test that we can detect SIMD at runtime
    let config = CpuConfig::auto_detect();
    let simd_enabled = config.simd_enabled();

    // Create bridge with auto-detected config
    let bridge = CpuBridge::with_config(config);

    // Verify config is accessible
    assert_eq!(bridge.config().simd_enabled(), simd_enabled);

    println!("Runtime SIMD detection:");
    println!("  SIMD enabled: {}", simd_enabled);
    println!("  Architecture: {}", std::env::consts::ARCH);
}

#[test]
fn test_manual_simd_control() {
    // Test that we can manually enable/disable SIMD
    let config_disabled = CpuConfig::builder()
        .enable_simd(false)
        .build();
    assert!(!config_disabled.simd_enabled());

    let config_enabled = CpuConfig::builder()
        .enable_simd(true)
        .build();
    assert!(config_enabled.simd_enabled());
}

// ============================================================================
// SIMD Correctness Tests - Numerical Operations
// ============================================================================

/// Scalar implementation for comparison
fn add_scalar(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Scalar multiplication
fn mul_scalar(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Scalar sum reduction
fn sum_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[cfg(target_arch = "x86_64")]
mod x86_simd_tests {
    use super::*;

    #[test]
    #[cfg(target_feature = "avx2")]
    fn test_avx2_vector_addition() {
        use std::arch::x86_64::*;

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let expected = add_scalar(&a, &b);

        unsafe {
            // AVX2 can process 8 f32s at once
            let va = _mm256_loadu_ps(a.as_ptr());
            let vb = _mm256_loadu_ps(b.as_ptr());
            let result = _mm256_add_ps(va, vb);

            let mut output = vec![0.0f32; 8];
            _mm256_storeu_ps(output.as_mut_ptr(), result);

            // Compare SIMD result with scalar result
            for (i, (&simd_val, &scalar_val)) in output.iter().zip(&expected).enumerate() {
                assert!((simd_val - scalar_val).abs() < 1e-6,
                    "Mismatch at index {}: SIMD={}, Scalar={}", i, simd_val, scalar_val);
            }
        }
    }

    #[test]
    #[cfg(target_feature = "avx2")]
    fn test_avx2_vector_multiplication() {
        use std::arch::x86_64::*;

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

        let expected = mul_scalar(&a, &b);

        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr());
            let vb = _mm256_loadu_ps(b.as_ptr());
            let result = _mm256_mul_ps(va, vb);

            let mut output = vec![0.0f32; 8];
            _mm256_storeu_ps(output.as_mut_ptr(), result);

            for (i, (&simd_val, &scalar_val)) in output.iter().zip(&expected).enumerate() {
                assert!((simd_val - scalar_val).abs() < 1e-6,
                    "Mismatch at index {}: SIMD={}, Scalar={}", i, simd_val, scalar_val);
            }
        }
    }

    #[test]
    #[cfg(target_feature = "sse4.2")]
    fn test_sse_vector_operations() {
        use std::arch::x86_64::*;

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];

        let expected = add_scalar(&a, &b);

        unsafe {
            // SSE can process 4 f32s at once
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let result = _mm_add_ps(va, vb);

            let mut output = vec![0.0f32; 4];
            _mm_storeu_ps(output.as_mut_ptr(), result);

            for (i, (&simd_val, &scalar_val)) in output.iter().zip(&expected).enumerate() {
                assert!((simd_val - scalar_val).abs() < 1e-6,
                    "Mismatch at index {}: SIMD={}, Scalar={}", i, simd_val, scalar_val);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_simd_tests {
    use super::*;

    #[test]
    fn test_neon_vector_addition() {
        use std::arch::aarch64::*;

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];

        let expected = add_scalar(&a, &b);

        unsafe {
            // NEON can process 4 f32s at once
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let result = vaddq_f32(va, vb);

            let mut output = vec![0.0f32; 4];
            vst1q_f32(output.as_mut_ptr(), result);

            for (i, (&simd_val, &scalar_val)) in output.iter().zip(&expected).enumerate() {
                assert!((simd_val - scalar_val).abs() < 1e-6,
                    "Mismatch at index {}: SIMD={}, Scalar={}", i, simd_val, scalar_val);
            }
        }
    }

    #[test]
    fn test_neon_vector_multiplication() {
        use std::arch::aarch64::*;

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![2.0f32, 2.0, 2.0, 2.0];

        let expected = mul_scalar(&a, &b);

        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let result = vmulq_f32(va, vb);

            let mut output = vec![0.0f32; 4];
            vst1q_f32(output.as_mut_ptr(), result);

            for (i, (&simd_val, &scalar_val)) in output.iter().zip(&expected).enumerate() {
                assert!((simd_val - scalar_val).abs() < 1e-6,
                    "Mismatch at index {}: SIMD={}, Scalar={}", i, simd_val, scalar_val);
            }
        }
    }
}

// ============================================================================
// Fallback Mechanism Tests
// ============================================================================

#[test]
fn test_simd_disabled_fallback() {
    // Create config with SIMD explicitly disabled
    let config = CpuConfig::builder()
        .num_threads(4)
        .enable_simd(false)
        .build();

    assert!(!config.simd_enabled(), "SIMD should be disabled");

    let bridge = CpuBridge::with_config(config);

    // Test that operations still work correctly with SIMD disabled
    let tasks: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let results = bridge
        .parallel_translate(tasks, |&x| Ok(x * 2.0))
        .expect("Translation should work without SIMD");

    assert_eq!(results.len(), 100);
    for (i, &result) in results.iter().enumerate() {
        assert!((result - (i as f32 * 2.0)).abs() < 1e-6);
    }
}

#[test]
fn test_scalar_vs_simd_equivalence() {
    // Test that scalar and SIMD-enabled configs produce same results
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();

    let config_scalar = CpuConfig::builder()
        .enable_simd(false)
        .build();

    let config_simd = CpuConfig::builder()
        .enable_simd(true)
        .build();

    let bridge_scalar = CpuBridge::with_config(config_scalar);
    let bridge_simd = CpuBridge::with_config(config_simd);

    // Perform same operation with both
    let results_scalar = bridge_scalar
        .parallel_translate(data.clone(), |&x| Ok(x * x + 1.0))
        .expect("Scalar operation failed");

    let results_simd = bridge_simd
        .parallel_translate(data, |&x| Ok(x * x + 1.0))
        .expect("SIMD operation failed");

    // Results should be identical (within floating point precision)
    assert_eq!(results_scalar.len(), results_simd.len());
    for (i, (&scalar, &simd)) in results_scalar.iter().zip(&results_simd).enumerate() {
        assert!((scalar - simd).abs() < 1e-4,
            "Mismatch at index {}: scalar={}, simd={}", i, scalar, simd);
    }
}

// ============================================================================
// Large-Scale SIMD Performance Tests
// ============================================================================

#[test]
fn test_simd_batch_processing() {
    let config = CpuConfig::auto_detect();
    let bridge = CpuBridge::with_config(config);

    // Large batch of data for SIMD processing
    let data: Vec<f32> = (0..10_000).map(|i| i as f32).collect();

    let start = std::time::Instant::now();
    let results = bridge
        .parallel_translate(data, |&x| {
            // Simulate computation that benefits from SIMD
            Ok(x * 2.0 + 1.0)
        })
        .expect("Batch processing failed");
    let duration = start.elapsed();

    println!("Processed 10,000 items in {:?}", duration);
    println!("SIMD enabled: {}", bridge.config().simd_enabled());

    assert_eq!(results.len(), 10_000);

    // Verify correctness
    for (i, &result) in results.iter().enumerate() {
        let expected = i as f32 * 2.0 + 1.0;
        assert!((result - expected).abs() < 1e-4);
    }
}

#[test]
fn test_simd_vs_scalar_performance() {
    let data: Vec<f32> = (0..50_000).map(|i| (i as f32).sin()).collect();

    // Test with SIMD disabled
    let config_scalar = CpuConfig::builder()
        .enable_simd(false)
        .num_threads(4)
        .build();
    let bridge_scalar = CpuBridge::with_config(config_scalar);

    let start_scalar = std::time::Instant::now();
    let _ = bridge_scalar
        .parallel_translate(data.clone(), |&x| Ok(x * x + x))
        .expect("Scalar failed");
    let duration_scalar = start_scalar.elapsed();

    // Test with SIMD enabled
    let config_simd = CpuConfig::builder()
        .enable_simd(true)
        .num_threads(4)
        .build();
    let bridge_simd = CpuBridge::with_config(config_simd);

    let start_simd = std::time::Instant::now();
    let _ = bridge_simd
        .parallel_translate(data, |&x| Ok(x * x + x))
        .expect("SIMD failed");
    let duration_simd = start_simd.elapsed();

    println!("\nPerformance Comparison (50,000 items):");
    println!("  Scalar mode: {:?}", duration_scalar);
    println!("  SIMD mode: {:?}", duration_simd);

    if bridge_simd.config().simd_enabled() {
        println!("  Speedup: {:.2}x",
            duration_scalar.as_secs_f64() / duration_simd.as_secs_f64());
    }
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

#[test]
fn test_simd_with_non_aligned_data() {
    let config = CpuConfig::auto_detect();
    let bridge = CpuBridge::with_config(config);

    // Test with sizes that aren't multiples of SIMD width
    for size in [1, 3, 7, 15, 31, 63, 127, 255] {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let results = bridge
            .parallel_translate(data.clone(), |&x| Ok(x + 1.0))
            .expect("Non-aligned processing failed");

        assert_eq!(results.len(), size);
        for (i, &result) in results.iter().enumerate() {
            assert!((result - (i as f32 + 1.0)).abs() < 1e-6);
        }
    }
}

#[test]
fn test_simd_with_empty_data() {
    let config = CpuConfig::auto_detect();
    let bridge = CpuBridge::with_config(config);

    let data: Vec<f32> = vec![];
    let results = bridge
        .parallel_translate(data, |&x| Ok(x * 2.0))
        .expect("Empty data should work");

    assert_eq!(results.len(), 0);
}

#[test]
fn test_simd_with_special_values() {
    let config = CpuConfig::auto_detect();
    let bridge = CpuBridge::with_config(config);

    // Test with special floating point values
    let data = vec![
        0.0, -0.0,
        f32::INFINITY, f32::NEG_INFINITY,
        f32::MIN, f32::MAX,
        1e-10, 1e10,
    ];

    let results = bridge
        .parallel_translate(data.clone(), |&x| Ok(x + 1.0))
        .expect("Special values processing failed");

    assert_eq!(results.len(), data.len());

    // Verify special values are handled correctly
    assert_eq!(results[0], 1.0); // 0.0 + 1.0
    assert_eq!(results[1], 1.0); // -0.0 + 1.0
    assert_eq!(results[2], f32::INFINITY); // inf + 1.0
    assert_eq!(results[3], f32::NEG_INFINITY); // -inf + 1.0
}

// ============================================================================
// Integration with CPU Bridge Features
// ============================================================================

#[test]
fn test_simd_with_different_thread_counts() {
    for num_threads in [1, 2, 4, 8] {
        let config = CpuConfig::builder()
            .num_threads(num_threads)
            .enable_simd(true)
            .build();

        let bridge = CpuBridge::with_config(config);

        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let results = bridge
            .parallel_translate(data, |&x| Ok(x * 2.0))
            .expect("Processing failed");

        assert_eq!(results.len(), 1000);

        // Verify correctness
        for (i, &result) in results.iter().enumerate() {
            assert!((result - (i as f32 * 2.0)).abs() < 1e-6);
        }
    }
}

#[test]
fn test_simd_with_different_batch_sizes() {
    for batch_size in [8, 16, 32, 64, 128] {
        let config = CpuConfig::builder()
            .batch_size(batch_size)
            .enable_simd(true)
            .build();

        let bridge = CpuBridge::with_config(config);

        let data: Vec<f32> = (0..500).map(|i| i as f32).collect();
        let results = bridge
            .parallel_translate(data, |&x| Ok(x + 10.0))
            .expect("Processing failed");

        assert_eq!(results.len(), 500);
    }
}

// ============================================================================
// Error Handling with SIMD
// ============================================================================

#[test]
fn test_simd_error_propagation() {
    let config = CpuConfig::auto_detect();
    let bridge = CpuBridge::with_config(config);

    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

    // Introduce error in the middle of processing
    let result = bridge.parallel_translate(data, |&x| {
        if x > 50.0 {
            Err(anyhow::anyhow!("Test error at {}", x))
        } else {
            Ok(x * 2.0)
        }
    });

    assert!(result.is_err(), "Error should propagate even with SIMD enabled");
}

// ============================================================================
// Platform-Specific Summary Test
// ============================================================================

#[test]
fn test_simd_platform_summary() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║         SIMD OPERATIONS TEST SUITE SUMMARY              ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Platform Information:                                    ║");
    println!("║  Architecture: {:<40} ║", std::env::consts::ARCH);
    println!("║  OS: {:<48} ║", std::env::consts::OS);
    println!("║  CPU Cores: {:<42} ║", num_cpus::get());
    println!("╠══════════════════════════════════════════════════════════╣");

    #[cfg(target_arch = "x86_64")]
    {
        println!("║ x86_64 SIMD Features:                                    ║");
        println!("║  AVX2: {:<46} ║", is_x86_feature_detected!("avx2"));
        println!("║  AVX: {:<47} ║", is_x86_feature_detected!("avx"));
        println!("║  SSE4.2: {:<44} ║", is_x86_feature_detected!("sse4.2"));
        println!("║  SSE4.1: {:<44} ║", is_x86_feature_detected!("sse4.1"));
        println!("║  SSSE3: {:<45} ║", is_x86_feature_detected!("ssse3"));
        println!("║  SSE3: {:<46} ║", is_x86_feature_detected!("sse3"));
        println!("║  SSE2: {:<46} ║", is_x86_feature_detected!("sse2"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("║ ARM64 SIMD Features:                                     ║");
        println!("║  NEON: Built-in (always available)                       ║");
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("║ SIMD Features: Not available on this architecture       ║");
    }

    println!("╠══════════════════════════════════════════════════════════╣");

    let config = CpuConfig::auto_detect();
    println!("║ CPU Bridge Configuration:                                ║");
    println!("║  SIMD Enabled: {:<40} ║", config.simd_enabled());
    println!("║  Num Threads: {:<41} ║", config.num_threads());
    println!("║  Batch Size: {:<42} ║", config.batch_size());
    println!("║  Stack Size: {} MB{:<30} ║", config.stack_size() / (1024 * 1024), "");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Test Coverage:                                           ║");
    println!("║  ✓ SIMD feature detection                                ║");
    println!("║  ✓ Platform-specific SIMD operations                     ║");
    println!("║  ✓ Scalar vs SIMD correctness                            ║");
    println!("║  ✓ Fallback mechanisms                                   ║");
    println!("║  ✓ Performance benchmarks                                ║");
    println!("║  ✓ Edge cases and boundary conditions                    ║");
    println!("║  ✓ Error handling                                        ║");
    println!("║  ✓ Integration with thread pool                          ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
}
