//! # SIMD Operations Demo
//!
//! This example demonstrates the SIMD optimization capabilities of the CPU Bridge.
//! Run with: cargo run --example simd_demo

use portalis_cpu_bridge::simd::{
    batch_string_contains, detect_cpu_capabilities, parallel_string_match, vectorized_char_count,
};
use std::time::Instant;

fn main() {
    println!("=== Portalis CPU Bridge - SIMD Demo ===\n");

    // Detect CPU capabilities
    print_cpu_capabilities();
    println!();

    // Demo 1: Batch string contains
    demo_batch_string_contains();
    println!();

    // Demo 2: Parallel string match
    demo_parallel_string_match();
    println!();

    // Demo 3: Vectorized char count
    demo_vectorized_char_count();
    println!();

    // Demo 4: Performance comparison
    demo_performance_comparison();
}

fn print_cpu_capabilities() {
    println!("📊 CPU Capabilities:");
    println!("─────────────────────");

    let caps = detect_cpu_capabilities();

    println!("Platform: {}", std::env::consts::ARCH);
    println!("Best SIMD: {}", caps.best_simd());
    println!();

    println!("Supported Instructions:");
    println!("  AVX2:   {}", if caps.avx2 { "✅ Yes" } else { "❌ No" });
    println!("  SSE4.2: {}", if caps.sse42 { "✅ Yes" } else { "❌ No" });
    println!("  NEON:   {}", if caps.neon { "✅ Yes" } else { "❌ No" });

    if !caps.has_simd() {
        println!("\n⚠️  No SIMD support detected - using scalar fallback");
    }
}

fn demo_batch_string_contains() {
    println!("🔍 Demo 1: Batch String Contains");
    println!("─────────────────────────────────");

    let source_lines = vec![
        "import std::io::Read",
        "use rayon::prelude::*",
        "import numpy as np",
        "from typing import List",
        "fn main() {",
        "import tensorflow as tf",
        "use serde::Serialize",
    ];

    println!("Source lines:");
    for (i, line) in source_lines.iter().enumerate() {
        println!("  [{}] {}", i, line);
    }

    let start = Instant::now();
    let contains_import = batch_string_contains(&source_lines, "import");
    let elapsed = start.elapsed();

    println!("\nSearching for 'import':");
    for (line, &contains) in source_lines.iter().zip(contains_import.iter()) {
        println!(
            "  {} {}",
            if contains { "✅" } else { "  " },
            line
        );
    }

    println!("\nProcessed {} lines in {:?}", source_lines.len(), elapsed);
}

fn demo_parallel_string_match() {
    println!("🎯 Demo 2: Parallel String Match");
    println!("─────────────────────────────────");

    let identifiers = vec![
        "test_function",
        "test_module",
        "example_test",
        "test_case_1",
        "main_function",
        "test_utils",
        "helper_function",
    ];

    println!("Identifiers:");
    for (i, id) in identifiers.iter().enumerate() {
        println!("  [{}] {}", i, id);
    }

    let start = Instant::now();
    let starts_with_test = parallel_string_match(&identifiers, "test_");
    let elapsed = start.elapsed();

    println!("\nMatching prefix 'test_':");
    for (id, &matches) in identifiers.iter().zip(starts_with_test.iter()) {
        println!(
            "  {} {}",
            if matches { "✅" } else { "  " },
            id
        );
    }

    let test_count = starts_with_test.iter().filter(|&&x| x).count();
    println!(
        "\nFound {} test identifiers in {:?}",
        test_count, elapsed
    );
}

fn demo_vectorized_char_count() {
    println!("📈 Demo 3: Vectorized Character Count");
    println!("──────────────────────────────────────");

    let code_samples = vec![
        "fn hello_world() { println!(\"Hello, world!\"); }",
        "let x = 42;",
        "struct User { name: String, age: u32 }",
        "impl Display for MyType { ... }",
        "match value { Ok(v) => v, Err(e) => panic!(e) }",
    ];

    println!("Code samples:");
    for (i, code) in code_samples.iter().enumerate() {
        println!("  [{}] {}", i, code);
    }

    // Count different characters
    let chars_to_count = vec!['(', ')', '{', '}', '_'];

    println!("\nCharacter frequency analysis:");
    for ch in chars_to_count {
        let start = Instant::now();
        let counts = vectorized_char_count(&code_samples, ch);
        let elapsed = start.elapsed();

        let total: usize = counts.iter().sum();
        println!(
            "  '{}': {} occurrences (counted in {:?})",
            ch, total, elapsed
        );
    }

    // Detailed breakdown for one character
    let underscore_counts = vectorized_char_count(&code_samples, '_');
    println!("\nDetailed underscore count per line:");
    for (code, count) in code_samples.iter().zip(underscore_counts.iter()) {
        println!("  {}: {}", count, code);
    }
}

fn demo_performance_comparison() {
    println!("⚡ Demo 4: Performance Comparison");
    println!("──────────────────────────────────");

    // Generate larger test dataset
    let size = 1000;
    let test_strings: Vec<String> = (0..size)
        .map(|i| format!("import module_{} from 'package_{}'", i, i % 100))
        .collect();
    let test_refs: Vec<&str> = test_strings.iter().map(|s| s.as_str()).collect();

    println!("Dataset: {} strings", size);
    println!();

    // Benchmark SIMD version
    let start = Instant::now();
    let simd_results = batch_string_contains(&test_refs, "import");
    let simd_time = start.elapsed();

    // Benchmark scalar equivalent
    let start = Instant::now();
    let scalar_results: Vec<bool> = test_refs.iter().map(|s| s.contains("import")).collect();
    let scalar_time = start.elapsed();

    // Verify results match
    assert_eq!(simd_results, scalar_results);

    println!("Results:");
    println!("  SIMD version:   {:?}", simd_time);
    println!("  Scalar version: {:?}", scalar_time);

    if scalar_time > simd_time && simd_time.as_nanos() > 0 {
        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        println!("\n  🚀 Speedup: {:.2}x", speedup);
    } else if simd_time > scalar_time {
        let slowdown = simd_time.as_nanos() as f64 / scalar_time.as_nanos() as f64;
        println!("\n  ⚠️  SIMD overhead: {:.2}x (batch too small)", slowdown);
    } else {
        println!("\n  ≈ Performance similar (very fast operation)");
    }

    let caps = detect_cpu_capabilities();
    println!("\n  Using: {}", caps.best_simd());
    println!("  Matches found: {}", simd_results.iter().filter(|&&x| x).count());
}
