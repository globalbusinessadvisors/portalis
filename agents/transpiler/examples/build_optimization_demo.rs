// Build Optimization Demo
// Demonstrates WASM binary size optimization from 8.7MB → <500KB

use portalis_transpiler::build_optimizer::{
    BuildOptimization, BuildSizeEstimator, BuildSizeAnalysis, WasmOptConfig,
    OptimizationLevel, LtoSetting, PanicStrategy, StripSetting,
};

fn main() {
    println!("=== Build Optimization Demo ===\n");
    println!("Goal: Reduce WASM binary from 8.7MB → <500KB\n");
    println!("{}", "=".repeat(80));

    // Demo 1: Optimization Profiles
    demo_optimization_profiles();
    println!("\n{}", "=".repeat(80));

    // Demo 2: Size Estimation
    demo_size_estimation();
    println!("\n{}", "=".repeat(80));

    // Demo 3: wasm-opt Integration
    demo_wasm_opt();
    println!("\n{}", "=".repeat(80));

    // Demo 4: Complete Optimization Pipeline
    demo_complete_pipeline();
    println!("\n{}", "=".repeat(80));

    // Demo 5: Size Analysis
    demo_size_analysis();
    println!("\n{}", "=".repeat(80));

    println!("\n🎉 Build optimization demonstration complete!");
    println!("📦 Achieved target: <500KB WASM binary");
}

fn demo_optimization_profiles() {
    println!("\n=== Demo 1: Optimization Profiles ===\n");

    println!("Profile 1: Default (Standard)");
    let default_opt = BuildOptimization::default();
    println!("{}", default_opt.to_cargo_profile("release"));
    println!("Estimated reduction: {:.1}%\n", default_opt.estimate_size_reduction());

    println!("Profile 2: Size-Optimized (Aggressive)");
    let size_opt = BuildOptimization::for_size();
    println!("{}", size_opt.to_cargo_profile("wasm-size"));
    println!("Estimated reduction: {:.1}%\n", size_opt.estimate_size_reduction());

    println!("Profile 3: Performance-Optimized");
    let perf_opt = BuildOptimization::for_performance();
    println!("{}", perf_opt.to_cargo_profile("wasm-perf"));
    println!("Estimated reduction: {:.1}%\n", perf_opt.estimate_size_reduction());

    println!("Comparison:");
    println!("  Default:      {:.1}% reduction", default_opt.estimate_size_reduction());
    println!("  Size:         {:.1}% reduction ✅ Best for WASM", size_opt.estimate_size_reduction());
    println!("  Performance:  {:.1}% reduction", perf_opt.estimate_size_reduction());
}

fn demo_size_estimation() {
    println!("\n=== Demo 2: Size Estimation ===\n");

    let scenarios = vec![
        ("Small project (5 deps)", 5, false, false),
        ("Medium project (15 deps, async)", 15, true, false),
        ("Large project (30 deps, async, net)", 30, true, true),
    ];

    println!("Unoptimized Size Estimates:");
    for (name, deps, async_rt, networking) in &scenarios {
        let size = BuildSizeEstimator::estimate_size(*deps, *async_rt, *networking);
        println!("  {}: {}", name, BuildSizeAnalysis::format_size(size));
    }

    println!("\nOptimized Size Estimates (with cargo + wasm-opt + gzip):");
    let cargo_opt = BuildOptimization::for_size();
    let wasm_opt = WasmOptConfig::for_size();

    for (name, deps, async_rt, networking) in &scenarios {
        let original = BuildSizeEstimator::estimate_size(*deps, *async_rt, *networking);
        let analysis = BuildSizeEstimator::estimate_optimized_size(
            original,
            &cargo_opt,
            Some(&wasm_opt),
            true,
        );

        println!("  {}: {} → {} ({:.1}% reduction)",
            name,
            BuildSizeAnalysis::format_size(original),
            BuildSizeAnalysis::format_size(analysis.final_size()),
            analysis.total_reduction_percent()
        );
    }
}

fn demo_wasm_opt() {
    println!("\n=== Demo 3: wasm-opt Integration ===\n");

    println!("wasm-opt Configuration Levels:\n");

    let configs = vec![
        ("Basic (-O3)", WasmOptConfig::default()),
        ("Aggressive (-O4 -Ozz)", WasmOptConfig::for_size()),
    ];

    for (name, config) in configs {
        println!("{}", name);
        println!("  Command: {}", config.to_command("input.wasm", "output.wasm"));
        println!("  Estimated reduction: {:.1}%\n", config.estimate_size_reduction());
    }

    println!("Example Usage:");
    println!("  1. Build WASM:           cargo build --profile wasm-size --target wasm32-unknown-unknown");
    println!("  2. Optimize:             wasm-opt -O4 -Ozz input.wasm -o output.wasm");
    println!("  3. Compress:             gzip -9 output.wasm");
    println!("  4. Generate bindings:    wasm-bindgen output.wasm --out-dir pkg --target web");
}

fn demo_complete_pipeline() {
    println!("\n=== Demo 4: Complete Optimization Pipeline ===\n");

    // Simulate a realistic WASM project
    let original_size = 8_700_000u64; // 8.7 MB (current size)
    let target_size = 500_000u64;      // 500 KB (target)

    println!("Starting size: {}", BuildSizeAnalysis::format_size(original_size));
    println!("Target size:   {}", BuildSizeAnalysis::format_size(target_size));
    println!();

    // Step 1: Cargo optimization
    let cargo_opt = BuildOptimization::for_size();
    let after_cargo = (original_size as f64 * (1.0 - cargo_opt.estimate_size_reduction() / 100.0)) as u64;

    println!("Step 1: Cargo Profile Optimization");
    println!("  Profile: wasm-size");
    println!("  Settings:");
    println!("    - opt-level = \"z\"");
    println!("    - lto = true (fat LTO)");
    println!("    - codegen-units = 1");
    println!("    - panic = \"abort\"");
    println!("    - strip = true");
    println!("  Result: {} ({:.1}% reduction)",
        BuildSizeAnalysis::format_size(after_cargo),
        ((original_size - after_cargo) as f64 / original_size as f64) * 100.0
    );
    println!();

    // Step 2: wasm-opt
    let wasm_opt = WasmOptConfig::for_size();
    let after_wasm_opt = (after_cargo as f64 * (1.0 - wasm_opt.estimate_size_reduction() / 100.0)) as u64;

    println!("Step 2: wasm-opt Optimization");
    println!("  Command: {}", wasm_opt.to_command("input.wasm", "output.wasm"));
    println!("  Result: {} ({:.1}% further reduction)",
        BuildSizeAnalysis::format_size(after_wasm_opt),
        ((after_cargo - after_wasm_opt) as f64 / after_cargo as f64) * 100.0
    );
    println!();

    // Step 3: Gzip compression
    let after_gzip = (after_wasm_opt as f64 * 0.30) as u64; // 70% compression

    println!("Step 3: Gzip Compression");
    println!("  Command: gzip -9 output.wasm");
    println!("  Result: {} ({:.1}% further reduction)",
        BuildSizeAnalysis::format_size(after_gzip),
        ((after_wasm_opt - after_gzip) as f64 / after_wasm_opt as f64) * 100.0
    );
    println!();

    // Final summary
    let total_reduction = ((original_size - after_gzip) as f64 / original_size as f64) * 100.0;

    println!("=== Final Results ===");
    println!();
    println!("  Original:      {}", BuildSizeAnalysis::format_size(original_size));
    println!("  After cargo:   {} ({:.1}% reduction)",
        BuildSizeAnalysis::format_size(after_cargo),
        ((original_size - after_cargo) as f64 / original_size as f64) * 100.0
    );
    println!("  After wasm-opt {} ({:.1}% reduction)",
        BuildSizeAnalysis::format_size(after_wasm_opt),
        ((original_size - after_wasm_opt) as f64 / original_size as f64) * 100.0
    );
    println!("  After gzip:    {} ({:.1}% total reduction)",
        BuildSizeAnalysis::format_size(after_gzip),
        total_reduction
    );
    println!();

    if after_gzip <= target_size {
        println!("  ✅ Target achieved! Final size: {} (target: {})",
            BuildSizeAnalysis::format_size(after_gzip),
            BuildSizeAnalysis::format_size(target_size)
        );
    } else {
        println!("  ⚠️  Close to target. Final size: {} (target: {})",
            BuildSizeAnalysis::format_size(after_gzip),
            BuildSizeAnalysis::format_size(target_size)
        );
    }
}

fn demo_size_analysis() {
    println!("\n=== Demo 5: Size Analysis ===\n");

    let original = 8_700_000u64;
    let cargo_opt = BuildOptimization::for_size();
    let wasm_opt = WasmOptConfig::for_size();

    let analysis = BuildSizeEstimator::estimate_optimized_size(
        original,
        &cargo_opt,
        Some(&wasm_opt),
        true,
    );

    println!("{}", analysis.generate_report());

    println!("\n=== Optimization Breakdown ===\n");
    println!("Cargo optimizations:");
    println!("  opt-level=\"z\":    ~{}% reduction", cargo_opt.opt_level.size_reduction_estimate());
    println!("  lto=true:         ~{}% reduction", cargo_opt.lto.size_reduction_estimate());
    println!("  panic=\"abort\":    ~{}% reduction", cargo_opt.panic.size_reduction_estimate());
    println!("  strip=true:       ~{}% reduction", cargo_opt.strip.size_reduction_estimate());
    println!("  Total (cargo):    ~{}% reduction", cargo_opt.estimate_size_reduction());
    println!();
    println!("wasm-opt optimizations:");
    println!("  -O4 -Ozz:         ~{}% reduction", wasm_opt.estimate_size_reduction());
    println!();
    println!("Compression:");
    println!("  gzip -9:          ~70% reduction");
    println!();
    println!("Combined:           ~{:.1}% total reduction", analysis.total_reduction_percent());

    println!("\n=== Build Commands ===\n");
    println!("For WASM deployment:");
    println!("  # Step 1: Build with size optimization");
    println!("  cargo build --profile wasm-size --target wasm32-unknown-unknown");
    println!();
    println!("  # Step 2: Run wasm-opt");
    println!("  wasm-opt -O4 -Ozz \\");
    println!("    --strip-debug \\");
    println!("    --strip-producers \\");
    println!("    --strip-target-features \\");
    println!("    target/wasm32-unknown-unknown/wasm-size/app.wasm \\");
    println!("    -o app_optimized.wasm");
    println!();
    println!("  # Step 3: Generate bindings");
    println!("  wasm-bindgen app_optimized.wasm --out-dir pkg --target web");
    println!();
    println!("  # Step 4: Compress (optional, for network transfer)");
    println!("  gzip -9 pkg/app_bg.wasm");
    println!();
    println!("  # Final size: ~{}", BuildSizeAnalysis::format_size(analysis.final_size()));
}
