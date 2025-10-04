// Code Splitting Demo
// Demonstrates WASM module splitting for lazy loading and optimization

use portalis_transpiler::code_splitter::{
    CodeSplitter, SplittingStrategy, LazyLoadConfig,
};
use std::collections::HashMap;

fn main() {
    println!("=== Code Splitting Demo ===\n");
    println!("Demonstrates: Module Splitting, Lazy Loading, Chunk Optimization\n");
    println!("{}", "=".repeat(80));

    // Demo 1: No Splitting (Baseline)
    demo_no_splitting();
    println!("\n{}", "=".repeat(80));

    // Demo 2: Size-Based Splitting
    demo_size_based_splitting();
    println!("\n{}", "=".repeat(80));

    // Demo 3: Feature-Based Splitting
    demo_feature_based_splitting();
    println!("\n{}", "=".repeat(80));

    // Demo 4: Automatic Splitting
    demo_automatic_splitting();
    println!("\n{}", "=".repeat(80));

    // Demo 5: Lazy Loading Strategies
    demo_lazy_loading_strategies();
    println!("\n{}", "=".repeat(80));

    // Demo 6: Real-World Example
    demo_real_world_example();
    println!("\n{}", "=".repeat(80));

    println!("\n🎉 Code splitting demonstration complete!");
}

fn demo_no_splitting() {
    println!("\n=== Demo 1: No Splitting (Baseline) ===\n");

    let splitter = CodeSplitter::new(SplittingStrategy::None);
    let modules = create_sample_modules();

    let analysis = splitter.analyze(&modules);

    println!("{}", analysis.generate_report());

    println!("\nUse Case:");
    println!("  - Small applications (<200 KB)");
    println!("  - Simple utilities");
    println!("  - Minimal overhead needed");
}

fn demo_size_based_splitting() {
    println!("\n=== Demo 2: Size-Based Splitting ===\n");

    let splitter = CodeSplitter::new(SplittingStrategy::BySize);
    let modules = create_large_app_modules();

    let analysis = splitter.analyze(&modules);

    println!("{}", analysis.generate_report());

    println!("\nBenefits:");
    println!("  ✅ Reduced initial load: {} → {}",
        format_size(analysis.original_size),
        format_size(analysis.initial_load_size)
    );
    println!("  ✅ Size reduction: {:.1}%", analysis.size_reduction_percent());
    println!("  ✅ {} lazy-loadable chunks", analysis.chunks.len() - 1);

    println!("\nUsage:");
    println!("  1. Initial page load: Only 'main' chunk ({})",
        format_size(analysis.initial_load_size));
    println!("  2. On-demand loading: Remaining {} chunks as needed",
        analysis.chunks.len() - 1);
}

fn demo_feature_based_splitting() {
    println!("\n=== Demo 3: Feature-Based Splitting ===\n");

    let splitter = CodeSplitter::new(SplittingStrategy::ByFeature);
    let modules = create_feature_modules();

    let analysis = splitter.analyze(&modules);

    println!("{}", analysis.generate_report());

    println!("\nFeature Organization:");
    for chunk in &analysis.chunks {
        println!("  {} chunk:", chunk.name);
        println!("    - Size: {}", format_size(chunk.size));
        println!("    - Loading: {}", chunk.loading_strategy());
        println!("    - Modules: {}", chunk.modules.len());
    }

    println!("\nBest For:");
    println!("  - Large applications with distinct features");
    println!("  - Multi-page applications");
    println!("  - Applications with optional features");
}

fn demo_automatic_splitting() {
    println!("\n=== Demo 4: Automatic Splitting ===\n");

    println!("Testing automatic splitting with different bundle sizes:\n");

    let test_cases = vec![
        ("Small bundle", 150_000),   // 150 KB
        ("Medium bundle", 300_000),  // 300 KB
        ("Large bundle", 600_000),   // 600 KB
    ];

    for (name, size) in test_cases {
        let mut modules = HashMap::new();
        modules.insert("main".to_string(), size / 3);
        modules.insert("module_a".to_string(), size / 3);
        modules.insert("module_b".to_string(), size / 3);

        let splitter = CodeSplitter::new(SplittingStrategy::Automatic);
        let analysis = splitter.analyze(&modules);

        println!("{}: {} → {} chunks",
            name,
            format_size(analysis.original_size),
            analysis.chunks.len()
        );
        println!("  Strategy: {:?}", analysis.strategy);
        println!("  Initial load: {} ({:.1}%)",
            format_size(analysis.initial_load_size),
            (analysis.initial_load_size as f64 / analysis.original_size as f64) * 100.0
        );
        println!();
    }

    println!("Automatic Strategy:");
    println!("  - Small (<200 KB):  No splitting");
    println!("  - Medium (200-500 KB): Default splitting");
    println!("  - Large (>500 KB):  Aggressive splitting");
}

fn demo_lazy_loading_strategies() {
    println!("\n=== Demo 5: Lazy Loading Strategies ===\n");

    let configs = vec![
        ("Conservative", LazyLoadConfig::conservative()),
        ("Default", LazyLoadConfig::default()),
        ("Aggressive", LazyLoadConfig::aggressive()),
    ];

    let modules = create_large_app_modules();

    for (name, config) in configs {
        let splitter = CodeSplitter::with_config(SplittingStrategy::BySize, config);
        let analysis = splitter.analyze(&modules);

        println!("{} Configuration:", name);
        println!("  Chunks created: {}", analysis.chunks.len());
        println!("  Initial load: {}", format_size(analysis.initial_load_size));
        println!("  Splitting overhead: {}", format_size(analysis.splitting_overhead()));
        println!();
    }

    println!("Configuration Comparison:");
    println!("                  Conservative  Default  Aggressive");
    println!("  Size threshold: 100 KB        50 KB    30 KB");
    println!("  Max chunks:     10            20       50");
    println!("  Min chunk size: 20 KB         10 KB    5 KB");
    println!("  Route-based:    No            No       Yes");
}

fn demo_real_world_example() {
    println!("\n=== Demo 6: Real-World Example ===\n");

    println!("Scenario: E-commerce WASM Application\n");

    let mut modules = HashMap::new();

    // Core functionality (always loaded)
    modules.insert("core/bootstrap".to_string(), 30_000);
    modules.insert("core/router".to_string(), 15_000);
    modules.insert("core/state".to_string(), 20_000);

    // Product catalog (preload)
    modules.insert("ui/product_list".to_string(), 40_000);
    modules.insert("ui/product_card".to_string(), 25_000);

    // Shopping cart (lazy)
    modules.insert("ui/cart".to_string(), 35_000);
    modules.insert("data/cart_api".to_string(), 20_000);

    // Checkout (lazy, only when needed)
    modules.insert("ui/checkout".to_string(), 50_000);
    modules.insert("data/payment_api".to_string(), 30_000);

    // Admin panel (lazy, rarely used)
    modules.insert("ui/admin_panel".to_string(), 60_000);
    modules.insert("data/admin_api".to_string(), 40_000);

    let splitter = CodeSplitter::new(SplittingStrategy::ByFeature);
    let analysis = splitter.analyze(&modules);

    println!("{}", analysis.generate_report());

    println!("\nLoading Timeline:");
    println!("  T=0ms:    Load core chunk ({}) - EAGER",
        format_size(analysis.chunks.iter().find(|c| c.is_entry).unwrap().size));
    println!("  T=100ms:  Preload UI chunk - PREFETCH");
    println!("  T=user:   Load cart when user adds item - LAZY");
    println!("  T=user:   Load checkout when user checks out - LAZY");
    println!("  T=user:   Load admin only for admins - LAZY");

    println!("\nPerformance Impact:");
    let original = analysis.original_size;
    let initial = analysis.initial_load_size;
    println!("  Traditional (no splitting):  {} loaded upfront",
        format_size(original));
    println!("  With code splitting:         {} loaded upfront",
        format_size(initial));
    println!("  Improvement:                 {:.1}% faster initial load",
        ((original - initial) as f64 / original as f64) * 100.0);

    println!("\nWebpack Configuration:");
    println!("{}", splitter.generate_webpack_config(&analysis));

    println!("\nDynamic Import Example:");
    if analysis.chunks.len() > 1 {
        let chunk = &analysis.chunks[1];
        println!("{}", splitter.generate_dynamic_import(chunk));
    }
}

// Helper functions

fn create_sample_modules() -> HashMap<String, u64> {
    let mut modules = HashMap::new();
    modules.insert("main".to_string(), 100_000);
    modules.insert("utils".to_string(), 20_000);
    modules
}

fn create_large_app_modules() -> HashMap<String, u64> {
    let mut modules = HashMap::new();
    modules.insert("main".to_string(), 50_000);
    modules.insert("feature_a".to_string(), 80_000);
    modules.insert("feature_b".to_string(), 70_000);
    modules.insert("feature_c".to_string(), 60_000);
    modules.insert("feature_d".to_string(), 90_000);
    modules.insert("utils".to_string(), 30_000);
    modules
}

fn create_feature_modules() -> HashMap<String, u64> {
    let mut modules = HashMap::new();
    modules.insert("core".to_string(), 60_000);
    modules.insert("ui_components".to_string(), 80_000);
    modules.insert("ui_layout".to_string(), 40_000);
    modules.insert("data_api".to_string(), 50_000);
    modules.insert("data_cache".to_string(), 30_000);
    modules
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
