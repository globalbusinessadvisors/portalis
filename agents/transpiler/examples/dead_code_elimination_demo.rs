// Dead Code Elimination and wasm-opt Demo
// Demonstrates comprehensive code optimization and tree-shaking

use portalis_transpiler::dead_code_eliminator::{
    DeadCodeEliminator, WasmOptPass, TreeShakingAnalysis,
};
use std::collections::{HashMap, HashSet};

fn main() {
    println!("=== Dead Code Elimination & wasm-opt Demo ===\n");
    println!("Demonstrates: Dead Code Analysis, Tree-Shaking, wasm-opt Optimization\n");
    println!("{}", "=".repeat(80));

    // Demo 1: Dead Code Analysis
    demo_dead_code_analysis();
    println!("\n{}", "=".repeat(80));

    // Demo 2: wasm-opt Optimization Passes
    demo_wasm_opt_passes();
    println!("\n{}", "=".repeat(80));

    // Demo 3: Tree-Shaking Analysis
    demo_tree_shaking();
    println!("\n{}", "=".repeat(80));

    // Demo 4: Complete Optimization Pipeline
    demo_complete_optimization();
    println!("\n{}", "=".repeat(80));

    // Demo 5: Optimization Recommendations
    demo_recommendations();
    println!("\n{}", "=".repeat(80));

    println!("\n🎉 Dead code elimination demonstration complete!");
}

fn demo_dead_code_analysis() {
    println!("\n=== Demo 1: Dead Code Analysis ===\n");

    let sample_code = r#"
use std::collections::HashMap;
use std::fs::File;  // Unused import

fn process_data(data: &str) -> String {
    data.to_uppercase()
}

fn unused_helper() {
    println!("This function is never called");
}

fn another_unused() {
    let _x = 42;
}

async fn fetch_data() {
    // Used in async context
    let result = process_data("test");
    println!("{}", result);
}

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(fetch_data());
}
"#;

    println!("Analyzing Rust code for dead code...\n");
    let analysis = DeadCodeEliminator::analyze_rust_code(sample_code);

    println!("{}", analysis.generate_report());

    let recommendations = DeadCodeEliminator::get_recommendations(&analysis);
    println!("Recommendations:");
    for (idx, rec) in recommendations.iter().enumerate() {
        println!("  {}. {}", idx + 1, rec);
    }
}

fn demo_wasm_opt_passes() {
    println!("\n=== Demo 2: wasm-opt Optimization Passes ===\n");

    let all_passes = WasmOptPass::all_passes();

    println!("Available Optimization Passes ({}):\n", all_passes.len());

    // Group by category
    let mut categories: HashMap<&str, Vec<&WasmOptPass>> = HashMap::new();

    for pass in &all_passes {
        let category = if pass.name.contains("remove") || pass.name == "dce" || pass.name == "vacuum" {
            "Dead Code Elimination"
        } else if pass.name.contains("inline") {
            "Inlining"
        } else if pass.name.contains("strip") {
            "Size Reduction"
        } else {
            "Code Optimization"
        };

        categories.entry(category).or_insert_with(Vec::new).push(pass);
    }

    for (category, passes) in categories {
        println!("{}:", category);
        for pass in passes {
            let marker = if pass.default { "✅" } else { "  " };
            println!("  {} --{:<35} Impact: {:.0}% - {}",
                marker,
                pass.name,
                pass.impact * 100.0,
                pass.description
            );
        }
        println!();
    }

    println!("Size Optimization Preset:");
    let size_passes = WasmOptPass::size_optimization_passes();
    for pass in &size_passes {
        println!("  --{}", pass);
    }
    println!("\nPerformance Optimization Preset:");
    let perf_passes = WasmOptPass::performance_optimization_passes();
    for pass in &perf_passes {
        println!("  --{}", pass);
    }
}

fn demo_tree_shaking() {
    println!("\n=== Demo 3: Tree-Shaking Analysis ===\n");

    // Simulate a project with dependencies
    let mut dependencies = HashMap::new();

    dependencies.insert("serde".to_string(), vec![
        "Serialize".to_string(),
        "Deserialize".to_string(),
        "ser::Error".to_string(),
        "de::Error".to_string(),
    ]);

    dependencies.insert("tokio".to_string(), vec![
        "runtime::Runtime".to_string(),
        "spawn".to_string(),
        "time::sleep".to_string(),
        "net::TcpListener".to_string(),
        "io::AsyncRead".to_string(),
    ]);

    dependencies.insert("reqwest".to_string(), vec![
        "Client".to_string(),
        "get".to_string(),
        "post".to_string(),
        "Response".to_string(),
    ]);

    dependencies.insert("chrono".to_string(), vec![
        "DateTime".to_string(),
        "Utc".to_string(),
        "Duration".to_string(),
    ]);

    dependencies.insert("unused_crate".to_string(), vec![
        "Function1".to_string(),
        "Function2".to_string(),
    ]);

    // Simulate used items in code
    let mut used_items = HashSet::new();
    used_items.insert("Serialize".to_string());
    used_items.insert("Deserialize".to_string());
    used_items.insert("runtime::Runtime".to_string());
    used_items.insert("spawn".to_string());
    used_items.insert("Client".to_string());
    used_items.insert("get".to_string());

    println!("Analyzing tree-shaking opportunities...\n");
    let analysis = DeadCodeEliminator::analyze_tree_shaking(&dependencies, &used_items);

    println!("{}", analysis.generate_report());

    println!("Optimization Opportunities:");
    if !analysis.unused_dependencies.is_empty() {
        println!("  1. Remove unused dependencies:");
        for dep in &analysis.unused_dependencies {
            println!("     - Remove '{}' from Cargo.toml", dep);
        }
    }

    if !analysis.partially_used.is_empty() {
        println!("  2. Optimize feature flags for partially used crates:");
        for item in &analysis.partially_used {
            println!("     - {}: Only using {}/{} features",
                item.dependency,
                item.used_features.len(),
                item.total_features
            );
            println!("       Consider using feature flags to reduce binary size");
        }
    }
}

fn demo_complete_optimization() {
    println!("\n=== Demo 4: Complete Optimization Pipeline ===\n");

    println!("Step 1: Analyze Source Code");
    println!("  - Detect unused functions, types, imports");
    println!("  - Estimate potential size reduction");
    println!();

    println!("Step 2: Configure Cargo.toml");
    println!("  [profile.wasm-size]");
    println!("  opt-level = \"z\"");
    println!("  lto = true");
    println!("  codegen-units = 1");
    println!("  panic = \"abort\"");
    println!("  strip = true");
    println!();

    println!("Step 3: Build WASM");
    println!("  $ cargo build --profile wasm-size --target wasm32-unknown-unknown");
    println!();

    println!("Step 4: Run wasm-opt with Dead Code Elimination");
    let cmd = DeadCodeEliminator::generate_wasm_opt_command(
        "target/wasm32-unknown-unknown/wasm-size/app.wasm",
        "app_optimized.wasm",
        4,
    );
    println!("  $ {}", cmd);
    println!();

    println!("Step 5: Analyze Results");
    println!("  Original WASM size:      8.7 MB");
    println!("  After cargo:             1.3 MB (85% reduction)");
    println!("  After wasm-opt + DCE:    450 KB (95% reduction)");
    println!("  After gzip:              150 KB (98% reduction)");
    println!();

    println!("Optimization Pass Breakdown:");
    println!("  Dead Code Elimination:");
    println!("    - dce:                         ~25% reduction");
    println!("    - remove-unused-names:         ~10% reduction");
    println!("    - remove-unused-module-elements: ~15% reduction");
    println!("  Size Reduction:");
    println!("    - strip-debug:                 ~30% reduction");
    println!("    - strip-producers:             ~5% reduction");
    println!("  Code Optimization:");
    println!("    - inlining:                    ~20% reduction");
    println!("    - simplify-locals:             ~15% reduction");
    println!("    - vacuum:                      ~15% reduction");
}

fn demo_recommendations() {
    println!("\n=== Demo 5: Optimization Recommendations ===\n");

    let sample_code = r#"
use std::collections::HashMap;
use std::fs::File;
use serde::{Serialize, Deserialize};

fn helper1() { }
fn helper2() { }
fn helper3() { }

fn main() {
    let _map = HashMap::new();
}
"#;

    let analysis = DeadCodeEliminator::analyze_rust_code(sample_code);
    let recommendations = DeadCodeEliminator::get_recommendations(&analysis);

    println!("Automated Recommendations:\n");
    for (idx, rec) in recommendations.iter().enumerate() {
        println!("{}. {}", idx + 1, rec);
    }

    println!("\n\nManual Optimization Checklist:\n");
    println!("Cargo Configuration:");
    println!("  ☑ Use opt-level = \"z\" for size");
    println!("  ☑ Enable LTO (Link Time Optimization)");
    println!("  ☑ Set codegen-units = 1");
    println!("  ☑ Use panic = \"abort\"");
    println!("  ☑ Enable strip = true");
    println!();

    println!("Code Optimization:");
    println!("  ☑ Remove unused functions and types");
    println!("  ☑ Remove unused imports");
    println!("  ☑ Use &str instead of String where possible");
    println!("  ☑ Avoid unnecessary allocations");
    println!("  ☑ Use const fn for compile-time computation");
    println!();

    println!("Dependency Management:");
    println!("  ☑ Remove unused dependencies");
    println!("  ☑ Use minimal feature flags");
    println!("  ☑ Prefer lightweight alternatives");
    println!("  ☑ Check for duplicate dependencies");
    println!();

    println!("WASM-Specific:");
    println!("  ☑ Run wasm-opt with aggressive optimization");
    println!("  ☑ Strip debug information");
    println!("  ☑ Use wasm-bindgen with minimal features");
    println!("  ☑ Enable tree-shaking in bundlers");
    println!("  ☑ Compress with gzip/brotli for transfer");
    println!();

    println!("Expected Size Reduction:");
    println!("  Cargo optimization:    60-85%");
    println!("  wasm-opt + DCE:        Additional 40-50%");
    println!("  Gzip compression:      Additional 60-70%");
    println!("  Total:                 95-98% reduction");
}
