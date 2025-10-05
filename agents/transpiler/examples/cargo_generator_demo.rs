// Cargo.toml Generator Demo
// Demonstrates auto-generation of Cargo.toml from Python code

use portalis_transpiler::{
    import_analyzer::ImportAnalyzer,
    cargo_generator::{CargoGenerator, CargoConfig},
};

fn main() {
    println!("=== Cargo.toml Auto-Generator Demo ===\n");

    // Example Python code with various imports
    let python_code = r#"
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import http.client
import hashlib
from decimal import Decimal
from collections import deque
import uuid
"#;

    println!("Python Code:");
    println!("{}", python_code);
    println!("\n{}", "=".repeat(60));

    // Step 1: Analyze imports
    let analyzer = ImportAnalyzer::new();
    let analysis = analyzer.analyze(python_code);

    println!("\n=== Import Analysis ===");
    println!("Detected {} Python imports", analysis.python_imports.len());
    println!("Mapped to {} Rust crates", analysis.rust_dependencies.len());

    // Step 2: Generate Cargo.toml with default config
    println!("\n=== Generated Cargo.toml (Default Config) ===\n");

    let generator = CargoGenerator::new();
    let cargo_toml = generator.generate(&analysis);
    println!("{}", cargo_toml);

    println!("\n{}", "=".repeat(60));

    // Step 3: Generate with custom config
    println!("\n=== Generated Cargo.toml (Custom Config) ===\n");

    let custom_config = CargoConfig {
        package_name: "my_wasm_app".to_string(),
        version: "1.0.0".to_string(),
        edition: "2021".to_string(),
        authors: vec![
            "Alice <alice@example.com>".to_string(),
            "Bob <bob@example.com>".to_string(),
        ],
        description: Some("WASM application transpiled from Python".to_string()),
        license: Some("Apache-2.0".to_string()),
        wasm_optimized: true,
        wasi_support: true,
        features: vec!["experimental".to_string()],
        repository: Some("https://github.com/example/my_wasm_app".to_string()),
        homepage: Some("https://example.com/my_wasm_app".to_string()),
        documentation: Some("https://docs.rs/my_wasm_app".to_string()),
        keywords: vec!["wasm".to_string(), "python".to_string(), "transpiler".to_string()],
        categories: vec!["wasm".to_string(), "development-tools".to_string()],
        generate_binary: false,
        generate_benchmarks: false,
        rust_version: Some("1.70".to_string()),
    };

    let custom_generator = CargoGenerator::with_config(custom_config);
    let custom_cargo_toml = custom_generator.generate(&analysis);
    println!("{}", custom_cargo_toml);

    println!("\n{}", "=".repeat(60));

    // Step 4: Generate .cargo/config.toml
    println!("\n=== Generated .cargo/config.toml ===\n");

    let cargo_config = generator.generate_cargo_config(&analysis);
    println!("{}", cargo_config);

    println!("\n{}", "=".repeat(60));

    // Step 5: Show WASM compatibility info
    println!("\n=== WASM Deployment Info ===\n");

    if analysis.wasm_compatibility.fully_compatible {
        println!("✅ Fully WASM compatible - deploy anywhere!");
    } else {
        println!("WASM Requirements:");
        if analysis.wasm_compatibility.needs_wasi {
            println!("  📁 Requires WASI (filesystem access)");
            println!("     → Deploy to: Wasmtime, Wasmer");
        }
        if analysis.wasm_compatibility.needs_js_interop {
            println!("  🌐 Requires JS Interop (browser APIs)");
            println!("     → Deploy to: Browser, Node.js");
        }
        if analysis.wasm_compatibility.has_incompatible {
            println!("  ❌ Has incompatible modules");
            println!("     → Some features won't work in WASM");
        }
    }

    println!("\n{}", "=".repeat(60));

    // Step 6: Build instructions
    println!("\n=== Build Instructions ===\n");
    println!("1. Save Cargo.toml to your project directory");
    println!("2. Build for WASM:");
    println!("   cargo build --target wasm32-unknown-unknown --release");
    println!("\n3. Optimize WASM binary:");
    println!("   wasm-opt -Oz output.wasm -o optimized.wasm");
    println!("\n4. Generate JS bindings:");
    println!("   wasm-bindgen output.wasm --out-dir pkg --target web");

    if analysis.wasm_compatibility.needs_wasi {
        println!("\nFor WASI support:");
        println!("   cargo build --target wasm32-wasi --release");
        println!("   wasmtime run output.wasm");
    }

    println!("\n{}", "=".repeat(60));

    // Step 7: Size estimates
    println!("\n=== Estimated WASM Binary Size ===\n");
    let num_deps = analysis.rust_dependencies.len();
    let base_size = 50; // KB
    let size_per_dep = 20; // KB
    let estimated_size = base_size + (num_deps * size_per_dep);

    println!("Base size:        ~{} KB", base_size);
    println!("Dependencies:     {} crates × {} KB = {} KB", num_deps, size_per_dep, num_deps * size_per_dep);
    println!("Estimated total:  ~{} KB (unoptimized)", estimated_size);
    println!("After wasm-opt:   ~{} KB (optimized)", estimated_size / 2);
    println!("After gzip:       ~{} KB (compressed)", estimated_size / 4);

    println!("\n{}", "=".repeat(60));

    // Step 8: Binary executable example
    println!("\n=== Binary Executable Example ===\n");

    let binary_generator = CargoGenerator::new()
        .with_package_name("cli-tool".to_string())
        .with_description("Command-line tool transpiled from Python".to_string())
        .with_binary(true);

    let cli_code = r#"
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()
    print(f"Processing {args.input}")
"#;

    let cli_analysis = analyzer.analyze(cli_code);
    let binary_cargo_toml = binary_generator.generate(&cli_analysis);

    println!("Python CLI application:");
    println!("{}", cli_code);
    println!("\nGenerated Cargo.toml with binary target:");
    println!("{}", binary_cargo_toml);

    println!("\n{}", "=".repeat(60));

    // Step 9: Library with benchmarks
    println!("\n=== Library with Benchmarks Example ===\n");

    let bench_generator = CargoGenerator::new()
        .with_package_name("fast-lib".to_string())
        .with_description("High-performance library".to_string())
        .with_benchmarks(true);

    let lib_code = r#"
import pytest
import numpy as np

def fast_algorithm(data):
    return np.sum(data ** 2)
"#;

    let lib_analysis = analyzer.analyze(lib_code);
    let bench_cargo_toml = bench_generator.generate(&lib_analysis);

    println!("Python library with tests:");
    println!("{}", lib_code);
    println!("\nGenerated Cargo.toml with dev-dependencies and benchmarks:");
    println!("{}", bench_cargo_toml);

    println!("\n{}", "=".repeat(60));

    // Step 10: Builder pattern example
    println!("\n=== Using Builder Pattern ===\n");

    let builder_example = CargoGenerator::new()
        .with_package_name("awesome-project".to_string())
        .with_version("2.0.0".to_string())
        .with_authors(vec!["Developer <dev@example.com>".to_string()])
        .with_description("Showcasing the builder pattern".to_string())
        .with_license("MIT".to_string())
        .with_repository("https://github.com/user/awesome-project".to_string())
        .with_keywords(vec!["awesome".to_string(), "example".to_string()])
        .with_categories(vec!["development-tools".to_string()])
        .with_rust_version("1.70".to_string())
        .with_binary(true)
        .with_benchmarks(true);

    let builder_cargo_toml = builder_example.generate(&analysis);

    println!("Fluent builder API creates comprehensive Cargo.toml:");
    println!("{}", builder_cargo_toml);

    println!("\n{}", "=".repeat(60));
    println!("\n✅ All Cargo.toml generation examples completed!");
}
