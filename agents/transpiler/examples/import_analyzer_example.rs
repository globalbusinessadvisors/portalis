// Import Analyzer Example
// Demonstrates how to use the import analyzer to analyze Python code

use portalis_transpiler::import_analyzer::ImportAnalyzer;

fn main() {
    let analyzer = ImportAnalyzer::new();

    let python_code = r#"
import json
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import hashlib
"#;

    println!("Analyzing Python imports...\n");

    let analysis = analyzer.analyze(python_code);

    println!("=== Python Imports Detected ===");
    for import in &analysis.python_imports {
        println!("  - {} (type: {:?})", import.module, import.import_type);
        if !import.items.is_empty() {
            println!("    Items: {}", import.items.join(", "));
        }
    }

    println!("\n=== Rust Dependencies Required ===");
    for dep in &analysis.rust_dependencies {
        println!("  - {} = \"{}\"", dep.crate_name, dep.version);
        println!("    WASM Compat: {:?}", dep.wasm_compat);
        if let Some(ref notes) = dep.notes {
            println!("    Notes: {}", notes);
        }
    }

    println!("\n=== Rust Use Statements ===");
    for use_stmt in &analysis.rust_use_statements {
        println!("  {}", use_stmt);
    }

    println!("\n=== WASM Compatibility Summary ===");
    let compat = &analysis.wasm_compatibility;

    if compat.fully_compatible {
        println!("  ✅ Fully WASM Compatible");
    } else {
        println!("  Status:");
        if compat.needs_wasi {
            println!("    ⚠️  Requires WASI (filesystem/OS)");
        }
        if compat.needs_js_interop {
            println!("    🌐 Requires JS Interop (browser APIs)");
        }
        if compat.has_incompatible {
            println!("    ❌ Has incompatible modules");
        }
    }

    println!("\n=== Module-by-Module Compatibility ===");
    let mut modules: Vec<_> = compat.modules_by_compat.iter().collect();
    modules.sort_by_key(|(name, _)| *name);

    for (module, compat_level) in modules {
        let icon = match compat_level {
            portalis_transpiler::stdlib_mapper::WasmCompatibility::Full => "✅",
            portalis_transpiler::stdlib_mapper::WasmCompatibility::Partial => "🟡",
            portalis_transpiler::stdlib_mapper::WasmCompatibility::RequiresWasi => "📁",
            portalis_transpiler::stdlib_mapper::WasmCompatibility::RequiresJsInterop => "🌐",
            portalis_transpiler::stdlib_mapper::WasmCompatibility::Incompatible => "❌",
        };
        println!("  {} {} - {:?}", icon, module, compat_level);
    }

    if !analysis.unmapped_modules.is_empty() {
        println!("\n=== ⚠️  Unmapped Modules ===");
        for module in &analysis.unmapped_modules {
            println!("  - {}", module);
        }
    }

    println!("\n=== Generated Cargo.toml ===");
    let cargo_toml = analyzer.generate_cargo_toml_deps(&analysis);
    println!("{}", cargo_toml);

    println!("\n=== Compatibility Report ===");
    let report = analyzer.generate_compatibility_report(&analysis);
    println!("{}", report);
}
