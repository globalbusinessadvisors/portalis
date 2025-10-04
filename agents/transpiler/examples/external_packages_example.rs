// External Package Support Demo
// Shows how external PyPI packages map to Rust crates

use portalis_transpiler::{
    import_analyzer::ImportAnalyzer,
    external_packages::ExternalPackageRegistry,
    cargo_generator::CargoGenerator,
};

fn main() {
    println!("=== External Package Support Demo ===\n");

    // Python code using popular external packages
    let python_code = r#"
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pydantic import BaseModel
import click
"#;

    println!("Python Code with External Packages:");
    println!("{}", python_code);
    println!("\n{}", "=".repeat(70));

    // Analyze imports
    let analyzer = ImportAnalyzer::new();
    let analysis = analyzer.analyze(python_code);

    println!("\n=== Detected Python Packages ===\n");
    for import in &analysis.python_imports {
        println!("  📦 {} (type: {:?})", import.module, import.import_type);
        if !import.items.is_empty() {
            println!("     Items: {}", import.items.join(", "));
        }
    }

    println!("\n=== Rust Crate Mappings ===\n");
    for dep in &analysis.rust_dependencies {
        println!("  {} = \"{}\"", dep.crate_name, dep.version);

        if !dep.features.is_empty() {
            println!("    Features: [{}]", dep.features.join(", "));
        }

        println!("    WASM: {:?}", dep.wasm_compat);

        if let Some(ref notes) = dep.notes {
            println!("    Notes: {}", notes);
        }
        println!();
    }

    println!("{}", "=".repeat(70));

    // Show WASM compatibility breakdown
    println!("\n=== WASM Compatibility Analysis ===\n");

    let registry = ExternalPackageRegistry::new();
    let stats = registry.stats();

    println!("Total External Packages Mapped: {}", stats.total_packages);
    println!("├─ ✅ Full WASM Compatible: {}", stats.full_wasm_compat);
    println!("├─ 🟡 Partial WASM Compatible: {}", stats.partial_wasm_compat);
    println!("├─ 🌐 Requires JS Interop: {}", stats.requires_js_interop);
    println!("└─ ❌ Incompatible: {}", stats.incompatible);

    println!("\n=== Package-by-Package Compatibility ===\n");

    let packages = vec![
        ("numpy", "ndarray"),
        ("pandas", "polars"),
        ("requests", "reqwest"),
        ("pillow", "image"),
        ("sklearn", "linfa"),
        ("scipy", "nalgebra"),
        ("matplotlib", "plotters"),
        ("pydantic", "serde"),
        ("click", "clap"),
    ];

    for (py_pkg, rust_crate) in packages {
        if let Some(mapping) = registry.get_package(py_pkg) {
            let icon = match mapping.wasm_compatible {
                portalis_transpiler::stdlib_mapper::WasmCompatibility::Full => "✅",
                portalis_transpiler::stdlib_mapper::WasmCompatibility::Partial => "🟡",
                portalis_transpiler::stdlib_mapper::WasmCompatibility::RequiresWasi => "📁",
                portalis_transpiler::stdlib_mapper::WasmCompatibility::RequiresJsInterop => "🌐",
                portalis_transpiler::stdlib_mapper::WasmCompatibility::Incompatible => "❌",
            };

            println!("{} {} → {}", icon, py_pkg, rust_crate);
            println!("   Compatibility: {:?}", mapping.wasm_compatible);

            if let Some(ref notes) = mapping.notes {
                println!("   {}", notes);
            }
            println!();
        }
    }

    println!("{}", "=".repeat(70));

    // Generate Cargo.toml
    println!("\n=== Generated Cargo.toml ===\n");

    let generator = CargoGenerator::new()
        .with_package_name("data_science_app".to_string());
    let cargo_toml = generator.generate(&analysis);

    println!("{}", cargo_toml);

    println!("{}", "=".repeat(70));

    // Usage examples
    println!("\n=== Python → Rust Translation Examples ===\n");

    println!("NumPy:");
    println!("  Python:  arr = np.array([1, 2, 3])");
    println!("  Rust:    let arr = arr1(&[1, 2, 3]);");
    println!();

    println!("Pandas:");
    println!("  Python:  df = pd.DataFrame({{'a': [1,2,3]}})");
    println!("  Rust:    let df = DataFrame::new(vec![Series::new(\"a\", &[1,2,3])])?;");
    println!();

    println!("Requests:");
    println!("  Python:  response = requests.get(url)");
    println!("  Rust:    let response = reqwest::blocking::get(url)?;");
    println!();

    println!("Scikit-learn:");
    println!("  Python:  model = LinearRegression()");
    println!("  Rust:    let model = LinearRegression::new();");
    println!();

    println!("{}", "=".repeat(70));

    // Deployment guidance
    println!("\n=== Deployment Guidance ===\n");

    if analysis.wasm_compatibility.fully_compatible {
        println!("✅ All packages are fully WASM compatible!");
        println!("   Deploy to: Browser, WASI, Edge Compute");
    } else {
        println!("Deployment Requirements:\n");

        if analysis.wasm_compatibility.needs_js_interop {
            println!("🌐 Requires JavaScript Interop");
            println!("   - Packages: requests, matplotlib");
            println!("   - Deploy to: Browser (with wasm-bindgen), Node.js");
            println!("   - Uses: fetch() API, Canvas rendering\n");
        }

        if analysis.wasm_compatibility.needs_wasi {
            println!("📁 Requires WASI (Filesystem)");
            println!("   - Packages: pandas (for file I/O)");
            println!("   - Deploy to: Wasmtime, Wasmer");
            println!("   - Alternative: Use in-memory data or IndexedDB in browser\n");
        }

        if analysis.wasm_compatibility.has_incompatible {
            println!("❌ Some packages incompatible with WASM");
            println!("   - Consider alternatives or server-side execution");
        }
    }

    println!("\n{}", "=".repeat(70));

    // Statistics
    println!("\n=== Platform Statistics ===\n");
    println!("📊 Coverage:");
    println!("   - Standard Library: 50 modules mapped");
    println!("   - External Packages: {} packages mapped", stats.total_packages);
    println!("   - Total: {} Python → Rust mappings", 50 + stats.total_packages);
    println!();
    println!("🎯 Top PyPI Packages Supported:");
    println!("   ✅ NumPy, Pandas, Requests, Pillow, Scikit-learn");
    println!("   ✅ SciPy, Matplotlib, Pydantic, Click, PyTest");
    println!();
    println!("🚀 WASM Deployment Targets:");
    println!("   - Browser (via wasm-bindgen)");
    println!("   - WASI runtimes (Wasmtime, Wasmer)");
    println!("   - Edge compute (Cloudflare Workers, Fastly)");
    println!("   - Node.js (via wasm-bindgen)");
}
