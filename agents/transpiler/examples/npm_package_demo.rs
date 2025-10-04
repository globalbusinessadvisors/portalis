// NPM Package Generation Demo
// Demonstrates comprehensive NPM package creation for WASM modules

use portalis_transpiler::npm_package_generator::{
    NpmPackageGenerator, NpmPackageConfig, ModuleType,
};

fn main() {
    println!("=== NPM Package Generation Demo ===\n");
    println!("Demonstrates: NPM package creation, README generation, TypeScript definitions\n");
    println!("{}", "=".repeat(80));

    // Demo 1: Basic NPM Package
    demo_basic_package();
    println!("\n{}", "=".repeat(80));

    // Demo 2: Scoped Package
    demo_scoped_package();
    println!("\n{}", "=".repeat(80));

    // Demo 3: Dual Module Package
    demo_dual_module_package();
    println!("\n{}", "=".repeat(80));

    // Demo 4: Package with Full Metadata
    demo_full_metadata_package();
    println!("\n{}", "=".repeat(80));

    // Demo 5: Package Files Breakdown
    demo_package_files();
    println!("\n{}", "=".repeat(80));

    // Demo 6: Publishing Workflow
    demo_publishing_workflow();
    println!("\n{}", "=".repeat(80));

    println!("\n🎉 NPM package generation demonstration complete!");
}

fn demo_basic_package() {
    println!("\n=== Demo 1: Basic NPM Package ===\n");

    let config = NpmPackageConfig {
        name: "simple-wasm-module".to_string(),
        version: "1.0.0".to_string(),
        description: "A simple WASM module for mathematical operations".to_string(),
        author: "John Doe".to_string(),
        license: "MIT".to_string(),
        keywords: vec![
            "wasm".to_string(),
            "webassembly".to_string(),
            "math".to_string(),
        ],
        ..Default::default()
    };

    let generator = NpmPackageGenerator::new(config);
    let package = generator.generate(150_000, 15_000);

    println!("Generated package.json:");
    println!("{}", "-".repeat(80));
    println!("{}", package.package_json);
    println!("{}", "-".repeat(80));

    println!("\nPackage Summary:");
    println!("  Size: {} KB", package.estimated_size / 1024);
    println!("  Files: {} examples + package.json + README + .npmignore", package.examples.len());
    println!("  TypeScript: {}", if package.typescript_defs.is_some() { "✓" } else { "✗" });
}

fn demo_scoped_package() {
    println!("\n=== Demo 2: Scoped Package ===\n");

    let config = NpmPackageConfig {
        name: "@myorg/ml-wasm".to_string(),
        version: "2.1.0".to_string(),
        description: "Machine learning algorithms compiled to WebAssembly".to_string(),
        author: "MyOrg Team <team@myorg.com>".to_string(),
        license: "Apache-2.0".to_string(),
        repository: Some("https://github.com/myorg/ml-wasm".to_string()),
        homepage: Some("https://ml-wasm.myorg.com".to_string()),
        bugs: Some("https://github.com/myorg/ml-wasm/issues".to_string()),
        keywords: vec![
            "wasm".to_string(),
            "webassembly".to_string(),
            "machine-learning".to_string(),
            "ml".to_string(),
            "ai".to_string(),
        ],
        module_type: ModuleType::ESModule,
        include_examples: true,
        include_types: true,
        targets: vec!["web".to_string(), "node".to_string(), "deno".to_string()],
    };

    let generator = NpmPackageGenerator::new(config);
    let package = generator.generate(500_000, 50_000);

    println!("Scoped Package Configuration:");
    println!("{}", "-".repeat(80));
    println!("Package name: @myorg/ml-wasm (scoped)");
    println!("Repository: https://github.com/myorg/ml-wasm");
    println!("Homepage: https://ml-wasm.myorg.com");
    println!("License: Apache-2.0");
    println!("{}", "-".repeat(80));

    println!("\npackage.json excerpt:");
    let lines: Vec<&str> = package.package_json.lines().take(15).collect();
    for line in lines {
        println!("{}", line);
    }
    println!("  ...");

    println!("\nPublishing:");
    println!("  npm publish --access public");
}

fn demo_dual_module_package() {
    println!("\n=== Demo 3: Dual Module Package (ESM + CommonJS) ===\n");

    let config = NpmPackageConfig {
        name: "universal-wasm".to_string(),
        version: "1.5.0".to_string(),
        description: "Universal WASM module supporting both ESM and CommonJS".to_string(),
        author: "Universal Team".to_string(),
        license: "MIT".to_string(),
        module_type: ModuleType::Dual,
        ..Default::default()
    };

    let generator = NpmPackageGenerator::new(config);
    let pkg_json = generator.generate_package_json();

    println!("Dual Module Support:");
    println!("{}", "-".repeat(80));

    // Extract exports section
    let exports_start = pkg_json.find("\"exports\"").unwrap_or(0);
    let exports_section: Vec<&str> = pkg_json[exports_start..]
        .lines()
        .take(8)
        .collect();

    for line in exports_section {
        println!("{}", line);
    }
    println!("{}", "-".repeat(80));

    println!("\nCompatibility:");
    println!("  ESM import:   import mod from 'universal-wasm'");
    println!("  CJS require:  const mod = require('universal-wasm')");
    println!("  TypeScript:   Full type definitions included");
}

fn demo_full_metadata_package() {
    println!("\n=== Demo 4: Package with Full Metadata ===\n");

    let config = NpmPackageConfig {
        name: "advanced-wasm-toolkit".to_string(),
        version: "3.2.1".to_string(),
        description: "Advanced WebAssembly toolkit with comprehensive features".to_string(),
        author: "WASM Labs <info@wasmlabs.io>".to_string(),
        license: "MIT".to_string(),
        repository: Some("https://github.com/wasmlabs/advanced-toolkit".to_string()),
        homepage: Some("https://toolkit.wasmlabs.io".to_string()),
        bugs: Some("https://github.com/wasmlabs/advanced-toolkit/issues".to_string()),
        keywords: vec![
            "wasm".to_string(),
            "webassembly".to_string(),
            "toolkit".to_string(),
            "advanced".to_string(),
            "performance".to_string(),
        ],
        module_type: ModuleType::ESModule,
        include_examples: true,
        include_types: true,
        targets: vec!["web".to_string(), "node".to_string(), "worker".to_string()],
    };

    let generator = NpmPackageGenerator::new(config);
    let package = generator.generate(750_000, 75_000);

    println!("README.md Preview:");
    println!("{}", "=".repeat(80));
    let readme_lines: Vec<&str> = package.readme.lines().take(40).collect();
    for line in readme_lines {
        println!("{}", line);
    }
    println!("\n... (truncated)");
    println!("{}", "=".repeat(80));

    println!("\n.npmignore Preview:");
    println!("{}", "-".repeat(80));
    let npmignore_lines: Vec<&str> = package.npmignore.lines().take(15).collect();
    for line in npmignore_lines {
        println!("{}", line);
    }
    println!("... (truncated)");
    println!("{}", "-".repeat(80));
}

fn demo_package_files() {
    println!("\n=== Demo 5: Package Files Breakdown ===\n");

    let config = NpmPackageConfig {
        name: "example-wasm".to_string(),
        version: "1.0.0".to_string(),
        description: "Example WASM package".to_string(),
        author: "Example Author".to_string(),
        license: "MIT".to_string(),
        include_examples: true,
        include_types: true,
        ..Default::default()
    };

    let generator = NpmPackageGenerator::new(config);
    let package = generator.generate(200_000, 20_000);

    println!("Package Contents:");
    println!("{}", "=".repeat(80));

    println!("\n1. Core Files:");
    println!("   ✓ package.json        ({} bytes)", package.package_json.len());
    println!("   ✓ README.md           ({} bytes)", package.readme.len());
    println!("   ✓ .npmignore          ({} bytes)", package.npmignore.len());

    if let Some(ref defs) = package.typescript_defs {
        println!("   ✓ index.d.ts          ({} bytes)", defs.len());
    }

    println!("\n2. Example Files ({} total):", package.examples.len());
    for (name, content) in &package.examples {
        println!("   ✓ {}   ({} bytes)", name, content.len());
    }

    println!("\n3. WASM Assets (provided separately):");
    println!("   ✓ index.wasm          (200 KB - binary)");
    println!("   ✓ index.js            (20 KB - glue code)");

    println!("\nTotal Estimated Size: {} KB", package.estimated_size / 1024);
    println!("{}", "=".repeat(80));

    // Show TypeScript definitions
    if let Some(ref defs) = package.typescript_defs {
        println!("\nTypeScript Definitions (index.d.ts):");
        println!("{}", "-".repeat(80));
        println!("{}", defs);
        println!("{}", "-".repeat(80));
    }

    // Show example file
    if let Some(example) = package.examples.get("example-nodejs.js") {
        println!("\nExample File (example-nodejs.js):");
        println!("{}", "-".repeat(80));
        println!("{}", example);
        println!("{}", "-".repeat(80));
    }
}

fn demo_publishing_workflow() {
    println!("\n=== Demo 6: Publishing Workflow ===\n");

    let config = NpmPackageConfig {
        name: "@company/production-wasm".to_string(),
        version: "4.0.0".to_string(),
        description: "Production-ready WASM module".to_string(),
        author: "Company Engineering".to_string(),
        license: "MIT".to_string(),
        repository: Some("https://github.com/company/production-wasm".to_string()),
        homepage: Some("https://production-wasm.company.com".to_string()),
        bugs: Some("https://github.com/company/production-wasm/issues".to_string()),
        keywords: vec![
            "wasm".to_string(),
            "production".to_string(),
            "enterprise".to_string(),
        ],
        module_type: ModuleType::ESModule,
        include_examples: true,
        include_types: true,
        ..Default::default()
    };

    let generator = NpmPackageGenerator::new(config);
    let package = generator.generate(300_000, 30_000);

    println!("Complete Publishing Workflow:");
    println!("{}", "=".repeat(80));

    println!("\n📋 Pre-Publishing Checklist:");
    println!("   ☑ package.json configured");
    println!("   ☑ README.md generated");
    println!("   ☑ TypeScript definitions included");
    println!("   ☑ Examples provided");
    println!("   ☑ .npmignore configured");
    println!("   ☑ License file included");

    println!("\n🔧 Build Pipeline:");
    println!("   1. Compile Rust to WASM:");
    println!("      $ cargo build --profile wasm-size --target wasm32-unknown-unknown");
    println!("\n   2. Optimize WASM:");
    println!("      $ wasm-opt -O4 input.wasm -o output.wasm");
    println!("\n   3. Generate bindings:");
    println!("      $ wasm-bindgen output.wasm --out-dir pkg --target web");
    println!("\n   4. Generate NPM package:");
    println!("      $ portalis-transpiler generate-npm-package");

    println!("\n📦 Package Validation:");
    println!("   $ npm pack --dry-run");
    println!("\n   Files to be published:");
    println!("     - package.json");
    println!("     - README.md");
    println!("     - index.js (glue code)");
    println!("     - index.wasm (binary)");
    println!("     - index.d.ts (TypeScript)");
    println!("     - LICENSE");

    println!("\n🚀 Publishing:");
    println!("   1. Login to npm:");
    println!("      $ npm login");
    println!("\n   2. Verify package:");
    println!("      $ npm publish --dry-run");
    println!("\n   3. Publish package:");
    println!("      $ npm publish --access public");
    println!("\n   4. Verify published:");
    println!("      $ npm view @company/production-wasm");

    println!("\n📊 Post-Publishing:");
    println!("   - Package URL: https://www.npmjs.com/package/@company/production-wasm");
    println!("   - Install: npm install @company/production-wasm");
    println!("   - Estimated download size: {} KB", package.estimated_size / 1024);

    println!("\n{}", "=".repeat(80));

    // Show publish script
    println!("\nGenerated Publish Script:");
    println!("{}", "-".repeat(80));
    let script = generator.generate_publish_script();
    println!("{}", script);
    println!("{}", "-".repeat(80));

    // Show .npmrc
    println!("\nGenerated .npmrc:");
    println!("{}", "-".repeat(80));
    let npmrc = generator.generate_npmrc();
    println!("{}", npmrc);
    println!("{}", "-".repeat(80));

    // Show package report
    println!("\nPackage Report:");
    println!("{}", "=".repeat(80));
    let report = generator.generate_package_report(&package);
    println!("{}", report);
    println!("{}", "=".repeat(80));
}
