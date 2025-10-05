// WASM Bundler with JS Glue Code Demo
// Demonstrates deployment pipeline for different targets

use portalis_transpiler::wasm_bundler::{
    WasmBundler, DeploymentTarget, BundleConfig, OptimizationLevel, CompressionFormat,
};

fn main() {
    println!("=== WASM Bundler & JS Glue Code Demo ===\n");
    println!("Demonstrates: Multi-target deployment, JS glue code generation, bundling pipeline\n");
    println!("{}", "=".repeat(80));

    // Demo 1: Web (ES Modules) Deployment
    demo_web_deployment();
    println!("\n{}", "=".repeat(80));

    // Demo 2: Node.js Deployment
    demo_nodejs_deployment();
    println!("\n{}", "=".repeat(80));

    // Demo 3: Bundler (Webpack/Rollup) Deployment
    demo_bundler_deployment();
    println!("\n{}", "=".repeat(80));

    // Demo 4: Deno Deployment
    demo_deno_deployment();
    println!("\n{}", "=".repeat(80));

    // Demo 5: No-modules (Classic) Deployment
    demo_no_modules_deployment();
    println!("\n{}", "=".repeat(80));

    // Demo 6: Complete Deployment Pipeline
    demo_complete_pipeline();
    println!("\n{}", "=".repeat(80));

    // Demo 7: Production vs Development Builds
    demo_build_configurations();
    println!("\n{}", "=".repeat(80));

    // Demo 8: Optimization Levels
    demo_optimization_levels();
    println!("\n{}", "=".repeat(80));

    // Demo 9: Compression & Size Analysis
    demo_compression_analysis();
    println!("\n{}", "=".repeat(80));

    // Demo 10: CDN Deployment
    demo_cdn_deployment();
    println!("\n{}", "=".repeat(80));

    // Demo 11: README Generation
    demo_readme_generation();
    println!("\n{}", "=".repeat(80));

    println!("\n🎉 WASM bundler demonstration complete!");
}

fn demo_web_deployment() {
    println!("\n=== Demo 1: Web (ES Modules) Deployment ===\n");

    let bundler = WasmBundler::new(BundleConfig {
        output_dir: "dist/web".to_string(),
        package_name: "my_wasm_app".to_string(),
        target: DeploymentTarget::Web,
        typescript: true,
        source_maps: true,
        optimize_size: true,
        debug: false,
        weak_refs: true,
        reference_types: true,
        optimization_level: OptimizationLevel::Size,
        compression: CompressionFormat::Gzip,
        generate_readme: true,
        code_splitting: false,
        minify_js: true,
    });

    println!("Target: Modern Web Browsers (ES2020+)");
    println!("Module System: ES Modules\n");

    // 1. wasm-bindgen command
    println!("Step 1: Generate wasm-bindgen command");
    let wasm_bindgen_cmd = bundler.generate_wasm_bindgen_command("target/wasm32-unknown-unknown/release/my_app.wasm");
    println!("$ {}\n", wasm_bindgen_cmd);

    // 2. JS glue code
    println!("Step 2: Generated JS glue code (my_wasm_app.js):");
    println!("{}", "-".repeat(80));
    println!("{}", bundler.generate_web_glue("my_wasm_app"));
    println!("{}", "-".repeat(80));

    // 3. package.json
    println!("\nStep 3: Generated package.json:");
    println!("{}", "-".repeat(80));
    println!("{}", bundler.generate_package_json());
    println!("{}", "-".repeat(80));

    // 4. HTML loader
    println!("\nStep 4: Generated HTML loader (index.html):");
    println!("{}", "-".repeat(80));
    println!("{}", bundler.generate_html_loader("my_wasm_app"));
    println!("{}", "-".repeat(80));

    println!("\nUsage in Browser:");
    println!(r#"  <script type="module">
    import {{ initWasm }} from './my_wasm_app.js';

    async function main() {{
      await initWasm();
      // Call WASM functions here
    }}
    main();
  </script>"#);
}

fn demo_nodejs_deployment() {
    println!("\n=== Demo 2: Node.js Deployment ===\n");

    let mut config = BundleConfig::development();
    config.output_dir = "dist/nodejs".to_string();
    config.package_name = "my_wasm_app".to_string();
    config.target = DeploymentTarget::NodeJs;
    let bundler = WasmBundler::new(config);

    println!("Target: Node.js (v16+)");
    println!("Module System: CommonJS\n");

    // 1. wasm-bindgen command
    println!("Step 1: Generate wasm-bindgen command");
    let wasm_bindgen_cmd = bundler.generate_wasm_bindgen_command("target/wasm32-unknown-unknown/release/my_app.wasm");
    println!("$ {}\n", wasm_bindgen_cmd);

    // 2. JS glue code
    println!("Step 2: Generated JS glue code (my_wasm_app.js):");
    println!("{}", "-".repeat(80));
    println!("{}", bundler.generate_nodejs_glue("my_wasm_app"));
    println!("{}", "-".repeat(80));

    // 3. package.json
    println!("\nStep 3: Generated package.json:");
    println!("{}", "-".repeat(80));
    println!("{}", bundler.generate_package_json());
    println!("{}", "-".repeat(80));

    println!("\nUsage in Node.js:");
    println!(r#"  const {{ initWasm }} = require('./my_wasm_app');

  async function main() {{
    await initWasm();
    // Call WASM functions here
  }}
  main();"#);
}

fn demo_bundler_deployment() {
    println!("\n=== Demo 3: Bundler (Webpack/Rollup) Deployment ===\n");

    let mut config = BundleConfig::production();
    config.output_dir = "dist/bundler".to_string();
    config.package_name = "my_wasm_app".to_string();
    config.target = DeploymentTarget::Bundler;
    let bundler = WasmBundler::new(config);

    println!("Target: Webpack/Rollup/Vite");
    println!("Module System: ES Modules (optimized for bundlers)\n");

    // 1. wasm-bindgen command
    println!("Step 1: Generate wasm-bindgen command");
    let wasm_bindgen_cmd = bundler.generate_wasm_bindgen_command("target/wasm32-unknown-unknown/release/my_app.wasm");
    println!("$ {}\n", wasm_bindgen_cmd);

    // 2. JS glue code
    println!("Step 2: Generated JS glue code (my_wasm_app.js):");
    println!("{}", "-".repeat(80));
    println!("{}", bundler.generate_web_glue("my_wasm_app"));
    println!("{}", "-".repeat(80));

    println!("\nWebpack Configuration:");
    println!("{}", "-".repeat(80));
    println!(r#"// webpack.config.js
module.exports = {{
  experiments: {{
    asyncWebAssembly: true,
    topLevelAwait: true,
  }},
  module: {{
    rules: [
      {{
        test: /\.wasm$/,
        type: 'webassembly/async',
      }},
    ],
  }},
}};"#);
    println!("{}", "-".repeat(80));

    println!("\nVite Configuration:");
    println!("{}", "-".repeat(80));
    println!(r#"// vite.config.js
export default {{
  optimizeDeps: {{
    exclude: ['my_wasm_app']
  }},
  server: {{
    fs: {{
      allow: ['..']
    }}
  }}
}};"#);
    println!("{}", "-".repeat(80));
}

fn demo_deno_deployment() {
    println!("\n=== Demo 4: Deno Deployment ===\n");

    let mut config = BundleConfig::production();
    config.output_dir = "dist/deno".to_string();
    config.package_name = "my_wasm_app".to_string();
    config.target = DeploymentTarget::Deno;
    let bundler = WasmBundler::new(config);

    println!("Target: Deno (v1.30+)");
    println!("Module System: ES Modules with Deno APIs\n");

    // 1. wasm-bindgen command
    println!("Step 1: Generate wasm-bindgen command");
    let wasm_bindgen_cmd = bundler.generate_wasm_bindgen_command("target/wasm32-unknown-unknown/release/my_app.wasm");
    println!("$ {}\n", wasm_bindgen_cmd);

    // 2. JS glue code (Deno uses web glue)
    println!("Step 2: Generated JS glue code (my_wasm_app.js):");
    println!("{}", "-".repeat(80));
    println!("{}", bundler.generate_web_glue("my_wasm_app"));
    println!("{}", "-".repeat(80));

    println!("\nUsage in Deno:");
    println!(r#"  import {{ initWasm }} from './my_wasm_app.js';

  await initWasm();
  // Call WASM functions here"#);

    println!("\nRun with:");
    println!("  $ deno run --allow-read --allow-net main.ts");
}

fn demo_no_modules_deployment() {
    println!("\n=== Demo 5: No-modules (Classic) Deployment ===\n");

    let mut config = BundleConfig::production();
    config.output_dir = "dist/no-modules".to_string();
    config.package_name = "my_wasm_app".to_string();
    config.target = DeploymentTarget::NoModules;
    config.typescript = false;
    config.weak_refs = false;
    config.reference_types = false;
    let bundler = WasmBundler::new(config);

    println!("Target: Legacy Browsers (IE11+)");
    println!("Module System: None (global namespace)\n");

    // 1. wasm-bindgen command
    println!("Step 1: Generate wasm-bindgen command");
    let wasm_bindgen_cmd = bundler.generate_wasm_bindgen_command("target/wasm32-unknown-unknown/release/my_app.wasm");
    println!("$ {}\n", wasm_bindgen_cmd);

    println!("Step 2: Include in HTML:");
    println!("{}", "-".repeat(80));
    println!(r#"<!DOCTYPE html>
<html>
<head>
  <script src="my_wasm_app.js"></script>
</head>
<body>
  <script>
    // WASM available in global namespace
    wasm_bindgen('./my_wasm_app_bg.wasm').then(function() {{
      // Call WASM functions here
      console.log('WASM initialized');
    }});
  </script>
</body>
</html>"#);
    println!("{}", "-".repeat(80));
}

fn demo_complete_pipeline() {
    println!("\n=== Demo 6: Complete Deployment Pipeline ===\n");

    let mut prod_config = BundleConfig::production();
    prod_config.package_name = "my_wasm_app".to_string();
    prod_config.output_dir = "dist/web".to_string();

    let bundler = WasmBundler::new(prod_config);

    println!("Complete build and deployment pipeline:\n");

    let pipeline = bundler.generate_deployment_script();

    println!("{}", "=".repeat(80));
    println!("{}", pipeline);
    println!("{}", "=".repeat(80));

    println!("\nPipeline Stages:");
    println!("  1. Cargo build (optimized for WASM)");
    println!("  2. wasm-bindgen (generate JS bindings)");
    println!("  3. wasm-opt (optimize WASM binary)");
    println!("  4. Copy artifacts to output directory");
    println!("  5. Generate package.json and HTML loader");

    println!("\nRun the pipeline:");
    println!("  $ bash deploy_wasm.sh");

    // Simulate bundle creation
    println!("\n\nSimulating bundle creation...\n");
    let bundle = bundler.create_bundle_simulation(450_000);

    println!("Bundle Output Summary:");
    println!("{}", "=".repeat(80));
    println!("Output directory: {}", bundle.output_dir);
    println!("Target: {:?}", bundle.target);
    println!("\nGenerated Files:");
    for file in &bundle.files {
        println!("  {} - {} ({} bytes)",
            file.path,
            file.file_type,
            file.size
        );
    }
    println!("\nSize Breakdown:");
    println!("  WASM binary:  {} KB", bundle.wasm_size / 1024);
    println!("  JS glue code: {} KB", bundle.js_size / 1024);
    println!("  Total size:   {} KB", bundle.total_size / 1024);
    println!("{}", "=".repeat(80));
}

fn demo_build_configurations() {
    println!("\n=== Demo 7: Production vs Development Builds ===\n");

    // Development build
    let mut dev_config = BundleConfig::development();
    dev_config.package_name = "my_wasm_app".to_string();

    println!("Development Configuration:");
    println!("{}", "-".repeat(80));
    println!("  Debug: {}", dev_config.debug);
    println!("  Source maps: {}", dev_config.source_maps);
    println!("  Optimize size: {}", dev_config.optimize_size);
    println!("  TypeScript: {}", dev_config.typescript);
    println!("{}", "-".repeat(80));

    let dev_bundler = WasmBundler::new(dev_config);
    let dev_cmd = dev_bundler.generate_wasm_bindgen_command("target/wasm32-unknown-unknown/debug/my_app.wasm");
    println!("\nDevelopment wasm-bindgen command:");
    println!("  $ {}\n", dev_cmd);

    // Production build
    let mut prod_config = BundleConfig::production();
    prod_config.package_name = "my_wasm_app".to_string();

    println!("Production Configuration:");
    println!("{}", "-".repeat(80));
    println!("  Debug: {}", prod_config.debug);
    println!("  Source maps: {}", prod_config.source_maps);
    println!("  Optimize size: {}", prod_config.optimize_size);
    println!("  TypeScript: {}", prod_config.typescript);
    println!("  Weak refs: {}", prod_config.weak_refs);
    println!("  Reference types: {}", prod_config.reference_types);
    println!("{}", "-".repeat(80));

    let prod_bundler = WasmBundler::new(prod_config);
    let prod_cmd = prod_bundler.generate_wasm_bindgen_command("target/wasm32-unknown-unknown/release/my_app.wasm");
    println!("\nProduction wasm-bindgen command:");
    println!("  $ {}\n", prod_cmd);

    println!("Key Differences:");
    println!("  Development:");
    println!("    - Debug symbols included");
    println!("    - Source maps enabled");
    println!("    - No size optimization");
    println!("    - Faster build times\n");
    println!("  Production:");
    println!("    - No debug symbols");
    println!("    - Optional source maps");
    println!("    - Aggressive size optimization");
    println!("    - Weak refs & reference types enabled");
    println!("    - Optimized for deployment");

    // Size comparison
    println!("\n\nEstimated Bundle Sizes:");
    println!("{}", "=".repeat(80));

    let dev_bundle = dev_bundler.create_bundle_simulation(850_000);
    println!("Development: {} KB total", dev_bundle.total_size / 1024);
    println!("  WASM: {} KB", dev_bundle.wasm_size / 1024);
    println!("  JS:   {} KB\n", dev_bundle.js_size / 1024);

    let prod_bundle = prod_bundler.create_bundle_simulation(450_000);
    println!("Production:  {} KB total", prod_bundle.total_size / 1024);
    println!("  WASM: {} KB", prod_bundle.wasm_size / 1024);
    println!("  JS:   {} KB", prod_bundle.js_size / 1024);

    let reduction = ((dev_bundle.total_size - prod_bundle.total_size) as f64 / dev_bundle.total_size as f64) * 100.0;
    println!("\nSize Reduction: {:.1}%", reduction);
    println!("{}", "=".repeat(80));
}

fn demo_optimization_levels() {
    println!("\n=== Demo 8: Optimization Levels ===\n");

    println!("Testing different optimization levels with wasm-opt:\n");

    let levels = vec![
        (OptimizationLevel::None, "No optimization - fastest build", 2_500_000),
        (OptimizationLevel::Basic, "Basic (-O1)", 1_800_000),
        (OptimizationLevel::Standard, "Standard (-O2)", 1_200_000),
        (OptimizationLevel::Aggressive, "Aggressive (-O3)", 900_000),
        (OptimizationLevel::Size, "Size (-Oz)", 600_000),
        (OptimizationLevel::MaxSize, "Max Size (-Ozz)", 450_000),
    ];

    for (level, desc, expected_size) in levels {
        let config = BundleConfig {
            optimization_level: level,
            package_name: "optimized_app".to_string(),
            ..BundleConfig::default()
        };

        let bundler = WasmBundler::new(config);
        let cmd = bundler.generate_wasm_opt_command("input.wasm", "output.wasm");

        println!("{:?} - {}", level, desc);
        println!("  Flag: {}", level.wasm_opt_flag());
        println!("  Expected size: {:.1} MB", expected_size as f64 / 1_000_000.0);
        println!("  Command: {}", cmd);
        println!();
    }

    println!("Optimization vs Build Time Trade-off:");
    println!("  None      → 0.1s build, 2.5 MB output");
    println!("  Basic     → 0.5s build, 1.8 MB output");
    println!("  Standard  → 2s build, 1.2 MB output");
    println!("  Aggressive→ 10s build, 900 KB output");
    println!("  Size      → 30s build, 600 KB output");
    println!("  MaxSize   → 60s build, 450 KB output (BEST)");
}

fn demo_compression_analysis() {
    println!("\n=== Demo 9: Compression & Size Analysis ===\n");

    println!("Analyzing compression strategies:\n");

    let formats = vec![
        (CompressionFormat::None, "No compression"),
        (CompressionFormat::Gzip, "Gzip only"),
        (CompressionFormat::Brotli, "Brotli only"),
        (CompressionFormat::Both, "Gzip + Brotli"),
    ];

    let original_size = 450_000_u64;

    for (format, desc) in formats {
        let config = BundleConfig {
            compression: format,
            optimization_level: OptimizationLevel::MaxSize,
            package_name: "compressed_app".to_string(),
            ..BundleConfig::default()
        };

        let bundler = WasmBundler::new(config);
        let (gzip_size, brotli_size) = bundler.estimate_compressed_size(original_size);

        println!("{:?} - {}", format, desc);
        println!("  Original: {:.1} KB", original_size as f64 / 1024.0);

        match format {
            CompressionFormat::None => {
                println!("  Deployed: {:.1} KB", original_size as f64 / 1024.0);
            },
            CompressionFormat::Gzip => {
                println!("  Gzip: {:.1} KB ({:.1}% reduction)",
                    gzip_size as f64 / 1024.0,
                    ((original_size - gzip_size) as f64 / original_size as f64) * 100.0);
            },
            CompressionFormat::Brotli => {
                println!("  Brotli: {:.1} KB ({:.1}% reduction)",
                    brotli_size as f64 / 1024.0,
                    ((original_size - brotli_size) as f64 / original_size as f64) * 100.0);
            },
            CompressionFormat::Both => {
                println!("  Gzip: {:.1} KB ({:.1}% reduction)",
                    gzip_size as f64 / 1024.0,
                    ((original_size - gzip_size) as f64 / original_size as f64) * 100.0);
                println!("  Brotli: {:.1} KB ({:.1}% reduction)",
                    brotli_size as f64 / 1024.0,
                    ((original_size - brotli_size) as f64 / original_size as f64) * 100.0);
            },
        }

        let commands = bundler.generate_compression_commands("output.wasm");
        if !commands.is_empty() {
            println!("  Commands:");
            for cmd in commands {
                println!("    $ {}", cmd);
            }
        }
        println!();
    }

    // Optimization report
    println!("\nComplete Optimization Pipeline:");
    println!("{}", "=".repeat(80));

    let bundler = WasmBundler::new(BundleConfig::production());
    let report = bundler.generate_optimization_report(2_500_000, 450_000);
    println!("{}", report);
}

fn demo_cdn_deployment() {
    println!("\n=== Demo 10: CDN-Optimized Deployment ===\n");

    let cdn_config = BundleConfig::cdn_optimized();
    let bundler = WasmBundler::new(cdn_config);

    println!("CDN Configuration:");
    println!("{}", "-".repeat(80));
    println!("  Output: cdn/");
    println!("  Optimization: {:?}", OptimizationLevel::MaxSize);
    println!("  Compression: {:?}", CompressionFormat::Both);
    println!("  Minify JS: Yes");
    println!("  Code Splitting: Yes");
    println!("{}", "-".repeat(80));

    println!("\nGenerated Files:");
    let bundle = bundler.create_bundle_simulation(450_000);
    println!("{}", bundle.generate_report());

    println!("\nCDN Deployment Instructions:");
    println!("{}", "=".repeat(80));
    println!("1. Upload bundle to CDN:");
    println!("   $ aws s3 sync cdn/ s3://my-cdn-bucket/wasm-app/v1.0.0/");
    println!("\n2. Configure Cache-Control:");
    println!("   Cache-Control: public, max-age=31536000, immutable");
    println!("\n3. Configure CORS:");
    println!(r#"   {{
     "AllowedOrigins": ["*"],
     "AllowedMethods": ["GET"],
     "AllowedHeaders": ["*"]
   }}"#);
    println!("\n4. Enable Compression:");
    println!("   Content-Encoding: br (for .wasm.br files)");
    println!("   Content-Encoding: gzip (for .wasm.gz files)");
    println!("\n5. Set Content-Type:");
    println!("   .wasm files: application/wasm");
    println!("   .js files: application/javascript");
    println!("{}", "=".repeat(80));

    println!("\nUsage from CDN:");
    println!(r#"  <script type="module">
    import init from 'https://cdn.example.com/wasm-app/v1.0.0/compressed_app.js';

    async function main() {{
      await init();
      console.log('WASM loaded from CDN!');
    }}
    main();
  </script>"#);

    println!("\nPerformance Benefits:");
    println!("  ✓ Global edge distribution");
    println!("  ✓ Aggressive caching (1 year)");
    println!("  ✓ Automatic compression selection (Brotli > Gzip)");
    println!("  ✓ HTTP/2 multiplexing");
    println!("  ✓ Reduced server load");
}

fn demo_readme_generation() {
    println!("\n=== Demo 11: README Generation ===\n");

    let bundler = WasmBundler::new(BundleConfig {
        package_name: "awesome_wasm_app".to_string(),
        optimization_level: OptimizationLevel::MaxSize,
        compression: CompressionFormat::Both,
        generate_readme: true,
        ..BundleConfig::production()
    });

    println!("Generated README.md:");
    println!("{}", "=".repeat(80));
    println!("{}", bundler.generate_readme());
    println!("{}", "=".repeat(80));

    println!("\nAdditional Generated Files:");
    println!("  - .gitignore");
    println!("{}", "-".repeat(40));
    println!("{}", bundler.generate_gitignore());

    println!("\nBundle Contents:");
    println!("  ✓ WASM binary (optimized)");
    println!("  ✓ JS glue code (minified)");
    println!("  ✓ TypeScript definitions");
    println!("  ✓ package.json");
    println!("  ✓ README.md");
    println!("  ✓ index.html (for web target)");
    println!("  ✓ .gitignore");
    println!("  ✓ Compressed files (.gz, .br)");
}
