// Multi-Target Deployment Demo
// Demonstrates building for Browser, Node.js, and Edge runtimes

use portalis_transpiler::multi_target_builder::{
    MultiTargetBuilder, TargetConfig, TargetRuntime,
};

fn main() {
    println!("=== Multi-Target Deployment Demo ===\n");
    println!("Demonstrates: Browser, Node.js, Cloudflare Workers, Deno Deploy, Vercel Edge\n");
    println!("{}", "=".repeat(80));

    // Demo 1: Target Runtime Overview
    demo_target_overview();
    println!("\n{}", "=".repeat(80));

    // Demo 2: Browser Deployment
    demo_browser_deployment();
    println!("\n{}", "=".repeat(80));

    // Demo 3: Node.js Deployment
    demo_nodejs_deployment();
    println!("\n{}", "=".repeat(80));

    // Demo 4: Cloudflare Workers Deployment
    demo_cloudflare_workers();
    println!("\n{}", "=".repeat(80));

    // Demo 5: Deno Deploy
    demo_deno_deploy();
    println!("\n{}", "=".repeat(80));

    // Demo 6: Vercel Edge Functions
    demo_vercel_edge();
    println!("\n{}", "=".repeat(80));

    // Demo 7: Multi-Target Build Pipeline
    demo_multi_target_pipeline();
    println!("\n{}", "=".repeat(80));

    // Demo 8: Compatibility Report
    demo_compatibility_report();
    println!("\n{}", "=".repeat(80));

    println!("\n🎉 Multi-target deployment demonstration complete!");
}

fn demo_target_overview() {
    println!("\n=== Demo 1: Target Runtime Overview ===\n");

    let targets = vec![
        TargetRuntime::Browser,
        TargetRuntime::NodeJs,
        TargetRuntime::CloudflareWorkers,
        TargetRuntime::DenoDeploy,
        TargetRuntime::VercelEdge,
        TargetRuntime::AwsLambdaEdge,
        TargetRuntime::FastlyCompute,
    ];

    println!("Supported Target Runtimes:\n");
    println!("{:<25} {:<15} {:<15} {:<20}", "Runtime", "Streaming", "Dynamic Import", "Max Bundle Size");
    println!("{}", "-".repeat(75));

    for target in targets {
        let streaming = if target.supports_streaming() { "✓" } else { "✗" };
        let dynamic_import = if target.supports_dynamic_import() { "✓" } else { "✗" };
        let max_size = target.max_bundle_size()
            .map(|s| format!("{} KB", s / 1024))
            .unwrap_or_else(|| "No limit".to_string());

        println!("{:<25} {:<15} {:<15} {:<20}",
            target.display_name(),
            streaming,
            dynamic_import,
            max_size
        );
    }
}

fn demo_browser_deployment() {
    println!("\n=== Demo 2: Browser Deployment ===\n");

    let config = TargetConfig::browser();
    let builder = MultiTargetBuilder::new();

    println!("Browser Configuration:");
    println!("{}", "-".repeat(80));
    println!("  Runtime: {}", config.runtime.display_name());
    println!("  Optimize Size: {}", config.optimize_size);
    println!("  Streaming: {}", config.enable_streaming);
    println!("  Code Splitting: {}", config.enable_code_splitting);
    println!("  Features: {}", config.features.join(", "));
    println!("{}", "-".repeat(80));

    let build = builder.build_target("my_module", 300_000, &config);

    println!("\nBuild Output:");
    println!("  Output Directory: {}", build.output_dir);
    println!("  Total Size: {} KB", build.total_size / 1024);
    println!("\n  Files:");
    for file in &build.files {
        let required = if file.required { "required" } else { "optional" };
        println!("    {} - {} ({} bytes, {})",
            file.path,
            file.file_type,
            file.size,
            required
        );
    }

    println!("\nGenerated Glue Code:");
    println!("{}", "-".repeat(80));
    let glue = builder.generate_browser_glue("my_module");
    println!("{}", glue);
    println!("{}", "-".repeat(80));

    println!("\nUsage:");
    println!(r#"  <script type="module">
    import {{ initWasm }} from './my_module.js';

    const wasm = await initWasm();
    // Use WASM functions
  </script>"#);
}

fn demo_nodejs_deployment() {
    println!("\n=== Demo 3: Node.js Deployment ===\n");

    let config = TargetConfig::nodejs();
    let builder = MultiTargetBuilder::new();

    println!("Node.js Configuration:");
    println!("{}", "-".repeat(80));
    println!("  Runtime: {}", config.runtime.display_name());
    println!("  Optimize Size: {}", config.optimize_size);
    println!("  Features: {}", config.features.join(", "));
    println!("{}", "-".repeat(80));

    let build = builder.build_target("my_module", 350_000, &config);

    println!("\nBuild Output:");
    println!("  Output Directory: {}", build.output_dir);
    println!("  Total Size: {} KB\n", build.total_size / 1024);

    println!("Generated Glue Code:");
    println!("{}", "-".repeat(80));
    let glue = builder.generate_nodejs_glue("my_module");
    println!("{}", glue);
    println!("{}", "-".repeat(80));

    println!("\nUsage:");
    println!(r#"  const {{ init, initSync }} = require('./my_module');

  // Async initialization
  await init();

  // Sync initialization (faster)
  initSync('./my_module.wasm');"#);
}

fn demo_cloudflare_workers() {
    println!("\n=== Demo 4: Cloudflare Workers Deployment ===\n");

    let config = TargetConfig::cloudflare_workers();
    let builder = MultiTargetBuilder::new();

    println!("Cloudflare Workers Configuration:");
    println!("{}", "-".repeat(80));
    println!("  Runtime: {}", config.runtime.display_name());
    println!("  Max Bundle Size: {} KB", config.runtime.max_bundle_size().unwrap() / 1024);
    println!("  Streaming: {}", config.enable_streaming);
    println!("  Features: {}", config.features.join(", "));
    println!("{}", "-".repeat(80));

    let build = builder.build_target("my_module", 250_000, &config);

    println!("\nBuild Output:");
    println!("  Total Size: {} KB", build.total_size / 1024);
    println!("  Within Size Limit: {}",
        if build.total_size < config.runtime.max_bundle_size().unwrap() {
            "✓"
        } else {
            "✗ EXCEEDS LIMIT"
        }
    );

    println!("\nGenerated wrangler.toml:");
    println!("{}", "-".repeat(80));
    let wrangler = builder.generate_wrangler_config("my_module");
    println!("{}", wrangler);
    println!("{}", "-".repeat(80));

    println!("\nGenerated Worker Code:");
    println!("{}", "-".repeat(80));
    let glue = builder.generate_cloudflare_workers_glue("my_module");
    println!("{}", glue);
    println!("{}", "-".repeat(80));

    println!("\nDeployment:");
    println!("  1. wrangler login");
    println!("  2. wrangler deploy");
}

fn demo_deno_deploy() {
    println!("\n=== Demo 5: Deno Deploy ===\n");

    let config = TargetConfig::deno_deploy();
    let builder = MultiTargetBuilder::new();

    println!("Deno Deploy Configuration:");
    println!("{}", "-".repeat(80));
    println!("  Runtime: {}", config.runtime.display_name());
    println!("  Streaming: {}", config.enable_streaming);
    println!("  Dynamic Import: {}", config.runtime.supports_dynamic_import());
    println!("  Features: {}", config.features.join(", "));
    println!("{}", "-".repeat(80));

    println!("\nGenerated deno.json:");
    println!("{}", "-".repeat(80));
    let deno_config = builder.generate_deno_config("my_module");
    println!("{}", deno_config);
    println!("{}", "-".repeat(80));

    println!("\nGenerated Handler Code:");
    println!("{}", "-".repeat(80));
    let glue = builder.generate_deno_deploy_glue("my_module");
    println!("{}", glue);
    println!("{}", "-".repeat(80));

    println!("\nDeployment:");
    println!("  deployctl deploy --project=my-project index.ts");
}

fn demo_vercel_edge() {
    println!("\n=== Demo 6: Vercel Edge Functions ===\n");

    let config = TargetConfig::vercel_edge();
    let builder = MultiTargetBuilder::new();

    println!("Vercel Edge Configuration:");
    println!("{}", "-".repeat(80));
    println!("  Runtime: {}", config.runtime.display_name());
    println!("  Max Bundle Size: {} KB", config.runtime.max_bundle_size().unwrap() / 1024);
    println!("  Streaming: {}", config.enable_streaming);
    println!("{}", "-".repeat(80));

    println!("\nGenerated Middleware Code:");
    println!("{}", "-".repeat(80));
    let glue = builder.generate_vercel_edge_glue("my_module");
    println!("{}", glue);
    println!("{}", "-".repeat(80));

    println!("\nDeployment:");
    println!("  1. Place WASM in public/ directory");
    println!("  2. vercel --prod");
}

fn demo_multi_target_pipeline() {
    println!("\n=== Demo 7: Multi-Target Build Pipeline ===\n");

    let mut builder = MultiTargetBuilder::new();

    // Add all targets
    builder.add_target(TargetConfig::browser());
    builder.add_target(TargetConfig::nodejs());
    builder.add_target(TargetConfig::cloudflare_workers());
    builder.add_target(TargetConfig::deno_deploy());
    builder.add_target(TargetConfig::vercel_edge());

    println!("Building for 5 targets...\n");

    let builds = builder.build_all("universal_module", 280_000);

    println!("Build Results:");
    println!("{}", "=".repeat(80));
    println!("{:<25} {:<20} {:<15} {:<20}", "Target", "Output Dir", "Size", "Files");
    println!("{}", "-".repeat(80));

    for build in &builds {
        println!("{:<25} {:<20} {:<15} {:<20}",
            build.target.display_name(),
            build.output_dir,
            format!("{} KB", build.total_size / 1024),
            build.files.len()
        );
    }

    println!("\n{}", "=".repeat(80));

    println!("\nUnified Deployment Script:");
    println!("{}", "-".repeat(80));
    println!(r#"#!/bin/bash
# Multi-target deployment script

set -e

echo "=== Multi-Target WASM Deployment ==="

# Build WASM once
echo "Building WASM..."
cargo build --profile wasm-size --target wasm32-unknown-unknown
WASM_FILE=target/wasm32-unknown-unknown/wasm-size/module.wasm

# Optimize
echo "Optimizing..."
wasm-opt -O4 $WASM_FILE -o module_opt.wasm

# Deploy to each target
echo "Deploying to Browser..."
cp module_opt.wasm dist/browser/

echo "Deploying to Node.js..."
cp module_opt.wasm dist/nodejs/

echo "Deploying to Cloudflare Workers..."
wrangler deploy

echo "Deploying to Deno Deploy..."
deployctl deploy --project=my-project index.ts

echo "Deploying to Vercel Edge..."
cp module_opt.wasm public/
vercel --prod

echo "✓ All deployments complete!"
"#);
    println!("{}", "-".repeat(80));
}

fn demo_compatibility_report() {
    println!("\n=== Demo 8: Compatibility Report ===\n");

    let mut builder = MultiTargetBuilder::new();
    builder.add_target(TargetConfig::browser());
    builder.add_target(TargetConfig::nodejs());
    builder.add_target(TargetConfig::cloudflare_workers());
    builder.add_target(TargetConfig::deno_deploy());
    builder.add_target(TargetConfig::vercel_edge());

    let builds = builder.build_all("compat_module", 320_000);

    let report = builder.generate_compatibility_report(&builds);

    println!("{}", report);

    println!("\n## Size Optimization by Target\n");
    println!("| Target | Original | Optimized | Reduction |");
    println!("|--------|----------|-----------|-----------|");

    for build in &builds {
        let original = 320_000_u64;
        let optimized = build.total_size;
        let reduction = ((original - optimized) as f64 / original as f64) * 100.0;

        println!("| {} | {} KB | {} KB | {:.1}% |",
            build.target.display_name(),
            original / 1024,
            optimized / 1024,
            reduction
        );
    }

    println!("\n## Deployment Commands\n");
    for build in &builds {
        println!("### {}\n", build.target.display_name());
        let script = builder.generate_deploy_script(build.target);
        println!("```bash");
        println!("{}", script.trim());
        println!("```\n");
    }
}
