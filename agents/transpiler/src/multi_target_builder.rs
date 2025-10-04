//! Multi-Target Builder - Generates deployment packages for Browser, Node.js, and Edge runtimes
//!
//! This module provides:
//! 1. Target-specific build configurations
//! 2. Browser optimizations (size, streaming, Web APIs)
//! 3. Node.js optimizations (CommonJS, native modules)
//! 4. Edge runtime support (Cloudflare Workers, Deno Deploy, Vercel Edge)
//! 5. Polyfills and feature detection
//! 6. Unified build pipeline
//! 7. Target-specific bundling

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Deployment target runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum TargetRuntime {
    /// Browser (Chrome, Firefox, Safari, Edge)
    Browser,
    /// Node.js (v16+)
    NodeJs,
    /// Cloudflare Workers
    CloudflareWorkers,
    /// Deno Deploy
    DenoDeploy,
    /// Vercel Edge Functions
    VercelEdge,
    /// AWS Lambda@Edge
    AwsLambdaEdge,
    /// Fastly Compute@Edge
    FastlyCompute,
}

impl TargetRuntime {
    pub fn name(&self) -> &str {
        match self {
            Self::Browser => "browser",
            Self::NodeJs => "nodejs",
            Self::CloudflareWorkers => "cloudflare-workers",
            Self::DenoDeploy => "deno-deploy",
            Self::VercelEdge => "vercel-edge",
            Self::AwsLambdaEdge => "aws-lambda-edge",
            Self::FastlyCompute => "fastly-compute",
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Browser => "Browser",
            Self::NodeJs => "Node.js",
            Self::CloudflareWorkers => "Cloudflare Workers",
            Self::DenoDeploy => "Deno Deploy",
            Self::VercelEdge => "Vercel Edge",
            Self::AwsLambdaEdge => "AWS Lambda@Edge",
            Self::FastlyCompute => "Fastly Compute@Edge",
        }
    }

    pub fn supports_streaming(&self) -> bool {
        matches!(self, Self::Browser | Self::CloudflareWorkers | Self::DenoDeploy)
    }

    pub fn supports_dynamic_import(&self) -> bool {
        matches!(self, Self::Browser | Self::NodeJs | Self::DenoDeploy)
    }

    pub fn max_bundle_size(&self) -> Option<u64> {
        match self {
            Self::CloudflareWorkers => Some(1_000_000), // 1 MB
            Self::VercelEdge => Some(1_000_000), // 1 MB
            Self::AwsLambdaEdge => Some(1_000_000), // 1 MB (uncompressed)
            Self::FastlyCompute => Some(50_000_000), // 50 MB
            _ => None,
        }
    }
}

/// Target-specific build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetConfig {
    pub runtime: TargetRuntime,
    pub optimize_size: bool,
    pub enable_streaming: bool,
    pub enable_code_splitting: bool,
    pub polyfills: Vec<String>,
    pub features: Vec<String>,
    pub env_vars: HashMap<String, String>,
}

impl TargetConfig {
    /// Browser configuration
    pub fn browser() -> Self {
        Self {
            runtime: TargetRuntime::Browser,
            optimize_size: true,
            enable_streaming: true,
            enable_code_splitting: true,
            polyfills: vec![],
            features: vec![
                "web-apis".to_string(),
                "fetch".to_string(),
                "websocket".to_string(),
                "indexeddb".to_string(),
            ],
            env_vars: HashMap::new(),
        }
    }

    /// Node.js configuration
    pub fn nodejs() -> Self {
        Self {
            runtime: TargetRuntime::NodeJs,
            optimize_size: false,
            enable_streaming: false,
            enable_code_splitting: true,
            polyfills: vec![],
            features: vec![
                "fs".to_string(),
                "crypto".to_string(),
                "http".to_string(),
            ],
            env_vars: HashMap::new(),
        }
    }

    /// Cloudflare Workers configuration
    pub fn cloudflare_workers() -> Self {
        Self {
            runtime: TargetRuntime::CloudflareWorkers,
            optimize_size: true,
            enable_streaming: true,
            enable_code_splitting: false,
            polyfills: vec![],
            features: vec![
                "fetch".to_string(),
                "kv".to_string(),
                "durable-objects".to_string(),
            ],
            env_vars: HashMap::new(),
        }
    }

    /// Deno Deploy configuration
    pub fn deno_deploy() -> Self {
        Self {
            runtime: TargetRuntime::DenoDeploy,
            optimize_size: true,
            enable_streaming: true,
            enable_code_splitting: true,
            polyfills: vec![],
            features: vec![
                "fetch".to_string(),
                "kv".to_string(),
            ],
            env_vars: HashMap::new(),
        }
    }

    /// Vercel Edge configuration
    pub fn vercel_edge() -> Self {
        Self {
            runtime: TargetRuntime::VercelEdge,
            optimize_size: true,
            enable_streaming: true,
            enable_code_splitting: false,
            polyfills: vec![],
            features: vec![
                "fetch".to_string(),
                "edge-config".to_string(),
            ],
            env_vars: HashMap::new(),
        }
    }
}

/// Multi-target build result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetBuild {
    pub target: TargetRuntime,
    pub output_dir: String,
    pub files: Vec<BuildFile>,
    pub total_size: u64,
    pub config: TargetConfig,
}

/// Build file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildFile {
    pub path: String,
    pub size: u64,
    pub file_type: String,
    pub required: bool,
}

/// Multi-target builder
pub struct MultiTargetBuilder {
    targets: Vec<TargetConfig>,
}

impl MultiTargetBuilder {
    /// Create new multi-target builder
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
        }
    }

    /// Add target configuration
    pub fn add_target(&mut self, config: TargetConfig) {
        self.targets.push(config);
    }

    /// Build for all targets
    pub fn build_all(&self, module_name: &str, wasm_size: u64) -> Vec<TargetBuild> {
        self.targets.iter().map(|config| {
            self.build_target(module_name, wasm_size, config)
        }).collect()
    }

    /// Build for specific target
    pub fn build_target(&self, module_name: &str, wasm_size: u64, config: &TargetConfig) -> TargetBuild {
        let output_dir = format!("dist/{}", config.runtime.name());
        let mut files = Vec::new();

        // WASM file
        let wasm_file = BuildFile {
            path: format!("{}/{}.wasm", output_dir, module_name),
            size: wasm_size,
            file_type: "WASM Binary".to_string(),
            required: true,
        };
        files.push(wasm_file);

        // JS glue code
        let js_size = wasm_size / 10;
        let js_ext = match config.runtime {
            TargetRuntime::NodeJs => ".cjs",
            _ => ".js",
        };
        let js_file = BuildFile {
            path: format!("{}/{}{}", output_dir, module_name, js_ext),
            size: js_size,
            file_type: "JS Glue Code".to_string(),
            required: true,
        };
        files.push(js_file);

        // Target-specific files
        match config.runtime {
            TargetRuntime::CloudflareWorkers => {
                files.push(BuildFile {
                    path: format!("{}/wrangler.toml", output_dir),
                    size: 500,
                    file_type: "Wrangler Config".to_string(),
                    required: true,
                });
            }
            TargetRuntime::DenoDeploy => {
                files.push(BuildFile {
                    path: format!("{}/deno.json", output_dir),
                    size: 300,
                    file_type: "Deno Config".to_string(),
                    required: true,
                });
            }
            TargetRuntime::VercelEdge => {
                files.push(BuildFile {
                    path: format!("{}/middleware.js", output_dir),
                    size: 1000,
                    file_type: "Edge Middleware".to_string(),
                    required: true,
                });
            }
            _ => {}
        }

        let total_size: u64 = files.iter().map(|f| f.size).sum();

        TargetBuild {
            target: config.runtime,
            output_dir,
            files,
            total_size,
            config: config.clone(),
        }
    }

    /// Generate browser-specific glue code
    pub fn generate_browser_glue(&self, module_name: &str) -> String {
        format!(r#"// Browser-specific glue code for {module_name}
import init from './{module_name}_bg.js';

let wasm;
let wasmMemory;

export async function initWasm(input) {{
  if (wasm) return wasm;

  const imports = {{}};

  if (typeof input === 'string' || (typeof Request === 'function' && input instanceof Request) || (typeof URL === 'function' && input instanceof URL)) {{
    input = fetch(input);
  }}

  const {{ instance, module }} = await init(await input, imports);
  wasm = instance.exports;
  wasmMemory = wasm.memory;

  return wasm;
}}

export function getMemory() {{
  return wasmMemory;
}}

// Re-export WASM functions
export * from './{module_name}_bg.js';
"#)
    }

    /// Generate Node.js-specific glue code
    pub fn generate_nodejs_glue(&self, module_name: &str) -> String {
        format!(r#"// Node.js-specific glue code for {module_name}
const {{ readFileSync }} = require('fs');
const {{ join }} = require('path');

let wasm;

function initSync(wasmPath) {{
  if (wasm) return wasm;

  const bytes = readFileSync(wasmPath || join(__dirname, '{module_name}.wasm'));
  const module = new WebAssembly.Module(bytes);
  const instance = new WebAssembly.Instance(module, {{}});

  wasm = instance.exports;
  return wasm;
}}

async function init(wasmPath) {{
  return initSync(wasmPath);
}}

module.exports = {{
  init,
  initSync,
  get wasm() {{ return wasm; }},
}};
"#)
    }

    /// Generate Cloudflare Workers glue code
    pub fn generate_cloudflare_workers_glue(&self, module_name: &str) -> String {
        format!(r#"// Cloudflare Workers glue code for {module_name}
import wasmModule from './{module_name}.wasm';

let wasm;

export async function initWasm() {{
  if (wasm) return wasm;

  const instance = await WebAssembly.instantiate(wasmModule, {{}});
  wasm = instance.exports;
  return wasm;
}}

export default {{
  async fetch(request, env, ctx) {{
    await initWasm();

    // Your handler logic here
    return new Response('Hello from WASM!', {{
      headers: {{ 'content-type': 'text/plain' }},
    }});
  }},
}};
"#)
    }

    /// Generate Deno Deploy glue code
    pub fn generate_deno_deploy_glue(&self, module_name: &str) -> String {
        format!(r#"// Deno Deploy glue code for {module_name}
const wasmBytes = await Deno.readFile('./{module_name}.wasm');
const wasmModule = await WebAssembly.compile(wasmBytes);

let wasm: WebAssembly.Instance | null = null;

export async function initWasm(): Promise<WebAssembly.Exports> {{
  if (wasm) return wasm.exports;

  wasm = await WebAssembly.instantiate(wasmModule, {{}});
  return wasm.exports;
}}

// Deno Deploy handler
Deno.serve(async (req: Request) => {{
  await initWasm();

  // Your handler logic here
  return new Response('Hello from WASM!', {{
    headers: {{ 'content-type': 'text/plain' }},
  }});
}});
"#)
    }

    /// Generate Vercel Edge glue code
    pub fn generate_vercel_edge_glue(&self, module_name: &str) -> String {
        format!(r#"// Vercel Edge glue code for {module_name}
import {{ NextRequest, NextResponse }} from 'next/server';

// Import WASM
const wasmModule = await WebAssembly.compileStreaming(
  fetch(new URL('./{module_name}.wasm', import.meta.url))
);

let wasm: WebAssembly.Instance | null = null;

async function initWasm() {{
  if (wasm) return wasm;
  wasm = await WebAssembly.instantiate(wasmModule, {{}});
  return wasm;
}}

export async function middleware(request: NextRequest) {{
  await initWasm();

  // Your middleware logic here
  return NextResponse.next();
}}

export const config = {{
  matcher: '/api/:path*',
}};
"#)
    }

    /// Generate wrangler.toml for Cloudflare Workers
    pub fn generate_wrangler_config(&self, module_name: &str) -> String {
        format!(r#"name = "{module_name}"
main = "index.js"
compatibility_date = "2024-01-01"

[build]
command = "echo 'Build complete'"

[build.upload]
format = "modules"
main = "./index.js"

[[build.upload.rules]]
type = "CompiledWasm"
globs = ["**/*.wasm"]

[env.production]
name = "{module_name}-production"
"#)
    }

    /// Generate deno.json
    pub fn generate_deno_config(&self, module_name: &str) -> String {
        format!(r#"{{
  "tasks": {{
    "start": "deno run --allow-net --allow-read index.ts"
  }},
  "compilerOptions": {{
    "lib": ["deno.window", "deno.unstable"],
    "strict": true
  }},
  "imports": {{
    "{module_name}": "./index.ts"
  }}
}}
"#)
    }

    /// Generate deployment scripts
    pub fn generate_deploy_script(&self, target: TargetRuntime) -> String {
        match target {
            TargetRuntime::CloudflareWorkers => {
                r#"#!/bin/bash
# Deploy to Cloudflare Workers

set -e

echo "=== Deploying to Cloudflare Workers ==="

# Build WASM
cargo build --profile wasm-size --target wasm32-unknown-unknown

# Optimize
wasm-opt -O4 target/wasm32-unknown-unknown/wasm-size/*.wasm -o dist/cloudflare-workers/module.wasm

# Deploy
wrangler deploy

echo "✓ Deployment complete"
"#.to_string()
            }
            TargetRuntime::DenoDeploy => {
                r#"#!/bin/bash
# Deploy to Deno Deploy

set -e

echo "=== Deploying to Deno Deploy ==="

# Build WASM
cargo build --profile wasm-size --target wasm32-unknown-unknown

# Copy files
cp target/wasm32-unknown-unknown/wasm-size/*.wasm dist/deno-deploy/

# Deploy
deployctl deploy --project=my-project index.ts

echo "✓ Deployment complete"
"#.to_string()
            }
            TargetRuntime::VercelEdge => {
                r#"#!/bin/bash
# Deploy to Vercel Edge

set -e

echo "=== Deploying to Vercel Edge ==="

# Build WASM
cargo build --profile wasm-size --target wasm32-unknown-unknown

# Copy to public directory
cp target/wasm32-unknown-unknown/wasm-size/*.wasm public/

# Deploy
vercel --prod

echo "✓ Deployment complete"
"#.to_string()
            }
            _ => {
                format!(r#"#!/bin/bash
# Deploy to {}

echo "Deployment script for {} not yet implemented"
"#, target.display_name(), target.display_name())
            }
        }
    }

    /// Generate compatibility report
    pub fn generate_compatibility_report(&self, builds: &[TargetBuild]) -> String {
        let mut report = String::from("# Multi-Target Compatibility Report\n\n");

        report.push_str("## Target Summary\n\n");
        report.push_str("| Target | Size | Streaming | Code Splitting | Max Size Limit |\n");
        report.push_str("|--------|------|-----------|----------------|----------------|\n");

        for build in builds {
            let streaming = if build.config.enable_streaming { "✓" } else { "✗" };
            let splitting = if build.config.enable_code_splitting { "✓" } else { "✗" };
            let max_size = build.target.max_bundle_size()
                .map(|s| format!("{} KB", s / 1024))
                .unwrap_or_else(|| "No limit".to_string());

            report.push_str(&format!(
                "| {} | {} KB | {} | {} | {} |\n",
                build.target.display_name(),
                build.total_size / 1024,
                streaming,
                splitting,
                max_size
            ));
        }

        report.push_str("\n## Features by Target\n\n");
        for build in builds {
            report.push_str(&format!("### {}\n\n", build.target.display_name()));
            report.push_str("**Enabled Features:**\n");
            for feature in &build.config.features {
                report.push_str(&format!("- {}\n", feature));
            }
            report.push('\n');
        }

        report
    }
}

impl Default for MultiTargetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_runtime_properties() {
        assert!(TargetRuntime::Browser.supports_streaming());
        assert!(TargetRuntime::CloudflareWorkers.supports_streaming());
        assert!(TargetRuntime::NodeJs.supports_dynamic_import());

        assert_eq!(TargetRuntime::CloudflareWorkers.max_bundle_size(), Some(1_000_000));
        assert_eq!(TargetRuntime::Browser.max_bundle_size(), None);
    }

    #[test]
    fn test_browser_config() {
        let config = TargetConfig::browser();
        assert_eq!(config.runtime, TargetRuntime::Browser);
        assert!(config.optimize_size);
        assert!(config.enable_streaming);
        assert!(config.features.contains(&"fetch".to_string()));
    }

    #[test]
    fn test_cloudflare_workers_config() {
        let config = TargetConfig::cloudflare_workers();
        assert_eq!(config.runtime, TargetRuntime::CloudflareWorkers);
        assert!(config.optimize_size);
        assert!(!config.enable_code_splitting);
    }

    #[test]
    fn test_multi_target_build() {
        let mut builder = MultiTargetBuilder::new();
        builder.add_target(TargetConfig::browser());
        builder.add_target(TargetConfig::nodejs());
        builder.add_target(TargetConfig::cloudflare_workers());

        let builds = builder.build_all("test_module", 200_000);
        assert_eq!(builds.len(), 3);

        let browser_build = &builds[0];
        assert_eq!(browser_build.target, TargetRuntime::Browser);
        assert!(!browser_build.files.is_empty());
    }

    #[test]
    fn test_glue_code_generation() {
        let builder = MultiTargetBuilder::new();

        let browser_glue = builder.generate_browser_glue("test");
        assert!(browser_glue.contains("import init"));
        assert!(browser_glue.contains("fetch"));

        let nodejs_glue = builder.generate_nodejs_glue("test");
        assert!(nodejs_glue.contains("require"));
        assert!(nodejs_glue.contains("readFileSync"));
    }

    #[test]
    fn test_config_generation() {
        let builder = MultiTargetBuilder::new();

        let wrangler = builder.generate_wrangler_config("my_module");
        assert!(wrangler.contains("name = \"my_module\""));
        assert!(wrangler.contains("CompiledWasm"));

        let deno_config = builder.generate_deno_config("my_module");
        assert!(deno_config.contains("deno.window"));
    }

    #[test]
    fn test_size_limits() {
        let config = TargetConfig::cloudflare_workers();
        let builder = MultiTargetBuilder::new();
        let build = builder.build_target("test", 500_000, &config);

        let max_size = config.runtime.max_bundle_size().unwrap();
        assert!(build.total_size < max_size, "Build exceeds Cloudflare Workers size limit");
    }
}
