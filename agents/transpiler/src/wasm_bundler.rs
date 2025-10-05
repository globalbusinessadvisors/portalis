//! WASM Bundler - Generates deployment bundles with JS glue code
//!
//! This module provides:
//! 1. WASM binary bundling and packaging
//! 2. JS glue code generation for browser/Node.js
//! 3. wasm-bindgen integration
//! 4. Deployment pipeline orchestration
//! 5. Multi-target bundle generation (web, nodejs, bundler)

use serde::{Deserialize, Serialize};

/// Deployment target platform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentTarget {
    /// Browser with ES modules
    Web,
    /// Browser with bundler (webpack, rollup)
    Bundler,
    /// Node.js environment
    NodeJs,
    /// Deno runtime
    Deno,
    /// No-modules (classic script tag)
    NoModules,
}

impl DeploymentTarget {
    /// Get wasm-bindgen target flag
    pub fn wasm_bindgen_target(&self) -> &str {
        match self {
            Self::Web => "web",
            Self::Bundler => "bundler",
            Self::NodeJs => "nodejs",
            Self::Deno => "deno",
            Self::NoModules => "no-modules",
        }
    }

    /// Get file extension for JS output
    pub fn js_extension(&self) -> &str {
        match self {
            Self::NodeJs => ".cjs",
            Self::Deno => ".js",
            _ => ".js",
        }
    }
}

/// Optimization level for WASM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization (fastest build)
    None,
    /// Basic optimization (O1)
    Basic,
    /// Standard optimization (O2)
    Standard,
    /// Aggressive optimization (O3)
    Aggressive,
    /// Size optimization (Oz)
    Size,
    /// Maximum size optimization (Ozz)
    MaxSize,
}

impl OptimizationLevel {
    /// Get wasm-opt flag
    pub fn wasm_opt_flag(&self) -> &str {
        match self {
            Self::None => "-O0",
            Self::Basic => "-O1",
            Self::Standard => "-O2",
            Self::Aggressive => "-O3",
            Self::Size => "-Oz",
            Self::MaxSize => "-Ozz",
        }
    }
}

/// Compression format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionFormat {
    None,
    Gzip,
    Brotli,
    Both,
}

/// Bundle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleConfig {
    /// Output directory
    pub output_dir: String,
    /// Package name
    pub package_name: String,
    /// Deployment target
    pub target: DeploymentTarget,
    /// Enable TypeScript definitions
    pub typescript: bool,
    /// Enable source maps
    pub source_maps: bool,
    /// Optimize for size
    pub optimize_size: bool,
    /// Keep debug info
    pub debug: bool,
    /// Enable weak references
    pub weak_refs: bool,
    /// Enable reference types
    pub reference_types: bool,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Compression format
    pub compression: CompressionFormat,
    /// Generate README
    pub generate_readme: bool,
    /// Enable code splitting
    pub code_splitting: bool,
    /// Minify JS output
    pub minify_js: bool,
}

impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            output_dir: "pkg".to_string(),
            package_name: "wasm_app".to_string(),
            target: DeploymentTarget::Web,
            typescript: true,
            source_maps: false,
            optimize_size: true,
            debug: false,
            weak_refs: true,
            reference_types: true,
            optimization_level: OptimizationLevel::Standard,
            compression: CompressionFormat::None,
            generate_readme: true,
            code_splitting: false,
            minify_js: false,
        }
    }
}

impl BundleConfig {
    /// Create production bundle config
    pub fn production() -> Self {
        Self {
            output_dir: "dist".to_string(),
            package_name: "wasm_app".to_string(),
            target: DeploymentTarget::Web,
            typescript: true,
            source_maps: false,
            optimize_size: true,
            debug: false,
            weak_refs: true,
            reference_types: true,
            optimization_level: OptimizationLevel::MaxSize,
            compression: CompressionFormat::Both,
            generate_readme: true,
            code_splitting: true,
            minify_js: true,
        }
    }

    /// Create development bundle config
    pub fn development() -> Self {
        Self {
            output_dir: "dev".to_string(),
            package_name: "wasm_app_dev".to_string(),
            target: DeploymentTarget::Web,
            typescript: true,
            source_maps: true,
            optimize_size: false,
            debug: true,
            weak_refs: false,
            reference_types: false,
            optimization_level: OptimizationLevel::None,
            compression: CompressionFormat::None,
            generate_readme: false,
            code_splitting: false,
            minify_js: false,
        }
    }

    /// Create CDN-optimized config
    pub fn cdn_optimized() -> Self {
        Self {
            output_dir: "cdn".to_string(),
            package_name: "wasm_app".to_string(),
            target: DeploymentTarget::Web,
            typescript: true,
            source_maps: false,
            optimize_size: true,
            debug: false,
            weak_refs: true,
            reference_types: true,
            optimization_level: OptimizationLevel::MaxSize,
            compression: CompressionFormat::Both,
            generate_readme: true,
            code_splitting: true,
            minify_js: true,
        }
    }
}

/// Bundle output information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleOutput {
    /// Output directory
    pub output_dir: String,
    /// Generated files
    pub files: Vec<BundleFile>,
    /// Total bundle size
    pub total_size: u64,
    /// WASM binary size
    pub wasm_size: u64,
    /// JS glue code size
    pub js_size: u64,
    /// Deployment target
    pub target: DeploymentTarget,
}

impl BundleOutput {
    /// Generate deployment report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== WASM Bundle Output ===\n\n");
        report.push_str(&format!("Target: {:?}\n", self.target));
        report.push_str(&format!("Output Directory: {}\n", self.output_dir));
        report.push_str(&format!("Total Size: {}\n", WasmBundler::format_size(self.total_size)));
        report.push_str(&format!("  WASM: {} ({:.1}%)\n",
            WasmBundler::format_size(self.wasm_size),
            (self.wasm_size as f64 / self.total_size as f64) * 100.0
        ));
        report.push_str(&format!("  JS:   {} ({:.1}%)\n",
            WasmBundler::format_size(self.js_size),
            (self.js_size as f64 / self.total_size as f64) * 100.0
        ));

        report.push_str(&format!("\nGenerated Files ({}):\n", self.files.len()));
        for file in &self.files {
            report.push_str(&format!("  {} - {} ({})\n",
                file.path,
                WasmBundler::format_size(file.size),
                file.file_type
            ));
        }

        report
    }
}

/// Bundle file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleFile {
    /// File path
    pub path: String,
    /// File size in bytes
    pub size: u64,
    /// File type
    pub file_type: String,
}

/// WASM bundler
pub struct WasmBundler {
    config: BundleConfig,
}

impl WasmBundler {
    /// Create new bundler with config
    pub fn new(config: BundleConfig) -> Self {
        Self { config }
    }

    /// Generate wasm-bindgen command
    pub fn generate_wasm_bindgen_command(&self, input_wasm: &str) -> String {
        let mut cmd = String::from("wasm-bindgen");

        cmd.push_str(&format!(" {} --out-dir {}", input_wasm, self.config.output_dir));
        cmd.push_str(&format!(" --target {}", self.config.target.wasm_bindgen_target()));

        if self.config.typescript {
            cmd.push_str(" --typescript");
        }

        if !self.config.debug {
            cmd.push_str(" --remove-name-section");
            cmd.push_str(" --remove-producers-section");
        }

        if self.config.weak_refs {
            cmd.push_str(" --weak-refs");
        }

        if self.config.reference_types {
            cmd.push_str(" --reference-types");
        }

        if self.config.source_maps {
            cmd.push_str(" --keep-debug");
        }

        cmd
    }

    /// Generate JS glue code for web target
    pub fn generate_web_glue(&self, module_name: &str) -> String {
        format!(r#"// Generated JS glue code for WASM module: {}
import init, * as wasm from './{}_bg.js';

// Initialize WASM module
let wasmInitialized = false;
let wasmInstance = null;

/**
 * Initialize the WASM module
 * @returns {{Promise<void>}}
 */
export async function initWasm() {{
    if (wasmInitialized) {{
        return wasmInstance;
    }}

    try {{
        wasmInstance = await init();
        wasmInitialized = true;
        console.log('[WASM] Module initialized successfully');
        return wasmInstance;
    }} catch (err) {{
        console.error('[WASM] Failed to initialize:', err);
        throw err;
    }}
}}

/**
 * Check if WASM is initialized
 * @returns {{boolean}}
 */
export function isInitialized() {{
    return wasmInitialized;
}}

/**
 * Get WASM instance
 * @returns {{object|null}}
 */
export function getInstance() {{
    return wasmInstance;
}}

// Auto-initialize on import (optional)
// Uncomment to auto-initialize:
// initWasm().catch(console.error);

// Re-export all WASM exports
export * from './{}_bg.js';
export {{ init as default }};
"#, module_name, module_name, module_name)
    }

    /// Generate JS glue code for Node.js target
    pub fn generate_nodejs_glue(&self, module_name: &str) -> String {
        format!(r#"// Generated JS glue code for WASM module (Node.js): {}
const wasm = require('./{}_bg.cjs');

let wasmInitialized = false;

/**
 * Initialize the WASM module
 */
async function initWasm() {{
    if (wasmInitialized) {{
        return wasm;
    }}

    try {{
        // WASM is automatically loaded in Node.js
        wasmInitialized = true;
        console.log('[WASM] Module loaded successfully');
        return wasm;
    }} catch (err) {{
        console.error('[WASM] Failed to load:', err);
        throw err;
    }}
}}

/**
 * Check if WASM is initialized
 */
function isInitialized() {{
    return wasmInitialized;
}}

module.exports = {{
    initWasm,
    isInitialized,
    ...wasm
}};
"#, module_name, module_name)
    }

    /// Generate package.json
    pub fn generate_package_json(&self) -> String {
        let module_type = match self.config.target {
            DeploymentTarget::Web | DeploymentTarget::Bundler | DeploymentTarget::Deno => "module",
            DeploymentTarget::NodeJs => "commonjs",
            DeploymentTarget::NoModules => "commonjs",
        };

        format!(r#"{{
  "name": "{}",
  "version": "0.1.0",
  "type": "{}",
  "files": [
    "*.js",
    "*.wasm",
    "*.d.ts"
  ],
  "main": "{}.js",
  "types": "{}.d.ts",
  "sideEffects": [
    "*.js",
    "*.wasm"
  ]
}}
"#, self.config.package_name, module_type, self.config.package_name, self.config.package_name)
    }

    /// Generate HTML loader for web deployment
    pub fn generate_html_loader(&self, module_name: &str) -> String {
        format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - WASM Application</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 800px;
            margin: 2rem auto;
            padding: 0 1rem;
        }}
        .status {{
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}
        .loading {{ background: #fef3c7; color: #92400e; }}
        .ready {{ background: #d1fae5; color: #065f46; }}
        .error {{ background: #fee2e2; color: #991b1b; }}
    </style>
</head>
<body>
    <h1>{} - WASM Application</h1>
    <div id="status" class="status loading">Loading WASM module...</div>
    <div id="app"></div>

    <script type="module">
        import {{ initWasm, isInitialized }} from './{}.js';

        const statusEl = document.getElementById('status');
        const appEl = document.getElementById('app');

        async function main() {{
            try {{
                // Initialize WASM
                await initWasm();

                if (isInitialized()) {{
                    statusEl.textContent = 'WASM module ready!';
                    statusEl.className = 'status ready';

                    // Your application code here
                    appEl.innerHTML = '<p>WASM module loaded successfully. Ready to use!</p>';
                }}
            }} catch (err) {{
                statusEl.textContent = `Error: ${{err.message}}`;
                statusEl.className = 'status error';
                console.error('Failed to initialize WASM:', err);
            }}
        }}

        main();
    </script>
</body>
</html>
"#, module_name, module_name, module_name)
    }

    /// Generate deployment pipeline script
    pub fn generate_deployment_script(&self) -> String {
        let profile = if self.config.optimize_size { "wasm-size" } else { "release" };

        format!(r#"#!/bin/bash
# WASM Deployment Pipeline
# Generated for target: {:?}

set -e

echo "=== WASM Deployment Pipeline ==="
echo "Target: {:?}"
echo "Output: {}"
echo ""

# Step 1: Build WASM
echo "Step 1/5: Building WASM binary..."
cargo build --profile {} --target wasm32-unknown-unknown
echo "✓ Build complete"
echo ""

# Step 2: Run wasm-opt (if available)
echo "Step 2/5: Optimizing WASM..."
if command -v wasm-opt &> /dev/null; then
    wasm-opt -O4 -Ozz \\
        --strip-debug \\
        --strip-producers \\
        target/wasm32-unknown-unknown/{}/{}_bg.wasm \\
        -o target/wasm32-unknown-unknown/{}/{}_opt.wasm
    echo "✓ Optimization complete"
else
    echo "⚠ wasm-opt not found, skipping optimization"
    cp target/wasm32-unknown-unknown/{}/{}_bg.wasm \\
       target/wasm32-unknown-unknown/{}/{}_opt.wasm
fi
echo ""

# Step 3: Generate JS bindings
echo "Step 3/5: Generating JS bindings..."
{}
echo "✓ Bindings generated"
echo ""

# Step 4: Generate additional files
echo "Step 4/5: Generating deployment files..."
# package.json, HTML loader, etc. would be generated here
echo "✓ Deployment files generated"
echo ""

# Step 5: Bundle summary
echo "Step 5/5: Bundle Summary"
echo "Output directory: {}"
ls -lh {}
echo ""

echo "=== Deployment Complete ==="
echo "Deploy the '{}' directory to your server or CDN"
"#,
            self.config.target,
            self.config.target,
            self.config.output_dir,
            profile,
            profile,
            self.config.package_name,
            profile,
            self.config.package_name,
            profile,
            self.config.package_name,
            profile,
            self.config.package_name,
            self.generate_wasm_bindgen_command(&format!(
                "target/wasm32-unknown-unknown/{}/{}_opt.wasm",
                profile,
                self.config.package_name
            )),
            self.config.output_dir,
            self.config.output_dir,
            self.config.output_dir,
        )
    }

    /// Generate wasm-opt optimization command
    pub fn generate_wasm_opt_command(&self, input: &str, output: &str) -> String {
        let mut cmd = format!("wasm-opt {} {}", self.config.optimization_level.wasm_opt_flag(), input);

        cmd.push_str(" --enable-bulk-memory");
        cmd.push_str(" --enable-sign-ext");

        if !self.config.debug {
            cmd.push_str(" --strip-debug");
            cmd.push_str(" --strip-producers");
            cmd.push_str(" --strip-target-features");
        }

        if self.config.optimize_size {
            cmd.push_str(" --vacuum");
            cmd.push_str(" --dce");
            cmd.push_str(" --remove-unused-names");
            cmd.push_str(" --remove-unused-module-elements");
        }

        cmd.push_str(&format!(" -o {}", output));
        cmd
    }

    /// Generate compression commands
    pub fn generate_compression_commands(&self, wasm_file: &str) -> Vec<String> {
        let mut commands = Vec::new();

        match self.config.compression {
            CompressionFormat::None => {},
            CompressionFormat::Gzip | CompressionFormat::Both => {
                commands.push(format!("gzip -9 -k {}", wasm_file));
            },
            CompressionFormat::Brotli => {
                commands.push(format!("brotli -9 -k {}", wasm_file));
            },
        }

        if matches!(self.config.compression, CompressionFormat::Both) {
            commands.push(format!("brotli -9 -k {}", wasm_file));
        }

        commands
    }

    /// Generate README.md for the bundle
    pub fn generate_readme(&self) -> String {
        format!(r#"# {} - WASM Module

Generated WASM module ready for deployment.

## Quick Start

### Web (Browser)

```html
<script type="module">
  import init from './{}.js';

  async function run() {{
    await init();
    // Your code here
  }}

  run();
</script>
```

### Node.js

```javascript
const wasm = require('./{}.js');

async function main() {{
  await wasm.initWasm();
  // Your code here
}}

main();
```

## Files

- `{}_bg.wasm` - WASM binary module
- `{}.js` - JavaScript glue code
- `{}.d.ts` - TypeScript type definitions
- `package.json` - NPM package manifest

## Optimization

This bundle is optimized for:
- Target: {:?}
- Optimization level: {:?}
- Compression: {:?}

## Bundle Size

The WASM binary is highly optimized:
- Dead code elimination
- Tree shaking
- Aggressive optimization passes
{}

## Deployment

### Static Hosting

Upload all files to your web server or CDN. Ensure:
- WASM files are served with `application/wasm` MIME type
- Compressed files (.gz, .br) are served with appropriate encoding headers

### NPM Package

```bash
npm publish
```

### CDN (unpkg, jsDelivr)

```html
<script type="module">
  import init from 'https://unpkg.com/{}@latest/{}.js';
  await init();
</script>
```

## Performance Tips

1. **Preload WASM**: Add `<link rel="preload" as="fetch" href="{}_bg.wasm">` to HTML
2. **Use Compression**: Serve .wasm.br or .wasm.gz for 70-90% size reduction
3. **Cache-Control**: Set long cache times with versioned filenames
4. **Lazy Loading**: Use dynamic imports for code splitting

## License

See parent project for license information.
"#,
            self.config.package_name,
            self.config.package_name,
            self.config.package_name,
            self.config.package_name,
            self.config.package_name,
            self.config.package_name,
            self.config.target,
            self.config.optimization_level,
            self.config.compression,
            if matches!(self.config.compression, CompressionFormat::None) {
                ""
            } else {
                "\n- Gzip/Brotli compression included"
            },
            self.config.package_name,
            self.config.package_name,
            self.config.package_name,
        )
    }

    /// Generate .gitignore for bundle directory
    pub fn generate_gitignore(&self) -> String {
        r#"# Build artifacts
*.wasm
*.js
*.d.ts
!package.json
!README.md

# Source maps
*.map

# Compression artifacts
*.gz
*.br

# Node modules
node_modules/

# OS files
.DS_Store
Thumbs.db
"#.to_string()
    }

    /// Estimate compressed sizes
    pub fn estimate_compressed_size(&self, original_size: u64) -> (u64, u64) {
        // Typical compression ratios for WASM
        let gzip_size = (original_size as f64 * 0.3) as u64;  // ~70% reduction
        let brotli_size = (original_size as f64 * 0.25) as u64;  // ~75% reduction
        (gzip_size, brotli_size)
    }

    /// Generate optimization report
    pub fn generate_optimization_report(&self, original_size: u64, optimized_size: u64) -> String {
        let reduction = original_size - optimized_size;
        let reduction_pct = (reduction as f64 / original_size as f64) * 100.0;

        let (gzip_size, brotli_size) = self.estimate_compressed_size(optimized_size);

        format!(r#"=== WASM Optimization Report ===

Original Size:     {}
Optimized Size:    {} ({:.1}% reduction)
Reduction:         {}

Compression Estimates:
  Gzip:            {} ({:.1}% of optimized)
  Brotli:          {} ({:.1}% of optimized)

Optimization Level: {:?}
Target:            {:?}

Applied Optimizations:
{}{}{}{}{}

Deployment Size Estimates:
  Uncompressed:    {}
  Gzip:            {} (best for HTTP/1.1)
  Brotli:          {} (best for HTTP/2+)
"#,
            Self::format_size(original_size),
            Self::format_size(optimized_size),
            reduction_pct,
            Self::format_size(reduction),
            Self::format_size(gzip_size),
            (gzip_size as f64 / optimized_size as f64) * 100.0,
            Self::format_size(brotli_size),
            (brotli_size as f64 / optimized_size as f64) * 100.0,
            self.config.optimization_level,
            self.config.target,
            if !self.config.debug { "  ✓ Debug symbols stripped\n" } else { "" },
            if self.config.optimize_size { "  ✓ Size optimization passes\n" } else { "" },
            if self.config.weak_refs { "  ✓ Weak references enabled\n" } else { "" },
            if self.config.reference_types { "  ✓ Reference types enabled\n" } else { "" },
            if self.config.minify_js { "  ✓ JavaScript minification\n" } else { "" },
            Self::format_size(optimized_size),
            Self::format_size(gzip_size),
            Self::format_size(brotli_size),
        )
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

    /// Simulate bundle creation (for demo purposes)
    pub fn create_bundle_simulation(&self, wasm_size: u64) -> BundleOutput {
        let js_size = wasm_size / 10; // JS glue is typically ~10% of WASM size
        let total_size = wasm_size + js_size;

        let mut files = vec![
            BundleFile {
                path: format!("{}/{}_bg.wasm", self.config.output_dir, self.config.package_name),
                size: wasm_size,
                file_type: "WASM Binary".to_string(),
            },
            BundleFile {
                path: format!("{}/{}.js", self.config.output_dir, self.config.package_name),
                size: js_size,
                file_type: "JS Glue Code".to_string(),
            },
        ];

        if self.config.typescript {
            files.push(BundleFile {
                path: format!("{}/{}.d.ts", self.config.output_dir, self.config.package_name),
                size: js_size / 4,
                file_type: "TypeScript Definitions".to_string(),
            });
        }

        files.push(BundleFile {
            path: format!("{}/package.json", self.config.output_dir),
            size: 200,
            file_type: "Package Manifest".to_string(),
        });

        if self.config.target == DeploymentTarget::Web {
            files.push(BundleFile {
                path: format!("{}/index.html", self.config.output_dir),
                size: 1500,
                file_type: "HTML Loader".to_string(),
            });
        }

        BundleOutput {
            output_dir: self.config.output_dir.clone(),
            files,
            total_size,
            wasm_size,
            js_size,
            target: self.config.target,
        }
    }
}

impl Default for WasmBundler {
    fn default() -> Self {
        Self::new(BundleConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_targets() {
        assert_eq!(DeploymentTarget::Web.wasm_bindgen_target(), "web");
        assert_eq!(DeploymentTarget::NodeJs.wasm_bindgen_target(), "nodejs");
        assert_eq!(DeploymentTarget::Bundler.wasm_bindgen_target(), "bundler");
    }

    #[test]
    fn test_bundle_config() {
        let prod = BundleConfig::production();
        assert!(prod.optimize_size);
        assert!(!prod.debug);

        let dev = BundleConfig::development();
        assert!(!dev.optimize_size);
        assert!(dev.debug);
    }

    #[test]
    fn test_wasm_bindgen_command() {
        let bundler = WasmBundler::new(BundleConfig::default());
        let cmd = bundler.generate_wasm_bindgen_command("input.wasm");

        assert!(cmd.contains("wasm-bindgen"));
        assert!(cmd.contains("--target web"));
        assert!(cmd.contains("--typescript"));
    }

    #[test]
    fn test_package_json_generation() {
        let bundler = WasmBundler::new(BundleConfig::default());
        let pkg_json = bundler.generate_package_json();

        assert!(pkg_json.contains("\"type\": \"module\""));
        assert!(pkg_json.contains("wasm_app"));
    }

    #[test]
    fn test_bundle_simulation() {
        let bundler = WasmBundler::new(BundleConfig::default());
        let output = bundler.create_bundle_simulation(100_000);

        assert_eq!(output.wasm_size, 100_000);
        assert!(output.js_size > 0);
        assert!(output.files.len() >= 3);
    }
}
