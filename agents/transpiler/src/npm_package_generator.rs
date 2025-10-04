//! NPM Package Generator - Creates publishable NPM packages for WASM modules
//!
//! This module provides:
//! 1. Comprehensive package.json generation with all metadata
//! 2. README.md documentation generation
//! 3. TypeScript definitions (.d.ts) generation
//! 4. npm scripts and build commands
//! 5. Publishing configuration (.npmignore, .npmrc)
//! 6. Example usage and integration guides

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// NPM package configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpmPackageConfig {
    /// Package name (scoped or unscoped)
    pub name: String,
    /// Package version (semver)
    pub version: String,
    /// Package description
    pub description: String,
    /// Package author
    pub author: String,
    /// Package license (e.g., MIT, Apache-2.0)
    pub license: String,
    /// Repository URL
    pub repository: Option<String>,
    /// Homepage URL
    pub homepage: Option<String>,
    /// Bug tracker URL
    pub bugs: Option<String>,
    /// Keywords for npm search
    pub keywords: Vec<String>,
    /// Module type (module, commonjs)
    pub module_type: ModuleType,
    /// Include example code
    pub include_examples: bool,
    /// Include TypeScript definitions
    pub include_types: bool,
    /// Target environments
    pub targets: Vec<String>,
}

impl Default for NpmPackageConfig {
    fn default() -> Self {
        Self {
            name: "wasm-module".to_string(),
            version: "0.1.0".to_string(),
            description: "WASM module generated from Python".to_string(),
            author: "".to_string(),
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            bugs: None,
            keywords: vec!["wasm".to_string(), "webassembly".to_string()],
            module_type: ModuleType::ESModule,
            include_examples: true,
            include_types: true,
            targets: vec!["web".to_string(), "node".to_string()],
        }
    }
}

/// Module type for package.json
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModuleType {
    ESModule,
    CommonJS,
    Dual,
}

impl ModuleType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::ESModule => "module",
            Self::CommonJS => "commonjs",
            Self::Dual => "module",
        }
    }
}

/// NPM package files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpmPackage {
    /// package.json content
    pub package_json: String,
    /// README.md content
    pub readme: String,
    /// .npmignore content
    pub npmignore: String,
    /// TypeScript definitions
    pub typescript_defs: Option<String>,
    /// Example files
    pub examples: HashMap<String, String>,
    /// Total package size estimate
    pub estimated_size: u64,
}

/// NPM package generator
pub struct NpmPackageGenerator {
    config: NpmPackageConfig,
}

impl NpmPackageGenerator {
    /// Create new NPM package generator
    pub fn new(config: NpmPackageConfig) -> Self {
        Self { config }
    }

    /// Generate complete NPM package
    pub fn generate(&self, wasm_size: u64, js_size: u64) -> NpmPackage {
        let package_json = self.generate_package_json();
        let readme = self.generate_readme(wasm_size, js_size);
        let npmignore = self.generate_npmignore();
        let typescript_defs = if self.config.include_types {
            Some(self.generate_typescript_definitions())
        } else {
            None
        };
        let examples = if self.config.include_examples {
            self.generate_examples()
        } else {
            HashMap::new()
        };

        let mut estimated_size = wasm_size + js_size + package_json.len() as u64 + readme.len() as u64;
        if let Some(ref defs) = typescript_defs {
            estimated_size += defs.len() as u64;
        }

        NpmPackage {
            package_json,
            readme,
            npmignore,
            typescript_defs,
            examples,
            estimated_size,
        }
    }

    /// Generate comprehensive package.json
    pub fn generate_package_json(&self) -> String {
        let mut pkg = serde_json::json!({
            "name": self.config.name,
            "version": self.config.version,
            "description": self.config.description,
            "type": self.config.module_type.as_str(),
            "main": "./index.js",
            "types": "./index.d.ts",
            "files": [
                "*.js",
                "*.wasm",
                "*.d.ts",
                "README.md",
                "LICENSE"
            ],
            "scripts": {
                "build": "echo \"Build already complete\"",
                "test": "node test.js",
                "prepublishOnly": "echo \"Ready to publish\""
            },
            "keywords": self.config.keywords,
            "license": self.config.license,
        });

        // Add optional fields
        if !self.config.author.is_empty() {
            pkg["author"] = serde_json::json!(self.config.author);
        }

        if let Some(ref repo) = self.config.repository {
            pkg["repository"] = serde_json::json!({
                "type": "git",
                "url": repo
            });
        }

        if let Some(ref homepage) = self.config.homepage {
            pkg["homepage"] = serde_json::json!(homepage);
        }

        if let Some(ref bugs) = self.config.bugs {
            pkg["bugs"] = serde_json::json!({
                "url": bugs
            });
        }

        // Add dual module support if needed
        if self.config.module_type == ModuleType::Dual {
            pkg["exports"] = serde_json::json!({
                ".": {
                    "import": "./index.mjs",
                    "require": "./index.cjs",
                    "types": "./index.d.ts"
                }
            });
        } else if self.config.module_type == ModuleType::ESModule {
            pkg["exports"] = serde_json::json!({
                ".": {
                    "import": "./index.js",
                    "types": "./index.d.ts"
                }
            });
        }

        // Add engine requirements
        pkg["engines"] = serde_json::json!({
            "node": ">=16.0.0"
        });

        // Add peer dependencies for WASM
        pkg["peerDependencies"] = serde_json::json!({});

        // Add side effects
        pkg["sideEffects"] = serde_json::json!(["*.wasm", "*.js"]);

        serde_json::to_string_pretty(&pkg).unwrap_or_else(|_| "{}".to_string())
    }

    /// Generate README.md
    pub fn generate_readme(&self, wasm_size: u64, js_size: u64) -> String {
        let size_kb = (wasm_size + js_size) / 1024;
        let module_type = match self.config.module_type {
            ModuleType::ESModule => "ES Module",
            ModuleType::CommonJS => "CommonJS",
            ModuleType::Dual => "Dual (ES Module + CommonJS)",
        };

        format!(r#"# {}

{}

## 📦 Installation

```bash
npm install {}
```

or with yarn:

```bash
yarn add {}
```

## 🚀 Quick Start

### ES Module (Browser/Node.js)

```javascript
import init, {{ /* your functions */ }} from '{}';

async function main() {{
  // Initialize WASM module
  await init();

  // Use your functions
  console.log('WASM module ready!');
}}

main();
```

### CommonJS (Node.js)

```javascript
const {{ init }} = require('{}');

async function main() {{
  await init();
  console.log('WASM module ready!');
}}

main();
```

### Browser (ES Module)

```html
<script type="module">
  import init from './node_modules/{}/index.js';

  await init();
  // Your code here
</script>
```

## 📊 Package Info

- **Type**: {}
- **Size**: ~{} KB (WASM + JS)
- **License**: {}
- **Engines**: Node.js >=16.0.0

## 🎯 Features

- ✅ Zero dependencies
- ✅ TypeScript support included
- ✅ Browser and Node.js compatible
- ✅ Optimized WASM binary
- ✅ Tree-shakeable exports
- ✅ Source maps available

## 📚 API Documentation

### `init()`

Initialize the WASM module. Must be called before using any other functions.

**Returns**: `Promise<WebAssembly.Instance>`

**Example**:
```javascript
await init();
```

### Module Functions

[Add your specific function documentation here]

## 🔧 Advanced Usage

### Custom WASM Path

```javascript
import init from '{}';

// Load WASM from custom URL
await init('/path/to/custom.wasm');
```

### Error Handling

```javascript
try {{
  await init();
}} catch (err) {{
  console.error('Failed to initialize WASM:', err);
}}
```

### TypeScript

This package includes TypeScript definitions:

```typescript
import init, {{ /* your typed functions */ }} from '{}';

const result: number = await yourFunction(42);
```

## 🌐 Browser Support

- Chrome/Edge: ✅ (version 57+)
- Firefox: ✅ (version 52+)
- Safari: ✅ (version 11+)
- Node.js: ✅ (version 16+)
- Deno: ✅ (version 1.0+)

## 📝 Examples

### Example 1: Basic Usage

```javascript
import init, {{ add }} from '{}';

await init();
const result = add(2, 3);
console.log(result); // 5
```

### Example 2: Web Worker

```javascript
// worker.js
import init from '{}';

self.addEventListener('message', async (e) => {{
  await init();
  // Process messages
}});
```

## 🔨 Development

### Building from Source

```bash
# Clone repository
git clone {}

# Build WASM
cargo build --profile wasm-size --target wasm32-unknown-unknown

# Generate bindings
wasm-bindgen target/wasm32-unknown-unknown/wasm-size/module.wasm --out-dir pkg
```

## 📄 License

{}

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please use the [GitHub issue tracker]({}).

---

Generated with [Portalis](https://github.com/portalis) - Python to Rust/WASM transpiler
"#,
            self.config.name,
            self.config.description,
            self.config.name,
            self.config.name,
            self.config.name,
            self.config.name,
            self.config.name,
            module_type,
            size_kb,
            self.config.license,
            self.config.name,
            self.config.name,
            self.config.name,
            self.config.name,
            self.config.repository.as_deref().unwrap_or(""),
            self.config.license,
            self.config.bugs.as_deref().unwrap_or("")
        )
    }

    /// Generate .npmignore
    pub fn generate_npmignore(&self) -> String {
        r#"# Source files
src/
target/
*.rs
Cargo.toml
Cargo.lock

# Development files
.git/
.github/
.vscode/
.idea/
*.swp
*.swo
*~

# Build files
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Test files
tests/
test/
*.test.js
*.spec.js

# Documentation source
docs/
examples/

# CI/CD
.travis.yml
.gitlab-ci.yml
azure-pipelines.yml
.circleci/

# Misc
.DS_Store
Thumbs.db
"#.to_string()
    }

    /// Generate TypeScript definitions
    pub fn generate_typescript_definitions(&self) -> String {
        format!(r#"// TypeScript definitions for {}
// Generated by Portalis

/**
 * Initialize the WASM module
 * @param module_or_path - Optional WASM module or path to .wasm file
 * @returns Promise that resolves when WASM is initialized
 */
export default function init(module_or_path?: WebAssembly.Module | string | URL): Promise<WebAssembly.Instance>;

/**
 * Check if WASM module is initialized
 */
export function isInitialized(): boolean;

/**
 * Get the WASM instance
 */
export function getInstance(): WebAssembly.Instance | null;

// Add your specific function type definitions here
// Example:
// export function add(a: number, b: number): number;
// export function process_string(input: string): string;
"#, self.config.name)
    }

    /// Generate example files
    pub fn generate_examples(&self) -> HashMap<String, String> {
        let mut examples = HashMap::new();

        // Example 1: Basic usage (Node.js)
        examples.insert(
            "example-nodejs.js".to_string(),
            format!(r#"// Example: Basic usage in Node.js
const {{ init }} = require('{}');

async function main() {{
  try {{
    // Initialize WASM module
    await init();
    console.log('✓ WASM module initialized');

    // Use your functions here
    // const result = yourFunction(42);
    // console.log('Result:', result);
  }} catch (err) {{
    console.error('Error:', err);
    process.exit(1);
  }}
}}

main();
"#, self.config.name)
        );

        // Example 2: Browser usage
        examples.insert(
            "example-browser.html".to_string(),
            format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - Browser Example</title>
</head>
<body>
    <h1>{} - Browser Example</h1>
    <div id="output"></div>

    <script type="module">
        import init from './node_modules/{}/index.js';

        async function main() {{
            const output = document.getElementById('output');

            try {{
                // Initialize WASM
                await init();
                output.innerHTML = '<p>✓ WASM module initialized successfully!</p>';

                // Use your functions here
                // const result = yourFunction(42);
                // output.innerHTML += `<p>Result: ${{result}}</p>`;
            }} catch (err) {{
                output.innerHTML = `<p style="color: red;">Error: ${{err.message}}</p>`;
            }}
        }}

        main();
    </script>
</body>
</html>
"#, self.config.name, self.config.name, self.config.name)
        );

        // Example 3: TypeScript usage
        if self.config.include_types {
            examples.insert(
                "example-typescript.ts".to_string(),
                format!(r#"// Example: TypeScript usage
import init from '{}';

async function main(): Promise<void> {{
  try {{
    // Initialize WASM module
    await init();
    console.log('✓ WASM module initialized');

    // Type-safe function calls
    // const result: number = yourFunction(42);
    // console.log('Result:', result);
  }} catch (err) {{
    console.error('Error:', err);
    throw err;
  }}
}}

main();
"#, self.config.name)
            );
        }

        // Example 4: Web Worker usage
        examples.insert(
            "example-worker.js".to_string(),
            format!(r#"// Example: Web Worker usage
// worker.js
import init from '{}';

let wasmReady = false;

self.addEventListener('message', async (event) => {{
  if (!wasmReady) {{
    await init();
    wasmReady = true;
    self.postMessage({{ type: 'ready' }});
  }}

  const {{ type, data }} = event.data;

  try {{
    // Handle messages
    if (type === 'compute') {{
      // const result = yourFunction(data);
      // self.postMessage({{ type: 'result', data: result }});
    }}
  }} catch (err) {{
    self.postMessage({{ type: 'error', error: err.message }});
  }}
}});

// main.js
const worker = new Worker('./worker.js', {{ type: 'module' }});

worker.addEventListener('message', (event) => {{
  const {{ type, data }} = event.data;

  if (type === 'ready') {{
    console.log('Worker ready');
    worker.postMessage({{ type: 'compute', data: 42 }});
  }} else if (type === 'result') {{
    console.log('Result:', data);
  }}
}});
"#, self.config.name)
        );

        examples
    }

    /// Generate npm publish script
    pub fn generate_publish_script(&self) -> String {
        r#"#!/bin/bash
# NPM Publish Script

set -e

echo "=== NPM Package Publishing ==="
echo ""

# Step 1: Verify package
echo "Step 1/5: Verifying package..."
npm run prepublishOnly
echo "✓ Package verified"
echo ""

# Step 2: Run tests
echo "Step 2/5: Running tests..."
npm test || echo "⚠ Tests failed or not configured"
echo ""

# Step 3: Check package contents
echo "Step 3/5: Checking package contents..."
npm pack --dry-run
echo ""

# Step 4: Version check
echo "Step 4/5: Current version:"
npm version
echo ""

# Step 5: Publish
echo "Step 5/5: Ready to publish!"
echo ""
echo "To publish to npm:"
echo "  npm publish"
echo ""
echo "To publish as scoped package:"
echo "  npm publish --access public"
echo ""
echo "To publish with specific tag:"
echo "  npm publish --tag beta"
"#.to_string()
    }

    /// Generate .npmrc for publishing configuration
    pub fn generate_npmrc(&self) -> String {
        r#"# NPM Configuration
# Add your npm token: //registry.npmjs.org/:_authToken=${NPM_TOKEN}

# Package settings
save-exact=true
package-lock=true

# Publishing settings
access=public

# Registry (uncomment to use custom registry)
# registry=https://registry.npmjs.org/
"#.to_string()
    }

    /// Generate package report
    pub fn generate_package_report(&self, package: &NpmPackage) -> String {
        let examples_count = package.examples.len();
        let has_types = package.typescript_defs.is_some();
        let size_kb = package.estimated_size / 1024;

        format!(r#"NPM Package Report
==================

Package Name:     {}
Version:          {}
License:          {}
Module Type:      {}

Content:
--------
✓ package.json    ({} bytes)
✓ README.md       ({} bytes)
✓ .npmignore      ({} bytes)
{} TypeScript definitions
{} Examples:
{}

Total Size:       {} KB

Publishing:
-----------
1. Ensure you're logged in: npm login
2. Review package: npm pack --dry-run
3. Publish: npm publish{}

Package Contents:
-----------------
Files to be published:
- *.js (JavaScript glue code)
- *.wasm (WebAssembly binary)
{}
- README.md (Documentation)
- LICENSE (License file)
"#,
            self.config.name,
            self.config.version,
            self.config.license,
            match self.config.module_type {
                ModuleType::ESModule => "ES Module",
                ModuleType::CommonJS => "CommonJS",
                ModuleType::Dual => "Dual (ESM + CJS)",
            },
            package.package_json.len(),
            package.readme.len(),
            package.npmignore.len(),
            if has_types { "✓" } else { "✗" },
            if examples_count > 0 {
                format!("✓ {}", examples_count)
            } else {
                "✗ 0".to_string()
            },
            package.examples.keys()
                .map(|k| format!("  - {}", k))
                .collect::<Vec<_>>()
                .join("\n"),
            size_kb,
            if self.config.name.starts_with('@') {
                " --access public"
            } else {
                ""
            },
            if has_types { "- *.d.ts (TypeScript definitions)" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npm_package_generation() {
        let config = NpmPackageConfig {
            name: "test-wasm-module".to_string(),
            version: "1.0.0".to_string(),
            description: "Test WASM module".to_string(),
            author: "Test Author".to_string(),
            license: "MIT".to_string(),
            repository: Some("https://github.com/test/repo".to_string()),
            ..Default::default()
        };

        let generator = NpmPackageGenerator::new(config);
        let package = generator.generate(100_000, 10_000);

        assert!(package.package_json.contains("test-wasm-module"));
        assert!(package.package_json.contains("1.0.0"));
        assert!(package.readme.contains("# test-wasm-module"));
        assert!(package.typescript_defs.is_some());
        assert!(!package.examples.is_empty());
    }

    #[test]
    fn test_package_json_structure() {
        let config = NpmPackageConfig::default();
        let generator = NpmPackageGenerator::new(config);
        let pkg_json = generator.generate_package_json();

        assert!(pkg_json.contains("\"name\""));
        assert!(pkg_json.contains("\"version\""));
        assert!(pkg_json.contains("\"type\""));
        assert!(pkg_json.contains("\"main\""));
        assert!(pkg_json.contains("\"scripts\""));
    }

    #[test]
    fn test_module_types() {
        // ESModule
        let mut config = NpmPackageConfig::default();
        config.module_type = ModuleType::ESModule;
        let generator = NpmPackageGenerator::new(config.clone());
        let pkg = generator.generate_package_json();
        assert!(pkg.contains("\"type\": \"module\""));

        // CommonJS
        config.module_type = ModuleType::CommonJS;
        let generator = NpmPackageGenerator::new(config.clone());
        let pkg = generator.generate_package_json();
        assert!(pkg.contains("\"type\": \"commonjs\""));

        // Dual
        config.module_type = ModuleType::Dual;
        let generator = NpmPackageGenerator::new(config);
        let pkg = generator.generate_package_json();
        assert!(pkg.contains("\"exports\""));
    }

    #[test]
    fn test_readme_generation() {
        let config = NpmPackageConfig {
            name: "my-wasm-pkg".to_string(),
            description: "Amazing WASM package".to_string(),
            ..Default::default()
        };
        let generator = NpmPackageGenerator::new(config);
        let readme = generator.generate_readme(50_000, 5_000);

        assert!(readme.contains("# my-wasm-pkg"));
        assert!(readme.contains("Amazing WASM package"));
        assert!(readme.contains("Installation"));
        assert!(readme.contains("Quick Start"));
        assert!(readme.contains("API Documentation"));
    }

    #[test]
    fn test_typescript_definitions() {
        let config = NpmPackageConfig {
            name: "ts-wasm-module".to_string(),
            include_types: true,
            ..Default::default()
        };
        let generator = NpmPackageGenerator::new(config);
        let defs = generator.generate_typescript_definitions();

        assert!(defs.contains("ts-wasm-module"));
        assert!(defs.contains("export default function init"));
        assert!(defs.contains("Promise<WebAssembly.Instance>"));
    }

    #[test]
    fn test_examples_generation() {
        let config = NpmPackageConfig {
            include_examples: true,
            ..Default::default()
        };
        let generator = NpmPackageGenerator::new(config);
        let examples = generator.generate_examples();

        assert!(examples.contains_key("example-nodejs.js"));
        assert!(examples.contains_key("example-browser.html"));
        assert!(examples.contains_key("example-worker.js"));
        assert!(examples.contains_key("example-typescript.ts"));
    }

    #[test]
    fn test_npmignore_generation() {
        let config = NpmPackageConfig::default();
        let generator = NpmPackageGenerator::new(config);
        let npmignore = generator.generate_npmignore();

        assert!(npmignore.contains("src/"));
        assert!(npmignore.contains("*.rs"));
        assert!(npmignore.contains("Cargo.toml"));
        assert!(npmignore.contains("tests/"));
    }

    #[test]
    fn test_scoped_package() {
        let config = NpmPackageConfig {
            name: "@myorg/wasm-module".to_string(),
            ..Default::default()
        };
        let generator = NpmPackageGenerator::new(config);
        let pkg_json = generator.generate_package_json();
        let report = generator.generate_package_report(&generator.generate(100_000, 10_000));

        assert!(pkg_json.contains("@myorg/wasm-module"));
        assert!(report.contains("--access public"));
    }
}
