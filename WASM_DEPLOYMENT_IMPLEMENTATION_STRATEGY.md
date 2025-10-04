# WASM Deployment Implementation Strategy

**Project**: Portalis Python → Rust → WASM Transpiler
**Date**: October 4, 2025
**Current Status**: 191/219 tests passing (87.2%)
**Objective**: Production-ready WASM deployment pipeline

---

## Executive Summary

This strategy provides a detailed 15-day roadmap to build a complete WASM deployment infrastructure for the Portalis transpiler. The approach is **incremental, testable, and production-focused**, with clear success criteria and rollback procedures at each phase.

### Success Metrics
- **Phase 1**: WASM build infrastructure operational (100% tests passing in WASM target)
- **Phase 2**: End-to-end browser + Node.js demos functional
- **Phase 3**: High-priority parsing gaps closed (import system, tuple unpacking)
- **Phase 4**: Medium-priority features implemented (decorators, stdlib mapping)

### Current State Analysis

**Strengths:**
- ✅ Solid transpiler core (87.2% test pass rate)
- ✅ 150+ Python features already implemented
- ✅ Comprehensive test suite (219 tests)
- ✅ Rust toolchain ready (rustc 1.89.0)
- ✅ Node.js ecosystem available (v22.18.0)

**Gaps:**
- ❌ No wasm-pack installed
- ❌ No WASM-specific build configuration
- ❌ No browser/Node.js integration examples
- ⚠️ 28 failing tests (mostly parsing edge cases)
- ⚠️ Import system incomplete
- ⚠️ Tuple unpacking partial

---

## Phase 1: WASM Infrastructure (Days 1-3)

### Day 1: Toolchain & Build Setup

#### Morning: Install WASM Toolchain
**Tasks:**
1. Install wasm-pack: `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh`
2. Install wasm32 target: `rustup target add wasm32-unknown-unknown`
3. Verify installation: `wasm-pack --version`

**Time Estimate**: 1 hour
**Risk**: Low (standard installation)
**Validation**: `wasm-pack --version` outputs valid version

#### Afternoon: Add WASM Dependencies
**Tasks:**
1. Update `agents/transpiler/Cargo.toml`:
```toml
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
serde-wasm-bindgen = "0.6"
js-sys = "0.3"
console_error_panic_hook = "0.1"

[dev-dependencies]
wasm-bindgen-test = "0.3"

[features]
default = []
wasm = ["wasm-bindgen"]
```

2. Create `agents/transpiler/build.rs`:
```rust
fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        println!("cargo:rustc-cfg=wasm");
    }
}
```

**Time Estimate**: 2 hours
**Dependencies**: None
**Risk**: Low (additive changes)
**Validation**: `cargo check --target wasm32-unknown-unknown` succeeds
**Rollback**: `git checkout agents/transpiler/Cargo.toml`

#### Evening: WASM Bindings Module
**Tasks:**
1. Create `agents/transpiler/src/wasm.rs`:
```rust
use wasm_bindgen::prelude::*;
use crate::feature_translator::FeatureTranslator;

#[wasm_bindgen]
pub struct WasmTranspiler {
    translator: FeatureTranslator,
}

#[wasm_bindgen]
impl WasmTranspiler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            translator: FeatureTranslator::new(),
        }
    }

    #[wasm_bindgen]
    pub fn translate(&mut self, python_code: &str) -> Result<String, JsValue> {
        self.translator
            .translate(python_code)
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))
    }

    #[wasm_bindgen]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}
```

2. Update `agents/transpiler/src/lib.rs`:
```rust
#[cfg(target_arch = "wasm32")]
pub mod wasm;
```

**Time Estimate**: 2 hours
**Dependencies**: WASM dependencies installed
**Risk**: Low (isolated module)
**Validation**: `cargo check --target wasm32-unknown-unknown --features wasm` succeeds
**Rollback**: Remove `wasm.rs`, revert `lib.rs`

### Day 2: Build Scripts & NPM Packaging

#### Morning: wasm-pack Configuration
**Tasks:**
1. Create `agents/transpiler/Cargo.toml` metadata:
```toml
[package.metadata.wasm-pack.profile.release]
wasm-opt = ["-O3", "--enable-mutable-globals"]
```

2. Create build script `scripts/build-wasm.sh`:
```bash
#!/bin/bash
set -e

cd agents/transpiler

echo "Building WASM for web target..."
wasm-pack build --target web --out-dir ../../wasm-pkg/web

echo "Building WASM for Node.js target..."
wasm-pack build --target nodejs --out-dir ../../wasm-pkg/nodejs

echo "Building WASM for bundler target..."
wasm-pack build --target bundler --out-dir ../../wasm-pkg/bundler

echo "WASM builds complete!"
ls -lh ../../wasm-pkg/*/
```

**Time Estimate**: 2 hours
**Dependencies**: wasm-pack installed
**Risk**: Medium (first build may reveal issues)
**Validation**: All three builds succeed, .wasm files generated
**Rollback**: N/A (build artifacts only)

#### Afternoon: NPM Package Setup
**Tasks:**
1. Create `wasm-pkg/package.json`:
```json
{
  "name": "@portalis/transpiler-wasm",
  "version": "0.1.0",
  "description": "Python to Rust transpiler compiled to WASM",
  "main": "nodejs/portalis_transpiler.js",
  "module": "bundler/portalis_transpiler.js",
  "browser": "web/portalis_transpiler.js",
  "types": "web/portalis_transpiler.d.ts",
  "files": [
    "web/*",
    "nodejs/*",
    "bundler/*"
  ],
  "keywords": ["python", "rust", "transpiler", "wasm"],
  "license": "MIT"
}
```

2. Create `wasm-pkg/README.md` with usage examples

**Time Estimate**: 2 hours
**Dependencies**: WASM builds successful
**Risk**: Low (metadata only)
**Validation**: `npm pack` creates valid tarball
**Rollback**: Delete `wasm-pkg/` directory

#### Evening: CI/CD Integration
**Tasks:**
1. Create `.github/workflows/wasm-build.yml`:
```yaml
name: WASM Build

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-wasm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: wasm32-unknown-unknown
      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
      - name: Build WASM
        run: ./scripts/build-wasm.sh
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: wasm-packages
          path: wasm-pkg/
```

**Time Estimate**: 2 hours
**Dependencies**: GitHub repository
**Risk**: Low (CI configuration)
**Validation**: Workflow runs successfully on push
**Rollback**: Remove workflow file

### Day 3: Initial WASM Tests

#### Morning: WASM Test Infrastructure
**Tasks:**
1. Create `agents/transpiler/tests/wasm_tests.rs`:
```rust
#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use portalis_transpiler::wasm::WasmTranspiler;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_transpiler_creation() {
    let transpiler = WasmTranspiler::new();
    assert_eq!(transpiler.version(), env!("CARGO_PKG_VERSION"));
}

#[wasm_bindgen_test]
fn test_simple_function() {
    let mut transpiler = WasmTranspiler::new();
    let python = "def add(a, b):\n    return a + b";
    let result = transpiler.translate(python);
    assert!(result.is_ok());
    let rust = result.unwrap();
    assert!(rust.contains("pub fn add"));
}

#[wasm_bindgen_test]
fn test_invalid_syntax() {
    let mut transpiler = WasmTranspiler::new();
    let python = "def add(a, b\n    return a + b"; // Missing closing paren
    let result = transpiler.translate(python);
    assert!(result.is_err());
}
```

2. Run tests: `wasm-pack test --headless --firefox`

**Time Estimate**: 3 hours
**Dependencies**: WASM builds working
**Risk**: Medium (browser testing setup)
**Validation**: All WASM tests pass
**Rollback**: Remove test file

#### Afternoon: Performance Baseline
**Tasks:**
1. Create `agents/transpiler/benches/wasm_bench.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use portalis_transpiler::feature_translator::FeatureTranslator;

fn bench_fibonacci(c: &mut Criterion) {
    let python = r#"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"#;

    c.bench_function("translate_fibonacci", |b| {
        b.iter(|| {
            let mut translator = FeatureTranslator::new();
            translator.translate(black_box(python)).unwrap()
        })
    });
}

criterion_group!(benches, bench_fibonacci);
criterion_main!(benches);
```

2. Run benchmarks: `cargo bench --package portalis-transpiler`
3. Document baseline metrics in `docs/WASM_PERFORMANCE.md`

**Time Estimate**: 2 hours
**Dependencies**: None
**Risk**: Low (documentation)
**Validation**: Benchmarks run and document baseline
**Rollback**: N/A (documentation only)

#### Evening: Documentation
**Tasks:**
1. Create `docs/WASM_DEPLOYMENT_GUIDE.md` with:
   - Installation instructions
   - Build process
   - Testing procedures
   - Troubleshooting guide
2. Update main README with WASM section

**Time Estimate**: 2 hours
**Dependencies**: All Phase 1 tasks complete
**Risk**: None
**Validation**: Documentation review
**Rollback**: N/A

### Phase 1 Success Criteria
- ✅ wasm-pack builds succeed for web, nodejs, bundler targets
- ✅ WASM bindings compile without errors
- ✅ Basic WASM tests pass (3+ tests)
- ✅ Performance baseline documented
- ✅ CI/CD pipeline operational
- ✅ NPM package structure created

### Phase 1 Risk Mitigation
- **Build failures**: Extensive error handling, graceful degradation
- **Performance issues**: Profile early, optimize critical paths
- **Browser compatibility**: Test on Chrome, Firefox, Safari
- **Memory limits**: Monitor WASM heap usage, implement size limits

---

## Phase 2: End-to-End Examples (Days 4-5)

### Day 4: Browser Demo

#### Morning: HTML/JavaScript Harness
**Tasks:**
1. Create `examples/browser-demo/index.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Portalis WASM Transpiler Demo</title>
    <style>
        body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        textarea { width: 100%; height: 400px; font-family: monospace; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <h1>Python → Rust Transpiler (WASM)</h1>
    <div class="container">
        <div>
            <h2>Python Input</h2>
            <textarea id="python-input">def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)</textarea>
            <button id="translate-btn">Translate to Rust</button>
        </div>
        <div>
            <h2>Rust Output</h2>
            <textarea id="rust-output" readonly></textarea>
            <div id="status"></div>
        </div>
    </div>
    <script type="module" src="./app.js"></script>
</body>
</html>
```

2. Create `examples/browser-demo/app.js`:
```javascript
import init, { WasmTranspiler } from '../../wasm-pkg/web/portalis_transpiler.js';

async function run() {
    await init();

    const transpiler = new WasmTranspiler();
    const translateBtn = document.getElementById('translate-btn');
    const pythonInput = document.getElementById('python-input');
    const rustOutput = document.getElementById('rust-output');
    const status = document.getElementById('status');

    translateBtn.addEventListener('click', () => {
        try {
            const python = pythonInput.value;
            const rust = transpiler.translate(python);
            rustOutput.value = rust;
            status.innerHTML = '<span class="success">✓ Translation successful</span>';
        } catch (err) {
            status.innerHTML = `<span class="error">✗ Error: ${err}</span>`;
        }
    });

    console.log('Transpiler version:', transpiler.version());
}

run();
```

**Time Estimate**: 3 hours
**Dependencies**: Phase 1 complete
**Risk**: Low (standard web development)
**Validation**: Demo loads in browser, translations work
**Rollback**: Delete `examples/browser-demo/`

#### Afternoon: Development Server
**Tasks:**
1. Create `examples/browser-demo/package.json`:
```json
{
  "name": "portalis-browser-demo",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "devDependencies": {
    "vite": "^5.0.0"
  }
}
```

2. Create `examples/browser-demo/vite.config.js`:
```javascript
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    fs: {
      allow: ['../..']
    }
  }
});
```

3. Run: `npm install && npm run dev`

**Time Estimate**: 2 hours
**Dependencies**: Browser demo created
**Risk**: Low
**Validation**: `http://localhost:5173` serves demo
**Rollback**: N/A (dev dependency)

#### Evening: Example Library
**Tasks:**
1. Create `examples/browser-demo/examples.js` with 10+ Python code examples:
   - Fibonacci
   - Data processing
   - String manipulation
   - List comprehensions
   - Exception handling
   - Classes
2. Add example selector dropdown to UI

**Time Estimate**: 2 hours
**Dependencies**: Browser demo working
**Risk**: None
**Validation**: All examples translate successfully
**Rollback**: N/A (feature addition)

### Day 5: Node.js Integration

#### Morning: CLI Tool
**Tasks:**
1. Create `examples/nodejs-cli/transpile.js`:
```javascript
#!/usr/bin/env node

const fs = require('fs');
const { WasmTranspiler } = require('../../wasm-pkg/nodejs/portalis_transpiler.js');

const args = process.argv.slice(2);
if (args.length === 0) {
    console.error('Usage: transpile.js <input.py> [output.rs]');
    process.exit(1);
}

const inputFile = args[0];
const outputFile = args[1] || inputFile.replace('.py', '.rs');

const pythonCode = fs.readFileSync(inputFile, 'utf8');
const transpiler = new WasmTranspiler();

try {
    const rustCode = transpiler.translate(pythonCode);
    fs.writeFileSync(outputFile, rustCode);
    console.log(`✓ Translated ${inputFile} → ${outputFile}`);
} catch (err) {
    console.error(`✗ Translation failed: ${err}`);
    process.exit(1);
}
```

2. Make executable: `chmod +x transpile.js`
3. Test: `./transpile.js ../../examples/fibonacci.py`

**Time Estimate**: 2 hours
**Dependencies**: Node.js WASM build
**Risk**: Low
**Validation**: CLI translates example files
**Rollback**: Delete CLI script

#### Afternoon: NPM Package Integration
**Tasks:**
1. Create `examples/nodejs-integration/package.json`:
```json
{
  "name": "portalis-nodejs-example",
  "private": true,
  "dependencies": {
    "@portalis/transpiler-wasm": "file:../../wasm-pkg"
  }
}
```

2. Create `examples/nodejs-integration/batch-translate.js`:
```javascript
const fs = require('fs').promises;
const path = require('path');
const { WasmTranspiler } = require('@portalis/transpiler-wasm');

async function translateDirectory(inputDir, outputDir) {
    const files = await fs.readdir(inputDir);
    const transpiler = new WasmTranspiler();
    const results = [];

    for (const file of files) {
        if (!file.endsWith('.py')) continue;

        const inputPath = path.join(inputDir, file);
        const outputPath = path.join(outputDir, file.replace('.py', '.rs'));

        try {
            const python = await fs.readFile(inputPath, 'utf8');
            const rust = transpiler.translate(python);
            await fs.writeFile(outputPath, rust);
            results.push({ file, status: 'success' });
        } catch (err) {
            results.push({ file, status: 'error', error: err.message });
        }
    }

    return results;
}

(async () => {
    const results = await translateDirectory('../../examples', './output');
    console.log('Translation Results:');
    console.table(results);
})();
```

**Time Estimate**: 2 hours
**Dependencies**: NPM package created
**Risk**: Low
**Validation**: Batch translation processes example directory
**Rollback**: Delete integration example

#### Evening: End-to-End Validation
**Tasks:**
1. Run full pipeline test:
   - Translate Python examples to Rust (WASM)
   - Compile Rust to binary
   - Execute and verify outputs match
2. Create `scripts/e2e-test.sh`
3. Document results in `docs/E2E_VALIDATION.md`

**Time Estimate**: 3 hours
**Dependencies**: All examples working
**Risk**: Medium (complex integration)
**Validation**: 100% of examples pass end-to-end test
**Rollback**: N/A (validation only)

### Phase 2 Success Criteria
- ✅ Browser demo functional and visually polished
- ✅ Node.js CLI tool works for single files
- ✅ Batch translation processes directories
- ✅ End-to-end validation passes (Python → Rust → WASM → execution)
- ✅ Performance measurements documented
- ✅ At least 10 working examples

### Phase 2 Risk Mitigation
- **Browser compatibility**: Test on 3+ browsers
- **WASM loading issues**: Implement fallbacks, clear error messages
- **Performance**: Profile and optimize hot paths
- **User experience**: Clear error messages, helpful documentation

---

## Phase 3: High Priority Features (Days 6-10)

### Day 6: Import System - Part 1

#### Morning: Simple Import Parsing
**Tasks:**
1. Update `agents/transpiler/src/python_ast.rs`:
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    // Existing variants...
    Import {
        module: String,
        items: Option<Vec<String>>,
        alias: Option<String>,
    },
    FromImport {
        module: String,
        items: Vec<(String, Option<String>)>, // (name, alias)
    },
}
```

2. Update parser to handle:
   - `import math`
   - `import math as m`
   - `from math import sqrt`
   - `from math import sqrt, pow`
   - `from math import sqrt as square_root`

**Time Estimate**: 4 hours
**Dependencies**: None
**Risk**: Medium (core parser changes)
**Validation**: Unit tests for import parsing
**Rollback**: `git revert` parser changes

#### Afternoon: Rust Use Statement Generation
**Tasks:**
1. Update `agents/transpiler/src/python_to_rust.rs`:
```rust
fn translate_import(&mut self, module: &str, items: &Option<Vec<String>>, alias: &Option<String>) -> String {
    match module {
        "math" => {
            if let Some(items) = items {
                // from math import sqrt, pow
                items.iter()
                    .map(|item| self.map_stdlib_function(module, item))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                // import math
                "// std::f64::consts for math module".to_string()
            }
        }
        "sys" => "// std::env for sys module".to_string(),
        "os" => "// std::fs, std::env for os module".to_string(),
        _ => format!("// Unsupported module: {}", module),
    }
}
```

**Time Estimate**: 3 hours
**Dependencies**: Parser updated
**Risk**: Low (additive)
**Validation**: Tests generate correct use statements
**Rollback**: Revert translator changes

#### Evening: Tests
**Tasks:**
1. Create 20+ tests in `day_31_import_tests.rs`:
   - Simple imports
   - Import with aliases
   - From imports
   - Multiple items
   - Nested modules (if supported)

**Time Estimate**: 2 hours
**Dependencies**: Import translation working
**Risk**: None
**Validation**: All import tests pass
**Rollback**: N/A

### Day 7: Import System - Part 2

#### Morning: Standard Library Mapping
**Tasks:**
1. Create `agents/transpiler/src/stdlib_map.rs`:
```rust
use std::collections::HashMap;

pub struct StdlibMapper {
    mappings: HashMap<String, HashMap<String, String>>,
}

impl StdlibMapper {
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // Math module
        let mut math = HashMap::new();
        math.insert("sqrt".to_string(), "f64::sqrt".to_string());
        math.insert("pow".to_string(), "f64::powf".to_string());
        math.insert("abs".to_string(), "f64::abs".to_string());
        math.insert("floor".to_string(), "f64::floor".to_string());
        math.insert("ceil".to_string(), "f64::ceil".to_string());
        math.insert("pi".to_string(), "std::f64::consts::PI".to_string());
        math.insert("e".to_string(), "std::f64::consts::E".to_string());
        mappings.insert("math".to_string(), math);

        // Sys module
        let mut sys = HashMap::new();
        sys.insert("exit".to_string(), "std::process::exit".to_string());
        sys.insert("argv".to_string(), "std::env::args().collect::<Vec<_>>()".to_string());
        mappings.insert("sys".to_string(), sys);

        // OS module
        let mut os = HashMap::new();
        os.insert("getcwd".to_string(), "std::env::current_dir().unwrap().display().to_string()".to_string());
        mappings.insert("os".to_string(), os);

        Self { mappings }
    }

    pub fn map(&self, module: &str, item: &str) -> Option<String> {
        self.mappings.get(module)?.get(item).cloned()
    }
}
```

**Time Estimate**: 4 hours
**Dependencies**: Import parsing working
**Risk**: Low (isolated module)
**Validation**: Stdlib mapping tests
**Rollback**: Remove module

#### Afternoon: Integration
**Tasks:**
1. Integrate `StdlibMapper` into translator
2. Update function call translation to use mapped names
3. Add use statements to generated Rust code

**Time Estimate**: 3 hours
**Dependencies**: StdlibMapper created
**Risk**: Medium (affects code generation)
**Validation**: Examples using math.sqrt() translate correctly
**Rollback**: Disable stdlib mapping

#### Evening: Documentation
**Tasks:**
1. Create `docs/STDLIB_SUPPORT.md` listing:
   - Supported modules
   - Supported functions per module
   - Rust equivalents
   - Unsupported features
2. Update examples with import usage

**Time Estimate**: 2 hours
**Dependencies**: None
**Risk**: None
**Validation**: Documentation review
**Rollback**: N/A

### Day 8: Multiple Assignment & Tuple Unpacking

#### Morning: Parser Fixes
**Tasks:**
1. Fix multiple assignment: `a = b = c = 0`
   - Update parser to handle chained assignments
   - Generate sequential `let` statements

2. Fix tuple unpacking: `a, b = 1, 2`
   - Parse tuple patterns
   - Generate Rust destructuring

**Time Estimate**: 4 hours
**Dependencies**: None
**Risk**: High (complex parsing)
**Validation**: 4 failing tests now pass
**Rollback**: Revert parser changes

#### Afternoon: Advanced Patterns
**Tasks:**
1. Nested tuple unpacking: `(a, b), c = (1, 2), 3`
2. Extended unpacking: `a, *rest = [1, 2, 3]`
3. For loop unpacking: `for x, y in pairs:`

**Time Estimate**: 4 hours
**Dependencies**: Basic unpacking working
**Risk**: High (complex patterns)
**Validation**: 6 more failing tests pass
**Rollback**: Partial rollback to basic unpacking

#### Evening: Tests
**Tasks:**
1. Add 30+ tuple unpacking tests
2. Verify all Day 26-27 tests pass
3. Run full test suite: expect 201/219 passing (92%)

**Time Estimate**: 2 hours
**Dependencies**: Unpacking implemented
**Risk**: None
**Validation**: Test pass rate increases to 92%
**Rollback**: N/A

### Day 9: With Statement / Context Managers

#### Morning: Basic With Support
**Tasks:**
1. Add `With` statement to AST:
```rust
pub enum Statement {
    // Existing...
    With {
        context: Box<Expression>,
        alias: Option<String>,
        body: Vec<Statement>,
    },
}
```

2. Parse `with` statements:
```python
with open('file.txt') as f:
    content = f.read()
```

**Time Estimate**: 4 hours
**Dependencies**: None
**Risk**: Medium (new statement type)
**Validation**: With statement parsing tests
**Rollback**: Remove With variant

#### Afternoon: Rust RAII Translation
**Tasks:**
1. Translate `with` to Rust scoped blocks:
```rust
{
    let f = std::fs::File::open("file.txt").unwrap();
    let content = std::fs::read_to_string("file.txt").unwrap();
}
```

2. Handle common context managers:
   - `open()` → `File::open()`
   - Custom contexts → scoped blocks

**Time Estimate**: 4 hours
**Dependencies**: With parsing
**Risk**: Medium (translation complexity)
**Validation**: File I/O examples work
**Rollback**: Generate comments for unsupported contexts

#### Evening: Tests & Examples
**Tasks:**
1. Add 15+ with statement tests
2. Create file I/O examples
3. Document limitations

**Time Estimate**: 2 hours
**Dependencies**: With translation working
**Risk**: None
**Validation**: All with tests pass
**Rollback**: N/A

### Day 10: Parser Fixes & Validation

#### Morning: Fix Remaining Parsing Issues
**Tasks:**
1. Nested slices: `data[1:3][0]`
2. Lambda edge cases: `lambda: 42`
3. Nested built-in functions: `max(min(x, y), z)`
4. Chained string methods: `s.strip().lower().split()`

**Time Estimate**: 4 hours
**Dependencies**: None
**Risk**: High (multiple parser issues)
**Validation**: 8+ failing tests fixed
**Rollback**: Individual reverts per fix

#### Afternoon: Comprehensive Testing
**Tasks:**
1. Run full test suite
2. Analyze failures
3. Prioritize remaining issues
4. Fix quick wins

**Time Estimate**: 4 hours
**Dependencies**: Parser fixes applied
**Risk**: Medium
**Validation**: Test pass rate ≥ 95% (208/219)
**Rollback**: Revert problematic fixes

#### Evening: WASM Build Validation
**Tasks:**
1. Rebuild WASM with all Phase 3 changes
2. Test browser demo with new features
3. Test Node.js CLI with new features
4. Update examples

**Time Estimate**: 2 hours
**Dependencies**: All Phase 3 changes complete
**Risk**: Low
**Validation**: WASM builds succeed, demos work
**Rollback**: N/A (validation only)

### Phase 3 Success Criteria
- ✅ Import system supports math, sys, os modules
- ✅ Stdlib mapping for 20+ common functions
- ✅ Multiple assignment parsing fixed
- ✅ Tuple unpacking fully functional
- ✅ With statements supported (basic)
- ✅ Parser edge cases resolved
- ✅ Test pass rate ≥ 95% (208+ tests passing)
- ✅ WASM builds include all new features

### Phase 3 Risk Mitigation
- **Parser complexity**: Incremental changes, thorough testing
- **Breaking changes**: Feature flags for experimental features
- **Regression**: Continuous test execution
- **WASM compatibility**: Test after each major change

---

## Phase 4: Medium/Low Priority (Days 11-15)

### Day 11: Decorator Support - Part 1

#### Morning: Decorator Parsing
**Tasks:**
1. Add decorator support to AST:
```rust
pub struct Function {
    pub decorators: Vec<String>,
    pub name: String,
    // ...existing fields
}
```

2. Parse decorators:
```python
@staticmethod
@property
def method():
    pass
```

**Time Estimate**: 4 hours
**Dependencies**: None
**Risk**: Medium
**Validation**: Decorator parsing tests
**Rollback**: Remove decorator fields

#### Afternoon: Common Decorator Translation
**Tasks:**
1. `@property` → Rust getter pattern
2. `@staticmethod` → Remove `self` parameter
3. `@classmethod` → Comment (not directly supported)
4. Custom decorators → Comment annotation

**Time Estimate**: 4 hours
**Dependencies**: Decorator parsing
**Risk**: Medium (partial support)
**Validation**: Property/staticmethod examples work
**Rollback**: Generate comments only

#### Evening: Tests
**Tasks:**
1. Add 20+ decorator tests
2. Document supported/unsupported decorators

**Time Estimate**: 2 hours
**Dependencies**: Decorator translation
**Risk**: None
**Validation**: Decorator tests pass
**Rollback**: N/A

### Day 12: Generator & Yield Support

#### Morning: Yield Parsing
**Tasks:**
1. Add `Yield` expression to AST
2. Parse `yield` statements
3. Detect generator functions

**Time Estimate**: 4 hours
**Dependencies**: None
**Risk**: High (complex feature)
**Validation**: Yield parsing tests
**Rollback**: Remove yield support

#### Afternoon: Iterator Translation
**Tasks:**
1. Translate generator to Rust iterator:
```python
def count(n):
    i = 0
    while i < n:
        yield i
        i += 1
```

Becomes:
```rust
fn count(n: i32) -> impl Iterator<Item = i32> {
    (0..n)
}
```

2. Handle simple patterns, comment complex ones

**Time Estimate**: 4 hours
**Dependencies**: Yield parsing
**Risk**: High (limited Rust equivalent)
**Validation**: Simple generator examples work
**Rollback**: Generate comments for generators

#### Evening: Tests
**Tasks:**
1. Add 15+ generator tests
2. Document limitations
3. Provide workaround examples

**Time Estimate**: 2 hours
**Dependencies**: Generator translation
**Risk**: None
**Validation**: Generator tests pass
**Rollback**: N/A

### Day 13: Async/Await (If Time Permits)

#### Morning: Async Function Detection
**Tasks:**
1. Parse `async def` functions
2. Parse `await` expressions
3. Detect async context

**Time Estimate**: 4 hours
**Dependencies**: None
**Risk**: High (complex feature)
**Validation**: Async parsing tests
**Rollback**: Skip async support

#### Afternoon: Basic Async Translation
**Tasks:**
1. `async def` → `async fn`
2. `await` → `.await`
3. Add `#[tokio::main]` where appropriate
4. Simple examples only

**Time Estimate**: 4 hours
**Dependencies**: Async parsing
**Risk**: Very High (requires runtime)
**Validation**: Basic async examples work
**Rollback**: Comment out async features

#### Evening: Async Runtime Integration
**Tasks:**
1. Add `tokio` to generated Cargo.toml
2. Test async examples
3. Document async limitations

**Time Estimate**: 2 hours
**Dependencies**: Async translation
**Risk**: Very High
**Validation**: Async examples compile and run
**Rollback**: Remove async support entirely

### Day 14: Enhanced Stdlib Mapping

#### All Day: Expand Stdlib Coverage
**Tasks:**
1. Add 50+ more stdlib mappings:
   - `collections` module
   - `itertools` patterns
   - `functools` patterns
   - `datetime` basics
   - `json` module
   - `re` (regex) module

2. Create comprehensive mapping table
3. Add tests for each module
4. Update documentation

**Time Estimate**: 8 hours
**Dependencies**: Stdlib mapper infrastructure
**Risk**: Low (additive)
**Validation**: 50+ new stdlib functions supported
**Rollback**: Remove new mappings

### Day 15: Final Integration & Polish

#### Morning: Full Test Suite Run
**Tasks:**
1. Run all 219 tests
2. Fix any regressions
3. Document known issues
4. Update test statistics

**Time Estimate**: 4 hours
**Dependencies**: All Phase 4 complete
**Risk**: Medium (may reveal integration issues)
**Validation**: Test pass rate ≥ 96% (210+ passing)
**Rollback**: Revert problematic features

#### Afternoon: WASM Rebuild & Validation
**Tasks:**
1. Rebuild all WASM targets
2. Update browser demo with new features
3. Update Node.js examples
4. Performance regression testing

**Time Estimate**: 3 hours
**Dependencies**: Tests passing
**Risk**: Low
**Validation**: WASM builds succeed, no regressions
**Rollback**: N/A

#### Evening: Documentation & Release Prep
**Tasks:**
1. Update all documentation
2. Create release notes
3. Prepare NPM package for publishing
4. Create migration guide (if needed)
5. Final review

**Time Estimate**: 3 hours
**Dependencies**: All features complete
**Risk**: None
**Validation**: Documentation complete and accurate
**Rollback**: N/A

### Phase 4 Success Criteria
- ✅ Decorator support (property, staticmethod)
- ✅ Generator/yield support (basic patterns)
- ✅ Async/await support (if time permits)
- ✅ 70+ stdlib functions mapped
- ✅ Test pass rate ≥ 96% (210+ tests)
- ✅ All WASM targets build successfully
- ✅ Complete documentation

### Phase 4 Risk Mitigation
- **Scope creep**: Time-box each feature, skip if blocked
- **Complexity**: Implement simple cases first, document limitations
- **Async runtime**: Make async support optional/feature-gated
- **Testing**: Continuous validation, quick rollback

---

## Rollback Procedures

### Emergency Rollback
If critical issues arise at any phase:

1. **Identify the change**: Use `git log` to find problematic commits
2. **Revert commit**: `git revert <commit-hash>`
3. **Rebuild**: `cargo clean && cargo build`
4. **Validate**: Run test suite
5. **Document**: Record issue in rollback log

### Phase-Level Rollback
If an entire phase needs rollback:

**Phase 1 (WASM Infrastructure):**
```bash
git checkout main
git revert <phase1-start>..<phase1-end>
rm -rf wasm-pkg/
rm -rf agents/transpiler/src/wasm.rs
git restore agents/transpiler/Cargo.toml
```

**Phase 2 (Examples):**
```bash
rm -rf examples/browser-demo/
rm -rf examples/nodejs-integration/
git revert <phase2-start>..<phase2-end>
```

**Phase 3 (Features):**
```bash
git revert <phase3-start>..<phase3-end>
cargo test  # Validate rollback
```

**Phase 4 (Enhancements):**
```bash
git revert <phase4-start>..<phase4-end>
cargo test  # Validate rollback
```

### Partial Rollback
For individual features:

1. Create feature branch: `git checkout -b rollback/feature-name`
2. Revert specific commits: `git revert <commit-hash>`
3. Test thoroughly
4. Merge back: `git checkout main && git merge rollback/feature-name`

---

## Testing Strategy

### Unit Tests
- **Location**: `agents/transpiler/src/*_test.rs`
- **Run**: `cargo test --package portalis-transpiler`
- **Target**: 96%+ pass rate (210+ tests)
- **Frequency**: After every change

### WASM Tests
- **Location**: `agents/transpiler/tests/wasm_tests.rs`
- **Run**: `wasm-pack test --headless --chrome`
- **Target**: 100% pass rate
- **Frequency**: Daily during Phase 1-2, weekly after

### Integration Tests
- **Browser**: Manual testing in Chrome, Firefox, Safari
- **Node.js**: Automated CLI tests
- **End-to-End**: Full pipeline validation script
- **Frequency**: End of each phase

### Performance Tests
- **Benchmarks**: `cargo bench --package portalis-transpiler`
- **Metrics**: Translation time, WASM bundle size, runtime performance
- **Target**: No regressions, <10% variance
- **Frequency**: End of each phase

### Regression Tests
- **Strategy**: Git bisect for finding regressions
- **Process**: Run full test suite on each commit
- **Automation**: CI/CD pipeline
- **Target**: Zero regressions

---

## Dependency Management

### Critical Dependencies
```toml
wasm-bindgen = "0.2"           # WASM bindings
serde-wasm-bindgen = "0.6"     # Serialization
js-sys = "0.3"                 # JavaScript interop
console_error_panic_hook = "0.1"  # Error handling
```

### Development Dependencies
```toml
wasm-bindgen-test = "0.3"      # WASM testing
criterion = "0.5"               # Benchmarking
```

### Tooling
- **wasm-pack**: v0.12+ (WASM packaging)
- **Node.js**: v18+ (runtime)
- **Vite**: v5+ (dev server)

### Version Pinning
- Pin all dependencies to minor versions
- Document version compatibility in `Cargo.toml`
- Test with dependency updates monthly

---

## Performance Targets

### Build Performance
- **WASM build time**: <60 seconds (optimized)
- **Incremental rebuild**: <10 seconds
- **CI/CD pipeline**: <5 minutes total

### Runtime Performance
- **Translation speed**: >1000 LOC/sec (native), >500 LOC/sec (WASM)
- **Memory usage**: <50MB for typical programs
- **WASM bundle size**: <500KB compressed

### Quality Metrics
- **Test coverage**: >90% of core translator
- **Test pass rate**: >96% (210+ tests)
- **Documentation coverage**: 100% of public API

---

## Communication Plan

### Daily Standup
- **When**: End of each day
- **Format**: Written summary in `DAILY_PROGRESS.md`
- **Content**: Completed tasks, blockers, next steps

### Phase Reviews
- **When**: End of each phase
- **Format**: Detailed report document
- **Content**: Achievements, metrics, lessons learned, next phase plan

### Issue Tracking
- **Tool**: GitHub Issues
- **Labels**: phase1, phase2, phase3, phase4, bug, enhancement, documentation
- **Triage**: Daily review and prioritization

### Stakeholder Updates
- **Frequency**: Weekly
- **Format**: Email summary + demo video
- **Content**: Progress metrics, demo, upcoming milestones

---

## Success Criteria Summary

### Phase 1 (Days 1-3): WASM Infrastructure
- ✅ WASM builds for web, nodejs, bundler
- ✅ NPM package structure
- ✅ CI/CD pipeline
- ✅ Basic WASM tests passing

### Phase 2 (Days 4-5): End-to-End Examples
- ✅ Browser demo functional
- ✅ Node.js CLI tool
- ✅ 10+ working examples
- ✅ E2E validation passing

### Phase 3 (Days 6-10): High Priority Features
- ✅ Import system operational
- ✅ Stdlib mapping (20+ functions)
- ✅ Tuple unpacking fixed
- ✅ With statement support
- ✅ 95%+ test pass rate (208+ tests)

### Phase 4 (Days 11-15): Medium Priority Features
- ✅ Decorator support (basic)
- ✅ Generator support (simple cases)
- ✅ Async/await (optional)
- ✅ 70+ stdlib functions
- ✅ 96%+ test pass rate (210+ tests)
- ✅ Production-ready release

### Overall Success Metrics
- **Test pass rate**: 96%+ (210/219 tests)
- **Feature coverage**: 200+ Python features
- **WASM bundle size**: <500KB
- **Performance**: 10-50x faster than Python (WASM runtime)
- **Documentation**: Complete and accurate
- **Examples**: 20+ working demos
- **Release**: NPM package published

---

## Risk Assessment Matrix

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| WASM build failures | Low | High | Incremental testing, feature flags | Disable problematic features |
| Parser regression | Medium | High | Comprehensive test suite, CI/CD | Quick rollback procedures |
| Performance degradation | Medium | Medium | Continuous benchmarking | Optimize critical paths |
| Browser compatibility | Low | Medium | Multi-browser testing | Polyfills, fallbacks |
| Scope creep | High | Medium | Time-boxing, prioritization | Skip low-priority features |
| Documentation lag | Medium | Low | Concurrent documentation | Documentation sprint Day 15 |
| Dependency conflicts | Low | Medium | Version pinning, testing | Lock file management |
| WASM memory limits | Low | High | Size optimization, monitoring | Streaming, chunking |

---

## Timeline Overview

```
Days 1-3:  ████████░░░░░░░░░░░░░░░░░░  Phase 1: Infrastructure
Days 4-5:  ░░░░░░░░████░░░░░░░░░░░░░░  Phase 2: Examples
Days 6-10: ░░░░░░░░░░░░██████████░░░░  Phase 3: High Priority
Days 11-15:░░░░░░░░░░░░░░░░░░░░░░████  Phase 4: Polish

Day 1:  Toolchain, WASM deps, bindings
Day 2:  Build scripts, NPM package, CI/CD
Day 3:  WASM tests, benchmarks, docs
Day 4:  Browser demo
Day 5:  Node.js integration, E2E validation
Day 6:  Import parsing & translation
Day 7:  Stdlib mapping
Day 8:  Tuple unpacking fixes
Day 9:  With statement support
Day 10: Parser fixes, validation
Day 11: Decorator support
Day 12: Generator/yield support
Day 13: Async/await (optional)
Day 14: Enhanced stdlib mapping
Day 15: Final integration, release prep
```

---

## Deliverables Checklist

### Code
- [ ] WASM bindings module (`agents/transpiler/src/wasm.rs`)
- [ ] Build scripts (`scripts/build-wasm.sh`)
- [ ] Browser demo (`examples/browser-demo/`)
- [ ] Node.js CLI (`examples/nodejs-cli/`)
- [ ] Import system implementation
- [ ] Stdlib mapper (`agents/transpiler/src/stdlib_map.rs`)
- [ ] Parser fixes (tuple unpacking, multiple assignment)
- [ ] With statement support
- [ ] Decorator support
- [ ] Generator support (if time permits)
- [ ] Async support (if time permits)

### Testing
- [ ] WASM test suite (10+ tests)
- [ ] Integration tests (browser, Node.js)
- [ ] E2E validation script
- [ ] Performance benchmarks
- [ ] Regression test suite

### Documentation
- [ ] WASM Deployment Guide
- [ ] Stdlib Support Matrix
- [ ] API Documentation
- [ ] Migration Guide
- [ ] Troubleshooting Guide
- [ ] Performance Report
- [ ] Release Notes

### Infrastructure
- [ ] CI/CD pipeline
- [ ] NPM package
- [ ] Multi-target WASM builds
- [ ] Development server setup

### Examples
- [ ] 10+ browser examples
- [ ] 5+ Node.js examples
- [ ] E2E validation examples
- [ ] Performance comparison examples

---

## Post-Implementation

### Maintenance Plan
- **Bug fixes**: Weekly triage, priority-based fixing
- **Dependency updates**: Monthly review and updates
- **Security patches**: Immediate application
- **Feature requests**: Quarterly review and prioritization

### Future Enhancements
- **Phase 5**: Full async/await with runtime selection
- **Phase 6**: Advanced OOP (inheritance, metaclasses)
- **Phase 7**: Complete stdlib mapping (200+ functions)
- **Phase 8**: Python 3.13 feature support
- **Phase 9**: Source maps and debugging tools
- **Phase 10**: WASM SIMD optimization

### Community Engagement
- **Open source release**: Publish to GitHub
- **NPM publication**: Publish package
- **Documentation site**: Deploy to GitHub Pages
- **Blog posts**: Technical deep-dives
- **Conference talks**: Present at Rust/WASM meetups

---

## Conclusion

This implementation strategy provides a **clear, actionable roadmap** for deploying Portalis as a production-ready WASM application. The **incremental approach** with continuous testing and validation ensures high quality while maintaining the ability to quickly rollback problematic changes.

**Key Success Factors:**
1. **Incremental delivery**: Each phase builds on previous success
2. **Continuous testing**: Catch regressions early
3. **Clear rollback paths**: Minimize risk
4. **Performance focus**: Monitor and optimize throughout
5. **Comprehensive documentation**: Enable adoption and maintenance

**Expected Outcome:**
- Production-ready Python → Rust → WASM transpiler
- 210+ tests passing (96%+ pass rate)
- Browser and Node.js deployment examples
- 200+ Python features supported
- Complete documentation and examples
- Published NPM package
- 10-50x performance improvement over Python

The strategy is **realistic, achievable, and provides clear value** at each phase, ensuring stakeholder confidence and project success.
