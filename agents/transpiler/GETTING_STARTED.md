# Getting Started with Portalis Transpiler

This guide will help you install, configure, and use the Portalis Transpiler to convert Python code to Rust and deploy it as WebAssembly.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Your First Translation](#your-first-translation)
5. [WASM Deployment](#wasm-deployment)
6. [Common Workflows](#common-workflows)
7. [Next Steps](#next-steps)

---

## Prerequisites

### Required

- **Rust**: 1.70+ (stable)
- **Cargo**: Latest version

```bash
# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version
```

### Optional (for WASM deployment)

- **wasm-bindgen-cli**: WASM/JavaScript bindings
- **wasm-pack**: Build automation
- **binaryen**: WASM optimization (wasm-opt)

```bash
# Install WASM tools
cargo install wasm-bindgen-cli
cargo install wasm-pack

# Install binaryen (platform-specific)
# macOS
brew install binaryen

# Ubuntu/Debian
sudo apt-get install binaryen

# Windows (download from releases)
# https://github.com/WebAssembly/binaryen/releases
```

---

## Installation

### Option 1: Add as Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
portalis-transpiler = "1.0.0"
```

### Option 2: Clone from Source

```bash
# Clone repository
git clone https://github.com/portalis/transpiler.git
cd transpiler/agents/transpiler

# Build
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example async_runtime_demo
```

### Option 3: Install Binary (future)

```bash
# Will be available in v1.1+
cargo install portalis-transpiler
portalis --version
```

---

## Quick Start

### 1. Create a New Project

```bash
cargo new my_transpiler_project
cd my_transpiler_project
```

### 2. Add Dependency

Edit `Cargo.toml`:

```toml
[dependencies]
portalis-transpiler = "1.0.0"
```

### 3. Write Your First Transpiler

Edit `src/main.rs`:

```rust
use portalis_transpiler::PyToRustTranspiler;

fn main() {
    let python_code = r#"
def greet(name: str) -> str:
    return f"Hello, {name}!"

result = greet("World")
print(result)
"#;

    let mut transpiler = PyToRustTranspiler::new();
    let rust_code = transpiler.translate(python_code);

    println!("=== Rust Output ===");
    println!("{}", rust_code);
}
```

### 4. Run It

```bash
cargo run
```

**Expected Output**:
```rust
=== Rust Output ===
fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

fn main() {
    let result = greet("World");
    println!("{}", result);
}
```

---

## Your First Translation

### Example: Fibonacci Function

**Create `fibonacci.py`**:
```python
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Test it
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
```

**Translate It** (`src/main.rs`):
```rust
use portalis_transpiler::PyToRustTranspiler;
use std::fs;

fn main() {
    // Read Python file
    let python_code = fs::read_to_string("fibonacci.py")
        .expect("Failed to read fibonacci.py");

    // Translate
    let mut transpiler = PyToRustTranspiler::new();
    let rust_code = transpiler.translate(&python_code);

    // Write output
    fs::write("fibonacci.rs", &rust_code)
        .expect("Failed to write fibonacci.rs");

    println!("✅ Translation complete: fibonacci.rs");
    println!("\n{}", rust_code);
}
```

**Run**:
```bash
cargo run
```

**Output** (`fibonacci.rs`):
```rust
/// Calculate nth Fibonacci number
fn fibonacci(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}

fn main() {
    for i in 0..10 {
        println!("fib({}) = {}", i, fibonacci(i));
    }
}
```

---

## WASM Deployment

### Step 1: Set Up WASM Target

```bash
# Add WASM target
rustup target add wasm32-unknown-unknown
rustup target add wasm32-wasi
```

### Step 2: Translate and Bundle

**Create `deploy_wasm.rs`**:
```rust
use portalis_transpiler::{
    PyToRustTranspiler,
    WasmBundler,
    BundleConfig,
    DeploymentTarget,
    OptimizationLevel,
    CompressionFormat,
};
use std::fs;

fn main() {
    // Step 1: Translate Python to Rust
    let python_code = fs::read_to_string("my_script.py")
        .expect("Failed to read my_script.py");

    let mut transpiler = PyToRustTranspiler::new();
    let rust_code = transpiler.translate(&python_code);

    fs::write("src/lib.rs", &rust_code)
        .expect("Failed to write src/lib.rs");

    println!("✅ Translation complete");

    // Step 2: Configure WASM bundling
    let mut config = BundleConfig::production();
    config.package_name = "my_wasm_app".to_string();
    config.target = DeploymentTarget::Web;
    config.optimization_level = OptimizationLevel::MaxSize;
    config.compression = CompressionFormat::Brotli;
    config.generate_readme = true;

    // Step 3: Generate bundle
    let bundler = WasmBundler::new(config);
    let report = bundler.generate_bundle("my_wasm_app");

    println!("\n{}", report);
    println!("\n✅ WASM bundle ready in dist/web/");
}
```

### Step 3: Build WASM

```bash
# Build the transpiler
cargo build --release

# Run the WASM deployment
cargo run --bin deploy_wasm

# Your WASM is now in: dist/web/my_wasm_app.wasm
```

### Step 4: Use in Browser

**Generated `dist/web/index.html`**:
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>My WASM App</title>
</head>
<body>
    <script type="module">
        import init, { greet } from './my_wasm_app.js';

        async function run() {
            await init();
            const result = greet("World");
            console.log(result);
        }

        run();
    </script>
</body>
</html>
```

**Serve it**:
```bash
# Install simple HTTP server
python3 -m http.server 8000 -d dist/web

# Or use Node.js
npx http-server dist/web -p 8000

# Open http://localhost:8000
```

---

## Common Workflows

### Workflow 1: Simple Script Translation

```rust
use portalis_transpiler::PyToRustTranspiler;

let mut transpiler = PyToRustTranspiler::new();
let rust_code = transpiler.translate("def add(a, b): return a + b");
println!("{}", rust_code);
```

### Workflow 2: File-to-File Translation

```rust
use portalis_transpiler::PyToRustTranspiler;
use std::fs;

fn translate_file(input: &str, output: &str) {
    let python = fs::read_to_string(input).unwrap();
    let mut transpiler = PyToRustTranspiler::new();
    let rust = transpiler.translate(&python);
    fs::write(output, rust).unwrap();
}

translate_file("input.py", "output.rs");
```

### Workflow 3: Batch Translation

```rust
use portalis_transpiler::PyToRustTranspiler;
use std::fs;

fn translate_directory(input_dir: &str, output_dir: &str) {
    let mut transpiler = PyToRustTranspiler::new();

    for entry in fs::read_dir(input_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.extension() == Some("py".as_ref()) {
            let python = fs::read_to_string(&path).unwrap();
            let rust = transpiler.translate(&python);

            let output_path = format!(
                "{}/{}.rs",
                output_dir,
                path.file_stem().unwrap().to_str().unwrap()
            );

            fs::write(output_path, rust).unwrap();
        }
    }
}

translate_directory("python_src", "rust_src");
```

### Workflow 4: Async HTTP Client

**Python** (`fetch_data.py`):
```python
import asyncio
import aiohttp

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def main():
    data = await fetch_data("https://api.example.com/data")
    print(data)

asyncio.run(main())
```

**Translate and Build**:
```rust
use portalis_transpiler::{PyToRustTranspiler, CargoGenerator, CargoConfig};
use std::fs;

fn main() {
    // Translate
    let python = fs::read_to_string("fetch_data.py").unwrap();
    let mut transpiler = PyToRustTranspiler::new();
    let rust = transpiler.translate(&python);

    fs::write("src/main.rs", rust).unwrap();

    // Generate Cargo.toml with dependencies
    let mut config = CargoConfig::default();
    config.package_name = "fetch_data".to_string();
    config.is_async = true; // Adds tokio

    let generator = CargoGenerator::new(config);
    let cargo_toml = generator.generate();

    fs::write("Cargo.toml", cargo_toml).unwrap();

    println!("✅ Ready to build: cargo run");
}
```

### Workflow 5: Data Processing Pipeline

**Python** (`process_data.py`):
```python
import pandas as pd

def process_sales_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df['total'] = df['quantity'] * df['price']
    summary = df.groupby('category')['total'].sum()
    return summary.sort_values(ascending=False)
```

**Translate**:
```rust
use portalis_transpiler::PyToRustTranspiler;
use std::fs;

fn main() {
    let python = fs::read_to_string("process_data.py").unwrap();
    let mut transpiler = PyToRustTranspiler::new();
    let rust = transpiler.translate(&python);

    println!("{}", rust);
}
```

**Output**:
```rust
use polars::prelude::*;

fn process_sales_data(filename: &str) -> Result<DataFrame, PolarsError> {
    let mut df = CsvReader::from_path(filename)?.finish()?;

    df = df.lazy()
        .with_column(
            (col("quantity") * col("price")).alias("total")
        )
        .collect()?;

    df.lazy()
        .groupby([col("category")])
        .agg([col("total").sum()])
        .sort("total", SortOptions::default().with_order_descending(true))
        .collect()
}
```

---

## Next Steps

### Learn More

1. **[User Guide](./USER_GUIDE.md)** - Comprehensive feature documentation
2. **[API Reference](./API_REFERENCE.md)** - Detailed API documentation
3. **[Examples](./EXAMPLES.md)** - More code examples and tutorials
4. **[Migration Guide](./MIGRATION_GUIDE.md)** - Python → Rust migration patterns

### Explore Examples

```bash
# Run all examples
cargo run --example async_runtime_demo
cargo run --example asyncio_translation_example
cargo run --example wasm_bundler_demo
cargo run --example dead_code_elimination_demo
cargo run --example websocket_example

# See all examples
ls examples/
```

### Common Tasks

- **Add library support**: See [USER_GUIDE.md - Library Mapping](./USER_GUIDE.md#library-mapping)
- **Optimize WASM size**: See [USER_GUIDE.md - Optimization](./USER_GUIDE.md#optimization)
- **Handle errors**: See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- **Advanced features**: See [USER_GUIDE.md - Advanced](./USER_GUIDE.md#advanced-features)

### Get Help

- **Issues**: Report bugs or request features on GitHub
- **Questions**: Ask in GitHub Discussions
- **Documentation**: Browse full docs at [docs.portalis.dev](https://docs.portalis.dev)

---

## Quick Reference

### Basic Translation

```rust
use portalis_transpiler::PyToRustTranspiler;

let mut transpiler = PyToRustTranspiler::new();
let rust = transpiler.translate(python_code);
```

### With Cargo Generation

```rust
use portalis_transpiler::{PyToRustTranspiler, CargoGenerator, CargoConfig};

let mut transpiler = PyToRustTranspiler::new();
let rust = transpiler.translate(python_code);

let config = CargoConfig::default();
let generator = CargoGenerator::new(config);
let cargo_toml = generator.generate();
```

### WASM Deployment

```rust
use portalis_transpiler::{WasmBundler, BundleConfig, DeploymentTarget};

let mut config = BundleConfig::production();
config.target = DeploymentTarget::Web;

let bundler = WasmBundler::new(config);
bundler.generate_bundle("my_app");
```

### Optimization

```rust
use portalis_transpiler::{DeadCodeEliminator, OptimizationStrategy};

let mut eliminator = DeadCodeEliminator::new();
let optimized = eliminator.analyze_with_strategy(
    rust_code,
    OptimizationStrategy::Aggressive
);
```

---

**You're ready to start transpiling Python to Rust!** 🚀

For more advanced usage, continue to the [User Guide](./USER_GUIDE.md).
