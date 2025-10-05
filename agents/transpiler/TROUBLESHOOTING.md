# Troubleshooting Guide - Portalis Transpiler

Solutions to common issues and errors when using the Portalis Transpiler.

---

## Table of Contents

1. [Translation Errors](#translation-errors)
2. [Type Inference Issues](#type-inference-issues)
3. [Build Errors](#build-errors)
4. [WASM Issues](#wasm-issues)
5. [Performance Problems](#performance-problems)
6. [Library Compatibility](#library-compatibility)
7. [Runtime Errors](#runtime-errors)
8. [Getting Help](#getting-help)

---

## Translation Errors

### Error: Parse Error - Invalid Python Syntax

**Symptom**:
```
Error: ParseError("invalid syntax at line 15")
```

**Cause**: Python code has syntax errors or uses unsupported Python version

**Solutions**:

1. **Verify Python syntax**:
```bash
python3 -m py_compile your_script.py
```

2. **Check Python version compatibility**:
   - Portalis supports Python 3.8+
   - Some Python 3.11+ features may not be supported yet

3. **Look for common issues**:
```python
# Problem: Missing colon
def my_function()
    pass

# Fix:
def my_function():
    pass
```

---

### Error: Unsupported Feature

**Symptom**:
```
Error: UnsupportedFeature("metaclasses are not supported")
```

**Cause**: Python feature not yet implemented in transpiler

**Solutions**:

1. **Check supported features**:
   - See [README.md](./README.md) for full list
   - Avoid: metaclasses, eval(), exec(), dynamic imports

2. **Refactor code**:
```python
# Problem: Metaclass
class MyMeta(type):
    pass

class MyClass(metaclass=MyMeta):
    pass

# Fix: Use regular class with trait pattern
class MyClass:
    pass
```

3. **Submit feature request**:
   - Open GitHub issue for unsupported features
   - Provide minimal example

---

### Error: Import Not Found

**Symptom**:
```
Error: ImportError("module 'scikit-learn' not found in mappings")
```

**Cause**: Python library not mapped to Rust equivalent

**Solutions**:

1. **Check supported libraries**:
```rust
// Supported libraries (see README.md):
// - requests → reqwest
// - numpy → ndarray
// - pandas → polars
// - asyncio → tokio
// etc.
```

2. **Manual translation**:
```python
# If library not supported, translate manually
import unsupported_lib

def my_function():
    unsupported_lib.do_something()
```

Becomes:
```rust
// Find Rust equivalent and write manually
use rust_equivalent;

fn my_function() {
    rust_equivalent::do_something();
}
```

3. **Add custom mapping** (advanced):
```rust
use portalis_transpiler::ImportAnalyzer;

let mut analyzer = ImportAnalyzer::new();
analyzer.add_mapping("my_python_lib", "my_rust_crate");
```

---

## Type Inference Issues

### Error: Type Mismatch

**Symptom**:
```
Error: TypeError("expected i32, found String at line 42")
```

**Cause**: Type inference couldn't reconcile conflicting types

**Solutions**:

1. **Add explicit type annotations**:
```python
# Problem: Ambiguous types
def process(x):
    return x * 2

# Fix: Add type hints
def process(x: int) -> int:
    return x * 2
```

2. **Avoid type changes**:
```python
# Problem: Variable changes type
x = 42
x = "hello"  # Type changed!

# Fix: Use different variables
num = 42
text = "hello"
```

3. **Use Union types**:
```python
from typing import Union

def process(x: Union[int, str]) -> str:
    if isinstance(x, int):
        return str(x)
    return x
```

---

### Error: Cannot Infer Lifetime

**Symptom**:
```
Error: LifetimeError("cannot infer lifetime for return value")
```

**Cause**: Lifetime analysis couldn't determine reference lifetime

**Solutions**:

1. **Simplify function**:
```python
# Problem: Complex references
def get_first(items):
    if items:
        return items[0]
    return None

# Fix: Use explicit return types
from typing import Optional, List

def get_first(items: List[str]) -> Optional[str]:
    return items[0] if items else None
```

2. **Avoid returning references to local data**:
```python
# Problem:
def create_list():
    items = [1, 2, 3]
    return items[0]  # OK: returns value, not reference

# Problem:
def get_reference():
    text = "hello"
    return text  # May cause issues
```

3. **Review generated Rust**:
   - Check if lifetimes are needed
   - Manually add lifetime annotations if necessary

---

## Build Errors

### Error: Cargo Build Failed

**Symptom**:
```
error[E0425]: cannot find function `some_function` in this scope
```

**Cause**: Generated Rust code has compilation errors

**Solutions**:

1. **Check generated code**:
```bash
# Review generated Rust
cat src/lib.rs

# Try building with verbose output
cargo build --verbose
```

2. **Common fixes**:

```rust
// Problem: Missing import
fn main() {
    let json = serde_json::json!({"key": "value"});  // Error
}

// Fix: Add use statement
use serde_json::json;

fn main() {
    let json = json!({"key": "value"});
}
```

3. **Regenerate Cargo.toml**:
```rust
use portalis_transpiler::{CargoGenerator, CargoConfig};

let mut config = CargoConfig::default();
config.is_async = true;  // Add missing features
config.http_client = true;

let generator = CargoGenerator::new(config);
std::fs::write("Cargo.toml", generator.generate()).unwrap();
```

---

### Error: Missing Dependencies

**Symptom**:
```
error[E0432]: unresolved import `tokio`
```

**Cause**: Required crate not in Cargo.toml

**Solutions**:

1. **Add missing dependency**:
```toml
# Add to Cargo.toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
```

2. **Use CargoGenerator with all features**:
```rust
let mut config = CargoConfig::default();
config.is_async = true;
config.http_client = true;
config.wasm_target = true;
```

3. **Check for typos**:
```toml
# Wrong:
reqwuest = "0.11"  # Typo

# Correct:
reqwest = "0.11"
```

---

## WASM Issues

### Error: wasm-bindgen Not Found

**Symptom**:
```
Error: wasm-bindgen: command not found
```

**Cause**: wasm-bindgen-cli not installed

**Solutions**:

```bash
# Install wasm-bindgen
cargo install wasm-bindgen-cli

# Verify installation
wasm-bindgen --version

# If version mismatch:
cargo install wasm-bindgen-cli --force
```

---

### Error: WASM Build Fails

**Symptom**:
```
error: linking with `rust-lld` failed
```

**Cause**: WASM target not added or wrong target used

**Solutions**:

1. **Add WASM target**:
```bash
rustup target add wasm32-unknown-unknown
rustup target add wasm32-wasi
```

2. **Verify target**:
```bash
rustup target list --installed
```

3. **Check Cargo.toml**:
```toml
[lib]
crate-type = ["cdylib", "rlib"]  # Required for WASM
```

4. **Clean and rebuild**:
```bash
cargo clean
cargo build --target wasm32-unknown-unknown --release
```

---

### Error: WASM Module Too Large

**Symptom**:
```
warning: WASM binary is 5.2 MB (expected < 1 MB)
```

**Cause**: No optimization applied

**Solutions**:

1. **Enable optimization**:
```rust
use portalis_transpiler::{BundleConfig, OptimizationLevel, CompressionFormat};

let mut config = BundleConfig::production();
config.optimization_level = OptimizationLevel::MaxSize;
config.optimize_size = true;
config.compression = CompressionFormat::Brotli;
```

2. **Run dead code elimination**:
```rust
use portalis_transpiler::{DeadCodeEliminator, OptimizationStrategy};

let mut eliminator = DeadCodeEliminator::new();
let optimized = eliminator.analyze_with_strategy(
    rust_code,
    OptimizationStrategy::Aggressive
);
```

3. **Use wasm-opt**:
```bash
wasm-opt -Ozz input.wasm -o output.wasm
```

4. **Check dependency sizes**:
```bash
# Analyze binary size
cargo bloat --release --target wasm32-unknown-unknown
```

**Expected sizes**:
- Without optimization: 1-5 MB
- With dead code elimination: 400-800 KB
- With wasm-opt: 200-400 KB
- With Brotli compression: 50-150 KB

---

## Performance Problems

### Issue: Slow Translation

**Symptom**: Translation takes >10 seconds for medium files

**Solutions**:

1. **Profile the translation**:
```bash
cargo build --release
time cargo run --release -- translate input.py
```

2. **Optimize type inference**:
```python
# Add type hints to speed up inference
def process_data(items: List[int]) -> int:
    return sum(items)
```

3. **Split large files**:
```python
# Instead of one 5000-line file
# Split into multiple modules

# module1.py
def function1():
    pass

# module2.py
def function2():
    pass
```

4. **Use caching** (future feature):
```rust
// Will be available in v1.1+
transpiler.enable_cache(true);
```

---

### Issue: High Memory Usage

**Symptom**: Translation uses >1GB RAM

**Solutions**:

1. **Process in chunks**:
```rust
// Instead of:
let all_code = read_all_files();
transpiler.translate(&all_code);

// Do:
for file in files {
    let code = read_file(file);
    let rust = transpiler.translate(&code);
    write_output(rust);
}
```

2. **Reduce AST depth**:
```python
# Avoid deeply nested structures
# Problem:
def outer():
    def middle():
        def inner():
            def deepest():
                pass

# Better: Flatten hierarchy
```

---

## Library Compatibility

### Issue: NumPy Operation Not Supported

**Symptom**: Some NumPy operations don't translate correctly

**Solutions**:

1. **Check supported operations**:
```python
# Supported:
arr = np.array([1, 2, 3])
arr.sum()
arr.mean()
arr * 2

# Not supported (yet):
np.fft.fft(arr)  # FFT
np.linalg.svd(matrix)  # Advanced linear algebra
```

2. **Use basic operations**:
```python
# Instead of complex operation
result = np.complicated_operation(arr)

# Use simpler alternatives
result = arr.sum() / len(arr)
```

3. **Manual translation**:
```rust
// Translate complex NumPy manually
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

let arr = array![1.0, 2.0, 3.0, 4.0, 5.0];
let median = arr.quantile_mut(0.5, &Midpoint).unwrap();
```

---

### Issue: Async Library Mismatch

**Symptom**: async code doesn't work correctly

**Solutions**:

1. **Ensure consistent runtime**:
```rust
// Use same runtime throughout
#[tokio::main]
async fn main() {
    // All async code uses tokio
}

// Don't mix runtimes:
// #[async_std::main]  // Different runtime!
```

2. **Add runtime features**:
```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
```

3. **Check async translation**:
```python
# Python asyncio
async def fetch():
    async with aiohttp.ClientSession() as session:
        ...

# Becomes Rust tokio
async fn fetch() -> Result<_, Error> {
    let response = reqwest::get(...).await?;
    ...
}
```

---

## Runtime Errors

### Error: WASM Panic

**Symptom**:
```
panicked at 'index out of bounds'
```

**Cause**: Array access without bounds checking

**Solutions**:

1. **Add bounds checking**:
```rust
// Problem:
let item = arr[index];  // May panic

// Fix:
let item = arr.get(index).unwrap_or(&default);

// Or:
if index < arr.len() {
    let item = arr[index];
}
```

2. **Use Result types**:
```rust
fn get_item(arr: &[i32], index: usize) -> Result<i32, String> {
    arr.get(index)
        .copied()
        .ok_or_else(|| format!("Index {} out of bounds", index))
}
```

3. **Enable panic hooks** (WASM):
```rust
use std::panic;

#[wasm_bindgen(start)]
pub fn main() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}
```

---

### Error: Async Function Not Awaited

**Symptom**:
```
warning: unused implementer of `Future` that must be used
```

**Cause**: Forgot to `.await` async function

**Solutions**:

```rust
// Problem:
let result = fetch_data(url);  // Missing .await

// Fix:
let result = fetch_data(url).await?;
```

---

## Getting Help

### Debug Checklist

Before asking for help, try:

- [ ] Check error message carefully
- [ ] Review this troubleshooting guide
- [ ] Read relevant documentation section
- [ ] Search GitHub issues
- [ ] Create minimal reproducible example
- [ ] Check transpiler version

### Creating a Minimal Example

**Good bug report**:
```python
# Input (minimal.py)
def add(a: int, b: int) -> int:
    return a + b

# Error:
# TypeError: expected i32, found String
```

**Bad bug report**:
```
My 5000-line project doesn't work
Error somewhere in the code
Please fix
```

### Reporting Issues

**Include**:
1. Python code (minimal example)
2. Generated Rust code (if applicable)
3. Full error message
4. Transpiler version
5. Rust version (`rustc --version`)
6. OS and architecture

**Template**:
```markdown
## Description
Brief description of the issue

## Python Code
```python
def my_function():
    pass
```

## Expected Output
What you expected to happen

## Actual Output
What actually happened (error message)

## Environment
- Portalis version: 1.0.0
- Rust version: 1.75.0
- OS: Ubuntu 22.04
```

---

## Common Error Reference

### Quick Solutions

| Error | Quick Fix |
|-------|-----------|
| `ParseError` | Check Python syntax |
| `TypeError` | Add type hints |
| `ImportError` | Check supported libraries |
| `LifetimeError` | Simplify function |
| `wasm-bindgen not found` | `cargo install wasm-bindgen-cli` |
| `target not found` | `rustup target add wasm32-unknown-unknown` |
| `panic in WASM` | Add bounds checking |
| `missing await` | Add `.await` |

---

## Debugging Tools

### Useful Commands

```bash
# Check Python syntax
python3 -m py_compile script.py

# Format Python
black script.py

# Check Rust
cargo check

# Format Rust
cargo fmt

# Lint Rust
cargo clippy

# Test Rust
cargo test

# Build with verbose output
cargo build --verbose

# Profile build
cargo build --timings

# Check binary size
cargo bloat --release
```

### Environment Variables

```bash
# Rust logging
export RUST_LOG=debug
export RUST_BACKTRACE=1

# WASM debugging
export WASM_BINDGEN_DEBUG=1
```

---

## FAQ

**Q: Why is my translation failing?**
A: Check Python syntax first, then verify the feature is supported.

**Q: Can I translate any Python library?**
A: No, only supported libraries. See [README.md](./README.md) for full list.

**Q: How do I optimize WASM size?**
A: Use `OptimizationLevel::MaxSize`, dead code elimination, and Brotli compression.

**Q: Is Python 2 supported?**
A: No, only Python 3.8+.

**Q: Can I contribute new library mappings?**
A: Yes! See [ARCHITECTURE.md](./ARCHITECTURE.md) for extension points.

**Q: Why is translation slow?**
A: Add type hints, split large files, use release build.

**Q: How do I report bugs?**
A: GitHub Issues with minimal reproducible example.

---

## Resources

- **Documentation**: [README.md](./README.md), [USER_GUIDE.md](./USER_GUIDE.md)
- **Examples**: [EXAMPLES.md](./EXAMPLES.md)
- **API Reference**: [API_REFERENCE.md](./API_REFERENCE.md)
- **GitHub**: https://github.com/portalis/transpiler/issues
- **Discussions**: https://github.com/portalis/transpiler/discussions

---

**Still stuck?** Ask in GitHub Discussions or open an issue with a minimal reproducible example.
