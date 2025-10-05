# Portalis Transpiler - User Guide

Complete guide to using the Portalis Transpiler for Python to Rust translation and WASM deployment.

---

## Table of Contents

1. [Core Translation](#core-translation)
2. [Type System](#type-system)
3. [Library Mapping](#library-mapping)
4. [Advanced Features](#advanced-features)
5. [WASM Deployment](#wasm-deployment)
6. [Optimization](#optimization)
7. [Package Management](#package-management)
8. [Configuration](#configuration)
9. [Best Practices](#best-practices)
10. [Limitations](#limitations)

---

## Core Translation

### Basic Usage

```rust
use portalis_transpiler::PyToRustTranspiler;

let mut transpiler = PyToRustTranspiler::new();
let rust_code = transpiler.translate(python_code);
```

### Function Translation

**Python**:
```python
def add(a: int, b: int) -> int:
    return a + b

def multiply(x: float, y: float) -> float:
    return x * y

def greet(name: str) -> str:
    return f"Hello, {name}!"
```

**Rust Output**:
```rust
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn multiply(x: f64, y: f64) -> f64 {
    x * y
}

fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
```

### Class Translation

**Python**:
```python
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def move(self, dx: int, dy: int):
        self.x += dx
        self.y += dy
```

**Rust Output**:
```rust
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }

    fn distance_from_origin(&self) -> f64 {
        ((self.x.pow(2) + self.y.pow(2)) as f64).sqrt()
    }

    fn move_point(&mut self, dx: i32, dy: i32) {
        self.x += dx;
        self.y += dy;
    }
}
```

### Control Flow

**Python**:
```python
def classify_number(n: int) -> str:
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    else:
        return "positive"

def sum_range(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total

def factorial(n: int) -> int:
    result = 1
    while n > 1:
        result *= n
        n -= 1
    return result
```

**Rust Output**:
```rust
fn classify_number(n: i32) -> &'static str {
    if n < 0 {
        "negative"
    } else if n == 0 {
        "zero"
    } else {
        "positive"
    }
}

fn sum_range(n: i32) -> i32 {
    let mut total = 0;
    for i in 0..n {
        total += i;
    }
    total
}

fn factorial(n: i32) -> i32 {
    let mut result = 1;
    let mut n = n;
    while n > 1 {
        result *= n;
        n -= 1;
    }
    result
}
```

### Comprehensions

**Python**:
```python
# List comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested
matrix = [[i*j for j in range(5)] for i in range(5)]

# Dict comprehension
word_lengths = {word: len(word) for word in ["hello", "world"]}
```

**Rust Output**:
```rust
// Iterator chain
let squares: Vec<i32> = (0..10).map(|x| x.pow(2)).collect();

// With filter
let evens: Vec<i32> = (0..20).filter(|x| x % 2 == 0).collect();

// Nested
let matrix: Vec<Vec<i32>> = (0..5)
    .map(|i| (0..5).map(|j| i * j).collect())
    .collect();

// HashMap
use std::collections::HashMap;
let word_lengths: HashMap<&str, usize> =
    vec!["hello", "world"]
        .iter()
        .map(|&word| (word, word.len()))
        .collect();
```

---

## Type System

### Type Inference

The transpiler uses **Hindley-Milner type inference** to automatically determine types when not explicitly annotated.

**Python**:
```python
def process_data(items):
    # No type annotations
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result
```

**Rust Output** (inferred types):
```rust
fn process_data(items: &[i32]) -> Vec<i32> {
    let mut result = Vec::new();
    for &item in items {
        if item > 0 {
            result.push(item * 2);
        }
    }
    result
}
```

### Generic Types

**Python**:
```python
from typing import List, Optional, Dict, Tuple

def find_first(items: List[int], predicate) -> Optional[int]:
    for item in items:
        if predicate(item):
            return item
    return None

def group_by_key(items: List[Tuple[str, int]]) -> Dict[str, List[int]]:
    groups = {}
    for key, value in items:
        if key not in groups:
            groups[key] = []
        groups[key].append(value)
    return groups
```

**Rust Output**:
```rust
fn find_first<F>(items: &[i32], predicate: F) -> Option<i32>
where
    F: Fn(i32) -> bool,
{
    for &item in items {
        if predicate(item) {
            return Some(item);
        }
    }
    None
}

use std::collections::HashMap;

fn group_by_key(items: Vec<(String, i32)>) -> HashMap<String, Vec<i32>> {
    let mut groups = HashMap::new();
    for (key, value) in items {
        groups.entry(key).or_insert_with(Vec::new).push(value);
    }
    groups
}
```

### Lifetime Analysis

The transpiler automatically infers lifetimes and chooses between owned and borrowed types.

**Python**:
```python
def longest(s1: str, s2: str) -> str:
    if len(s1) > len(s2):
        return s1
    return s2

def first_word(s: str) -> str:
    for i, char in enumerate(s):
        if char == ' ':
            return s[:i]
    return s
```

**Rust Output**:
```rust
fn longest<'a>(s1: &'a str, s2: &'a str) -> &'a str {
    if s1.len() > s2.len() {
        s1
    } else {
        s2
    }
}

fn first_word(s: &str) -> &str {
    for (i, ch) in s.chars().enumerate() {
        if ch == ' ' {
            return &s[..i];
        }
    }
    s
}
```

---

## Library Mapping

### Standard Library

| Python | Rust | Usage |
|--------|------|-------|
| `os.path` | `std::path::Path` | Path operations |
| `sys.argv` | `std::env::args()` | Command-line args |
| `json` | `serde_json` | JSON serialization |
| `datetime` | `chrono` | Date/time handling |
| `re` | `regex` | Regular expressions |
| `collections` | `std::collections` | Data structures |
| `logging` | `log` | Logging |
| `threading` | `std::thread` | Threading |
| `asyncio` | `tokio` | Async runtime |

### HTTP / Requests

**Python**:
```python
import requests

# GET request
response = requests.get("https://api.example.com/data")
data = response.json()

# POST request
response = requests.post(
    "https://api.example.com/submit",
    json={"key": "value"},
    headers={"Authorization": "Bearer token"}
)

# Session
session = requests.Session()
session.headers.update({"User-Agent": "MyApp"})
response = session.get("https://api.example.com/profile")
```

**Rust Output**:
```rust
use reqwest;

#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    // GET request
    let response = reqwest::get("https://api.example.com/data")
        .await?
        .json::<serde_json::Value>()
        .await?;

    // POST request
    let client = reqwest::Client::new();
    let response = client
        .post("https://api.example.com/submit")
        .json(&serde_json::json!({"key": "value"}))
        .header("Authorization", "Bearer token")
        .send()
        .await?;

    // Session
    let client = reqwest::Client::builder()
        .user_agent("MyApp")
        .build()?;
    let response = client
        .get("https://api.example.com/profile")
        .send()
        .await?;

    Ok(())
}
```

### Async/Await (asyncio)

**Python**:
```python
import asyncio

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def process_urls(urls: list[str]) -> list[dict]:
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

async def main():
    urls = ["https://api.example.com/1", "https://api.example.com/2"]
    results = await process_urls(urls)
    print(results)

asyncio.run(main())
```

**Rust Output**:
```rust
use reqwest;
use serde_json::Value;

async fn fetch_data(url: &str) -> Result<Value, reqwest::Error> {
    reqwest::get(url).await?.json().await
}

async fn process_urls(urls: Vec<&str>) -> Vec<Result<Value, reqwest::Error>> {
    let tasks: Vec<_> = urls
        .into_iter()
        .map(|url| fetch_data(url))
        .collect();

    futures::future::join_all(tasks).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let urls = vec!["https://api.example.com/1", "https://api.example.com/2"];
    let results = process_urls(urls).await;
    println!("{:?}", results);
    Ok(())
}
```

### NumPy / ndarray

**Python**:
```python
import numpy as np

# Array creation
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))

# Operations
doubled = arr * 2
sum_all = np.sum(arr)
mean = np.mean(arr)

# Matrix operations
matrix = np.array([[1, 2], [3, 4]])
transposed = matrix.T
```

**Rust Output**:
```rust
use ndarray::{array, Array, Array2};

// Array creation
let arr = array![1, 2, 3, 4, 5];
let zeros = Array2::<f64>::zeros((3, 3));
let ones = Array2::<f64>::ones((2, 4));

// Operations
let doubled = &arr * 2;
let sum_all = arr.sum();
let mean = arr.mean().unwrap();

// Matrix operations
let matrix = array![[1, 2], [3, 4]];
let transposed = matrix.t();
```

### Pandas / Polars

**Python**:
```python
import pandas as pd

# Read CSV
df = pd.read_csv("data.csv")

# Operations
filtered = df[df['age'] > 18]
grouped = df.groupby('category')['value'].sum()
sorted_df = df.sort_values('date', ascending=False)

# Add column
df['total'] = df['quantity'] * df['price']
```

**Rust Output**:
```rust
use polars::prelude::*;

// Read CSV
let df = CsvReader::from_path("data.csv")?.finish()?;

// Operations
let filtered = df.filter(&df.column("age")?.gt(18)?)?;

let grouped = df
    .groupby(["category"])?
    .select(["value"])
    .sum()?;

let sorted_df = df.sort(["date"], vec![true])?;

// Add column
let df = df.lazy()
    .with_column(
        (col("quantity") * col("price")).alias("total")
    )
    .collect()?;
```

### Testing (pytest)

**Python**:
```python
import pytest

def test_addition():
    assert 1 + 1 == 2

def test_string_length():
    assert len("hello") == 5

@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15

@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input, expected):
    assert input ** 2 == expected
```

**Rust Output**:
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_addition() {
        assert_eq!(1 + 1, 2);
    }

    #[test]
    fn test_string_length() {
        assert_eq!("hello".len(), 5);
    }

    fn sample_data() -> Vec<i32> {
        vec![1, 2, 3, 4, 5]
    }

    #[test]
    fn test_sum() {
        let data = sample_data();
        assert_eq!(data.iter().sum::<i32>(), 15);
    }

    #[test]
    fn test_square_2() {
        assert_eq!(2_i32.pow(2), 4);
    }

    #[test]
    fn test_square_3() {
        assert_eq!(3_i32.pow(2), 9);
    }

    #[test]
    fn test_square_4() {
        assert_eq!(4_i32.pow(2), 16);
    }
}
```

---

## Advanced Features

### Decorators

**Python**:
```python
from functools import wraps
import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"

@property
def full_name(self):
    return f"{self.first_name} {self.last_name}"
```

**Rust Output**:
```rust
use std::time::Instant;

fn timer<F, R>(func: F) -> impl Fn() -> R
where
    F: Fn() -> R,
{
    move || {
        let start = Instant::now();
        let result = func();
        let duration = start.elapsed();
        println!("Function took {:.2?}", duration);
        result
    }
}

fn slow_function() -> &'static str {
    std::thread::sleep(std::time::Duration::from_secs(1));
    "done"
}

// Property
impl Person {
    fn full_name(&self) -> String {
        format!("{} {}", self.first_name, self.last_name)
    }
}
```

### Generators

**Python**:
```python
def fibonacci_gen(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def filter_positive(numbers):
    for num in numbers:
        if num > 0:
            yield num
```

**Rust Output**:
```rust
fn fibonacci_gen(n: usize) -> impl Iterator<Item = i32> {
    let mut a = 0;
    let mut b = 1;
    (0..n).map(move |_| {
        let current = a;
        let next = a + b;
        a = b;
        b = next;
        current
    })
}

fn filter_positive(numbers: Vec<i32>) -> impl Iterator<Item = i32> {
    numbers.into_iter().filter(|&num| num > 0)
}
```

### Context Managers

**Python**:
```python
with open("file.txt", "r") as f:
    content = f.read()

from contextlib import contextmanager

@contextmanager
def transaction():
    print("Begin transaction")
    yield
    print("Commit transaction")

with transaction():
    print("Do work")
```

**Rust Output**:
```rust
use std::fs::File;
use std::io::Read;

fn read_file() -> std::io::Result<String> {
    let mut f = File::open("file.txt")?;
    let mut content = String::new();
    f.read_to_string(&mut content)?;
    Ok(content)
}

// Transaction pattern
struct Transaction;

impl Transaction {
    fn new() -> Self {
        println!("Begin transaction");
        Transaction
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        println!("Commit transaction");
    }
}

fn do_work() {
    let _tx = Transaction::new();
    println!("Do work");
    // Automatically commits when _tx goes out of scope
}
```

### Error Handling

**Python**:
```python
def divide(a: int, b: int) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return 0.0
    except Exception as e:
        print(f"Error: {e}")
        raise

def read_config(filename: str) -> dict:
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
```

**Rust Output**:
```rust
fn divide(a: i32, b: i32) -> f64 {
    match b {
        0 => {
            println!("Cannot divide by zero");
            0.0
        }
        _ => a as f64 / b as f64,
    }
}

use std::fs::File;
use serde_json;

fn read_config(filename: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    match File::open(filename) {
        Ok(file) => {
            serde_json::from_reader(file)
                .map_err(|e| format!("Invalid JSON: {}", e).into())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Ok(serde_json::json!({}))
        }
        Err(e) => Err(e.into()),
    }
}
```

---

## WASM Deployment

### Basic WASM Bundle

```rust
use portalis_transpiler::{WasmBundler, BundleConfig, DeploymentTarget};

let mut config = BundleConfig::default();
config.package_name = "my_app".to_string();
config.target = DeploymentTarget::Web;

let bundler = WasmBundler::new(config);
bundler.generate_bundle("my_app");
```

### Deployment Targets

```rust
pub enum DeploymentTarget {
    Web,        // Browser with ES modules
    NodeJs,     // Node.js environment
    Bundler,    // Webpack/Rollup
    Deno,       // Deno runtime
    NoModules,  // Legacy browsers
}
```

**Example for each target**:

```rust
// Web (modern browsers)
let mut config = BundleConfig::production();
config.target = DeploymentTarget::Web;
bundler.generate_bundle("my_app");
// Output: dist/web/my_app.wasm + my_app.js

// Node.js
config.target = DeploymentTarget::NodeJs;
bundler.generate_bundle("my_app");
// Output: dist/nodejs/my_app.wasm + package.json

// CDN deployment
config.target = DeploymentTarget::Web;
config.optimize_size = true;
config.compression = CompressionFormat::Brotli;
bundler.generate_bundle("my_app");
// Output: dist/web/my_app.wasm.br (highly compressed)
```

### Optimization Levels

```rust
pub enum OptimizationLevel {
    None,       // -O0: No optimization (fastest build)
    Basic,      // -O1: Basic optimizations
    Standard,   // -O2: Standard optimizations
    Aggressive, // -O3: Aggressive optimizations
    Size,       // -Os: Optimize for size
    MaxSize,    // -Ozz: Maximum size reduction
}
```

**Usage**:

```rust
let mut config = BundleConfig::default();

// Development (fast builds)
config.optimization_level = OptimizationLevel::None;

// Production (balanced)
config.optimization_level = OptimizationLevel::Standard;

// CDN (smallest size)
config.optimization_level = OptimizationLevel::MaxSize;

let bundler = WasmBundler::new(config);
```

### Compression

```rust
pub enum CompressionFormat {
    None,   // No compression
    Gzip,   // Gzip compression (~70% reduction)
    Brotli, // Brotli compression (~80% reduction)
    Both,   // Generate both formats
}
```

**Example**:

```rust
let mut config = BundleConfig::production();
config.compression = CompressionFormat::Both;

let bundler = WasmBundler::new(config);
bundler.generate_bundle("my_app");

// Generates:
// - my_app.wasm         (original)
// - my_app.wasm.gz      (gzip)
// - my_app.wasm.br      (brotli)
```

---

## Optimization

### Dead Code Elimination

```rust
use portalis_transpiler::{DeadCodeEliminator, OptimizationStrategy};

let mut eliminator = DeadCodeEliminator::new();

// Conservative (only remove clearly unused private code)
let optimized = eliminator.analyze_with_strategy(
    rust_code,
    OptimizationStrategy::Conservative
);

// Moderate (remove private and pub(crate) unused code)
let optimized = eliminator.analyze_with_strategy(
    rust_code,
    OptimizationStrategy::Moderate
);

// Aggressive (remove ALL unreachable code)
let optimized = eliminator.analyze_with_strategy(
    rust_code,
    OptimizationStrategy::Aggressive
);
```

**Results**:
- Conservative: 30-40% code reduction
- Moderate: 50-60% code reduction
- Aggressive: 60-70% code reduction

### Call Graph Analysis

```rust
let eliminator = DeadCodeEliminator::new();
let call_graph = eliminator.build_call_graph(rust_code);

println!("Entry points: {:?}", call_graph.entry_points);
println!("Total functions: {}", call_graph.nodes.len());

let reachable = call_graph.compute_reachable();
println!("Reachable functions: {}", reachable.len());
```

### Tree-Shaking

Automatically removes unused dependencies:

```rust
use portalis_transpiler::DependencyResolver;

let mut resolver = DependencyResolver::new();
resolver.analyze_usage(rust_code);

// Only includes actually used dependencies in Cargo.toml
let used_deps = resolver.get_used_dependencies();
```

### Code Splitting

```rust
use portalis_transpiler::{CodeSplitter, SplitStrategy};

let splitter = CodeSplitter::new();

// Split by module
let chunks = splitter.split(rust_code, SplitStrategy::ByModule);

// Split by size
let chunks = splitter.split(rust_code, SplitStrategy::BySize(100_000)); // 100KB chunks

// Lazy loading
let chunks = splitter.split(rust_code, SplitStrategy::LazyLoad);
```

---

## Package Management

### Cargo.toml Generation

```rust
use portalis_transpiler::{CargoGenerator, CargoConfig};

let mut config = CargoConfig::default();
config.package_name = "my_project".to_string();
config.version = "1.0.0".to_string();
config.authors = vec!["Your Name <you@example.com>".to_string()];
config.edition = "2021".to_string();

// Add metadata
config.repository = Some("https://github.com/user/repo".to_string());
config.homepage = Some("https://example.com".to_string());
config.keywords = vec!["python".to_string(), "transpiler".to_string()];

// Add features
config.is_async = true;     // Adds tokio
config.http_client = true;  // Adds reqwest
config.wasm_target = true;  // Adds wasm-bindgen

let generator = CargoGenerator::new(config);
let cargo_toml = generator.generate();
```

**Generated Cargo.toml**:
```toml
[package]
name = "my_project"
version = "1.0.0"
authors = ["Your Name <you@example.com>"]
edition = "2021"
repository = "https://github.com/user/repo"
homepage = "https://example.com"
keywords = ["python", "transpiler"]

[dependencies]
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
wasm-bindgen = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[lib]
crate-type = ["cdylib", "rlib"]
```

### Dependency Resolution

```rust
use portalis_transpiler::{DependencyResolver, VersionConstraint};

let mut resolver = DependencyResolver::new();

// Add dependencies
resolver.add_dependency("tokio", VersionConstraint::Exact("1.35.0"));
resolver.add_dependency("serde", VersionConstraint::Caret("1.0"));
resolver.add_dependency("reqwest", VersionConstraint::Tilde("0.11"));

// Resolve versions
let resolved = resolver.resolve()?;
```

---

## Configuration

### BundleConfig Options

```rust
pub struct BundleConfig {
    pub package_name: String,
    pub output_dir: String,
    pub target: DeploymentTarget,
    pub optimization_level: OptimizationLevel,
    pub optimize_size: bool,
    pub code_splitting: bool,
    pub compression: CompressionFormat,
    pub generate_readme: bool,
    pub generate_package_json: bool,
}
```

**Preset Configurations**:

```rust
// Production (optimized for deployment)
let config = BundleConfig::production();
// - OptimizationLevel::Aggressive
// - optimize_size: true
// - compression: Brotli
// - code_splitting: true

// Development (fast iteration)
let config = BundleConfig::development();
// - OptimizationLevel::None
// - optimize_size: false
// - compression: None
// - code_splitting: false

// CDN (maximum optimization)
let config = BundleConfig::cdn_optimized();
// - OptimizationLevel::MaxSize
// - optimize_size: true
// - compression: Both
// - code_splitting: true
```

---

## Best Practices

### 1. Use Type Annotations

**Good**:
```python
def process_data(items: List[int]) -> int:
    return sum(items)
```

**Avoid**:
```python
def process_data(items):  # Type inference may be less accurate
    return sum(items)
```

### 2. Prefer Standard Library

Python standard library functions translate more reliably than third-party libraries.

**Good**:
```python
import json
data = json.loads(text)
```

**May Need Review**:
```python
import ujson  # Custom JSON library
data = ujson.loads(text)
```

### 3. Use Async/Await Consistently

**Good**:
```python
async def fetch_all(urls: List[str]):
    tasks = [fetch_url(url) for url in urls]
    return await asyncio.gather(*tasks)
```

**Avoid mixing**:
```python
def fetch_all(urls):  # Sync wrapper around async
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(fetch_async(urls))
```

### 4. Handle Errors Explicitly

**Good**:
```python
def read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""
```

**Less Optimal**:
```python
def read_file(path: str) -> str:
    with open(path) as f:  # May panic
        return f.read()
```

### 5. Avoid Dynamic Features

**Good**:
```python
def calculate(x: int) -> int:
    return x * 2
```

**Not Supported**:
```python
def calculate(x: int) -> int:
    return eval(f"{x} * 2")  # Dynamic code execution
```

---

## Limitations

### What's NOT Supported

1. **C Extensions**: Native libraries (OpenCV, TensorFlow)
2. **Dynamic Code**: `eval()`, `exec()`, `compile()`
3. **Metaclasses**: Complex metaprogramming
4. **GUI**: tkinter, PyQt, wxPython
5. **Some ML**: Training libraries (scikit-learn, PyTorch training)
6. **Runtime Imports**: `importlib` dynamic loading

### Workarounds

**For C Extensions**:
- Use Rust equivalents (e.g., `image` crate instead of PIL)
- Compile C code to WASM separately

**For Dynamic Code**:
- Refactor to static functions
- Generate code at build time, not runtime

**For GUI**:
- Use web-based UI (HTML/CSS/JS with WASM backend)

**For ML Training**:
- Train in Python, export model, load in Rust for inference

---

## Next Steps

- **[API Reference](./API_REFERENCE.md)** - Detailed API docs
- **[Examples](./EXAMPLES.md)** - More code examples
- **[Migration Guide](./MIGRATION_GUIDE.md)** - Python → Rust patterns
- **[Troubleshooting](./TROUBLESHOOTING.md)** - Common issues

For questions or issues, see [GitHub Issues](https://github.com/portalis/transpiler/issues).
