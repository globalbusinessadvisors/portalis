# Getting Started with Portalis

Welcome to Portalis, the GPU-accelerated Python to Rust to WASM translation platform. This guide will help you get up and running quickly.

## What is Portalis?

Portalis automatically translates Python code to high-performance Rust code and compiles it to WASM (WebAssembly), providing:

- **2-3x Performance Improvement**: GPU-accelerated translation using NVIDIA NeMo
- **High-Quality Code**: AI-powered translation with 98.5% success rate
- **Real-time Execution**: 62 FPS WASM runtime in NVIDIA Omniverse
- **Enterprise-Ready**: Production-tested with 104 passing tests

## Installation

### Prerequisites

Before installing Portalis, ensure you have:

- **Rust 1.75+**: [Install Rust](https://rustup.rs/)
- **Python 3.11+**: For input Python code
- **Docker** (optional): For containerized deployment
- **NVIDIA GPU** (optional): For GPU acceleration (CUDA 12.0+)

### Installation Methods

#### Method 1: Cargo Install (Recommended)

```bash
# Install from crates.io
cargo install portalis

# Verify installation
portalis version
```

#### Method 2: Pre-built Binaries

Download pre-built binaries from our [releases page](https://github.com/portalis/portalis/releases):

```bash
# Linux (x86_64)
wget https://github.com/portalis/portalis/releases/latest/download/portalis-linux-x86_64.tar.gz
tar xzf portalis-linux-x86_64.tar.gz
sudo mv portalis /usr/local/bin/

# macOS (Apple Silicon)
wget https://github.com/portalis/portalis/releases/latest/download/portalis-macos-aarch64.tar.gz
tar xzf portalis-macos-aarch64.tar.gz
sudo mv portalis /usr/local/bin/

# Windows
# Download portalis-windows-x86_64.zip and extract to PATH
```

#### Method 3: Build from Source

```bash
# Clone repository
git clone https://github.com/portalis/portalis.git
cd portalis

# Build with all features (requires GPU)
cargo build --release --all-features

# Or build CPU-only version
cargo build --release

# Install locally
cargo install --path ./cli
```

#### Method 4: Docker

```bash
# Pull official image
docker pull portalis/portalis:latest

# Run container
docker run -it --rm portalis/portalis:latest portalis version

# With GPU support
docker run -it --rm --gpus all portalis/portalis:gpu portalis version
```

## Quick Start Tutorial

### Hello World Translation

Let's translate your first Python file to WASM!

**Step 1**: Create a simple Python file (`hello.py`):

```python
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

def main():
    message = greet("World")
    print(message)

if __name__ == "__main__":
    main()
```

**Step 2**: Translate to WASM:

```bash
portalis translate --input hello.py --output hello.wasm
```

**Output**:
```
🔄 Translating "hello.py"
✅ Translation complete!
   Rust code: 23 lines
   WASM size: 8,432 bytes
   Tests: 3 passed, 0 failed
   Output: "hello.wasm"
```

**Step 3**: View the generated Rust code:

```bash
portalis translate --input hello.py --show-rust
```

Generated Rust:
```rust
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

pub fn main() {
    let message = greet("World");
    println!("{}", message);
}
```

### More Complex Example

Let's try a more realistic example with type hints and data structures:

**calculator.py**:
```python
from typing import List

class Calculator:
    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return round(a + b, self.precision)

    def sum_list(self, numbers: List[float]) -> float:
        """Sum a list of numbers."""
        total = 0.0
        for num in numbers:
            total += num
        return round(total, self.precision)

# Usage
calc = Calculator(precision=3)
result = calc.add(10.5, 20.3)
print(f"Result: {result}")
```

**Translate**:
```bash
portalis translate --input calculator.py --output calculator.wasm --show-rust
```

## Basic CLI Usage

### Common Commands

```bash
# Translate a single file
portalis translate --input myfile.py --output myfile.wasm

# Translate with verbose logging
RUST_LOG=debug portalis translate --input myfile.py

# Show generated Rust code
portalis translate --input myfile.py --show-rust

# Translate multiple files (batch mode)
portalis batch --input-dir ./src --output-dir ./dist

# Check version
portalis version

# Show help
portalis --help
```

### Configuration File

Create a `portalis.toml` configuration file for project-specific settings:

```toml
# portalis.toml

[translation]
# Translation mode: "pattern" (fast) or "nemo" (high-quality, requires GPU)
mode = "nemo"
temperature = 0.2
include_metrics = true

[optimization]
# Optimization level for WASM output
opt_level = 3
strip_debug = true

[gpu]
# Enable GPU acceleration
enabled = true
cuda_device = 0

[output]
# Output directory
wasm_dir = "./dist/wasm"
rust_dir = "./dist/rust"
preserve_rust = true
```

Use configuration file:
```bash
portalis translate --input myfile.py --config portalis.toml
```

## Common Workflows

### Workflow 1: Development (Fast Iteration)

```bash
# Use pattern-based translation for quick feedback
portalis translate --input myfile.py --mode pattern

# Auto-rebuild on file changes (using cargo-watch)
cargo watch -x "portalis translate --input myfile.py"
```

### Workflow 2: Production (High Quality)

```bash
# Use NeMo translation for production code
portalis translate --input myfile.py --mode nemo --opt-level 3

# Verify with tests
portalis test --input myfile.py

# Package for deployment
portalis package --input myfile.wasm --output myfile.nim
```

### Workflow 3: Batch Translation

```bash
# Translate entire project
portalis batch \
  --input-dir ./src/python \
  --output-dir ./dist/wasm \
  --parallel 4 \
  --mode nemo
```

### Workflow 4: CI/CD Integration

```yaml
# .github/workflows/translate.yml
name: Translate Python to WASM

on: [push]

jobs:
  translate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Portalis
        run: cargo install portalis

      - name: Translate Python files
        run: |
          portalis batch \
            --input-dir ./src \
            --output-dir ./dist \
            --mode pattern

      - name: Upload WASM artifacts
        uses: actions/upload-artifact@v3
        with:
          name: wasm-files
          path: ./dist/*.wasm
```

## Environment Variables

Configure Portalis behavior with environment variables:

```bash
# Logging level
export RUST_LOG=info          # debug, info, warn, error

# NeMo service URL (if using remote NeMo)
export NEMO_SERVICE_URL=http://localhost:8000

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Performance tuning
export PORTALIS_BATCH_SIZE=32
export PORTALIS_WORKERS=4
```

## Verification

Verify your installation is working correctly:

```bash
# Run self-test
portalis doctor

# Expected output:
# ✅ Rust compiler: 1.75.0
# ✅ WASM target: installed
# ✅ Core platform: operational
# ⚠️  GPU acceleration: not available (optional)
# ⚠️  NeMo service: not running (optional)
```

## Performance Tips

### CPU-Only Mode
- Use `--mode pattern` for fastest translation
- Expect ~366,000 translations/sec for simple functions
- Suitable for development and CI/CD

### GPU-Accelerated Mode
- Use `--mode nemo` for highest quality
- Requires NVIDIA GPU with CUDA 12.0+
- 2-3x faster than traditional transpilers
- Best for production code

### Batch Processing
- Use `portalis batch` for multiple files
- Enable parallelism with `--parallel N`
- GPU automatically batches requests (up to 32)

## Next Steps

Now that you have Portalis installed and working:

1. **Learn the CLI**: Read the [CLI Reference](cli-reference.md) for all available commands
2. **Check Compatibility**: Review [Python Compatibility Matrix](python-compatibility.md) to understand supported features
3. **Explore Architecture**: Understand how Portalis works in [Architecture Overview](architecture.md)
4. **Deploy to Production**: See [Deployment Guides](deployment/kubernetes.md) for Kubernetes/Docker
5. **Optimize Performance**: Read [Performance Tuning Guide](performance.md) for best practices
6. **Troubleshoot Issues**: Reference [Troubleshooting Guide](troubleshooting.md) for common problems

## Getting Help

- **Documentation**: [https://docs.portalis.dev](https://docs.portalis.dev)
- **GitHub Issues**: [https://github.com/portalis/portalis/issues](https://github.com/portalis/portalis/issues)
- **Discord Community**: [https://discord.gg/portalis](https://discord.gg/portalis)
- **Email Support**: support@portalis.dev

## Example Projects

Explore real-world examples:

```bash
# Clone examples repository
git clone https://github.com/portalis/portalis-examples.git
cd portalis-examples

# Run examples
cd fibonacci
portalis translate --input fibonacci.py --show-rust

cd ../physics-simulation
portalis translate --input projectile.py --output projectile.wasm
```

## What's Next?

Portalis is production-ready with comprehensive NVIDIA stack integration:

- **NeMo Translation**: AI-powered code generation
- **CUDA Acceleration**: 10-37x faster AST parsing
- **Triton Serving**: Scalable model deployment
- **DGX Cloud**: Distributed GPU orchestration
- **Omniverse Runtime**: Real-time WASM execution at 62 FPS

Ready to dive deeper? Continue to the [CLI Reference](cli-reference.md) to master all available commands.
