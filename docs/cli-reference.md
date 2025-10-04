# CLI Reference

Complete command-line reference for Portalis.

## Installation

See [Getting Started](getting-started.md#installation) for installation instructions.

## Global Options

These options are available for all commands:

```bash
portalis [OPTIONS] <COMMAND>
```

| Option | Description | Default |
|--------|-------------|---------|
| `--help`, `-h` | Show help information | - |
| `--version`, `-V` | Show version information | - |
| `--config <FILE>` | Path to configuration file | `./portalis.toml` |
| `--verbose`, `-v` | Increase logging verbosity | `info` |
| `--quiet`, `-q` | Suppress non-error output | - |
| `--color <WHEN>` | Color output: always, auto, never | `auto` |

### Environment Variables

```bash
# Logging configuration
RUST_LOG=debug              # Log level: trace, debug, info, warn, error
RUST_BACKTRACE=1            # Enable backtrace on errors

# Service endpoints
NEMO_SERVICE_URL=http://localhost:8000      # NeMo translation service
TRITON_URL=http://localhost:8001            # Triton inference server
DGX_CLOUD_URL=https://api.dgx-cloud.nvidia.com

# GPU configuration
CUDA_VISIBLE_DEVICES=0      # CUDA device selection
CUDA_LAUNCH_BLOCKING=1      # Synchronous CUDA operations (debugging)

# Performance tuning
PORTALIS_BATCH_SIZE=32      # Batch size for GPU operations
PORTALIS_WORKERS=4          # Parallel workers for batch translation
PORTALIS_CACHE_DIR=~/.cache/portalis    # Cache directory
```

## Commands

### `translate` - Translate Python to WASM

Translate a single Python file to WASM.

```bash
portalis translate [OPTIONS] --input <FILE>
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input <FILE>` | `-i` | Input Python file (required) | - |
| `--output <FILE>` | `-o` | Output WASM file | `<input>.wasm` |
| `--mode <MODE>` | `-m` | Translation mode: pattern, nemo | `pattern` |
| `--show-rust` | | Display generated Rust code | `false` |
| `--save-rust <FILE>` | | Save Rust code to file | - |
| `--temperature <FLOAT>` | | NeMo sampling temperature (0.0-1.0) | `0.2` |
| `--opt-level <N>` | `-O` | Optimization level: 0, 1, 2, 3, s, z | `3` |
| `--strip-debug` | | Strip debug symbols | `false` |
| `--run-tests` | | Run conformance tests | `true` |
| `--no-tests` | | Skip conformance tests | - |

#### Examples

```bash
# Basic translation
portalis translate --input hello.py

# High-quality translation with NeMo
portalis translate --input app.py --mode nemo --output app.wasm

# Save intermediate Rust code
portalis translate --input lib.py --save-rust lib.rs --show-rust

# Maximum optimization
portalis translate --input compute.py -O z --strip-debug

# Fast development mode (skip tests)
portalis translate --input draft.py --no-tests --mode pattern
```

#### Translation Modes

**Pattern-Based Mode** (`--mode pattern`):
- Fast CPU-based translation
- ~366,000 translations/second
- Best for simple functions and development
- No GPU required
- Limited to well-defined patterns

**NeMo Mode** (`--mode nemo`):
- AI-powered translation with NVIDIA NeMo
- 2-3x overall speedup
- Handles complex Python features
- Requires GPU and NeMo service
- 98.5% translation success rate

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Translation failed |
| 2 | Invalid input file |
| 3 | Tests failed |
| 4 | Service unavailable (NeMo) |

---

### `batch` - Batch Translation

Translate multiple Python files in batch.

```bash
portalis batch [OPTIONS] --input-dir <DIR>
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input-dir <DIR>` | `-i` | Input directory with Python files | - |
| `--output-dir <DIR>` | `-o` | Output directory for WASM files | `./dist` |
| `--pattern <GLOB>` | `-p` | File pattern to match | `**/*.py` |
| `--parallel <N>` | `-j` | Number of parallel workers | CPU cores |
| `--mode <MODE>` | `-m` | Translation mode | `pattern` |
| `--recursive` | `-r` | Recursively search subdirectories | `true` |
| `--preserve-structure` | | Maintain directory structure | `true` |
| `--fail-fast` | | Stop on first error | `false` |
| `--continue-on-error` | | Continue after errors | `true` |

#### Examples

```bash
# Translate all Python files in a directory
portalis batch --input-dir ./src --output-dir ./dist

# Parallel translation with 8 workers
portalis batch -i ./src -o ./dist --parallel 8

# Translate specific files
portalis batch -i ./src -p "**/*_lib.py" --mode nemo

# Fast CI/CD translation
portalis batch -i ./src -o ./build --mode pattern --fail-fast
```

---

### `test` - Run Conformance Tests

Verify Python-to-Rust translation correctness.

```bash
portalis test [OPTIONS] --input <FILE>
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input <FILE>` | `-i` | Python file to test | - |
| `--wasm <FILE>` | `-w` | WASM file to test | `<input>.wasm` |
| `--test-cases <FILE>` | `-t` | Custom test cases (JSON) | Auto-generated |
| `--coverage` | | Generate coverage report | `false` |
| `--benchmark` | | Run performance benchmarks | `false` |

#### Examples

```bash
# Test translated WASM
portalis test --input calculator.py --wasm calculator.wasm

# Generate coverage report
portalis test --input app.py --coverage

# Benchmark performance
portalis test --input compute.py --benchmark
```

---

### `package` - Package for Deployment

Package WASM for deployment (NIM containers, Triton, etc.).

```bash
portalis package [OPTIONS] --input <FILE>
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input <FILE>` | `-i` | Input WASM file | - |
| `--output <FILE>` | `-o` | Output package file | - |
| `--format <FORMAT>` | `-f` | Package format: nim, docker, helm | `nim` |
| `--registry <URL>` | | Container registry URL | - |
| `--tag <TAG>` | `-t` | Image tag | `latest` |
| `--gpu` | | Include GPU support | `false` |
| `--triton` | | Package for Triton Inference Server | `false` |

#### Package Formats

**NIM (NVIDIA Inference Microservice)**:
```bash
portalis package --input model.wasm --format nim --output model.nim --gpu
```

**Docker Container**:
```bash
portalis package -i app.wasm -f docker -o app:v1.0 --registry docker.io/myorg
```

**Helm Chart**:
```bash
portalis package -i service.wasm -f helm -o mychart/ --gpu
```

---

### `serve` - Run Translation Service

Start HTTP server for translation API.

```bash
portalis serve [OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host <IP>` | `-H` | Bind address | `0.0.0.0` |
| `--port <PORT>` | `-p` | Listen port | `8080` |
| `--workers <N>` | `-w` | Worker processes | CPU cores |
| `--mode <MODE>` | `-m` | Default translation mode | `pattern` |
| `--gpu` | | Enable GPU acceleration | `false` |
| `--max-requests <N>` | | Max concurrent requests | `100` |
| `--timeout <SECONDS>` | | Request timeout | `30` |

#### Examples

```bash
# Start basic server
portalis serve --port 8080

# Production server with GPU
portalis serve --host 0.0.0.0 --port 80 --workers 4 --gpu --mode nemo

# Development server
portalis serve --port 3000 --mode pattern
```

#### API Endpoints

**POST /api/v1/translate**:
```bash
curl -X POST http://localhost:8080/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "python_code": "def add(a: int, b: int) -> int:\n    return a + b",
    "mode": "nemo",
    "temperature": 0.2
  }'
```

Response:
```json
{
  "rust_code": "pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}",
  "wasm_bytes": "<base64-encoded>",
  "confidence": 0.98,
  "metrics": {
    "total_time_ms": 145.2,
    "gpu_utilization": 0.85
  }
}
```

---

### `doctor` - System Diagnostics

Check installation and dependencies.

```bash
portalis doctor [OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--verbose` | Show detailed diagnostics | `false` |
| `--fix` | Attempt to fix issues | `false` |

#### Example Output

```
Portalis System Diagnostics
============================

✅ Rust compiler: 1.75.0
✅ Cargo: 1.75.0
✅ WASM target (wasm32-wasi): installed
✅ Core platform: operational

GPU Acceleration:
⚠️  CUDA: not detected (optional)
⚠️  NeMo service: not running (optional)
   Start with: docker-compose up nemo-service

Deployment:
✅ Docker: 24.0.7
✅ Kubernetes: kubectl 1.28.0
⚠️  Helm: not installed (optional)

Status: Ready for CPU-based translation
Recommendation: Install GPU for 2-3x performance improvement
```

---

### `version` - Version Information

Show version and build information.

```bash
portalis version [OPTIONS]
```

#### Options

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--check-updates` | Check for newer versions |

#### Example Output

```
Portalis v0.1.0
Python → Rust → WASM Translation Platform

Build Information:
  Commit: 453a9d02
  Date: 2025-10-03
  Features: nemo, cuda, triton

Components:
  Core: 6,550 LOC
  NVIDIA Stack: 35,000+ LOC
  Tests: 104 passing

NVIDIA Integration:
  ✅ NeMo: GPU translation service
  ✅ CUDA: AST parsing acceleration
  ✅ Triton: Model serving
  ✅ NIM: Microservice packaging
  ✅ DGX Cloud: Distributed orchestration
  ✅ Omniverse: WASM runtime (62 FPS)

Documentation: https://docs.portalis.dev
```

---

## Configuration File

### portalis.toml

Full configuration file example:

```toml
# Portalis Configuration File

[translation]
# Default translation mode
mode = "nemo"                    # "pattern" or "nemo"
temperature = 0.2                # 0.0 (deterministic) to 1.0 (creative)
include_metrics = true           # Collect performance metrics
run_tests = true                 # Run conformance tests

[optimization]
# WASM optimization settings
opt_level = 3                    # 0, 1, 2, 3, 's' (size), 'z' (aggressive size)
strip_debug = true               # Remove debug symbols
lto = true                       # Link-time optimization

[gpu]
# GPU acceleration settings
enabled = true                   # Enable GPU features
cuda_device = 0                  # CUDA device ID (0-7)
batch_size = 32                  # GPU batch size
memory_limit_mb = 8192           # GPU memory limit

[services]
# External service URLs
nemo_url = "http://localhost:8000"
triton_url = "http://localhost:8001"
dgx_cloud_url = "https://api.dgx-cloud.nvidia.com"

[services.nemo]
# NeMo-specific configuration
timeout_seconds = 30
retry_attempts = 3
retry_delay_ms = 1000

[output]
# Output settings
wasm_dir = "./dist/wasm"
rust_dir = "./dist/rust"
preserve_rust = false            # Keep intermediate Rust files
preserve_artifacts = true        # Keep build artifacts

[logging]
# Logging configuration
level = "info"                   # trace, debug, info, warn, error
format = "pretty"                # pretty, json, compact
output = "stdout"                # stdout, stderr, file
file = "./portalis.log"          # Log file (if output=file)

[cache]
# Caching settings
enabled = true
directory = "~/.cache/portalis"
max_size_mb = 1024
ttl_seconds = 86400              # 24 hours

[performance]
# Performance tuning
parallel_workers = 4             # Number of parallel workers
max_concurrent_requests = 100
request_timeout_seconds = 30
```

---

## Examples by Use Case

### Development

```bash
# Fast iteration
portalis translate -i app.py --mode pattern --no-tests

# Auto-rebuild on changes (with cargo-watch)
cargo watch -x "portalis translate -i app.py"
```

### Testing

```bash
# Comprehensive testing
portalis test -i app.py --coverage --benchmark

# CI/CD pipeline
portalis batch -i ./src -o ./dist --fail-fast
```

### Production

```bash
# High-quality translation
portalis translate -i service.py --mode nemo -O z --strip-debug

# Package for Kubernetes
portalis package -i service.wasm -f helm --gpu
```

### Debugging

```bash
# Verbose logging
RUST_LOG=debug portalis translate -i debug.py --show-rust

# With backtrace
RUST_BACKTRACE=1 portalis translate -i error.py
```

---

## Shell Completion

Generate shell completion scripts:

```bash
# Bash
portalis completion bash > /etc/bash_completion.d/portalis

# Zsh
portalis completion zsh > /usr/local/share/zsh/site-functions/_portalis

# Fish
portalis completion fish > ~/.config/fish/completions/portalis.fish

# PowerShell
portalis completion powershell > $PROFILE
```

---

## See Also

- [Getting Started Guide](getting-started.md)
- [Python Compatibility Matrix](python-compatibility.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Performance Tuning](performance.md)
- [API Reference](api-reference.md)
