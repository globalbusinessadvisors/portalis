# Troubleshooting Guide

Common issues and solutions when using Portalis.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Translation Errors](#translation-errors)
- [GPU and CUDA Problems](#gpu-and-cuda-problems)
- [Build Failures](#build-failures)
- [WASM Runtime Issues](#wasm-runtime-issues)
- [Performance Problems](#performance-problems)
- [Service Connectivity](#service-connectivity)
- [Getting Help](#getting-help)

---

## Installation Issues

### Issue: `cargo install portalis` fails

**Symptoms**:
```
error: failed to compile `portalis v0.1.0`
```

**Solutions**:

1. **Update Rust toolchain**:
```bash
rustup update stable
rustc --version  # Should be 1.75+
```

2. **Install required dependencies**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential pkg-config libssl-dev

# macOS
brew install openssl pkg-config

# Windows (use Visual Studio Build Tools)
# Download from: https://visualstudio.microsoft.com/downloads/
```

3. **Check WASM target**:
```bash
rustup target add wasm32-wasi
rustup target list | grep wasm32-wasi
```

### Issue: GPU features not available

**Symptoms**:
```
⚠️  GPU acceleration: not available
⚠️  CUDA: not detected
```

**This is normal if**:
- You don't have an NVIDIA GPU
- CUDA is not installed
- Running in CPU-only mode

**Solutions**:

**Option 1**: Use CPU-only mode (no action needed)
- Pattern-based translation works without GPU
- Suitable for development and testing

**Option 2**: Install CUDA for GPU acceleration
```bash
# Check GPU
nvidia-smi

# Install CUDA 12.0+ from NVIDIA website
# Then rebuild with GPU features
cargo install portalis --features gpu
```

---

## Translation Errors

### Issue: Translation failed with "Unsupported Python feature"

**Symptoms**:
```
Error: Translation failed
Reason: Unsupported feature: **kwargs at line 42
```

**Solution**:
Refactor code to use supported patterns. See [Python Compatibility Matrix](python-compatibility.md).

**Example - Keyword Arguments**:
```python
# Unsupported
def configure(**kwargs):
    pass

# Workaround - use dataclass
from dataclasses import dataclass

@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080

def configure(config: Config):
    pass
```

### Issue: Type inference failures

**Symptoms**:
```
Error: Cannot infer type for variable 'x' at line 15
```

**Solution**:
Add explicit type hints:

```python
# Before - may fail
def process(data):
    result = transform(data)
    return result

# After - works
def process(data: List[int]) -> List[int]:
    result: List[int] = transform(data)
    return result
```

### Issue: Import errors

**Symptoms**:
```
Error: Cannot translate module 'custom_module'
```

**Solution**:

1. **Check compatibility**: Ensure module is supported
2. **Use explicit imports**: Avoid wildcard imports
```python
# Avoid
from module import *

# Prefer
from module import specific_function, SpecificClass
```

3. **Inline stdlib usage**: Some stdlib modules have Rust equivalents
```python
# Instead of importing datetime
import datetime

# Use type hints that translate
from datetime import datetime as DateTime
```

### Issue: NeMo translation timeout

**Symptoms**:
```
Error: NeMo service timeout after 30s
```

**Solutions**:

1. **Increase timeout**:
```bash
portalis translate --input large.py --timeout 60
```

2. **Break into smaller files**:
```bash
# Split large file into modules
portalis batch --input-dir ./modules --parallel 4
```

3. **Use pattern mode for large files**:
```bash
portalis translate --input large.py --mode pattern
```

---

## GPU and CUDA Problems

### Issue: CUDA out of memory

**Symptoms**:
```
CUDA Error: out of memory (code: 2)
GPU memory: 7854MB / 8192MB
```

**Solutions**:

1. **Reduce batch size**:
```toml
# portalis.toml
[gpu]
batch_size = 16  # Reduce from 32
```

2. **Clear GPU cache**:
```bash
# Restart translation service
docker-compose restart nemo-service
```

3. **Limit GPU memory**:
```toml
[gpu]
memory_limit_mb = 6144  # Leave headroom
```

### Issue: CUDA version mismatch

**Symptoms**:
```
Error: CUDA 11.8 detected, but 12.0+ required
```

**Solution**:
Update CUDA toolkit:
```bash
# Check current version
nvidia-smi

# Download CUDA 12.0+ from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
```

### Issue: GPU not detected

**Symptoms**:
```bash
portalis doctor

⚠️  CUDA: not detected
⚠️  GPU: not available
```

**Solutions**:

1. **Check NVIDIA driver**:
```bash
nvidia-smi
# Should show GPU info

# If not, install/update driver
sudo ubuntu-drivers autoinstall
```

2. **Check CUDA installation**:
```bash
nvcc --version
# Should show CUDA 12.0+
```

3. **Set CUDA device**:
```bash
export CUDA_VISIBLE_DEVICES=0
portalis translate --input test.py
```

---

## Build Failures

### Issue: WASM compilation failed

**Symptoms**:
```
Error: wasm-ld: error: unknown argument: --shared-memory
```

**Solution**:

1. **Update WASM target**:
```bash
rustup target add wasm32-wasi --force
```

2. **Check Rust version**:
```bash
rustc --version  # Should be 1.75+
rustup update stable
```

### Issue: "Cannot find wasm-ld"

**Symptoms**:
```
Error: linker `rust-lld` not found
```

**Solution**:

```bash
# Reinstall Rust toolchain
rustup toolchain install stable
rustup default stable

# Verify
which rust-lld
```

### Issue: Link-time optimization (LTO) errors

**Symptoms**:
```
Error: LTO failed during linking
```

**Solution**:

Disable LTO temporarily:
```bash
portalis translate --input test.py --no-lto

# Or in config
# portalis.toml
[optimization]
lto = false
```

---

## WASM Runtime Issues

### Issue: WASM module instantiation failed

**Symptoms**:
```
Error: WebAssembly.instantiate(): Import #0 module="env" error
```

**Solution**:

1. **Check WASI compatibility**:
```bash
# Use wasmtime to test
wasmtime run output.wasm
```

2. **Verify module exports**:
```bash
wasm-objdump -x output.wasm | grep export
```

### Issue: Stack overflow in WASM

**Symptoms**:
```
RuntimeError: call stack exhausted
```

**Solution**:

Increase stack size:
```bash
# Compile with larger stack
portalis translate --input recursive.py --stack-size 1048576
```

Or refactor to avoid deep recursion:
```python
# Before - recursive
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# After - iterative
def factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

---

## Performance Problems

### Issue: Slow translation performance

**Symptoms**:
```
⏱️  Translation took 45.3s (expected ~5s)
```

**Solutions**:

1. **Enable GPU acceleration**:
```bash
# Check if GPU is available
portalis doctor

# Use NeMo mode
portalis translate --input slow.py --mode nemo
```

2. **Use batch mode**:
```bash
# Translate multiple files in parallel
portalis batch --input-dir ./src --parallel 8
```

3. **Reduce optimization level**:
```bash
# For development
portalis translate --input dev.py -O 0
```

### Issue: High memory usage

**Symptoms**:
```
System memory: 14.2GB / 16GB
Warning: High memory pressure
```

**Solutions**:

1. **Reduce parallelism**:
```bash
portalis batch --input-dir ./src --parallel 2
```

2. **Process files sequentially**:
```bash
for file in src/*.py; do
    portalis translate --input "$file"
done
```

3. **Enable streaming mode**:
```toml
[translation]
streaming = true
```

### Issue: Low GPU utilization

**Symptoms**:
```
GPU utilization: 15% (expected 70%+)
```

**Solutions**:

1. **Increase batch size**:
```toml
[gpu]
batch_size = 64  # Increase from 32
```

2. **Use batch translation**:
```bash
portalis batch --input-dir ./src --mode nemo
```

3. **Check GPU isn't throttling**:
```bash
nvidia-smi --query-gpu=temperature.gpu --format=csv
# Temperature should be <85°C
```

---

## Service Connectivity

### Issue: Cannot connect to NeMo service

**Symptoms**:
```
Error: Failed to connect to NeMo service at http://localhost:8000
Connection refused
```

**Solutions**:

1. **Start NeMo service**:
```bash
# Using Docker Compose
docker-compose up -d nemo-service

# Check status
docker-compose ps
```

2. **Check service URL**:
```bash
# Test connectivity
curl http://localhost:8000/health

# Set correct URL
export NEMO_SERVICE_URL=http://localhost:8000
```

3. **Check firewall**:
```bash
# Allow port 8000
sudo ufw allow 8000/tcp
```

### Issue: NeMo service crashes

**Symptoms**:
```
Error: NeMo service returned 500 Internal Server Error
```

**Solutions**:

1. **Check service logs**:
```bash
docker-compose logs nemo-service
```

2. **Restart service**:
```bash
docker-compose restart nemo-service
```

3. **Check GPU memory**:
```bash
nvidia-smi
# Free up GPU memory if needed
```

### Issue: Triton Inference Server timeout

**Symptoms**:
```
Error: Triton request timeout after 30s
```

**Solutions**:

1. **Increase timeout**:
```bash
export TRITON_TIMEOUT=60
```

2. **Check Triton health**:
```bash
curl http://localhost:8001/v2/health/ready
```

3. **Scale Triton instances**:
```bash
kubectl scale deployment triton --replicas=3
```

---

## Debugging Techniques

### Enable Verbose Logging

```bash
# Maximum verbosity
RUST_LOG=trace portalis translate --input debug.py

# Component-specific logging
RUST_LOG=portalis_transpiler=debug portalis translate --input debug.py
```

### Get Stack Traces

```bash
# Full backtrace on errors
RUST_BACKTRACE=full portalis translate --input error.py

# Colored backtrace
RUST_BACKTRACE=1 COLORBT_SHOW_HIDDEN=1 portalis translate --input error.py
```

### Profile Performance

```bash
# Built-in profiler
portalis translate --input slow.py --profile

# Output:
# Profiling Results:
#   Ingest:     145ms  (12%)
#   Analysis:   234ms  (19%)
#   Translation: 723ms (61%)
#   Build:      98ms   (8%)
```

### Inspect Intermediate Output

```bash
# Save Rust code
portalis translate --input test.py --save-rust test.rs

# Keep build artifacts
portalis translate --input test.py --preserve-artifacts

# Show all intermediate steps
portalis translate --input test.py --verbose --show-rust
```

---

## Common Error Messages

### "feature `nemo` not enabled"

**Cause**: Trying to use NeMo without GPU features.

**Solution**:
```bash
# Rebuild with GPU support
cargo install portalis --features gpu

# Or use pattern mode
portalis translate --input test.py --mode pattern
```

### "WASM validation failed"

**Cause**: Generated WASM is invalid.

**Solution**:
```bash
# Validate WASM
wasm-validate output.wasm

# Regenerate with validation
portalis translate --input test.py --validate-wasm
```

### "Type annotation required"

**Cause**: Missing type hints.

**Solution**: Add type annotations to all functions and variables:
```python
def process(items: List[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in items:
        result[item] = len(item)
    return result
```

---

## Known Issues and Workarounds

### Issue: Large file translation is slow

**Status**: Known limitation

**Workaround**: Split into smaller modules
```python
# Instead of one large file (10,000+ lines)
# Split into multiple modules

# calculations.py (500 lines)
# data_structures.py (300 lines)
# utilities.py (200 lines)
```

### Issue: Some NumPy operations unsupported

**Status**: Partial support

**Workaround**: Use supported operations or Rust equivalents
```python
# Instead of NumPy
import numpy as np
arr = np.array([1, 2, 3])

# Use lists
from typing import List
arr: List[int] = [1, 2, 3]
```

---

## Collecting Diagnostic Information

When reporting issues, include:

```bash
# System diagnostics
portalis doctor --verbose > diagnostics.txt

# Version information
portalis version --json > version.json

# Translation attempt with logs
RUST_LOG=debug portalis translate --input problematic.py 2>&1 | tee error.log

# Environment
env | grep -E '(RUST|CUDA|PORTALIS)' > env.txt
```

---

## Getting Help

### Documentation
- [Getting Started](getting-started.md)
- [CLI Reference](cli-reference.md)
- [Python Compatibility](python-compatibility.md)
- [Performance Guide](performance.md)

### Community Support
- **GitHub Issues**: [github.com/portalis/portalis/issues](https://github.com/portalis/portalis/issues)
- **Discord**: [discord.gg/portalis](https://discord.gg/portalis)
- **Stack Overflow**: Tag with `portalis`

### Enterprise Support
- **Email**: support@portalis.dev
- **SLA**: 24-hour response for enterprise customers
- **Priority Support**: Available with Enterprise license

### Reporting Bugs

Include in bug reports:
1. Portalis version (`portalis version`)
2. Operating system and version
3. Rust version (`rustc --version`)
4. GPU information (if applicable - `nvidia-smi`)
5. Minimal reproduction case
6. Expected vs. actual behavior
7. Diagnostic information (see above)

**Bug report template**:
```markdown
### Environment
- Portalis version: 0.1.0
- OS: Ubuntu 22.04
- Rust: 1.75.0
- GPU: NVIDIA RTX 4090 (if applicable)

### Description
Brief description of the issue

### Reproduction
Steps to reproduce:
1. Create file `test.py` with...
2. Run `portalis translate --input test.py`
3. Error occurs

### Expected Behavior
What should happen

### Actual Behavior
What actually happens (include error messages)

### Diagnostics
Attach diagnostics.txt, error.log, version.json
```

---

## FAQ

**Q: Can I translate Python 2 code?**
A: No, only Python 3.9+ is supported. Use tools like `2to3` to upgrade.

**Q: Does Portalis support Jupyter notebooks?**
A: Not directly. Extract Python code from notebooks first.

**Q: Can I translate packages with dependencies?**
A: Yes, but dependencies must also be translated. Use `portalis batch` for projects.

**Q: Is multithreading supported?**
A: Python threading translates to Tokio async tasks. For true parallelism, use async/await.

**Q: Can I use Portalis in CI/CD?**
A: Yes! See [Getting Started](getting-started.md#workflow-4-cicd-integration) for examples.

---

**Still having issues?** Join our [Discord community](https://discord.gg/portalis) for real-time help!
