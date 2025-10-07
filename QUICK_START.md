# Portalis Quick Start

Get started with Portalis in under 5 minutes!

## Installation

```bash
# Install from crates.io (once published)
cargo install portalis

# Verify installation
portalis --version
```

## Your First Conversion

**Option 1: Convert current directory** (easiest)

```bash
# Navigate to your Python project
cd my-python-project/

# Convert everything to WASM
portalis convert

# That's it! Portalis auto-detects and converts all .py files
```

**Option 2: Convert a specific file**

```python
# Create calculator.py
def add(a: int, b: int) -> int:
    return a + b
```

```bash
portalis convert calculator.py
```

**Option 3: Convert a Python package**

```bash
# If your directory has __init__.py or setup.py
portalis convert ./mylib/

# Creates a Rust crate with WASM output
```

That's it! You'll see:

```
Portalis - Python to Rust/WASM Converter

Converting: calculator.py

├─ Reading Python code... ✓
├─ Translating to Rust... ✓
├─ Compiling to WASM... ✓ (./dist/calculator.wasm)
└─ Running tests... ✓

✅ Conversion complete!

Next steps:
  Run with Node.js:
    node -e "require('./dist/calculator.wasm')"
```

## Common Workflows

### 1. Convert Current Directory (Most Common)

```bash
cd my-python-project/
portalis convert              # Defaults to current directory
# or explicitly:
portalis convert .
```

### 2. Convert a Single File

```bash
portalis convert script.py

# Save Rust code too
portalis convert script.py --format both

# Custom output location
portalis convert script.py -o ./build
```

### 3. Convert a Python Library/Package

If your directory has `__init__.py`, `setup.py`, or `pyproject.toml`:

```bash
portalis convert ./mylib/
```

Output: Complete Rust crate with WASM

### 4. Convert an Entire Project

```bash
# Directory with multiple .py files
portalis convert ./src/

# Fast mode (skip tests)
portalis convert ./src/ --fast
```

### 5. Analyze Before Converting

```bash
portalis convert complex_app.py --analyze
```

Shows:
- Supported features
- Compatibility score
- Estimated output size

## Input Types (Auto-Detected)

Portalis automatically detects what you're converting:

| You Type | Portalis Detects | What Happens |
|----------|------------------|--------------|
| `portalis convert` | Current directory | Converts all `.py` files |
| `portalis convert .` | Current directory | Same as above |
| `portalis convert script.py` | Single Python file | Creates `script.wasm` |
| `portalis convert mylib/` (has `__init__.py`) | Python package | Creates Rust crate + WASM |
| `portalis convert src/` | Directory with `.py` files | Converts each file to WASM |

**No configuration needed** - Portalis figures it out!

## Output Formats

```bash
# WASM only (default)
portalis convert app.py

# Rust only
portalis convert app.py --format rust

# Both Rust and WASM
portalis convert app.py --format both
```

## Examples

Try the included examples:

```bash
# Clone the repo to get examples
git clone https://github.com/portalis/portalis
cd portalis/examples

# Convert the simple example
portalis convert add.py

# Convert fibonacci
portalis convert fibonacci.py

# Convert with analysis
portalis convert test_classes.py --analyze
```

## Next Steps

- Read the [full documentation](https://portalis.dev/docs)
- Check out [examples](https://github.com/portalis/portalis/tree/main/examples)
- Join our [community](https://github.com/portalis/portalis/discussions)

## Troubleshooting

**"Command not found"**
```bash
# Make sure cargo bin is in PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

**"File not found"**
```bash
# List available Python files
ls *.py

# Portalis will suggest similar files
portalis convert typo.py
```

**Get Help**
```bash
portalis --help
portalis convert --help
```

## Publishing Your Package (Maintainers Only)

If you're a Portalis maintainer with credentials:

```bash
# Dry run
./publish.sh --dry-run

# Publish to PyPI and crates.io
./publish.sh
```

---

**Portalis** - Making Python fast, one conversion at a time 🚀
