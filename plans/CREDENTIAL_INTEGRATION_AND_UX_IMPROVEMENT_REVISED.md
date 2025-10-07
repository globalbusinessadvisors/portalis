# Portalis Credential Integration & UX Improvement Plan (REVISED)

**Status:** ✅ IMPLEMENTED - See IMPLEMENTATION_SUMMARY.md
**Priority:** High
**Target:** Enable platform publishing + Simplify end-user CLI workflow
**Created:** 2025-10-06
**Revised:** 2025-10-06
**Completed:** 2025-10-06

---

## ⚠️ NOTE: This was the original plan. See IMPLEMENTATION_SUMMARY.md for actual implementation.

**Key Changes from Plan:**
1. ✅ `convert` command **defaults to current directory** (zero-friction)
2. ✅ Full **Python package conversion** implemented (creates Rust crate)
3. ✅ **WASM build** using wasm-pack with cargo fallback
4. ✅ Complete auto-detection for files/packages/directories
5. ✅ Real-world examples in USE_CASES.md

---

---

## Critical Clarifications

### ❌ INCORRECT ASSUMPTIONS IN FIRST PLAN:
1. **Credentials are NOT for end-users** - They are for YOU (platform maintainer) to publish the Portalis platform itself
2. **Platform NOT YET PUBLISHED** - Portalis is not on crates.io or PyPI yet
3. **End-user workflow unclear** - How do users specify which Python library/script to convert?
4. **WASM deployment status unclear** - Is the transpiler itself compiled to WASM?

### ✅ CORRECTED UNDERSTANDING:
1. **Your credentials** = Publish the Portalis PLATFORM to crates.io and PyPI
2. **End-users** = Developers who will `cargo install portalis` or `pip install portalis` AFTER you publish
3. **End-user workflow** = They need a simple way to point Portalis at their Python code
4. **Two separate concerns:**
   - **A) Publishing Portalis** (your credentials, one-time setup)
   - **B) Using Portalis** (end-user experience, ongoing)

---

## Part A: Publishing Portalis Platform (Your Credentials)

### A.1 Current State

**NOT YET PUBLISHED:**
- ❌ `portalis` not on crates.io (cargo search returns nothing)
- ❌ `portalis-nemo-integration` not on PyPI
- ❌ `portalis-dgx-cloud` not on PyPI
- ✅ Code is ready (21,000+ LOC, 104 tests passing)
- ✅ You have credentials in `.env`

**YOUR .env credentials are for:**
```env
TWINE_USERNAME=__token__              # To publish Python packages to PyPI
TWINE_PASSWORD=pypi-xxx               # PyPI API token
CARGO_REGISTRY_TOKEN=crates-io-xxx    # To publish Rust crates to crates.io
```

### A.2 What Gets Published

#### To crates.io (Rust):
```
portalis                    # Main CLI tool (what users will `cargo install`)
portalis-core               # Core library
portalis-transpiler         # The actual transpiler (main value)
portalis-orchestration      # Pipeline orchestration
(+ other agent crates as needed)
```

#### To PyPI (Python):
```
portalis-nemo-integration   # NeMo-based translation (optional GPU feature)
portalis-dgx-cloud          # DGX Cloud integration (optional)
```

### A.3 Publishing Workflow (Platform Maintainer - YOU)

**Simple Publishing Commands:**

Create a root-level `Makefile` or `publish.sh` script:

```bash
#!/bin/bash
# publish.sh - Publish Portalis platform to registries

set -e

echo "🚀 Publishing Portalis Platform"
echo "================================"

# Load credentials from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "❌ .env file not found. Create it with TWINE_* and CARGO_REGISTRY_TOKEN"
    exit 1
fi

# 1. Publish Python packages to PyPI
echo "📦 Publishing Python packages to PyPI..."

cd nemo-integration
python -m build
twine upload dist/* --skip-existing
cd ..

cd dgx-cloud
python -m build
twine upload dist/* --skip-existing
cd ..

# 2. Publish Rust crates to crates.io
echo "🦀 Publishing Rust crates to crates.io..."

# Publish in dependency order
CRATES=(
    "core"
    "orchestration"
    "agents/ingest"
    "agents/analysis"
    "agents/transpiler"
    "agents/build"
    "agents/test"
    "agents/packaging"
    "cli"
)

for crate in "${CRATES[@]}"; do
    echo "Publishing $crate..."
    cd "$crate"
    cargo publish --token "$CARGO_REGISTRY_TOKEN" || true
    cd -
    sleep 5  # Rate limiting
done

echo "✅ Publishing complete!"
echo ""
echo "Users can now install with:"
echo "  cargo install portalis"
echo "  pip install portalis-nemo-integration  # optional GPU"
```

**Usage (YOU run this):**
```bash
# Dry run first
./publish.sh --dry-run

# Real publish
./publish.sh
```

**Key Points:**
- ✅ Credentials stay in `.env` (gitignored)
- ✅ Script reads from `.env` automatically
- ✅ One command publishes everything
- ✅ End-users NEVER see or need these credentials
- ✅ End-users just `cargo install portalis` after you publish

---

## Part B: End-User Experience (Using Portalis)

### B.1 The Actual Problem

**Current confusion for end-users:**

After installing `cargo install portalis`, how do they:
1. Specify which Python library or script to convert?
2. Understand what gets converted?
3. Know where the output goes?
4. Use the WASM output?

**Example end-user scenarios:**

**Scenario 1: Convert a single Python script**
```bash
# User has: my_script.py
# User wants: my_script.wasm

# What command do they run?
portalis ??? my_script.py ???
```

**Scenario 2: Convert a Python library**
```bash
# User has: mylib/ (Python package with __init__.py, multiple .py files)
# User wants: Convert entire library to Rust crate or WASM

# What command do they run?
portalis ??? mylib/ ???
```

**Scenario 3: Convert dependencies**
```bash
# User has: app.py that imports numpy, requests, pandas
# User wants: Standalone WASM with dependencies handled

# What command do they run?
portalis ??? app.py --with-deps ???
```

### B.2 Proposed End-User CLI

**Primary Command: `portalis convert`**

```bash
portalis convert <INPUT> [OPTIONS]

Arguments:
  <INPUT>                   Python file (.py) or package directory

Options:
  -o, --output <PATH>       Output path (default: ./dist)
  --format <FORMAT>         Output format: wasm, rust, both [default: wasm]
  --with-stdlib             Include Python stdlib mappings [default: true]
  --analyze                 Show compatibility analysis first
  --verbose                 Show detailed progress

Examples:
  # Convert single script to WASM
  portalis convert my_script.py

  # Convert to Rust source only
  portalis convert my_script.py --format rust

  # Convert entire package
  portalis convert ./mylib/

  # Analyze before converting
  portalis convert complex_app.py --analyze

  # Custom output location
  portalis convert app.py -o ./build/app.wasm
```

**Output:**
```
Converting: my_script.py
├─ Analyzing Python code... ✓
├─ Mapping stdlib imports... ✓
├─ Translating to Rust... ✓
├─ Compiling to WASM... ✓
└─ Output: ./dist/my_script.wasm (245 KB)

Next steps:
  Run with Node.js: node -e "require('./dist/my_script.wasm')"
  Run with browser: <script src="dist/my_script.wasm"></script>
  Test: portalis test dist/my_script.wasm
```

### B.3 Clear Input Detection

**Auto-detection logic:**

```rust
fn detect_input_type(path: &str) -> InputType {
    let path = Path::new(path);

    if path.is_file() && path.extension() == Some("py") {
        // Single Python file
        InputType::SingleFile(path.to_path_buf())
    } else if path.is_dir() {
        // Check if it's a Python package
        if path.join("__init__.py").exists() ||
           path.join("setup.py").exists() ||
           path.join("pyproject.toml").exists() {
            InputType::PythonPackage(path.to_path_buf())
        } else {
            // Directory with Python files
            InputType::Directory(path.to_path_buf())
        }
    } else {
        InputType::Invalid
    }
}
```

**Clear error messages:**

```bash
$ portalis convert not_found.py
❌ Error: File not found: not_found.py

Did you mean one of these?
  ./script.py
  ./examples/add.py
  ./tests/test_simple.py

$ portalis convert binary_file.exe
❌ Error: Input must be a Python file (.py) or directory

Usage: portalis convert <file.py|directory>
```

### B.4 Example End-User Workflows

**Workflow 1: Simple script conversion**
```bash
# Create Python file
cat > calculator.py << EOF
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
EOF

# Convert to WASM
portalis convert calculator.py

# Output
# → ./dist/calculator.wasm
# → ./dist/calculator.js (bindings)
```

**Workflow 2: Library conversion**
```bash
# User has a Python package
mylib/
  __init__.py
  core.py
  utils.py

# Convert entire package
portalis convert mylib/

# Output
# → ./dist/mylib.wasm
# OR
# → ./dist/mylib/ (Rust crate)
```

**Workflow 3: Analysis first**
```bash
portalis convert complex_app.py --analyze

# Output:
# Analyzing complex_app.py...
#
# ✓ Supported features (85%):
#   - Functions, classes, decorators
#   - List comprehensions
#   - Type hints
#
# ⚠ Partial support (10%):
#   - asyncio (requires runtime)
#   - File I/O (WASI only)
#
# ✗ Unsupported (5%):
#   - C extensions (numpy, pandas)
#   - Dynamic imports
#
# Recommendation: Proceed with conversion
# Estimated output size: ~500 KB WASM
#
# Convert now? [y/N]: y
```

---

## Part C: Is Portalis Itself WASM?

### C.1 Current Architecture

**What you have:**
- ✅ Portalis CLI (Rust binary)
- ✅ Portalis transpiler (Rust library, can compile to WASM)
- ✅ Example WASM outputs (fibonacci.wasm, etc.)

**The transpiler Cargo.toml shows:**
```toml
[lib]
crate-type = ["cdylib", "rlib"]

[features]
wasm = ["wasm-bindgen", "wasm-bindgen-futures", ...]
wasi = ["dep:wasi", "tokio"]
```

**This means:**
- `rlib` = Used by CLI (native Rust)
- `cdylib` = Can be compiled to WASM

### C.2 Two Deployment Models

**Model 1: Native CLI (Current - Primary)**
```bash
# User installs native binary
cargo install portalis

# Runs natively (fast)
portalis convert app.py
```

**Model 2: WASM Transpiler (Optional - Web/Cloud)**
```javascript
// Run Portalis transpiler in browser or Node.js
import init, { transpile } from './portalis_transpiler.wasm';

await init();
const result = transpile('def add(a, b): return a + b');
console.log(result.rust_code);
```

### C.3 Recommendation

**Primary distribution: Native binary**
- Faster execution
- Better tooling integration
- Simpler for most users

**Secondary distribution: WASM module**
- For web-based IDEs
- Cloud serverless functions
- Browser-based tools

**Publishing both:**
```bash
# Native binary (crates.io)
cargo install portalis

# WASM module (npm)
npm install @portalis/transpiler-wasm
```

---

## Revised Implementation Plan

### Phase 1: Platform Publishing (Week 1)

**Goal:** Get Portalis published so users can install it

**Day 1-2: Prepare for publishing**
- [ ] Review all Cargo.toml metadata (description, keywords, license)
- [ ] Review all setup.py/pyproject.toml metadata
- [ ] Ensure README.md is clear for crates.io/PyPI
- [ ] Run all tests: `cargo test --all`
- [ ] Check licenses are correct

**Day 3-4: Create publishing automation**
- [ ] Create `publish.sh` script
- [ ] Test with `--dry-run` on test registries
- [ ] Document publishing process
- [ ] Create GitHub Actions workflow (optional)

**Day 5: Publish to registries**
- [ ] Publish Python packages to PyPI
- [ ] Publish Rust crates to crates.io (in dependency order)
- [ ] Verify installations work: `cargo install portalis`
- [ ] Create release announcement

### Phase 2: End-User UX (Week 2)

**Goal:** Make `portalis convert` intuitive and clear

**Day 1-2: Implement smart input detection**
- [ ] Create input type detection logic
- [ ] Add clear error messages
- [ ] Test with various input types (file, dir, package)

**Day 3-4: Simplify commands**
- [ ] Implement simplified `convert` command
- [ ] Add `--analyze` flag
- [ ] Add auto-detection of output format
- [ ] Improve progress feedback

**Day 5-7: Documentation**
- [ ] Update getting-started.md with clear examples
- [ ] Create tutorial videos/GIFs
- [ ] Document all common workflows
- [ ] Add troubleshooting guide

### Phase 3: WASM Deployment (Week 3) - Optional

**Goal:** Provide WASM version of transpiler for web use

**If you want web-based transpiler:**
- [ ] Build transpiler as WASM: `wasm-pack build agents/transpiler`
- [ ] Publish to npm: `npm publish @portalis/transpiler-wasm`
- [ ] Create web demo
- [ ] Document web API

---

## Files to Create

### 1. `publish.sh` (Root of repo)
```bash
#!/bin/bash
# One-command publishing for Portalis platform
```

### 2. `.github/workflows/publish.yml` (CI/CD)
```yaml
name: Publish Portalis
on:
  release:
    types: [created]
env:
  CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}
  TWINE_USERNAME: ${{ secrets.TWINE_USERNAME }}
  TWINE_PASSWORD: ${{ secrets.TWINE_PASSWORD }}
```

### 3. `docs/publishing-maintainer-guide.md`
- How YOU publish Portalis
- Credential setup
- Release checklist

### 4. `docs/user-guide-simple.md`
- How END-USERS use Portalis after installation
- Clear examples
- Common workflows

### 5. Update existing files
- `README.md` - Clear installation instructions
- `getting-started.md` - Simplify end-user examples
- `cli/src/main.rs` - Simplify commands

---

## Key Differences from First Plan

| First Plan (WRONG) | Revised Plan (CORRECT) |
|---------------------|------------------------|
| Credentials for end-users | Credentials for YOU to publish |
| Complex `publish` subcommand in CLI | Simple `publish.sh` script for maintainer |
| Unclear input specification | Clear: `portalis convert <file-or-dir>` |
| Assumed already published | Need to publish FIRST |
| Mixed concerns | Separated: (A) Publishing vs (B) Using |

---

## Success Criteria

### For Publishing (Your Goal):
- ✅ `cargo install portalis` works globally
- ✅ `pip install portalis-nemo-integration` works (optional)
- ✅ Users never need YOUR credentials
- ✅ Automated publishing script exists

### For End-Users (Their Goal):
- ✅ `portalis convert script.py` just works
- ✅ Clear error messages
- ✅ Auto-detects input type
- ✅ Obvious output location
- ✅ 5-minute quick start

---

## Next Steps

1. **Review this revised plan** - Confirm understanding is correct
2. **Choose priority:**
   - Option A: Publish platform FIRST, then improve UX
   - Option B: Improve UX FIRST, then publish
3. **Clarify WASM strategy:**
   - Native CLI only?
   - Both native + WASM transpiler?
4. **Decide on timeline:**
   - Publish this week?
   - Improve UX first?

---

**Questions for you:**

1. Do you want to publish Portalis to crates.io/PyPI NOW (this week)?
2. Is the transpiler itself meant to run as WASM, or just produce WASM?
3. What should `portalis convert mylib/` output - WASM or Rust crate?
4. Are there any examples of the CLI actually working on the example Python files?

