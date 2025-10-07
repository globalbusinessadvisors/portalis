# Portalis Use Cases - Real-World Examples

## Use Case 1: Convert a Single Python Script

**Scenario:** You have a Python utility script you want to run in the browser.

**Input:**
```python
# price_calculator.py
def calculate_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

def format_price(amount: float) -> str:
    return f"${amount:.2f}"
```

**Command:**
```bash
portalis convert price_calculator.py
```

**Output:**
```
dist/price_calculator.wasm
dist/price_calculator.js (bindings)
```

**Use it:**
```html
<script type="module">
  import init, { calculate_tax } from './dist/price_calculator.js';
  await init();
  console.log(calculate_tax(100.0, 0.07)); // 107.0
</script>
```

---

## Use Case 2: Convert Your Current Python Project

**Scenario:** You're in your Python project directory and want to convert everything.

**Directory structure:**
```
my-project/
├── main.py
├── utils.py
├── models.py
└── config.py
```

**Command:**
```bash
cd my-project/
portalis convert
# or: portalis convert .
```

**Output:**
```
dist/
├── main.wasm
├── utils.wasm
├── models.wasm
└── config.wasm
```

**What happens:**
- Portalis finds all `.py` files in current directory
- Converts each to a separate WASM module
- Preserves your project structure

---

## Use Case 3: Convert a Python Library/Package

**Scenario:** You maintain a Python library and want a WASM version.

**Input:**
```
mylib/
├── __init__.py
├── core.py
├── helpers.py
└── utils.py
```

**Command:**
```bash
portalis convert mylib/
```

**Output:**
```
dist/mylib/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── core.rs
│   ├── helpers.rs
│   └── utils.rs
└── pkg/
    ├── mylib_bg.wasm
    └── mylib.js
```

**What happens:**
- Detects it's a package (has `__init__.py`)
- Creates a complete Rust crate
- Each Python module becomes a Rust module
- Builds as a single WASM library
- Exports all public functions

**Use it:**
```javascript
import init, * as mylib from './dist/mylib/pkg/mylib.js';
await init();
mylib.someFunction();
```

---

## Use Case 4: Convert Django/Flask App Business Logic

**Scenario:** Extract business logic from web framework for edge deployment.

**Input:**
```
backend/
├── api.py          # Skip (framework-specific)
├── business.py     # Convert this
├── validators.py   # Convert this
└── calculations.py # Convert this
```

**Command:**
```bash
# Convert only the business logic files
portalis convert business.py
portalis convert validators.py
portalis convert calculations.py

# Or create a new directory and convert
mkdir logic/
cp business.py validators.py calculations.py logic/
portalis convert logic/
```

**Output:**
```
dist/
├── business.wasm
├── validators.wasm
└── calculations.wasm
```

**Result:**
- Run validation logic in browser
- Execute calculations on edge workers
- No Python runtime needed

---

## Use Case 5: Convert Data Processing Pipeline

**Scenario:** You have a data pipeline that needs to run fast.

**Input:**
```python
# pipeline.py
from typing import List, Dict

def clean_data(records: List[Dict]) -> List[Dict]:
    return [r for r in records if r.get('valid')]

def transform_data(records: List[Dict]) -> List[Dict]:
    return [
        {**r, 'processed': True}
        for r in records
    ]

def aggregate_data(records: List[Dict]) -> Dict:
    return {
        'total': len(records),
        'valid': sum(1 for r in records if r['valid'])
    }
```

**Command:**
```bash
portalis convert pipeline.py --format both
```

**Output:**
```
dist/
├── pipeline.wasm    # Run anywhere
└── pipeline.rs      # Rust source (for inspection/modification)
```

**Performance:**
- Python: ~100ms for 10k records
- WASM: ~10ms for 10k records (10x faster)

---

## Use Case 6: Convert Machine Learning Inference

**Scenario:** Model inference code (not training) for edge deployment.

**Input:**
```python
# inference.py
import numpy as np  # Portalis maps to Rust equivalents

def preprocess(data: List[float]) -> np.ndarray:
    return np.array(data) / 255.0

def predict(features: np.ndarray) -> int:
    # Simple model logic
    return np.argmax(features)

def postprocess(prediction: int) -> str:
    labels = ['cat', 'dog', 'bird']
    return labels[prediction]
```

**Command:**
```bash
portalis convert inference.py --analyze
# Review compatibility
portalis convert inference.py
```

**Output:**
```
dist/inference.wasm
```

**Deploy:**
- Browser: Fast client-side predictions
- Edge: Cloudflare Workers, Vercel Edge
- Mobile: WebView with WASM

---

## Use Case 7: Convert Entire Multi-File Project

**Scenario:** Complex project with multiple modules.

**Input:**
```
ecommerce/
├── setup.py
├── ecommerce/
│   ├── __init__.py
│   ├── cart.py
│   ├── pricing.py
│   ├── inventory.py
│   └── shipping.py
└── tests/
    └── test_cart.py
```

**Command:**
```bash
portalis convert ecommerce/
```

**What Portalis Does:**

1. **Detects:** Python package (has `setup.py`)
2. **Analyzes:** Maps all modules and dependencies
3. **Converts:** Each module to Rust
4. **Organizes:** Creates proper Rust crate structure
5. **Builds:** Compiles to single WASM library

**Output:**
```
dist/ecommerce/
├── Cargo.toml
├── src/
│   ├── lib.rs       # Exports all modules
│   ├── cart.rs
│   ├── pricing.rs
│   ├── inventory.rs
│   └── shipping.rs
└── pkg/
    └── ecommerce_bg.wasm
```

**Use it:**
```javascript
import init, { calculatePrice, addToCart } from './dist/ecommerce/pkg/ecommerce.js';
await init();

const cart = addToCart({id: 1, name: "Widget"});
const price = calculatePrice(cart, "USD");
```

---

## Use Case 8: Fast Iteration Workflow

**Scenario:** Developing and testing conversions quickly.

**Workflow:**
```bash
# 1. Quick convert without tests
portalis convert . --fast

# 2. Check specific file compatibility
portalis convert complex_module.py --analyze

# 3. Convert with Rust source for debugging
portalis convert problem.py --format both

# 4. Full convert with tests
portalis convert .
```

---

## Use Case 9: Convert CLI Tools

**Scenario:** Python CLI tool → WASM for universal deployment.

**Input:**
```python
# markdown_parser.py
def parse_markdown(text: str) -> str:
    # Parsing logic
    return html

def convert_file(input_path: str) -> str:
    with open(input_path) as f:
        return parse_markdown(f.read())
```

**Command:**
```bash
portalis convert markdown_parser.py
```

**Deploy:**
- Browser extension
- Desktop app (Tauri, Electron)
- Server (Cloudflare Workers)
- Mobile (React Native with WASM)

---

## Use Case 10: Legacy Code Migration

**Scenario:** Old Python 2 code you want to modernize and speed up.

**Input:**
```python
# legacy_stats.py (Python 2 style)
def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    return sorted_nums[n/2]
```

**Workflow:**
```bash
# 1. Analyze compatibility
portalis convert legacy_stats.py --analyze

# 2. See issues and recommendations
# 3. Fix obvious issues (type hints, division)
# 4. Convert

portalis convert legacy_stats.py
```

**Benefits:**
- Forces type clarity
- Identifies bugs (integer division)
- Creates fast, modern version
- No Python runtime dependency

---

## Quick Reference

| What You Have | Command | Output |
|---------------|---------|--------|
| Single script | `portalis convert script.py` | `dist/script.wasm` |
| Current project | `portalis convert` | Each `.py` → `.wasm` |
| Python package | `portalis convert mylib/` | Rust crate + WASM |
| Specific files | `portalis convert a.py b.py` | Multiple WASM |
| Need Rust too | `portalis convert . --format both` | `.rs` + `.wasm` |
| Quick test | `portalis convert . --fast` | Skip tests |

---

## Common Patterns

### Pattern 1: Microservice → Edge Function
```bash
# Extract business logic from microservice
portalis convert business_logic/
# Deploy WASM to Cloudflare Workers
```

### Pattern 2: Python Package → NPM Package
```bash
portalis convert mypackage/
cd dist/mypackage
wasm-pack publish
# Now installable: npm install mypackage-wasm
```

### Pattern 3: Prototype in Python → Production in WASM
```bash
# 1. Build Python prototype
# 2. Test and validate
# 3. Convert to WASM for production
portalis convert prototype.py
# 4. Deploy fast WASM version
```

### Pattern 4: Multi-language Monorepo
```bash
monorepo/
├── python/       # Source
│   └── lib/
└── rust/         # Generated
    └── lib/

cd python/lib
portalis convert -o ../../rust/lib
```

---

**Key Insight:** Portalis removes the friction from Python → WASM conversion by:
1. **No explicit input needed** - defaults to current directory
2. **Smart detection** - figures out file vs package vs project
3. **Complete conversion** - handles everything from script to full library
4. **One command** - `portalis convert` just works
