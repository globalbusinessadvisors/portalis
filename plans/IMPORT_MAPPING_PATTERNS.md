# Import Mapping Patterns - Quick Reference

**Purpose**: Common patterns for mapping Python imports to Rust crates
**Audience**: Mapping engineers adding new modules/packages
**See Also**: `IMPORT_MAPPING_REQUIREMENTS_ANALYSIS.md` for full requirements

---

## Module Mapping Patterns

### Pattern 1: Direct Stdlib Mapping (No Crate)

**When**: Python stdlib module maps to Rust std library

```rust
ModuleMapping {
    python_module: "math".to_string(),
    rust_crate: None,  // No external crate needed
    rust_use: "std::f64::consts".to_string(),
    dependencies: vec![],
    version: "*".to_string(),
    wasm_compatible: WasmCompatibility::Full,
    notes: Some("Pure computation, fully WASM compatible".to_string()),
}
```

**Examples**: `math`, `json` (with serde_json), `collections`, `itertools`

### Pattern 2: External Crate Mapping

**When**: Python module maps to Rust crate on crates.io

```rust
ModuleMapping {
    python_module: "csv".to_string(),
    rust_crate: Some("csv".to_string()),  // Crate name
    rust_use: "csv".to_string(),
    dependencies: vec!["serde".to_string()],  // Additional deps
    version: "1".to_string(),
    wasm_compatible: WasmCompatibility::Partial,
    notes: Some("Works with strings, file I/O requires WASI".to_string()),
}
```

**Examples**: `requests` → `reqwest`, `pillow` → `image`, `numpy` → `ndarray`

### Pattern 3: WASI Required Mapping

**When**: Module needs filesystem or OS access

```rust
ModuleMapping {
    python_module: "pathlib".to_string(),
    rust_crate: None,
    rust_use: "std::path::Path".to_string(),
    dependencies: vec![],
    version: "*".to_string(),
    wasm_compatible: WasmCompatibility::RequiresWasi,
    notes: Some("Requires WASI for filesystem access in browser".to_string()),
}
```

**Examples**: `pathlib`, `tempfile`, `glob`, `os.path`, `shutil`

### Pattern 4: JS Interop Required

**When**: Module needs browser APIs (fetch, crypto, etc.)

```rust
ModuleMapping {
    python_module: "requests".to_string(),
    rust_crate: Some("reqwest".to_string()),
    rust_use: "reqwest".to_string(),
    dependencies: vec![],
    version: "0.11".to_string(),
    wasm_compatible: WasmCompatibility::RequiresJsInterop,
    notes: Some("Uses fetch() API in browser WASM".to_string()),
}
```

**Examples**: `requests`, `urllib.request`, `secrets`, `time.time()`, `datetime.now()`

### Pattern 5: Incompatible Mapping

**When**: Module cannot work in WASM

```rust
ModuleMapping {
    python_module: "subprocess".to_string(),
    rust_crate: None,
    rust_use: "std::process".to_string(),
    dependencies: vec![],
    version: "*".to_string(),
    wasm_compatible: WasmCompatibility::Incompatible,
    notes: Some("Process spawning not available in WASM - use JS interop for browser, WASI for server (limited)".to_string()),
}
```

**Examples**: `subprocess`, `socket` (raw), `multiprocessing` (full), `signal`

---

## Function Mapping Patterns

### Pattern 1: Direct Function Call

**When**: Python function → Rust function (same semantics)

```rust
FunctionMapping {
    python_name: "sqrt".to_string(),
    rust_equiv: "f64::sqrt".to_string(),
    requires_use: None,
    wasm_compatible: WasmCompatibility::Full,
    transform_notes: None,
}
```

**Examples**: `math.sqrt` → `f64::sqrt`, `json.dumps` → `serde_json::to_string`

### Pattern 2: Method Call

**When**: Python function → Rust method on type

```rust
FunctionMapping {
    python_name: "heappush".to_string(),
    rust_equiv: "heap.push".to_string(),
    requires_use: Some("std::collections::BinaryHeap".to_string()),
    wasm_compatible: WasmCompatibility::Full,
    transform_notes: Some("BinaryHeap is max-heap, use Reverse for min-heap".to_string()),
}
```

**Examples**: `heapq.heappush` → `heap.push`, `list.append` → `vec.push`

### Pattern 3: Different API Pattern

**When**: Rust API is significantly different from Python

```rust
FunctionMapping {
    python_name: "reduce".to_string(),
    rust_equiv: "iterator.fold".to_string(),
    requires_use: None,
    wasm_compatible: WasmCompatibility::Full,
    transform_notes: Some("Use .fold() or .reduce() on iterators".to_string()),
}
```

**Examples**: `functools.reduce` → `iter.fold`, `map()` → `iter.map()`, `filter()` → `iter.filter()`

### Pattern 4: Constructor/Builder

**When**: Python class constructor → Rust builder pattern

```rust
FunctionMapping {
    python_name: "EmailMessage".to_string(),
    rust_equiv: "lettre::Message::builder".to_string(),
    requires_use: Some("lettre::Message".to_string()),
    wasm_compatible: WasmCompatibility::Full,
    transform_notes: Some("Use builder pattern for message construction".to_string()),
}
```

**Examples**: `email.EmailMessage()` → `Message::builder()`, `DataFrame()` → `DataFrame::new()`

### Pattern 5: Macro or Attribute

**When**: Python decorator/feature → Rust macro/attribute

```rust
FunctionMapping {
    python_name: "dataclass".to_string(),
    rust_equiv: "#[derive(Debug, Clone)]".to_string(),
    requires_use: None,
    wasm_compatible: WasmCompatibility::Full,
    transform_notes: Some("Use struct with derive attributes".to_string()),
}
```

**Examples**: `@dataclass` → `#[derive(...)]`, `@lru_cache` → `#[cached]`, `@test` → `#[test]`

---

## Class Mapping Patterns

### Pattern 1: Simple Class → Struct

**When**: Python class with __init__ → Rust struct with impl

```rust
ClassMapping {
    python_class: "Point".to_string(),
    rust_type: "Point".to_string(),
    constructor: ConstructorMapping {
        python_init: "__init__(self, x, y)".to_string(),
        rust_equiv: "Point::new".to_string(),
        params: vec![
            ParamMapping {
                python_param: "x".to_string(),
                python_type: "float".to_string(),
                rust_param: "x".to_string(),
                rust_type: "f64".to_string(),
                default_value: None,
            },
            // ... y param
        ],
    },
    methods: vec![
        MethodMapping {
            python_method: "distance".to_string(),
            rust_method: "distance".to_string(),
            params: vec![/* other: Point */],
            return_type: TypeMapping {
                python_type: "float".to_string(),
                rust_type: "f64".to_string(),
            },
            transform_notes: None,
        },
    ],
    properties: vec![],
    wasm_compatible: WasmCompatibility::Full,
}
```

### Pattern 2: Class with Properties → Struct with Getters

**When**: Python @property → Rust getter methods

```rust
ClassMapping {
    python_class: "Circle".to_string(),
    rust_type: "Circle".to_string(),
    // ... constructor
    methods: vec![
        MethodMapping {
            python_method: "area".to_string(),
            rust_method: "area".to_string(),
            params: vec![],
            return_type: TypeMapping {
                python_type: "float".to_string(),
                rust_type: "f64".to_string(),
            },
            transform_notes: Some("@property in Python, method in Rust".to_string()),
        },
    ],
    properties: vec![
        PropertyMapping {
            python_name: "radius".to_string(),
            rust_field: "radius".to_string(),
            python_type: "float".to_string(),
            rust_type: "f64".to_string(),
            getter: Some("radius".to_string()),
            setter: Some("set_radius".to_string()),
        },
    ],
    wasm_compatible: WasmCompatibility::Full,
}
```

### Pattern 3: Enum Class → Rust Enum

**When**: Python Enum class → Rust enum

```rust
ClassMapping {
    python_class: "Color".to_string(),
    rust_type: "enum Color".to_string(),
    constructor: ConstructorMapping {
        python_init: "Color.RED".to_string(),
        rust_equiv: "Color::Red".to_string(),
        params: vec![],
    },
    methods: vec![],
    properties: vec![],
    wasm_compatible: WasmCompatibility::Full,
}
```

### Pattern 4: Container Class → Generic Struct

**When**: Python container (DataFrame, Array) → Rust generic struct

```rust
ClassMapping {
    python_class: "pandas.DataFrame".to_string(),
    rust_type: "polars::DataFrame".to_string(),
    constructor: ConstructorMapping {
        python_init: "__init__(data, columns=None, ...)".to_string(),
        rust_equiv: "DataFrame::new".to_string(),
        params: vec![
            ParamMapping {
                python_param: "data".to_string(),
                python_type: "dict | list | ndarray".to_string(),
                rust_param: "series".to_string(),
                rust_type: "Vec<Series>".to_string(),
                default_value: None,
            },
        ],
    },
    methods: vec![
        MethodMapping {
            python_method: "head".to_string(),
            rust_method: "head".to_string(),
            params: vec![
                ParamMapping {
                    python_param: "n".to_string(),
                    python_type: "int".to_string(),
                    rust_param: "n".to_string(),
                    rust_type: "Option<usize>".to_string(),
                    default_value: Some("Some(5)".to_string()),
                },
            ],
            return_type: TypeMapping {
                python_type: "DataFrame".to_string(),
                rust_type: "DataFrame".to_string(),
            },
            transform_notes: Some("Returns owned DataFrame, not view".to_string()),
        },
        // ... 100+ more methods
    ],
    properties: vec![
        PropertyMapping {
            python_name: "shape".to_string(),
            rust_field: "N/A".to_string(),
            python_type: "tuple[int, int]".to_string(),
            rust_type: "(usize, usize)".to_string(),
            getter: Some("shape".to_string()),
            setter: None,
        },
    ],
    wasm_compatible: WasmCompatibility::Partial,
}
```

---

## Type Mapping Patterns

### Pattern 1: Primitive Types

| Python Type | Rust Type | Notes |
|------------|-----------|-------|
| `int` | `i64` or `i32` | Context-dependent size |
| `float` | `f64` | Standard 64-bit float |
| `str` | `String` or `&str` | Owned vs borrowed |
| `bytes` | `Vec<u8>` or `&[u8]` | Owned vs borrowed |
| `bool` | `bool` | Direct mapping |
| `None` | `()` or `Option<T>` | Context-dependent |

### Pattern 2: Collection Types

| Python Type | Rust Type | Notes |
|------------|-----------|-------|
| `list[T]` | `Vec<T>` | Generic preserved |
| `tuple[T1, T2]` | `(T1, T2)` | Fixed-size tuples |
| `set[T]` | `HashSet<T>` | Generic preserved |
| `dict[K, V]` | `HashMap<K, V>` | Generic preserved |
| `frozenset[T]` | `HashSet<T>` | Immutable by default |

### Pattern 3: Optional & Union Types

| Python Type | Rust Type | Notes |
|------------|-----------|-------|
| `Optional[T]` | `Option<T>` | Direct mapping |
| `T | None` | `Option<T>` | Union with None |
| `Union[T1, T2]` | `enum { T1(T1), T2(T2) }` | Custom enum |
| `Any` | `Box<dyn Any>` | Dynamic typing |

### Pattern 4: Callable Types

| Python Type | Rust Type | Notes |
|------------|-----------|-------|
| `Callable[[int], str]` | `Fn(i64) -> String` | Function trait |
| `Callable[[], T]` | `Fn() -> T` | No args |
| `Callable[[...], None]` | `Fn(...) -> ()` | Void return |

### Pattern 5: NumPy Types

| Python Type | Rust Type (ndarray) | Notes |
|------------|-------------------|-------|
| `np.int8` | `Array<i8, D>` | 8-bit int array |
| `np.int64` | `Array<i64, D>` | 64-bit int array |
| `np.float32` | `Array<f32, D>` | 32-bit float array |
| `np.float64` | `Array<f64, D>` | 64-bit float array |
| `np.ndarray` | `ArrayD<T>` | Dynamic dimensions |
| `np.matrix` | `Array2<T>` | Always 2D |

---

## WASM Compatibility Patterns

### Pattern 1: Full WASM Compatible

**Characteristics**:
- Pure computation
- No I/O, no OS calls
- Works in all WASM targets

**Example Modules**: `math`, `json`, `base64`, `hashlib`, `regex`, `collections`

**Template**:
```rust
ModuleMapping {
    // ... other fields
    wasm_compatible: WasmCompatibility::Full,
    notes: Some("Pure computation, fully WASM compatible".to_string()),
}
```

### Pattern 2: Requires WASI

**Characteristics**:
- Needs filesystem access
- Needs environment variables
- Needs process info

**Example Modules**: `pathlib`, `tempfile`, `glob`, `os`, `shutil`

**Template**:
```rust
ModuleMapping {
    // ... other fields
    wasm_compatible: WasmCompatibility::RequiresWasi,
    notes: Some("Requires WASI for filesystem access".to_string()),
}

// In transpiled code:
#[cfg(target_os = "wasi")]
use std::fs;

#[cfg(not(target_os = "wasi"))]
compile_error!("This module requires WASI");
```

### Pattern 3: Requires JS Interop

**Characteristics**:
- Needs browser APIs (fetch, crypto, storage)
- Needs network access
- Needs timing/performance APIs

**Example Modules**: `requests`, `urllib.request`, `secrets`, `time.time()`, `datetime.now()`

**Template**:
```rust
ModuleMapping {
    // ... other fields
    wasm_compatible: WasmCompatibility::RequiresJsInterop,
    notes: Some("Uses browser fetch() API in WASM".to_string()),
}

// In transpiled code:
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
pub async fn fetch_data(url: &str) -> Result<String> {
    let window = web_sys::window().unwrap();
    // ... fetch implementation
}
```

### Pattern 4: Partial WASM

**Characteristics**:
- Some features work (in-memory operations)
- Some features don't work (I/O operations)

**Example Modules**: `csv` (strings ok, files no), `zipfile` (memory ok, files no), `pandas` (operations ok, I/O no)

**Template**:
```rust
ModuleMapping {
    // ... other fields
    wasm_compatible: WasmCompatibility::Partial,
    notes: Some("✅ WASM: In-memory operations work. ❌ WASM: File I/O requires WASI or JS interop".to_string()),
}

// Document what works and what doesn't:
FunctionMapping {
    python_name: "read_csv".to_string(),
    rust_equiv: "polars::CsvReader::from_reader".to_string(),
    wasm_compatible: WasmCompatibility::RequiresWasi,
    transform_notes: Some("❌ WASM: File I/O requires WASI. Alternative: pass data as string or use IndexedDB in browser".to_string()),
}

FunctionMapping {
    python_name: "DataFrame.head".to_string(),
    rust_equiv: ".head(Some(n))".to_string(),
    wasm_compatible: WasmCompatibility::Full,
    transform_notes: Some("✅ WASM: Operations work everywhere".to_string()),
}
```

### Pattern 5: Incompatible

**Characteristics**:
- Cannot work in WASM sandbox
- Raw sockets, process spawning, signals, etc.

**Example Modules**: `subprocess`, `socket` (raw), `multiprocessing`, `signal`

**Template**:
```rust
ModuleMapping {
    // ... other fields
    wasm_compatible: WasmCompatibility::Incompatible,
    notes: Some("Process spawning not available in WASM - use JS interop for browser, WASI for server (limited)".to_string()),
}

// Provide alternatives:
// "Alternative: Use web_sys::Window::post_message for IPC in browser"
// "Alternative: Use WASI process_spawn in server-side WASM (limited)"
```

---

## API Granularity Patterns

### Level 1: Module-Only (Minimum)

**When**: Just need to map the import, no function details

```rust
// Just module mapping, no function mappings
vec![(
    ModuleMapping { /* ... */ },
    vec![]  // Empty function list
)]
```

**Use Case**: Rarely used modules, placeholder for future expansion

### Level 2: Basic (5-10 APIs)

**When**: Map the most common functions

```rust
vec![(
    ModuleMapping { /* ... */ },
    vec![
        FunctionMapping { python_name: "open", /* ... */ },
        FunctionMapping { python_name: "read", /* ... */ },
        FunctionMapping { python_name: "write", /* ... */ },
        FunctionMapping { python_name: "close", /* ... */ },
        FunctionMapping { python_name: "exists", /* ... */ },
    ]
)]
```

**Use Case**: Initial mapping, cover 80% use cases

### Level 3: Comprehensive (15-25 APIs)

**When**: Map all commonly used functions

```rust
vec![(
    ModuleMapping { /* ... */ },
    vec![
        // 15-25 function mappings covering:
        // - Constructors/factories
        // - Core operations
        // - Utility functions
        // - Common workflows
    ]
)]
```

**Use Case**: Production-quality mapping, 95% use cases

### Level 4: Complete (40+ APIs)

**When**: Map entire API surface

```rust
vec![(
    ModuleMapping { /* ... */ },
    vec![
        // 40+ function mappings covering:
        // - All public functions
        // - All public classes (as functions)
        // - All public methods
        // - All edge cases
    ]
)]
```

**Use Case**: Critical packages (NumPy, Pandas, requests)

---

## External Package Patterns

### Pattern 1: Data Science Package

**Example**: NumPy → ndarray

```rust
ExternalPackageMapping {
    python_package: "numpy".to_string(),
    rust_crate: "ndarray".to_string(),
    version: "0.15".to_string(),
    features: vec![],
    wasm_compatible: WasmCompatibility::Full,
    api_mappings: vec![
        ApiMapping {
            python_api: "numpy.array".to_string(),
            rust_equiv: "ndarray::arr1".to_string(),
            requires_use: Some("ndarray".to_string()),
            transform_notes: Some("Use arr1 for 1D, arr2 for 2D".to_string()),
            wasm_compatible: WasmCompatibility::Full,
        },
        // ... 40+ more API mappings
    ],
    notes: Some("Pure computation, fully WASM compatible".to_string()),
    alternatives: vec!["nalgebra".to_string()],
}
```

### Pattern 2: Web Framework Package

**Example**: Flask → actix-web

```rust
ExternalPackageMapping {
    python_package: "flask".to_string(),
    rust_crate: "actix-web".to_string(),
    version: "4".to_string(),
    features: vec![],
    wasm_compatible: WasmCompatibility::Incompatible,
    api_mappings: vec![],
    notes: Some("Use warp or axum for server. Not applicable in browser WASM.".to_string()),
    alternatives: vec!["warp".to_string(), "axum".to_string()],
}
```

### Pattern 3: HTTP Client Package

**Example**: requests → reqwest

```rust
ExternalPackageMapping {
    python_package: "requests".to_string(),
    rust_crate: "reqwest".to_string(),
    version: "0.11".to_string(),
    features: vec!["json".to_string(), "blocking".to_string()],
    wasm_compatible: WasmCompatibility::RequiresJsInterop,
    api_mappings: vec![
        ApiMapping {
            python_api: "requests.get".to_string(),
            rust_equiv: "reqwest::blocking::get".to_string(),
            requires_use: Some("reqwest".to_string()),
            transform_notes: Some("Use async version in WASM with wasm-bindgen-futures".to_string()),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
        },
    ],
    notes: Some("Uses fetch() API in browser WASM".to_string()),
    alternatives: vec!["ureq".to_string()],
}
```

### Pattern 4: Database Package

**Example**: SQLAlchemy → diesel

```rust
ExternalPackageMapping {
    python_package: "sqlalchemy".to_string(),
    rust_crate: "diesel".to_string(),
    version: "2".to_string(),
    features: vec!["sqlite".to_string()],
    wasm_compatible: WasmCompatibility::Partial,
    api_mappings: vec![],
    notes: Some("✅ WASM: SQLite works with sql.js in browser. ❌ WASM: PostgreSQL/MySQL require network access via JS interop or server proxy.".to_string()),
    alternatives: vec!["sqlx".to_string(), "sea-orm".to_string()],
}
```

### Pattern 5: ML Package

**Example**: PyTorch → burn

```rust
ExternalPackageMapping {
    python_package: "torch".to_string(),
    rust_crate: "burn".to_string(),
    version: "0.11".to_string(),
    features: vec![],
    wasm_compatible: WasmCompatibility::Partial,
    api_mappings: vec![],
    notes: Some("✅ WASM: Inference works with pre-trained models. Training works for small models. ❌ WASM: Large model training limited by memory.".to_string()),
    alternatives: vec!["tch".to_string(), "tract".to_string()],
}
```

---

## Common Transformation Patterns

### Pattern 1: List Comprehension → Iterator

**Python**:
```python
result = [x * 2 for x in numbers if x > 0]
```

**Rust**:
```rust
let result: Vec<i64> = numbers.iter()
    .filter(|&&x| x > 0)
    .map(|&x| x * 2)
    .collect();
```

**Transform Notes**: "Use .iter().filter().map().collect() pattern"

### Pattern 2: Dictionary Comprehension → Iterator

**Python**:
```python
result = {k: v * 2 for k, v in items.items() if v > 0}
```

**Rust**:
```rust
let result: HashMap<String, i64> = items.iter()
    .filter(|(_, &v)| v > 0)
    .map(|(k, &v)| (k.clone(), v * 2))
    .collect();
```

**Transform Notes**: "Use .iter().filter().map().collect() into HashMap"

### Pattern 3: With Statement → RAII

**Python**:
```python
with open('file.txt') as f:
    content = f.read()
```

**Rust**:
```rust
let content = std::fs::read_to_string("file.txt")?;
// Or with explicit scope:
{
    let file = File::open("file.txt")?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
}
// File automatically closed at end of scope
```

**Transform Notes**: "Use RAII - resources automatically cleaned up at end of scope"

### Pattern 4: Try/Except → Result

**Python**:
```python
try:
    result = risky_operation()
except ValueError as e:
    result = default_value
```

**Rust**:
```rust
let result = risky_operation()
    .unwrap_or(default_value);
// Or with error handling:
let result = match risky_operation() {
    Ok(val) => val,
    Err(e) if e.is::<ValueError>() => default_value,
    Err(e) => return Err(e),
};
```

**Transform Notes**: "Use Result<T, E> and match or combinators (.unwrap_or, .map_err, etc.)"

### Pattern 5: Async/Await → Async/Await (different runtime)

**Python**:
```python
async def fetch_data():
    response = await client.get(url)
    return await response.json()
```

**Rust**:
```rust
async fn fetch_data() -> Result<Value> {
    let response = client.get(url).await?;
    let data = response.json().await?;
    Ok(data)
}
```

**Transform Notes**: "Use tokio or async-std runtime, Result instead of exceptions"

---

## Testing Patterns

### Pattern 1: Unit Test for Mapping

```rust
#[test]
fn test_numpy_array_mapping() {
    let registry = ExternalPackageRegistry::new();
    let numpy = registry.get_package("numpy").unwrap();

    assert_eq!(numpy.rust_crate, "ndarray");
    assert_eq!(numpy.wasm_compatible, WasmCompatibility::Full);

    let array_api = numpy.api_mappings.iter()
        .find(|m| m.python_api == "numpy.array")
        .unwrap();
    assert_eq!(array_api.rust_equiv, "ndarray::arr1");
}
```

### Pattern 2: Integration Test for Transpilation

```rust
#[test]
fn test_transpile_numpy_code() {
    let python = r#"
import numpy as np
arr = np.array([1, 2, 3])
result = np.sum(arr)
    "#;

    let rust = transpile(python).unwrap();

    assert!(rust.contains("use ndarray"));
    assert!(rust.contains("arr1(&[1, 2, 3])"));
    assert!(rust.contains(".sum()"));
}
```

### Pattern 3: WASM Compatibility Test

```rust
#[test]
fn test_wasm_compatibility() {
    let mapper = StdlibMapper::new();

    // Full WASM
    assert_eq!(
        mapper.get_wasm_compatibility("math"),
        Some(WasmCompatibility::Full)
    );

    // Requires WASI
    assert_eq!(
        mapper.get_wasm_compatibility("pathlib"),
        Some(WasmCompatibility::RequiresWasi)
    );

    // Incompatible
    assert_eq!(
        mapper.get_wasm_compatibility("subprocess"),
        Some(WasmCompatibility::Incompatible)
    );
}
```

---

## Documentation Patterns

### Pattern 1: Module Documentation

```markdown
## Module: {python_module}

**Rust Equivalent**: {rust_crate} {version}
**WASM Compatibility**: {compatibility_level}

### Browser (wasm32-unknown-unknown)
- ✅ Works: {list features}
- ❌ Doesn't work: {list features}
- 🔧 Workaround: {alternatives}

### Example
\`\`\`python
# Python
{python_example}
\`\`\`

\`\`\`rust
// Rust
{rust_example}
\`\`\`

### Dependencies
\`\`\`toml
[dependencies]
{crate} = "{version}"
\`\`\`
```

### Pattern 2: API Documentation

```markdown
## API: {python_api}

**Python**: `{python_signature}`
**Rust**: `{rust_signature}`

### Parameters
| Python | Rust | Type Mapping | Notes |
|--------|------|--------------|-------|
| {param} | {param} | {py_type} → {rust_type} | {notes} |

### Returns
| Python | Rust | Notes |
|--------|------|-------|
| {py_type} | {rust_type} | {notes} |

### Transform Notes
{transformation_details}
```

---

## Checklist for New Mappings

### Before Adding a Mapping

- [ ] Check if module/package is in top 1000 by usage
- [ ] Search for existing Rust crate alternatives
- [ ] Evaluate WASM compatibility
- [ ] Identify key APIs (aim for 15+ APIs minimum)
- [ ] Document transformation patterns
- [ ] Check for multiple Rust alternatives

### While Adding a Mapping

- [ ] Create `ModuleMapping` or `ExternalPackageMapping`
- [ ] Set correct `WasmCompatibility` level
- [ ] Add detailed notes (especially for partial/incompatible)
- [ ] Add 15+ API mappings (or justify why fewer)
- [ ] Add transform notes for non-obvious mappings
- [ ] List alternative crates if available

### After Adding a Mapping

- [ ] Add unit test for the mapping
- [ ] Add integration test for transpilation
- [ ] Add WASM compatibility test
- [ ] Document with examples
- [ ] Update coverage statistics
- [ ] Add to migration guide (if complex)

---

## Quick Reference

### WASM Compatibility Decision Tree

```
Does it do pure computation (no I/O, no OS)?
├─ Yes → Full
└─ No → Does it need filesystem/OS access?
    ├─ Yes → Does it work with WASI?
    │   ├─ Yes → RequiresWasi
    │   └─ No → Incompatible
    └─ No → Does it need network/browser APIs?
        ├─ Yes → RequiresJsInterop
        └─ No → Check if some features work
            ├─ Some work → Partial
            └─ None work → Incompatible
```

### Rust Crate Selection Criteria

1. **Quality**: Well-maintained, active development
2. **WASM Support**: Works in wasm32-unknown-unknown
3. **API Similarity**: Close to Python API
4. **Performance**: Good benchmarks
5. **Community**: Popular, good docs, examples
6. **License**: Compatible license (MIT, Apache 2.0)

### Coverage Targets

| Priority | APIs per Module | Use Case |
|----------|----------------|----------|
| P0 (Critical) | 25+ | Top 50 packages |
| P1 (High) | 15+ | Top 200 packages |
| P2 (Medium) | 10+ | Top 500 packages |
| P3 (Low) | 5+ | Top 1000 packages |

---

**Quick Start**: Copy the relevant pattern from this guide and customize for your module/package.

**Full Requirements**: See `IMPORT_MAPPING_REQUIREMENTS_ANALYSIS.md`

**Current Status**: See `IMPORT_MAPPING_EXECUTIVE_SUMMARY.md`
