# Python Compatibility Matrix

This document details the Python language features supported by Portalis and their translation quality levels.

## Translation Quality Levels

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **Full** | Complete translation with high fidelity | 95%+ correctness, production-ready |
| **Partial** | Functional translation with limitations | 80-95% correctness, may need manual review |
| **Unsupported** | Not yet implemented | Workarounds available or future roadmap |

## Version Support

| Python Version | Support Level | Notes |
|----------------|---------------|-------|
| **3.11** | Full | Recommended, fully tested |
| **3.10** | Full | All features supported |
| **3.9** | Partial | Most features, some stdlib limitations |
| **3.8** | Partial | Core features only |
| <3.8 | Unsupported | Use Python 3.9+ |

## Core Language Features

### Data Types

| Feature | Quality | Rust Equivalent | Notes |
|---------|---------|-----------------|-------|
| `int` | Full | `i64` | Arbitrary precision not supported |
| `float` | Full | `f64` | IEEE 754 double precision |
| `str` | Full | `String` | UTF-8 encoding |
| `bool` | Full | `bool` | Direct mapping |
| `bytes` | Full | `Vec<u8>` | Byte arrays |
| `list` | Full | `Vec<T>` | Homogeneous types only |
| `tuple` | Full | `(T1, T2, ...)` | Fixed-size tuples |
| `dict` | Partial | `HashMap<K, V>` | Requires hashable keys |
| `set` | Partial | `HashSet<T>` | Requires hashable values |
| `None` | Full | `Option::None` | Optional types |

### Control Flow

| Feature | Quality | Notes |
|---------|---------|-------|
| `if/elif/else` | Full | Direct translation |
| `for` loop | Full | Iterator-based |
| `while` loop | Full | Direct translation |
| `break` | Full | Supported |
| `continue` | Full | Supported |
| `match` (3.10+) | Partial | Simple patterns only |
| `pass` | Full | Empty block |
| `return` | Full | Early returns supported |

### Functions

| Feature | Quality | Notes |
|---------|---------|-------|
| Function definitions | Full | `def func():` → `pub fn func()` |
| Type hints | Full | Strongly encouraged |
| Default arguments | Full | Compile-time defaults |
| Keyword arguments | Partial | Requires named parameters |
| `*args` | Partial | Translated to `Vec<T>` |
| `**kwargs` | Unsupported | Use structs instead |
| Lambda functions | Partial | Simple lambdas only |
| Closures | Partial | Limited capture support |
| Decorators | Partial | Common decorators only |
| Async functions | Partial | Tokio runtime required |

### Classes and OOP

| Feature | Quality | Rust Equivalent | Notes |
|---------|---------|-----------------|-------|
| Class definition | Full | `struct` + `impl` | Direct translation |
| `__init__` | Full | Associated function | Constructor |
| Instance methods | Full | `&self` methods | |
| Class methods | Full | `Self` parameter | `@classmethod` |
| Static methods | Full | Associated functions | `@staticmethod` |
| Properties | Partial | Getter/setter methods | `@property` |
| Inheritance | Partial | Trait-based | Single inheritance |
| Multiple inheritance | Unsupported | Use composition |
| `super()` | Partial | Explicit parent calls | |
| Magic methods | Partial | See table below | |

### Magic Methods

| Python | Rust | Quality | Notes |
|--------|------|---------|-------|
| `__str__` | `Display` trait | Full | String representation |
| `__repr__` | `Debug` trait | Full | Debug representation |
| `__eq__` | `PartialEq` trait | Full | Equality |
| `__lt__`, `__gt__` | `PartialOrd` trait | Full | Comparison |
| `__add__`, `__sub__` | Operator traits | Full | Arithmetic |
| `__len__` | Method | Full | Length/size |
| `__getitem__` | Index trait | Partial | Array access |
| `__iter__` | `Iterator` trait | Partial | Iteration |
| `__call__` | `Fn` trait | Unsupported | Use methods |

## Type Annotations

### Built-in Types

| Python Type | Rust Type | Quality | Notes |
|-------------|-----------|---------|-------|
| `int` | `i64` | Full | Default integer |
| `float` | `f64` | Full | Default float |
| `str` | `String` | Full | UTF-8 strings |
| `bool` | `bool` | Full | Boolean |
| `None` | `Option::None` | Full | Null value |
| `Any` | Generic `T` | Partial | Type erasure limitations |

### Generic Types

| Python Type | Rust Type | Quality | Notes |
|-------------|-----------|---------|-------|
| `List[T]` | `Vec<T>` | Full | Homogeneous lists |
| `Tuple[T1, T2]` | `(T1, T2)` | Full | Fixed-size tuples |
| `Dict[K, V]` | `HashMap<K, V>` | Full | Hash-based maps |
| `Set[T]` | `HashSet<T>` | Full | Hash-based sets |
| `Optional[T]` | `Option<T>` | Full | Nullable types |
| `Union[T1, T2]` | `enum` | Partial | Tagged unions |
| `Callable[[Args], Ret]` | `Fn(Args) -> Ret` | Partial | Function types |

### Typing Module

| Python | Rust Equivalent | Quality | Notes |
|--------|-----------------|---------|-------|
| `List` | `Vec` | Full | |
| `Dict` | `HashMap` | Full | |
| `Set` | `HashSet` | Full | |
| `Optional` | `Option` | Full | |
| `Union` | `enum` | Partial | |
| `Literal` | `const` | Partial | Compile-time values |
| `TypeVar` | Generic `<T>` | Partial | Generic type parameters |
| `Protocol` | Trait | Partial | Structural typing |
| `NewType` | Type alias | Full | |

## Standard Library

### Commonly Supported Modules

| Module | Quality | Rust Equivalent | Notes |
|--------|---------|-----------------|-------|
| `math` | Full | `std::f64` | Math functions |
| `random` | Full | `rand` crate | Random numbers |
| `datetime` | Partial | `chrono` crate | Date/time handling |
| `json` | Full | `serde_json` | JSON serialization |
| `re` | Partial | `regex` crate | Regular expressions |
| `os.path` | Partial | `std::path` | Path manipulation |
| `collections` | Full | `std::collections` | Data structures |

### Limited Support

| Module | Quality | Workaround | Notes |
|--------|---------|------------|-------|
| `sys` | Partial | Platform-specific | System info limited |
| `io` | Partial | `std::io` | File I/O supported |
| `threading` | Unsupported | Use Tokio async | No GIL in Rust |
| `multiprocessing` | Unsupported | Use async/await | Process-based concurrency |
| `ctypes` | Unsupported | FFI directly | Use Rust FFI |
| `pickle` | Unsupported | Use `serde` | Serialization |

### Unsupported Modules

These modules cannot be directly translated:

- `tkinter` - GUI framework (use web UI)
- `asyncio` - Python-specific (use Tokio)
- `importlib` - Dynamic imports unsupported
- `inspect` - Reflection limited in Rust
- `gc` - Rust has different memory model

## Operators

### Arithmetic Operators

| Operator | Quality | Notes |
|----------|---------|-------|
| `+` | Full | Addition |
| `-` | Full | Subtraction |
| `*` | Full | Multiplication |
| `/` | Full | Float division |
| `//` | Full | Integer division |
| `%` | Full | Modulo |
| `**` | Full | Power (via `powi`/`powf`) |

### Comparison Operators

| Operator | Quality | Notes |
|----------|---------|-------|
| `==` | Full | Equality |
| `!=` | Full | Inequality |
| `<`, `>` | Full | Less/greater than |
| `<=`, `>=` | Full | Less/greater or equal |
| `is` | Partial | Pointer equality |
| `is not` | Partial | Pointer inequality |
| `in` | Partial | Collection membership |

### Logical Operators

| Operator | Quality | Notes |
|----------|---------|-------|
| `and` | Full | Logical AND |
| `or` | Full | Logical OR |
| `not` | Full | Logical NOT |

### Bitwise Operators

| Operator | Quality | Notes |
|----------|---------|-------|
| `&` | Full | Bitwise AND |
| `|` | Full | Bitwise OR |
| `^` | Full | Bitwise XOR |
| `~` | Full | Bitwise NOT |
| `<<` | Full | Left shift |
| `>>` | Full | Right shift |

## Comprehensions

| Feature | Quality | Rust Equivalent | Notes |
|---------|---------|-----------------|-------|
| List comprehension | Full | `iter().map().collect()` | `[x for x in items]` |
| Dict comprehension | Partial | `iter().collect()` | Simple cases only |
| Set comprehension | Partial | `iter().collect()` | Simple cases only |
| Generator expression | Partial | `Iterator` | Lazy evaluation |

**Example**:
```python
# Python
squares = [x**2 for x in range(10) if x % 2 == 0]

# Translated Rust
let squares: Vec<i64> = (0..10)
    .filter(|x| x % 2 == 0)
    .map(|x| x.pow(2))
    .collect();
```

## Exception Handling

| Feature | Quality | Rust Equivalent | Notes |
|---------|---------|-----------------|-------|
| `try/except` | Partial | `Result<T, E>` | Error handling |
| `finally` | Partial | Drop trait | Cleanup |
| `raise` | Partial | `Err(...)` | Error propagation |
| Custom exceptions | Partial | Custom error types | |
| Exception hierarchy | Unsupported | Flat error types | |

**Example**:
```python
# Python
try:
    result = risky_operation()
except ValueError as e:
    print(f"Error: {e}")
    result = None

# Translated Rust
let result = match risky_operation() {
    Ok(val) => Some(val),
    Err(e) => {
        println!("Error: {}", e);
        None
    }
};
```

## Async/Await

| Feature | Quality | Rust Equivalent | Notes |
|---------|---------|-----------------|-------|
| `async def` | Partial | `async fn` | Tokio runtime |
| `await` | Partial | `.await` | Async operations |
| `asyncio.gather` | Partial | `tokio::join!` | Parallel execution |
| `asyncio.create_task` | Partial | `tokio::spawn` | Task spawning |

## Known Limitations

### 1. Dynamic Typing
**Issue**: Python's dynamic nature doesn't map directly to Rust.

**Workaround**: Use comprehensive type hints.
```python
# Good - translates well
def add(a: int, b: int) -> int:
    return a + b

# Poor - translation may fail
def add(a, b):
    return a + b
```

### 2. Duck Typing
**Issue**: Structural typing is limited.

**Workaround**: Use explicit traits/protocols.

### 3. Monkey Patching
**Issue**: Runtime modification unsupported.

**Workaround**: Refactor to use composition.

### 4. `eval()` and `exec()`
**Issue**: Dynamic code execution impossible.

**Workaround**: Redesign using static approaches.

### 5. Multiple Inheritance
**Issue**: Rust supports single inheritance via traits.

**Workaround**: Use trait composition.

### 6. Global Interpreter Lock (GIL)
**Issue**: GIL semantics don't exist in Rust.

**Benefit**: True parallelism with threads!

## Workarounds for Unsupported Features

### Keyword Arguments (`**kwargs`)

```python
# Python - unsupported
def configure(**kwargs):
    pass

# Workaround - use struct
from dataclasses import dataclass

@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080

def configure(config: Config):
    pass
```

### Dynamic Attributes

```python
# Python - unsupported
obj.new_attr = "value"

# Workaround - use HashMap
from typing import Dict, Any

class DynamicObject:
    def __init__(self):
        self.attrs: Dict[str, Any] = {}

    def set_attr(self, key: str, value: Any):
        self.attrs[key] = value
```

### Metaclasses

```python
# Python - unsupported
class Meta(type):
    pass

# Workaround - use procedural macros in Rust
# Or refactor to avoid metaclasses
```

## Future Roadmap

Features planned for future releases:

### Short Term (3-6 months)
- [ ] Full `async/await` support
- [ ] Generator functions
- [ ] Context managers (`with` statement)
- [ ] More magic methods
- [ ] Enhanced type inference

### Medium Term (6-12 months)
- [ ] `dataclasses` full support
- [ ] `pathlib` integration
- [ ] More stdlib modules
- [ ] Protocol/structural typing
- [ ] Advanced comprehensions

### Long Term (12+ months)
- [ ] Partial `**kwargs` support
- [ ] Limited reflection
- [ ] Dynamic typing fallback
- [ ] CPython extension interop
- [ ] NumPy array translation

## Testing Compatibility

Use our compatibility checker:

```bash
portalis check-compat --input myfile.py

# Output:
# ✅ 45 features fully supported
# ⚠️  3 features partially supported
# ❌ 2 features unsupported
#
# Partial support:
#   - Line 23: Dict comprehension (simple patterns only)
#   - Line 45: Property decorator (basic support)
#   - Line 67: Multiple inheritance (use traits)
#
# Unsupported:
#   - Line 89: **kwargs (use struct)
#   - Line 102: eval() (refactor to static)
#
# Recommendation: 93% compatible - ready for translation
```

## Best Practices

### 1. Use Type Hints Everywhere
```python
# Good
def calculate(x: float, y: float) -> float:
    return x + y

# Avoid
def calculate(x, y):
    return x + y
```

### 2. Prefer Composition Over Inheritance
```python
# Good
class Logger:
    pass

class Service:
    def __init__(self, logger: Logger):
        self.logger = logger

# Avoid deep inheritance hierarchies
```

### 3. Use Standard Library
```python
# Good - translates to Rust easily
from dataclasses import dataclass
from typing import List

@dataclass
class Point:
    x: float
    y: float
```

### 4. Avoid Dynamic Features
```python
# Avoid
exec("x = 5")
getattr(obj, "dynamic_attr")
setattr(obj, "dynamic_attr", value)

# Prefer static access
x = 5
obj.static_attr
obj.static_attr = value
```

## See Also

- [Getting Started Guide](getting-started.md)
- [CLI Reference](cli-reference.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Architecture Overview](architecture.md)
