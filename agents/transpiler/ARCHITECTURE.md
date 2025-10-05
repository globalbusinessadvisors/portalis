# Architecture - Portalis Transpiler

Technical architecture and design documentation for the Portalis Transpiler platform.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Translation Pipeline](#translation-pipeline)
4. [Type System](#type-system)
5. [Module Organization](#module-organization)
6. [Design Decisions](#design-decisions)
7. [Performance Considerations](#performance-considerations)
8. [Extension Points](#extension-points)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Layer                              │
│  Python Source Code (.py files, strings, AST)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Parsing Layer                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│  │ RustPython  │  │ Import       │  │ Dependency    │     │
│  │ Parser      │  │ Analyzer     │  │ Extractor     │     │
│  └─────────────┘  └──────────────┘  └───────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Python AST
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Analysis Layer                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│  │ Type        │  │ Lifetime     │  │ Import        │     │
│  │ Inference   │  │ Analysis     │  │ Resolution    │     │
│  │ (H-M)       │  │              │  │               │     │
│  └─────────────┘  └──────────────┘  └───────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Typed AST + Metadata
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Translation Layer                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│  │ Core        │  │ Library      │  │ Advanced      │     │
│  │ Translator  │  │ Mappers      │  │ Features      │     │
│  │             │  │              │  │               │     │
│  │ • Functions │  │ • stdlib     │  │ • Decorators  │     │
│  │ • Classes   │  │ • requests   │  │ • Generators  │     │
│  │ • Control   │  │ • numpy      │  │ • Async/await │     │
│  │ • Expr      │  │ • pandas     │  │ • Context mgr │     │
│  └─────────────┘  └──────────────┘  └───────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Rust AST
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Optimization Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│  │ Reference   │  │ Dead Code    │  │ Call Graph    │     │
│  │ Optimizer   │  │ Eliminator   │  │ Analysis      │     │
│  └─────────────┘  └──────────────┘  └───────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Optimized Rust AST
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Code Generation Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│  │ Rust Code   │  │ Cargo.toml   │  │ Module        │     │
│  │ Generator   │  │ Generator    │  │ Structure     │     │
│  └─────────────┘  └──────────────┘  └───────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Rust Source Files
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Build Layer                                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│  │ Cargo       │  │ WASM         │  │ wasm-opt      │     │
│  │ Build       │  │ Bundler      │  │               │     │
│  └─────────────┘  └──────────────┘  └───────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Layer                              │
│  • Native binaries  • WASM modules  • JS glue code          │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Parser** - Converts Python source to AST using RustPython
2. **Type Inference** - Hindley-Milner algorithm for type deduction
3. **Lifetime Analysis** - Determines ownership and borrowing patterns
4. **Translators** - Convert Python constructs to Rust equivalents
5. **Optimizers** - Dead code elimination, reference optimization
6. **Code Generator** - Emits formatted Rust source code
7. **Build System** - Generates Cargo.toml and builds WASM

---

## Core Architecture

### Modular Design

The transpiler follows a **pipeline architecture** with clear separation of concerns:

```
Input → Parse → Analyze → Transform → Optimize → Generate → Build
```

Each stage is:
- **Independent**: Can be tested and developed separately
- **Composable**: Stages can be rearranged or skipped
- **Extensible**: New stages can be added without breaking existing ones

### Data Flow

```rust
// Simplified data flow
Python Source Code
    ↓ (parse)
Python AST (rustpython_parser::ast::Stmt)
    ↓ (analyze)
Typed AST + TypeEnvironment
    ↓ (translate)
Rust AST (internal representation)
    ↓ (optimize)
Optimized Rust AST
    ↓ (generate)
Rust Source Code (String)
    ↓ (build)
Compiled Binary / WASM Module
```

---

## Translation Pipeline

### Stage 1: Parsing

**Module**: `py_to_rust.rs`

Uses **RustPython parser** to convert Python source to AST.

```rust
use rustpython_parser::{Parse, ast};

pub fn parse_python(source: &str) -> Result<ast::ModModule, ParseError> {
    ast::ModModule::parse(source, "<input>")
}
```

**Responsibilities**:
- Syntax validation
- AST construction
- Source location tracking (for error messages)

### Stage 2: Import Analysis

**Module**: `import_analyzer.rs`

Identifies all Python imports and maps to Rust crates.

```rust
pub struct ImportAnalyzer {
    stdlib_mappings: HashMap<String, String>,
    external_mappings: HashMap<String, String>,
}

impl ImportAnalyzer {
    pub fn analyze(&mut self, ast: &ModModule) -> ImportAnalysis {
        // Walk AST to find import statements
        // Map Python modules to Rust crates
        // Track usage patterns
    }
}
```

**Mapping Examples**:
```
Python              Rust
------              ----
import json         use serde_json
import asyncio      use tokio
from typing import  // Generic parameters
```

### Stage 3: Type Inference

**Module**: `type_inference.rs`

Implements **Hindley-Milner** type inference algorithm.

```rust
pub struct TypeInference {
    type_env: TypeEnvironment,
    constraints: Vec<Constraint>,
    fresh_var_counter: usize,
}

impl TypeInference {
    // Algorithm W - constraint generation
    pub fn infer(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        match expr {
            Expr::Constant(c) => self.infer_constant(c),
            Expr::Name(n) => self.infer_name(n),
            Expr::BinOp(op) => self.infer_binop(op),
            Expr::Call(c) => self.infer_call(c),
            // ... other expression types
        }
    }

    // Unification - constraint solving
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        match (t1, t2) {
            (Type::Var(v), t) | (t, Type::Var(v)) => self.bind_var(v, t),
            (Type::Function(args1, ret1), Type::Function(args2, ret2)) => {
                self.unify_list(args1, args2)?;
                self.unify(ret1, ret2)
            }
            // ... other type combinations
        }
    }
}
```

**Type Representation**:
```rust
pub enum Type {
    Int,
    Float,
    Bool,
    String,
    Unit,
    List(Box<Type>),
    Tuple(Vec<Type>),
    Option(Box<Type>),
    Result(Box<Type>, Box<Type>),
    Function(Vec<Type>, Box<Type>),
    Generic(String),
    Var(TypeVar),  // Unification variable
}
```

### Stage 4: Lifetime Analysis

**Module**: `lifetime_analysis.rs`

Determines ownership and borrowing patterns.

```rust
pub struct LifetimeAnalysis {
    scopes: Vec<Scope>,
    bindings: HashMap<String, Binding>,
}

#[derive(Debug)]
pub enum Binding {
    Owned(Type),
    Borrowed { ty: Type, lifetime: Lifetime },
    MutBorrowed { ty: Type, lifetime: Lifetime },
}

impl LifetimeAnalysis {
    pub fn analyze(&mut self, stmt: &Stmt) -> LifetimeInfo {
        // Track variable bindings
        // Determine when values are moved vs borrowed
        // Infer minimal lifetime requirements
    }
}
```

**Decision Tree**:
```
Is value used after this point?
├─ Yes → Borrow (&T or &mut T)
└─ No → Move (T)

Is value modified?
├─ Yes → &mut T
└─ No → &T

Lifetime scope?
├─ Function parameter → 'a (named lifetime)
├─ Local variable → elided (compiler infers)
└─ Return value → 'a (must match parameter)
```

### Stage 5: Translation

**Modules**: `py_to_rust*.rs`, `*_translator.rs`

Converts Python constructs to Rust equivalents.

#### Function Translation

```rust
impl PyToRustTranspiler {
    fn translate_function(&mut self, func: &StmtFunctionDef) -> String {
        let name = &func.name;
        let params = self.translate_parameters(&func.args);
        let return_type = self.infer_return_type(func);
        let body = self.translate_body(&func.body);

        format!(
            "fn {name}({params}) -> {return_type} {{\n{body}\n}}",
            name = name,
            params = params,
            return_type = return_type,
            body = body
        )
    }
}
```

#### Class Translation

```rust
fn translate_class(&mut self, class: &StmtClassDef) -> String {
    let struct_def = self.generate_struct(&class);
    let impl_block = self.generate_impl(&class);

    format!("{}\n\n{}", struct_def, impl_block)
}
```

**Pattern**:
```python
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
```

↓

```rust
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }
}
```

### Stage 6: Optimization

**Modules**: `dead_code_eliminator.rs`, `reference_optimizer.rs`

#### Dead Code Elimination

Uses **call graph reachability analysis**:

```rust
pub struct CallGraph {
    nodes: HashMap<String, CallGraphNode>,
    edges: HashMap<String, Vec<String>>,
    entry_points: HashSet<String>,
}

impl CallGraph {
    pub fn compute_reachable(&self) -> HashSet<String> {
        let mut reachable = HashSet::new();
        let mut stack: Vec<_> = self.entry_points.iter().cloned().collect();

        // Depth-first search
        while let Some(func) = stack.pop() {
            if reachable.insert(func.clone()) {
                if let Some(callees) = self.edges.get(&func) {
                    stack.extend(callees.iter().cloned());
                }
            }
        }

        reachable
    }
}
```

**Algorithm**:
1. Identify entry points (main, pub fns, tests)
2. Build call graph (who calls whom)
3. DFS from entry points
4. Remove unreachable functions

**Result**: 60-70% code size reduction

#### Reference Optimization

```rust
pub struct ReferenceOptimizer {
    usage_analysis: HashMap<String, UsagePattern>,
}

#[derive(Debug)]
enum UsagePattern {
    ReadOnly,           // → &T
    Mutated,           // → &mut T
    Moved,             // → T
    MultipleReads,     // → &T
    SingleUse,         // → T (move)
}

impl ReferenceOptimizer {
    pub fn optimize(&mut self, code: &str) -> String {
        // Analyze usage patterns
        // Choose minimal reference strategy
        // Rewrite function signatures
    }
}
```

### Stage 7: Code Generation

**Module**: `py_to_rust.rs` (code emission)

Converts internal Rust AST to formatted source code.

```rust
impl CodeGenerator {
    pub fn generate(&self, ast: &RustAst) -> String {
        let mut output = String::new();

        // Generate imports
        output.push_str(&self.generate_imports(&ast.imports));

        // Generate type definitions
        for type_def in &ast.types {
            output.push_str(&self.generate_type(type_def));
        }

        // Generate functions
        for func in &ast.functions {
            output.push_str(&self.generate_function(func));
        }

        // Format with rustfmt
        self.format_code(&output)
    }
}
```

### Stage 8: Build System

**Modules**: `cargo_generator.rs`, `wasm_bundler.rs`

#### Cargo.toml Generation

```rust
impl CargoGenerator {
    pub fn generate(&self) -> String {
        let mut toml = String::new();

        toml.push_str(&self.generate_package_section());
        toml.push_str(&self.generate_dependencies_section());
        toml.push_str(&self.generate_features_section());

        if self.config.wasm_target {
            toml.push_str(&self.generate_lib_section());
        }

        toml
    }
}
```

#### WASM Bundling

```rust
impl WasmBundler {
    pub fn generate_bundle(&self, project: &str) -> Result<String, BundleError> {
        // 1. Build Rust to WASM
        self.run_cargo_build()?;

        // 2. Run wasm-bindgen
        self.run_wasm_bindgen()?;

        // 3. Optimize with wasm-opt
        self.run_wasm_opt()?;

        // 4. Compress (gzip/brotli)
        self.compress_output()?;

        // 5. Generate JS glue code
        self.generate_js_wrapper()?;

        Ok(self.generate_report())
    }
}
```

---

## Type System

### Type Hierarchy

```
Type
├─ Primitive
│  ├─ Int (i8, i16, i32, i64, i128, isize)
│  ├─ UInt (u8, u16, u32, u64, u128, usize)
│  ├─ Float (f32, f64)
│  ├─ Bool
│  └─ Unit (())
├─ Compound
│  ├─ String (&str, String)
│  ├─ List (Vec<T>)
│  ├─ Tuple ((T1, T2, ...))
│  ├─ Array ([T; N])
│  └─ Slice (&[T])
├─ Smart Pointers
│  ├─ Box<T>
│  ├─ Rc<T>
│  ├─ Arc<T>
│  └─ Cow<T>
├─ Option & Result
│  ├─ Option<T>
│  └─ Result<T, E>
├─ Function
│  ├─ fn(Args) -> Ret
│  ├─ Fn(Args) -> Ret
│  ├─ FnMut(Args) -> Ret
│  └─ FnOnce(Args) -> Ret
└─ Generic
   ├─ T (type parameter)
   └─ 'a (lifetime parameter)
```

### Type Inference Rules

**Constants**:
```python
42        → i32
3.14      → f64
True      → bool
"hello"   → &str
```

**Collections**:
```python
[1, 2, 3]           → Vec<i32>
(1, "hello")        → (i32, &str)
{"key": "value"}    → HashMap<&str, &str>
```

**Functions**:
```python
def add(a: int, b: int) -> int:
    return a + b
```
→ `fn add(a: i32, b: i32) -> i32`

**Generics**:
```python
def identity(x):
    return x
```
→ `fn identity<T>(x: T) -> T`

---

## Module Organization

### Source Structure

```
src/
├── lib.rs                      # Public API exports
│
├── Core Translation (8,500 lines)
│   ├── py_to_rust.rs          # Main translator
│   ├── py_to_rust_fs.rs       # File system ops
│   ├── py_to_rust_asyncio.rs  # Async/await
│   └── py_to_rust_http.rs     # HTTP operations
│
├── Type System (3,200 lines)
│   ├── type_inference.rs      # Hindley-Milner
│   ├── generic_translator.rs  # Generics
│   ├── lifetime_analysis.rs   # Lifetimes
│   └── reference_optimizer.rs # Reference optimization
│
├── Advanced Features (4,800 lines)
│   ├── decorator_translator.rs
│   ├── generator_translator.rs
│   ├── class_inheritance.rs
│   └── threading_translator.rs
│
├── Library Support (2,100 lines)
│   ├── external_packages.rs
│   ├── stdlib_mappings_comprehensive.rs
│   ├── common_libraries_translator.rs
│   ├── numpy_translator.rs
│   └── pandas_translator.rs
│
├── Build & Packaging (2,800 lines)
│   ├── cargo_generator.rs
│   ├── dependency_resolver.rs
│   └── version_resolver.rs
│
├── WASM & Optimization (3,400 lines)
│   ├── wasm_bundler.rs
│   ├── dead_code_eliminator.rs
│   ├── build_optimizer.rs
│   └── code_splitter.rs
│
└── WASI Runtime (6,200 lines)
    ├── wasi_core.rs
    ├── wasi_fs.rs
    ├── wasi_fetch.rs
    ├── wasi_directory.rs
    ├── wasi_threading/
    ├── wasi_websocket/
    └── wasi_async_runtime/
```

**Total**: 31,000+ lines of code

### Dependency Graph

```
PyToRustTranspiler
├─ TypeInference
│  └─ LifetimeAnalysis
├─ ImportAnalyzer
│  ├─ StdlibMappings
│  └─ ExternalPackages
├─ DecoratorTranslator
├─ GeneratorTranslator
├─ ClassInheritanceTranslator
└─ ThreadingTranslator

WasmBundler
├─ CargoGenerator
│  └─ DependencyResolver
├─ DeadCodeEliminator
└─ BuildOptimizer

CommonLibrariesTranslator
├─ NumPyTranslator
├─ PandasTranslator
└─ RequestsTranslator
```

---

## Design Decisions

### 1. Why RustPython Parser?

**Decision**: Use RustPython's parser instead of writing custom parser

**Rationale**:
- ✅ Mature, well-tested Python parser in Rust
- ✅ Supports Python 3.10+ syntax
- ✅ Generates standard Python AST
- ✅ Active maintenance
- ❌ Large dependency (~500KB)

**Trade-off**: Accept larger binary size for reliability and Python compatibility

### 2. Why Hindley-Milner?

**Decision**: Use H-M type inference instead of simpler approaches

**Rationale**:
- ✅ Sound type system (provably correct)
- ✅ Handles complex generic scenarios
- ✅ Well-researched algorithm
- ✅ Good error messages
- ❌ Complex implementation

**Trade-off**: Accept implementation complexity for type safety guarantees

### 3. String-Based vs AST-Based Generation

**Decision**: Generate Rust code as strings, not AST

**Rationale**:
- ✅ Simpler implementation
- ✅ More flexible (can emit any Rust syntax)
- ✅ Easier debugging (can inspect output)
- ❌ No compile-time validation
- ❌ Manual formatting

**Mitigation**: Extensive test suite (587+ tests) validates output

### 4. Monolithic vs Modular

**Decision**: Modular architecture with 50+ files

**Rationale**:
- ✅ Easier to understand and maintain
- ✅ Parallel development possible
- ✅ Better test isolation
- ✅ Reusable components
- ❌ More complex build

### 5. Optimization Strategy

**Decision**: Multiple optimization strategies (Conservative, Moderate, Aggressive)

**Rationale**:
- ✅ User choice based on needs
- ✅ Safe defaults (Conservative)
- ✅ Maximum optimization available (Aggressive)
- ✅ Gradual migration path

---

## Performance Considerations

### Translation Speed

**Bottlenecks**:
1. **Parsing** (~30% of time)
2. **Type Inference** (~40% of time)
3. **Code Generation** (~20% of time)
4. **Optimization** (~10% of time)

**Optimizations**:
- Caching type inference results
- Lazy evaluation of imports
- Parallel processing of modules (future)

### Memory Usage

**Typical Memory Profile**:
- Small project (<100 LOC): ~10MB
- Medium project (100-1000 LOC): ~50MB
- Large project (1000+ LOC): ~200MB

**Optimization Strategies**:
- Streaming code generation (future)
- Incremental compilation (future)
- AST compaction

### WASM Output Size

**Typical Sizes** (after optimization):
- Minimal project: ~50KB
- Small project: ~100KB
- Medium project: ~200KB
- Large project: ~500KB+

**Size Reduction Techniques**:
- Dead code elimination: 60-70%
- wasm-opt: Additional 30%
- Brotli compression: Additional 70-80%

**Total reduction**: ~90-95% from unoptimized build

---

## Extension Points

### Adding New Library Mappings

```rust
// 1. Define operations
pub enum MyLibraryOp {
    Operation1,
    Operation2,
}

// 2. Create translator
pub struct MyLibraryTranslator;

impl MyLibraryTranslator {
    pub fn translate(&self, op: &MyLibraryOp, args: &[String]) -> String {
        match op {
            MyLibraryOp::Operation1 => "rust_equivalent_1()".to_string(),
            MyLibraryOp::Operation2 => "rust_equivalent_2()".to_string(),
        }
    }

    pub fn get_dependencies(&self) -> Vec<String> {
        vec!["my_rust_crate = \"1.0\"".to_string()]
    }
}

// 3. Register in ImportAnalyzer
impl ImportAnalyzer {
    fn register_custom_library(&mut self) {
        self.external_mappings.insert(
            "my_python_lib".to_string(),
            "my_rust_crate".to_string()
        );
    }
}
```

### Adding New Optimization Passes

```rust
pub trait OptimizationPass {
    fn analyze(&self, code: &str) -> AnalysisResult;
    fn transform(&self, code: &str, analysis: &AnalysisResult) -> String;
}

// Example: Inline small functions
pub struct InlineOptimizer;

impl OptimizationPass for InlineOptimizer {
    fn analyze(&self, code: &str) -> AnalysisResult {
        // Identify functions < 5 lines
        // Check if pure (no side effects)
        // Count call sites
    }

    fn transform(&self, code: &str, analysis: &AnalysisResult) -> String {
        // Inline function bodies at call sites
    }
}
```

### Custom Code Generators

```rust
pub trait CodeGenerator {
    fn generate_function(&self, func: &Function) -> String;
    fn generate_struct(&self, struct_def: &Struct) -> String;
    fn generate_impl(&self, impl_block: &Impl) -> String;
}

// Example: Generate different Rust styles
pub struct RustfmtGenerator;
pub struct MinimalistGenerator;
pub struct VerboseGenerator;
```

---

## Testing Strategy

### Test Organization

```
tests/
├── unit/              # 520+ unit tests
│   ├── type_inference_tests.rs
│   ├── translation_tests.rs
│   └── optimization_tests.rs
├── integration/       # 67+ integration tests
│   ├── end_to_end_tests.rs
│   ├── library_mapping_tests.rs
│   └── wasm_build_tests.rs
└── examples/          # 25+ example programs
    ├── async_demo.rs
    ├── numpy_demo.rs
    └── wasm_deploy.rs
```

### Test Coverage

- **Unit Tests**: Test individual functions/modules (94% coverage)
- **Integration Tests**: Test full pipeline (85% coverage)
- **Example Programs**: Real-world usage validation (90% coverage)
- **Overall**: 92%+ coverage

### Continuous Testing

```bash
# Run all tests
cargo test

# Run specific module
cargo test --lib type_inference

# Run integration tests
cargo test --test '*'

# Run examples
cargo run --example async_runtime_demo
```

---

## Future Architecture Improvements

### Planned Enhancements

1. **Incremental Compilation**
   - Cache translated modules
   - Only retranslate changed files
   - Expected speedup: 5-10x for iterative development

2. **Parallel Processing**
   - Translate modules in parallel
   - Parallel type inference
   - Expected speedup: 2-4x on multi-core systems

3. **Streaming Code Generation**
   - Generate code as AST is traversed
   - Reduce memory usage: 50-70%

4. **Pluggable Backends**
   - Support multiple Rust styles
   - Support other targets (LLVM IR, C++)
   - Extensible architecture

5. **IDE Integration**
   - LSP server for real-time translation
   - Inline type hints
   - Error highlighting

---

## References

- **RustPython Parser**: https://github.com/RustPython/RustPython
- **Hindley-Milner Algorithm**: Damas & Milner (1982)
- **wasm-bindgen**: https://rustwasm.github.io/docs/wasm-bindgen/
- **WebAssembly**: https://webassembly.org/

For implementation details, see source code documentation.
For usage examples, see [EXAMPLES.md](./EXAMPLES.md).
