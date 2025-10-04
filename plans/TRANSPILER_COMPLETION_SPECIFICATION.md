# PORTALIS Transpiler Completion Specification

**Version**: 1.0
**Date**: 2025-10-04
**Status**: Planning Document
**Purpose**: Comprehensive roadmap to achieve full "any Python library/script to Rust to WASM" capability

---

## Executive Summary

### Current State
PORTALIS has successfully built a **production-ready MVP transpiler** with:
- **87.2% test pass rate** (191/219 tests passing)
- **150+ Python features** implemented (~28.5% of full Python spec)
- **Full WASM pipeline** working (Python → Rust → WASM)
- **Pattern-based code generation** for 15+ hardcoded functions
- **Basic async support** via py_to_rust_asyncio.rs
- **WASI core filesystem** implementation complete
- **External package registry** with 200+ PyPI package mappings

### Vision
Transform PORTALIS into a **comprehensive, production-grade transpiler** capable of:
- Parsing **any valid Python script** (full AST support)
- Translating **all 527 Python language features**
- Handling **real-world libraries** (NumPy, Pandas, Flask, Django, etc.)
- Generating **optimized, idiomatic Rust code**
- Producing **production-ready WASM modules**
- Supporting **both browser and server-side** WASM deployments

### Strategic Approach
Rather than attempting to implement all features immediately, this specification defines a **phased, incremental approach**:
1. **Complete the foundation** (real AST parser, generic code generator)
2. **Build core infrastructure** (WASI modules, async runtime)
3. **Expand language coverage** (remaining Python features)
4. **Enhance package ecosystem** (external dependencies)
5. **Production hardening** (optimization, testing, deployment)

### Timeline Estimate
**Total Duration**: 16-24 weeks (4-6 months)
- **Phase 1**: Foundation (4 weeks)
- **Phase 2**: Core Infrastructure (4 weeks)
- **Phase 3**: Language Coverage (6 weeks)
- **Phase 4**: Package Ecosystem (3 weeks)
- **Phase 5**: Production Hardening (3 weeks)

**Resource Estimate**: 2-3 senior Rust engineers working full-time

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    PORTALIS Transpiler                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Python AST Parser (rustpython-parser integration)       │
│     • Full Python 3.x syntax support                         │
│     • Source location tracking                               │
│     • Error recovery and reporting                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Semantic Analysis Layer                                  │
│     • Type inference engine                                  │
│     • Symbol table construction                              │
│     • Import resolution                                      │
│     • Control flow analysis                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Translation Engine                                       │
│     • Expression translator                                  │
│     • Statement translator                                   │
│     • Type mapper (Python → Rust)                            │
│     • Pattern matcher (idioms)                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Code Generator                                           │
│     • Rust AST construction                                  │
│     • Lifetime inference                                     │
│     • Borrow checker compliance                              │
│     • Optimization passes                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. WASI Runtime Layer                                       │
│     • wasi_fs (filesystem)                                   │
│     • wasi_threading (threads/workers)                       │
│     • wasi_websocket (networking)                            │
│     • wasi_async_runtime (async/await)                       │
│     • wasi_fetch (HTTP client)                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Package Resolution                                       │
│     • External package mapper (numpy→ndarray)                │
│     • Dependency graph builder                               │
│     • Version compatibility checker                          │
│     • Cargo.toml generator                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  7. WASM Pipeline                                            │
│     • wasm-pack integration                                  │
│     • wasm-bindgen code generation                           │
│     • Optimization (wasm-opt)                                │
│     • Bundling and deployment                                │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Python Source (.py)
    │
    ├─► [Parser] → Python AST
    │                  │
    │                  ├─► [Semantic Analysis] → Typed AST
    │                  │                            │
    │                  │                            ├─► [Type Inference]
    │                  │                            ├─► [Symbol Tables]
    │                  │                            └─► [Import Resolution]
    │                  │
    │                  └─► [Translator] → Rust AST
    │                                        │
    │                                        ├─► [Expression Translation]
    │                                        ├─► [Statement Translation]
    │                                        └─► [Pattern Matching]
    │
    └─► [Code Generator] → Rust Source (.rs)
                                │
                                ├─► [Optimization Passes]
                                └─► [Formatting]
                                      │
                                      ├─► Cargo.toml (generated)
                                      └─► .rs files
                                            │
                                            └─► [wasm-pack] → WASM Module (.wasm)
                                                                    │
                                                                    ├─► Browser
                                                                    ├─► Node.js
                                                                    └─► Server (WASI)
```

---

## Phase-by-Phase Implementation Plan

### Phase 1: Foundation (Weeks 1-4)

**Goal**: Replace basic parser with full Python AST support and build generic code generation framework.

#### 1.1 Real Python AST Parser Integration (Week 1)
**Objective**: Integrate rustpython-parser for complete Python 3.x syntax support.

**Current State**:
- `simple_parser.rs` (302 lines) - handles only basic literals and assignments
- `indented_parser.rs` (~900 lines) - improved but still limited
- No source location tracking
- Limited error recovery

**Target State**:
- Full Python 3.10+ AST support via rustpython-parser
- Accurate source location tracking (line/column)
- Comprehensive error reporting with context
- Support for all Python constructs (modules, functions, classes, etc.)

**Implementation Approach**:
```rust
// New parser module structure
mod python_parser {
    use rustpython_parser::{ast, Parse};

    pub struct PythonParser {
        source: String,
        filename: String,
    }

    impl PythonParser {
        pub fn parse(&self) -> Result<ast::Mod, ParseError> {
            // Use rustpython_parser::parse_program
            ast::Suite::parse(&self.source, &self.filename)
                .map_err(|e| ParseError::from_rustpython(e))
        }

        pub fn parse_expression(&self) -> Result<ast::Expr, ParseError> {
            ast::Expression::parse(&self.source, &self.filename)
                .map_err(|e| ParseError::from_rustpython(e))
        }
    }
}
```

**Required Dependencies**:
```toml
[dependencies]
rustpython-parser = "0.3"  # Already in Cargo.toml
rustpython-ast = "0.3"
```

**Deliverables**:
- [ ] `src/python_parser.rs` - rustpython-parser wrapper
- [ ] `src/ast_converter.rs` - Convert rustpython AST → internal AST
- [ ] `src/parse_error.rs` - Rich error type with source context
- [ ] Unit tests for all Python syntax constructs
- [ ] Integration tests with real Python files

**Success Criteria**:
- Parse any valid Python 3.x file without errors
- Provide line/column information for all AST nodes
- Handle syntax errors with helpful messages
- 95%+ test coverage for parser module

**Complexity**: Medium (10-15 person-days)

---

#### 1.2 Generic Expression Translator (Week 2)
**Objective**: Build a comprehensive expression translator that goes beyond 15 hardcoded patterns.

**Current State**:
- `code_generator.rs` - 15 hardcoded function patterns (fibonacci, factorial, etc.)
- `python_to_rust.rs` - Some expression translation but incomplete
- No generic strategy for arbitrary expressions

**Target State**:
- Translate **any** Python expression to Rust
- Handle nested expressions recursively
- Support all operators (arithmetic, logical, comparison, bitwise)
- Handle method calls, attribute access, indexing, slicing
- Type-aware translation with proper casting

**Implementation Approach**:
```rust
pub struct ExpressionTranslator {
    type_ctx: TypeContext,
    temp_counter: usize,
}

impl ExpressionTranslator {
    pub fn translate_expr(&mut self, expr: &PyExpr) -> Result<RustExpr> {
        match expr {
            // Literals
            PyExpr::Constant(value) => self.translate_constant(value),

            // Binary operations
            PyExpr::BinOp { left, op, right } => {
                let left_rust = self.translate_expr(left)?;
                let right_rust = self.translate_expr(right)?;
                self.translate_binop(left_rust, op, right_rust)
            }

            // Function calls
            PyExpr::Call { func, args, keywords } => {
                self.translate_call(func, args, keywords)
            }

            // Attribute access (obj.attr)
            PyExpr::Attribute { value, attr, ctx } => {
                self.translate_attribute(value, attr)
            }

            // Subscript (obj[index])
            PyExpr::Subscript { value, slice, ctx } => {
                self.translate_subscript(value, slice)
            }

            // List/Dict/Set comprehensions
            PyExpr::ListComp { elt, generators } => {
                self.translate_list_comp(elt, generators)
            }

            // Lambda expressions
            PyExpr::Lambda { args, body } => {
                self.translate_lambda(args, body)
            }

            // And many more...
            _ => Err(TranslationError::UnsupportedExpression(expr.clone()))
        }
    }

    fn translate_binop(&mut self, left: RustExpr, op: &Operator, right: RustExpr)
        -> Result<RustExpr> {
        // Type-aware operator translation
        let result_type = self.infer_binop_type(&left, op, &right)?;

        match op {
            Operator::Add => {
                // String concatenation vs numeric addition
                if result_type.is_string() {
                    RustExpr::MethodCall {
                        receiver: left,
                        method: "push_str".to_string(),
                        args: vec![right],
                    }
                } else {
                    RustExpr::Binary {
                        left: Box::new(left),
                        op: RustOp::Add,
                        right: Box::new(right),
                    }
                }
            }
            // ... other operators
        }
    }
}
```

**Translation Strategies**:

1. **Arithmetic Operations**:
   - `a + b` → `a + b` (for integers/floats)
   - `a + b` → `format!("{}{}", a, b)` (for strings)
   - `a / b` → `a / b` (always float in Python 3)
   - `a // b` → `(a / b).floor()` (floor division)
   - `a ** b` → `a.powf(b)` (exponentiation)

2. **Comparison Operations**:
   - `a == b` → `a == b`
   - `a is b` → `std::ptr::eq(&a, &b)` (identity)
   - `a in b` → `b.contains(&a)` (membership)

3. **Logical Operations**:
   - `a and b` → `a && b` (but with short-circuit evaluation)
   - `a or b` → `a || b`
   - `not a` → `!a`

4. **Method Calls**:
   - `obj.method(args)` → Translate based on type of `obj`
   - String methods: `s.upper()` → `s.to_uppercase()`
   - List methods: `lst.append(x)` → `lst.push(x)`
   - Dict methods: `d.get(k)` → `d.get(&k)`

5. **Complex Expressions**:
   - List comprehensions: `[x*2 for x in range(10)]` → `(0..10).map(|x| x*2).collect::<Vec<_>>()`
   - Generator expressions: `(x*2 for x in range(10))` → `(0..10).map(|x| x*2)`
   - Lambda: `lambda x: x+1` → `|x| x+1`

**Deliverables**:
- [ ] `src/expr_translator.rs` - Generic expression translator
- [ ] `src/operator_mapper.rs` - Python → Rust operator mapping
- [ ] `src/method_mapper.rs` - Method translation rules
- [ ] Comprehensive test suite (500+ expression tests)

**Success Criteria**:
- Translate 95%+ of Python expressions correctly
- Handle deeply nested expressions (10+ levels)
- Proper type inference and casting
- Generate idiomatic Rust code

**Complexity**: High (15-20 person-days)

---

#### 1.3 Generic Statement Translator (Week 3)
**Objective**: Translate all Python statement types to Rust.

**Current State**:
- `python_to_rust.rs` - Basic statement translation (if, while, for, etc.)
- Limited control flow support
- No advanced features (with, async, decorators)

**Target State**:
- Translate **all** Python statement types
- Handle complex control flow (nested loops, break/continue, else clauses)
- Support context managers (with statements)
- Handle async/await properly
- Implement decorator translation

**Statement Coverage**:

| Python Statement | Rust Translation | Status | Priority |
|------------------|------------------|--------|----------|
| Assignment | `let x = value;` | ✅ Complete | - |
| Augmented assignment | `x += value;` | ✅ Complete | - |
| If/elif/else | `if {} else {}` | ✅ Complete | - |
| While loop | `while {}` | ✅ Complete | - |
| For loop | `for x in iter {}` | ✅ Complete | - |
| Break/continue | `break;` / `continue;` | ✅ Complete | - |
| Return | `return value;` | ✅ Complete | - |
| Try/except/finally | Error handling | ⚠️ Partial | HIGH |
| With statement | RAII pattern | ❌ Missing | HIGH |
| Async def | `async fn {}` | ⚠️ Partial | HIGH |
| Await expression | `.await` | ⚠️ Partial | HIGH |
| Yield expression | Generator pattern | ❌ Missing | MEDIUM |
| Class definition | `struct` + `impl` | ⚠️ Partial | HIGH |
| Decorator | Attribute/macro | ❌ Missing | MEDIUM |
| Global/nonlocal | Variable scoping | ❌ Missing | LOW |
| Import statements | Module system | ⚠️ Partial | HIGH |
| Assert | `assert!()` | ✅ Complete | - |
| Pass | `// pass` | ✅ Complete | - |
| Raise | `panic!()` / `Err()` | ⚠️ Partial | MEDIUM |
| Delete | Not applicable | ❌ Missing | LOW |

**Implementation Strategy**:

```rust
pub struct StatementTranslator {
    expr_translator: ExpressionTranslator,
    scope_stack: Vec<Scope>,
    loop_depth: usize,
    async_context: bool,
}

impl StatementTranslator {
    pub fn translate_stmt(&mut self, stmt: &PyStmt) -> Result<RustStmt> {
        match stmt {
            // Function definition
            PyStmt::FunctionDef {
                name, args, body, decorator_list, returns, is_async
            } => {
                self.translate_function(name, args, body, decorator_list, returns, *is_async)
            }

            // Class definition
            PyStmt::ClassDef { name, bases, body, decorator_list } => {
                self.translate_class(name, bases, body, decorator_list)
            }

            // With statement (context manager)
            PyStmt::With { items, body } => {
                self.translate_with(items, body)
            }

            // Import statements
            PyStmt::Import { names } => {
                self.translate_import(names)
            }

            PyStmt::ImportFrom { module, names, level } => {
                self.translate_import_from(module, names, *level)
            }

            // Try/except/finally
            PyStmt::Try { body, handlers, orelse, finalbody } => {
                self.translate_try(body, handlers, orelse, finalbody)
            }

            // ... more statements
        }
    }

    fn translate_with(&mut self, items: &[WithItem], body: &[PyStmt])
        -> Result<RustStmt> {
        // Translate context manager to RAII pattern
        // Python: with open("file.txt") as f: ...
        // Rust: { let mut f = File::open("file.txt")?; ... } // f auto-closes

        let mut rust_stmts = Vec::new();

        for item in items {
            let context_expr = self.expr_translator.translate_expr(&item.context_expr)?;
            let var_name = item.optional_vars.as_ref()
                .ok_or(TranslationError::ContextManagerNeedsVar)?;

            rust_stmts.push(RustStmt::Let {
                pattern: RustPattern::Ident(var_name.clone()),
                ty: None, // Infer
                init: Some(context_expr),
                mutable: true,
            });
        }

        // Translate body
        for stmt in body {
            rust_stmts.push(self.translate_stmt(stmt)?);
        }

        Ok(RustStmt::Block {
            stmts: rust_stmts,
            // Variables will be dropped at end of block (RAII)
        })
    }
}
```

**Deliverables**:
- [ ] `src/stmt_translator.rs` - Complete statement translator
- [ ] `src/context_manager.rs` - With statement support
- [ ] `src/decorator_translator.rs` - Decorator handling
- [ ] `src/exception_handler.rs` - Try/except translation
- [ ] Test suite for all statement types

**Success Criteria**:
- Translate 95%+ of Python statements correctly
- Proper scoping and variable lifetime
- Idiomatic Rust patterns (RAII, Result types, etc.)
- Full test coverage

**Complexity**: High (15-20 person-days)

---

#### 1.4 Advanced Type Inference (Week 4)
**Objective**: Build a sophisticated type inference engine beyond primitives.

**Current State**:
- `python_to_rust.rs` - Basic type inference (i32, String, bool, f64)
- Limited support for collections (Vec, HashMap)
- No generics, traits, or lifetimes

**Target State**:
- Infer complex types (generics, traits, lifetimes)
- Handle collection types properly (Vec<T>, HashMap<K,V>, Option<T>, Result<T,E>)
- Infer function signatures
- Support trait bounds
- Lifetime inference for references

**Type Inference Algorithm**:

```rust
pub struct TypeInferenceEngine {
    // Type environment: variable → type
    type_env: HashMap<String, RustType>,

    // Constraints collected during inference
    constraints: Vec<TypeConstraint>,

    // Generic type variables
    type_vars: HashMap<String, TypeVar>,
}

#[derive(Debug, Clone)]
pub enum RustType {
    // Primitives
    I32, I64, F64, Bool, String, Unit,

    // Collections
    Vec(Box<RustType>),
    HashMap(Box<RustType>, Box<RustType>),
    HashSet(Box<RustType>),

    // Option/Result
    Option(Box<RustType>),
    Result(Box<RustType>, Box<RustType>),

    // Tuples
    Tuple(Vec<RustType>),

    // Functions
    Fn {
        params: Vec<RustType>,
        ret: Box<RustType>,
    },

    // References
    Ref {
        lifetime: Option<String>,
        mutable: bool,
        inner: Box<RustType>,
    },

    // Generic type variable
    TypeVar(String),

    // Custom types (structs, enums)
    Custom {
        name: String,
        generics: Vec<RustType>,
    },

    // Trait object
    Trait {
        name: String,
        bounds: Vec<TraitBound>,
    },

    // Unknown (needs inference)
    Unknown,
}

impl TypeInferenceEngine {
    pub fn infer_type(&mut self, expr: &PyExpr) -> Result<RustType> {
        match expr {
            // Literal → concrete type
            PyExpr::Constant(value) => self.infer_literal_type(value),

            // Variable → lookup in environment
            PyExpr::Name { id, ctx } => {
                self.type_env.get(id)
                    .cloned()
                    .unwrap_or(RustType::Unknown)
            }

            // Binary operation → type of operands
            PyExpr::BinOp { left, op, right } => {
                let left_ty = self.infer_type(left)?;
                let right_ty = self.infer_type(right)?;
                self.infer_binop_result_type(&left_ty, op, &right_ty)
            }

            // Function call → return type
            PyExpr::Call { func, args, keywords } => {
                let func_ty = self.infer_type(func)?;
                match func_ty {
                    RustType::Fn { ret, .. } => Ok(*ret),
                    _ => Err(TypeError::NotCallable(func_ty))
                }
            }

            // List → Vec<T> where T is element type
            PyExpr::List { elts } => {
                if elts.is_empty() {
                    Ok(RustType::Vec(Box::new(RustType::Unknown)))
                } else {
                    // Infer from first element, unify with rest
                    let elem_ty = self.infer_type(&elts[0])?;
                    for elt in &elts[1..] {
                        let elt_ty = self.infer_type(elt)?;
                        self.unify(elem_ty.clone(), elt_ty)?;
                    }
                    Ok(RustType::Vec(Box::new(elem_ty)))
                }
            }

            // Dict → HashMap<K, V>
            PyExpr::Dict { keys, values } => {
                if keys.is_empty() {
                    Ok(RustType::HashMap(
                        Box::new(RustType::Unknown),
                        Box::new(RustType::Unknown)
                    ))
                } else {
                    let key_ty = self.infer_type(&keys[0])?;
                    let val_ty = self.infer_type(&values[0])?;
                    Ok(RustType::HashMap(Box::new(key_ty), Box::new(val_ty)))
                }
            }

            // List comprehension
            PyExpr::ListComp { elt, generators } => {
                let elem_ty = self.infer_type(elt)?;
                Ok(RustType::Vec(Box::new(elem_ty)))
            }

            // ... more cases
        }
    }

    // Unify two types (Hindley-Milner style)
    fn unify(&mut self, t1: RustType, t2: RustType) -> Result<RustType> {
        match (t1.clone(), t2.clone()) {
            // Same type
            (a, b) if a == b => Ok(a),

            // Unknown can be unified with anything
            (RustType::Unknown, t) | (t, RustType::Unknown) => Ok(t),

            // Type variable unification
            (RustType::TypeVar(v), t) | (t, RustType::TypeVar(v)) => {
                self.type_vars.insert(v, TypeVar::Bound(t.clone()));
                Ok(t)
            }

            // Collection types must unify element types
            (RustType::Vec(t1), RustType::Vec(t2)) => {
                let unified = self.unify(*t1, *t2)?;
                Ok(RustType::Vec(Box::new(unified)))
            }

            // Incompatible types
            (t1, t2) => Err(TypeError::CannotUnify(t1, t2))
        }
    }
}
```

**Special Cases**:

1. **None → Option<T>**:
   ```python
   x = None
   x = 42  # x must be Option<i32>
   ```

2. **Exception Handling → Result<T, E>**:
   ```python
   def may_fail():
       try:
           return risky_operation()
       except ValueError:
           return None
   # → fn may_fail() -> Result<i32, ValueError>
   ```

3. **Iterators**:
   ```python
   for x in range(10):  # x: i32
       pass

   for x in ["a", "b"]:  # x: &str
       pass
   ```

4. **Generic Functions**:
   ```python
   def identity(x):
       return x
   # → fn identity<T>(x: T) -> T
   ```

**Deliverables**:
- [ ] `src/type_inference.rs` - Comprehensive type inference engine
- [ ] `src/type_unification.rs` - Hindley-Milner unification
- [ ] `src/lifetime_inference.rs` - Lifetime analysis
- [ ] `src/trait_inference.rs` - Trait bound inference
- [ ] Test suite with complex type scenarios

**Success Criteria**:
- Correctly infer 90%+ of types
- Generate valid Rust type annotations
- Handle generics and trait bounds
- Proper lifetime annotations

**Complexity**: Very High (20-25 person-days)

---

### Phase 2: Core Infrastructure (Weeks 5-8)

**Goal**: Complete WASI module implementations and async runtime support.

#### 2.1 Complete wasi_fs Implementation (Week 5)
**Objective**: Build full filesystem operations beyond current core implementation.

**Current State**:
- `wasi_core.rs` (1066 lines) - Basic filesystem operations complete
- `wasi_fs.rs` exists (stub)
- `wasi_directory.rs` exists (stub)
- Core operations: fd_read, fd_write, fd_seek, path_open, path_create_directory

**Target State**:
- Full POSIX-like filesystem API
- Directory operations (list, traverse, stat)
- File metadata (permissions, timestamps)
- Symlinks and hardlinks
- File locking
- Memory-mapped files (mmap)
- Virtual filesystem support

**API Coverage**:

| Operation | WASI Function | Status | Priority |
|-----------|---------------|--------|----------|
| Open file | `path_open` | ✅ Complete | - |
| Read file | `fd_read` | ✅ Complete | - |
| Write file | `fd_write` | ✅ Complete | - |
| Seek | `fd_seek` | ✅ Complete | - |
| Tell | `fd_tell` | ✅ Complete | - |
| Close | `fd_close` | ✅ Complete | - |
| Create dir | `path_create_directory` | ✅ Complete | - |
| Remove file | `path_unlink_file` | ✅ Complete | - |
| File stat | `path_filestat_get` | ✅ Complete | - |
| Read dir | `fd_readdir` | ❌ Missing | HIGH |
| Rename | `path_rename` | ❌ Missing | HIGH |
| Symlink | `path_symlink` | ❌ Missing | MEDIUM |
| Readlink | `path_readlink` | ❌ Missing | MEDIUM |
| Remove dir | `path_remove_directory` | ❌ Missing | HIGH |
| File metadata | `fd_filestat_get` | ❌ Missing | HIGH |
| Set permissions | `path_filestat_set_times` | ❌ Missing | MEDIUM |
| File lock | `fd_advise`, `fd_allocate` | ❌ Missing | LOW |
| Prestat | `fd_prestat_get` | ❌ Missing | MEDIUM |

**Implementation**:

```rust
// wasi_fs.rs - Extended filesystem operations

impl WasiFilesystem {
    /// fd_readdir: Read directory entries
    pub fn fd_readdir(
        &self,
        fd: Fd,
        buf: &mut [u8],
        cookie: u64,
    ) -> Result<usize, WasiErrno> {
        // Verify fd is a directory
        let entry = self.fd_table.get(fd)?;
        if !entry.path.is_dir() {
            return Err(WasiErrno::NotDir);
        }

        // Read directory entries
        let entries = std::fs::read_dir(&entry.path)
            .map_err(|e| WasiErrno::from_io_error(&e))?;

        // Serialize entries into buffer (WASI format)
        let mut offset = 0;
        for (idx, entry) in entries.enumerate().skip(cookie as usize) {
            let entry = entry.map_err(|e| WasiErrno::from_io_error(&e))?;

            // Encode dirent structure
            let dirent = Dirent {
                d_next: cookie + idx as u64 + 1,
                d_ino: 0, // Placeholder
                d_namlen: entry.file_name().len() as u32,
                d_type: if entry.file_type()?.is_dir() {
                    FILETYPE_DIRECTORY
                } else {
                    FILETYPE_REGULAR_FILE
                },
            };

            if offset + size_of::<Dirent>() + dirent.d_namlen as usize > buf.len() {
                break; // Buffer full
            }

            // Write dirent + name
            // ... serialization logic

            offset += size_of::<Dirent>() + dirent.d_namlen as usize;
        }

        Ok(offset)
    }

    /// path_rename: Rename or move a file/directory
    pub fn path_rename(
        &self,
        old_fd: Fd,
        old_path: &Path,
        new_fd: Fd,
        new_path: &Path,
    ) -> Result<(), WasiErrno> {
        // Resolve both paths
        let old_dir = self.fd_table.get(old_fd)?;
        let new_dir = self.fd_table.get(new_fd)?;

        let old_full = PathResolver::resolve(&old_dir.path, old_path)?;
        let new_full = PathResolver::resolve(&new_dir.path, new_path)?;

        // Perform rename
        std::fs::rename(&old_full, &new_full)
            .map_err(|e| WasiErrno::from_io_error(&e))
    }

    /// fd_filestat_get: Get file/directory metadata
    pub fn fd_filestat_get(&self, fd: Fd) -> Result<Filestat, WasiErrno> {
        self.fd_table.get_mut(fd, |entry| {
            let metadata = if let Some(ref file) = entry.file {
                file.metadata().map_err(|e| WasiErrno::from_io_error(&e))?
            } else {
                std::fs::metadata(&entry.path)
                    .map_err(|e| WasiErrno::from_io_error(&e))?
            };

            Ok(Filestat {
                dev: 0,  // Device ID
                ino: 0,  // Inode
                filetype: if metadata.is_dir() {
                    FILETYPE_DIRECTORY
                } else {
                    FILETYPE_REGULAR_FILE
                },
                nlink: 1,  // Number of hard links
                size: metadata.len(),
                atim: system_time_to_wasi(metadata.accessed()?),
                mtim: system_time_to_wasi(metadata.modified()?),
                ctim: system_time_to_wasi(metadata.created()?),
            })
        })
    }
}
```

**Translation Examples**:

```python
# Python filesystem operations
import os

# List directory
for entry in os.listdir('/path'):
    print(entry)

# Rename file
os.rename('old.txt', 'new.txt')

# Get file stats
stat = os.stat('file.txt')
print(f"Size: {stat.st_size}")
```

```rust
// Generated Rust with WASI
use portalis_wasi::WasiFilesystem;

let wasi_fs = WasiFilesystem::new();
let dirfd = wasi_fs.add_preopen("/path")?;

// List directory
let mut buf = [0u8; 4096];
let bytes_read = wasi_fs.fd_readdir(dirfd, &mut buf, 0)?;
// Parse entries from buffer...

// Rename file
wasi_fs.path_rename(dirfd, Path::new("old.txt"), dirfd, Path::new("new.txt"))?;

// Get file stats
let stat = wasi_fs.path_filestat_get(dirfd, Path::new("file.txt"))?;
println!("Size: {}", stat.size);
```

**Deliverables**:
- [ ] Complete `wasi_fs.rs` implementation
- [ ] `wasi_directory.rs` - Directory operations
- [ ] `wasi_filestat.rs` - Metadata operations
- [ ] Virtual filesystem support (in-memory)
- [ ] Comprehensive test suite

**Success Criteria**:
- All WASI filesystem functions implemented
- Pass WASI test suite
- Support both native and WASM targets
- Full documentation

**Complexity**: High (15-20 person-days)

---

#### 2.2 Complete wasi_threading Implementation (Week 6)
**Objective**: Full threading and Web Workers support.

**Current State**:
- `wasi_threading/` directory exists (stubs)
- `web_workers/` directory exists (stubs)
- No actual threading implementation

**Target State**:
- POSIX-like threading API
- Web Workers for browser WASM
- Thread pools for native WASM
- Synchronization primitives (mutexes, semaphores, condition variables)
- Thread-local storage
- Atomic operations

**Threading Model**:

```
┌─────────────────────────────────────────────────────┐
│           Python Threading/Multiprocessing           │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Portalis Threading Layer                │
│  • Detects target (browser vs native vs server)      │
│  • Maps to appropriate backend                       │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Web Workers  │ │ WASI Threads │ │ std::thread  │
│  (Browser)   │ │  (WASI)      │ │  (Native)    │
└──────────────┘ └──────────────┘ └──────────────┘
```

**API Coverage**:

| Python API | Rust Translation | Browser | WASI | Native |
|------------|------------------|---------|------|--------|
| `threading.Thread` | `spawn_thread()` | Worker | wasi_thread | std::thread |
| `threading.Lock` | `Mutex<()>` | ✅ | ✅ | ✅ |
| `threading.RLock` | `parking_lot::RwLock` | ✅ | ✅ | ✅ |
| `threading.Semaphore` | `Semaphore` | ✅ | ✅ | ✅ |
| `threading.Event` | `Event` | ✅ | ✅ | ✅ |
| `threading.Condition` | `Condvar` | ⚠️ | ✅ | ✅ |
| `queue.Queue` | `crossbeam::channel` | ✅ | ✅ | ✅ |
| `multiprocessing.Pool` | `rayon::ThreadPool` | ❌ | ✅ | ✅ |

**Implementation**:

```rust
// wasi_threading/mod.rs

#[cfg(target_arch = "wasm32")]
pub mod web_worker_backend;

#[cfg(not(target_arch = "wasm32"))]
pub mod native_thread_backend;

pub struct PortalisThread {
    #[cfg(target_arch = "wasm32")]
    worker: web_sys::Worker,

    #[cfg(not(target_arch = "wasm32"))]
    handle: std::thread::JoinHandle<()>,
}

impl PortalisThread {
    pub fn spawn<F>(f: F) -> Result<Self, ThreadError>
    where
        F: FnOnce() + Send + 'static,
    {
        #[cfg(target_arch = "wasm32")]
        {
            // Create Web Worker
            let worker = web_sys::Worker::new("./worker.js")?;

            // Serialize closure and send to worker
            let closure = wasm_bindgen::closure::Closure::wrap(
                Box::new(f) as Box<dyn FnOnce()>
            );

            worker.post_message(&closure)?;

            Ok(Self { worker })
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Use std::thread
            let handle = std::thread::spawn(f);
            Ok(Self { handle })
        }
    }

    pub fn join(self) -> Result<(), ThreadError> {
        #[cfg(target_arch = "wasm32")]
        {
            // Wait for worker to finish (using Promise)
            // ... implementation
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.handle.join()
                .map_err(|_| ThreadError::PanickedThread)?;
            Ok(())
        }
    }
}

// Synchronization primitives
pub struct PortalisMutex<T> {
    #[cfg(target_arch = "wasm32")]
    inner: parking_lot::Mutex<T>,  // Single-threaded in browser

    #[cfg(not(target_arch = "wasm32"))]
    inner: std::sync::Mutex<T>,
}

impl<T> PortalisMutex<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
        }
    }

    pub fn lock(&self) -> Result<MutexGuard<T>, LockError> {
        self.inner.lock()
            .map_err(|_| LockError::Poisoned)
    }
}
```

**Translation Examples**:

```python
# Python threading
import threading

def worker(n):
    print(f"Worker {n}")

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

```rust
// Generated Rust
use portalis_wasi::threading::PortalisThread;

let mut threads = Vec::new();

for i in 0..5 {
    let t = PortalisThread::spawn(move || {
        println!("Worker {}", i);
    })?;
    threads.push(t);
}

for t in threads {
    t.join()?;
}
```

**Deliverables**:
- [ ] `wasi_threading/mod.rs` - Core threading API
- [ ] `web_workers/worker_pool.rs` - Web Worker pool
- [ ] `wasi_threading/sync.rs` - Synchronization primitives
- [ ] `wasi_threading/channel.rs` - Message passing
- [ ] Browser integration (worker.js template)
- [ ] Test suite (native + WASM)

**Success Criteria**:
- Spawn threads in browser (Web Workers)
- Spawn threads in WASI runtime
- All sync primitives working
- Message passing between threads
- No data races

**Complexity**: Very High (20-25 person-days)

---

#### 2.3 Complete wasi_websocket Implementation (Week 7)
**Objective**: WebSocket client and server support.

**Current State**:
- `wasi_websocket/` directory exists (stubs)
- `WEBSOCKET_IMPLEMENTATION.md` planning document
- No actual implementation

**Target State**:
- WebSocket client (browser + native)
- WebSocket server (native only)
- Binary and text message support
- Connection lifecycle management
- Error handling and reconnection
- SSL/TLS support

**Implementation**:

```rust
// wasi_websocket/client.rs

#[cfg(target_arch = "wasm32")]
use web_sys::WebSocket as BrowserWebSocket;

#[cfg(not(target_arch = "wasm32"))]
use tokio_tungstenite::connect_async;

pub struct WebSocketClient {
    #[cfg(target_arch = "wasm32")]
    inner: BrowserWebSocket,

    #[cfg(not(target_arch = "wasm32"))]
    inner: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>
    >,
}

impl WebSocketClient {
    pub async fn connect(url: &str) -> Result<Self, WebSocketError> {
        #[cfg(target_arch = "wasm32")]
        {
            let ws = BrowserWebSocket::new(url)?;

            // Setup event handlers
            let onopen = Closure::wrap(Box::new(move || {
                // Connection opened
            }) as Box<dyn FnMut()>);
            ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
            onopen.forget();

            Ok(Self { inner: ws })
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (stream, _) = connect_async(url).await?;
            Ok(Self { inner: stream })
        }
    }

    pub async fn send(&mut self, msg: Message) -> Result<(), WebSocketError> {
        #[cfg(target_arch = "wasm32")]
        {
            match msg {
                Message::Text(text) => self.inner.send_with_str(&text)?,
                Message::Binary(data) => {
                    let array = js_sys::Uint8Array::from(&data[..]);
                    self.inner.send_with_array_buffer(&array.buffer())?;
                }
            }
            Ok(())
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            use futures_util::SinkExt;
            self.inner.send(msg.into()).await?;
            Ok(())
        }
    }

    pub async fn recv(&mut self) -> Result<Message, WebSocketError> {
        #[cfg(target_arch = "wasm32")]
        {
            // Use onmessage callback + channel
            // ... implementation
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            use futures_util::StreamExt;
            let msg = self.inner.next().await
                .ok_or(WebSocketError::ConnectionClosed)??;
            Ok(msg.into())
        }
    }
}
```

**Translation Examples**:

```python
# Python WebSocket
import websocket

ws = websocket.WebSocket()
ws.connect("wss://echo.websocket.org")
ws.send("Hello, WebSocket!")
result = ws.recv()
print(result)
ws.close()
```

```rust
// Generated Rust
use portalis_wasi::websocket::WebSocketClient;

let mut ws = WebSocketClient::connect("wss://echo.websocket.org").await?;
ws.send(Message::Text("Hello, WebSocket!".to_string())).await?;
let result = ws.recv().await?;
println!("{:?}", result);
ws.close().await?;
```

**Deliverables**:
- [ ] `wasi_websocket/client.rs` - WebSocket client
- [ ] `wasi_websocket/server.rs` - WebSocket server (native)
- [ ] `wasi_websocket/message.rs` - Message types
- [ ] `wasi_websocket/error.rs` - Error handling
- [ ] Examples and tests

**Success Criteria**:
- Connect to WebSocket servers (browser + native)
- Send/receive text and binary messages
- Handle connection lifecycle
- SSL/TLS support

**Complexity**: Medium (10-15 person-days)

---

#### 2.4 Complete wasi_async_runtime (Week 8)
**Objective**: Full async/await runtime support.

**Current State**:
- `wasi_async_runtime/` directory exists (stubs)
- `py_to_rust_asyncio.rs` (953 lines) - Translation patterns complete
- `ASYNC_RUNTIME_COMPLETE.md` planning document
- No actual runtime implementation

**Target State**:
- Tokio integration for native WASM
- wasm-bindgen-futures for browser
- Event loop management
- Async I/O (files, network)
- Timers and intervals
- Signal handling

**Implementation**:

```rust
// wasi_async_runtime/runtime.rs

pub struct AsyncRuntime {
    #[cfg(target_arch = "wasm32")]
    _phantom: std::marker::PhantomData<()>,  // Browser uses browser event loop

    #[cfg(not(target_arch = "wasm32"))]
    runtime: tokio::runtime::Runtime,
}

impl AsyncRuntime {
    pub fn new() -> Result<Self, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            Ok(Self { _phantom: std::marker::PhantomData })
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(num_cpus::get())
                .enable_all()
                .build()?;

            Ok(Self { runtime })
        }
    }

    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: std::future::Future,
    {
        #[cfg(target_arch = "wasm32")]
        {
            // Use wasm_bindgen_futures::spawn_local
            wasm_bindgen_futures::spawn_local(async move {
                future.await
            });
            // Note: Can't actually block in browser, returns ()
            panic!("Cannot block_on in browser WASM");
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.runtime.block_on(future)
        }
    }

    pub fn spawn<F>(&self, future: F) -> TaskHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(future);
            TaskHandle::new()  // Dummy handle
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let handle = self.runtime.spawn(future);
            TaskHandle { inner: handle }
        }
    }
}
```

**Translation Examples**:

```python
# Python async
import asyncio

async def fetch_data(url):
    # Fetch data
    return data

async def main():
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

asyncio.run(main())
```

```rust
// Generated Rust
use portalis_wasi::async_runtime::AsyncRuntime;

async fn fetch_data(url: String) -> Result<Data> {
    // Fetch data
    Ok(data)
}

async fn main() -> Result<()> {
    let tasks: Vec<_> = urls.iter()
        .map(|url| fetch_data(url.clone()))
        .collect();

    let results = futures::future::join_all(tasks).await;
    Ok(())
}

// Entry point
fn run() {
    let runtime = AsyncRuntime::new().unwrap();
    runtime.block_on(main()).unwrap();
}
```

**Deliverables**:
- [ ] `wasi_async_runtime/runtime.rs` - Runtime implementation
- [ ] `wasi_async_runtime/executor.rs` - Task executor
- [ ] `wasi_async_runtime/timer.rs` - Async timers
- [ ] `wasi_async_runtime/io.rs` - Async I/O
- [ ] Integration with py_to_rust_asyncio.rs

**Success Criteria**:
- Run async functions in browser and native
- Spawn concurrent tasks
- Async I/O working
- Timers and intervals
- Full tokio integration

**Complexity**: High (15-20 person-days)

---

### Phase 3: Language Coverage (Weeks 9-14)

**Goal**: Expand Python language feature coverage from 28.5% to 70%+.

#### 3.1 Advanced Control Flow (Week 9)
**Objective**: Implement remaining control flow features.

**Features to Implement**:

1. **Match Statements (Python 3.10+)**:
   ```python
   match value:
       case 1:
           return "one"
       case 2:
           return "two"
       case _:
           return "other"
   ```
   ```rust
   match value {
       1 => "one".to_string(),
       2 => "two".to_string(),
       _ => "other".to_string(),
   }
   ```

2. **Walrus Operator (:=)**:
   ```python
   if (n := len(items)) > 10:
       print(f"Too many items: {n}")
   ```
   ```rust
   let n = items.len();
   if n > 10 {
       println!("Too many items: {}", n);
   }
   ```

3. **While/For Else Clauses** (already done, needs testing):
   ```python
   for item in items:
       if found(item):
           break
   else:
       # Only executes if no break
       print("Not found")
   ```
   ```rust
   let mut found = false;
   for item in items {
       if found(item) {
           found = true;
           break;
       }
   }
   if !found {
       println!("Not found");
   }
   ```

4. **Named Expressions in Comprehensions**:
   ```python
   results = [(y := x * 2, y + 1) for x in range(10)]
   ```

**Deliverables**:
- [ ] Match statement translator
- [ ] Walrus operator support
- [ ] Enhanced else clause handling
- [ ] Tests for all features

**Complexity**: Medium (8-10 person-days)

---

#### 3.2 Advanced Functions (Week 10)
**Objective**: Implement remaining function features.

**Features to Implement**:

1. **Default Parameters**:
   ```python
   def greet(name, greeting="Hello"):
       return f"{greeting}, {name}!"
   ```
   ```rust
   fn greet(name: String, greeting: Option<String>) -> String {
       let greeting = greeting.unwrap_or_else(|| "Hello".to_string());
       format!("{}, {}!", greeting, name)
   }
   ```

2. ***args and **kwargs**:
   ```python
   def variadic(required, *args, **kwargs):
       print(required, args, kwargs)
   ```
   ```rust
   fn variadic(required: i32, args: Vec<Value>, kwargs: HashMap<String, Value>) {
       println!("{:?} {:?} {:?}", required, args, kwargs);
   }
   ```

3. **Keyword-Only Arguments**:
   ```python
   def func(pos, *, kwonly):
       pass
   ```
   ```rust
   fn func(pos: i32, kwonly: i32) {
       // In Rust, all args are positional or named at call site
   }
   ```

4. **Type Hints (Full PEP 484)**:
   ```python
   from typing import List, Optional, Union

   def process(items: List[int]) -> Optional[int]:
       pass
   ```
   ```rust
   fn process(items: Vec<i32>) -> Option<i32> {
       None
   }
   ```

5. **Decorators**:
   ```python
   @property
   def name(self):
       return self._name

   @lru_cache(maxsize=128)
   def expensive(n):
       return n * 2
   ```
   ```rust
   // @property → getter method
   impl MyStruct {
       fn name(&self) -> &String {
           &self._name
       }
   }

   // @lru_cache → use cached crate
   #[cached::cached(size=128)]
   fn expensive(n: i32) -> i32 {
       n * 2
   }
   ```

**Deliverables**:
- [ ] Default parameter translator
- [ ] Variadic args support
- [ ] Type hint parser
- [ ] Decorator translator
- [ ] Tests

**Complexity**: High (12-15 person-days)

---

#### 3.3 Advanced Classes (Week 11)
**Objective**: Complete class feature support.

**Features to Implement**:

1. **Class Inheritance**:
   ```python
   class Animal:
       def speak(self):
           pass

   class Dog(Animal):
       def speak(self):
           return "Woof!"
   ```
   ```rust
   trait Animal {
       fn speak(&self) -> String;
   }

   struct Dog;

   impl Animal for Dog {
       fn speak(&self) -> String {
           "Woof!".to_string()
       }
   }
   ```

2. **Multiple Inheritance**:
   ```python
   class C(A, B):
       pass
   ```
   ```rust
   // Use trait composition
   struct C {
       a_data: A,
       b_data: B,
   }

   impl ATrait for C { /* delegate to a_data */ }
   impl BTrait for C { /* delegate to b_data */ }
   ```

3. **Properties**:
   ```python
   class Person:
       @property
       def age(self):
           return self._age

       @age.setter
       def age(self, value):
           if value < 0:
               raise ValueError("Age cannot be negative")
           self._age = value
   ```
   ```rust
   impl Person {
       fn age(&self) -> i32 {
           self._age
       }

       fn set_age(&mut self, value: i32) -> Result<(), ValueError> {
           if value < 0 {
               return Err(ValueError::new("Age cannot be negative"));
           }
           self._age = value;
           Ok(())
       }
   }
   ```

4. **Class Methods and Static Methods**:
   ```python
   class MyClass:
       @classmethod
       def from_string(cls, s):
           return cls(int(s))

       @staticmethod
       def helper(x):
           return x * 2
   ```
   ```rust
   impl MyClass {
       fn from_string(s: String) -> Self {
           Self::new(s.parse().unwrap())
       }

       fn helper(x: i32) -> i32 {
           x * 2
       }
   }
   ```

5. **Abstract Base Classes**:
   ```python
   from abc import ABC, abstractmethod

   class Shape(ABC):
       @abstractmethod
       def area(self):
           pass
   ```
   ```rust
   trait Shape {
       fn area(&self) -> f64;
   }
   ```

6. **Dataclasses**:
   ```python
   from dataclasses import dataclass

   @dataclass
   class Point:
       x: int
       y: int
   ```
   ```rust
   #[derive(Debug, Clone, PartialEq)]
   struct Point {
       x: i32,
       y: i32,
   }
   ```

**Deliverables**:
- [ ] Inheritance translator
- [ ] Property translator
- [ ] Class/static method support
- [ ] ABC translator
- [ ] Dataclass support
- [ ] Tests

**Complexity**: Very High (18-22 person-days)

---

#### 3.4 Comprehensions and Generators (Week 12)
**Objective**: Full comprehension and generator support.

**Features to Implement**:

1. **Dict Comprehensions**:
   ```python
   squares = {x: x**2 for x in range(10)}
   ```
   ```rust
   let squares: HashMap<i32, i32> = (0..10)
       .map(|x| (x, x.pow(2)))
       .collect();
   ```

2. **Set Comprehensions**:
   ```python
   unique = {x % 10 for x in range(100)}
   ```
   ```rust
   let unique: HashSet<i32> = (0..100)
       .map(|x| x % 10)
       .collect();
   ```

3. **Nested Comprehensions**:
   ```python
   matrix = [[i*j for j in range(5)] for i in range(5)]
   ```
   ```rust
   let matrix: Vec<Vec<i32>> = (0..5)
       .map(|i| (0..5).map(|j| i * j).collect())
       .collect();
   ```

4. **Generator Functions (yield)**:
   ```python
   def fibonacci():
       a, b = 0, 1
       while True:
           yield a
           a, b = b, a + b
   ```
   ```rust
   use std::iter;

   fn fibonacci() -> impl Iterator<Item = i32> {
       iter::successors(Some((0, 1)), |(a, b)| Some((*b, a + b)))
           .map(|(a, _)| a)
   }
   ```

5. **Generator Expressions**:
   ```python
   gen = (x**2 for x in range(1000000))
   ```
   ```rust
   let gen = (0..1000000).map(|x| x.pow(2));
   ```

6. **Async Generators**:
   ```python
   async def async_gen():
       for i in range(10):
           await asyncio.sleep(0.1)
           yield i
   ```
   ```rust
   use futures::stream::{self, Stream};

   fn async_gen() -> impl Stream<Item = i32> {
       stream::iter(0..10).then(|i| async move {
           tokio::time::sleep(Duration::from_millis(100)).await;
           i
       })
   }
   ```

**Deliverables**:
- [ ] Dict/set comprehension translator
- [ ] Nested comprehension support
- [ ] Generator function translator
- [ ] Async generator support
- [ ] Tests

**Complexity**: High (15-18 person-days)

---

#### 3.5 Context Managers (Week 13)
**Objective**: Full `with` statement support.

**Features to Implement**:

1. **Basic With**:
   ```python
   with open("file.txt") as f:
       content = f.read()
   ```
   ```rust
   {
       let mut f = File::open("file.txt")?;
       let mut content = String::new();
       f.read_to_string(&mut content)?;
       // f automatically closed (RAII)
   }
   ```

2. **Multiple Context Managers**:
   ```python
   with open("in.txt") as fin, open("out.txt", "w") as fout:
       fout.write(fin.read())
   ```
   ```rust
   {
       let mut fin = File::open("in.txt")?;
       let mut fout = File::create("out.txt")?;
       let mut content = String::new();
       fin.read_to_string(&mut content)?;
       fout.write_all(content.as_bytes())?;
   }
   ```

3. **Custom Context Managers**:
   ```python
   class MyContext:
       def __enter__(self):
           print("Entering")
           return self

       def __exit__(self, exc_type, exc_val, exc_tb):
           print("Exiting")
           return False
   ```
   ```rust
   struct MyContext;

   impl MyContext {
       fn enter(&mut self) -> &mut Self {
           println!("Entering");
           self
       }
   }

   impl Drop for MyContext {
       fn drop(&mut self) {
           println!("Exiting");
       }
   }
   ```

4. **Async Context Managers**:
   ```python
   async with aiohttp.ClientSession() as session:
       async with session.get(url) as response:
           return await response.text()
   ```
   ```rust
   {
       let session = ClientSession::new().await?;
       let response = session.get(url).await?;
       let text = response.text().await?;
       text
       // session and response automatically dropped
   }
   ```

**Deliverables**:
- [ ] With statement translator
- [ ] RAII pattern generator
- [ ] Custom context manager support
- [ ] Async with support
- [ ] Tests

**Complexity**: Medium (10-12 person-days)

---

#### 3.6 Advanced Iterators (Week 14)
**Objective**: Full iterator protocol support.

**Features to Implement**:

1. **Iterator Protocol**:
   ```python
   class Counter:
       def __init__(self, max):
           self.max = max
           self.n = 0

       def __iter__(self):
           return self

       def __next__(self):
           if self.n >= self.max:
               raise StopIteration
           self.n += 1
           return self.n
   ```
   ```rust
   struct Counter {
       max: i32,
       n: i32,
   }

   impl Iterator for Counter {
       type Item = i32;

       fn next(&mut self) -> Option<Self::Item> {
           if self.n >= self.max {
               None
           } else {
               self.n += 1;
               Some(self.n)
           }
       }
   }
   ```

2. **Iterator Tools**:
   ```python
   from itertools import chain, zip_longest, islice

   # Chain iterators
   combined = chain(iter1, iter2)

   # Zip with padding
   paired = zip_longest(list1, list2, fillvalue=0)

   # Slice iterator
   first_10 = islice(infinite_iterator, 10)
   ```
   ```rust
   use itertools::Itertools;

   // Chain iterators
   let combined = iter1.chain(iter2);

   // Zip with padding
   let paired = iter1.zip_longest(iter2)
       .map(|p| p.or_both(0));

   // Slice iterator
   let first_10 = infinite_iterator.take(10);
   ```

**Deliverables**:
- [ ] Iterator protocol translator
- [ ] itertools integration
- [ ] Custom iterator support
- [ ] Tests

**Complexity**: Medium (8-10 person-days)

---

### Phase 4: Package Ecosystem (Weeks 15-17)

**Goal**: Enable transpilation of real-world Python libraries.

#### 4.1 Enhanced Package Resolution (Week 15)
**Objective**: Improve external package mapping beyond current 200+ packages.

**Current State**:
- `external_packages.rs` - 200+ PyPI package mappings
- Basic API mappings for common packages

**Target State**:
- 500+ package mappings
- API-level function mappings
- Automatic dependency resolution
- Version compatibility checking
- Alternative crate suggestions

**Key Packages to Add**:

| Python Package | Rust Crate | Priority | Complexity |
|----------------|------------|----------|------------|
| beautifulsoup4 | scraper | HIGH | Medium |
| lxml | roxmltree | HIGH | Medium |
| cryptography | ring / rustls | HIGH | High |
| pyyaml | serde_yaml | HIGH | Low |
| boto3 | rusoto | MEDIUM | High |
| sqlalchemy | diesel / sqlx | HIGH | Very High |
| celery | N/A (build custom) | MEDIUM | Very High |
| pytest | Built-in #[test] | HIGH | Medium |
| black | rustfmt | LOW | Low |
| mypy | cargo check | LOW | Low |

**Implementation**:

```rust
// Enhanced package resolution
pub struct PackageResolver {
    registry: ExternalPackageRegistry,
    dependency_graph: DependencyGraph,
}

impl PackageResolver {
    pub fn resolve_imports(&self, imports: &[Import])
        -> Result<ResolvedDependencies> {
        let mut deps = Vec::new();

        for import in imports {
            // Check if it's a stdlib module
            if let Some(stdlib_mapping) = self.resolve_stdlib(&import.module) {
                deps.push(Dependency::Stdlib(stdlib_mapping));
                continue;
            }

            // Check external package registry
            if let Some(pkg) = self.registry.get_package(&import.module) {
                deps.push(Dependency::External {
                    python_pkg: import.module.clone(),
                    rust_crate: pkg.rust_crate.clone(),
                    version: pkg.version.clone(),
                    features: pkg.features.clone(),
                });
                continue;
            }

            // Unknown package - warn user
            deps.push(Dependency::Unknown(import.module.clone()));
        }

        // Build dependency graph and check for conflicts
        let graph = self.dependency_graph.build(&deps)?;

        // Resolve version conflicts
        let resolved = self.resolve_versions(graph)?;

        Ok(resolved)
    }

    pub fn generate_cargo_toml(&self, deps: &ResolvedDependencies)
        -> String {
        let mut cargo = String::from("[dependencies]\n");

        for dep in &deps.external {
            cargo.push_str(&format!(
                "{} = {{ version = \"{}\", features = {:?} }}\n",
                dep.rust_crate,
                dep.version,
                dep.features
            ));
        }

        cargo
    }
}
```

**Deliverables**:
- [ ] Expand package registry to 500+
- [ ] API-level function mappings
- [ ] Version resolution algorithm
- [ ] Cargo.toml generator enhancements
- [ ] Compatibility matrix

**Success Criteria**:
- Resolve 90%+ of common Python imports
- Generate valid Cargo.toml
- Handle version conflicts
- Suggest alternatives for unmapped packages

**Complexity**: Medium (10-12 person-days)

---

#### 4.2 NumPy/SciPy Support (Week 16)
**Objective**: Deep integration with ndarray and nalgebra for scientific computing.

**Implementation Strategy**:

```python
# Python NumPy
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
squared = arr ** 2
mean = np.mean(squared)
```

```rust
// Generated Rust
use ndarray::{Array1, Array};

let arr = Array1::from_vec(vec![1, 2, 3, 4, 5]);
let squared = arr.mapv(|x| x.pow(2));
let mean = squared.mean().unwrap();
```

**API Mappings**:

| NumPy Function | ndarray Equivalent |
|----------------|-------------------|
| `np.array()` | `Array::from_vec()` |
| `np.zeros()` | `Array::zeros()` |
| `np.ones()` | `Array::ones()` |
| `np.arange()` | `Array::range()` |
| `np.linspace()` | `Array::linspace()` |
| `arr.reshape()` | `.into_shape()` |
| `arr.transpose()` | `.t()` |
| `np.dot()` | `.dot()` |
| `np.matmul()` | `.dot()` |
| `np.linalg.inv()` | nalgebra |

**Deliverables**:
- [ ] NumPy API mapper
- [ ] ndarray code generator
- [ ] SciPy function mappings
- [ ] Test suite with scientific workloads

**Complexity**: High (12-15 person-days)

---

#### 4.3 Pandas/Polars Support (Week 17)
**Objective**: DataFrame operations for data science.

**Implementation Strategy**:

```python
# Python Pandas
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
})

adults = df[df['age'] >= 30]
```

```rust
// Generated Rust with Polars
use polars::prelude::*;

let df = DataFrame::new(vec![
    Series::new("name", &["Alice", "Bob", "Charlie"]),
    Series::new("age", &[25, 30, 35]),
])?;

let adults = df.filter(&df.column("age")?.gt(30)?)?;
```

**Deliverables**:
- [ ] Pandas API mapper
- [ ] Polars code generator
- [ ] DataFrame operation translator
- [ ] Test suite

**Complexity**: High (12-15 person-days)

---

### Phase 5: Production Hardening (Weeks 18-20)

**Goal**: Production-ready transpiler with optimization, testing, and deployment.

#### 5.1 Code Optimization (Week 18)
**Objective**: Generate optimized, idiomatic Rust code.

**Optimization Passes**:

1. **Dead Code Elimination**:
   - Remove unused variables
   - Remove unreachable code
   - Inline constants

2. **Expression Simplification**:
   - Constant folding
   - Algebraic simplification
   - Common subexpression elimination

3. **Lifetime Optimization**:
   - Minimize borrow lifetimes
   - Use references instead of clones
   - Optimize ownership transfers

4. **Iterator Fusion**:
   ```python
   result = list(map(lambda x: x * 2, filter(lambda x: x > 0, items)))
   ```
   ```rust
   // Before
   let result: Vec<_> = items.iter()
       .filter(|x| **x > 0)
       .map(|x| x * 2)
       .collect();

   // After (already optimized by rustc, but good to generate this way)
   let result: Vec<_> = items.iter()
       .filter_map(|x| if *x > 0 { Some(x * 2) } else { None })
       .collect();
   ```

**Deliverables**:
- [ ] `build_optimizer.rs` enhancement
- [ ] Dead code eliminator
- [ ] Expression simplifier
- [ ] Lifetime optimizer
- [ ] Benchmarks

**Complexity**: High (12-15 person-days)

---

#### 5.2 Testing and Validation (Week 19)
**Objective**: Comprehensive test suite and validation.

**Test Coverage Goals**:
- Unit tests: 95%+ coverage
- Integration tests: 50+ real Python scripts
- E2E tests: Full pipeline (Python → Rust → WASM)
- Performance tests: Benchmarks vs Python

**Deliverables**:
- [ ] Expand unit test suite (500+ tests)
- [ ] Integration test suite (50+ real scripts)
- [ ] E2E test pipeline
- [ ] Performance benchmarks
- [ ] Fuzzing infrastructure

**Complexity**: Medium (10-12 person-days)

---

#### 5.3 Documentation and Deployment (Week 20)
**Objective**: Production-ready documentation and deployment tools.

**Documentation**:
- User guide
- API reference
- Translation guide (Python → Rust patterns)
- Troubleshooting guide
- Example gallery

**Deployment Tools**:
- CLI tool enhancements
- VS Code extension
- GitHub Actions integration
- Docker container
- Package registry (crates.io)

**Deliverables**:
- [ ] Complete documentation
- [ ] CLI tool
- [ ] VS Code extension
- [ ] CI/CD templates
- [ ] Docker image

**Complexity**: Medium (10-12 person-days)

---

## Integration Points

### 1. Parser → Translator
```
rustpython AST → Internal AST → Type Inference → Translation
```

### 2. Translator → Code Generator
```
Rust AST → Optimization Passes → Code Formatting → Output
```

### 3. External Packages → Dependencies
```
Python imports → Package Resolution → Cargo.toml → Rust crates
```

### 4. WASI Runtime → Generated Code
```
Python stdlib calls → WASI functions → Runtime implementation
```

### 5. Async Runtime → Generated Async Code
```
Python asyncio → Tokio/wasm-bindgen-futures → Async Rust
```

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| rustpython-parser limitations | Medium | High | Contribute upstream or fork |
| Type inference edge cases | High | Medium | Incremental approach, extensive testing |
| WASI spec changes | Low | Medium | Track WASI standardization |
| Browser compatibility | Medium | Medium | Feature detection, polyfills |
| Performance overhead | Low | Low | Optimization passes, benchmarks |
| Lifetime inference complexity | High | High | Conservative borrowing, clone when needed |
| External package API drift | High | Medium | Version pinning, compatibility matrix |
| Memory safety violations | Low | Very High | Extensive testing, fuzzing, miri |

### Mitigation Strategies

1. **Technical**:
   - Incremental implementation with continuous testing
   - Feature flags for experimental features
   - Extensive fuzzing and property-based testing
   - Miri for undefined behavior detection

2. **Process**:
   - Weekly progress reviews
   - Monthly milestone deliverables
   - Continuous integration with comprehensive test suite
   - Code reviews for all changes

3. **Resource**:
   - Cross-training team members
   - Documentation as code
   - Pair programming for complex features
   - External expert consultation

---

## Testing Strategy

### Unit Testing
- **Coverage Goal**: 95%+
- **Approach**: Test each component in isolation
- **Tools**: cargo test, proptest (property-based)
- **Categories**:
  - Parser tests (valid and invalid Python)
  - Type inference tests
  - Expression translation tests
  - Statement translation tests
  - WASI function tests
  - Optimization pass tests

### Integration Testing
- **Coverage Goal**: 50+ real Python scripts
- **Approach**: Full pipeline testing
- **Test Cases**:
  - Data processing scripts
  - Web scrapers
  - ML model inference
  - CLI utilities
  - Web servers (subset)

### End-to-End Testing
- **Approach**: Python → Rust → WASM → Execution
- **Validation**:
  - Correct output
  - Performance benchmarks
  - Memory usage
  - WASM size

### Performance Testing
- **Benchmarks**:
  - Translation speed (lines/sec)
  - Generated code performance vs Python
  - WASM module size
  - Runtime overhead

### Fuzzing
- **Tools**: cargo-fuzz, AFL
- **Targets**:
  - Parser (malformed Python)
  - Type inference
  - Code generator

---

## Performance Considerations

### Translation Performance
- **Target**: Translate 1000+ lines/sec
- **Optimizations**:
  - Parallel AST traversal
  - Cached type inference
  - Incremental compilation

### Generated Code Performance
- **Target**: 10-50x faster than Python
- **Optimizations**:
  - Inline small functions
  - Use stack allocation when possible
  - Minimize allocations
  - Use iterators instead of collections

### WASM Module Size
- **Target**: <500KB for typical library
- **Optimizations**:
  - Tree shaking (dead code elimination)
  - wasm-opt aggressive optimization
  - Code splitting for large libraries

### Runtime Overhead
- **Target**: <5% overhead vs native Rust
- **Optimizations**:
  - Minimize dynamic dispatch
  - Use static dispatch when possible
  - Avoid unnecessary boxing

---

## Timeline Estimate

### Detailed Schedule

| Phase | Weeks | Person-Weeks | Dependencies | Deliverables |
|-------|-------|--------------|--------------|-------------|
| **Phase 1: Foundation** | 1-4 | 16-20 | None | AST parser, Generic translators, Type inference |
| 1.1 AST Parser | 1 | 2-3 | - | rustpython integration |
| 1.2 Expression Translator | 2 | 3-4 | 1.1 | Generic expression support |
| 1.3 Statement Translator | 3 | 3-4 | 1.1, 1.2 | All statement types |
| 1.4 Type Inference | 4 | 4-5 | 1.1, 1.2 | Advanced type system |
| **Phase 2: Infrastructure** | 5-8 | 16-20 | Phase 1 | Complete WASI, Async runtime |
| 2.1 wasi_fs | 5 | 3-4 | 1.4 | Full filesystem |
| 2.2 wasi_threading | 6 | 4-5 | 1.4 | Threading support |
| 2.3 wasi_websocket | 7 | 2-3 | 2.2 | WebSocket support |
| 2.4 wasi_async_runtime | 8 | 3-4 | 2.2, 2.3 | Async runtime |
| **Phase 3: Language Coverage** | 9-14 | 24-30 | Phase 2 | 70%+ Python coverage |
| 3.1 Advanced Control Flow | 9 | 2 | 1.3 | Match, walrus operator |
| 3.2 Advanced Functions | 10 | 3 | 1.3, 1.4 | Defaults, *args, decorators |
| 3.3 Advanced Classes | 11 | 4-5 | 1.3, 1.4 | Inheritance, properties |
| 3.4 Comprehensions | 12 | 3-4 | 1.2, 1.4 | Dict/set comps, generators |
| 3.5 Context Managers | 13 | 2-3 | 1.3 | With statements |
| 3.6 Iterators | 14 | 2 | 1.2 | Iterator protocol |
| **Phase 4: Package Ecosystem** | 15-17 | 12-15 | Phase 3 | 500+ packages, NumPy, Pandas |
| 4.1 Package Resolution | 15 | 2-3 | 1.1 | 500+ packages |
| 4.2 NumPy Support | 16 | 3-4 | 4.1 | Scientific computing |
| 4.3 Pandas Support | 17 | 3-4 | 4.1 | Data science |
| **Phase 5: Production** | 18-20 | 12-15 | Phase 4 | Production ready |
| 5.1 Optimization | 18 | 3-4 | All | Code optimization |
| 5.2 Testing | 19 | 2-3 | All | Comprehensive tests |
| 5.3 Documentation | 20 | 2-3 | All | Docs, deployment |

**Total Timeline**: 16-20 weeks (4-5 months)
**Total Effort**: 80-100 person-weeks
**Team Size**: 2-3 senior Rust engineers

### Parallelizable Work Streams

- **Stream A** (Parser/Translator): 1.1 → 1.2 → 1.3 → 3.x
- **Stream B** (Type System): 1.4 → 3.2 → 3.3 → 4.x
- **Stream C** (Runtime): 2.x → 5.1
- **Stream D** (Testing/Docs): 5.2 → 5.3 (ongoing)

---

## Success Metrics

### Development Metrics
- **Code Coverage**: 95%+ unit test coverage
- **Test Pass Rate**: 95%+ (vs current 87.2%)
- **Build Time**: <2 minutes for full rebuild
- **CI/CD Time**: <10 minutes end-to-end

### Functional Metrics
- **Python Feature Coverage**: 70%+ of 527 features
- **Package Coverage**: 500+ PyPI packages mapped
- **Translation Success Rate**: 90%+ of valid Python scripts
- **Generated Code Compilation**: 95%+ compiles without errors

### Performance Metrics
- **Translation Speed**: 1000+ lines/second
- **Generated Code Performance**: 10-50x faster than Python
- **WASM Module Size**: <500KB for typical library
- **Runtime Overhead**: <5% vs native Rust

### Production Readiness
- **Documentation**: Complete (user guide, API ref, examples)
- **Deployment**: One-command deployment to browser/server
- **Ecosystem**: Integration with VS Code, GitHub Actions, Docker
- **Community**: 100+ GitHub stars, 10+ contributors

---

## Top 3 Priorities

### 1. Real Python AST Parser (Week 1)
**Why**: Foundation for everything else. Without full AST support, we're limited to toy examples.

**Impact**: Unlock ability to parse any Python file, enabling work on all other features.

**Dependencies**: None (can start immediately)

### 2. Generic Expression/Statement Translators (Weeks 2-3)
**Why**: Move beyond 15 hardcoded patterns to handle arbitrary Python code.

**Impact**: Enable translation of real-world Python scripts, not just demos.

**Dependencies**: AST parser (Week 1)

### 3. Complete WASI Filesystem (Week 5)
**Why**: Most Python scripts do file I/O. Without this, many real scripts fail.

**Impact**: Support Python scripts that read/write files, which is 80%+ of use cases.

**Dependencies**: Foundation (Weeks 1-4)

---

## Top 3 Risks

### 1. Type Inference Complexity (Probability: High, Impact: High)
**Risk**: Python's dynamic typing makes full type inference extremely difficult. Edge cases abound.

**Mitigation**:
- Start with simple cases (primitives, collections)
- Use type hints when available
- Fall back to runtime typing (Box<dyn Any>) when needed
- Conservative approach: clone instead of borrow when uncertain
- Extensive testing with real Python code

**Contingency**: If type inference proves too complex, use a hybrid approach with optional type annotations and runtime type checks.

### 2. WASI Spec Instability (Probability: Medium, Impact: Medium)
**Risk**: WASI is still evolving (currently preview1, preview2 in development). APIs may change.

**Mitigation**:
- Use stable preview1 APIs
- Abstract WASI calls behind our own API layer
- Monitor WASI standardization process
- Provide migration path for spec changes
- Version compatibility layer

**Contingency**: Maintain multiple WASI versions, provide feature flags for different targets.

### 3. Lifetime Inference Failures (Probability: High, Impact: High)
**Risk**: Rust's borrow checker is strict. Python's reference semantics don't map directly. May generate uncompilable code.

**Mitigation**:
- Conservative borrowing strategy (clone by default)
- Use Rc/Arc for shared ownership
- Explicit lifetime annotations
- Runtime reference counting when needed
- Extensive testing with borrow checker

**Contingency**: Provide "escape hatches" using unsafe code for edge cases, with runtime checks to ensure safety.

---

## Incremental Delivery Approach

### Week 4 Milestone: "Foundation Complete"
**Deliverables**:
- Parse any valid Python 3.x file
- Translate basic expressions and statements
- Generate compilable Rust code for simple scripts
- Type inference for primitives and collections

**Success Criteria**:
- Translate 50+ Python scripts successfully
- Generated code compiles and runs correctly
- Test pass rate: 90%+

### Week 8 Milestone: "Runtime Complete"
**Deliverables**:
- Full WASI filesystem support
- Threading and Web Workers working
- WebSocket client/server
- Async runtime integration

**Success Criteria**:
- Translate Python scripts using files, threads, network
- WASM modules run in browser and server
- Test pass rate: 92%+

### Week 14 Milestone: "Language Complete"
**Deliverables**:
- 70%+ of Python features implemented
- Advanced classes, functions, comprehensions
- Context managers, iterators, generators
- Decorator support

**Success Criteria**:
- Translate complex Python libraries
- Support OOP patterns
- Test pass rate: 95%+

### Week 17 Milestone: "Ecosystem Ready"
**Deliverables**:
- 500+ package mappings
- NumPy/Pandas support
- Dependency resolution
- Real-world library translation

**Success Criteria**:
- Translate data science workloads
- Scientific computing libraries
- Web scraping scripts
- Test pass rate: 95%+

### Week 20 Milestone: "Production Ready"
**Deliverables**:
- Optimized code generation
- Comprehensive documentation
- Deployment tools
- Full test suite

**Success Criteria**:
- 10-50x performance vs Python
- <500KB WASM modules
- Complete documentation
- 95%+ test coverage
- 95%+ test pass rate

---

## Conclusion

This specification provides a **comprehensive, actionable roadmap** to transform PORTALIS from a capable MVP transpiler (87.2% test pass rate, 28.5% Python coverage) to a **production-grade, industrial-strength** transpiler capable of handling any Python library or script.

### Key Takeaways

1. **Phased Approach**: 5 phases over 16-20 weeks, building incrementally
2. **Focus on Foundation**: Weeks 1-4 are critical (AST parser, generic translation, type inference)
3. **Parallel Work Streams**: Multiple engineers can work concurrently
4. **Incremental Milestones**: Usable deliverables every 4 weeks
5. **Risk Mitigation**: Identified top risks with concrete mitigation strategies
6. **Realistic Timeline**: 4-5 months with 2-3 senior engineers

### Expected Outcomes

- **70%+ Python feature coverage** (vs current 28.5%)
- **95%+ test pass rate** (vs current 87.2%)
- **500+ package mappings** (vs current 200+)
- **10-50x performance** vs Python
- **Production-ready WASM pipeline**
- **Full browser and server support**

### Next Steps

1. **Immediate**: Start Week 1 (AST parser integration)
2. **Short-term**: Complete Phase 1 (Foundation) in 4 weeks
3. **Medium-term**: Deliver "Runtime Complete" milestone at Week 8
4. **Long-term**: Production release at Week 20

This is an ambitious but achievable plan that will establish PORTALIS as the **definitive Python-to-WASM transpiler** in the Rust ecosystem.

---

**Document End**
