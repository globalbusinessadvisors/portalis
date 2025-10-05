# API Reference - Portalis Transpiler

Complete API documentation for all public interfaces in the Portalis Transpiler.

---

## Table of Contents

1. [PyToRustTranspiler](#pytorusttranspiler)
2. [Type System](#type-system)
3. [Library Translators](#library-translators)
4. [WASM Bundler](#wasm-bundler)
5. [Dead Code Eliminator](#dead-code-eliminator)
6. [Cargo Generator](#cargo-generator)
7. [Dependency Resolver](#dependency-resolver)
8. [Import Analyzer](#import-analyzer)

---

## PyToRustTranspiler

Main translation engine for converting Python code to Rust.

### Struct

```rust
pub struct PyToRustTranspiler {
    // Internal state
}
```

### Methods

#### `new() -> Self`

Creates a new transpiler instance.

```rust
let mut transpiler = PyToRustTranspiler::new();
```

#### `translate(&mut self, python_code: &str) -> String`

Translates Python code to Rust.

**Parameters:**
- `python_code: &str` - Python source code

**Returns:** Rust source code as `String`

**Example:**
```rust
let rust_code = transpiler.translate("def add(a, b): return a + b");
```

#### `translate_file(&mut self, path: &str) -> String`

Translates a Python file to Rust.

**Parameters:**
- `path: &str` - Path to Python file

**Returns:** Rust source code as `String`

**Example:**
```rust
let rust_code = transpiler.translate_file("my_script.py");
```

#### `translate_module(&mut self, python_code: &str, module_name: &str) -> String`

Translates Python code as a named module.

**Parameters:**
- `python_code: &str` - Python source code
- `module_name: &str` - Module name

**Returns:** Rust module as `String`

**Example:**
```rust
let rust_module = transpiler.translate_module(code, "my_module");
```

#### `set_options(&mut self, options: TranslationOptions)`

Configures translation options.

**Parameters:**
- `options: TranslationOptions` - Configuration options

**Example:**
```rust
let options = TranslationOptions {
    preserve_comments: true,
    infer_lifetimes: true,
    optimize_references: true,
    ..Default::default()
};
transpiler.set_options(options);
```

### TranslationOptions

```rust
pub struct TranslationOptions {
    pub preserve_comments: bool,      // Keep Python comments
    pub infer_lifetimes: bool,        // Automatic lifetime inference
    pub optimize_references: bool,    // Reference optimization
    pub generate_tests: bool,         // Convert pytest to #[test]
    pub async_runtime: AsyncRuntime,  // tokio, async-std, smol
    pub target: CompilationTarget,    // native, wasm32-unknown, wasm32-wasi
}

impl Default for TranslationOptions {
    fn default() -> Self {
        TranslationOptions {
            preserve_comments: true,
            infer_lifetimes: true,
            optimize_references: true,
            generate_tests: true,
            async_runtime: AsyncRuntime::Tokio,
            target: CompilationTarget::Native,
        }
    }
}
```

---

## Type System

### TypeInference

Hindley-Milner type inference engine.

#### Struct

```rust
pub struct TypeInference {
    // Internal type environment
}
```

#### Methods

##### `new() -> Self`

Creates a new type inference engine.

```rust
let mut inference = TypeInference::new();
```

##### `infer_type(&mut self, expr: &Expr) -> Result<Type, TypeError>`

Infers the type of an expression.

**Parameters:**
- `expr: &Expr` - Expression to analyze

**Returns:** `Result<Type, TypeError>`

**Example:**
```rust
let expr = parse_expression("lambda x: x + 1");
let type_result = inference.infer_type(&expr)?;
```

##### `unify(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError>`

Unifies two types.

**Parameters:**
- `t1: &Type` - First type
- `t2: &Type` - Second type

**Returns:** `Result<(), TypeError>`

**Example:**
```rust
inference.unify(&Type::Int, &Type::Int)?;
```

### Type

Type representation.

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
    Custom(String),
}
```

### LifetimeAnalysis

Lifetime inference for references.

#### Struct

```rust
pub struct LifetimeAnalysis {
    // Lifetime tracking
}
```

#### Methods

##### `analyze(&mut self, code: &str) -> LifetimeInfo`

Analyzes lifetime requirements.

**Parameters:**
- `code: &str` - Rust code to analyze

**Returns:** `LifetimeInfo` with lifetime annotations

**Example:**
```rust
let analyzer = LifetimeAnalysis::new();
let info = analyzer.analyze(rust_code);
```

### ReferenceOptimizer

Optimizes reference usage.

#### Methods

##### `optimize(&mut self, code: &str) -> String`

Optimizes reference and ownership patterns.

**Parameters:**
- `code: &str` - Rust code

**Returns:** Optimized Rust code

**Example:**
```rust
let optimizer = ReferenceOptimizer::new();
let optimized = optimizer.optimize(rust_code);
```

---

## Library Translators

### CommonLibrariesTranslator

Translates common Python libraries to Rust equivalents.

#### Enum: PythonLibrary

```rust
pub enum PythonLibrary {
    Requests,     // → reqwest
    Pytest,       // → #[test]
    Pydantic,     // → serde + validator
    Logging,      // → log
    Argparse,     // → clap
    Json,         // → serde_json
    Datetime,     // → chrono
    Pathlib,      // → std::path
    Regex,        // → regex
    Os,           // → std::fs, std::env
    Sys,          // → std::env
    Collections,  // → std::collections
}
```

#### Methods

##### `translate_requests(&mut self, op: &RequestsOp, args: &[String]) -> String`

Translates requests operations.

**Operations:**
```rust
pub enum RequestsOp {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
    Session,
    Headers,
    Timeout,
    Auth,
}
```

**Example:**
```rust
let translator = CommonLibrariesTranslator::new();
let rust_code = translator.translate_requests(
    &RequestsOp::Get,
    &["https://api.example.com".to_string()]
);
// Output: reqwest::get("https://api.example.com").await?.text().await?
```

##### `translate_pytest(&mut self, op: &PytestOp, args: &[String]) -> String`

Translates pytest patterns.

**Operations:**
```rust
pub enum PytestOp {
    Test,
    Fixture,
    Parametrize,
    Mark,
    Raises,
    Skip,
    Xfail,
}
```

**Example:**
```rust
let rust_test = translator.translate_pytest(
    &PytestOp::Test,
    &["my_function".to_string()]
);
// Output: #[test]\nfn test_my_function() { ... }
```

##### `get_cargo_dependencies(&self, library: &PythonLibrary) -> Vec<String>`

Returns Rust dependencies for a Python library.

**Example:**
```rust
let deps = translator.get_cargo_dependencies(&PythonLibrary::Requests);
// Returns: vec!["reqwest = { version = \"0.11\", features = [\"json\"] }"]
```

### NumPyTranslator

Translates NumPy operations to ndarray.

#### Methods

##### `translate_operation(&mut self, op: &NumPyOp, args: &[String]) -> String`

**Operations:**
```rust
pub enum NumPyOp {
    Array,
    Zeros,
    Ones,
    Arange,
    Linspace,
    Sum,
    Mean,
    Std,
    Dot,
    Transpose,
}
```

### PandasTranslator

Translates Pandas operations to Polars.

#### Methods

##### `translate_dataframe_op(&mut self, op: &DataFrameOp, args: &[String]) -> String`

**Operations:**
```rust
pub enum DataFrameOp {
    ReadCsv,
    Filter,
    GroupBy,
    Sort,
    Select,
    WithColumn,
    Join,
}
```

---

## WASM Bundler

Generates optimized WASM bundles.

### Struct

```rust
pub struct WasmBundler {
    config: BundleConfig,
}
```

### BundleConfig

```rust
pub struct BundleConfig {
    pub package_name: String,
    pub output_dir: String,
    pub target: DeploymentTarget,
    pub optimization_level: OptimizationLevel,
    pub optimize_size: bool,
    pub code_splitting: bool,
    pub compression: CompressionFormat,
    pub generate_readme: bool,
    pub generate_package_json: bool,
}
```

### Methods

#### `new(config: BundleConfig) -> Self`

Creates a new WASM bundler.

```rust
let bundler = WasmBundler::new(config);
```

#### `production() -> BundleConfig`

Returns production-optimized configuration.

```rust
let config = BundleConfig::production();
```

#### `development() -> BundleConfig`

Returns development-friendly configuration.

```rust
let config = BundleConfig::development();
```

#### `cdn_optimized() -> BundleConfig`

Returns CDN-optimized configuration (maximum compression).

```rust
let config = BundleConfig::cdn_optimized();
```

#### `generate_bundle(&self, project_name: &str) -> String`

Generates a complete WASM bundle.

**Parameters:**
- `project_name: &str` - Project name

**Returns:** Status report as `String`

**Example:**
```rust
let report = bundler.generate_bundle("my_app");
println!("{}", report);
```

#### `generate_wasm_opt_command(&self, input: &str, output: &str) -> String`

Generates wasm-opt optimization command.

**Parameters:**
- `input: &str` - Input WASM file
- `output: &str` - Output WASM file

**Returns:** Shell command as `String`

**Example:**
```rust
let cmd = bundler.generate_wasm_opt_command("input.wasm", "output.wasm");
// Output: "wasm-opt -Ozz --vacuum --dce input.wasm -o output.wasm"
```

### Enums

#### DeploymentTarget

```rust
pub enum DeploymentTarget {
    Web,        // Browser with ES modules
    NodeJs,     // Node.js environment
    Bundler,    // Webpack/Rollup
    Deno,       // Deno runtime
    NoModules,  // Legacy browsers
}
```

#### OptimizationLevel

```rust
pub enum OptimizationLevel {
    None,       // -O0
    Basic,      // -O1
    Standard,   // -O2
    Aggressive, // -O3
    Size,       // -Os
    MaxSize,    // -Ozz
}

impl OptimizationLevel {
    pub fn wasm_opt_flag(&self) -> &str;
}
```

#### CompressionFormat

```rust
pub enum CompressionFormat {
    None,
    Gzip,
    Brotli,
    Both,
}
```

---

## Dead Code Eliminator

Removes unused code using call graph analysis.

### Struct

```rust
pub struct DeadCodeEliminator {
    // Internal state
}
```

### Methods

#### `new() -> Self`

Creates a new dead code eliminator.

```rust
let eliminator = DeadCodeEliminator::new();
```

#### `analyze(&mut self, code: &str) -> DeadCodeReport`

Analyzes code for dead code.

**Parameters:**
- `code: &str` - Rust source code

**Returns:** `DeadCodeReport` with analysis results

**Example:**
```rust
let report = eliminator.analyze(rust_code);
println!("Dead functions: {:?}", report.dead_functions);
```

#### `analyze_with_strategy(&mut self, code: &str, strategy: OptimizationStrategy) -> String`

Removes dead code using a specific strategy.

**Parameters:**
- `code: &str` - Rust source code
- `strategy: OptimizationStrategy` - Optimization strategy

**Returns:** Optimized Rust code

**Example:**
```rust
let optimized = eliminator.analyze_with_strategy(
    rust_code,
    OptimizationStrategy::Aggressive
);
```

#### `build_call_graph(&mut self, code: &str) -> CallGraph`

Builds a call graph from code.

**Parameters:**
- `code: &str` - Rust source code

**Returns:** `CallGraph`

**Example:**
```rust
let graph = eliminator.build_call_graph(rust_code);
let reachable = graph.compute_reachable();
```

### CallGraph

```rust
pub struct CallGraph {
    pub nodes: HashMap<String, CallGraphNode>,
    pub edges: HashMap<String, Vec<String>>,
    pub entry_points: HashSet<String>,
}

impl CallGraph {
    pub fn compute_reachable(&self) -> HashSet<String>;
    pub fn get_dead_code(&self) -> Vec<String>;
}
```

### OptimizationStrategy

```rust
pub enum OptimizationStrategy {
    Conservative,  // Only remove clearly unused private code
    Moderate,      // Remove private and pub(crate) unused code
    Aggressive,    // Remove ALL unreachable code
}
```

### DeadCodeReport

```rust
pub struct DeadCodeReport {
    pub total_functions: usize,
    pub dead_functions: Vec<String>,
    pub live_functions: Vec<String>,
    pub size_reduction: f64,  // Percentage
}
```

---

## Cargo Generator

Generates Cargo.toml files with dependencies and metadata.

### Struct

```rust
pub struct CargoGenerator {
    config: CargoConfig,
}
```

### CargoConfig

```rust
pub struct CargoConfig {
    pub package_name: String,
    pub version: String,
    pub authors: Vec<String>,
    pub edition: String,
    pub description: Option<String>,
    pub license: Option<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub documentation: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub rust_version: Option<String>,
    pub dependencies: Vec<String>,
    pub is_async: bool,
    pub http_client: bool,
    pub wasm_target: bool,
    pub generate_binary: bool,
    pub generate_benchmarks: bool,
}

impl Default for CargoConfig {
    fn default() -> Self {
        CargoConfig {
            package_name: "my_project".to_string(),
            version: "0.1.0".to_string(),
            authors: vec![],
            edition: "2021".to_string(),
            description: None,
            license: Some("MIT".to_string()),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec![],
            categories: vec![],
            rust_version: None,
            dependencies: vec![],
            is_async: false,
            http_client: false,
            wasm_target: false,
            generate_binary: false,
            generate_benchmarks: false,
        }
    }
}
```

### Methods

#### `new(config: CargoConfig) -> Self`

Creates a new Cargo generator.

```rust
let generator = CargoGenerator::new(config);
```

#### `generate(&self) -> String`

Generates Cargo.toml content.

**Returns:** Cargo.toml as `String`

**Example:**
```rust
let cargo_toml = generator.generate();
std::fs::write("Cargo.toml", cargo_toml)?;
```

#### Builder Methods

```rust
impl CargoGenerator {
    pub fn with_description(mut self, desc: String) -> Self;
    pub fn with_license(mut self, license: String) -> Self;
    pub fn with_repository(mut self, repo: String) -> Self;
    pub fn with_homepage(mut self, homepage: String) -> Self;
    pub fn with_documentation(mut self, docs: String) -> Self;
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self;
    pub fn with_categories(mut self, categories: Vec<String>) -> Self;
    pub fn with_rust_version(mut self, version: String) -> Self;
    pub fn with_binary(mut self, enable: bool) -> Self;
    pub fn with_benchmarks(mut self, enable: bool) -> Self;
    pub fn with_async_runtime(mut self, enable: bool) -> Self;
    pub fn with_http_client(mut self, enable: bool) -> Self;
    pub fn with_wasm_target(mut self, enable: bool) -> Self;
}
```

**Example:**
```rust
let generator = CargoGenerator::new(CargoConfig::default())
    .with_description("My awesome project".to_string())
    .with_license("MIT".to_string())
    .with_repository("https://github.com/user/repo".to_string())
    .with_async_runtime(true)
    .with_http_client(true);

let cargo_toml = generator.generate();
```

---

## Dependency Resolver

Resolves and manages Rust dependencies.

### Struct

```rust
pub struct DependencyResolver {
    // Dependency graph
}
```

### Methods

#### `new() -> Self`

Creates a new dependency resolver.

```rust
let resolver = DependencyResolver::new();
```

#### `add_dependency(&mut self, name: &str, constraint: VersionConstraint)`

Adds a dependency with version constraint.

**Parameters:**
- `name: &str` - Crate name
- `constraint: VersionConstraint` - Version constraint

**Example:**
```rust
resolver.add_dependency("tokio", VersionConstraint::Caret("1.0"));
resolver.add_dependency("serde", VersionConstraint::Exact("1.0.195"));
```

#### `resolve(&self) -> Result<Vec<ResolvedDependency>, ResolverError>`

Resolves all dependencies.

**Returns:** `Result<Vec<ResolvedDependency>, ResolverError>`

**Example:**
```rust
let resolved = resolver.resolve()?;
for dep in resolved {
    println!("{} = \"{}\"", dep.name, dep.version);
}
```

### VersionConstraint

```rust
pub enum VersionConstraint {
    Exact(String),      // "1.0.0"
    Caret(String),      // "^1.0" (compatible)
    Tilde(String),      // "~1.0" (patch updates)
    Wildcard(String),   // "1.*"
    Range(String, String), // ">= 1.0, < 2.0"
}
```

### ResolvedDependency

```rust
pub struct ResolvedDependency {
    pub name: String,
    pub version: String,
    pub features: Vec<String>,
    pub optional: bool,
}
```

---

## Import Analyzer

Analyzes Python imports and maps to Rust dependencies.

### Struct

```rust
pub struct ImportAnalyzer {
    // Import tracking
}
```

### Methods

#### `analyze(&mut self, code: &str) -> ImportAnalysis`

Analyzes all imports in Python code.

**Parameters:**
- `code: &str` - Python source code

**Returns:** `ImportAnalysis`

**Example:**
```rust
let analyzer = ImportAnalyzer::new();
let analysis = analyzer.analyze(python_code);

for import in analysis.imports {
    println!("Import: {} → {}", import.python_module, import.rust_crate);
}
```

### ImportAnalysis

```rust
pub struct ImportAnalysis {
    pub imports: Vec<ImportInfo>,
    pub cargo_dependencies: Vec<String>,
    pub unsupported: Vec<String>,
}

pub struct ImportInfo {
    pub python_module: String,
    pub rust_crate: String,
    pub items: Vec<String>,
    pub alias: Option<String>,
}
```

---

## Error Types

### TranspilerError

```rust
pub enum TranspilerError {
    ParseError(String),
    TypeError(String),
    UnsupportedFeature(String),
    ImportError(String),
    GenerationError(String),
}

impl std::fmt::Display for TranspilerError;
impl std::error::Error for TranspilerError;
```

### Usage

```rust
use portalis_transpiler::{PyToRustTranspiler, TranspilerError};

fn translate_code(python: &str) -> Result<String, TranspilerError> {
    let mut transpiler = PyToRustTranspiler::new();
    Ok(transpiler.translate(python))
}
```

---

## Complete Example

Combining multiple APIs:

```rust
use portalis_transpiler::{
    PyToRustTranspiler,
    CargoGenerator,
    CargoConfig,
    WasmBundler,
    BundleConfig,
    DeploymentTarget,
    DeadCodeEliminator,
    OptimizationStrategy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Translate Python to Rust
    let python_code = std::fs::read_to_string("my_script.py")?;
    let mut transpiler = PyToRustTranspiler::new();
    let rust_code = transpiler.translate(&python_code);

    // 2. Optimize (remove dead code)
    let mut eliminator = DeadCodeEliminator::new();
    let optimized = eliminator.analyze_with_strategy(
        &rust_code,
        OptimizationStrategy::Aggressive
    );

    // 3. Write Rust source
    std::fs::write("src/lib.rs", optimized)?;

    // 4. Generate Cargo.toml
    let cargo_config = CargoConfig::default();
    let generator = CargoGenerator::new(cargo_config)
        .with_description("My WASM app".to_string())
        .with_async_runtime(true)
        .with_wasm_target(true);

    std::fs::write("Cargo.toml", generator.generate())?;

    // 5. Build WASM bundle
    let mut bundle_config = BundleConfig::production();
    bundle_config.target = DeploymentTarget::Web;

    let bundler = WasmBundler::new(bundle_config);
    let report = bundler.generate_bundle("my_app");

    println!("{}", report);
    Ok(())
}
```

---

## Feature Flags

Optional features available via Cargo:

```toml
[dependencies]
portalis-transpiler = { version = "1.0", features = ["full"] }
```

Available features:
- `async-tokio` - Tokio async runtime support
- `async-async-std` - async-std runtime support
- `numpy` - NumPy translation
- `pandas` - Pandas translation
- `wasm` - WASM bundling and optimization
- `optimization` - Dead code elimination
- `full` - All features

---

For more examples, see [EXAMPLES.md](./EXAMPLES.md).
For usage patterns, see [USER_GUIDE.md](./USER_GUIDE.md).
