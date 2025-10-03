# Testing Strategy: London School TDD for Python→Rust→WASM Platform

## 1. Overall Test Philosophy

### London School TDD Application

This testing strategy applies **London School (Mockist) TDD** principles to the Python→Rust→WASM translation pipeline:

**Core Principles:**
- **Outside-in Development**: Start with user-facing acceptance tests (Script Mode, Library Mode) and work inward
- **Mockist Testing**: Test each component in isolation using mocks/stubs for dependencies
- **Behavior Verification**: Focus on interactions and contracts rather than internal state
- **Test-First Design**: Write tests before implementation to drive API design
- **Clear Boundaries**: Establish well-defined interfaces between agents and components

**Strategic Benefits:**
1. **Early Design Feedback**: Tests reveal architectural issues before implementation
2. **Component Isolation**: Each agent/module can be developed and tested independently
3. **Parallel Development**: Teams can work on different components simultaneously
4. **Regression Protection**: Behavioral contracts prevent breaking changes
5. **Documentation**: Tests serve as executable specifications

### Test Pyramid Structure

```
                    /\
                   /  \  E2E Tests (Script/Library Mode)
                  /____\
                 /      \
                / Integration \ (NVIDIA Stack, Multi-Agent)
               /____________\
              /              \
             /  Unit Tests    \ (Individual Agents, Pure Functions)
            /__________________\
```

**Distribution:**
- 70% Unit Tests (fast, isolated, mockist)
- 20% Integration Tests (component boundaries)
- 10% E2E Tests (full pipeline validation)

---

## 2. Acceptance Test Scenarios

### 2.1 Script Mode Acceptance Tests

**User Story**: As a developer, I want to convert a single Python script to WASM so I can deploy it as a portable microservice.

**Acceptance Criteria:**

#### AC-SCRIPT-001: Simple Pure Function Translation
```gherkin
Given a Python script "fibonacci.py" containing a pure function
When I run the pipeline in Script Mode
Then I receive a WASM module "fibonacci.wasm"
And the WASM module exports the same function signature
And executing the WASM function with input 10 returns the same output as Python
And the translation completes in < 30 seconds
```

**Test Data:**
```python
# fibonacci.py
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Golden Vectors:**
```
Input: 0 → Output: 0
Input: 1 → Output: 1
Input: 10 → Output: 55
Input: 20 → Output: 6765
```

---

#### AC-SCRIPT-002: Script with External Dependencies
```gherkin
Given a Python script "data_processor.py" using numpy/pandas
When I run the pipeline in Script Mode
Then the dependency analyzer identifies external libraries
And the pipeline generates Rust equivalents using ndarray/polars
And the WASM module includes all necessary dependencies
And conformance tests validate numerical parity within 1e-6 tolerance
```

**Test Data:**
```python
# data_processor.py
import numpy as np

def normalize_array(data: list[float]) -> list[float]:
    arr = np.array(data)
    return ((arr - arr.mean()) / arr.std()).tolist()
```

**Golden Vectors:**
```
Input: [1.0, 2.0, 3.0, 4.0, 5.0]
Output: [-1.414, -0.707, 0.0, 0.707, 1.414] (±1e-6)
```

---

#### AC-SCRIPT-003: NIM Service Deployment
```gherkin
Given a successfully translated WASM module
When I request NIM packaging
Then a Docker container is created with Triton configuration
And the container exposes HTTP/gRPC endpoints
And health checks pass
And the service responds to inference requests in < 100ms (p95)
```

---

### 2.2 Library Mode Acceptance Tests

**User Story**: As a team lead, I want to convert an entire Python library to Rust/WASM to achieve performance and portability for enterprise deployment.

#### AC-LIB-001: Multi-Module Library Translation
```gherkin
Given a Python library "mylib" with 3 modules and 50 functions
When I run the pipeline in Library Mode
Then a Rust workspace is generated with corresponding crates
And each Python module maps to a Rust module
And all public APIs are preserved in the Rust interface
And an API parity report shows 100% coverage
```

**Test Structure:**
```
mylib/
├── core.py          → mylib-core (crate)
├── utils.py         → mylib-utils (crate)
└── algorithms.py    → mylib-algorithms (crate)
```

---

#### AC-LIB-002: Test Suite Translation
```gherkin
Given a Python library with pytest test suite (100 tests)
When the pipeline translates tests to Rust
Then Rust tests are generated using #[test] and proptest
And at least 95% of Python tests have Rust equivalents
And all generated Rust tests pass
And a test coverage report is generated
```

---

#### AC-LIB-003: Performance Benchmarking
```gherkin
Given a translated library in Rust/WASM
When performance benchmarks are executed
Then the Rust implementation is at least as fast as Python
And a performance report compares execution times
And memory usage is documented
And the report identifies optimization opportunities
```

**Benchmark Categories:**
- CPU-bound operations (target: 5-10x faster than Python)
- I/O-bound operations (target: parity with Python)
- Memory allocation (target: 50% reduction)

---

## 3. Unit Test Strategy per Component

### 3.1 Ingest & Analyze Agents

**Component**: `IngestionAgent`, `DependencyAnalyzer`, `APIExtractor`

**Testing Approach:**
```rust
// Test doubles for dependencies
trait FileSystem {
    fn read_file(&self, path: &str) -> Result<String, Error>;
}

struct MockFileSystem {
    files: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingestion_agent_parses_simple_python_file() {
        // Arrange
        let mut mock_fs = MockFileSystem::new();
        mock_fs.add_file("test.py", "def add(a, b): return a + b");
        let agent = IngestionAgent::new(Box::new(mock_fs));

        // Act
        let ast = agent.parse("test.py").unwrap();

        // Assert
        assert_eq!(ast.functions.len(), 1);
        assert_eq!(ast.functions[0].name, "add");
        assert_eq!(ast.functions[0].params.len(), 2);
    }

    #[test]
    fn dependency_analyzer_detects_stdlib_imports() {
        // Arrange
        let analyzer = DependencyAnalyzer::new();
        let code = "import os\nimport sys";

        // Act
        let deps = analyzer.analyze(code).unwrap();

        // Assert
        assert!(deps.stdlib.contains(&"os"));
        assert!(deps.stdlib.contains(&"sys"));
        assert!(deps.external.is_empty());
    }

    #[test]
    fn api_extractor_builds_function_signature() {
        // Arrange
        let extractor = APIExtractor::new();
        let ast = parse_python("def greet(name: str) -> str: pass");

        // Act
        let signature = extractor.extract_signature(&ast).unwrap();

        // Assert
        assert_eq!(signature.name, "greet");
        assert_eq!(signature.params[0].type_annotation, "str");
        assert_eq!(signature.return_type, "str");
    }
}
```

**Mock Boundaries:**
- File system access
- Network calls (for package metadata)
- External process execution (AST parsers)

---

### 3.2 Spec Generator (NeMo Integration)

**Component**: `SpecGeneratorAgent`, `NeMoClient`, `RustTypeMapper`

**Testing Approach:**
```rust
trait LLMClient {
    fn generate(&self, prompt: &str) -> Result<String, Error>;
}

struct MockNeMoClient {
    responses: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn spec_generator_creates_rust_struct_from_python_class() {
        // Arrange
        let mut mock_nemo = MockNeMoClient::new();
        mock_nemo.set_response(
            "generate_struct",
            "struct User { name: String, age: u32 }"
        );
        let generator = SpecGeneratorAgent::new(Box::new(mock_nemo));
        let python_class = "class User:\n    def __init__(self, name: str, age: int)";

        // Act
        let rust_spec = generator.generate_spec(python_class).unwrap();

        // Assert
        assert!(rust_spec.contains("struct User"));
        assert!(rust_spec.contains("name: String"));
        assert!(rust_spec.contains("age: u32"));
    }

    #[test]
    fn type_mapper_converts_python_types_to_rust() {
        let mapper = RustTypeMapper::new();

        assert_eq!(mapper.map("int"), "i64");
        assert_eq!(mapper.map("str"), "String");
        assert_eq!(mapper.map("list[int]"), "Vec<i64>");
        assert_eq!(mapper.map("dict[str, int]"), "HashMap<String, i64>");
    }

    #[test]
    fn spec_generator_handles_nemo_timeout() {
        let mut mock_nemo = MockNeMoClient::new();
        mock_nemo.set_timeout_mode(true);
        let generator = SpecGeneratorAgent::new(Box::new(mock_nemo));

        let result = generator.generate_spec("class Test: pass");

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::Timeout));
    }
}
```

**Property-Based Tests:**
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn type_mapper_preserves_collection_nesting(depth in 1..5usize) {
        let nested_type = build_nested_list("int", depth);
        let rust_type = RustTypeMapper::new().map(&nested_type);

        // Property: nesting depth is preserved
        assert_eq!(count_angle_brackets(&rust_type), depth);
    }
}
```

---

### 3.3 Transpiler Agents

**Component**: `FunctionTranspiler`, `ClassTranspiler`, `WABIGenerator`

**Testing Approach:**
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn function_transpiler_converts_simple_function() {
        // Arrange
        let transpiler = FunctionTranspiler::new();
        let python_fn = PythonFunction {
            name: "add",
            params: vec![("a", "int"), ("b", "int")],
            body: "return a + b",
            return_type: "int",
        };

        // Act
        let rust_fn = transpiler.transpile(&python_fn).unwrap();

        // Assert
        assert!(rust_fn.contains("fn add(a: i64, b: i64) -> i64"));
        assert!(rust_fn.contains("a + b"));
    }

    #[test]
    fn class_transpiler_generates_impl_blocks() {
        let transpiler = ClassTranspiler::new();
        let python_class = PythonClass {
            name: "Counter",
            methods: vec![
                ("increment", vec![], "self.count += 1"),
                ("get", vec![], "return self.count"),
            ],
            fields: vec![("count", "int")],
        };

        let rust_code = transpiler.transpile(&python_class).unwrap();

        assert!(rust_code.contains("struct Counter"));
        assert!(rust_code.contains("impl Counter"));
        assert!(rust_code.contains("fn increment(&mut self)"));
        assert!(rust_code.contains("fn get(&self) -> i64"));
    }

    #[test]
    fn wabi_generator_creates_wasm_exports() {
        let generator = WABIGenerator::new();
        let function = RustFunction {
            name: "process",
            signature: "fn process(data: Vec<u8>) -> Vec<u8>",
        };

        let wasm_export = generator.generate_export(&function).unwrap();

        assert!(wasm_export.contains("#[no_mangle]"));
        assert!(wasm_export.contains("pub extern \"C\" fn process"));
        assert!(wasm_export.contains("wasm_bindgen"));
    }
}
```

**Mutation Testing:**
Use `cargo-mutants` to ensure tests catch subtle errors:
```bash
cargo mutants --test-threads 4
```

---

### 3.4 CUDA-Accelerated Engines

**Component**: `CUDAParser`, `EmbeddingSimilarity`, `TestPrioritizer`

**Testing Approach:**
```rust
// Mock GPU computation for fast testing
trait GPUCompute {
    fn parallel_parse(&self, sources: Vec<String>) -> Vec<AST>;
    fn compute_embeddings(&self, texts: Vec<String>) -> Vec<Vec<f32>>;
}

struct MockGPUCompute {
    parse_delay: Duration,
}

#[cfg(test)]
mod tests {
    #[test]
    fn cuda_parser_processes_files_in_parallel() {
        let mock_gpu = MockGPUCompute::new();
        let parser = CUDAParser::new(Box::new(mock_gpu));
        let files = vec!["file1.py", "file2.py", "file3.py"];

        let start = Instant::now();
        let asts = parser.parse_batch(files).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(asts.len(), 3);
        // Should be faster than sequential (3x parse_delay)
        assert!(elapsed < Duration::from_millis(300));
    }

    #[test]
    fn embedding_similarity_ranks_translations() {
        let similarity = EmbeddingSimilarity::new();
        let original = "Calculate the sum of two numbers";
        let candidates = vec![
            "fn add(a: i64, b: i64) -> i64 { a + b }",
            "fn multiply(a: i64, b: i64) -> i64 { a * b }",
            "fn sum(x: i64, y: i64) -> i64 { x + y }",
        ];

        let ranked = similarity.rank(original, candidates).unwrap();

        // Should rank correct implementations higher
        assert!(ranked[0].score > 0.8);
        assert_eq!(ranked[0].candidate, candidates[0]);
    }
}
```

**Integration Tests with Real GPU (CI/CD):**
```rust
#[cfg(feature = "cuda-integration")]
mod cuda_integration_tests {
    #[test]
    #[ignore] // Only run in CI with GPU
    fn real_cuda_acceleration_benchmark() {
        let parser = CUDAParser::with_real_gpu();
        let large_codebase = load_test_files(1000);

        let start = Instant::now();
        let results = parser.parse_batch(large_codebase).unwrap();
        let elapsed = start.elapsed();

        // Should complete 1000 files in < 10 seconds
        assert!(elapsed < Duration::from_secs(10));
        assert_eq!(results.len(), 1000);
    }
}
```

---

### 3.5 Build & Test Agents

**Component**: `RustWorkspaceBuilder`, `TestTranslator`, `BenchmarkRunner`

**Testing Approach:**
```rust
trait BuildTool {
    fn compile(&self, workspace: &Path) -> Result<CompileResult, Error>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn workspace_builder_creates_cargo_toml() {
        let builder = RustWorkspaceBuilder::new();
        let spec = WorkspaceSpec {
            name: "mylib",
            crates: vec!["core", "utils"],
        };

        let workspace = builder.build(&spec).unwrap();

        assert!(workspace.cargo_toml.exists());
        assert!(workspace.crates.contains_key("core"));
        assert!(workspace.crates.contains_key("utils"));
    }

    #[test]
    fn test_translator_converts_pytest_to_rust() {
        let translator = TestTranslator::new();
        let pytest = r#"
def test_addition():
    assert add(2, 3) == 5
"#;

        let rust_test = translator.translate(pytest).unwrap();

        assert!(rust_test.contains("#[test]"));
        assert!(rust_test.contains("fn test_addition()"));
        assert!(rust_test.contains("assert_eq!(add(2, 3), 5)"));
    }

    #[test]
    fn benchmark_runner_executes_criterion_benches() {
        let mut mock_build = MockBuildTool::new();
        mock_build.set_bench_result(BenchResult {
            name: "fibonacci_10",
            time: Duration::from_micros(42),
        });

        let runner = BenchmarkRunner::new(Box::new(mock_build));
        let results = runner.run_benchmarks("./workspace").unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].name, "fibonacci_10");
    }
}
```

---

### 3.6 Packaging Agents

**Component**: `WASMCompiler`, `NIMBuilder`, `TritonRegistrar`

**Testing Approach:**
```rust
trait ContainerRuntime {
    fn build_image(&self, dockerfile: &Path) -> Result<ImageId, Error>;
    fn run_container(&self, image: &ImageId) -> Result<ContainerId, Error>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn wasm_compiler_produces_valid_wasm() {
        let compiler = WASMCompiler::new();
        let rust_code = "pub fn add(a: i32, b: i32) -> i32 { a + b }";

        let wasm_bytes = compiler.compile_to_wasm(rust_code).unwrap();

        // Validate WASM magic number
        assert_eq!(&wasm_bytes[0..4], b"\0asm");
        // Validate version
        assert_eq!(&wasm_bytes[4..8], &[0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn nim_builder_creates_docker_image() {
        let mut mock_runtime = MockContainerRuntime::new();
        mock_runtime.expect_build().returning(|_| Ok(ImageId("test-123")));

        let builder = NIMBuilder::new(Box::new(mock_runtime));
        let config = NIMConfig {
            name: "fibonacci-service",
            wasm_path: "./fibonacci.wasm",
        };

        let image = builder.build(&config).unwrap();

        assert_eq!(image.id, "test-123");
    }

    #[test]
    fn triton_registrar_registers_endpoint() {
        let mut mock_triton = MockTritonClient::new();
        mock_triton.expect_register().returning(|_| Ok(Endpoint {
            url: "http://triton:8000/v2/models/fibonacci",
        }));

        let registrar = TritonRegistrar::new(Box::new(mock_triton));
        let endpoint = registrar.register("fibonacci-service").unwrap();

        assert!(endpoint.url.contains("/v2/models/fibonacci"));
    }
}
```

---

## 4. Integration Test Plan

### 4.1 Multi-Agent Integration

**Test Scenario**: Ingest → Spec → Transpile Pipeline

```rust
#[test]
#[ignore] // Integration test
fn full_python_to_rust_pipeline() {
    // Arrange
    let pipeline = Pipeline::new()
        .with_ingestion()
        .with_spec_generation()
        .with_transpilation();

    let python_code = r#"
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"#;

    // Act
    let rust_code = pipeline.execute(python_code).unwrap();

    // Assert
    assert!(rust_code.contains("fn factorial(n: i64) -> i64"));

    // Compile and verify
    let compiled = compile_rust(&rust_code).unwrap();
    assert!(compiled.compiles);
}
```

### 4.2 NVIDIA Stack Integration

**Test Categories:**

#### NeMo LLM Integration
```rust
#[test]
#[cfg(feature = "nemo-integration")]
fn nemo_generates_valid_rust_code() {
    let nemo_client = NeMoClient::connect(NEMO_ENDPOINT).unwrap();
    let prompt = build_translation_prompt("def add(a, b): return a + b");

    let response = nemo_client.generate(prompt, GenerationConfig::default()).unwrap();

    // Validate syntactic correctness
    assert!(syn::parse_file(&response).is_ok());
    // Validate semantic correctness
    assert!(response.contains("fn add"));
}
```

#### CUDA Acceleration Integration
```rust
#[test]
#[cfg(feature = "cuda-integration")]
fn cuda_embedding_computation() {
    let cuda_engine = CUDAEmbeddingEngine::new().unwrap();
    let texts = vec![
        "Calculate sum",
        "Compute product",
        "Find maximum",
    ];

    let embeddings = cuda_engine.embed(texts).unwrap();

    assert_eq!(embeddings.len(), 3);
    assert_eq!(embeddings[0].len(), 768); // Embedding dimension
}
```

#### Triton Deployment Integration
```rust
#[test]
#[cfg(feature = "triton-integration")]
fn deploy_and_query_triton_model() {
    let triton = TritonClient::connect(TRITON_URL).unwrap();
    let model_path = "./models/fibonacci.wasm";

    // Deploy
    triton.load_model("fibonacci", model_path).unwrap();

    // Query
    let input = json!({"n": 10});
    let response = triton.infer("fibonacci", input).unwrap();

    assert_eq!(response["result"], 55);
}
```

### 4.3 Contract Testing

**Producer-Consumer Contracts:**

```rust
// Producer: SpecGeneratorAgent
// Consumer: TranspilerAgent
#[test]
fn spec_generator_produces_valid_contract() {
    let spec = SpecGeneratorAgent::new(mock_nemo())
        .generate_spec(SAMPLE_PYTHON)
        .unwrap();

    // Validate contract schema
    assert!(spec.validate_schema().is_ok());

    // Consumer validates it can process
    let transpiler = TranspilerAgent::new();
    assert!(transpiler.can_process(&spec));
}
```

---

## 5. Conformance Testing Approach

### 5.1 Python vs Rust Parity

**Goal**: Ensure translated Rust code produces identical outputs to Python

**Strategy:**

#### Golden Vector Testing
```rust
struct GoldenTest {
    input: serde_json::Value,
    expected_output: serde_json::Value,
    tolerance: Option<f64>, // For floating-point comparisons
}

fn load_golden_vectors(path: &str) -> Vec<GoldenTest> {
    // Load from JSON/YAML file
}

#[test]
fn conformance_test_fibonacci() {
    let golden_tests = load_golden_vectors("tests/golden/fibonacci.json");

    for test in golden_tests {
        let python_result = execute_python("fibonacci.py", &test.input);
        let rust_result = execute_rust("fibonacci.wasm", &test.input);

        assert_eq!(python_result, rust_result);
    }
}
```

#### Differential Testing
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn differential_test_normalize_array(
        data in prop::collection::vec(any::<f64>(), 1..100)
    ) {
        let python_output = run_python_normalize(&data);
        let rust_output = run_rust_normalize(&data);

        // Compare with tolerance
        for (py, rs) in python_output.iter().zip(rust_output.iter()) {
            assert!((py - rs).abs() < 1e-6);
        }
    }
}
```

#### Equivalence Classes
```
Input Partitions:
1. Empty collections
2. Single-element collections
3. Small collections (2-10 elements)
4. Large collections (1000+ elements)
5. Edge cases (None, NaN, Infinity for numeric types)
6. Boundary values (min/max for integers)
```

### 5.2 API Surface Parity

**Parity Report Generator:**
```rust
struct ParityReport {
    total_functions: usize,
    translated: usize,
    skipped: Vec<(String, String)>, // (name, reason)
    signature_mismatches: Vec<SignatureMismatch>,
}

fn generate_parity_report(
    python_api: &APISpec,
    rust_api: &APISpec,
) -> ParityReport {
    // Compare public APIs
    let mut report = ParityReport::default();

    for py_fn in &python_api.functions {
        if let Some(rs_fn) = rust_api.find_function(&py_fn.name) {
            if py_fn.signature != rs_fn.signature {
                report.signature_mismatches.push(/* ... */);
            }
            report.translated += 1;
        } else {
            report.skipped.push((py_fn.name.clone(), "Not translated".into()));
        }
    }

    report.total_functions = python_api.functions.len();
    report
}

#[test]
fn library_mode_achieves_95_percent_parity() {
    let python_api = extract_python_api("./mylib");
    let rust_api = extract_rust_api("./rust-workspace");

    let report = generate_parity_report(&python_api, &rust_api);

    let coverage = (report.translated as f64) / (report.total_functions as f64);
    assert!(coverage >= 0.95, "API coverage: {:.2}%", coverage * 100.0);
}
```

---

## 6. Performance Testing Strategy

### 6.1 Benchmark Categories

#### Microbenchmarks (Criterion.rs)
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_fibonacci(c: &mut Criterion) {
    c.bench_function("fibonacci_10", |b| {
        b.iter(|| fibonacci(black_box(10)))
    });

    c.bench_function("fibonacci_20", |b| {
        b.iter(|| fibonacci(black_box(20)))
    });
}

criterion_group!(benches, bench_fibonacci);
criterion_main!(benches);
```

#### Comparative Benchmarks (Python vs Rust)
```yaml
# benchmarks.yaml
benchmarks:
  - name: fibonacci
    scenarios:
      - input: 10
        python_baseline: 5.2µs
        rust_target: "<1µs"
      - input: 20
        python_baseline: 520µs
        rust_target: "<100µs"

  - name: matrix_multiply
    scenarios:
      - input: {size: 100}
        python_baseline: 12ms
        rust_target: "<2ms"
```

#### Load Testing (Triton/NIM)
```rust
use goose::prelude::*;

async fn triton_inference(user: &mut GooseUser) -> TransactionResult {
    let request = json!({
        "inputs": [{"name": "input", "data": [1, 2, 3]}]
    });

    user.post("/v2/models/fibonacci/infer", &request).await?;
    Ok(())
}

#[tokio::test]
async fn load_test_triton_endpoint() {
    GooseAttack::initialize()?
        .register_scenario(
            scenario!("InferenceLoad")
                .register_transaction(transaction!(triton_inference).set_weight(10)?)
        )
        .execute()
        .await?;
}
```

### 6.2 Performance Regression Detection

**CI/CD Integration:**
```yaml
# .github/workflows/performance.yml
name: Performance Regression

on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest-gpu
    steps:
      - uses: actions/checkout@v2
      - run: cargo bench --bench=all -- --save-baseline=pr-${{ github.event.number }}
      - run: |
          cargo bench --bench=all -- --baseline=main
          if [ $? -ne 0 ]; then
            echo "Performance regression detected!"
            exit 1
          fi
```

**Performance Budget:**
```toml
# performance_budget.toml
[budgets]
fibonacci_10 = { max_time = "1µs", max_memory = "1KB" }
matrix_multiply_100 = { max_time = "5ms", max_memory = "100KB" }
full_pipeline_script = { max_time = "30s", max_memory = "500MB" }
```

---

## 7. Test Tooling and Infrastructure

### 7.1 Test Frameworks

**Rust:**
- **Unit Tests**: Built-in `#[test]`
- **Property Tests**: `proptest`, `quickcheck`
- **Benchmarks**: `criterion`
- **Mocking**: `mockall`, `mockito`
- **Mutation Testing**: `cargo-mutants`
- **Coverage**: `cargo-tarpaulin`

**Python:**
- **Unit Tests**: `pytest`
- **Property Tests**: `hypothesis`
- **Benchmarks**: `pytest-benchmark`
- **Mocking**: `unittest.mock`

**Integration:**
- **Contract Tests**: `pact`
- **Load Tests**: `goose` (Rust), `locust` (Python)
- **E2E Tests**: Custom test harness

### 7.2 Test Data Management

**Golden Vectors Repository:**
```
tests/
├── golden/
│   ├── fibonacci.json
│   ├── matrix_ops.json
│   └── data_processing.json
├── fixtures/
│   ├── sample_scripts/
│   └── sample_libraries/
└── mocks/
    ├── nemo_responses.json
    └── triton_responses.json
```

**Test Data Generator:**
```rust
struct TestDataGenerator {
    seed: u64,
}

impl TestDataGenerator {
    fn generate_numeric_array(&self, size: usize) -> Vec<f64> {
        (0..size).map(|i| (i as f64) * 1.5).collect()
    }

    fn generate_python_function(&self, complexity: usize) -> String {
        // Generate random but valid Python functions
    }
}
```

### 7.3 CI/CD Pipeline

**Test Execution Strategy:**

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test --lib
      - run: pytest tests/unit
    timeout-minutes: 10

  integration-tests:
    runs-on: ubuntu-latest-gpu
    steps:
      - run: cargo test --test integration_* --features cuda-integration
    timeout-minutes: 30

  conformance-tests:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test --test conformance_*
      - run: python scripts/generate_parity_report.py
    timeout-minutes: 20

  e2e-tests:
    runs-on: ubuntu-latest-gpu
    steps:
      - run: cargo test --test e2e_* --features full-stack
    timeout-minutes: 60
```

### 7.4 Test Observability

**Metrics Collection:**
```rust
use prometheus::{Counter, Histogram, Registry};

lazy_static! {
    static ref TEST_DURATION: Histogram = Histogram::new(
        "test_duration_seconds",
        "Test execution duration"
    ).unwrap();

    static ref TEST_FAILURES: Counter = Counter::new(
        "test_failures_total",
        "Total test failures"
    ).unwrap();
}

#[test]
fn instrumented_test() {
    let _timer = TEST_DURATION.start_timer();

    // Test logic

    if result.is_err() {
        TEST_FAILURES.inc();
    }
}
```

**Test Report Dashboard:**
- Test execution times (trend over time)
- Coverage percentage per component
- Flaky test detection
- Performance regression alerts

---

## 8. Test-First Development Workflow

### 8.1 London School TDD Cycle

**Red-Green-Refactor with Mocks:**

```
1. Write Acceptance Test (Red)
   ├── Define expected behavior from user perspective
   ├── Mock all dependencies
   └── Test fails (not implemented)

2. Write Unit Tests (Red)
   ├── Break down acceptance test into component behaviors
   ├── Mock collaborators
   └── All tests fail

3. Implement Minimum Code (Green)
   ├── Make unit tests pass
   ├── Verify acceptance test progresses
   └── All tests pass

4. Refactor (Green)
   ├── Improve design without changing behavior
   ├── All tests still pass
   └── Clean up mocks if needed

5. Integration Test
   ├── Replace some mocks with real implementations
   ├── Verify component boundaries
   └── Iterate if issues found
```

### 8.2 Example Workflow: Implementing Function Transpiler

#### Step 1: Acceptance Test
```rust
#[test]
#[ignore] // Acceptance test
fn transpile_python_function_to_rust() {
    let pipeline = TranspilationPipeline::new();
    let python = "def add(a: int, b: int) -> int:\n    return a + b";

    let rust = pipeline.transpile(python).unwrap();

    assert!(rust.contains("fn add(a: i64, b: i64) -> i64"));
    assert!(compile_rust(&rust).is_ok());
}
```

#### Step 2: Unit Tests with Mocks
```rust
#[test]
fn function_transpiler_uses_type_mapper() {
    // Arrange
    let mut mock_mapper = MockTypeMapper::new();
    mock_mapper.expect_map()
        .with(eq("int"))
        .returning(|_| "i64".to_string());

    let transpiler = FunctionTranspiler::new(Box::new(mock_mapper));
    let function = PythonFunction {
        name: "add",
        params: vec![("a", "int"), ("b", "int")],
        return_type: "int",
    };

    // Act
    let result = transpiler.transpile(&function).unwrap();

    // Assert
    assert!(result.contains("i64"));
}

#[test]
fn function_transpiler_delegates_body_translation() {
    let mut mock_body_translator = MockBodyTranslator::new();
    mock_body_translator.expect_translate()
        .with(eq("return a + b"))
        .returning(|_| "a + b".to_string());

    let transpiler = FunctionTranspiler::new_with_body_translator(
        Box::new(mock_body_translator)
    );

    // Act & Assert
    let result = transpiler.transpile(&SAMPLE_FUNCTION).unwrap();
    assert!(result.contains("a + b"));
}
```

#### Step 3: Implementation
```rust
pub struct FunctionTranspiler {
    type_mapper: Box<dyn TypeMapper>,
    body_translator: Box<dyn BodyTranslator>,
}

impl FunctionTranspiler {
    pub fn transpile(&self, func: &PythonFunction) -> Result<String, Error> {
        let params = func.params.iter()
            .map(|(name, ty)| format!("{}: {}", name, self.type_mapper.map(ty)))
            .collect::<Vec<_>>()
            .join(", ");

        let return_type = self.type_mapper.map(&func.return_type);
        let body = self.body_translator.translate(&func.body)?;

        Ok(format!(
            "fn {}({}) -> {} {{ {} }}",
            func.name, params, return_type, body
        ))
    }
}
```

#### Step 4: Integration Test
```rust
#[test]
fn integration_test_real_type_mapper() {
    // Replace mock with real implementation
    let transpiler = FunctionTranspiler::new(
        Box::new(RustTypeMapper::new()),
        Box::new(SimpleBodyTranslator::new()),
    );

    let result = transpiler.transpile(&SAMPLE_FUNCTION).unwrap();

    // Verify with real compiler
    assert!(syn::parse_file(&result).is_ok());
}
```

### 8.3 Testing Checklist

**Per Feature:**
- [ ] Acceptance test written (outside-in)
- [ ] Unit tests written (mockist, isolated)
- [ ] All tests red initially
- [ ] Implementation makes tests green
- [ ] Code refactored for clarity
- [ ] Integration test validates boundaries
- [ ] Conformance test added (if applicable)
- [ ] Performance benchmark added (if applicable)
- [ ] Documentation updated
- [ ] CI pipeline passes

---

## 9. Success Metrics

### 9.1 Test Quality Metrics

**Coverage Targets:**
- Line Coverage: >80%
- Branch Coverage: >75%
- Mutation Score: >70%

**Reliability Metrics:**
- Flaky Test Rate: <2%
- Test Execution Time: <10 minutes (full suite)
- Mean Time to Detect Defect: <1 day

### 9.2 Conformance Metrics

**Parity Targets:**
- API Coverage: 100% for Script Mode, >95% for Library Mode
- Behavioral Parity: >99% (golden vector tests)
- Numerical Tolerance: <1e-6 for floating-point operations

### 9.3 Performance Metrics

**Benchmark Targets:**
- CPU-bound: 5-10x faster than Python
- I/O-bound: Parity with Python
- Memory: 50% reduction vs Python
- Latency (NIM): <100ms p95

---

## 10. Risk Mitigation Through Testing

### 10.1 Dynamic Python Semantics

**Risk**: Python's dynamic typing and runtime behavior is hard to capture in Rust

**Mitigation**:
- Extensive property-based testing with `proptest`
- Runtime tracing integration for API discovery
- Type inference validation tests
- Edge case enumeration (None, empty collections, etc.)

### 10.2 NVIDIA Stack Dependencies

**Risk**: NeMo, CUDA, Triton availability in CI/CD

**Mitigation**:
- Mock-first approach for unit tests (no GPU needed)
- Optional integration tests (`#[cfg(feature = "cuda-integration")]`)
- Nightly GPU-enabled CI runs
- Fallback to CPU implementations for development

### 10.3 Performance Regressions

**Risk**: Code changes accidentally slow down pipeline

**Mitigation**:
- Automated benchmark comparison in CI
- Performance budgets enforced
- Historical trend tracking
- Alerts on >10% regression

### 10.4 Numerical Correctness

**Risk**: Floating-point errors accumulate, causing divergence

**Mitigation**:
- Tolerance-based assertions
- Golden vector validation
- Differential testing against Python
- IEEE 754 compliance checks

---

## Conclusion

This testing strategy establishes a comprehensive, London School TDD-based approach for the Python→Rust→WASM platform. By emphasizing **mockist testing**, **outside-in development**, and **behavior verification**, we enable:

1. **Rapid iteration** through isolated component testing
2. **Parallel development** via clear interface boundaries
3. **High confidence** through acceptance and conformance testing
4. **Performance validation** through continuous benchmarking
5. **Enterprise readiness** through integration testing with NVIDIA stack

The strategy is **actionable** (specific test patterns and tools), **measurable** (clear metrics), and **aligned with TDD principles** (test-first workflow). This foundation ensures the platform can deliver on its promise of seamless Python-to-WASM translation with correctness and performance guarantees.
