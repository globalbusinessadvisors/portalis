# PORTALIS: Test Agent Pseudocode
## SPARC Phase 2: Pseudocode

**Agent:** Test Agent
**Version:** 1.0
**Date:** 2025-10-03
**Methodology:** SPARC (Specification, Pseudocode, Architecture, Refinement, Completion)
**Testing Approach:** London School TDD (Outside-In, Mockist)

---

## Table of Contents

1. [Agent Overview](#1-agent-overview)
2. [Data Structures](#2-data-structures)
3. [Core Algorithms](#3-core-algorithms)
4. [Input/Output Contracts](#4-inputoutput-contracts)
5. [Error Handling Strategy](#5-error-handling-strategy)
6. [London School TDD Test Points](#6-london-school-tdd-test-points)

---

## 1. Agent Overview

### 1.1 Purpose

The Test Agent is responsible for validating the correctness, parity, and performance of translated Rust/WASM implementations against their Python originals. It implements four primary testing strategies:

1. **Conformance Testing**: Translating Python tests to Rust, generating golden vectors
2. **Property-Based Testing**: Using quickcheck/proptest for contract validation
3. **Parity Validation**: Comparing Python vs Rust/WASM outputs for semantic equivalence
4. **Performance Benchmarking**: Measuring and comparing execution metrics

### 1.2 Functional Requirements Coverage

- **FR-2.6.1**: Conformance Testing (test translation, golden vectors, test execution, coverage)
- **FR-2.6.2**: Property-Based Testing (quickcheck/proptest, property derivation, boundary testing)
- **FR-2.6.3**: Parity Validation (equivalent workloads, output comparison, semantic differences)
- **FR-2.6.4**: Performance Benchmarking (Python vs Rust/WASM, metrics, regression detection)

### 1.3 Dependencies

**Inputs from Other Agents:**
- Analysis Agent: API contracts, function signatures, type information
- Specification Generator: Rust type mappings, contract specifications
- Transpiler Agent: Generated Rust code
- Build Agent: Compiled WASM modules

**External Dependencies:**
- Python interpreter (for golden vector generation)
- Rust test framework (cargo test)
- Wasmtime/Wasmer runtime (for WASM execution)
- Criterion.rs (for benchmarking)

---

## 2. Data Structures

### 2.1 Core Data Types

```rust
// ============================================================================
// Test Suite Representation
// ============================================================================

struct TestSuite {
    name: String,
    mode: TestMode,
    test_cases: Vec<TestCase>,
    coverage_target: CoverageTarget,
    metadata: TestMetadata,
}

enum TestMode {
    Conformance,
    PropertyBased,
    Parity,
    Performance,
}

struct TestCase {
    id: TestCaseId,
    name: String,
    description: String,
    source: TestSource,
    inputs: Vec<TestInput>,
    expected_output: ExpectedOutput,
    constraints: Vec<Constraint>,
    timeout_ms: u64,
}

enum TestSource {
    TranslatedPythonTest {
        original_path: PathBuf,
        original_line: usize,
        python_ast: PythonTestAST,
    },
    GeneratedProperty {
        property_type: PropertyType,
        derived_from: ContractId,
    },
    ManualGoldenVector {
        vector_id: VectorId,
    },
}

struct TestInput {
    parameter_name: String,
    value: Value,
    value_type: TypeInfo,
}

enum ExpectedOutput {
    ExactValue(Value),
    WithinTolerance {
        value: Value,
        tolerance: Tolerance,
    },
    MatchesPredicate {
        predicate: PredicateFn,
        description: String,
    },
    RaisesException {
        exception_type: String,
        message_pattern: Option<Regex>,
    },
}

struct Tolerance {
    absolute: Option<f64>,
    relative: Option<f64>,
    ulp: Option<u32>,  // Units in Last Place for floating-point
}

// ============================================================================
// Golden Test Vectors
// ============================================================================

struct GoldenVector {
    id: VectorId,
    function_id: FunctionId,
    inputs: Vec<TestInput>,
    python_output: PythonExecutionResult,
    rust_output: Option<RustExecutionResult>,
    wasm_output: Option<WasmExecutionResult>,
    metadata: VectorMetadata,
}

struct PythonExecutionResult {
    stdout: String,
    stderr: String,
    return_value: Option<Value>,
    exception: Option<ExceptionInfo>,
    execution_time_ms: f64,
    memory_peak_bytes: u64,
}

struct RustExecutionResult {
    stdout: String,
    stderr: String,
    return_value: Option<Value>,
    error: Option<ErrorInfo>,
    execution_time_ms: f64,
    memory_peak_bytes: u64,
}

struct WasmExecutionResult {
    stdout: String,
    stderr: String,
    return_value: Option<Value>,
    error: Option<ErrorInfo>,
    execution_time_ms: f64,
    memory_peak_bytes: u64,
    wasm_fuel_consumed: u64,
}

// ============================================================================
// Property-Based Testing
// ============================================================================

struct PropertyTest {
    id: PropertyTestId,
    name: String,
    property_type: PropertyType,
    generators: Vec<Generator>,
    property_predicate: PropertyPredicate,
    shrink_strategy: ShrinkStrategy,
    num_iterations: usize,
}

enum PropertyType {
    // Algebraic properties
    Commutative,       // f(a, b) == f(b, a)
    Associative,       // f(f(a, b), c) == f(a, f(b, c))
    Identity,          // f(a, identity) == a
    Inverse,           // f(f_inv(a)) == a
    Idempotent,        // f(f(a)) == f(a)

    // Relational properties
    Monotonic,         // a < b => f(a) <= f(b)
    Symmetric,         // f(a, b) == f(b, a)
    Transitive,        // f(a, b) && f(b, c) => f(a, c)

    // Contract properties
    Precondition,      // Input constraints
    Postcondition,     // Output guarantees
    Invariant,         // State invariants maintained

    // Behavioral properties
    Deterministic,     // Same inputs => same outputs
    Pure,              // No side effects
    BoundedOutput,     // Output within specified range

    // Metamorphic properties
    Metamorphic {
        source_inputs: Vec<TestInput>,
        transformed_inputs: Vec<TestInput>,
        relation: MetamorphicRelation,
    },
}

struct Generator {
    parameter_name: String,
    strategy: GenerationStrategy,
}

enum GenerationStrategy {
    IntRange { min: i64, max: i64 },
    FloatRange { min: f64, max: f64 },
    StringPattern { pattern: Regex, max_len: usize },
    ListOf { element_gen: Box<GenerationStrategy>, min_len: usize, max_len: usize },
    DictOf { key_gen: Box<GenerationStrategy>, value_gen: Box<GenerationStrategy> },
    Custom { generator_fn: GeneratorFn },
    BoundaryValues { base_gen: Box<GenerationStrategy> },  // Include edge cases
}

struct PropertyPredicate {
    predicate_fn: PredicateFn,
    description: String,
    python_reference: Option<PathBuf>,
}

// ============================================================================
// Parity Validation
// ============================================================================

struct ParityReport {
    timestamp: DateTime,
    total_tests: usize,
    passed: usize,
    failed: usize,
    mismatches: Vec<ParityMismatch>,
    coverage: CoverageMetrics,
    summary: ParitySummary,
}

struct ParityMismatch {
    test_case_id: TestCaseId,
    function_id: FunctionId,
    inputs: Vec<TestInput>,
    python_output: Value,
    rust_output: Value,
    wasm_output: Option<Value>,
    difference: DifferenceReport,
    severity: MismatchSeverity,
}

enum MismatchSeverity {
    Critical,      // Wrong result, will break functionality
    Major,         // Significant difference, may impact correctness
    Minor,         // Small numeric difference, within acceptable tolerance
    Informational, // Different representation but semantically equivalent
}

struct DifferenceReport {
    diff_type: DiffType,
    details: String,
    metric: Option<f64>,  // Quantitative measure of difference
}

enum DiffType {
    TypeMismatch { expected: String, actual: String },
    ValueMismatch { expected: Value, actual: Value },
    FloatingPointDrift { expected: f64, actual: f64, ulp_distance: u32 },
    CollectionSizeDiff { expected: usize, actual: usize },
    CollectionElementDiff { index: usize, expected: Value, actual: Value },
    ExceptionDiff { expected: String, actual: String },
    MissingOutput,
    ExtraOutput,
}

// ============================================================================
// Performance Benchmarking
// ============================================================================

struct BenchmarkSuite {
    name: String,
    benchmarks: Vec<Benchmark>,
    baseline: Option<BaselineMetrics>,
}

struct Benchmark {
    id: BenchmarkId,
    name: String,
    function_id: FunctionId,
    workload: Workload,
    python_metrics: ExecutionMetrics,
    rust_metrics: ExecutionMetrics,
    wasm_metrics: ExecutionMetrics,
    comparison: PerformanceComparison,
}

struct Workload {
    description: String,
    input_size: usize,
    iterations: usize,
    warmup_iterations: usize,
    inputs: Vec<TestInput>,
}

struct ExecutionMetrics {
    mean_time_ns: f64,
    median_time_ns: f64,
    std_dev_ns: f64,
    min_time_ns: f64,
    max_time_ns: f64,
    percentile_95_ns: f64,
    percentile_99_ns: f64,
    throughput_ops_per_sec: f64,
    memory_peak_bytes: u64,
    memory_avg_bytes: u64,
    allocations_count: u64,
}

struct PerformanceComparison {
    speedup_factor: f64,  // rust_time / python_time (>1 means faster)
    memory_improvement: f64,  // python_memory / rust_memory
    throughput_improvement: f64,
    regression_detected: bool,
    regression_threshold: f64,
}

// ============================================================================
// Coverage Metrics
// ============================================================================

struct CoverageMetrics {
    api_coverage: ApiCoverage,
    line_coverage: LineCoverage,
    branch_coverage: BranchCoverage,
    test_coverage: TestCoverage,
}

struct ApiCoverage {
    total_functions: usize,
    tested_functions: usize,
    coverage_percentage: f64,
    untested_functions: Vec<FunctionId>,
}

struct LineCoverage {
    total_lines: usize,
    covered_lines: usize,
    coverage_percentage: f64,
    uncovered_ranges: Vec<LineRange>,
}

struct BranchCoverage {
    total_branches: usize,
    covered_branches: usize,
    coverage_percentage: f64,
    uncovered_branches: Vec<BranchId>,
}

struct TestCoverage {
    total_test_cases: usize,
    passing_test_cases: usize,
    failing_test_cases: usize,
    skipped_test_cases: usize,
    pass_rate: f64,
}

// ============================================================================
// Test Execution Context
// ============================================================================

struct TestExecutionContext {
    test_suite_id: TestSuiteId,
    python_runtime: PythonRuntime,
    rust_runtime: RustRuntime,
    wasm_runtime: WasmRuntime,
    timeout_config: TimeoutConfig,
    resource_limits: ResourceLimits,
}

struct PythonRuntime {
    python_path: PathBuf,
    version: PythonVersion,
    virtual_env: Option<PathBuf>,
    sys_path: Vec<PathBuf>,
}

struct RustRuntime {
    cargo_path: PathBuf,
    rust_version: RustVersion,
    target_triple: String,
    features: Vec<String>,
}

struct WasmRuntime {
    runtime_type: WasmRuntimeType,
    runtime_path: PathBuf,
    wasi_config: WasiConfig,
    resource_limits: WasmResourceLimits,
}

enum WasmRuntimeType {
    Wasmtime,
    Wasmer,
    Browser,
}

struct ResourceLimits {
    max_memory_bytes: u64,
    max_execution_time_ms: u64,
    max_stack_size_bytes: u64,
}

struct WasmResourceLimits {
    max_memory_pages: u32,
    max_table_elements: u32,
    max_instances: u32,
    max_fuel: Option<u64>,
}
```

### 2.2 Supporting Data Types

```rust
// ============================================================================
// Type System Support
// ============================================================================

enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(HashMap<Value, Value>),
    Set(HashSet<Value>),
    Custom {
        type_name: String,
        fields: HashMap<String, Value>,
    },
}

struct TypeInfo {
    base_type: BaseType,
    generic_params: Vec<TypeInfo>,
    nullable: bool,
    constraints: Vec<TypeConstraint>,
}

enum BaseType {
    Primitive(PrimitiveType),
    Collection(CollectionType),
    Custom(String),
}

enum PrimitiveType {
    Bool, Int, Float, String, Bytes, None,
}

enum CollectionType {
    List, Tuple, Dict, Set,
}

// ============================================================================
// Error Information
// ============================================================================

struct ExceptionInfo {
    exception_type: String,
    message: String,
    traceback: Vec<StackFrame>,
}

struct ErrorInfo {
    error_type: String,
    message: String,
    backtrace: Vec<StackFrame>,
}

struct StackFrame {
    file: PathBuf,
    line: usize,
    function: String,
    code_snippet: Option<String>,
}

// ============================================================================
// Metadata
// ============================================================================

struct TestMetadata {
    created_at: DateTime,
    source_commit: String,
    python_version: String,
    rust_version: String,
    tags: Vec<String>,
}

struct VectorMetadata {
    generated_at: DateTime,
    generator_version: String,
    deterministic: bool,
    seed: Option<u64>,
}
```

---

## 3. Core Algorithms

### 3.1 Test Translation Algorithm

```rust
// ============================================================================
// ALGORITHM: Translate Python Tests to Rust
// ============================================================================
// Converts Python unit tests (pytest, unittest) to Rust tests
// Preserves test structure, assertions, and setup/teardown logic
// ============================================================================

fn translate_python_tests(
    python_test_files: Vec<PathBuf>,
    python_ast_forest: ASTForest,
    rust_module_map: ModuleMap,
    type_mappings: TypeMappings,
) -> Result<Vec<RustTestFile>, TestTranslationError> {

    let mut rust_test_files = Vec::new()

    for python_test_file in python_test_files {
        // Step 1: Parse Python test file
        let python_ast = parse_python_file(python_test_file)?

        // Step 2: Identify test framework (pytest, unittest, doctest)
        let framework = detect_test_framework(python_ast)?

        // Step 3: Extract test functions/classes
        let test_entities = extract_test_entities(python_ast, framework)?

        let mut rust_tests = Vec::new()

        for test_entity in test_entities {
            match test_entity {
                TestEntity::Function(test_fn) => {
                    // Step 4: Translate individual test function
                    let rust_test = translate_test_function(
                        test_fn,
                        rust_module_map,
                        type_mappings,
                        framework,
                    )?

                    rust_tests.push(rust_test)
                }

                TestEntity::Class(test_class) => {
                    // Step 5: Translate test class (setUp/tearDown methods)
                    let rust_test_module = translate_test_class(
                        test_class,
                        rust_module_map,
                        type_mappings,
                        framework,
                    )?

                    rust_tests.extend(rust_test_module.tests)
                }
            }
        }

        // Step 6: Generate Rust test file
        let rust_test_file = RustTestFile {
            path: compute_rust_test_path(python_test_file),
            module_name: extract_module_name(python_test_file),
            imports: generate_test_imports(rust_tests, rust_module_map),
            tests: rust_tests,
            helper_functions: Vec::new(),
        }

        rust_test_files.push(rust_test_file)
    }

    Ok(rust_test_files)
}

// ============================================================================
// SUB-ALGORITHM: Translate Individual Test Function
// ============================================================================

fn translate_test_function(
    python_test: PythonTestFunction,
    rust_module_map: ModuleMap,
    type_mappings: TypeMappings,
    framework: TestFramework,
) -> Result<RustTestFunction, TestTranslationError> {

    // Step 1: Extract test metadata
    let test_name = python_test.name
    let docstring = python_test.docstring
    let decorators = python_test.decorators

    // Step 2: Translate test attributes (skip, xfail, parametrize)
    let rust_attributes = translate_test_attributes(decorators, framework)?

    // Step 3: Translate test body
    let mut rust_statements = Vec::new()

    for python_stmt in python_test.body {
        match python_stmt {
            // Arrange phase: variable assignments, setup
            PythonStmt::Assign { target, value } => {
                let rust_binding = translate_assignment(
                    target,
                    value,
                    type_mappings,
                )?
                rust_statements.push(rust_binding)
            }

            // Act phase: function call under test
            PythonStmt::Expr { value: PythonExpr::Call { func, args, kwargs } } => {
                let rust_call = translate_function_call(
                    func,
                    args,
                    kwargs,
                    rust_module_map,
                    type_mappings,
                )?
                rust_statements.push(rust_call)
            }

            // Assert phase: assertions
            PythonStmt::Assert { test, msg } => {
                let rust_assertion = translate_assertion(
                    test,
                    msg,
                    framework,
                )?
                rust_statements.push(rust_assertion)
            }

            // pytest-specific assertions (assert x == y)
            PythonStmt::Expr { value: PythonExpr::Compare { left, ops, comparators } } => {
                let rust_assertion = translate_comparison_assertion(
                    left,
                    ops,
                    comparators,
                )?
                rust_statements.push(rust_assertion)
            }

            // Other statements
            _ => {
                let rust_stmt = translate_statement(
                    python_stmt,
                    rust_module_map,
                    type_mappings,
                )?
                rust_statements.push(rust_stmt)
            }
        }
    }

    // Step 4: Generate Rust test function
    Ok(RustTestFunction {
        name: sanitize_test_name(test_name),
        attributes: rust_attributes,
        docstring: docstring,
        body: rust_statements,
        expected_panic: detect_expected_exception(python_test),
    })
}

// ============================================================================
// SUB-ALGORITHM: Translate Assertions
// ============================================================================

fn translate_assertion(
    python_assertion: PythonExpr,
    message: Option<String>,
    framework: TestFramework,
) -> Result<RustStmt, TestTranslationError> {

    match (python_assertion, framework) {
        // unittest: self.assertEqual(a, b)
        (PythonExpr::Call {
            func: PythonExpr::Attribute { value, attr: "assertEqual" },
            args: [left, right],
            ..
        }, TestFramework::Unittest) => {
            Ok(RustStmt::MacroCall {
                name: "assert_eq",
                args: vec![
                    translate_expression(left)?,
                    translate_expression(right)?,
                ],
                message: message,
            })
        }

        // pytest: assert a == b
        (PythonExpr::Compare {
            left,
            ops: [CompOp::Eq],
            comparators: [right],
        }, TestFramework::Pytest) => {
            Ok(RustStmt::MacroCall {
                name: "assert_eq",
                args: vec![
                    translate_expression(left)?,
                    translate_expression(right)?,
                ],
                message: message,
            })
        }

        // unittest: self.assertTrue(x)
        (PythonExpr::Call {
            func: PythonExpr::Attribute { attr: "assertTrue", .. },
            args: [expr],
            ..
        }, TestFramework::Unittest) => {
            Ok(RustStmt::MacroCall {
                name: "assert",
                args: vec![translate_expression(expr)?],
                message: message,
            })
        }

        // unittest: self.assertRaises(Exception)
        (PythonExpr::Call {
            func: PythonExpr::Attribute { attr: "assertRaises", .. },
            args: [exception_type],
            ..
        }, TestFramework::Unittest) => {
            // Translate to #[should_panic] attribute
            Ok(RustStmt::Attribute {
                name: "should_panic",
                expected: Some(extract_exception_name(exception_type)?),
            })
        }

        // pytest: approx for floating-point comparisons
        (PythonExpr::Compare {
            left,
            ops: [CompOp::Eq],
            comparators: [PythonExpr::Call { func, args: [right], .. }],
        }, TestFramework::Pytest) if is_approx_call(func) => {
            Ok(RustStmt::MacroCall {
                name: "assert_approx_eq",
                args: vec![
                    translate_expression(left)?,
                    translate_expression(right)?,
                    RustExpr::Literal("1e-6"),  // Default tolerance
                ],
                message: message,
            })
        }

        _ => {
            // Fallback: generic assert
            Ok(RustStmt::MacroCall {
                name: "assert",
                args: vec![translate_expression(python_assertion)?],
                message: message,
            })
        }
    }
}
```

### 3.2 Golden Vector Generation Algorithm

```rust
// ============================================================================
// ALGORITHM: Generate Golden Test Vectors
// ============================================================================
// Executes Python functions with generated inputs to create reference outputs
// Stores input/output pairs for Rust/WASM validation
// ============================================================================

fn generate_golden_vectors(
    python_module: PythonModule,
    api_surface: ApiSurface,
    generation_config: VectorGenerationConfig,
) -> Result<Vec<GoldenVector>, VectorGenerationError> {

    let mut golden_vectors = Vec::new()

    // Step 1: Initialize Python runtime
    let python_runtime = initialize_python_runtime(
        python_module.path,
        python_module.dependencies,
    )?

    // Step 2: For each public function in API surface
    for function in api_surface.public_functions {
        let function_vectors = generate_vectors_for_function(
            function,
            python_runtime,
            generation_config,
        )?

        golden_vectors.extend(function_vectors)
    }

    // Step 3: Serialize golden vectors to disk
    save_golden_vectors(golden_vectors, generation_config.output_path)?

    Ok(golden_vectors)
}

// ============================================================================
// SUB-ALGORITHM: Generate Vectors for Single Function
// ============================================================================

fn generate_vectors_for_function(
    function: FunctionSignature,
    python_runtime: PythonRuntime,
    config: VectorGenerationConfig,
) -> Result<Vec<GoldenVector>, VectorGenerationError> {

    let mut vectors = Vec::new()

    // Step 1: Generate input combinations based on parameter types
    let input_generator = create_input_generator(function.parameters, config)?

    // Step 2: Generate N test vectors
    for i in 0..config.vectors_per_function {
        // Generate random inputs
        let inputs = input_generator.generate_inputs()?

        // Step 3: Execute Python function with inputs
        let execution_result = execute_python_function(
            python_runtime,
            function.qualified_name,
            inputs.clone(),
        )?

        // Step 4: Create golden vector
        let vector = GoldenVector {
            id: VectorId::new(),
            function_id: function.id,
            inputs: inputs,
            python_output: execution_result,
            rust_output: None,  // Will be filled during validation
            wasm_output: None,  // Will be filled during validation
            metadata: VectorMetadata {
                generated_at: now(),
                generator_version: VERSION,
                deterministic: config.deterministic,
                seed: config.seed,
            },
        }

        vectors.push(vector)
    }

    // Step 5: Add boundary value test vectors
    if config.include_boundary_values {
        let boundary_vectors = generate_boundary_vectors(
            function,
            python_runtime,
        )?
        vectors.extend(boundary_vectors)
    }

    Ok(vectors)
}

// ============================================================================
// SUB-ALGORITHM: Execute Python Function
// ============================================================================

fn execute_python_function(
    runtime: PythonRuntime,
    function_name: QualifiedName,
    inputs: Vec<TestInput>,
) -> Result<PythonExecutionResult, ExecutionError> {

    // Step 1: Prepare Python execution environment
    let temp_script = create_temp_execution_script(function_name, inputs)?

    // Step 2: Execute Python script with resource monitoring
    let start_time = Instant::now()
    let start_memory = get_memory_usage()

    let process = Command::new(runtime.python_path)
        .arg(temp_script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?

    // Step 3: Wait for completion with timeout
    let output = wait_with_timeout(
        process,
        Duration::from_millis(config.execution_timeout_ms),
    )?

    let execution_time = start_time.elapsed()
    let peak_memory = get_peak_memory_usage() - start_memory

    // Step 4: Parse output
    let result = if output.status.success() {
        // Parse stdout as JSON-serialized return value
        let return_value = parse_json_value(output.stdout)?

        PythonExecutionResult {
            stdout: String::from_utf8(output.stdout)?,
            stderr: String::from_utf8(output.stderr)?,
            return_value: Some(return_value),
            exception: None,
            execution_time_ms: execution_time.as_millis() as f64,
            memory_peak_bytes: peak_memory,
        }
    } else {
        // Parse exception from stderr
        let exception = parse_python_exception(output.stderr)?

        PythonExecutionResult {
            stdout: String::from_utf8(output.stdout)?,
            stderr: String::from_utf8(output.stderr)?,
            return_value: None,
            exception: Some(exception),
            execution_time_ms: execution_time.as_millis() as f64,
            memory_peak_bytes: peak_memory,
        }
    }

    // Step 5: Clean up temporary files
    remove_file(temp_script)?

    Ok(result)
}

// ============================================================================
// SUB-ALGORITHM: Generate Boundary Value Inputs
// ============================================================================

fn generate_boundary_vectors(
    function: FunctionSignature,
    runtime: PythonRuntime,
) -> Result<Vec<GoldenVector>, VectorGenerationError> {

    let mut boundary_vectors = Vec::new()

    for parameter in function.parameters {
        // Generate boundary values based on parameter type
        let boundary_values = match parameter.type_info {
            TypeInfo::Int { min, max } => {
                vec![
                    min,
                    min + 1,
                    0,
                    max - 1,
                    max,
                ]
            }

            TypeInfo::Float { min, max } => {
                vec![
                    min,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    max,
                    f64::MIN,
                    f64::MAX,
                    f64::EPSILON,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    // Note: NaN handled separately
                ]
            }

            TypeInfo::String { max_length } => {
                vec![
                    "",  // Empty string
                    "a",  // Single character
                    "a".repeat(max_length),  // Maximum length
                    "unicode: \u{1F600}",  // Unicode characters
                ]
            }

            TypeInfo::List { element_type, max_size } => {
                vec![
                    Vec::new(),  // Empty list
                    vec![default_value(element_type)],  // Single element
                    vec![default_value(element_type); max_size],  // Maximum size
                ]
            }

            TypeInfo::Optional { inner_type } => {
                vec![
                    None,
                    Some(default_value(inner_type)),
                ]
            }

            _ => Vec::new(),
        }

        // For each boundary value, create a test vector
        for value in boundary_values {
            let inputs = create_inputs_with_boundary_value(
                function.parameters.clone(),
                parameter.name.clone(),
                value,
            )?

            let execution_result = execute_python_function(
                runtime,
                function.qualified_name.clone(),
                inputs.clone(),
            )?

            boundary_vectors.push(GoldenVector {
                id: VectorId::new(),
                function_id: function.id,
                inputs: inputs,
                python_output: execution_result,
                rust_output: None,
                wasm_output: None,
                metadata: VectorMetadata {
                    generated_at: now(),
                    generator_version: VERSION,
                    deterministic: true,
                    seed: None,
                },
            })
        }
    }

    Ok(boundary_vectors)
}
```

### 3.3 Parity Validation Algorithm

```rust
// ============================================================================
// ALGORITHM: Validate Parity (Python vs Rust/WASM)
// ============================================================================
// Executes equivalent workloads in Python and Rust/WASM
// Compares outputs with tolerance-aware comparison
// Detects semantic differences
// ============================================================================

fn validate_parity(
    golden_vectors: Vec<GoldenVector>,
    rust_binary: PathBuf,
    wasm_module: PathBuf,
    comparison_config: ComparisonConfig,
) -> Result<ParityReport, ParityValidationError> {

    let mut mismatches = Vec::new()
    let mut passed = 0
    let mut failed = 0

    // Step 1: Initialize Rust and WASM runtimes
    let rust_runtime = initialize_rust_runtime(rust_binary)?
    let wasm_runtime = initialize_wasm_runtime(wasm_module)?

    // Step 2: For each golden vector
    for mut vector in golden_vectors {
        // Step 3: Execute Rust implementation
        let rust_result = execute_rust_function(
            rust_runtime,
            vector.function_id,
            vector.inputs.clone(),
        )?

        vector.rust_output = Some(rust_result.clone())

        // Step 4: Execute WASM implementation
        let wasm_result = execute_wasm_function(
            wasm_runtime,
            vector.function_id,
            vector.inputs.clone(),
        )?

        vector.wasm_output = Some(wasm_result.clone())

        // Step 5: Compare outputs
        let python_output = vector.python_output.return_value.clone()
        let rust_output = rust_result.return_value.clone()
        let wasm_output = wasm_result.return_value.clone()

        // Python vs Rust comparison
        if let Some(mismatch) = compare_outputs(
            python_output.clone(),
            rust_output.clone(),
            comparison_config,
            "Rust",
        )? {
            mismatches.push(ParityMismatch {
                test_case_id: vector.id,
                function_id: vector.function_id,
                inputs: vector.inputs.clone(),
                python_output: python_output.clone().unwrap_or(Value::None),
                rust_output: rust_output.clone().unwrap_or(Value::None),
                wasm_output: None,
                difference: mismatch.clone(),
                severity: classify_mismatch_severity(mismatch),
            })
            failed += 1
        } else {
            passed += 1
        }

        // Python vs WASM comparison
        if let Some(mismatch) = compare_outputs(
            python_output.clone(),
            wasm_output.clone(),
            comparison_config,
            "WASM",
        )? {
            mismatches.push(ParityMismatch {
                test_case_id: vector.id,
                function_id: vector.function_id,
                inputs: vector.inputs.clone(),
                python_output: python_output.clone().unwrap_or(Value::None),
                rust_output: Value::None,
                wasm_output: wasm_output.clone(),
                difference: mismatch.clone(),
                severity: classify_mismatch_severity(mismatch),
            })
        }
    }

    // Step 6: Calculate coverage metrics
    let coverage = calculate_coverage_metrics(golden_vectors)?

    // Step 7: Generate parity report
    Ok(ParityReport {
        timestamp: now(),
        total_tests: golden_vectors.len(),
        passed: passed,
        failed: failed,
        mismatches: mismatches,
        coverage: coverage,
        summary: generate_parity_summary(passed, failed, mismatches),
    })
}

// ============================================================================
// SUB-ALGORITHM: Compare Outputs with Tolerance
// ============================================================================

fn compare_outputs(
    expected: Option<Value>,
    actual: Option<Value>,
    config: ComparisonConfig,
    runtime_name: &str,
) -> Result<Option<DifferenceReport>, ComparisonError> {

    // Step 1: Handle missing outputs
    match (expected, actual) {
        (None, None) => return Ok(None),  // Both None, equal
        (Some(_), None) => {
            return Ok(Some(DifferenceReport {
                diff_type: DiffType::MissingOutput,
                details: format!("{} produced no output", runtime_name),
                metric: None,
            }))
        }
        (None, Some(_)) => {
            return Ok(Some(DifferenceReport {
                diff_type: DiffType::ExtraOutput,
                details: format!("{} produced unexpected output", runtime_name),
                metric: None,
            }))
        }
        (Some(exp), Some(act)) => {
            // Continue to value comparison
            return compare_values(exp, act, config)
        }
    }
}

// ============================================================================
// SUB-ALGORITHM: Deep Value Comparison
// ============================================================================

fn compare_values(
    expected: Value,
    actual: Value,
    config: ComparisonConfig,
) -> Result<Option<DifferenceReport>, ComparisonError> {

    match (expected, actual) {
        // Exact equality for simple types
        (Value::None, Value::None) => Ok(None),
        (Value::Bool(a), Value::Bool(b)) if a == b => Ok(None),
        (Value::Int(a), Value::Int(b)) if a == b => Ok(None),
        (Value::String(a), Value::String(b)) if a == b => Ok(None),
        (Value::Bytes(a), Value::Bytes(b)) if a == b => Ok(None),

        // Floating-point comparison with tolerance
        (Value::Float(a), Value::Float(b)) => {
            compare_floats(a, b, config.float_tolerance)
        }

        // Collection comparisons
        (Value::List(a), Value::List(b)) => {
            compare_lists(a, b, config)
        }

        (Value::Tuple(a), Value::Tuple(b)) => {
            compare_tuples(a, b, config)
        }

        (Value::Dict(a), Value::Dict(b)) => {
            compare_dicts(a, b, config)
        }

        (Value::Set(a), Value::Set(b)) => {
            compare_sets(a, b, config)
        }

        // Custom type comparison
        (Value::Custom { type_name: tn1, fields: f1 },
         Value::Custom { type_name: tn2, fields: f2 }) => {
            if tn1 != tn2 {
                return Ok(Some(DifferenceReport {
                    diff_type: DiffType::TypeMismatch {
                        expected: tn1,
                        actual: tn2,
                    },
                    details: "Custom type names differ".to_string(),
                    metric: None,
                }))
            }
            compare_dicts(f1, f2, config)
        }

        // Type mismatch
        (expected, actual) => {
            Ok(Some(DifferenceReport {
                diff_type: DiffType::TypeMismatch {
                    expected: type_name(&expected),
                    actual: type_name(&actual),
                },
                details: format!(
                    "Expected {:?} but got {:?}",
                    expected, actual
                ),
                metric: None,
            }))
        }
    }
}

// ============================================================================
// SUB-ALGORITHM: Floating-Point Comparison with ULP Distance
// ============================================================================

fn compare_floats(
    expected: f64,
    actual: f64,
    tolerance: Tolerance,
) -> Result<Option<DifferenceReport>, ComparisonError> {

    // Step 1: Handle special values
    if expected.is_nan() && actual.is_nan() {
        return Ok(None)  // Both NaN, consider equal
    }

    if expected.is_infinite() && actual.is_infinite() {
        if expected.is_sign_positive() == actual.is_sign_positive() {
            return Ok(None)  // Both +Inf or both -Inf
        } else {
            return Ok(Some(DifferenceReport {
                diff_type: DiffType::FloatingPointDrift {
                    expected: expected,
                    actual: actual,
                    ulp_distance: u32::MAX,
                },
                details: "Infinity sign mismatch".to_string(),
                metric: Some(f64::INFINITY),
            }))
        }
    }

    // Step 2: Check absolute tolerance
    if let Some(abs_tol) = tolerance.absolute {
        let abs_diff = (expected - actual).abs()
        if abs_diff <= abs_tol {
            return Ok(None)
        }
    }

    // Step 3: Check relative tolerance
    if let Some(rel_tol) = tolerance.relative {
        let rel_diff = ((expected - actual) / expected).abs()
        if rel_diff <= rel_tol {
            return Ok(None)
        }
    }

    // Step 4: Check ULP (Units in Last Place) tolerance
    if let Some(max_ulp) = tolerance.ulp {
        let ulp_distance = calculate_ulp_distance(expected, actual)

        if ulp_distance <= max_ulp {
            return Ok(None)
        }

        return Ok(Some(DifferenceReport {
            diff_type: DiffType::FloatingPointDrift {
                expected: expected,
                actual: actual,
                ulp_distance: ulp_distance,
            },
            details: format!(
                "ULP distance {} exceeds tolerance {}",
                ulp_distance, max_ulp
            ),
            metric: Some(ulp_distance as f64),
        }))
    }

    // Step 5: Default exact comparison
    if expected == actual {
        Ok(None)
    } else {
        Ok(Some(DifferenceReport {
            diff_type: DiffType::FloatingPointDrift {
                expected: expected,
                actual: actual,
                ulp_distance: calculate_ulp_distance(expected, actual),
            },
            details: format!(
                "Values differ: expected {}, got {}",
                expected, actual
            ),
            metric: Some((expected - actual).abs()),
        }))
    }
}

// ============================================================================
// HELPER: Calculate ULP Distance Between Floats
// ============================================================================

fn calculate_ulp_distance(a: f64, b: f64) -> u32 {
    // Convert floats to their IEEE 754 bit representation
    let a_bits = a.to_bits()
    let b_bits = b.to_bits()

    // Handle sign differences
    if (a_bits ^ b_bits) >> 63 != 0 {
        // Different signs - if both close to zero, distance is small
        if a.abs() < f64::EPSILON && b.abs() < f64::EPSILON {
            return 0
        }
        return u32::MAX  // Different signs, large distance
    }

    // Same sign - calculate bit difference
    let diff = if a_bits > b_bits {
        a_bits - b_bits
    } else {
        b_bits - a_bits
    }

    // ULP distance is the difference in bit representation
    min(diff as u32, u32::MAX)
}

// ============================================================================
// SUB-ALGORITHM: Compare Lists
// ============================================================================

fn compare_lists(
    expected: Vec<Value>,
    actual: Vec<Value>,
    config: ComparisonConfig,
) -> Result<Option<DifferenceReport>, ComparisonError> {

    // Step 1: Check length
    if expected.len() != actual.len() {
        return Ok(Some(DifferenceReport {
            diff_type: DiffType::CollectionSizeDiff {
                expected: expected.len(),
                actual: actual.len(),
            },
            details: format!(
                "List length mismatch: expected {}, got {}",
                expected.len(), actual.len()
            ),
            metric: Some((expected.len() as i64 - actual.len() as i64).abs() as f64),
        }))
    }

    // Step 2: Compare elements
    for (index, (exp_elem, act_elem)) in expected.iter().zip(actual.iter()).enumerate() {
        if let Some(diff) = compare_values(
            exp_elem.clone(),
            act_elem.clone(),
            config,
        )? {
            return Ok(Some(DifferenceReport {
                diff_type: DiffType::CollectionElementDiff {
                    index: index,
                    expected: exp_elem.clone(),
                    actual: act_elem.clone(),
                },
                details: format!(
                    "List element at index {} differs: {}",
                    index, diff.details
                ),
                metric: diff.metric,
            }))
        }
    }

    Ok(None)  // All elements equal
}
```

### 3.4 Property-Based Testing Algorithm

```rust
// ============================================================================
// ALGORITHM: Generate Property-Based Tests
// ============================================================================
// Derives properties from function contracts
// Generates quickcheck/proptest tests for Rust
// Tests invariants, algebraic properties, and metamorphic relations
// ============================================================================

fn generate_property_tests(
    api_surface: ApiSurface,
    contracts: Vec<Contract>,
    config: PropertyTestConfig,
) -> Result<Vec<PropertyTest>, PropertyGenerationError> {

    let mut property_tests = Vec::new()

    // Step 1: For each function with contracts
    for function in api_surface.public_functions {
        let function_contracts = contracts
            .iter()
            .filter(|c| c.function_id == function.id)
            .collect::<Vec<_>>()

        // Step 2: Derive properties from contracts
        let derived_properties = derive_properties_from_contracts(
            function.clone(),
            function_contracts,
        )?

        property_tests.extend(derived_properties)

        // Step 3: Infer algebraic properties from function type
        let algebraic_properties = infer_algebraic_properties(
            function.clone(),
        )?

        property_tests.extend(algebraic_properties)

        // Step 4: Generate metamorphic properties
        let metamorphic_properties = generate_metamorphic_properties(
            function.clone(),
            config,
        )?

        property_tests.extend(metamorphic_properties)
    }

    Ok(property_tests)
}

// ============================================================================
// SUB-ALGORITHM: Derive Properties from Contracts
// ============================================================================

fn derive_properties_from_contracts(
    function: FunctionSignature,
    contracts: Vec<&Contract>,
) -> Result<Vec<PropertyTest>, PropertyGenerationError> {

    let mut properties = Vec::new()

    for contract in contracts {
        match contract.contract_type {
            // Precondition: test that violations are rejected
            ContractType::Precondition { condition } => {
                let property = PropertyTest {
                    id: PropertyTestId::new(),
                    name: format!("{}_precondition", function.name),
                    property_type: PropertyType::Precondition,
                    generators: create_generators_from_params(function.parameters.clone()),
                    property_predicate: PropertyPredicate {
                        predicate_fn: Box::new(move |inputs| {
                            // If precondition is violated, function should return error
                            if !evaluate_condition(condition.clone(), inputs) {
                                match execute_function(function.id, inputs) {
                                    Err(_) => true,  // Correctly rejected
                                    Ok(_) => false,  // Should have failed
                                }
                            } else {
                                true  // Precondition satisfied, don't test
                            }
                        }),
                        description: format!(
                            "Function rejects inputs that violate: {}",
                            condition
                        ),
                        python_reference: None,
                    },
                    shrink_strategy: ShrinkStrategy::Default,
                    num_iterations: config.iterations_per_property,
                }

                properties.push(property)
            }

            // Postcondition: test that outputs satisfy guarantees
            ContractType::Postcondition { condition } => {
                let property = PropertyTest {
                    id: PropertyTestId::new(),
                    name: format!("{}_postcondition", function.name),
                    property_type: PropertyType::Postcondition,
                    generators: create_generators_from_params(function.parameters.clone()),
                    property_predicate: PropertyPredicate {
                        predicate_fn: Box::new(move |inputs| {
                            match execute_function(function.id, inputs.clone()) {
                                Ok(output) => {
                                    // Check postcondition on output
                                    evaluate_condition_with_output(
                                        condition.clone(),
                                        inputs,
                                        output,
                                    )
                                }
                                Err(_) => true,  // Error is acceptable
                            }
                        }),
                        description: format!(
                            "Function output satisfies: {}",
                            condition
                        ),
                        python_reference: None,
                    },
                    shrink_strategy: ShrinkStrategy::Default,
                    num_iterations: config.iterations_per_property,
                }

                properties.push(property)
            }

            // Invariant: test that state invariants are maintained
            ContractType::Invariant { condition } => {
                let property = PropertyTest {
                    id: PropertyTestId::new(),
                    name: format!("{}_invariant", function.name),
                    property_type: PropertyType::Invariant,
                    generators: create_generators_from_params(function.parameters.clone()),
                    property_predicate: PropertyPredicate {
                        predicate_fn: Box::new(move |inputs| {
                            let state_before = capture_state()
                            let result = execute_function(function.id, inputs.clone())
                            let state_after = capture_state()

                            // Check invariant holds before and after
                            evaluate_condition(condition.clone(), state_before) &&
                            evaluate_condition(condition.clone(), state_after)
                        }),
                        description: format!(
                            "Invariant maintained: {}",
                            condition
                        ),
                        python_reference: None,
                    },
                    shrink_strategy: ShrinkStrategy::Default,
                    num_iterations: config.iterations_per_property,
                }

                properties.push(property)
            }
        }
    }

    Ok(properties)
}

// ============================================================================
// SUB-ALGORITHM: Infer Algebraic Properties
// ============================================================================

fn infer_algebraic_properties(
    function: FunctionSignature,
) -> Result<Vec<PropertyTest>, PropertyGenerationError> {

    let mut properties = Vec::new()

    // Check if function signature suggests algebraic properties

    // Commutativity: f(a, b) == f(b, a)
    if function.parameters.len() == 2 &&
       function.parameters[0].type_info == function.parameters[1].type_info {

        let property = PropertyTest {
            id: PropertyTestId::new(),
            name: format!("{}_commutative", function.name),
            property_type: PropertyType::Commutative,
            generators: create_generators_from_params(function.parameters.clone()),
            property_predicate: PropertyPredicate {
                predicate_fn: Box::new(move |inputs| {
                    let result_ab = execute_function(
                        function.id,
                        vec![inputs[0].clone(), inputs[1].clone()],
                    )
                    let result_ba = execute_function(
                        function.id,
                        vec![inputs[1].clone(), inputs[0].clone()],
                    )

                    match (result_ab, result_ba) {
                        (Ok(a), Ok(b)) => compare_values_exact(a, b),
                        (Err(_), Err(_)) => true,  // Both error, acceptable
                        _ => false,
                    }
                }),
                description: format!("{}(a, b) == {}(b, a)", function.name, function.name),
                python_reference: None,
            },
            shrink_strategy: ShrinkStrategy::Default,
            num_iterations: config.iterations_per_property,
        }

        properties.push(property)
    }

    // Idempotence: f(f(a)) == f(a)
    if function.parameters.len() == 1 &&
       function.return_type == function.parameters[0].type_info {

        let property = PropertyTest {
            id: PropertyTestId::new(),
            name: format!("{}_idempotent", function.name),
            property_type: PropertyType::Idempotent,
            generators: create_generators_from_params(function.parameters.clone()),
            property_predicate: PropertyPredicate {
                predicate_fn: Box::new(move |inputs| {
                    let result_once = execute_function(function.id, inputs.clone())

                    match result_once {
                        Ok(output_once) => {
                            let result_twice = execute_function(
                                function.id,
                                vec![TestInput {
                                    parameter_name: function.parameters[0].name.clone(),
                                    value: output_once.clone(),
                                    value_type: function.return_type.clone(),
                                }],
                            )

                            match result_twice {
                                Ok(output_twice) => compare_values_exact(output_once, output_twice),
                                Err(_) => false,
                            }
                        }
                        Err(_) => true,  // If first call fails, acceptable
                    }
                }),
                description: format!("{}({}(a)) == {}(a)", function.name, function.name, function.name),
                python_reference: None,
            },
            shrink_strategy: ShrinkStrategy::Default,
            num_iterations: config.iterations_per_property,
        }

        properties.push(property)
    }

    // Associativity: f(f(a, b), c) == f(a, f(b, c))
    if function.parameters.len() == 2 &&
       function.parameters[0].type_info == function.parameters[1].type_info &&
       function.return_type == function.parameters[0].type_info {

        let property = PropertyTest {
            id: PropertyTestId::new(),
            name: format!("{}_associative", function.name),
            property_type: PropertyType::Associative,
            generators: {
                let mut gens = create_generators_from_params(function.parameters.clone())
                // Add third parameter of same type
                gens.push(gens[0].clone())
                gens
            },
            property_predicate: PropertyPredicate {
                predicate_fn: Box::new(move |inputs| {
                    // f(f(a, b), c)
                    let ab = execute_function(
                        function.id,
                        vec![inputs[0].clone(), inputs[1].clone()],
                    )
                    let ab_c = match ab {
                        Ok(ab_val) => execute_function(
                            function.id,
                            vec![
                                TestInput {
                                    parameter_name: function.parameters[0].name.clone(),
                                    value: ab_val,
                                    value_type: function.return_type.clone(),
                                },
                                inputs[2].clone(),
                            ],
                        ),
                        Err(_) => return true,  // Skip if first call fails
                    }

                    // f(a, f(b, c))
                    let bc = execute_function(
                        function.id,
                        vec![inputs[1].clone(), inputs[2].clone()],
                    )
                    let a_bc = match bc {
                        Ok(bc_val) => execute_function(
                            function.id,
                            vec![
                                inputs[0].clone(),
                                TestInput {
                                    parameter_name: function.parameters[1].name.clone(),
                                    value: bc_val,
                                    value_type: function.return_type.clone(),
                                },
                            ],
                        ),
                        Err(_) => return true,
                    }

                    match (ab_c, a_bc) {
                        (Ok(left), Ok(right)) => compare_values_exact(left, right),
                        (Err(_), Err(_)) => true,
                        _ => false,
                    }
                }),
                description: format!(
                    "{}({}(a, b), c) == {}(a, {}(b, c))",
                    function.name, function.name, function.name, function.name
                ),
                python_reference: None,
            },
            shrink_strategy: ShrinkStrategy::Default,
            num_iterations: config.iterations_per_property,
        }

        properties.push(property)
    }

    Ok(properties)
}

// ============================================================================
// SUB-ALGORITHM: Generate Metamorphic Properties
// ============================================================================

fn generate_metamorphic_properties(
    function: FunctionSignature,
    config: PropertyTestConfig,
) -> Result<Vec<PropertyTest>, PropertyGenerationError> {

    let mut properties = Vec::new()

    // Metamorphic testing: instead of checking absolute correctness,
    // check relations between outputs for related inputs

    // Example: sorting
    if function.name.contains("sort") {
        // Metamorphic relation: reverse(sort(reverse(x))) == sort(x)
        let property = PropertyTest {
            id: PropertyTestId::new(),
            name: format!("{}_metamorphic_reverse", function.name),
            property_type: PropertyType::Metamorphic {
                source_inputs: vec![],  // Will be generated
                transformed_inputs: vec![],  // Will be generated
                relation: MetamorphicRelation::Custom("reverse invariance".to_string()),
            },
            generators: create_generators_from_params(function.parameters.clone()),
            property_predicate: PropertyPredicate {
                predicate_fn: Box::new(move |inputs| {
                    // sort(x)
                    let sorted = execute_function(function.id, inputs.clone())

                    // reverse(x)
                    let reversed = reverse_list(inputs[0].value.clone())

                    // sort(reverse(x))
                    let sort_reversed = execute_function(
                        function.id,
                        vec![TestInput {
                            parameter_name: inputs[0].parameter_name.clone(),
                            value: reversed,
                            value_type: inputs[0].value_type.clone(),
                        }],
                    )

                    match (sorted, sort_reversed) {
                        (Ok(a), Ok(b)) => compare_values_exact(a, b),
                        _ => false,
                    }
                }),
                description: "sort(x) == sort(reverse(x))".to_string(),
                python_reference: None,
            },
            shrink_strategy: ShrinkStrategy::Default,
            num_iterations: config.iterations_per_property,
        }

        properties.push(property)
    }

    Ok(properties)
}
```

### 3.5 Performance Benchmarking Algorithm

```rust
// ============================================================================
// ALGORITHM: Performance Benchmarking
// ============================================================================
// Executes benchmarks for Python vs Rust/WASM
// Measures execution time, memory, throughput
// Detects performance regressions
// ============================================================================

fn run_performance_benchmarks(
    benchmark_suite: BenchmarkSuite,
    python_module: PathBuf,
    rust_binary: PathBuf,
    wasm_module: PathBuf,
    config: BenchmarkConfig,
) -> Result<BenchmarkResults, BenchmarkError> {

    let mut results = Vec::new()

    // Step 1: Initialize runtimes
    let python_runtime = initialize_python_runtime(python_module)?
    let rust_runtime = initialize_rust_runtime(rust_binary)?
    let wasm_runtime = initialize_wasm_runtime(wasm_module)?

    // Step 2: For each benchmark
    for benchmark_spec in benchmark_suite.benchmarks {
        // Step 3: Run Python benchmark
        let python_metrics = benchmark_python_function(
            python_runtime,
            benchmark_spec.function_id,
            benchmark_spec.workload.clone(),
            config,
        )?

        // Step 4: Run Rust benchmark
        let rust_metrics = benchmark_rust_function(
            rust_runtime,
            benchmark_spec.function_id,
            benchmark_spec.workload.clone(),
            config,
        )?

        // Step 5: Run WASM benchmark
        let wasm_metrics = benchmark_wasm_function(
            wasm_runtime,
            benchmark_spec.function_id,
            benchmark_spec.workload.clone(),
            config,
        )?

        // Step 6: Compare metrics
        let comparison = compare_performance(
            python_metrics.clone(),
            rust_metrics.clone(),
            wasm_metrics.clone(),
            config.regression_threshold,
        )?

        results.push(Benchmark {
            id: benchmark_spec.id,
            name: benchmark_spec.name,
            function_id: benchmark_spec.function_id,
            workload: benchmark_spec.workload,
            python_metrics: python_metrics,
            rust_metrics: rust_metrics,
            wasm_metrics: wasm_metrics,
            comparison: comparison,
        })
    }

    Ok(BenchmarkResults {
        suite_name: benchmark_suite.name,
        benchmarks: results,
        summary: generate_benchmark_summary(results),
    })
}

// ============================================================================
// SUB-ALGORITHM: Benchmark Single Function
// ============================================================================

fn benchmark_python_function(
    runtime: PythonRuntime,
    function_id: FunctionId,
    workload: Workload,
    config: BenchmarkConfig,
) -> Result<ExecutionMetrics, BenchmarkError> {

    let mut execution_times = Vec::new()
    let mut memory_samples = Vec::new()

    // Step 1: Warmup iterations
    for _ in 0..workload.warmup_iterations {
        execute_python_function(
            runtime,
            function_id.qualified_name.clone(),
            workload.inputs.clone(),
        )?
    }

    // Step 2: Measured iterations
    for _ in 0..workload.iterations {
        // Start timing
        let start = Instant::now()
        let start_memory = get_memory_usage()

        // Execute function
        let result = execute_python_function(
            runtime,
            function_id.qualified_name.clone(),
            workload.inputs.clone(),
        )?

        // End timing
        let elapsed = start.elapsed()
        let peak_memory = get_peak_memory_usage() - start_memory

        execution_times.push(elapsed.as_nanos() as f64)
        memory_samples.push(peak_memory)
    }

    // Step 3: Calculate statistics
    Ok(calculate_execution_metrics(
        execution_times,
        memory_samples,
        workload.iterations,
    ))
}

// ============================================================================
// SUB-ALGORITHM: Calculate Execution Metrics
// ============================================================================

fn calculate_execution_metrics(
    execution_times: Vec<f64>,  // nanoseconds
    memory_samples: Vec<u64>,   // bytes
    iterations: usize,
) -> ExecutionMetrics {

    // Sort for percentile calculation
    let mut sorted_times = execution_times.clone()
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap())

    let mean_time = sorted_times.iter().sum::<f64>() / sorted_times.len() as f64
    let median_time = sorted_times[sorted_times.len() / 2]
    let min_time = sorted_times[0]
    let max_time = sorted_times[sorted_times.len() - 1]

    // Calculate standard deviation
    let variance = sorted_times
        .iter()
        .map(|&t| (t - mean_time).powi(2))
        .sum::<f64>() / sorted_times.len() as f64
    let std_dev = variance.sqrt()

    // Calculate percentiles
    let p95_index = (sorted_times.len() as f64 * 0.95) as usize
    let p99_index = (sorted_times.len() as f64 * 0.99) as usize
    let percentile_95 = sorted_times[p95_index]
    let percentile_99 = sorted_times[p99_index]

    // Calculate throughput (operations per second)
    let throughput = 1_000_000_000.0 / mean_time  // 1 billion ns = 1 second

    // Memory statistics
    let peak_memory = *memory_samples.iter().max().unwrap_or(&0)
    let avg_memory = memory_samples.iter().sum::<u64>() / memory_samples.len() as u64

    ExecutionMetrics {
        mean_time_ns: mean_time,
        median_time_ns: median_time,
        std_dev_ns: std_dev,
        min_time_ns: min_time,
        max_time_ns: max_time,
        percentile_95_ns: percentile_95,
        percentile_99_ns: percentile_99,
        throughput_ops_per_sec: throughput,
        memory_peak_bytes: peak_memory,
        memory_avg_bytes: avg_memory,
        allocations_count: 0,  // Would need instrumentation
    }
}

// ============================================================================
// SUB-ALGORITHM: Compare Performance and Detect Regressions
// ============================================================================

fn compare_performance(
    python_metrics: ExecutionMetrics,
    rust_metrics: ExecutionMetrics,
    wasm_metrics: ExecutionMetrics,
    regression_threshold: f64,
) -> Result<PerformanceComparison, ComparisonError> {

    // Step 1: Calculate speedup factors
    let rust_speedup = python_metrics.mean_time_ns / rust_metrics.mean_time_ns
    let wasm_speedup = python_metrics.mean_time_ns / wasm_metrics.mean_time_ns

    // Step 2: Calculate memory improvement
    let rust_memory_improvement =
        python_metrics.memory_peak_bytes as f64 / rust_metrics.memory_peak_bytes as f64
    let wasm_memory_improvement =
        python_metrics.memory_peak_bytes as f64 / wasm_metrics.memory_peak_bytes as f64

    // Step 3: Calculate throughput improvement
    let rust_throughput_improvement =
        rust_metrics.throughput_ops_per_sec / python_metrics.throughput_ops_per_sec
    let wasm_throughput_improvement =
        wasm_metrics.throughput_ops_per_sec / python_metrics.throughput_ops_per_sec

    // Step 4: Detect regressions
    // Regression = Rust/WASM is slower than Python beyond threshold
    let rust_regression_detected = rust_speedup < (1.0 - regression_threshold)
    let wasm_regression_detected = wasm_speedup < (1.0 - regression_threshold)

    Ok(PerformanceComparison {
        speedup_factor: rust_speedup,
        memory_improvement: rust_memory_improvement,
        throughput_improvement: rust_throughput_improvement,
        regression_detected: rust_regression_detected || wasm_regression_detected,
        regression_threshold: regression_threshold,
    })
}
```

---

## 4. Input/Output Contracts

### 4.1 Test Agent Input Contract

```yaml
INPUT:
  # From Analysis Agent
  api_surface:
    public_functions: [FunctionSignature]
    public_classes: [ClassSignature]
    type_information: TypeMappings

  # From Specification Generator
  contracts:
    preconditions: [Contract]
    postconditions: [Contract]
    invariants: [Contract]
    type_constraints: [TypeConstraint]

  # From Transpiler Agent
  rust_workspace:
    source_files: [PathBuf]
    module_map: ModuleMap
    generated_types: [RustType]

  # From Build Agent
  build_artifacts:
    rust_binary: PathBuf
    wasm_module: PathBuf
    debug_symbols: Option<PathBuf>

  # Original Python source
  python_source:
    module_path: PathBuf
    test_files: [PathBuf]
    dependencies: [Dependency]

  # Configuration
  test_config:
    modes: ["conformance", "property", "parity", "performance"]
    conformance_config:
      translate_existing_tests: bool
      generate_golden_vectors: bool
      vectors_per_function: usize
      include_boundary_values: bool
    property_config:
      enabled: bool
      iterations_per_property: usize
      generate_metamorphic: bool
    parity_config:
      float_tolerance:
        absolute: f64
        relative: f64
        ulp: u32
      comparison_depth: "shallow" | "deep"
    benchmark_config:
      enabled: bool
      iterations: usize
      warmup_iterations: usize
      regression_threshold: f64  # e.g., 0.1 = 10% slower is regression
```

### 4.2 Test Agent Output Contract

```yaml
OUTPUT:
  status: "success" | "partial_success" | "failure"

  # Generated test artifacts
  test_artifacts:
    rust_test_files: [PathBuf]
    golden_vectors: PathBuf  # JSON/YAML file with test vectors
    property_tests: [PathBuf]
    benchmark_definitions: PathBuf

  # Test execution results
  conformance_results:
    translated_tests:
      total: usize
      passed: usize
      failed: usize
      skipped: usize
    golden_vector_tests:
      total: usize
      passed: usize
      failed: usize
    test_failures: [TestFailure]

  property_test_results:
    total_properties: usize
    passed: usize
    failed: usize
    counterexamples: [Counterexample]

  parity_report:
    total_comparisons: usize
    matches: usize
    mismatches: usize
    severity_breakdown:
      critical: usize
      major: usize
      minor: usize
      informational: usize
    mismatch_details: [ParityMismatch]

  benchmark_results:
    benchmarks: [Benchmark]
    summary:
      avg_rust_speedup: f64
      avg_wasm_speedup: f64
      avg_memory_improvement: f64
      regressions_detected: usize

  # Coverage metrics
  coverage:
    api_coverage_percentage: f64
    line_coverage_percentage: f64
    branch_coverage_percentage: f64
    untested_functions: [FunctionId]

  # Summary
  summary:
    overall_pass_rate: f64
    parity_score: f64  # 0.0 to 1.0
    performance_score: f64  # Average speedup
    recommendation: "ready_for_production" | "needs_review" | "requires_fixes"

  # Error/Warning reporting
  errors: [ErrorReport]
  warnings: [Warning]
```

---

## 5. Error Handling Strategy

### 5.1 Error Categories

```rust
enum TestAgentError {
    // Test Translation Errors
    TestTranslationError {
        source_file: PathBuf,
        line: usize,
        reason: String,
        severity: ErrorSeverity,
    },

    // Execution Errors
    PythonExecutionError {
        function_name: String,
        inputs: Vec<TestInput>,
        stderr: String,
        exit_code: i32,
    },

    RustExecutionError {
        function_name: String,
        inputs: Vec<TestInput>,
        error: String,
    },

    WasmExecutionError {
        function_name: String,
        inputs: Vec<TestInput>,
        trap: String,
    },

    // Comparison Errors
    ValueComparisonError {
        expected_type: String,
        actual_type: String,
        context: String,
    },

    // Resource Errors
    TimeoutError {
        function_name: String,
        timeout_ms: u64,
    },

    MemoryLimitExceeded {
        function_name: String,
        limit_bytes: u64,
        actual_bytes: u64,
    },

    // Configuration Errors
    InvalidConfiguration {
        field: String,
        reason: String,
    },

    // I/O Errors
    IoError {
        operation: String,
        path: PathBuf,
        source: std::io::Error,
    },
}

enum ErrorSeverity {
    Critical,   // Cannot proceed with testing
    Major,      // Can continue but results unreliable
    Minor,      // Can continue with partial results
    Warning,    // Informational, does not affect results
}
```

### 5.2 Error Recovery Strategies

```rust
// ============================================================================
// Error Recovery Strategy
// ============================================================================

fn handle_test_execution_error(
    error: TestAgentError,
    context: TestExecutionContext,
) -> RecoveryAction {

    match error {
        // Timeout: Skip test, mark as timeout failure
        TestAgentError::TimeoutError { function_name, timeout_ms } => {
            log_warning(format!(
                "Function {} timed out after {}ms, skipping",
                function_name, timeout_ms
            ))
            RecoveryAction::SkipTest {
                reason: "timeout",
                continue_testing: true,
            }
        }

        // Memory limit: Skip test, may indicate infinite allocation
        TestAgentError::MemoryLimitExceeded { function_name, limit_bytes, actual_bytes } => {
            log_error(format!(
                "Function {} exceeded memory limit: {} > {}",
                function_name, actual_bytes, limit_bytes
            ))
            RecoveryAction::SkipTest {
                reason: "memory_limit",
                continue_testing: true,
            }
        }

        // Python execution error: Record as test failure, continue
        TestAgentError::PythonExecutionError { function_name, inputs, stderr, .. } => {
            log_error(format!(
                "Python execution failed for {}: {}",
                function_name, stderr
            ))
            RecoveryAction::RecordFailure {
                test_id: generate_test_id(function_name, inputs),
                reason: "python_execution_error",
                continue_testing: true,
            }
        }

        // WASM trap: Record as test failure, continue
        TestAgentError::WasmExecutionError { function_name, inputs, trap } => {
            log_error(format!(
                "WASM trap in {}: {}",
                function_name, trap
            ))
            RecoveryAction::RecordFailure {
                test_id: generate_test_id(function_name, inputs),
                reason: "wasm_trap",
                continue_testing: true,
            }
        }

        // Critical errors: Abort testing
        TestAgentError::InvalidConfiguration { field, reason } => {
            log_critical(format!(
                "Invalid configuration for {}: {}",
                field, reason
            ))
            RecoveryAction::Abort {
                reason: "invalid_configuration",
            }
        }

        // I/O errors: Retry with backoff, then abort
        TestAgentError::IoError { operation, path, source } => {
            if context.retry_count < MAX_RETRIES {
                log_warning(format!(
                    "I/O error during {} on {:?}, retrying ({}/{})",
                    operation, path, context.retry_count + 1, MAX_RETRIES
                ))
                RecoveryAction::Retry {
                    backoff_ms: calculate_backoff(context.retry_count),
                }
            } else {
                log_critical(format!(
                    "I/O error during {} on {:?} after {} retries: {}",
                    operation, path, MAX_RETRIES, source
                ))
                RecoveryAction::Abort {
                    reason: "io_error_max_retries",
                }
            }
        }

        _ => {
            log_error(format!("Unhandled error: {:?}", error))
            RecoveryAction::SkipTest {
                reason: "unhandled_error",
                continue_testing: true,
            }
        }
    }
}

enum RecoveryAction {
    Continue,
    SkipTest { reason: &'static str, continue_testing: bool },
    RecordFailure { test_id: TestCaseId, reason: &'static str, continue_testing: bool },
    Retry { backoff_ms: u64 },
    Abort { reason: &'static str },
}
```

### 5.3 Graceful Degradation

```rust
// ============================================================================
// Graceful Degradation Strategy
// ============================================================================

fn execute_test_suite_with_degradation(
    test_suite: TestSuite,
    context: TestExecutionContext,
) -> TestResults {

    let mut results = TestResults::new()

    // Attempt full test suite execution
    match execute_test_suite(test_suite.clone(), context.clone()) {
        Ok(full_results) => return full_results,
        Err(error) => {
            log_warning(format!(
                "Full test suite execution failed: {:?}, attempting degraded mode",
                error
            ))
        }
    }

    // Degradation Level 1: Skip performance benchmarks
    if test_suite.mode.contains(TestMode::Performance) {
        log_info("Skipping performance benchmarks due to execution errors")

        let degraded_suite = TestSuite {
            mode: test_suite.mode.without(TestMode::Performance),
            ..test_suite.clone()
        }

        match execute_test_suite(degraded_suite, context.clone()) {
            Ok(partial_results) => {
                results = partial_results
                results.metadata.degradation_level = 1
                return results
            }
            Err(_) => {}
        }
    }

    // Degradation Level 2: Skip property-based tests
    if test_suite.mode.contains(TestMode::PropertyBased) {
        log_info("Skipping property-based tests due to execution errors")

        let degraded_suite = TestSuite {
            mode: test_suite.mode.without(TestMode::PropertyBased),
            ..test_suite.clone()
        }

        match execute_test_suite(degraded_suite, context.clone()) {
            Ok(partial_results) => {
                results = partial_results
                results.metadata.degradation_level = 2
                return results
            }
            Err(_) => {}
        }
    }

    // Degradation Level 3: Only run conformance tests
    log_warning("Running minimal conformance tests only")

    let minimal_suite = TestSuite {
        mode: TestMode::Conformance,
        ..test_suite.clone()
    }

    match execute_test_suite(minimal_suite, context) {
        Ok(minimal_results) => {
            results = minimal_results
            results.metadata.degradation_level = 3
            results
        }
        Err(error) => {
            log_critical(format!("All test modes failed: {:?}", error))
            results.metadata.degradation_level = 4
            results.status = TestStatus::Failed
            results
        }
    }
}
```

---

## 6. London School TDD Test Points

### 6.1 Test Hierarchy

Following London School TDD (Outside-In, Mockist approach), we test the Test Agent through:

```
Acceptance Tests (End-to-End)
    │
    ├─→ Test Suite Translation Acceptance
    │   - Given Python test file, When translate, Then Rust test compiles
    │   - Given pytest parametrize, When translate, Then Rust macro generated
    │
    ├─→ Golden Vector Generation Acceptance
    │   - Given Python function, When generate vectors, Then N vectors created
    │   - Given boundary config, When generate, Then edge cases included
    │
    ├─→ Parity Validation Acceptance
    │   - Given mismatched outputs, When validate, Then mismatch reported
    │   - Given floating-point diff, When compare with tolerance, Then pass
    │
    └─→ Performance Benchmarking Acceptance
        - Given Python and Rust implementations, When benchmark, Then speedup calculated
        - Given regression threshold, When regression detected, Then report flagged

Component Tests (Agent-Level)
    │
    ├─→ TestTranslator Component
    │   - Mock: AST parser, Rust code generator
    │   - Verify: Correct assertion translation
    │   - Verify: Test attributes preserved
    │
    ├─→ GoldenVectorGenerator Component
    │   - Mock: Python runtime, input generator
    │   - Verify: Execution results captured
    │   - Verify: Boundary values generated
    │
    ├─→ ParityValidator Component
    │   - Mock: Python/Rust/WASM runtimes
    │   - Verify: Tolerance-based comparison
    │   - Verify: ULP distance calculation
    │
    └─→ PerformanceBenchmarker Component
        - Mock: Runtime environments
        - Verify: Statistical metrics calculated
        - Verify: Regression detection logic

Unit Tests (Algorithm-Level)
    │
    ├─→ Assertion Translation
    │   - Test: unittest.assertEqual → assert_eq!
    │   - Test: pytest approx → assert_approx_eq!
    │   - Test: assertRaises → #[should_panic]
    │
    ├─→ Value Comparison
    │   - Test: Floating-point ULP distance
    │   - Test: Nested collection comparison
    │   - Test: Type mismatch detection
    │
    ├─→ Property Derivation
    │   - Test: Commutativity detection
    │   - Test: Idempotence detection
    │   - Test: Metamorphic relation generation
    │
    └─→ Metrics Calculation
        - Test: Mean/median/percentile calculation
        - Test: Speedup factor calculation
        - Test: Regression threshold logic
```

### 6.2 Mock Boundaries

```rust
// ============================================================================
// Mock Interfaces (Test Doubles)
// ============================================================================

trait PythonRuntimeMock {
    fn execute_function(&self, name: String, inputs: Vec<TestInput>)
        -> Result<PythonExecutionResult, ExecutionError>

    fn get_version(&self) -> PythonVersion
}

trait RustRuntimeMock {
    fn execute_function(&self, name: String, inputs: Vec<TestInput>)
        -> Result<RustExecutionResult, ExecutionError>

    fn compile_test_binary(&self, source: PathBuf)
        -> Result<PathBuf, CompilationError>
}

trait WasmRuntimeMock {
    fn instantiate_module(&self, module_path: PathBuf)
        -> Result<WasmInstance, InstantiationError>

    fn call_function(&self, instance: WasmInstance, name: String, inputs: Vec<Value>)
        -> Result<Value, TrapError>
}

trait InputGeneratorMock {
    fn generate(&self, param_type: TypeInfo)
        -> Result<Value, GenerationError>

    fn generate_boundary_values(&self, param_type: TypeInfo)
        -> Result<Vec<Value>, GenerationError>
}

trait MetricsCollectorMock {
    fn start_measurement(&mut self, operation: &str)

    fn end_measurement(&mut self, operation: &str) -> ExecutionMetrics

    fn get_memory_usage(&self) -> u64
}
```

### 6.3 Example Test Cases

```rust
// ============================================================================
// ACCEPTANCE TEST: Golden Vector Generation
// ============================================================================

#[test]
fn test_golden_vector_generation_end_to_end() {
    // Arrange
    let python_module = create_test_python_module("simple_math.py", r#"
def add(a: int, b: int) -> int:
    return a + b
    "#)

    let api_surface = ApiSurface {
        public_functions: vec![
            FunctionSignature {
                id: FunctionId::new(),
                name: "add".to_string(),
                qualified_name: QualifiedName::from("simple_math.add"),
                parameters: vec![
                    Parameter { name: "a", type_info: TypeInfo::Int { min: i64::MIN, max: i64::MAX } },
                    Parameter { name: "b", type_info: TypeInfo::Int { min: i64::MIN, max: i64::MAX } },
                ],
                return_type: TypeInfo::Int { min: i64::MIN, max: i64::MAX },
            }
        ],
    }

    let config = VectorGenerationConfig {
        vectors_per_function: 10,
        include_boundary_values: true,
        deterministic: true,
        seed: Some(42),
        ..Default::default()
    }

    // Act
    let result = generate_golden_vectors(python_module, api_surface, config)

    // Assert
    assert!(result.is_ok())
    let vectors = result.unwrap()

    // Verify we have at least 10 vectors (plus boundary values)
    assert!(vectors.len() >= 10)

    // Verify all vectors have valid Python outputs
    for vector in &vectors {
        assert!(vector.python_output.return_value.is_some())
        assert!(vector.python_output.exception.is_none())
    }

    // Verify boundary values are included
    let has_zero = vectors.iter().any(|v|
        v.inputs.iter().any(|i| matches!(i.value, Value::Int(0)))
    )
    assert!(has_zero, "Should include zero as boundary value")
}

// ============================================================================
// COMPONENT TEST: Parity Validator
// ============================================================================

#[test]
fn test_parity_validator_detects_floating_point_drift() {
    // Arrange
    let mut python_runtime_mock = MockPythonRuntime::new()
    let mut rust_runtime_mock = MockRustRuntime::new()

    // Python returns 0.1 + 0.2 = 0.30000000000000004 (floating-point error)
    python_runtime_mock
        .expect_execute_function()
        .returning(|_, _| Ok(PythonExecutionResult {
            return_value: Some(Value::Float(0.30000000000000004)),
            ..Default::default()
        }))

    // Rust returns 0.3 (exact)
    rust_runtime_mock
        .expect_execute_function()
        .returning(|_, _| Ok(RustExecutionResult {
            return_value: Some(Value::Float(0.3)),
            ..Default::default()
        }))

    let validator = ParityValidator::new(
        python_runtime_mock,
        rust_runtime_mock,
    )

    let tolerance = Tolerance {
        absolute: Some(1e-10),
        relative: None,
        ulp: Some(2),  // Allow 2 ULPs difference
    }

    // Act
    let result = validator.compare_outputs(
        Some(Value::Float(0.30000000000000004)),
        Some(Value::Float(0.3)),
        ComparisonConfig { float_tolerance: tolerance, ..Default::default() },
        "Rust",
    )

    // Assert - Should pass with ULP tolerance
    assert!(result.is_ok())
    assert!(result.unwrap().is_none(), "Should not detect mismatch within ULP tolerance")
}

// ============================================================================
// UNIT TEST: Floating-Point ULP Distance Calculation
// ============================================================================

#[test]
fn test_calculate_ulp_distance_same_value() {
    let a = 1.0
    let b = 1.0
    let distance = calculate_ulp_distance(a, b)
    assert_eq!(distance, 0)
}

#[test]
fn test_calculate_ulp_distance_adjacent_floats() {
    let a = 1.0
    let b = f64::from_bits(a.to_bits() + 1)  // Next representable float
    let distance = calculate_ulp_distance(a, b)
    assert_eq!(distance, 1)
}

#[test]
fn test_calculate_ulp_distance_different_signs() {
    let a = 1.0
    let b = -1.0
    let distance = calculate_ulp_distance(a, b)
    assert_eq!(distance, u32::MAX)  // Different signs = large distance
}

#[test]
fn test_calculate_ulp_distance_near_zero() {
    let a = 0.0
    let b = f64::EPSILON
    let distance = calculate_ulp_distance(a, b)
    assert!(distance < 10, "Should be very close to zero")
}

// ============================================================================
// UNIT TEST: Property Derivation
// ============================================================================

#[test]
fn test_detect_commutative_property() {
    // Arrange
    let function = FunctionSignature {
        name: "add".to_string(),
        parameters: vec![
            Parameter { name: "a", type_info: TypeInfo::Int { .. } },
            Parameter { name: "b", type_info: TypeInfo::Int { .. } },
        ],
        return_type: TypeInfo::Int { .. },
        ..Default::default()
    }

    // Act
    let properties = infer_algebraic_properties(function).unwrap()

    // Assert
    assert!(properties.iter().any(|p|
        matches!(p.property_type, PropertyType::Commutative)
    ), "Should detect commutative property for binary function with same parameter types")
}

#[test]
fn test_detect_idempotent_property() {
    // Arrange
    let function = FunctionSignature {
        name: "normalize".to_string(),
        parameters: vec![
            Parameter { name: "s", type_info: TypeInfo::String { .. } },
        ],
        return_type: TypeInfo::String { .. },
        ..Default::default()
    }

    // Act
    let properties = infer_algebraic_properties(function).unwrap()

    // Assert
    assert!(properties.iter().any(|p|
        matches!(p.property_type, PropertyType::Idempotent)
    ), "Should detect idempotent property for unary function with same input/output type")
}

// ============================================================================
// UNIT TEST: Benchmark Metrics Calculation
// ============================================================================

#[test]
fn test_calculate_execution_metrics_percentiles() {
    // Arrange
    let execution_times = vec![
        100.0, 150.0, 200.0, 250.0, 300.0,
        350.0, 400.0, 450.0, 500.0, 550.0,
    ]  // nanoseconds
    let memory_samples = vec![1024; 10]  // bytes

    // Act
    let metrics = calculate_execution_metrics(execution_times.clone(), memory_samples, 10)

    // Assert
    assert_eq!(metrics.mean_time_ns, 325.0)
    assert_eq!(metrics.median_time_ns, 300.0)
    assert_eq!(metrics.min_time_ns, 100.0)
    assert_eq!(metrics.max_time_ns, 550.0)

    // 95th percentile should be around 500-550
    assert!(metrics.percentile_95_ns >= 500.0)
    assert!(metrics.percentile_95_ns <= 550.0)
}

#[test]
fn test_detect_performance_regression() {
    // Arrange
    let python_metrics = ExecutionMetrics {
        mean_time_ns: 1000.0,
        ..Default::default()
    }

    let rust_metrics = ExecutionMetrics {
        mean_time_ns: 1200.0,  // 20% slower than Python - REGRESSION!
        ..Default::default()
    }

    let regression_threshold = 0.1  // 10% tolerance

    // Act
    let comparison = compare_performance(
        python_metrics,
        rust_metrics,
        Default::default(),
        regression_threshold,
    ).unwrap()

    // Assert
    assert!(comparison.regression_detected, "Should detect regression when Rust is >10% slower")
    assert!(comparison.speedup_factor < 1.0, "Speedup factor should be < 1.0 (slowdown)")
}

// ============================================================================
// INTEGRATION TEST: Full Test Suite Execution
// ============================================================================

#[test]
fn test_full_test_suite_execution_integration() {
    // This would be a more comprehensive integration test
    // combining multiple components with real (not mocked) dependencies

    // Arrange: Create a real Python module and Rust translation
    let temp_dir = tempdir().unwrap()
    let python_file = temp_dir.path().join("math_ops.py")
    std::fs::write(&python_file, r#"
def multiply(a: int, b: int) -> int:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b
    "#).unwrap()

    let rust_file = temp_dir.path().join("math_ops.rs")
    std::fs::write(&rust_file, r#"
pub fn multiply(a: i64, b: i64) -> i64 {
    a * b
}

pub fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}
    "#).unwrap()

    // Compile Rust to library
    let rust_binary = compile_rust_library(&rust_file).unwrap()

    // Act: Run full test suite
    let test_agent = TestAgent::new(TestAgentConfig::default())
    let results = test_agent.execute_full_suite(
        python_file,
        rust_binary,
        None,  // No WASM for this test
    ).unwrap()

    // Assert: Verify comprehensive results
    assert_eq!(results.status, TestStatus::Success)
    assert!(results.conformance_results.total > 0)
    assert!(results.parity_report.matches > 0)
    assert_eq!(results.parity_report.mismatches.len(), 0)
}
```

### 6.4 Test Coverage Requirements

```yaml
Test Coverage Requirements:
  Unit Tests:
    - All comparison algorithms: 100%
    - Floating-point ULP calculation: 100%
    - Property derivation logic: 100%
    - Metrics calculation: 100%
    - Boundary value generation: 100%

  Component Tests:
    - TestTranslator: >90% coverage
    - GoldenVectorGenerator: >90% coverage
    - ParityValidator: >95% coverage (critical for correctness)
    - PerformanceBenchmarker: >85% coverage

  Integration Tests:
    - End-to-end test translation: >80%
    - End-to-end parity validation: >85%
    - End-to-end benchmarking: >75%

  Acceptance Tests:
    - All FR-2.6.* requirements: 100%
    - All error scenarios: 100%
    - All degradation paths: 100%
```

---

## 7. Summary

### 7.1 Agent Capabilities

The Test Agent provides:

1. **Test Translation**: Converts Python tests to idiomatic Rust tests
2. **Golden Vector Generation**: Creates reference test data from Python execution
3. **Parity Validation**: Tolerance-aware comparison of Python vs Rust/WASM outputs
4. **Property-Based Testing**: Derives and tests algebraic/metamorphic properties
5. **Performance Benchmarking**: Measures and compares execution metrics
6. **Comprehensive Reporting**: Generates detailed test, parity, and performance reports

### 7.2 Key Algorithms

- **Test Translation**: AST-based translation preserving test semantics
- **Golden Vector Generation**: Input generation with boundary value coverage
- **Floating-Point Comparison**: ULP-based tolerance for robust comparison
- **Property Derivation**: Type-driven inference of algebraic properties
- **Performance Metrics**: Statistical analysis with regression detection

### 7.3 Quality Assurance

- **Error Recovery**: Graceful degradation with multiple fallback levels
- **Tolerance Configuration**: Flexible comparison thresholds
- **Meta-Testing**: London School TDD with comprehensive test coverage
- **Observability**: Detailed logging and metrics collection

---

**SPARC Phase 2 (Pseudocode): COMPLETE**

This pseudocode defines **HOW** the Test Agent operates. Next phase will detail the implementation architecture.
