# PORTALIS - SPARC Phase 5 (Completion) Action Plan
## Executable Roadmap to Production

**Plan Date:** 2025-10-03
**Target Completion:** Week 37 (2025-06-13)
**Owner:** Engineering Leadership
**Status:** 🟢 **READY TO EXECUTE**

---

## EXECUTIVE SUMMARY

This action plan provides a **week-by-week executable roadmap** to complete SPARC Phase 5 (Completion) for the Portalis project. The plan transforms 52,360 lines of documentation and architectural planning into a **working production system in 37 weeks**.

**Key Milestones:**
- **Week 0:** Proof-of-concept and stakeholder approval
- **Week 3:** Foundation complete (agent framework operational)
- **Week 11:** MVP complete (Script Mode working)
- **Week 21:** Library Mode complete
- **Week 29:** NVIDIA acceleration integrated
- **Week 37:** Production deployment

**Investment:** ~$500K-900K (engineering + infrastructure)

---

## PHASE 0: FOUNDATION (WEEKS 0-3)

### Week 0: Approval & Proof-of-Concept

**Objective:** Secure approval and validate core assumptions

#### Day 1-2: Stakeholder Approval

**Actions:**
- [ ] Present consolidated completion report to leadership
- [ ] Review 37-week timeline and $500K-900K budget
- [ ] Secure commitment to allocate 3-engineer team
- [ ] Approve Phase 0-1 priorities (no GPU work until Phase 3)

**Deliverables:**
- Executive approval document
- Team allocation confirmation
- Budget approval

**Success Criteria:**
- ✅ Go/No-Go decision made
- ✅ Team assigned (2 Rust engineers, 1 Python engineer)
- ✅ Budget approved

**Owner:** VP Engineering / CTO

---

#### Day 3-5: Proof-of-Concept

**Objective:** Build simplest possible translator to validate assumptions

**Implementation:**
```rust
// Simple POC: fibonacci.py → fibonacci.rs → fibonacci.wasm

// Step 1: Parse (manual AST construction for POC)
fn parse_python(source: &str) -> PythonAST { ... }

// Step 2: Translate (simple pattern matching)
fn translate_to_rust(ast: PythonAST) -> String { ... }

// Step 3: Compile (invoke rustc)
fn compile_to_wasm(rust_code: &str) -> Result<Vec<u8>> { ... }

// Step 4: Execute WASM
fn execute_wasm(wasm: &[u8], input: i32) -> i32 { ... }
```

**Test Case:**
```python
# fibonacci.py
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Expected Output:**
```rust
// fibonacci.rs (generated)
pub fn fibonacci(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}
```

**Validation:**
- Python: `fibonacci(10)` → `55`
- WASM: `fibonacci(10)` → `55`

**Success Criteria:**
- ✅ POC translates fibonacci.py → WASM
- ✅ WASM output matches Python output
- ✅ Total time: 3-5 days (one engineer)
- ✅ Architectural assumptions validated

**Go/No-Go Decision:**
- **GO:** If POC succeeds, proceed to Phase 0
- **NO-GO:** If POC fails, reassess approach or pivot

**Owner:** Senior Rust Engineer

---

### Week 1: Rust Workspace Setup

**Objective:** Create project structure and build system

#### Actions

**1. Initialize Rust Workspace**
```bash
cd /workspace/portalis
cargo init --name portalis
mkdir -p src/{agents,core,orchestration}
mkdir -p tests/{unit,integration,e2e}
```

**2. Configure Cargo.toml**
```toml
[workspace]
members = [
    "agents/ingest",
    "agents/analysis",
    "agents/specgen",
    "agents/transpiler",
    "agents/build",
    "agents/test",
    "agents/packaging",
    "core",
    "orchestration",
]

[workspace.dependencies]
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
async-trait = "0.1"
```

**3. Create Agent Trait**
```rust
// core/src/agent.rs
#[async_trait]
pub trait Agent: Send + Sync {
    type Input: Send + Sync;
    type Output: Send + Sync;

    async fn execute(&self, input: Self::Input) -> Result<Self::Output>;
    fn name(&self) -> &str;
    fn capabilities(&self) -> Vec<Capability>;
}

pub enum Capability {
    Parsing,
    TypeInference,
    CodeGeneration,
    Compilation,
    Testing,
}
```

**4. Setup Testing Infrastructure**
```rust
// tests/conftest.rs
pub fn setup_test_agent<A: Agent>() -> A { ... }
pub fn mock_input() -> MockInput { ... }
pub fn assert_output(output: Output) { ... }
```

**5. CI/CD Integration**
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all
      - run: cargo clippy -- -D warnings
```

**Deliverables:**
- [ ] Rust workspace with 7 agent crates
- [ ] Core library with Agent trait
- [ ] Test infrastructure
- [ ] CI/CD pipeline operational

**Success Criteria:**
- ✅ `cargo build` succeeds
- ✅ `cargo test` passes (empty tests OK)
- ✅ CI/CD pipeline runs successfully
- ✅ All crates stubbed out

**Team:** 1 Rust engineer (full-time)

**Estimated Lines of Code:** ~500 lines (boilerplate)

---

### Week 2: Agent Framework Implementation

**Objective:** Implement core agent infrastructure

#### Actions

**1. Message Bus**
```rust
// core/src/message_bus.rs
pub struct MessageBus {
    channels: HashMap<AgentId, Sender<Message>>,
}

impl MessageBus {
    pub async fn send(&self, to: AgentId, msg: Message) -> Result<()> { ... }
    pub async fn broadcast(&self, msg: Message) -> Result<()> { ... }
    pub fn subscribe(&mut self, agent: AgentId) -> Receiver<Message> { ... }
}
```

**2. Agent Registry**
```rust
// core/src/registry.rs
pub struct AgentRegistry {
    agents: HashMap<AgentId, Box<dyn Agent>>,
}

impl AgentRegistry {
    pub fn register<A: Agent>(&mut self, agent: A) { ... }
    pub fn get(&self, id: AgentId) -> Option<&dyn Agent> { ... }
    pub fn list_by_capability(&self, cap: Capability) -> Vec<AgentId> { ... }
}
```

**3. State Management**
```rust
// core/src/state.rs
#[derive(Debug, Clone)]
pub struct PipelineState {
    pub phase: Phase,
    pub artifacts: HashMap<String, Artifact>,
    pub errors: Vec<Error>,
}

pub enum Phase {
    Ingesting,
    Analyzing,
    Generating,
    Transpiling,
    Building,
    Testing,
    Packaging,
    Complete,
}
```

**4. Unit Tests (London School TDD)**
```rust
// tests/unit/test_message_bus.rs
#[tokio::test]
async fn test_message_bus_send() {
    let bus = MessageBus::new();
    let (tx, rx) = channel();
    bus.register_agent("agent1", tx);

    bus.send("agent1", Message::Start).await.unwrap();

    let msg = rx.recv().await.unwrap();
    assert_eq!(msg, Message::Start);
}
```

**Deliverables:**
- [ ] Message bus implementation
- [ ] Agent registry
- [ ] State management
- [ ] Unit tests (>80% coverage)

**Success Criteria:**
- ✅ Agents can communicate via message bus
- ✅ State transitions tracked correctly
- ✅ All tests passing
- ✅ Coverage >80%

**Team:** 2 Rust engineers

**Estimated Lines of Code:** ~1,000 lines

---

### Week 3: Pipeline Orchestration

**Objective:** Coordinate agent execution

#### Actions

**1. Pipeline Coordinator**
```rust
// orchestration/src/pipeline.rs
pub struct Pipeline {
    registry: AgentRegistry,
    bus: MessageBus,
    state: PipelineState,
}

impl Pipeline {
    pub async fn execute(&mut self, input: Input) -> Result<Output> {
        self.transition_to(Phase::Ingesting);
        let ast = self.run_agent::<IngestAgent>(input).await?;

        self.transition_to(Phase::Analyzing);
        let analysis = self.run_agent::<AnalysisAgent>(ast).await?;

        // ... continue for all agents

        Ok(output)
    }

    async fn run_agent<A: Agent>(&mut self, input: A::Input) -> Result<A::Output> {
        let agent = self.registry.get::<A>()?;
        agent.execute(input).await
    }
}
```

**2. Error Recovery**
```rust
// orchestration/src/recovery.rs
pub struct ErrorRecovery {
    checkpoints: Vec<Checkpoint>,
}

impl ErrorRecovery {
    pub async fn rollback(&mut self, to_phase: Phase) -> Result<()> { ... }
    pub fn create_checkpoint(&mut self, state: &PipelineState) { ... }
}
```

**3. Integration Tests**
```rust
// tests/integration/test_pipeline.rs
#[tokio::test]
async fn test_dummy_pipeline() {
    let pipeline = Pipeline::new();
    let input = DummyInput::new();

    let output = pipeline.execute(input).await.unwrap();

    assert_eq!(output.phase, Phase::Complete);
}
```

**Deliverables:**
- [ ] Pipeline orchestrator
- [ ] Error recovery mechanism
- [ ] Integration tests
- [ ] Dummy workflow executes successfully

**Success Criteria:**
- ✅ Dummy workflow executes end-to-end
- ✅ Error recovery works (rollback tested)
- ✅ Integration tests passing
- ✅ Ready for agent implementation

**Team:** 2 Rust engineers

**Estimated Lines of Code:** ~500 lines

---

### Phase 0 Gate Review (End of Week 3)

**Criteria:**
- ✅ Rust workspace builds successfully
- ✅ Agent framework operational (message bus, registry, state)
- ✅ Pipeline orchestrator executes dummy workflow
- ✅ CI/CD running on every commit
- ✅ Test coverage >80%
- ✅ All Phase 0 deliverables complete

**Decision:**
- **PASS:** Proceed to Phase 1
- **FAIL:** Extend Phase 0, address blockers

**Review Meeting:** Leadership + Engineering Team

---

## PHASE 1: MVP SCRIPT MODE (WEEKS 4-11)

### Week 4-5: Ingest Agent (AST Parser)

**Objective:** Parse Python source code into AST

#### Implementation

**1. Python Parser**
```rust
// agents/ingest/src/parser.rs
use rustpython_parser::{parse_program, ast};

pub struct PythonParser;

impl PythonParser {
    pub fn parse(&self, source: &str) -> Result<ast::Module> {
        parse_program(source, "<input>")
            .map_err(|e| Error::ParseError(e))
    }
}
```

**2. AST Representation**
```rust
// agents/ingest/src/ast.rs
#[derive(Debug, Clone)]
pub struct PortalisAST {
    pub functions: Vec<Function>,
    pub classes: Vec<Class>,
    pub imports: Vec<Import>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Vec<Statement>,
}
```

**3. Ingest Agent Implementation**
```rust
// agents/ingest/src/agent.rs
pub struct IngestAgent {
    parser: PythonParser,
}

#[async_trait]
impl Agent for IngestAgent {
    type Input = String; // Python source code
    type Output = PortalisAST;

    async fn execute(&self, source: String) -> Result<PortalisAST> {
        let python_ast = self.parser.parse(&source)?;
        let portalis_ast = self.convert(python_ast)?;
        Ok(portalis_ast)
    }
}
```

**4. Tests (London School TDD)**
```rust
// tests/unit/test_ingest_agent.rs
#[tokio::test]
async fn test_parse_simple_function() {
    let agent = IngestAgent::new();
    let source = r#"
def add(a, b):
    return a + b
"#;

    let ast = agent.execute(source.to_string()).await.unwrap();

    assert_eq!(ast.functions.len(), 1);
    assert_eq!(ast.functions[0].name, "add");
    assert_eq!(ast.functions[0].params.len(), 2);
}

#[tokio::test]
async fn test_parse_with_imports() {
    // Test import extraction
}

#[tokio::test]
async fn test_parse_error_handling() {
    // Test syntax error handling
}
```

**Test Scenarios:**
- Simple function (10-20 LOC)
- Function with type hints
- Multiple functions
- Imports and dependencies
- Syntax errors
- Edge cases (empty file, comments only)

**Deliverables:**
- [ ] Python parser (using rustpython-parser)
- [ ] AST conversion to Portalis format
- [ ] Ingest Agent implementation
- [ ] 15+ unit tests (>85% coverage)

**Success Criteria:**
- ✅ Parse simple Python functions
- ✅ Extract imports and dependencies
- ✅ Handle syntax errors gracefully
- ✅ Test coverage >85%
- ✅ Integration with pipeline

**Team:** 1 Python engineer, 1 Rust engineer

**Estimated Lines of Code:** ~2,000 lines

---

### Week 6-7: Analysis Agent (Type Inference)

**Objective:** Infer types and analyze code semantics

#### Implementation

**1. Type Inference Engine**
```rust
// agents/analysis/src/inference.rs
pub struct TypeInferenceEngine {
    nemo_client: Option<NeMoClient>, // Optional LLM assistance
}

impl TypeInferenceEngine {
    pub fn infer_function_types(&self, func: &Function) -> Result<TypedFunction> {
        // 1. Check for type hints (Python 3.5+)
        if let Some(hint) = func.return_type {
            return Ok(self.map_python_type(hint)?);
        }

        // 2. Infer from usage patterns
        let inferred = self.analyze_usage(func)?;

        // 3. Fallback to NeMo if ambiguous
        if inferred.confidence < 0.8 {
            return self.nemo_client.as_ref()
                .ok_or(Error::NoNeMo)?
                .infer_types(func).await?;
        }

        Ok(inferred)
    }
}
```

**2. API Extraction**
```rust
// agents/analysis/src/api.rs
pub struct APIExtractor;

impl APIExtractor {
    pub fn extract(&self, ast: &PortalisAST) -> Result<APIContract> {
        let mut api = APIContract::default();

        for func in &ast.functions {
            api.functions.push(FunctionSignature {
                name: func.name.clone(),
                params: self.extract_params(func)?,
                return_type: self.extract_return(func)?,
            });
        }

        Ok(api)
    }
}
```

**3. Analysis Agent**
```rust
// agents/analysis/src/agent.rs
pub struct AnalysisAgent {
    inference: TypeInferenceEngine,
    extractor: APIExtractor,
}

#[async_trait]
impl Agent for AnalysisAgent {
    type Input = PortalisAST;
    type Output = Analysis;

    async fn execute(&self, ast: PortalisAST) -> Result<Analysis> {
        let typed_ast = self.inference.infer_types(&ast).await?;
        let api = self.extractor.extract(&typed_ast)?;

        Ok(Analysis {
            typed_ast,
            api,
            confidence: self.compute_confidence(&typed_ast),
        })
    }
}
```

**4. Tests**
```rust
// tests/unit/test_analysis_agent.rs
#[tokio::test]
async fn test_infer_types_from_hints() {
    let agent = AnalysisAgent::new();
    let ast = create_typed_ast(); // Helper with type hints

    let analysis = agent.execute(ast).await.unwrap();

    assert_eq!(analysis.typed_ast.functions[0].return_type, Type::Int);
    assert!(analysis.confidence > 0.9);
}

#[tokio::test]
async fn test_infer_types_from_usage() {
    // Test without type hints
}

#[tokio::test]
async fn test_nemo_fallback() {
    // Test NeMo integration for ambiguous cases
    let agent = AnalysisAgent::with_nemo(mock_nemo_client());
    // ...
}
```

**Deliverables:**
- [ ] Type inference engine (rule-based)
- [ ] API extraction
- [ ] NeMo integration (optional fallback)
- [ ] Analysis Agent implementation
- [ ] 20+ unit tests (>85% coverage)

**Success Criteria:**
- ✅ Infer types from hints (100% accuracy)
- ✅ Infer types from usage (>80% accuracy)
- ✅ Extract API contracts correctly
- ✅ NeMo fallback works for ambiguous cases
- ✅ Test coverage >85%

**Team:** 2 Rust engineers

**Estimated Lines of Code:** ~3,000 lines

---

### Week 8-9: Specification Generator & Transpiler Agent

**Objective:** Generate Rust code from analyzed Python

#### Implementation

**1. Specification Generator**
```rust
// agents/specgen/src/generator.rs
pub struct SpecificationGenerator;

impl SpecificationGenerator {
    pub fn generate_spec(&self, analysis: &Analysis) -> Result<RustSpec> {
        let mut spec = RustSpec::default();

        for func in &analysis.typed_ast.functions {
            spec.items.push(self.generate_function_spec(func)?);
        }

        Ok(spec)
    }

    fn generate_function_spec(&self, func: &TypedFunction) -> Result<RustItem> {
        Ok(RustItem::Function {
            name: func.name.clone(),
            params: self.map_params(&func.params)?,
            return_type: self.map_type(&func.return_type)?,
        })
    }
}
```

**2. Transpiler Agent**
```rust
// agents/transpiler/src/agent.rs
pub struct TranspilerAgent {
    specgen: SpecificationGenerator,
    codegen: CodeGenerator,
    nemo_client: Option<NeMoClient>,
}

#[async_trait]
impl Agent for TranspilerAgent {
    type Input = Analysis;
    type Output = RustCode;

    async fn execute(&self, analysis: Analysis) -> Result<RustCode> {
        // Generate Rust specification
        let spec = self.specgen.generate_spec(&analysis)?;

        // Generate Rust code (rule-based)
        let mut code = self.codegen.generate(&spec)?;

        // Optionally enhance with NeMo
        if let Some(nemo) = &self.nemo_client {
            code = nemo.enhance_code(&code).await?;
        }

        Ok(code)
    }
}
```

**3. Code Generator**
```rust
// agents/transpiler/src/codegen.rs
pub struct CodeGenerator;

impl CodeGenerator {
    pub fn generate(&self, spec: &RustSpec) -> Result<String> {
        let mut output = String::new();

        for item in &spec.items {
            output.push_str(&self.generate_item(item)?);
            output.push('\n');
        }

        Ok(output)
    }

    fn generate_item(&self, item: &RustItem) -> Result<String> {
        match item {
            RustItem::Function { name, params, return_type } => {
                Ok(format!(
                    "pub fn {}({}) -> {} {{\n    todo!()\n}}",
                    name,
                    self.format_params(params),
                    return_type
                ))
            }
        }
    }
}
```

**4. Tests**
```rust
// tests/unit/test_transpiler_agent.rs
#[tokio::test]
async fn test_transpile_simple_function() {
    let agent = TranspilerAgent::new();
    let analysis = create_analysis(); // fibonacci function

    let rust_code = agent.execute(analysis).await.unwrap();

    assert!(rust_code.contains("pub fn fibonacci"));
    assert!(rust_code.contains("-> i32"));
}

#[tokio::test]
async fn test_transpile_with_nemo() {
    // Test NeMo-enhanced code generation
}
```

**Deliverables:**
- [ ] Specification generator
- [ ] Code generator (rule-based)
- [ ] Transpiler Agent implementation
- [ ] NeMo integration (optional)
- [ ] 25+ unit tests (>85% coverage)

**Success Criteria:**
- ✅ Generate valid Rust code
- ✅ Handle common Python patterns
- ✅ Preserve semantics
- ✅ NeMo enhancement works
- ✅ Test coverage >85%

**Team:** 2 Rust engineers

**Estimated Lines of Code:** ~4,000 lines

---

### Week 10: Build Agent (WASM Compilation)

**Objective:** Compile Rust code to WASM

#### Implementation

**1. Build Agent**
```rust
// agents/build/src/agent.rs
use std::process::Command;

pub struct BuildAgent {
    workspace: PathBuf,
}

#[async_trait]
impl Agent for BuildAgent {
    type Input = RustCode;
    type Output = WasmArtifact;

    async fn execute(&self, code: RustCode) -> Result<WasmArtifact> {
        // 1. Write Rust code to file
        let src_path = self.workspace.join("src/lib.rs");
        fs::write(&src_path, code.source)?;

        // 2. Compile to WASM
        let output = Command::new("cargo")
            .args(&["build", "--target", "wasm32-wasi", "--release"])
            .current_dir(&self.workspace)
            .output()?;

        if !output.status.success() {
            return Err(Error::CompilationFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }

        // 3. Read WASM binary
        let wasm_path = self.workspace.join("target/wasm32-wasi/release/lib.wasm");
        let wasm_bytes = fs::read(wasm_path)?;

        Ok(WasmArtifact {
            bytes: wasm_bytes,
            metadata: self.extract_metadata(&output)?,
        })
    }
}
```

**2. Tests**
```rust
// tests/unit/test_build_agent.rs
#[tokio::test]
async fn test_compile_simple_rust() {
    let agent = BuildAgent::new();
    let rust_code = RustCode {
        source: "pub fn add(a: i32, b: i32) -> i32 { a + b }".to_string(),
    };

    let artifact = agent.execute(rust_code).await.unwrap();

    assert!(!artifact.bytes.is_empty());
    assert!(artifact.bytes.starts_with(b"\0asm")); // WASM magic number
}
```

**Deliverables:**
- [ ] Build agent implementation
- [ ] Cargo integration
- [ ] Error handling for compilation failures
- [ ] 10+ unit tests (>80% coverage)

**Success Criteria:**
- ✅ Compile simple Rust code to WASM
- ✅ Handle compilation errors gracefully
- ✅ WASM artifact validated
- ✅ Test coverage >80%

**Team:** 1 Rust engineer

**Estimated Lines of Code:** ~1,500 lines

---

### Week 11: Test Agent & Packaging Agent

**Objective:** Validate translations and create artifacts

#### Implementation

**1. Test Agent**
```rust
// agents/test/src/agent.rs
use wasmtime::*;

pub struct TestAgent {
    engine: Engine,
}

#[async_trait]
impl Agent for TestAgent {
    type Input = (WasmArtifact, TestCases);
    type Output = TestResults;

    async fn execute(&self, (artifact, tests): Self::Input) -> Result<TestResults> {
        let mut results = TestResults::default();

        // Load WASM module
        let module = Module::from_binary(&self.engine, &artifact.bytes)?;
        let mut store = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;

        // Run test cases
        for test in tests {
            let func = instance.get_typed_func::<i32, i32>(&mut store, &test.function)?;
            let result = func.call(&mut store, test.input)?;

            results.add(TestResult {
                name: test.name,
                passed: result == test.expected,
                actual: result,
                expected: test.expected,
            });
        }

        Ok(results)
    }
}
```

**2. Packaging Agent**
```rust
// agents/packaging/src/agent.rs
pub struct PackagingAgent;

#[async_trait]
impl Agent for PackagingAgent {
    type Input = (WasmArtifact, TestResults);
    type Output = Package;

    async fn execute(&self, (artifact, results): Self::Input) -> Result<Package> {
        if results.fail_rate() > 0.1 {
            return Err(Error::TooManyFailures(results.fail_rate()));
        }

        Ok(Package {
            wasm: artifact.bytes,
            metadata: PackageMetadata {
                version: "0.1.0".to_string(),
                test_results: results,
                timestamp: Utc::now(),
            },
        })
    }
}
```

**Deliverables:**
- [ ] Test agent implementation
- [ ] Packaging agent implementation
- [ ] WASM execution via wasmtime
- [ ] 15+ unit tests (>80% coverage)

**Success Criteria:**
- ✅ Execute WASM modules
- ✅ Validate test results
- ✅ Create deployment packages
- ✅ Test coverage >80%

**Team:** 2 Rust engineers

**Estimated Lines of Code:** ~2,000 lines

---

### Phase 1 Gate Review (End of Week 11)

**MVP Demonstration:**
```bash
# End-to-end test
cargo run -- translate fibonacci.py --output fibonacci.wasm

# Verify
wasmtime fibonacci.wasm fibonacci 10
# Expected: 55
```

**Test Scripts (8/10 must pass):**
1. fibonacci.py ✅
2. factorial.py ✅
3. sum_list.py ✅
4. binary_search.py ✅
5. bubble_sort.py ✅
6. string_reverse.py ✅
7. prime_check.py ⚠️
8. palindrome.py ✅
9. gcd.py ⚠️
10. power.py ✅

**Gate Criteria:**
- ✅ 8/10 test scripts translate successfully
- ✅ Generated Rust compiles without errors
- ✅ WASM modules execute correctly
- ✅ Test pass rate >90%
- ✅ End-to-end time <5 minutes per script
- ✅ Test coverage >80% across all agents
- ✅ Demo-able to stakeholders

**Decision:**
- **PASS:** Proceed to Phase 2 (Library Mode)
- **CONDITIONAL PASS:** Fix 2 failing scripts, then proceed
- **FAIL:** Extend Phase 1, address blockers

**Review Meeting:** Leadership + Engineering + Product

---

## PHASE 2: LIBRARY MODE (WEEKS 12-21)

### Week 12-14: Multi-File Support

**Objective:** Handle Python packages with multiple files

#### Implementation

**1. Package Parser**
```rust
// agents/ingest/src/package.rs
pub struct PackageParser {
    parser: PythonParser,
}

impl PackageParser {
    pub fn parse_package(&self, root: &Path) -> Result<PackageAST> {
        let mut package = PackageAST::default();

        // Parse setup.py / pyproject.toml
        let manifest = self.parse_manifest(root)?;
        package.metadata = manifest;

        // Parse all Python files
        for entry in WalkDir::new(root).into_iter() {
            let entry = entry?;
            if entry.path().extension() == Some("py") {
                let module = self.parser.parse_file(entry.path())?;
                package.modules.insert(entry.path().to_path_buf(), module);
            }
        }

        // Resolve imports
        package.dependencies = self.resolve_dependencies(&package)?;

        Ok(package)
    }
}
```

**2. Dependency Resolver**
```rust
// agents/analysis/src/dependencies.rs
pub struct DependencyResolver;

impl DependencyResolver {
    pub fn resolve(&self, package: &PackageAST) -> Result<DependencyGraph> {
        let mut graph = DependencyGraph::default();

        for (path, module) in &package.modules {
            for import in &module.imports {
                if let Some(dep) = self.find_module(&import, package) {
                    graph.add_edge(path, dep);
                }
            }
        }

        // Detect cycles
        if graph.has_cycles() {
            return Err(Error::CircularDependency);
        }

        Ok(graph)
    }
}
```

**3. Workspace Generator**
```rust
// agents/build/src/workspace.rs
pub struct WorkspaceGenerator;

impl WorkspaceGenerator {
    pub fn generate(&self, package: &PackageAST) -> Result<CargoWorkspace> {
        let mut workspace = CargoWorkspace::new(&package.metadata.name);

        // Create crate for each module
        for (path, module) in &package.modules {
            let crate_name = self.module_to_crate_name(path);
            workspace.add_crate(crate_name, self.generate_crate(module)?);
        }

        // Add inter-crate dependencies
        for edge in &package.dependencies.edges {
            workspace.add_dependency(edge.from, edge.to);
        }

        Ok(workspace)
    }
}
```

**Deliverables:**
- [ ] Package parser (setup.py, pyproject.toml)
- [ ] Cross-file dependency resolution
- [ ] Cargo workspace generation
- [ ] 20+ unit tests (>80% coverage)

**Success Criteria:**
- ✅ Parse packages with 5-10 files
- ✅ Resolve cross-file dependencies
- ✅ Generate multi-crate workspace
- ✅ Test coverage >80%

**Team:** 2 Rust engineers, 1 Python engineer

**Estimated Lines of Code:** ~3,000 lines

---

### Week 15-17: Class Translation

**Objective:** Translate Python classes to Rust structs + traits

#### Implementation

**1. Class Analyzer**
```rust
// agents/analysis/src/class_analyzer.rs
pub struct ClassAnalyzer;

impl ClassAnalyzer {
    pub fn analyze_class(&self, class: &Class) -> Result<ClassAnalysis> {
        Ok(ClassAnalysis {
            name: class.name.clone(),
            fields: self.extract_fields(class)?,
            methods: self.analyze_methods(class)?,
            inheritance: self.analyze_inheritance(class)?,
        })
    }

    fn analyze_inheritance(&self, class: &Class) -> Result<InheritanceStrategy> {
        if class.bases.is_empty() {
            return Ok(InheritanceStrategy::None);
        }

        // Map inheritance to Rust traits
        Ok(InheritanceStrategy::Trait {
            trait_name: format!("{}Trait", class.name),
            implementations: class.bases.clone(),
        })
    }
}
```

**2. Class Transpiler**
```rust
// agents/transpiler/src/class_transpiler.rs
pub struct ClassTranspiler;

impl ClassTranspiler {
    pub fn transpile(&self, analysis: &ClassAnalysis) -> Result<RustCode> {
        let mut code = String::new();

        // Generate struct
        code.push_str(&self.generate_struct(analysis)?);

        // Generate trait (if inheritance)
        if let InheritanceStrategy::Trait { trait_name, .. } = &analysis.inheritance {
            code.push_str(&self.generate_trait(trait_name, analysis)?);
        }

        // Generate impl block
        code.push_str(&self.generate_impl(analysis)?);

        Ok(RustCode { source: code })
    }

    fn generate_struct(&self, analysis: &ClassAnalysis) -> Result<String> {
        Ok(format!(
            "#[derive(Debug, Clone)]\npub struct {} {{\n{}\n}}",
            analysis.name,
            analysis.fields.iter()
                .map(|f| format!("    pub {}: {}", f.name, f.rust_type))
                .collect::<Vec<_>>()
                .join(",\n")
        ))
    }
}
```

**Deliverables:**
- [ ] Class analyzer
- [ ] Struct generation
- [ ] Method translation
- [ ] Trait-based inheritance
- [ ] 30+ unit tests (>80% coverage)

**Success Criteria:**
- ✅ Translate simple classes
- ✅ Handle methods (instance and static)
- ✅ Handle properties
- ✅ Map inheritance to traits
- ✅ Test coverage >80%

**Team:** 2 Rust engineers

**Estimated Lines of Code:** ~3,000 lines

---

### Week 18-21: Standard Library Mapping & Integration

**Objective:** Map Python stdlib to Rust equivalents

#### Implementation

**1. Stdlib Mapper**
```rust
// agents/transpiler/src/stdlib_mapper.rs
pub struct StdlibMapper {
    mappings: HashMap<String, StdlibMapping>,
}

impl StdlibMapper {
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // Common mappings
        mappings.insert("len".to_string(), StdlibMapping {
            rust_equivalent: ".len()".to_string(),
            requires_import: None,
        });

        mappings.insert("range".to_string(), StdlibMapping {
            rust_equivalent: "..".to_string(),
            requires_import: None,
        });

        mappings.insert("str.split".to_string(), StdlibMapping {
            rust_equivalent: ".split()".to_string(),
            requires_import: None,
        });

        // ... 50+ mappings

        Self { mappings }
    }
}
```

**2. Integration Testing**
```rust
// tests/e2e/test_library_translation.rs
#[tokio::test]
async fn test_translate_small_library() {
    let pipeline = Pipeline::new();
    let library_path = "tests/fixtures/sample_library/"; // 10K LOC

    let result = pipeline.translate_library(library_path).await.unwrap();

    assert!(result.success);
    assert_eq!(result.crates.len(), 5);
    assert!(result.api_coverage > 0.8);
    assert!(result.test_pass_rate > 0.9);
}
```

**Deliverables:**
- [ ] Standard library mapper (50+ mappings)
- [ ] I/O operations mapping
- [ ] String/collection operations
- [ ] End-to-end library translation
- [ ] 40+ tests (>80% coverage)

**Success Criteria:**
- ✅ Translate 1 real library (>10K LOC)
- ✅ 80%+ API coverage
- ✅ 90%+ test pass rate
- ✅ Compilation success rate >95%
- ✅ Multi-crate workspace generates correctly

**Team:** 3 Rust engineers, 1 Python engineer

**Estimated Lines of Code:** ~3,000 lines

---

### Phase 2 Gate Review (End of Week 21)

**Target Library:** Choose 1 real Python library (10K+ LOC)

**Candidates:**
- `requests` subset (HTTP client)
- `click` subset (CLI framework)
- `pydantic` subset (data validation)

**Gate Criteria:**
- ✅ Library translates successfully
- ✅ Multi-crate workspace generated
- ✅ 80%+ API coverage
- ✅ 90%+ test pass rate
- ✅ Compilation success >95%
- ✅ Core platform production-quality

**Decision:**
- **PASS:** Proceed to Phase 3 (NVIDIA Integration)
- **CONDITIONAL PASS:** Fix specific issues, validate again
- **FAIL:** Extend Phase 2

**Review Meeting:** Leadership + Engineering + Pilot Customers

---

## PHASE 3: NVIDIA ACCELERATION (WEEKS 22-29)

### Week 22-24: NeMo Integration

**Objective:** Connect NeMo LLM to transpiler for enhanced translation

#### Implementation

**1. NeMo Client Integration**
```rust
// agents/transpiler/src/nemo_client.rs
pub struct NeMoTranspilerClient {
    service: NeMoService,
}

impl NeMoTranspilerClient {
    pub async fn enhance_translation(&self, context: &TranslationContext) -> Result<RustCode> {
        let prompt = self.build_prompt(context)?;
        let response = self.service.translate(prompt).await?;

        // Validate generated code
        self.validate_rust_code(&response)?;

        Ok(RustCode { source: response })
    }

    fn build_prompt(&self, context: &TranslationContext) -> Result<String> {
        Ok(format!(
            "Translate the following Python function to idiomatic Rust:\n\n\
            Python:\n{}\n\n\
            Context:\n- Input types: {}\n- Return type: {}\n\
            Rust:",
            context.python_code,
            context.param_types.join(", "),
            context.return_type
        ))
    }
}
```

**2. A/B Testing Framework**
```rust
// agents/transpiler/src/ab_test.rs
pub struct ABTestFramework {
    rule_based: RuleBasedTranspiler,
    nemo_based: NeMoTranspilerClient,
}

impl ABTestFramework {
    pub async fn compare(&self, context: &TranslationContext) -> Result<ABTestResult> {
        // Generate both versions
        let rule_based = self.rule_based.transpile(context)?;
        let nemo_based = self.nemo_based.enhance_translation(context).await?;

        // Compare quality
        let rule_score = self.evaluate_quality(&rule_based)?;
        let nemo_score = self.evaluate_quality(&nemo_based)?;

        Ok(ABTestResult {
            rule_based: (rule_based, rule_score),
            nemo_based: (nemo_based, nemo_score),
            winner: if nemo_score > rule_score { "nemo" } else { "rule" },
        })
    }
}
```

**Deliverables:**
- [ ] NeMo client integration
- [ ] A/B testing framework
- [ ] Confidence scoring
- [ ] 20+ tests (>80% coverage)

**Success Criteria:**
- ✅ NeMo enhances translation quality
- ✅ Confidence scores >90%
- ✅ A/B tests show improvement
- ✅ Fallback to rule-based works

**Team:** 1 Rust engineer, 1 ML engineer

**Estimated Lines of Code:** ~2,000 lines

---

### Week 25-26: CUDA Acceleration

**Objective:** GPU-accelerate analysis and parsing

#### Implementation

**1. CUDA-Accelerated Similarity Search**
```rust
// agents/analysis/src/cuda_similarity.rs
use cuda_acceleration::embeddings::EmbeddingEngine;

pub struct CudaSimilaritySearch {
    engine: EmbeddingEngine,
}

impl CudaSimilaritySearch {
    pub fn find_similar_patterns(&self, func: &Function) -> Result<Vec<Pattern>> {
        // Embed function using CUDA-accelerated model
        let embedding = self.engine.embed(func)?;

        // GPU-accelerated similarity search
        let similar = self.engine.search_similar(&embedding, top_k: 10)?;

        Ok(similar)
    }
}
```

**2. Parallel AST Processing**
```rust
// agents/ingest/src/cuda_parser.rs
use cuda_acceleration::parallel::ParallelProcessor;

pub struct CudaParallelParser {
    processor: ParallelProcessor,
}

impl CudaParallelParser {
    pub fn parse_large_codebase(&self, files: Vec<PathBuf>) -> Result<Vec<PortalisAST>> {
        // Batch files for GPU processing
        let batches = self.processor.batch(files, batch_size: 32)?;

        // Parse in parallel on GPU
        let asts = batches.par_iter()
            .map(|batch| self.parse_batch(batch))
            .collect()?;

        Ok(asts)
    }
}
```

**Deliverables:**
- [ ] CUDA-accelerated similarity search
- [ ] Parallel AST processing
- [ ] Performance benchmarks
- [ ] 15+ tests (>75% coverage)

**Success Criteria:**
- ✅ 10x+ speedup on large files (>10K LOC)
- ✅ GPU utilization >70%
- ✅ Batch processing scales linearly
- ✅ CPU fallback works

**Team:** 1 Rust engineer, 1 CUDA engineer

**Estimated Lines of Code:** ~1,500 lines

---

### Week 27-28: Triton & NIM Integration

**Objective:** Deploy models and create microservices

#### Implementation

**1. Triton Model Deployment**
```bash
# Deploy NeMo translation model to Triton
cd deployment/triton
docker-compose up -d

# Verify
curl http://localhost:8000/v2/health/ready
```

**2. NIM Microservices Integration**
```rust
// nim-microservices/src/routes/translate.rs
#[post("/translate")]
async fn translate(
    req: Json<TranslateRequest>,
    pipeline: Data<Pipeline>,
) -> Result<Json<TranslateResponse>> {
    let result = pipeline.translate(&req.source_code).await?;

    Ok(Json(TranslateResponse {
        rust_code: result.rust_code,
        wasm_artifact: result.wasm_bytes,
        metadata: result.metadata,
    }))
}
```

**Deliverables:**
- [ ] Triton deployment operational
- [ ] NIM microservices integrated
- [ ] Load balancing configured
- [ ] 10+ integration tests

**Success Criteria:**
- ✅ Triton handles 100+ req/sec
- ✅ NIM API responds <100ms (P95)
- ✅ All services healthy
- ✅ Monitoring operational

**Team:** 1 Rust engineer, 1 DevOps engineer

**Estimated Lines of Code:** ~1,000 lines

---

### Week 29: Performance Validation & Benchmarking

**Objective:** Validate all SLA targets

#### Benchmarks

**1. Execute Benchmark Suite**
```bash
# NeMo translation performance
python benchmarks/benchmark_nemo.py

# End-to-end pipeline
python benchmarks/benchmark_e2e.py

# Load testing
locust -f load-tests/locust_scenarios.py --host=http://localhost:8080
```

**2. Validate SLA Metrics**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| NeMo P95 Latency (100 LOC) | <500ms | ??? | ⏳ |
| CUDA Speedup vs CPU | >10x | ??? | ⏳ |
| Triton Max QPS | >100 | ??? | ⏳ |
| NIM API P95 Latency | <100ms | ??? | ⏳ |
| E2E Translation (100 LOC) | <500ms | ??? | ⏳ |

**Deliverables:**
- [ ] All 20 SLA metrics validated
- [ ] Performance report with actual data
- [ ] Optimization recommendations
- [ ] Bottleneck analysis

**Success Criteria:**
- ✅ All SLA targets met or exceeded
- ✅ 10x+ CUDA speedup confirmed
- ✅ Load testing passes (1000 concurrent users)
- ✅ Cost <$0.10/translation

**Team:** 2 engineers (1 performance, 1 DevOps)

---

### Phase 3 Gate Review (End of Week 29)

**Gate Criteria:**
- ✅ All NVIDIA integrations operational
- ✅ All 20 SLA metrics validated
- ✅ 10x+ speedup demonstrated
- ✅ Triton handles 100+ req/sec
- ✅ NIM API <100ms P95
- ✅ Load testing successful
- ✅ Cost targets met

**Decision:**
- **PASS:** Proceed to Phase 4 (Production)
- **CONDITIONAL PASS:** Address performance gaps
- **FAIL:** Extend Phase 3, optimize

**Review Meeting:** Leadership + Engineering + Finance

---

## PHASE 4: PRODUCTION DEPLOYMENT (WEEKS 30+)

### Week 30-32: Production Readiness

**Objective:** Prepare for production deployment

#### Actions

**1. Production Deployment Guide**
- [ ] Kubernetes manifests (production-grade)
- [ ] Scaling strategies (HPA, cluster autoscaling)
- [ ] Security hardening (RBAC, network policies)
- [ ] Disaster recovery procedures
- [ ] Backup and restore runbooks

**2. Security Validation**
```bash
# Security scans
bandit -r . -f json -o security-report.json
safety check --json
cargo audit
trivy scan --severity CRITICAL,HIGH

# Fix all critical vulnerabilities
```

**3. Monitoring & Alerting**
- [ ] Production Grafana dashboards
- [ ] PagerDuty integration
- [ ] Alert rules tuned for production
- [ ] On-call rotation established
- [ ] Incident response runbooks

**Deliverables:**
- [ ] Production deployment guide (1,500+ lines)
- [ ] Security audit complete (zero critical vulns)
- [ ] Monitoring operational
- [ ] Incident response procedures

**Success Criteria:**
- ✅ Production environment deployed
- ✅ Zero critical security vulnerabilities
- ✅ Monitoring and alerting operational
- ✅ Team trained on incident response

**Team:** 2 DevOps engineers, 1 Security engineer

---

### Week 33-34: Customer Pilot

**Objective:** Validate with real customers

#### Pilot Program

**Pilot Customers:** 3-5 early adopters

**Pilot Goals:**
- Validate translation accuracy on real codebases
- Gather feedback on usability
- Identify edge cases and bugs
- Validate SLA compliance in production

**Success Metrics:**
- >90% translation success rate
- >80% customer satisfaction
- <5 critical bugs found
- SLA compliance >95%

**Deliverables:**
- [ ] Pilot customer onboarding
- [ ] Feedback collection and analysis
- [ ] Bug fixes and improvements
- [ ] Pilot success report

**Success Criteria:**
- ✅ 3+ customers using successfully
- ✅ >90% translation success rate
- ✅ >80% satisfaction scores
- ✅ Zero critical bugs in production

**Team:** 3 engineers (support), 1 PM

---

### Week 35-37: GA Preparation & Launch

**Objective:** General Availability launch

#### Final Preparations

**1. Documentation Finalization**
- [ ] API reference (generated)
- [ ] User tutorials (step-by-step)
- [ ] Integration guides
- [ ] FAQ and troubleshooting

**2. Marketing & Launch**
- [ ] Launch blog post
- [ ] Demo videos
- [ ] Benchmark results published
- [ ] Case studies from pilots

**3. Support Infrastructure**
- [ ] Support ticket system
- [ ] Knowledge base
- [ ] Community forum
- [ ] Office hours

**Deliverables:**
- [ ] Complete documentation site
- [ ] Launch materials
- [ ] Support infrastructure
- [ ] GA release (v1.0.0)

**Success Criteria:**
- ✅ GA release deployed
- ✅ Documentation complete
- ✅ Support infrastructure operational
- ✅ SPARC Phase 5 (Completion) ACHIEVED

**Team:** Full team (7 engineers + PM + Support)

---

## SPARC PHASE 5 COMPLETION VALIDATION

### Final Checklist

**Functional Completeness:**
- [ ] All 7 agents implemented and tested
- [ ] Script Mode working (100%)
- [ ] Library Mode working (80%+ coverage)
- [ ] NVIDIA acceleration integrated
- [ ] End-to-end pipeline reliable

**Quality Standards:**
- [ ] 80%+ code coverage (London School TDD)
- [ ] All integration tests passing
- [ ] Performance benchmarks meet targets
- [ ] Security vulnerabilities addressed
- [ ] Zero critical bugs

**Production Readiness:**
- [ ] Error handling comprehensive
- [ ] Logging and monitoring operational
- [ ] Deployment automation working
- [ ] Rollback procedures tested
- [ ] Documentation complete (actual results)

**Customer Validation:**
- [ ] 3+ pilot customers successful
- [ ] >90% translation success rate
- [ ] >80% customer satisfaction
- [ ] SLA compliance >95%
- [ ] Production deployment stable

**SPARC Methodology:**
- [ ] Phase 1 (Specification) - Complete
- [ ] Phase 2 (Pseudocode) - Complete
- [ ] Phase 3 (Architecture) - Complete
- [ ] Phase 4 (Refinement) - Complete
- [ ] Phase 5 (Completion) - **ACHIEVED**

---

## RISK MANAGEMENT

### Continuous Risk Monitoring

**Weekly Risk Review:**
- Identify new risks
- Update probability and impact
- Review mitigation effectiveness
- Escalate critical risks

**Monthly Risk Report:**
- Risk register update
- Trend analysis
- Mitigation plan adjustments
- Stakeholder communication

### Contingency Plans

**If Timeline Slips (>2 weeks):**
1. Descope non-critical features
2. Add contractors for specialized skills
3. Extend specific phase, compress others
4. Communicate revised timeline to stakeholders

**If Budget Exceeds (+20%):**
1. Review ROI and business case
2. Seek additional funding
3. Phase deployment (GA with basic features)
4. Defer enterprise features to v2.0

**If Technical Blockers:**
1. Spike to prototype solution (3-5 days)
2. Bring in external experts
3. Consider alternative approaches
4. Escalate to leadership if unresolvable

---

## COMMUNICATION PLAN

### Weekly Updates

**Audience:** Engineering Team
**Format:** Standup + Weekly Demo
**Content:** Progress, blockers, next week's goals

### Monthly Reviews

**Audience:** Leadership
**Format:** Gate Review Meeting
**Content:** Phase completion, metrics, risks, budget

### Quarterly Business Reviews

**Audience:** Executives
**Format:** QBR Presentation
**Content:** Strategic progress, ROI, market fit

---

## SUCCESS METRICS TRACKING

### Weekly KPIs

- Lines of code written
- Test coverage %
- Bug count (open/closed)
- Velocity (story points)

### Monthly KPIs

- Phase completion %
- Budget utilization
- SLA compliance
- Team morale

### Quarterly KPIs

- Customer adoption
- Translation success rate
- Revenue/ARR
- NPS score

---

## CONCLUSION

This action plan provides a **clear, executable roadmap** to complete SPARC Phase 5 (Completion) for the Portalis project. The plan is:

✅ **Realistic:** 37 weeks with 3-7 engineers
✅ **Validated:** Based on comprehensive swarm analysis
✅ **Gated:** Strict success criteria at each phase
✅ **Risk-Aware:** Contingency plans for common scenarios
✅ **London School TDD:** >80% coverage throughout

**Next Steps:**
1. **This Week:** Secure stakeholder approval
2. **Week 0:** Execute proof-of-concept
3. **Weeks 1-3:** Phase 0 foundation sprint
4. **Weeks 4-11:** Phase 1 MVP implementation

**Critical Success Factors:**
- Start implementation IMMEDIATELY (after POC)
- Maintain TDD discipline (>80% coverage)
- Weekly demos and monthly gates
- No scope creep - defer to later phases

**Expected Outcome:**
- **Week 37:** Production-ready Portalis platform
- **SPARC Phase 5:** COMPLETE
- **Business Value:** Python → Rust → WASM translation at scale

---

**Status:** ✅ ACTION PLAN READY
**Approval Required:** Stakeholder sign-off
**Next Review:** After Phase 0 (Week 3)

---

*Plan prepared by Claude Flow Swarm*
*Ready for execution upon stakeholder approval*
