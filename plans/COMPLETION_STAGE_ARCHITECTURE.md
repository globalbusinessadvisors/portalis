# PORTALIS - Completion Stage Architecture Design
## SPARC Phase 5: London TDD Implementation Framework

**Author:** System Architect
**Date:** 2025-10-03
**Status:** ACTIVE IMPLEMENTATION
**Version:** 1.0

---

## Executive Summary

This document defines the complete architecture for the Completion stage of the Portalis SPARC London TDD framework build. It integrates the existing Rust core (18 files) with the comprehensive NVIDIA acceleration stack (21,000+ LOC) to deliver a production-ready Python-to-Rust-to-WASM translation platform.

### Current State Assessment

**Existing Foundation (COMPLETE):**
- ✅ Rust core agent framework (portalis-core)
- ✅ 7 agent implementations (Ingest, Analysis, SpecGen, Transpiler, Build, Test, Packaging)
- ✅ Pipeline orchestration layer
- ✅ Message bus for agent communication
- ✅ NVIDIA integration stack (NeMo, CUDA, Triton, NIM, DGX Cloud, Omniverse)
- ✅ CI/CD pipeline with comprehensive test infrastructure

**Missing Components (TO IMPLEMENT):**
- ❌ Integration glue between Rust agents and NVIDIA Python services
- ❌ Comprehensive test coverage for core agents (London TDD)
- ❌ End-to-end validation workflows
- ❌ Production deployment configurations
- ❌ Monitoring and observability for core platform

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Component Design](#2-core-component-design)
3. [NVIDIA Integration Patterns](#3-nvidia-integration-patterns)
4. [Testing Architecture (London TDD)](#4-testing-architecture-london-tdd)
5. [Deployment & CI/CD Pipeline](#5-deployment--cicd-pipeline)
6. [Monitoring & Observability](#6-monitoring--observability)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. Architecture Overview

### 1.1 System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                           │
│  CLI (Rust) │ REST API (Python/NIM) │ gRPC API (Python/NIM)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Orchestration Layer (Rust)                     │
│  Pipeline Manager │ State Machine │ Agent Coordinator           │
│  Message Bus │ Error Recovery │ Progress Tracking               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Layer (Rust)                           │
│  Ingest → Analysis → SpecGen → Transpiler → Build → Test →     │
│                           Package                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Integration Bridge (Rust ↔ Python)                 │
│  FFI Bindings │ gRPC Clients │ REST Clients │ Serialization    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             NVIDIA Acceleration Layer (Python)                  │
│  NeMo │ CUDA │ Triton │ NIM │ DGX Cloud │ Omniverse           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow Architecture

```
INPUT: Python Source Code
         ↓
    ┌────────────┐
    │   Ingest   │ ──→ Validates, parses → PythonAST
    └────────────┘
         ↓
    ┌────────────┐
    │  Analysis  │ ──→ Type inference → TypedFunctions + APIContract
    └────────────┘      │
         ↓              └──→ [NVIDIA: NeMo Embeddings for API clustering]
    ┌────────────┐
    │  SpecGen   │ ──→ Rust spec generation → RustSpec
    └────────────┘      │
         ↓              └──→ [NVIDIA: NeMo LLM for trait generation]
    ┌────────────┐
    │ Transpiler │ ──→ Code translation → RustCode
    └────────────┘      │
         ↓              └──→ [NVIDIA: NeMo LLM + CUDA ranking]
    ┌────────────┐
    │   Build    │ ──→ Compilation → WasmBinary
    └────────────┘
         ↓
    ┌────────────┐
    │    Test    │ ──→ Validation → TestResults
    └────────────┘      │
         ↓              └──→ [NVIDIA: CUDA parallel test execution]
    ┌────────────┐
    │  Package   │ ──→ Packaging → NIMContainer
    └────────────┘      │
                        └──→ [NVIDIA: Triton deployment]
         ↓
OUTPUT: WASM Module + NIM Container + Test Report
```

---

## 2. Core Component Design

### 2.1 Rust Agent Framework (portalis-core)

**Status:** ✅ IMPLEMENTED

**Location:** `/workspace/portalis/core/src/`

**Components:**
- `agent.rs` - Agent trait and base abstractions
- `message.rs` - Message bus for inter-agent communication
- `types.rs` - Pipeline data types (Artifact, Phase, PipelineState)
- `error.rs` - Error handling framework

**Design Patterns:**
- **London School TDD:** Agents communicate via message passing
- **Dependency Injection:** All dependencies injectable for testing
- **Async/Await:** Full async support with tokio runtime
- **Strong Typing:** Type-safe artifact passing between agents

**Key Interfaces:**

```rust
#[async_trait]
pub trait Agent: Send + Sync {
    type Input: Send + Sync;
    type Output: Send + Sync;

    async fn execute(&self, input: Self::Input) -> Result<Self::Output>;
    fn id(&self) -> AgentId;
    fn name(&self) -> &str;
    fn capabilities(&self) -> Vec<AgentCapability>;
    fn validate_input(&self, input: &Self::Input) -> Result<()>;
}
```

### 2.2 Agent Implementations

#### 2.2.1 Ingest Agent

**Status:** ✅ IMPLEMENTED (Basic)
**Location:** `/workspace/portalis/agents/ingest/src/lib.rs`
**Completion:** 70% (Needs production Python parser)

**Current Implementation:**
- Regex-based Python parsing (POC only)
- Function and import extraction
- Parameter type hint parsing

**Required Enhancements:**
- Replace with `rustpython-parser` or `tree-sitter-python`
- Add class and decorator support
- Implement dependency graph extraction
- Add validation for Python syntax errors

**Integration Points:**
- None (pure Rust, no NVIDIA dependencies)

**Test Coverage Target:** 90%
- Unit tests: 20+ (parsing edge cases)
- Integration tests: 5+ (various Python files)

#### 2.2.2 Analysis Agent

**Status:** ✅ IMPLEMENTED (Basic)
**Location:** `/workspace/portalis/agents/analysis/src/lib.rs`
**Completion:** 60% (Needs advanced type inference)

**Current Implementation:**
- Python type hint to Rust type mapping
- API contract extraction
- Confidence scoring

**Required Enhancements:**
- Add type flow analysis for untyped code
- Implement control flow graph generation
- Add data dependency tracking
- Integrate with NeMo for semantic analysis

**Integration Points:**
- **NVIDIA NeMo:** Code embeddings for API clustering
- **CUDA:** Parallel AST traversal for large files

**Test Coverage Target:** 85%
- Unit tests: 25+ (type inference scenarios)
- Integration tests: 8+ (with NeMo mocks)

#### 2.2.3 SpecGen Agent

**Status:** ⚠️ STUB IMPLEMENTATION
**Location:** `/workspace/portalis/agents/specgen/src/lib.rs`
**Completion:** 20% (Needs full implementation)

**Required Implementation:**
- Rust trait generation from Python classes
- ABI design for WASM/WASI interfaces
- Error type mapping (Python exceptions → Rust Result)
- Memory model specification

**Integration Points:**
- **NVIDIA NeMo:** LLM-guided trait generation
- **Template Engine:** Rust code templates

**Test Coverage Target:** 80%
- Unit tests: 20+ (spec generation patterns)
- Integration tests: 6+ (with NeMo)
- Golden tests: 10+ (reference specs)

#### 2.2.4 Transpiler Agent

**Status:** ✅ IMPLEMENTED (Basic)
**Location:** `/workspace/portalis/agents/transpiler/src/lib.rs`
**Completion:** 50% (Needs advanced translation)

**Current Implementation:**
- Template-based code generation
- Simple function translation (add, fibonacci)
- Parameter and return type formatting

**Required Enhancements:**
- Full Python expression translation
- Control flow translation (if/else, loops)
- Error handling (try/except → Result)
- Iterator and comprehension support
- Class to struct/trait translation

**Integration Points:**
- **NVIDIA NeMo:** LLM-based translation
- **CUDA:** Translation candidate ranking
- **Fallback:** Template-based generation

**Test Coverage Target:** 90%
- Unit tests: 40+ (translation patterns)
- Integration tests: 15+ (with NeMo)
- Property tests: 10+ (equivalence checking)
- Golden tests: 20+ (reference translations)

#### 2.2.5 Build Agent

**Status:** ✅ IMPLEMENTED (Basic)
**Location:** `/workspace/portalis/agents/build/src/lib.rs`
**Completion:** 70% (Needs error handling)

**Current Implementation:**
- Cargo project scaffolding
- WASM target compilation
- Basic error capture

**Required Enhancements:**
- Better error message parsing
- Incremental compilation support
- Dependency resolution
- Optimization level configuration

**Integration Points:**
- **Rust Toolchain:** cargo build --target wasm32-wasi
- **File System:** Temporary project directories

**Test Coverage Target:** 85%
- Unit tests: 15+ (project setup)
- Integration tests: 10+ (compilation scenarios)
- E2E tests: 5+ (full builds)

#### 2.2.6 Test Agent

**Status:** ⚠️ STUB IMPLEMENTATION
**Location:** `/workspace/portalis/agents/test/src/lib.rs`
**Completion:** 30% (Needs test harness)

**Required Implementation:**
- WASM module loading and execution
- Test case generation from Python tests
- Golden output validation
- Performance benchmarking
- Property-based testing

**Integration Points:**
- **CUDA:** Parallel test execution
- **WASM Runtime:** wasmtime or wasmer
- **Python Interpreter:** For golden outputs

**Test Coverage Target:** 80%
- Unit tests: 18+ (test execution)
- Integration tests: 12+ (various test types)

#### 2.2.7 Packaging Agent

**Status:** ⚠️ STUB IMPLEMENTATION
**Location:** `/workspace/portalis/agents/packaging/src/lib.rs`
**Completion:** 25% (Needs full implementation)

**Required Implementation:**
- NIM container image creation
- Triton model repository structure
- Manifest generation (config.pbtxt)
- Artifact bundling

**Integration Points:**
- **NIM Services:** Container building
- **Triton:** Model deployment
- **Docker:** Container runtime

**Test Coverage Target:** 75%
- Unit tests: 12+ (packaging logic)
- Integration tests: 8+ (with NIM/Triton)

### 2.3 Orchestration Layer

**Status:** ✅ IMPLEMENTED (Basic)
**Location:** `/workspace/portalis/orchestration/src/lib.rs`
**Completion:** 80% (Needs error recovery)

**Current Implementation:**
- Sequential pipeline execution
- Phase transition tracking
- Artifact passing between agents

**Required Enhancements:**
- Parallel agent execution where possible
- Error recovery and retry logic
- Pipeline checkpointing
- Progress callbacks for UI
- Resource management (memory, GPU)

**Architecture:**

```rust
pub struct Pipeline {
    ingest: IngestAgent,
    analysis: AnalysisAgent,
    transpiler: TranspilerAgent,
    build: BuildAgent,
    test: TestAgent,
    packaging: PackagingAgent,
    state: PipelineState,
}

impl Pipeline {
    pub async fn translate(
        &mut self,
        source_path: PathBuf,
        source_code: String
    ) -> Result<PipelineOutput>;
}
```

**Enhancement: Async Parallel Execution**

```rust
// Future enhancement: Execute independent agents in parallel
async fn execute_parallel(&mut self) -> Result<()> {
    // Analysis can run while Ingest completes
    let (ingest_result, analysis_prep) = tokio::join!(
        self.ingest.execute(input),
        self.prepare_analysis()
    );

    // Test generation can run parallel to compilation
    let (build_result, test_prep) = tokio::join!(
        self.build.execute(build_input),
        self.generate_tests(spec)
    );
}
```

---

## 3. NVIDIA Integration Patterns

### 3.1 Integration Architecture

**Challenge:** Bridge Rust core platform with Python-based NVIDIA services

**Solutions:**

#### 3.1.1 gRPC Integration (PREFERRED)

**Status:** ✅ INFRASTRUCTURE EXISTS (NIM microservices)
**Location:** `/workspace/portalis/nim-microservices/grpc/`

**Pattern:**
```
Rust Agent → gRPC Client → NIM gRPC Server → NeMo/CUDA
```

**Implementation:**

```rust
// In agents that need NVIDIA services
use tonic::transport::Channel;
use portalis_nim_grpc::translation_service_client::TranslationServiceClient;

pub struct TranspilerAgent {
    id: AgentId,
    nemo_client: Option<TranslationServiceClient<Channel>>,
}

impl TranspilerAgent {
    async fn translate_with_nemo(&self, code: &str) -> Result<String> {
        if let Some(client) = &self.nemo_client {
            let request = TranslationRequest {
                source_code: code.to_string(),
                target_language: "rust".to_string(),
            };

            let response = client.translate(request).await?;
            Ok(response.into_inner().translated_code)
        } else {
            // Fallback to template-based
            self.template_translate(code)
        }
    }
}
```

**Required Work:**
- Generate Rust gRPC stubs from Python proto definitions
- Add connection pooling for gRPC clients
- Implement retry and timeout logic
- Add circuit breaker for service failures

#### 3.1.2 REST Integration (FALLBACK)

**Status:** ✅ INFRASTRUCTURE EXISTS (NIM REST API)
**Location:** `/workspace/portalis/nim-microservices/api/`

**Pattern:**
```
Rust Agent → reqwest HTTP Client → NIM REST API → NeMo/CUDA
```

**Implementation:**

```rust
use reqwest::Client;

pub struct NeMoClient {
    client: Client,
    base_url: String,
}

impl NeMoClient {
    pub async fn translate(&self, code: &str) -> Result<String> {
        let response = self.client
            .post(&format!("{}/api/v1/translate", self.base_url))
            .json(&serde_json::json!({
                "source_code": code,
                "target_language": "rust"
            }))
            .send()
            .await?;

        let result: TranslationResponse = response.json().await?;
        Ok(result.rust_code)
    }
}
```

#### 3.1.3 FFI Integration (FUTURE)

**Status:** 🔮 FUTURE ENHANCEMENT

**Pattern:** Direct FFI calls to CUDA kernels from Rust

**Use Case:** Ultra-low-latency parallel operations

### 3.2 NVIDIA Service Integration Points

#### 3.2.1 NeMo Integration

**Service:** Translation and Embedding Generation
**Location:** `/workspace/portalis/nemo-integration/`
**Status:** ✅ COMPLETE

**Usage in Agents:**

| Agent | NeMo Use Case | Priority |
|-------|---------------|----------|
| Analysis | Code embeddings for API clustering | MEDIUM |
| SpecGen | LLM-guided trait generation | HIGH |
| Transpiler | Python→Rust translation | CRITICAL |
| Test | Test case generation | LOW |

**Configuration:**

```toml
# config/nemo.toml
[nemo]
model_path = "/models/nemo/codegen-7b"
batch_size = 32
max_length = 512
temperature = 0.2
enable_gpu = true
gpu_id = 0

[fallback]
enable = true
template_dir = "/templates/rust"
```

#### 3.2.2 CUDA Acceleration

**Service:** Parallel Processing Kernels
**Location:** `/workspace/portalis/cuda-acceleration/`
**Status:** ✅ COMPLETE

**Usage in Agents:**

| Agent | CUDA Use Case | Priority |
|-------|---------------|----------|
| Analysis | Parallel AST traversal | MEDIUM |
| Transpiler | Translation candidate ranking | HIGH |
| Test | Parallel test execution | MEDIUM |

**Integration Pattern:**

```rust
// CUDA is called via Python service (no direct Rust-CUDA FFI for MVP)
// Analysis Agent calls CUDA through NeMo service
async fn analyze_with_cuda(&self, ast: &PythonAst) -> Result<Analysis> {
    let request = CudaAnalysisRequest {
        ast_nodes: serialize_ast(ast),
        parallel: true,
    };

    let response = self.cuda_client.analyze(request).await?;
    deserialize_analysis(response)
}
```

#### 3.2.3 Triton Deployment

**Service:** Model Serving Infrastructure
**Location:** `/workspace/portalis/deployment/triton/`
**Status:** ✅ COMPLETE

**Usage:** Packaging Agent deploys WASM modules to Triton

**Integration:**

```rust
pub struct TritonDeployer {
    base_url: String,
    model_repository: PathBuf,
}

impl TritonDeployer {
    pub async fn deploy(&self, wasm: &[u8], manifest: &Manifest) -> Result<()> {
        // 1. Create model directory structure
        let model_dir = self.model_repository.join(&manifest.name);
        std::fs::create_dir_all(&model_dir)?;

        // 2. Write WASM binary
        let version_dir = model_dir.join("1");
        std::fs::create_dir_all(&version_dir)?;
        std::fs::write(version_dir.join("model.wasm"), wasm)?;

        // 3. Generate config.pbtxt
        let config = self.generate_triton_config(manifest)?;
        std::fs::write(model_dir.join("config.pbtxt"), config)?;

        // 4. Notify Triton of new model
        self.reload_model_repository().await?;

        Ok(())
    }
}
```

#### 3.2.4 NIM Container Building

**Service:** Microservice Packaging
**Location:** `/workspace/portalis/nim-microservices/`
**Status:** ✅ COMPLETE

**Usage:** Packaging Agent creates deployable containers

**Integration:**

```rust
pub struct NIMPackager {
    docker_client: Docker,
    registry: String,
}

impl NIMPackager {
    pub async fn package(&self, wasm: &[u8], manifest: &Manifest) -> Result<String> {
        // 1. Create Dockerfile
        let dockerfile = self.generate_dockerfile(manifest)?;

        // 2. Build image
        let image_tag = format!("{}/{}:{}", self.registry, manifest.name, manifest.version);
        let build_result = self.docker_client.build_image(
            BuildOptions {
                dockerfile: dockerfile.into(),
                t: image_tag.clone(),
                ..Default::default()
            }
        ).await?;

        // 3. Push to registry
        self.docker_client.push_image(&image_tag).await?;

        Ok(image_tag)
    }
}
```

#### 3.2.5 DGX Cloud Integration

**Service:** Distributed Workload Orchestration
**Location:** `/workspace/portalis/dgx-cloud/`
**Status:** ✅ COMPLETE (Library Mode)

**Usage:** Future enhancement for large library translation

**Pattern:** Submit jobs to DGX Cloud for parallel translation

#### 3.2.6 Omniverse Integration

**Service:** WASM Runtime for Simulation
**Location:** `/workspace/portalis/omniverse-integration/`
**Status:** ✅ COMPLETE

**Usage:** Deploy WASM modules to Omniverse for testing

---

## 4. Testing Architecture (London TDD)

### 4.1 London School TDD Principles

**Core Tenets:**
1. **Outside-In Development:** Start with acceptance tests
2. **Interaction Testing:** Test how objects collaborate
3. **Mock Collaborators:** Test in isolation with mocks
4. **Emergent Design:** Design emerges from tests

### 4.2 Test Pyramid

```
         ╱╲
        ╱  ╲
       ╱ E2E╲          5% - Full stack (10 tests)
      ╱──────╲
     ╱        ╲
    ╱Integration╲     25% - Agent interactions (50 tests)
   ╱────────────╲
  ╱              ╲
 ╱  Unit Tests   ╲   70% - Individual agents (140 tests)
╱────────────────╲
```

**Total Target: 200+ tests**

### 4.3 Test Structure

#### 4.3.1 Acceptance Tests (E2E)

**Location:** `/workspace/portalis/tests/acceptance/`
**Framework:** Rust integration tests

**Test Scenarios:**

```rust
#[tokio::test]
async fn test_fibonacci_script_translation() {
    // Given: A Python fibonacci script
    let python_code = include_str!("fixtures/fibonacci.py");

    // When: Full pipeline execution
    let mut pipeline = Pipeline::new();
    let result = pipeline.translate(
        PathBuf::from("fibonacci.py"),
        python_code.to_string()
    ).await.unwrap();

    // Then: WASM module is valid and tests pass
    assert!(!result.rust_code.is_empty());
    assert!(result.wasm_bytes.len() > 0);
    assert_eq!(result.test_failed, 0);

    // And: WASM execution produces correct results
    let wasm_result = execute_wasm(&result.wasm_bytes, "fibonacci", &[10]).await;
    assert_eq!(wasm_result, 55);
}

#[tokio::test]
async fn test_class_translation_with_methods() {
    // Test class to struct/trait translation
    let python_code = include_str!("fixtures/calculator.py");
    let result = translate_and_validate(python_code).await.unwrap();

    assert!(result.rust_code.contains("pub struct Calculator"));
    assert!(result.rust_code.contains("impl Calculator"));
}

#[tokio::test]
async fn test_error_handling_translation() {
    // Test Python exceptions to Rust Result
    let python_code = include_str!("fixtures/divide.py");
    let result = translate_and_validate(python_code).await.unwrap();

    assert!(result.rust_code.contains("Result<"));
    assert!(result.rust_code.contains("ZeroDivisionError"));
}
```

**Test Fixtures:**
- `fibonacci.py` - Simple recursive function
- `calculator.py` - Class with methods
- `divide.py` - Error handling
- `data_processor.py` - List comprehensions
- `async_example.py` - Async/await (future)

#### 4.3.2 Integration Tests

**Location:** `/workspace/portalis/tests/integration/`

**Test Categories:**

**A. Agent-to-Agent Communication**

```rust
#[tokio::test]
async fn test_ingest_to_analysis_flow() {
    // Given: Ingest agent output
    let ingest = IngestAgent::new();
    let analysis = AnalysisAgent::new();

    let ingest_result = ingest.execute(IngestInput {
        source_path: PathBuf::from("test.py"),
        source_code: "def add(a: int, b: int) -> int: return a + b".to_string(),
    }).await.unwrap();

    // When: Passing to Analysis agent
    let ast_json = serde_json::to_value(&ingest_result.ast).unwrap();
    let analysis_result = analysis.execute(AnalysisInput { ast: ast_json }).await.unwrap();

    // Then: Type information is extracted
    assert_eq!(analysis_result.typed_functions.len(), 1);
    assert_eq!(analysis_result.typed_functions[0].params.len(), 2);
}
```

**B. NVIDIA Service Integration (with Mocks)**

```rust
#[tokio::test]
async fn test_transpiler_with_nemo_mock() {
    // Given: Mock NeMo service
    let mock_server = MockNeMoServer::start().await;
    mock_server.expect_translate()
        .with_input("def add(a, b): return a + b")
        .returns("pub fn add(a: i32, b: i32) -> i32 { a + b }");

    // When: Transpiler calls NeMo
    let transpiler = TranspilerAgent::with_nemo_url(mock_server.url());
    let result = transpiler.translate_function(...).await.unwrap();

    // Then: Translation matches mock response
    assert!(result.contains("pub fn add"));

    // And: Mock was called correctly
    mock_server.verify();
}
```

**C. Pipeline State Machine**

```rust
#[tokio::test]
async fn test_pipeline_phase_transitions() {
    let mut pipeline = Pipeline::new();

    assert_eq!(pipeline.state().phase, Phase::Idle);

    pipeline.start_ingest();
    assert_eq!(pipeline.state().phase, Phase::Ingesting);

    // ... test all phase transitions
}
```

#### 4.3.3 Unit Tests

**Location:** Embedded in each agent module (`#[cfg(test)]`)

**Coverage Requirements:**

| Agent | Unit Tests | Coverage |
|-------|-----------|----------|
| Core (agent.rs) | 15+ | 95% |
| Core (message.rs) | 12+ | 95% |
| Core (types.rs) | 10+ | 90% |
| Ingest | 20+ | 90% |
| Analysis | 25+ | 85% |
| SpecGen | 20+ | 80% |
| Transpiler | 40+ | 90% |
| Build | 15+ | 85% |
| Test | 18+ | 80% |
| Packaging | 12+ | 75% |
| Orchestration | 13+ | 85% |

**Example: Ingest Agent Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_function_with_type_hints() { ... }

    #[tokio::test]
    async fn test_parse_function_without_type_hints() { ... }

    #[tokio::test]
    async fn test_parse_class_with_methods() { ... }

    #[tokio::test]
    async fn test_parse_imports() { ... }

    #[tokio::test]
    async fn test_parse_decorators() { ... }

    #[tokio::test]
    async fn test_invalid_syntax_error() { ... }

    #[tokio::test]
    async fn test_empty_file() { ... }

    // ... 13+ more tests
}
```

### 4.4 Mock Infrastructure

**Pattern:** London School mocks for all external dependencies

#### 4.4.1 NeMo Service Mock

**Location:** `/workspace/portalis/tests/mocks/nemo_mock.rs`

```rust
pub struct MockNeMoService {
    expectations: Vec<Expectation>,
    call_count: Arc<Mutex<usize>>,
}

impl MockNeMoService {
    pub fn new() -> Self { ... }

    pub fn expect_translate(&mut self) -> &mut Expectation {
        let exp = Expectation::new();
        self.expectations.push(exp);
        self.expectations.last_mut().unwrap()
    }

    pub async fn translate(&self, code: &str) -> String {
        // Match against expectations
        for exp in &self.expectations {
            if exp.matches(code) {
                return exp.response();
            }
        }
        panic!("Unexpected call to translate");
    }

    pub fn verify(&self) {
        assert!(self.expectations.iter().all(|e| e.was_called()));
    }
}
```

#### 4.4.2 File System Mock

```rust
pub struct MockFileSystem {
    files: HashMap<PathBuf, String>,
}

impl FileSystem for MockFileSystem {
    fn read(&self, path: &Path) -> Result<String> {
        self.files.get(path)
            .cloned()
            .ok_or_else(|| Error::FileNotFound)
    }

    fn write(&mut self, path: &Path, content: &str) -> Result<()> {
        self.files.insert(path.to_path_buf(), content.to_string());
        Ok(())
    }
}
```

#### 4.4.3 Cargo/Compiler Mock

```rust
pub struct MockCompiler {
    should_succeed: bool,
    wasm_output: Vec<u8>,
}

impl Compiler for MockCompiler {
    async fn compile(&self, code: &str) -> Result<Vec<u8>> {
        if self.should_succeed {
            Ok(self.wasm_output.clone())
        } else {
            Err(Error::Compilation("Mock compilation failed".into()))
        }
    }
}
```

### 4.5 Property-Based Testing

**Framework:** proptest

**Use Case:** Verify translation correctness properties

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_addition_translation_equivalence(a: i32, b: i32) {
        // Property: Python add(a, b) == Rust add(a, b)
        let python_result = run_python_add(a, b);
        let rust_result = run_translated_add(a, b);

        prop_assert_eq!(python_result, rust_result);
    }

    #[test]
    fn test_list_operations_preserve_length(items: Vec<i32>) {
        // Property: List length is preserved through translation
        let python_len = run_python_len(&items);
        let rust_len = run_translated_len(&items);

        prop_assert_eq!(python_len, rust_len);
    }
}
```

### 4.6 Golden Tests

**Pattern:** Reference outputs for regression testing

**Location:** `/workspace/portalis/tests/golden/`

```
golden/
├── inputs/
│   ├── fibonacci.py
│   ├── calculator.py
│   └── data_processor.py
└── expected/
    ├── fibonacci.rs
    ├── calculator.rs
    └── data_processor.rs
```

```rust
#[test]
fn test_golden_fibonacci() {
    let input = include_str!("../golden/inputs/fibonacci.py");
    let expected = include_str!("../golden/expected/fibonacci.rs");

    let result = translate(input).unwrap();

    // Normalize whitespace and comments
    let normalized_result = normalize_rust_code(&result);
    let normalized_expected = normalize_rust_code(expected);

    assert_eq!(normalized_result, normalized_expected);
}
```

---

## 5. Deployment & CI/CD Pipeline

### 5.1 CI/CD Architecture

**Platform:** GitHub Actions
**Configuration:** `/workspace/portalis/.github/workflows/`

**Status:** ✅ PARTIAL (NVIDIA stack only)

### 5.2 CI Pipeline Stages

```yaml
# .github/workflows/ci.yml
name: Portalis CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  # Stage 1: Rust Core Tests
  rust-core-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          components: clippy, rustfmt

      - name: Run clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Run rustfmt
        run: cargo fmt -- --check

      - name: Run unit tests
        run: cargo test --all-features --lib

      - name: Run doc tests
        run: cargo test --doc

  # Stage 2: Integration Tests (Rust + NVIDIA)
  integration-tests:
    runs-on: ubuntu-latest
    needs: rust-core-tests
    services:
      nemo-mock:
        image: mockserver/mockserver
        ports:
          - 1080:1080

    steps:
      - name: Run integration tests
        run: cargo test --test integration -- --test-threads=1
        env:
          NEMO_MOCK_URL: http://localhost:1080
          MOCK_MODE: true

  # Stage 3: E2E Tests
  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests

    steps:
      - name: Setup WASM target
        run: rustup target add wasm32-wasi

      - name: Run E2E tests
        run: cargo test --test e2e

  # Stage 4: NVIDIA Stack Tests (existing)
  nvidia-stack-tests:
    runs-on: ubuntu-latest
    needs: rust-core-tests
    steps:
      - name: Run NVIDIA tests
        run: pytest tests/ -m "unit or integration" --cov

  # Stage 5: Security Scan
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Run cargo audit
        run: cargo audit

      - name: Run Python security scan
        run: |
          pip install bandit safety
          bandit -r nemo-integration/ nim-microservices/
          safety check

  # Stage 6: Build Docker Images
  build-images:
    runs-on: ubuntu-latest
    needs: [e2e-tests, nvidia-stack-tests]
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Build NIM image
        run: docker build -t portalis/nim:latest nim-microservices/

      - name: Build Triton image
        run: docker build -t portalis/triton:latest deployment/triton/

  # Stage 7: Deploy (Production)
  deploy:
    runs-on: ubuntu-latest
    needs: build-images
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - name: Deploy to DGX Cloud
        run: kubectl apply -f deployment/k8s/
```

### 5.3 Deployment Configurations

#### 5.3.1 Docker Compose (Development)

**Location:** `/workspace/portalis/docker-compose.yaml`

```yaml
version: '3.8'

services:
  # Core Portalis service (Rust CLI + orchestration)
  portalis-core:
    build:
      context: .
      dockerfile: Dockerfile.core
    volumes:
      - ./workspace:/workspace
    depends_on:
      - nemo-service
      - nim-api

  # NeMo translation service
  nemo-service:
    build:
      context: nemo-integration
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/models/nemo
    volumes:
      - nemo-models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # NIM API gateway
  nim-api:
    build:
      context: nim-microservices
    ports:
      - "8080:8080"
      - "50051:50051"
    depends_on:
      - nemo-service

  # Triton inference server
  triton:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    ports:
      - "8000:8000"
      - "8001:8001"
    volumes:
      - ./models:/models
    command: tritonserver --model-repository=/models

  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  nemo-models:
```

#### 5.3.2 Kubernetes (Production)

**Location:** `/workspace/portalis/deployment/k8s/`

**Manifests:**
- `namespace.yaml` - Portalis namespace
- `configmap.yaml` - Configuration
- `secret.yaml` - Credentials
- `deployment-core.yaml` - Core Rust service
- `deployment-nemo.yaml` - NeMo service
- `deployment-nim.yaml` - NIM API
- `deployment-triton.yaml` - Triton server
- `service.yaml` - Service definitions
- `ingress.yaml` - External access
- `hpa.yaml` - Auto-scaling

**Example: Core Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: portalis-core
  namespace: portalis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: portalis-core
  template:
    metadata:
      labels:
        app: portalis-core
    spec:
      containers:
      - name: core
        image: portalis/core:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: NEMO_SERVICE_URL
          value: "http://nemo-service:8080"
        - name: NIM_API_URL
          value: "http://nim-api:8080"
        - name: TRITON_URL
          value: "http://triton:8000"
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## 6. Monitoring & Observability

### 6.1 Metrics Architecture

**Stack:** Prometheus + Grafana + OpenTelemetry

**Status:** ✅ INFRASTRUCTURE EXISTS (NVIDIA only)

### 6.2 Core Platform Metrics

**Required Addition:** Rust agent metrics

#### 6.2.1 Agent Metrics

```rust
use prometheus::{Counter, Histogram, Registry};

pub struct AgentMetrics {
    executions_total: Counter,
    execution_duration: Histogram,
    errors_total: Counter,
    artifacts_produced: Counter,
}

impl AgentMetrics {
    pub fn new(agent_name: &str, registry: &Registry) -> Self {
        Self {
            executions_total: Counter::new(
                format!("{}_executions_total", agent_name),
                "Total number of agent executions"
            ).unwrap(),

            execution_duration: Histogram::new(
                format!("{}_execution_duration_seconds", agent_name),
                "Agent execution duration"
            ).unwrap(),

            errors_total: Counter::new(
                format!("{}_errors_total", agent_name),
                "Total errors"
            ).unwrap(),

            artifacts_produced: Counter::new(
                format!("{}_artifacts_total", agent_name),
                "Artifacts produced"
            ).unwrap(),
        }
    }

    pub fn record_execution(&self, duration: f64, success: bool) {
        self.executions_total.inc();
        self.execution_duration.observe(duration);
        if !success {
            self.errors_total.inc();
        }
    }
}
```

#### 6.2.2 Pipeline Metrics

**Metrics to Track:**

- `pipeline_executions_total` - Total pipeline runs
- `pipeline_duration_seconds` - End-to-end duration
- `pipeline_success_rate` - Success percentage
- `pipeline_phase_duration_seconds{phase}` - Duration per phase
- `pipeline_artifacts_size_bytes{type}` - Artifact sizes
- `pipeline_queue_depth` - Pending translations

#### 6.2.3 NVIDIA Integration Metrics

**Existing Metrics (from NVIDIA stack):**

- `nemo_inference_duration_seconds` - NeMo latency
- `cuda_kernel_execution_time_ms` - CUDA kernel perf
- `triton_request_duration_seconds` - Triton serving
- `gpu_utilization_percent` - GPU usage
- `gpu_memory_used_bytes` - GPU memory

### 6.3 Logging Architecture

**Framework:** `tracing` (Rust) + structured JSON logs

```rust
use tracing::{info, error, warn, debug, instrument};

#[instrument(skip(self, input))]
async fn execute(&self, input: Self::Input) -> Result<Self::Output> {
    info!(
        agent = %self.name(),
        input_size = input.size(),
        "Starting agent execution"
    );

    let start = Instant::now();

    match self.process(input).await {
        Ok(output) => {
            info!(
                duration_ms = start.elapsed().as_millis(),
                output_size = output.size(),
                "Agent execution succeeded"
            );
            Ok(output)
        }
        Err(e) => {
            error!(
                error = %e,
                duration_ms = start.elapsed().as_millis(),
                "Agent execution failed"
            );
            Err(e)
        }
    }
}
```

### 6.4 Tracing (Distributed)

**Framework:** OpenTelemetry

**Pattern:** Trace requests across Rust → Python → GPU

```rust
use opentelemetry::trace::{Tracer, Span};

async fn translate_with_nemo(&self, code: &str) -> Result<String> {
    let tracer = global::tracer("portalis");
    let mut span = tracer.start("nemo_translation");

    span.set_attribute("code_length", code.len() as i64);

    let result = self.nemo_client.translate(code).await;

    match &result {
        Ok(rust_code) => {
            span.set_attribute("output_length", rust_code.len() as i64);
            span.set_attribute("success", true);
        }
        Err(e) => {
            span.set_attribute("success", false);
            span.set_attribute("error", e.to_string());
        }
    }

    span.end();
    result
}
```

### 6.5 Dashboards

**Grafana Dashboards:**

1. **Pipeline Overview**
   - Success rate gauge
   - Throughput graph (translations/hour)
   - Average duration line chart
   - Phase breakdown pie chart

2. **Agent Performance**
   - Execution duration heatmap
   - Error rate by agent
   - Artifact size distributions

3. **NVIDIA Stack**
   - GPU utilization
   - NeMo inference latency
   - CUDA kernel performance
   - Triton serving metrics

4. **SLA Tracking**
   - P50, P95, P99 latencies
   - Availability percentage
   - Error budget remaining

---

## 7. Implementation Roadmap

### 7.1 Phase 1: Core Platform Completion (Week 1-2)

**Goal:** Complete all agent implementations

**Tasks:**

1. **Ingest Agent Enhancement** (2 days)
   - Replace regex parser with `tree-sitter-python`
   - Add class and decorator support
   - Implement dependency extraction
   - Write 20+ unit tests

2. **Analysis Agent Enhancement** (3 days)
   - Add type flow analysis
   - Implement control flow graphs
   - Integrate NeMo for embeddings (mock)
   - Write 25+ unit tests

3. **SpecGen Agent Implementation** (3 days)
   - Implement trait generation
   - Add ABI design logic
   - Error type mapping
   - Write 20+ unit tests + golden tests

4. **Transpiler Enhancement** (4 days)
   - Expression translation
   - Control flow (if/for/while)
   - Error handling (try/except)
   - Class translation
   - Write 40+ unit tests + property tests

5. **Build Agent Enhancement** (1 day)
   - Better error parsing
   - Dependency resolution
   - Write 15+ unit tests

6. **Test Agent Implementation** (3 days)
   - WASM execution harness
   - Test generation
   - Golden validation
   - Write 18+ unit tests

7. **Packaging Agent Implementation** (2 days)
   - NIM container building
   - Triton deployment
   - Write 12+ unit tests

**Deliverables:**
- All agents functional (basic)
- 140+ unit tests passing
- 50+ integration tests passing
- 10+ E2E tests passing

### 7.2 Phase 2: NVIDIA Integration (Week 3-4)

**Goal:** Connect Rust agents to NVIDIA services

**Tasks:**

1. **gRPC Client Generation** (2 days)
   - Generate Rust stubs from protobuf
   - Implement connection pooling
   - Add retry logic

2. **NeMo Integration** (3 days)
   - Connect SpecGen to NeMo
   - Connect Transpiler to NeMo
   - Add fallback mechanisms
   - Integration tests with mock

3. **CUDA Integration** (2 days)
   - Connect Analysis to CUDA via NeMo
   - Parallel test execution
   - Integration tests

4. **Triton Deployment** (2 days)
   - Implement Triton deployer in Packaging
   - Model repository management
   - Integration tests

5. **NIM Container Building** (2 days)
   - Docker API integration
   - Container building
   - Registry push

6. **Configuration Management** (1 day)
   - TOML config files
   - Environment variable override
   - Validation

**Deliverables:**
- All NVIDIA services integrated
- Integration tests passing
- Configuration system in place

### 7.3 Phase 3: Testing & Validation (Week 5-6)

**Goal:** Achieve comprehensive test coverage

**Tasks:**

1. **Acceptance Test Suite** (3 days)
   - 10+ E2E scenarios
   - Golden test fixtures
   - Property-based tests

2. **Mock Infrastructure** (2 days)
   - NeMo service mock
   - File system mock
   - Compiler mock

3. **Performance Testing** (2 days)
   - Load tests
   - Stress tests
   - Benchmark suite

4. **Security Testing** (1 day)
   - Input validation tests
   - Injection attack tests
   - Dependency scanning

5. **Coverage Analysis** (1 day)
   - Generate coverage reports
   - Identify gaps
   - Add missing tests

6. **CI/CD Pipeline Update** (2 days)
   - Add Rust tests to CI
   - Integration with NVIDIA tests
   - Deployment automation

**Deliverables:**
- 200+ total tests passing
- 85%+ code coverage
- CI pipeline green
- Security scan passing

### 7.4 Phase 4: Production Readiness (Week 7-8)

**Goal:** Deploy to production

**Tasks:**

1. **Monitoring Integration** (2 days)
   - Add Prometheus metrics
   - Create Grafana dashboards
   - OpenTelemetry tracing

2. **Documentation** (3 days)
   - API documentation
   - User guide
   - Deployment guide
   - Architecture diagrams

3. **Performance Optimization** (2 days)
   - Profile bottlenecks
   - Optimize hot paths
   - Parallel execution

4. **Deployment** (2 days)
   - Kubernetes manifests
   - Helm charts
   - Deploy to staging

5. **Production Validation** (2 days)
   - Canary deployment
   - Load testing in prod
   - Monitor metrics

6. **Handoff** (1 day)
   - Documentation review
   - Knowledge transfer
   - Runbook creation

**Deliverables:**
- Production deployment
- Monitoring dashboards live
- Documentation complete
- SLA metrics tracked

---

## 8. Success Criteria

### 8.1 Functional Requirements

- ✅ All 7 agents implemented and tested
- ✅ End-to-end pipeline operational
- ✅ NVIDIA integration complete
- ✅ Script mode: fibonacci.py → WASM in <60s
- ✅ Library mode: <5K LOC in <15 min

### 8.2 Quality Requirements

- ✅ 200+ tests passing (70% unit, 25% integration, 5% E2E)
- ✅ 85%+ code coverage
- ✅ 0 critical security vulnerabilities
- ✅ CI pipeline <30 minutes
- ✅ All clippy warnings resolved

### 8.3 Performance Requirements

- ✅ Pipeline P95 latency <5 minutes
- ✅ Throughput >10 translations/hour
- ✅ GPU utilization >70%
- ✅ NeMo P95 latency <500ms

### 8.4 Operational Requirements

- ✅ Kubernetes deployment manifests
- ✅ Prometheus metrics exported
- ✅ Grafana dashboards created
- ✅ Documentation complete
- ✅ Runbooks written

---

## 9. Conclusion

This architecture provides a comprehensive blueprint for completing the Portalis SPARC London TDD framework build. The design:

1. **Leverages Existing Work:** Builds on the 18 Rust files and 21,000 LOC NVIDIA stack already implemented
2. **Follows London TDD:** Outside-in development with comprehensive mocking
3. **Enables NVIDIA Acceleration:** Clear integration patterns with NeMo, CUDA, Triton, NIM
4. **Production Ready:** Full CI/CD, monitoring, and deployment strategy
5. **Incremental Delivery:** 8-week phased implementation with clear milestones

**Next Steps:**
1. Review and approve this architecture
2. Begin Phase 1 (Core Platform Completion)
3. Weekly progress reviews against roadmap
4. Adapt plan based on learnings

---

## Appendix A: File Structure

```
/workspace/portalis/
├── core/                          # ✅ Core abstractions (Rust)
│   ├── src/
│   │   ├── agent.rs              # Agent trait
│   │   ├── message.rs            # Message bus
│   │   ├── types.rs              # Pipeline types
│   │   └── error.rs              # Error handling
│   └── Cargo.toml
│
├── agents/                        # ✅ Agent implementations (Rust)
│   ├── ingest/src/lib.rs         # 70% complete
│   ├── analysis/src/lib.rs       # 60% complete
│   ├── specgen/src/lib.rs        # 20% complete
│   ├── transpiler/src/lib.rs     # 50% complete
│   ├── build/src/lib.rs          # 70% complete
│   ├── test/src/lib.rs           # 30% complete
│   └── packaging/src/lib.rs      # 25% complete
│
├── orchestration/                 # ✅ Pipeline orchestration (Rust)
│   ├── src/lib.rs                # 80% complete
│   └── Cargo.toml
│
├── cli/                           # ✅ CLI interface (Rust)
│   ├── src/main.rs               # Basic implementation
│   └── Cargo.toml
│
├── nemo-integration/              # ✅ NeMo service (Python)
│   ├── src/
│   │   ├── translation/
│   │   │   ├── nemo_service.py   # Model wrapper
│   │   │   └── translator.py     # Translation logic
│   │   ├── mapping/
│   │   │   ├── type_mapper.py    # Type conversion
│   │   │   └── error_mapper.py   # Error mapping
│   │   └── validation/
│   │       └── validator.py      # Output validation
│   └── tests/
│
├── cuda-acceleration/             # ✅ CUDA kernels (C++/Python)
│   ├── kernels/
│   │   ├── ast_parallel.cu       # Parallel AST ops
│   │   └── embedding.cu          # Embedding generation
│   └── bindings/python/
│
├── nim-microservices/             # ✅ NIM API (Python)
│   ├── api/
│   │   ├── routes/
│   │   │   ├── translation.py    # REST endpoints
│   │   │   └── health.py         # Health checks
│   │   └── main.py               # FastAPI app
│   ├── grpc/
│   │   └── server.py             # gRPC server
│   └── tests/
│
├── deployment/                    # ✅ Deployment configs
│   ├── triton/
│   │   ├── configs/              # Triton configs
│   │   ├── docker-compose.yaml   # Local dev
│   │   └── models/               # Model repository
│   └── k8s/                      # ❌ TO CREATE
│       ├── namespace.yaml
│       ├── deployment-*.yaml
│       └── service.yaml
│
├── tests/                         # ⚠️ PARTIAL
│   ├── unit/                     # ❌ TO CREATE (Rust)
│   ├── integration/              # ✅ EXISTS (Python only)
│   ├── acceptance/               # ❌ TO CREATE (Rust)
│   ├── mocks/                    # ❌ TO CREATE
│   └── golden/                   # ❌ TO CREATE
│       ├── inputs/
│       └── expected/
│
├── monitoring/                    # ✅ Monitoring stack (Python)
│   ├── prometheus/
│   └── grafana/
│
├── benchmarks/                    # ✅ Performance tests (Python)
│   ├── benchmark_nemo.py
│   └── benchmark_e2e.py
│
├── .github/workflows/             # ⚠️ PARTIAL
│   ├── test.yml                  # ✅ NVIDIA tests only
│   └── ci.yml                    # ❌ TO CREATE (Rust + Python)
│
├── Cargo.toml                     # ✅ Workspace definition
├── docker-compose.yaml            # ❌ TO CREATE
└── README.md                      # ✅ EXISTS

Legend:
✅ Complete
⚠️ Partial (needs enhancement)
❌ Missing (needs creation)
```

---

## Appendix B: Technology Stack Summary

| Layer | Technology | Status | Purpose |
|-------|-----------|--------|---------|
| **Core Platform** | Rust | ✅ 80% | Agents, orchestration, CLI |
| **Build System** | Cargo | ✅ Complete | Rust workspace management |
| **Async Runtime** | Tokio | ✅ Complete | Async execution |
| **Serialization** | Serde | ✅ Complete | Data interchange |
| **Logging** | tracing | ✅ Complete | Structured logging |
| **NeMo Service** | Python | ✅ Complete | LLM translation |
| **CUDA Kernels** | C++/CUDA | ✅ Complete | GPU acceleration |
| **NIM API** | FastAPI | ✅ Complete | REST/gRPC endpoints |
| **Triton** | NVIDIA Triton | ✅ Complete | Model serving |
| **Container** | Docker | ✅ Complete | Containerization |
| **Orchestration** | Kubernetes | ⚠️ Partial | Production deployment |
| **Monitoring** | Prometheus | ✅ Complete | Metrics collection |
| **Dashboards** | Grafana | ✅ Complete | Visualization |
| **Tracing** | OpenTelemetry | ❌ To Add | Distributed tracing |
| **CI/CD** | GitHub Actions | ⚠️ Partial | Automation |
| **Testing** | cargo test, pytest | ⚠️ Partial | Test execution |

---

**End of Architecture Document**

*This architecture is ready for implementation. All components are designed to integrate seamlessly with the existing NVIDIA stack while following London School TDD principles.*
