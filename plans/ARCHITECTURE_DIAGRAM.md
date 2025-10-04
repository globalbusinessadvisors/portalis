# PORTALIS Architecture Diagrams
## Visual Reference for Completion Stage

---

## System Architecture - Full Stack View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  CLI (Rust)  │  │  REST API    │  │  gRPC API    │  │   Web UI     │   │
│  │              │  │  (FastAPI)   │  │  (Python)    │  │  (Future)    │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
└─────────┼──────────────────┼──────────────────┼──────────────────┼──────────┘
          │                  │                  │                  │
          └──────────────────┴──────────────────┴──────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER (Rust)                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        Pipeline Manager                                 │ │
│  │  • State Machine (Idle → Ingesting → ... → Complete)                  │ │
│  │  • Agent Coordinator (sequential + parallel execution)                │ │
│  │  • Error Recovery & Retry Logic                                       │ │
│  │  • Progress Tracking & Telemetry                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Message Bus (Tokio)                            │ │
│  │  • Async message passing between agents                               │ │
│  │  • Artifact routing (PythonAST → RustCode → WASM)                    │ │
│  │  • Event broadcasting (metrics, logs, errors)                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT LAYER (Rust)                                   │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Ingest  │→ │ Analysis │→ │ SpecGen  │→ │Transpiler│→ │  Build   │ →  │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│       ↓             ↓             ↓             ↓             ↓            │
│  PythonAST    TypedFuncs    RustSpec     RustCode      WasmBinary        │
│                                                                              │
│  ┌──────────┐  ┌──────────┐                                                │
│  │   Test   │→ │ Package  │ → FINAL OUTPUT                                │
│  │  Agent   │  │  Agent   │                                                │
│  └──────────┘  └──────────┘                                                │
│       ↓             ↓                                                       │
│  TestResults   NIMContainer                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                   INTEGRATION BRIDGE LAYER                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │  gRPC Clients  │  │  REST Clients  │  │  FFI Bindings  │               │
│  │  (tonic)       │  │  (reqwest)     │  │  (future)      │               │
│  └────────────────┘  └────────────────┘  └────────────────┘               │
│                                                                              │
│  • Connection Pooling                                                       │
│  • Retry Logic & Circuit Breakers                                          │
│  • Request/Response Serialization (protobuf, JSON)                         │
│  • Fallback Mechanisms (template-based when NVIDIA unavailable)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                  NVIDIA ACCELERATION LAYER (Python)                          │
│                                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │    NeMo    │  │    CUDA    │  │   Triton   │  │    NIM     │           │
│  │ (LLM Trans)│  │(GPU Kernels)│  │ (Serving)  │  │ (Containers)│          │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘           │
│                                                                              │
│  ┌────────────┐  ┌────────────┐                                            │
│  │ DGX Cloud  │  │ Omniverse  │                                            │
│  │(Distributed)│  │(WASM Runtime)│                                          │
│  └────────────┘  └────────────┘                                            │
│                                                                              │
│  GPU Cluster: NVIDIA A100/H100 GPUs                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Pipeline - Detailed Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           INPUT: Python Source Code                           │
│  def fibonacci(n: int) -> int:                                               │
│      if n <= 1:                                                              │
│          return n                                                            │
│      return fibonacci(n-1) + fibonacci(n-2)                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                          ┌──────────────────┐
                          │   Ingest Agent   │
                          │  • Parse Python  │
                          │  • Extract AST   │
                          │  • Validate      │
                          └──────────────────┘
                                    ↓
                          ┌──────────────────────────────────────┐
                          │        PythonAST Artifact            │
                          │  {                                   │
                          │    functions: [{                     │
                          │      name: "fibonacci",              │
                          │      params: [{name: "n", type: int}]│
                          │      return_type: "int"              │
                          │    }],                               │
                          │    classes: [],                      │
                          │    imports: []                       │
                          │  }                                   │
                          └──────────────────────────────────────┘
                                    ↓
                          ┌──────────────────┐
                          │  Analysis Agent  │◄───┐
                          │  • Type inference│    │ NeMo Embeddings
                          │  • API extraction│    │ (optional)
                          │  • CFG analysis  │    │
                          └──────────────────┘    │
                                    ↓             │
                          ┌──────────────────────────────────────┐
                          │    TypedFunctions + APIContract      │
                          │  typed_functions: [{               │
                          │    name: "fibonacci",              │
                          │    params: [{name: "n", type: I32}]│
                          │    return_type: I32,               │
                          │    confidence: 0.95                │
                          │  }]                                │
                          │  api_contract: {                  │
                          │    signatures: ["fn(i32)->i32"]   │
                          │  }                                │
                          └──────────────────────────────────────┘
                                    ↓
                          ┌──────────────────┐
                          │  SpecGen Agent   │◄───┐
                          │  • Trait gen     │    │ NeMo LLM
                          │  • ABI design    │    │ (trait generation)
                          │  • Error mapping │    │
                          └──────────────────┘    │
                                    ↓             │
                          ┌──────────────────────────────────────┐
                          │          RustSpec Artifact           │
                          │  pub trait Fibonacci {               │
                          │    fn fibonacci(n: i32) -> i32;      │
                          │  }                                   │
                          │                                      │
                          │  // WASI ABI specification           │
                          │  #[wasm_bindgen]                     │
                          │  pub fn fibonacci(n: i32) -> i32     │
                          └──────────────────────────────────────┘
                                    ↓
                          ┌──────────────────┐
                          │ Transpiler Agent │◄───┐
                          │  • Code gen      │    │ NeMo LLM
                          │  • Expression xlat│    │ (translation)
                          │  • Control flow  │    │
                          └──────────────────┘    │ CUDA Ranking
                                    ↓             │ (best candidate)
                          ┌──────────────────────────────────────┐
                          │         RustCode Artifact            │
                          │  pub fn fibonacci(n: i32) -> i32 {   │
                          │      if n <= 1 {                     │
                          │          return n;                   │
                          │      }                               │
                          │      fibonacci(n - 1) + fibonacci(n - 2)│
                          │  }                                   │
                          └──────────────────────────────────────┘
                                    ↓
                          ┌──────────────────┐
                          │   Build Agent    │
                          │  • Cargo setup   │
                          │  • wasm32 compile│
                          │  • Optimization  │
                          └──────────────────┘
                                    ↓
                          ┌──────────────────────────────────────┐
                          │        WasmBinary Artifact           │
                          │  [0x00, 0x61, 0x73, 0x6d, ...       │
                          │   compiled WASM bytecode]            │
                          │  size: 1.2 KB                        │
                          └──────────────────────────────────────┘
                                    ↓
                          ┌──────────────────┐
                          │    Test Agent    │◄───┐
                          │  • WASM exec     │    │ CUDA Parallel
                          │  • Golden test   │    │ (test execution)
                          │  • Perf bench    │    │
                          └──────────────────┘    │
                                    ↓             │
                          ┌──────────────────────────────────────┐
                          │        TestResults Artifact          │
                          │  passed: 10                          │
                          │  failed: 0                           │
                          │  details: [                          │
                          │    {name: "fib_0", result: 0},       │
                          │    {name: "fib_5", result: 5},       │
                          │    {name: "fib_10", result: 55}      │
                          │  ]                                   │
                          └──────────────────────────────────────┘
                                    ↓
                          ┌──────────────────┐
                          │  Package Agent   │◄───┐
                          │  • NIM container │    │ NIM Builder
                          │  • Triton deploy │    │
                          │  • Manifest gen  │    │ Triton API
                          └──────────────────┘    │
                                    ↓             │
┌──────────────────────────────────────────────────────────────────────────────┐
│                     OUTPUT: Deployable Artifacts                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │  WASM Module   │  │ NIM Container  │  │  Test Report   │                │
│  │ fibonacci.wasm │  │ fibonacci:v1   │  │  100% pass     │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## NVIDIA Integration - Service Interaction

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Rust Agent Layer                                │
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│  │ Analysis │   │ SpecGen  │   │Transpiler│   │ Package  │           │
│  │  Agent   │   │  Agent   │   │  Agent   │   │  Agent   │           │
│  └─────┬────┘   └─────┬────┘   └─────┬────┘   └─────┬────┘           │
└────────┼──────────────┼──────────────┼──────────────┼────────────────┘
         │              │              │              │
         │ gRPC         │ gRPC         │ gRPC         │ REST/Docker
         ↓              ↓              ↓              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      NIM Microservices (Python)                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    gRPC Server (port 50051)                        │ │
│  │  service TranslationService {                                     │ │
│  │    rpc Translate(TranslationRequest) → TranslationResponse        │ │
│  │    rpc GenerateEmbeddings(CodeRequest) → EmbeddingResponse        │ │
│  │    rpc RankCandidates(RankingRequest) → RankingResponse           │ │
│  │  }                                                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                             ↓                                            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                   REST API (port 8080)                             │ │
│  │  POST /api/v1/translate                                            │ │
│  │  POST /api/v1/embeddings                                           │ │
│  │  POST /api/v1/rank                                                 │ │
│  │  POST /api/v1/deploy                                               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                             ↓                                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐           │
│  │ NeMo Service   │  │ CUDA Service   │  │ Triton Client  │           │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘           │
└──────────┼───────────────────┼───────────────────┼────────────────────┘
           │                   │                   │
           ↓                   ↓                   ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA Acceleration Services                          │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    NeMo LLM Service                          │      │
│  │  • Model: codegen-7b-mono                                    │      │
│  │  • GPU: A100 (40GB)                                          │      │
│  │  • Batch Size: 32                                            │      │
│  │  • Operations:                                               │      │
│  │    - Python → Rust translation                               │      │
│  │    - Code embedding generation                               │      │
│  │    - Trait/interface generation                              │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    CUDA Parallel Kernels                     │      │
│  │  • GPU: Multi-GPU (A100 x 2)                                 │      │
│  │  • Operations:                                               │      │
│  │    - Parallel AST traversal (10,000 nodes/ms)               │      │
│  │    - Translation candidate ranking (1000 candidates/ms)     │      │
│  │    - Parallel test execution (100 tests/sec)                │      │
│  │    - Embedding similarity (cosine distance, batched)        │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              Triton Inference Server (port 8000)             │      │
│  │  • Serves WASM modules as models                             │      │
│  │  • HTTP/gRPC endpoints                                       │      │
│  │  • Model repository: /models/                                │      │
│  │  • Auto-scaling based on load                                │      │
│  │  • Metrics: requests/sec, latency, GPU usage                 │      │
│  └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Testing Architecture - London TDD

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Test Pyramid (200+ Tests)                            │
│                                                                          │
│                            ╱╲                                            │
│                           ╱  ╲                                           │
│                          ╱ E2E╲          10 Tests (5%)                   │
│                         ╱──────╲        Full stack validation           │
│                        ╱        ╲                                        │
│                       ╱Integration╲    50 Tests (25%)                    │
│                      ╱────────────╲   Agent communication               │
│                     ╱              ╲                                     │
│                    ╱   Unit Tests  ╲  140 Tests (70%)                   │
│                   ╱────────────────╲ Individual logic                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        Test Organization                                 │
│                                                                          │
│  Unit Tests (Isolated, Fast < 1s)                                       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  #[cfg(test)]                                                      │ │
│  │  mod tests {                                                       │ │
│  │    use super::*;                                                   │ │
│  │    use MockNeMoService;  // London TDD mock                        │ │
│  │                                                                    │ │
│  │    #[tokio::test]                                                  │ │
│  │    async fn test_transpiler_with_mock_nemo() {                    │ │
│  │      // Given: Mock NeMo returns specific translation            │ │
│  │      let mock = MockNeMoService::new();                           │ │
│  │      mock.expect_translate()                                      │ │
│  │          .with("def add(a, b): return a + b")                     │ │
│  │          .returns("pub fn add(a: i32, b: i32) -> i32 { a + b }"); │ │
│  │                                                                    │ │
│  │      // When: Transpiler uses mock                                │ │
│  │      let transpiler = TranspilerAgent::with_nemo(mock);           │ │
│  │      let result = transpiler.translate(...).await.unwrap();       │ │
│  │                                                                    │ │
│  │      // Then: Mock interaction verified                           │ │
│  │      assert!(result.contains("pub fn add"));                      │ │
│  │      mock.verify();                                               │ │
│  │    }                                                               │ │
│  │  }                                                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Integration Tests (Services, Medium 1-30s)                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  // tests/integration/test_agent_flow.rs                           │ │
│  │  #[tokio::test]                                                    │ │
│  │  async fn test_ingest_to_analysis_flow() {                        │ │
│  │    let ingest = IngestAgent::new();                               │ │
│  │    let analysis = AnalysisAgent::new();                           │ │
│  │                                                                    │ │
│  │    // Ingest → Analysis                                           │ │
│  │    let ast = ingest.execute(...).await.unwrap();                  │ │
│  │    let typed = analysis.execute(ast).await.unwrap();              │ │
│  │                                                                    │ │
│  │    assert_eq!(typed.functions.len(), 1);                          │ │
│  │  }                                                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  E2E Tests (Full pipeline, Slow 30s-5min)                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  // tests/e2e/test_fibonacci.rs                                    │ │
│  │  #[tokio::test]                                                    │ │
│  │  async fn test_fibonacci_end_to_end() {                           │ │
│  │    // Given: Python fibonacci                                     │ │
│  │    let python = include_str!("fixtures/fibonacci.py");            │ │
│  │                                                                    │ │
│  │    // When: Full pipeline                                         │ │
│  │    let mut pipeline = Pipeline::new();                            │ │
│  │    let result = pipeline.translate(...).await.unwrap();           │ │
│  │                                                                    │ │
│  │    // Then: WASM executes correctly                               │ │
│  │    let output = execute_wasm(result.wasm, "fibonacci", &[10]);    │ │
│  │    assert_eq!(output, 55); // 10th Fibonacci number              │ │
│  │  }                                                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      Kubernetes Production Cluster                        │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                          Ingress Controller                         │ │
│  │  • HTTPS termination                                                │ │
│  │  • Load balancing                                                   │ │
│  │  • Rate limiting                                                    │ │
│  └──────────────────┬──────────────────┬──────────────────────────────┘ │
│                     │                  │                                  │
│      ┌──────────────┴───────┐   ┌─────┴──────────┐                      │
│      │                      │   │                │                       │
│  ┌───▼──────────┐  ┌────────▼───┐  ┌─▼────────────┐                    │
│  │ Portalis     │  │ NIM API    │  │ Triton       │                     │
│  │ Core Service │  │ Service    │  │ Inference    │                     │
│  │ (Rust)       │  │ (Python)   │  │ Server       │                     │
│  │              │  │            │  │              │                     │
│  │ Replicas: 3  │  │ Replicas:5 │  │ Replicas: 2  │                     │
│  │ CPU: 2 cores │  │ CPU: 4     │  │ GPU: A100 x2 │                     │
│  │ RAM: 4GB     │  │ RAM: 8GB   │  │ RAM: 32GB    │                     │
│  └───────┬──────┘  └─────┬──────┘  └───┬──────────┘                    │
│          │               │             │                                 │
│          └───────────────┴─────────────┘                                 │
│                          │                                               │
│  ┌───────────────────────▼──────────────────────┐                       │
│  │          Persistent Storage (PVC)            │                        │
│  │  • Models: /models (100GB)                   │                        │
│  │  • Artifacts: /workspace (50GB)              │                        │
│  │  • Logs: /logs (20GB)                        │                        │
│  └──────────────────────────────────────────────┘                       │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                      Observability Stack                            │ │
│  │                                                                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │ │
│  │  │ Prometheus   │  │   Grafana    │  │ OpenTelemetry│             │ │
│  │  │ (Metrics)    │  │ (Dashboard)  │  │  (Tracing)   │             │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘

Local Development (Docker Compose)
┌──────────────────────────────────────────────────────────────────────────┐
│  docker-compose.yaml                                                      │
│                                                                           │
│  services:                                                                │
│    portalis-core:    # Rust CLI + orchestration                         │
│    nemo-service:     # NeMo LLM (GPU required)                           │
│    nim-api:          # REST/gRPC gateway                                 │
│    triton:           # Inference server                                  │
│    prometheus:       # Metrics collection                                │
│    grafana:          # Dashboards                                        │
│                                                                           │
│  Volumes:                                                                 │
│    nemo-models:/models                                                   │
│    workspace:/workspace                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## CI/CD Pipeline

```
GitHub Push/PR
      ↓
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions Workflow                       │
│                                                                  │
│  Stage 1: Rust Core Tests (5 min)                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • cargo clippy (linting)                                 │  │
│  │ • cargo fmt --check (formatting)                         │  │
│  │ • cargo test --all-features (unit + doc tests)           │  │
│  │ • cargo audit (security)                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  Stage 2: Integration Tests (10 min)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Start mock NeMo service (MockServer)                   │  │
│  │ • cargo test --test integration                          │  │
│  │ • Verify agent-to-agent communication                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  Stage 3: NVIDIA Stack Tests (15 min)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • pytest tests/ (Python unit + integration)              │  │
│  │ • Mock GPU tests (no actual GPU required)                │  │
│  │ • Coverage report (target: 85%+)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  Stage 4: E2E Tests (Docker) (20 min)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • docker-compose -f docker-compose.test.yaml up          │  │
│  │ • Run full pipeline tests                                │  │
│  │ • Validate WASM output                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  Stage 5: Security Scan (5 min)                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • cargo audit (Rust dependencies)                        │  │
│  │ • bandit (Python security)                               │  │
│  │ • safety check (Python dependencies)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  Stage 6: Build Images (10 min) [main branch only]             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • docker build portalis/core:latest                      │  │
│  │ • docker build portalis/nim:latest                       │  │
│  │ • docker push to registry                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  Stage 7: Deploy (5 min) [main branch only]                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • kubectl apply -f deployment/k8s/                       │  │
│  │ • Canary deployment (10% traffic)                        │  │
│  │ • Monitor metrics                                        │  │
│  │ • Auto-rollback on errors                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Total Pipeline Time: ~30 minutes (target)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Monitoring & Observability

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Metrics Flow                                   │
│                                                                          │
│  Rust Agents                  Python Services              Infrastructure│
│  ┌──────────┐                ┌──────────┐                ┌──────────┐  │
│  │ Ingest   │──metrics──┐    │  NeMo    │──metrics──┐    │   GPU    │  │
│  │ Analysis │──────────►│    │  Service │──────────►│    │  Metrics │  │
│  │ Transpiler│           │    │  NIM API │           │    │  (DCGM)  │  │
│  │ Build    │           │    │  Triton  │           │    └─────┬────┘  │
│  │ Test     │           │    └──────────┘           │          │       │
│  │ Package  │           │                           │          │       │
│  └──────────┘           │                           │          │       │
│                         ↓                           ↓          ↓       │
│              ┌────────────────────────────────────────────────────┐    │
│              │           Prometheus (port 9090)                   │    │
│              │  • Scrapes metrics every 15s                       │    │
│              │  • Retention: 30 days                              │    │
│              │  • Alerting rules configured                       │    │
│              └────────────────┬───────────────────────────────────┘    │
│                               │                                         │
│                               ↓                                         │
│              ┌────────────────────────────────────────────────────┐    │
│              │              Grafana (port 3000)                   │    │
│              │  Dashboards:                                       │    │
│              │  • Pipeline Overview (success rate, throughput)    │    │
│              │  • Agent Performance (latency, errors)             │    │
│              │  • NVIDIA Stack (GPU usage, NeMo latency)          │    │
│              │  • SLA Tracking (P50/P95/P99)                      │    │
│              └────────────────────────────────────────────────────┘    │
│                                                                          │
│  Key Metrics:                                                            │
│  • pipeline_executions_total                                            │
│  • pipeline_duration_seconds (P50, P95, P99)                            │
│  • pipeline_success_rate                                                │
│  • agent_execution_duration_seconds{agent="transpiler"}                 │
│  • nemo_inference_duration_seconds                                      │
│  • cuda_kernel_execution_time_ms                                        │
│  • gpu_utilization_percent                                              │
│  • wasm_binary_size_bytes                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure Map

```
/workspace/portalis/
│
├── core/                           ✅ Rust core abstractions
│   ├── src/
│   │   ├── agent.rs               Agent trait + metadata
│   │   ├── message.rs             Message bus
│   │   ├── types.rs               Pipeline types
│   │   ├── error.rs               Error handling
│   │   └── lib.rs                 Public exports
│   └── Cargo.toml
│
├── agents/                         ✅ Agent implementations
│   ├── ingest/src/lib.rs          Python parser (70%)
│   ├── analysis/src/lib.rs        Type inference (60%)
│   ├── specgen/src/lib.rs         Rust spec gen (20%)
│   ├── transpiler/src/lib.rs      Code translation (50%)
│   ├── build/src/lib.rs           WASM compilation (70%)
│   ├── test/src/lib.rs            Validation (30%)
│   └── packaging/src/lib.rs       NIM/Triton deploy (25%)
│
├── orchestration/                  ✅ Pipeline coordination
│   ├── src/lib.rs                 Pipeline manager (80%)
│   └── Cargo.toml
│
├── cli/                            ✅ Command-line interface
│   ├── src/main.rs                CLI entry point
│   └── Cargo.toml
│
├── nemo-integration/               ✅ NeMo LLM service
│   ├── src/
│   │   ├── translation/           Translation logic
│   │   ├── mapping/               Type/error mapping
│   │   └── validation/            Output validation
│   └── tests/
│
├── cuda-acceleration/              ✅ GPU kernels
│   ├── kernels/
│   │   ├── ast_parallel.cu        Parallel AST ops
│   │   └── embedding.cu           Embedding generation
│   └── bindings/python/
│
├── nim-microservices/              ✅ REST/gRPC API
│   ├── api/
│   │   ├── routes/                REST endpoints
│   │   └── main.py                FastAPI app
│   ├── grpc/
│   │   └── server.py              gRPC server
│   └── tests/
│
├── deployment/                     ✅ Deployment configs
│   ├── triton/
│   │   ├── docker-compose.yaml    Local dev
│   │   └── configs/               Triton configs
│   └── k8s/                       ❌ To create
│       ├── deployment-*.yaml
│       └── service.yaml
│
├── tests/                          ⚠️ Partial
│   ├── unit/                      ❌ To create (Rust)
│   ├── integration/               ✅ Exists (Python)
│   ├── e2e/                       ❌ To create (Rust)
│   ├── mocks/                     ❌ To create
│   └── golden/                    ❌ To create
│
├── monitoring/                     ✅ Observability
│   ├── prometheus/
│   └── grafana/
│
├── .github/workflows/              ⚠️ Partial
│   ├── test.yml                   ✅ NVIDIA tests
│   └── ci.yml                     ❌ To create (full)
│
├── COMPLETION_STAGE_ARCHITECTURE.md  ✅ Full spec (1806 lines)
├── ARCHITECTURE_SUMMARY.md           ✅ This summary
├── ARCHITECTURE_DIAGRAM.md           ✅ Visual diagrams
└── Cargo.toml                        ✅ Workspace def
```

---

**Legend:**
- ✅ Complete
- ⚠️ Partial (needs enhancement)
- ❌ Missing (needs creation)
- Percentages indicate implementation completion

---

*These diagrams provide a visual reference for the architecture described in [COMPLETION_STAGE_ARCHITECTURE.md](COMPLETION_STAGE_ARCHITECTURE.md). Use them for team communication, onboarding, and implementation planning.*

**Last Updated:** 2025-10-03
**Status:** ARCHITECTURE FINALIZED
