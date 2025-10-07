# Wassette Integration Visual Diagrams

## 1. High-Level System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        PORTALIS PLATFORM                              │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   PRESENTATION LAYER                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │
│  │  │   CLI    │  │ REST API │  │  Web UI  │  │Omniverse │   │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              ORCHESTRATION & MESSAGE BUS                    │    │
│  │     Pipeline Controller | State Machine | Error Handler     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    AGENT SWARM LAYER                        │    │
│  │  ┌──────┐  ┌────────┐  ┌────────┐  ┌──────┐  ┌──────┐     │    │
│  │  │Ingest│→│Analysis│→│SpecGen │→│Trans-│→│Build │     │    │
│  │  │      │  │        │  │        │  │piler │  │      │     │    │
│  │  └──────┘  └────────┘  └────────┘  └──────┘  └──────┘     │    │
│  │                                         ↓          ↓        │    │
│  │  ┌──────┐  ┌─────────┐        ┌────────────────────────┐  │    │
│  │  │Test  │  │Packaging│   ┌───→│  WASSETTE BRIDGE  ⭐  │  │    │
│  │  │      │  │         │   │    │  - WASM Validation    │  │    │
│  │  └──────┘  └─────────┘   │    │  - Component Loading  │  │    │
│  │      ↓           ↓        │    │  - Secure Execution   │  │    │
│  │      └───────────┴────────┘    │  - Permission Control │  │    │
│  │                                 └────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              NVIDIA ACCELERATION (Optional)                 │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │
│  │  │  NeMo    │  │   CUDA   │  │  Triton  │  │   DGX    │   │    │
│  │  │  Bridge  │  │  Bridge  │  │  Server  │  │  Cloud   │   │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              WASSETTE RUNTIME ENVIRONMENT  ⭐               │    │
│  │  ┌────────────────────────────────────────────────────────┐ │    │
│  │  │           Wasmtime Security Sandbox                    │ │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │ │    │
│  │  │  │ WASM Component│ │ WASM Component│ │WASM Component││ │    │
│  │  │  │   (Python→   │  │   (Rust→     │  │  (JS→      │ │ │    │
│  │  │  │    WASM)     │  │    WASM)     │  │   WASM)    │ │ │    │
│  │  │  └──────────────┘  └──────────────┘  └─────────────┘ │ │    │
│  │  │                                                        │ │    │
│  │  │  Permission Boundaries:                               │ │    │
│  │  │  [Filesystem] [Network] [Environment] [Memory]        │ │    │
│  │  └────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘

Legend: ⭐ = New Wassette Integration
```

---

## 2. Wassette Bridge Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              PORTALIS WASSETTE BRIDGE ARCHITECTURE              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PUBLIC API LAYER                             │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  WassetteClient                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │ │
│  │  │    new()    │  │load_component│ │  validate   │      │ │
│  │  │  default()  │  │  execute()   │  │is_available │      │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   INTERNAL MODULES                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Runtime    │  │  Component   │  │  Validator   │         │
│  │   Module     │  │   Module     │  │   Module     │         │
│  │              │  │              │  │              │         │
│  │ - Initialize │  │ - Handle     │  │ - Validate   │         │
│  │ - Execute    │  │ - Metadata   │  │ - Report     │         │
│  │ - Cleanup    │  │ - Lifecycle  │  │ - Errors     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │ Permissions  │  │     MCP      │                           │
│  │   Module     │  │   Module     │                           │
│  │              │  │              │                           │
│  │ - Configure  │  │ - Protocol   │                           │
│  │ - Enforce    │  │ - Integration│                           │
│  │ - Audit      │  │ - AI Tools   │                           │
│  └──────────────┘  └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   WASMTIME LAYER                                │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  Wasmtime Engine                          │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │ │
│  │  │ Module  │  │Instance │  │  Store  │  │  WASI   │     │ │
│  │  │  Load   │  │ Create  │  │  Manage │  │ Context │     │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Integration Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│           PYTHON TO WASM PIPELINE WITH WASSETTE                 │
└─────────────────────────────────────────────────────────────────┘

User Input: Python Script (calculator.py)
     │
     ↓
┌──────────────┐
│ Ingest Agent │  Parse Python → AST
└──────────────┘
     │
     ↓
┌──────────────┐
│Analysis Agent│  Extract Types & Contracts
└──────────────┘
     │
     ↓
┌──────────────┐
│ SpecGen Agent│  Generate Rust Specifications
└──────────────┘
     │
     ↓
┌──────────────┐
│  Transpiler  │  Translate Python → Rust
│    Agent     │  Compile Rust → WASM
└──────────────┘
     │
     ↓─────────────────────────────────────┐
     │                                     │
     ↓                                     ↓
┌──────────────┐                    ┌──────────────┐
│  WASM File   │ ───────────────→   │   WASSETTE   │  ⭐ NEW
│calculator.wasm│                    │    BRIDGE    │
└──────────────┘                    └──────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ↓                      ↓                      ↓
              ┌───────────┐         ┌───────────┐         ┌───────────┐
              │ Validate  │         │   Load    │         │  Execute  │
              │  - Magic  │         │ Component │         │ Component │
              │  - Format │         │  - Parse  │         │  - Run    │
              │  - Size   │         │  - Store  │         │  - Capture│
              └───────────┘         └───────────┘         └───────────┘
                    │                      │                      │
                    └──────────────────────┴──────────────────────┘
                                           │
                                           ↓
                                  ┌─────────────────┐
                                  │ Validation      │
                                  │    Report       │
                                  │                 │
                                  │ ✓ is_valid      │
                                  │ ✓ exports[]     │
                                  │ ✓ imports[]     │
                                  │ ✓ metadata      │
                                  └─────────────────┘
                                           │
     ┌─────────────────────────────────────┘
     │
     ↓
┌──────────────┐
│  Test Agent  │  Run Tests in Sandbox  ⭐ Uses Wassette
└──────────────┘
     │
     ↓
┌──────────────┐
│  Packaging   │  Verify WASM Package   ⭐ Uses Wassette
│    Agent     │
└──────────────┘
     │
     ↓
Final Output:
  - Rust source code
  - Validated WASM binary ✓
  - Test results
  - Deployment package
```

---

## 4. Wassette Bridge Class Diagram

```
┌────────────────────────────────────────────────────────────┐
│                    WassetteClient                          │
├────────────────────────────────────────────────────────────┤
│ - runtime: Option<WassetteRuntime>                        │
│ - config: WassetteConfig                                   │
├────────────────────────────────────────────────────────────┤
│ + new(config: WassetteConfig) -> Result<Self>             │
│ + default() -> Result<Self>                               │
│ + load_component(path: &Path) -> Result<ComponentHandle>  │
│ + execute_component(...) -> Result<ExecutionResult>       │
│ + validate_component(...) -> Result<ValidationReport>     │
│ + is_available() -> bool                                   │
└────────────────────────────────────────────────────────────┘
                    │
                    │ contains
                    ↓
┌────────────────────────────────────────────────────────────┐
│                  WassetteConfig                            │
├────────────────────────────────────────────────────────────┤
│ + enable_sandbox: bool                                     │
│ + max_memory_mb: usize                                     │
│ + max_execution_time_secs: u64                             │
│ + permissions: ComponentPermissions                        │
└────────────────────────────────────────────────────────────┘
                    │
                    │ has-a
                    ↓
┌────────────────────────────────────────────────────────────┐
│              ComponentPermissions                          │
├────────────────────────────────────────────────────────────┤
│ + allow_fs: bool                                           │
│ + allow_network: bool                                      │
│ + allow_env: bool                                          │
│ + allowed_paths: Vec<String>                               │
│ + allowed_hosts: Vec<String>                               │
│ + allowed_env_vars: Vec<String>                            │
├────────────────────────────────────────────────────────────┤
│ + restrictive() -> Self                                    │
│ + permissive() -> Self                                     │
│ + for_testing() -> Self                                    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                 WassetteRuntime                            │
├────────────────────────────────────────────────────────────┤
│ - engine: Engine                                           │
│ - config: WassetteConfig                                   │
├────────────────────────────────────────────────────────────┤
│ + new(config: &WassetteConfig) -> Result<Self>            │
│ + load_component(path: &Path) -> Result<ComponentHandle>  │
│ + execute_component(...) -> Result<ExecutionResult>       │
│ - extract_metadata(bytes: &[u8]) -> Result<Metadata>      │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                  ComponentHandle                           │
├────────────────────────────────────────────────────────────┤
│ + id: String                                               │
│ + path: PathBuf                                            │
│ + metadata: ComponentMetadata                              │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│               ValidationReport                             │
├────────────────────────────────────────────────────────────┤
│ + is_valid: bool                                           │
│ + errors: Vec<String>                                      │
│ + warnings: Vec<String>                                    │
│ + metadata: ComponentMetadata                              │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│               ExecutionResult                              │
├────────────────────────────────────────────────────────────┤
│ + success: bool                                            │
│ + stdout: String                                           │
│ + stderr: String                                           │
│ + exit_code: Option<i32>                                   │
│ + execution_time_ms: f64                                   │
│ + memory_used_bytes: usize                                 │
└────────────────────────────────────────────────────────────┘
```

---

## 5. Agent Integration Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSPILER AGENT                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Core Transpilation Logic (Existing)                 │  │
│  │  - Parse Python                                      │  │
│  │  - Generate Rust                                     │  │
│  │  - Compile to WASM                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  WASM Output (calculator.wasm)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  #[cfg(feature = "wassette")]                        │  │
│  │  Validate WASM Using Wassette Bridge  ⭐ NEW        │  │
│  │                                                       │  │
│  │  let client = WassetteClient::default()?;            │  │
│  │  let report = client.validate_component(&wasm)?;     │  │
│  │  if !report.is_valid {                               │  │
│  │      return Err(...);                                │  │
│  │  }                                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Validated WASM Output ✓                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       TEST AGENT                            │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Test Preparation (Existing)                         │  │
│  │  - Load test cases                                   │  │
│  │  - Prepare test data                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  #[cfg(feature = "wassette")]                        │  │
│  │  Execute WASM Tests in Sandbox  ⭐ NEW              │  │
│  │                                                       │  │
│  │  let config = WassetteConfig {                       │  │
│  │      permissions: ComponentPermissions::for_testing() │  │
│  │  };                                                   │  │
│  │  let client = WassetteClient::new(config)?;          │  │
│  │  let component = client.load_component(&wasm)?;      │  │
│  │  let result = client.execute_component(component)?;  │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Test Results with Sandbox Isolation ✓               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PACKAGING AGENT                           │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Package Preparation (Existing)                      │  │
│  │  - Create NIM container                              │  │
│  │  - Prepare artifacts                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  #[cfg(feature = "wassette")]                        │  │
│  │  Verify WASM Before Deployment  ⭐ NEW              │  │
│  │                                                       │  │
│  │  let client = WassetteClient::default()?;            │  │
│  │  let report = client.validate_component(&wasm)?;     │  │
│  │  log_component_metadata(&report.metadata);           │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Verified Deployment Package ✓                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Security Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│              WASM COMPONENT IN WASSETTE SANDBOX             │
└─────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  WASM Component │
                    │  (User Code)    │
                    └─────────────────┘
                            ↕
              ┌─────────────────────────┐
              │  Permission Boundaries  │
              └─────────────────────────┘
                            ↕
    ┌───────────┬───────────┴───────────┬───────────┐
    ↓           ↓                       ↓           ↓
┌─────────┐ ┌─────────┐           ┌─────────┐ ┌─────────┐
│Filesystem│ │ Network │           │Environment│ │ Memory │
│         │ │         │           │         │ │        │
│ ✓ /tmp  │ │ ✗ Denied│           │ ✓ TEST_*│ │ ✓ 128MB│
│ ✗ /etc  │ │         │           │ ✗ Other │ │ ✗ More │
└─────────┘ └─────────┘           └─────────┘ └─────────┘

Permission Levels:
┌──────────────────────────────────────────────────────────┐
│ RESTRICTIVE (Default)                                    │
│ - Filesystem: Denied                                     │
│ - Network: Denied                                        │
│ - Environment: Denied                                    │
│ - Memory: 128MB                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ TESTING                                                  │
│ - Filesystem: /tmp only                                  │
│ - Network: Denied                                        │
│ - Environment: TEST_* only                               │
│ - Memory: 256MB                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ PRODUCTION (Configured)                                  │
│ - Filesystem: /app/data                                  │
│ - Network: api.example.com                               │
│ - Environment: APP_*                                     │
│ - Memory: 512MB                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

Development Environment:
┌──────────────────────────────────────────────────────┐
│  Local Machine                                       │
│  ┌────────────────────────────────────────────────┐ │
│  │  Portalis CLI                                  │ │
│  │  └─→ Transpiler Agent                          │ │
│  │      └─→ Wassette Bridge (validation)          │ │
│  │          └─→ Local WASM Output ✓               │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘

CI/CD Pipeline:
┌──────────────────────────────────────────────────────┐
│  GitHub Actions / Jenkins                            │
│  ┌────────────────────────────────────────────────┐ │
│  │  1. Build with Wassette Features               │ │
│  │     cargo build --features wassette             │ │
│  │                                                 │ │
│  │  2. Run Integration Tests                      │ │
│  │     cargo test --features wassette             │ │
│  │                                                 │ │
│  │  3. Validate WASM Artifacts                    │ │
│  │     portalis validate-wasm output.wasm         │ │
│  │                                                 │ │
│  │  4. Package for Deployment                     │ │
│  │     docker build -t portalis:latest .          │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘

Production Environment:
┌──────────────────────────────────────────────────────┐
│  Kubernetes Cluster / DGX Cloud                      │
│  ┌────────────────────────────────────────────────┐ │
│  │  Portalis Services (Pods)                      │ │
│  │  ┌──────────────────────────────────────────┐ │ │
│  │  │  Transpiler Service                      │ │ │
│  │  │  - Wassette Bridge enabled               │ │ │
│  │  │  - Production permissions                │ │ │
│  │  │  - Monitoring enabled                    │ │ │
│  │  └──────────────────────────────────────────┘ │ │
│  │                                                 │ │
│  │  ┌──────────────────────────────────────────┐ │ │
│  │  │  WASM Component Registry                 │ │ │
│  │  │  - Validated WASM artifacts              │ │ │
│  │  │  - Version control                       │ │ │
│  │  │  - Metadata storage                      │ │ │
│  │  └──────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## 8. Error Handling Flow

```
┌─────────────────────────────────────────────────────────────┐
│              ERROR HANDLING IN WASSETTE BRIDGE              │
└─────────────────────────────────────────────────────────────┘

WASM Validation Request
        │
        ↓
    File Exists?
        │
    ┌───┴───┐
   No      Yes
    │       │
    ↓       ↓
  Error   Read File
    ↓       │
    │   Valid WASM?
    │       │
    │   ┌───┴───┐
    │  No      Yes
    │   │       │
    │   ↓       ↓
    │ Error  Wasmtime
    │        Validate
    │           │
    │       ┌───┴───┐
    │     Fail    Pass
    │       │       │
    │       ↓       ↓
    └──→ Error   Success
            ↓       ↓
      ┌─────────────────┐
      │ ValidationReport│
      │ - is_valid: bool│
      │ - errors: [...]  │
      │ - warnings: [...] │
      └─────────────────┘

Error Recovery Strategy:

┌──────────────────────────────────────────────────────┐
│ Level 1: Graceful Degradation                        │
│ - If wassette unavailable, continue without validation│
│ - Log warning, don't fail pipeline                   │
└──────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│ Level 2: Retry Logic                                 │
│ - Transient errors (network, resource): retry 3x     │
│ - Permanent errors (invalid WASM): fail immediately  │
└──────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│ Level 3: Logging & Monitoring                        │
│ - Log all errors to tracing                          │
│ - Emit metrics for failures                          │
│ - Alert on high error rates                          │
└──────────────────────────────────────────────────────┘
```

---

## 9. Build and Test Flow

```
┌─────────────────────────────────────────────────────────────┐
│              BUILD & TEST WORKFLOW                          │
└─────────────────────────────────────────────────────────────┘

Developer Workflow:

1. Code Changes
   │
   ↓
2. Local Build
   $ cargo build -p portalis-wassette-bridge
   │
   ↓
3. Unit Tests
   $ cargo test -p portalis-wassette-bridge
   │
   ↓
4. Integration Tests (with runtime feature)
   $ cargo test -p portalis-wassette-bridge --features runtime
   │
   ↓
5. E2E Tests (full pipeline)
   $ cargo test --workspace --features wassette
   │
   ↓
6. Manual Validation
   $ portalis validate-wasm test.wasm
   │
   ↓
7. Commit & Push
   $ git commit -m "feat: Add wassette validation"


CI/CD Workflow:

┌────────────────────────────────────────────────────┐
│  GitHub Actions Trigger (on PR)                    │
└────────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────┐
│  Job 1: Build                                      │
│  - cargo build --workspace                         │
│  - cargo build --workspace --features wassette     │
└────────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────┐
│  Job 2: Unit Tests                                 │
│  - cargo test --lib                                │
└────────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────┐
│  Job 3: Integration Tests                          │
│  - cargo test --test '*'                           │
│  - cargo test --features wassette                  │
└────────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────┐
│  Job 4: E2E Tests                                  │
│  - Full pipeline with WASM validation              │
│  - Performance benchmarks                          │
└────────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────┐
│  Job 5: Security Scan                              │
│  - cargo audit                                     │
│  - WASM security validation                        │
└────────────────────────────────────────────────────┘
                    ↓
            ┌───────┴────────┐
          Pass            Fail
            │                │
            ↓                ↓
        Merge PR      Block Merge
                      Notify Team
```

---

## Legend

```
Symbols:
  →   Flow direction
  ↓   Data/control flow
  ⭐  New integration point
  ✓   Validation passed
  ✗   Denied/blocked
  [ ] Component boundary
  ┌─┐ Container/module

Components:
  [Box]         Module or service
  (Circle)      Process or action
  <Diamond>     Decision point
  {Cylinder}    Data storage
```

---

**Note**: All diagrams use ASCII art for universal compatibility.
See the full architecture document for detailed specifications.
