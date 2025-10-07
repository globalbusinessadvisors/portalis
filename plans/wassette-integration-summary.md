# Wassette Integration Summary
## Quick Reference Guide

**Date**: 2025-10-07
**Status**: Design Complete - Ready for Implementation

---

## What is Wassette?

Wassette is Microsoft's security-oriented WebAssembly Component runtime built on Wasmtime. It provides:
- Browser-grade security sandbox for WASM execution
- Granular permission management (filesystem, network, env vars)
- MCP (Model Context Protocol) integration
- Multi-language WASM component support

**GitHub**: https://github.com/microsoft/wassette

---

## Integration Approach

### Strategy: Workspace Member + Bridge Agent Pattern

Following the same pattern as `nemo-bridge` and `cuda-bridge`:

```
portalis/
└── agents/
    ├── nemo-bridge/        # Existing - NVIDIA NeMo integration
    ├── cuda-bridge/        # Existing - CUDA acceleration
    └── wassette-bridge/    # NEW - Wassette WASM runtime
```

### Integration Type: Cargo Dependency

```toml
# agents/wassette-bridge/Cargo.toml
[dependencies]
wassette = { git = "https://github.com/microsoft/wassette", optional = true }
wasmtime = { version = "24.0", optional = true }
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 PORTALIS AGENTS                         │
│                                                         │
│  Transpiler → Build → Test → Packaging                │
│       ↓         ↓       ↓         ↓                    │
│       └─────────┴───────┴─────────┘                    │
│                   ↓                                     │
│         ┌──────────────────────┐                       │
│         │  WASSETTE BRIDGE     │  ← NEW               │
│         │  - Validate WASM     │                       │
│         │  - Execute WASM      │                       │
│         │  - Manage Components │                       │
│         └──────────────────────┘                       │
│                   ↓                                     │
│         ┌──────────────────────┐                       │
│         │  WASMTIME SANDBOX    │                       │
│         │  Security Isolation  │                       │
│         └──────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

---

## Key Integration Points

### 1. Transpiler Agent
**Purpose**: Validate generated WASM output

```rust
// agents/transpiler/src/lib.rs
#[cfg(feature = "wassette")]
use portalis_wassette_bridge::WassetteClient;

async fn validate_wasm(&self, wasm_path: &Path) -> Result<()> {
    let client = WassetteClient::default()?;
    let report = client.validate_component(wasm_path)?;
    if !report.is_valid {
        return Err(Error::Validation(format!("WASM invalid: {:?}", report.errors)));
    }
    Ok(())
}
```

### 2. Test Agent
**Purpose**: Execute WASM tests in isolated sandbox

```rust
// agents/test/src/lib.rs
#[cfg(feature = "wassette")]
async fn run_wasm_tests(&self, wasm_path: &Path) -> Result<TestResults> {
    let client = WassetteClient::new(WassetteConfig {
        permissions: ComponentPermissions::for_testing(),
        ..Default::default()
    })?;
    let component = client.load_component(wasm_path)?;
    let result = client.execute_component(&component, vec![])?;
    Ok(TestResults::from(result))
}
```

### 3. Packaging Agent
**Purpose**: Verify WASM before deployment

```rust
// agents/packaging/src/lib.rs
#[cfg(feature = "wassette")]
async fn verify_package(&self, wasm_path: &Path) -> Result<()> {
    let client = WassetteClient::default()?;
    let report = client.validate_component(wasm_path)?;
    tracing::info!("Component exports: {:?}", report.metadata.exports);
    Ok(())
}
```

---

## Directory Structure

### New Files

```
agents/wassette-bridge/
├── Cargo.toml                    # Package manifest
├── src/
│   ├── lib.rs                    # Main API
│   ├── runtime.rs                # Wasmtime wrapper
│   ├── component.rs              # Component types
│   ├── validator.rs              # WASM validation
│   ├── permissions.rs            # Permission management
│   └── mcp.rs                    # MCP integration (future)
├── tests/
│   ├── integration_tests.rs
│   └── fixtures/
│       ├── valid.wasm
│       └── invalid.wasm
└── README.md
```

### Modified Files

```
Cargo.toml                        # Add wassette-bridge member
agents/transpiler/Cargo.toml      # Add wassette feature
agents/transpiler/src/lib.rs      # Add WASM validation
agents/test/Cargo.toml            # Add wassette feature
agents/test/src/lib.rs            # Add WASM execution
agents/packaging/Cargo.toml       # Add wassette feature
agents/packaging/src/lib.rs       # Add WASM verification
```

---

## Core API Design

### WassetteClient

```rust
pub struct WassetteClient {
    runtime: Option<WassetteRuntime>,
    config: WassetteConfig,
}

impl WassetteClient {
    // Create client with configuration
    pub fn new(config: WassetteConfig) -> Result<Self>;

    // Create with defaults
    pub fn default() -> Result<Self>;

    // Load WASM component
    pub fn load_component(&self, path: &Path) -> Result<ComponentHandle>;

    // Execute component
    pub fn execute_component(
        &self,
        component: &ComponentHandle,
        args: Vec<String>
    ) -> Result<ExecutionResult>;

    // Validate component
    pub fn validate_component(&self, path: &Path) -> Result<ValidationReport>;

    // Check availability
    pub fn is_available(&self) -> bool;
}
```

### Configuration

```rust
pub struct WassetteConfig {
    pub enable_sandbox: bool,              // Default: true
    pub max_memory_mb: usize,              // Default: 128
    pub max_execution_time_secs: u64,      // Default: 30
    pub permissions: ComponentPermissions,
}

pub struct ComponentPermissions {
    pub allow_fs: bool,
    pub allow_network: bool,
    pub allow_env: bool,
    pub allowed_paths: Vec<String>,
    pub allowed_hosts: Vec<String>,
    pub allowed_env_vars: Vec<String>,
}
```

### Results

```rust
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub metadata: ComponentMetadata,
}

pub struct ExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub execution_time_ms: f64,
    pub memory_used_bytes: usize,
}
```

---

## Build System Changes

### Workspace Cargo.toml

```toml
[workspace]
members = [
    # ... existing members ...
    "agents/wassette-bridge",  # ADD THIS
]

[workspace.dependencies]
# ... existing dependencies ...
wassette = { git = "https://github.com/microsoft/wassette", optional = true }
wasmtime = { version = "24.0", optional = true }
```

### Wassette Bridge Cargo.toml

```toml
[package]
name = "portalis-wassette-bridge"
version.workspace = true
edition.workspace = true
# ... workspace settings ...

[dependencies]
portalis-core = { version = "0.1.0", path = "../../core" }
serde.workspace = true
serde_json.workspace = true
tracing.workspace = true
anyhow.workspace = true
thiserror.workspace = true

# Wassette runtime (optional)
wassette = { git = "https://github.com/microsoft/wassette", optional = true }
wasmtime = { version = "24.0", optional = true }
wasmtime-wasi = { version = "24.0", optional = true }

[features]
default = []
runtime = ["wassette", "wasmtime", "wasmtime-wasi"]
```

### Transpiler Agent Update

```toml
# agents/transpiler/Cargo.toml
[dependencies]
portalis-wassette-bridge = { version = "0.1.0", path = "../wassette-bridge", optional = true }

[features]
wassette = ["portalis-wassette-bridge/runtime"]
```

---

## CLI Integration

### New Commands

```bash
# Validate WASM component
portalis validate-wasm output.wasm

# Execute WASM component
portalis run-wasm output.wasm --args "arg1 arg2"

# Convert with validation
portalis convert script.py --validate-wasm

# Component metadata
portalis wasm-info output.wasm
```

### Usage Examples

```bash
# Enable wassette features during build
cargo build --features wassette

# Convert Python to WASM with validation
portalis convert calculator.py --validate-wasm

# Run WASM in sandbox
portalis run-wasm calculator.wasm --args "2 + 2"
```

---

## Testing Strategy

### Test Levels

1. **Unit Tests**: Mock wassette runtime
2. **Integration Tests**: Real WASM files with Wasmtime
3. **E2E Tests**: Full pipeline with validation

### Example Test

```rust
#[test]
fn test_validate_valid_wasm() {
    let client = WassetteClient::default().unwrap();
    let report = client.validate_component(Path::new("test.wasm")).unwrap();
    assert!(report.is_valid);
}

#[test]
#[cfg(feature = "runtime")]
fn test_execute_wasm() {
    let client = WassetteClient::default().unwrap();
    let component = client.load_component(Path::new("test.wasm")).unwrap();
    let result = client.execute_component(&component, vec![]).unwrap();
    assert!(result.success);
}
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1)
- Create `agents/wassette-bridge` directory
- Add to workspace members
- Implement skeleton API
- Write unit tests

### Phase 2: Runtime Integration (Week 2)
- Add wassette/wasmtime dependencies
- Implement WassetteRuntime
- Implement validation logic
- Add integration tests

### Phase 3: Agent Integration (Week 3)
- Integrate with transpiler agent
- Integrate with test agent
- Integrate with packaging agent
- Update orchestration pipeline

### Phase 4: Advanced Features (Week 4)
- Permission management
- MCP integration (optional)
- Performance metrics
- Documentation

### Phase 5: Production (Week 5)
- Performance benchmarking
- Security audit
- Monitoring/logging
- Release preparation

---

## Feature Flags

### Compile-time Features

```toml
# Minimal build (no wassette)
cargo build

# With wassette runtime
cargo build --features wassette

# Transpiler with wassette validation
cd agents/transpiler
cargo build --features wassette
```

### Runtime Configuration

```bash
# Enable via environment
export PORTALIS_ENABLE_WASSETTE=1
export PORTALIS_WASSETTE_MAX_MEMORY_MB=256

# Or via config file
# config/portalis.toml
[wassette]
enabled = true
max_memory_mb = 256
enable_sandbox = true
```

---

## Security Considerations

### Default Security Posture

```rust
// Restrictive by default
ComponentPermissions {
    allow_fs: false,
    allow_network: false,
    allow_env: false,
    allowed_paths: vec![],
    allowed_hosts: vec![],
    allowed_env_vars: vec![],
}
```

### Testing Permissions

```rust
// Relaxed for testing
ComponentPermissions::for_testing() {
    allow_fs: true,        // Limited paths only
    allow_network: false,
    allow_env: true,       // TEST_* vars only
    allowed_paths: vec!["/tmp"],
}
```

### Production Permissions

```rust
// Configured per deployment
ComponentPermissions {
    allow_fs: true,
    allow_network: true,
    allow_env: true,
    allowed_paths: vec!["/app/data"],
    allowed_hosts: vec!["api.example.com"],
    allowed_env_vars: vec!["APP_*"],
}
```

---

## Performance Metrics

### Expected Overhead

- **Validation**: 5-20ms per component
- **Loading**: 10-50ms per component
- **Execution**: Variable (depends on component)
- **Memory**: ~50MB base + component memory

### Optimization Strategies

1. Component caching
2. Lazy loading
3. Parallel validation
4. Runtime pooling

---

## Monitoring

### Prometheus Metrics

```rust
portalis_wassette_validations_total      // Total validations
portalis_wassette_executions_total       // Total executions
portalis_wassette_validation_duration_seconds  // Validation time
portalis_wassette_execution_duration_seconds   // Execution time
```

### Logging

```rust
tracing::info!(
    component = %path.display(),
    valid = %report.is_valid,
    "WASM validation complete"
);
```

---

## Success Criteria

### Functional
- [ ] Wassette-bridge compiles successfully
- [ ] WASM validation works
- [ ] Component execution succeeds
- [ ] All agents integrated
- [ ] CLI commands functional

### Non-Functional
- [ ] Validation <100ms for typical WASM
- [ ] Memory overhead <100MB
- [ ] Test coverage >80%
- [ ] Documentation complete
- [ ] Zero security vulnerabilities

### Integration
- [ ] No breaking changes to existing code
- [ ] Optional (feature flags work)
- [ ] Backward compatible
- [ ] Consistent error handling

---

## Quick Start for Developers

### 1. Create Bridge Crate

```bash
cd /workspace/Portalis
mkdir -p agents/wassette-bridge/src
cd agents/wassette-bridge
```

### 2. Create Cargo.toml

```toml
[package]
name = "portalis-wassette-bridge"
version = "0.1.0"
edition = "2021"

[dependencies]
portalis-core = { path = "../../core" }
anyhow = "1.0"
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
tracing = "0.1"

[features]
default = []
runtime = ["wasmtime"]
```

### 3. Create src/lib.rs

```rust
//! Wassette Bridge - WASM Component Runtime Integration

pub mod runtime;
pub mod component;
pub mod validator;

use portalis_core::Result;
use std::path::Path;

pub struct WassetteClient;

impl WassetteClient {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub fn validate_component(&self, path: &Path) -> Result<ValidationReport> {
        // Implementation here
    }
}

pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
}
```

### 4. Add to Workspace

```toml
# /workspace/Portalis/Cargo.toml
[workspace]
members = [
    # ... existing ...
    "agents/wassette-bridge",
]
```

### 5. Build and Test

```bash
cd /workspace/Portalis
cargo build -p portalis-wassette-bridge
cargo test -p portalis-wassette-bridge
```

---

## Troubleshooting

### "Wassette runtime not available"
**Solution**: Build with `--features wassette`

### "WASM validation failed"
**Solution**: Check WASM file is valid with `wasm-validate output.wasm`

### "Permission denied"
**Solution**: Adjust `ComponentPermissions` configuration

### High memory usage
**Solution**: Reduce `max_memory_mb` in config

---

## Resources

### Documentation
- Full Architecture: [wassette-integration-architecture.md](./wassette-integration-architecture.md)
- Portalis Architecture: [architecture.md](./architecture.md)
- Agent Patterns: [specification.md](./specification.md)

### External Links
- [Wassette GitHub](https://github.com/microsoft/wassette)
- [Wasmtime Docs](https://docs.wasmtime.dev/)
- [WebAssembly Component Model](https://github.com/WebAssembly/component-model)
- [MCP Specification](https://modelcontextprotocol.io/)

---

## Next Steps

1. Review architecture document
2. Create `agents/wassette-bridge` crate
3. Implement core API
4. Add tests
5. Integrate with agents
6. Documentation and release

---

**Status**: Ready for Implementation
**Owner**: Development Team
**Priority**: High
**Timeline**: 5 weeks
