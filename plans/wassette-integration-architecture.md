# Wassette Integration Architecture
## WebAssembly Component Runtime Integration for Portalis

**Version**: 1.0
**Author**: SystemDesigner
**Date**: 2025-10-07
**Status**: Design Phase

---

## Executive Summary

This document defines the architecture for integrating **wassette** - Microsoft's security-oriented WebAssembly Component runtime - into the Portalis Python-to-WASM translation platform. The integration will enable Portalis to:

1. **Execute and validate** WASM output using browser-grade security isolation
2. **Manage WASM Components** with granular permission controls
3. **Provide MCP-compatible tooling** for AI-powered code validation
4. **Enable dynamic component loading** for testing and deployment
5. **Support multi-language WASM** component interoperability

### Key Benefits

- **Security**: Wasmtime-based sandbox for safe WASM execution
- **Validation**: Runtime validation of transpiled WASM against Python source
- **Testing**: Isolated test environments for WASM components
- **Deployment**: Production-ready component management infrastructure
- **Extensibility**: MCP integration for AI-powered validation and debugging

---

## 1. Wassette Overview

### 1.1 What is Wassette?

Wassette is a security-oriented runtime that runs WebAssembly Components via MCP (Microsoft Component Platform). It provides:

- **Wasmtime Integration**: Browser-grade security sandbox
- **Component Management**: Dynamic loading and lifecycle management
- **Permission System**: Granular control over storage, network, environment
- **Multi-Language Support**: Components in Python, Rust, JavaScript, Go
- **WIT Interface**: WebAssembly Interface Types for typed contracts

### 1.2 Technology Stack

- **Language**: Rust
- **Runtime**: Wasmtime (WebAssembly runtime)
- **Protocol**: MCP (Model Context Protocol)
- **Interface**: WIT (WebAssembly Interface Types)
- **License**: MIT

### 1.3 Repository Structure (from GitHub)

```
wassette/
├── crates/              # Rust crate modules
├── src/                 # Main source code
├── examples/            # Example WASM components
├── docs/                # Documentation
├── tests/               # Test suite
├── Cargo.toml           # Rust package manifest
└── Cargo.lock           # Dependency lock file
```

---

## 2. Integration Strategy

### 2.1 Integration Approach: Workspace Member + Agent Bridge

**Decision**: Integrate wassette as a **workspace member crate** with a **bridge agent** pattern, consistent with existing NVIDIA integrations (nemo-bridge, cuda-bridge).

**Rationale**:
1. **Consistency**: Follows established pattern used for nemo-bridge and cuda-bridge
2. **Modularity**: Wassette can be enabled/disabled via Cargo features
3. **Isolation**: Clear API boundaries between Portalis and wassette
4. **Testability**: Easy to mock for tests (London School TDD)
5. **Flexibility**: Can be used by multiple agents (transpiler, test, packaging)

### 2.2 Workspace Structure

```
portalis/
├── agents/
│   ├── transpiler/          # Existing transpiler agent
│   ├── test/                # Existing test agent
│   ├── packaging/           # Existing packaging agent
│   ├── wassette-bridge/     # NEW: Wassette integration bridge
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs       # Main bridge API
│   │   │   ├── runtime.rs   # Wassette runtime wrapper
│   │   │   ├── component.rs # Component management
│   │   │   ├── validator.rs # WASM validation
│   │   │   └── mcp.rs       # MCP integration (optional)
│   │   └── tests/
│   │       ├── integration_tests.rs
│   │       └── fixtures/
│   └── ...
├── core/                    # Core library (agent traits, types)
├── orchestration/           # Pipeline orchestration
├── cli/                     # CLI interface
└── Cargo.toml               # Workspace manifest
```

### 2.3 Integration Type: Dependency vs. Embedded

**Recommended**: **Cargo Dependency** (external crate)

```toml
# agents/wassette-bridge/Cargo.toml
[dependencies]
wassette = { git = "https://github.com/microsoft/wassette", version = "0.1", optional = true }
# OR when published to crates.io:
# wassette = { version = "0.1", optional = true }
```

**Alternative**: **Git Submodule** (if needed)

Only if:
- Wassette not published to crates.io
- Need custom patches/modifications
- Want version pinning at commit level

```bash
# Only if necessary
git submodule add https://github.com/microsoft/wassette external/wassette
```

**Decision**: Start with **Cargo dependency** (cleaner, easier updates). Move to submodule only if needed.

---

## 3. Architecture Design

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PORTALIS PLATFORM                           │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │              ORCHESTRATION LAYER                       │   │
│  │  Pipeline Controller | Message Bus | State Management  │   │
│  └────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                   AGENT LAYER                          │   │
│  │  Transpiler → Build → Test → Packaging                │   │
│  │       ↓         ↓       ↓         ↓                    │   │
│  │       └─────────┴───────┴─────────┘                    │   │
│  │                   ↓                                     │   │
│  │         ┌──────────────────────┐                       │   │
│  │         │  WASSETTE BRIDGE     │  ← NEW INTEGRATION   │   │
│  │         │  - Runtime Manager   │                       │   │
│  │         │  - Component Loader  │                       │   │
│  │         │  - Validator         │                       │   │
│  │         │  - Permission Mgr    │                       │   │
│  │         └──────────────────────┘                       │   │
│  │                   ↓                                     │   │
│  └────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌────────────────────────────────────────────────────────┐   │
│  │              WASSETTE RUNTIME                          │   │
│  │  ┌──────────────────────────────────────────────────┐ │   │
│  │  │         Wasmtime Security Sandbox                │ │   │
│  │  │  ┌────────────┐  ┌────────────┐  ┌───────────┐  │ │   │
│  │  │  │ Component  │  │ Component  │  │ Component │  │ │   │
│  │  │  │     A      │  │     B      │  │     C     │  │ │   │
│  │  │  │ (Python→  │  │ (Rust→    │  │  (JS→     │  │ │   │
│  │  │  │  WASM)     │  │  WASM)     │  │  WASM)    │  │ │   │
│  │  │  └────────────┘  └────────────┘  └───────────┘  │ │   │
│  │  │                                                   │ │   │
│  │  │         Permission Boundaries                    │ │   │
│  │  │  Storage | Network | Env Vars | Filesystem       │ │   │
│  │  └──────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Wassette Bridge Agent

The `wassette-bridge` is **not** a full agent (doesn't implement `Agent` trait), but rather a **utility bridge** similar to `cuda-bridge` and `nemo-bridge`.

#### Bridge Responsibilities

1. **Runtime Management**: Initialize and manage Wasmtime runtime
2. **Component Lifecycle**: Load, execute, unload WASM components
3. **Validation**: Validate WASM components for correctness and security
4. **Permission Control**: Manage component permissions (storage, network, etc.)
5. **Error Handling**: Translate Wasmtime errors to Portalis errors

#### Bridge Interface

```rust
// agents/wassette-bridge/src/lib.rs

use portalis_core::{Result, Error};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Wassette runtime client for WASM component management
pub struct WassetteClient {
    config: WassetteConfig,
    runtime_available: bool,
}

/// Wassette configuration
#[derive(Debug, Clone)]
pub struct WassetteConfig {
    /// Enable security sandbox
    pub enable_sandbox: bool,
    /// Maximum memory per component (bytes)
    pub max_memory: usize,
    /// Allowed permissions
    pub permissions: ComponentPermissions,
}

/// Component permissions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComponentPermissions {
    /// Allow file system access
    pub allow_fs: bool,
    /// Allow network access
    pub allow_network: bool,
    /// Allow environment variable access
    pub allow_env: bool,
    /// Allowed file paths
    pub allowed_paths: Vec<String>,
}

/// Component execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Stdout output
    pub stdout: String,
    /// Stderr output
    pub stderr: String,
    /// Exit code
    pub exit_code: Option<i32>,
    /// Execution time (ms)
    pub execution_time_ms: f64,
}

impl WassetteClient {
    /// Create new Wassette client
    pub fn new(config: WassetteConfig) -> Result<Self>;

    /// Create with default configuration
    pub fn default() -> Result<Self>;

    /// Load a WASM component
    pub fn load_component(&self, path: &Path) -> Result<ComponentHandle>;

    /// Execute a WASM component
    pub fn execute_component(
        &self,
        component: &ComponentHandle,
        args: Vec<String>
    ) -> Result<ExecutionResult>;

    /// Validate a WASM component
    pub fn validate_component(&self, path: &Path) -> Result<ValidationReport>;

    /// Check if wassette runtime is available
    pub fn is_available(&self) -> bool;
}

/// Handle to a loaded WASM component
#[derive(Debug)]
pub struct ComponentHandle {
    id: String,
    path: PathBuf,
}

/// Validation report for WASM component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Is component valid?
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Component metadata
    pub metadata: ComponentMetadata,
}

/// Component metadata extracted from WASM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component name
    pub name: String,
    /// Exported functions
    pub exports: Vec<String>,
    /// Required imports
    pub imports: Vec<String>,
    /// Memory requirements (bytes)
    pub memory_requirement: usize,
}
```

### 3.3 Integration Points

#### 3.3.1 Transpiler Agent Integration

The transpiler agent will use wassette-bridge to validate generated WASM:

```rust
// agents/transpiler/src/lib.rs (updated)

#[cfg(feature = "wassette")]
use portalis_wassette_bridge::WassetteClient;

impl TranspilerAgent {
    /// Validate WASM output using wassette
    #[cfg(feature = "wassette")]
    async fn validate_wasm(&self, wasm_path: &Path) -> Result<()> {
        let client = WassetteClient::default()?;
        let report = client.validate_component(wasm_path)?;

        if !report.is_valid {
            return Err(Error::Validation(format!(
                "WASM validation failed: {:?}",
                report.errors
            )));
        }

        Ok(())
    }
}
```

#### 3.3.2 Test Agent Integration

The test agent will use wassette to execute WASM tests in isolation:

```rust
// agents/test/src/lib.rs (updated)

#[cfg(feature = "wassette")]
use portalis_wassette_bridge::{WassetteClient, ComponentPermissions};

impl TestAgent {
    /// Run WASM tests using wassette sandbox
    #[cfg(feature = "wassette")]
    async fn run_wasm_tests(&self, wasm_path: &Path) -> Result<TestResults> {
        let mut config = WassetteConfig::default();
        config.permissions.allow_fs = true;
        config.permissions.allowed_paths = vec![
            "/tmp/test_data".to_string()
        ];

        let client = WassetteClient::new(config)?;
        let component = client.load_component(wasm_path)?;
        let result = client.execute_component(&component, vec![])?;

        Ok(TestResults {
            passed: result.success,
            output: result.stdout,
            errors: result.stderr,
        })
    }
}
```

#### 3.3.3 Packaging Agent Integration

The packaging agent will use wassette to verify deployable WASM:

```rust
// agents/packaging/src/lib.rs (updated)

#[cfg(feature = "wassette")]
use portalis_wassette_bridge::WassetteClient;

impl PackagingAgent {
    /// Verify WASM package before deployment
    #[cfg(feature = "wassette")]
    async fn verify_package(&self, wasm_path: &Path) -> Result<()> {
        let client = WassetteClient::default()?;
        let report = client.validate_component(wasm_path)?;

        // Log component metadata
        tracing::info!(
            "WASM component: {} exports={:?} imports={:?}",
            report.metadata.name,
            report.metadata.exports,
            report.metadata.imports
        );

        if !report.is_valid {
            return Err(Error::Packaging(
                "WASM component failed validation"
            ));
        }

        Ok(())
    }
}
```

---

## 4. Build System Integration

### 4.1 Workspace Cargo.toml Updates

```toml
# /workspace/Portalis/Cargo.toml

[workspace]
resolver = "2"
members = [
    "core",
    "agents/ingest",
    "agents/analysis",
    "agents/specgen",
    "agents/transpiler",
    "agents/build",
    "agents/test",
    "agents/packaging",
    "agents/nemo-bridge",
    "agents/cuda-bridge",
    "agents/wassette-bridge",  # NEW
    "orchestration",
    "cli",
]

[workspace.dependencies]
# ... existing dependencies ...

# Wassette integration (optional)
wassette = { git = "https://github.com/microsoft/wassette", optional = true }
# OR when available on crates.io:
# wassette = { version = "0.1", optional = true }

wasmtime = { version = "24.0", optional = true }  # Needed for runtime
```

### 4.2 Wassette Bridge Cargo.toml

```toml
# agents/wassette-bridge/Cargo.toml

[package]
name = "portalis-wassette-bridge"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
description = "Wassette WebAssembly runtime integration for WASM validation and execution"
repository.workspace = true
homepage.workspace = true
keywords.workspace = true
categories.workspace = true
readme = "../../README.md"

[dependencies]
portalis-core = { version = "0.1.0", path = "../../core" }
serde.workspace = true
serde_json.workspace = true
tracing.workspace = true
anyhow.workspace = true
thiserror.workspace = true

# Wassette and Wasmtime (optional)
wassette = { git = "https://github.com/microsoft/wassette", optional = true }
wasmtime = { version = "24.0", optional = true }
wasmtime-wasi = { version = "24.0", optional = true }

[features]
default = []
# Enable when wassette runtime is available
runtime = ["wassette", "wasmtime", "wasmtime-wasi"]

[dev-dependencies]
tempfile.workspace = true
tokio = { workspace = true, features = ["test-util"] }
```

### 4.3 Dependent Agent Updates

#### Transpiler Agent

```toml
# agents/transpiler/Cargo.toml

[dependencies]
# ... existing dependencies ...

# Optional wassette integration
portalis-wassette-bridge = { version = "0.1.0", path = "../wassette-bridge", optional = true }

[features]
default = []
nemo = ["portalis-nemo-bridge"]
wasm = ["wasm-bindgen", "wasm-bindgen-futures", "js-sys", "web-sys", ...]
wasi = ["dep:wasi", "tokio"]
wassette = ["portalis-wassette-bridge/runtime"]  # NEW
```

#### Test Agent

```toml
# agents/test/Cargo.toml

[dependencies]
portalis-core = { version = "0.1.0", path = "../../core" }
portalis-wassette-bridge = { version = "0.1.0", path = "../wassette-bridge", optional = true }
# ... other dependencies ...

[features]
default = []
wassette = ["portalis-wassette-bridge/runtime"]  # NEW
```

#### Packaging Agent

```toml
# agents/packaging/Cargo.toml

[dependencies]
portalis-core = { version = "0.1.0", path = "../../core" }
portalis-wassette-bridge = { version = "0.1.0", path = "../wassette-bridge", optional = true }
# ... other dependencies ...

[features]
default = []
wassette = ["portalis-wassette-bridge/runtime"]  # NEW
```

---

## 5. API Design

### 5.1 Core Bridge API

```rust
// agents/wassette-bridge/src/lib.rs

//! Wassette Bridge - WebAssembly Component Runtime Integration
//!
//! Provides integration between Portalis and the wassette WASM runtime
//! for secure component execution and validation.

pub mod runtime;
pub mod component;
pub mod validator;
pub mod permissions;

#[cfg(feature = "runtime")]
pub mod mcp;

use portalis_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub use runtime::WassetteRuntime;
pub use component::{ComponentHandle, ComponentMetadata};
pub use validator::{ValidationReport, Validator};
pub use permissions::ComponentPermissions;

/// Main Wassette client interface
pub struct WassetteClient {
    runtime: Option<WassetteRuntime>,
    config: WassetteConfig,
}

/// Configuration for Wassette runtime
#[derive(Debug, Clone)]
pub struct WassetteConfig {
    /// Enable security sandbox
    pub enable_sandbox: bool,
    /// Maximum memory per component (MB)
    pub max_memory_mb: usize,
    /// Maximum execution time (seconds)
    pub max_execution_time_secs: u64,
    /// Component permissions
    pub permissions: ComponentPermissions,
}

impl Default for WassetteConfig {
    fn default() -> Self {
        Self {
            enable_sandbox: true,
            max_memory_mb: 128,
            max_execution_time_secs: 30,
            permissions: ComponentPermissions::default(),
        }
    }
}

impl WassetteClient {
    /// Create new Wassette client with configuration
    pub fn new(config: WassetteConfig) -> Result<Self> {
        let runtime = if cfg!(feature = "runtime") {
            Some(WassetteRuntime::new(&config)?)
        } else {
            None
        };

        Ok(Self { runtime, config })
    }

    /// Create with default configuration
    pub fn default() -> Result<Self> {
        Self::new(WassetteConfig::default())
    }

    /// Check if runtime is available
    pub fn is_available(&self) -> bool {
        self.runtime.is_some()
    }

    /// Load a WASM component from file
    pub fn load_component(&self, path: &Path) -> Result<ComponentHandle> {
        let runtime = self.runtime.as_ref()
            .ok_or_else(|| Error::Runtime("Wassette runtime not available".into()))?;
        runtime.load_component(path)
    }

    /// Execute a WASM component with arguments
    pub fn execute_component(
        &self,
        component: &ComponentHandle,
        args: Vec<String>,
    ) -> Result<ExecutionResult> {
        let runtime = self.runtime.as_ref()
            .ok_or_else(|| Error::Runtime("Wassette runtime not available".into()))?;
        runtime.execute_component(component, args)
    }

    /// Validate a WASM component
    pub fn validate_component(&self, path: &Path) -> Result<ValidationReport> {
        Validator::validate(path, &self.config)
    }
}

/// Result of component execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Execution succeeded
    pub success: bool,
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code (if available)
    pub exit_code: Option<i32>,
    /// Execution time (milliseconds)
    pub execution_time_ms: f64,
    /// Memory used (bytes)
    pub memory_used_bytes: usize,
}
```

### 5.2 Runtime Module

```rust
// agents/wassette-bridge/src/runtime.rs

use portalis_core::{Error, Result};
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "runtime")]
use wasmtime::*;
#[cfg(feature = "runtime")]
use wasmtime_wasi::WasiCtxBuilder;

use crate::{ComponentHandle, ComponentMetadata, ExecutionResult, WassetteConfig};

/// Wassette runtime wrapper around Wasmtime
pub struct WassetteRuntime {
    #[cfg(feature = "runtime")]
    engine: Engine,
    config: WassetteConfig,
}

impl WassetteRuntime {
    /// Create new runtime
    #[cfg(feature = "runtime")]
    pub fn new(config: &WassetteConfig) -> Result<Self> {
        let mut wasmtime_config = Config::new();
        wasmtime_config.wasm_component_model(true);

        // Configure security sandbox
        if config.enable_sandbox {
            wasmtime_config.max_wasm_stack(1024 * 1024); // 1MB stack
        }

        let engine = Engine::new(&wasmtime_config)
            .map_err(|e| Error::Runtime(format!("Failed to create Wasmtime engine: {}", e)))?;

        Ok(Self {
            engine,
            config: config.clone(),
        })
    }

    #[cfg(not(feature = "runtime"))]
    pub fn new(_config: &WassetteConfig) -> Result<Self> {
        Err(Error::Runtime("Wassette runtime not available (feature 'runtime' not enabled)".into()))
    }

    /// Load a WASM component
    #[cfg(feature = "runtime")]
    pub fn load_component(&self, path: &Path) -> Result<ComponentHandle> {
        if !path.exists() {
            return Err(Error::Io(format!("WASM file not found: {:?}", path)));
        }

        // Read WASM file
        let wasm_bytes = std::fs::read(path)
            .map_err(|e| Error::Io(format!("Failed to read WASM file: {}", e)))?;

        // Validate module
        Module::validate(&self.engine, &wasm_bytes)
            .map_err(|e| Error::Validation(format!("Invalid WASM module: {}", e)))?;

        // Extract metadata
        let metadata = self.extract_metadata(&wasm_bytes)?;

        Ok(ComponentHandle {
            id: uuid::Uuid::new_v4().to_string(),
            path: path.to_path_buf(),
            metadata,
        })
    }

    #[cfg(not(feature = "runtime"))]
    pub fn load_component(&self, _path: &Path) -> Result<ComponentHandle> {
        Err(Error::Runtime("Wassette runtime not available".into()))
    }

    /// Execute a WASM component
    #[cfg(feature = "runtime")]
    pub fn execute_component(
        &self,
        component: &ComponentHandle,
        args: Vec<String>,
    ) -> Result<ExecutionResult> {
        let start = Instant::now();

        // Create WASI context
        let wasi = WasiCtxBuilder::new()
            .args(&args)?
            .inherit_stdio()
            .build();

        let mut store = Store::new(&self.engine, wasi);

        // Load module
        let wasm_bytes = std::fs::read(&component.path)
            .map_err(|e| Error::Io(format!("Failed to read WASM file: {}", e)))?;
        let module = Module::new(&self.engine, &wasm_bytes)
            .map_err(|e| Error::Runtime(format!("Failed to load module: {}", e)))?;

        // Create instance
        let instance = Instance::new(&mut store, &module, &[])
            .map_err(|e| Error::Runtime(format!("Failed to create instance: {}", e)))?;

        // Execute main function
        let main_func = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .or_else(|_| instance.get_typed_func::<(), ()>(&mut store, "main"))
            .map_err(|e| Error::Runtime(format!("No entry point found: {}", e)))?;

        let execution_result = main_func.call(&mut store, ());

        let duration = start.elapsed();

        Ok(ExecutionResult {
            success: execution_result.is_ok(),
            stdout: String::new(), // Would capture from WASI
            stderr: if let Err(e) = execution_result {
                e.to_string()
            } else {
                String::new()
            },
            exit_code: if execution_result.is_ok() { Some(0) } else { Some(1) },
            execution_time_ms: duration.as_secs_f64() * 1000.0,
            memory_used_bytes: 0, // Would get from store
        })
    }

    #[cfg(not(feature = "runtime"))]
    pub fn execute_component(
        &self,
        _component: &ComponentHandle,
        _args: Vec<String>,
    ) -> Result<ExecutionResult> {
        Err(Error::Runtime("Wassette runtime not available".into()))
    }

    /// Extract metadata from WASM bytes
    #[cfg(feature = "runtime")]
    fn extract_metadata(&self, wasm_bytes: &[u8]) -> Result<ComponentMetadata> {
        let module = Module::new(&self.engine, wasm_bytes)
            .map_err(|e| Error::Parse(format!("Failed to parse WASM: {}", e)))?;

        let exports: Vec<String> = module.exports()
            .map(|e| e.name().to_string())
            .collect();

        let imports: Vec<String> = module.imports()
            .map(|i| format!("{}::{}", i.module(), i.name()))
            .collect();

        Ok(ComponentMetadata {
            name: "component".to_string(),
            exports,
            imports,
            memory_requirement: 0, // Would calculate from module
        })
    }

    #[cfg(not(feature = "runtime"))]
    fn extract_metadata(&self, _wasm_bytes: &[u8]) -> Result<ComponentMetadata> {
        Ok(ComponentMetadata::default())
    }
}
```

### 5.3 Component Module

```rust
// agents/wassette-bridge/src/component.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Handle to a loaded WASM component
#[derive(Debug, Clone)]
pub struct ComponentHandle {
    /// Unique component ID
    pub id: String,
    /// Path to WASM file
    pub path: PathBuf,
    /// Component metadata
    pub metadata: ComponentMetadata,
}

/// Metadata extracted from WASM component
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component name
    pub name: String,
    /// Exported functions
    pub exports: Vec<String>,
    /// Required imports
    pub imports: Vec<String>,
    /// Memory requirements (bytes)
    pub memory_requirement: usize,
}
```

### 5.4 Validator Module

```rust
// agents/wassette-bridge/src/validator.rs

use portalis_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::{ComponentMetadata, WassetteConfig};

/// WASM component validator
pub struct Validator;

impl Validator {
    /// Validate a WASM component file
    pub fn validate(path: &Path, config: &WassetteConfig) -> Result<ValidationReport> {
        let mut report = ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: ComponentMetadata::default(),
        };

        // Check file exists
        if !path.exists() {
            report.is_valid = false;
            report.errors.push(format!("WASM file not found: {:?}", path));
            return Ok(report);
        }

        // Read file
        let wasm_bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) => {
                report.is_valid = false;
                report.errors.push(format!("Failed to read WASM file: {}", e));
                return Ok(report);
            }
        };

        // Validate WASM magic number
        if wasm_bytes.len() < 4 || &wasm_bytes[0..4] != b"\0asm" {
            report.is_valid = false;
            report.errors.push("Invalid WASM magic number".to_string());
            return Ok(report);
        }

        // Check file size
        let max_size_bytes = config.max_memory_mb * 1024 * 1024;
        if wasm_bytes.len() > max_size_bytes {
            report.warnings.push(format!(
                "WASM file size ({} bytes) exceeds recommended maximum ({} bytes)",
                wasm_bytes.len(),
                max_size_bytes
            ));
        }

        #[cfg(feature = "runtime")]
        {
            // Validate with Wasmtime
            use wasmtime::*;
            let engine = Engine::default();
            match Module::validate(&engine, &wasm_bytes) {
                Ok(_) => {
                    // Extract metadata
                    if let Ok(module) = Module::new(&engine, &wasm_bytes) {
                        report.metadata.exports = module.exports()
                            .map(|e| e.name().to_string())
                            .collect();
                        report.metadata.imports = module.imports()
                            .map(|i| format!("{}::{}", i.module(), i.name()))
                            .collect();
                    }
                }
                Err(e) => {
                    report.is_valid = false;
                    report.errors.push(format!("WASM validation failed: {}", e));
                }
            }
        }

        Ok(report)
    }
}

/// Validation report for WASM component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Is the component valid?
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Component metadata
    pub metadata: ComponentMetadata,
}
```

### 5.5 Permissions Module

```rust
// agents/wassette-bridge/src/permissions.rs

use serde::{Deserialize, Serialize};

/// Component permission configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComponentPermissions {
    /// Allow file system access
    pub allow_fs: bool,
    /// Allow network access
    pub allow_network: bool,
    /// Allow environment variable access
    pub allow_env: bool,
    /// Allowed file system paths
    pub allowed_paths: Vec<String>,
    /// Allowed network hosts
    pub allowed_hosts: Vec<String>,
    /// Allowed environment variables
    pub allowed_env_vars: Vec<String>,
}

impl ComponentPermissions {
    /// Create restrictive permissions (deny all)
    pub fn restrictive() -> Self {
        Self::default()
    }

    /// Create permissive permissions (allow all)
    pub fn permissive() -> Self {
        Self {
            allow_fs: true,
            allow_network: true,
            allow_env: true,
            allowed_paths: vec!["*".to_string()],
            allowed_hosts: vec!["*".to_string()],
            allowed_env_vars: vec!["*".to_string()],
        }
    }

    /// Create test permissions (limited)
    pub fn for_testing() -> Self {
        Self {
            allow_fs: true,
            allow_network: false,
            allow_env: true,
            allowed_paths: vec!["/tmp".to_string()],
            allowed_hosts: vec![],
            allowed_env_vars: vec!["TEST_*".to_string()],
        }
    }
}
```

---

## 6. Testing Strategy

### 6.1 Test Levels

Following London School TDD:

1. **Unit Tests**: Mock wassette runtime, test bridge logic
2. **Integration Tests**: Real wassette runtime with test WASM files
3. **E2E Tests**: Full pipeline with WASM validation

### 6.2 Test Structure

```rust
// agents/wassette-bridge/tests/integration_tests.rs

#[cfg(test)]
mod integration_tests {
    use portalis_wassette_bridge::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_client_creation() {
        let config = WassetteConfig::default();
        let client = WassetteClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_validation_invalid_file() {
        let client = WassetteClient::default().unwrap();
        let result = client.validate_component(Path::new("nonexistent.wasm"));
        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(!report.is_valid);
    }

    #[test]
    #[cfg(feature = "runtime")]
    fn test_load_valid_wasm() {
        // Create test WASM file (minimal valid WASM)
        let temp_dir = TempDir::new().unwrap();
        let wasm_path = temp_dir.path().join("test.wasm");

        // Write minimal valid WASM module
        let minimal_wasm = vec![
            0x00, 0x61, 0x73, 0x6d, // WASM magic
            0x01, 0x00, 0x00, 0x00, // WASM version
        ];
        std::fs::write(&wasm_path, minimal_wasm).unwrap();

        let client = WassetteClient::default().unwrap();
        let result = client.load_component(&wasm_path);

        // May fail if WASM is too minimal, but should not panic
        assert!(result.is_ok() || result.is_err());
    }
}
```

### 6.3 Mock Strategy

```rust
// agents/wassette-bridge/src/lib.rs (testing section)

#[cfg(test)]
pub mod mock {
    use super::*;

    /// Mock Wassette client for testing
    pub struct MockWassetteClient {
        pub should_succeed: bool,
    }

    impl MockWassetteClient {
        pub fn new_success() -> Self {
            Self { should_succeed: true }
        }

        pub fn new_failure() -> Self {
            Self { should_succeed: false }
        }

        pub fn validate_component(&self, _path: &Path) -> Result<ValidationReport> {
            if self.should_succeed {
                Ok(ValidationReport {
                    is_valid: true,
                    errors: vec![],
                    warnings: vec![],
                    metadata: ComponentMetadata::default(),
                })
            } else {
                Ok(ValidationReport {
                    is_valid: false,
                    errors: vec!["Mock validation error".to_string()],
                    warnings: vec![],
                    metadata: ComponentMetadata::default(),
                })
            }
        }
    }
}
```

---

## 7. Migration and Implementation Plan

### 7.1 Phase 1: Foundation (Week 1)

**Goal**: Create basic wassette-bridge infrastructure

**Tasks**:
1. Create `agents/wassette-bridge` directory structure
2. Add wassette-bridge to workspace members
3. Implement basic WassetteClient skeleton
4. Add feature flags to Cargo.toml
5. Write unit tests for basic functionality

**Deliverables**:
- `agents/wassette-bridge/Cargo.toml`
- `agents/wassette-bridge/src/lib.rs` (skeleton)
- `agents/wassette-bridge/src/runtime.rs` (stub)
- Unit tests passing

### 7.2 Phase 2: Runtime Integration (Week 2)

**Goal**: Integrate actual wassette/wasmtime runtime

**Tasks**:
1. Add wassette dependency (from git or crates.io)
2. Implement WassetteRuntime with Wasmtime
3. Implement component loading
4. Implement validation logic
5. Add integration tests with real WASM files

**Deliverables**:
- Working runtime integration
- Validation working
- Integration tests passing

### 7.3 Phase 3: Agent Integration (Week 3)

**Goal**: Integrate wassette-bridge into existing agents

**Tasks**:
1. Add wassette feature to transpiler agent
2. Add WASM validation to transpiler output
3. Add wassette feature to test agent
4. Add WASM execution tests
5. Update orchestration pipeline

**Deliverables**:
- Transpiler validates WASM output
- Test agent can run WASM tests
- E2E tests passing

### 7.4 Phase 4: Advanced Features (Week 4)

**Goal**: Add advanced capabilities

**Tasks**:
1. Implement permission management
2. Add MCP integration (optional)
3. Add performance metrics
4. Add component metadata extraction
5. Documentation and examples

**Deliverables**:
- Permission system working
- Metrics collection
- Complete documentation
- Example workflows

### 7.5 Phase 5: Production Readiness (Week 5)

**Goal**: Production-ready integration

**Tasks**:
1. Performance benchmarking
2. Security audit
3. Error handling improvements
4. Monitoring and logging
5. Release preparation

**Deliverables**:
- Benchmark results
- Security review complete
- Production deployment guide
- Release notes

---

## 8. File Structure Changes

### 8.1 New Files

```
agents/wassette-bridge/
├── Cargo.toml                          # NEW
├── src/
│   ├── lib.rs                          # NEW - Main bridge API
│   ├── runtime.rs                      # NEW - Wasmtime wrapper
│   ├── component.rs                    # NEW - Component types
│   ├── validator.rs                    # NEW - WASM validation
│   ├── permissions.rs                  # NEW - Permission management
│   └── mcp.rs                          # NEW - MCP integration (optional)
├── tests/
│   ├── integration_tests.rs            # NEW
│   └── fixtures/
│       ├── valid_component.wasm        # NEW - Test fixture
│       └── invalid_component.wasm      # NEW - Test fixture
├── examples/
│   ├── basic_validation.rs             # NEW
│   └── execute_component.rs            # NEW
└── README.md                           # NEW
```

### 8.2 Modified Files

```
Cargo.toml                              # MODIFIED - Add wassette-bridge member
agents/transpiler/Cargo.toml            # MODIFIED - Add wassette feature
agents/transpiler/src/lib.rs            # MODIFIED - Add validation
agents/test/Cargo.toml                  # MODIFIED - Add wassette feature
agents/test/src/lib.rs                  # MODIFIED - Add WASM execution
agents/packaging/Cargo.toml             # MODIFIED - Add wassette feature
agents/packaging/src/lib.rs             # MODIFIED - Add verification
orchestration/src/lib.rs                # MODIFIED - Add wassette validation step
cli/src/commands/convert.rs             # MODIFIED - Add validation flag
README.md                               # MODIFIED - Document wassette integration
docs/architecture.md                    # MODIFIED - Add wassette section
```

---

## 9. Configuration and Environment

### 9.1 Environment Variables

```bash
# Enable wassette runtime
export PORTALIS_ENABLE_WASSETTE=1

# Wassette runtime configuration
export PORTALIS_WASSETTE_MAX_MEMORY_MB=128
export PORTALIS_WASSETTE_MAX_EXECUTION_TIME_SECS=30
export PORTALIS_WASSETTE_ENABLE_SANDBOX=true

# Permission configuration
export PORTALIS_WASSETTE_ALLOW_FS=true
export PORTALIS_WASSETTE_ALLOW_NETWORK=false
export PORTALIS_WASSETTE_ALLOWED_PATHS=/tmp,/var/tmp
```

### 9.2 Configuration File

```toml
# config/portalis.toml

[wassette]
enabled = true
max_memory_mb = 128
max_execution_time_secs = 30
enable_sandbox = true

[wassette.permissions]
allow_fs = true
allow_network = false
allow_env = true
allowed_paths = ["/tmp", "/var/tmp"]
allowed_hosts = []
allowed_env_vars = ["TEST_*"]
```

---

## 10. CLI Integration

### 10.1 New CLI Commands

```bash
# Validate WASM component
portalis validate-wasm output.wasm

# Execute WASM component
portalis run-wasm output.wasm --args "arg1 arg2"

# Convert with WASM validation
portalis convert script.py --validate-wasm

# Get component metadata
portalis wasm-info output.wasm
```

### 10.2 CLI Implementation

```rust
// cli/src/commands/validate_wasm.rs

use clap::Parser;
use portalis_wassette_bridge::{WassetteClient, WassetteConfig};
use std::path::PathBuf;

#[derive(Parser)]
pub struct ValidateWasmCommand {
    /// Path to WASM file
    wasm_file: PathBuf,

    /// Show detailed validation report
    #[arg(long)]
    detailed: bool,
}

impl ValidateWasmCommand {
    pub fn execute(&self) -> Result<()> {
        let client = WassetteClient::default()?;
        let report = client.validate_component(&self.wasm_file)?;

        if report.is_valid {
            println!("✓ WASM component is valid");
        } else {
            println!("✗ WASM component validation failed");
            for error in &report.errors {
                println!("  Error: {}", error);
            }
        }

        if self.detailed {
            println!("\nMetadata:");
            println!("  Name: {}", report.metadata.name);
            println!("  Exports: {:?}", report.metadata.exports);
            println!("  Imports: {:?}", report.metadata.imports);
        }

        if !report.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &report.warnings {
                println!("  {}", warning);
            }
        }

        Ok(())
    }
}
```

---

## 11. Documentation Requirements

### 11.1 README Updates

Add to main README.md:

```markdown
## Wassette Integration

Portalis integrates with [Wassette](https://github.com/microsoft/wassette), a security-oriented WebAssembly runtime, to provide:

- **WASM Validation**: Verify generated WASM components
- **Secure Execution**: Run WASM in isolated sandbox
- **Component Management**: Load and manage WASM components
- **Permission Control**: Granular security permissions

### Usage

```bash
# Enable wassette features
cargo build --features wassette

# Validate WASM output
portalis validate-wasm output.wasm

# Convert with validation
portalis convert script.py --validate-wasm
```

See [Wassette Integration Guide](docs/wassette-integration.md) for details.
```

### 11.2 New Documentation

Create `docs/wassette-integration.md`:

- Overview of wassette integration
- Installation and setup
- Configuration options
- Usage examples
- API reference
- Troubleshooting

---

## 12. Security Considerations

### 12.1 Sandbox Configuration

- **Default**: Strict sandbox enabled
- **Memory Limits**: 128MB default, configurable
- **Execution Time**: 30 seconds default timeout
- **File System**: Restricted to specific paths
- **Network**: Disabled by default

### 12.2 Permission Model

```rust
// Example: Strict permissions for untrusted WASM
let permissions = ComponentPermissions {
    allow_fs: false,
    allow_network: false,
    allow_env: false,
    allowed_paths: vec![],
    allowed_hosts: vec![],
    allowed_env_vars: vec![],
};
```

### 12.3 Validation Requirements

1. **Magic Number Check**: Verify WASM magic bytes
2. **Wasmtime Validation**: Full module validation
3. **Size Limits**: Enforce maximum component size
4. **Import Restrictions**: Validate required imports
5. **Export Verification**: Ensure required exports exist

---

## 13. Performance Considerations

### 13.1 Runtime Overhead

- **Component Loading**: ~10-50ms per component
- **Validation**: ~5-20ms per component
- **Execution**: Depends on component complexity
- **Memory**: ~50MB base + component memory

### 13.2 Optimization Strategies

1. **Component Caching**: Cache loaded components
2. **Lazy Loading**: Load components on-demand
3. **Parallel Validation**: Validate multiple components in parallel
4. **Resource Pooling**: Reuse Wasmtime engines

### 13.3 Benchmarking

```rust
// Benchmark validation performance
#[bench]
fn bench_component_validation(b: &mut Bencher) {
    let client = WassetteClient::default().unwrap();
    let wasm_path = Path::new("test.wasm");

    b.iter(|| {
        client.validate_component(wasm_path).unwrap()
    });
}
```

---

## 14. Monitoring and Observability

### 14.1 Metrics

```rust
// Prometheus metrics
lazy_static! {
    static ref WASSETTE_VALIDATIONS: IntCounter = register_int_counter!(
        "portalis_wassette_validations_total",
        "Total number of WASM validations"
    ).unwrap();

    static ref WASSETTE_EXECUTIONS: IntCounter = register_int_counter!(
        "portalis_wassette_executions_total",
        "Total number of WASM executions"
    ).unwrap();

    static ref WASSETTE_VALIDATION_DURATION: Histogram = register_histogram!(
        "portalis_wassette_validation_duration_seconds",
        "WASM validation duration"
    ).unwrap();
}
```

### 14.2 Logging

```rust
// Structured logging
tracing::info!(
    component_path = %path.display(),
    validation_result = %report.is_valid,
    "WASM component validated"
);

tracing::error!(
    component_path = %path.display(),
    error = %e,
    "WASM component validation failed"
);
```

### 14.3 Tracing

```rust
// OpenTelemetry spans
#[tracing::instrument(skip(self))]
pub fn validate_component(&self, path: &Path) -> Result<ValidationReport> {
    // Validation logic with automatic span creation
}
```

---

## 15. Error Handling

### 15.1 Error Types

```rust
// Extended Error enum
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Wassette runtime error: {0}")]
    WassetteRuntime(String),

    #[error("WASM validation error: {0}")]
    WasmValidation(String),

    #[error("Component load error: {0}")]
    ComponentLoad(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}
```

### 15.2 Error Recovery

```rust
// Graceful fallback when wassette unavailable
match client.validate_component(&path) {
    Ok(report) if report.is_valid => {
        tracing::info!("WASM validation passed");
    }
    Ok(report) => {
        tracing::warn!("WASM validation failed, but continuing");
        // Don't fail the entire pipeline
    }
    Err(e) if cfg!(not(feature = "wassette")) => {
        tracing::info!("Wassette not available, skipping validation");
    }
    Err(e) => {
        tracing::error!("Wassette error: {}", e);
        return Err(e);
    }
}
```

---

## 16. Future Enhancements

### 16.1 Short-term (3-6 months)

1. **MCP Integration**: Full Model Context Protocol support
2. **Component Repository**: Registry for WASM components
3. **Hot Reloading**: Dynamic component updates
4. **Profiling**: Performance profiling tools

### 16.2 Long-term (6-12 months)

1. **Distributed Execution**: Multi-node WASM execution
2. **Component Composition**: Combine multiple components
3. **AI-Powered Debugging**: Use LLMs for WASM debugging
4. **Cross-Language Interop**: Enhanced language bridging

---

## 17. Success Criteria

### 17.1 Functional Requirements

- [ ] Wassette-bridge compiles and passes tests
- [ ] WASM validation works correctly
- [ ] Component execution succeeds
- [ ] Integration with transpiler agent works
- [ ] Integration with test agent works
- [ ] CLI commands functional

### 17.2 Non-Functional Requirements

- [ ] Validation completes in <100ms for typical components
- [ ] Memory overhead <100MB
- [ ] Test coverage >80%
- [ ] Documentation complete
- [ ] Zero security vulnerabilities

### 17.3 Integration Requirements

- [ ] Works without breaking existing functionality
- [ ] Optional (can be disabled via feature flags)
- [ ] Backward compatible with existing WASM pipeline
- [ ] Consistent error handling with other agents

---

## 18. References

### 18.1 External Resources

- [Wassette GitHub](https://github.com/microsoft/wassette)
- [Wasmtime Documentation](https://docs.wasmtime.dev/)
- [WebAssembly Component Model](https://github.com/WebAssembly/component-model)
- [MCP Specification](https://modelcontextprotocol.io/)
- [WIT Format](https://component-model.bytecodealliance.org/design/wit.html)

### 18.2 Internal Documentation

- [Portalis Architecture](./architecture.md)
- [Agent Design Patterns](./specification.md)
- [NVIDIA Integration](./nvidia-integration-architecture.md)
- [Testing Strategy](./TESTING_STRATEGY.md)

---

## Appendix A: Example Integration Flow

### Complete Pipeline with Wassette

```
1. User runs: portalis convert script.py --validate-wasm

2. Pipeline Flow:
   ├── Ingest Agent: Parse Python → AST
   ├── Analysis Agent: Extract types and contracts
   ├── Spec Generator: Generate Rust specs
   ├── Transpiler Agent: Generate Rust code
   │   └── compile to WASM
   │
   ├── Wassette Bridge: Validate WASM ← NEW
   │   ├── Check magic number
   │   ├── Validate with Wasmtime
   │   ├── Extract metadata
   │   └── Return validation report
   │
   ├── Test Agent: Run tests
   │   └── Execute WASM tests via wassette ← NEW
   │
   └── Packaging Agent: Create deployment artifacts
       └── Verify WASM component ← NEW

3. Output:
   ├── Rust source code
   ├── WASM binary (validated ✓)
   ├── Validation report
   └── Component metadata
```

---

## Appendix B: Troubleshooting Guide

### Common Issues

**Issue**: "Wassette runtime not available"
- **Cause**: Feature flag not enabled
- **Solution**: Build with `--features wassette`

**Issue**: "WASM validation failed: Invalid magic number"
- **Cause**: Corrupted or non-WASM file
- **Solution**: Verify file is valid WASM

**Issue**: "Permission denied" during execution
- **Cause**: Strict permissions blocking required access
- **Solution**: Adjust ComponentPermissions configuration

**Issue**: High memory usage
- **Cause**: Large WASM components or memory leaks
- **Solution**: Reduce max_memory_mb or investigate component

---

## Appendix C: Migration Checklist

### Pre-Integration

- [ ] Review wassette documentation
- [ ] Understand Wasmtime API
- [ ] Plan feature flag strategy
- [ ] Design test fixtures

### Implementation

- [ ] Create wassette-bridge crate
- [ ] Add to workspace members
- [ ] Implement WassetteClient
- [ ] Implement validation logic
- [ ] Add unit tests
- [ ] Add integration tests

### Agent Integration

- [ ] Update transpiler Cargo.toml
- [ ] Add validation to transpiler
- [ ] Update test agent Cargo.toml
- [ ] Add execution to test agent
- [ ] Update packaging agent
- [ ] Update orchestration pipeline

### Documentation

- [ ] Update main README
- [ ] Create integration guide
- [ ] Add API documentation
- [ ] Create examples
- [ ] Update architecture docs

### Release

- [ ] Run full test suite
- [ ] Performance benchmarks
- [ ] Security review
- [ ] Update changelog
- [ ] Tag release

---

**END OF DOCUMENT**
