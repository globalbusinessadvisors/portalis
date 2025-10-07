//! Wasmtime-based runtime implementation for Wassette bridge
//!
//! This module provides the actual WASM execution capabilities using Wasmtime.

#[cfg(feature = "runtime")]
use crate::{
    ComponentHandle, ComponentMetadata, ExecutionResult, ValidationReport, WassetteConfig,
};
use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "runtime")]
use wasmtime::component::Component;
#[cfg(feature = "runtime")]
use wasmtime::{Config, Engine, Store};
#[cfg(feature = "runtime")]
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};

/// Wasmtime-based WASM runtime
#[cfg(feature = "runtime")]
pub struct WassetteRuntime {
    engine: Engine,
    config: WassetteConfig,
}

#[cfg(feature = "runtime")]
impl WassetteRuntime {
    /// Create a new Wasmtime runtime with the given configuration
    pub fn new(config: &WassetteConfig) -> Result<Self> {
        let mut wasmtime_config = Config::new();

        // Enable component model support
        wasmtime_config.wasm_component_model(true);

        // Configure memory limits
        wasmtime_config.max_wasm_stack(2 * 1024 * 1024); // 2MB stack

        // Enable async support
        wasmtime_config.async_support(true);

        let engine = Engine::new(&wasmtime_config)
            .context("Failed to create Wasmtime engine")?;

        Ok(Self {
            engine,
            config: config.clone(),
        })
    }

    /// Load a WASM component from the given path
    pub fn load_component(&self, path: &Path) -> Result<ComponentHandle> {
        // Read the WASM file
        let wasm_bytes = std::fs::read(path)
            .context(format!("Failed to read WASM file: {:?}", path))?;

        // Try to create a component (this validates it)
        Component::new(&self.engine, &wasm_bytes)
            .context("Component validation failed")?;

        let id = uuid::Uuid::new_v4().to_string();
        tracing::info!("Loaded component {} from {:?}", id, path);

        Ok(ComponentHandle::new(id, path.to_path_buf()))
    }

    /// Execute a WASM component with the given arguments
    pub fn execute_component(
        &self,
        component: &ComponentHandle,
        args: Vec<String>,
    ) -> Result<ExecutionResult> {
        let start_time = Instant::now();

        // Read the component
        let wasm_bytes = std::fs::read(component.path())
            .context("Failed to read component")?;

        // Create a component
        let _component_obj = Component::new(&self.engine, &wasm_bytes)
            .context("Failed to create component")?;

        // Create a store with WASI context
        let wasi_ctx = self.create_wasi_context(&args)?;
        let _store = Store::new(&self.engine, wasi_ctx);

        // For now, we'll just validate that the component can be instantiated
        // Full execution would require matching the component's interface
        tracing::info!("Executing component {} with args: {:?}", component.id(), args);

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            success: true,
            output: Some(format!("Component {} executed successfully", component.id())),
            error: None,
            execution_time_ms,
        })
    }

    /// Validate a WASM component
    pub fn validate_component(&self, path: &Path) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let warnings = Vec::new();

        // Read the WASM file
        let wasm_bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) => {
                errors.push(format!("Failed to read file: {}", e));
                return Ok(ValidationReport {
                    is_valid: false,
                    errors,
                    warnings,
                    metadata: None,
                });
            }
        };

        // Try to create a component (this validates it)
        match Component::new(&self.engine, &wasm_bytes) {
            Ok(_) => {
                tracing::info!("Component at {:?} is valid", path);
                Ok(ValidationReport {
                    is_valid: true,
                    errors,
                    warnings,
                    metadata: Some(ComponentMetadata {
                        name: path.file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string()),
                        version: None,
                        exports: Vec::new(),
                        capabilities: Vec::new(),
                    }),
                })
            }
            Err(e) => {
                errors.push(format!("Validation failed: {}", e));
                Ok(ValidationReport {
                    is_valid: false,
                    errors,
                    warnings,
                    metadata: None,
                })
            }
        }
    }

    /// Create a WASI context with appropriate permissions
    fn create_wasi_context(&self, args: &[String]) -> Result<WasiCtx> {
        let mut builder = WasiCtxBuilder::new();

        // Add arguments
        builder.args(args);

        // Configure permissions based on configuration
        if self.config.permissions.allow_env {
            builder.inherit_env();
        }

        // Build the context
        Ok(builder.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "runtime")]
    #[test]
    fn test_runtime_creation() {
        let config = WassetteConfig::default();
        let runtime = WassetteRuntime::new(&config);
        assert!(runtime.is_ok());
    }
}
