//! Wassette WebAssembly Runtime Bridge for Portalis
//!
//! This module provides integration between Portalis and the Wassette WASM runtime,
//! enabling secure execution of WebAssembly components with fine-grained permissions.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "runtime")]
mod runtime;

#[cfg(feature = "runtime")]
pub use runtime::WassetteRuntime;

/// Configuration for the Wassette client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WassetteConfig {
    /// Enable sandboxing (default: true)
    pub enable_sandbox: bool,
    /// Maximum memory in MB (default: 128)
    pub max_memory_mb: usize,
    /// Maximum execution time in seconds (default: 30)
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

/// Permission configuration for WASM components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPermissions {
    /// Allow filesystem access
    pub allow_fs: bool,
    /// Allow network access
    pub allow_network: bool,
    /// Allow environment variable access
    pub allow_env: bool,
    /// Allowed filesystem paths
    pub allowed_paths: Vec<String>,
    /// Allowed network hosts
    pub allowed_hosts: Vec<String>,
    /// Allowed environment variables
    pub allowed_env_vars: Vec<String>,
}

impl Default for ComponentPermissions {
    fn default() -> Self {
        Self {
            allow_fs: false,
            allow_network: false,
            allow_env: false,
            allowed_paths: Vec::new(),
            allowed_hosts: Vec::new(),
            allowed_env_vars: Vec::new(),
        }
    }
}

/// Validation report for WASM components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Whether the component is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Component metadata
    pub metadata: Option<ComponentMetadata>,
}

/// Metadata extracted from a WASM component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component name
    pub name: Option<String>,
    /// Component version
    pub version: Option<String>,
    /// Exported functions
    pub exports: Vec<String>,
    /// Required capabilities
    pub capabilities: Vec<String>,
}

/// Handle to a loaded WASM component
#[derive(Debug)]
pub struct ComponentHandle {
    id: String,
    path: std::path::PathBuf,
}

impl ComponentHandle {
    pub fn new(id: String, path: std::path::PathBuf) -> Self {
        Self { id, path }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Result of executing a WASM component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Output from the component
    pub output: Option<String>,
    /// Error message if execution failed
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Main Wassette client interface
pub struct WassetteClient {
    #[cfg(feature = "runtime")]
    runtime: Option<WassetteRuntime>,
    config: WassetteConfig,
}

impl WassetteClient {
    /// Create a new Wassette client with the given configuration
    pub fn new(config: WassetteConfig) -> Result<Self> {
        #[cfg(feature = "runtime")]
        {
            let runtime = Some(WassetteRuntime::new(&config)?);
            Ok(Self { runtime, config })
        }
        #[cfg(not(feature = "runtime"))]
        {
            Ok(Self { config })
        }
    }

    /// Create a new Wassette client with default configuration
    pub fn default() -> Result<Self> {
        Self::new(WassetteConfig::default())
    }

    /// Load a WASM component from the given path
    pub fn load_component(&self, path: &Path) -> Result<ComponentHandle> {
        #[cfg(feature = "runtime")]
        {
            if let Some(runtime) = &self.runtime {
                return runtime.load_component(path);
            }
        }

        // Mock implementation when runtime is not available
        let id = uuid::Uuid::new_v4().to_string();
        tracing::info!("Mock: Loading component from {:?}", path);
        Ok(ComponentHandle::new(id, path.to_path_buf()))
    }

    /// Execute a WASM component with the given arguments
    pub fn execute_component(
        &self,
        component: &ComponentHandle,
        args: Vec<String>,
    ) -> Result<ExecutionResult> {
        #[cfg(feature = "runtime")]
        {
            if let Some(runtime) = &self.runtime {
                return runtime.execute_component(component, args);
            }
        }

        // Mock implementation when runtime is not available
        tracing::info!("Mock: Executing component {} with args: {:?}", component.id(), args);
        Ok(ExecutionResult {
            success: true,
            output: Some("Mock execution result".to_string()),
            error: None,
            execution_time_ms: 0,
        })
    }

    /// Validate a WASM component
    pub fn validate_component(&self, path: &Path) -> Result<ValidationReport> {
        #[cfg(feature = "runtime")]
        {
            if let Some(runtime) = &self.runtime {
                return runtime.validate_component(path);
            }
        }

        // Mock implementation when runtime is not available
        tracing::info!("Mock: Validating component at {:?}", path);
        Ok(ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: Some(ComponentMetadata {
                name: Some("mock-component".to_string()),
                version: Some("0.1.0".to_string()),
                exports: vec!["main".to_string()],
                capabilities: Vec::new(),
            }),
        })
    }

    /// Check if the Wassette runtime is available
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "runtime")]
        {
            self.runtime.is_some()
        }
        #[cfg(not(feature = "runtime"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WassetteConfig::default();
        assert!(config.enable_sandbox);
        assert_eq!(config.max_memory_mb, 128);
        assert_eq!(config.max_execution_time_secs, 30);
        assert!(!config.permissions.allow_fs);
        assert!(!config.permissions.allow_network);
    }

    #[test]
    fn test_client_creation() {
        let client = WassetteClient::default();
        assert!(client.is_ok());
    }

    #[test]
    fn test_component_handle_creation() {
        let handle = ComponentHandle::new(
            "test-id".to_string(),
            std::path::PathBuf::from("/test/path.wasm"),
        );
        assert_eq!(handle.id(), "test-id");
        assert_eq!(handle.path(), Path::new("/test/path.wasm"));
    }

    #[test]
    fn test_default_permissions() {
        let perms = ComponentPermissions::default();
        assert!(!perms.allow_fs);
        assert!(!perms.allow_network);
        assert!(!perms.allow_env);
        assert!(perms.allowed_paths.is_empty());
        assert!(perms.allowed_hosts.is_empty());
    }
}
