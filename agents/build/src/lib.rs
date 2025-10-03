//! Build Agent - Rust to WASM Compilation
//!
//! Compiles generated Rust code to WebAssembly.

use async_trait::async_trait;
use portalis_core::{Agent, AgentCapability, AgentId, ArtifactMetadata, Error, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Input from Transpiler Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInput {
    pub rust_code: String,
}

/// Compiled WASM output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildOutput {
    pub wasm_bytes: Vec<u8>,
    pub compilation_log: String,
    pub metadata: ArtifactMetadata,
}

/// Build Agent implementation
pub struct BuildAgent {
    id: AgentId,
    workspace_dir: PathBuf,
}

impl BuildAgent {
    pub fn new() -> Self {
        Self {
            id: AgentId::new(),
            workspace_dir: std::env::temp_dir().join(format!("portalis-build-{}", uuid::Uuid::new_v4())),
        }
    }

    pub fn with_workspace(workspace_dir: PathBuf) -> Self {
        Self {
            id: AgentId::new(),
            workspace_dir,
        }
    }

    /// Create a temporary Rust project for compilation
    fn setup_project(&self, rust_code: &str) -> Result<PathBuf> {
        fs::create_dir_all(&self.workspace_dir)
            .map_err(|e| Error::Compilation(format!("Failed to create workspace: {}", e)))?;

        // Create Cargo.toml
        let cargo_toml = r#"[package]
name = "portalis-generated"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
"#;

        fs::write(self.workspace_dir.join("Cargo.toml"), cargo_toml)
            .map_err(|e| Error::Compilation(format!("Failed to write Cargo.toml: {}", e)))?;

        // Create src directory and lib.rs
        let src_dir = self.workspace_dir.join("src");
        fs::create_dir_all(&src_dir)
            .map_err(|e| Error::Compilation(format!("Failed to create src dir: {}", e)))?;

        fs::write(src_dir.join("lib.rs"), rust_code)
            .map_err(|e| Error::Compilation(format!("Failed to write lib.rs: {}", e)))?;

        Ok(self.workspace_dir.clone())
    }

    /// Compile Rust code to WASM
    fn compile_to_wasm(&self) -> Result<(Vec<u8>, String)> {
        // Build with cargo
        let output = Command::new("cargo")
            .args(&["build", "--release", "--target", "wasm32-unknown-unknown"])
            .current_dir(&self.workspace_dir)
            .output()
            .map_err(|e| Error::Compilation(format!("Failed to run cargo: {}", e)))?;

        let log = format!(
            "STDOUT:\n{}\n\nSTDERR:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        if !output.status.success() {
            return Err(Error::Compilation(format!(
                "Compilation failed:\n{}",
                log
            )));
        }

        // Read the compiled WASM file
        let wasm_path = self.workspace_dir
            .join("target/wasm32-unknown-unknown/release/portalis_generated.wasm");

        let wasm_bytes = fs::read(&wasm_path)
            .map_err(|e| Error::Compilation(format!("Failed to read WASM: {}", e)))?;

        Ok((wasm_bytes, log))
    }

    /// Clean up temporary workspace
    fn cleanup(&self) -> Result<()> {
        if self.workspace_dir.exists() {
            fs::remove_dir_all(&self.workspace_dir)
                .map_err(|e| Error::Compilation(format!("Failed to cleanup: {}", e)))?;
        }
        Ok(())
    }
}

impl Default for BuildAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BuildAgent {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

#[async_trait]
impl Agent for BuildAgent {
    type Input = BuildInput;
    type Output = BuildOutput;

    async fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        tracing::info!("Building Rust code to WASM");

        // Set up project
        self.setup_project(&input.rust_code)?;

        // Compile to WASM
        let (wasm_bytes, log) = self.compile_to_wasm()?;

        // Validate WASM magic number
        if !wasm_bytes.starts_with(&[0x00, 0x61, 0x73, 0x6D]) {
            return Err(Error::Compilation("Invalid WASM binary".into()));
        }

        let metadata = ArtifactMetadata::new(self.name())
            .with_tag("size", wasm_bytes.len().to_string())
            .with_tag("workspace", self.workspace_dir.display().to_string());

        Ok(BuildOutput {
            wasm_bytes,
            compilation_log: log,
            metadata,
        })
    }

    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        "BuildAgent"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::Compilation]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Rust toolchain and wasm32 target
    async fn test_compile_simple_rust() {
        let agent = BuildAgent::new();

        let rust_code = r#"
#[no_mangle]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;

        let input = BuildInput {
            rust_code: rust_code.to_string(),
        };

        let output = agent.execute(input).await.unwrap();

        assert!(!output.wasm_bytes.is_empty());
        assert!(output.wasm_bytes.starts_with(&[0x00, 0x61, 0x73, 0x6D]));
    }
}
