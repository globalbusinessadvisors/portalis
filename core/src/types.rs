//! Core types for the Portalis pipeline
//!
//! Defines the data structures that flow through the translation pipeline.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Pipeline execution phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Initial state
    Idle,
    /// Ingesting Python source code
    Ingesting,
    /// Analyzing code structure and types
    Analyzing,
    /// Generating Rust specifications
    GeneratingSpec,
    /// Transpiling Python to Rust
    Transpiling,
    /// Building Rust to WASM
    Building,
    /// Testing the translation
    Testing,
    /// Packaging artifacts
    Packaging,
    /// Successfully completed
    Complete,
    /// Failed with error
    Failed,
}

/// Artifact types produced by agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Artifact {
    /// Python source code
    PythonSource {
        path: PathBuf,
        content: String,
    },
    /// Parsed Python AST
    PythonAst {
        ast: serde_json::Value,
        metadata: ArtifactMetadata,
    },
    /// Type analysis results
    TypeAnalysis {
        types: HashMap<String, String>,
        api: serde_json::Value,
        metadata: ArtifactMetadata,
    },
    /// Rust specification
    RustSpec {
        spec: String,
        metadata: ArtifactMetadata,
    },
    /// Generated Rust code
    RustCode {
        source: String,
        metadata: ArtifactMetadata,
    },
    /// Compiled WASM binary
    WasmBinary {
        bytes: Vec<u8>,
        metadata: ArtifactMetadata,
    },
    /// Test results
    TestResults {
        passed: usize,
        failed: usize,
        details: Vec<TestResult>,
        metadata: ArtifactMetadata,
    },
    /// Final package
    Package {
        wasm: Vec<u8>,
        manifest: serde_json::Value,
        metadata: ArtifactMetadata,
    },
}

/// Metadata attached to artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub created_at: i64,
    pub created_by: String,
    pub version: String,
    pub tags: HashMap<String, String>,
}

impl ArtifactMetadata {
    pub fn new(created_by: impl Into<String>) -> Self {
        Self {
            created_at: chrono::Utc::now().timestamp(),
            created_by: created_by.into(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            tags: HashMap::new(),
        }
    }

    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub actual: Option<serde_json::Value>,
    pub expected: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// Pipeline state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineState {
    pub phase: Phase,
    pub artifacts: HashMap<String, Artifact>,
    pub errors: Vec<String>,
    pub started_at: i64,
    pub completed_at: Option<i64>,
}

impl PipelineState {
    /// Create a new pipeline state
    pub fn new() -> Self {
        Self {
            phase: Phase::Idle,
            artifacts: HashMap::new(),
            errors: Vec::new(),
            started_at: chrono::Utc::now().timestamp(),
            completed_at: None,
        }
    }

    /// Transition to a new phase
    pub fn transition(&mut self, phase: Phase) {
        self.phase = phase;
        if matches!(phase, Phase::Complete | Phase::Failed) {
            self.completed_at = Some(chrono::Utc::now().timestamp());
        }
    }

    /// Add an artifact to the state
    pub fn add_artifact(&mut self, key: impl Into<String>, artifact: Artifact) {
        self.artifacts.insert(key.into(), artifact);
    }

    /// Add an error to the state
    pub fn add_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
    }

    /// Check if the pipeline has failed
    pub fn has_failed(&self) -> bool {
        self.phase == Phase::Failed || !self.errors.is_empty()
    }

    /// Get the duration in seconds (if completed)
    pub fn duration_secs(&self) -> Option<i64> {
        self.completed_at.map(|end| end - self.started_at)
    }
}

impl Default for PipelineState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_state_creation() {
        let state = PipelineState::new();
        assert_eq!(state.phase, Phase::Idle);
        assert_eq!(state.artifacts.len(), 0);
        assert_eq!(state.errors.len(), 0);
    }

    #[test]
    fn test_pipeline_state_transition() {
        let mut state = PipelineState::new();
        state.transition(Phase::Ingesting);
        assert_eq!(state.phase, Phase::Ingesting);
        assert!(state.completed_at.is_none());

        state.transition(Phase::Complete);
        assert_eq!(state.phase, Phase::Complete);
        assert!(state.completed_at.is_some());
    }

    #[test]
    fn test_pipeline_state_error_tracking() {
        let mut state = PipelineState::new();
        assert!(!state.has_failed());

        state.add_error("Test error");
        assert!(state.has_failed());
        assert_eq!(state.errors.len(), 1);
    }

    #[test]
    fn test_artifact_metadata() {
        let metadata = ArtifactMetadata::new("test-agent")
            .with_tag("type", "python")
            .with_tag("version", "3.11");

        assert_eq!(metadata.created_by, "test-agent");
        assert_eq!(metadata.tags.get("type"), Some(&"python".to_string()));
        assert_eq!(metadata.tags.get("version"), Some(&"3.11".to_string()));
    }
}
