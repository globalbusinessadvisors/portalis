//! Packaging Agent - Artifact Assembly
//!
//! Creates deployable packages from WASM artifacts.

use async_trait::async_trait;
use portalis_core::{Agent, AgentCapability, AgentId, ArtifactMetadata, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackagingInput {
    pub wasm_bytes: Vec<u8>,
    pub test_results: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackagingOutput {
    pub package_bytes: Vec<u8>,
    pub manifest: serde_json::Value,
    pub metadata: ArtifactMetadata,
}

pub struct PackagingAgent {
    id: AgentId,
}

impl PackagingAgent {
    pub fn new() -> Self {
        Self { id: AgentId::new() }
    }
}

impl Default for PackagingAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for PackagingAgent {
    type Input = PackagingInput;
    type Output = PackagingOutput;

    async fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        tracing::info!("Packaging WASM artifact");

        let manifest = serde_json::json!({
            "version": "0.1.0",
            "wasm_size": input.wasm_bytes.len(),
            "test_results": input.test_results,
        });

        let metadata = ArtifactMetadata::new(self.name())
            .with_tag("size", input.wasm_bytes.len().to_string());

        Ok(PackagingOutput {
            package_bytes: input.wasm_bytes,
            manifest,
            metadata,
        })
    }

    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        "PackagingAgent"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::Packaging]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_package_creation() {
        let agent = PackagingAgent::new();

        let wasm_bytes = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        let test_results = json!({
            "passed": 5,
            "failed": 0
        });

        let input = PackagingInput {
            wasm_bytes: wasm_bytes.clone(),
            test_results,
        };

        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.package_bytes, wasm_bytes);
        assert_eq!(output.manifest["version"], "0.1.0");
        assert_eq!(output.manifest["wasm_size"], 8);
    }

    #[tokio::test]
    async fn test_package_includes_test_results() {
        let agent = PackagingAgent::new();

        let test_results = json!({
            "passed": 3,
            "failed": 1
        });

        let input = PackagingInput {
            wasm_bytes: vec![0x00; 100],
            test_results: test_results.clone(),
        };

        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.manifest["test_results"], test_results);
        assert_eq!(output.manifest["wasm_size"], 100);
    }

    #[test]
    fn test_agent_metadata() {
        let agent = PackagingAgent::new();

        assert_eq!(agent.name(), "PackagingAgent");
        assert_eq!(agent.capabilities(), vec![AgentCapability::Packaging]);
    }
}
