//! Test Agent - WASM Validation
//!
//! Executes and validates WASM modules.

use async_trait::async_trait;
use portalis_core::{Agent, AgentCapability, AgentId, ArtifactMetadata, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestInput {
    pub wasm_bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestOutput {
    pub passed: usize,
    pub failed: usize,
    pub metadata: ArtifactMetadata,
}

pub struct TestAgent {
    id: AgentId,
}

impl TestAgent {
    pub fn new() -> Self {
        Self { id: AgentId::new() }
    }
}

impl Default for TestAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for TestAgent {
    type Input = TestInput;
    type Output = TestOutput;

    async fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        tracing::info!("Testing WASM module");

        // Simplified validation for POC
        let passed = if input.wasm_bytes.starts_with(&[0x00, 0x61, 0x73, 0x6D]) {
            1
        } else {
            0
        };

        let failed = 1 - passed;

        let metadata = ArtifactMetadata::new(self.name())
            .with_tag("passed", passed.to_string())
            .with_tag("failed", failed.to_string());

        Ok(TestOutput {
            passed,
            failed,
            metadata,
        })
    }

    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        "TestAgent"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::Testing]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_valid_wasm_passes() {
        let agent = TestAgent::new();

        // Valid WASM magic number
        let wasm_bytes = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];

        let input = TestInput { wasm_bytes };
        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.passed, 1);
        assert_eq!(output.failed, 0);
    }

    #[tokio::test]
    async fn test_invalid_wasm_fails() {
        let agent = TestAgent::new();

        // Invalid WASM (wrong magic number)
        let wasm_bytes = vec![0xFF, 0xFF, 0xFF, 0xFF];

        let input = TestInput { wasm_bytes };
        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.passed, 0);
        assert_eq!(output.failed, 1);
    }

    #[tokio::test]
    async fn test_empty_wasm_fails() {
        let agent = TestAgent::new();

        let input = TestInput { wasm_bytes: vec![] };
        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.passed, 0);
        assert_eq!(output.failed, 1);
    }

    #[test]
    fn test_agent_metadata() {
        let agent = TestAgent::new();

        assert_eq!(agent.name(), "TestAgent");
        assert_eq!(agent.capabilities(), vec![AgentCapability::Testing]);
    }
}
