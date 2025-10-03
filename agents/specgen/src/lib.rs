//! Specification Generator Agent
//!
//! Generates Rust type specifications from Python analysis.

use async_trait::async_trait;
use portalis_core::{Agent, AgentCapability, AgentId, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecGenInput {
    pub analysis: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecGenOutput {
    pub specification: String,
}

pub struct SpecGenAgent {
    id: AgentId,
}

impl SpecGenAgent {
    pub fn new() -> Self {
        Self { id: AgentId::new() }
    }
}

impl Default for SpecGenAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for SpecGenAgent {
    type Input = SpecGenInput;
    type Output = SpecGenOutput;

    async fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        // Simplified for POC - delegates to transpiler
        Ok(SpecGenOutput {
            specification: serde_json::to_string_pretty(&input.analysis)?,
        })
    }

    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        "SpecGenAgent"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::SpecificationGeneration]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_specification_generation() {
        let agent = SpecGenAgent::new();

        let analysis = json!({
            "functions": [{
                "name": "add",
                "return_type": "i32"
            }]
        });

        let input = SpecGenInput {
            analysis: analysis.clone(),
        };

        let output = agent.execute(input).await.unwrap();

        assert!(!output.specification.is_empty());
        assert!(output.specification.contains("add"));
    }

    #[tokio::test]
    async fn test_specification_preserves_structure() {
        let agent = SpecGenAgent::new();

        let analysis = json!({
            "functions": [],
            "classes": [],
            "metadata": {"version": "1.0"}
        });

        let input = SpecGenInput {
            analysis: analysis.clone(),
        };

        let output = agent.execute(input).await.unwrap();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&output.specification).unwrap();
        assert_eq!(parsed, analysis);
    }

    #[test]
    fn test_agent_metadata() {
        let agent = SpecGenAgent::new();

        assert_eq!(agent.name(), "SpecGenAgent");
        assert_eq!(agent.capabilities(), vec![AgentCapability::SpecificationGeneration]);
    }
}
