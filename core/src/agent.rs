//! Agent trait and core abstractions
//!
//! Defines the Agent trait that all specialized agents implement.
//! Following London School TDD: agents are tested via mocked interactions.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

use crate::Result;

/// Unique identifier for agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(Uuid);

impl AgentId {
    /// Create a new random agent ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Capabilities that agents can provide
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentCapability {
    /// Parse Python source code into AST
    Parsing,
    /// Analyze code and infer types
    TypeInference,
    /// Extract API contracts
    ApiExtraction,
    /// Generate Rust specifications
    SpecificationGeneration,
    /// Transpile Python to Rust
    CodeGeneration,
    /// Compile Rust to WASM
    Compilation,
    /// Execute tests and validate
    Testing,
    /// Package artifacts for deployment
    Packaging,
}

/// Core Agent trait that all specialized agents implement
///
/// Following London School TDD:
/// - Agents communicate via message passing
/// - Dependencies are injected and easily mocked
/// - Behavior is tested via interaction testing
#[async_trait]
pub trait Agent: Send + Sync {
    /// Input type for this agent
    type Input: Send + Sync;

    /// Output type for this agent
    type Output: Send + Sync;

    /// Execute the agent's primary function
    ///
    /// This is the main entry point for agent execution.
    /// Implementations should be pure and side-effect free where possible.
    async fn execute(&self, input: Self::Input) -> Result<Self::Output>;

    /// Get the agent's unique identifier
    fn id(&self) -> AgentId;

    /// Get the agent's human-readable name
    fn name(&self) -> &str;

    /// Get the capabilities this agent provides
    fn capabilities(&self) -> Vec<AgentCapability>;

    /// Validate that the agent can handle the given input
    ///
    /// Default implementation always returns Ok(())
    fn validate_input(&self, _input: &Self::Input) -> Result<()> {
        Ok(())
    }
}

/// Agent metadata for registration and discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    pub id: AgentId,
    pub name: String,
    pub capabilities: Vec<AgentCapability>,
    pub version: String,
}

impl AgentMetadata {
    /// Create new agent metadata
    pub fn new(id: AgentId, name: impl Into<String>, capabilities: Vec<AgentCapability>) -> Self {
        Self {
            id,
            name: name.into(),
            capabilities,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id_creation() {
        let id1 = AgentId::new();
        let id2 = AgentId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_agent_id_display() {
        let id = AgentId::new();
        let display = format!("{}", id);
        assert!(!display.is_empty());
    }

    #[test]
    fn test_agent_metadata_creation() {
        let id = AgentId::new();
        let metadata = AgentMetadata::new(
            id,
            "TestAgent",
            vec![AgentCapability::Parsing],
        );

        assert_eq!(metadata.id, id);
        assert_eq!(metadata.name, "TestAgent");
        assert_eq!(metadata.capabilities.len(), 1);
    }
}
