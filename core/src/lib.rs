//! Portalis Core Library
//!
//! Core abstractions and types for the Portalis Python → Rust → WASM translation platform.
//! Following London School TDD principles with outside-in development.

pub mod agent;
pub mod message;
pub mod types;
pub mod error;

pub use agent::{Agent, AgentCapability, AgentId, AgentMetadata};
pub use message::{Message, MessageBus, MessageId, MessagePayload};
pub use types::{Artifact, Phase, PipelineState, ArtifactMetadata, TestResult};
pub use error::{Error, Result};
