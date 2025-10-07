//! Portalis Core Library
//!
//! Core abstractions and types for the Portalis Python → Rust → WASM translation platform.
//! Following London School TDD principles with outside-in development.

pub mod agent;
pub mod message;
pub mod types;
pub mod error;
pub mod metrics;
pub mod middleware;
pub mod telemetry;
pub mod logging;
pub mod assessment;
pub mod acceleration;
// TODO: Re-enable when sqlx dependency is added
// pub mod rbac;
// pub mod sso;
// pub mod quota;

pub use agent::{Agent, AgentCapability, AgentId, AgentMetadata};
pub use message::{Message, MessageBus, MessageId, MessagePayload};
pub use types::{Artifact, Phase, PipelineState, ArtifactMetadata, TestResult};
pub use error::{Error, Result};
pub use metrics::PortalisMetrics;
pub use telemetry::init_telemetry;
pub use acceleration::AccelerationConfig;
