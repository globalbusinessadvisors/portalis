//! Error types for Portalis
//!
//! Following Rust best practices with thiserror for ergonomic error handling.

use thiserror::Error;

/// Result type alias for Portalis operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Portalis operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Type inference error: {0}")]
    TypeInference(String),

    #[error("Code generation error: {0}")]
    CodeGeneration(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Test execution error: {0}")]
    TestExecution(String),

    #[error("Message bus error: {0}")]
    MessageBus(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),
}
