//! Ingest Agent - Python AST Parser
//!
//! Parses Python source code into a Portalis AST representation.
//! Following London School TDD with outside-in development.

mod enhanced_parser;
pub mod project_parser;

use async_trait::async_trait;
use enhanced_parser::EnhancedParser;
use portalis_core::{Agent, AgentCapability, AgentId, ArtifactMetadata, Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// Re-export for external use
pub use project_parser::{ProjectParser, PythonProject, PythonModule};

/// Input for the Ingest Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestInput {
    pub source_path: PathBuf,
    pub source_code: String,
}

/// Output from the Ingest Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestOutput {
    pub ast: PythonAst,
    pub metadata: ArtifactMetadata,
}

/// Simplified Python AST representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonAst {
    pub functions: Vec<PythonFunction>,
    pub classes: Vec<PythonClass>,
    pub imports: Vec<PythonImport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonFunction {
    pub name: String,
    pub params: Vec<PythonParameter>,
    pub return_type: Option<String>,
    pub body: String,
    pub decorators: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonParameter {
    pub name: String,
    pub type_hint: Option<String>,
    pub default: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonClass {
    pub name: String,
    pub bases: Vec<String>,
    pub methods: Vec<PythonFunction>,
    pub attributes: Vec<PythonAttribute>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonAttribute {
    pub name: String,
    pub type_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonImport {
    pub module: String,
    pub items: Vec<String>,
    pub alias: Option<String>,
}

/// Ingest Agent implementation
pub struct IngestAgent {
    id: AgentId,
    parser: EnhancedParser,
    fallback_regex: bool, // Use regex parser as fallback
}

impl IngestAgent {
    pub fn new() -> Self {
        Self {
            id: AgentId::new(),
            parser: EnhancedParser::new(),
            fallback_regex: false,
        }
    }

    /// Enable regex fallback mode for testing
    pub fn with_regex_fallback(mut self) -> Self {
        self.fallback_regex = true;
        self
    }

    /// Parse Python source using enhanced rustpython-parser
    fn parse_python(&self, source: &str) -> Result<PythonAst> {
        // Try enhanced parser first
        match self.parser.parse(source) {
            Ok(ast) => Ok(ast),
            Err(e) if self.fallback_regex => {
                tracing::warn!("Enhanced parser failed, falling back to regex: {}", e);
                self.parse_python_regex(source)
            }
            Err(e) => Err(e),
        }
    }

    /// Simple Python parser (proof-of-concept fallback)
    /// For MVP, we'll use regex-based parsing for simple functions
    fn parse_python_regex(&self, source: &str) -> Result<PythonAst> {
        let mut ast = PythonAst {
            functions: Vec::new(),
            classes: Vec::new(),
            imports: Vec::new(),
        };

        // Parse imports
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                ast.imports.push(self.parse_import(trimmed)?);
            }
        }

        // Parse functions (simple regex-based for POC)
        let func_regex = regex::Regex::new(r"def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(\w+))?\s*:")
            .map_err(|e| Error::Parse(e.to_string()))?;

        for cap in func_regex.captures_iter(source) {
            let name = cap.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
            let params_str = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            let return_type = cap.get(3).map(|m| m.as_str().to_string());

            let params = self.parse_parameters(params_str)?;

            ast.functions.push(PythonFunction {
                name,
                params,
                return_type,
                body: String::new(), // Simplified for POC
                decorators: Vec::new(),
            });
        }

        Ok(ast)
    }

    fn parse_import(&self, line: &str) -> Result<PythonImport> {
        if line.starts_with("import ") {
            let module = line.strip_prefix("import ")
                .unwrap_or("")
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_string();

            Ok(PythonImport {
                module,
                items: Vec::new(),
                alias: None,
            })
        } else {
            // from X import Y
            Ok(PythonImport {
                module: "unknown".to_string(),
                items: Vec::new(),
                alias: None,
            })
        }
    }

    fn parse_parameters(&self, params_str: &str) -> Result<Vec<PythonParameter>> {
        let mut params = Vec::new();

        for param in params_str.split(',') {
            let trimmed = param.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Simple parsing: name or name: type
            let parts: Vec<&str> = trimmed.split(':').collect();
            let name = parts[0].trim().to_string();
            let type_hint = parts.get(1).map(|t| t.trim().to_string());

            params.push(PythonParameter {
                name,
                type_hint,
                default: None,
            });
        }

        Ok(params)
    }
}

impl Default for IngestAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for IngestAgent {
    type Input = IngestInput;
    type Output = IngestOutput;

    async fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        tracing::info!("Ingesting Python source from {:?}", input.source_path);

        let ast = self.parse_python(&input.source_code)?;

        let metadata = ArtifactMetadata::new(self.name())
            .with_tag("source", input.source_path.display().to_string())
            .with_tag("functions", ast.functions.len().to_string())
            .with_tag("classes", ast.classes.len().to_string());

        Ok(IngestOutput { ast, metadata })
    }

    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        "IngestAgent"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::Parsing]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_simple_function() {
        let agent = IngestAgent::new();
        let source = r#"
def add(a: int, b: int) -> int:
    return a + b
"#;

        let input = IngestInput {
            source_path: PathBuf::from("test.py"),
            source_code: source.to_string(),
        };

        let output = agent.execute(input).await.unwrap();
        assert_eq!(output.ast.functions.len(), 1);
        assert_eq!(output.ast.functions[0].name, "add");
        assert_eq!(output.ast.functions[0].params.len(), 2);
        assert_eq!(output.ast.functions[0].return_type, Some("int".to_string()));
    }

    #[tokio::test]
    async fn test_parse_function_without_types() {
        let agent = IngestAgent::new();
        let source = r#"
def multiply(x, y):
    return x * y
"#;

        let input = IngestInput {
            source_path: PathBuf::from("test.py"),
            source_code: source.to_string(),
        };

        let output = agent.execute(input).await.unwrap();
        assert_eq!(output.ast.functions.len(), 1);
        assert_eq!(output.ast.functions[0].name, "multiply");
        assert_eq!(output.ast.functions[0].return_type, None);
    }

    #[tokio::test]
    async fn test_parse_imports() {
        let agent = IngestAgent::new();
        let source = r#"
import sys
import os

def main():
    pass
"#;

        let input = IngestInput {
            source_path: PathBuf::from("test.py"),
            source_code: source.to_string(),
        };

        let output = agent.execute(input).await.unwrap();
        assert_eq!(output.ast.imports.len(), 2);
    }

    #[tokio::test]
    async fn test_parse_empty_file() {
        let agent = IngestAgent::new();

        let input = IngestInput {
            source_path: PathBuf::from("empty.py"),
            source_code: "".to_string(),
        };

        let output = agent.execute(input).await.unwrap();
        assert_eq!(output.ast.functions.len(), 0);
        assert_eq!(output.ast.classes.len(), 0);
        assert_eq!(output.ast.imports.len(), 0);
    }

    #[tokio::test]
    async fn test_multiple_parameters() {
        let agent = IngestAgent::new();
        let source = r#"
def process(a: int, b: str, c: float, d: bool) -> bool:
    return True
"#;

        let input = IngestInput {
            source_path: PathBuf::from("test.py"),
            source_code: source.to_string(),
        };

        let output = agent.execute(input).await.unwrap();
        assert_eq!(output.ast.functions[0].params.len(), 4);
        assert_eq!(output.ast.functions[0].params[0].name, "a");
        assert_eq!(output.ast.functions[0].params[1].name, "b");
    }

    #[test]
    fn test_agent_capabilities() {
        let agent = IngestAgent::new();
        assert_eq!(agent.capabilities(), vec![AgentCapability::Parsing]);
    }
}
