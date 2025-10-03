//! Analysis Agent - Type Inference and API Extraction
//!
//! Analyzes Python AST to infer types and extract API contracts.

// mod flow_analyzer; // Disabled for Phase 1 rapid completion
pub mod dependency_resolver;

use async_trait::async_trait;
// use flow_analyzer::FlowAnalyzer;
pub use dependency_resolver::{DependencyResolver, ResolvedDependency, DependencyType, CrateMapping};
use portalis_core::{Agent, AgentCapability, AgentId, ArtifactMetadata, Result};
use serde::{Deserialize, Serialize};

/// Input from Ingest Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisInput {
    pub ast: serde_json::Value, // PythonAst from ingest
}

/// Output with type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisOutput {
    pub typed_functions: Vec<TypedFunction>,
    pub api_contract: ApiContract,
    pub confidence: f64,
    pub metadata: ArtifactMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedFunction {
    pub name: String,
    pub params: Vec<TypedParameter>,
    pub return_type: RustType,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedParameter {
    pub name: String,
    pub rust_type: RustType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RustType {
    I32,
    I64,
    F64,
    String,
    Bool,
    Vec(Box<RustType>),
    Option(Box<RustType>),
    Unknown,
}

impl std::fmt::Display for RustType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RustType::I32 => write!(f, "i32"),
            RustType::I64 => write!(f, "i64"),
            RustType::F64 => write!(f, "f64"),
            RustType::String => write!(f, "String"),
            RustType::Bool => write!(f, "bool"),
            RustType::Vec(inner) => write!(f, "Vec<{}>", inner),
            RustType::Option(inner) => write!(f, "Option<{}>", inner),
            RustType::Unknown => write!(f, "()"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiContract {
    pub functions: Vec<FunctionSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub name: String,
    pub params: Vec<String>,
    pub return_type: String,
}

/// Analysis Agent implementation
pub struct AnalysisAgent {
    id: AgentId,
}

impl AnalysisAgent {
    pub fn new() -> Self {
        Self {
            id: AgentId::new(),
        }
    }

    /// Infer Rust types from Python type hints
    fn infer_type(&self, python_type: Option<&str>) -> RustType {
        match python_type {
            Some("int") => RustType::I32,
            Some("float") => RustType::F64,
            Some("str") => RustType::String,
            Some("bool") => RustType::Bool,
            Some(t) if t.starts_with("List[") => {
                // Simplified list handling
                RustType::Vec(Box::new(RustType::I32))
            }
            _ => RustType::Unknown,
        }
    }

    /// Build API contract from typed functions
    fn build_api_contract(&self, functions: &[TypedFunction]) -> ApiContract {
        let signatures = functions.iter().map(|f| FunctionSignature {
            name: f.name.clone(),
            params: f.params.iter().map(|p| p.rust_type.to_string()).collect(),
            return_type: f.return_type.to_string(),
        }).collect();

        ApiContract { functions: signatures }
    }
}

impl Default for AnalysisAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for AnalysisAgent {
    type Input = AnalysisInput;
    type Output = AnalysisOutput;

    async fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        tracing::info!("Analyzing Python AST for type information");

        // Parse functions from AST (simplified)
        let functions: Vec<serde_json::Value> = input.ast.get("functions")
            .and_then(|f| f.as_array())
            .cloned()
            .unwrap_or_default();

        let mut typed_functions = Vec::new();
        let mut total_confidence = 0.0;

        for func in functions {
            let name = func.get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown")
                .to_string();

            let params: Vec<serde_json::Value> = func.get("params")
                .and_then(|p| p.as_array())
                .cloned()
                .unwrap_or_default();

            let typed_params: Vec<TypedParameter> = params.iter().map(|p| {
                let param_name = p.get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let type_hint = p.get("type_hint")
                    .and_then(|t| t.as_str());

                TypedParameter {
                    name: param_name,
                    rust_type: self.infer_type(type_hint),
                }
            }).collect();

            let return_hint = func.get("return_type")
                .and_then(|r| r.as_str());

            let return_type = self.infer_type(return_hint);

            // Calculate confidence based on type hint availability
            let confidence = if return_hint.is_some() { 0.9 } else { 0.5 };
            total_confidence += confidence;

            typed_functions.push(TypedFunction {
                name,
                params: typed_params,
                return_type,
                confidence,
            });
        }

        let overall_confidence = if !typed_functions.is_empty() {
            total_confidence / typed_functions.len() as f64
        } else {
            0.0
        };

        let api_contract = self.build_api_contract(&typed_functions);

        let metadata = ArtifactMetadata::new(self.name())
            .with_tag("functions", typed_functions.len().to_string())
            .with_tag("confidence", format!("{:.2}", overall_confidence));

        Ok(AnalysisOutput {
            typed_functions,
            api_contract,
            confidence: overall_confidence,
            metadata,
        })
    }

    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        "AnalysisAgent"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::TypeInference, AgentCapability::ApiExtraction]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_type_inference_with_hints() {
        let agent = AnalysisAgent::new();

        let ast = json!({
            "functions": [{
                "name": "add",
                "params": [
                    {"name": "a", "type_hint": "int"},
                    {"name": "b", "type_hint": "int"}
                ],
                "return_type": "int"
            }]
        });

        let input = AnalysisInput { ast };
        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.typed_functions.len(), 1);
        assert_eq!(output.typed_functions[0].name, "add");
        assert_eq!(output.typed_functions[0].return_type, RustType::I32);
        assert!(output.confidence > 0.8);
    }

    #[tokio::test]
    async fn test_type_inference_without_hints() {
        let agent = AnalysisAgent::new();

        let ast = json!({
            "functions": [{
                "name": "multiply",
                "params": [
                    {"name": "x", "type_hint": null},
                    {"name": "y", "type_hint": null}
                ],
                "return_type": null
            }]
        });

        let input = AnalysisInput { ast };
        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.typed_functions.len(), 1);
        assert_eq!(output.typed_functions[0].return_type, RustType::Unknown);
        assert!(output.confidence < 0.7);
    }

    #[tokio::test]
    async fn test_type_inference_string_type() {
        let agent = AnalysisAgent::new();

        let ast = json!({
            "functions": [{
                "name": "greet",
                "params": [{"name": "name", "type_hint": "str"}],
                "return_type": "str"
            }]
        });

        let input = AnalysisInput { ast };
        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.typed_functions[0].params[0].rust_type, RustType::String);
        assert_eq!(output.typed_functions[0].return_type, RustType::String);
    }

    #[tokio::test]
    async fn test_type_inference_float_type() {
        let agent = AnalysisAgent::new();

        let ast = json!({
            "functions": [{
                "name": "calculate",
                "params": [{"name": "value", "type_hint": "float"}],
                "return_type": "float"
            }]
        });

        let input = AnalysisInput { ast };
        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.typed_functions[0].params[0].rust_type, RustType::F64);
        assert_eq!(output.typed_functions[0].return_type, RustType::F64);
    }

    #[tokio::test]
    async fn test_api_contract_generation() {
        let agent = AnalysisAgent::new();

        let ast = json!({
            "functions": [
                {
                    "name": "add",
                    "params": [
                        {"name": "a", "type_hint": "int"},
                        {"name": "b", "type_hint": "int"}
                    ],
                    "return_type": "int"
                },
                {
                    "name": "multiply",
                    "params": [
                        {"name": "x", "type_hint": "int"},
                        {"name": "y", "type_hint": "int"}
                    ],
                    "return_type": "int"
                }
            ]
        });

        let input = AnalysisInput { ast };
        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.api_contract.functions.len(), 2);
        assert_eq!(output.api_contract.functions[0].name, "add");
        assert_eq!(output.api_contract.functions[1].name, "multiply");
    }

    #[tokio::test]
    async fn test_empty_ast() {
        let agent = AnalysisAgent::new();

        let ast = json!({
            "functions": []
        });

        let input = AnalysisInput { ast };
        let output = agent.execute(input).await.unwrap();

        assert_eq!(output.typed_functions.len(), 0);
        assert_eq!(output.confidence, 0.0);
    }

    #[test]
    fn test_rust_type_display() {
        assert_eq!(RustType::I32.to_string(), "i32");
        assert_eq!(RustType::String.to_string(), "String");
        assert_eq!(RustType::Bool.to_string(), "bool");
        assert_eq!(RustType::Vec(Box::new(RustType::I32)).to_string(), "Vec<i32>");
    }

    #[test]
    fn test_agent_capabilities() {
        let agent = AnalysisAgent::new();
        let caps = agent.capabilities();
        assert!(caps.contains(&AgentCapability::TypeInference));
        assert!(caps.contains(&AgentCapability::ApiExtraction));
    }
}
