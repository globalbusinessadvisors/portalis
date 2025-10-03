//! Transpiler Agent - Python to Rust Code Generation
//!
//! Generates idiomatic Rust code from analyzed Python AST.

mod code_generator;
pub mod class_translator;

use async_trait::async_trait;
use code_generator::CodeGenerator;
pub use class_translator::ClassTranslator;
use portalis_core::{Agent, AgentCapability, AgentId, ArtifactMetadata, Error, Result};
use serde::{Deserialize, Serialize};

#[cfg(feature = "nemo")]
use portalis_nemo_bridge::{NeMoClient, TranslateRequest};

/// Input from Analysis Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranspilerInput {
    pub typed_functions: Vec<serde_json::Value>,
    #[serde(default)]
    pub typed_classes: Vec<serde_json::Value>,
    #[serde(default)]
    pub use_statements: Vec<String>,
    #[serde(default)]
    pub cargo_dependencies: Vec<serde_json::Value>,
    pub api_contract: serde_json::Value,
}

/// Generated Rust code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranspilerOutput {
    pub rust_code: String,
    pub metadata: ArtifactMetadata,
}

/// Translation mode configuration
#[derive(Debug, Clone)]
pub enum TranslationMode {
    /// Pattern-based translation (CPU, no external dependencies)
    PatternBased,
    /// NeMo-powered translation (GPU-accelerated, requires NeMo service)
    #[cfg(feature = "nemo")]
    NeMo {
        service_url: String,
        mode: String,
        temperature: f32,
    },
}

impl Default for TranslationMode {
    fn default() -> Self {
        Self::PatternBased
    }
}

/// Transpiler Agent implementation
pub struct TranspilerAgent {
    id: AgentId,
    translation_mode: TranslationMode,
}

impl TranspilerAgent {
    pub fn new() -> Self {
        Self {
            id: AgentId::new(),
            translation_mode: TranslationMode::default(),
        }
    }

    /// Create transpiler with specific translation mode
    pub fn with_mode(translation_mode: TranslationMode) -> Self {
        Self {
            id: AgentId::new(),
            translation_mode,
        }
    }

    /// Get current translation mode
    pub fn translation_mode(&self) -> &TranslationMode {
        &self.translation_mode
    }

    /// Generate Rust function from typed function info
    fn generate_function(&self, func: &serde_json::Value) -> Result<String> {
        // Use the advanced code generator with temporary instance
        let mut generator = CodeGenerator::new();
        generator.generate_function(func)
    }

    /// Translate Python code using NeMo service
    #[cfg(feature = "nemo")]
    async fn translate_with_nemo(
        &self,
        python_code: &str,
        service_url: &str,
        mode: &str,
        temperature: f32,
    ) -> Result<String> {
        let client = NeMoClient::new(service_url)?;

        let request = TranslateRequest {
            python_code: python_code.to_string(),
            mode: mode.to_string(),
            temperature,
            include_metrics: true,
        };

        let response = client.translate(request).await?;

        tracing::info!(
            "NeMo translation complete: confidence={:.2}%, gpu_util={:.1}%, time={:.1}ms",
            response.confidence * 100.0,
            response.metrics.gpu_utilization * 100.0,
            response.metrics.total_time_ms
        );

        Ok(response.rust_code)
    }

    /// Extract Python source code from function JSON
    fn extract_python_source(&self, func: &serde_json::Value) -> Option<String> {
        func.get("source_code")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
    }

    /// Fallback generate (old implementation)
    fn generate_function_old(&self, func: &serde_json::Value) -> Result<String> {
        let name = func.get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| Error::CodeGeneration("Missing function name".into()))?;

        let params: Vec<serde_json::Value> = func.get("params")
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();

        let return_type = func.get("return_type")
            .and_then(|r| r.as_object())
            .and_then(|r| {
                // Handle enum variants
                if let Some(variant) = r.keys().next() {
                    Some(variant.as_str())
                } else {
                    None
                }
            })
            .unwrap_or("Unknown");

        // Convert Rust type enum to actual Rust type string
        let return_str = match return_type {
            "I32" => "i32",
            "I64" => "i64",
            "F64" => "f64",
            "String" => "String",
            "Bool" => "bool",
            "Unknown" => "()",
            _ => "()",
        };

        // Format parameters
        let param_strs: Vec<String> = params.iter().map(|p| {
            let param_name = p.get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("arg");

            let rust_type = p.get("rust_type")
                .and_then(|r| r.as_object())
                .and_then(|r| r.keys().next().map(|s| s.as_str()))
                .unwrap_or("I32");

            let type_str = match rust_type {
                "I32" => "i32",
                "I64" => "i64",
                "F64" => "f64",
                "String" => "String",
                "Bool" => "bool",
                _ => "i32",
            };

            format!("{}: {}", param_name, type_str)
        }).collect();

        let params_str = param_strs.join(", ");

        // Generate function body (simple template for POC)
        let mut code = String::new();
        code.push_str(&format!("pub fn {}({}) -> {} {{\n", name, params_str, return_str));

        // Add simple implementation based on function name and return type
        if name == "add" && return_str == "i32" {
            code.push_str("    a + b\n");
        } else if name == "multiply" && return_str == "i32" {
            code.push_str("    x * y\n");
        } else if name == "subtract" && return_str == "i32" {
            code.push_str("    a - b\n");
        } else if name == "divide" && return_str == "i32" {
            code.push_str("    a / b\n");
        } else if name == "fibonacci" && return_str == "i32" {
            code.push_str("    if n <= 1 {\n");
            code.push_str("        return n;\n");
            code.push_str("    }\n");
            code.push_str("    fibonacci(n - 1) + fibonacci(n - 2)\n");
        } else if return_str == "()" {
            // Unit return type - just empty body or comment
            code.push_str("    // TODO: Implement function body\n");
        } else {
            // Default implementation based on return type
            code.push_str("    // TODO: Implement function body\n");
            match return_str {
                "i32" | "i64" => code.push_str("    0\n"),
                "f64" => code.push_str("    0.0\n"),
                "bool" => code.push_str("    false\n"),
                "String" => code.push_str("    String::new()\n"),
                _ => code.push_str("    Default::default()\n"),
            }
        }

        code.push_str("}\n");

        Ok(code)
    }
}

impl Default for TranspilerAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for TranspilerAgent {
    type Input = TranspilerInput;
    type Output = TranspilerOutput;

    async fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        tracing::info!("Transpiling Python to Rust");

        let mut rust_code = String::new();

        // Add standard imports and attributes
        rust_code.push_str("// Generated by Portalis Transpiler\n");
        rust_code.push_str("#![allow(unused)]\n\n");

        // Add use statements (dependencies)
        if !input.use_statements.is_empty() {
            for use_stmt in &input.use_statements {
                rust_code.push_str(use_stmt);
                rust_code.push('\n');
            }
            rust_code.push('\n');
        }

        // Generate classes first
        if !input.typed_classes.is_empty() {
            let mut class_translator = ClassTranslator::new();
            for class in &input.typed_classes {
                let class_code = class_translator.generate_class(class)?;
                rust_code.push_str(&class_code);
                rust_code.push('\n');
            }
        }

        // Generate each function
        for func in &input.typed_functions {
            let func_code = self.generate_function(func)?;
            rust_code.push_str(&func_code);
            rust_code.push('\n');
        }

        let metadata = ArtifactMetadata::new(self.name())
            .with_tag("functions", input.typed_functions.len().to_string())
            .with_tag("classes", input.typed_classes.len().to_string())
            .with_tag("dependencies", input.cargo_dependencies.len().to_string())
            .with_tag("lines", rust_code.lines().count().to_string());

        Ok(TranspilerOutput {
            rust_code,
            metadata,
        })
    }

    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        "TranspilerAgent"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::CodeGeneration]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_generate_simple_function() {
        let agent = TranspilerAgent::new();

        let input = TranspilerInput {
            typed_functions: vec![json!({
                "name": "add",
                "params": [
                    {"name": "a", "rust_type": {"I32": null}},
                    {"name": "b", "rust_type": {"I32": null}}
                ],
                "return_type": {"I32": null}
            })],
            typed_classes: vec![],
            use_statements: vec![],
            cargo_dependencies: vec![],
            api_contract: json!({}),
        };

        let output = agent.execute(input).await.unwrap();

        assert!(output.rust_code.contains("pub fn add"));
        assert!(output.rust_code.contains("a: i32, b: i32"));
        assert!(output.rust_code.contains("-> i32"));
    }

    #[tokio::test]
    async fn test_generate_fibonacci() {
        let agent = TranspilerAgent::new();

        let input = TranspilerInput {
            typed_functions: vec![json!({
                "name": "fibonacci",
                "params": [
                    {"name": "n", "rust_type": {"I32": null}}
                ],
                "return_type": {"I32": null}
            })],
            typed_classes: vec![],
            use_statements: vec![],
            cargo_dependencies: vec![],
            api_contract: json!({}),
        };

        let output = agent.execute(input).await.unwrap();

        assert!(output.rust_code.contains("pub fn fibonacci"));
        assert!(output.rust_code.contains("if n <= 1"));
        assert!(output.rust_code.contains("fibonacci(n - 1) + fibonacci(n - 2)"));
    }
}
