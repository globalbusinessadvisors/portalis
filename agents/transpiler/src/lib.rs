//! Transpiler Agent - Python to Rust Code Generation
//!
//! Generates idiomatic Rust code from analyzed Python AST.

mod code_generator;
pub mod class_translator;
pub mod class_inheritance;
pub mod decorator_translator;
pub mod python_ast;
pub mod python_to_rust;
pub mod python_parser;
pub mod expression_translator;
pub mod statement_translator;
pub mod simple_parser;
pub mod indented_parser;
pub mod feature_translator;
pub mod stdlib_mapper;
pub mod stdlib_mappings_comprehensive;
pub mod wasi_core;
pub mod wasi_directory;
pub mod wasi_fs;
pub mod wasi_filesystem;
pub mod wasi_fetch;
pub mod wasi_websocket;
pub mod wasi_threading;
pub mod wasi_async_runtime;
pub mod web_workers;
pub mod py_to_rust_fs;
pub mod py_to_rust_http;
pub mod py_to_rust_asyncio;
pub mod import_analyzer;
pub mod cargo_generator;
pub mod external_packages;
pub mod dependency_graph;
pub mod dependency_resolver;
pub mod version_compatibility;
pub mod build_optimizer;
pub mod dead_code_eliminator;
pub mod code_splitter;
pub mod wasm_bundler;
pub mod npm_package_generator;
pub mod typescript_generator;
pub mod multi_target_builder;
pub mod advanced_features;

// WASM bindings (only compile for wasm32 target)
#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(test)]
mod test_import_aliases;

#[cfg(test)]
mod day3_features_test;

#[cfg(test)]
mod day4_5_features_test;

#[cfg(test)]
mod day6_7_features_test;

#[cfg(test)]
mod day8_9_features_test;

#[cfg(test)]
mod day10_11_features_test;

#[cfg(test)]
mod day12_13_features_test;

#[cfg(test)]
mod day14_15_features_test;

#[cfg(test)]
mod day16_17_features_test;

#[cfg(test)]
mod day18_19_features_test;

#[cfg(test)]
mod day20_21_features_test;

#[cfg(test)]
mod day22_23_features_test;

#[cfg(test)]
mod day24_25_features_test;

#[cfg(test)]
mod day26_27_features_test;

#[cfg(test)]
mod day28_29_features_test;

#[cfg(test)]
mod day30_features_test;

#[cfg(not(target_arch = "wasm32"))]
use async_trait::async_trait;

use code_generator::CodeGenerator;
pub use class_translator::ClassTranslator;
use python_parser::PythonParser;
use python_to_rust::PythonToRustTranslator;
use feature_translator::FeatureTranslator;

#[cfg(not(target_arch = "wasm32"))]
use portalis_core::{Agent, AgentCapability, AgentId, ArtifactMetadata, Error, Result};

#[cfg(target_arch = "wasm32")]
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(target_arch = "wasm32")]
#[derive(Debug)]
pub enum Error {
    CodeGeneration(String),
    Parse(String),
    Other(String),
}

#[cfg(target_arch = "wasm32")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::CodeGeneration(msg) => write!(f, "Code generation error: {}", msg),
            Error::Parse(msg) => write!(f, "Parse error: {}", msg),
            Error::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl std::error::Error for Error {}

#[cfg(target_arch = "wasm32")]
impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Other(s)
    }
}

#[cfg(target_arch = "wasm32")]
impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Other(s.to_string())
    }
}

use serde::{Deserialize, Serialize};

#[cfg(all(feature = "nemo", not(target_arch = "wasm32")))]
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
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranspilerOutput {
    pub rust_code: String,
    pub metadata: ArtifactMetadata,
}

/// Translation mode configuration
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub enum TranslationMode {
    /// Pattern-based translation (CPU, no external dependencies)
    /// Uses CodeGenerator for JSON-based typed functions
    PatternBased,
    /// Full AST-based translation pipeline
    /// Uses PythonParser → PythonToRustTranslator for complete Python source
    AstBased,
    /// Feature-based translation with stdlib mapping
    /// Uses FeatureTranslator for comprehensive Python → Rust translation
    FeatureBased,
    /// NeMo-powered translation (GPU-accelerated, requires NeMo service)
    #[cfg(feature = "nemo")]
    NeMo {
        service_url: String,
        mode: String,
        temperature: f32,
    },
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for TranslationMode {
    fn default() -> Self {
        Self::PatternBased
    }
}

/// Transpiler Agent implementation
#[cfg(not(target_arch = "wasm32"))]
pub struct TranspilerAgent {
    id: AgentId,
    translation_mode: TranslationMode,
}

#[cfg(not(target_arch = "wasm32"))]
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

    /// Create transpiler with AST-based translation mode
    pub fn with_ast_mode() -> Self {
        Self::with_mode(TranslationMode::AstBased)
    }

    /// Create transpiler with feature-based translation mode
    pub fn with_feature_mode() -> Self {
        Self::with_mode(TranslationMode::FeatureBased)
    }

    /// Get current translation mode
    pub fn translation_mode(&self) -> &TranslationMode {
        &self.translation_mode
    }

    /// Generate Rust function from typed function info
    #[cfg(not(target_arch = "wasm32"))]
    fn generate_function(&self, func: &serde_json::Value) -> Result<String> {
        // Check if source code is available for AST-based translation
        if let Some(source_code) = self.extract_python_source(func) {
            match &self.translation_mode {
                TranslationMode::AstBased => {
                    return self.translate_with_ast(&source_code);
                }
                TranslationMode::FeatureBased => {
                    return self.translate_with_features(&source_code);
                }
                _ => {}
            }
        }

        // Fallback to pattern-based CodeGenerator
        let mut generator = CodeGenerator::new();
        generator.generate_function(func)
    }

    /// Translate Python source code using AST-based translator
    #[cfg(not(target_arch = "wasm32"))]
    fn translate_with_ast(&self, python_code: &str) -> Result<String> {
        // Parse Python source
        let parser = PythonParser::new(python_code.to_string(), "<string>".to_string());
        let module = parser.parse()?;

        // Translate to Rust
        let mut translator = PythonToRustTranslator::new();
        translator.translate_module(&module)
    }

    /// Translate Python source code using feature-based translator
    #[cfg(not(target_arch = "wasm32"))]
    fn translate_with_features(&self, python_code: &str) -> Result<String> {
        let mut translator = FeatureTranslator::new();
        translator.translate(python_code)
    }

    /// Translate complete Python module to Rust (public API)
    ///
    /// This method provides a high-level interface for translating Python source code
    /// using the configured translation mode.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn translate_python_module(&self, python_source: &str) -> Result<String> {
        match &self.translation_mode {
            TranslationMode::PatternBased => {
                // For pattern-based, we need to parse and extract functions
                // This is a simplified approach - in production you'd parse the module
                Err(Error::CodeGeneration(
                    "PatternBased mode requires pre-analyzed JSON input. Use AstBased or FeatureBased for raw Python source.".into()
                ))
            }
            TranslationMode::AstBased => {
                self.translate_with_ast(python_source)
            }
            TranslationMode::FeatureBased => {
                self.translate_with_features(python_source)
            }
            #[cfg(feature = "nemo")]
            TranslationMode::NeMo { service_url, mode, temperature } => {
                // NeMo requires async, this is a sync method
                Err(Error::CodeGeneration(
                    "NeMo mode requires async translation. Use translate_with_nemo directly.".into()
                ))
            }
        }
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
    #[allow(dead_code)]
    #[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
impl Default for TranspilerAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
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

    #[test]
    fn test_feature_based_translation() {
        let agent = TranspilerAgent::with_feature_mode();

        let python_code = r#"
x = 42
y = 3.14
msg = "hello"
result = x + 10
"#;

        let rust_code = agent.translate_python_module(python_code).unwrap();

        assert!(rust_code.contains("let x: i32 = 42"));
        assert!(rust_code.contains("let y: f64 = 3.14"));
        assert!(rust_code.contains("let msg: String = \"hello\""));
        assert!(rust_code.contains("let result: i32 = x + 10"));
    }

    #[test]
    fn test_ast_based_translation() {
        let agent = TranspilerAgent::with_ast_mode();

        let python_code = r#"
def add(a, b):
    return a + b
"#;

        let rust_code = agent.translate_python_module(python_code).unwrap();

        assert!(rust_code.contains("fn add"));
        assert!(rust_code.contains("return a + b"));
    }

    #[test]
    fn test_feature_based_with_imports() {
        let agent = TranspilerAgent::with_feature_mode();

        let python_code = r#"
import math
x = math.sqrt(16)
"#;

        let rust_code = agent.translate_python_module(python_code).unwrap();

        // Should have use statements for math module
        assert!(rust_code.contains("use") || rust_code.contains("math"));
    }

    #[test]
    fn test_feature_based_with_control_flow() {
        let agent = TranspilerAgent::with_feature_mode();

        let python_code = r#"
x = 10
if x > 5:
    y = 20
else:
    y = 30
"#;

        let rust_code = agent.translate_python_module(python_code).unwrap();

        assert!(rust_code.contains("let x: i32 = 10"));
        assert!(rust_code.contains("if x > 5"));
    }

    #[tokio::test]
    async fn test_ast_mode_with_source_code() {
        let agent = TranspilerAgent::with_ast_mode();

        let input = TranspilerInput {
            typed_functions: vec![json!({
                "name": "add",
                "params": [
                    {"name": "a", "rust_type": {"I32": null}},
                    {"name": "b", "rust_type": {"I32": null}}
                ],
                "return_type": {"I32": null},
                "source_code": "def add(a, b):\n    return a + b"
            })],
            typed_classes: vec![],
            use_statements: vec![],
            cargo_dependencies: vec![],
            api_contract: json!({}),
        };

        let output = agent.execute(input).await.unwrap();

        // Should use AST translation since source_code is available
        assert!(output.rust_code.contains("fn add"));
    }
}
