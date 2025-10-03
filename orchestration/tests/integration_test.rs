//! End-to-end integration test for Phase 2
//!
//! Tests the complete translation pipeline with a real multi-file Python project.

use portalis_ingest::{IngestAgent, IngestInput, ProjectParser};
use portalis_analysis::DependencyResolver;
use portalis_transpiler::{TranspilerAgent, TranspilerInput, ClassTranslator};
use portalis_orchestration::{WorkspaceGenerator, WorkspaceConfig, CrateInfo, ExternalDependency};
use portalis_core::Agent;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

#[tokio::test]
async fn test_end_to_end_library_translation() {
    // This test demonstrates the complete Phase 2 pipeline:
    // 1. Parse multi-file Python project
    // 2. Extract classes and functions
    // 3. Resolve dependencies
    // 4. Generate Rust code
    // 5. Create Cargo workspace

    println!("=== Phase 2 End-to-End Integration Test ===\n");

    // Test project already exists at examples/test_project
    let test_project_path = Path::new("examples/test_project");

    if !test_project_path.exists() {
        println!("Test project not found, skipping integration test");
        return;
    }

    // Step 1: Parse the project
    println!("Step 1: Parsing multi-file Python project...");
    let parser = ProjectParser::new();
    let project = parser.parse_project(test_project_path).unwrap();

    println!("  ✓ Found {} modules", project.modules.len());
    for (name, module) in &project.modules {
        println!("    - {}: {} functions, {} classes, {} imports",
            name,
            module.ast.functions.len(),
            module.ast.classes.len(),
            module.imports.len()
        );
    }

    // Step 2: Resolve dependencies
    println!("\nStep 2: Resolving dependencies...");
    let mut resolver = DependencyResolver::new();

    // Register internal modules
    let module_names: Vec<String> = project.modules.keys().cloned().collect();
    resolver.register_internal_modules(module_names);

    println!("  ✓ Registered {} internal modules", project.modules.len());

    // Step 3: Generate Rust code for each module
    println!("\nStep 3: Generating Rust code...");
    let transpiler = TranspilerAgent::new();

    let mut crates = Vec::new();
    for (module_name, module) in &project.modules {
        // Convert classes and functions to JSON for transpiler
        let typed_classes = serde_json::to_value(&module.ast.classes).unwrap();
        let typed_functions = serde_json::to_value(&module.ast.functions).unwrap();

        let input = TranspilerInput {
            typed_functions: typed_functions.as_array().cloned().unwrap_or_default(),
            typed_classes: typed_classes.as_array().cloned().unwrap_or_default(),
            use_statements: vec![],
            cargo_dependencies: vec![],
            api_contract: serde_json::json!({}),
        };

        let output = transpiler.execute(input).await.unwrap();

        println!("  ✓ Generated {} for module '{}'",
            if module.ast.classes.is_empty() && module.ast.functions.is_empty() {
                "placeholder code"
            } else {
                "Rust code"
            },
            module_name
        );

        // Create crate info
        let crate_name = module_name.replace('.', "_");
        crates.push(CrateInfo {
            name: crate_name.clone(),
            path: crate_name,
            rust_code: output.rust_code,
            dependencies: vec![],
            external_deps: vec![],
        });
    }

    // Step 4: Generate workspace
    println!("\nStep 4: Generating Cargo workspace...");
    let temp_dir = TempDir::new().unwrap();
    let workspace_path = temp_dir.path().join("translated_project");

    let generator = WorkspaceGenerator::new(workspace_path.clone());

    let config = WorkspaceConfig {
        name: "translated_project".to_string(),
        crates,
    };

    generator.generate(&config).unwrap();

    // Verify workspace structure
    assert!(workspace_path.exists(), "Workspace directory should exist");
    assert!(workspace_path.join("Cargo.toml").exists(), "Root Cargo.toml should exist");
    assert!(workspace_path.join("README.md").exists(), "README should exist");

    for crate_info in &config.crates {
        let crate_path = workspace_path.join(&crate_info.path);
        assert!(crate_path.exists(), "Crate directory should exist: {}", crate_info.path);
        assert!(crate_path.join("Cargo.toml").exists(), "Crate Cargo.toml should exist");
        assert!(crate_path.join("src/lib.rs").exists(), "Crate lib.rs should exist");
    }

    println!("  ✓ Created workspace with {} crates", config.crates.len());
    println!("  ✓ All files generated successfully");

    println!("\n=== Phase 2 Integration Test PASSED ===");
}

#[tokio::test]
async fn test_class_translation_pipeline() {
    println!("=== Class Translation Pipeline Test ===\n");

    // Test translating a Python class through the full pipeline
    let python_code = r#"
class Calculator:
    def __init__(self, precision: int):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        return a + b

    def multiply(self, a: float, b: float) -> float:
        return a * b
"#;

    println!("Step 1: Parsing Python class...");
    let ingest = IngestAgent::new();
    let ingest_output = ingest.execute(IngestInput {
        source_path: PathBuf::from("calculator.py"),
        source_code: python_code.to_string(),
    }).await.unwrap();

    assert_eq!(ingest_output.ast.classes.len(), 1);
    println!("  ✓ Parsed 1 class with {} methods",
        ingest_output.ast.classes[0].methods.len());

    println!("\nStep 2: Transpiling to Rust...");
    let transpiler = TranspilerAgent::new();
    let classes_json = serde_json::to_value(&ingest_output.ast.classes).unwrap();

    let transpiler_output = transpiler.execute(TranspilerInput {
        typed_functions: vec![],
        typed_classes: classes_json.as_array().cloned().unwrap_or_default(),
        use_statements: vec![],
        cargo_dependencies: vec![],
        api_contract: serde_json::json!({}),
    }).await.unwrap();

    println!("  ✓ Generated Rust code:");
    println!("{}", transpiler_output.rust_code);

    // Verify the generated code
    assert!(transpiler_output.rust_code.contains("pub struct Calculator"));
    assert!(transpiler_output.rust_code.contains("pub fn new(precision: i32)"));
    assert!(transpiler_output.rust_code.contains("pub fn add(&self, a: f64, b: f64) -> f64"));
    assert!(transpiler_output.rust_code.contains("pub fn multiply(&self, a: f64, b: f64) -> f64"));

    println!("\n=== Class Translation Pipeline PASSED ===");
}

#[tokio::test]
async fn test_workspace_generation_with_dependencies() {
    println!("=== Workspace with Dependencies Test ===\n");

    let temp_dir = TempDir::new().unwrap();
    let workspace_path = temp_dir.path().join("app_with_deps");

    let generator = WorkspaceGenerator::new(workspace_path.clone());

    // Create a realistic workspace with multiple crates and dependencies
    let config = WorkspaceConfig {
        name: "my_application".to_string(),
        crates: vec![
            CrateInfo {
                name: "core".to_string(),
                path: "core".to_string(),
                rust_code: r#"
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    pub version: String,
}
"#.to_string(),
                dependencies: vec![],
                external_deps: vec![
                    ExternalDependency {
                        name: "serde".to_string(),
                        version: "1.0".to_string(),
                        features: vec!["derive".to_string()],
                    },
                ],
            },
            CrateInfo {
                name: "app".to_string(),
                path: "app".to_string(),
                rust_code: r#"
use core::Config;
use serde_json::Value;

pub fn load_config() -> Config {
    Config {
        name: "MyApp".to_string(),
        version: "1.0.0".to_string(),
    }
}
"#.to_string(),
                dependencies: vec!["core".to_string()],
                external_deps: vec![
                    ExternalDependency {
                        name: "serde_json".to_string(),
                        version: "1.0".to_string(),
                        features: vec![],
                    },
                ],
            },
        ],
    };

    generator.generate(&config).unwrap();

    println!("Workspace generated at: {:?}", workspace_path);

    // Verify workspace Cargo.toml
    let workspace_toml = std::fs::read_to_string(workspace_path.join("Cargo.toml")).unwrap();
    println!("\nWorkspace Cargo.toml:");
    println!("{}", workspace_toml);

    assert!(workspace_toml.contains("[workspace]"));
    assert!(workspace_toml.contains("members"));
    assert!(workspace_toml.contains("\"core\""));
    assert!(workspace_toml.contains("\"app\""));
    assert!(workspace_toml.contains("serde"));
    assert!(workspace_toml.contains("serde_json"));

    // Verify app depends on core
    let app_toml = std::fs::read_to_string(workspace_path.join("app/Cargo.toml")).unwrap();
    println!("\nApp Cargo.toml:");
    println!("{}", app_toml);

    assert!(app_toml.contains("core = { path = \"../core\" }"));

    println!("\n=== Workspace with Dependencies Test PASSED ===");
}
