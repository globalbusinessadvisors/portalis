//! Integration tests for dependency resolution

use portalis_transpiler::{TranspilerAgent, TranspilerInput};
use portalis_core::Agent;
use serde_json::json;

#[tokio::test]
async fn test_transpile_with_stdlib_dependencies() {
    let agent = TranspilerAgent::new();

    // Simulating code that uses math and collections
    let input = TranspilerInput {
        typed_functions: vec![json!({
            "name": "calculate_distance",
            "params": [
                {"name": "x1", "type_hint": "float"},
                {"name": "y1", "type_hint": "float"},
                {"name": "x2", "type_hint": "float"},
                {"name": "y2", "type_hint": "float"}
            ],
            "return_type": "float"
        })],
        typed_classes: vec![],
        use_statements: vec![
            "use std::f64::{sqrt, pow};".to_string(),
            "use std::collections::HashMap;".to_string(),
        ],
        cargo_dependencies: vec![],
        api_contract: json!({}),
    };

    let output = agent.execute(input).await.unwrap();

    println!("Generated code:\n{}", output.rust_code);

    // Verify use statements are included
    assert!(output.rust_code.contains("use std::f64"));
    assert!(output.rust_code.contains("use std::collections::HashMap"));

    // Verify function is generated
    assert!(output.rust_code.contains("pub fn calculate_distance"));
}

#[tokio::test]
async fn test_transpile_with_external_dependencies() {
    let agent = TranspilerAgent::new();

    // Simulating code that uses external crates
    let input = TranspilerInput {
        typed_functions: vec![],
        typed_classes: vec![json!({
            "name": "DataProcessor",
            "attributes": [
                {"name": "data", "type_hint": "Vec"}
            ],
            "methods": [
                {
                    "name": "__init__",
                    "params": [
                        {"name": "self"},
                        {"name": "data", "type_hint": "Vec"}
                    ],
                    "return_type": null
                }
            ]
        })],
        use_statements: vec![
            "use serde_json;".to_string(),
            "use chrono::prelude::*;".to_string(),
        ],
        cargo_dependencies: vec![
            json!({
                "crate_name": "serde_json",
                "version": "1.0",
                "features": []
            }),
            json!({
                "crate_name": "chrono",
                "version": "0.4",
                "features": []
            })
        ],
        api_contract: json!({}),
    };

    let output = agent.execute(input).await.unwrap();

    println!("Generated code:\n{}", output.rust_code);

    // Verify use statements
    assert!(output.rust_code.contains("use serde_json"));
    assert!(output.rust_code.contains("use chrono::prelude"));

    // Verify struct generated
    assert!(output.rust_code.contains("pub struct DataProcessor"));

    // Verify metadata includes dependencies
    let deps_tag = output.metadata.tags.get("dependencies");
    assert!(deps_tag.is_some());
}

#[tokio::test]
async fn test_transpile_with_multiple_dependencies() {
    let agent = TranspilerAgent::new();

    let input = TranspilerInput {
        typed_functions: vec![json!({
            "name": "process_request",
            "params": [
                {"name": "url", "type_hint": "str"}
            ],
            "return_type": "str"
        })],
        typed_classes: vec![],
        use_statements: vec![
            "use std::collections::HashMap;".to_string(),
            "use serde_json::Value;".to_string(),
            "use regex::Regex;".to_string(),
        ],
        cargo_dependencies: vec![
            json!({
                "crate_name": "serde_json",
                "version": "1.0",
                "features": []
            }),
            json!({
                "crate_name": "regex",
                "version": "1.0",
                "features": []
            })
        ],
        api_contract: json!({}),
    };

    let output = agent.execute(input).await.unwrap();

    println!("Generated code:\n{}", output.rust_code);

    // Verify all use statements present
    assert!(output.rust_code.contains("use std::collections::HashMap"));
    assert!(output.rust_code.contains("use serde_json::Value"));
    assert!(output.rust_code.contains("use regex::Regex"));

    // Use statements should come before function definitions
    let use_pos = output.rust_code.find("use std::collections").unwrap();
    let fn_pos = output.rust_code.find("pub fn process_request").unwrap();
    assert!(use_pos < fn_pos, "Use statements should come before functions");
}
