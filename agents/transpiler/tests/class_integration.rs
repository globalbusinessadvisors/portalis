//! End-to-end class translation integration tests

use portalis_transpiler::{ClassTranslator, TranspilerAgent, TranspilerInput};
use portalis_core::Agent;
use serde_json::json;

#[tokio::test]
async fn test_translate_calculator_class() {
    let agent = TranspilerAgent::new();

    let input = TranspilerInput {
        typed_functions: vec![],
        typed_classes: vec![json!({
            "name": "Calculator",
            "attributes": [
                {"name": "precision", "type_hint": "int"}
            ],
            "methods": [
                {
                    "name": "__init__",
                    "params": [
                        {"name": "self"},
                        {"name": "precision", "type_hint": "int"}
                    ],
                    "return_type": null
                },
                {
                    "name": "add",
                    "params": [
                        {"name": "self"},
                        {"name": "a", "type_hint": "float"},
                        {"name": "b", "type_hint": "float"}
                    ],
                    "return_type": "float"
                },
                {
                    "name": "subtract",
                    "params": [
                        {"name": "self"},
                        {"name": "a", "type_hint": "float"},
                        {"name": "b", "type_hint": "float"}
                    ],
                    "return_type": "float"
                }
            ]
        })],
        use_statements: vec![],
        cargo_dependencies: vec![],
        api_contract: json!({}),
    };

    let output = agent.execute(input).await.unwrap();

    println!("Generated code:\n{}", output.rust_code);

    // Verify struct generation
    assert!(output.rust_code.contains("pub struct Calculator"));
    assert!(output.rust_code.contains("pub precision: i32"));

    // Verify impl block
    assert!(output.rust_code.contains("impl Calculator"));

    // Verify constructor
    assert!(output.rust_code.contains("pub fn new(precision: i32) -> Self"));

    // Verify instance methods
    assert!(output.rust_code.contains("pub fn add(&self, a: f64, b: f64) -> f64"));
    assert!(output.rust_code.contains("pub fn subtract(&self, a: f64, b: f64) -> f64"));
}

#[tokio::test]
async fn test_translate_counter_class() {
    let agent = TranspilerAgent::new();

    let input = TranspilerInput {
        typed_functions: vec![],
        typed_classes: vec![json!({
            "name": "Counter",
            "attributes": [
                {"name": "count", "type_hint": "int"}
            ],
            "methods": [
                {
                    "name": "__init__",
                    "params": [
                        {"name": "self"}
                    ],
                    "return_type": null
                },
                {
                    "name": "increment",
                    "params": [
                        {"name": "self"}
                    ],
                    "return_type": "int"
                },
                {
                    "name": "get_count",
                    "params": [
                        {"name": "self"}
                    ],
                    "return_type": "int"
                }
            ]
        })],
        use_statements: vec![],
        cargo_dependencies: vec![],
        api_contract: json!({}),
    };

    let output = agent.execute(input).await.unwrap();

    println!("Generated code:\n{}", output.rust_code);

    // Verify struct
    assert!(output.rust_code.contains("pub struct Counter"));
    assert!(output.rust_code.contains("pub count: i32"));

    // Verify methods
    assert!(output.rust_code.contains("pub fn increment(&self) -> i32"));
    assert!(output.rust_code.contains("pub fn get_count(&self) -> i32"));
}

#[test]
fn test_class_translator_directly() {
    let mut translator = ClassTranslator::new();

    let class = json!({
        "name": "Rectangle",
        "attributes": [
            {"name": "width", "type_hint": "float"},
            {"name": "height", "type_hint": "float"}
        ],
        "methods": [
            {
                "name": "__init__",
                "params": [
                    {"name": "self"},
                    {"name": "width", "type_hint": "float"},
                    {"name": "height", "type_hint": "float"}
                ],
                "return_type": null
            },
            {
                "name": "area",
                "params": [
                    {"name": "self"}
                ],
                "return_type": "float"
            }
        ]
    });

    let result = translator.generate_class(&class).unwrap();

    println!("Generated:\n{}", result);

    assert!(result.contains("pub struct Rectangle"));
    assert!(result.contains("pub width: f64"));
    assert!(result.contains("pub height: f64"));
    assert!(result.contains("pub fn new(width: f64, height: f64) -> Self"));
    assert!(result.contains("pub fn area(&self) -> f64"));
}
