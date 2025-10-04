//! Day 16-17 Feature Tests: Lambda Expressions
//!
//! Tests for:
//! - Simple lambda expressions
//! - Lambda with multiple arguments
//! - Lambda in map/filter operations
//! - Lambda with complex expressions

use super::feature_translator::FeatureTranslator;

#[test]
fn test_simple_lambda() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
add_one = lambda x: x + 1
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("add_one"));
    assert!(rust.contains("|x|") || rust.contains("lambda"));
}

#[test]
fn test_lambda_two_args() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
add = lambda x, y: x + y
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("add"));
    assert!(rust.contains("x") && rust.contains("y"));
}

#[test]
fn test_lambda_no_args() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
get_value = lambda: 42
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("get_value"));
    assert!(rust.contains("42"));
}

#[test]
fn test_lambda_with_multiplication() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
double = lambda x: x * 2
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("double"));
    assert!(rust.contains("x * 2"));
}

#[test]
fn test_lambda_with_comparison() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
is_positive = lambda x: x > 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("is_positive"));
    assert!(rust.contains("x > 0"));
}

#[test]
fn test_lambda_with_multiple_operations() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
calc = lambda x, y: x * 2 + y
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("calc"));
    assert!(rust.contains("x") && rust.contains("y"));
}

#[test]
fn test_lambda_complex_expression() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
f = lambda x, y, z: x + y * z
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("f"));
    assert!(rust.contains("x") && rust.contains("y") && rust.contains("z"));
}

#[test]
fn test_lambda_with_call() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
square = lambda x: x * x
result = square(5)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("square"));
    assert!(rust.contains("result"));
}

#[test]
fn test_multiple_lambdas() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
add = lambda x, y: x + y
sub = lambda x, y: x - y
mul = lambda x, y: x * y
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("add"));
    assert!(rust.contains("sub"));
    assert!(rust.contains("mul"));
}

#[test]
fn test_lambda_assigned_to_variable() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
increment = lambda x: x + 1
value = 10
new_value = increment(value)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("increment"));
    assert!(rust.contains("value"));
    assert!(rust.contains("new_value"));
}
