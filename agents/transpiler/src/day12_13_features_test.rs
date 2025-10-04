//! Day 12-13 Feature Tests: Assert Statements and Error Handling
//!
//! Tests for:
//! - Basic assert statements
//! - Assert with messages
//! - Assert with complex conditions

use super::feature_translator::FeatureTranslator;

#[test]
fn test_simple_assert() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
assert True
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("assert!"));
}

#[test]
fn test_assert_with_condition() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 5
assert x > 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("assert!"));
    assert!(rust.contains("x > 0"));
}

#[test]
fn test_assert_with_message() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 5
assert x > 0, "x must be positive"
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("assert!"));
    assert!(rust.contains("x > 0"));
    assert!(rust.contains("must be positive"));
}

#[test]
fn test_assert_equality() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
result = 42
assert result == 42
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("assert!"));
    assert!(rust.contains("result == 42"));
}

#[test]
fn test_assert_not_equal() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
value = 10
assert value != 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("assert!"));
    assert!(rust.contains("value != 0"));
}

#[test]
fn test_assert_in_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def divide(a, b):
    assert b != 0, "Cannot divide by zero"
    return a / b
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("fn divide"));
    assert!(rust.contains("assert!"));
    assert!(rust.contains("b != 0"));
    assert!(rust.contains("Cannot divide by zero"));
}

#[test]
fn test_multiple_asserts() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 5
y = 10
assert x > 0
assert y > x
assert x + y == 15
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    // Should contain multiple assert statements
    let assert_count = rust.matches("assert!").count();
    assert!(assert_count >= 3);
}

#[test]
fn test_assert_with_and() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 5
y = 10
assert x > 0 and y > 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("assert!"));
    assert!(rust.contains("x > 0") && rust.contains("y > 0"));
}

#[test]
fn test_assert_with_or() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 5
assert x < 0 or x > 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("assert!"));
    assert!(rust.contains("x < 0") || rust.contains("x > 0"));
}

#[test]
fn test_assert_with_not() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 5
assert not (x == 0)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("assert!"));
}
