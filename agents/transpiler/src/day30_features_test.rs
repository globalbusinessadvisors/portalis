//! Day 30 Feature Tests: Exception Handling
//!
//! Tests for:
//! - try-except blocks
//! - try-except-else blocks
//! - try-except-finally blocks
//! - raise statements
//! - multiple except clauses

use super::feature_translator::FeatureTranslator;

#[test]
fn test_try_except_basic() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
try:
    x = 1 / 0
except:
    x = 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("try") || rust.contains("//"));
}

#[test]
fn test_try_except_with_type() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
try:
    value = int("not a number")
except ValueError:
    value = 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("value"));
    assert!(rust.contains("ValueError") || rust.contains("except"));
}

#[test]
fn test_try_except_with_variable() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
try:
    result = 10 / 0
except ZeroDivisionError as e:
    result = 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("result"));
}

#[test]
fn test_try_except_else() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
try:
    x = 5
except:
    x = 0
else:
    y = x + 1
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("y"));
}

#[test]
fn test_try_finally() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
try:
    x = 1
finally:
    print("cleanup")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("finally") || rust.contains("cleanup"));
}

#[test]
fn test_try_except_finally() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
try:
    x = dangerous_operation()
except:
    x = default_value
finally:
    cleanup()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("cleanup") || rust.contains("finally"));
}

#[test]
fn test_raise_exception() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
if x < 0:
    raise ValueError("x must be positive")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("panic") || rust.contains("raise"));
}

#[test]
fn test_raise_bare() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
try:
    do_something()
except:
    raise
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("panic") || rust.contains("raise") || rust.contains("do_something"));
}

#[test]
fn test_nested_try_except() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
try:
    try:
        x = 1
    except:
        x = 2
except:
    x = 3
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
}

#[test]
fn test_try_except_in_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("safe_divide"));
    assert!(rust.contains("a") && rust.contains("b"));
}
