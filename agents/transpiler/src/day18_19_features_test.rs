//! Day 18-19 Feature Tests: Built-in Functions
//!
//! Tests for:
//! - len() function
//! - max() and min() functions
//! - sum() function
//! - abs() function
//! - sorted() and reversed() functions

use super::feature_translator::FeatureTranslator;

#[test]
fn test_len_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
count = len(numbers)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("count"));
    assert!(rust.contains(".len()"));
}

#[test]
fn test_max_function_list() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 5, 3, 9, 2]
biggest = max(numbers)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("biggest"));
    assert!(rust.contains("max") || rust.contains(".iter()"));
}

#[test]
fn test_max_function_args() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
result = max(5, 10, 3)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("result"));
    assert!(rust.contains("5") && rust.contains("10") && rust.contains("3"));
}

#[test]
fn test_min_function_list() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 5, 3, 9, 2]
smallest = min(numbers)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("smallest"));
    assert!(rust.contains("min") || rust.contains(".iter()"));
}

#[test]
fn test_min_function_args() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
result = min(5, 10, 3)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("result"));
    assert!(rust.contains("5") && rust.contains("10") && rust.contains("3"));
}

#[test]
fn test_sum_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("total"));
    assert!(rust.contains("sum") || rust.contains(".iter()"));
}

#[test]
fn test_abs_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = -10
positive = abs(x)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("positive"));
    assert!(rust.contains("abs"));
}

#[test]
fn test_sorted_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [3, 1, 4, 1, 5]
ordered = sorted(numbers)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("ordered"));
    assert!(rust.contains("sort") || rust.contains("sorted"));
}

#[test]
fn test_reversed_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
backwards = reversed(numbers)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("backwards"));
    assert!(rust.contains("reverse") || rust.contains("reversed"));
}

#[test]
fn test_multiple_builtin_functions() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
count = len(numbers)
total = sum(numbers)
maximum = max(numbers)
minimum = min(numbers)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("count"));
    assert!(rust.contains("total"));
    assert!(rust.contains("maximum"));
    assert!(rust.contains("minimum"));
}

#[test]
fn test_nested_builtin_functions() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, -5, 3, -2, 4]
max_abs = max(abs(x) for x in numbers)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("max_abs"));
}

#[test]
fn test_len_with_string() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "Hello"
length = len(text)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("length"));
    assert!(rust.contains(".len()"));
}
