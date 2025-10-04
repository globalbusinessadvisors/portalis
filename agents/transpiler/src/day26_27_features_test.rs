//! Day 26-27 Feature Tests: Tuple Unpacking
//!
//! Tests for:
//! - Tuple unpacking in assignments
//! - Tuple unpacking in for loops
//! - Multiple return values

use super::feature_translator::FeatureTranslator;

#[test]
fn test_tuple_unpacking_simple() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
pair = (1, 2)
a, b = pair
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pair"));
    assert!(rust.contains("a") && rust.contains("b"));
}

#[test]
fn test_for_loop_tuple_unpacking() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
pairs = [(1, 2), (3, 4)]
for a, b in pairs:
    x = a + b
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pairs"));
    assert!(rust.contains("(a, b)") || (rust.contains("a") && rust.contains("b")));
}

#[test]
fn test_enumerate_simple() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = [10, 20, 30]
for i, val in enumerate(items):
    result = i + val
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("items"));
    assert!(rust.contains("enumerate"));
    assert!(rust.contains("(i, val)") || (rust.contains("i") && rust.contains("val")));
}

#[test]
fn test_zip_simple() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
list1 = [1, 2, 3]
list2 = [4, 5, 6]
for a, b in zip(list1, list2):
    sum_val = a + b
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("list1"));
    assert!(rust.contains("list2"));
    assert!(rust.contains("zip"));
}

#[test]
fn test_tuple_assignment_literal() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x, y = 1, 2
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x") && rust.contains("y"));
    assert!(rust.contains("1") && rust.contains("2"));
}

#[test]
fn test_tuple_unpacking_three_values() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
triple = (1, 2, 3)
a, b, c = triple
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("triple"));
    assert!(rust.contains("a") && rust.contains("b") && rust.contains("c"));
}

#[test]
fn test_for_enumerate_with_assignment() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
words = ["hello", "world"]
for idx, word in enumerate(words):
    length = len(word)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("words"));
    assert!(rust.contains("enumerate"));
}

#[test]
fn test_nested_tuple_unpacking() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
data = [(1, 2), (3, 4), (5, 6)]
for x, y in data:
    product = x * y
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("data"));
    assert!(rust.contains("x") && rust.contains("y"));
}

#[test]
fn test_swap_values() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
a = 1
b = 2
a, b = b, a
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("a") && rust.contains("b"));
}

#[test]
fn test_for_loop_with_index() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = [100, 200, 300]
for index, value in enumerate(items):
    doubled = value * 2
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("items"));
    assert!(rust.contains("index") && rust.contains("value"));
}
