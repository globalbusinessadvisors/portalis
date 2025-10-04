//! Day 10-11 Feature Tests: Advanced Data Structures and Operations
//!
//! Tests for:
//! - String concatenation
//! - Dictionary literals and operations
//! - List comprehensions (basic)
//! - More built-in functions (len, range with step)
//! - Import statements

use super::feature_translator::FeatureTranslator;

#[test]
fn test_string_concatenation() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
greeting = "Hello" + " " + "World"
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("greeting"));
    assert!(rust.contains("Hello") && rust.contains("World"));
}

#[test]
fn test_string_concatenation_with_variables() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
first = "Hello"
second = "World"
message = first + " " + second
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("first") && rust.contains("second"));
    assert!(rust.contains("message"));
}

#[test]
fn test_dictionary_literal_empty() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
d = {}
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("HashMap") || rust.contains("{}"));
}

#[test]
fn test_dictionary_literal_with_values() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
person = {"name": "Alice", "age": 30}
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("person"));
    assert!(rust.contains("Alice") && rust.contains("30"));
}

#[test]
fn test_dictionary_access() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
person = {"name": "Alice"}
name = person["name"]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("person"));
    assert!(rust.contains("name"));
}

#[test]
fn test_list_comprehension_simple() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
squares = [x * x for x in range(5)]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("squares"));
    // Should contain some form of iteration
    assert!(rust.contains("for") || rust.contains("map") || rust.contains("collect"));
}

#[test]
fn test_list_comprehension_with_condition() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
evens = [x for x in range(10) if x % 2 == 0]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("evens"));
    assert!(rust.contains("if") || rust.contains("filter"));
}

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
    assert!(rust.contains("len") || rust.contains("length"));
}

#[test]
fn test_range_with_step() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
for i in range(0, 10, 2):
    print(i)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("for i in"));
    // Should handle step parameter
    assert!(rust.contains("step") || rust.contains("..") || rust.contains("2"));
}

#[test]
fn test_multiple_assignment() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = y = z = 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x") && rust.contains("y") && rust.contains("z"));
}

#[test]
fn test_augmented_assignment_all_operators() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 10
x += 5
x -= 2
x *= 3
x /= 2
x %= 4
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x += 5"));
    assert!(rust.contains("x -= 2"));
    assert!(rust.contains("x *= 3"));
    assert!(rust.contains("x /= 2"));
    assert!(rust.contains("x %= 4"));
}

#[test]
fn test_string_methods() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "hello"
upper = text.upper()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("upper"));
}

#[test]
fn test_list_append() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3]
numbers.append(4)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("append") || rust.contains("push"));
}
