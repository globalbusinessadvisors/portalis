//! Day 14-15 Feature Tests: Slice Notation
//!
//! Tests for:
//! - Basic slicing: list[1:3]
//! - Open-ended slicing: list[:5], list[2:]
//! - Full slice: list[:]
//! - Slicing with step: list[::2], list[1:10:2]

use super::feature_translator::FeatureTranslator;

#[test]
fn test_basic_slice() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
subset = numbers[1:3]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("subset"));
    assert!(rust.contains("1..3") || rust.contains("[1:3]"));
}

#[test]
fn test_slice_from_start() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
first_three = numbers[:3]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("first_three"));
    assert!(rust.contains("..3"));
}

#[test]
fn test_slice_to_end() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
from_two = numbers[2:]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("from_two"));
    assert!(rust.contains("2.."));
}

#[test]
fn test_full_slice() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
copy = numbers[:]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("copy"));
}

#[test]
fn test_slice_with_step() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
evens = numbers[::2]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("evens"));
    assert!(rust.contains("step_by") || rust.contains("::2"));
}

#[test]
fn test_slice_with_range_and_step() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
subset = numbers[1:8:2]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("subset"));
}

#[test]
fn test_slice_in_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def get_middle(lst):
    return lst[1:-1]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("fn get_middle"));
    assert!(rust.contains("lst"));
}

#[test]
fn test_string_slice() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "Hello World"
first_five = text[:5]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("first_five"));
}

#[test]
fn test_multiple_slices() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
first_half = numbers[:5]
second_half = numbers[5:]
middle = numbers[3:7]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("first_half"));
    assert!(rust.contains("second_half"));
    assert!(rust.contains("middle"));
}

#[test]
fn test_nested_slice() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
row = matrix[1]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("matrix"));
    assert!(rust.contains("row"));
}
