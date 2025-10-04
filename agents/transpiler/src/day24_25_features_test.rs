//! Day 24-25 Feature Tests: Enumerate and Zip
//!
//! Tests for:
//! - enumerate() function
//! - zip() function with two or more iterables
//! - Iteration patterns with enumerate and zip

use super::feature_translator::FeatureTranslator;

#[test]
fn test_enumerate_basic() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = ["a", "b", "c"]
for i, item in enumerate(items):
    print(i, item)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("items"));
    assert!(rust.contains("enumerate"));
}

#[test]
fn test_enumerate_assignment() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = [10, 20, 30]
indexed = list(enumerate(items))
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("items"));
    assert!(rust.contains("indexed"));
    assert!(rust.contains("enumerate"));
}

#[test]
fn test_zip_two_lists() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
names = ["Alice", "Bob"]
ages = [30, 25]
for name, age in zip(names, ages):
    print(name, age)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("names"));
    assert!(rust.contains("ages"));
    assert!(rust.contains("zip"));
}

#[test]
fn test_zip_assignment() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
list1 = [1, 2, 3]
list2 = [4, 5, 6]
paired = list(zip(list1, list2))
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("list1"));
    assert!(rust.contains("list2"));
    assert!(rust.contains("paired"));
    assert!(rust.contains("zip"));
}

#[test]
fn test_zip_three_lists() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
a = [1, 2]
b = [3, 4]
c = [5, 6]
result = list(zip(a, b, c))
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("a"));
    assert!(rust.contains("b"));
    assert!(rust.contains("c"));
    assert!(rust.contains("zip"));
}

#[test]
fn test_enumerate_in_list_comprehension() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = ["a", "b", "c"]
indexed = [(i, x) for i, x in enumerate(items)]
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("items"));
    assert!(rust.contains("indexed"));
}

#[test]
fn test_zip_with_range() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = ["a", "b", "c"]
for i, item in zip(range(len(items)), items):
    print(i, item)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("items"));
    assert!(rust.contains("zip") || rust.contains("enumerate"));
}

#[test]
fn test_enumerate_with_index() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
words = ["hello", "world"]
for index, word in enumerate(words):
    print(index)
    print(word)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("words"));
    assert!(rust.contains("index") || rust.contains("word"));
}

#[test]
fn test_zip_unequal_length() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
short = [1, 2]
long = [3, 4, 5, 6]
pairs = list(zip(short, long))
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("short"));
    assert!(rust.contains("long"));
    assert!(rust.contains("pairs"));
    assert!(rust.contains("zip"));
}

#[test]
fn test_enumerate_start_index() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = ["a", "b", "c"]
for i, item in enumerate(items):
    result = i + 1
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("items"));
    assert!(rust.contains("enumerate") || rust.contains("for"));
}
