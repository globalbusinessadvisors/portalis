//! Day 22-23 Feature Tests: String Methods
//!
//! Tests for:
//! - Case conversion: upper(), lower()
//! - Whitespace: strip(), lstrip(), rstrip()
//! - Splitting and joining: split(), join()
//! - Search: find(), count(), startswith(), endswith()
//! - Modification: replace()

use super::feature_translator::FeatureTranslator;

#[test]
fn test_string_upper() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "hello"
upper_text = text.upper()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("upper_text"));
    assert!(rust.contains("to_uppercase"));
}

#[test]
fn test_string_lower() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "HELLO"
lower_text = text.lower()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("lower_text"));
    assert!(rust.contains("to_lowercase"));
}

#[test]
fn test_string_strip() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "  hello  "
stripped = text.strip()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("stripped"));
    assert!(rust.contains("trim"));
}

#[test]
fn test_string_lstrip() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "  hello"
left_stripped = text.lstrip()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("left_stripped"));
    assert!(rust.contains("trim_start"));
}

#[test]
fn test_string_rstrip() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "hello  "
right_stripped = text.rstrip()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("right_stripped"));
    assert!(rust.contains("trim_end"));
}

#[test]
fn test_string_split_no_args() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "hello world foo"
words = text.split()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("words"));
    assert!(rust.contains("split") || rust.contains("split_whitespace"));
}

#[test]
fn test_string_split_with_delimiter() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "a,b,c"
parts = text.split(",")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("parts"));
    assert!(rust.contains("split"));
}

#[test]
fn test_string_join() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
words = ["hello", "world"]
text = " ".join(words)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("words"));
    assert!(rust.contains("text"));
    assert!(rust.contains("join"));
}

#[test]
fn test_string_replace() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "hello world"
new_text = text.replace("world", "Rust")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("new_text"));
    assert!(rust.contains("replace"));
}

#[test]
fn test_string_startswith() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "hello world"
result = text.startswith("hello")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("result"));
    assert!(rust.contains("starts_with"));
}

#[test]
fn test_string_endswith() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "hello world"
result = text.endswith("world")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("result"));
    assert!(rust.contains("ends_with"));
}

#[test]
fn test_string_find() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "hello world"
index = text.find("world")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("index"));
    assert!(rust.contains("find"));
}

#[test]
fn test_string_count() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "hello hello"
count = text.count("hello")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("count"));
    assert!(rust.contains("matches") || rust.contains("count"));
}

#[test]
fn test_chained_string_methods() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "  HELLO  "
result = text.strip().lower()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("text"));
    assert!(rust.contains("result"));
}

#[test]
fn test_multiple_string_operations() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
text = "Hello World"
upper = text.upper()
lower = text.lower()
stripped = text.strip()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("upper"));
    assert!(rust.contains("lower"));
    assert!(rust.contains("stripped"));
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
    assert!(rust.contains("push") || rust.contains("append"));
}

#[test]
fn test_list_pop() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3]
last = numbers.pop()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("last"));
    assert!(rust.contains("pop"));
}

#[test]
fn test_list_sort() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [3, 1, 2]
numbers.sort()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("sort"));
}

#[test]
fn test_list_reverse() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3]
numbers.reverse()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("reverse"));
}
