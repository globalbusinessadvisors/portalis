//! Day 28-29 Feature Tests: Additional Built-in Functions and Type Conversions
//!
//! Tests for:
//! - any() and all() functions
//! - Type conversion functions: int(), float(), str(), bool(), list(), dict()

use super::feature_translator::FeatureTranslator;

#[test]
fn test_any_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
values = [False, False, True, False]
result = any(values)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("values"));
    assert!(rust.contains("result"));
    assert!(rust.contains("any") || rust.contains(".iter()"));
}

#[test]
fn test_all_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
values = [True, True, True]
result = all(values)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("values"));
    assert!(rust.contains("result"));
    assert!(rust.contains("all") || rust.contains(".iter()"));
}

#[test]
fn test_any_with_condition() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
has_even = any([n % 2 == 0 for n in numbers])
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("has_even"));
}

#[test]
fn test_all_with_condition() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [2, 4, 6, 8]
all_even = all([n % 2 == 0 for n in numbers])
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("all_even"));
}

#[test]
fn test_int_conversion() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 3.14
y = int(x)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("y"));
    assert!(rust.contains("as i32") || rust.contains("int"));
}

#[test]
fn test_float_conversion() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 5
y = float(x)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("y"));
    assert!(rust.contains("as f64") || rust.contains("float"));
}

#[test]
fn test_str_conversion() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 42
text = str(x)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("text"));
    assert!(rust.contains("to_string") || rust.contains("str"));
}

#[test]
fn test_bool_conversion() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 1
flag = bool(x)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x"));
    assert!(rust.contains("flag"));
}

#[test]
fn test_list_constructor_empty() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = list()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("items"));
    assert!(rust.contains("Vec::new") || rust.contains("vec!"));
}

#[test]
fn test_list_from_range() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = list(range(5))
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("numbers"));
    assert!(rust.contains("collect") || rust.contains("Vec"));
}

#[test]
fn test_dict_constructor_empty() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
data = dict()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("data"));
    assert!(rust.contains("HashMap"));
}

#[test]
fn test_any_empty_list() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
empty = []
result = any(empty)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("empty"));
    assert!(rust.contains("result"));
}

#[test]
fn test_all_empty_list() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
empty = []
result = all(empty)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("empty"));
    assert!(rust.contains("result"));
}

#[test]
fn test_type_conversion_chain() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = "42"
y = int(x)
z = float(y)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("x") && rust.contains("y") && rust.contains("z"));
}

#[test]
fn test_list_from_enumerate() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = ["a", "b", "c"]
indexed = list(enumerate(items))
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("items"));
    assert!(rust.contains("indexed"));
    assert!(rust.contains("enumerate"));
}

#[test]
fn test_list_from_zip() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
a = [1, 2, 3]
b = [4, 5, 6]
pairs = list(zip(a, b))
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("a") && rust.contains("b"));
    assert!(rust.contains("pairs"));
    assert!(rust.contains("zip"));
}
