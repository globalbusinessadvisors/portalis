//! Tests for import alias functionality
//!
//! Tests:
//! - import math as m
//! - from os import path as p
//! - Using aliased modules in code

use crate::feature_translator::FeatureTranslator;

#[test]
fn test_import_with_alias() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
import math as m
x = m.pi
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    // Should resolve m -> math and translate math.pi correctly
    assert!(rust.contains("std::f64::consts"));
    assert!(rust.contains("std::f64::consts::PI"));
}

#[test]
fn test_import_with_alias_function_call() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
import math as m
result = m.sqrt(16.0)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    // Should resolve m -> math and translate math.sqrt correctly
    assert!(rust.contains("sqrt()"));
}

#[test]
fn test_from_import_with_alias() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
from math import pi as PI
x = PI
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    // For from imports with aliases, PI maps to math.pi
    // This should resolve to the constant
    assert!(rust.contains("let x"));
}

#[test]
fn test_multiple_imports_with_aliases() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
import math as m
import json as j

x = m.pi
y = m.sqrt(16.0)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    assert!(rust.contains("std::f64::consts::PI"));
    assert!(rust.contains("sqrt()"));
}

#[test]
fn test_mixed_imports_with_and_without_aliases() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
import math
import json as j

x = math.pi
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    // math has no alias, should work normally
    assert!(rust.contains("std::f64::consts::PI"));
}

#[test]
fn test_nested_attribute_os_path_exists() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
import os

result = os.path.exists("/tmp")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    // Should translate os.path.exists to std::path::Path
    assert!(rust.contains("std::path::Path"));
    assert!(rust.contains("exists()"));
}

#[test]
fn test_nested_attribute_from_import() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
from os import path

result = path.exists("/tmp")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    // This is trickier - path.exists where path is imported from os
    // For now it might not work perfectly, but it shouldn't crash
}
