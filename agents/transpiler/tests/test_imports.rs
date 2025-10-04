//! Integration tests for import system and attribute translation

use portalis_transpiler::feature_translator::FeatureTranslator;

#[test]
fn test_import_math_constant() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
import math
x = math.pi
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    // Should generate use statement
    assert!(rust.contains("use std::f64::consts"));

    // Should translate math.pi to std::f64::consts::PI
    assert!(rust.contains("std::f64::consts::PI") || rust.contains("PI"));
}

#[test]
fn test_import_math_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
import math
result = math.sqrt(16.0)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    // Should translate math.sqrt() properly
    assert!(rust.contains("sqrt"));
}

#[test]
fn test_from_import() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
from os import path
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    // Should generate use statement
    assert!(rust.contains("use"));
}
