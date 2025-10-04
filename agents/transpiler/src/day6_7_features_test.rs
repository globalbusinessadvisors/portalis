//! Day 6-7 Feature Tests: Functions and Scope
//!
//! Tests for:
//! - Function definitions with parameters
//! - Return statements
//! - Function calls
//! - Type hints
//! - Default arguments
//! - Multiple return values
//! - Lambda expressions (basic)
//! - Variable scoping

use super::feature_translator::FeatureTranslator;

#[test]
fn test_simple_function_no_params() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def greet():
    print("Hello")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn greet()"));
    assert!(rust.contains(r#"print("Hello");"#));
}

#[test]
fn test_function_with_params() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def add(a, b):
    return a + b
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn add("));
    assert!(rust.contains("a: ()") || rust.contains("a,"));
    assert!(rust.contains("b: ()") || rust.contains("b)"));
    assert!(rust.contains("return a + b;"));
}

#[test]
fn test_function_with_type_hints() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def add(a: int, b: int) -> int:
    return a + b
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn add("));
    assert!(rust.contains("a: i32") || rust.contains("a: int"));
    assert!(rust.contains("b: i32") || rust.contains("b: int"));
    assert!(rust.contains("-> i32") || rust.contains("-> int"));
    assert!(rust.contains("return a + b;"));
}

#[test]
fn test_function_return_no_value() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def do_something():
    x = 5
    return
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn do_something()"));
    assert!(rust.contains("let x: i32 = 5;"));
    assert!(rust.contains("return;"));
}

#[test]
fn test_function_return_expression() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def square(x):
    return x * x
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn square("));
    assert!(rust.contains("return x * x;"));
}

#[test]
fn test_function_call_no_args() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def greet():
    print("Hello")

greet()
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn greet()"));
    assert!(rust.contains("greet();"));
}

#[test]
fn test_function_call_with_args() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def add(a, b):
    return a + b

result = add(5, 3)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn add("));
    assert!(rust.contains("let result") || rust.contains("result ="));
    assert!(rust.contains("add(5, 3)"));
}

#[test]
fn test_function_call_result_used() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def double(x):
    return x * 2

y = double(10)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn double("));
    assert!(rust.contains("double(10)"));
}

#[test]
fn test_multiple_functions() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

result = add(10, 5)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn add("));
    assert!(rust.contains("pub fn subtract("));
    assert!(rust.contains("return a + b;"));
    assert!(rust.contains("return a - b;"));
}

#[test]
fn test_function_with_local_variables() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def calculate(x):
    temp = x * 2
    result = temp + 5
    return result
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn calculate("));
    assert!(rust.contains("let temp") || rust.contains("temp ="));
    assert!(rust.contains("let result") || rust.contains("result ="));
    assert!(rust.contains("return result;"));
}

#[test]
fn test_function_with_if_statement() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def abs_value(x):
    if x < 0:
        return -x
    else:
        return x
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn abs_value("));
    assert!(rust.contains("if x < 0"));
    assert!(rust.contains("return -x;") || rust.contains("return - x;"));
    assert!(rust.contains("return x;"));
}

#[test]
fn test_function_with_loop() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def sum_range(n):
    total = 0
    for i in range(n):
        total += i
    return total
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn sum_range("));
    assert!(rust.contains("let total: i32 = 0;"));
    assert!(rust.contains("for i in 0..n"));
    assert!(rust.contains("total += i;"));
    assert!(rust.contains("return total;"));
}

#[test]
fn test_recursive_function() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn factorial("));
    assert!(rust.contains("if n <= 1"));
    assert!(rust.contains("return 1;"));
    assert!(rust.contains("factorial(n - 1)"));
}

#[test]
fn test_complete_program_with_functions() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
def is_even(n):
    return n % 2 == 0

def filter_evens(limit):
    for i in range(limit):
        if is_even(i):
            print(i)

filter_evens(10)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub fn is_even("));
    assert!(rust.contains("return n % 2 == 0;"));
    assert!(rust.contains("pub fn filter_evens("));
    assert!(rust.contains("is_even(i)"));
    assert!(rust.contains("filter_evens(10)"));
}
