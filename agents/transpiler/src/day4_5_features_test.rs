//! Day 4-5 Feature Tests: Control Flow and Additional Operators
//!
//! Tests for:
//! - Multi-line if/elif/else blocks
//! - For loops with range()
//! - While loops
//! - All arithmetic operators (-, *, /, %, **)
//! - Nested control structures

use super::feature_translator::FeatureTranslator;

#[test]
fn test_if_statement() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 10
if x > 5:
    y = 20
"#;
    let rust = translator.translate(python).unwrap();

    assert!(rust.contains("let x: i32 = 10;"));
    assert!(rust.contains("if x > 5"));
    assert!(rust.contains("let y: i32 = 20;"));
}

#[test]
fn test_if_else() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 10
if x > 5:
    y = 20
else:
    y = 0
"#;
    let rust = translator.translate(python).unwrap();

    assert!(rust.contains("if x > 5"));
    assert!(rust.contains("} else {"));
    assert!(rust.contains("let y: i32 = 20;"));
    assert!(rust.contains("let y: i32 = 0;"));
}

#[test]
fn test_if_elif_else() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 10
if x > 15:
    y = 30
elif x > 5:
    y = 20
else:
    y = 0
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("if x > 15"));
    assert!(rust.contains("if x > 5"));  // elif becomes nested if
    assert!(rust.contains("else {"));
}

#[test]
fn test_for_loop_range() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
total = 0
for i in range(5):
    total += i
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);

    assert!(rust.contains("let total: i32 = 0;"));
    assert!(rust.contains("for i in 0..5"));
    assert!(rust.contains("total += i;"));
}

#[test]
fn test_for_loop_range_start_stop() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
total = 0
for i in range(2, 8):
    total += i
"#;
    let rust = translator.translate(python).unwrap();

    assert!(rust.contains("for i in 2..8"));
}

#[test]
fn test_while_loop() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
x = 0
while x < 10:
    x += 1
"#;
    let rust = translator.translate(python).unwrap();

    assert!(rust.contains("let x: i32 = 0;"));
    assert!(rust.contains("while x < 10"));
    assert!(rust.contains("x += 1;"));
}

#[test]
fn test_subtraction_operator() {
    let mut translator = FeatureTranslator::new();
    let python = "result = 10 - 3";
    let rust = translator.translate(python).unwrap();

    assert!(rust.contains("let result: i32 = 10 - 3;"));
}

#[test]
fn test_multiplication_operator() {
    let mut translator = FeatureTranslator::new();
    let python = "result = 5 * 4";
    let rust = translator.translate(python).unwrap();

    assert!(rust.contains("let result: i32 = 5 * 4;"));
}

#[test]
fn test_division_operator() {
    let mut translator = FeatureTranslator::new();
    let python = "result = 20 / 4";
    let rust = translator.translate(python).unwrap();

    assert!(rust.contains("let result: f64 = 20 / 4;"));
}

#[test]
fn test_modulo_operator() {
    let mut translator = FeatureTranslator::new();
    let python = "result = 17 % 5";
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("let result: i32 = 17 % 5;"));
}

#[test]
fn test_nested_if_in_loop() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
for i in range(10):
    if i > 5:
        x = i
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("for i in 0..10"));
    assert!(rust.contains("if i > 5"));
    assert!(rust.contains("x = i;") || rust.contains("let x: i32 = i;"));
}

#[test]
fn test_nested_loops() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
for i in range(3):
    for j in range(2):
        x = i + j
"#;
    let rust = translator.translate(python).unwrap();

    assert!(rust.contains("for i in 0..3"));
    assert!(rust.contains("for j in 0..2"));
    assert!(rust.contains("let x: i32 = i + j;"));
}

#[test]
fn test_complete_program_fibonacci() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
# Fibonacci sequence
a = 0
b = 1
for i in range(10):
    print(a)
    temp = a + b
    a = b
    b = temp
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust code:\n{}", rust);

    assert!(rust.contains("let a: i32 = 0;"));
    assert!(rust.contains("let b: i32 = 1;"));
    assert!(rust.contains("for i in 0..10"));
    assert!(rust.contains("print(a);"));
    assert!(rust.contains("let temp: i32 = a + b;"));
}

#[test]
fn test_complete_program_fizzbuzz() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
for i in range(1, 16):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust code:\n{}", rust);

    assert!(rust.contains("for i in 1..16"));
    assert!(rust.contains("if i % 15 == 0"));
    assert!(rust.contains("if i % 3 == 0"));  // elif becomes nested if
    assert!(rust.contains(r#"print("FizzBuzz");"#));
    assert!(rust.contains(r#"print("Fizz");"#));
    assert!(rust.contains(r#"print("Buzz");"#));
}

#[test]
fn test_all_arithmetic_operators() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
a = 10 + 5
b = 10 - 5
c = 10 * 5
d = 10 / 5
e = 10 % 3
"#;
    let rust = translator.translate(python).unwrap();

    assert!(rust.contains("let a: i32 = 10 + 5;"));
    assert!(rust.contains("let b: i32 = 10 - 5;"));
    assert!(rust.contains("let c: i32 = 10 * 5;"));
    assert!(rust.contains("let d: f64 = 10 / 5;"));
    assert!(rust.contains("let e: i32 = 10 % 3;"));
}
