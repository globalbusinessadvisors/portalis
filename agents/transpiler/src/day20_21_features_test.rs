//! Day 20-21 Feature Tests: Loop Else Clauses
//!
//! Tests for:
//! - for...else statements
//! - while...else statements
//! - Else clause execution only when loop completes normally (no break)

use super::feature_translator::FeatureTranslator;

#[test]
fn test_for_else_no_break() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
for i in range(3):
    print(i)
else:
    print("Loop completed")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("for i in"));
    assert!(rust.contains("Loop completed"));
    assert!(rust.contains("_loop_completed") || rust.contains("if"));
}

#[test]
fn test_for_else_with_break() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
for i in range(10):
    if i == 5:
        break
else:
    print("Never executed")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("for i in"));
    assert!(rust.contains("break"));
    assert!(rust.contains("Never executed"));
}

#[test]
fn test_while_else_no_break() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
i = 0
while i < 3:
    print(i)
    i += 1
else:
    print("Loop finished")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("while i < 3"));
    assert!(rust.contains("Loop finished"));
}

#[test]
fn test_while_else_with_break() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
i = 0
while i < 10:
    if i == 5:
        break
    i += 1
else:
    print("Not reached")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("while"));
    assert!(rust.contains("break"));
    assert!(rust.contains("Not reached"));
}

#[test]
fn test_for_else_search_pattern() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
numbers = [1, 2, 3, 4, 5]
for n in numbers:
    if n == 10:
        print("Found it!")
        break
else:
    print("Not found")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("for n in"));
    assert!(rust.contains("Found it"));
    assert!(rust.contains("Not found"));
}

#[test]
fn test_nested_for_with_else() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
for i in range(3):
    for j in range(3):
        print(i, j)
    else:
        print("Inner loop done")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("for i in"));
    assert!(rust.contains("for j in"));
    assert!(rust.contains("Inner loop done"));
}

#[test]
fn test_for_else_with_continue() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
for i in range(5):
    if i == 2:
        continue
    print(i)
else:
    print("Done")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("for i in"));
    assert!(rust.contains("continue"));
    assert!(rust.contains("Done"));
}

#[test]
fn test_while_else_countdown() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
count = 5
while count > 0:
    print(count)
    count -= 1
else:
    print("Blast off!")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("while count > 0"));
    assert!(rust.contains("Blast off"));
}

#[test]
fn test_for_else_empty_sequence() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
items = []
for item in items:
    print(item)
else:
    print("No items")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("for item in"));
    assert!(rust.contains("No items"));
}

#[test]
fn test_for_else_with_assignment_in_else() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
found = False
for i in range(5):
    if i == 10:
        found = True
        break
else:
    print("Setting default")
    found = False
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("for i in"));
    assert!(rust.contains("found"));
    assert!(rust.contains("Setting default"));
}
