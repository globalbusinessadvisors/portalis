//! Integration tests for the generic Python-to-Rust translator
//!
//! These tests verify end-to-end translation from Python source to Rust code

use portalis_transpiler::python_parser::PythonParser;
use portalis_transpiler::expression_translator::{ExpressionTranslator, TranslationContext};
use portalis_transpiler::statement_translator::StatementTranslator;

#[test]
fn test_translate_simple_function() {
    let source = r#"
def add(a: int, b: int) -> int:
    return a + b
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn add"));
    assert!(rust_code.contains("a: i64"));
    assert!(rust_code.contains("b: i64"));
    assert!(rust_code.contains("-> i64"));
    assert!(rust_code.contains("return a + b"));
}

#[test]
fn test_translate_fibonacci_function() {
    let source = r#"
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn fibonacci"));
    assert!(rust_code.contains("n: i64"));
    assert!(rust_code.contains("if n <= 1"));
    assert!(rust_code.contains("return n"));
    assert!(rust_code.contains("fibonacci(n - 1) + fibonacci(n - 2)"));
}

#[test]
fn test_translate_for_loop_with_range() {
    let source = r#"
def sum_range(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn sum_range"));
    assert!(rust_code.contains("let mut total = 0"));
    assert!(rust_code.contains("for i in"));
    assert!(rust_code.contains("0.."));
    assert!(rust_code.contains("total += i"));
}

#[test]
fn test_translate_list_operations() {
    let source = r#"
def process_list(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn process_list"));
    assert!(rust_code.contains("vec![]"));
    assert!(rust_code.contains("for item in"));
    assert!(rust_code.contains("if item > 0"));
    assert!(rust_code.contains(".push("));
}

#[test]
fn test_translate_string_operations() {
    let source = r#"
def greet(name: str) -> str:
    greeting = "Hello, "
    return greeting + name
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn greet"));
    assert!(rust_code.contains("name: String"));
    assert!(rust_code.contains("-> String"));
    assert!(rust_code.contains("\"Hello, \""));
    // String concatenation is translated to format!
    assert!(rust_code.contains("format!") || rust_code.contains("+"));
}

#[test]
fn test_translate_dict_operations() {
    let source = r#"
def create_mapping():
    data = {"a": 1, "b": 2, "c": 3}
    return data
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn create_mapping"));
    assert!(rust_code.contains("HashMap"));
}

#[test]
fn test_translate_conditional_expression() {
    let source = r#"
def max_val(a: int, b: int) -> int:
    return a if a > b else b
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn max_val"));
    assert!(rust_code.contains("if a > b { a } else { b }"));
}

#[test]
fn test_translate_list_comprehension() {
    let source = r#"
def squares(n: int):
    return [x * x for x in range(n)]
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn squares"));
    assert!(rust_code.contains(".map("));
    assert!(rust_code.contains("collect"));
}

#[test]
fn test_translate_multiple_functions() {
    let source = r#"
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

def multiply(a: int, b: int) -> int:
    return a * b
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let mut rust_code = String::new();
    for stmt in &module.statements {
        rust_code.push_str(&translator.translate(stmt).expect("Failed to translate"));
        rust_code.push('\n');
    }

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn add"));
    assert!(rust_code.contains("pub fn subtract"));
    assert!(rust_code.contains("pub fn multiply"));
    assert!(rust_code.contains("return a + b"));
    assert!(rust_code.contains("return a - b"));
    assert!(rust_code.contains("return a * b"));
}

#[test]
fn test_translate_nested_if_statements() {
    let source = r#"
def classify(x: int) -> str:
    if x > 0:
        if x > 100:
            return "large positive"
        else:
            return "small positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn classify"));
    assert!(rust_code.contains("if x > 0"));
    assert!(rust_code.contains("if x > 100"));
    assert!(rust_code.contains("else if x < 0"));
    assert!(rust_code.contains("\"large positive\""));
    assert!(rust_code.contains("\"small positive\""));
    assert!(rust_code.contains("\"negative\""));
    assert!(rust_code.contains("\"zero\""));
}

#[test]
fn test_translate_while_loop() {
    let source = r#"
def countdown(n: int) -> int:
    while n > 0:
        n -= 1
    return n
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn countdown"));
    assert!(rust_code.contains("while n > 0"));
    assert!(rust_code.contains("n -= 1"));
}

#[test]
fn test_translate_builtin_functions() {
    let source = r#"
def demo():
    numbers = [1, 2, 3, 4, 5]
    length = len(numbers)
    total = sum(numbers)
    maximum = max(numbers)
    minimum = min(numbers)
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn demo"));
    assert!(rust_code.contains(".len()"));
    assert!(rust_code.contains(".sum()") || rust_code.contains("iter().sum()"));
}

#[test]
fn test_translate_lambda() {
    let source = r#"
def create_adder(x: int):
    return lambda y: x + y
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn create_adder"));
    assert!(rust_code.contains("|y| x + y"));
}

#[test]
fn test_translate_boolean_operators() {
    let source = r#"
def check(a: bool, b: bool, c: bool) -> bool:
    return a and b or c
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn check"));
    assert!(rust_code.contains("&&") || rust_code.contains("||"));
}

#[test]
fn test_translate_complex_expression() {
    let source = r#"
def complex_calc(x: int, y: int) -> int:
    return (x + y) * (x - y) + x ** 2
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn complex_calc"));
    assert!(rust_code.contains("+"));
    assert!(rust_code.contains("-"));
    assert!(rust_code.contains("*"));
    assert!(rust_code.contains("powf") || rust_code.contains("**"));
}

#[test]
fn test_translate_real_world_function() {
    let source = r#"
def process_data(items: list, threshold: int) -> dict:
    result = {}
    count = 0

    for item in items:
        if item > threshold:
            result[count] = item * 2
            count += 1

    return result
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let rust_code = translator.translate(&module.statements[0]).expect("Failed to translate");

    println!("Generated Rust code:\n{}", rust_code);

    assert!(rust_code.contains("pub fn process_data"));
    assert!(rust_code.contains("HashMap") || rust_code.contains("result"));
    assert!(rust_code.contains("for item in"));
    assert!(rust_code.contains("if item > threshold"));
}
