//! Standalone demonstration of the generic Python-to-Rust translator
//!
//! This example shows how to use the new generic translator system to convert
//! Python code to Rust, handling any Python expression or statement.
//!
//! Run with: cargo run --example generic_translator_demo

use portalis_transpiler::python_parser::PythonParser;
use portalis_transpiler::expression_translator::{ExpressionTranslator, TranslationContext};
use portalis_transpiler::statement_translator::StatementTranslator;

fn main() {
    println!("=== Generic Python-to-Rust Translator Demo ===\n");

    // Example 1: Simple function
    demo_simple_function();

    // Example 2: Fibonacci
    demo_fibonacci();

    // Example 3: List operations
    demo_list_operations();

    // Example 4: Complex expressions
    demo_complex_expressions();

    // Example 5: Class definition
    demo_class_definition();
}

fn demo_simple_function() {
    println!("--- Example 1: Simple Function ---");

    let python_code = r#"
def add(a: int, b: int) -> int:
    return a + b
"#;

    match translate_python(python_code) {
        Ok(rust_code) => {
            println!("Python:");
            println!("{}", python_code);
            println!("Rust:");
            println!("{}", rust_code);
        }
        Err(e) => println!("Translation error: {}", e),
    }

    println!();
}

fn demo_fibonacci() {
    println!("--- Example 2: Fibonacci ---");

    let python_code = r#"
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"#;

    match translate_python(python_code) {
        Ok(rust_code) => {
            println!("Python:");
            println!("{}", python_code);
            println!("Rust:");
            println!("{}", rust_code);
        }
        Err(e) => println!("Translation error: {}", e),
    }

    println!();
}

fn demo_list_operations() {
    println!("--- Example 3: List Operations ---");

    let python_code = r#"
def process_numbers(numbers):
    result = []
    for n in numbers:
        if n > 0:
            result.append(n * 2)
    return result
"#;

    match translate_python(python_code) {
        Ok(rust_code) => {
            println!("Python:");
            println!("{}", python_code);
            println!("Rust:");
            println!("{}", rust_code);
        }
        Err(e) => println!("Translation error: {}", e),
    }

    println!();
}

fn demo_complex_expressions() {
    println!("--- Example 4: Complex Expressions ---");

    let python_code = r#"
def calculate(x: int, y: int) -> int:
    squares = [i * i for i in range(10)]
    result = x if x > y else y
    total = sum(squares)
    return result + total
"#;

    match translate_python(python_code) {
        Ok(rust_code) => {
            println!("Python:");
            println!("{}", python_code);
            println!("Rust:");
            println!("{}", rust_code);
        }
        Err(e) => println!("Translation error: {}", e),
    }

    println!();
}

fn demo_class_definition() {
    println!("--- Example 5: Class Definition ---");

    let python_code = r#"
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b
"#;

    match translate_python(python_code) {
        Ok(rust_code) => {
            println!("Python:");
            println!("{}", python_code);
            println!("Rust:");
            println!("{}", rust_code);
        }
        Err(e) => println!("Translation error: {}", e),
    }

    println!();
}

/// Helper function to translate Python code to Rust
fn translate_python(python_code: &str) -> Result<String, String> {
    // Parse Python code
    let parser = PythonParser::new(python_code, "<demo>");
    let module = parser.parse().map_err(|e| format!("Parse error: {}", e))?;

    // Translate to Rust
    let mut ctx = TranslationContext::new();
    let mut translator = StatementTranslator::new(&mut ctx);

    let mut rust_code = String::new();
    for stmt in &module.statements {
        rust_code.push_str(&translator.translate(stmt).map_err(|e| format!("Translation error: {}", e))?);
    }

    Ok(rust_code)
}
