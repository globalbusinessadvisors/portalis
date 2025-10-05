//! Generator to Iterator Translation Examples
//!
//! Demonstrates how Python generators (functions with yield) are translated
//! to Rust iterators using the Iterator trait and state machines.

use portalis_transpiler::generator_translator::*;
use portalis_transpiler::python_ast::{PyExpr, PyLiteral, PyStmt, TypeAnnotation};

fn main() {
    println!("=== Python Generator → Rust Iterator Translation Examples ===\n");

    // Example 1: Simple generator with single yield
    example_simple_generator();

    // Example 2: Generator with multiple yields (state machine)
    example_multiple_yields();

    // Example 3: Range generator
    example_range_generator();

    // Example 4: Fibonacci generator
    example_fibonacci_generator();

    // Example 5: Generator expressions
    example_generator_expressions();

    // Example 6: Infinite generator
    example_infinite_generator();

    // Example 7: Generator with parameters
    example_parameterized_generator();

    // Example 8: Helper functions
    example_helper_functions();
}

fn example_simple_generator() {
    println!("## Example 1: Simple Generator (Single Yield)\n");
    println!("Python:");
    println!(r#"
def simple_gen():
    yield 42
"#);

    let mut translator = GeneratorTranslator::new();

    let info = GeneratorInfo {
        name: "simple_gen".to_string(),
        params: vec![],
        yield_type: Some(TypeAnnotation::Name("int".to_string())),
        body: vec![PyStmt::Expr(PyExpr::Yield(Some(Box::new(
            PyExpr::Literal(PyLiteral::Int(42)),
        ))))],
        is_async: false,
    };

    let rust_code = translator.translate_generator(&info);

    println!("\nRust (simple closure-based):");
    println!("{}", rust_code);
    println!("\nUsage:");
    println!("for value in simple_gen() {{");
    println!("    println!(\"{{:?}}\", value); // Prints: 42");
    println!("}}");

    println!("\n{}\n", "=".repeat(80));
}

fn example_multiple_yields() {
    println!("## Example 2: Generator with Multiple Yields\n");
    println!("Python:");
    println!(r#"
def count_to_three():
    yield 1
    yield 2
    yield 3
"#);

    let mut translator = GeneratorTranslator::new();

    let info = GeneratorInfo {
        name: "count_to_three".to_string(),
        params: vec![],
        yield_type: Some(TypeAnnotation::Name("int".to_string())),
        body: vec![
            PyStmt::Expr(PyExpr::Yield(Some(Box::new(PyExpr::Literal(
                PyLiteral::Int(1),
            ))))),
            PyStmt::Expr(PyExpr::Yield(Some(Box::new(PyExpr::Literal(
                PyLiteral::Int(2),
            ))))),
            PyStmt::Expr(PyExpr::Yield(Some(Box::new(PyExpr::Literal(
                PyLiteral::Int(3),
            ))))),
        ],
        is_async: false,
    };

    let rust_code = translator.translate_generator(&info);

    println!("\nRust (state machine):");
    println!("{}", rust_code);
    println!("\nNote: Uses state enum to track position between yields");
    println!("Each yield point becomes a state transition");

    println!("\n{}\n", "=".repeat(80));
}

fn example_range_generator() {
    println!("## Example 3: Range Generator\n");
    println!("Python:");
    println!(r#"
def my_range(start: int, end: int):
    i = start
    while i < end:
        yield i
        i += 1
"#);

    println!("\nRust (idiomatic):");
    println!(r#"pub fn my_range(start: i32, end: i32) -> impl Iterator<Item = i32> {{
    start..end
}}

// Or using iterator adapters:
pub fn my_range_step(start: i32, end: i32, step: i32) -> impl Iterator<Item = i32> {{
    (start..end).step_by(step as usize)
}}"#);

    println!("\nUsage:");
    println!("for i in my_range(0, 5) {{");
    println!("    println!(\"{{:?}}\", i);");
    println!("}}");
    println!("// Prints: 0, 1, 2, 3, 4");

    println!("\n{}\n", "=".repeat(80));
}

fn example_fibonacci_generator() {
    println!("## Example 4: Fibonacci Generator\n");
    println!("Python:");
    println!(r#"
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
"#);

    println!("\nRust (using state in struct):");
    println!(r#"#[derive(Debug)]
pub struct Fibonacci {{
    a: u64,
    b: u64,
}}

impl Iterator for Fibonacci {{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {{
        let current = self.a;
        let next = self.a + self.b;
        self.a = self.b;
        self.b = next;
        Some(current)
    }}
}}

pub fn fibonacci() -> Fibonacci {{
    Fibonacci {{ a: 0, b: 1 }}
}}

// With take() to limit:
pub fn fibonacci_n(n: usize) -> impl Iterator<Item = u64> {{
    fibonacci().take(n)
}}"#);

    println!("\nUsage:");
    println!("for fib in fibonacci().take(10) {{");
    println!("    println!(\"{{:?}}\", fib);");
    println!("}}");
    println!("// Prints: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34");

    println!("\n{}\n", "=".repeat(80));
}

fn example_generator_expressions() {
    println!("## Example 5: Generator Expressions\n");
    println!("Python:");
    println!(r#"
# Generator expression
squares = (x * x for x in range(10))

# Equivalent to:
def squares_gen():
    for x in range(10):
        yield x * x
"#);

    println!("\nRust (using iterator adapters):");
    println!(r#"// Generator expression → map()
let squares = (0..10).map(|x| x * x);

// Chain multiple operations
let even_squares = (0..10)
    .filter(|x| x % 2 == 0)
    .map(|x| x * x);

// Collect to vector if needed
let vec: Vec<_> = squares.collect();"#);

    println!("\nCommon patterns:");
    println!(r#"
// filter → .filter()
evens = (x for x in range(10) if x % 2 == 0)
→ (0..10).filter(|x| x % 2 == 0)

// map → .map()
doubled = (x * 2 for x in range(10))
→ (0..10).map(|x| x * 2)

// flat_map for nested
pairs = ((x, y) for x in range(3) for y in range(3))
→ (0..3).flat_map(|x| (0..3).map(move |y| (x, y)))
"#);

    println!("{}\n", "=".repeat(80));
}

fn example_infinite_generator() {
    println!("## Example 6: Infinite Generator\n");
    println!("Python:");
    println!(r#"
def count_from(start: int):
    i = start
    while True:
        yield i
        i += 1
"#);

    println!("\nRust:");
    println!(r#"pub fn count_from(start: i32) -> impl Iterator<Item = i32> {{
    start..
}}

// Or explicitly:
pub fn count_from_v2(start: i32) -> impl Iterator<Item = i32> {{
    std::iter::successors(Some(start), |&n| Some(n + 1))
}}

// With custom step:
pub fn count_by(start: i32, step: i32) -> impl Iterator<Item = i32> {{
    (0..).map(move |i| start + i * step)
}}"#);

    println!("\nUsage (with take to limit):");
    println!("for i in count_from(100).take(5) {{");
    println!("    println!(\"{{:?}}\", i);");
    println!("}}");
    println!("// Prints: 100, 101, 102, 103, 104");

    println!("\n{}\n", "=".repeat(80));
}

fn example_parameterized_generator() {
    println!("## Example 7: Generator with Parameters\n");
    println!("Python:");
    println!(r#"
def repeat_value(value: int, times: int):
    for _ in range(times):
        yield value
"#);

    let mut translator = GeneratorTranslator::new();

    let info = GeneratorInfo {
        name: "repeat_value".to_string(),
        params: vec![
            ("value".to_string(), Some(TypeAnnotation::Name("int".to_string()))),
            ("times".to_string(), Some(TypeAnnotation::Name("int".to_string()))),
        ],
        yield_type: Some(TypeAnnotation::Name("int".to_string())),
        body: vec![PyStmt::Expr(PyExpr::Yield(Some(Box::new(
            PyExpr::Name("value".to_string()),
        ))))],
        is_async: false,
    };

    let rust_code = translator.translate_generator(&info);

    println!("\nRust (generated):");
    println!("{}", rust_code);

    println!("\nIdiomatic Rust:");
    println!(r#"pub fn repeat_value(value: i32, times: i32) -> impl Iterator<Item = i32> {{
    std::iter::repeat(value).take(times as usize)
}}"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_helper_functions() {
    println!("## Example 8: Generator Helper Functions\n");
    println!("Common Rust iterator patterns for Python generators:\n");

    let helpers = generate_generator_helpers();
    println!("{}", helpers);

    println!("\nCommon Iterator Methods:");
    println!(r#"
// Transformation
.map(|x| x * 2)              // Apply function to each element
.filter(|x| x % 2 == 0)      // Keep only matching elements
.flat_map(|x| x..)           // Flatten nested iterators

// Combination
.zip(other_iter)             // Combine two iterators
.chain(other_iter)           // Concatenate iterators
.enumerate()                 // Add index (0, 1, 2, ...)

// Consumption
.collect::<Vec<_>>()         // Collect into vector
.fold(0, |acc, x| acc + x)   // Reduce/accumulate
.sum(), .product()           // Sum or multiply all elements

// Limiting
.take(n)                     // Take first n elements
.skip(n)                     // Skip first n elements
.take_while(|x| condition)   // Take while condition is true

// Inspection
.any(|x| condition)          // Check if any match
.all(|x| condition)          // Check if all match
.find(|x| condition)         // Find first matching element
"#);

    println!("{}\n", "=".repeat(80));
}
