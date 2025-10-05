//! Decorator Translation Examples
//!
//! Demonstrates how Python decorators are translated to Rust using
//! various patterns including wrapper functions, macros, and attributes.

use portalis_transpiler::decorator_translator::*;
use portalis_transpiler::python_ast::{PyExpr, PyLiteral, TypeAnnotation};

fn main() {
    println!("=== Python Decorator → Rust Translation Examples ===\n");

    // Example 1: Simple decorators
    example_simple_decorators();

    // Example 2: Decorators with arguments
    example_decorators_with_args();

    // Example 3: Stacked decorators
    example_stacked_decorators();

    // Example 4: Property decorators
    example_property_decorators();

    // Example 5: Timing decorator
    example_timing_decorator();

    // Example 6: Caching decorator
    example_caching_decorator();

    // Example 7: Retry decorator
    example_retry_decorator();

    // Example 8: Helper functions
    example_helper_functions();
}

fn example_simple_decorators() {
    println!("## Example 1: Simple Decorators\n");
    println!("Python:");
    println!(r#"
@staticmethod
def calculate(x: int, y: int) -> int:
    return x + y

@property
def value(self) -> int:
    return self._value
"#);

    let translator = DecoratorTranslator::new();

    // Test @staticmethod
    let static_decorator = PyExpr::Name("staticmethod".to_string());
    let static_rust = translator.translate_decorator(&static_decorator);

    // Test @property
    let property_decorator = PyExpr::Name("property".to_string());
    let property_rust = translator.translate_decorator(&property_decorator);

    println!("\nRust:");
    println!("{}", static_rust);
    println!("pub fn calculate(x: i32, y: i32) -> i32 {{");
    println!("    x + y");
    println!("}}\n");

    println!("{}", property_rust);
    println!("pub fn value(&self) -> i32 {{");
    println!("    self._value");
    println!("}}");

    println!("\n{}\n", "=".repeat(80));
}

fn example_decorators_with_args() {
    println!("## Example 2: Decorators with Arguments\n");
    println!("Python:");
    println!(r#"
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@retry(max_attempts=3, delay=0.1)
def fetch_data(url: str) -> str:
    return requests.get(url).text
"#);

    let translator = DecoratorTranslator::new();

    // Test @lru_cache(maxsize=128)
    let cache_decorator = PyExpr::Call {
        func: Box::new(PyExpr::Name("lru_cache".to_string())),
        args: vec![],
        kwargs: vec![("maxsize".to_string(), PyExpr::Literal(PyLiteral::Int(128)))].into_iter().collect(),
    };
    let cache_rust = translator.translate_decorator(&cache_decorator);

    println!("\nRust:");
    println!("{}", cache_rust);
    println!("// Use lazy_static + HashMap or the cached crate");
    println!("pub fn fibonacci(n: i32) -> i32 {{");
    println!("    if n < 2 {{ return n; }}");
    println!("    fibonacci(n-1) + fibonacci(n-2)");
    println!("}}");

    println!("\n{}\n", "=".repeat(80));
}

fn example_stacked_decorators() {
    println!("## Example 3: Stacked Decorators\n");
    println!("Python:");
    println!(r#"
@log
@timeit
@retry
def complex_operation(data: list) -> dict:
    # Process data
    return {{"result": len(data)}}
"#);

    let translator = DecoratorTranslator::new();

    // Stacked decorators are applied bottom-up
    let decorators = vec![
        PyExpr::Name("log".to_string()),
        PyExpr::Name("timeit".to_string()),
        PyExpr::Name("retry".to_string()),
    ];

    println!("\nRust (decorators applied bottom-up):");
    for decorator in decorators.iter().rev() {
        let rust = translator.translate_decorator(decorator);
        println!("{}", rust);
    }

    println!("pub fn complex_operation(data: Vec<T>) -> HashMap<String, usize> {{");
    println!("    // Wrapped with retry, then timing, then logging");
    println!("    HashMap::from([(\"result\".to_string(), data.len())])");
    println!("}}");

    println!("\n{}\n", "=".repeat(80));
}

fn example_property_decorators() {
    println!("## Example 4: Property Decorators\n");
    println!("Python:");
    println!(r#"
class Person:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @classmethod
    def from_dict(cls, data: dict) -> 'Person':
        return cls(data['name'])
"#);

    let translator = DecoratorTranslator::new();

    let property_decorator = PyExpr::Name("property".to_string());
    let classmethod_decorator = PyExpr::Name("classmethod".to_string());

    let property_rust = translator.translate_decorator(&property_decorator);
    let classmethod_rust = translator.translate_decorator(&classmethod_decorator);

    println!("\nRust:");
    println!("pub struct Person {{");
    println!("    _name: String,");
    println!("}}\n");
    println!("impl Person {{");
    println!("    pub fn new(name: String) -> Self {{");
    println!("        Self {{ _name: name }}");
    println!("    }}\n");
    println!("    {}", property_rust);
    println!("    pub fn name(&self) -> &str {{");
    println!("        &self._name");
    println!("    }}\n");
    println!("    pub fn set_name(&mut self, value: String) {{");
    println!("        self._name = value;");
    println!("    }}\n");
    println!("    {}", classmethod_rust);
    println!("    pub fn from_dict(data: HashMap<String, String>) -> Self {{");
    println!("        Self::new(data[\"name\"].clone())");
    println!("    }}");
    println!("}}");

    println!("\n{}\n", "=".repeat(80));
}

fn example_timing_decorator() {
    println!("## Example 5: Timing Decorator with Wrapper\n");
    println!("Python:");
    println!(r#"
@timeit
def slow_function(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total
"#);

    let translator = DecoratorTranslator::new();

    let params = vec![
        ("n".to_string(), Some(TypeAnnotation::Name("int".to_string()))),
    ];
    let return_type = Some(TypeAnnotation::Name("int".to_string()));
    let body = r#"let mut total = 0;
for i in 0..n {
    total += i;
}
total"#;

    let wrapper = translator.generate_wrapper_function(
        "timeit",
        "slow_function",
        &params,
        &return_type,
        body,
    );

    println!("\nRust (with timing wrapper):");
    println!("{}", wrapper);

    println!("\n{}\n", "=".repeat(80));
}

fn example_caching_decorator() {
    println!("## Example 6: Caching Decorator\n");
    println!("Python:");
    println!(r#"
@lru_cache
def expensive_computation(x: int, y: int) -> int:
    # Simulate expensive operation
    return x ** y
"#);

    let translator = DecoratorTranslator::new();

    let params = vec![
        ("x".to_string(), Some(TypeAnnotation::Name("int".to_string()))),
        ("y".to_string(), Some(TypeAnnotation::Name("int".to_string()))),
    ];
    let return_type = Some(TypeAnnotation::Name("int".to_string()));
    let body = "x.pow(y as u32)";

    let wrapper = translator.generate_wrapper_function(
        "lru_cache",
        "expensive_computation",
        &params,
        &return_type,
        body,
    );

    println!("\nRust (with caching wrapper):");
    println!("{}", wrapper);

    println!("\n{}\n", "=".repeat(80));
}

fn example_retry_decorator() {
    println!("## Example 7: Retry Decorator\n");
    println!("Python:");
    println!(r#"
@retry
def unstable_operation(data: str) -> bool:
    # May fail occasionally
    return process(data)
"#);

    let translator = DecoratorTranslator::new();

    let params = vec![
        ("data".to_string(), Some(TypeAnnotation::Name("str".to_string()))),
    ];
    let return_type = Some(TypeAnnotation::Name("bool".to_string()));
    let body = "process(data)";

    let wrapper = translator.generate_wrapper_function(
        "retry",
        "unstable_operation",
        &params,
        &return_type,
        body,
    );

    println!("\nRust (with retry wrapper):");
    println!("{}", wrapper);

    println!("\n{}\n", "=".repeat(80));
}

fn example_helper_functions() {
    println!("## Example 8: Decorator Helper Functions\n");
    println!("Python decorators often require helper functions in Rust.\n");

    let helpers = generate_decorator_helpers();

    println!("Rust helper functions:");
    println!("{}", helpers);

    println!("Usage example:");
    println!(r#"
// Use helper for timing
pub fn my_function() -> i32 {{
    time_function("my_function", || {{
        // Your code here
        42
    }})
}}

// Use helper for logging
pub fn another_function(x: i32) -> i32 {{
    log_function("another_function", || {{
        x * 2
    }})
}}
"#);

    println!("{}\n", "=".repeat(80));
}
