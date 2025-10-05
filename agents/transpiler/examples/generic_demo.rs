//! Generic Type Support Examples
//!
//! Demonstrates translation of Python's generic types to Rust's generic system,
//! including type parameters, bounds, variance, and associated types.

use portalis_transpiler::generic_translator::*;

fn main() {
    println!("=== Generic Type Support Examples ===\n");

    // Example 1: Basic generic types
    example_basic_generics();

    // Example 2: Generic functions
    example_generic_functions();

    // Example 3: Generic classes/structs
    example_generic_structs();

    // Example 4: Type bounds and constraints
    example_type_bounds();

    // Example 5: Multiple type parameters
    example_multiple_params();

    // Example 6: Where clauses
    example_where_clauses();

    // Example 7: Associated types
    example_associated_types();

    // Example 8: Common generic patterns
    example_common_patterns();
}

fn example_basic_generics() {
    println!("## Example 1: Basic Generic Types\n");
    println!("Python typing module → Rust std types:\n");

    let translator = GenericTranslator::new();

    let type_mappings = vec![
        ("List[int]", "Python list of integers"),
        ("Dict[str, int]", "Python dictionary"),
        ("Set[str]", "Python set"),
        ("Optional[str]", "Optional string"),
        ("Tuple[int, str, bool]", "Tuple with multiple types"),
    ];

    println!("Python Type           →  Rust Type");
    println!("{}", "-".repeat(50));
    for (py_type, description) in type_mappings {
        let rust_type = translator.translate_type(py_type);
        println!("{:<20}  →  {}", py_type, rust_type);
    }

    println!("\nKey mappings:");
    println!("- List[T] → Vec<T>");
    println!("- Dict[K, V] → HashMap<K, V>");
    println!("- Set[T] → HashSet<T>");
    println!("- Optional[T] → Option<T>");
    println!("- Tuple[T, U, V] → (T, U, V)");

    println!("\n{}\n", "=".repeat(80));
}

fn example_generic_functions() {
    println!("## Example 2: Generic Functions\n");

    println!("Python:");
    println!(r#"
def identity[T](x: T) -> T:
    return x
"#);

    let mut translator = GenericTranslator::new();
    let mut func = translator.parse_generic_function("def identity[T](x: T) -> T:");
    func.params.push(("x".to_string(), "T".to_string()));
    func.return_type = Some("T".to_string());

    println!("Rust:");
    println!("{} {{", func.to_rust());
    println!("    x");
    println!("}}");
    println!();

    println!("Python:");
    println!(r#"
def first[T](items: List[T]) -> T:
    return items[0]
"#);

    let mut func = GenericFunction::new("first");
    func.type_params.push(TypeParam::new("T"));
    func.params.push(("items".to_string(), "Vec<T>".to_string()));
    func.return_type = Some("T".to_string());

    println!("Rust:");
    println!("{} {{", func.to_rust());
    println!("    items[0].clone()");
    println!("}}");

    println!("\n{}\n", "=".repeat(80));
}

fn example_generic_structs() {
    println!("## Example 3: Generic Classes/Structs\n");

    println!("Python:");
    println!(r#"
class Container[T]:
    def __init__(self):
        self.items: List[T] = []

    def add(self, item: T):
        self.items.append(item)
"#);

    let mut translator = GenericTranslator::new();
    let mut struct_def = translator.parse_generic_class("class Container[T]:");
    struct_def.fields.push(("items".to_string(), "Vec<T>".to_string()));
    struct_def.derives.push("Debug".to_string());
    struct_def.derives.push("Clone".to_string());

    println!("Rust:");
    print!("{}", struct_def.to_rust());
    println!();

    println!("impl<T> Container<T> {{");
    println!("    pub fn new() -> Self {{");
    println!("        Self {{ items: Vec::new() }}");
    println!("    }}");
    println!();
    println!("    pub fn add(&mut self, item: T) {{");
    println!("        self.items.push(item);");
    println!("    }}");
    println!("}}");

    println!("\n{}\n", "=".repeat(80));
}

fn example_type_bounds() {
    println!("## Example 4: Type Bounds and Constraints\n");

    println!("Python:");
    println!(r#"
from typing import Protocol

class Comparable(Protocol):
    def __lt__(self, other) -> bool: ...

def max_value[T: Comparable](a: T, b: T) -> T:
    return a if a > b else b
"#);

    let mut func = GenericFunction::new("max_value");
    func.type_params.push(
        TypeParam::new("T")
            .with_bound(TypeBound::Trait("Ord".to_string()))
            .with_bound(TypeBound::Trait("Clone".to_string()))
    );
    func.params.push(("a".to_string(), "T".to_string()));
    func.params.push(("b".to_string(), "T".to_string()));
    func.return_type = Some("T".to_string());

    println!("\nRust:");
    println!("{} {{", func.to_rust());
    println!("    if a > b {{ a }} else {{ b }}");
    println!("}}");
    println!();

    println!("Common trait bounds:");
    println!("- Comparable → Ord (total ordering)");
    println!("- Hashable → Hash (hash support)");
    println!("- Iterable → IntoIterator (iteration)");
    println!("- Clone → Clone (value cloning)");
    println!("- Copy → Copy (bitwise copy)");

    println!("\n{}\n", "=".repeat(80));
}

fn example_multiple_params() {
    println!("## Example 5: Multiple Type Parameters\n");

    println!("Python:");
    println!(r#"
class Pair[T, U]:
    def __init__(self, first: T, second: U):
        self.first = first
        self.second = second

    def swap(self) -> 'Pair[U, T]':
        return Pair(self.second, self.first)
"#);

    let mut struct_def = GenericStruct::new("Pair");
    struct_def.type_params.push(TypeParam::new("T"));
    struct_def.type_params.push(TypeParam::new("U"));
    struct_def.fields.push(("first".to_string(), "T".to_string()));
    struct_def.fields.push(("second".to_string(), "U".to_string()));
    struct_def.derives.push("Debug".to_string());
    struct_def.derives.push("Clone".to_string());

    println!("\nRust:");
    print!("{}", struct_def.to_rust());
    println!();

    println!("impl<T, U> Pair<T, U> {{");
    println!("    pub fn new(first: T, second: U) -> Self {{");
    println!("        Self {{ first, second }}");
    println!("    }}");
    println!();
    println!("    pub fn swap(self) -> Pair<U, T> {{");
    println!("        Pair::new(self.second, self.first)");
    println!("    }}");
    println!("}}");

    println!("\n{}\n", "=".repeat(80));
}

fn example_where_clauses() {
    println!("## Example 6: Where Clauses\n");

    println!("Where clauses are used for complex type constraints.\n");

    println!("Python:");
    println!(r#"
def process[T: Comparable & Hashable](items: List[T]) -> Dict[T, int]:
    """Count occurrences of each item"""
    ...
"#);

    let mut func = GenericFunction::new("process");
    func.type_params.push(TypeParam::new("T"));
    func.params.push(("items".to_string(), "Vec<T>".to_string()));
    func.return_type = Some("HashMap<T, usize>".to_string());

    func.where_clauses.push(
        WhereClause::new("T")
            .with_bound(TypeBound::Trait("Ord".to_string()))
            .with_bound(TypeBound::Trait("Hash".to_string()))
            .with_bound(TypeBound::Trait("Eq".to_string()))
    );

    println!("\nRust:");
    println!("{} {{", func.to_rust());
    println!("    let mut counts = HashMap::new();");
    println!("    for item in items {{");
    println!("        *counts.entry(item).or_insert(0) += 1;");
    println!("    }}");
    println!("    counts");
    println!("}}");
    println!();

    println!("Benefits of where clauses:");
    println!("- Cleaner syntax for complex bounds");
    println!("- Required for associated type constraints");
    println!("- Better readability for multiple bounds");

    println!("\n{}\n", "=".repeat(80));
}

fn example_associated_types() {
    println!("## Example 7: Associated Types\n");

    println!("Associated types allow traits to define placeholder types.\n");

    println!("Python (Protocol with associated type):");
    println!(r#"
from typing import Protocol, TypeVar

T = TypeVar('T')

class Container(Protocol):
    Item: type

    def get(self, index: int) -> Item:
        ...

    def len(self) -> int:
        ...
"#);

    let translator = GenericTranslator::new();
    let trait_def = translator.generate_associated_type_trait(
        "Container",
        &[("Item", vec![])],
        &[
            {
                let mut func = GenericFunction::new("get");
                func.params.push(("self".to_string(), "&self".to_string()));
                func.params.push(("index".to_string(), "usize".to_string()));
                func.return_type = Some("Option<&Self::Item>".to_string());
                func
            },
            {
                let mut func = GenericFunction::new("len");
                func.params.push(("self".to_string(), "&self".to_string()));
                func.return_type = Some("usize".to_string());
                func
            },
        ],
    );

    println!("\nRust:");
    print!("{}", trait_def);
    println!();

    println!("Implementation for Vec:");
    println!("impl<T> Container for Vec<T> {{");
    println!("    type Item = T;");
    println!();
    println!("    fn get(&self, index: usize) -> Option<&Self::Item> {{");
    println!("        self.get(index)");
    println!("    }}");
    println!();
    println!("    fn len(&self) -> usize {{");
    println!("        self.len()");
    println!("    }}");
    println!("}}");

    println!("Associated types vs generic parameters:");
    println!("- Associated: one Item type per implementation");
    println!("- Generic: can have multiple different types");
    println!("- Associated types reduce type annotations");

    println!("\n{}\n", "=".repeat(80));
}

fn example_common_patterns() {
    println!("## Example 8: Common Generic Patterns\n");

    println!("### Pattern 1: Generic Container");
    println!("{}", GenericPatterns::container_pattern());

    println!("\n### Pattern 2: Generic Pair");
    println!("{}", GenericPatterns::pair_pattern());

    println!("\n### Pattern 3: Associated Type Trait");
    let trait_code = GenericPatterns::associated_type_pattern();
    println!("{}", &trait_code[..trait_code.len().min(500)]);
    println!("...");

    println!("\n### Pattern 4: Bounded Function");
    println!("{}", GenericPatterns::bounded_function_pattern());

    println!("\n### Pattern 5: Result-like Enum");
    let result_code = GenericPatterns::result_pattern();
    println!("{}", &result_code[..result_code.len().min(400)]);
    println!("...");

    println!("\n### Pattern 6: PhantomData (Zero-sized Type Parameter)");
    println!("{}", GenericPatterns::phantom_data_pattern());

    println!("\nUse cases:");
    println!("- Container: collections, wrappers");
    println!("- Pair: tuples, key-value pairs");
    println!("- Associated types: iterators, containers");
    println!("- Bounded functions: comparisons, operations");
    println!("- Result: error handling, optional success");
    println!("- PhantomData: type-safe IDs, markers");

    println!("\n{}\n", "=".repeat(80));
}
