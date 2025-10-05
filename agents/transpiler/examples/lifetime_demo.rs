//! Lifetime Analysis and Insertion Examples
//!
//! Demonstrates how Python code with references is translated to Rust with
//! proper lifetime annotations, elision, and borrow checking.

use portalis_transpiler::lifetime_analysis::*;

fn main() {
    println!("=== Lifetime Analysis and Insertion Examples ===\n");

    // Example 1: Basic lifetime inference
    example_basic_lifetime();

    // Example 2: Lifetime elision
    example_lifetime_elision();

    // Example 3: Struct with references
    example_struct_with_references();

    // Example 4: Multiple lifetimes
    example_multiple_lifetimes();

    // Example 5: Lifetime bounds
    example_lifetime_bounds();

    // Example 6: Common lifetime patterns
    example_common_patterns();

    // Example 7: Python to Rust lifetime mapping
    example_python_to_rust();

    // Example 8: Complex lifetime scenarios
    example_complex_lifetimes();
}

fn example_basic_lifetime() {
    println!("## Example 1: Basic Lifetime Inference\n");
    println!("Python:");
    println!(r#"
def first_word(s):
    """Returns reference to first word in string"""
    return s.split()[0]
"#);

    let mut analyzer = LifetimeAnalyzer::new();

    // Analyze function that returns a reference
    let params = vec![
        ("s".to_string(), TypeWithLifetime::reference("str", false, false)),
    ];
    let return_type = Some(TypeWithLifetime::reference("str", false, false));

    let sig = analyzer.analyze_function("first_word", params, return_type);

    println!("\nRust:");
    println!("{}\n", sig.to_rust());

    println!("Analysis:");
    println!("- Parameter 's' is a reference (&str)");
    println!("- Return type is also a reference (&str)");
    println!("- Lifetime 'a connects input to output");
    println!("- Compiler can verify the returned reference is valid");

    println!("\n{}\n", "=".repeat(80));
}

fn example_lifetime_elision() {
    println!("## Example 2: Lifetime Elision\n");
    println!("Rust has lifetime elision rules that allow omitting lifetime annotations in common cases.\n");

    let mut analyzer = LifetimeAnalyzer::new();

    println!("### Rule 1: Single input reference");
    let params = vec![
        ("x".to_string(), TypeWithLifetime::reference("i32", false, false)),
    ];
    let return_type = Some(TypeWithLifetime::reference("i32", false, false));
    let sig = analyzer.analyze_function("get_ref", params, return_type);

    println!("Without elision: {}", sig.to_rust());
    println!("With elision:    fn get_ref(x: &i32) -> &i32");
    println!();

    println!("### Rule 2: Multiple inputs - need explicit lifetimes");
    let params = vec![
        ("x".to_string(), TypeWithLifetime::reference("i32", false, false)),
        ("y".to_string(), TypeWithLifetime::reference("i32", false, false)),
    ];
    let return_type = Some(TypeWithLifetime::reference("i32", false, false));
    let sig = analyzer.analyze_function("choose", params, return_type);

    println!("{}", sig.to_rust());
    println!("Cannot use elision - ambiguous which lifetime to use");
    println!();

    println!("### Rule 3: Method with &self");
    let params = vec![
        ("self".to_string(), TypeWithLifetime::reference("Self", false, false)),
    ];
    let return_type = Some(TypeWithLifetime::reference("str", false, false));
    let sig = analyzer.analyze_function("name", params, return_type);

    println!("Without elision: {}", sig.to_rust());
    println!("With elision:    fn name(&self) -> &str");

    println!("\n{}\n", "=".repeat(80));
}

fn example_struct_with_references() {
    println!("## Example 3: Struct with References\n");
    println!("Python:");
    println!(r#"
class Buffer:
    def __init__(self, data):
        self.data = data  # Reference to external data
        self.size = len(data)
"#);

    let mut analyzer = LifetimeAnalyzer::new();

    let fields = vec![
        ("data".to_string(), TypeWithLifetime::reference("str", false, false)),
        ("size".to_string(), TypeWithLifetime::value("usize")),
    ];

    let struct_def = analyzer.analyze_struct("Buffer", fields);

    println!("\nRust:");
    println!("{}", struct_def.to_rust());
    println!();

    println!("Features:");
    println!("- Struct has lifetime parameter 'a");
    println!("- Field 'data' uses lifetime 'a");
    println!("- Field 'size' is a value, no lifetime needed");
    println!("- All instances must declare lifetime: Buffer<'a>");

    println!("\n{}\n", "=".repeat(80));
}

fn example_multiple_lifetimes() {
    println!("## Example 4: Multiple Lifetimes\n");
    println!("Python:");
    println!(r#"
def combine(first, second, use_first):
    """Returns either first or second based on flag"""
    if use_first:
        return first
    else:
        return second
"#);

    let mut analyzer = LifetimeAnalyzer::new();

    let params = vec![
        ("first".to_string(), TypeWithLifetime::reference("str", false, false)),
        ("second".to_string(), TypeWithLifetime::reference("str", false, false)),
        ("use_first".to_string(), TypeWithLifetime::value("bool")),
    ];
    let return_type = Some(TypeWithLifetime::reference("str", false, false));

    let sig = analyzer.analyze_function("combine", params, return_type);

    println!("\nRust:");
    println!("{}", sig.to_rust());
    println!();

    println!("Analysis:");
    println!("- 'first' has lifetime 'a");
    println!("- 'second' has lifetime 'b");
    println!("- Return type must be valid for BOTH lifetimes");
    println!("- Rust uses the shorter of the two lifetimes for safety");
    println!();

    println!("Alternative with same lifetime:");
    let params = vec![
        ("first".to_string(), TypeWithLifetime::with_lifetime("str", &Lifetime::new("a"), false, false)),
        ("second".to_string(), TypeWithLifetime::with_lifetime("str", &Lifetime::new("a"), false, false)),
        ("use_first".to_string(), TypeWithLifetime::value("bool")),
    ];
    let return_type = Some(TypeWithLifetime::with_lifetime("str", &Lifetime::new("a"), false, false));

    let mut analyzer = LifetimeAnalyzer::new();
    let sig = analyzer.analyze_function("combine_same", params, return_type);
    println!("{}", sig.to_rust());
    println!("Both inputs must live at least as long as the output");

    println!("\n{}\n", "=".repeat(80));
}

fn example_lifetime_bounds() {
    println!("## Example 5: Lifetime Bounds\n");
    println!("Lifetime bounds constrain relationships between lifetimes.\n");

    let mut analyzer = LifetimeAnalyzer::new();

    println!("### Example: Inner reference outlives outer");
    let params = vec![
        ("outer".to_string(), TypeWithLifetime::with_lifetime("Wrapper", &Lifetime::new("a"), false, false)),
    ];
    let return_type = Some(TypeWithLifetime::with_lifetime("str", &Lifetime::new("b"), false, false));

    let mut sig = analyzer.analyze_function("extract_inner", params, return_type);

    // Add constraint that 'b must outlive 'a
    sig.constraints.push(LifetimeConstraint::Outlives {
        longer: Lifetime::new("b"),
        shorter: Lifetime::new("a"),
    });

    println!("{}", sig.to_rust());
    println!();

    println!("The constraint 'b: 'a means:");
    println!("- Lifetime 'b must be at least as long as 'a");
    println!("- Inner data (&'b str) outlives the wrapper (&'a Wrapper)");
    println!("- Ensures extracted reference is always valid");

    println!("\n{}\n", "=".repeat(80));
}

fn example_common_patterns() {
    println!("## Example 6: Common Lifetime Patterns\n");

    let patterns = LifetimePatterns::common_patterns();

    println!("### Pattern 1: Return Parameter Reference");
    println!("{}", patterns.return_param_ref);
    println!();

    println!("### Pattern 2: Struct with Reference");
    println!("{}", patterns.struct_with_ref);
    println!();

    println!("### Pattern 3: Multiple Lifetimes");
    println!("{}", patterns.multiple_lifetimes);
    println!();

    println!("### Pattern 4: Lifetime Bounds");
    println!("{}", patterns.lifetime_bounds);
    println!();

    println!("### Pattern 5: Static Lifetime");
    println!("{}", patterns.static_lifetime);
    println!("- 'static means the data lives for the entire program");
    println!("- String literals have 'static lifetime");
    println!();

    println!("### Pattern 6: Mutable Reference");
    println!("{}", patterns.mutable_ref);
    println!("- Only one mutable reference allowed at a time");
    println!("- Prevents data races");

    println!("\n{}\n", "=".repeat(80));
}

fn example_python_to_rust() {
    println!("## Example 7: Python to Rust Lifetime Mapping\n");

    println!("Python uses garbage collection, Rust uses lifetimes.\n");

    println!("### Python: Object References");
    println!(r#"
class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node  # Reference to another node
"#);

    let mut analyzer = LifetimeAnalyzer::new();
    let fields = vec![
        ("value".to_string(), TypeWithLifetime::value("i32")),
        ("next".to_string(), TypeWithLifetime::reference("Node", false, false)),
    ];
    let struct_def = analyzer.analyze_struct("Node", fields);

    println!("Rust (with lifetimes):");
    println!("{}", struct_def.to_rust());
    println!();

    println!("Alternative: Use smart pointers to avoid lifetimes");
    println!("struct Node {{");
    println!("    value: i32,");
    println!("    next: Option<Box<Node>>,  // Owned reference");
    println!("}}");
    println!();

    println!("### Python: String Slicing");
    println!(r#"
def get_extension(filename):
    return filename.split('.')[-1]  # Returns slice of input
"#);

    let params = vec![
        ("filename".to_string(), TypeWithLifetime::reference("str", false, false)),
    ];
    let return_type = Some(TypeWithLifetime::reference("str", false, false));
    let sig = analyzer.analyze_function("get_extension", params, return_type);

    println!("Rust:");
    println!("{}", sig.to_rust());
    println!("The lifetime ensures the slice is valid as long as the original string");

    println!("\n{}\n", "=".repeat(80));
}

fn example_complex_lifetimes() {
    println!("## Example 8: Complex Lifetime Scenarios\n");

    let mut analyzer = LifetimeAnalyzer::new();

    println!("### Scenario 1: Iterator with borrowed data");
    println!(r#"
Python:
class Items:
    def __iter__(self):
        return iter(self.data)
"#);

    let fields = vec![
        ("data".to_string(), TypeWithLifetime::reference("[T]", false, false)),
        ("index".to_string(), TypeWithLifetime::value("usize")),
    ];
    let iter_struct = analyzer.analyze_struct("Items", fields);
    println!("Rust:");
    println!("{}", iter_struct.to_rust());
    println!();

    println!("### Scenario 2: Closure capturing references");
    let params = vec![
        ("data".to_string(), TypeWithLifetime::reference("Vec<i32>", false, false)),
    ];
    let return_type = Some(TypeWithLifetime::with_lifetime(
        "impl Fn(i32) -> bool",
        &Lifetime::new("a"),
        false,
        false,
    ));
    let sig = analyzer.analyze_function("make_filter", params, return_type);

    println!("{}", sig.to_rust());
    println!("The closure captures 'data and inherits its lifetime");
    println!();

    println!("### Scenario 3: Higher-ranked trait bounds (HRTB)");
    println!("fn apply<F>(f: F)");
    println!("where");
    println!("    F: for<'a> Fn(&'a str) -> &'a str,");
    println!("{{");
    println!("    // F works with ANY lifetime 'a");
    println!("}}");
    println!();
    println!("HRTB allows functions to work with references of any lifetime");

    println!("\n{}\n", "=".repeat(80));
}
