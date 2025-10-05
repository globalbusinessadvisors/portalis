//! Reference Optimization Examples
//!
//! Demonstrates optimization of reference usage in transpiled Rust code,
//! including clone elimination, string optimizations, and smart pointer selection.

use portalis_transpiler::reference_optimizer::*;

fn main() {
    println!("=== Reference Optimization Examples ===\n");

    // Example 1: Parameter optimization
    example_parameter_optimization();

    // Example 2: Clone elimination
    example_clone_elimination();

    // Example 3: String slice optimization
    example_string_optimization();

    // Example 4: Iterator chain optimization
    example_iterator_optimization();

    // Example 5: Slice parameters
    example_slice_parameters();

    // Example 6: Return type optimization
    example_return_optimization();

    // Example 7: Cow optimization
    example_cow_optimization();

    // Example 8: Smart pointer selection
    example_smart_pointer_selection();
}

fn example_parameter_optimization() {
    println!("## Example 1: Parameter Optimization\n");

    println!("Python:");
    println!(r#"
def greet(name):
    print(f"Hello, {{name}}")
"#);

    let mut optimizer = ReferenceOptimizer::new();

    println!("\nNaive Rust translation:");
    println!("fn greet(name: String) {{");
    println!("    println!(\"Hello, {{}}\", name);");
    println!("}}");

    let optimized = optimizer.optimize_parameter(
        "name",
        "String",
        "println!(\"Hello, {}\", name)"
    );

    println!("\nOptimized Rust:");
    println!("fn {} {{", optimized);
    println!("    println!(\"Hello, {{}}\", name);");
    println!("}}");

    println!("\nBenefit: Caller can pass &str without allocating String");
    println!("Usage: greet(\"Alice\") instead of greet(\"Alice\".to_string())");

    println!("\n{}\n", "=".repeat(80));
}

fn example_clone_elimination() {
    println!("## Example 2: Clone Elimination\n");

    let (before, after) = OptimizationPatterns::clone_elimination_example();

    println!("❌ Inefficient (unnecessary clone):");
    println!("{}", before);

    println!("✅ Optimized (clone eliminated):");
    println!("{}", after);

    let mut optimizer = ReferenceOptimizer::new();
    optimizer.track_read("data");

    let result = optimizer.eliminate_clone("result", "data.clone()");

    println!("Optimizer suggestion:");
    println!("  Original:  data.clone()");
    println!("  Optimized: {}", result);
    println!("\nReason: 'data' is consumed and not used again, clone is unnecessary");

    println!("\n{}\n", "=".repeat(80));
}

fn example_string_optimization() {
    println!("## Example 3: String Slice Optimization\n");

    let (before, after) = OptimizationPatterns::string_slice_example();

    println!("❌ Inefficient (String allocation):");
    println!("{}", before);

    println!("✅ Optimized (&str - no allocation):");
    println!("{}", after);

    let mut optimizer = ReferenceOptimizer::new();

    let optimized = optimizer.optimize_string("greeting", "Hello", true);
    println!("\nOptimizer suggestion:");
    println!("{}", optimized);

    println!("\nString vs &str guidelines:");
    println!("- Use &str for: function parameters, string literals, borrowed strings");
    println!("- Use String for: owned data, building strings, returning new strings");
    println!("- Memory impact: &str is zero-cost, String allocates on heap");

    println!("\n{}\n", "=".repeat(80));
}

fn example_iterator_optimization() {
    println!("## Example 4: Iterator Chain Optimization\n");

    let (before, after) = OptimizationPatterns::iterator_chain_example();

    println!("❌ Inefficient (intermediate allocation):");
    println!("{}", before);

    println!("✅ Optimized (direct chain):");
    println!("{}", after);

    let mut optimizer = ReferenceOptimizer::new();

    let code = "items.iter().map(|x| x * 2).collect::<Vec<_>>().iter().filter(|x| **x > 10)";
    let optimized = optimizer.optimize_iterator("chain", code);

    println!("\nOptimizer detected:");
    println!("  Pattern: .collect::<Vec<_>>().iter()");
    println!("  Issue: Unnecessary intermediate Vec allocation");
    println!("  Fix: Chain iterators directly");

    println!("\nIterator optimization benefits:");
    println!("- Zero intermediate allocations");
    println!("- Lazy evaluation (only compute what's needed)");
    println!("- Better cache locality");
    println!("- Compiler can inline and optimize better");

    println!("\n{}\n", "=".repeat(80));
}

fn example_slice_parameters() {
    println!("## Example 5: Slice Parameters\n");

    let (before, after) = OptimizationPatterns::slice_parameter_example();

    println!("❌ Inefficient (takes ownership):");
    println!("{}", before);

    println!("✅ Optimized (borrows via slice):");
    println!("{}", after);

    let mut optimizer = ReferenceOptimizer::new();

    let optimized = optimizer.optimize_parameter(
        "numbers",
        "Vec<i32>",
        "numbers.iter().sum()"
    );

    println!("Optimizer suggestion: {}", optimized);

    println!("\nSlice benefits:");
    println!("- Works with Vec, arrays, and other slices");
    println!("- Caller retains ownership");
    println!("- No cloning needed");
    println!("- More flexible API");

    println!("\nExamples:");
    println!("  fn process(data: &[i32])  // Works with Vec, array, slice");
    println!("  process(&vec);            // Pass Vec");
    println!("  process(&arr);            // Pass array");
    println!("  process(&slice[1..5]);    // Pass slice");

    println!("\n{}\n", "=".repeat(80));
}

fn example_return_optimization() {
    println!("## Example 6: Return Type Optimization\n");

    println!("Python:");
    println!(r#"
def get_first_word(text):
    return text.split()[0]
"#);

    println!("\n❌ Inefficient:");
    println!("fn get_first_word(text: &str) -> String {{");
    println!("    text.split_whitespace().next().unwrap().to_string()");
    println!("}}");

    println!("\n✅ Optimized:");
    println!("fn get_first_word(text: &str) -> &str {{");
    println!("    text.split_whitespace().next().unwrap()");
    println!("}}");

    println!("\nWith lifetime annotation:");
    println!("fn get_first_word<'a>(text: &'a str) -> &'a str {{");
    println!("    text.split_whitespace().next().unwrap()");
    println!("}}");

    println!("\nBenefits:");
    println!("- No allocation (zero-cost)");
    println!("- Returns slice of input");
    println!("- Lifetime ties output to input");

    println!("\n{}\n", "=".repeat(80));
}

fn example_cow_optimization() {
    println!("## Example 7: Cow (Clone on Write) Optimization\n");

    let (before, after) = OptimizationPatterns::cow_example();

    println!("❌ Inefficient (always allocates):");
    println!("{}", before);

    println!("✅ Optimized (conditional allocation):");
    println!("{}", after);

    let mut optimizer = ReferenceOptimizer::new();

    let suggestion = optimizer.suggest_cow("result", true, true);
    println!("Optimizer suggestion: {}", suggestion);

    println!("\nCow use cases:");
    println!("- Sometimes return borrowed, sometimes owned");
    println!("- Delay allocation until necessary");
    println!("- API accepts both &str and String");

    println!("\nExample usage:");
    println!(r#"
use std::borrow::Cow;

fn process(s: &str, uppercase: bool) -> Cow<str> {{
    if uppercase {{
        Cow::Owned(s.to_uppercase())  // Allocate when needed
    }} else {{
        Cow::Borrowed(s)              // Zero-cost when not
    }}
}}

let result = process("hello", false);  // Borrowed, no allocation
let result = process("hello", true);   // Owned, allocated
"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_smart_pointer_selection() {
    println!("## Example 8: Smart Pointer Selection\n");

    println!("{}", OptimizationPatterns::smart_pointer_example());

    println!("\nOptimizer decision tree:");
    println!("┌─ Need to share ownership?");
    println!("│  ├─ No  → Use Box<T> (or just T)");
    println!("│  └─ Yes ─┬─ Multi-threaded?");
    println!("│          ├─ No  ─┬─ Need mutation?");
    println!("│          │       ├─ No  → Rc<T>");
    println!("│          │       └─ Yes → Rc<RefCell<T>>");
    println!("│          └─ Yes ─┬─ Need mutation?");
    println!("│                  ├─ No  → Arc<T>");
    println!("│                  └─ Yes → Arc<Mutex<T>>");

    let mut optimizer = ReferenceOptimizer::new();

    println!("\nExamples:");

    let single = optimizer.optimize_smart_pointer("data", false, false, false);
    println!("  Single ownership: {}", single);

    let shared = optimizer.optimize_smart_pointer("data", true, false, false);
    println!("  Shared, immutable: {}", shared);

    let shared_mut = optimizer.optimize_smart_pointer("data", true, true, false);
    println!("  Shared, mutable (single-thread): {}", shared_mut);

    let thread_safe = optimizer.optimize_smart_pointer("data", true, true, true);
    println!("  Shared, mutable (multi-thread): {}", thread_safe);

    println!("\nPerformance characteristics:");
    println!("- Box<T>: Heap allocation, single owner, no overhead");
    println!("- Rc<T>: Reference counting, cheap clone, not thread-safe");
    println!("- Arc<T>: Atomic reference counting, thread-safe, slight overhead");
    println!("- RefCell<T>: Runtime borrow checking, panics on violation");
    println!("- Mutex<T>: Thread-safe mutation, blocks on contention");

    println!("\n{}\n", "=".repeat(80));
}
