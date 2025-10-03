//! Benchmark for transpiler translation performance
//!
//! Measures baseline CPU performance for pattern-based translation

use portalis_core::Agent;
use portalis_transpiler::{TranspilerAgent, TranspilerInput};
use serde_json::json;
use std::time::Instant;

fn create_simple_function_input() -> TranspilerInput {
    TranspilerInput {
        typed_functions: vec![json!({
            "name": "add",
            "params": [
                {"name": "a", "rust_type": {"I32": null}},
                {"name": "b", "rust_type": {"I32": null}}
            ],
            "return_type": {"I32": null},
            "docstring": "Add two numbers"
        })],
        typed_classes: vec![],
        use_statements: vec![],
        cargo_dependencies: vec![],
        api_contract: json!({}),
    }
}

fn create_complex_function_input() -> TranspilerInput {
    TranspilerInput {
        typed_functions: vec![json!({
            "name": "fibonacci",
            "params": [
                {"name": "n", "rust_type": {"I32": null}}
            ],
            "return_type": {"I32": null},
            "docstring": "Calculate fibonacci number"
        })],
        typed_classes: vec![],
        use_statements: vec![],
        cargo_dependencies: vec![],
        api_contract: json!({}),
    }
}

fn create_class_input() -> TranspilerInput {
    TranspilerInput {
        typed_functions: vec![],
        typed_classes: vec![json!({
            "name": "Calculator",
            "attributes": [
                {"name": "precision", "type_hint": "int"}
            ],
            "methods": [
                {
                    "name": "__init__",
                    "params": [
                        {"name": "self", "rust_type": null},
                        {"name": "precision", "rust_type": {"I32": null}}
                    ],
                    "return_type": null
                },
                {
                    "name": "add",
                    "params": [
                        {"name": "self", "rust_type": null},
                        {"name": "a", "rust_type": {"F64": null}},
                        {"name": "b", "rust_type": {"F64": null}}
                    ],
                    "return_type": {"F64": null}
                }
            ]
        })],
        use_statements: vec![],
        cargo_dependencies: vec![],
        api_contract: json!({}),
    }
}

#[tokio::main]
async fn main() {
    println!("=== Portalis Transpiler Benchmark ===\n");
    println!("Measuring baseline CPU performance for pattern-based translation\n");

    let agent = TranspilerAgent::new();

    // Benchmark 1: Simple function
    println!("1. Simple Function Translation (add)");
    let input = create_simple_function_input();
    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = agent.execute(input.clone()).await;
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("   Iterations: {}", iterations);
    println!("   Total time: {:.2?}", elapsed);
    println!("   Average: {:.3}ms per translation", avg_ms);
    println!("   Throughput: {:.0} translations/sec\n", 1000.0 / avg_ms);

    // Benchmark 2: Complex function
    println!("2. Complex Function Translation (fibonacci)");
    let input = create_complex_function_input();
    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = agent.execute(input.clone()).await;
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("   Iterations: {}", iterations);
    println!("   Total time: {:.2?}", elapsed);
    println!("   Average: {:.3}ms per translation", avg_ms);
    println!("   Throughput: {:.0} translations/sec\n", 1000.0 / avg_ms);

    // Benchmark 3: Class translation
    println!("3. Class Translation (Calculator)");
    let input = create_class_input();
    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = agent.execute(input.clone()).await;
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("   Iterations: {}", iterations);
    println!("   Total time: {:.2?}", elapsed);
    println!("   Average: {:.3}ms per translation", avg_ms);
    println!("   Throughput: {:.0} translations/sec\n", 1000.0 / avg_ms);

    println!("=== Benchmark Complete ===");
    println!("\nNote: These are baseline CPU measurements.");
    println!("GPU-accelerated NeMo translation benchmarks will be added in future phases.");
}
