// Advanced Python Features Demo
// Demonstrates: generators, comprehensions, walrus operator, pattern matching, *args/**kwargs

use portalis_transpiler::advanced_features::{
    AdvancedFeaturesTranslator, Generator, Parameter, Comprehension,
    ComprehensionType, IterClause, WalrusExpr, MatchCase, Pattern,
};

fn main() {
    println!("=== Advanced Python Features Translation Demo ===\n");
    println!("Demonstrates: Generators, Comprehensions, Walrus, Pattern Matching, *args/**kwargs\n");
    println!("{}", "=".repeat(80));

    // Demo 1: Generator Functions
    demo_generators();
    println!("\n{}", "=".repeat(80));

    // Demo 2: Dict Comprehensions
    demo_dict_comprehensions();
    println!("\n{}", "=".repeat(80));

    // Demo 3: Set Comprehensions
    demo_set_comprehensions();
    println!("\n{}", "=".repeat(80));

    // Demo 4: Generator Expressions
    demo_generator_expressions();
    println!("\n{}", "=".repeat(80));

    // Demo 5: Walrus Operator
    demo_walrus_operator();
    println!("\n{}", "=".repeat(80));

    // Demo 6: Pattern Matching
    demo_pattern_matching();
    println!("\n{}", "=".repeat(80));

    // Demo 7: *args and **kwargs
    demo_args_kwargs();
    println!("\n{}", "=".repeat(80));

    // Demo 8: Real-World Examples
    demo_real_world_examples();
    println!("\n{}", "=".repeat(80));

    println!("\n🎉 Advanced features demonstration complete!");
}

fn demo_generators() {
    println!("\n=== Demo 1: Generator Functions ===\n");

    println!("Python Generator:");
    println!("{}", "-".repeat(80));
    println!(r#"def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        yield a
        a, b = b, a + b

# Usage
for num in fibonacci(10):
    print(num)"#);
    println!("{}", "-".repeat(80));

    let gen = Generator {
        name: "fibonacci".to_string(),
        params: vec![
            Parameter {
                name: "n".to_string(),
                param_type: "i32".to_string(),
                is_args: false,
                is_kwargs: false,
                default_value: None,
            }
        ],
        yield_type: "i32".to_string(),
        body: vec![
            "let mut i = 0".to_string(),
            "yield self.a".to_string(),
            "let temp = self.a + self.b".to_string(),
            "self.a = self.b".to_string(),
            "self.b = temp".to_string(),
        ],
        is_async: false,
    };

    println!("\nTranslated to Rust:");
    println!("{}", "-".repeat(80));
    let rust_code = AdvancedFeaturesTranslator::translate_generator(&gen);
    println!("{}", rust_code);
    println!("{}", "-".repeat(80));

    println!("\nUsage:");
    println!(r#"  for num in fibonacci(10) {{
      println!("{{}}", num);
  }}"#);

    println!("\nBenefits:");
    println!("  ✅ Zero-cost abstraction - compiles to efficient iterator");
    println!("  ✅ Lazy evaluation - values computed on demand");
    println!("  ✅ Memory efficient - no intermediate storage");
}

fn demo_dict_comprehensions() {
    println!("\n=== Demo 2: Dict Comprehensions ===\n");

    println!("Python Dict Comprehension:");
    println!("{}", "-".repeat(80));
    println!("{{k: v**2 for k, v in items.items() if v > 0}}");
    println!("{}", "-".repeat(80));

    let comp = Comprehension {
        comp_type: ComprehensionType::Dict,
        expression: "v * v".to_string(),
        key_expr: Some("k".to_string()),
        iterables: vec![IterClause {
            target: "(k, v)".to_string(),
            iter_expr: "items.iter()".to_string(),
        }],
        conditions: vec!["*v > 0".to_string()],
    };

    println!("\nTranslated to Rust:");
    println!("{}", "-".repeat(80));
    let rust_code = AdvancedFeaturesTranslator::translate_dict_comprehension(&comp);
    println!("{}", rust_code);
    println!("{}", "-".repeat(80));

    println!("\nMore Examples:");
    println!("\n1. Key transformation:");
    println!(r#"   Python: {{k.upper(): v for k, v in items.items()}}"#);
    println!(r#"   Rust:   {{"#);
    println!(r#"             let mut result = HashMap::new();"#);
    println!(r#"             for (k, v) in items.iter() {{"#);
    println!(r#"                 result.insert(k.to_uppercase(), v.clone());"#);
    println!(r#"             }}"#);
    println!(r#"             result"#);
    println!(r#"           }}"#);

    println!("\n2. Filtering and mapping:");
    println!(r#"   Python: {{k: v*2 for k, v in items.items() if v % 2 == 0}}"#);
    println!(r#"   Rust:   items.iter()"#);
    println!(r#"               .filter(|(_, v)| *v % 2 == 0)"#);
    println!(r#"               .map(|(k, v)| (k.clone(), v * 2))"#);
    println!(r#"               .collect::<HashMap<_, _>>()"#);
}

fn demo_set_comprehensions() {
    println!("\n=== Demo 3: Set Comprehensions ===\n");

    println!("Python Set Comprehension:");
    println!("{}", "-".repeat(80));
    println!("{{x**2 for x in range(10) if x % 2 == 0}}");
    println!("{}", "-".repeat(80));

    let comp = Comprehension {
        comp_type: ComprehensionType::Set,
        expression: "x * x".to_string(),
        key_expr: None,
        iterables: vec![IterClause {
            target: "x".to_string(),
            iter_expr: "0..10".to_string(),
        }],
        conditions: vec!["x % 2 == 0".to_string()],
    };

    println!("\nTranslated to Rust:");
    println!("{}", "-".repeat(80));
    let rust_code = AdvancedFeaturesTranslator::translate_dict_comprehension(&comp);
    println!("{}", rust_code);
    println!("{}", "-".repeat(80));

    println!("\nIterator Chain Alternative:");
    println!("{}", "-".repeat(80));
    println!(r#"(0..10)
    .filter(|x| x % 2 == 0)
    .map(|x| x * x)
    .collect::<HashSet<_>>()"#);
    println!("{}", "-".repeat(80));
}

fn demo_generator_expressions() {
    println!("\n=== Demo 4: Generator Expressions ===\n");

    println!("Python Generator Expression:");
    println!("{}", "-".repeat(80));
    println!(r#"(x**2 for x in range(10) if x % 2 == 0)"#);
    println!("{}", "-".repeat(80));

    let comp = Comprehension {
        comp_type: ComprehensionType::Generator,
        expression: "x * x".to_string(),
        key_expr: None,
        iterables: vec![IterClause {
            target: "x".to_string(),
            iter_expr: "0..10".to_string(),
        }],
        conditions: vec!["x % 2 == 0".to_string()],
    };

    println!("\nTranslated to Rust (Iterator Chain):");
    println!("{}", "-".repeat(80));
    let rust_code = AdvancedFeaturesTranslator::translate_dict_comprehension(&comp);
    println!("{}", rust_code);
    println!("{}", "-".repeat(80));

    println!("\nUsage:");
    println!(r#"  for value in (0..10).filter(|x| x % 2 == 0).map(|x| x * x) {{
      println!("{{}}", value);
  }}"#);

    println!("\nBenefits:");
    println!("  ✅ Lazy evaluation - computed on demand");
    println!("  ✅ Composable - chain multiple operations");
    println!("  ✅ Zero allocations until .collect()");
}

fn demo_walrus_operator() {
    println!("\n=== Demo 5: Walrus Operator (:=) ===\n");

    println!("Python Walrus Operator:");
    println!("{}", "-".repeat(80));
    println!(r#"if (n := len(data)) > 10:
    process(n)

while (line := file.readline()):
    print(line)

[y for x in range(10) if (y := x**2) > 20]"#);
    println!("{}", "-".repeat(80));

    println!("\nTranslated to Rust:");
    println!("{}", "-".repeat(80));

    let expr1 = WalrusExpr {
        target: "n".to_string(),
        value: "data.len()".to_string(),
    };
    println!("// if (n := len(data)) > 10:");
    println!("{}", AdvancedFeaturesTranslator::translate_walrus(&expr1, "n > 10"));
    println!();

    println!("// while (line := file.readline()):");
    println!(r#"while {{
    let line = file.read_line();
    !line.is_empty()
}} {{
    println!("{{}}", line);
}}"#);
    println!();

    println!("// List comprehension with walrus:");
    println!(r#"(0..10)
    .filter_map(|x| {{
        let y = x * x;
        if y > 20 {{ Some(y) }} else {{ None }}
    }})
    .collect::<Vec<_>>()"#);

    println!("{}", "-".repeat(80));
}

fn demo_pattern_matching() {
    println!("\n=== Demo 6: Pattern Matching (match statement) ===\n");

    println!("Python Pattern Matching (3.10+):");
    println!("{}", "-".repeat(80));
    println!(r#"match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y-axis at {{y}}")
    case (x, 0):
        print(f"X-axis at {{x}}")
    case (x, y) if x == y:
        print(f"Diagonal at {{x}}")
    case (x, y):
        print(f"Point at ({{x}}, {{y}})")"#);
    println!("{}", "-".repeat(80));

    let cases = vec![
        MatchCase {
            pattern: Pattern::Tuple(vec![
                Pattern::Literal("0".to_string()),
                Pattern::Literal("0".to_string()),
            ]),
            guard: None,
            body: vec![r#"println!("Origin")"#.to_string()],
        },
        MatchCase {
            pattern: Pattern::Tuple(vec![
                Pattern::Literal("0".to_string()),
                Pattern::Variable("y".to_string()),
            ]),
            guard: None,
            body: vec![r#"println!("Y-axis at {}", y)"#.to_string()],
        },
        MatchCase {
            pattern: Pattern::Tuple(vec![
                Pattern::Variable("x".to_string()),
                Pattern::Literal("0".to_string()),
            ]),
            guard: None,
            body: vec![r#"println!("X-axis at {}", x)"#.to_string()],
        },
        MatchCase {
            pattern: Pattern::Tuple(vec![
                Pattern::Variable("x".to_string()),
                Pattern::Variable("y".to_string()),
            ]),
            guard: Some("x == y".to_string()),
            body: vec![r#"println!("Diagonal at {}", x)"#.to_string()],
        },
        MatchCase {
            pattern: Pattern::Tuple(vec![
                Pattern::Variable("x".to_string()),
                Pattern::Variable("y".to_string()),
            ]),
            guard: None,
            body: vec![r#"println!("Point at ({}, {})", x, y)"#.to_string()],
        },
    ];

    println!("\nTranslated to Rust:");
    println!("{}", "-".repeat(80));
    let rust_code = AdvancedFeaturesTranslator::translate_pattern_match("point", &cases);
    println!("{}", rust_code);
    println!("{}", "-".repeat(80));

    println!("\nMore Pattern Types:");
    println!("\n1. Struct patterns:");
    println!("   Python: case Point(x=0, y=0): ...");
    println!("   Rust:   Point {{ x: 0, y: 0 }} => ...");

    println!("\n2. Or patterns:");
    println!("   Python: case 'a' | 'e' | 'i' | 'o' | 'u': ...");
    println!("   Rust:   'a' | 'e' | 'i' | 'o' | 'u' => ...");

    println!("\n3. Wildcard:");
    println!("   Python: case _: ...");
    println!("   Rust:   _ => ...");
}

fn demo_args_kwargs() {
    println!("\n=== Demo 7: *args and **kwargs ===\n");

    println!("Python Variadic Parameters:");
    println!("{}", "-".repeat(80));
    println!(r#"def process(*args, **kwargs):
    for arg in args:
        print(arg)
    for k, v in kwargs.items():
        print(f"{{k}}: {{v}}")

process(1, 2, 3, name="Alice", age=30)"#);
    println!("{}", "-".repeat(80));

    let params = vec![
        Parameter {
            name: "args".to_string(),
            param_type: "i32".to_string(),
            is_args: true,
            is_kwargs: false,
            default_value: None,
        },
        Parameter {
            name: "kwargs".to_string(),
            param_type: "String".to_string(),
            is_args: false,
            is_kwargs: true,
            default_value: None,
        },
    ];

    let (params_str, _) = AdvancedFeaturesTranslator::translate_args_kwargs(&params);

    println!("\nTranslated to Rust:");
    println!("{}", "-".repeat(80));
    println!("pub fn process({}) {{", params_str);
    println!("    for arg in args {{");
    println!("        println!(\"{{}}\", arg);");
    println!("    }}");
    println!("    for (k, v) in kwargs {{");
    println!("        println!(\"{{}}: {{}}\", k, v);");
    println!("    }}");
    println!("}}");
    println!("{}", "-".repeat(80));

    println!("\nUsage:");
    println!("{}", "-".repeat(80));
    println!(r#"let args = vec![1, 2, 3];
let mut kwargs = HashMap::new();
kwargs.insert("name".to_string(), "Alice".to_string());
kwargs.insert("age".to_string(), "30".to_string());

process(args, kwargs);"#);
    println!("{}", "-".repeat(80));

    println!("\nAlternative: Builder Pattern:");
    println!("{}", "-".repeat(80));
    println!(r#"ProcessBuilder::new()
    .arg(1)
    .arg(2)
    .arg(3)
    .kwarg("name", "Alice")
    .kwarg("age", "30")
    .build()
    .process();"#);
    println!("{}", "-".repeat(80));
}

fn demo_real_world_examples() {
    println!("\n=== Demo 8: Real-World Examples ===\n");

    let examples = AdvancedFeaturesTranslator::generate_examples();

    for (i, (name, python, rust)) in examples.iter().enumerate() {
        println!("Example {}: {}\n", i + 1, name);
        println!("Python:");
        println!("{}", "-".repeat(80));
        println!("{}", python);
        println!("{}", "-".repeat(80));
        println!("\nRust:");
        println!("{}", "-".repeat(80));
        println!("{}", rust);
        println!("{}", "-".repeat(80));
        if i < examples.len() - 1 {
            println!();
        }
    }

    println!("\n\nTranslation Summary:");
    println!("{}", "=".repeat(80));
    println!("Feature              | Python Syntax           | Rust Equivalent");
    println!("{}", "-".repeat(80));
    println!("Generator            | def f(): yield x        | Iterator trait + struct");
    println!("Dict Comprehension   | {{k:v for k,v in ...}}    | HashMap + for loop");
    println!("Set Comprehension    | {{x for x in ...}}        | HashSet + for loop");
    println!("Generator Expr       | (x for x in ...)        | Iterator chain");
    println!("Walrus Operator      | if (n := expr):         | {{ let n = expr; ... }}");
    println!("Pattern Matching     | match case:             | match {{ pattern => }}");
    println!("*args                | def f(*args):           | Vec<T>");
    println!("**kwargs             | def f(**kwargs):        | HashMap<String, T>");
    println!("{}", "=".repeat(80));
}
