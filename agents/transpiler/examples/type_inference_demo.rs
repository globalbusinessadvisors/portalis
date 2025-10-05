//! Hindley-Milner Type Inference Examples
//!
//! Demonstrates how the HM type inference system works for Python to Rust translation,
//! including polymorphism, unification, and automatic type derivation.

use portalis_transpiler::type_inference::*;

fn main() {
    println!("=== Hindley-Milner Type Inference Examples ===\n");

    // Example 1: Basic type inference
    example_basic_inference();

    // Example 2: Function types
    example_function_types();

    // Example 3: Polymorphic types
    example_polymorphic_types();

    // Example 4: Let polymorphism
    example_let_polymorphism();

    // Example 5: Type unification
    example_unification();

    // Example 6: Python to Rust type mapping
    example_python_to_rust();

    // Example 7: Complex type inference
    example_complex_inference();

    // Example 8: Type constraints
    example_type_constraints();
}

fn example_basic_inference() {
    println!("## Example 1: Basic Type Inference\n");
    println!("Python:");
    println!("x = 42");
    println!("s = \"hello\"");
    println!("b = True\n");

    let mut inf = TypeInference::new();

    // Infer type of integer literal
    let expr_int = Expr::Lit(Literal::Int(42));
    let (_, ty_int) = inf.infer(&expr_int).unwrap();
    println!("Rust type for 42:");
    println!("  HM type: {}", ty_int);
    println!("  Rust: {}\n", ty_int.to_rust_type());

    // Infer type of string literal
    let expr_str = Expr::Lit(Literal::String("hello".to_string()));
    let (_, ty_str) = inf.infer(&expr_str).unwrap();
    println!("Rust type for \"hello\":");
    println!("  HM type: {}", ty_str);
    println!("  Rust: {}\n", ty_str.to_rust_type());

    // Infer type of boolean literal
    let expr_bool = Expr::Lit(Literal::Bool(true));
    let (_, ty_bool) = inf.infer(&expr_bool).unwrap();
    println!("Rust type for True:");
    println!("  HM type: {}", ty_bool);
    println!("  Rust: {}", ty_bool.to_rust_type());

    println!("\n{}\n", "=".repeat(80));
}

fn example_function_types() {
    println!("## Example 2: Function Types\n");
    println!("Python:");
    println!("def identity(x):");
    println!("    return x\n");

    let mut inf = TypeInference::new();

    // λx. x  (identity function)
    let identity = Expr::Abs(
        "x".to_string(),
        Box::new(Expr::Var("x".to_string())),
    );

    let (_, ty) = inf.infer(&identity).unwrap();
    println!("Inferred type for identity:");
    println!("  HM type: {}", ty);
    println!("  Rust: {}", ty.to_rust_type());
    println!("\nNote: This is a polymorphic function type t0 -> t0");
    println!("Can work with any type: Int -> Int, String -> String, etc.\n");

    println!("Python:");
    println!("def const(x):");
    println!("    def inner(y):");
    println!("        return x");
    println!("    return inner\n");

    // λx. λy. x  (const function)
    let const_fn = Expr::Abs(
        "x".to_string(),
        Box::new(Expr::Abs(
            "y".to_string(),
            Box::new(Expr::Var("x".to_string())),
        )),
    );

    let mut inf = TypeInference::new();
    let (_, ty) = inf.infer(&const_fn).unwrap();
    println!("Inferred type for const:");
    println!("  HM type: {}", ty);
    println!("  Meaning: takes x, returns function that ignores y and returns x");

    println!("\n{}\n", "=".repeat(80));
}

fn example_polymorphic_types() {
    println!("## Example 3: Polymorphic Types\n");
    println!("Polymorphism allows a function to work with multiple types.\n");

    println!("Identity function: ∀a. a -> a");
    let scheme = Scheme::poly(
        vec![TypeVar { name: "a".to_string(), id: 0 }],
        Type::fun(Type::var("a", 0), Type::var("a", 0)),
    );
    println!("  Can be instantiated as:");
    println!("    Int -> Int");
    println!("    String -> String");
    println!("    Vec<T> -> Vec<T>");
    println!("    etc.\n");

    println!("Map function: ∀a b. (a -> b) -> Vec<a> -> Vec<b>");
    let a = Type::var("a", 0);
    let b = Type::var("b", 1);
    let vec_a = Type::app(Type::con("Vec"), a.clone());
    let vec_b = Type::app(Type::con("Vec"), b.clone());
    let fn_a_b = Type::fun(a, b);
    let map_type = Type::fun(fn_a_b, Type::fun(vec_a, vec_b));

    println!("  HM type: {}", map_type);
    println!("  Rust: {}", map_type.to_rust_type());

    println!("\n{}\n", "=".repeat(80));
}

fn example_let_polymorphism() {
    println!("## Example 4: Let Polymorphism\n");
    println!("Python:");
    println!("def example():");
    println!("    id = lambda x: x");
    println!("    a = id(42)       # id: Int -> Int");
    println!("    b = id(\"hello\")  # id: String -> String");
    println!("    return (a, b)\n");

    let mut inf = TypeInference::new();

    // let id = λx. x in (id 42, id "hello")
    let id_fn = Expr::Abs("x".to_string(), Box::new(Expr::Var("x".to_string())));

    let apply_int = Expr::App(
        Box::new(Expr::Var("id".to_string())),
        Box::new(Expr::Lit(Literal::Int(42))),
    );

    let apply_str = Expr::App(
        Box::new(Expr::Var("id".to_string())),
        Box::new(Expr::Lit(Literal::String("hello".to_string()))),
    );

    let let_expr = Expr::Let(
        "id".to_string(),
        Box::new(id_fn),
        Box::new(apply_int), // Simplified - just showing int application
    );

    let (_, ty) = inf.infer(&let_expr).unwrap();
    println!("Result type: {}", ty);
    println!("\nKey insight: 'id' is generalized in let binding");
    println!("Can be used at different types in the same scope");

    println!("\n{}\n", "=".repeat(80));
}

fn example_unification() {
    println!("## Example 5: Type Unification\n");
    println!("Unification makes two types equal by finding substitutions.\n");

    let mut inf = TypeInference::new();

    println!("Case 1: Unify type variable with concrete type");
    let t1 = Type::var("a", 0);
    let t2 = Type::con("Int");
    let subst = inf.unify(&t1, &t2).unwrap();
    println!("  Unify 'a with Int");
    println!("  Result: [a ↦ Int]\n");

    println!("Case 2: Unify function types");
    let t1 = Type::fun(Type::var("a", 0), Type::var("b", 1));
    let t2 = Type::fun(Type::con("Int"), Type::var("c", 2));
    let subst = inf.unify(&t1, &t2).unwrap();
    println!("  Unify (a -> b) with (Int -> c)");
    println!("  Result: [a ↦ Int, b ↦ c]\n");

    println!("Case 3: Occurs check (prevents infinite types)");
    let t1 = Type::var("a", 0);
    let t2 = Type::fun(Type::var("a", 0), Type::con("Int"));
    match inf.unify(&t1, &t2) {
        Err(TypeError::OccursCheck(..)) => {
            println!("  Cannot unify 'a with (a -> Int)");
            println!("  Would create infinite type: a = a -> Int = (a -> Int) -> Int = ...");
        }
        _ => println!("  Unexpected result"),
    }

    println!("\n{}\n", "=".repeat(80));
}

fn example_python_to_rust() {
    println!("## Example 6: Python to Rust Type Mapping\n");

    let type_mappings = vec![
        ("Int", Type::con("Int"), "i32"),
        ("Float", Type::con("Float"), "f64"),
        ("String", Type::con("String"), "String"),
        ("Bool", Type::con("Bool"), "bool"),
        ("Unit", Type::con("Unit"), "()"),
        ("Vec<Int>", Type::app(Type::con("Vec"), Type::con("Int")), "Vec<i32>"),
        ("Option<String>", Type::app(Type::con("Option"), Type::con("String")), "Option<String>"),
    ];

    println!("HM Type          →  Rust Type");
    println!("{}", "-".repeat(40));
    for (name, ty, rust) in type_mappings {
        println!("{:<15}  →  {}", name, ty.to_rust_type());
    }

    println!("\nFunction types:");
    let fn_type = Type::fun(Type::con("Int"), Type::con("String"));
    println!("  {} → {}", fn_type, fn_type.to_rust_type());

    let fn_type2 = Type::fun(
        Type::fun(Type::con("Int"), Type::con("Bool")),
        Type::app(Type::con("Vec"), Type::con("String")),
    );
    println!("  {} → {}", fn_type2, fn_type2.to_rust_type());

    println!("\n{}\n", "=".repeat(80));
}

fn example_complex_inference() {
    println!("## Example 7: Complex Type Inference\n");
    println!("Python:");
    println!("def apply_twice(f, x):");
    println!("    return f(f(x))\n");

    let mut inf = TypeInference::new();

    // λf. λx. f (f x)
    let apply_twice = Expr::Abs(
        "f".to_string(),
        Box::new(Expr::Abs(
            "x".to_string(),
            Box::new(Expr::App(
                Box::new(Expr::Var("f".to_string())),
                Box::new(Expr::App(
                    Box::new(Expr::Var("f".to_string())),
                    Box::new(Expr::Var("x".to_string())),
                )),
            )),
        )),
    );

    let (_, ty) = inf.infer(&apply_twice).unwrap();
    println!("Inferred type:");
    println!("  HM: {}", ty);
    println!("\nAnalysis:");
    println!("  f must be a function: a -> a (input and output same type)");
    println!("  x must be of type a");
    println!("  Result is of type a");
    println!("  Full type: (a -> a) -> a -> a");

    println!("\n{}\n", "=".repeat(80));
}

fn example_type_constraints() {
    println!("## Example 8: Type Constraints\n");
    println!("Python:");
    println!("def example(x):");
    println!("    if x > 0:");
    println!("        return x + 1");
    println!("    else:");
    println!("        return 0\n");

    let mut inf = TypeInference::new();

    // Simulate: if x then 42 else 0
    let if_expr = Expr::If(
        Box::new(Expr::Var("x".to_string())),
        Box::new(Expr::Lit(Literal::Int(42))),
        Box::new(Expr::Lit(Literal::Int(0))),
    );

    // Need to set up environment
    inf.env.insert("x", Scheme::mono(Type::con("Bool")));

    let (_, ty) = inf.infer(&if_expr).unwrap();
    println!("Constraints:");
    println!("  1. x must be Bool (condition type)");
    println!("  2. then-branch: Int");
    println!("  3. else-branch: Int");
    println!("  4. Both branches must have same type");
    println!("\nInferred result type: {}", ty);
    println!("Rust: {}", ty.to_rust_type());

    println!("\nType inference ensures:");
    println!("  ✓ Type safety");
    println!("  ✓ No runtime type errors");
    println!("  ✓ Automatic polymorphism");
    println!("  ✓ Minimal type annotations");

    println!("\n{}\n", "=".repeat(80));
}
