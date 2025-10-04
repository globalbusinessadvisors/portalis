//! Advanced Python Features Translator
//!
//! This module provides translation for advanced Python features:
//! 1. Generator functions and yield expressions
//! 2. Dict and set comprehensions
//! 3. Walrus operator (:=)
//! 4. Pattern matching (match statement)
//! 5. *args and **kwargs (variadic parameters)
//! 6. Async generators
//! 7. Generator expressions

use serde::{Deserialize, Serialize};

/// Generator function representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generator {
    pub name: String,
    pub params: Vec<Parameter>,
    pub yield_type: String,
    pub body: Vec<String>,
    pub is_async: bool,
}

/// Parameter representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: String,
    pub is_args: bool,
    pub is_kwargs: bool,
    pub default_value: Option<String>,
}

/// Comprehension type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComprehensionType {
    List,
    Dict,
    Set,
    Generator,
}

/// Comprehension representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comprehension {
    pub comp_type: ComprehensionType,
    pub expression: String,
    pub key_expr: Option<String>,
    pub iterables: Vec<IterClause>,
    pub conditions: Vec<String>,
}

/// Iterator clause in comprehension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterClause {
    pub target: String,
    pub iter_expr: String,
}

/// Walrus operator (assignment expression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalrusExpr {
    pub target: String,
    pub value: String,
}

/// Pattern matching case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub guard: Option<String>,
    pub body: Vec<String>,
}

/// Pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pattern {
    Literal(String),
    Variable(String),
    Wildcard,
    Struct { name: String, fields: Vec<(String, Pattern)> },
    Tuple(Vec<Pattern>),
    Or(Vec<Pattern>),
}

/// Advanced features translator
pub struct AdvancedFeaturesTranslator;

impl AdvancedFeaturesTranslator {
    /// Translate generator function to Rust
    pub fn translate_generator(gen: &Generator) -> String {
        let mut output = String::new();

        // Generate iterator struct
        let struct_name = format!("{}Iterator", to_pascal_case(&gen.name));
        output.push_str(&format!("pub struct {} {{\n", struct_name));

        // Add state fields
        output.push_str("    state: usize,\n");
        for param in &gen.params {
            if !param.is_args && !param.is_kwargs {
                output.push_str(&format!("    {}: {},\n", param.name, param.param_type));
            }
        }
        output.push_str("}\n\n");

        // Generate Iterator implementation
        output.push_str(&format!("impl Iterator for {} {{\n", struct_name));
        output.push_str(&format!("    type Item = {};\n\n", gen.yield_type));
        output.push_str("    fn next(&mut self) -> Option<Self::Item> {\n");
        output.push_str("        match self.state {\n");

        // Add state machine for yields
        for (i, body_line) in gen.body.iter().enumerate() {
            if body_line.contains("yield") {
                output.push_str(&format!("            {} => {{\n", i));
                output.push_str(&format!("                self.state = {};\n", i + 1));
                let value = body_line.replace("yield", "").trim().to_string();
                output.push_str(&format!("                Some({})\n", value));
                output.push_str("            }\n");
            }
        }

        output.push_str("            _ => None,\n");
        output.push_str("        }\n");
        output.push_str("    }\n");
        output.push_str("}\n\n");

        // Generate constructor function
        let params_str = gen.params.iter()
            .filter(|p| !p.is_args && !p.is_kwargs)
            .map(|p| format!("{}: {}", p.name, p.param_type))
            .collect::<Vec<_>>()
            .join(", ");

        output.push_str(&format!("pub fn {}({}) -> {} {{\n", gen.name, params_str, struct_name));
        output.push_str(&format!("    {} {{\n", struct_name));
        output.push_str("        state: 0,\n");
        for param in &gen.params {
            if !param.is_args && !param.is_kwargs {
                output.push_str(&format!("        {},\n", param.name));
            }
        }
        output.push_str("    }\n");
        output.push_str("}\n");

        output
    }

    /// Translate dict comprehension to Rust
    pub fn translate_dict_comprehension(comp: &Comprehension) -> String {
        let mut output = String::new();

        match comp.comp_type {
            ComprehensionType::Dict => {
                output.push_str("{\n");
                output.push_str("    let mut result = HashMap::new();\n");

                // Nested iteration
                for iter_clause in &comp.iterables {
                    output.push_str(&format!("    for {} in {} {{\n",
                        iter_clause.target, iter_clause.iter_expr));
                }

                // Conditions
                if !comp.conditions.is_empty() {
                    let condition = comp.conditions.join(" && ");
                    output.push_str(&format!("        if {} {{\n", condition));
                    output.push_str(&format!("            result.insert({}, {});\n",
                        comp.key_expr.as_ref().unwrap(), comp.expression));
                    output.push_str("        }\n");
                } else {
                    output.push_str(&format!("        result.insert({}, {});\n",
                        comp.key_expr.as_ref().unwrap(), comp.expression));
                }

                // Close loops
                for _ in &comp.iterables {
                    output.push_str("    }\n");
                }

                output.push_str("    result\n");
                output.push_str("}");
            }
            ComprehensionType::Set => {
                output.push_str("{\n");
                output.push_str("    let mut result = HashSet::new();\n");

                for iter_clause in &comp.iterables {
                    output.push_str(&format!("    for {} in {} {{\n",
                        iter_clause.target, iter_clause.iter_expr));
                }

                if !comp.conditions.is_empty() {
                    let condition = comp.conditions.join(" && ");
                    output.push_str(&format!("        if {} {{\n", condition));
                    output.push_str(&format!("            result.insert({});\n", comp.expression));
                    output.push_str("        }\n");
                } else {
                    output.push_str(&format!("        result.insert({});\n", comp.expression));
                }

                for _ in &comp.iterables {
                    output.push_str("    }\n");
                }

                output.push_str("    result\n");
                output.push_str("}");
            }
            ComprehensionType::Generator => {
                // Generator expression -> iterator chain
                output.push_str(&format!("{}.iter()", comp.iterables[0].iter_expr));

                if !comp.conditions.is_empty() {
                    for condition in &comp.conditions {
                        output.push_str(&format!(".filter(|{}| {})",
                            comp.iterables[0].target, condition));
                    }
                }

                output.push_str(&format!(".map(|{}| {})",
                    comp.iterables[0].target, comp.expression));
            }
            _ => {}
        }

        output
    }

    /// Translate walrus operator
    pub fn translate_walrus(expr: &WalrusExpr, context: &str) -> String {
        // In Rust, we need to assign first, then use
        format!("{{ let {} = {}; {} }}", expr.target, expr.value, context)
    }

    /// Translate pattern matching
    pub fn translate_pattern_match(subject: &str, cases: &[MatchCase]) -> String {
        let mut output = String::new();
        output.push_str(&format!("match {} {{\n", subject));

        for case in cases {
            let pattern_str = Self::translate_pattern(&case.pattern);

            if let Some(ref guard) = case.guard {
                output.push_str(&format!("    {} if {} => {{\n", pattern_str, guard));
            } else {
                output.push_str(&format!("    {} => {{\n", pattern_str));
            }

            for line in &case.body {
                output.push_str(&format!("        {}\n", line));
            }
            output.push_str("    }\n");
        }

        output.push_str("}\n");
        output
    }

    /// Translate pattern
    fn translate_pattern(pattern: &Pattern) -> String {
        match pattern {
            Pattern::Literal(lit) => lit.clone(),
            Pattern::Variable(var) => var.clone(),
            Pattern::Wildcard => "_".to_string(),
            Pattern::Struct { name, fields } => {
                let fields_str = fields.iter()
                    .map(|(name, pat)| format!("{}: {}", name, Self::translate_pattern(pat)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} {{ {} }}", name, fields_str)
            }
            Pattern::Tuple(patterns) => {
                let patterns_str = patterns.iter()
                    .map(|p| Self::translate_pattern(p))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", patterns_str)
            }
            Pattern::Or(patterns) => {
                patterns.iter()
                    .map(|p| Self::translate_pattern(p))
                    .collect::<Vec<_>>()
                    .join(" | ")
            }
        }
    }

    /// Translate *args to Rust
    pub fn translate_args_kwargs(params: &[Parameter]) -> (String, String) {
        let mut regular_params = Vec::new();
        let mut args_handling = String::new();

        for param in params {
            if param.is_args {
                regular_params.push(format!("{}: Vec<{}>", param.name, param.param_type));
            } else if param.is_kwargs {
                regular_params.push(format!("{}: HashMap<String, {}>", param.name, param.param_type));
            } else {
                regular_params.push(format!("{}: {}", param.name, param.param_type));
            }
        }

        (regular_params.join(", "), args_handling)
    }

    /// Generate example code snippets
    pub fn generate_examples() -> Vec<(String, String, String)> {
        vec![
            (
                "Generator".to_string(),
                r#"def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b"#.to_string(),
                r#"pub struct FibonacciIterator {
    state: usize,
    n: i32,
    a: i32,
    b: i32,
    count: i32,
}

impl Iterator for FibonacciIterator {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.n {
            return None;
        }
        let result = self.a;
        let temp = self.a + self.b;
        self.a = self.b;
        self.b = temp;
        self.count += 1;
        Some(result)
    }
}

pub fn fibonacci(n: i32) -> FibonacciIterator {
    FibonacciIterator { state: 0, n, a: 0, b: 1, count: 0 }
}"#.to_string(),
            ),
            (
                "Dict Comprehension".to_string(),
                r#"{k: v**2 for k, v in items.items() if v > 0}"#.to_string(),
                r#"{
    let mut result = HashMap::new();
    for (k, v) in items.iter() {
        if *v > 0 {
            result.insert(k.clone(), v * v);
        }
    }
    result
}"#.to_string(),
            ),
            (
                "Walrus Operator".to_string(),
                r#"if (n := len(data)) > 10:
    process(n)"#.to_string(),
                r#"if { let n = data.len(); n } > 10 {
    process(n);
}"#.to_string(),
            ),
            (
                "Pattern Matching".to_string(),
                r#"match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y-axis at {y}")
    case (x, 0):
        print(f"X-axis at {x}")
    case (x, y):
        print(f"Point at ({x}, {y})")"#.to_string(),
                r#"match point {
    (0, 0) => println!("Origin"),
    (0, y) => println!("Y-axis at {}", y),
    (x, 0) => println!("X-axis at {}", x),
    (x, y) => println!("Point at ({}, {})", x, y),
}"#.to_string(),
            ),
            (
                "*args/**kwargs".to_string(),
                r#"def process(*args, **kwargs):
    for arg in args:
        print(arg)
    for k, v in kwargs.items():
        print(f"{k}: {v}")"#.to_string(),
                r#"pub fn process(args: Vec<String>, kwargs: HashMap<String, String>) {
    for arg in args {
        println!("{}", arg);
    }
    for (k, v) in kwargs {
        println!("{}: {}", k, v);
    }
}"#.to_string(),
            ),
        ]
    }
}

/// Helper function to convert snake_case to PascalCase
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_translation() {
        let gen = Generator {
            name: "count_up".to_string(),
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
                "yield i".to_string(),
                "i += 1".to_string(),
                "yield i".to_string(),
            ],
            is_async: false,
        };

        let rust_code = AdvancedFeaturesTranslator::translate_generator(&gen);
        assert!(rust_code.contains("CountUpIterator"));
        assert!(rust_code.contains("impl Iterator"));
        assert!(rust_code.contains("type Item = i32"));
    }

    #[test]
    fn test_dict_comprehension() {
        let comp = Comprehension {
            comp_type: ComprehensionType::Dict,
            expression: "v * 2".to_string(),
            key_expr: Some("k".to_string()),
            iterables: vec![IterClause {
                target: "(k, v)".to_string(),
                iter_expr: "items.iter()".to_string(),
            }],
            conditions: vec!["v > 0".to_string()],
        };

        let rust_code = AdvancedFeaturesTranslator::translate_dict_comprehension(&comp);
        assert!(rust_code.contains("HashMap::new()"));
        assert!(rust_code.contains("result.insert"));
        assert!(rust_code.contains("if v > 0"));
    }

    #[test]
    fn test_set_comprehension() {
        let comp = Comprehension {
            comp_type: ComprehensionType::Set,
            expression: "x * x".to_string(),
            key_expr: None,
            iterables: vec![IterClause {
                target: "x".to_string(),
                iter_expr: "numbers.iter()".to_string(),
            }],
            conditions: vec![],
        };

        let rust_code = AdvancedFeaturesTranslator::translate_dict_comprehension(&comp);
        assert!(rust_code.contains("HashSet::new()"));
        assert!(rust_code.contains("result.insert"));
    }

    #[test]
    fn test_walrus_operator() {
        let expr = WalrusExpr {
            target: "n".to_string(),
            value: "data.len()".to_string(),
        };

        let rust_code = AdvancedFeaturesTranslator::translate_walrus(&expr, "n > 10");
        assert!(rust_code.contains("let n = data.len()"));
    }

    #[test]
    fn test_pattern_matching() {
        let cases = vec![
            MatchCase {
                pattern: Pattern::Literal("0".to_string()),
                guard: None,
                body: vec!["println!(\"Zero\")".to_string()],
            },
            MatchCase {
                pattern: Pattern::Variable("n".to_string()),
                guard: Some("n > 0".to_string()),
                body: vec!["println!(\"Positive: {}\", n)".to_string()],
            },
            MatchCase {
                pattern: Pattern::Wildcard,
                guard: None,
                body: vec!["println!(\"Other\")".to_string()],
            },
        ];

        let rust_code = AdvancedFeaturesTranslator::translate_pattern_match("value", &cases);
        assert!(rust_code.contains("match value"));
        assert!(rust_code.contains("if n > 0"));
        assert!(rust_code.contains("_ =>"));
    }

    #[test]
    fn test_args_kwargs() {
        let params = vec![
            Parameter {
                name: "args".to_string(),
                param_type: "String".to_string(),
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
        assert!(params_str.contains("Vec<String>"));
        assert!(params_str.contains("HashMap<String, String>"));
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("my_function"), "MyFunction");
        assert_eq!(to_pascal_case("fibonacci"), "Fibonacci");
        assert_eq!(to_pascal_case("count_up_to"), "CountUpTo");
    }
}
