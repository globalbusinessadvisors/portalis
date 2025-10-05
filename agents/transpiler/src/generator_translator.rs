//! Python Generator to Rust Iterator Translation
//!
//! This module translates Python generators (functions with yield) to Rust iterators
//! using the Iterator trait and state machines.

use crate::python_ast::{PyExpr, PyStmt, TypeAnnotation};
use std::collections::HashSet;

/// Information about a generator function
#[derive(Debug, Clone)]
pub struct GeneratorInfo {
    pub name: String,
    pub params: Vec<(String, Option<TypeAnnotation>)>,
    pub yield_type: Option<TypeAnnotation>,
    pub body: Vec<PyStmt>,
    pub is_async: bool,
}

/// Represents the state of a generator
#[derive(Debug, Clone)]
pub struct GeneratorState {
    /// State enum variants (one per yield point)
    pub states: Vec<String>,
    /// Variables that need to be preserved across yields
    pub state_vars: HashSet<String>,
}

/// Strategy for translating generators
#[derive(Debug, Clone)]
pub enum GeneratorStrategy {
    /// Simple iterator using a closure
    SimpleClosure,
    /// State machine with explicit states
    StateMachine,
    /// Using gen! macro (if available)
    GenMacro,
    /// Stream for async generators
    AsyncStream,
}

/// Main generator translator
pub struct GeneratorTranslator {
    /// Current state counter for generating state names
    state_counter: usize,
}

impl Default for GeneratorTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratorTranslator {
    pub fn new() -> Self {
        Self { state_counter: 0 }
    }

    /// Determine the best strategy for translating a generator
    pub fn choose_strategy(&self, info: &GeneratorInfo) -> GeneratorStrategy {
        if info.is_async {
            GeneratorStrategy::AsyncStream
        } else if self.count_yields(&info.body) <= 1 {
            GeneratorStrategy::SimpleClosure
        } else {
            GeneratorStrategy::StateMachine
        }
    }

    /// Count yield statements in the body
    fn count_yields(&self, body: &[PyStmt]) -> usize {
        let mut count = 0;
        for stmt in body {
            count += self.count_yields_in_stmt(stmt);
        }
        count
    }

    fn count_yields_in_stmt(&self, stmt: &PyStmt) -> usize {
        match stmt {
            PyStmt::Expr(expr) => self.count_yields_in_expr(expr),
            PyStmt::If { body, orelse, .. } => {
                let mut count = 0;
                for s in body {
                    count += self.count_yields_in_stmt(s);
                }
                for s in orelse {
                    count += self.count_yields_in_stmt(s);
                }
                count
            }
            PyStmt::While { body, .. } | PyStmt::For { body, .. } => {
                let mut count = 0;
                for s in body {
                    count += self.count_yields_in_stmt(s);
                }
                count
            }
            _ => 0,
        }
    }

    fn count_yields_in_expr(&self, expr: &PyExpr) -> usize {
        match expr {
            PyExpr::Yield(_) => 1,
            _ => 0,
        }
    }

    /// Translate a generator function to an iterator
    pub fn translate_generator(&mut self, info: &GeneratorInfo) -> String {
        let strategy = self.choose_strategy(info);

        match strategy {
            GeneratorStrategy::SimpleClosure => self.translate_simple_closure(info),
            GeneratorStrategy::StateMachine => self.translate_state_machine(info),
            GeneratorStrategy::GenMacro => self.translate_gen_macro(info),
            GeneratorStrategy::AsyncStream => self.translate_async_stream(info),
        }
    }

    /// Translate simple generator to closure-based iterator
    fn translate_simple_closure(&self, info: &GeneratorInfo) -> String {
        let params = self.format_params(&info.params);
        let yield_type = self.extract_yield_type(&info.yield_type);

        // Extract the yield value
        let yield_value = self.extract_first_yield_value(&info.body);

        format!(
            r#"pub fn {}<{}>({}) -> impl Iterator<Item = {}> {{
    std::iter::once({})
}}"#,
            info.name,
            self.extract_generic_params(&info.params),
            params,
            yield_type,
            yield_value
        )
    }

    /// Translate generator to state machine iterator
    fn translate_state_machine(&mut self, info: &GeneratorInfo) -> String {
        let struct_name = self.to_pascal_case(&info.name);
        let params = self.format_params(&info.params);
        let yield_type = self.extract_yield_type(&info.yield_type);

        // Analyze generator to extract state information
        let state_info = self.analyze_generator_state(info);

        let mut code = String::new();

        // Generate state enum
        code.push_str(&self.generate_state_enum(&struct_name, &state_info));
        code.push_str("\n\n");

        // Generate iterator struct
        code.push_str(&self.generate_iterator_struct(&struct_name, info, &state_info));
        code.push_str("\n\n");

        // Generate Iterator impl
        code.push_str(&self.generate_iterator_impl(&struct_name, info, &state_info, &yield_type));
        code.push_str("\n\n");

        // Generate constructor function
        code.push_str(&format!(
            r#"pub fn {}<{}>({}) -> {} {{
    {} {{
        state: {}State::Start,
{}
    }}
}}"#,
            info.name,
            self.extract_generic_params(&info.params),
            params,
            struct_name,
            struct_name,
            struct_name,
            self.format_struct_init_fields(&info.params)
        ));

        code
    }

    /// Generate state enum
    fn generate_state_enum(&self, struct_name: &str, state_info: &GeneratorState) -> String {
        let mut code = format!("#[derive(Debug, Clone)]\nenum {}State {{\n", struct_name);
        code.push_str("    Start,\n");

        for (i, state) in state_info.states.iter().enumerate() {
            code.push_str(&format!("    {},\n", state));
        }

        code.push_str("    Done,\n");
        code.push_str("}");

        code
    }

    /// Generate iterator struct
    fn generate_iterator_struct(
        &self,
        struct_name: &str,
        info: &GeneratorInfo,
        state_info: &GeneratorState,
    ) -> String {
        let mut code = format!("#[derive(Debug)]\npub struct {} {{\n", struct_name);
        code.push_str(&format!("    state: {}State,\n", struct_name));

        // Add parameter fields
        for (name, type_ann) in &info.params {
            let rust_type = self.python_type_to_rust(type_ann.as_ref());
            code.push_str(&format!("    {}: {},\n", name, rust_type));
        }

        // Add state variables
        for var in &state_info.state_vars {
            code.push_str(&format!("    {}: Option<i32>, // TODO: infer type\n", var));
        }

        code.push_str("}");
        code
    }

    /// Generate Iterator trait implementation
    fn generate_iterator_impl(
        &mut self,
        struct_name: &str,
        info: &GeneratorInfo,
        state_info: &GeneratorState,
        yield_type: &str,
    ) -> String {
        let mut code = format!(
            "impl Iterator for {} {{\n    type Item = {};\n\n    fn next(&mut self) -> Option<Self::Item> {{\n",
            struct_name, yield_type
        );

        code.push_str("        match self.state {\n");
        code.push_str("            Self::State::Start => {\n");

        // Generate code for initial state
        let (first_yield, next_state) = self.extract_first_yield_and_state(&info.body);
        code.push_str(&format!("                self.state = {}State::{};\n", struct_name, next_state));
        code.push_str(&format!("                Some({})\n", first_yield));
        code.push_str("            }\n");

        // Generate code for intermediate states
        for (i, state) in state_info.states.iter().enumerate() {
            code.push_str(&format!("            {}State::{} => {{\n", struct_name, state));
            code.push_str("                // TODO: Generate state transition\n");
            code.push_str("                self.state = {}State::Done;\n");
            code.push_str("                None\n");
            code.push_str("            }\n");
        }

        code.push_str(&format!("            {}State::Done => None,\n", struct_name));
        code.push_str("        }\n");
        code.push_str("    }\n");
        code.push_str("}");

        code
    }

    /// Translate using gen! macro (hypothetical)
    fn translate_gen_macro(&self, info: &GeneratorInfo) -> String {
        let params = self.format_params(&info.params);
        let yield_type = self.extract_yield_type(&info.yield_type);

        format!(
            r#"pub fn {}<{}>({}) -> impl Iterator<Item = {}> {{
    gen! {{
        // Generator body
        // TODO: Translate body with yield statements
    }}
}}"#,
            info.name,
            self.extract_generic_params(&info.params),
            params,
            yield_type
        )
    }

    /// Translate async generator to Stream
    fn translate_async_stream(&self, info: &GeneratorInfo) -> String {
        let params = self.format_params(&info.params);
        let yield_type = self.extract_yield_type(&info.yield_type);

        format!(
            r#"pub async fn {}<{}>({}) -> impl Stream<Item = {}> {{
    stream! {{
        // Async generator body
        // TODO: Translate body with yield statements
    }}
}}"#,
            info.name,
            self.extract_generic_params(&info.params),
            params,
            yield_type
        )
    }

    /// Analyze generator to extract state information
    fn analyze_generator_state(&mut self, info: &GeneratorInfo) -> GeneratorState {
        let yield_count = self.count_yields(&info.body);
        let mut states = Vec::new();

        for i in 0..yield_count.saturating_sub(1) {
            states.push(format!("State{}", i + 1));
        }

        GeneratorState {
            states,
            state_vars: HashSet::new(),
        }
    }

    /// Extract first yield value from body
    fn extract_first_yield_value(&self, body: &[PyStmt]) -> String {
        for stmt in body {
            if let Some(value) = self.find_yield_value(stmt) {
                return value;
            }
        }
        "()".to_string()
    }

    fn find_yield_value(&self, stmt: &PyStmt) -> Option<String> {
        match stmt {
            PyStmt::Expr(expr) => {
                if let PyExpr::Yield(Some(val)) = expr {
                    return Some(self.simple_expr_to_string(val));
                }
                None
            }
            PyStmt::If { body, orelse, .. } => {
                for s in body {
                    if let Some(v) = self.find_yield_value(s) {
                        return Some(v);
                    }
                }
                for s in orelse {
                    if let Some(v) = self.find_yield_value(s) {
                        return Some(v);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn extract_first_yield_and_state(&self, body: &[PyStmt]) -> (String, String) {
        let value = self.extract_first_yield_value(body);
        (value, "State1".to_string())
    }

    fn simple_expr_to_string(&self, expr: &PyExpr) -> String {
        match expr {
            PyExpr::Literal(lit) => format!("{:?}", lit),
            PyExpr::Name(name) => name.clone(),
            _ => "()".to_string(),
        }
    }

    // Helper functions

    fn format_params(&self, params: &[(String, Option<TypeAnnotation>)]) -> String {
        params
            .iter()
            .map(|(name, type_ann)| {
                let rust_type = self.python_type_to_rust(type_ann.as_ref());
                format!("{}: {}", name, rust_type)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn format_struct_init_fields(&self, params: &[(String, Option<TypeAnnotation>)]) -> String {
        params
            .iter()
            .map(|(name, _)| format!("        {},", name))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn extract_generic_params(&self, _params: &[(String, Option<TypeAnnotation>)]) -> String {
        // For now, no generics
        String::new()
    }

    fn extract_yield_type(&self, yield_type: &Option<TypeAnnotation>) -> String {
        match yield_type {
            Some(t) => self.python_type_to_rust(Some(t)),
            None => "()".to_string(),
        }
    }

    fn python_type_to_rust(&self, type_ann: Option<&TypeAnnotation>) -> String {
        match type_ann {
            Some(TypeAnnotation::Name(name)) => self.python_type_to_rust_name(name),
            Some(TypeAnnotation::Generic { base, args }) => {
                let base_type = self.python_type_to_rust(Some(base));
                let arg_types = args
                    .iter()
                    .map(|arg| self.python_type_to_rust(Some(arg)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}<{}>", base_type, arg_types)
            }
            None => "()".to_string(),
        }
    }

    fn python_type_to_rust_name(&self, name: &str) -> String {
        match name {
            "int" => "i32",
            "float" => "f64",
            "str" => "String",
            "bool" => "bool",
            "list" => "Vec",
            "dict" => "HashMap",
            "set" => "HashSet",
            "tuple" => "tuple",
            "None" => "()",
            _ => name,
        }
        .to_string()
    }

    fn to_pascal_case(&self, snake_case: &str) -> String {
        snake_case
            .split('_')
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                    None => String::new(),
                }
            })
            .collect()
    }
}

/// Generate helper code for generators
pub fn generate_generator_helpers() -> String {
    r#"// Generator helper utilities

/// Create an iterator from a range
pub fn range(start: i32, end: i32) -> impl Iterator<Item = i32> {
    start..end
}

/// Create an iterator with a step
pub fn range_step(start: i32, end: i32, step: i32) -> impl Iterator<Item = i32> {
    (start..end).step_by(step.abs() as usize)
}

/// Infinite iterator from a value
pub fn infinite_from(start: i32) -> impl Iterator<Item = i32> {
    (start..).into_iter()
}

/// Repeat a value n times
pub fn repeat_n<T: Clone>(value: T, n: usize) -> impl Iterator<Item = T> {
    std::iter::repeat(value).take(n)
}
"#
    .to_string()
}

/// Translate generator expressions (list comprehensions that yield)
pub fn translate_generator_expr(
    element: &PyExpr,
    generators: &[(PyExpr, PyExpr)],
) -> String {
    // Simple translation for generator expressions
    if generators.len() == 1 {
        let (target, iter) = &generators[0];
        format!(
            "{}.map(|{}| {})",
            format!("{:?}", iter), // TODO: proper translation
            format!("{:?}", target),
            format!("{:?}", element)
        )
    } else {
        // Nested generators
        format!("// TODO: nested generator expression")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_yields() {
        let translator = GeneratorTranslator::new();
        let body = vec![PyStmt::Expr(PyExpr::Yield(Some(Box::new(
            PyExpr::Literal(crate::python_ast::PyLiteral::Int(42)),
        ))))];

        assert_eq!(translator.count_yields(&body), 1);
    }

    #[test]
    fn test_choose_strategy_simple() {
        let translator = GeneratorTranslator::new();
        let info = GeneratorInfo {
            name: "simple_gen".to_string(),
            params: vec![],
            yield_type: Some(TypeAnnotation::Name("int".to_string())),
            body: vec![PyStmt::Expr(PyExpr::Yield(Some(Box::new(
                PyExpr::Literal(crate::python_ast::PyLiteral::Int(42)),
            ))))],
            is_async: false,
        };

        match translator.choose_strategy(&info) {
            GeneratorStrategy::SimpleClosure => (),
            _ => panic!("Expected SimpleClosure strategy"),
        }
    }

    #[test]
    fn test_to_pascal_case() {
        let translator = GeneratorTranslator::new();
        assert_eq!(translator.to_pascal_case("my_generator"), "MyGenerator");
        assert_eq!(translator.to_pascal_case("simple"), "Simple");
    }
}
