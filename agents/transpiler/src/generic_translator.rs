//! Generic Type Translation
//!
//! Translates Python's generic types to Rust's generic system, including:
//! - Type parameters (T, K, V)
//! - Generic functions and classes
//! - Type bounds and constraints
//! - Variance annotations
//! - Associated types

use std::collections::{HashMap, HashSet};

/// Represents a generic type parameter
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeParam {
    pub name: String,
    pub bounds: Vec<TypeBound>,
    pub default: Option<String>,
    pub variance: Variance,
}

impl TypeParam {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bounds: Vec::new(),
            default: None,
            variance: Variance::Invariant,
        }
    }

    pub fn with_bound(mut self, bound: TypeBound) -> Self {
        self.bounds.push(bound);
        self
    }

    pub fn with_bounds(mut self, bounds: Vec<TypeBound>) -> Self {
        self.bounds = bounds;
        self
    }

    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default = Some(default.into());
        self
    }

    pub fn with_variance(mut self, variance: Variance) -> Self {
        self.variance = variance;
        self
    }

    /// Convert to Rust generic parameter syntax
    pub fn to_rust(&self) -> String {
        let mut result = self.name.clone();

        if !self.bounds.is_empty() {
            let bounds_str: Vec<String> = self.bounds.iter()
                .map(|b| b.to_rust())
                .collect();
            result.push_str(": ");
            result.push_str(&bounds_str.join(" + "));
        }

        result
    }
}

/// Type bounds/constraints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeBound {
    /// Trait bound (e.g., T: Clone)
    Trait(String),
    /// Lifetime bound (e.g., T: 'a)
    Lifetime(String),
    /// Sized bound
    Sized,
    /// ?Sized relaxation
    MaybeSized,
    /// Multiple bounds combined
    Combined(Vec<TypeBound>),
}

impl TypeBound {
    pub fn to_rust(&self) -> String {
        match self {
            TypeBound::Trait(name) => name.clone(),
            TypeBound::Lifetime(lt) => format!("'{}", lt),
            TypeBound::Sized => "Sized".to_string(),
            TypeBound::MaybeSized => "?Sized".to_string(),
            TypeBound::Combined(bounds) => {
                bounds.iter()
                    .map(|b| b.to_rust())
                    .collect::<Vec<_>>()
                    .join(" + ")
            }
        }
    }
}

/// Variance annotations for generic types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Variance {
    /// Covariant (out in Python, default in Rust)
    Covariant,
    /// Contravariant (in in Python)
    Contravariant,
    /// Invariant (default in Python)
    Invariant,
}

/// Generic function signature
#[derive(Debug, Clone)]
pub struct GenericFunction {
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<(String, String)>,  // (name, type)
    pub return_type: Option<String>,
    pub where_clauses: Vec<WhereClause>,
}

impl GenericFunction {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            type_params: Vec::new(),
            params: Vec::new(),
            return_type: None,
            where_clauses: Vec::new(),
        }
    }

    pub fn to_rust(&self) -> String {
        let mut result = String::new();

        // Function signature
        result.push_str("fn ");
        result.push_str(&self.name);

        // Type parameters
        if !self.type_params.is_empty() {
            result.push('<');
            let params_str: Vec<String> = self.type_params.iter()
                .map(|p| p.to_rust())
                .collect();
            result.push_str(&params_str.join(", "));
            result.push('>');
        }

        // Function parameters
        result.push('(');
        let params_str: Vec<String> = self.params.iter()
            .map(|(name, ty)| format!("{}: {}", name, ty))
            .collect();
        result.push_str(&params_str.join(", "));
        result.push(')');

        // Return type
        if let Some(ret) = &self.return_type {
            result.push_str(" -> ");
            result.push_str(ret);
        }

        // Where clauses
        if !self.where_clauses.is_empty() {
            result.push_str("\nwhere\n");
            for clause in &self.where_clauses {
                result.push_str("    ");
                result.push_str(&clause.to_rust());
                result.push_str(",\n");
            }
        }

        result
    }
}

/// Where clause for complex constraints
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub type_name: String,
    pub bounds: Vec<TypeBound>,
}

impl WhereClause {
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            bounds: Vec::new(),
        }
    }

    pub fn with_bound(mut self, bound: TypeBound) -> Self {
        self.bounds.push(bound);
        self
    }

    pub fn to_rust(&self) -> String {
        let bounds_str: Vec<String> = self.bounds.iter()
            .map(|b| b.to_rust())
            .collect();
        format!("{}: {}", self.type_name, bounds_str.join(" + "))
    }
}

/// Generic struct/class definition
#[derive(Debug, Clone)]
pub struct GenericStruct {
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub fields: Vec<(String, String)>,  // (name, type)
    pub derives: Vec<String>,
    pub where_clauses: Vec<WhereClause>,
}

impl GenericStruct {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            type_params: Vec::new(),
            fields: Vec::new(),
            derives: Vec::new(),
            where_clauses: Vec::new(),
        }
    }

    pub fn to_rust(&self) -> String {
        let mut result = String::new();

        // Derives
        if !self.derives.is_empty() {
            result.push_str("#[derive(");
            result.push_str(&self.derives.join(", "));
            result.push_str(")]\n");
        }

        // Struct definition
        result.push_str("pub struct ");
        result.push_str(&self.name);

        // Type parameters
        if !self.type_params.is_empty() {
            result.push('<');
            let params_str: Vec<String> = self.type_params.iter()
                .map(|p| p.to_rust())
                .collect();
            result.push_str(&params_str.join(", "));
            result.push('>');
        }

        // Where clauses
        if !self.where_clauses.is_empty() {
            result.push_str("\nwhere\n");
            for clause in &self.where_clauses {
                result.push_str("    ");
                result.push_str(&clause.to_rust());
                result.push_str(",\n");
            }
        }

        // Fields
        result.push_str(" {\n");
        for (name, ty) in &self.fields {
            result.push_str("    pub ");
            result.push_str(name);
            result.push_str(": ");
            result.push_str(ty);
            result.push_str(",\n");
        }
        result.push_str("}\n");

        result
    }
}

/// Generic type translator
pub struct GenericTranslator {
    /// Mapping of Python generic types to Rust
    type_mappings: HashMap<String, String>,
    /// Active type parameters in scope
    active_params: HashSet<String>,
    /// Python typing module imports
    typing_imports: HashSet<String>,
}

impl GenericTranslator {
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // Python typing → Rust std types
        mappings.insert("List".to_string(), "Vec".to_string());
        mappings.insert("Dict".to_string(), "HashMap".to_string());
        mappings.insert("Set".to_string(), "HashSet".to_string());
        mappings.insert("Tuple".to_string(), "tuple".to_string());
        mappings.insert("Optional".to_string(), "Option".to_string());
        mappings.insert("Union".to_string(), "enum".to_string());
        mappings.insert("Callable".to_string(), "Fn".to_string());
        mappings.insert("Iterable".to_string(), "IntoIterator".to_string());
        mappings.insert("Iterator".to_string(), "Iterator".to_string());
        mappings.insert("Sequence".to_string(), "Vec".to_string());
        mappings.insert("Mapping".to_string(), "HashMap".to_string());

        Self {
            type_mappings: mappings,
            active_params: HashSet::new(),
            typing_imports: HashSet::new(),
        }
    }

    /// Translate Python generic type to Rust
    pub fn translate_type(&self, python_type: &str) -> String {
        // Handle simple types
        if let Some(rust_type) = self.type_mappings.get(python_type) {
            return rust_type.clone();
        }

        // Handle parameterized types (e.g., List[int])
        if python_type.contains('[') {
            return self.translate_parameterized_type(python_type);
        }

        // Check if it's a type parameter
        if self.active_params.contains(python_type) {
            return python_type.to_string();
        }

        // Default: use as-is
        python_type.to_string()
    }

    /// Translate parameterized type (e.g., List[int] → Vec<i32>)
    fn translate_parameterized_type(&self, python_type: &str) -> String {
        let parts: Vec<&str> = python_type.splitn(2, '[').collect();
        if parts.len() != 2 {
            return python_type.to_string();
        }

        let base = parts[0].trim();
        let params_str = parts[1].trim_end_matches(']');

        let rust_base = self.type_mappings.get(base)
            .unwrap_or(&base.to_string())
            .clone();

        // Handle multiple parameters (e.g., Dict[str, int])
        let params: Vec<String> = params_str.split(',')
            .map(|p| self.translate_type(p.trim()))
            .collect();

        // Special handling for specific types
        match base {
            "Optional" => {
                format!("Option<{}>", params[0])
            }
            "Union" => {
                // Union types become enums or Result in Rust
                if params.len() == 2 && params.contains(&"None".to_string()) {
                    let other = params.iter()
                        .find(|p| *p != "None")
                        .unwrap();
                    format!("Option<{}>", other)
                } else {
                    format!("/* Union<{}> */", params.join(", "))
                }
            }
            "Callable" => {
                if params.len() == 2 {
                    let args = &params[0];
                    let ret = &params[1];
                    format!("impl Fn({}) -> {}", args, ret)
                } else {
                    "Box<dyn Fn()>".to_string()
                }
            }
            "Tuple" => {
                format!("({})", params.join(", "))
            }
            _ => {
                format!("{}<{}>", rust_base, params.join(", "))
            }
        }
    }

    /// Parse Python generic function definition
    pub fn parse_generic_function(&mut self, python_def: &str) -> GenericFunction {
        // Simplified parser for: def foo[T](x: T) -> T:
        // or: def foo[T: Comparable](x: T) -> T:

        let mut func = GenericFunction::new("generic_fn");

        // Extract function name
        if let Some(name_end) = python_def.find('[').or_else(|| python_def.find('(')) {
            let name_start = python_def.find("def ").unwrap_or(0) + 4;
            func.name = python_def[name_start..name_end].trim().to_string();
        }

        // Extract type parameters [T, K, V]
        if let Some(start) = python_def.find('[') {
            if let Some(end) = python_def.find(']') {
                let params_str = &python_def[start + 1..end];
                for param_def in params_str.split(',') {
                    let param_def = param_def.trim();

                    if param_def.contains(':') {
                        // Has bound: T: Comparable
                        let parts: Vec<&str> = param_def.splitn(2, ':').collect();
                        let name = parts[0].trim();
                        let bound = parts[1].trim();

                        let type_param = TypeParam::new(name)
                            .with_bound(self.python_bound_to_rust(bound));
                        self.active_params.insert(name.to_string());
                        func.type_params.push(type_param);
                    } else {
                        // Simple type parameter
                        let type_param = TypeParam::new(param_def);
                        self.active_params.insert(param_def.to_string());
                        func.type_params.push(type_param);
                    }
                }
            }
        }

        func
    }

    /// Parse Python generic class definition
    pub fn parse_generic_class(&mut self, python_def: &str) -> GenericStruct {
        // Simplified parser for: class Container[T]:
        // or: class Pair[T, U]:

        let mut struct_def = GenericStruct::new("GenericStruct");

        // Extract class name
        if let Some(name_end) = python_def.find('[').or_else(|| python_def.find(':')) {
            let name_start = python_def.find("class ").unwrap_or(0) + 6;
            struct_def.name = python_def[name_start..name_end].trim().to_string();
        }

        // Extract type parameters
        if let Some(start) = python_def.find('[') {
            if let Some(end) = python_def.find(']') {
                let params_str = &python_def[start + 1..end];
                for param_def in params_str.split(',') {
                    let param_def = param_def.trim();

                    if param_def.contains(':') {
                        let parts: Vec<&str> = param_def.splitn(2, ':').collect();
                        let name = parts[0].trim();
                        let bound = parts[1].trim();

                        let type_param = TypeParam::new(name)
                            .with_bound(self.python_bound_to_rust(bound));
                        self.active_params.insert(name.to_string());
                        struct_def.type_params.push(type_param);
                    } else {
                        let type_param = TypeParam::new(param_def);
                        self.active_params.insert(param_def.to_string());
                        struct_def.type_params.push(type_param);
                    }
                }
            }
        }

        struct_def
    }

    /// Convert Python type bound to Rust trait bound
    fn python_bound_to_rust(&self, python_bound: &str) -> TypeBound {
        match python_bound {
            "Comparable" | "SupportsLt" => TypeBound::Trait("Ord".to_string()),
            "Hashable" => TypeBound::Trait("Hash".to_string()),
            "Sized" => TypeBound::Sized,
            "Iterable" => TypeBound::Trait("IntoIterator".to_string()),
            "Callable" => TypeBound::Trait("Fn()".to_string()),
            _ => TypeBound::Trait(python_bound.to_string()),
        }
    }

    /// Add common trait bounds based on usage patterns
    pub fn infer_bounds(&self, type_param: &str, usage_patterns: &[&str]) -> Vec<TypeBound> {
        let mut bounds = Vec::new();

        for pattern in usage_patterns {
            match *pattern {
                "clone" => bounds.push(TypeBound::Trait("Clone".to_string())),
                "copy" => bounds.push(TypeBound::Trait("Copy".to_string())),
                "compare" | "<" | ">" | "<=" | ">=" => {
                    if !bounds.iter().any(|b| matches!(b, TypeBound::Trait(s) if s == "Ord")) {
                        bounds.push(TypeBound::Trait("Ord".to_string()));
                    }
                }
                "==" | "!=" => {
                    if !bounds.iter().any(|b| matches!(b, TypeBound::Trait(s) if s == "PartialEq")) {
                        bounds.push(TypeBound::Trait("PartialEq".to_string()));
                    }
                }
                "hash" => bounds.push(TypeBound::Trait("Hash".to_string())),
                "print" | "format" => bounds.push(TypeBound::Trait("std::fmt::Display".to_string())),
                "iter" => bounds.push(TypeBound::Trait("IntoIterator".to_string())),
                _ => {}
            }
        }

        bounds
    }

    /// Generate impl block for generic struct
    pub fn generate_impl_block(
        &self,
        struct_name: &str,
        type_params: &[TypeParam],
        methods: &[(String, GenericFunction)],
    ) -> String {
        let mut result = String::new();

        // impl<T> StructName<T>
        result.push_str("impl<");
        let params_str: Vec<String> = type_params.iter()
            .map(|p| p.to_rust())
            .collect();
        result.push_str(&params_str.join(", "));
        result.push_str("> ");
        result.push_str(struct_name);
        result.push('<');
        result.push_str(&type_params.iter()
            .map(|p| p.name.clone())
            .collect::<Vec<_>>()
            .join(", "));
        result.push_str("> {\n");

        // Methods
        for (method_name, func) in methods {
            result.push_str("    pub ");
            result.push_str(&func.to_rust());
            result.push_str(" {\n");
            result.push_str("        todo!()\n");
            result.push_str("    }\n\n");
        }

        result.push_str("}\n");

        result
    }

    /// Generate associated type trait
    pub fn generate_associated_type_trait(
        &self,
        trait_name: &str,
        associated_types: &[(&str, Vec<TypeBound>)],
        methods: &[GenericFunction],
    ) -> String {
        let mut result = String::new();

        result.push_str("pub trait ");
        result.push_str(trait_name);
        result.push_str(" {\n");

        // Associated types
        for (type_name, bounds) in associated_types {
            result.push_str("    type ");
            result.push_str(type_name);
            if !bounds.is_empty() {
                result.push_str(": ");
                let bounds_str: Vec<String> = bounds.iter()
                    .map(|b| b.to_rust())
                    .collect();
                result.push_str(&bounds_str.join(" + "));
            }
            result.push_str(";\n");
        }

        result.push('\n');

        // Methods
        for method in methods {
            result.push_str("    ");
            result.push_str(&method.to_rust());
            result.push_str(";\n");
        }

        result.push_str("}\n");

        result
    }

    /// Clear active type parameters (when exiting scope)
    pub fn clear_params(&mut self) {
        self.active_params.clear();
    }
}

impl Default for GenericTranslator {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for common generic patterns
pub struct GenericPatterns;

impl GenericPatterns {
    /// Generic container pattern
    pub fn container_pattern() -> String {
        r#"
#[derive(Debug, Clone)]
pub struct Container<T> {
    pub items: Vec<T>,
}

impl<T> Container<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }
}
"#.to_string()
    }

    /// Generic pair pattern
    pub fn pair_pattern() -> String {
        r#"
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair<T, U> {
    pub first: T,
    pub second: U,
}

impl<T, U> Pair<T, U> {
    pub fn new(first: T, second: U) -> Self {
        Self { first, second }
    }

    pub fn swap(self) -> Pair<U, T> {
        Pair::new(self.second, self.first)
    }
}
"#.to_string()
    }

    /// Generic trait with associated type
    pub fn associated_type_pattern() -> String {
        r#"
pub trait Container {
    type Item;

    fn get(&self, index: usize) -> Option<&Self::Item>;
    fn len(&self) -> usize;
}

impl<T> Container for Vec<T> {
    type Item = T;

    fn get(&self, index: usize) -> Option<&Self::Item> {
        self.get(index)
    }

    fn len(&self) -> usize {
        self.len()
    }
}
"#.to_string()
    }

    /// Generic function with multiple bounds
    pub fn bounded_function_pattern() -> String {
        r#"
fn compare_and_print<T>(a: T, b: T) -> T
where
    T: Ord + Clone + std::fmt::Display,
{
    let max = if a > b { a.clone() } else { b.clone() };
    println!("Max: {}", max);
    max
}
"#.to_string()
    }

    /// Generic Result-like enum
    pub fn result_pattern() -> String {
        r#"
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> Result<T, E> {
    pub fn is_ok(&self) -> bool {
        matches!(self, Result::Ok(_))
    }

    pub fn unwrap(self) -> T {
        match self {
            Result::Ok(value) => value,
            Result::Err(_) => panic!("called unwrap on Err"),
        }
    }
}
"#.to_string()
    }

    /// Phantom data pattern for zero-sized type parameters
    pub fn phantom_data_pattern() -> String {
        r#"
use std::marker::PhantomData;

pub struct TypedId<T> {
    id: u64,
    _phantom: PhantomData<T>,
}

impl<T> TypedId<T> {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }
}
"#.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_param_to_rust() {
        let param = TypeParam::new("T")
            .with_bound(TypeBound::Trait("Clone".to_string()))
            .with_bound(TypeBound::Trait("Debug".to_string()));

        assert_eq!(param.to_rust(), "T: Clone + Debug");
    }

    #[test]
    fn test_translate_simple_type() {
        let translator = GenericTranslator::new();
        assert_eq!(translator.translate_type("List"), "Vec");
        assert_eq!(translator.translate_type("Dict"), "HashMap");
        assert_eq!(translator.translate_type("Optional"), "Option");
    }

    #[test]
    fn test_translate_parameterized_type() {
        let translator = GenericTranslator::new();
        assert_eq!(translator.translate_type("List[int]"), "Vec<int>");
        assert_eq!(translator.translate_type("Dict[str, int]"), "HashMap<str, int>");
        assert_eq!(translator.translate_type("Optional[str]"), "Option<str>");
    }

    #[test]
    fn test_generic_function() {
        let func = GenericFunction::new("identity")
            .type_params.push(TypeParam::new("T"));

        let mut func = GenericFunction::new("identity");
        func.type_params.push(TypeParam::new("T"));
        func.params.push(("x".to_string(), "T".to_string()));
        func.return_type = Some("T".to_string());

        let rust = func.to_rust();
        assert!(rust.contains("fn identity<T>"));
        assert!(rust.contains("x: T"));
        assert!(rust.contains("-> T"));
    }

    #[test]
    fn test_infer_bounds() {
        let translator = GenericTranslator::new();
        let bounds = translator.infer_bounds("T", &["clone", "compare", "print"]);

        assert!(bounds.iter().any(|b| matches!(b, TypeBound::Trait(s) if s == "Clone")));
        assert!(bounds.iter().any(|b| matches!(b, TypeBound::Trait(s) if s == "Ord")));
        assert!(bounds.iter().any(|b| matches!(b, TypeBound::Trait(s) if s == "std::fmt::Display")));
    }
}
