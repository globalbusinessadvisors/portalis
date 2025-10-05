//! Lifetime Analysis and Insertion
//!
//! Analyzes Python code patterns and inserts appropriate Rust lifetime annotations
//! for references, ensuring memory safety and preventing dangling references.

use std::collections::{HashMap, HashSet};

/// Represents a Rust lifetime
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lifetime {
    pub name: String,
}

impl Lifetime {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn anonymous() -> Self {
        Self::new("_")
    }

    pub fn static_lifetime() -> Self {
        Self::new("static")
    }

    pub fn to_string(&self) -> String {
        format!("'{}", self.name)
    }
}

/// Type with optional lifetime annotations
#[derive(Debug, Clone)]
pub struct TypeWithLifetime {
    pub base_type: String,
    pub lifetimes: Vec<Lifetime>,
    pub is_reference: bool,
    pub is_mutable: bool,
}

impl TypeWithLifetime {
    pub fn new(base_type: impl Into<String>) -> Self {
        Self {
            base_type: base_type.into(),
            lifetimes: vec![],
            is_reference: false,
            is_mutable: false,
        }
    }

    pub fn with_lifetime(mut self, lifetime: Lifetime) -> Self {
        self.lifetimes.push(lifetime);
        self
    }

    pub fn as_reference(mut self) -> Self {
        self.is_reference = true;
        self
    }

    pub fn as_mut_reference(mut self) -> Self {
        self.is_reference = true;
        self.is_mutable = true;
        self
    }

    pub fn to_rust_string(&self) -> String {
        let mut result = String::new();

        if self.is_reference {
            result.push('&');
            if !self.lifetimes.is_empty() {
                result.push_str(&self.lifetimes[0].to_string());
                result.push(' ');
            }
            if self.is_mutable {
                result.push_str("mut ");
            }
        }

        result.push_str(&self.base_type);

        // Generic lifetime parameters
        if !self.is_reference && !self.lifetimes.is_empty() {
            result.push('<');
            let lifetime_strs: Vec<_> = self.lifetimes.iter().map(|l| l.to_string()).collect();
            result.push_str(&lifetime_strs.join(", "));
            result.push('>');
        }

        result
    }
}

/// Lifetime constraint between lifetimes
#[derive(Debug, Clone)]
pub enum LifetimeConstraint {
    /// 'a outlives 'b ('a: 'b)
    Outlives(Lifetime, Lifetime),
    /// 'a is the same as 'b
    Equal(Lifetime, Lifetime),
    /// 'a must be 'static
    Static(Lifetime),
}

/// Function signature with lifetime annotations
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub lifetime_params: Vec<Lifetime>,
    pub params: Vec<(String, TypeWithLifetime)>,
    pub return_type: Option<TypeWithLifetime>,
    pub constraints: Vec<LifetimeConstraint>,
}

impl FunctionSignature {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            lifetime_params: vec![],
            params: vec![],
            return_type: None,
            constraints: vec![],
        }
    }

    pub fn add_lifetime_param(&mut self, lifetime: Lifetime) {
        if !self.lifetime_params.contains(&lifetime) {
            self.lifetime_params.push(lifetime);
        }
    }

    pub fn add_param(&mut self, name: String, ty: TypeWithLifetime) {
        // Collect lifetimes from parameter type
        for lifetime in &ty.lifetimes {
            self.add_lifetime_param(lifetime.clone());
        }
        self.params.push((name, ty));
    }

    pub fn set_return_type(&mut self, ty: TypeWithLifetime) {
        // Collect lifetimes from return type
        for lifetime in &ty.lifetimes {
            self.add_lifetime_param(lifetime.clone());
        }
        self.return_type = Some(ty);
    }

    pub fn to_rust_string(&self) -> String {
        let mut result = format!("fn {}", self.name);

        // Lifetime parameters
        if !self.lifetime_params.is_empty() {
            result.push('<');
            let lifetime_strs: Vec<_> = self
                .lifetime_params
                .iter()
                .map(|l| l.to_string())
                .collect();
            result.push_str(&lifetime_strs.join(", "));
            result.push('>');
        }

        // Function parameters
        result.push('(');
        let param_strs: Vec<_> = self
            .params
            .iter()
            .map(|(name, ty)| format!("{}: {}", name, ty.to_rust_string()))
            .collect();
        result.push_str(&param_strs.join(", "));
        result.push(')');

        // Return type
        if let Some(ret_ty) = &self.return_type {
            result.push_str(" -> ");
            result.push_str(&ret_ty.to_rust_string());
        }

        result
    }
}

/// Struct definition with lifetime parameters
#[derive(Debug, Clone)]
pub struct StructDefinition {
    pub name: String,
    pub lifetime_params: Vec<Lifetime>,
    pub fields: Vec<(String, TypeWithLifetime)>,
}

impl StructDefinition {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            lifetime_params: vec![],
            fields: vec![],
        }
    }

    pub fn add_field(&mut self, name: String, ty: TypeWithLifetime) {
        // Collect lifetimes from field type
        for lifetime in &ty.lifetimes {
            if !self.lifetime_params.contains(lifetime) {
                self.lifetime_params.push(lifetime.clone());
            }
        }
        self.fields.push((name, ty));
    }

    pub fn to_rust_string(&self) -> String {
        let mut result = format!("struct {}", self.name);

        // Lifetime parameters
        if !self.lifetime_params.is_empty() {
            result.push('<');
            let lifetime_strs: Vec<_> = self
                .lifetime_params
                .iter()
                .map(|l| l.to_string())
                .collect();
            result.push_str(&lifetime_strs.join(", "));
            result.push('>');
        }

        result.push_str(" {\n");

        // Fields
        for (name, ty) in &self.fields {
            result.push_str(&format!("    {}: {},\n", name, ty.to_rust_string()));
        }

        result.push('}');
        result
    }
}

/// Lifetime analyzer
pub struct LifetimeAnalyzer {
    /// Counter for generating fresh lifetime names
    lifetime_counter: usize,
    /// Current scope's lifetime information
    scope_lifetimes: HashMap<String, Lifetime>,
    /// Detected reference patterns
    reference_patterns: Vec<ReferencePattern>,
}

/// Reference pattern detected in code
#[derive(Debug, Clone)]
pub enum ReferencePattern {
    /// Borrowing a variable
    Borrow { var: String, is_mutable: bool },
    /// Returning a reference from a function
    ReturnRef { param: String },
    /// Storing a reference in a struct
    StructRef { field: String, source: String },
    /// Reference in a collection
    CollectionRef { container: String, element: String },
}

impl LifetimeAnalyzer {
    pub fn new() -> Self {
        Self {
            lifetime_counter: 0,
            scope_lifetimes: HashMap::new(),
            reference_patterns: Vec::new(),
        }
    }

    /// Generate a fresh lifetime
    pub fn fresh_lifetime(&mut self) -> Lifetime {
        let name = format!("a{}", self.lifetime_counter);
        self.lifetime_counter += 1;
        Lifetime::new(name)
    }

    /// Analyze function and determine lifetime annotations
    pub fn analyze_function(
        &mut self,
        params: &[(String, String, bool)], // (name, type, is_reference)
        return_type: Option<(&str, bool)>,  // (type, is_reference)
        returns_param: Option<&str>,        // which param is returned
    ) -> FunctionSignature {
        let mut sig = FunctionSignature::new("function");

        // Apply elision rules first
        if let Some(elided) = self.try_elision(params, return_type, returns_param) {
            return elided;
        }

        // Assign lifetimes to reference parameters
        let mut param_lifetimes: HashMap<String, Lifetime> = HashMap::new();

        for (name, ty, is_ref) in params {
            let param_ty = if *is_ref {
                let lifetime = self.fresh_lifetime();
                param_lifetimes.insert(name.clone(), lifetime.clone());
                TypeWithLifetime::new(ty.clone())
                    .as_reference()
                    .with_lifetime(lifetime)
            } else {
                TypeWithLifetime::new(ty.clone())
            };

            sig.add_param(name.clone(), param_ty);
        }

        // Handle return type
        if let Some((ret_ty, is_ref)) = return_type {
            if is_ref {
                // If returning a reference, it must come from a parameter
                if let Some(param_name) = returns_param {
                    if let Some(lifetime) = param_lifetimes.get(param_name) {
                        let return_ty = TypeWithLifetime::new(ret_ty)
                            .as_reference()
                            .with_lifetime(lifetime.clone());
                        sig.set_return_type(return_ty);
                    }
                } else {
                    // Generic lifetime for return
                    let lifetime = self.fresh_lifetime();
                    let return_ty = TypeWithLifetime::new(ret_ty)
                        .as_reference()
                        .with_lifetime(lifetime);
                    sig.set_return_type(return_ty);
                }
            } else {
                sig.set_return_type(TypeWithLifetime::new(ret_ty));
            }
        }

        sig
    }

    /// Try to apply lifetime elision rules
    fn try_elision(
        &mut self,
        params: &[(String, String, bool)],
        return_type: Option<(&str, bool)>,
        returns_param: Option<&str>,
    ) -> Option<FunctionSignature> {
        let ref_params: Vec<_> = params.iter().filter(|(_, _, is_ref)| *is_ref).collect();

        // Rule 1: Each elided lifetime in function arguments gets a distinct parameter
        // Rule 2: If there's exactly one input lifetime, it's assigned to all output lifetimes
        // Rule 3: If there are multiple input lifetimes and one is &self or &mut self, the lifetime of self is assigned to all output lifetimes

        if ref_params.is_empty() {
            // No references, no lifetimes needed
            let mut sig = FunctionSignature::new("function");
            for (name, ty, _) in params {
                sig.add_param(name.clone(), TypeWithLifetime::new(ty.clone()));
            }
            if let Some((ret_ty, _)) = return_type {
                sig.set_return_type(TypeWithLifetime::new(ret_ty));
            }
            return Some(sig);
        }

        if ref_params.len() == 1 {
            // Single reference parameter - elision applies
            if let Some((ret_ty, true)) = return_type {
                let mut sig = FunctionSignature::new("function");

                // Use elided lifetime (not explicitly written)
                for (name, ty, is_ref) in params {
                    if *is_ref {
                        sig.add_param(name.clone(), TypeWithLifetime::new(ty.clone()).as_reference());
                    } else {
                        sig.add_param(name.clone(), TypeWithLifetime::new(ty.clone()));
                    }
                }

                sig.set_return_type(TypeWithLifetime::new(ret_ty).as_reference());
                return Some(sig);
            }
        }

        None
    }

    /// Analyze struct and determine lifetime parameters
    pub fn analyze_struct(&mut self, fields: &[(String, String, bool)]) -> StructDefinition {
        let mut struct_def = StructDefinition::new("Struct");

        // Fields with references need lifetime parameters
        let mut has_references = false;
        let lifetime = self.fresh_lifetime();

        for (name, ty, is_ref) in fields {
            let field_ty = if *is_ref {
                has_references = true;
                TypeWithLifetime::new(ty.clone())
                    .as_reference()
                    .with_lifetime(lifetime.clone())
            } else {
                TypeWithLifetime::new(ty.clone())
            };

            struct_def.add_field(name.clone(), field_ty);
        }

        struct_def
    }

    /// Generate lifetime bounds for trait implementations
    pub fn generate_trait_bounds(&self, lifetimes: &[Lifetime]) -> Vec<String> {
        let mut bounds = Vec::new();

        for lifetime in lifetimes {
            // Common bound: 'a: 'static (lifetime outlives 'static)
            // Or 'a: 'b (one lifetime outlives another)
            bounds.push(format!("{}: 'static", lifetime.to_string()));
        }

        bounds
    }

    /// Detect common lifetime patterns
    pub fn detect_pattern(&mut self, pattern: ReferencePattern) {
        self.reference_patterns.push(pattern);
    }

    /// Generate lifetime annotations for detected patterns
    pub fn generate_annotations(&self) -> Vec<String> {
        let mut annotations = Vec::new();

        for pattern in &self.reference_patterns {
            match pattern {
                ReferencePattern::Borrow { var, is_mutable } => {
                    let mutability = if *is_mutable { "mut " } else { "" };
                    annotations.push(format!("&{}{}", mutability, var));
                }
                ReferencePattern::ReturnRef { param } => {
                    annotations.push(format!("// Returns reference to {}", param));
                }
                ReferencePattern::StructRef { field, source } => {
                    annotations.push(format!("// {} references {}", field, source));
                }
                ReferencePattern::CollectionRef { container, element } => {
                    annotations.push(format!("// {} contains references to {}", container, element));
                }
            }
        }

        annotations
    }
}

impl Default for LifetimeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Common lifetime patterns and their translations
pub struct LifetimePatterns;

impl LifetimePatterns {
    /// Pattern: Returning a reference to a parameter
    pub fn return_param_ref() -> String {
        r#"fn first<'a>(x: &'a str, y: &str) -> &'a str {
    x
}"#
        .to_string()
    }

    /// Pattern: Struct with reference field
    pub fn struct_with_ref() -> String {
        r#"struct Container<'a> {
    data: &'a str,
}"#
        .to_string()
    }

    /// Pattern: Multiple lifetimes
    pub fn multiple_lifetimes() -> String {
        r#"fn combine<'a, 'b>(x: &'a str, y: &'b str) -> String {
    format!("{}{}", x, y)
}"#
        .to_string()
    }

    /// Pattern: Lifetime bounds
    pub fn lifetime_bounds() -> String {
        r#"struct Wrapper<'a, T: 'a> {
    data: &'a T,
}"#
        .to_string()
    }

    /// Pattern: Static lifetime
    pub fn static_lifetime() -> String {
        r#"const MESSAGE: &'static str = "Hello, World!";

fn get_message() -> &'static str {
    MESSAGE
}"#
        .to_string()
    }

    /// Pattern: Lifetime elision (implicit)
    pub fn elision() -> String {
        r#"// With elision (no explicit lifetimes needed):
fn process(s: &str) -> &str {
    s.trim()
}

// Equivalent to:
fn process_explicit<'a>(s: &'a str) -> &'a str {
    s.trim()
}"#
        .to_string()
    }

    /// Pattern: Self lifetime
    pub fn self_lifetime() -> String {
        r#"impl<'a> Container<'a> {
    fn get_data(&self) -> &'a str {
        self.data
    }
}"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifetime_creation() {
        let lifetime = Lifetime::new("a");
        assert_eq!(lifetime.to_string(), "'a");
    }

    #[test]
    fn test_type_with_lifetime() {
        let ty = TypeWithLifetime::new("str")
            .as_reference()
            .with_lifetime(Lifetime::new("a"));

        assert_eq!(ty.to_rust_string(), "&'a str");
    }

    #[test]
    fn test_function_signature() {
        let mut sig = FunctionSignature::new("example");
        sig.add_param(
            "x".to_string(),
            TypeWithLifetime::new("str")
                .as_reference()
                .with_lifetime(Lifetime::new("a")),
        );
        sig.set_return_type(
            TypeWithLifetime::new("str")
                .as_reference()
                .with_lifetime(Lifetime::new("a")),
        );

        let rust = sig.to_rust_string();
        assert!(rust.contains("fn example"));
        assert!(rust.contains("'a"));
    }

    #[test]
    fn test_elision_single_param() {
        let mut analyzer = LifetimeAnalyzer::new();
        let sig = analyzer.analyze_function(
            &[("s".to_string(), "str".to_string(), true)],
            Some(("str", true)),
            Some("s"),
        );

        // Should use elision (no explicit lifetime in signature)
        let rust = sig.to_rust_string();
        assert!(rust.contains("&str"));
    }

    #[test]
    fn test_struct_with_references() {
        let mut analyzer = LifetimeAnalyzer::new();
        let struct_def = analyzer.analyze_struct(&[
            ("name".to_string(), "str".to_string(), true),
            ("age".to_string(), "i32".to_string(), false),
        ]);

        let rust = struct_def.to_rust_string();
        assert!(rust.contains("'a"));
        assert!(rust.contains("&'a str"));
    }
}
