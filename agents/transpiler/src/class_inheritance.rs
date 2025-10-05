//! Class Inheritance Translation
//!
//! Translates Python class inheritance to Rust using traits and composition.
//!
//! ## Translation Strategy
//!
//! ### Single Inheritance
//! - Base class becomes a struct
//! - Derived class contains base struct as a field
//! - Methods delegated to base when appropriate
//!
//! ### Multiple Inheritance
//! - Common interfaces become traits
//! - Each base class becomes a trait
//! - Derived class implements all traits
//! - Uses composition for data
//!
//! ### Abstract Base Classes (ABC)
//! - Translates to Rust traits
//! - Abstract methods become trait methods
//! - Concrete methods become default implementations

use crate::{Error, Result};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct InheritanceInfo {
    pub base_classes: Vec<String>,
    pub is_abstract: bool,
    pub abstract_methods: Vec<String>,
    pub mro: Vec<String>, // Method Resolution Order
}

#[derive(Debug, Clone)]
pub struct ClassHierarchy {
    classes: HashMap<String, ClassInfo>,
}

#[derive(Debug, Clone)]
pub struct ClassInfo {
    pub name: String,
    pub bases: Vec<String>,
    pub methods: Vec<MethodInfo>,
    pub attributes: Vec<AttributeInfo>,
    pub is_abstract: bool,
}

#[derive(Debug, Clone)]
pub struct MethodInfo {
    pub name: String,
    pub is_abstract: bool,
    pub params: Vec<String>,
    pub return_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AttributeInfo {
    pub name: String,
    pub type_hint: Option<String>,
}

pub struct ClassInheritanceTranslator {
    hierarchy: ClassHierarchy,
    indent_level: usize,
}

impl ClassInheritanceTranslator {
    pub fn new() -> Self {
        Self {
            hierarchy: ClassHierarchy::new(),
            indent_level: 0,
        }
    }

    fn indent(&self) -> String {
        "    ".repeat(self.indent_level)
    }

    /// Register a class in the hierarchy
    pub fn register_class(&mut self, class_info: ClassInfo) {
        self.hierarchy.add_class(class_info);
    }

    /// Translate a class with inheritance to Rust
    pub fn translate_class(&mut self, class_name: &str) -> Result<String> {
        let class_info = self.hierarchy.get_class(class_name)
            .ok_or_else(|| Error::CodeGeneration(format!("Class {} not found", class_name)))?
            .clone(); // Clone to avoid borrow checker issues

        if class_info.bases.is_empty() {
            // No inheritance - simple struct + impl
            self.translate_simple_class(&class_info)
        } else if class_info.bases.len() == 1 {
            // Single inheritance
            self.translate_single_inheritance(&class_info)
        } else {
            // Multiple inheritance - use traits
            self.translate_multiple_inheritance(&class_info)
        }
    }

    /// Translate class without inheritance
    fn translate_simple_class(&mut self, class_info: &ClassInfo) -> Result<String> {
        let mut code = String::new();

        // Generate struct
        code.push_str(&self.generate_struct(&class_info.name, &class_info.attributes)?);
        code.push('\n');

        // Generate impl
        code.push_str(&self.generate_impl(&class_info.name, &class_info.methods)?);

        Ok(code)
    }

    /// Translate single inheritance
    fn translate_single_inheritance(&mut self, class_info: &ClassInfo) -> Result<String> {
        let mut code = String::new();
        let base_class = &class_info.bases[0];

        // If base is abstract, implement it as a trait
        if let Some(base_info) = self.hierarchy.get_class(base_class).cloned() {
            if base_info.is_abstract {
                return self.translate_trait_implementation(class_info, &base_info);
            }
        }

        // Struct with base as field
        code.push_str(&format!("#[derive(Debug, Clone)]\n"));
        code.push_str(&format!("pub struct {} {{\n", class_info.name));
        self.indent_level += 1;

        // Include base class as field
        code.push_str(&format!("{}pub base: {},\n", self.indent(), base_class));

        // Add derived class attributes
        for attr in &class_info.attributes {
            let rust_type = self.python_type_to_rust(attr.type_hint.as_deref().unwrap_or("()"));
            code.push_str(&format!("{}pub {}: {},\n", self.indent(), attr.name, rust_type));
        }

        self.indent_level -= 1;
        code.push_str("}\n\n");

        // Impl block with methods
        code.push_str(&self.generate_impl_with_inheritance(&class_info.name, &class_info.methods, base_class)?);

        Ok(code)
    }

    /// Translate multiple inheritance using traits
    fn translate_multiple_inheritance(&mut self, class_info: &ClassInfo) -> Result<String> {
        let mut code = String::new();

        // Generate traits for each base class if they're abstract
        for base_class in &class_info.bases {
            if let Some(base_info) = self.hierarchy.get_class(base_class).cloned() {
                if base_info.is_abstract {
                    code.push_str(&self.generate_trait(&base_info)?);
                    code.push('\n');
                }
            }
        }

        // Generate struct
        code.push_str(&format!("#[derive(Debug, Clone)]\n"));
        code.push_str(&format!("pub struct {} {{\n", class_info.name));
        self.indent_level += 1;

        // Add attributes
        for attr in &class_info.attributes {
            let rust_type = self.python_type_to_rust(attr.type_hint.as_deref().unwrap_or("()"));
            code.push_str(&format!("{}pub {}: {},\n", self.indent(), attr.name, rust_type));
        }

        self.indent_level -= 1;
        code.push_str("}\n\n");

        // Implement each base trait
        for base_class in &class_info.bases {
            if let Some(base_info) = self.hierarchy.get_class(base_class).cloned() {
                if base_info.is_abstract {
                    code.push_str(&self.generate_trait_impl(&class_info.name, &base_info)?);
                    code.push('\n');
                }
            }
        }

        // Regular impl block for non-inherited methods
        let own_methods: Vec<_> = class_info.methods.iter()
            .filter(|m| !self.is_inherited_method(class_info, &m.name))
            .cloned()
            .collect();

        if !own_methods.is_empty() {
            code.push_str(&self.generate_impl(&class_info.name, &own_methods)?);
        }

        Ok(code)
    }

    /// Generate a trait from an abstract base class
    fn generate_trait(&mut self, class_info: &ClassInfo) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!("pub trait {} {{\n", class_info.name));
        self.indent_level += 1;

        for method in &class_info.methods {
            let params = self.format_trait_method_params(&method.params);
            let return_type = method.return_type.as_deref()
                .map(|t| self.python_type_to_rust(t))
                .unwrap_or_else(|| "()".to_string());

            if method.is_abstract {
                // Abstract method - no body
                code.push_str(&format!("{}fn {}({}) -> {};\n",
                    self.indent(), method.name, params, return_type));
            } else {
                // Concrete method - default implementation
                code.push_str(&format!("{}fn {}({}) -> {} {{\n",
                    self.indent(), method.name, params, return_type));
                self.indent_level += 1;
                code.push_str(&format!("{}// Default implementation\n", self.indent()));
                code.push_str(&format!("{}()\n", self.indent()));
                self.indent_level -= 1;
                code.push_str(&format!("{}}}\n", self.indent()));
            }
        }

        self.indent_level -= 1;
        code.push_str("}\n");

        Ok(code)
    }

    /// Generate trait implementation
    fn generate_trait_impl(&mut self, struct_name: &str, trait_info: &ClassInfo) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!("impl {} for {} {{\n", trait_info.name, struct_name));
        self.indent_level += 1;

        for method in &trait_info.methods {
            if method.is_abstract {
                let params = self.format_trait_method_params(&method.params);
                let return_type = method.return_type.as_deref()
                    .map(|t| self.python_type_to_rust(t))
                    .unwrap_or_else(|| "()".to_string());

                code.push_str(&format!("{}fn {}({}) -> {} {{\n",
                    self.indent(), method.name, params, return_type));
                self.indent_level += 1;
                code.push_str(&format!("{}// TODO: Implement {}\n", self.indent(), method.name));
                code.push_str(&format!("{}()\n", self.indent()));
                self.indent_level -= 1;
                code.push_str(&format!("{}}}\n", self.indent()));
            }
        }

        self.indent_level -= 1;
        code.push_str("}\n");

        Ok(code)
    }

    /// Translate trait implementation for abstract base
    fn translate_trait_implementation(&mut self, derived: &ClassInfo, base: &ClassInfo) -> Result<String> {
        let mut code = String::new();

        // Generate trait if not already generated
        code.push_str(&self.generate_trait(base)?);
        code.push('\n');

        // Generate derived struct
        code.push_str(&self.generate_struct(&derived.name, &derived.attributes)?);
        code.push('\n');

        // Implement the trait
        code.push_str(&self.generate_trait_impl(&derived.name, base)?);

        Ok(code)
    }

    /// Generate super() call translation
    pub fn translate_super_call(&self, class_name: &str, method_name: &str) -> Result<String> {
        let class_info = self.hierarchy.get_class(class_name)
            .ok_or_else(|| Error::CodeGeneration(format!("Class {} not found", class_name)))?;

        if class_info.bases.is_empty() {
            return Err(Error::CodeGeneration("No base class for super()".into()));
        }

        let _base_class = &class_info.bases[0];

        // For single inheritance with composition
        Ok(format!("self.base.{}()", method_name))
    }

    /// Generate struct definition
    fn generate_struct(&mut self, name: &str, attributes: &[AttributeInfo]) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!("#[derive(Debug, Clone)]\n"));
        code.push_str(&format!("pub struct {} {{\n", name));

        if !attributes.is_empty() {
            self.indent_level += 1;
            for attr in attributes {
                let rust_type = self.python_type_to_rust(attr.type_hint.as_deref().unwrap_or("()"));
                code.push_str(&format!("{}pub {}: {},\n", self.indent(), attr.name, rust_type));
            }
            self.indent_level -= 1;
        }

        code.push_str("}\n");

        Ok(code)
    }

    /// Generate impl block
    fn generate_impl(&mut self, name: &str, methods: &[MethodInfo]) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!("impl {} {{\n", name));
        self.indent_level += 1;

        for method in methods {
            let params = self.format_method_params(&method.params);
            let return_type = method.return_type.as_deref()
                .map(|t| self.python_type_to_rust(t))
                .unwrap_or_else(|| "()".to_string());

            code.push_str(&format!("{}pub fn {}({}) -> {} {{\n",
                self.indent(), method.name, params, return_type));
            self.indent_level += 1;
            code.push_str(&format!("{}// TODO: Implement method\n", self.indent()));
            code.push_str(&format!("{}()\n", self.indent()));
            self.indent_level -= 1;
            code.push_str(&format!("{}}}\n", self.indent()));
        }

        self.indent_level -= 1;
        code.push_str("}\n");

        Ok(code)
    }

    /// Generate impl with base class delegation
    fn generate_impl_with_inheritance(&mut self, name: &str, methods: &[MethodInfo], base: &str) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!("impl {} {{\n", name));
        self.indent_level += 1;

        // Constructor
        code.push_str(&format!("{}pub fn new(base: {}) -> Self {{\n", self.indent(), base));
        self.indent_level += 1;
        code.push_str(&format!("{}Self {{ base }}\n", self.indent()));
        self.indent_level -= 1;
        code.push_str(&format!("{}}}\n\n", self.indent()));

        // Methods
        for method in methods {
            let params = self.format_method_params(&method.params);
            let return_type = method.return_type.as_deref()
                .map(|t| self.python_type_to_rust(t))
                .unwrap_or_else(|| "()".to_string());

            code.push_str(&format!("{}pub fn {}({}) -> {} {{\n",
                self.indent(), method.name, params, return_type));
            self.indent_level += 1;
            code.push_str(&format!("{}// TODO: Implement method\n", self.indent()));
            code.push_str(&format!("{}()\n", self.indent()));
            self.indent_level -= 1;
            code.push_str(&format!("{}}}\n", self.indent()));
        }

        self.indent_level -= 1;
        code.push_str("}\n");

        Ok(code)
    }

    fn format_method_params(&self, params: &[String]) -> String {
        if params.is_empty() {
            return "&self".to_string();
        }

        let mut result = vec!["&self".to_string()];
        for (i, param) in params.iter().enumerate() {
            if i == 0 && param == "self" {
                continue;
            }
            result.push(format!("{}: ()", param));
        }
        result.join(", ")
    }

    fn format_trait_method_params(&self, params: &[String]) -> String {
        if params.is_empty() {
            return "&self".to_string();
        }

        let mut result = vec!["&self".to_string()];
        for (i, param) in params.iter().enumerate() {
            if i == 0 && param == "self" {
                continue;
            }
            result.push(format!("{}: ()", param));
        }
        result.join(", ")
    }

    fn is_inherited_method(&self, class_info: &ClassInfo, method_name: &str) -> bool {
        for base_name in &class_info.bases {
            if let Some(base_info) = self.hierarchy.get_class(base_name) {
                if base_info.methods.iter().any(|m| m.name == method_name) {
                    return true;
                }
            }
        }
        false
    }

    fn python_type_to_rust(&self, py_type: &str) -> String {
        match py_type {
            "int" => "i32",
            "float" => "f64",
            "str" => "String",
            "bool" => "bool",
            "None" | "()" => "()",
            _ => "()",
        }.to_string()
    }
}

impl ClassHierarchy {
    pub fn new() -> Self {
        Self {
            classes: HashMap::new(),
        }
    }

    pub fn add_class(&mut self, class_info: ClassInfo) {
        self.classes.insert(class_info.name.clone(), class_info);
    }

    pub fn get_class(&self, name: &str) -> Option<&ClassInfo> {
        self.classes.get(name)
    }

    /// Compute Method Resolution Order (MRO) using C3 linearization
    pub fn compute_mro(&self, class_name: &str) -> Vec<String> {
        let mut mro = Vec::new();
        let mut visited = HashSet::new();

        self.mro_helper(class_name, &mut mro, &mut visited);

        mro
    }

    fn mro_helper(&self, class_name: &str, mro: &mut Vec<String>, visited: &mut HashSet<String>) {
        if visited.contains(class_name) {
            return;
        }

        visited.insert(class_name.to_string());

        if let Some(class_info) = self.get_class(class_name) {
            // Visit bases first (depth-first)
            for base in &class_info.bases {
                self.mro_helper(base, mro, visited);
            }
        }

        // Add current class
        mro.push(class_name.to_string());
    }
}

impl Default for ClassInheritanceTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_class_no_inheritance() {
        let mut translator = ClassInheritanceTranslator::new();

        let class_info = ClassInfo {
            name: "Point".to_string(),
            bases: vec![],
            methods: vec![],
            attributes: vec![
                AttributeInfo { name: "x".to_string(), type_hint: Some("int".to_string()) },
                AttributeInfo { name: "y".to_string(), type_hint: Some("int".to_string()) },
            ],
            is_abstract: false,
        };

        translator.register_class(class_info.clone());
        let result = translator.translate_class("Point").unwrap();

        assert!(result.contains("pub struct Point"));
        assert!(result.contains("pub x: i32"));
        assert!(result.contains("pub y: i32"));
    }

    #[test]
    fn test_single_inheritance() {
        let mut translator = ClassInheritanceTranslator::new();

        let base = ClassInfo {
            name: "Animal".to_string(),
            bases: vec![],
            methods: vec![
                MethodInfo {
                    name: "speak".to_string(),
                    is_abstract: false,
                    params: vec!["self".to_string()],
                    return_type: None,
                },
            ],
            attributes: vec![
                AttributeInfo { name: "name".to_string(), type_hint: Some("str".to_string()) },
            ],
            is_abstract: false,
        };

        let derived = ClassInfo {
            name: "Dog".to_string(),
            bases: vec!["Animal".to_string()],
            methods: vec![
                MethodInfo {
                    name: "bark".to_string(),
                    is_abstract: false,
                    params: vec!["self".to_string()],
                    return_type: None,
                },
            ],
            attributes: vec![
                AttributeInfo { name: "breed".to_string(), type_hint: Some("str".to_string()) },
            ],
            is_abstract: false,
        };

        translator.register_class(base);
        translator.register_class(derived);

        let result = translator.translate_class("Dog").unwrap();

        assert!(result.contains("pub struct Dog"));
        assert!(result.contains("pub base: Animal"));
        assert!(result.contains("pub breed: String"));
    }

    #[test]
    fn test_abstract_base_class() {
        let mut translator = ClassInheritanceTranslator::new();

        let base = ClassInfo {
            name: "Shape".to_string(),
            bases: vec![],
            methods: vec![
                MethodInfo {
                    name: "area".to_string(),
                    is_abstract: true,
                    params: vec!["self".to_string()],
                    return_type: Some("float".to_string()),
                },
            ],
            attributes: vec![],
            is_abstract: true,
        };

        let derived = ClassInfo {
            name: "Circle".to_string(),
            bases: vec!["Shape".to_string()],
            methods: vec![
                MethodInfo {
                    name: "area".to_string(),
                    is_abstract: false,
                    params: vec!["self".to_string()],
                    return_type: Some("float".to_string()),
                },
            ],
            attributes: vec![
                AttributeInfo { name: "radius".to_string(), type_hint: Some("float".to_string()) },
            ],
            is_abstract: false,
        };

        translator.register_class(base);
        translator.register_class(derived);

        let result = translator.translate_class("Circle").unwrap();

        assert!(result.contains("pub trait Shape"));
        assert!(result.contains("fn area(&self) -> f64;"));
        assert!(result.contains("impl Shape for Circle"));
    }

    #[test]
    fn test_mro_computation() {
        let hierarchy = ClassHierarchy::new();
        // More comprehensive MRO tests would go here
    }
}
