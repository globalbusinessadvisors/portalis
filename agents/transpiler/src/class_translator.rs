//! Class Translation Module
//!
//! Translates Python classes to idiomatic Rust structs with impl blocks.

use crate::{Error, Result};
use serde_json::Value;

pub struct ClassTranslator {
    indent_level: usize,
}

impl ClassTranslator {
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    fn indent(&self) -> String {
        "    ".repeat(self.indent_level)
    }

    /// Generate complete Rust struct + impl from Python class
    pub fn generate_class(&mut self, class: &Value) -> Result<String> {
        let name = class.get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| Error::CodeGeneration("Missing class name".into()))?;

        let attributes: Vec<Value> = class.get("attributes")
            .and_then(|a| a.as_array())
            .cloned()
            .unwrap_or_default();

        let methods: Vec<Value> = class.get("methods")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();

        let mut code = String::new();

        // Generate struct definition
        code.push_str(&self.generate_struct(name, &attributes)?);
        code.push('\n');

        // Generate impl block
        code.push_str(&self.generate_impl(name, &methods)?);

        Ok(code)
    }

    /// Generate Rust struct definition
    fn generate_struct(&mut self, name: &str, attributes: &[Value]) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!("pub struct {} {{", name));

        if attributes.is_empty() {
            code.push_str("}\n");
            return Ok(code);
        }

        code.push('\n');
        self.indent_level += 1;

        for attr in attributes {
            let attr_name = attr.get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| Error::CodeGeneration("Missing attribute name".into()))?;

            let rust_type = self.infer_attribute_type(attr);

            code.push_str(&format!("{}pub {}: {},\n", self.indent(), attr_name, rust_type));
        }

        self.indent_level -= 1;
        code.push_str("}\n");

        Ok(code)
    }

    /// Generate impl block with methods
    fn generate_impl(&mut self, name: &str, methods: &[Value]) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!("impl {} {{\n", name));
        self.indent_level += 1;

        for (i, method) in methods.iter().enumerate() {
            let method_code = self.generate_method(method)?;
            code.push_str(&method_code);

            if i < methods.len() - 1 {
                code.push('\n');
            }
        }

        self.indent_level -= 1;
        code.push_str("}\n");

        Ok(code)
    }

    /// Generate a single method
    fn generate_method(&mut self, method: &Value) -> Result<String> {
        let name = method.get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| Error::CodeGeneration("Missing method name".into()))?;

        // Handle __init__ -> new() conversion
        if name == "__init__" {
            return self.generate_constructor(method);
        }

        let params: Vec<Value> = method.get("params")
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();

        let return_type = self.extract_return_type(method);

        // Build parameter list (skip 'self')
        let mut param_strs = Vec::new();
        let has_self = params.first()
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .map(|n| n == "self")
            .unwrap_or(false);

        if has_self {
            param_strs.push("&self".to_string());
        }

        for param in params.iter().skip(if has_self { 1 } else { 0 }) {
            let param_name = param.get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("arg");
            let param_type = self.extract_param_type(param);
            param_strs.push(format!("{}: {}", param_name, param_type));
        }

        let params_str = param_strs.join(", ");

        // Generate method signature
        let mut code = String::new();
        code.push_str(&format!("{}pub fn {}({}) -> {} {{\n",
            self.indent(), name, params_str, return_type));

        self.indent_level += 1;

        // Generate method body (pattern-based for now)
        let body = self.generate_method_body(method)?;
        code.push_str(&body);

        self.indent_level -= 1;
        code.push_str(&format!("{}}}\n", self.indent()));

        Ok(code)
    }

    /// Generate constructor (from __init__)
    fn generate_constructor(&mut self, method: &Value) -> Result<String> {
        let params: Vec<Value> = method.get("params")
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();

        // Skip 'self' parameter
        let init_params: Vec<_> = params.iter()
            .skip(1)
            .collect();

        let mut param_strs = Vec::new();
        for param in &init_params {
            let param_name = param.get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("arg");
            let param_type = self.extract_param_type(param);
            param_strs.push(format!("{}: {}", param_name, param_type));
        }

        let params_str = param_strs.join(", ");

        let mut code = String::new();
        code.push_str(&format!("{}pub fn new({}) -> Self {{\n",
            self.indent(), params_str));

        self.indent_level += 1;

        // Generate struct initialization
        code.push_str(&format!("{}Self {{\n", self.indent()));
        self.indent_level += 1;

        // Map parameters to fields (simple case: same names)
        for param in &init_params {
            if let Some(param_name) = param.get("name").and_then(|n| n.as_str()) {
                code.push_str(&format!("{}{},\n", self.indent(), param_name));
            }
        }

        self.indent_level -= 1;
        code.push_str(&format!("{}}}\n", self.indent()));

        self.indent_level -= 1;
        code.push_str(&format!("{}}}\n", self.indent()));

        Ok(code)
    }

    /// Generate method body (simplified for now)
    fn generate_method_body(&self, _method: &Value) -> Result<String> {
        // For now, return a default implementation
        // Later we'll add pattern matching for specific methods
        Ok(format!("{}// TODO: Implement method body\n{}()\n",
            self.indent(), self.indent()))
    }

    /// Infer Rust type for attribute
    fn infer_attribute_type(&self, attr: &Value) -> String {
        if let Some(type_hint) = attr.get("type_hint").and_then(|t| t.as_str()) {
            self.python_type_to_rust(type_hint)
        } else {
            "()".to_string() // Default type
        }
    }

    /// Extract return type from method
    fn extract_return_type(&self, method: &Value) -> String {
        method.get("return_type")
            .and_then(|rt| rt.as_str())
            .map(|t| self.python_type_to_rust(t))
            .unwrap_or_else(|| "()".to_string())
    }

    /// Extract parameter type
    fn extract_param_type(&self, param: &Value) -> String {
        if let Some(type_hint) = param.get("type_hint").and_then(|t| t.as_str()) {
            self.python_type_to_rust(type_hint)
        } else {
            "()".to_string()
        }
    }

    /// Convert Python type to Rust type
    fn python_type_to_rust(&self, py_type: &str) -> String {
        match py_type {
            "int" => "i32",
            "float" => "f64",
            "str" => "String",
            "bool" => "bool",
            "None" => "()",
            _ => "()",
        }.to_string()
    }
}

impl Default for ClassTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_generate_simple_struct() {
        let mut translator = ClassTranslator::new();

        let class = json!({
            "name": "Point",
            "attributes": [
                {"name": "x", "type_hint": "int"},
                {"name": "y", "type_hint": "int"}
            ],
            "methods": []
        });

        let result = translator.generate_class(&class).unwrap();
        assert!(result.contains("pub struct Point"));
        assert!(result.contains("pub x: i32"));
        assert!(result.contains("pub y: i32"));
    }

    #[test]
    fn test_generate_constructor() {
        let mut translator = ClassTranslator::new();

        let class = json!({
            "name": "Counter",
            "attributes": [
                {"name": "count", "type_hint": "int"}
            ],
            "methods": [
                {
                    "name": "__init__",
                    "params": [
                        {"name": "self"},
                        {"name": "initial", "type_hint": "int"}
                    ],
                    "return_type": null
                }
            ]
        });

        let result = translator.generate_class(&class).unwrap();
        assert!(result.contains("pub fn new(initial: i32) -> Self"));
        assert!(result.contains("Self {"));
        assert!(result.contains("initial,"));
    }

    #[test]
    fn test_generate_instance_method() {
        let mut translator = ClassTranslator::new();

        let class = json!({
            "name": "Calculator",
            "attributes": [],
            "methods": [
                {
                    "name": "add",
                    "params": [
                        {"name": "self"},
                        {"name": "a", "type_hint": "int"},
                        {"name": "b", "type_hint": "int"}
                    ],
                    "return_type": "int"
                }
            ]
        });

        let result = translator.generate_class(&class).unwrap();
        assert!(result.contains("pub fn add(&self, a: i32, b: i32) -> i32"));
        assert!(result.contains("impl Calculator"));
    }
}
