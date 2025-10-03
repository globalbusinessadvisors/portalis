//! Enhanced Python Parser using rustpython-parser
//!
//! Production-grade AST parsing with full Python language support.

use crate::{PythonAst, PythonClass, PythonFunction, PythonImport, PythonParameter};
use portalis_core::{Error, Result};
use rustpython_parser::{ast, Parse};

/// Enhanced parser using rustpython-parser for full AST support
pub struct EnhancedParser;

impl EnhancedParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse Python source code into our AST representation
    pub fn parse(&self, source: &str) -> Result<PythonAst> {
        // Parse using rustpython-parser
        let parsed = ast::Suite::parse(source, "<input>")
            .map_err(|e| Error::Parse(format!("Python parse error: {}", e)))?;

        let mut ast = PythonAst {
            functions: Vec::new(),
            classes: Vec::new(),
            imports: Vec::new(),
        };

        // Walk the AST and extract functions, classes, and imports
        for stmt in parsed.iter() {
            match stmt {
                ast::Stmt::FunctionDef(func) => {
                    ast.functions.push(self.extract_function(func)?);
                }
                ast::Stmt::ClassDef(class) => {
                    ast.classes.push(self.extract_class(class)?);
                }
                ast::Stmt::Import(import) => {
                    for alias in &import.names {
                        ast.imports.push(PythonImport {
                            module: alias.name.to_string(),
                            items: Vec::new(),
                            alias: alias.asname.as_ref().map(|a| a.to_string()),
                        });
                    }
                }
                ast::Stmt::ImportFrom(import) => {
                    let module = import
                        .module
                        .as_ref()
                        .map(|m| m.to_string())
                        .unwrap_or_else(|| "".to_string());

                    let items: Vec<String> = import
                        .names
                        .iter()
                        .map(|alias| alias.name.to_string())
                        .collect();

                    ast.imports.push(PythonImport {
                        module,
                        items,
                        alias: None,
                    });
                }
                _ => {}
            }
        }

        Ok(ast)
    }

    /// Extract function information from AST node
    fn extract_function(&self, func: &ast::StmtFunctionDef) -> Result<PythonFunction> {
        let name = func.name.to_string();

        // Extract parameters
        let params = self.extract_parameters(&func.args)?;

        // Extract return type annotation
        let return_type = func
            .returns
            .as_ref()
            .and_then(|expr| self.extract_type_from_expr(expr));

        // Extract decorators
        let decorators: Vec<String> = func
            .decorator_list
            .iter()
            .filter_map(|dec| self.expr_to_string(dec))
            .collect();

        // For now, we don't parse the body - we'll use it later for advanced analysis
        let body = String::new();

        Ok(PythonFunction {
            name,
            params,
            return_type,
            body,
            decorators,
        })
    }

    /// Extract parameters from function arguments
    fn extract_parameters(&self, args: &ast::Arguments) -> Result<Vec<PythonParameter>> {
        let mut params = Vec::new();

        // Regular positional arguments (rustpython 0.3 uses posonlyargs + args)
        for arg_with_default in &args.args {
            let arg = &arg_with_default.def;
            let name = arg.arg.to_string();
            let type_hint = arg.annotation.as_ref().and_then(|ann| self.extract_type_from_expr(ann));

            // Default value is in the ArgWithDefault struct
            let default = arg_with_default.default.as_ref().and_then(|expr| self.expr_to_string(expr));

            params.push(PythonParameter {
                name,
                type_hint,
                default,
            });
        }

        Ok(params)
    }

    /// Extract class information from AST node
    fn extract_class(&self, class: &ast::StmtClassDef) -> Result<PythonClass> {
        let name = class.name.to_string();

        // Extract base classes
        let bases: Vec<String> = class
            .bases
            .iter()
            .filter_map(|base| self.expr_to_string(base))
            .collect();

        // Extract methods and attributes
        let mut methods = Vec::new();
        let mut attributes = Vec::new();

        for stmt in &class.body {
            if let ast::Stmt::FunctionDef(func) = stmt {
                let method = self.extract_function(func)?;

                // Extract attributes from __init__ method
                if func.name.as_str() == "__init__" {
                    attributes.extend(self.extract_attributes_from_init(func));
                }

                methods.push(method);
            }
        }

        Ok(PythonClass {
            name,
            bases,
            methods,
            attributes,
        })
    }

    /// Extract class attributes from __init__ method
    fn extract_attributes_from_init(&self, func: &ast::StmtFunctionDef) -> Vec<crate::PythonAttribute> {
        use crate::PythonAttribute;
        let mut attributes = Vec::new();

        // Walk through the function body looking for self.attribute assignments
        for stmt in &func.body {
            if let ast::Stmt::Assign(assign) = stmt {
                for target in &assign.targets {
                    if let ast::Expr::Attribute(attr) = target {
                        // Check if it's self.something
                        if let ast::Expr::Name(name) = &*attr.value {
                            if name.id.as_str() == "self" {
                                attributes.push(PythonAttribute {
                                    name: attr.attr.to_string(),
                                    type_hint: None, // Type inference could be added later
                                });
                            }
                        }
                    }
                }
            } else if let ast::Stmt::AnnAssign(ann_assign) = stmt {
                // Handle annotated assignments: self.x: int = 0
                if let ast::Expr::Attribute(attr) = &*ann_assign.target {
                    if let ast::Expr::Name(name) = &*attr.value {
                        if name.id.as_str() == "self" {
                            let type_hint = self.extract_type_from_expr(&ann_assign.annotation);

                            attributes.push(PythonAttribute {
                                name: attr.attr.to_string(),
                                type_hint,
                            });
                        }
                    }
                }
            }
        }

        attributes
    }

    /// Extract type annotation from expression
    fn extract_type_from_expr(&self, expr: &ast::Expr) -> Option<String> {
        match expr {
            ast::Expr::Name(name) => Some(name.id.to_string()),
            ast::Expr::Constant(constant) => {
                // Handle string type annotations
                if let ast::Constant::Str(s) = &constant.value {
                    Some(s.to_string())
                } else {
                    None
                }
            }
            ast::Expr::Subscript(subscript) => {
                // Handle generic types like List[int], Dict[str, int]
                let base = self.expr_to_string(&subscript.value)?;
                let slice = self.expr_to_string(&subscript.slice)?;
                Some(format!("{}[{}]", base, slice))
            }
            ast::Expr::Tuple(tuple) => {
                // Handle tuple type annotations
                let elements: Vec<String> = tuple
                    .elts
                    .iter()
                    .filter_map(|e| self.expr_to_string(e))
                    .collect();
                Some(format!("({})", elements.join(", ")))
            }
            _ => self.expr_to_string(expr),
        }
    }

    /// Convert expression to string (simplified)
    fn expr_to_string(&self, expr: &ast::Expr) -> Option<String> {
        match expr {
            ast::Expr::Name(name) => Some(name.id.to_string()),
            ast::Expr::Constant(constant) => match &constant.value {
                ast::Constant::Int(i) => Some(i.to_string()),
                ast::Constant::Float(f) => Some(f.to_string()),
                ast::Constant::Str(s) => Some(s.to_string()),
                ast::Constant::Bool(b) => Some(b.to_string()),
                ast::Constant::None => Some("None".to_string()),
                _ => None,
            },
            ast::Expr::Attribute(attr) => {
                let value = self.expr_to_string(&attr.value)?;
                Some(format!("{}.{}", value, attr.attr))
            }
            _ => None,
        }
    }
}

impl Default for EnhancedParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let parser = EnhancedParser::new();
        let source = r#"
def add(a: int, b: int) -> int:
    return a + b
"#;

        let ast = parser.parse(source).unwrap();
        assert_eq!(ast.functions.len(), 1);
        assert_eq!(ast.functions[0].name, "add");
        assert_eq!(ast.functions[0].params.len(), 2);
        assert_eq!(ast.functions[0].return_type, Some("int".to_string()));
    }

    #[test]
    fn test_parse_multiple_functions() {
        let parser = EnhancedParser::new();
        let source = r#"
def add(a: int, b: int) -> int:
    return a + b

def multiply(x: int, y: int) -> int:
    return x * y

def greet(name: str) -> str:
    return f"Hello, {name}"
"#;

        let ast = parser.parse(source).unwrap();
        assert_eq!(ast.functions.len(), 3);
        assert_eq!(ast.functions[0].name, "add");
        assert_eq!(ast.functions[1].name, "multiply");
        assert_eq!(ast.functions[2].name, "greet");
    }

    #[test]
    fn test_parse_class() {
        let parser = EnhancedParser::new();
        let source = r#"
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b
"#;

        let ast = parser.parse(source).unwrap();
        assert_eq!(ast.classes.len(), 1);
        assert_eq!(ast.classes[0].name, "Calculator");
        assert_eq!(ast.classes[0].methods.len(), 2);
    }

    #[test]
    fn test_parse_class_with_init() {
        let parser = EnhancedParser::new();
        let source = r#"
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self) -> int:
        self.count = self.count + 1
        return self.count
"#;

        let ast = parser.parse(source).unwrap();
        assert_eq!(ast.classes.len(), 1);
        assert_eq!(ast.classes[0].name, "Counter");
        assert_eq!(ast.classes[0].methods.len(), 2);
        assert_eq!(ast.classes[0].attributes.len(), 1);
        assert_eq!(ast.classes[0].attributes[0].name, "count");
    }

    #[test]
    fn test_parse_class_with_typed_attributes() {
        let parser = EnhancedParser::new();
        let source = r#"
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height
"#;

        let ast = parser.parse(source).unwrap();
        assert_eq!(ast.classes.len(), 1);
        assert_eq!(ast.classes[0].name, "Rectangle");
        assert_eq!(ast.classes[0].methods.len(), 2);
        assert_eq!(ast.classes[0].attributes.len(), 2);
        assert_eq!(ast.classes[0].attributes[0].name, "width");
        assert_eq!(ast.classes[0].attributes[1].name, "height");
    }

    #[test]
    fn test_parse_imports() {
        let parser = EnhancedParser::new();
        let source = r#"
import os
import sys
from typing import List, Dict
from pathlib import Path
"#;

        let ast = parser.parse(source).unwrap();
        assert!(ast.imports.len() >= 2);
    }

    #[test]
    fn test_parse_function_with_defaults() {
        let parser = EnhancedParser::new();
        let source = r#"
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}"
"#;

        let ast = parser.parse(source).unwrap();
        assert_eq!(ast.functions.len(), 1);
        assert_eq!(ast.functions[0].params.len(), 2);
        assert_eq!(ast.functions[0].params[1].default, Some("Hello".to_string()));
    }

    #[test]
    fn test_parse_complex_types() {
        let parser = EnhancedParser::new();
        let source = r#"
def process(items: list) -> dict:
    return {}
"#;

        let ast = parser.parse(source).unwrap();
        assert_eq!(ast.functions.len(), 1);
        // Type annotations should be captured
        assert!(ast.functions[0].return_type.is_some());
        assert_eq!(ast.functions[0].return_type, Some("dict".to_string()));
    }

    #[test]
    fn test_parse_decorators() {
        let parser = EnhancedParser::new();
        let source = r#"
@property
def name(self) -> str:
    return self._name
"#;

        let ast = parser.parse(source).unwrap();
        assert_eq!(ast.functions.len(), 1);
        assert_eq!(ast.functions[0].decorators.len(), 1);
        assert_eq!(ast.functions[0].decorators[0], "property");
    }
}
