//! Python to Rust translator
//!
//! Translates Python AST to Rust code, implementing the 527 Python language features.
//! Starting with Low complexity features for Phase 1.

use crate::python_ast::*;
use crate::stdlib_mapper::StdlibMapper;
use crate::{Error, Result};
use std::collections::HashMap;

/// Type inference engine for Python → Rust translation
#[derive(Debug, Clone)]
pub struct TypeInference {
    /// Variable name → inferred Rust type
    type_map: HashMap<String, RustType>,
}

/// Rust type representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RustType {
    I32,
    I64,
    F64,
    Bool,
    String,
    Unit,
    Vec(Box<RustType>),
    Option(Box<RustType>),
    Unknown,
}

impl RustType {
    /// Convert to Rust type string
    pub fn to_rust_str(&self) -> String {
        match self {
            RustType::I32 => "i32".to_string(),
            RustType::I64 => "i64".to_string(),
            RustType::F64 => "f64".to_string(),
            RustType::Bool => "bool".to_string(),
            RustType::String => "String".to_string(),
            RustType::Unit => "()".to_string(),
            RustType::Vec(inner) => format!("Vec<{}>", inner.to_rust_str()),
            RustType::Option(inner) => format!("Option<{}>", inner.to_rust_str()),
            RustType::Unknown => "()".to_string(),
        }
    }
}

impl TypeInference {
    pub fn new() -> Self {
        Self {
            type_map: HashMap::new(),
        }
    }

    /// Infer type from Python literal
    pub fn infer_from_literal(&self, lit: &PyLiteral) -> RustType {
        match lit {
            PyLiteral::Int(n) => {
                // Use i32 for small integers, i64 for large
                if *n >= i32::MIN as i64 && *n <= i32::MAX as i64 {
                    RustType::I32
                } else {
                    RustType::I64
                }
            }
            PyLiteral::Float(_) => RustType::F64,
            PyLiteral::String(_) => RustType::String,
            PyLiteral::Bool(_) => RustType::Bool,
            PyLiteral::None => RustType::Option(Box::new(RustType::Unknown)),
            PyLiteral::Bytes(_) => RustType::Vec(Box::new(RustType::I32)), // Vec<u8> simplified to Vec<i32>
        }
    }

    /// Infer type from Python expression
    pub fn infer_expr(&mut self, expr: &PyExpr) -> RustType {
        match expr {
            PyExpr::Literal(lit) => self.infer_from_literal(lit),
            PyExpr::Name(name) => self
                .type_map
                .get(name)
                .cloned()
                .unwrap_or(RustType::Unknown),
            PyExpr::BinOp { left, op, right } => {
                let left_type = self.infer_expr(left);
                let right_type = self.infer_expr(right);

                // Simplistic type inference
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mult | BinOp::Mod => {
                        // Numeric operations
                        if left_type == RustType::F64 || right_type == RustType::F64 {
                            RustType::F64
                        } else if left_type == RustType::I64 || right_type == RustType::I64 {
                            RustType::I64
                        } else {
                            RustType::I32
                        }
                    }
                    BinOp::Div => RustType::F64, // Division always returns float in Python 3
                    BinOp::FloorDiv => RustType::I32,
                    _ => RustType::Unknown,
                }
            }
            PyExpr::UnaryOp { op, operand } => {
                let operand_type = self.infer_expr(operand);
                match op {
                    UnaryOp::Not => RustType::Bool,
                    UnaryOp::USub | UnaryOp::UAdd => operand_type,
                    _ => RustType::Unknown,
                }
            }
            PyExpr::List(elements) => {
                if elements.is_empty() {
                    RustType::Vec(Box::new(RustType::Unknown))
                } else {
                    let first_type = self.infer_expr(&elements[0]);
                    RustType::Vec(Box::new(first_type))
                }
            }
            PyExpr::Tuple(elements) => {
                if elements.is_empty() {
                    RustType::Unknown
                } else {
                    // For simplicity, tuples become unit type in many contexts
                    // but when unpacked, elements get individual types
                    RustType::Unknown
                }
            }
            PyExpr::Compare { .. } => {
                // Comparison operators always return bool
                RustType::Bool
            }
            PyExpr::Call { func, args, .. } => {
                // Check if it's a method call (func is an Attribute)
                if let PyExpr::Attribute { value: _, attr } = func.as_ref() {
                    // Method call - infer based on method name
                    match attr.as_str() {
                        "split" | "splitlines" => RustType::Vec(Box::new(RustType::String)),
                        "strip" | "lstrip" | "rstrip" | "lower" | "upper" | "replace" => RustType::String,
                        "append" | "extend" | "remove" | "clear" | "sort" | "reverse" => {
                            RustType::Unknown // Mutating methods return ()
                        }
                        "pop" => {
                            // Return element type of the collection
                            RustType::Unknown // Would need collection tracking
                        }
                        "join" => RustType::String,
                        "format" => RustType::String,
                        "items" | "keys" | "values" => RustType::Unknown, // Dict methods
                        "count" | "index" | "find" => RustType::I32,
                        _ => RustType::Unknown,
                    }
                } else if let PyExpr::Name(name) = func.as_ref() {
                    // Function call - infer based on function name
                    match name.as_str() {
                        "len" => RustType::I32,
                        "sum" => RustType::I32, // Could be smarter based on input
                        "min" | "max" => {
                            // Return element type of the collection
                            if !args.is_empty() {
                                let arg_type = self.infer_expr(&args[0]);
                                if let RustType::Vec(inner) = arg_type {
                                    *inner
                                } else {
                                    arg_type // For non-Vec args like max(a, b, c)
                                }
                            } else {
                                RustType::I32
                            }
                        }
                        "abs" => {
                            if !args.is_empty() {
                                self.infer_expr(&args[0])
                            } else {
                                RustType::I32
                            }
                        }
                        "int" => RustType::I32,
                        "float" => RustType::F64,
                        "str" => RustType::String,
                        "bool" => RustType::Bool,
                        "range" => RustType::Unknown, // range is an iterator
                        "enumerate" | "zip" => RustType::Unknown, // iterators
                        "sorted" | "reversed" => {
                            // Return Vec of same type
                            if !args.is_empty() {
                                self.infer_expr(&args[0])
                            } else {
                                RustType::Vec(Box::new(RustType::Unknown))
                            }
                        }
                        "list" => {
                            // list(iterable) returns Vec
                            if !args.is_empty() {
                                if let RustType::Vec(inner) = self.infer_expr(&args[0]) {
                                    RustType::Vec(inner)
                                } else {
                                    RustType::Vec(Box::new(RustType::Unknown))
                                }
                            } else {
                                RustType::Vec(Box::new(RustType::Unknown))
                            }
                        }
                        "any" | "all" => RustType::Bool,
                        _ => RustType::Unknown,
                    }
                } else {
                    RustType::Unknown
                }
            }
            PyExpr::Attribute { value, attr } => {
                // Infer type based on method call
                match attr.as_str() {
                    "len" => RustType::I32,
                    "abs" => self.infer_expr(value),
                    "split" => RustType::Vec(Box::new(RustType::String)),
                    "strip" | "lower" | "upper" | "replace" => RustType::String,
                    "append" | "extend" | "remove" | "pop" | "clear" | "sort" | "reverse" => {
                        RustType::Unknown // Mutating methods return ()
                    }
                    _ => RustType::Unknown,
                }
            }
            _ => RustType::Unknown,
        }
    }

    /// Record type for a variable
    pub fn record_type(&mut self, name: String, rust_type: RustType) {
        self.type_map.insert(name, rust_type);
    }
}

impl Default for TypeInference {
    fn default() -> Self {
        Self::new()
    }
}

/// Python to Rust code generator
pub struct PythonToRustTranslator {
    type_inference: TypeInference,
    indent_level: usize,
    stdlib_mapper: StdlibMapper,
    imported_modules: Vec<String>,
    /// Maps aliases to actual module names (alias -> module)
    module_aliases: HashMap<String, String>,
}

impl PythonToRustTranslator {
    pub fn new() -> Self {
        Self {
            type_inference: TypeInference::new(),
            indent_level: 0,
            stdlib_mapper: StdlibMapper::new(),
            imported_modules: Vec::new(),
            module_aliases: HashMap::new(),
        }
    }

    /// Set imported modules for attribute resolution
    pub fn set_imports(&mut self, imports: Vec<String>) {
        self.imported_modules = imports;
    }

    /// Set module aliases (alias -> actual module name)
    pub fn set_aliases(&mut self, aliases: HashMap<String, String>) {
        self.module_aliases = aliases;
    }

    /// Resolve module name from alias or return original
    fn resolve_module_name(&self, name: &str) -> String {
        self.module_aliases.get(name).cloned().unwrap_or_else(|| name.to_string())
    }

    /// Extract full module path from nested attributes
    /// e.g., os.path.exists -> "os.path"
    /// Returns (module_path, final_attr)
    fn extract_module_path(&self, expr: &PyExpr) -> Option<(String, String)> {
        match expr {
            PyExpr::Attribute { value, attr } => {
                match value.as_ref() {
                    // Simple case: module.attr (e.g., math.pi)
                    PyExpr::Name(name) => {
                        let resolved = self.resolve_module_name(name);
                        Some((resolved, attr.clone()))
                    }
                    // Nested case: module.submodule.attr (e.g., os.path.exists)
                    PyExpr::Attribute { value: inner_value, attr: inner_attr } => {
                        if let PyExpr::Name(module_name) = inner_value.as_ref() {
                            let resolved = self.resolve_module_name(module_name);
                            let full_module = format!("{}.{}", resolved, inner_attr);
                            Some((full_module, attr.clone()))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Get indentation string
    fn indent(&self) -> String {
        "    ".repeat(self.indent_level)
    }

    /// Translate Python module to Rust code
    pub fn translate_module(&mut self, module: &PyModule) -> Result<String> {
        let mut code = String::new();

        // Add header
        code.push_str("// Generated by Portalis Python → Rust Translator\n");
        code.push_str("#![allow(unused)]\n\n");

        // Translate statements
        for stmt in &module.statements {
            code.push_str(&self.translate_stmt(stmt)?);
        }

        Ok(code)
    }

    /// Translate Python statement to Rust
    pub fn translate_stmt(&mut self, stmt: &PyStmt) -> Result<String> {
        match stmt {
            PyStmt::Expr(expr) => {
                let expr_code = self.translate_expr(expr)?;
                Ok(format!("{}{};\n", self.indent(), expr_code))
            }

            PyStmt::Assign {
                target,
                value,
            } => {
                let value_code = self.translate_expr(value)?;

                // Infer type from value
                let rust_type = self.type_inference.infer_expr(value);
                let type_str = rust_type.to_rust_str();

                // Get target name (simplified - assumes Name expression)
                let target_name = match target {
                    PyExpr::Name(name) => name.clone(),
                    _ => {
                        return Err(Error::CodeGeneration(
                            "Complex assignment targets not yet supported".to_string()
                        ))
                    }
                };

                // Record type for future reference
                self.type_inference
                    .record_type(target_name.clone(), rust_type.clone());

                Ok(format!(
                    "{}let {}: {} = {};\n",
                    self.indent(),
                    target_name,
                    type_str,
                    value_code
                ))
            }

            PyStmt::AugAssign { target, op, value } => {
                let target_code = self.translate_expr(target)?;
                let value_code = self.translate_expr(value)?;
                let op_str = self.binop_to_rust(*op);
                Ok(format!(
                    "{}{} {}= {};\n",
                    self.indent(),
                    target_code,
                    op_str,
                    value_code
                ))
            }

            PyStmt::FunctionDef {
                name,
                params,
                body,
                return_type,
                decorators,
                is_async,
            } => {
                let mut code = String::new();

                // Translate decorators to Rust attributes
                for decorator in decorators {
                    let rust_attr = self.translate_decorator(decorator);
                    if !rust_attr.is_empty() {
                        code.push_str(&format!("{}{}\n", self.indent(), rust_attr));
                    }
                }

                // Function signature
                let async_keyword = if *is_async { "async " } else { "" };
                code.push_str(&format!("{}pub {}fn {}(", self.indent(), async_keyword, name));

                // Parameters
                let param_strs: Vec<String> = params
                    .iter()
                    .map(|param| {
                        let rust_type = if let Some(annotation) = &param.type_annotation {
                            self.type_annotation_to_rust(annotation)
                        } else {
                            RustType::Unknown
                        };
                        format!("{}: {}", param.name, rust_type.to_rust_str())
                    })
                    .collect();
                code.push_str(&param_strs.join(", "));

                // Return type
                let ret_type = if let Some(annotation) = return_type {
                    self.type_annotation_to_rust(annotation)
                } else {
                    RustType::Unit
                };
                code.push_str(&format!(") -> {} {{\n", ret_type.to_rust_str()));

                // Body
                self.indent_level += 1;
                for stmt in body {
                    code.push_str(&self.translate_stmt(stmt)?);
                }
                self.indent_level -= 1;

                code.push_str(&format!("{}}}\n\n", self.indent()));

                Ok(code)
            }

            PyStmt::Return { value } => {
                if let Some(e) = value {
                    let expr_code = self.translate_expr(e)?;
                    Ok(format!("{}return {};\n", self.indent(), expr_code))
                } else {
                    Ok(format!("{}return;\n", self.indent()))
                }
            }

            PyStmt::Assert { test, msg } => {
                let test_code = self.translate_expr(test)?;
                if let Some(msg_expr) = msg {
                    let msg_code = self.translate_expr(msg_expr)?;
                    Ok(format!("{}assert!({}, {});\n", self.indent(), test_code, msg_code))
                } else {
                    Ok(format!("{}assert!({});\n", self.indent(), test_code))
                }
            }

            PyStmt::If { test, body, orelse } => {
                let mut code = String::new();
                let test_code = self.translate_expr(test)?;

                code.push_str(&format!("{}if {} {{\n", self.indent(), test_code));

                self.indent_level += 1;
                for stmt in body {
                    code.push_str(&self.translate_stmt(stmt)?);
                }
                self.indent_level -= 1;

                code.push_str(&format!("{}}}", self.indent()));

                if !orelse.is_empty() {
                    code.push_str(" else {\n");
                    self.indent_level += 1;
                    for stmt in orelse {
                        code.push_str(&self.translate_stmt(stmt)?);
                    }
                    self.indent_level -= 1;
                    code.push_str(&format!("{}}}", self.indent()));
                }

                code.push('\n');
                Ok(code)
            }

            PyStmt::While { test, body, orelse } => {
                let mut code = String::new();
                let test_code = self.translate_expr(test)?;

                // While-else requires a flag to track if loop completed normally
                if !orelse.is_empty() {
                    code.push_str(&format!("{}let mut _loop_completed = true;\n", self.indent()));
                }

                code.push_str(&format!("{}while {} {{\n", self.indent(), test_code));

                self.indent_level += 1;
                for stmt in body {
                    // If body contains break and we have an else clause, set flag to false BEFORE break
                    if !orelse.is_empty() && matches!(stmt, PyStmt::Break) {
                        code.push_str(&format!("{}_loop_completed = false;\n", self.indent()));
                    }
                    let stmt_code = self.translate_stmt(stmt)?;
                    code.push_str(&stmt_code);
                }
                self.indent_level -= 1;

                code.push_str(&format!("{}}}\n", self.indent()));

                // Handle else clause - executes only if loop completed normally (no break)
                if !orelse.is_empty() {
                    code.push_str(&format!("{}if _loop_completed {{\n", self.indent()));
                    self.indent_level += 1;
                    for stmt in orelse {
                        code.push_str(&self.translate_stmt(stmt)?);
                    }
                    self.indent_level -= 1;
                    code.push_str(&format!("{}}}\n", self.indent()));
                }

                Ok(code)
            }

            PyStmt::For { target, iter, body, orelse } => {
                let mut code = String::new();
                // Translate the iterator expression (range() is already handled in translate_expr)
                let rust_iter = self.translate_expr(iter)?;

                // Get target name (simplified - assumes Name expression)
                let rust_target = match target {
                    PyExpr::Name(name) => {
                        // Register loop variable type (assume i32 for range iterations)
                        self.type_inference.type_map.insert(name.clone(), RustType::I32);
                        name.clone()
                    }
                    _ => {
                        return Err(Error::CodeGeneration(
                            "Complex for loop targets not yet supported".to_string()
                        ))
                    }
                };

                // For-else requires a flag to track if loop completed normally
                if !orelse.is_empty() {
                    code.push_str(&format!("{}let mut _loop_completed = true;\n", self.indent()));
                }

                code.push_str(&format!("{}for {} in {} {{\n", self.indent(), rust_target, rust_iter));

                self.indent_level += 1;
                for stmt in body {
                    // If body contains break and we have an else clause, set flag to false BEFORE break
                    if !orelse.is_empty() && matches!(stmt, PyStmt::Break) {
                        code.push_str(&format!("{}_loop_completed = false;\n", self.indent()));
                    }
                    let stmt_code = self.translate_stmt(stmt)?;
                    code.push_str(&stmt_code);
                }
                self.indent_level -= 1;

                code.push_str(&format!("{}}}\n", self.indent()));

                // Handle else clause - executes only if loop completed normally (no break)
                if !orelse.is_empty() {
                    code.push_str(&format!("{}if _loop_completed {{\n", self.indent()));
                    self.indent_level += 1;
                    for stmt in orelse {
                        code.push_str(&self.translate_stmt(stmt)?);
                    }
                    self.indent_level -= 1;
                    code.push_str(&format!("{}}}\n", self.indent()));
                }

                Ok(code)
            }

            PyStmt::Pass => Ok(format!("{}// pass\n", self.indent())),

            PyStmt::Break => Ok(format!("{}break;\n", self.indent())),

            PyStmt::Continue => Ok(format!("{}continue;\n", self.indent())),

            PyStmt::ClassDef {
                name,
                bases: _,
                body,
                decorators: _,
            } => {
                let mut code = String::new();

                // Struct definition
                code.push_str(&format!("{}pub struct {} {{\n", self.indent(), name));

                // Find __init__ to extract attributes
                let mut attributes = vec![];
                for stmt in body {
                    if let PyStmt::FunctionDef {
                        name: func_name,
                        body: func_body,
                        ..
                    } = stmt
                    {
                        if func_name == "__init__" {
                            // Extract self.x = ... assignments
                            for init_stmt in func_body {
                                if let PyStmt::Assign { target, value } = init_stmt {
                                    // Extract attribute name from target expression
                                    if let PyExpr::Attribute { value: obj, attr } = target {
                                        if let PyExpr::Name(name) = obj.as_ref() {
                                            if name == "self" {
                                                let attr_type = self.type_inference.infer_expr(value);
                                                attributes.push((attr.clone(), attr_type));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Generate struct fields
                self.indent_level += 1;
                for (attr_name, attr_type) in &attributes {
                    code.push_str(&format!(
                        "{}pub {}: {},\n",
                        self.indent(),
                        attr_name,
                        attr_type.to_rust_str()
                    ));
                }
                self.indent_level -= 1;
                code.push_str(&format!("{}}}\n\n", self.indent()));

                // Impl block
                code.push_str(&format!("{}impl {} {{\n", self.indent(), name));
                self.indent_level += 1;

                // Translate methods
                for stmt in body {
                    if let PyStmt::FunctionDef {
                        name: func_name,
                        params,
                        body: func_body,
                        return_type,
                        ..
                    } = stmt
                    {
                        // Skip __init__ - it becomes the struct
                        if func_name == "__init__" {
                            // Generate new() constructor
                            code.push_str(&format!("{}pub fn new(", self.indent()));

                            // Parameters (skip self)
                            let param_strs: Vec<String> = params
                                .iter()
                                .filter(|param| param.name != "self")
                                .map(|param| {
                                    let rust_type = if let Some(annotation) = &param.type_annotation {
                                        self.type_annotation_to_rust(annotation)
                                    } else {
                                        RustType::Unknown
                                    };
                                    format!("{}: {}", param.name, rust_type.to_rust_str())
                                })
                                .collect();
                            code.push_str(&param_strs.join(", "));
                            code.push_str(&format!(") -> Self {{\n"));

                            // Body: create struct
                            self.indent_level += 1;
                            code.push_str(&format!("{}Self {{\n", self.indent()));
                            self.indent_level += 1;
                            for (attr_name, _) in &attributes {
                                code.push_str(&format!("{}{},\n", self.indent(), attr_name));
                            }
                            self.indent_level -= 1;
                            code.push_str(&format!("{}}}\n", self.indent()));
                            self.indent_level -= 1;
                            code.push_str(&format!("{}}}\n\n", self.indent()));
                        } else {
                            // Regular method
                            code.push_str(&format!("{}pub fn {}(", self.indent(), func_name));

                            // Parameters (including &self or &mut self)
                            let param_strs: Vec<String> = params
                                .iter()
                                .map(|param| {
                                    if param.name == "self" {
                                        "&self".to_string()
                                    } else {
                                        let rust_type = if let Some(annotation) = &param.type_annotation {
                                            self.type_annotation_to_rust(annotation)
                                        } else {
                                            RustType::Unknown
                                        };
                                        format!("{}: {}", param.name, rust_type.to_rust_str())
                                    }
                                })
                                .collect();
                            code.push_str(&param_strs.join(", "));

                            // Return type
                            let ret_type = if let Some(hint) = return_type {
                                self.type_annotation_to_rust(hint)
                            } else {
                                RustType::Unit
                            };
                            code.push_str(&format!(") -> {} {{\n", ret_type.to_rust_str()));

                            // Body
                            self.indent_level += 1;
                            for body_stmt in func_body {
                                code.push_str(&self.translate_stmt(body_stmt)?);
                            }
                            self.indent_level -= 1;
                            code.push_str(&format!("{}}}\n\n", self.indent()));
                        }
                    }
                }

                self.indent_level -= 1;
                code.push_str(&format!("{}}}\n\n", self.indent()));

                Ok(code)
            }

            PyStmt::Try { body, handlers, orelse, finalbody } => {
                let mut code = String::new();

                // In Rust, we'll use a closure + match pattern for try-except
                // try: body -> (|| { body })()
                // except: handlers -> match on Result or panic handling

                if !finalbody.is_empty() {
                    // With finally, we need a more complex pattern
                    code.push_str(&format!("{}// try-except-finally\n", self.indent()));
                    code.push_str(&format!("{}{{\n", self.indent()));
                    self.indent_level += 1;

                    // Try body with panic catching (simplified)
                    code.push_str(&format!("{}let _result = std::panic::catch_unwind(|| {{\n", self.indent()));
                    self.indent_level += 1;
                    for stmt in body {
                        code.push_str(&self.translate_stmt(stmt)?);
                    }
                    self.indent_level -= 1;
                    code.push_str(&format!("{}}});\n\n", self.indent()));

                    // Exception handlers
                    if !handlers.is_empty() {
                        code.push_str(&format!("{}if _result.is_err() {{\n", self.indent()));
                        self.indent_level += 1;
                        for handler in handlers {
                            for stmt in &handler.body {
                                code.push_str(&self.translate_stmt(stmt)?);
                            }
                        }
                        self.indent_level -= 1;
                        code.push_str(&format!("{}}}\n", self.indent()));
                    }

                    // Else clause (executes if no exception)
                    if !orelse.is_empty() {
                        code.push_str(&format!("{}if _result.is_ok() {{\n", self.indent()));
                        self.indent_level += 1;
                        for stmt in orelse {
                            code.push_str(&self.translate_stmt(stmt)?);
                        }
                        self.indent_level -= 1;
                        code.push_str(&format!("{}}}\n", self.indent()));
                    }

                    // Finally block (always executes)
                    code.push_str(&format!("{}// finally\n", self.indent()));
                    for stmt in finalbody {
                        code.push_str(&self.translate_stmt(stmt)?);
                    }

                    self.indent_level -= 1;
                    code.push_str(&format!("{}}}\n", self.indent()));
                } else {
                    // Simpler pattern without finally
                    code.push_str(&format!("{}// try-except\n", self.indent()));
                    for stmt in body {
                        code.push_str(&self.translate_stmt(stmt)?);
                    }

                    // For now, just add comments for except blocks
                    for (_i, handler) in handlers.iter().enumerate() {
                        if let Some(exc_type_expr) = &handler.exception_type {
                            let exc_type = self.translate_expr(exc_type_expr)?;
                            code.push_str(&format!("{}// except {}\n", self.indent(), exc_type));
                        } else {
                            code.push_str(&format!("{}// except (bare)\n", self.indent()));
                        }
                    }
                }

                Ok(code)
            }

            PyStmt::Raise { exception } => {
                if let Some(e) = exception {
                    let exc_code = self.translate_expr(e)?;
                    Ok(format!("{}panic!(\"{{:?}}\", {});\n", self.indent(), exc_code))
                } else {
                    Ok(format!("{}panic!(\"Exception raised\");\n", self.indent()))
                }
            }

            PyStmt::Import { modules: _ } => {
                // Import statements are handled at module level
                // modules is Vec<(String, Option<String>)> - (module_name, optional_alias)
                Ok(String::new())
            }

            PyStmt::ImportFrom { module: _, names: _, level: _ } => {
                // From-import statements are handled at module level
                // module is Option<String>, names is Vec<(String, Option<String>)>, level is usize
                Ok(String::new())
            }

            PyStmt::With { items, body } => {
                let mut code = String::new();

                // Translate context managers to Rust scoped blocks
                // For file operations, we'll use explicit scope + drop
                // For locks/other resources, similar pattern

                for (idx, item) in items.iter().enumerate() {
                    let context_code = self.translate_expr(&item.context_expr)?;

                    // Determine if this is a file operation or other resource
                    let is_file_like = context_code.contains("open(")
                        || context_code.contains("File::open")
                        || context_code.contains("File::create");

                    if let Some(var) = &item.optional_vars {
                        let var_name = self.translate_expr(var)?;
                        if is_file_like {
                            // File operations: let var = File::open(...)?;
                            code.push_str(&format!(
                                "{}let mut {} = {};\n",
                                self.indent(),
                                var_name,
                                context_code
                            ));
                        } else {
                            // Generic resource: create scoped binding
                            code.push_str(&format!(
                                "{}let {} = {};\n",
                                self.indent(),
                                var_name,
                                context_code
                            ));
                        }
                    } else {
                        // No variable binding - just execute
                        code.push_str(&format!(
                            "{}{};\n",
                            self.indent(),
                            context_code
                        ));
                    }

                    // Open scope for resource cleanup
                    if idx == items.len() - 1 {
                        code.push_str(&format!("{}{{\n", self.indent()));
                    }
                }

                // Translate body
                self.indent_level += 1;
                for stmt in body {
                    code.push_str(&self.translate_stmt(stmt)?);
                }
                self.indent_level -= 1;

                // Close scope (resources will be dropped)
                code.push_str(&format!("{}}}\n", self.indent()));
                code.push_str(&format!("{}// End of with block\n", self.indent()));

                Ok(code)
            }

            _ => Err(Error::CodeGeneration(format!(
                "Statement type not yet implemented: {:?}",
                stmt
            ))),
        }
    }

    /// Translate Python expression to Rust
    pub fn translate_expr(&mut self, expr: &PyExpr) -> Result<String> {
        match expr {
            PyExpr::Literal(lit) => self.translate_literal(lit),

            PyExpr::Name(name) => Ok(name.clone()),

            PyExpr::Await(value) => {
                let value_code = self.translate_expr(value)?;
                Ok(format!("{}.await", value_code))
            }

            PyExpr::BinOp { left, op, right } => {
                let left_code = self.translate_expr(left)?;
                let right_code = self.translate_expr(right)?;
                let op_str = self.binop_to_rust(*op);
                Ok(format!("{} {} {}", left_code, op_str, right_code))
            }

            PyExpr::UnaryOp { op, operand } => {
                let operand_code = self.translate_expr(operand)?;
                let op_str = self.unaryop_to_rust(*op);
                Ok(format!("{}{}", op_str, operand_code))
            }

            PyExpr::Call { func, args, .. } => {
                // Check if it's a method call (object.method())
                if let PyExpr::Attribute { value, attr } = func.as_ref() {
                    // Try to extract module path (handles nested like os.path.exists())
                    if let Some((module_path, final_attr)) = self.extract_module_path(func) {
                        let parts: Vec<&str> = module_path.split('.').collect();
                        let base_module = parts[0];

                        if self.imported_modules.contains(&module_path) || self.imported_modules.contains(&base_module.to_string()) {
                            let args_code: Vec<String> = args
                                .iter()
                                .map(|arg| self.translate_expr(arg))
                                .collect::<Result<Vec<_>>>()?;

                            // Handle special cases for known modules
                            if module_path == "math" && final_attr == "sqrt" {
                                if !args_code.is_empty() {
                                    return Ok(format!("({} as f64).sqrt()", args_code[0]));
                                }
                            }
                            if module_path == "math" && final_attr == "pow" {
                                if args_code.len() >= 2 {
                                    return Ok(format!("({} as f64).powf({} as f64)", args_code[0], args_code[1]));
                                }
                            }
                            if module_path == "json" && final_attr == "dumps" {
                                if !args_code.is_empty() {
                                    return Ok(format!("serde_json::to_string(&{})?", args_code[0]));
                                }
                            }
                            if module_path == "json" && final_attr == "loads" {
                                if !args_code.is_empty() {
                                    return Ok(format!("serde_json::from_str({})?", args_code[0]));
                                }
                            }
                            // Handle nested module functions like os.path.exists()
                            if module_path == "os.path" && final_attr == "exists" {
                                if !args_code.is_empty() {
                                    return Ok(format!("std::path::Path::new(&{}).exists()", args_code[0]));
                                }
                            }
                            if module_path == "os.path" && final_attr == "join" {
                                if args_code.len() >= 2 {
                                    return Ok(format!("std::path::Path::new(&{}).join({})", args_code[0], args_code[1]));
                                }
                            }
                        }
                    }

                    let value_code = self.translate_expr(value)?;
                    let args_code: Vec<String> = args
                        .iter()
                        .map(|arg| self.translate_expr(arg))
                        .collect::<Result<Vec<_>>>()?;

                    // Translate common string methods
                    match attr.as_str() {
                        "upper" => return Ok(format!("{}.to_uppercase()", value_code)),
                        "lower" => return Ok(format!("{}.to_lowercase()", value_code)),
                        "strip" => return Ok(format!("{}.trim()", value_code)),
                        "lstrip" => return Ok(format!("{}.trim_start()", value_code)),
                        "rstrip" => return Ok(format!("{}.trim_end()", value_code)),
                        "split" => {
                            if args_code.is_empty() {
                                return Ok(format!("{}.split_whitespace().collect::<Vec<_>>()", value_code));
                            } else {
                                return Ok(format!("{}.split({}).collect::<Vec<_>>()", value_code, args_code[0]));
                            }
                        }
                        "join" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.join(&{})", args_code[0], value_code));
                            }
                        }
                        "replace" => {
                            if args_code.len() == 2 {
                                return Ok(format!("{}.replace({}, {})", value_code, args_code[0], args_code[1]));
                            }
                        }
                        "startswith" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.starts_with({})", value_code, args_code[0]));
                            }
                        }
                        "endswith" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.ends_with({})", value_code, args_code[0]));
                            }
                        }
                        "find" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.find({}).unwrap_or(-1_isize) as i32", value_code, args_code[0]));
                            }
                        }
                        "count" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.matches({}).count()", value_code, args_code[0]));
                            }
                        }
                        // List methods
                        "append" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.push({})", value_code, args_code[0]));
                            }
                        }
                        "extend" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.extend({})", value_code, args_code[0]));
                            }
                        }
                        "pop" => {
                            if args_code.is_empty() {
                                return Ok(format!("{}.pop().unwrap()", value_code));
                            } else {
                                return Ok(format!("{}.remove({} as usize)", value_code, args_code[0]));
                            }
                        }
                        "remove" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{{ if let Some(pos) = {}.iter().position(|x| x == &{}) {{ {}.remove(pos); }} }}",
                                    value_code, args_code[0], value_code));
                            }
                        }
                        "clear" => return Ok(format!("{}.clear()", value_code)),
                        "reverse" => return Ok(format!("{}.reverse()", value_code)),
                        "sort" => return Ok(format!("{}.sort()", value_code)),
                        _ => {
                            // Default method call
                            return Ok(format!("{}.{}({})", value_code, attr, args_code.join(", ")));
                        }
                    }
                }

                // Check if it's a built-in function that needs special translation
                if let PyExpr::Name(func_name) = func.as_ref() {
                    let args_code: Vec<String> = args
                        .iter()
                        .map(|arg| self.translate_expr(arg))
                        .collect::<Result<Vec<_>>>()?;

                    match func_name.as_str() {
                        // Built-in functions with special translation
                        "len" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.len()", args_code[0]));
                            }
                        }
                        "max" => {
                            if args_code.len() == 1 {
                                // max(list) -> *list.iter().max().unwrap()
                                return Ok(format!("*{}.iter().max().unwrap()", args_code[0]));
                            } else if args_code.len() > 1 {
                                // max(a, b, c) -> *[a, b, c].iter().max().unwrap()
                                return Ok(format!("*[{}].iter().max().unwrap()", args_code.join(", ")));
                            }
                        }
                        "min" => {
                            if args_code.len() == 1 {
                                return Ok(format!("*{}.iter().min().unwrap()", args_code[0]));
                            } else if args_code.len() > 1 {
                                return Ok(format!("*[{}].iter().min().unwrap()", args_code.join(", ")));
                            }
                        }
                        "sum" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.iter().sum::<i32>()", args_code[0]));
                            }
                        }
                        "abs" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.abs()", args_code[0]));
                            }
                        }
                        "sorted" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{{ let mut v = {}.clone(); v.sort(); v }}", args_code[0]));
                            }
                        }
                        "reversed" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{{ let mut v = {}.clone(); v.reverse(); v }}", args_code[0]));
                            }
                        }
                        "print" => {
                            return Ok(format!("println!(\"{{:?}}\", {})", args_code.join(", ")));
                        }
                        "range" => {
                            // Translate range() to Rust range syntax
                            // Note: Do NOT add parentheses - they'll be added by caller if needed
                            if args_code.len() == 1 {
                                return Ok(format!("0..{}", args_code[0]));
                            } else if args_code.len() == 2 {
                                return Ok(format!("{}..{}", args_code[0], args_code[1]));
                            } else if args_code.len() == 3 {
                                // range(start, stop, step) - step requires wrapping
                                return Ok(format!("({}..{}).step_by({} as usize)", args_code[0], args_code[1], args_code[2]));
                            }
                        }
                        "enumerate" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.iter().enumerate()", args_code[0]));
                            }
                        }
                        "zip" => {
                            if args_code.len() == 2 {
                                return Ok(format!("{}.iter().zip({}.iter())", args_code[0], args_code[1]));
                            } else if args_code.len() > 2 {
                                // For multiple iterables, chain zip calls
                                let mut result = format!("{}.iter().zip({}.iter())", args_code[0], args_code[1]);
                                for arg in &args_code[2..] {
                                    result = format!("{}.zip({}.iter())", result, arg);
                                }
                                return Ok(result);
                            }
                        }
                        "any" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.iter().any(|x| *x)", args_code[0]));
                            }
                        }
                        "all" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.iter().all(|x| *x)", args_code[0]));
                            }
                        }
                        "isinstance" => {
                            // isinstance is complex, just generate a comment for now
                            return Ok(format!("/* isinstance({}, {}) */true", args_code.join(", "), ""));
                        }
                        "type" => {
                            if args_code.len() == 1 {
                                return Ok(format!("/* type({}) */", args_code[0]));
                            }
                        }
                        "list" => {
                            // list() constructor - convert iterator to Vec
                            if args_code.len() == 1 {
                                // Check if it's already an iterator expression
                                if args_code[0].contains(".iter()") || args_code[0].contains("..") {
                                    return Ok(format!("{}.collect::<Vec<_>>()", args_code[0]));
                                } else {
                                    return Ok(format!("{}.iter().collect::<Vec<_>>()", args_code[0]));
                                }
                            } else if args_code.is_empty() {
                                return Ok("Vec::new()".to_string());
                            }
                        }
                        "dict" => {
                            if args_code.is_empty() {
                                return Ok("HashMap::new()".to_string());
                            }
                        }
                        "str" => {
                            if args_code.len() == 1 {
                                return Ok(format!("{}.to_string()", args_code[0]));
                            }
                        }
                        "int" => {
                            if args_code.len() == 1 {
                                return Ok(format!("({} as i32)", args_code[0]));
                            }
                        }
                        "float" => {
                            if args_code.len() == 1 {
                                return Ok(format!("({} as f64)", args_code[0]));
                            }
                        }
                        "bool" => {
                            if args_code.len() == 1 {
                                return Ok(format!("({} as bool)", args_code[0]));
                            }
                        }
                        _ => {
                            // Default: translate as regular function call
                            return Ok(format!("{}({})", func_name, args_code.join(", ")));
                        }
                    }
                }

                // Fallback for non-Name function expressions
                let func_code = self.translate_expr(func)?;
                let args_code: Vec<String> = args
                    .iter()
                    .map(|arg| self.translate_expr(arg))
                    .collect::<Result<Vec<_>>>()?;
                Ok(format!("{}({})", func_code, args_code.join(", ")))
            }

            PyExpr::List(elements) => {
                let elements_code: Vec<String> = elements
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(format!("vec![{}]", elements_code.join(", ")))
            }

            PyExpr::Tuple(elements) => {
                let elements_code: Vec<String> = elements
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(format!("({})", elements_code.join(", ")))
            }

            PyExpr::Compare { left, op, right } => {
                let left_code = self.translate_expr(left)?;
                let right_code = self.translate_expr(right)?;
                let op_str = self.cmpop_to_rust(*op);
                Ok(format!("{} {} {}", left_code, op_str, right_code))
            }

            PyExpr::Subscript { value, index } => {
                let value_code = self.translate_expr(value)?;
                let index_code = self.translate_expr(index)?;
                Ok(format!("{}[{}]", value_code, index_code))
            }

            PyExpr::Slice { value, lower, upper, step } => {
                let value_code = self.translate_expr(value)?;

                // Translate Python slices to Rust slice syntax
                // Python: list[1:3] -> Rust: &list[1..3]
                // Python: list[:5] -> Rust: &list[..5]
                // Python: list[2:] -> Rust: &list[2..]
                // Python: list[::2] -> Rust: list.iter().step_by(2).copied().collect::<Vec<_>>()
                // Python: list[1:10:2] -> more complex

                if step.is_some() {
                    // Slicing with step requires iterator approach
                    let step_code = self.translate_expr(step.as_ref().unwrap())?;

                    let range_str = if lower.is_some() && upper.is_some() {
                        let lower_code = self.translate_expr(lower.as_ref().unwrap())?;
                        let upper_code = self.translate_expr(upper.as_ref().unwrap())?;
                        format!("{}..{}", lower_code, upper_code)
                    } else if lower.is_some() {
                        let lower_code = self.translate_expr(lower.as_ref().unwrap())?;
                        format!("{}..", lower_code)
                    } else if upper.is_some() {
                        let upper_code = self.translate_expr(upper.as_ref().unwrap())?;
                        format!("..{}", upper_code)
                    } else {
                        "..".to_string()
                    };

                    Ok(format!(
                        "{}[{}].iter().step_by({} as usize).copied().collect::<Vec<_>>()",
                        value_code, range_str, step_code
                    ))
                } else {
                    // Simple slice without step
                    let slice_str = if lower.is_some() && upper.is_some() {
                        let lower_code = self.translate_expr(lower.as_ref().unwrap())?;
                        let upper_code = self.translate_expr(upper.as_ref().unwrap())?;
                        format!("{}..{}", lower_code, upper_code)
                    } else if lower.is_some() {
                        let lower_code = self.translate_expr(lower.as_ref().unwrap())?;
                        format!("{}..", lower_code)
                    } else if upper.is_some() {
                        let upper_code = self.translate_expr(upper.as_ref().unwrap())?;
                        format!("..{}", upper_code)
                    } else {
                        "..".to_string()
                    };

                    Ok(format!("&{}[{}]", value_code, slice_str))
                }
            }

            PyExpr::Attribute { value, attr } => {
                // Try to extract module path (handles nested attributes like os.path.exists)
                if let Some((module_path, final_attr)) = self.extract_module_path(&PyExpr::Attribute {
                    value: value.clone(),
                    attr: attr.clone()
                }) {
                    // Check if this module path was imported (either "os" or "os.path")
                    let parts: Vec<&str> = module_path.split('.').collect();
                    let base_module = parts[0];

                    if self.imported_modules.contains(&module_path) || self.imported_modules.contains(&base_module.to_string()) {
                        // Try to translate using stdlib mapper with full path
                        if let Some(rust_equiv) = self.stdlib_mapper.get_function(&module_path, &final_attr) {
                            // For constants like math.pi, return the full path
                            if rust_equiv.contains("::") {
                                return Ok(rust_equiv.clone());
                            }
                        }

                        // Check for module-level constants
                        if let Some(_mapping) = self.stdlib_mapper.get_module(&module_path) {
                            // Special cases for constants
                            if module_path == "math" && final_attr == "pi" {
                                return Ok("std::f64::consts::PI".to_string());
                            }
                            if module_path == "math" && final_attr == "e" {
                                return Ok("std::f64::consts::E".to_string());
                            }
                        }
                    }
                }

                // Default attribute access
                let value_code = self.translate_expr(value)?;
                Ok(format!("{}.{}", value_code, attr))
            }

            PyExpr::Dict { keys, values } => {
                if keys.is_empty() {
                    return Ok("HashMap::new()".to_string());
                }

                let mut pairs = vec![];
                for (key, value) in keys.iter().zip(values.iter()) {
                    let key_code = self.translate_expr(key)?;
                    let value_code = self.translate_expr(value)?;
                    pairs.push(format!("({}, {})", key_code, value_code));
                }

                Ok(format!(
                    "HashMap::from([{}])",
                    pairs.join(", ")
                ))
            }

            PyExpr::ListComp { element, generators } => {
                // Translate to iterator chain: iter.map(...).filter(...).collect()
                if generators.is_empty() {
                    return Err(Error::CodeGeneration("List comprehension with no generators".to_string()));
                }

                let comp = &generators[0];
                let iter_code = self.translate_expr(&comp.iter)?;
                let element_code = self.translate_expr(element)?;

                // Convert range() to Rust range
                let rust_iter = if iter_code.starts_with("range(") {
                    let args = &iter_code[6..iter_code.len() - 1];
                    if args.contains(',') {
                        let parts: Vec<&str> = args.split(',').collect();
                        format!("({}..{})", parts[0].trim(), parts[1].trim())
                    } else {
                        format!("(0..{})", args)
                    }
                } else {
                    iter_code
                };

                // Build the iterator chain
                let target_var = self.translate_expr(&comp.target)?;
                let mut code = format!("{}.map(|{}| {})", rust_iter, target_var, element_code);

                // Add filter if there are conditions
                if !comp.ifs.is_empty() {
                    for condition in &comp.ifs {
                        let condition_code = self.translate_expr(condition)?;
                        code = format!("{}.filter(|{}| {})", code, target_var, condition_code);
                    }
                }

                // Collect into Vec
                code = format!("{}.collect::<Vec<_>>()", code);

                Ok(code)
            }

            PyExpr::Lambda { args, body } => {
                // Translate Python lambda to Rust closure
                // Python: lambda x: x + 1 -> Rust: |x| x + 1
                // Python: lambda x, y: x + y -> Rust: |x, y| x + y

                let args_str = args.join(", ");
                let body_code = self.translate_expr(body)?;

                Ok(format!("|{}| {}", args_str, body_code))
            }

            _ => Err(Error::CodeGeneration(format!(
                "Expression type not yet implemented: {:?}",
                expr
            ))),
        }
    }

    /// Convert Python comparison operator to Rust
    fn cmpop_to_rust(&self, op: CmpOp) -> &str {
        match op {
            CmpOp::Eq => "==",
            CmpOp::NotEq => "!=",
            CmpOp::Lt => "<",
            CmpOp::LtE => "<=",
            CmpOp::Gt => ">",
            CmpOp::GtE => ">=",
            CmpOp::Is => "==", // Simplified
            CmpOp::IsNot => "!=", // Simplified
            CmpOp::In => "contains", // Needs special handling
            CmpOp::NotIn => "!contains", // Needs special handling
        }
    }

    /// Translate Python literal to Rust
    fn translate_literal(&self, lit: &PyLiteral) -> Result<String> {
        match lit {
            PyLiteral::Int(n) => Ok(n.to_string()),
            PyLiteral::Float(f) => Ok(f.to_string()),
            PyLiteral::String(s) => Ok(format!("\"{}\"", s)),
            PyLiteral::Bool(b) => Ok(b.to_string()),
            PyLiteral::None => Ok("None".to_string()), // Will need Option handling
            PyLiteral::Bytes(_) => Ok("vec![]".to_string()), // Simplified
        }
    }

    /// Convert Python binary operator to Rust
    fn binop_to_rust(&self, op: BinOp) -> &str {
        match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mult => "*",
            BinOp::Div => "/",
            BinOp::FloorDiv => "/", // Need to add integer division handling
            BinOp::Mod => "%",
            BinOp::Pow => "pow", // Need function call
            BinOp::LShift => "<<",
            BinOp::RShift => ">>",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::BitAnd => "&",
            BinOp::MatMult => "*", // Simplified
        }
    }

    /// Convert Python unary operator to Rust
    fn unaryop_to_rust(&self, op: UnaryOp) -> &str {
        match op {
            UnaryOp::Invert => "!",
            UnaryOp::Not => "!",
            UnaryOp::UAdd => "+",
            UnaryOp::USub => "-",
        }
    }

    /// Convert TypeAnnotation to Rust type
    fn type_annotation_to_rust(&self, annotation: &TypeAnnotation) -> RustType {
        match annotation {
            TypeAnnotation::Name(name) => self.python_type_to_rust(name),
            TypeAnnotation::Generic { base, args } => {
                // Handle generic types like List[int], Dict[str, int]
                let base_type = if let TypeAnnotation::Name(name) = base.as_ref() {
                    name.as_str()
                } else {
                    return RustType::Unknown;
                };

                match base_type {
                    "List" | "list" => {
                        if let Some(inner) = args.first() {
                            let inner_type = self.type_annotation_to_rust(inner);
                            RustType::Vec(Box::new(inner_type))
                        } else {
                            RustType::Vec(Box::new(RustType::Unknown))
                        }
                    }
                    "Optional" | "Option" => {
                        if let Some(inner) = args.first() {
                            let inner_type = self.type_annotation_to_rust(inner);
                            RustType::Option(Box::new(inner_type))
                        } else {
                            RustType::Option(Box::new(RustType::Unknown))
                        }
                    }
                    _ => RustType::Unknown,
                }
            }
        }
    }

    /// Convert Python type hint string to Rust type
    fn python_type_to_rust(&self, hint: &str) -> RustType {
        match hint {
            "int" => RustType::I32,
            "float" => RustType::F64,
            "str" => RustType::String,
            "bool" => RustType::Bool,
            _ => RustType::Unknown,
        }
    }

    /// Translate Python decorator to Rust attribute
    fn translate_decorator(&self, decorator: &PyExpr) -> String {
        // Extract decorator name from expression
        let decorator_name = match decorator {
            PyExpr::Name(name) => name.as_str(),
            PyExpr::Call { func, .. } => {
                // For decorator calls like @lru_cache(maxsize=128)
                if let PyExpr::Name(name) = func.as_ref() {
                    name.as_str()
                } else {
                    return String::new();
                }
            }
            _ => return String::new(),
        };

        match decorator_name {
            // Common Python decorators -> Rust attributes
            "staticmethod" => "#[allow(non_snake_case)]".to_string(),
            "classmethod" => "#[allow(non_snake_case)]".to_string(),
            "property" => "#[inline]".to_string(),
            "abstractmethod" => "".to_string(), // No direct equivalent
            "dataclass" => "#[derive(Debug, Clone)]".to_string(),
            "lru_cache" | "cache" => "// TODO: Add caching".to_string(),
            "override" => "#[inline]".to_string(),
            "deprecated" => "#[deprecated]".to_string(),
            "async" | "asyncio.coroutine" => "#[tokio::main]".to_string(),
            "pytest.fixture" => "#[test]".to_string(),
            "unittest.mock.patch" => "// Mock decorator".to_string(),
            _ => {
                // For unknown decorators, add as comment
                format!("// @{}", decorator_name)
            }
        }
    }
}

impl Default for PythonToRustTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_simple_assignment() {
        let mut translator = PythonToRustTranslator::new();

        let module = PyModule {
            statements: vec![PyStmt::Assign {
                target: PyExpr::Name("x".to_string()),
                value: PyExpr::Literal(PyLiteral::Int(42)),
            }],
        };

        let result = translator.translate_module(&module).unwrap();
        assert!(result.contains("let x: i32 = 42;"));
    }

    #[test]
    fn test_translate_function() {
        let mut translator = PythonToRustTranslator::new();

        let module = PyModule {
            statements: vec![PyStmt::FunctionDef {
                name: "add".to_string(),
                params: vec![
                    FunctionParam {
                        name: "a".to_string(),
                        type_annotation: Some(TypeAnnotation::Name("int".to_string())),
                        default_value: None,
                    },
                    FunctionParam {
                        name: "b".to_string(),
                        type_annotation: Some(TypeAnnotation::Name("int".to_string())),
                        default_value: None,
                    },
                ],
                body: vec![PyStmt::Return {
                    value: Some(PyExpr::BinOp {
                        left: Box::new(PyExpr::Name("a".to_string())),
                        op: BinOp::Add,
                        right: Box::new(PyExpr::Name("b".to_string())),
                    })
                }],
                return_type: Some(TypeAnnotation::Name("int".to_string())),
                decorators: vec![],
                is_async: false,
            }],
        };

        let result = translator.translate_module(&module).unwrap();
        assert!(result.contains("pub fn add(a: i32, b: i32) -> i32"));
        assert!(result.contains("return a + b;"));
    }

    #[test]
    fn test_type_inference_int() {
        let inference = TypeInference::new();
        let lit = PyLiteral::Int(42);
        assert_eq!(inference.infer_from_literal(&lit), RustType::I32);
    }

    #[test]
    fn test_type_inference_float() {
        let inference = TypeInference::new();
        let lit = PyLiteral::Float(3.14);
        assert_eq!(inference.infer_from_literal(&lit), RustType::F64);
    }

    #[test]
    fn test_type_inference_string() {
        let inference = TypeInference::new();
        let lit = PyLiteral::String("hello".to_string());
        assert_eq!(inference.infer_from_literal(&lit), RustType::String);
    }
}
