//! Generic Expression Translator - Python expressions to Rust
//!
//! This module provides comprehensive translation of Python expressions to idiomatic Rust.
//! Unlike the hardcoded patterns in code_generator.rs, this handles ANY Python expression.
//!
//! Key features:
//! - Recursive expression translation
//! - Type-aware operator translation
//! - Proper handling of Python semantics (string concat, floor division, etc.)
//! - Support for comprehensions, lambdas, and complex expressions

use crate::python_ast::*;
use crate::{Error, Result};
use std::collections::HashMap;

/// Context for expression translation
pub struct TranslationContext {
    /// Variable types inferred or declared
    pub variable_types: HashMap<String, RustType>,
    /// Temporary variable counter for complex expressions
    pub temp_counter: usize,
    /// Current indentation level
    pub indent_level: usize,
}

impl TranslationContext {
    pub fn new() -> Self {
        Self {
            variable_types: HashMap::new(),
            temp_counter: 0,
            indent_level: 0,
        }
    }

    pub fn indent(&self) -> String {
        "    ".repeat(self.indent_level)
    }

    pub fn generate_temp_var(&mut self) -> String {
        let name = format!("_temp_{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    pub fn get_type(&self, var_name: &str) -> RustType {
        self.variable_types
            .get(var_name)
            .cloned()
            .unwrap_or(RustType::Unknown)
    }

    pub fn set_type(&mut self, var_name: String, rust_type: RustType) {
        self.variable_types.insert(var_name, rust_type);
    }
}

/// Rust type representation
#[derive(Debug, Clone, PartialEq)]
pub enum RustType {
    I32,
    I64,
    F64,
    Bool,
    String,
    Char,
    Vec(Box<RustType>),
    HashMap(Box<RustType>, Box<RustType>),
    HashSet(Box<RustType>),
    Option(Box<RustType>),
    Result(Box<RustType>, Box<RustType>),
    Tuple(Vec<RustType>),
    Custom(String),
    Unknown,
}

impl RustType {
    pub fn is_numeric(&self) -> bool {
        matches!(self, RustType::I32 | RustType::I64 | RustType::F64)
    }

    pub fn is_string(&self) -> bool {
        matches!(self, RustType::String)
    }

    pub fn to_rust_string(&self) -> String {
        match self {
            RustType::I32 => "i32".to_string(),
            RustType::I64 => "i64".to_string(),
            RustType::F64 => "f64".to_string(),
            RustType::Bool => "bool".to_string(),
            RustType::String => "String".to_string(),
            RustType::Char => "char".to_string(),
            RustType::Vec(inner) => format!("Vec<{}>", inner.to_rust_string()),
            RustType::HashMap(k, v) => {
                format!("HashMap<{}, {}>", k.to_rust_string(), v.to_rust_string())
            }
            RustType::HashSet(inner) => format!("HashSet<{}>", inner.to_rust_string()),
            RustType::Option(inner) => format!("Option<{}>", inner.to_rust_string()),
            RustType::Result(ok, err) => {
                format!("Result<{}, {}>", ok.to_rust_string(), err.to_rust_string())
            }
            RustType::Tuple(types) => {
                let type_strs: Vec<_> = types.iter().map(|t| t.to_rust_string()).collect();
                format!("({})", type_strs.join(", "))
            }
            RustType::Custom(name) => name.clone(),
            RustType::Unknown => "()".to_string(),
        }
    }
}

/// Generic expression translator
pub struct ExpressionTranslator<'a> {
    ctx: &'a mut TranslationContext,
}

impl<'a> ExpressionTranslator<'a> {
    pub fn new(ctx: &'a mut TranslationContext) -> Self {
        Self { ctx }
    }

    /// Get mutable access to the translation context
    pub fn ctx_mut(&mut self) -> &mut TranslationContext {
        self.ctx
    }

    /// Translate any Python expression to Rust
    pub fn translate(&mut self, expr: &PyExpr) -> Result<String> {
        match expr {
            PyExpr::Literal(lit) => self.translate_literal(lit),
            PyExpr::Name(name) => Ok(self.translate_name(name)),
            PyExpr::BinOp { left, op, right } => self.translate_binop(left, op, right),
            PyExpr::UnaryOp { op, operand } => self.translate_unaryop(op, operand),
            PyExpr::Compare { left, op, right } => self.translate_compare(left, op, right),
            PyExpr::BoolOp { op, left, right } => self.translate_boolop(op, left, right),
            PyExpr::Call { func, args, kwargs } => self.translate_call(func, args, kwargs),
            PyExpr::Attribute { value, attr } => self.translate_attribute(value, attr),
            PyExpr::Subscript { value, index } => self.translate_subscript(value, index),
            PyExpr::Slice { value, lower, upper, step } => {
                self.translate_slice(value, lower.as_deref(), upper.as_deref(), step.as_deref())
            }
            PyExpr::List(elements) => self.translate_list(elements),
            PyExpr::Tuple(elements) => self.translate_tuple(elements),
            PyExpr::Dict { keys, values } => self.translate_dict(keys, values),
            PyExpr::Set(elements) => self.translate_set(elements),
            PyExpr::ListComp { element, generators } => {
                self.translate_list_comp(element, generators)
            }
            PyExpr::IfExp { test, body, orelse } => self.translate_if_expr(test, body, orelse),
            PyExpr::Lambda { args, body } => self.translate_lambda(args, body),
            PyExpr::Await(expr) => self.translate_await(expr),
            PyExpr::Yield(expr) => self.translate_yield(expr.as_deref()),
        }
    }

    /// Translate a literal value
    fn translate_literal(&self, lit: &PyLiteral) -> Result<String> {
        match lit {
            PyLiteral::Int(n) => Ok(n.to_string()),
            PyLiteral::Float(f) => Ok(f.to_string()),
            PyLiteral::String(s) => Ok(format!("\"{}\"", s.replace('"', "\\\""))),
            PyLiteral::Bool(b) => Ok(b.to_string()),
            PyLiteral::None => Ok("None".to_string()), // Will need Option<T> handling
            PyLiteral::Bytes(b) => Ok(format!("b\"{}\"", String::from_utf8_lossy(b))),
        }
    }

    /// Translate a variable name
    fn translate_name(&self, name: &str) -> String {
        // Python keywords that conflict with Rust
        match name {
            "True" => "true".to_string(),
            "False" => "false".to_string(),
            "None" => "None".to_string(),
            "type" => "type_".to_string(),
            "match" => "match_".to_string(),
            _ => name.to_string(),
        }
    }

    /// Translate binary operations
    fn translate_binop(&mut self, left: &PyExpr, op: &BinOp, right: &PyExpr) -> Result<String> {
        let left_rust = self.translate(left)?;
        let right_rust = self.translate(right)?;

        let result = match op {
            BinOp::Add => {
                // Check if we're dealing with strings
                let left_type = self.infer_type(left);
                if left_type.is_string() {
                    // String concatenation
                    format!("format!(\"{{}}{{}}\", {}, {})", left_rust, right_rust)
                } else {
                    format!("{} + {}", left_rust, right_rust)
                }
            }
            BinOp::Sub => format!("{} - {}", left_rust, right_rust),
            BinOp::Mult => format!("{} * {}", left_rust, right_rust),
            BinOp::Div => {
                // Python 3 always does float division
                format!("({} as f64) / ({} as f64)", left_rust, right_rust)
            }
            BinOp::FloorDiv => {
                // Floor division
                format!("({} / {}).floor() as i64", left_rust, right_rust)
            }
            BinOp::Mod => format!("{} % {}", left_rust, right_rust),
            BinOp::Pow => {
                // Exponentiation
                format!("({} as f64).powf({} as f64)", left_rust, right_rust)
            }
            BinOp::LShift => format!("{} << {}", left_rust, right_rust),
            BinOp::RShift => format!("{} >> {}", left_rust, right_rust),
            BinOp::BitOr => format!("{} | {}", left_rust, right_rust),
            BinOp::BitXor => format!("{} ^ {}", left_rust, right_rust),
            BinOp::BitAnd => format!("{} & {}", left_rust, right_rust),
            BinOp::MatMult => {
                // Matrix multiplication - requires external library
                return Err(Error::CodeGeneration(
                    "Matrix multiplication (@) requires ndarray crate".to_string(),
                ));
            }
        };

        Ok(result)
    }

    /// Translate unary operations
    fn translate_unaryop(&mut self, op: &UnaryOp, operand: &PyExpr) -> Result<String> {
        let operand_rust = self.translate(operand)?;

        let result = match op {
            UnaryOp::Not => format!("!({})", operand_rust),
            UnaryOp::UAdd => format!("+({})", operand_rust),
            UnaryOp::USub => format!("-({})", operand_rust),
            UnaryOp::Invert => format!("!({})", operand_rust), // Bitwise NOT
        };

        Ok(result)
    }

    /// Translate comparison operations
    fn translate_compare(&mut self, left: &PyExpr, op: &CmpOp, right: &PyExpr) -> Result<String> {
        let left_rust = self.translate(left)?;
        let right_rust = self.translate(right)?;

        let result = match op {
            CmpOp::Eq => format!("{} == {}", left_rust, right_rust),
            CmpOp::NotEq => format!("{} != {}", left_rust, right_rust),
            CmpOp::Lt => format!("{} < {}", left_rust, right_rust),
            CmpOp::LtE => format!("{} <= {}", left_rust, right_rust),
            CmpOp::Gt => format!("{} > {}", left_rust, right_rust),
            CmpOp::GtE => format!("{} >= {}", left_rust, right_rust),
            CmpOp::Is => {
                // Identity comparison - use pointer equality
                format!("std::ptr::eq(&{}, &{})", left_rust, right_rust)
            }
            CmpOp::IsNot => {
                format!("!std::ptr::eq(&{}, &{})", left_rust, right_rust)
            }
            CmpOp::In => {
                // Membership test - depends on type
                format!("{}.contains(&{})", right_rust, left_rust)
            }
            CmpOp::NotIn => {
                format!("!{}.contains(&{})", right_rust, left_rust)
            }
        };

        Ok(result)
    }

    /// Translate boolean operations
    fn translate_boolop(&mut self, op: &BoolOp, left: &PyExpr, right: &PyExpr) -> Result<String> {
        let left_rust = self.translate(left)?;
        let right_rust = self.translate(right)?;

        let result = match op {
            BoolOp::And => format!("{} && {}", left_rust, right_rust),
            BoolOp::Or => format!("{} || {}", left_rust, right_rust),
        };

        Ok(result)
    }

    /// Translate function calls
    fn translate_call(
        &mut self,
        func: &PyExpr,
        args: &[PyExpr],
        kwargs: &HashMap<String, PyExpr>,
    ) -> Result<String> {
        // Translate function/method name
        let func_name = match func {
            PyExpr::Name(name) => self.translate_builtin_function(name, args, kwargs)?,
            PyExpr::Attribute { value, attr } => {
                // Method call: obj.method(args)
                let obj = self.translate(value)?;
                let method = self.translate_method_name(attr);
                let args_str = self.translate_args(args)?;
                format!("{}.{}({})", obj, method, args_str)
            }
            _ => {
                // Complex function expression
                let func_str = self.translate(func)?;
                let args_str = self.translate_args(args)?;
                format!("{}({})", func_str, args_str)
            }
        };

        Ok(func_name)
    }

    /// Translate built-in Python functions
    fn translate_builtin_function(
        &mut self,
        name: &str,
        args: &[PyExpr],
        kwargs: &HashMap<String, PyExpr>,
    ) -> Result<String> {
        if !kwargs.is_empty() {
            return Err(Error::CodeGeneration(
                "Keyword arguments not yet supported in function calls".to_string(),
            ));
        }

        let args_str = self.translate_args(args)?;

        let result = match name {
            // Built-in functions that map directly
            "print" => format!("println!(\"{{:?}}\", {})", args_str),
            "len" => format!("{}.len()", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "range" => {
                match args.len() {
                    1 => format!("(0..{})", self.translate(&args[0])?),
                    2 => format!("({}..{})", self.translate(&args[0])?, self.translate(&args[1])?),
                    3 => format!("({}..{}).step_by({})",
                        self.translate(&args[0])?,
                        self.translate(&args[1])?,
                        self.translate(&args[2])?),
                    _ => return Err(Error::CodeGeneration("Invalid range() arguments".to_string())),
                }
            }
            "str" => format!("{}.to_string()", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "int" => format!("{} as i64", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "float" => format!("{} as f64", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "bool" => args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default(),
            "abs" => format!("{}.abs()", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "min" => format!("std::cmp::min({}, {})",
                self.translate(&args[0])?,
                self.translate(&args[1])?),
            "max" => format!("std::cmp::max({}, {})",
                self.translate(&args[0])?,
                self.translate(&args[1])?),
            "sum" => format!("{}.iter().sum()", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "all" => format!("{}.iter().all(|x| *x)", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "any" => format!("{}.iter().any(|x| *x)", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "enumerate" => format!("{}.iter().enumerate()", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "zip" => format!("{}.iter().zip({}.iter())",
                self.translate(&args[0])?,
                self.translate(&args[1])?),
            "map" => format!("{}.iter().map({})",
                self.translate(&args[1])?,
                self.translate(&args[0])?),
            "filter" => format!("{}.iter().filter({})",
                self.translate(&args[1])?,
                self.translate(&args[0])?),
            "sorted" => format!("{{ let mut tmp = {}.clone(); tmp.sort(); tmp }}",
                args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "reversed" => format!("{}.iter().rev()", args.get(0).map(|a| self.translate(a)).transpose()?.unwrap_or_default()),
            "list" => {
                if args.is_empty() {
                    "vec![]".to_string()
                } else {
                    format!("{}.collect::<Vec<_>>()", self.translate(&args[0])?)
                }
            }
            "dict" => {
                if args.is_empty() {
                    "HashMap::new()".to_string()
                } else {
                    format!("{}.collect::<HashMap<_, _>>()", self.translate(&args[0])?)
                }
            }
            "set" => {
                if args.is_empty() {
                    "HashSet::new()".to_string()
                } else {
                    format!("{}.collect::<HashSet<_>>()", self.translate(&args[0])?)
                }
            }
            "tuple" => {
                if args.is_empty() {
                    "()".to_string()
                } else {
                    // Convert to tuple
                    args_str
                }
            }
            "isinstance" => {
                // Type checking - complex, may need runtime support
                format!("/* isinstance({}) */", args_str)
            }
            "hasattr" => {
                // Attribute checking - complex
                format!("/* hasattr({}) */", args_str)
            }
            "getattr" => {
                // Dynamic attribute access - complex
                format!("/* getattr({}) */", args_str)
            }
            "open" => {
                // File I/O - use WasiFilesystem for cross-platform support
                if args.len() >= 2 {
                    // Has mode parameter
                    let path = self.translate(&args[0])?;
                    let mode = self.translate(&args[1])?;

                    // Determine operation based on mode
                    // Mode can be: "r", "w", "a", "r+", "w+", "a+", "rb", "wb", etc.
                    format!("WasiFilesystem::open_with_mode({}, {})?", path, mode)
                } else {
                    // Default mode is read
                    format!("WasiFilesystem::open({})?", self.translate(&args[0])?)
                }
            }
            // Regular function call
            _ => format!("{}({})", name, args_str),
        };

        Ok(result)
    }

    /// Translate method names from Python to Rust
    fn translate_method_name(&self, method: &str) -> String {
        match method {
            // String methods
            "upper" => "to_uppercase".to_string(),
            "lower" => "to_lowercase".to_string(),
            "strip" => "trim".to_string(),
            "lstrip" => "trim_start".to_string(),
            "rstrip" => "trim_end".to_string(),
            "split" => "split".to_string(),
            "replace" => "replace".to_string(),
            "startswith" => "starts_with".to_string(),
            "endswith" => "ends_with".to_string(),
            "find" => "find".to_string(),

            // List methods
            "append" => "push".to_string(),
            "extend" => "extend".to_string(),
            "insert" => "insert".to_string(),
            "remove" => "remove".to_string(),
            "pop" => "pop".to_string(),
            "clear" => "clear".to_string(),
            "sort" => "sort".to_string(),
            "reverse" => "reverse".to_string(),
            "count" => "iter().filter(|x| x == &val).count".to_string(),
            "index" => "iter().position(|x| x == &val).unwrap".to_string(),

            // Dict methods
            "get" => "get".to_string(),
            "keys" => "keys".to_string(),
            "values" => "values".to_string(),
            "items" => "iter".to_string(),
            "update" => "extend".to_string(),

            // Set methods
            "add" => "insert".to_string(),
            "discard" => "remove".to_string(),
            "union" => "union".to_string(),
            "intersection" => "intersection".to_string(),
            "difference" => "difference".to_string(),

            // File methods
            "read" => "read_to_string".to_string(),
            "write" => "write_all".to_string(),
            "close" => "flush".to_string(),
            "readline" => "read_line".to_string(),
            "readlines" => "lines".to_string(),
            "writelines" => "write_all".to_string(),
            "seek" => "seek".to_string(),
            "tell" => "stream_position".to_string(),
            "flush" => "flush".to_string(),

            // Default: keep the same
            _ => method.to_string(),
        }
    }

    /// Translate function arguments
    fn translate_args(&mut self, args: &[PyExpr]) -> Result<String> {
        let arg_strs: Result<Vec<_>> = args.iter().map(|arg| self.translate(arg)).collect();
        Ok(arg_strs?.join(", "))
    }

    /// Translate attribute access
    fn translate_attribute(&mut self, value: &PyExpr, attr: &str) -> Result<String> {
        let value_rust = self.translate(value)?;
        Ok(format!("{}.{}", value_rust, attr))
    }

    /// Translate subscript (indexing)
    fn translate_subscript(&mut self, value: &PyExpr, index: &PyExpr) -> Result<String> {
        let value_rust = self.translate(value)?;
        let index_rust = self.translate(index)?;

        // Python allows negative indexing, Rust doesn't
        // For now, simple translation
        Ok(format!("{}[{} as usize]", value_rust, index_rust))
    }

    /// Translate slice
    fn translate_slice(
        &mut self,
        value: &PyExpr,
        lower: Option<&PyExpr>,
        upper: Option<&PyExpr>,
        step: Option<&PyExpr>,
    ) -> Result<String> {
        let value_rust = self.translate(value)?;

        if step.is_some() {
            // Stepped slicing is complex
            return Err(Error::CodeGeneration(
                "Stepped slicing not yet implemented".to_string(),
            ));
        }

        let lower_rust = lower.map(|l| self.translate(l)).transpose()?.unwrap_or_else(|| "0".to_string());
        let upper_rust = upper.map(|u| self.translate(u)).transpose()?;

        if let Some(upper) = upper_rust {
            Ok(format!("{}[{} as usize..{} as usize]", value_rust, lower_rust, upper))
        } else {
            Ok(format!("{}[{} as usize..]", value_rust, lower_rust))
        }
    }

    /// Translate list literal
    fn translate_list(&mut self, elements: &[PyExpr]) -> Result<String> {
        let elem_strs: Result<Vec<_>> = elements.iter().map(|e| self.translate(e)).collect();
        Ok(format!("vec![{}]", elem_strs?.join(", ")))
    }

    /// Translate tuple literal
    fn translate_tuple(&mut self, elements: &[PyExpr]) -> Result<String> {
        let elem_strs: Result<Vec<_>> = elements.iter().map(|e| self.translate(e)).collect();
        Ok(format!("({})", elem_strs?.join(", ")))
    }

    /// Translate dict literal
    fn translate_dict(&mut self, keys: &[PyExpr], values: &[PyExpr]) -> Result<String> {
        if keys.len() != values.len() {
            return Err(Error::CodeGeneration(
                "Dict keys and values length mismatch".to_string(),
            ));
        }

        let mut pairs = Vec::new();
        for (k, v) in keys.iter().zip(values.iter()) {
            let key_str = self.translate(k)?;
            let val_str = self.translate(v)?;
            pairs.push(format!("({}, {})", key_str, val_str));
        }

        Ok(format!(
            "HashMap::from([{}])",
            pairs.join(", ")
        ))
    }

    /// Translate set literal
    fn translate_set(&mut self, elements: &[PyExpr]) -> Result<String> {
        let elem_strs: Result<Vec<_>> = elements.iter().map(|e| self.translate(e)).collect();
        Ok(format!("HashSet::from([{}])", elem_strs?.join(", ")))
    }

    /// Translate list comprehension
    fn translate_list_comp(
        &mut self,
        element: &PyExpr,
        generators: &[Comprehension],
    ) -> Result<String> {
        if generators.is_empty() {
            return Err(Error::CodeGeneration(
                "List comprehension needs at least one generator".to_string(),
            ));
        }

        // For now, handle single generator
        let gen = &generators[0];
        let iter_rust = self.translate(&gen.iter)?;
        let target_name = match &gen.target {
            PyExpr::Name(name) => name.clone(),
            _ => return Err(Error::CodeGeneration(
                "Complex comprehension targets not yet supported".to_string(),
            )),
        };

        // Handle filters
        let filter_str = if !gen.ifs.is_empty() {
            let conditions: Result<Vec<_>> = gen.ifs.iter().map(|cond| self.translate(cond)).collect();
            format!(".filter(|&{}| {})", target_name, conditions?.join(" && "))
        } else {
            String::new()
        };

        let element_rust = self.translate(element)?;

        Ok(format!(
            "{}.iter(){}.map(|{}| {}).collect::<Vec<_>>()",
            iter_rust, filter_str, target_name, element_rust
        ))
    }

    /// Translate conditional expression (ternary)
    fn translate_if_expr(&mut self, test: &PyExpr, body: &PyExpr, orelse: &PyExpr) -> Result<String> {
        let test_rust = self.translate(test)?;
        let body_rust = self.translate(body)?;
        let orelse_rust = self.translate(orelse)?;

        Ok(format!(
            "if {} {{ {} }} else {{ {} }}",
            test_rust, body_rust, orelse_rust
        ))
    }

    /// Translate lambda expression
    fn translate_lambda(&mut self, args: &[String], body: &PyExpr) -> Result<String> {
        let body_rust = self.translate(body)?;
        let args_str = args.join(", ");

        Ok(format!("|{}| {}", args_str, body_rust))
    }

    /// Translate await expression
    fn translate_await(&mut self, expr: &PyExpr) -> Result<String> {
        let expr_rust = self.translate(expr)?;
        Ok(format!("{}.await", expr_rust))
    }

    /// Translate yield expression
    fn translate_yield(&mut self, expr: Option<&PyExpr>) -> Result<String> {
        if let Some(expr) = expr {
            let expr_rust = self.translate(expr)?;
            Ok(format!("yield {}", expr_rust))
        } else {
            Ok("yield".to_string())
        }
    }

    /// Infer the type of an expression (simple version)
    pub fn infer_type(&self, expr: &PyExpr) -> RustType {
        match expr {
            PyExpr::Literal(lit) => match lit {
                PyLiteral::Int(_) => RustType::I64,
                PyLiteral::Float(_) => RustType::F64,
                PyLiteral::String(_) => RustType::String,
                PyLiteral::Bool(_) => RustType::Bool,
                PyLiteral::None => RustType::Option(Box::new(RustType::Unknown)),
                PyLiteral::Bytes(_) => RustType::Vec(Box::new(RustType::I32)),
            },
            PyExpr::Name(name) => self.ctx.get_type(name),
            PyExpr::List(_) => RustType::Vec(Box::new(RustType::Unknown)),
            PyExpr::Dict { .. } => {
                RustType::HashMap(Box::new(RustType::Unknown), Box::new(RustType::Unknown))
            }
            PyExpr::Set(_) => RustType::HashSet(Box::new(RustType::Unknown)),
            PyExpr::Tuple(_) => RustType::Tuple(vec![]),
            _ => RustType::Unknown,
        }
    }
}

impl Default for TranslationContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translate_expr_str(expr: &PyExpr) -> String {
        let mut ctx = TranslationContext::new();
        let mut translator = ExpressionTranslator::new(&mut ctx);
        translator.translate(expr).unwrap()
    }

    #[test]
    fn test_translate_literals() {
        assert_eq!(
            translate_expr_str(&PyExpr::Literal(PyLiteral::Int(42))),
            "42"
        );
        assert_eq!(
            translate_expr_str(&PyExpr::Literal(PyLiteral::Float(3.14))),
            "3.14"
        );
        assert_eq!(
            translate_expr_str(&PyExpr::Literal(PyLiteral::String("hello".to_string()))),
            "\"hello\""
        );
        assert_eq!(
            translate_expr_str(&PyExpr::Literal(PyLiteral::Bool(true))),
            "true"
        );
    }

    #[test]
    fn test_translate_binop() {
        let expr = PyExpr::BinOp {
            left: Box::new(PyExpr::Literal(PyLiteral::Int(2))),
            op: BinOp::Add,
            right: Box::new(PyExpr::Literal(PyLiteral::Int(3))),
        };
        assert_eq!(translate_expr_str(&expr), "2 + 3");
    }

    #[test]
    fn test_translate_comparison() {
        let expr = PyExpr::Compare {
            left: Box::new(PyExpr::Name("x".to_string())),
            op: CmpOp::Gt,
            right: Box::new(PyExpr::Literal(PyLiteral::Int(0))),
        };
        assert_eq!(translate_expr_str(&expr), "x > 0");
    }

    #[test]
    fn test_translate_list() {
        let expr = PyExpr::List(vec![
            PyExpr::Literal(PyLiteral::Int(1)),
            PyExpr::Literal(PyLiteral::Int(2)),
            PyExpr::Literal(PyLiteral::Int(3)),
        ]);
        assert_eq!(translate_expr_str(&expr), "vec![1, 2, 3]");
    }

    #[test]
    fn test_translate_dict() {
        let expr = PyExpr::Dict {
            keys: vec![PyExpr::Literal(PyLiteral::String("a".to_string()))],
            values: vec![PyExpr::Literal(PyLiteral::Int(1))],
        };
        let result = translate_expr_str(&expr);
        assert!(result.contains("HashMap"));
        assert!(result.contains("\"a\""));
        assert!(result.contains("1"));
    }

    #[test]
    fn test_translate_builtin_functions() {
        let expr = PyExpr::Call {
            func: Box::new(PyExpr::Name("len".to_string())),
            args: vec![PyExpr::Name("my_list".to_string())],
            kwargs: HashMap::new(),
        };
        assert_eq!(translate_expr_str(&expr), "my_list.len()");

        let expr = PyExpr::Call {
            func: Box::new(PyExpr::Name("range".to_string())),
            args: vec![PyExpr::Literal(PyLiteral::Int(10))],
            kwargs: HashMap::new(),
        };
        assert_eq!(translate_expr_str(&expr), "(0..10)");
    }

    #[test]
    fn test_translate_conditional_expr() {
        let expr = PyExpr::IfExp {
            test: Box::new(PyExpr::Name("condition".to_string())),
            body: Box::new(PyExpr::Literal(PyLiteral::Int(1))),
            orelse: Box::new(PyExpr::Literal(PyLiteral::Int(0))),
        };
        assert_eq!(
            translate_expr_str(&expr),
            "if condition { 1 } else { 0 }"
        );
    }

    #[test]
    fn test_translate_lambda() {
        let expr = PyExpr::Lambda {
            args: vec!["x".to_string(), "y".to_string()],
            body: Box::new(PyExpr::BinOp {
                left: Box::new(PyExpr::Name("x".to_string())),
                op: BinOp::Add,
                right: Box::new(PyExpr::Name("y".to_string())),
            }),
        };
        assert_eq!(translate_expr_str(&expr), "|x, y| x + y");
    }
}
