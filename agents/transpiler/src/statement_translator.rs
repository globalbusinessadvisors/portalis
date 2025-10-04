//! Generic Statement Translator - Python statements to Rust
//!
//! This module provides comprehensive translation of Python statements to idiomatic Rust.
//! Handles all statement types: assignments, control flow, functions, classes, etc.
//!
//! Key features:
//! - Complete statement coverage (if, for, while, try, with, etc.)
//! - Proper Rust idioms (Result for exceptions, iterators for loops, etc.)
//! - Type inference and propagation
//! - Lifetime management assistance

use crate::expression_translator::{ExpressionTranslator, TranslationContext, RustType};
use crate::python_ast::*;
use crate::{Error, Result};

/// Generic statement translator
pub struct StatementTranslator<'a> {
    ctx: &'a mut TranslationContext,
    expr_translator: ExpressionTranslator<'a>,
}

impl<'a> StatementTranslator<'a> {
    pub fn new(ctx: &'a mut TranslationContext) -> Self {
        let expr_translator = ExpressionTranslator::new(ctx);
        Self {
            ctx,
            expr_translator,
        }
    }

    /// Translate any Python statement to Rust
    pub fn translate(&mut self, stmt: &PyStmt) -> Result<String> {
        match stmt {
            PyStmt::Expr(expr) => self.translate_expr_stmt(expr),
            PyStmt::Assign { target, value } => self.translate_assign(target, value),
            PyStmt::AugAssign { target, op, value } => {
                self.translate_aug_assign(target, op, value)
            }
            PyStmt::AnnAssign {
                target,
                annotation,
                value,
            } => self.translate_ann_assign(target, annotation, value.as_ref()),
            PyStmt::FunctionDef {
                name,
                params,
                body,
                return_type,
                decorators,
                is_async,
            } => self.translate_function_def(name, params, body, return_type.as_ref(), decorators, *is_async),
            PyStmt::Return { value } => self.translate_return(value.as_ref()),
            PyStmt::If { test, body, orelse } => self.translate_if(test, body, orelse),
            PyStmt::While { test, body, orelse } => self.translate_while(test, body, orelse),
            PyStmt::For {
                target,
                iter,
                body,
                orelse,
            } => self.translate_for(target, iter, body, orelse),
            PyStmt::Pass => Ok(format!("{}// pass\n", self.ctx.indent())),
            PyStmt::Break => Ok(format!("{}break;\n", self.ctx.indent())),
            PyStmt::Continue => Ok(format!("{}continue;\n", self.ctx.indent())),
            PyStmt::ClassDef {
                name,
                bases,
                body,
                decorators,
            } => self.translate_class_def(name, bases, body, decorators),
            PyStmt::Import { modules } => self.translate_import(modules),
            PyStmt::ImportFrom {
                module,
                names,
                level,
            } => self.translate_import_from(module.as_ref(), names, *level),
            PyStmt::Assert { test, msg } => self.translate_assert(test, msg.as_ref()),
            PyStmt::Try {
                body,
                handlers,
                orelse,
                finalbody,
            } => self.translate_try(body, handlers, orelse, finalbody),
            PyStmt::Raise { exception } => self.translate_raise(exception.as_ref()),
            PyStmt::With { items, body } => self.translate_with(items, body),
            PyStmt::Delete { targets } => self.translate_delete(targets),
            PyStmt::Global { names } => self.translate_global(names),
            PyStmt::Nonlocal { names } => self.translate_nonlocal(names),
        }
    }

    /// Translate multiple statements
    pub fn translate_block(&mut self, stmts: &[PyStmt]) -> Result<String> {
        let mut code = String::new();
        for stmt in stmts {
            code.push_str(&self.translate(stmt)?);
        }
        Ok(code)
    }

    /// Translate expression statement
    fn translate_expr_stmt(&mut self, expr: &PyExpr) -> Result<String> {
        let expr_rust = self.expr_translator.translate(expr)?;
        Ok(format!("{}{};\n", self.ctx.indent(), expr_rust))
    }

    /// Translate assignment
    fn translate_assign(&mut self, target: &PyExpr, value: &PyExpr) -> Result<String> {
        let target_name = match target {
            PyExpr::Name(name) => name.clone(),
            _ => {
                return Err(Error::CodeGeneration(
                    "Complex assignment targets not yet supported".to_string(),
                ))
            }
        };

        let value_rust = self.expr_translator.translate(value)?;

        // Infer type from value
        let inferred_type = self.expr_translator.infer_type(value);
        self.ctx.set_type(target_name.clone(), inferred_type);

        Ok(format!(
            "{}let mut {} = {};\n",
            self.ctx.indent(),
            target_name,
            value_rust
        ))
    }

    /// Translate augmented assignment
    fn translate_aug_assign(&mut self, target: &PyExpr, op: &BinOp, value: &PyExpr) -> Result<String> {
        let target_rust = self.expr_translator.translate(target)?;
        let value_rust = self.expr_translator.translate(value)?;

        let op_str = match op {
            BinOp::Add => "+=",
            BinOp::Sub => "-=",
            BinOp::Mult => "*=",
            BinOp::Div => "/=",
            BinOp::Mod => "%=",
            BinOp::BitAnd => "&=",
            BinOp::BitOr => "|=",
            BinOp::BitXor => "^=",
            BinOp::LShift => "<<=",
            BinOp::RShift => ">>=",
            _ => {
                return Err(Error::CodeGeneration(format!(
                    "Augmented assignment not supported for operator {:?}",
                    op
                )))
            }
        };

        Ok(format!(
            "{}{} {} {};\n",
            self.ctx.indent(),
            target_rust,
            op_str,
            value_rust
        ))
    }

    /// Translate annotated assignment
    fn translate_ann_assign(
        &mut self,
        target: &PyExpr,
        annotation: &TypeAnnotation,
        value: Option<&PyExpr>,
    ) -> Result<String> {
        let target_name = match target {
            PyExpr::Name(name) => name.clone(),
            _ => {
                return Err(Error::CodeGeneration(
                    "Complex assignment targets not yet supported".to_string(),
                ))
            }
        };

        let type_str = self.translate_type_annotation(annotation)?;

        if let Some(value) = value {
            let value_rust = self.expr_translator.translate(value)?;
            Ok(format!(
                "{}let mut {}: {} = {};\n",
                self.ctx.indent(),
                target_name,
                type_str,
                value_rust
            ))
        } else {
            // Declaration without initialization (not common in Rust)
            Ok(format!(
                "{}// let {}: {}; (uninitialized)\n",
                self.ctx.indent(),
                target_name,
                type_str
            ))
        }
    }

    /// Translate function definition
    fn translate_function_def(
        &mut self,
        name: &str,
        params: &[FunctionParam],
        body: &[PyStmt],
        return_type: Option<&TypeAnnotation>,
        decorators: &[PyExpr],
        is_async: bool,
    ) -> Result<String> {
        let mut code = String::new();

        // Handle decorators (as comments for now)
        for decorator in decorators {
            let dec_str = self.expr_translator.translate(decorator)?;
            code.push_str(&format!("{}// @{}\n", self.ctx.indent(), dec_str));
        }

        // Function signature
        let async_str = if is_async { "async " } else { "" };
        let params_str = self.translate_params(params)?;
        let return_str = if let Some(ret_type) = return_type {
            format!(" -> {}", self.translate_type_annotation(ret_type)?)
        } else {
            String::new()
        };

        code.push_str(&format!(
            "{}pub {}fn {}({}){} {{\n",
            self.ctx.indent(),
            async_str,
            name,
            params_str,
            return_str
        ));

        // Function body
        self.ctx.indent_level += 1;
        code.push_str(&self.translate_block(body)?);
        self.ctx.indent_level -= 1;

        code.push_str(&format!("{}}}\n", self.ctx.indent()));

        Ok(code)
    }

    /// Translate function parameters
    fn translate_params(&self, params: &[FunctionParam]) -> Result<String> {
        let param_strs: Result<Vec<_>> = params
            .iter()
            .map(|param| {
                let type_str = if let Some(annotation) = &param.type_annotation {
                    self.translate_type_annotation(annotation)?
                } else {
                    "i64".to_string() // Default type
                };

                Ok(format!("{}: {}", param.name, type_str))
            })
            .collect();

        Ok(param_strs?.join(", "))
    }

    /// Translate type annotation
    fn translate_type_annotation(&self, annotation: &TypeAnnotation) -> Result<String> {
        match annotation {
            TypeAnnotation::Name(name) => Ok(self.map_type_name(name)),
            TypeAnnotation::Generic { base, args } => {
                let base_str = self.translate_type_annotation(base)?;
                let args_strs: Result<Vec<_>> =
                    args.iter().map(|arg| self.translate_type_annotation(arg)).collect();
                Ok(format!("{}<{}>", base_str, args_strs?.join(", ")))
            }
        }
    }

    /// Map Python type names to Rust type names
    fn map_type_name(&self, py_type: &str) -> String {
        match py_type {
            "int" => "i64".to_string(),
            "float" => "f64".to_string(),
            "str" => "String".to_string(),
            "bool" => "bool".to_string(),
            "bytes" => "Vec<u8>".to_string(),
            "list" | "List" => "Vec".to_string(),
            "dict" | "Dict" => "HashMap".to_string(),
            "set" | "Set" => "HashSet".to_string(),
            "tuple" | "Tuple" => "tuple".to_string(),
            "Optional" | "Option" => "Option".to_string(),
            "Any" => "Box<dyn std::any::Any>".to_string(),
            _ => py_type.to_string(),
        }
    }

    /// Translate return statement
    fn translate_return(&mut self, value: Option<&PyExpr>) -> Result<String> {
        if let Some(expr) = value {
            let expr_rust = self.expr_translator.translate(expr)?;
            Ok(format!("{}return {};\n", self.ctx.indent(), expr_rust))
        } else {
            Ok(format!("{}return;\n", self.ctx.indent()))
        }
    }

    /// Translate if statement
    fn translate_if(&mut self, test: &PyExpr, body: &[PyStmt], orelse: &[PyStmt]) -> Result<String> {
        let mut code = String::new();

        let test_rust = self.expr_translator.translate(test)?;
        code.push_str(&format!("{}if {} {{\n", self.ctx.indent(), test_rust));

        self.ctx.indent_level += 1;
        code.push_str(&self.translate_block(body)?);
        self.ctx.indent_level -= 1;

        if !orelse.is_empty() {
            // Check if orelse is another if statement (elif)
            if orelse.len() == 1 {
                if let PyStmt::If { test, body, orelse: nested_else } = &orelse[0] {
                    // This is an elif
                    code.push_str(&format!("{}}} else ", self.ctx.indent()));
                    let elif_test = self.expr_translator.translate(test)?;
                    code.push_str(&format!("if {} {{\n", elif_test));

                    self.ctx.indent_level += 1;
                    code.push_str(&self.translate_block(body)?);
                    self.ctx.indent_level -= 1;

                    if !nested_else.is_empty() {
                        code.push_str(&format!("{}}} else {{\n", self.ctx.indent()));
                        self.ctx.indent_level += 1;
                        code.push_str(&self.translate_block(nested_else)?);
                        self.ctx.indent_level -= 1;
                    }

                    code.push_str(&format!("{}}}\n", self.ctx.indent()));
                    return Ok(code);
                }
            }

            // Regular else block
            code.push_str(&format!("{}}} else {{\n", self.ctx.indent()));
            self.ctx.indent_level += 1;
            code.push_str(&self.translate_block(orelse)?);
            self.ctx.indent_level -= 1;
        }

        code.push_str(&format!("{}}}\n", self.ctx.indent()));

        Ok(code)
    }

    /// Translate while loop
    fn translate_while(&mut self, test: &PyExpr, body: &[PyStmt], orelse: &[PyStmt]) -> Result<String> {
        let mut code = String::new();

        let test_rust = self.expr_translator.translate(test)?;
        code.push_str(&format!("{}while {} {{\n", self.ctx.indent(), test_rust));

        self.ctx.indent_level += 1;
        code.push_str(&self.translate_block(body)?);
        self.ctx.indent_level -= 1;

        code.push_str(&format!("{}}}\n", self.ctx.indent()));

        // Python's while-else is rare and complex in Rust
        if !orelse.is_empty() {
            code.push_str(&format!("{}// while-else block:\n", self.ctx.indent()));
            code.push_str(&self.translate_block(orelse)?);
        }

        Ok(code)
    }

    /// Translate for loop
    fn translate_for(
        &mut self,
        target: &PyExpr,
        iter: &PyExpr,
        body: &[PyStmt],
        orelse: &[PyStmt],
    ) -> Result<String> {
        let mut code = String::new();

        let target_name = match target {
            PyExpr::Name(name) => name.clone(),
            _ => {
                return Err(Error::CodeGeneration(
                    "Complex for loop targets not yet supported".to_string(),
                ))
            }
        };

        let iter_rust = self.expr_translator.translate(iter)?;

        code.push_str(&format!(
            "{}for {} in {} {{\n",
            self.ctx.indent(),
            target_name,
            iter_rust
        ));

        self.ctx.indent_level += 1;
        code.push_str(&self.translate_block(body)?);
        self.ctx.indent_level -= 1;

        code.push_str(&format!("{}}}\n", self.ctx.indent()));

        // Python's for-else is rare and complex in Rust
        if !orelse.is_empty() {
            code.push_str(&format!("{}// for-else block:\n", self.ctx.indent()));
            code.push_str(&self.translate_block(orelse)?);
        }

        Ok(code)
    }

    /// Translate class definition
    fn translate_class_def(
        &mut self,
        name: &str,
        bases: &[PyExpr],
        body: &[PyStmt],
        decorators: &[PyExpr],
    ) -> Result<String> {
        let mut code = String::new();

        // Decorators as comments
        for decorator in decorators {
            let dec_str = self.expr_translator.translate(decorator)?;
            code.push_str(&format!("{}// @{}\n", self.ctx.indent(), dec_str));
        }

        // Base classes as comment
        if !bases.is_empty() {
            let bases_strs: Result<Vec<_>> =
                bases.iter().map(|b| self.expr_translator.translate(b)).collect();
            code.push_str(&format!(
                "{}// Inherits from: {}\n",
                self.ctx.indent(),
                bases_strs?.join(", ")
            ));
        }

        code.push_str(&format!("{}pub struct {} {{\n", self.ctx.indent(), name));

        // Extract fields from __init__ if present
        self.ctx.indent_level += 1;

        // For now, simple struct with fields from body
        code.push_str(&format!("{}// TODO: Add fields\n", self.ctx.indent()));

        self.ctx.indent_level -= 1;
        code.push_str(&format!("{}}}\n\n", self.ctx.indent()));

        // Implementation block for methods
        code.push_str(&format!("{}impl {} {{\n", self.ctx.indent(), name));

        self.ctx.indent_level += 1;
        code.push_str(&self.translate_block(body)?);
        self.ctx.indent_level -= 1;

        code.push_str(&format!("{}}}\n", self.ctx.indent()));

        Ok(code)
    }

    /// Translate import statement
    fn translate_import(&self, modules: &[(String, Option<String>)]) -> Result<String> {
        let mut code = String::new();

        for (module, alias) in modules {
            let module_path = module.replace('.', "::");
            if let Some(alias_name) = alias {
                code.push_str(&format!(
                    "{}use {} as {};\n",
                    self.ctx.indent(),
                    module_path,
                    alias_name
                ));
            } else {
                code.push_str(&format!("{}use {};\n", self.ctx.indent(), module_path));
            }
        }

        Ok(code)
    }

    /// Translate from import statement
    fn translate_import_from(
        &self,
        module: Option<&String>,
        names: &[(String, Option<String>)],
        level: usize,
    ) -> Result<String> {
        let mut code = String::new();

        let module_path = if let Some(m) = module {
            m.replace('.', "::")
        } else {
            "super::".to_string()
        };

        for (name, alias) in names {
            if let Some(alias_name) = alias {
                code.push_str(&format!(
                    "{}use {}::{} as {};\n",
                    self.ctx.indent(),
                    module_path,
                    name,
                    alias_name
                ));
            } else {
                code.push_str(&format!(
                    "{}use {}::{};\n",
                    self.ctx.indent(),
                    module_path,
                    name
                ));
            }
        }

        Ok(code)
    }

    /// Translate assert statement
    fn translate_assert(&mut self, test: &PyExpr, msg: Option<&PyExpr>) -> Result<String> {
        let test_rust = self.expr_translator.translate(test)?;

        if let Some(msg_expr) = msg {
            let msg_rust = self.expr_translator.translate(msg_expr)?;
            Ok(format!(
                "{}assert!({}, {});\n",
                self.ctx.indent(),
                test_rust,
                msg_rust
            ))
        } else {
            Ok(format!("{}assert!({});\n", self.ctx.indent(), test_rust))
        }
    }

    /// Translate try-except statement
    fn translate_try(
        &mut self,
        body: &[PyStmt],
        handlers: &[ExceptHandler],
        orelse: &[PyStmt],
        finalbody: &[PyStmt],
    ) -> Result<String> {
        let mut code = String::new();

        // Rust doesn't have try-except, use Result<T, E> pattern
        code.push_str(&format!("{}// Try block (converted to Result pattern)\n", self.ctx.indent()));
        code.push_str(&format!("{}match (|| -> Result<(), Box<dyn std::error::Error>> {{\n", self.ctx.indent()));

        self.ctx.indent_level += 1;
        code.push_str(&self.translate_block(body)?);
        code.push_str(&format!("{}Ok(())\n", self.ctx.indent()));
        self.ctx.indent_level -= 1;

        code.push_str(&format!("{}}})() {{\n", self.ctx.indent()));

        self.ctx.indent_level += 1;

        // Exception handlers
        if !handlers.is_empty() {
            code.push_str(&format!("{}Err(e) => {{\n", self.ctx.indent()));
            self.ctx.indent_level += 1;

            for handler in handlers {
                if let Some(exc_type) = &handler.exception_type {
                    let exc_type_str = self.expr_translator.translate(exc_type)?;
                    code.push_str(&format!("{}// Handle {} exception\n", self.ctx.indent(), exc_type_str));
                }
                code.push_str(&self.translate_block(&handler.body)?);
            }

            self.ctx.indent_level -= 1;
            code.push_str(&format!("{}}},\n", self.ctx.indent()));
        }

        // Else block (executed if no exception)
        if !orelse.is_empty() {
            code.push_str(&format!("{}Ok(_) => {{\n", self.ctx.indent()));
            self.ctx.indent_level += 1;
            code.push_str(&self.translate_block(orelse)?);
            self.ctx.indent_level -= 1;
            code.push_str(&format!("{}}}\n", self.ctx.indent()));
        } else {
            code.push_str(&format!("{}Ok(_) => {{}}\n", self.ctx.indent()));
        }

        self.ctx.indent_level -= 1;
        code.push_str(&format!("{}}}\n", self.ctx.indent()));

        // Finally block
        if !finalbody.is_empty() {
            code.push_str(&format!("{}// Finally block\n", self.ctx.indent()));
            code.push_str(&self.translate_block(finalbody)?);
        }

        Ok(code)
    }

    /// Translate raise statement
    fn translate_raise(&mut self, exception: Option<&PyExpr>) -> Result<String> {
        if let Some(exc) = exception {
            let exc_rust = self.expr_translator.translate(exc)?;
            Ok(format!(
                "{}return Err(Box::new({}));\n",
                self.ctx.indent(),
                exc_rust
            ))
        } else {
            Ok(format!("{}return Err(Box::new(/* re-raise */));\n", self.ctx.indent()))
        }
    }

    /// Translate with statement (context manager)
    fn translate_with(&mut self, items: &[WithItem], body: &[PyStmt]) -> Result<String> {
        let mut code = String::new();

        // Rust doesn't have context managers, use RAII pattern
        for item in items {
            let ctx_expr = self.expr_translator.translate(&item.context_expr)?;

            if let Some(var) = &item.optional_vars {
                let var_name = match var {
                    PyExpr::Name(name) => name,
                    _ => return Err(Error::CodeGeneration("Complex with targets not supported".to_string())),
                };
                code.push_str(&format!("{}let {} = {};\n", self.ctx.indent(), var_name, ctx_expr));
            } else {
                code.push_str(&format!("{}let _ctx = {};\n", self.ctx.indent(), ctx_expr));
            }
        }

        // With body
        code.push_str(&format!("{}{{\n", self.ctx.indent()));
        self.ctx.indent_level += 1;
        code.push_str(&self.translate_block(body)?);
        self.ctx.indent_level -= 1;
        code.push_str(&format!("{}}}\n", self.ctx.indent()));

        Ok(code)
    }

    /// Translate delete statement
    fn translate_delete(&mut self, targets: &[PyExpr]) -> Result<String> {
        let mut code = String::new();

        for target in targets {
            let target_rust = self.expr_translator.translate(target)?;
            code.push_str(&format!(
                "{}// del {} (Rust uses drop/RAII)\n",
                self.ctx.indent(),
                target_rust
            ));
        }

        Ok(code)
    }

    /// Translate global declaration
    fn translate_global(&self, names: &[String]) -> Result<String> {
        Ok(format!(
            "{}// global {}\n",
            self.ctx.indent(),
            names.join(", ")
        ))
    }

    /// Translate nonlocal declaration
    fn translate_nonlocal(&self, names: &[String]) -> Result<String> {
        Ok(format!(
            "{}// nonlocal {}\n",
            self.ctx.indent(),
            names.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translate_stmt_str(stmt: &PyStmt) -> String {
        let mut ctx = TranslationContext::new();
        let mut translator = StatementTranslator::new(&mut ctx);
        translator.translate(stmt).unwrap()
    }

    #[test]
    fn test_translate_assignment() {
        let stmt = PyStmt::Assign {
            target: PyExpr::Name("x".to_string()),
            value: PyExpr::Literal(PyLiteral::Int(42)),
        };
        let result = translate_stmt_str(&stmt);
        assert!(result.contains("let mut x = 42"));
    }

    #[test]
    fn test_translate_if_statement() {
        let stmt = PyStmt::If {
            test: PyExpr::Name("condition".to_string()),
            body: vec![PyStmt::Return {
                value: Some(PyExpr::Literal(PyLiteral::Int(1))),
            }],
            orelse: vec![],
        };
        let result = translate_stmt_str(&stmt);
        assert!(result.contains("if condition"));
        assert!(result.contains("return 1"));
    }

    #[test]
    fn test_translate_for_loop() {
        let stmt = PyStmt::For {
            target: PyExpr::Name("i".to_string()),
            iter: PyExpr::Call {
                func: Box::new(PyExpr::Name("range".to_string())),
                args: vec![PyExpr::Literal(PyLiteral::Int(10))],
                kwargs: std::collections::HashMap::new(),
            },
            body: vec![PyStmt::Expr(PyExpr::Name("i".to_string()))],
            orelse: vec![],
        };
        let result = translate_stmt_str(&stmt);
        assert!(result.contains("for i in"));
        assert!(result.contains("0..10"));
    }

    #[test]
    fn test_translate_function_def() {
        let stmt = PyStmt::FunctionDef {
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
                }),
            }],
            return_type: Some(TypeAnnotation::Name("int".to_string())),
            decorators: vec![],
            is_async: false,
        };
        let result = translate_stmt_str(&stmt);
        assert!(result.contains("pub fn add"));
        assert!(result.contains("a: i64"));
        assert!(result.contains("b: i64"));
        assert!(result.contains("-> i64"));
        assert!(result.contains("return a + b"));
    }
}
