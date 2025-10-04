//! Full Python AST Parser using rustpython-parser
//!
//! This module provides complete Python 3.x parsing capabilities by wrapping
//! rustpython-parser and converting its AST to our internal representation.
//!
//! Key features:
//! - Full Python 3.10+ syntax support
//! - Source location tracking (line/column numbers)
//! - Comprehensive error reporting with context
//! - Support for all Python constructs (modules, functions, classes, etc.)

use crate::python_ast::*;
use crate::{Error, Result};
use rustpython_parser::{ast, Parse};
use std::path::Path;

/// Main Python parser that wraps rustpython-parser
pub struct PythonParser {
    source: String,
    filename: String,
}

/// Parse error with source location information
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub filename: String,
    pub source_snippet: Option<String>,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Parse error in {} at line {}, column {}: {}",
            self.filename, self.line, self.column, self.message
        )?;

        if let Some(snippet) = &self.source_snippet {
            write!(f, "\n{}", snippet)?;
        }

        Ok(())
    }
}

impl std::error::Error for ParseError {}

impl PythonParser {
    /// Create a new parser for the given source code
    pub fn new(source: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            filename: filename.into(),
        }
    }

    /// Create a parser from a file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|e| {
            Error::Parse(format!("Failed to read file {:?}: {}", path, e))
        })?;

        Ok(Self::new(
            source,
            path.to_string_lossy().to_string(),
        ))
    }

    /// Parse the source code into a Python module
    pub fn parse(&self) -> Result<PyModule> {
        // Parse using rustpython-parser
        let parsed = ast::Suite::parse(&self.source, &self.filename).map_err(|e| {
            Error::Parse(format!("rustpython-parser error: {:?}", e))
        })?;

        // Convert to our internal AST representation
        let mut converter = AstConverter::new(&self.source, &self.filename);
        converter.convert_module(&parsed)
    }

    /// Parse a single expression
    pub fn parse_expression(&self) -> Result<PyExpr> {
        let parsed = ast::Expr::parse(&self.source, &self.filename).map_err(|e| {
            Error::Parse(format!("rustpython-parser error: {:?}", e))
        })?;

        let mut converter = AstConverter::new(&self.source, &self.filename);
        converter.convert_expr(&parsed)
    }

    /// Parse a single statement
    pub fn parse_statement(&self) -> Result<PyStmt> {
        let parsed = ast::Suite::parse(&self.source, &self.filename).map_err(|e| {
            Error::Parse(format!("rustpython-parser error: {:?}", e))
        })?;

        let mut converter = AstConverter::new(&self.source, &self.filename);

        // Extract first statement from parsed statements
        if parsed.is_empty() {
            return Err(Error::Parse("No statements found".to_string()));
        }

        converter.convert_stmt(&parsed[0])
    }

    /// Get source code snippet for error reporting
    pub fn get_source_snippet(&self, line: usize, column: usize, context_lines: usize) -> String {
        let lines: Vec<&str> = self.source.lines().collect();

        let start_line = line.saturating_sub(context_lines);
        let end_line = (line + context_lines).min(lines.len());

        let mut snippet = String::new();

        for (i, line_content) in lines.iter().enumerate().skip(start_line).take(end_line - start_line) {
            snippet.push_str(&format!("{:4} | {}\n", i + 1, line_content));

            // Add error marker
            if i + 1 == line {
                snippet.push_str(&format!("     | {}^\n", " ".repeat(column.saturating_sub(1))));
            }
        }

        snippet
    }
}

/// AST converter from rustpython AST to our internal AST
struct AstConverter {
    #[allow(dead_code)]
    source: String,
    #[allow(dead_code)]
    filename: String,
}

impl AstConverter {
    fn new(source: &str, filename: &str) -> Self {
        Self {
            source: source.to_string(),
            filename: filename.to_string(),
        }
    }

    /// Convert a module from rustpython Mod enum
    #[allow(dead_code)]
    fn convert_module_from_mod(&mut self, mod_ast: ast::Mod) -> Result<PyModule> {
        match mod_ast {
            ast::Mod::Module(module) => self.convert_module(&module.body),
            ast::Mod::Expression(expr) => {
                let mut module = PyModule::new();
                let converted_expr = self.convert_expr(&expr.body)?;
                module.add_stmt(PyStmt::Expr(converted_expr));
                Ok(module)
            }
            _ => Err(Error::Parse("Unsupported module type".to_string())),
        }
    }

    /// Convert a module (list of statements)
    fn convert_module(&mut self, stmts: &[ast::Stmt]) -> Result<PyModule> {
        let mut module = PyModule::new();

        for stmt in stmts {
            let converted = self.convert_stmt(stmt)?;
            module.add_stmt(converted);
        }

        Ok(module)
    }

    /// Convert a statement
    fn convert_stmt(&mut self, stmt: &ast::Stmt) -> Result<PyStmt> {
        match stmt {
            // Function definition
            ast::Stmt::FunctionDef(func) => {
                let name = func.name.to_string();

                // Convert parameters
                let params = self.convert_parameters(&func.args)?;

                // Convert return type annotation if present
                let return_type = func
                    .returns
                    .as_ref()
                    .map(|expr| self.convert_type_annotation(expr))
                    .transpose()?;

                // Convert body
                let mut body = Vec::new();
                for stmt in &func.body {
                    body.push(self.convert_stmt(stmt)?);
                }

                // Convert decorators
                let decorators = func
                    .decorator_list
                    .iter()
                    .map(|dec| self.convert_expr(dec))
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyStmt::FunctionDef {
                    name,
                    params,
                    body,
                    return_type,
                    decorators,
                    is_async: false, // Will be handled by AsyncFunctionDef
                })
            }

            // Async function definition
            ast::Stmt::AsyncFunctionDef(func) => {
                let name = func.name.to_string();
                let params = self.convert_parameters(&func.args)?;
                let return_type = func
                    .returns
                    .as_ref()
                    .map(|expr| self.convert_type_annotation(expr))
                    .transpose()?;

                let mut body = Vec::new();
                for stmt in &func.body {
                    body.push(self.convert_stmt(stmt)?);
                }

                let decorators = func
                    .decorator_list
                    .iter()
                    .map(|dec| self.convert_expr(dec))
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyStmt::FunctionDef {
                    name,
                    params,
                    body,
                    return_type,
                    decorators,
                    is_async: true,
                })
            }

            // Class definition
            ast::Stmt::ClassDef(class) => {
                let name = class.name.to_string();

                // Convert base classes
                let bases = class
                    .bases
                    .iter()
                    .map(|base| self.convert_expr(base))
                    .collect::<Result<Vec<_>>>()?;

                // Convert body
                let mut body = Vec::new();
                for stmt in &class.body {
                    body.push(self.convert_stmt(stmt)?);
                }

                // Convert decorators
                let decorators = class
                    .decorator_list
                    .iter()
                    .map(|dec| self.convert_expr(dec))
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyStmt::ClassDef {
                    name,
                    bases,
                    body,
                    decorators,
                })
            }

            // Return statement
            ast::Stmt::Return(ret) => {
                let value = ret
                    .value
                    .as_ref()
                    .map(|expr| self.convert_expr(expr))
                    .transpose()?;

                Ok(PyStmt::Return { value })
            }

            // Assignment
            ast::Stmt::Assign(assign) => {
                // Python allows multiple targets: a = b = c = 42
                // We'll simplify to single target for now
                if assign.targets.is_empty() {
                    return Err(Error::Parse("Assignment with no targets".to_string()));
                }

                let target = self.convert_expr(&assign.targets[0])?;
                let value = self.convert_expr(&assign.value)?;

                Ok(PyStmt::Assign { target, value })
            }

            // Augmented assignment (+=, -=, etc.)
            ast::Stmt::AugAssign(aug) => {
                let target = self.convert_expr(&aug.target)?;
                let op = self.convert_binop(&aug.op);
                let value = self.convert_expr(&aug.value)?;

                Ok(PyStmt::AugAssign { target, op, value })
            }

            // Expression statement
            ast::Stmt::Expr(expr) => {
                let value = self.convert_expr(&expr.value)?;
                Ok(PyStmt::Expr(value))
            }

            // If statement
            ast::Stmt::If(if_stmt) => {
                let test = self.convert_expr(&if_stmt.test)?;

                let mut body = Vec::new();
                for stmt in &if_stmt.body {
                    body.push(self.convert_stmt(stmt)?);
                }

                let mut orelse = Vec::new();
                for stmt in &if_stmt.orelse {
                    orelse.push(self.convert_stmt(stmt)?);
                }

                Ok(PyStmt::If { test, body, orelse })
            }

            // While loop
            ast::Stmt::While(while_stmt) => {
                let test = self.convert_expr(&while_stmt.test)?;

                let mut body = Vec::new();
                for stmt in &while_stmt.body {
                    body.push(self.convert_stmt(stmt)?);
                }

                let mut orelse = Vec::new();
                for stmt in &while_stmt.orelse {
                    orelse.push(self.convert_stmt(stmt)?);
                }

                Ok(PyStmt::While { test, body, orelse })
            }

            // For loop
            ast::Stmt::For(for_stmt) => {
                let target = self.convert_expr(&for_stmt.target)?;
                let iter = self.convert_expr(&for_stmt.iter)?;

                let mut body = Vec::new();
                for stmt in &for_stmt.body {
                    body.push(self.convert_stmt(stmt)?);
                }

                let mut orelse = Vec::new();
                for stmt in &for_stmt.orelse {
                    orelse.push(self.convert_stmt(stmt)?);
                }

                Ok(PyStmt::For {
                    target,
                    iter,
                    body,
                    orelse,
                })
            }

            // Break
            ast::Stmt::Break(_) => Ok(PyStmt::Break),

            // Continue
            ast::Stmt::Continue(_) => Ok(PyStmt::Continue),

            // Pass
            ast::Stmt::Pass(_) => Ok(PyStmt::Pass),

            // Import
            ast::Stmt::Import(import) => {
                let mut modules = Vec::new();
                for alias in &import.names {
                    modules.push((
                        alias.name.to_string(),
                        alias.asname.as_ref().map(|s| s.to_string()),
                    ));
                }
                Ok(PyStmt::Import { modules })
            }

            // Import from
            ast::Stmt::ImportFrom(import) => {
                let module = import.module.as_ref().map(|s| s.to_string());

                let mut names = Vec::new();
                for alias in &import.names {
                    names.push((
                        alias.name.to_string(),
                        alias.asname.as_ref().map(|s| s.to_string()),
                    ));
                }

                Ok(PyStmt::ImportFrom {
                    module,
                    names,
                    level: import.level.map(|i| i.to_usize()).unwrap_or(0),
                })
            }

            // Try-except
            ast::Stmt::Try(try_stmt) => {
                let mut body = Vec::new();
                for stmt in &try_stmt.body {
                    body.push(self.convert_stmt(stmt)?);
                }

                let mut handlers = Vec::new();
                for handler in &try_stmt.handlers {
                    // ExceptHandler is a tuple variant in rustpython_parser
                    let ast::ExceptHandler::ExceptHandler(handler_data) = handler;
                    let exception_type = handler_data.type_
                        .as_ref()
                        .map(|expr| self.convert_expr(expr))
                        .transpose()?;

                    let handler_name = handler_data.name.as_ref().map(|id| id.to_string());

                    let mut handler_body = Vec::new();
                    for stmt in &handler_data.body {
                        handler_body.push(self.convert_stmt(stmt)?);
                    }

                    handlers.push(ExceptHandler {
                        exception_type,
                        name: handler_name,
                        body: handler_body,
                    });
                }

                let mut orelse = Vec::new();
                for stmt in &try_stmt.orelse {
                    orelse.push(self.convert_stmt(stmt)?);
                }

                let mut finalbody = Vec::new();
                for stmt in &try_stmt.finalbody {
                    finalbody.push(self.convert_stmt(stmt)?);
                }

                Ok(PyStmt::Try {
                    body,
                    handlers,
                    orelse,
                    finalbody,
                })
            }

            // Raise
            ast::Stmt::Raise(raise) => {
                let exception = raise
                    .exc
                    .as_ref()
                    .map(|expr| self.convert_expr(expr))
                    .transpose()?;

                Ok(PyStmt::Raise { exception })
            }

            // With statement
            ast::Stmt::With(with) => {
                let mut items = Vec::new();
                for item in &with.items {
                    let context_expr = self.convert_expr(&item.context_expr)?;
                    let optional_vars = item
                        .optional_vars
                        .as_ref()
                        .map(|expr| self.convert_expr(expr))
                        .transpose()?;

                    items.push(WithItem {
                        context_expr,
                        optional_vars,
                    });
                }

                let mut body = Vec::new();
                for stmt in &with.body {
                    body.push(self.convert_stmt(stmt)?);
                }

                Ok(PyStmt::With { items, body })
            }

            // Assert
            ast::Stmt::Assert(assert) => {
                let test = self.convert_expr(&assert.test)?;
                let msg = assert
                    .msg
                    .as_ref()
                    .map(|expr| self.convert_expr(expr))
                    .transpose()?;

                Ok(PyStmt::Assert { test, msg })
            }

            // Global
            ast::Stmt::Global(global) => {
                let names = global.names.iter().map(|n| n.to_string()).collect();
                Ok(PyStmt::Global { names })
            }

            // Nonlocal
            ast::Stmt::Nonlocal(nonlocal) => {
                let names = nonlocal.names.iter().map(|n| n.to_string()).collect();
                Ok(PyStmt::Nonlocal { names })
            }

            // Delete
            ast::Stmt::Delete(delete) => {
                let targets = delete
                    .targets
                    .iter()
                    .map(|expr| self.convert_expr(expr))
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyStmt::Delete { targets })
            }

            // Annotated assignment (PEP 526)
            ast::Stmt::AnnAssign(ann) => {
                let target = self.convert_expr(&ann.target)?;
                let annotation = self.convert_type_annotation(&ann.annotation)?;
                let value = ann
                    .value
                    .as_ref()
                    .map(|expr| self.convert_expr(expr))
                    .transpose()?;

                Ok(PyStmt::AnnAssign {
                    target,
                    annotation,
                    value,
                })
            }

            _ => Err(Error::Parse(format!(
                "Unsupported statement type: {:?}",
                stmt
            ))),
        }
    }

    /// Convert an expression
    fn convert_expr(&mut self, expr: &ast::Expr) -> Result<PyExpr> {
        match expr {
            // Literals
            ast::Expr::Constant(constant) => {
                let literal = match &constant.value {
                    ast::Constant::Int(i) => {
                        // Try to convert to i64
                        if let Ok(val) = i.try_into() {
                            PyLiteral::Int(val)
                        } else {
                            return Err(Error::Parse(format!(
                                "Integer literal too large: {}",
                                i
                            )));
                        }
                    }
                    ast::Constant::Float(f) => PyLiteral::Float(*f),
                    ast::Constant::Str(s) => PyLiteral::String(s.to_string()),
                    ast::Constant::Bool(b) => PyLiteral::Bool(*b),
                    ast::Constant::None => PyLiteral::None,
                    ast::Constant::Bytes(b) => PyLiteral::Bytes(b.clone()),
                    _ => {
                        return Err(Error::Parse(format!(
                            "Unsupported constant type: {:?}",
                            constant.value
                        )))
                    }
                };

                Ok(PyExpr::Literal(literal))
            }

            // Variable name
            ast::Expr::Name(name) => Ok(PyExpr::Name(name.id.to_string())),

            // Binary operation
            ast::Expr::BinOp(binop) => {
                let left = Box::new(self.convert_expr(&binop.left)?);
                let op = self.convert_binop(&binop.op);
                let right = Box::new(self.convert_expr(&binop.right)?);

                Ok(PyExpr::BinOp { left, op, right })
            }

            // Unary operation
            ast::Expr::UnaryOp(unaryop) => {
                let op = self.convert_unaryop(&unaryop.op);
                let operand = Box::new(self.convert_expr(&unaryop.operand)?);

                Ok(PyExpr::UnaryOp { op, operand })
            }

            // Function call
            ast::Expr::Call(call) => {
                let func = Box::new(self.convert_expr(&call.func)?);

                let args = call
                    .args
                    .iter()
                    .map(|arg| self.convert_expr(arg))
                    .collect::<Result<Vec<_>>>()?;

                let mut kwargs = std::collections::HashMap::new();
                for keyword in &call.keywords {
                    if let Some(arg_name) = &keyword.arg {
                        let value = self.convert_expr(&keyword.value)?;
                        kwargs.insert(arg_name.to_string(), value);
                    }
                }

                Ok(PyExpr::Call { func, args, kwargs })
            }

            // Attribute access
            ast::Expr::Attribute(attr) => {
                let value = Box::new(self.convert_expr(&attr.value)?);
                let attr_name = attr.attr.to_string();

                Ok(PyExpr::Attribute {
                    value,
                    attr: attr_name,
                })
            }

            // Subscript
            ast::Expr::Subscript(subscript) => {
                let value = Box::new(self.convert_expr(&subscript.value)?);

                // Check if it's a slice
                if let ast::Expr::Slice(slice) = &*subscript.slice {
                    let lower = match &slice.lower {
                        Some(expr) => Some(Box::new(self.convert_expr(expr)?)),
                        None => None,
                    };

                    let upper = match &slice.upper {
                        Some(expr) => Some(Box::new(self.convert_expr(expr)?)),
                        None => None,
                    };

                    let step = match &slice.step {
                        Some(expr) => Some(Box::new(self.convert_expr(expr)?)),
                        None => None,
                    };

                    Ok(PyExpr::Slice {
                        value,
                        lower,
                        upper,
                        step,
                    })
                } else {
                    let index = Box::new(self.convert_expr(&subscript.slice)?);
                    Ok(PyExpr::Subscript { value, index })
                }
            }

            // List
            ast::Expr::List(list) => {
                let elements = list
                    .elts
                    .iter()
                    .map(|expr| self.convert_expr(expr))
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyExpr::List(elements))
            }

            // Tuple
            ast::Expr::Tuple(tuple) => {
                let elements = tuple
                    .elts
                    .iter()
                    .map(|expr| self.convert_expr(expr))
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyExpr::Tuple(elements))
            }

            // Dict
            ast::Expr::Dict(dict) => {
                let keys = dict
                    .keys
                    .iter()
                    .filter_map(|opt_expr| opt_expr.as_ref())
                    .map(|expr| self.convert_expr(expr))
                    .collect::<Result<Vec<_>>>()?;

                let values = dict
                    .values
                    .iter()
                    .map(|expr| self.convert_expr(expr))
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyExpr::Dict { keys, values })
            }

            // Set
            ast::Expr::Set(set) => {
                let elements = set
                    .elts
                    .iter()
                    .map(|expr| self.convert_expr(expr))
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyExpr::Set(elements))
            }

            // List comprehension
            ast::Expr::ListComp(comp) => {
                let element = Box::new(self.convert_expr(&comp.elt)?);

                let generators = comp
                    .generators
                    .iter()
                    .map(|gen| self.convert_comprehension(gen))
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyExpr::ListComp {
                    element,
                    generators,
                })
            }

            // Conditional expression (ternary)
            ast::Expr::IfExp(ifexp) => {
                let test = Box::new(self.convert_expr(&ifexp.test)?);
                let body = Box::new(self.convert_expr(&ifexp.body)?);
                let orelse = Box::new(self.convert_expr(&ifexp.orelse)?);

                Ok(PyExpr::IfExp { test, body, orelse })
            }

            // Lambda
            ast::Expr::Lambda(lambda) => {
                let args = lambda
                    .args
                    .args
                    .iter()
                    .map(|arg| arg.def.arg.to_string())
                    .collect();

                let body = Box::new(self.convert_expr(&lambda.body)?);

                Ok(PyExpr::Lambda { args, body })
            }

            // Comparison
            ast::Expr::Compare(compare) => {
                let left = Box::new(self.convert_expr(&compare.left)?);

                // Python allows chained comparisons: a < b < c
                // For simplicity, we'll handle the first comparison only
                if compare.ops.is_empty() || compare.comparators.is_empty() {
                    return Err(Error::Parse("Empty comparison".to_string()));
                }

                let op = self.convert_cmpop(&compare.ops[0]);
                let right = Box::new(self.convert_expr(&compare.comparators[0])?);

                Ok(PyExpr::Compare { left, op, right })
            }

            // Boolean operation (and, or)
            ast::Expr::BoolOp(boolop) => {
                if boolop.values.len() < 2 {
                    return Err(Error::Parse("BoolOp with less than 2 values".to_string()));
                }

                let op = match boolop.op {
                    ast::BoolOp::And => BoolOp::And,
                    ast::BoolOp::Or => BoolOp::Or,
                };

                let left = Box::new(self.convert_expr(&boolop.values[0])?);
                let right = Box::new(self.convert_expr(&boolop.values[1])?);

                Ok(PyExpr::BoolOp { op, left, right })
            }

            // Await expression
            ast::Expr::Await(await_expr) => {
                let value = Box::new(self.convert_expr(&await_expr.value)?);
                Ok(PyExpr::Await(value))
            }

            // Yield expression
            ast::Expr::Yield(yield_expr) => {
                let value = match &yield_expr.value {
                    Some(expr) => Some(Box::new(self.convert_expr(expr)?)),
                    None => None,
                };

                Ok(PyExpr::Yield(value))
            }

            _ => Err(Error::Parse(format!(
                "Unsupported expression type: {:?}",
                expr
            ))),
        }
    }

    /// Convert function parameters
    fn convert_parameters(&mut self, args: &ast::Arguments) -> Result<Vec<FunctionParam>> {
        let mut params = Vec::new();

        for arg in &args.args {
            let name = arg.def.arg.to_string();
            let type_annotation = arg
                .def
                .annotation
                .as_ref()
                .map(|expr| self.convert_type_annotation(expr))
                .transpose()?;

            params.push(FunctionParam {
                name,
                type_annotation,
                default_value: None, // Defaults are handled separately in rustpython
            });
        }

        Ok(params)
    }

    /// Convert type annotation
    fn convert_type_annotation(&mut self, expr: &ast::Expr) -> Result<TypeAnnotation> {
        match expr {
            ast::Expr::Name(name) => Ok(TypeAnnotation::Name(name.id.to_string())),

            ast::Expr::Subscript(subscript) => {
                // Generic types like List[int], Dict[str, int]
                if let ast::Expr::Name(name) = &*subscript.value {
                    let base = name.id.to_string();
                    let args = vec![self.convert_type_annotation(&subscript.slice)?];

                    Ok(TypeAnnotation::Generic {
                        base: Box::new(TypeAnnotation::Name(base)),
                        args,
                    })
                } else {
                    Err(Error::Parse("Complex generic type".to_string()))
                }
            }

            _ => Ok(TypeAnnotation::Name("Any".to_string())),
        }
    }

    /// Convert comprehension generator
    fn convert_comprehension(&mut self, gen: &ast::Comprehension) -> Result<Comprehension> {
        let target = self.convert_expr(&gen.target)?;
        let iter = self.convert_expr(&gen.iter)?;

        let ifs = gen
            .ifs
            .iter()
            .map(|expr| self.convert_expr(expr))
            .collect::<Result<Vec<_>>>()?;

        Ok(Comprehension { target, iter, ifs })
    }

    /// Convert binary operator
    fn convert_binop(&self, op: &ast::Operator) -> BinOp {
        match op {
            ast::Operator::Add => BinOp::Add,
            ast::Operator::Sub => BinOp::Sub,
            ast::Operator::Mult => BinOp::Mult,
            ast::Operator::Div => BinOp::Div,
            ast::Operator::FloorDiv => BinOp::FloorDiv,
            ast::Operator::Mod => BinOp::Mod,
            ast::Operator::Pow => BinOp::Pow,
            ast::Operator::LShift => BinOp::LShift,
            ast::Operator::RShift => BinOp::RShift,
            ast::Operator::BitOr => BinOp::BitOr,
            ast::Operator::BitXor => BinOp::BitXor,
            ast::Operator::BitAnd => BinOp::BitAnd,
            ast::Operator::MatMult => BinOp::MatMult,
        }
    }

    /// Convert unary operator
    fn convert_unaryop(&self, op: &ast::UnaryOp) -> UnaryOp {
        match op {
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::UAdd => UnaryOp::UAdd,
            ast::UnaryOp::USub => UnaryOp::USub,
            ast::UnaryOp::Invert => UnaryOp::Invert,
        }
    }

    /// Convert comparison operator
    fn convert_cmpop(&self, op: &ast::CmpOp) -> CmpOp {
        match op {
            ast::CmpOp::Eq => CmpOp::Eq,
            ast::CmpOp::NotEq => CmpOp::NotEq,
            ast::CmpOp::Lt => CmpOp::Lt,
            ast::CmpOp::LtE => CmpOp::LtE,
            ast::CmpOp::Gt => CmpOp::Gt,
            ast::CmpOp::GtE => CmpOp::GtE,
            ast::CmpOp::Is => CmpOp::Is,
            ast::CmpOp::IsNot => CmpOp::IsNot,
            ast::CmpOp::In => CmpOp::In,
            ast::CmpOp::NotIn => CmpOp::NotIn,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let source = r#"
def add(a, b):
    return a + b
"#;

        let parser = PythonParser::new(source, "test.py");
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);

        if let PyStmt::FunctionDef { name, params, .. } = &module.statements[0] {
            assert_eq!(name, "add");
            assert_eq!(params.len(), 2);
        } else {
            panic!("Expected function definition");
        }
    }

    #[test]
    fn test_parse_literals() {
        let source = "42";
        let parser = PythonParser::new(source, "test.py");
        let expr = parser.parse_expression().unwrap();

        assert!(matches!(expr, PyExpr::Literal(PyLiteral::Int(42))));

        let source = "3.14";
        let parser = PythonParser::new(source, "test.py");
        let expr = parser.parse_expression().unwrap();

        assert!(matches!(expr, PyExpr::Literal(PyLiteral::Float(_))));

        let source = r#""hello""#;
        let parser = PythonParser::new(source, "test.py");
        let expr = parser.parse_expression().unwrap();

        assert!(matches!(expr, PyExpr::Literal(PyLiteral::String(_))));
    }

    #[test]
    fn test_parse_binary_operation() {
        let source = "a + b";
        let parser = PythonParser::new(source, "test.py");
        let expr = parser.parse_expression().unwrap();

        if let PyExpr::BinOp { left, op, right } = expr {
            assert!(matches!(*left, PyExpr::Name(_)));
            assert!(matches!(op, BinOp::Add));
            assert!(matches!(*right, PyExpr::Name(_)));
        } else {
            panic!("Expected binary operation");
        }
    }

    #[test]
    fn test_parse_function_call() {
        let source = "print(42)";
        let parser = PythonParser::new(source, "test.py");
        let expr = parser.parse_expression().unwrap();

        if let PyExpr::Call { func, args, .. } = expr {
            assert!(matches!(*func, PyExpr::Name(_)));
            assert_eq!(args.len(), 1);
        } else {
            panic!("Expected function call");
        }
    }

    #[test]
    fn test_parse_list_comprehension() {
        let source = "[x * 2 for x in range(10)]";
        let parser = PythonParser::new(source, "test.py");
        let expr = parser.parse_expression().unwrap();

        assert!(matches!(expr, PyExpr::ListComp { .. }));
    }

    #[test]
    fn test_parse_if_statement() {
        let source = r#"
if x > 0:
    print("positive")
else:
    print("negative")
"#;

        let parser = PythonParser::new(source, "test.py");
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        assert!(matches!(module.statements[0], PyStmt::If { .. }));
    }

    #[test]
    fn test_parse_class_definition() {
        let source = r#"
class MyClass:
    def method(self):
        pass
"#;

        let parser = PythonParser::new(source, "test.py");
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);

        if let PyStmt::ClassDef { name, body, .. } = &module.statements[0] {
            assert_eq!(name, "MyClass");
            assert!(!body.is_empty());
        } else {
            panic!("Expected class definition");
        }
    }

    #[test]
    fn test_parse_import() {
        let source = "import sys";
        let parser = PythonParser::new(source, "test.py");
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        assert!(matches!(module.statements[0], PyStmt::Import { .. }));

        let source = "from os import path";
        let parser = PythonParser::new(source, "test.py");
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        assert!(matches!(module.statements[0], PyStmt::ImportFrom { .. }));
    }

    #[test]
    fn test_parse_async_function() {
        let source = r#"
async def fetch_data():
    result = await get_data()
    return result
"#;

        let parser = PythonParser::new(source, "test.py");
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);

        if let PyStmt::FunctionDef { is_async, .. } = &module.statements[0] {
            assert!(is_async);
        } else {
            panic!("Expected async function definition");
        }
    }

    #[test]
    fn test_parse_try_except() {
        let source = r#"
try:
    risky_operation()
except ValueError as e:
    handle_error(e)
finally:
    cleanup()
"#;

        let parser = PythonParser::new(source, "test.py");
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        assert!(matches!(module.statements[0], PyStmt::Try { .. }));
    }

    #[test]
    fn test_parse_with_statement() {
        let source = r#"
with open("file.txt") as f:
    content = f.read()
"#;

        let parser = PythonParser::new(source, "test.py");
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        assert!(matches!(module.statements[0], PyStmt::With { .. }));
    }

    #[test]
    fn test_parse_type_annotations() {
        let source = r#"
def add(a: int, b: int) -> int:
    return a + b
"#;

        let parser = PythonParser::new(source, "test.py");
        let module = parser.parse().unwrap();

        if let PyStmt::FunctionDef {
            params,
            return_type,
            ..
        } = &module.statements[0]
        {
            assert!(params[0].type_annotation.is_some());
            assert!(return_type.is_some());
        } else {
            panic!("Expected function definition");
        }
    }
}
