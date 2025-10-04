//! Indentation-aware Python parser
//!
//! Handles Python's indentation-based block structure for:
//! - if/elif/else blocks
//! - for loops
//! - while loops
//! - function definitions

use crate::python_ast::*;
use crate::{Error, Result};

#[derive(Debug, Clone)]
struct Line {
    content: String,
    indent_level: usize,
    line_number: usize,
}

pub struct IndentedPythonParser {
    lines: Vec<Line>,
    current_line: usize,
    pending_decorators: Vec<PyExpr>,
}

impl IndentedPythonParser {
    pub fn new(source: &str) -> Self {
        let lines: Vec<Line> = source
            .lines()
            .enumerate()
            .map(|(idx, s)| {
                let indent_level = s.chars().take_while(|c| *c == ' ').count() / 4;
                Line {
                    content: s.trim().to_string(),
                    indent_level,
                    line_number: idx + 1,
                }
            })
            .collect();

        Self {
            lines,
            current_line: 0,
            pending_decorators: Vec::new(),
        }
    }

    pub fn parse(&mut self) -> Result<PyModule> {
        let mut module = PyModule::new();

        while self.current_line < self.lines.len() {
            let line = &self.lines[self.current_line];

            // Skip empty lines and comments
            if line.content.is_empty() || line.content.starts_with('#') {
                self.current_line += 1;
                continue;
            }

            // Parse statement at current indentation level
            if let Some(stmt) = self.parse_statement(0)? {
                module.add_stmt(stmt);
            }
        }

        Ok(module)
    }

    fn parse_statement(&mut self, expected_indent: usize) -> Result<Option<PyStmt>> {
        if self.current_line >= self.lines.len() {
            return Ok(None);
        }

        let line = &self.lines[self.current_line].clone();

        // Check indent level
        if line.indent_level != expected_indent {
            return Ok(None);
        }

        // If statement with block
        if line.content.starts_with("if ") && line.content.ends_with(':') {
            return self.parse_if_statement(expected_indent);
        }

        // While loop with block
        if line.content.starts_with("while ") && line.content.ends_with(':') {
            return self.parse_while_loop(expected_indent);
        }

        // For loop with block
        if line.content.starts_with("for ") && line.content.ends_with(':') {
            return self.parse_for_loop(expected_indent);
        }

        // Decorator (collects decorators for next function/class)
        if line.content.starts_with('@') {
            return self.parse_decorator(expected_indent);
        }

        // Class definition
        if line.content.starts_with("class ") && line.content.ends_with(':') {
            return self.parse_class_def(expected_indent);
        }

        // Async function definition
        if line.content.starts_with("async def ") && line.content.ends_with(':') {
            return self.parse_function_def(expected_indent);
        }

        // Function definition
        if line.content.starts_with("def ") && line.content.ends_with(':') {
            return self.parse_function_def(expected_indent);
        }

        // Import statement
        if line.content.starts_with("import ") {
            return self.parse_import_statement();
        }

        // From import statement
        if line.content.starts_with("from ") && line.content.contains(" import ") {
            return self.parse_from_import_statement();
        }

        // Try-except statement
        if line.content == "try:" {
            return self.parse_try_except(expected_indent);
        }

        // With statement (context manager)
        if line.content.starts_with("with ") && line.content.ends_with(':') {
            return self.parse_with_statement(expected_indent);
        }

        // Raise statement
        if line.content.starts_with("raise") {
            self.current_line += 1;
            let exc = if line.content.len() > 5 {
                Some(self.parse_expr(line.content[5..].trim())?)
            } else {
                None
            };
            return Ok(Some(PyStmt::Raise { exception: exc }));
        }

        // Return statement
        if line.content.starts_with("return") {
            self.current_line += 1;
            let expr = if line.content.len() > 6 {
                Some(self.parse_expr(line.content[6..].trim())?)
            } else {
                None
            };
            return Ok(Some(PyStmt::Return { value: expr }));
        }

        // Assert statement
        if line.content.starts_with("assert ") {
            self.current_line += 1;
            let rest = line.content[7..].trim(); // Skip "assert "

            // Check for optional message: assert condition, "message"
            let (test_str, msg) = if let Some(comma_pos) = rest.find(',') {
                let test_part = rest[..comma_pos].trim();
                let msg_part = rest[comma_pos + 1..].trim();
                let msg_expr = Some(self.parse_expr(msg_part)?);
                (test_part, msg_expr)
            } else {
                (rest, None)
            };

            let test = self.parse_expr(test_str)?;
            return Ok(Some(PyStmt::Assert { test, msg }));
        }

        // Simple statements (assignment, print, etc.)
        let stmt = self.parse_simple_statement(&line.content)?;
        self.current_line += 1;
        Ok(stmt)
    }

    fn parse_if_statement(&mut self, expected_indent: usize) -> Result<Option<PyStmt>> {
        let line = &self.lines[self.current_line].clone();

        // Handle both "if" and "elif"
        let (condition_str, is_elif) = if line.content.starts_with("elif ") {
            (line.content[5..line.content.len() - 1].trim(), true)
        } else {
            (line.content[3..line.content.len() - 1].trim(), false)
        };

        let condition = self.parse_expr(condition_str)?;

        self.current_line += 1;

        // Parse if/elif body
        let body = self.parse_block(expected_indent + 1)?;

        // Check for else or elif (but not after elif)
        let mut orelse = vec![];
        if self.current_line < self.lines.len() {
            let next_line = &self.lines[self.current_line];
            if next_line.indent_level == expected_indent
                && (next_line.content == "else:" || next_line.content.starts_with("elif "))
            {
                if next_line.content == "else:" {
                    self.current_line += 1;
                    orelse = self.parse_block(expected_indent + 1)?;
                } else if next_line.content.starts_with("elif ") {
                    // Treat elif as nested if in else block
                    if let Some(elif_stmt) = self.parse_if_statement(expected_indent)? {
                        orelse = vec![elif_stmt];
                    }
                }
            }
        }

        Ok(Some(PyStmt::If {
            test: condition,
            body,
            orelse,
        }))
    }

    fn parse_while_loop(&mut self, expected_indent: usize) -> Result<Option<PyStmt>> {
        let line = &self.lines[self.current_line].clone();
        let condition_str = line.content[6..line.content.len() - 1].trim();
        let condition = self.parse_expr(condition_str)?;

        self.current_line += 1;

        let body = self.parse_block(expected_indent + 1)?;

        // Check for optional else clause
        let orelse = if self.current_line < self.lines.len() {
            let line = &self.lines[self.current_line];
            if line.indent_level == expected_indent && line.content == "else:" {
                self.current_line += 1;
                self.parse_block(expected_indent + 1)?
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        Ok(Some(PyStmt::While {
            test: condition,
            body,
            orelse,
        }))
    }

    fn parse_for_loop(&mut self, expected_indent: usize) -> Result<Option<PyStmt>> {
        let line = &self.lines[self.current_line].clone();
        let for_content = &line.content[4..line.content.len() - 1].trim();

        // Parse: for VAR in EXPR or for (VAR1, VAR2) in EXPR or for VAR1, VAR2 in EXPR
        if let Some(in_pos) = for_content.find(" in ") {
            let target_str = for_content[..in_pos].trim();

            // Parse target - handle tuple unpacking
            let target = if target_str.contains(',') {
                // Tuple unpacking: "i, item" or "(i, item)"
                let clean_str = if target_str.starts_with('(') && target_str.ends_with(')') {
                    &target_str[1..target_str.len() - 1]
                } else {
                    target_str
                };

                let elements: Vec<PyExpr> = clean_str
                    .split(',')
                    .map(|s| PyExpr::Name(s.trim().to_string()))
                    .collect();
                PyExpr::Tuple(elements)
            } else {
                // Single target
                PyExpr::Name(target_str.to_string())
            };

            let iter_str = for_content[in_pos + 4..].trim();
            let iter = self.parse_expr(iter_str)?;

            self.current_line += 1;

            let body = self.parse_block(expected_indent + 1)?;

            // Check for optional else clause
            let orelse = if self.current_line < self.lines.len() {
                let line = &self.lines[self.current_line];
                if line.indent_level == expected_indent && line.content == "else:" {
                    self.current_line += 1;
                    self.parse_block(expected_indent + 1)?
                } else {
                    vec![]
                }
            } else {
                vec![]
            };

            return Ok(Some(PyStmt::For {
                target,
                iter,
                body,
                orelse,
            }));
        }

        Err(Error::CodeGeneration(
            "Invalid for loop syntax".to_string(),
        ))
    }

    fn parse_function_def(&mut self, expected_indent: usize) -> Result<Option<PyStmt>> {
        let line = &self.lines[self.current_line].clone();

        // Check if async function
        let is_async = line.content.starts_with("async def ");
        let def_start = if is_async { 10 } else { 4 }; // "async def " = 10, "def " = 4
        let def_content = &line.content[def_start..line.content.len() - 1].trim();

        // Parse: def NAME(ARGS) -> RETURN_TYPE or def NAME(ARGS)
        if let Some(paren_pos) = def_content.find('(') {
            let name = def_content[..paren_pos].trim().to_string();
            let close_paren = if let Some(pos) = def_content.find(')') {
                pos
            } else {
                return Err(Error::CodeGeneration("Missing closing paren".to_string()));
            };

            let args_str = &def_content[paren_pos + 1..close_paren];
            let params = self.parse_function_args(args_str)?;

            // Check for return type annotation: -> TYPE
            let return_type = if close_paren + 1 < def_content.len() {
                let after_paren = &def_content[close_paren + 1..].trim();
                if after_paren.starts_with("->") {
                    Some(TypeAnnotation::Name(after_paren[2..].trim().to_string()))
                } else {
                    None
                }
            } else {
                None
            };

            self.current_line += 1;

            let body = self.parse_block(expected_indent + 1)?;

            return Ok(Some(PyStmt::FunctionDef {
                name,
                params,
                body,
                return_type,
                decorators: std::mem::take(&mut self.pending_decorators),
                is_async,
            }));
        }

        Err(Error::CodeGeneration(
            "Invalid function definition".to_string(),
        ))
    }

    fn parse_class_def(&mut self, expected_indent: usize) -> Result<Option<PyStmt>> {
        let line = &self.lines[self.current_line].clone();
        let class_content = &line.content[6..line.content.len() - 1].trim();

        // Parse: class NAME or class NAME(BASE)
        let (name, bases) = if let Some(paren_pos) = class_content.find('(') {
            // Has base classes
            let name = class_content[..paren_pos].trim().to_string();
            let close_paren = if let Some(pos) = class_content.find(')') {
                pos
            } else {
                return Err(Error::CodeGeneration("Missing closing paren in class".to_string()));
            };

            let bases_str = &class_content[paren_pos + 1..close_paren];
            let bases: Vec<PyExpr> = if bases_str.trim().is_empty() {
                vec![]
            } else {
                bases_str.split(',').map(|s| PyExpr::Name(s.trim().to_string())).collect()
            };

            (name, bases)
        } else {
            // No base classes
            (class_content.to_string(), vec![] as Vec<PyExpr>)
        };

        self.current_line += 1;

        // Parse class body
        let body = self.parse_block(expected_indent + 1)?;

        Ok(Some(PyStmt::ClassDef {
            name,
            bases,
            body,
            decorators: std::mem::take(&mut self.pending_decorators),
        }))
    }

    fn parse_decorator(&mut self, _expected_indent: usize) -> Result<Option<PyStmt>> {
        let line = &self.lines[self.current_line].clone();

        // Parse decorator: @decorator_name or @decorator_name(args)
        let decorator_str = &line.content[1..].trim(); // Remove @ and trim

        // Extract just the decorator name (before any parentheses)
        let decorator_name = if let Some(paren_pos) = decorator_str.find('(') {
            decorator_str[..paren_pos].trim().to_string()
        } else {
            decorator_str.to_string()
        };

        self.pending_decorators.push(PyExpr::Name(decorator_name));
        self.current_line += 1;

        // Return None to continue parsing (decorators don't generate statements themselves)
        Ok(None)
    }

    fn parse_block(&mut self, expected_indent: usize) -> Result<Vec<PyStmt>> {
        let mut stmts = vec![];

        while self.current_line < self.lines.len() {
            let line = &self.lines[self.current_line];

            // Skip empty lines and comments
            if line.content.is_empty() || line.content.starts_with('#') {
                self.current_line += 1;
                continue;
            }

            // Check if we're still in the block
            if line.indent_level < expected_indent {
                break;
            }

            if line.indent_level > expected_indent {
                return Err(Error::CodeGeneration(format!(
                    "Unexpected indentation at line {}",
                    line.line_number
                )));
            }

            if let Some(stmt) = self.parse_statement(expected_indent)? {
                stmts.push(stmt);
            } else {
                break;
            }
        }

        Ok(stmts)
    }

    fn parse_function_args(&self, args_str: &str) -> Result<Vec<FunctionParam>> {
        if args_str.trim().is_empty() {
            return Ok(vec![]);
        }

        let params: Vec<FunctionParam> = args_str
            .split(',')
            .map(|arg| {
                let arg = arg.trim();
                // Simple parsing: name or name: type
                if let Some(colon_pos) = arg.find(':') {
                    let name = arg[..colon_pos].trim().to_string();
                    let type_annotation = Some(TypeAnnotation::Name(arg[colon_pos + 1..].trim().to_string()));
                    FunctionParam {
                        name,
                        type_annotation,
                        default_value: None,
                    }
                } else {
                    FunctionParam {
                        name: arg.to_string(),
                        type_annotation: None,
                        default_value: None,
                    }
                }
            })
            .collect();

        Ok(params)
    }

    fn parse_simple_statement(&self, line: &str) -> Result<Option<PyStmt>> {
        // Pass statement
        if line == "pass" {
            return Ok(Some(PyStmt::Pass));
        }

        // Break statement
        if line == "break" {
            return Ok(Some(PyStmt::Break));
        }

        // Continue statement
        if line == "continue" {
            return Ok(Some(PyStmt::Continue));
        }

        // Assignment (including chained assignment like x = y = z = 0)
        // First, check if this looks like an assignment by checking for '='
        // but exclude comparison operators
        if line.contains('=') && !line.contains("==") && !line.contains("!=")
            && !line.contains("<=") && !line.contains(">=") {

            // Split by '=' but be careful about comparison operators
            // For now, simple split - will handle edge cases later
            let parts: Vec<&str> = line.split('=')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            if parts.len() >= 2 {
                let last_part = parts[parts.len() - 1];

                // Check if first part looks like augmented assignment (x+=, x-=, etc.)
                if parts.len() == 2 {
                    let target = parts[0];
                    if let Some(op_char) = target.chars().last() {
                        if matches!(op_char, '+' | '-' | '*' | '/' | '%') {
                            let actual_target = target[..target.len() - 1].trim().to_string();
                            let op = match op_char {
                                '+' => BinOp::Add,
                                '-' => BinOp::Sub,
                                '*' => BinOp::Mult,
                                '/' => BinOp::Div,
                                '%' => BinOp::Mod,
                                _ => unreachable!(),
                            };
                            let value = self.parse_expr(last_part)?;
                            return Ok(Some(PyStmt::AugAssign {
                                target: PyExpr::Name(actual_target),
                                op,
                                value,
                            }));
                        }
                    }
                }

                // Regular assignment (take first target for chained assignments like a = b = 5)
                let target_str = parts[parts.len() - 2].trim();
                let value = self.parse_expr(last_part)?;

                // Parse target as expression
                let target = self.parse_expr(target_str)?;

                // Return assignment
                return Ok(Some(PyStmt::Assign {
                    target,
                    value,
                }));
            }
        }

        // Print function
        if line.starts_with("print(") && line.ends_with(')') {
            let args_str = &line[6..line.len() - 1];
            let expr = self.parse_expr(args_str)?;
            return Ok(Some(PyStmt::Expr(PyExpr::Call {
                func: Box::new(PyExpr::Name("print".to_string())),
                args: vec![expr],
                kwargs: std::collections::HashMap::new(),
            })));
        }

        // General function call as statement
        if line.contains('(') && line.ends_with(')') {
            // Try to parse as expression - if it's a Call, wrap it as Expr statement
            if let Ok(expr) = self.parse_expr(line) {
                if matches!(expr, PyExpr::Call { .. }) {
                    return Ok(Some(PyStmt::Expr(expr)));
                }
            }
        }

        Ok(None)
    }

    /// Find position of operator outside of parentheses
    fn find_op_outside_parens(&self, s: &str, op: &str) -> Option<usize> {
        let mut depth = 0;
        let mut i = 0;
        let chars: Vec<char> = s.chars().collect();

        while i < chars.len() {
            match chars[i] {
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                _ => {}
            }

            if depth == 0 && i + op.len() <= s.len() {
                if &s[i..i + op.len()] == op {
                    return Some(i);
                }
            }
            i += 1;
        }

        None
    }

    fn parse_expr(&self, s: &str) -> Result<PyExpr> {
        let s = s.trim();

        // Await expression: await expr
        if s.starts_with("await ") {
            let inner = &s[6..].trim();
            let value = self.parse_expr(inner)?;
            return Ok(PyExpr::Await(Box::new(value)));
        }

        // Lambda expressions: lambda x: x + 1, lambda x, y: x + y
        if s.starts_with("lambda ") {
            return self.parse_lambda(s);
        }

        // Boolean literals
        if s == "True" {
            return Ok(PyExpr::Literal(PyLiteral::Bool(true)));
        }
        if s == "False" {
            return Ok(PyExpr::Literal(PyLiteral::Bool(false)));
        }

        // None
        if s == "None" {
            return Ok(PyExpr::Literal(PyLiteral::None));
        }

        // range() function
        if s.starts_with("range(") && s.ends_with(')') {
            let args_str = &s[6..s.len() - 1];
            let args: Vec<PyExpr> = args_str
                .split(',')
                .map(|arg| self.parse_expr(arg.trim()))
                .collect::<Result<Vec<_>>>()?;

            return Ok(PyExpr::Call {
                func: Box::new(PyExpr::Name("range".to_string())),
                args,
                kwargs: std::collections::HashMap::new(),
            });
        }

        // List literals or list comprehension
        if s.starts_with('[') && s.ends_with(']') {
            let content = &s[1..s.len() - 1].trim();
            if content.is_empty() {
                return Ok(PyExpr::List(vec![]));
            }

            // Check for list comprehension: [element for var in iterable]
            if content.contains(" for ") {
                return self.parse_list_comprehension(content);
            }

            // Regular list literal - use depth-aware comma splitting
            let mut element_strings = Vec::new();
            let mut current = String::new();
            let mut depth = 0;

            for ch in content.chars() {
                match ch {
                    '(' | '[' | '{' => {
                        depth += 1;
                        current.push(ch);
                    }
                    ')' | ']' | '}' => {
                        depth -= 1;
                        current.push(ch);
                    }
                    ',' if depth == 0 => {
                        element_strings.push(current.trim().to_string());
                        current.clear();
                    }
                    _ => current.push(ch),
                }
            }

            if !current.trim().is_empty() {
                element_strings.push(current.trim().to_string());
            }

            let elements: Result<Vec<_>> = element_strings
                .iter()
                .map(|e| self.parse_expr(e))
                .collect();
            return Ok(PyExpr::List(elements?));
        }

        // Dictionary literals
        if s.starts_with('{') && s.ends_with('}') {
            let content = &s[1..s.len() - 1].trim();
            if content.is_empty() {
                return Ok(PyExpr::Dict {
                    keys: vec![],
                    values: vec![],
                });
            }

            // Parse key: value pairs - use depth-aware comma splitting
            let mut keys = vec![];
            let mut values = vec![];

            // Split pairs by comma at depth 0
            let mut pair_strings = Vec::new();
            let mut current = String::new();
            let mut depth = 0;

            for ch in content.chars() {
                match ch {
                    '(' | '[' | '{' => {
                        depth += 1;
                        current.push(ch);
                    }
                    ')' | ']' | '}' => {
                        depth -= 1;
                        current.push(ch);
                    }
                    ',' if depth == 0 => {
                        pair_strings.push(current.trim().to_string());
                        current.clear();
                    }
                    _ => current.push(ch),
                }
            }

            if !current.trim().is_empty() {
                pair_strings.push(current.trim().to_string());
            }

            for pair in pair_strings {
                if let Some(colon_pos) = pair.find(':') {
                    let key_str = &pair[..colon_pos].trim();
                    let value_str = &pair[colon_pos + 1..].trim();
                    keys.push(self.parse_expr(key_str)?);
                    values.push(self.parse_expr(value_str)?);
                }
            }

            return Ok(PyExpr::Dict { keys, values });
        }

        // Tuple literals
        if s.starts_with('(') && s.ends_with(')') {
            let content = &s[1..s.len() - 1].trim();
            if content.is_empty() {
                return Ok(PyExpr::Tuple(vec![]));
            }

            // Use depth-aware comma splitting
            let mut element_strings = Vec::new();
            let mut current = String::new();
            let mut depth = 0;

            for ch in content.chars() {
                match ch {
                    '(' | '[' | '{' => {
                        depth += 1;
                        current.push(ch);
                    }
                    ')' | ']' | '}' => {
                        depth -= 1;
                        current.push(ch);
                    }
                    ',' if depth == 0 => {
                        element_strings.push(current.trim().to_string());
                        current.clear();
                    }
                    _ => current.push(ch),
                }
            }

            if !current.trim().is_empty() {
                element_strings.push(current.trim().to_string());
            }

            let elements: Result<Vec<_>> = element_strings
                .iter()
                .map(|e| self.parse_expr(e))
                .collect();
            return Ok(PyExpr::Tuple(elements?));
        }

        // String literals
        if (s.starts_with('"') && s.ends_with('"'))
            || (s.starts_with('\'') && s.ends_with('\''))
        {
            let content = &s[1..s.len() - 1];
            return Ok(PyExpr::Literal(PyLiteral::String(content.to_string())));
        }

        // Float literals
        if s.contains('.') && s.parse::<f64>().is_ok() {
            return Ok(PyExpr::Literal(PyLiteral::Float(s.parse().unwrap())));
        }

        // Integer literals
        if let Ok(value) = s.parse::<i64>() {
            return Ok(PyExpr::Literal(PyLiteral::Int(value)));
        }

        // Logical operators
        for (op_str, is_and) in &[(" and ", true), (" or ", false)] {
            if let Some(pos) = self.find_op_outside_parens(s, op_str) {
                let left = self.parse_expr(&s[..pos])?;
                let right = self.parse_expr(&s[pos + op_str.len()..])?;
                let op = if *is_and {
                    BinOp::BitAnd
                } else {
                    BinOp::BitOr
                };
                return Ok(PyExpr::BinOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                });
            }
        }

        // Comparison operators
        for (op_str, cmp_op) in &[
            ("==", CmpOp::Eq),
            ("!=", CmpOp::NotEq),
            ("<=", CmpOp::LtE),
            (">=", CmpOp::GtE),
            ("<", CmpOp::Lt),
            (">", CmpOp::Gt),
        ] {
            if let Some(pos) = self.find_op_outside_parens(s, op_str) {
                let left = self.parse_expr(&s[..pos])?;
                let right = self.parse_expr(&s[pos + op_str.len()..])?;
                return Ok(PyExpr::Compare {
                    left: Box::new(left),
                    op: *cmp_op,
                    right: Box::new(right),
                });
            }
        }

        // Unary not
        if s.starts_with("not ") {
            let operand = self.parse_expr(&s[4..])?;
            return Ok(PyExpr::UnaryOp {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            });
        }

        // Arithmetic operators (in order of precedence)
        for (op_str, op) in &[
            (" + ", BinOp::Add),
            (" - ", BinOp::Sub),
            (" * ", BinOp::Mult),
            (" / ", BinOp::Div),
            (" // ", BinOp::FloorDiv),
            (" % ", BinOp::Mod),
            (" ** ", BinOp::Pow),
        ] {
            if let Some(pos) = self.find_op_outside_parens(s, op_str) {
                let left = self.parse_expr(&s[..pos])?;
                let right = self.parse_expr(&s[pos + op_str.len()..])?;
                return Ok(PyExpr::BinOp {
                    left: Box::new(left),
                    op: *op,
                    right: Box::new(right),
                });
            }
        }

        // List indexing or slicing
        if let Some(bracket_pos) = s.find('[') {
            if s.ends_with(']') {
                let value_str = &s[..bracket_pos];
                let index_str = &s[bracket_pos + 1..s.len() - 1];
                let value = self.parse_expr(value_str)?;

                // Check if it's a slice (contains ':')
                if index_str.contains(':') {
                    return self.parse_slice(value, index_str);
                }

                // Regular subscript
                let index = self.parse_expr(index_str)?;
                return Ok(PyExpr::Subscript {
                    value: Box::new(value),
                    index: Box::new(index),
                });
            }
        }

        // Function calls (general form: name(...))
        if let Some(paren_pos) = s.find('(') {
            if s.ends_with(')') {
                let func_name = &s[..paren_pos];
                // Make sure it's a valid identifier
                if func_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    let args_str = &s[paren_pos + 1..s.len() - 1];
                    let args: Vec<PyExpr> = if args_str.trim().is_empty() {
                        vec![]
                    } else {
                        // Split by comma at depth 0 only (depth-aware splitting)
                        let mut arg_strings = Vec::new();
                        let mut current = String::new();
                        let mut depth = 0;

                        for ch in args_str.chars() {
                            match ch {
                                '(' | '[' | '{' => {
                                    depth += 1;
                                    current.push(ch);
                                }
                                ')' | ']' | '}' => {
                                    depth -= 1;
                                    current.push(ch);
                                }
                                ',' if depth == 0 => {
                                    arg_strings.push(current.trim().to_string());
                                    current.clear();
                                }
                                _ => current.push(ch),
                            }
                        }

                        if !current.trim().is_empty() {
                            arg_strings.push(current.trim().to_string());
                        }

                        arg_strings
                            .iter()
                            .map(|arg| self.parse_expr(arg))
                            .collect::<Result<Vec<_>>>()?
                    };

                    return Ok(PyExpr::Call {
                        func: Box::new(PyExpr::Name(func_name.to_string())),
                        args,
                        kwargs: std::collections::HashMap::new(),
                    });
                }
            }
        }

        // Unary minus (e.g., -x, -5)
        if s.starts_with('-') && s.len() > 1 {
            let operand_str = &s[1..].trim();
            // Make sure it's not subtraction (has spaces around -)
            if !operand_str.is_empty() {
                let operand = self.parse_expr(operand_str)?;
                return Ok(PyExpr::UnaryOp {
                    op: UnaryOp::USub,
                    operand: Box::new(operand),
                });
            }
        }

        // Method call or attribute access (obj.method(...) or obj.attr or obj.attr.method(...))
        if let Some(dot_pos) = s.find('.') {
            let value_str = &s[..dot_pos];
            let attr_part = &s[dot_pos + 1..];

            // Check if attr_part has a method call at the end
            if let Some(paren_pos) = attr_part.rfind('(') {
                if attr_part.ends_with(')') {
                    // Find the method name (everything between last dot and paren, or from start if no dot)
                    let last_dot_in_attr = attr_part[..paren_pos].rfind('.');
                    let (nested_attrs, method_name) = if let Some(last_dot) = last_dot_in_attr {
                        // Has nested attrs like path.exists in os.path.exists(...)
                        (&attr_part[..last_dot], &attr_part[last_dot + 1..paren_pos])
                    } else {
                        // No nested attrs, like method in os.method(...)
                        ("", &attr_part[..paren_pos])
                    };

                    if method_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        let args_str = &attr_part[paren_pos + 1..attr_part.len() - 1];
                        let args: Vec<PyExpr> = if args_str.trim().is_empty() {
                            vec![]
                        } else {
                            // Use depth-aware comma splitting for method arguments
                            let mut arg_strings = Vec::new();
                            let mut current = String::new();
                            let mut depth = 0;
                            let mut in_string = false;
                            let mut string_char = ' ';

                            for ch in args_str.chars() {
                                match ch {
                                    '"' | '\'' if !in_string => {
                                        in_string = true;
                                        string_char = ch;
                                        current.push(ch);
                                    }
                                    c if in_string && c == string_char => {
                                        in_string = false;
                                        current.push(ch);
                                    }
                                    '(' | '[' | '{' if !in_string => {
                                        depth += 1;
                                        current.push(ch);
                                    }
                                    ')' | ']' | '}' if !in_string => {
                                        depth -= 1;
                                        current.push(ch);
                                    }
                                    ',' if depth == 0 && !in_string => {
                                        arg_strings.push(current.trim().to_string());
                                        current.clear();
                                    }
                                    _ => current.push(ch),
                                }
                            }

                            if !current.trim().is_empty() {
                                arg_strings.push(current.trim().to_string());
                            }

                            arg_strings
                                .iter()
                                .map(|arg| self.parse_expr(arg))
                                .collect::<Result<Vec<_>>>()?
                        };

                        // Build the nested attribute chain
                        let mut base = self.parse_expr(value_str)?;

                        // If there are nested attributes, build them up
                        if !nested_attrs.is_empty() {
                            for attr in nested_attrs.split('.') {
                                base = PyExpr::Attribute {
                                    value: Box::new(base),
                                    attr: attr.to_string(),
                                };
                            }
                        }

                        // Add the final method name
                        let method_attr = PyExpr::Attribute {
                            value: Box::new(base),
                            attr: method_name.to_string(),
                        };

                        return Ok(PyExpr::Call {
                            func: Box::new(method_attr),
                            args,
                            kwargs: std::collections::HashMap::new(),
                        });
                    }
                }
            } else {
                // Simple attribute access (no method call) - can be nested like os.path
                let value = self.parse_expr(value_str)?;

                // Handle nested attributes by splitting on dots
                let attrs: Vec<&str> = attr_part.split('.').collect();
                let mut result = value;

                for attr in attrs {
                    if attr.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        result = PyExpr::Attribute {
                            value: Box::new(result),
                            attr: attr.to_string(),
                        };
                    } else {
                        return Err(Error::CodeGeneration(format!(
                            "Invalid attribute name: {}",
                            attr
                        )));
                    }
                }

                return Ok(result);
            }
        }

        // Implicit tuple (comma-separated values without parentheses)
        // e.g., "i, item" or "1, 2, 3"
        // Only if there's a comma and we're not inside parentheses/brackets
        if s.contains(',') && !s.starts_with('(') {
            // Check for commas at depth 0 (not inside nested structures)
            let mut depth = 0;
            let mut has_top_level_comma = false;

            for ch in s.chars() {
                match ch {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => depth -= 1,
                    ',' if depth == 0 => {
                        has_top_level_comma = true;
                        break;
                    }
                    _ => {}
                }
            }

            if has_top_level_comma {
                // Split by comma at top level only
                let mut elements = Vec::new();
                let mut current = String::new();
                let mut depth = 0;

                for ch in s.chars() {
                    match ch {
                        '(' | '[' | '{' => {
                            depth += 1;
                            current.push(ch);
                        }
                        ')' | ']' | '}' => {
                            depth -= 1;
                            current.push(ch);
                        }
                        ',' if depth == 0 => {
                            elements.push(current.trim().to_string());
                            current.clear();
                        }
                        _ => current.push(ch),
                    }
                }

                // Don't forget the last element
                if !current.trim().is_empty() {
                    elements.push(current.trim().to_string());
                }

                // Parse each element
                let parsed_elements: Result<Vec<_>> = elements
                    .iter()
                    .map(|e| self.parse_expr(e))
                    .collect();

                return Ok(PyExpr::Tuple(parsed_elements?));
            }
        }

        // Variable name
        if s.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Ok(PyExpr::Name(s.to_string()));
        }

        Err(Error::CodeGeneration(format!(
            "Unable to parse expression: {}",
            s
        )))
    }

    fn parse_list_comprehension(&self, content: &str) -> Result<PyExpr> {
        // Parse: element for var in iterable [if condition]
        // Example: x * 2 for x in range(10) if x > 5

        // Find the first " for " (outside any parentheses)
        let for_pos = if let Some(pos) = content.find(" for ") {
            pos
        } else {
            return Err(Error::CodeGeneration("Invalid list comprehension: missing 'for'".to_string()));
        };

        let element_str = content[..for_pos].trim();
        let rest = content[for_pos + 5..].trim(); // Skip " for "

        // Find " in "
        let in_pos = if let Some(pos) = rest.find(" in ") {
            pos
        } else {
            return Err(Error::CodeGeneration("Invalid list comprehension: missing 'in'".to_string()));
        };

        let target = rest[..in_pos].trim().to_string();
        let after_in = rest[in_pos + 4..].trim(); // Skip " in "

        // Check for " if " condition
        let (iter_str, ifs) = if let Some(if_pos) = after_in.find(" if ") {
            let iter_part = after_in[..if_pos].trim();
            let condition_str = after_in[if_pos + 4..].trim();
            let condition = self.parse_expr(condition_str)?;
            (iter_part, vec![condition])
        } else {
            (after_in, vec![])
        };

        let element = self.parse_expr(element_str)?;
        let iter = self.parse_expr(iter_str)?;

        Ok(PyExpr::ListComp {
            element: Box::new(element),
            generators: vec![Comprehension {
                target: PyExpr::Name(target),
                iter,
                ifs,
            }],
        })
    }

    fn parse_slice(&self, value: PyExpr, slice_str: &str) -> Result<PyExpr> {
        // Parse slice notation: [lower:upper:step]
        // Examples: [1:3], [:5], [2:], [::2], [1:10:2]

        let parts: Vec<&str> = slice_str.split(':').collect();

        let lower = if parts[0].trim().is_empty() {
            None
        } else {
            Some(Box::new(self.parse_expr(parts[0].trim())?))
        };

        let upper = if parts.len() > 1 && !parts[1].trim().is_empty() {
            Some(Box::new(self.parse_expr(parts[1].trim())?))
        } else {
            None
        };

        let step = if parts.len() > 2 && !parts[2].trim().is_empty() {
            Some(Box::new(self.parse_expr(parts[2].trim())?))
        } else {
            None
        };

        Ok(PyExpr::Slice {
            value: Box::new(value),
            lower,
            upper,
            step,
        })
    }

    fn parse_lambda(&self, s: &str) -> Result<PyExpr> {
        // Parse: lambda args: body
        // Examples: lambda x: x + 1, lambda x, y: x + y, lambda: 42

        let rest = &s[7..].trim(); // Skip "lambda "

        let colon_pos = if let Some(pos) = rest.find(':') {
            pos
        } else {
            return Err(Error::CodeGeneration("Invalid lambda: missing ':'".to_string()));
        };

        let args_str = rest[..colon_pos].trim();
        let body_str = rest[colon_pos + 1..].trim();

        // Parse arguments
        let args: Vec<String> = if args_str.is_empty() {
            vec![]
        } else {
            args_str.split(',').map(|a| a.trim().to_string()).collect()
        };

        // Parse body
        let body = self.parse_expr(body_str)?;

        Ok(PyExpr::Lambda {
            args,
            body: Box::new(body),
        })
    }

    fn parse_try_except(&mut self, expected_indent: usize) -> Result<Option<PyStmt>> {
        self.current_line += 1; // Skip "try:"

        // Parse try body
        let body = self.parse_block(expected_indent + 1)?;

        let mut handlers = vec![];
        let mut orelse = vec![];
        let mut finalbody = vec![];

        // Parse except clauses and optional else/finally
        while self.current_line < self.lines.len() {
            let line = &self.lines[self.current_line];
            if line.indent_level != expected_indent {
                break;
            }

            if line.content.starts_with("except") {
                self.current_line += 1;

                // Parse: except ExceptionType as name:
                let except_str = line.content[6..line.content.len() - 1].trim();
                let (exc_type, name) = if except_str.is_empty() {
                    // bare except
                    (None, None)
                } else if let Some(as_pos) = except_str.find(" as ") {
                    let exc = Some(PyExpr::Name(except_str[..as_pos].trim().to_string()));
                    let var = Some(except_str[as_pos + 4..].trim().to_string());
                    (exc, var)
                } else {
                    (Some(PyExpr::Name(except_str.to_string())), None)
                };

                let handler_body = self.parse_block(expected_indent + 1)?;
                handlers.push(ExceptHandler {
                    exception_type: exc_type,
                    name,
                    body: handler_body,
                });
            } else if line.content == "else:" {
                self.current_line += 1;
                orelse = self.parse_block(expected_indent + 1)?;
            } else if line.content == "finally:" {
                self.current_line += 1;
                finalbody = self.parse_block(expected_indent + 1)?;
                break; // finally is always last
            } else {
                break;
            }
        }

        Ok(Some(PyStmt::Try {
            body,
            handlers,
            orelse,
            finalbody,
        }))
    }

    fn parse_with_statement(&mut self, expected_indent: usize) -> Result<Option<PyStmt>> {
        let line = &self.lines[self.current_line].clone();

        // Parse: with context_expr [as var]:
        let with_content = &line.content[5..line.content.len() - 1].trim();

        // Parse with items (can have multiple: with open(f1) as f, open(f2) as g:)
        let mut items = Vec::new();

        // Simple implementation: split by comma at depth 0
        let mut item_strings = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for ch in with_content.chars() {
            match ch {
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    item_strings.push(current.trim().to_string());
                    current.clear();
                }
                _ => current.push(ch),
            }
        }

        if !current.trim().is_empty() {
            item_strings.push(current.trim().to_string());
        }

        // Parse each item
        for item_str in item_strings {
            let (context_expr_str, optional_vars) = if let Some(as_pos) = item_str.find(" as ") {
                let expr = item_str[..as_pos].trim();
                let var = Some(PyExpr::Name(item_str[as_pos + 4..].trim().to_string()));
                (expr, var)
            } else {
                (item_str.as_str(), None)
            };

            let context_expr = self.parse_expr(context_expr_str)?;
            items.push(WithItem {
                context_expr,
                optional_vars,
            });
        }

        self.current_line += 1;

        // Parse body
        let body = self.parse_block(expected_indent + 1)?;

        Ok(Some(PyStmt::With { items, body }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_if_statement() {
        let source = r#"
if x > 0:
    print("positive")
"#;
        let mut parser = IndentedPythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::If { test, body, .. } => {
                assert_eq!(body.len(), 1);
            }
            _ => panic!("Expected if statement"),
        }
    }

    #[test]
    fn test_if_else() {
        let source = r#"
if x > 0:
    y = 1
else:
    y = -1
"#;
        let mut parser = IndentedPythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::If { body, orelse, .. } => {
                assert_eq!(body.len(), 1);
                assert_eq!(orelse.len(), 1);
            }
            _ => panic!("Expected if statement"),
        }
    }

    #[test]
    fn test_for_loop() {
        let source = r#"
for i in range(10):
    print(i)
"#;
        let mut parser = IndentedPythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::For { target, body, .. } => {
                assert_eq!(*target, PyExpr::Name("i".to_string()));
                assert_eq!(body.len(), 1);
            }
            _ => panic!("Expected for loop"),
        }
    }

    #[test]
    fn test_while_loop() {
        let source = r#"
while x < 10:
    x = x + 1
"#;
        let mut parser = IndentedPythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::While { test, body, .. } => {
                assert_eq!(body.len(), 1);
            }
            _ => panic!("Expected while loop"),
        }
    }

    #[test]
    fn test_function_definition() {
        let source = r#"
def add(a, b):
    return a + b
"#;
        let mut parser = IndentedPythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::FunctionDef { name, params, body, .. } => {
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
                assert_eq!(body.len(), 1);
            }
            _ => panic!("Expected function definition"),
        }
    }

    #[test]
    fn test_nested_blocks() {
        let source = r#"
if x > 0:
    if y > 0:
        print("both positive")
    else:
        print("x positive, y not")
"#;
        let mut parser = IndentedPythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::If { body, .. } => {
                assert_eq!(body.len(), 1);
                // Inner if statement
                match &body[0] {
                    PyStmt::If { orelse, .. } => {
                        assert_eq!(orelse.len(), 1);
                    }
                    _ => panic!("Expected nested if"),
                }
            }
            _ => panic!("Expected if statement"),
        }
    }

    #[test]
    fn test_import_statement() {
        let source = "import math";
        let mut parser = IndentedPythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::Import { modules } => {
                assert_eq!(modules.len(), 1);
                assert_eq!(modules[0].0, "math");
            }
            _ => panic!("Expected import statement"),
        }
    }

    #[test]
    fn test_from_import_statement() {
        let source = "from os import path";
        let mut parser = IndentedPythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::ImportFrom { module, names, .. } => {
                assert_eq!(module, &Some("os".to_string()));
                assert_eq!(names.len(), 1);
                assert_eq!(names[0].0, "path");
            }
            _ => panic!("Expected from-import statement"),
        }
    }
}

impl IndentedPythonParser {
    fn parse_import_statement(&mut self) -> Result<Option<PyStmt>> {
        let line = &self.lines[self.current_line].clone();
        self.current_line += 1;

        // Parse "import module1 [as alias1], module2 [as alias2]"
        let import_str = line.content[7..].trim(); // Skip "import "
        let mut names = Vec::new();
        let mut aliases = Vec::new();

        for part in import_str.split(',') {
            let part = part.trim();
            if part.contains(" as ") {
                let parts: Vec<&str> = part.split(" as ").collect();
                if parts.len() == 2 {
                    names.push(parts[0].trim().to_string());
                    aliases.push(Some(parts[1].trim().to_string()));
                }
            } else {
                names.push(part.to_string());
                aliases.push(None);
            }
        }

        // Combine names and aliases into modules: Vec<(String, Option<String>)>
        let modules: Vec<(String, Option<String>)> = names.into_iter().zip(aliases.into_iter()).collect();
        Ok(Some(PyStmt::Import { modules }))
    }

    fn parse_from_import_statement(&mut self) -> Result<Option<PyStmt>> {
        let line = &self.lines[self.current_line].clone();
        self.current_line += 1;

        // Parse "from module import name1 [as alias1], name2 [as alias2]"
        let parts: Vec<&str> = line.content.split(" import ").collect();
        if parts.len() != 2 {
            #[cfg(target_arch = "wasm32")]
            return Err(Error::Parse(format!(
                "Invalid from-import statement: {}",
                line.content
            )));

            #[cfg(not(target_arch = "wasm32"))]
            return Err(Error::CodeGeneration(format!(
                "Invalid from-import statement: {}",
                line.content
            )));
        }

        let module = parts[0][5..].trim().to_string(); // Skip "from "
        let mut names = Vec::new();
        let mut aliases = Vec::new();

        for part in parts[1].split(',') {
            let part = part.trim();
            if part.contains(" as ") {
                let alias_parts: Vec<&str> = part.split(" as ").collect();
                if alias_parts.len() == 2 {
                    names.push(alias_parts[0].trim().to_string());
                    aliases.push(Some(alias_parts[1].trim().to_string()));
                }
            } else {
                names.push(part.to_string());
                aliases.push(None);
            }
        }

        // Combine names and aliases into names: Vec<(String, Option<String>)>
        let name_pairs: Vec<(String, Option<String>)> = names.into_iter().zip(aliases.into_iter()).collect();
        Ok(Some(PyStmt::ImportFrom {
            module: Some(module),
            names: name_pairs,
            level: 0  // Direct import, not relative
        }))
    }
}
