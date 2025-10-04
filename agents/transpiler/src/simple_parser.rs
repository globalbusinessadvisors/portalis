//! Simple Python parser for Low complexity features
//!
//! This is a simplified parser that handles the first 10 Low complexity features:
//! 1. Integer literals
//! 2. Float literals
//! 3. String literals
//! 4. Boolean literals
//! 5. Simple assignment
//! 6. Multiple assignment
//! 7. Augmented assignment
//! 8. Print function
//! 9. Comments
//! 10. Function docstrings

use crate::python_ast::*;
use crate::{Error, Result};

pub struct SimplePythonParser {
    lines: Vec<String>,
    current_line: usize,
}

impl SimplePythonParser {
    pub fn new(source: &str) -> Self {
        let lines: Vec<String> = source.lines().map(|s| s.to_string()).collect();
        Self {
            lines,
            current_line: 0,
        }
    }

    pub fn parse(&mut self) -> Result<PyModule> {
        let mut module = PyModule::new();

        while self.current_line < self.lines.len() {
            let line = self.lines[self.current_line].trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                self.current_line += 1;
                continue;
            }

            // Parse statement
            if let Some(stmt) = self.parse_statement(line)? {
                module.add_stmt(stmt);
            }

            self.current_line += 1;
        }

        Ok(module)
    }

    fn parse_statement(&self, line: &str) -> Result<Option<PyStmt>> {
        // If statement: if condition:
        if line.starts_with("if ") && line.ends_with(':') {
            let condition_str = &line[3..line.len() - 1].trim();
            let condition = self.parse_expr(condition_str)?;

            return Ok(Some(PyStmt::If {
                test: condition,
                body: vec![], // Will be filled by multi-line parser
                orelse: vec![],
            }));
        }

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

        // Simple assignment: x = 42
        if let Some(pos) = line.find('=') {
            // Check it's not ==, !=, <=, >=
            if pos > 0 && line.len() > pos + 1 {
                let before = line.chars().nth(pos.saturating_sub(1));
                let after = line.chars().nth(pos + 1);

                if !matches!(before, Some('=') | Some('!') | Some('<') | Some('>'))
                    && !matches!(after, Some('='))
                {
                    let target = line[..pos].trim().to_string();
                    let value_str = line[pos + 1..].trim();

                    // Check for augmented assignment
                    if let Some(op_char) = target.chars().last() {
                        if matches!(op_char, '+' | '-' | '*' | '/' | '%' | '&' | '|' | '^') {
                            let actual_target = target[..target.len() - 1].trim().to_string();
                            let op = match op_char {
                                '+' => BinOp::Add,
                                '-' => BinOp::Sub,
                                '*' => BinOp::Mult,
                                '/' => BinOp::Div,
                                '%' => BinOp::Mod,
                                '&' => BinOp::BitAnd,
                                '|' => BinOp::BitOr,
                                '^' => BinOp::BitXor,
                                _ => unreachable!(),
                            };
                            let value = self.parse_expr(value_str)?;

                            return Ok(Some(PyStmt::AugAssign {
                                target: PyExpr::Name(actual_target),
                                op,
                                value,
                            }));
                        }
                    }

                    // Simple assignment
                    let value = self.parse_expr(value_str)?;
                    return Ok(Some(PyStmt::Assign {
                        target: PyExpr::Name(target),
                        value,
                    }));
                }
            }
        }

        // Print function: print(...)
        if line.starts_with("print(") && line.ends_with(')') {
            let args_str = &line[6..line.len() - 1];
            let expr = self.parse_expr(args_str)?;

            return Ok(Some(PyStmt::Expr(PyExpr::Call {
                func: Box::new(PyExpr::Name("print".to_string())),
                args: vec![expr],
                kwargs: std::collections::HashMap::new(),
            })));
        }

        Ok(None)
    }

    fn parse_expr(&self, s: &str) -> Result<PyExpr> {
        let s = s.trim();

        // Boolean literals
        if s == "True" {
            return Ok(PyExpr::Literal(PyLiteral::Bool(true)));
        }
        if s == "False" {
            return Ok(PyExpr::Literal(PyLiteral::Bool(false)));
        }

        // None literal
        if s == "None" {
            return Ok(PyExpr::Literal(PyLiteral::None));
        }

        // List literals: [1, 2, 3]
        if s.starts_with('[') && s.ends_with(']') {
            let content = &s[1..s.len() - 1].trim();
            if content.is_empty() {
                return Ok(PyExpr::List(vec![]));
            }
            let elements: Result<Vec<_>> = content
                .split(',')
                .map(|e| self.parse_expr(e.trim()))
                .collect();
            return Ok(PyExpr::List(elements?));
        }

        // Tuple literals: (1, 2, 3)
        if s.starts_with('(') && s.ends_with(')') {
            let content = &s[1..s.len() - 1].trim();
            if content.is_empty() {
                return Ok(PyExpr::Tuple(vec![]));
            }
            let elements: Result<Vec<_>> = content
                .split(',')
                .map(|e| self.parse_expr(e.trim()))
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

        // Float literals (must check before int - has decimal point)
        if s.contains('.') && s.parse::<f64>().is_ok() {
            let value = s.parse::<f64>().unwrap();
            return Ok(PyExpr::Literal(PyLiteral::Float(value)));
        }

        // Integer literals
        if let Ok(value) = s.parse::<i64>() {
            return Ok(PyExpr::Literal(PyLiteral::Int(value)));
        }

        // Logical operators (must check before comparison)
        for (op_str, python_op) in &[
            (" and ", "&&"),
            (" or ", "||"),
        ] {
            if let Some(pos) = s.find(op_str) {
                let left = self.parse_expr(&s[..pos])?;
                let right = self.parse_expr(&s[pos + op_str.len()..])?;
                // Store as BinOp for now, will be handled specially
                let op = if *python_op == "&&" {
                    BinOp::BitAnd // Reuse for logical and
                } else {
                    BinOp::BitOr // Reuse for logical or
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
            if let Some(pos) = s.find(op_str) {
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

        // Binary arithmetic operations
        for (op_str, op) in &[
            (" + ", BinOp::Add),
            (" - ", BinOp::Sub),
            (" * ", BinOp::Mult),
            (" / ", BinOp::Div),
            (" % ", BinOp::Mod),
        ] {
            if let Some(pos) = s.find(op_str) {
                let left = self.parse_expr(&s[..pos])?;
                let right = self.parse_expr(&s[pos + op_str.len()..])?;
                return Ok(PyExpr::BinOp {
                    left: Box::new(left),
                    op: *op,
                    right: Box::new(right),
                });
            }
        }

        // List indexing: list[index]
        if let Some(bracket_pos) = s.find('[') {
            if s.ends_with(']') {
                let value_str = &s[..bracket_pos];
                let index_str = &s[bracket_pos + 1..s.len() - 1];
                let value = self.parse_expr(value_str)?;
                let index = self.parse_expr(index_str)?;
                return Ok(PyExpr::Subscript {
                    value: Box::new(value),
                    index: Box::new(index),
                });
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_integer_literal() {
        let source = "x = 42";
        let mut parser = SimplePythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::Assign { target, value } => {
                assert_eq!(*target, PyExpr::Name("x".to_string()));
                assert_eq!(*value, PyExpr::Literal(PyLiteral::Int(42)));
            }
            _ => panic!("Expected assignment"),
        }
    }

    #[test]
    fn test_parse_float_literal() {
        let source = "pi = 3.14";
        let mut parser = SimplePythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::Assign { value, .. } => {
                assert_eq!(*value, PyExpr::Literal(PyLiteral::Float(3.14)));
            }
            _ => panic!("Expected assignment"),
        }
    }

    #[test]
    fn test_parse_string_literal() {
        let source = r#"msg = "hello""#;
        let mut parser = SimplePythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::Assign { value, .. } => {
                assert_eq!(
                    *value,
                    PyExpr::Literal(PyLiteral::String("hello".to_string()))
                );
            }
            _ => panic!("Expected assignment"),
        }
    }

    #[test]
    fn test_parse_boolean_literal() {
        let source = "flag = True";
        let mut parser = SimplePythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::Assign { value, .. } => {
                assert_eq!(*value, PyExpr::Literal(PyLiteral::Bool(true)));
            }
            _ => panic!("Expected assignment"),
        }
    }

    #[test]
    fn test_parse_augmented_assignment() {
        let source = "x += 5";
        let mut parser = SimplePythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::AugAssign { target, op, value } => {
                assert_eq!(*target, PyExpr::Name("x".to_string()));
                assert_eq!(*op, BinOp::Add);
                assert_eq!(*value, PyExpr::Literal(PyLiteral::Int(5)));
            }
            _ => panic!("Expected augmented assignment"),
        }
    }

    #[test]
    fn test_parse_print() {
        let source = r#"print("hello")"#;
        let mut parser = SimplePythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::Expr(PyExpr::Call { func, args, .. }) => {
                assert_eq!(**func, PyExpr::Name("print".to_string()));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected print call"),
        }
    }

    #[test]
    fn test_skip_comments() {
        let source = "# This is a comment\nx = 42\n# Another comment";
        let mut parser = SimplePythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1); // Only the assignment, comments skipped
    }

    #[test]
    fn test_parse_binary_operation() {
        let source = "result = 2 + 3";
        let mut parser = SimplePythonParser::new(source);
        let module = parser.parse().unwrap();

        assert_eq!(module.statements.len(), 1);
        match &module.statements[0] {
            PyStmt::Assign { value, .. } => match value {
                PyExpr::BinOp { left, op, right } => {
                    assert_eq!(**left, PyExpr::Literal(PyLiteral::Int(2)));
                    assert_eq!(*op, BinOp::Add);
                    assert_eq!(**right, PyExpr::Literal(PyLiteral::Int(3)));
                }
                _ => panic!("Expected binary operation"),
            },
            _ => panic!("Expected assignment"),
        }
    }
}
