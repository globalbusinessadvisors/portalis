//! Python AST types for translation to Rust
//!
//! Defines a simplified Python AST that can be easily translated to Rust.
//! Based on the 527 Python language features cataloged in PYTHON_LANGUAGE_FEATURES.md

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Python literal types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PyLiteral {
    /// Integer literal: 42, 0x2A, 0o52, 0b101010
    Int(i64),
    /// Float literal: 3.14, 1.0e10
    Float(f64),
    /// String literal: "hello", 'world', r"raw", f"formatted"
    String(String),
    /// Boolean literal: True, False
    Bool(bool),
    /// None literal
    None,
    /// Bytes literal: b"data"
    Bytes(Vec<u8>),
}

/// Python expression types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PyExpr {
    /// Literal value
    Literal(PyLiteral),
    /// Variable name
    Name(String),
    /// Binary operation: a + b, x * y
    BinOp {
        left: Box<PyExpr>,
        op: BinOp,
        right: Box<PyExpr>,
    },
    /// Unary operation: -x, not y
    UnaryOp {
        op: UnaryOp,
        operand: Box<PyExpr>,
    },
    /// Function call: func(args)
    Call {
        func: Box<PyExpr>,
        args: Vec<PyExpr>,
        kwargs: HashMap<String, PyExpr>,
    },
    /// Attribute access: obj.attr
    Attribute {
        value: Box<PyExpr>,
        attr: String,
    },
    /// Subscript: list[0], dict["key"]
    Subscript {
        value: Box<PyExpr>,
        index: Box<PyExpr>,
    },
    /// Slice: list[1:3], list[:5], list[2:], list[1:10:2]
    Slice {
        value: Box<PyExpr>,
        lower: Option<Box<PyExpr>>,
        upper: Option<Box<PyExpr>>,
        step: Option<Box<PyExpr>>,
    },
    /// List literal: [1, 2, 3]
    List(Vec<PyExpr>),
    /// Tuple literal: (1, 2, 3)
    Tuple(Vec<PyExpr>),
    /// Dict literal: {"a": 1, "b": 2}
    Dict {
        keys: Vec<PyExpr>,
        values: Vec<PyExpr>,
    },
    /// Set literal: {1, 2, 3}
    Set(Vec<PyExpr>),
    /// List comprehension: [x*2 for x in range(10)]
    ListComp {
        element: Box<PyExpr>,
        generators: Vec<Comprehension>,
    },
    /// Conditional expression: x if condition else y
    IfExp {
        test: Box<PyExpr>,
        body: Box<PyExpr>,
        orelse: Box<PyExpr>,
    },
    /// Lambda: lambda x: x + 1
    Lambda {
        args: Vec<String>,
        body: Box<PyExpr>,
    },
    /// Comparison: x == y, a < b
    Compare {
        left: Box<PyExpr>,
        op: CmpOp,
        right: Box<PyExpr>,
    },
    /// Await expression
    Await {
        value: Box<PyExpr>,
    },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,      // +
    Sub,      // -
    Mult,     // *
    Div,      // /
    FloorDiv, // //
    Mod,      // %
    Pow,      // **
    LShift,   // <<
    RShift,   // >>
    BitOr,    // |
    BitXor,   // ^
    BitAnd,   // &
    MatMult,  // @ (Python 3.5+)
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Invert, // ~
    Not,    // not
    UAdd,   // +
    USub,   // -
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    Eq,    // ==
    NotEq, // !=
    Lt,    // <
    LtE,   // <=
    Gt,    // >
    GtE,   // >=
    Is,    // is
    IsNot, // is not
    In,    // in
    NotIn, // not in
}

/// Comprehension clause: for x in iterable if condition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Comprehension {
    pub target: String,
    pub iter: PyExpr,
    pub ifs: Vec<PyExpr>,
}

/// Python statement types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PyStmt {
    /// Expression statement
    Expr(PyExpr),
    /// Assignment: x = 42 or chained: x = y = z = 42
    Assign {
        targets: Vec<String>,  // Support multiple targets for chained assignment
        value: PyExpr,
        type_hint: Option<String>,
    },
    /// Augmented assignment: x += 1
    AugAssign {
        target: String,
        op: BinOp,
        value: PyExpr,
    },
    /// Function definition
    FunctionDef {
        name: String,
        args: Vec<Arg>,
        body: Vec<PyStmt>,
        return_type: Option<String>,
        decorators: Vec<String>,
        is_async: bool,
    },
    /// Return statement
    Return(Option<PyExpr>),
    /// If statement
    If {
        test: PyExpr,
        body: Vec<PyStmt>,
        orelse: Vec<PyStmt>,
    },
    /// While loop
    While {
        test: PyExpr,
        body: Vec<PyStmt>,
        orelse: Vec<PyStmt>,
    },
    /// For loop (supports tuple unpacking: for i, item in ...)
    For {
        targets: Vec<String>,  // Can be ["i"] or ["i", "item"] for unpacking
        iter: PyExpr,
        body: Vec<PyStmt>,
        orelse: Vec<PyStmt>,
    },
    /// Pass statement
    Pass,
    /// Break statement
    Break,
    /// Continue statement
    Continue,
    /// Class definition
    ClassDef {
        name: String,
        bases: Vec<String>,
        body: Vec<PyStmt>,
        decorators: Vec<String>,
    },
    /// Import statement: import module [as alias]
    Import {
        names: Vec<String>,
        aliases: Vec<Option<String>>,
    },
    /// From import: from module import name [as alias]
    ImportFrom {
        module: String,
        names: Vec<String>,
        aliases: Vec<Option<String>>,
    },
    /// Assert statement: assert condition, message
    Assert {
        test: PyExpr,
        msg: Option<PyExpr>,
    },
    /// Try-except statement
    Try {
        body: Vec<PyStmt>,
        handlers: Vec<ExceptHandler>,
        orelse: Vec<PyStmt>,
        finalbody: Vec<PyStmt>,
    },
    /// Raise statement
    Raise {
        exc: Option<PyExpr>,
    },
    /// With statement (context manager)
    With {
        items: Vec<WithItem>,
        body: Vec<PyStmt>,
    },
}

/// With item for context managers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WithItem {
    pub context_expr: PyExpr,
    pub optional_vars: Option<String>,
}

/// Exception handler for try-except
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExceptHandler {
    pub exc_type: Option<String>,
    pub name: Option<String>,
    pub body: Vec<PyStmt>,
}

/// Function argument
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Arg {
    pub name: String,
    pub type_hint: Option<String>,
    pub default: Option<PyExpr>,
}

/// Python module (top-level)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PyModule {
    pub body: Vec<PyStmt>,
}

impl PyModule {
    pub fn new() -> Self {
        Self { body: vec![] }
    }

    pub fn add_stmt(&mut self, stmt: PyStmt) {
        self.body.push(stmt);
    }
}

impl Default for PyModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_creation() {
        let int_lit = PyLiteral::Int(42);
        assert_eq!(int_lit, PyLiteral::Int(42));

        let str_lit = PyLiteral::String("hello".to_string());
        assert_eq!(str_lit, PyLiteral::String("hello".to_string()));

        let bool_lit = PyLiteral::Bool(true);
        assert_eq!(bool_lit, PyLiteral::Bool(true));
    }

    #[test]
    fn test_simple_assignment() {
        let module = PyModule {
            body: vec![PyStmt::Assign {
                targets: vec!["x".to_string()],
                value: PyExpr::Literal(PyLiteral::Int(42)),
                type_hint: None,
            }],
        };

        assert_eq!(module.body.len(), 1);
    }

    #[test]
    fn test_binary_operation() {
        let expr = PyExpr::BinOp {
            left: Box::new(PyExpr::Literal(PyLiteral::Int(2))),
            op: BinOp::Add,
            right: Box::new(PyExpr::Literal(PyLiteral::Int(3))),
        };

        match expr {
            PyExpr::BinOp { left, op, right } => {
                assert_eq!(*left, PyExpr::Literal(PyLiteral::Int(2)));
                assert_eq!(op, BinOp::Add);
                assert_eq!(*right, PyExpr::Literal(PyLiteral::Int(3)));
            }
            _ => panic!("Expected BinOp"),
        }
    }

    #[test]
    fn test_function_definition() {
        let func = PyStmt::FunctionDef {
            name: "add".to_string(),
            args: vec![
                Arg {
                    name: "a".to_string(),
                    type_hint: Some("int".to_string()),
                    default: None,
                },
                Arg {
                    name: "b".to_string(),
                    type_hint: Some("int".to_string()),
                    default: None,
                },
            ],
            body: vec![PyStmt::Return(Some(PyExpr::BinOp {
                left: Box::new(PyExpr::Name("a".to_string())),
                op: BinOp::Add,
                right: Box::new(PyExpr::Name("b".to_string())),
            }))],
            return_type: Some("int".to_string()),
            decorators: vec![],
            is_async: false,
        };

        match func {
            PyStmt::FunctionDef { name, args, .. } => {
                assert_eq!(name, "add");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected FunctionDef"),
        }
    }
}
