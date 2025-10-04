//! Integration tests for the Python parser
//!
//! These tests verify that the Python parser correctly parses real Python code
//! and converts it to our internal AST representation.

use portalis_transpiler::python_parser::PythonParser;
use portalis_transpiler::python_ast::*;

#[test]
fn test_parse_simple_function() {
    let source = r#"
def add(a, b):
    return a + b
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    assert_eq!(module.statements.len(), 1);

    match &module.statements[0] {
        PyStmt::FunctionDef { name, params, body, .. } => {
            assert_eq!(name, "add");
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "a");
            assert_eq!(params[1].name, "b");
            assert_eq!(body.len(), 1);

            // Check return statement
            match &body[0] {
                PyStmt::Return { value } => {
                    assert!(value.is_some());
                }
                _ => panic!("Expected return statement"),
            }
        }
        _ => panic!("Expected function definition"),
    }
}

#[test]
fn test_parse_function_with_type_annotations() {
    let source = r#"
def multiply(x: int, y: int) -> int:
    return x * y
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::FunctionDef {
            name,
            params,
            return_type,
            ..
        } => {
            assert_eq!(name, "multiply");
            assert_eq!(params.len(), 2);

            // Check type annotations
            assert!(params[0].type_annotation.is_some());
            assert!(params[1].type_annotation.is_some());
            assert!(return_type.is_some());
        }
        _ => panic!("Expected function definition"),
    }
}

#[test]
fn test_parse_class_definition() {
    let source = r#"
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    assert_eq!(module.statements.len(), 1);

    match &module.statements[0] {
        PyStmt::ClassDef { name, body, .. } => {
            assert_eq!(name, "Calculator");
            assert_eq!(body.len(), 2); // Two methods

            // Check both are function definitions
            for stmt in body {
                assert!(matches!(stmt, PyStmt::FunctionDef { .. }));
            }
        }
        _ => panic!("Expected class definition"),
    }
}

#[test]
fn test_parse_if_statement() {
    let source = r#"
if x > 0:
    print("positive")
elif x < 0:
    print("negative")
else:
    print("zero")
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    assert_eq!(module.statements.len(), 1);

    match &module.statements[0] {
        PyStmt::If { test, body, orelse } => {
            assert!(matches!(*test, PyExpr::Compare { .. }));
            assert!(!body.is_empty());
            assert!(!orelse.is_empty()); // Should have elif/else
        }
        _ => panic!("Expected if statement"),
    }
}

#[test]
fn test_parse_for_loop() {
    let source = r#"
for i in range(10):
    print(i)
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::For { target, iter, body, .. } => {
            assert!(matches!(*target, PyExpr::Name(_)));
            assert!(matches!(*iter, PyExpr::Call { .. }));
            assert!(!body.is_empty());
        }
        _ => panic!("Expected for loop"),
    }
}

#[test]
fn test_parse_while_loop() {
    let source = r#"
while x < 10:
    x = x + 1
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::While { test, body, .. } => {
            assert!(matches!(*test, PyExpr::Compare { .. }));
            assert!(!body.is_empty());
        }
        _ => panic!("Expected while loop"),
    }
}

#[test]
fn test_parse_list_comprehension() {
    let source = "[x * 2 for x in range(10)]";

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    match expr {
        PyExpr::ListComp { element, generators } => {
            assert!(matches!(*element, PyExpr::BinOp { .. }));
            assert_eq!(generators.len(), 1);
        }
        _ => panic!("Expected list comprehension"),
    }
}

#[test]
fn test_parse_try_except() {
    let source = r#"
try:
    risky_operation()
except ValueError as e:
    print(e)
except Exception:
    print("Unknown error")
finally:
    cleanup()
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::Try { body, handlers, finalbody, .. } => {
            assert!(!body.is_empty());
            assert_eq!(handlers.len(), 2);
            assert!(!finalbody.is_empty());

            // Check first handler has exception type and name
            assert!(handlers[0].exception_type.is_some());
            assert!(handlers[0].name.is_some());

            // Check second handler has exception type but no name
            assert!(handlers[1].exception_type.is_some());
            assert!(handlers[1].name.is_none());
        }
        _ => panic!("Expected try-except statement"),
    }
}

#[test]
fn test_parse_with_statement() {
    let source = r#"
with open("file.txt") as f:
    content = f.read()
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::With { items, body } => {
            assert_eq!(items.len(), 1);
            assert!(items[0].optional_vars.is_some());
            assert!(!body.is_empty());
        }
        _ => panic!("Expected with statement"),
    }
}

#[test]
fn test_parse_import_statements() {
    let source = r#"
import sys
from os import path
from collections import Counter as C
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    assert_eq!(module.statements.len(), 3);

    // Check import
    match &module.statements[0] {
        PyStmt::Import { modules } => {
            assert_eq!(modules.len(), 1);
            assert_eq!(modules[0].0, "sys");
            assert!(modules[0].1.is_none());
        }
        _ => panic!("Expected import statement"),
    }

    // Check from import
    match &module.statements[1] {
        PyStmt::ImportFrom { module, names, .. } => {
            assert_eq!(module.as_ref().unwrap(), "os");
            assert_eq!(names.len(), 1);
            assert_eq!(names[0].0, "path");
        }
        _ => panic!("Expected from import statement"),
    }

    // Check from import with alias
    match &module.statements[2] {
        PyStmt::ImportFrom { names, .. } => {
            assert_eq!(names[0].0, "Counter");
            assert_eq!(names[0].1.as_ref().unwrap(), "C");
        }
        _ => panic!("Expected from import with alias"),
    }
}

#[test]
fn test_parse_async_function() {
    let source = r#"
async def fetch_data(url):
    result = await get_data(url)
    return result
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::FunctionDef { name, is_async, body, .. } => {
            assert_eq!(name, "fetch_data");
            assert!(is_async);

            // Check for await expression in body
            let has_await = body.iter().any(|stmt| {
                matches!(stmt, PyStmt::Assign { value: PyExpr::Await(_), .. })
            });
            assert!(has_await);
        }
        _ => panic!("Expected async function definition"),
    }
}

#[test]
fn test_parse_decorators() {
    let source = r#"
@decorator
@another_decorator(arg1, arg2)
def decorated_function():
    pass
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::FunctionDef { decorators, .. } => {
            assert_eq!(decorators.len(), 2);
            assert!(matches!(decorators[0], PyExpr::Name(_)));
            assert!(matches!(decorators[1], PyExpr::Call { .. }));
        }
        _ => panic!("Expected decorated function"),
    }
}

#[test]
fn test_parse_lambda() {
    let source = "lambda x, y: x + y";

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    match expr {
        PyExpr::Lambda { args, body } => {
            assert_eq!(args.len(), 2);
            assert!(matches!(*body, PyExpr::BinOp { .. }));
        }
        _ => panic!("Expected lambda expression"),
    }
}

#[test]
fn test_parse_dict_literal() {
    let source = r#"{"key1": "value1", "key2": "value2"}"#;

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    match expr {
        PyExpr::Dict { keys, values } => {
            assert_eq!(keys.len(), 2);
            assert_eq!(values.len(), 2);
        }
        _ => panic!("Expected dict literal"),
    }
}

#[test]
fn test_parse_set_literal() {
    let source = "{1, 2, 3, 4, 5}";

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    match expr {
        PyExpr::Set(elements) => {
            assert_eq!(elements.len(), 5);
        }
        _ => panic!("Expected set literal"),
    }
}

#[test]
fn test_parse_conditional_expression() {
    let source = "x if condition else y";

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    match expr {
        PyExpr::IfExp { test, body, orelse } => {
            assert!(matches!(*test, PyExpr::Name(_)));
            assert!(matches!(*body, PyExpr::Name(_)));
            assert!(matches!(*orelse, PyExpr::Name(_)));
        }
        _ => panic!("Expected conditional expression"),
    }
}

#[test]
fn test_parse_attribute_access() {
    let source = "obj.attr.nested";

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    match expr {
        PyExpr::Attribute { value, attr } => {
            assert_eq!(attr, "nested");
            assert!(matches!(*value, PyExpr::Attribute { .. }));
        }
        _ => panic!("Expected attribute access"),
    }
}

#[test]
fn test_parse_subscript() {
    let source = "list[0]";

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    match expr {
        PyExpr::Subscript { value, index } => {
            assert!(matches!(*value, PyExpr::Name(_)));
            assert!(matches!(*index, PyExpr::Literal(PyLiteral::Int(0))));
        }
        _ => panic!("Expected subscript"),
    }
}

#[test]
fn test_parse_slice() {
    let source = "list[1:10:2]";

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    match expr {
        PyExpr::Slice { value, lower, upper, step } => {
            assert!(matches!(*value, PyExpr::Name(_)));
            assert!(lower.is_some());
            assert!(upper.is_some());
            assert!(step.is_some());
        }
        _ => panic!("Expected slice"),
    }
}

#[test]
fn test_parse_boolean_operations() {
    let source = "a and b or c";

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    // Should parse as (a and b) or c
    match expr {
        PyExpr::BoolOp { op, .. } => {
            assert_eq!(op, BoolOp::Or);
        }
        _ => panic!("Expected boolean operation"),
    }
}

#[test]
fn test_parse_comparison_chain() {
    let source = "a < b";

    let parser = PythonParser::new(source, "test.py");
    let expr = parser.parse_expression().expect("Failed to parse");

    match expr {
        PyExpr::Compare { left, op, right } => {
            assert!(matches!(*left, PyExpr::Name(_)));
            assert_eq!(op, CmpOp::Lt);
            assert!(matches!(*right, PyExpr::Name(_)));
        }
        _ => panic!("Expected comparison"),
    }
}

#[test]
fn test_parse_augmented_assignment() {
    let source = "x += 1";

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::AugAssign { target, op, value } => {
            assert!(matches!(*target, PyExpr::Name(_)));
            assert_eq!(*op, BinOp::Add);
            assert!(matches!(*value, PyExpr::Literal(PyLiteral::Int(1))));
        }
        _ => panic!("Expected augmented assignment"),
    }
}

#[test]
fn test_parse_annotated_assignment() {
    let source = "x: int = 42";

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::AnnAssign { target, annotation, value } => {
            assert!(matches!(*target, PyExpr::Name(_)));
            assert!(matches!(annotation, TypeAnnotation::Name(_)));
            assert!(value.is_some());
        }
        _ => panic!("Expected annotated assignment"),
    }
}

#[test]
fn test_parse_assert_statement() {
    let source = r#"assert x > 0, "x must be positive""#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::Assert { test, msg } => {
            assert!(matches!(*test, PyExpr::Compare { .. }));
            assert!(msg.is_some());
        }
        _ => panic!("Expected assert statement"),
    }
}

#[test]
fn test_parse_raise_statement() {
    let source = "raise ValueError('Invalid value')";

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::Raise { exception } => {
            assert!(exception.is_some());
        }
        _ => panic!("Expected raise statement"),
    }
}

#[test]
fn test_parse_delete_statement() {
    let source = "del x, y, z";

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::Delete { targets } => {
            assert_eq!(targets.len(), 3);
        }
        _ => panic!("Expected delete statement"),
    }
}

#[test]
fn test_parse_global_statement() {
    let source = "global x, y";

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::Global { names } => {
            assert_eq!(names.len(), 2);
            assert_eq!(names[0], "x");
            assert_eq!(names[1], "y");
        }
        _ => panic!("Expected global statement"),
    }
}

#[test]
fn test_parse_nonlocal_statement() {
    let source = "nonlocal x, y";

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse");

    match &module.statements[0] {
        PyStmt::Nonlocal { names } => {
            assert_eq!(names.len(), 2);
            assert_eq!(names[0], "x");
            assert_eq!(names[1], "y");
        }
        _ => panic!("Expected nonlocal statement"),
    }
}

#[test]
fn test_parse_complex_real_world_code() {
    let source = r#"
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []

    async def process_items(self, items):
        results = []
        for item in items:
            try:
                result = await self.process_one(item)
                results.append(result)
            except Exception as e:
                print(f"Error processing {item}: {e}")
                continue
        return results

    def process_one(self, item):
        if item is None:
            raise ValueError("Item cannot be None")

        with self.get_context() as ctx:
            return ctx.transform(item)
"#;

    let parser = PythonParser::new(source, "test.py");
    let module = parser.parse().expect("Failed to parse complex code");

    // Should successfully parse the entire class
    assert_eq!(module.statements.len(), 1);

    match &module.statements[0] {
        PyStmt::ClassDef { name, body, .. } => {
            assert_eq!(name, "DataProcessor");
            assert_eq!(body.len(), 3); // __init__, process_items, process_one
        }
        _ => panic!("Expected class definition"),
    }
}
