//! Control Flow Analysis Module
//!
//! Analyzes Python control flow structures for advanced type inference and code generation.

use rustpython_parser::ast;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Control flow graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlFlowNode {
    pub id: usize,
    pub node_type: ControlFlowNodeType,
    pub successors: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlFlowNodeType {
    Entry,
    Exit,
    Statement,
    Condition,
    Loop,
    Return,
}

/// Variable usage tracking for type inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableUsage {
    pub name: String,
    pub assignments: Vec<AssignmentInfo>,
    pub usages: Vec<UsageInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentInfo {
    pub value_type: String,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub context: String,
    pub line: usize,
}

/// Flow analyzer for control flow analysis
pub struct FlowAnalyzer {
    variables: HashMap<String, VariableUsage>,
}

impl FlowAnalyzer {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Analyze a function body for control flow and variable usage
    pub fn analyze_function(&mut self, body: &[ast::Stmt]) {
        for stmt in body {
            self.analyze_statement(stmt);
        }
    }

    fn analyze_statement(&mut self, stmt: &ast::Stmt) {
        match stmt {
            ast::Stmt::Assign(assign) => {
                self.analyze_assignment(assign);
            }
            ast::Stmt::AugAssign(aug) => {
                self.analyze_aug_assignment(aug);
            }
            ast::Stmt::If(if_stmt) => {
                self.analyze_if(if_stmt);
            }
            ast::Stmt::For(for_stmt) => {
                self.analyze_for(for_stmt);
            }
            ast::Stmt::While(while_stmt) => {
                self.analyze_while(while_stmt);
            }
            ast::Stmt::Return(ret) => {
                if let Some(value) = &ret.value {
                    self.analyze_expr(value);
                }
            }
            _ => {}
        }
    }

    fn analyze_assignment(&mut self, assign: &ast::StmtAssign) {
        // Track variable assignments for type inference
        for target in &assign.targets {
            if let ast::Expr::Name(name) = target {
                let var_name = name.id.to_string();
                let entry = self.variables.entry(var_name.clone()).or_insert(VariableUsage {
                    name: var_name,
                    assignments: Vec::new(),
                    usages: Vec::new(),
                });

                entry.assignments.push(AssignmentInfo {
                    value_type: self.infer_expr_type(&assign.value),
                    line: 0, // TODO: extract line info
                });
            }
        }
    }

    fn analyze_aug_assignment(&mut self, aug: &ast::StmtAugAssign) {
        // x += 1, x *= 2, etc.
        if let ast::Expr::Name(name) = &*aug.target {
            let var_name = name.id.to_string();
            let entry = self.variables.entry(var_name.clone()).or_insert(VariableUsage {
                name: var_name,
                assignments: Vec::new(),
                usages: Vec::new(),
            });

            entry.usages.push(UsageInfo {
                context: "augmented_assignment".to_string(),
                line: 0,
            });
        }
    }

    fn analyze_if(&mut self, if_stmt: &ast::StmtIf) {
        self.analyze_expr(&if_stmt.test);
        for stmt in &if_stmt.body {
            self.analyze_statement(stmt);
        }
        for stmt in &if_stmt.orelse {
            self.analyze_statement(stmt);
        }
    }

    fn analyze_for(&mut self, for_stmt: &ast::StmtFor) {
        // Track loop variable
        if let ast::Expr::Name(name) = &*for_stmt.target {
            let var_name = name.id.to_string();
            self.variables.entry(var_name.clone()).or_insert(VariableUsage {
                name: var_name,
                assignments: vec![AssignmentInfo {
                    value_type: "int".to_string(), // Assume int from range
                    line: 0,
                }],
                usages: Vec::new(),
            });
        }

        for stmt in &for_stmt.body {
            self.analyze_statement(stmt);
        }
    }

    fn analyze_while(&mut self, while_stmt: &ast::StmtWhile) {
        self.analyze_expr(&while_stmt.test);
        for stmt in &while_stmt.body {
            self.analyze_statement(stmt);
        }
    }

    fn analyze_expr(&mut self, expr: &ast::Expr) {
        match expr {
            ast::Expr::Name(name) => {
                let var_name = name.id.to_string();
                let entry = self.variables.entry(var_name.clone()).or_insert(VariableUsage {
                    name: var_name,
                    assignments: Vec::new(),
                    usages: Vec::new(),
                });
                entry.usages.push(UsageInfo {
                    context: "expression".to_string(),
                    line: 0,
                });
            }
            ast::Expr::BinOp(binop) => {
                self.analyze_expr(&binop.left);
                self.analyze_expr(&binop.right);
            }
            ast::Expr::Compare(comp) => {
                self.analyze_expr(&comp.left);
                for comparator in &comp.comparators {
                    self.analyze_expr(comparator);
                }
            }
            ast::Expr::Call(call) => {
                self.analyze_expr(&call.func);
                for arg in &call.args {
                    self.analyze_expr(arg);
                }
            }
            _ => {}
        }
    }

    fn infer_expr_type(&mut self, expr: &ast::Expr) -> String {
        match expr {
            ast::Expr::Constant(constant) => match &constant.value {
                ast::Constant::Int(_) => "int".to_string(),
                ast::Constant::Float(_) => "float".to_string(),
                ast::Constant::Str(_) => "str".to_string(),
                ast::Constant::Bool(_) => "bool".to_string(),
                _ => "unknown".to_string(),
            },
            ast::Expr::BinOp(binop) => {
                // Analyze operands
                self.analyze_expr(&binop.left);
                self.analyze_expr(&binop.right);
                "int".to_string() // Simplified
            }
            ast::Expr::Compare(_) => "bool".to_string(),
            _ => "unknown".to_string(),
        }
    }

    pub fn get_variable_type(&self, name: &str) -> Option<String> {
        self.variables.get(name).and_then(|usage| {
            usage
                .assignments
                .last()
                .map(|assignment| assignment.value_type.clone())
        })
    }
}

impl Default for FlowAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::Parse;

    #[test]
    fn test_analyze_simple_assignment() {
        let mut analyzer = FlowAnalyzer::new();
        let source = "x = 5\ny = 10";
        let parsed = ast::Suite::parse(source, "<test>").unwrap();

        analyzer.analyze_function(&parsed);

        assert!(analyzer.get_variable_type("x").is_some());
        assert_eq!(analyzer.get_variable_type("x").unwrap(), "int");
    }

    #[test]
    fn test_analyze_if_statement() {
        let mut analyzer = FlowAnalyzer::new();
        let source = r#"
if x > 0:
    y = 1
else:
    y = 2
"#;
        let parsed = ast::Suite::parse(source, "<test>").unwrap();

        analyzer.analyze_function(&parsed);

        assert!(analyzer.variables.contains_key("y"));
    }

    #[test]
    fn test_analyze_for_loop() {
        let mut analyzer = FlowAnalyzer::new();
        let source = r#"
for i in range(10):
    x = i * 2
"#;
        let parsed = ast::Suite::parse(source, "<test>").unwrap();

        analyzer.analyze_function(&parsed);

        assert!(analyzer.variables.contains_key("i"));
        assert!(analyzer.variables.contains_key("x"));
    }
}
