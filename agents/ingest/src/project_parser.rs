//! Project-Level Parser for Multi-File Python Projects
//!
//! Handles parsing entire Python projects with multiple modules and dependencies.

use crate::{enhanced_parser::EnhancedParser, PythonAst};
use portalis_core::{Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Represents a complete Python project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonProject {
    pub root_path: PathBuf,
    pub modules: HashMap<String, PythonModule>,
    pub dependency_graph: DependencyGraph,
}

/// Represents a single Python module/file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonModule {
    pub name: String,
    pub path: PathBuf,
    pub ast: PythonAst,
    pub imports: Vec<ImportStatement>,
}

/// Import statement information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportStatement {
    pub module: String,
    pub items: Vec<String>,
    pub alias: Option<String>,
    pub is_relative: bool,
}

/// Dependency graph for module ordering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    pub nodes: HashMap<String, ModuleNode>,
    pub edges: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleNode {
    pub name: String,
    pub dependencies: Vec<String>,
    pub dependents: Vec<String>,
}

/// Project parser implementation
pub struct ProjectParser {
    parser: EnhancedParser,
}

impl ProjectParser {
    pub fn new() -> Self {
        Self {
            parser: EnhancedParser::new(),
        }
    }

    /// Parse an entire Python project from a root directory
    pub fn parse_project(&self, root_path: &Path) -> Result<PythonProject> {
        let mut modules = HashMap::new();

        // Discover all Python files
        let python_files = self.discover_python_files(root_path)?;

        // Parse each file
        for file_path in python_files {
            let module_name = self.path_to_module_name(root_path, &file_path)?;
            let source = std::fs::read_to_string(&file_path)?;

            let ast = self.parser.parse(&source)?;
            let imports = self.extract_imports(&ast);

            modules.insert(
                module_name.clone(),
                PythonModule {
                    name: module_name,
                    path: file_path,
                    ast,
                    imports,
                },
            );
        }

        // Build dependency graph
        let dependency_graph = self.build_dependency_graph(&modules)?;

        Ok(PythonProject {
            root_path: root_path.to_path_buf(),
            modules,
            dependency_graph,
        })
    }

    /// Discover all Python files in a directory tree
    fn discover_python_files(&self, root: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        if !root.exists() {
            return Err(Error::Parse(format!("Path does not exist: {:?}", root)));
        }

        self.walk_directory(root, &mut files)?;

        Ok(files)
    }

    fn walk_directory(&self, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }

        let entries = std::fs::read_dir(dir)?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Skip common directories
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') || name == "__pycache__" || name == "venv" {
                        continue;
                    }
                }
                self.walk_directory(&path, files)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("py") {
                files.push(path);
            }
        }

        Ok(())
    }

    /// Convert file path to module name
    fn path_to_module_name(&self, root: &Path, file_path: &Path) -> Result<String> {
        let relative = file_path
            .strip_prefix(root)
            .map_err(|_| Error::Parse("Invalid path".into()))?;

        let mut parts: Vec<String> = relative
            .iter()
            .map(|s| s.to_string_lossy().to_string())
            .collect();

        // Remove .py extension from last part
        if let Some(last) = parts.last_mut() {
            if last.ends_with(".py") {
                *last = last.trim_end_matches(".py").to_string();
            }
            // __init__.py becomes the parent module
            if *last == "__init__" {
                parts.pop();
            }
        }

        Ok(parts.join("."))
    }

    /// Extract import statements from AST
    fn extract_imports(&self, ast: &PythonAst) -> Vec<ImportStatement> {
        ast.imports
            .iter()
            .map(|imp| ImportStatement {
                module: imp.module.clone(),
                items: imp.items.clone(),
                alias: imp.alias.clone(),
                is_relative: imp.module.starts_with('.'),
            })
            .collect()
    }

    /// Build dependency graph from modules
    fn build_dependency_graph(
        &self,
        modules: &HashMap<String, PythonModule>,
    ) -> Result<DependencyGraph> {
        let mut nodes = HashMap::new();
        let mut edges = Vec::new();

        // Initialize nodes
        for module_name in modules.keys() {
            nodes.insert(
                module_name.clone(),
                ModuleNode {
                    name: module_name.clone(),
                    dependencies: Vec::new(),
                    dependents: Vec::new(),
                },
            );
        }

        // Build edges
        for (module_name, module) in modules {
            for import in &module.imports {
                // Only track internal dependencies
                if modules.contains_key(&import.module) {
                    edges.push((module_name.clone(), import.module.clone()));

                    // Update dependency lists
                    if let Some(node) = nodes.get_mut(module_name) {
                        node.dependencies.push(import.module.clone());
                    }
                    if let Some(dep_node) = nodes.get_mut(&import.module) {
                        dep_node.dependents.push(module_name.clone());
                    }
                }
            }
        }

        Ok(DependencyGraph { nodes, edges })
    }

    /// Get modules in topological order (dependencies first)
    pub fn topological_sort(&self, graph: &DependencyGraph) -> Result<Vec<String>> {
        let mut result = Vec::new();
        let mut visited = HashMap::new();
        let mut temp_mark = HashMap::new();

        for node_name in graph.nodes.keys() {
            if !visited.contains_key(node_name) {
                self.visit(
                    node_name,
                    &graph.nodes,
                    &mut visited,
                    &mut temp_mark,
                    &mut result,
                )?;
            }
        }

        // DFS post-order already gives us the correct topological order (dependencies first)
        Ok(result)
    }

    fn visit(
        &self,
        node: &str,
        nodes: &HashMap<String, ModuleNode>,
        visited: &mut HashMap<String, bool>,
        temp_mark: &mut HashMap<String, bool>,
        result: &mut Vec<String>,
    ) -> Result<()> {
        if temp_mark.get(node).copied().unwrap_or(false) {
            return Err(Error::Parse(format!("Circular dependency detected: {}", node)));
        }

        if visited.get(node).copied().unwrap_or(false) {
            return Ok(());
        }

        temp_mark.insert(node.to_string(), true);

        if let Some(module_node) = nodes.get(node) {
            for dep in &module_node.dependencies {
                self.visit(dep, nodes, visited, temp_mark, result)?;
            }
        }

        temp_mark.insert(node.to_string(), false);
        visited.insert(node.to_string(), true);
        result.push(node.to_string());

        Ok(())
    }
}

impl Default for ProjectParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_name_conversion() {
        let parser = ProjectParser::new();
        let root = Path::new("/project");
        let file = Path::new("/project/math/utils.py");

        let module_name = parser.path_to_module_name(root, file).unwrap();
        assert_eq!(module_name, "math.utils");
    }

    #[test]
    fn test_init_module_name() {
        let parser = ProjectParser::new();
        let root = Path::new("/project");
        let file = Path::new("/project/math/__init__.py");

        let module_name = parser.path_to_module_name(root, file).unwrap();
        assert_eq!(module_name, "math");
    }

    #[test]
    fn test_dependency_graph_simple() {
        let parser = ProjectParser::new();
        let mut modules = HashMap::new();

        // Create a simple dependency: module_a imports module_b
        modules.insert(
            "module_a".to_string(),
            PythonModule {
                name: "module_a".to_string(),
                path: PathBuf::from("module_a.py"),
                ast: PythonAst {
                    functions: vec![],
                    classes: vec![],
                    imports: vec![],
                },
                imports: vec![ImportStatement {
                    module: "module_b".to_string(),
                    items: vec![],
                    alias: None,
                    is_relative: false,
                }],
            },
        );

        modules.insert(
            "module_b".to_string(),
            PythonModule {
                name: "module_b".to_string(),
                path: PathBuf::from("module_b.py"),
                ast: PythonAst {
                    functions: vec![],
                    classes: vec![],
                    imports: vec![],
                },
                imports: vec![],
            },
        );

        let graph = parser.build_dependency_graph(&modules).unwrap();

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0], ("module_a".to_string(), "module_b".to_string()));
    }

    #[test]
    fn test_topological_sort() {
        let parser = ProjectParser::new();

        // Create graph: c depends on b, b depends on a
        let mut nodes = HashMap::new();
        nodes.insert(
            "a".to_string(),
            ModuleNode {
                name: "a".to_string(),
                dependencies: vec![],
                dependents: vec!["b".to_string()],
            },
        );
        nodes.insert(
            "b".to_string(),
            ModuleNode {
                name: "b".to_string(),
                dependencies: vec!["a".to_string()],
                dependents: vec!["c".to_string()],
            },
        );
        nodes.insert(
            "c".to_string(),
            ModuleNode {
                name: "c".to_string(),
                dependencies: vec!["b".to_string()],
                dependents: vec![],
            },
        );

        let graph = DependencyGraph {
            nodes,
            edges: vec![
                ("b".to_string(), "a".to_string()),
                ("c".to_string(), "b".to_string()),
            ],
        };

        let sorted = parser.topological_sort(&graph).unwrap();

        // a should come before b, b should come before c
        let a_pos = sorted.iter().position(|s| s == "a").unwrap();
        let b_pos = sorted.iter().position(|s| s == "b").unwrap();
        let c_pos = sorted.iter().position(|s| s == "c").unwrap();

        assert!(a_pos < b_pos);
        assert!(b_pos < c_pos);
    }
}
