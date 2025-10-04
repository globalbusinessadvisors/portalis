//! Dependency Graph - Directed graph of module dependencies
//!
//! This module provides:
//! 1. Dependency graph construction from Python imports
//! 2. Circular import detection using DFS
//! 3. Dependency resolution and traversal
//! 4. Import validation and optimization
//!
//! # Architecture
//! - Uses petgraph for efficient graph operations
//! - Nodes represent Python modules and files
//! - Edges represent import relationships
//! - Supports both internal (project) and external (library) dependencies

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::DfsPostOrder;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::import_analyzer::{ImportType, ImportedSymbol, PythonImport};
use crate::stdlib_mapper::StdlibMapper;
use crate::external_packages::ExternalPackageRegistry;

/// Represents a node in the dependency graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModuleNode {
    /// Module/file identifier (e.g., "myapp.models", "numpy")
    pub identifier: String,

    /// Node type (file, package, or external)
    pub node_type: NodeType,

    /// File path for local modules
    pub file_path: Option<String>,

    /// Whether this is a local project module
    pub is_local: bool,
}

/// Type of module node
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Python file (.py)
    File,

    /// Python package (directory with __init__.py)
    Package,

    /// External library (stdlib or third-party)
    External,
}

/// Represents an edge (import relationship) in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportEdge {
    /// Type of import (module, from, or star)
    pub import_type: ImportType,

    /// Specific items imported (for from imports) with their aliases
    pub items: Vec<ImportedSymbol>,

    /// Alias used (if any) for module-level imports
    pub alias: Option<String>,

    /// Line number where import occurs
    pub line_number: Option<usize>,
}

/// Circular dependency cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularDependency {
    /// Modules involved in the cycle
    pub cycle: Vec<String>,

    /// Suggested fix
    pub suggestion: String,
}

/// Import validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Issue type
    pub issue_type: IssueType,

    /// Module/import with the issue
    pub module: String,

    /// Description of the issue
    pub description: String,

    /// Suggested fix
    pub suggestion: Option<String>,

    /// Severity level
    pub severity: Severity,
}

/// Type of validation issue
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IssueType {
    /// Unknown or unmapped module
    UnknownModule,

    /// Wildcard import (from x import *)
    WildcardImport,

    /// Unused import
    UnusedImport,

    /// Import conflict (same alias, different modules)
    ImportConflict,

    /// Relative import depth too deep
    RelativeImportDepth,

    /// Circular dependency
    CircularDependency,
}

/// Severity level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Module being optimized
    pub module: String,

    /// Type of optimization
    pub optimization_type: OptimizationType,

    /// Description
    pub description: String,

    /// Suggested code change
    pub suggested_code: String,
}

/// Type of optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizationType {
    /// Remove unused import
    RemoveUnused,

    /// Combine imports from same module
    CombineImports,

    /// Replace star import with explicit imports
    ReplaceStarImport,

    /// Reorder imports (stdlib, external, local)
    ReorderImports,
}

/// Dependency graph for module analysis
pub struct DependencyGraph {
    /// The directed graph
    graph: DiGraph<ModuleNode, ImportEdge>,

    /// Map from module identifier to node index
    node_map: HashMap<String, NodeIndex>,

    /// Stdlib mapper for resolution
    stdlib_mapper: StdlibMapper,

    /// External package registry
    external_registry: ExternalPackageRegistry,

    /// Tracked usage of imported symbols
    #[allow(dead_code)]
    symbol_usage: HashMap<String, HashSet<String>>,
}

impl DependencyGraph {
    /// Create a new empty dependency graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            stdlib_mapper: StdlibMapper::new(),
            external_registry: ExternalPackageRegistry::new(),
            symbol_usage: HashMap::new(),
        }
    }

    /// Add a module node to the graph
    pub fn add_module(&mut self, identifier: String, node_type: NodeType, file_path: Option<String>) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&identifier) {
            return idx;
        }

        let is_local = matches!(node_type, NodeType::File | NodeType::Package);

        let node = ModuleNode {
            identifier: identifier.clone(),
            node_type,
            file_path,
            is_local,
        };

        let idx = self.graph.add_node(node);
        self.node_map.insert(identifier, idx);
        idx
    }

    /// Add an import edge between modules
    pub fn add_import(
        &mut self,
        from_module: &str,
        to_module: &str,
        import_type: ImportType,
        items: Vec<ImportedSymbol>,
        alias: Option<String>,
        line_number: Option<usize>,
    ) {
        let from_idx = self.node_map.get(from_module).copied();
        let to_idx = self.node_map.get(to_module).copied();

        if let (Some(from), Some(to)) = (from_idx, to_idx) {
            let edge = ImportEdge {
                import_type,
                items,
                alias,
                line_number,
            };

            self.graph.add_edge(from, to, edge);
        }
    }

    /// Build graph from Python imports for a module
    pub fn add_module_imports(&mut self, module_id: &str, file_path: Option<String>, imports: &[PythonImport]) {
        // Add the source module
        let source_idx = self.add_module(
            module_id.to_string(),
            NodeType::File,
            file_path,
        );

        // Add all imported modules and edges
        for import in imports {
            // Determine node type based on whether it's mapped
            let node_type = if self.stdlib_mapper.get_module(&import.module).is_some()
                || self.external_registry.get_package(&import.module).is_some() {
                NodeType::External
            } else {
                NodeType::Package
            };

            let target_idx = self.add_module(
                import.module.clone(),
                node_type,
                None,
            );

            // Add the import edge
            let edge = ImportEdge {
                import_type: import.import_type,
                items: import.items.clone(),
                alias: import.alias.clone(),
                line_number: None,
            };

            self.graph.add_edge(source_idx, target_idx, edge);
        }
    }

    /// Detect circular dependencies using DFS
    pub fn detect_circular_dependencies(&self) -> Vec<CircularDependency> {
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = Vec::new();

        // Check each node as a potential cycle start
        for node_idx in self.graph.node_indices() {
            if !visited.contains(&node_idx) {
                self.dfs_detect_cycle(node_idx, &mut visited, &mut rec_stack, &mut cycles);
            }
        }

        cycles
    }

    /// DFS helper for cycle detection
    fn dfs_detect_cycle(
        &self,
        node: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        rec_stack: &mut Vec<NodeIndex>,
        cycles: &mut Vec<CircularDependency>,
    ) {
        visited.insert(node);
        rec_stack.push(node);

        // Check all neighbors
        for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
            if !visited.contains(&neighbor) {
                self.dfs_detect_cycle(neighbor, visited, rec_stack, cycles);
            } else if rec_stack.contains(&neighbor) {
                // Found a cycle - extract it
                let cycle_start_pos = rec_stack.iter().position(|&n| n == neighbor).unwrap();
                let cycle_nodes: Vec<String> = rec_stack[cycle_start_pos..]
                    .iter()
                    .map(|&idx| self.graph[idx].identifier.clone())
                    .collect();

                let suggestion = self.suggest_cycle_fix(&cycle_nodes);

                cycles.push(CircularDependency {
                    cycle: cycle_nodes,
                    suggestion,
                });
            }
        }

        rec_stack.pop();
    }

    /// Suggest a fix for a circular dependency
    fn suggest_cycle_fix(&self, cycle: &[String]) -> String {
        if cycle.len() == 2 {
            format!(
                "Move shared code to a new module, or use delayed imports (import inside functions) in {}",
                cycle[0]
            )
        } else {
            format!(
                "Refactor to break the cycle: {} → ... → {}. Consider:\n\
                 1. Extract shared code to a new module\n\
                 2. Use dependency injection\n\
                 3. Move imports inside functions (delayed imports)",
                cycle.first().unwrap(),
                cycle.last().unwrap()
            )
        }
    }

    /// Validate imports and detect issues
    pub fn validate_imports(&self, used_symbols: &HashMap<String, HashSet<String>>) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Track wildcard imports
        for edge_idx in self.graph.edge_indices() {
            let (_, target_idx) = self.graph.edge_endpoints(edge_idx).unwrap();
            let edge = &self.graph[edge_idx];
            let target_node = &self.graph[target_idx];

            // Check for wildcard imports
            if edge.import_type == ImportType::StarImport {
                issues.push(ValidationIssue {
                    issue_type: IssueType::WildcardImport,
                    module: target_node.identifier.clone(),
                    description: format!(
                        "Wildcard import 'from {} import *' should be avoided",
                        target_node.identifier
                    ),
                    suggestion: Some("Use explicit imports instead".to_string()),
                    severity: Severity::Warning,
                });
            }

            // Check for unused imports
            if edge.import_type == ImportType::FromImport && !edge.items.is_empty() {
                let module_used = used_symbols.get(&target_node.identifier);

                for item in &edge.items {
                    if let Some(used) = module_used {
                        if !used.contains(&item.name) {
                            issues.push(ValidationIssue {
                                issue_type: IssueType::UnusedImport,
                                module: target_node.identifier.clone(),
                                description: format!(
                                    "Imported symbol '{}' from '{}' is not used",
                                    item.name, target_node.identifier
                                ),
                                suggestion: Some(format!("Remove unused import: {}", item.name)),
                                severity: Severity::Info,
                            });
                        }
                    }
                }
            }
        }

        // Check for unknown/unmapped modules
        for node_idx in self.graph.node_indices() {
            let node = &self.graph[node_idx];

            // Check Package nodes (potential external imports) that aren't locally defined
            // External nodes are already mapped, so we skip those
            if matches!(node.node_type, NodeType::Package) && node.file_path.is_none() {
                // Check if it's mapped to stdlib or external package
                if self.stdlib_mapper.get_module(&node.identifier).is_none()
                    && self.external_registry.get_package(&node.identifier).is_none()
                {
                    issues.push(ValidationIssue {
                        issue_type: IssueType::UnknownModule,
                        module: node.identifier.clone(),
                        description: format!(
                            "Module '{}' is not mapped to a Rust crate",
                            node.identifier
                        ),
                        suggestion: Some("Add mapping to stdlib_mapper or external_packages".to_string()),
                        severity: Severity::Warning,
                    });
                }
            }
        }

        // Check for import conflicts (same alias, different modules)
        let mut alias_map: HashMap<String, Vec<String>> = HashMap::new();

        for edge_idx in self.graph.edge_indices() {
            let (_, target_idx) = self.graph.edge_endpoints(edge_idx).unwrap();
            let edge = &self.graph[edge_idx];
            let target_node = &self.graph[target_idx];

            if let Some(ref alias) = edge.alias {
                alias_map.entry(alias.clone())
                    .or_default()
                    .push(target_node.identifier.clone());
            }
        }

        for (alias, modules) in alias_map {
            if modules.len() > 1 {
                issues.push(ValidationIssue {
                    issue_type: IssueType::ImportConflict,
                    module: modules.join(", "),
                    description: format!(
                        "Alias '{}' is used for multiple modules: {}",
                        alias,
                        modules.join(", ")
                    ),
                    suggestion: Some("Use different aliases for different modules".to_string()),
                    severity: Severity::Error,
                });
            }
        }

        issues
    }

    /// Generate optimization suggestions
    pub fn generate_optimizations(&self) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Group imports by module
        let mut module_imports: HashMap<String, Vec<(NodeIndex, &ImportEdge)>> = HashMap::new();

        for edge_idx in self.graph.edge_indices() {
            let (source, target) = self.graph.edge_endpoints(edge_idx).unwrap();
            let edge = &self.graph[edge_idx];
            let target_module = &self.graph[target].identifier;

            module_imports.entry(target_module.clone())
                .or_default()
                .push((source, edge));
        }

        // Suggest combining imports from same module
        for (module, imports) in module_imports {
            if imports.len() > 1 {
                // Check if they're all from the same source
                let sources: HashSet<_> = imports.iter().map(|(src, _)| *src).collect();

                if sources.len() == 1 {
                    let all_items: Vec<String> = imports.iter()
                        .flat_map(|(_, edge)| edge.items.iter().map(|s| s.name.clone()))
                        .collect();

                    if !all_items.is_empty() {
                        suggestions.push(OptimizationSuggestion {
                            module: module.clone(),
                            optimization_type: OptimizationType::CombineImports,
                            description: format!(
                                "Multiple imports from '{}' can be combined",
                                module
                            ),
                            suggested_code: format!(
                                "from {} import {}",
                                module,
                                all_items.join(", ")
                            ),
                        });
                    }
                }
            }
        }

        // Suggest replacing star imports
        for edge_idx in self.graph.edge_indices() {
            let (_, target_idx) = self.graph.edge_endpoints(edge_idx).unwrap();
            let edge = &self.graph[edge_idx];
            let target = &self.graph[target_idx];

            if edge.import_type == ImportType::StarImport {
                suggestions.push(OptimizationSuggestion {
                    module: target.identifier.clone(),
                    optimization_type: OptimizationType::ReplaceStarImport,
                    description: format!(
                        "Replace 'from {} import *' with explicit imports",
                        target.identifier
                    ),
                    suggested_code: format!(
                        "from {} import <specific_items>",
                        target.identifier
                    ),
                });
            }
        }

        suggestions
    }

    /// Get all dependencies of a module (transitive)
    pub fn get_dependencies(&self, module: &str) -> HashSet<String> {
        let mut deps = HashSet::new();

        if let Some(&node_idx) = self.node_map.get(module) {
            let mut queue = VecDeque::new();
            queue.push_back(node_idx);

            while let Some(idx) = queue.pop_front() {
                for neighbor in self.graph.neighbors_directed(idx, Direction::Outgoing) {
                    let dep_module = &self.graph[neighbor].identifier;

                    if deps.insert(dep_module.clone()) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        deps
    }

    /// Get all dependents of a module (who imports this module)
    pub fn get_dependents(&self, module: &str) -> HashSet<String> {
        let mut dependents = HashSet::new();

        if let Some(&node_idx) = self.node_map.get(module) {
            for neighbor in self.graph.neighbors_directed(node_idx, Direction::Incoming) {
                dependents.insert(self.graph[neighbor].identifier.clone());
            }
        }

        dependents
    }

    /// Get topological sort of modules (build order)
    pub fn topological_sort(&self) -> Result<Vec<String>, String> {
        let mut post_order = DfsPostOrder::new(&self.graph, self.graph.node_indices().next().unwrap_or(NodeIndex::new(0)));
        let mut sorted = Vec::new();

        while let Some(node) = post_order.next(&self.graph) {
            sorted.push(self.graph[node].identifier.clone());
        }

        sorted.reverse();
        Ok(sorted)
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let total_modules = self.graph.node_count();
        let total_imports = self.graph.edge_count();

        let local_modules = self.graph.node_weights()
            .filter(|n| n.is_local)
            .count();

        let external_modules = total_modules - local_modules;

        GraphStats {
            total_modules,
            local_modules,
            external_modules,
            total_imports,
        }
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_modules: usize,
    pub local_modules: usize,
    pub external_modules: usize,
    pub total_imports: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_module() {
        let mut graph = DependencyGraph::new();

        let idx1 = graph.add_module("myapp.models".to_string(), NodeType::File, None);
        let idx2 = graph.add_module("myapp.views".to_string(), NodeType::File, None);

        assert_ne!(idx1, idx2);
        assert_eq!(graph.graph.node_count(), 2);
    }

    #[test]
    fn test_add_import() {
        let mut graph = DependencyGraph::new();

        graph.add_module("myapp.models".to_string(), NodeType::File, None);
        graph.add_module("myapp.views".to_string(), NodeType::File, None);

        graph.add_import(
            "myapp.views",
            "myapp.models",
            ImportType::FromImport,
            vec![ImportedSymbol { name: "User".to_string(), alias: None }],
            None,
            Some(1),
        );

        assert_eq!(graph.graph.edge_count(), 1);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut graph = DependencyGraph::new();

        // Create a circular dependency: A -> B -> C -> A
        graph.add_module("module_a".to_string(), NodeType::File, None);
        graph.add_module("module_b".to_string(), NodeType::File, None);
        graph.add_module("module_c".to_string(), NodeType::File, None);

        graph.add_import("module_a", "module_b", ImportType::Module, vec![], None, None);
        graph.add_import("module_b", "module_c", ImportType::Module, vec![], None, None);
        graph.add_import("module_c", "module_a", ImportType::Module, vec![], None, None);

        let cycles = graph.detect_circular_dependencies();

        assert!(!cycles.is_empty(), "Should detect circular dependency");
        assert!(cycles[0].cycle.contains(&"module_a".to_string()));
        assert!(cycles[0].cycle.contains(&"module_b".to_string()));
    }

    #[test]
    fn test_validate_wildcard_import() {
        let mut graph = DependencyGraph::new();

        graph.add_module("main".to_string(), NodeType::File, None);
        graph.add_module("utils".to_string(), NodeType::Package, None);

        graph.add_import(
            "main",
            "utils",
            ImportType::StarImport,
            vec![],
            None,
            None,
        );

        let used_symbols = HashMap::new();
        let issues = graph.validate_imports(&used_symbols);

        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.issue_type == IssueType::WildcardImport));
    }

    #[test]
    fn test_get_dependencies() {
        let mut graph = DependencyGraph::new();

        graph.add_module("app".to_string(), NodeType::File, None);
        graph.add_module("models".to_string(), NodeType::File, None);
        graph.add_module("database".to_string(), NodeType::File, None);

        graph.add_import("app", "models", ImportType::Module, vec![], None, None);
        graph.add_import("models", "database", ImportType::Module, vec![], None, None);

        let deps = graph.get_dependencies("app");

        assert!(deps.contains("models"));
        assert!(deps.contains("database"));
    }

    #[test]
    fn test_optimization_suggestions() {
        let mut graph = DependencyGraph::new();

        graph.add_module("main".to_string(), NodeType::File, None);
        graph.add_module("utils".to_string(), NodeType::Package, None);

        // Add star import
        graph.add_import("main", "utils", ImportType::StarImport, vec![], None, None);

        let suggestions = graph.generate_optimizations();

        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.optimization_type == OptimizationType::ReplaceStarImport));
    }
}
