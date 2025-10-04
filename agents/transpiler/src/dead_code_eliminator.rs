//! Dead Code Eliminator - Analyzes and removes unused code for WASM optimization
//!
//! This module provides:
//! 1. Dead code detection in generated Rust code
//! 2. wasm-opt advanced optimization passes
//! 3. Tree-shaking analysis for dependencies
//! 4. Unused function and type detection
//! 5. Optimization reporting and recommendations

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Dead code analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadCodeAnalysis {
    /// Unused functions
    pub unused_functions: Vec<UnusedItem>,
    /// Unused types/structs
    pub unused_types: Vec<UnusedItem>,
    /// Unused imports
    pub unused_imports: Vec<String>,
    /// Unused dependencies
    pub unused_dependencies: Vec<String>,
    /// Total potential size reduction (bytes)
    pub potential_size_reduction: u64,
}

impl DeadCodeAnalysis {
    /// Calculate total items found
    pub fn total_unused_items(&self) -> usize {
        self.unused_functions.len()
            + self.unused_types.len()
            + self.unused_imports.len()
            + self.unused_dependencies.len()
    }

    /// Generate detailed report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Dead Code Analysis ===\n\n");

        if !self.unused_functions.is_empty() {
            report.push_str(&format!("Unused Functions ({}): \n", self.unused_functions.len()));
            for item in self.unused_functions.iter().take(10) {
                report.push_str(&format!("  - {} (line {}) - ~{} bytes\n",
                    item.name, item.line, item.estimated_size));
            }
            if self.unused_functions.len() > 10 {
                report.push_str(&format!("  ... and {} more\n", self.unused_functions.len() - 10));
            }
            report.push('\n');
        }

        if !self.unused_types.is_empty() {
            report.push_str(&format!("Unused Types ({}): \n", self.unused_types.len()));
            for item in self.unused_types.iter().take(10) {
                report.push_str(&format!("  - {} (line {}) - ~{} bytes\n",
                    item.name, item.line, item.estimated_size));
            }
            if self.unused_types.len() > 10 {
                report.push_str(&format!("  ... and {} more\n", self.unused_types.len() - 10));
            }
            report.push('\n');
        }

        if !self.unused_imports.is_empty() {
            report.push_str(&format!("Unused Imports ({}): \n", self.unused_imports.len()));
            for import in self.unused_imports.iter().take(10) {
                report.push_str(&format!("  - {}\n", import));
            }
            if self.unused_imports.len() > 10 {
                report.push_str(&format!("  ... and {} more\n", self.unused_imports.len() - 10));
            }
            report.push('\n');
        }

        if !self.unused_dependencies.is_empty() {
            report.push_str(&format!("Unused Dependencies ({}): \n", self.unused_dependencies.len()));
            for dep in &self.unused_dependencies {
                report.push_str(&format!("  - {}\n", dep));
            }
            report.push('\n');
        }

        report.push_str(&format!("Total Items: {}\n", self.total_unused_items()));
        report.push_str(&format!("Potential Size Reduction: {:.2} KB\n",
            self.potential_size_reduction as f64 / 1024.0));

        report
    }
}

/// Represents an unused code item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnusedItem {
    /// Item name
    pub name: String,
    /// Item type (function, struct, etc.)
    pub item_type: String,
    /// Source file
    pub file: String,
    /// Line number
    pub line: usize,
    /// Estimated size in bytes
    pub estimated_size: u64,
    /// Reason for being unused
    pub reason: String,
}

/// wasm-opt optimization pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmOptPass {
    /// Pass name
    pub name: String,
    /// Description
    pub description: String,
    /// Estimated impact (0.0-1.0)
    pub impact: f64,
    /// Enabled by default
    pub default: bool,
}

impl WasmOptPass {
    /// Get all available optimization passes
    pub fn all_passes() -> Vec<Self> {
        vec![
            // Dead code elimination
            Self {
                name: "dce".to_string(),
                description: "Dead Code Elimination - removes unreachable code".to_string(),
                impact: 0.25,
                default: true,
            },
            Self {
                name: "remove-unused-brs".to_string(),
                description: "Remove unused branch targets".to_string(),
                impact: 0.05,
                default: true,
            },
            Self {
                name: "remove-unused-names".to_string(),
                description: "Remove unused function/global names".to_string(),
                impact: 0.10,
                default: true,
            },
            Self {
                name: "remove-unused-module-elements".to_string(),
                description: "Remove unused module elements".to_string(),
                impact: 0.15,
                default: true,
            },
            // Inlining
            Self {
                name: "inlining".to_string(),
                description: "Inline small functions".to_string(),
                impact: 0.20,
                default: true,
            },
            Self {
                name: "inline-main".to_string(),
                description: "Inline the main function".to_string(),
                impact: 0.05,
                default: false,
            },
            // Code optimization
            Self {
                name: "simplify-locals".to_string(),
                description: "Simplify local variable usage".to_string(),
                impact: 0.15,
                default: true,
            },
            Self {
                name: "coalesce-locals".to_string(),
                description: "Coalesce local variables".to_string(),
                impact: 0.10,
                default: true,
            },
            Self {
                name: "merge-blocks".to_string(),
                description: "Merge basic blocks".to_string(),
                impact: 0.08,
                default: true,
            },
            Self {
                name: "optimize-instructions".to_string(),
                description: "Optimize individual instructions".to_string(),
                impact: 0.12,
                default: true,
            },
            // Size reduction
            Self {
                name: "strip".to_string(),
                description: "Strip debug information".to_string(),
                impact: 0.30,
                default: true,
            },
            Self {
                name: "strip-debug".to_string(),
                description: "Strip debug information (alias)".to_string(),
                impact: 0.30,
                default: true,
            },
            Self {
                name: "strip-producers".to_string(),
                description: "Strip producer metadata".to_string(),
                impact: 0.05,
                default: true,
            },
            Self {
                name: "strip-target-features".to_string(),
                description: "Strip target features section".to_string(),
                impact: 0.03,
                default: true,
            },
            // Advanced optimization
            Self {
                name: "precompute".to_string(),
                description: "Precompute constant expressions".to_string(),
                impact: 0.10,
                default: true,
            },
            Self {
                name: "vacuum".to_string(),
                description: "Remove duplicate/unused code after other optimizations".to_string(),
                impact: 0.15,
                default: true,
            },
            Self {
                name: "duplicate-function-elimination".to_string(),
                description: "Eliminate duplicate functions".to_string(),
                impact: 0.12,
                default: false,
            },
        ]
    }

    /// Get passes for aggressive size optimization
    pub fn size_optimization_passes() -> Vec<String> {
        vec![
            "dce".to_string(),
            "remove-unused-brs".to_string(),
            "remove-unused-names".to_string(),
            "remove-unused-module-elements".to_string(),
            "strip-debug".to_string(),
            "strip-producers".to_string(),
            "strip-target-features".to_string(),
            "vacuum".to_string(),
            "inlining".to_string(),
            "simplify-locals".to_string(),
            "coalesce-locals".to_string(),
            "merge-blocks".to_string(),
            "optimize-instructions".to_string(),
            "precompute".to_string(),
            "duplicate-function-elimination".to_string(),
        ]
    }

    /// Get passes for performance optimization
    pub fn performance_optimization_passes() -> Vec<String> {
        vec![
            "inlining".to_string(),
            "inline-main".to_string(),
            "optimize-instructions".to_string(),
            "precompute".to_string(),
            "simplify-locals".to_string(),
            "merge-blocks".to_string(),
        ]
    }
}

/// Tree-shaking analysis for dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeShakingAnalysis {
    /// Total dependencies
    pub total_dependencies: usize,
    /// Used dependencies
    pub used_dependencies: Vec<String>,
    /// Unused dependencies (can be removed)
    pub unused_dependencies: Vec<String>,
    /// Partially used dependencies
    pub partially_used: Vec<PartiallyUsedDependency>,
    /// Estimated size savings from tree-shaking
    pub estimated_savings: u64,
}

impl TreeShakingAnalysis {
    /// Generate report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Tree-Shaking Analysis ===\n\n");
        report.push_str(&format!("Total Dependencies: {}\n", self.total_dependencies));
        report.push_str(&format!("Fully Used: {}\n", self.used_dependencies.len()));
        report.push_str(&format!("Unused: {}\n", self.unused_dependencies.len()));
        report.push_str(&format!("Partially Used: {}\n\n", self.partially_used.len()));

        if !self.unused_dependencies.is_empty() {
            report.push_str("Unused Dependencies (can be removed):\n");
            for dep in &self.unused_dependencies {
                report.push_str(&format!("  ❌ {}\n", dep));
            }
            report.push('\n');
        }

        if !self.partially_used.is_empty() {
            report.push_str("Partially Used Dependencies:\n");
            for item in &self.partially_used {
                report.push_str(&format!("  ⚠️  {} - using {}/{} features\n",
                    item.dependency, item.used_features.len(), item.total_features));
            }
            report.push('\n');
        }

        report.push_str(&format!("Estimated Savings: {:.2} KB\n",
            self.estimated_savings as f64 / 1024.0));

        report
    }
}

/// Partially used dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartiallyUsedDependency {
    /// Dependency name
    pub dependency: String,
    /// Used features
    pub used_features: Vec<String>,
    /// Total available features
    pub total_features: usize,
    /// Unused feature names
    pub unused_features: Vec<String>,
}

/// Dead code eliminator
pub struct DeadCodeEliminator;

impl DeadCodeEliminator {
    /// Analyze Rust code for dead code
    pub fn analyze_rust_code(code: &str) -> DeadCodeAnalysis {
        let mut unused_functions = Vec::new();
        let unused_types = Vec::new();
        let mut unused_imports = Vec::new();

        // Simple heuristic analysis (in production, use proper AST parsing)
        let lines: Vec<&str> = code.lines().collect();

        // Track function definitions
        let mut defined_functions = HashSet::new();
        let mut used_functions = HashSet::new();

        for (idx, line) in lines.iter().enumerate() {
            // Find function definitions
            if line.contains("fn ") && !line.trim_start().starts_with("//") {
                if let Some(name) = Self::extract_function_name(line) {
                    defined_functions.insert((name.clone(), idx + 1));
                }
            }

            // Find function calls (simple heuristic)
            for (name, _) in &defined_functions {
                if line.contains(&format!("{}(", name)) && !line.contains(&format!("fn {}", name)) {
                    used_functions.insert(name.clone());
                }
            }

            // Find unused imports
            if line.trim_start().starts_with("use ") {
                if let Some(import) = Self::extract_import(line) {
                    // Check if import is used in code
                    let is_used = code.contains(&import) &&
                        code.matches(&import).count() > 1; // More than just the import line

                    if !is_used {
                        unused_imports.push(import);
                    }
                }
            }
        }

        // Identify unused functions
        for (name, line) in defined_functions {
            if !used_functions.contains(&name) && name != "main" {
                unused_functions.push(UnusedItem {
                    name: name.clone(),
                    item_type: "function".to_string(),
                    file: "generated.rs".to_string(),
                    line,
                    estimated_size: 100, // Estimated avg function size
                    reason: "Never called".to_string(),
                });
            }
        }

        let potential_reduction =
            (unused_functions.len() * 100) as u64 +
            (unused_types.len() * 50) as u64 +
            (unused_imports.len() * 20) as u64;

        DeadCodeAnalysis {
            unused_functions,
            unused_types,
            unused_imports,
            unused_dependencies: vec![],
            potential_size_reduction: potential_reduction,
        }
    }

    /// Extract function name from definition line
    fn extract_function_name(line: &str) -> Option<String> {
        let line = line.trim();
        if let Some(start) = line.find("fn ") {
            let after_fn = &line[start + 3..];
            if let Some(end) = after_fn.find('(') {
                let name = after_fn[..end].trim();
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
        None
    }

    /// Extract import name
    fn extract_import(line: &str) -> Option<String> {
        let line = line.trim();
        if line.starts_with("use ") {
            let import = line.trim_start_matches("use ").trim_end_matches(';').trim();
            Some(import.to_string())
        } else {
            None
        }
    }

    /// Generate wasm-opt command with dead code elimination
    pub fn generate_wasm_opt_command(
        input: &str,
        output: &str,
        optimization_level: u8,
    ) -> String {
        let passes = WasmOptPass::size_optimization_passes();

        let mut cmd = format!("wasm-opt -O{}", optimization_level);

        for pass in passes {
            cmd.push_str(&format!(" --{}", pass));
        }

        cmd.push_str(&format!(" {} -o {}", input, output));
        cmd
    }

    /// Analyze tree-shaking opportunities
    pub fn analyze_tree_shaking(
        dependencies: &HashMap<String, Vec<String>>,
        used_items: &HashSet<String>,
    ) -> TreeShakingAnalysis {
        let total_dependencies = dependencies.len();
        let mut used_dependencies = Vec::new();
        let mut unused_dependencies = Vec::new();
        let mut partially_used = Vec::new();

        for (dep_name, features) in dependencies {
            let used_features: Vec<String> = features
                .iter()
                .filter(|f| used_items.contains(*f))
                .cloned()
                .collect();

            if used_features.is_empty() {
                unused_dependencies.push(dep_name.clone());
            } else if used_features.len() == features.len() {
                used_dependencies.push(dep_name.clone());
            } else {
                let unused_features: Vec<String> = features
                    .iter()
                    .filter(|f| !used_items.contains(*f))
                    .cloned()
                    .collect();

                partially_used.push(PartiallyUsedDependency {
                    dependency: dep_name.clone(),
                    used_features: used_features.clone(),
                    total_features: features.len(),
                    unused_features,
                });
            }
        }

        let estimated_savings =
            (unused_dependencies.len() * 20000) as u64 + // 20KB per unused dep
            (partially_used.len() * 5000) as u64;        // 5KB per partially used

        TreeShakingAnalysis {
            total_dependencies,
            used_dependencies,
            unused_dependencies,
            partially_used,
            estimated_savings,
        }
    }

    /// Get optimization recommendations
    pub fn get_recommendations(analysis: &DeadCodeAnalysis) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !analysis.unused_functions.is_empty() {
            recommendations.push(format!(
                "Remove {} unused functions to save ~{:.1} KB",
                analysis.unused_functions.len(),
                (analysis.unused_functions.len() * 100) as f64 / 1024.0
            ));
        }

        if !analysis.unused_imports.is_empty() {
            recommendations.push(format!(
                "Remove {} unused imports",
                analysis.unused_imports.len()
            ));
        }

        if !analysis.unused_dependencies.is_empty() {
            recommendations.push(format!(
                "Remove {} unused dependencies from Cargo.toml",
                analysis.unused_dependencies.len()
            ));
        }

        if analysis.potential_size_reduction > 1024 {
            recommendations.push(format!(
                "Total potential size reduction: {:.2} KB",
                analysis.potential_size_reduction as f64 / 1024.0
            ));
        }

        recommendations.push(
            "Run wasm-opt with dead code elimination passes for maximum optimization".to_string()
        );

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dead_code_analysis() {
        let code = r#"
fn used_function() {
    println!("I am used");
}

fn unused_function() {
    println!("I am never called");
}

fn main() {
    used_function();
}
"#;

        let analysis = DeadCodeEliminator::analyze_rust_code(code);

        assert!(analysis.unused_functions.len() > 0);
        assert!(analysis.unused_functions.iter().any(|f| f.name == "unused_function"));
    }

    #[test]
    fn test_wasm_opt_passes() {
        let passes = WasmOptPass::all_passes();
        assert!(passes.len() > 10);

        let dce = passes.iter().find(|p| p.name == "dce");
        assert!(dce.is_some());
    }

    #[test]
    fn test_wasm_opt_command_generation() {
        let cmd = DeadCodeEliminator::generate_wasm_opt_command(
            "input.wasm",
            "output.wasm",
            4,
        );

        assert!(cmd.contains("wasm-opt"));
        assert!(cmd.contains("-O4"));
        assert!(cmd.contains("--dce"));
        assert!(cmd.contains("input.wasm"));
    }

    #[test]
    fn test_tree_shaking_analysis() {
        let mut deps = HashMap::new();
        deps.insert("serde".to_string(), vec!["Serialize".to_string(), "Deserialize".to_string()]);
        deps.insert("unused_crate".to_string(), vec!["Function1".to_string()]);

        let mut used_items = HashSet::new();
        used_items.insert("Serialize".to_string());

        let analysis = DeadCodeEliminator::analyze_tree_shaking(&deps, &used_items);

        assert_eq!(analysis.total_dependencies, 2);
        assert!(analysis.unused_dependencies.contains(&"unused_crate".to_string()));
        assert_eq!(analysis.partially_used.len(), 1);
    }

    #[test]
    fn test_size_optimization_passes() {
        let passes = WasmOptPass::size_optimization_passes();

        assert!(passes.contains(&"dce".to_string()));
        assert!(passes.contains(&"strip-debug".to_string()));
        assert!(passes.contains(&"vacuum".to_string()));
    }
}
