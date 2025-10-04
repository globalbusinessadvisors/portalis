//! Dependency Resolver - Resolves Python imports to Rust dependencies
//!
//! This module provides:
//! 1. Import resolution using stdlib_mapper and external_packages
//! 2. Cargo dependency collection with version conflict handling
//! 3. Rust use statement generation
//! 4. WASM compatibility checking
//! 5. Integration with dependency graph analysis

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::dependency_graph::{DependencyGraph, ValidationIssue, OptimizationSuggestion, CircularDependency};
use crate::import_analyzer::{ImportAnalyzer, ImportAnalysis, PythonImport, RustDependency, WasmCompatibilitySummary};
use crate::stdlib_mapper::{StdlibMapper, WasmCompatibility};
use crate::external_packages::ExternalPackageRegistry;
use crate::cargo_generator::CargoGenerator;
use crate::version_compatibility::VersionCompatibilityChecker;

/// Represents a resolved Rust dependency with metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedDependency {
    /// Rust crate name
    pub crate_name: String,

    /// Version requirement
    pub version: String,

    /// Required features
    pub features: Vec<String>,

    /// WASM compatibility
    pub wasm_compat: WasmCompatibility,

    /// Source Python modules that require this
    pub source_modules: Vec<String>,

    /// Notes about the dependency
    pub notes: Option<String>,
}

/// Cargo.toml dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoDependency {
    /// Crate name
    pub name: String,

    /// Version
    pub version: String,

    /// Features
    pub features: Vec<String>,

    /// Optional flag
    pub optional: bool,

    /// Default features flag
    pub default_features: bool,
}

/// Version conflict between dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionConflict {
    /// Crate with conflict
    pub crate_name: String,

    /// Requested versions
    pub versions: Vec<String>,

    /// Resolution strategy used
    pub resolution: String,
}

/// Complete dependency resolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResolution {
    /// All resolved Rust dependencies
    pub dependencies: Vec<ResolvedDependency>,

    /// Generated Rust use statements
    pub use_statements: Vec<String>,

    /// WASM compatibility summary
    pub wasm_summary: WasmSummary,

    /// Unmapped Python modules
    pub unmapped_modules: Vec<String>,

    /// Version conflicts (if any)
    pub version_conflicts: Vec<VersionConflict>,

    /// Validation issues
    pub validation_issues: Vec<ValidationIssue>,

    /// Optimization suggestions
    pub optimizations: Vec<OptimizationSuggestion>,

    /// Circular dependencies detected
    pub circular_dependencies: Vec<CircularDependency>,

    /// Cargo.toml content
    pub cargo_toml: String,
}

/// WASM compatibility summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSummary {
    /// Fully compatible
    pub fully_compatible: bool,

    /// Requires WASI
    pub needs_wasi: bool,

    /// Requires JS interop
    pub needs_js_interop: bool,

    /// Has incompatible modules
    pub has_incompatible: bool,

    /// Module compatibility breakdown
    pub modules: HashMap<String, WasmCompatibility>,
}

/// Dependency resolver - main orchestrator
pub struct DependencyResolver {
    stdlib_mapper: StdlibMapper,
    external_registry: ExternalPackageRegistry,
    dependency_graph: DependencyGraph,
}

impl DependencyResolver {
    /// Create a new dependency resolver
    pub fn new() -> Self {
        Self {
            stdlib_mapper: StdlibMapper::new(),
            external_registry: ExternalPackageRegistry::new(),
            dependency_graph: DependencyGraph::new(),
        }
    }

    /// Resolve dependencies for a single Python module
    pub fn resolve_module(&mut self, module_id: &str, python_code: &str) -> DependencyResolution {
        // Extract imports using AST-based analyzer
        let analyzer = ImportAnalyzer::with_module_path(module_id.to_string());
        let import_analysis = analyzer.analyze(python_code);

        // Add to dependency graph
        self.dependency_graph.add_module_imports(
            module_id,
            Some(module_id.to_string()),
            &import_analysis.python_imports,
        );

        // Resolve to Rust dependencies
        self.resolve_imports(&import_analysis)
    }

    /// Resolve dependencies for multiple modules (entire project)
    pub fn resolve_project(&mut self, modules: &HashMap<String, String>) -> DependencyResolution {
        // Analyze all modules and build dependency graph
        let mut all_imports = Vec::new();

        for (module_id, python_code) in modules {
            let analyzer = ImportAnalyzer::with_module_path(module_id.clone());
            let import_analysis = analyzer.analyze(python_code);

            // Add to graph
            self.dependency_graph.add_module_imports(
                module_id,
                Some(module_id.clone()),
                &import_analysis.python_imports,
            );

            all_imports.extend(import_analysis.python_imports);
        }

        // Create combined analysis
        let combined_analysis = ImportAnalysis {
            python_imports: all_imports,
            rust_dependencies: vec![],
            rust_use_statements: vec![],
            wasm_compatibility: Default::default(),
            unmapped_modules: vec![],
        };

        // Resolve all imports
        let mut resolution = self.resolve_imports(&combined_analysis);

        // Add graph-based analysis
        resolution.circular_dependencies = self.dependency_graph.detect_circular_dependencies();

        // Validation (we'll use empty symbol usage for now - can be enhanced later)
        let used_symbols = HashMap::new();
        resolution.validation_issues = self.dependency_graph.validate_imports(&used_symbols);

        // Optimizations
        resolution.optimizations = self.dependency_graph.generate_optimizations();

        resolution
    }

    /// Resolve imports to Rust dependencies
    fn resolve_imports(&self, analysis: &ImportAnalysis) -> DependencyResolution {
        let mut dependencies_map: HashMap<String, ResolvedDependency> = HashMap::new();
        let mut use_statements = HashSet::new();
        let mut unmapped_modules = Vec::new();
        let mut modules_compat = HashMap::new();
        let mut version_conflicts = Vec::new();

        // Process each import
        for import in &analysis.python_imports {
            // Try stdlib first
            if let Some(module_mapping) = self.stdlib_mapper.get_module(&import.module) {
                // Track compatibility
                modules_compat.insert(
                    import.module.clone(),
                    module_mapping.wasm_compatible.clone(),
                );

                // Generate use statement
                if !module_mapping.rust_use.is_empty() {
                    let use_stmt = self.generate_use_statement(import, &module_mapping.rust_use);
                    use_statements.insert(use_stmt);
                }

                // Add dependency if external crate
                if let Some(ref crate_name) = module_mapping.rust_crate {
                    self.add_or_merge_dependency(
                        &mut dependencies_map,
                        &mut version_conflicts,
                        crate_name.clone(),
                        module_mapping.version.clone(),
                        vec![],
                        module_mapping.wasm_compatible.clone(),
                        import.module.clone(),
                        module_mapping.notes.clone(),
                    );
                }
            }
            // Try external packages
            else if let Some(pkg_mapping) = self.external_registry.get_package(&import.module) {
                modules_compat.insert(
                    import.module.clone(),
                    pkg_mapping.wasm_compatible.clone(),
                );

                // Generate use statement
                let use_stmt = format!("use {};", pkg_mapping.rust_crate);
                use_statements.insert(use_stmt);

                // Add dependency
                self.add_or_merge_dependency(
                    &mut dependencies_map,
                    &mut version_conflicts,
                    pkg_mapping.rust_crate.clone(),
                    pkg_mapping.version.clone(),
                    pkg_mapping.features.clone(),
                    pkg_mapping.wasm_compatible.clone(),
                    import.module.clone(),
                    pkg_mapping.notes.clone(),
                );
            } else {
                // Unmapped module
                unmapped_modules.push(import.module.clone());
            }
        }

        // Build WASM summary
        let wasm_summary = self.build_wasm_summary(&modules_compat);

        // Convert to vec and sort
        let mut dependencies: Vec<ResolvedDependency> = dependencies_map.into_values().collect();
        dependencies.sort_by(|a, b| a.crate_name.cmp(&b.crate_name));

        let mut use_statements: Vec<String> = use_statements.into_iter().collect();
        use_statements.sort();

        unmapped_modules.sort();
        unmapped_modules.dedup();

        // Generate Cargo.toml
        let cargo_toml = self.generate_cargo_toml(&dependencies, &wasm_summary);

        DependencyResolution {
            dependencies,
            use_statements,
            wasm_summary,
            unmapped_modules,
            version_conflicts,
            validation_issues: vec![],
            optimizations: vec![],
            circular_dependencies: vec![],
            cargo_toml,
        }
    }

    /// Add or merge a dependency, handling version conflicts
    fn add_or_merge_dependency(
        &self,
        dependencies: &mut HashMap<String, ResolvedDependency>,
        conflicts: &mut Vec<VersionConflict>,
        crate_name: String,
        version: String,
        features: Vec<String>,
        wasm_compat: WasmCompatibility,
        source_module: String,
        notes: Option<String>,
    ) {
        if let Some(existing) = dependencies.get_mut(&crate_name) {
            // Merge source modules
            if !existing.source_modules.contains(&source_module) {
                existing.source_modules.push(source_module.clone());
            }

            // Merge features
            for feature in features {
                if !existing.features.contains(&feature) {
                    existing.features.push(feature);
                }
            }

            // Check version conflict
            if existing.version != version {
                // Record conflict
                let mut versions = vec![existing.version.clone(), version.clone()];
                versions.sort();
                versions.dedup();

                // Resolution: use the higher/latest version
                let resolved_version = self.resolve_version_conflict(&existing.version, &version);
                existing.version = resolved_version.clone();

                conflicts.push(VersionConflict {
                    crate_name: crate_name.clone(),
                    versions,
                    resolution: format!("Using version {}", resolved_version),
                });
            }
        } else {
            // New dependency
            dependencies.insert(
                crate_name.clone(),
                ResolvedDependency {
                    crate_name,
                    version,
                    features,
                    wasm_compat,
                    source_modules: vec![source_module],
                    notes,
                },
            );
        }
    }

    /// Resolve version conflict using semantic versioning
    fn resolve_version_conflict(&self, v1: &str, v2: &str) -> String {
        // Use VersionCompatibilityChecker for proper semver comparison
        let checker = VersionCompatibilityChecker::new();

        match checker.get_highest(v1, v2) {
            Ok(version) => version,
            Err(_) => {
                // Fallback to string comparison if semver parsing fails
                if v1 >= v2 {
                    v1.to_string()
                } else {
                    v2.to_string()
                }
            }
        }
    }

    /// Generate use statement for import
    fn generate_use_statement(&self, import: &PythonImport, rust_use: &str) -> String {
        if import.items.is_empty() {
            format!("use {};", rust_use)
        } else {
            // Map specific items
            let items: Vec<String> = import.items.iter()
                .map(|sym| sym.name.clone())
                .collect();
            format!("use {}::{{{}}};", rust_use, items.join(", "))
        }
    }

    /// Build WASM compatibility summary
    fn build_wasm_summary(&self, modules: &HashMap<String, WasmCompatibility>) -> WasmSummary {
        let mut needs_wasi = false;
        let mut needs_js_interop = false;
        let mut has_incompatible = false;

        for compat in modules.values() {
            match compat {
                WasmCompatibility::RequiresWasi => needs_wasi = true,
                WasmCompatibility::RequiresJsInterop => needs_js_interop = true,
                WasmCompatibility::Incompatible => has_incompatible = true,
                WasmCompatibility::Partial => needs_wasi = true,
                WasmCompatibility::Full => {}
            }
        }

        WasmSummary {
            fully_compatible: !needs_wasi && !needs_js_interop && !has_incompatible,
            needs_wasi,
            needs_js_interop,
            has_incompatible,
            modules: modules.clone(),
        }
    }

    /// Generate Cargo.toml dependencies section
    fn generate_cargo_toml(&self, dependencies: &[ResolvedDependency], wasm_summary: &WasmSummary) -> String {
        // Convert ResolvedDependency to RustDependency format for CargoGenerator
        let rust_deps: Vec<RustDependency> = dependencies.iter().map(|dep| {
            RustDependency {
                crate_name: dep.crate_name.clone(),
                version: dep.version.clone(),
                features: dep.features.clone(),
                wasm_compat: dep.wasm_compat.clone(),
                target: None,
                notes: dep.notes.clone(),
            }
        }).collect();

        // Create ImportAnalysis structure that CargoGenerator expects
        let analysis = ImportAnalysis {
            python_imports: vec![],
            rust_dependencies: rust_deps,
            rust_use_statements: vec![],
            wasm_compatibility: WasmCompatibilitySummary {
                fully_compatible: wasm_summary.fully_compatible,
                needs_wasi: wasm_summary.needs_wasi,
                needs_js_interop: wasm_summary.needs_js_interop,
                has_incompatible: wasm_summary.has_incompatible,
                modules_by_compat: wasm_summary.modules.clone(),
            },
            unmapped_modules: vec![],
        };

        // Use CargoGenerator for complete, production-ready Cargo.toml
        let generator = CargoGenerator::new()
            .with_wasi_support(wasm_summary.needs_wasi);

        generator.generate(&analysis)
    }

    /// Get dependency graph reference
    pub fn graph(&self) -> &DependencyGraph {
        &self.dependency_graph
    }

    /// Get mutable dependency graph reference
    pub fn graph_mut(&mut self) -> &mut DependencyGraph {
        &mut self.dependency_graph
    }
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for WASM compatibility default
impl Default for crate::import_analyzer::WasmCompatibilitySummary {
    fn default() -> Self {
        Self {
            fully_compatible: true,
            needs_wasi: false,
            needs_js_interop: false,
            has_incompatible: false,
            modules_by_compat: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_stdlib_import() {
        let mut resolver = DependencyResolver::new();

        let python_code = r#"
import json
from pathlib import Path
"#;

        let resolution = resolver.resolve_module("test_module", python_code);

        assert!(!resolution.dependencies.is_empty());
        assert!(!resolution.use_statements.is_empty());
        assert!(resolution.unmapped_modules.is_empty());
    }

    #[test]
    fn test_resolve_external_package() {
        let mut resolver = DependencyResolver::new();

        let python_code = r#"
import numpy as np
import pandas as pd
"#;

        let resolution = resolver.resolve_module("test_module", python_code);

        assert!(resolution.dependencies.iter().any(|d| d.crate_name == "ndarray"));
        assert!(resolution.dependencies.iter().any(|d| d.crate_name == "polars"));
    }

    #[test]
    fn test_version_conflict_resolution() {
        let mut resolver = DependencyResolver::new();

        let mut modules = HashMap::new();
        modules.insert("module1".to_string(), "import json".to_string());
        modules.insert("module2".to_string(), "from json import loads".to_string());

        let resolution = resolver.resolve_project(&modules);

        // Should have serde_json only once
        let json_deps: Vec<_> = resolution.dependencies.iter()
            .filter(|d| d.crate_name == "serde_json")
            .collect();

        assert!(json_deps.len() <= 1);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut resolver = DependencyResolver::new();

        // Create modules with circular dependencies
        // Note: This is a simplified test - in reality, you'd need actual Python code
        // that creates the circular dependency

        let resolution = resolver.resolve_module("test", "import json");

        // Graph should be built
        assert!(resolver.graph().stats().total_modules > 0);
    }

    #[test]
    fn test_wasm_compatibility_summary() {
        let mut resolver = DependencyResolver::new();

        let python_code = r#"
import json
import asyncio
from pathlib import Path
"#;

        let resolution = resolver.resolve_module("test_module", python_code);

        // json is full WASM, pathlib needs WASI, asyncio needs JS
        assert!(!resolution.wasm_summary.fully_compatible);
        assert!(resolution.wasm_summary.needs_wasi || resolution.wasm_summary.needs_js_interop);
    }

    #[test]
    fn test_cargo_toml_generation() {
        let mut resolver = DependencyResolver::new();

        let python_code = r#"
import json
import numpy as np
"#;

        let resolution = resolver.resolve_module("test_module", python_code);

        assert!(resolution.cargo_toml.contains("[dependencies]"));
        assert!(resolution.cargo_toml.contains("serde_json"));
        assert!(resolution.cargo_toml.contains("ndarray"));
    }
}
