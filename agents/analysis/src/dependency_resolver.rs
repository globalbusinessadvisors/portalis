//! Dependency Resolution Module
//!
//! Resolves Python imports to Rust crates and generates use statements.

use portalis_core::{Error, Result};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// Dependency resolver for Python imports
pub struct DependencyResolver {
    /// Mapping of Python modules to Rust crates
    crate_mappings: HashMap<String, CrateMapping>,
    /// Internal modules in the project
    internal_modules: HashSet<String>,
}

/// Information about a Rust crate mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateMapping {
    /// Rust crate name
    pub crate_name: String,
    /// Crate version
    pub version: String,
    /// Optional features to enable
    pub features: Vec<String>,
    /// Rust path for use statement
    pub rust_path: Option<String>,
}

/// Resolved dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedDependency {
    /// Original Python import
    pub python_import: String,
    /// Type of dependency
    pub dep_type: DependencyType,
    /// Rust use statement (if external)
    pub use_statement: Option<String>,
    /// Cargo.toml entry (if external)
    pub cargo_entry: Option<CrateMapping>,
}

/// Type of dependency
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Standard library (built into Rust)
    Stdlib,
    /// Internal module in the project
    Internal,
    /// External crate
    External,
    /// Unknown/unmapped
    Unknown,
}

impl DependencyResolver {
    pub fn new() -> Self {
        let mut resolver = Self {
            crate_mappings: HashMap::new(),
            internal_modules: HashSet::new(),
        };

        // Initialize with common Python stdlib → Rust mappings
        resolver.init_stdlib_mappings();
        // Initialize with popular package mappings
        resolver.init_external_mappings();

        resolver
    }

    /// Register internal modules in the project
    pub fn register_internal_modules(&mut self, modules: Vec<String>) {
        self.internal_modules.extend(modules);
    }

    /// Resolve a Python import statement
    pub fn resolve_import(&self, module: &str, items: &[String]) -> Result<ResolvedDependency> {
        // Check if it's an internal module
        if self.is_internal_module(module) {
            return Ok(ResolvedDependency {
                python_import: module.to_string(),
                dep_type: DependencyType::Internal,
                use_statement: Some(self.generate_internal_use(module, items)),
                cargo_entry: None,
            });
        }

        // Check if it's a known stdlib or external mapping
        if let Some(mapping) = self.get_mapping(module) {
            let dep_type = if mapping.crate_name == "std" {
                DependencyType::Stdlib
            } else {
                DependencyType::External
            };

            return Ok(ResolvedDependency {
                python_import: module.to_string(),
                dep_type: dep_type.clone(),
                use_statement: Some(self.generate_use_statement(module, items, mapping)?),
                cargo_entry: if dep_type == DependencyType::External {
                    Some(mapping.clone())
                } else {
                    None
                },
            });
        }

        // Unknown module
        Ok(ResolvedDependency {
            python_import: module.to_string(),
            dep_type: DependencyType::Unknown,
            use_statement: None,
            cargo_entry: None,
        })
    }

    /// Check if a module is internal to the project
    fn is_internal_module(&self, module: &str) -> bool {
        // Check exact match
        if self.internal_modules.contains(module) {
            return true;
        }

        // Check if it's a submodule of an internal module
        for internal in &self.internal_modules {
            if module.starts_with(&format!("{}.", internal)) {
                return true;
            }
        }

        false
    }

    /// Get crate mapping for a Python module
    fn get_mapping(&self, module: &str) -> Option<&CrateMapping> {
        // Try exact match first
        if let Some(mapping) = self.crate_mappings.get(module) {
            return Some(mapping);
        }

        // Try parent module (e.g., "os.path" → "os")
        let parts: Vec<&str> = module.split('.').collect();
        if parts.len() > 1 {
            if let Some(mapping) = self.crate_mappings.get(parts[0]) {
                return Some(mapping);
            }
        }

        None
    }

    /// Generate use statement for internal module
    fn generate_internal_use(&self, module: &str, items: &[String]) -> String {
        if items.is_empty() {
            format!("use crate::{};", module.replace('.', "::"))
        } else {
            let items_str = items.join(", ");
            format!("use crate::{}::{{{}}};", module.replace('.', "::"), items_str)
        }
    }

    /// Generate use statement for external dependency
    fn generate_use_statement(
        &self,
        _module: &str,
        items: &[String],
        mapping: &CrateMapping,
    ) -> Result<String> {
        let base_path = mapping.rust_path.as_ref()
            .ok_or_else(|| Error::Parse("Missing rust_path in mapping".into()))?;

        if items.is_empty() {
            Ok(format!("use {};", base_path))
        } else {
            let items_str = items.join(", ");
            Ok(format!("use {}::{{{}}};", base_path, items_str))
        }
    }

    /// Initialize standard library mappings
    fn init_stdlib_mappings(&mut self) {
        // Math functions
        self.crate_mappings.insert("math".to_string(), CrateMapping {
            crate_name: "std".to_string(),
            version: "".to_string(),
            features: vec![],
            rust_path: Some("std::f64".to_string()),
        });

        // OS and filesystem
        self.crate_mappings.insert("os".to_string(), CrateMapping {
            crate_name: "std".to_string(),
            version: "".to_string(),
            features: vec![],
            rust_path: Some("std::env".to_string()),
        });

        self.crate_mappings.insert("os.path".to_string(), CrateMapping {
            crate_name: "std".to_string(),
            version: "".to_string(),
            features: vec![],
            rust_path: Some("std::path".to_string()),
        });

        // Collections
        self.crate_mappings.insert("collections".to_string(), CrateMapping {
            crate_name: "std".to_string(),
            version: "".to_string(),
            features: vec![],
            rust_path: Some("std::collections".to_string()),
        });

        // JSON
        self.crate_mappings.insert("json".to_string(), CrateMapping {
            crate_name: "serde_json".to_string(),
            version: "1.0".to_string(),
            features: vec![],
            rust_path: Some("serde_json".to_string()),
        });

        // Time/datetime
        self.crate_mappings.insert("datetime".to_string(), CrateMapping {
            crate_name: "chrono".to_string(),
            version: "0.4".to_string(),
            features: vec![],
            rust_path: Some("chrono".to_string()),
        });

        // Random
        self.crate_mappings.insert("random".to_string(), CrateMapping {
            crate_name: "rand".to_string(),
            version: "0.8".to_string(),
            features: vec![],
            rust_path: Some("rand".to_string()),
        });

        // Typing (no direct equivalent, handled by type system)
        self.crate_mappings.insert("typing".to_string(), CrateMapping {
            crate_name: "std".to_string(),
            version: "".to_string(),
            features: vec![],
            rust_path: Some("std".to_string()),
        });
    }

    /// Initialize external package mappings
    fn init_external_mappings(&mut self) {
        // NumPy → ndarray
        self.crate_mappings.insert("numpy".to_string(), CrateMapping {
            crate_name: "ndarray".to_string(),
            version: "0.15".to_string(),
            features: vec![],
            rust_path: Some("ndarray".to_string()),
        });

        // Pandas → polars
        self.crate_mappings.insert("pandas".to_string(), CrateMapping {
            crate_name: "polars".to_string(),
            version: "0.35".to_string(),
            features: vec!["lazy".to_string()],
            rust_path: Some("polars::prelude".to_string()),
        });

        // Requests → reqwest
        self.crate_mappings.insert("requests".to_string(), CrateMapping {
            crate_name: "reqwest".to_string(),
            version: "0.11".to_string(),
            features: vec!["json".to_string()],
            rust_path: Some("reqwest".to_string()),
        });

        // Flask → actix-web
        self.crate_mappings.insert("flask".to_string(), CrateMapping {
            crate_name: "actix-web".to_string(),
            version: "4.0".to_string(),
            features: vec![],
            rust_path: Some("actix_web".to_string()),
        });

        // Pytest → (built-in testing)
        self.crate_mappings.insert("pytest".to_string(), CrateMapping {
            crate_name: "std".to_string(),
            version: "".to_string(),
            features: vec![],
            rust_path: Some("std".to_string()),
        });

        // Pathlib → std::path
        self.crate_mappings.insert("pathlib".to_string(), CrateMapping {
            crate_name: "std".to_string(),
            version: "".to_string(),
            features: vec![],
            rust_path: Some("std::path".to_string()),
        });

        // Regex
        self.crate_mappings.insert("re".to_string(), CrateMapping {
            crate_name: "regex".to_string(),
            version: "1.0".to_string(),
            features: vec![],
            rust_path: Some("regex".to_string()),
        });
    }

    /// Get all required Cargo dependencies
    pub fn get_cargo_dependencies(&self, resolved: &[ResolvedDependency]) -> Vec<CrateMapping> {
        let mut deps = Vec::new();
        let mut seen = HashSet::new();

        for dep in resolved {
            if let Some(ref cargo_entry) = dep.cargo_entry {
                if !seen.contains(&cargo_entry.crate_name) {
                    deps.push(cargo_entry.clone());
                    seen.insert(cargo_entry.crate_name.clone());
                }
            }
        }

        deps
    }

    /// Generate all use statements for a module
    pub fn generate_use_statements(&self, resolved: &[ResolvedDependency]) -> Vec<String> {
        let mut statements = Vec::new();
        let mut seen = HashSet::new();

        for dep in resolved {
            if let Some(ref use_stmt) = dep.use_statement {
                if !seen.contains(use_stmt) {
                    statements.push(use_stmt.clone());
                    seen.insert(use_stmt.clone());
                }
            }
        }

        statements.sort();
        statements
    }
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_stdlib_import() {
        let resolver = DependencyResolver::new();

        let result = resolver.resolve_import("math", &["sqrt".to_string()]).unwrap();

        assert_eq!(result.dep_type, DependencyType::Stdlib);
        assert!(result.use_statement.is_some());
        assert!(result.use_statement.unwrap().contains("std::f64"));
    }

    #[test]
    fn test_resolve_external_import() {
        let resolver = DependencyResolver::new();

        let result = resolver.resolve_import("numpy", &["array".to_string()]).unwrap();

        assert_eq!(result.dep_type, DependencyType::External);
        assert!(result.cargo_entry.is_some());
        assert_eq!(result.cargo_entry.unwrap().crate_name, "ndarray");
    }

    #[test]
    fn test_resolve_internal_import() {
        let mut resolver = DependencyResolver::new();
        resolver.register_internal_modules(vec!["mymodule".to_string()]);

        let result = resolver.resolve_import("mymodule", &["MyClass".to_string()]).unwrap();

        assert_eq!(result.dep_type, DependencyType::Internal);
        assert!(result.use_statement.is_some());
        assert!(result.use_statement.unwrap().contains("use crate::mymodule"));
    }

    #[test]
    fn test_resolve_unknown_import() {
        let resolver = DependencyResolver::new();

        let result = resolver.resolve_import("unknown_package", &[]).unwrap();

        assert_eq!(result.dep_type, DependencyType::Unknown);
        assert!(result.use_statement.is_none());
    }

    #[test]
    fn test_generate_use_statements() {
        let resolver = DependencyResolver::new();

        let resolved = vec![
            resolver.resolve_import("math", &["sqrt".to_string()]).unwrap(),
            resolver.resolve_import("numpy", &["array".to_string()]).unwrap(),
        ];

        let statements = resolver.generate_use_statements(&resolved);

        assert_eq!(statements.len(), 2);
        assert!(statements.iter().any(|s| s.contains("std::f64")));
        assert!(statements.iter().any(|s| s.contains("ndarray")));
    }

    #[test]
    fn test_get_cargo_dependencies() {
        let resolver = DependencyResolver::new();

        let resolved = vec![
            resolver.resolve_import("math", &[]).unwrap(),
            resolver.resolve_import("numpy", &[]).unwrap(),
            resolver.resolve_import("requests", &[]).unwrap(),
        ];

        let deps = resolver.get_cargo_dependencies(&resolved);

        // Should only include external deps (not stdlib)
        assert_eq!(deps.len(), 2);
        assert!(deps.iter().any(|d| d.crate_name == "ndarray"));
        assert!(deps.iter().any(|d| d.crate_name == "reqwest"));
    }

    #[test]
    fn test_submodule_resolution() {
        let mut resolver = DependencyResolver::new();
        resolver.register_internal_modules(vec!["mypackage".to_string()]);

        let result = resolver.resolve_import("mypackage.submodule", &[]).unwrap();

        assert_eq!(result.dep_type, DependencyType::Internal);
    }
}
