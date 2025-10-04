//! Import Analyzer - Detects Python imports and maps to Rust dependencies
//!
//! Analyzes Python source code to:
//! 1. Detect all import statements
//! 2. Map Python modules to Rust crates
//! 3. Track WASM compatibility
//! 4. Generate Cargo.toml dependencies

use crate::stdlib_mapper::{StdlibMapper, WasmCompatibility};
use crate::external_packages::ExternalPackageRegistry;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Represents a detected Python import
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PythonImport {
    /// Module name (e.g., "json", "pathlib")
    pub module: String,

    /// Specific items imported (e.g., ["Path", "exists"] from pathlib)
    pub items: Vec<String>,

    /// Import type
    pub import_type: ImportType,

    /// Alias if any (e.g., "import numpy as np")
    pub alias: Option<String>,
}

/// Type of import statement
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ImportType {
    /// import module
    Module,

    /// from module import item
    FromImport,

    /// from module import *
    StarImport,
}

/// Rust crate dependency with WASM info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustDependency {
    /// Crate name
    pub crate_name: String,

    /// Version requirement
    pub version: String,

    /// Features needed
    pub features: Vec<String>,

    /// WASM compatibility level
    pub wasm_compat: WasmCompatibility,

    /// Optional target-specific (e.g., only for wasm32)
    pub target: Option<String>,

    /// Additional notes
    pub notes: Option<String>,
}

/// Analysis result with dependencies and compatibility info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportAnalysis {
    /// Detected Python imports
    pub python_imports: Vec<PythonImport>,

    /// Required Rust dependencies
    pub rust_dependencies: Vec<RustDependency>,

    /// Rust use statements to add
    pub rust_use_statements: Vec<String>,

    /// WASM compatibility summary
    pub wasm_compatibility: WasmCompatibilitySummary,

    /// Unmapped modules (need manual mapping)
    pub unmapped_modules: Vec<String>,
}

/// WASM compatibility summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmCompatibilitySummary {
    /// Can compile to WASM without any issues
    pub fully_compatible: bool,

    /// Requires WASI support
    pub needs_wasi: bool,

    /// Requires JavaScript interop
    pub needs_js_interop: bool,

    /// Has incompatible modules
    pub has_incompatible: bool,

    /// Detailed breakdown
    pub modules_by_compat: HashMap<String, WasmCompatibility>,
}

/// Import Analyzer - analyzes Python imports
pub struct ImportAnalyzer {
    stdlib_mapper: StdlibMapper,
    external_registry: ExternalPackageRegistry,
}

impl ImportAnalyzer {
    /// Create new import analyzer
    pub fn new() -> Self {
        Self {
            stdlib_mapper: StdlibMapper::new(),
            external_registry: ExternalPackageRegistry::new(),
        }
    }

    /// Analyze Python source code and extract import information
    pub fn analyze(&self, python_code: &str) -> ImportAnalysis {
        let python_imports = self.extract_imports(python_code);
        let mut rust_dependencies = Vec::new();
        let mut rust_use_statements = Vec::new();
        let mut unmapped_modules = Vec::new();
        let mut modules_by_compat = HashMap::new();

        for import in &python_imports {
            // Try stdlib first
            if let Some(module_mapping) = self.stdlib_mapper.get_module(&import.module) {
                // Generate Rust dependency
                let rust_dep = RustDependency {
                    crate_name: module_mapping.rust_crate.clone()
                        .unwrap_or_else(|| "std".to_string()),
                    version: module_mapping.version.clone(),
                    features: vec![],
                    wasm_compat: module_mapping.wasm_compatible.clone(),
                    target: None,
                    notes: module_mapping.notes.clone(),
                };

                // Track compatibility
                modules_by_compat.insert(
                    import.module.clone(),
                    module_mapping.wasm_compatible.clone()
                );

                // Generate use statement
                if !module_mapping.rust_use.is_empty() {
                    let use_stmt = if import.items.is_empty() {
                        format!("use {};", module_mapping.rust_use)
                    } else {
                        // Map specific items
                        let items_str = import.items.join(", ");
                        format!("use {}::{{{}}};", module_mapping.rust_use, items_str)
                    };
                    rust_use_statements.push(use_stmt);
                }

                // Add dependency if not stdlib
                if module_mapping.rust_crate.is_some() {
                    rust_dependencies.push(rust_dep);
                }
            }
            // Try external packages
            else if let Some(pkg_mapping) = self.external_registry.get_package(&import.module) {
                let rust_dep = RustDependency {
                    crate_name: pkg_mapping.rust_crate.clone(),
                    version: pkg_mapping.version.clone(),
                    features: pkg_mapping.features.clone(),
                    wasm_compat: pkg_mapping.wasm_compatible.clone(),
                    target: None,
                    notes: pkg_mapping.notes.clone(),
                };

                modules_by_compat.insert(
                    import.module.clone(),
                    pkg_mapping.wasm_compatible.clone()
                );

                // Generate use statement
                rust_use_statements.push(format!("use {};", pkg_mapping.rust_crate));

                rust_dependencies.push(rust_dep);
            } else {
                // Module not yet mapped
                unmapped_modules.push(import.module.clone());
            }
        }

        // Deduplicate dependencies
        rust_dependencies.sort_by(|a, b| a.crate_name.cmp(&b.crate_name));
        rust_dependencies.dedup_by(|a, b| a.crate_name == b.crate_name);

        rust_use_statements.sort();
        rust_use_statements.dedup();

        unmapped_modules.sort();
        unmapped_modules.dedup();

        // Build WASM compatibility summary
        let wasm_compatibility = self.build_wasm_summary(&modules_by_compat);

        ImportAnalysis {
            python_imports,
            rust_dependencies,
            rust_use_statements,
            wasm_compatibility,
            unmapped_modules,
        }
    }

    /// Extract import statements from Python code
    fn extract_imports(&self, python_code: &str) -> Vec<PythonImport> {
        let mut imports = Vec::new();

        for line in python_code.lines() {
            let trimmed = line.trim();

            // Skip comments and empty lines
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Parse "import module" or "import module as alias"
            if trimmed.starts_with("import ") {
                if let Some(import) = self.parse_import_statement(trimmed) {
                    imports.push(import);
                }
            }

            // Parse "from module import item1, item2"
            else if trimmed.starts_with("from ") {
                if let Some(import) = self.parse_from_import(trimmed) {
                    imports.push(import);
                }
            }
        }

        imports
    }

    /// Parse "import module" or "import module as alias"
    fn parse_import_statement(&self, line: &str) -> Option<PythonImport> {
        let line = line.strip_prefix("import ")?.trim();

        // Handle "import module as alias"
        if let Some(as_pos) = line.find(" as ") {
            let module = line[..as_pos].trim().to_string();
            let alias = line[as_pos + 4..].trim().to_string();

            return Some(PythonImport {
                module,
                items: vec![],
                import_type: ImportType::Module,
                alias: Some(alias),
            });
        }

        // Handle "import module"
        let module = line.split(',').next()?.trim().to_string();

        Some(PythonImport {
            module,
            items: vec![],
            import_type: ImportType::Module,
            alias: None,
        })
    }

    /// Parse "from module import item" or "from module import *"
    fn parse_from_import(&self, line: &str) -> Option<PythonImport> {
        let line = line.strip_prefix("from ")?.trim();

        // Find "import" keyword
        let import_pos = line.find(" import ")?;
        let module = line[..import_pos].trim().to_string();
        let items_str = line[import_pos + 8..].trim();

        // Handle "from module import *"
        if items_str == "*" {
            return Some(PythonImport {
                module,
                items: vec![],
                import_type: ImportType::StarImport,
                alias: None,
            });
        }

        // Parse individual items
        let items: Vec<String> = items_str
            .split(',')
            .map(|s| {
                // Handle "item as alias" - just take the item name
                s.trim()
                    .split(" as ")
                    .next()
                    .unwrap_or(s.trim())
                    .to_string()
            })
            .collect();

        Some(PythonImport {
            module,
            items,
            import_type: ImportType::FromImport,
            alias: None,
        })
    }

    /// Build WASM compatibility summary
    fn build_wasm_summary(
        &self,
        modules_by_compat: &HashMap<String, WasmCompatibility>
    ) -> WasmCompatibilitySummary {
        let mut needs_wasi = false;
        let mut needs_js_interop = false;
        let mut has_incompatible = false;

        for compat in modules_by_compat.values() {
            match compat {
                WasmCompatibility::RequiresWasi => needs_wasi = true,
                WasmCompatibility::RequiresJsInterop => needs_js_interop = true,
                WasmCompatibility::Incompatible => has_incompatible = true,
                WasmCompatibility::Partial => {
                    // Some functions work, some don't
                    needs_wasi = true;
                }
                WasmCompatibility::Full => {}
            }
        }

        let fully_compatible = !needs_wasi && !needs_js_interop && !has_incompatible;

        WasmCompatibilitySummary {
            fully_compatible,
            needs_wasi,
            needs_js_interop,
            has_incompatible,
            modules_by_compat: modules_by_compat.clone(),
        }
    }

    /// Generate Cargo.toml dependencies section
    pub fn generate_cargo_toml_deps(&self, analysis: &ImportAnalysis) -> String {
        let mut output = String::new();
        output.push_str("[dependencies]\n");

        for dep in &analysis.rust_dependencies {
            if dep.target.is_none() {
                if dep.features.is_empty() {
                    output.push_str(&format!("{} = \"{}\"\n", dep.crate_name, dep.version));
                } else {
                    output.push_str(&format!(
                        "{} = {{ version = \"{}\", features = [{}] }}\n",
                        dep.crate_name,
                        dep.version,
                        dep.features.iter()
                            .map(|f| format!("\"{}\"", f))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }

        // Add target-specific dependencies
        output.push_str("\n[target.'cfg(target_arch = \"wasm32\")'.dependencies]\n");

        // Add wasm-bindgen if JS interop needed
        if analysis.wasm_compatibility.needs_js_interop {
            output.push_str("wasm-bindgen = \"0.2\"\n");
            output.push_str("wasm-bindgen-futures = \"0.4\"\n");
            output.push_str("js-sys = \"0.3\"\n");
        }

        // Add WASI if needed
        if analysis.wasm_compatibility.needs_wasi {
            output.push_str("\n[target.'cfg(target_arch = \"wasm32\")'.dependencies.wasi]\n");
            output.push_str("version = \"0.11\"\n");
            output.push_str("optional = true\n");
        }

        output
    }

    /// Generate compatibility report
    pub fn generate_compatibility_report(&self, analysis: &ImportAnalysis) -> String {
        let mut report = String::new();

        report.push_str("# WASM Compatibility Report\n\n");

        if analysis.wasm_compatibility.fully_compatible {
            report.push_str("✅ **Fully WASM Compatible** - No special requirements\n\n");
        } else {
            report.push_str("## Compatibility Status\n\n");

            if analysis.wasm_compatibility.needs_wasi {
                report.push_str("⚠️  **Requires WASI** - Needs filesystem/OS support\n");
            }

            if analysis.wasm_compatibility.needs_js_interop {
                report.push_str("🌐 **Requires JS Interop** - Needs browser APIs\n");
            }

            if analysis.wasm_compatibility.has_incompatible {
                report.push_str("❌ **Has Incompatible Modules** - Some features won't work in WASM\n");
            }

            report.push_str("\n");
        }

        // Module-by-module breakdown
        report.push_str("## Module Compatibility\n\n");

        let mut modules: Vec<_> = analysis.wasm_compatibility.modules_by_compat.iter().collect();
        modules.sort_by_key(|(name, _)| *name);

        for (module, compat) in modules {
            let icon = match compat {
                WasmCompatibility::Full => "✅",
                WasmCompatibility::Partial => "🟡",
                WasmCompatibility::RequiresWasi => "📁",
                WasmCompatibility::RequiresJsInterop => "🌐",
                WasmCompatibility::Incompatible => "❌",
            };

            report.push_str(&format!("{} `{}` - {:?}\n", icon, module, compat));
        }

        // Unmapped modules warning
        if !analysis.unmapped_modules.is_empty() {
            report.push_str("\n## ⚠️  Unmapped Modules\n\n");
            report.push_str("The following modules are not yet mapped to Rust crates:\n\n");

            for module in &analysis.unmapped_modules {
                report.push_str(&format!("- `{}`\n", module));
            }

            report.push_str("\nThese will need manual implementation or mapping.\n");
        }

        report
    }
}

impl Default for ImportAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_import() {
        let analyzer = ImportAnalyzer::new();
        let imports = analyzer.extract_imports("import json");

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "json");
        assert_eq!(imports[0].import_type, ImportType::Module);
        assert!(imports[0].items.is_empty());
    }

    #[test]
    fn test_parse_import_with_alias() {
        let analyzer = ImportAnalyzer::new();
        let imports = analyzer.extract_imports("import datetime as dt");

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "datetime");
        assert_eq!(imports[0].alias, Some("dt".to_string()));
    }

    #[test]
    fn test_parse_from_import() {
        let analyzer = ImportAnalyzer::new();
        let imports = analyzer.extract_imports("from pathlib import Path");

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "pathlib");
        assert_eq!(imports[0].items, vec!["Path"]);
        assert_eq!(imports[0].import_type, ImportType::FromImport);
    }

    #[test]
    fn test_parse_from_import_multiple() {
        let analyzer = ImportAnalyzer::new();
        let imports = analyzer.extract_imports("from datetime import datetime, timedelta");

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "datetime");
        assert_eq!(imports[0].items, vec!["datetime", "timedelta"]);
    }

    #[test]
    fn test_parse_star_import() {
        let analyzer = ImportAnalyzer::new();
        let imports = analyzer.extract_imports("from os import *");

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "os");
        assert_eq!(imports[0].import_type, ImportType::StarImport);
    }

    #[test]
    fn test_analyze_mapped_modules() {
        let analyzer = ImportAnalyzer::new();
        let code = r#"
import json
from pathlib import Path
import logging
"#;

        let analysis = analyzer.analyze(code);

        assert_eq!(analysis.python_imports.len(), 3);
        assert!(!analysis.rust_use_statements.is_empty());
        assert!(analysis.unmapped_modules.is_empty());
    }

    #[test]
    fn test_analyze_unmapped_module() {
        let analyzer = ImportAnalyzer::new();
        let code = "import nonexistent_module";

        let analysis = analyzer.analyze(code);

        assert_eq!(analysis.unmapped_modules.len(), 1);
        assert_eq!(analysis.unmapped_modules[0], "nonexistent_module");
    }

    #[test]
    fn test_wasm_compatibility_summary() {
        let analyzer = ImportAnalyzer::new();
        let code = r#"
import json
from pathlib import Path
import asyncio
"#;

        let analysis = analyzer.analyze(code);

        // json is full WASM, pathlib needs WASI, asyncio needs JS interop
        assert!(!analysis.wasm_compatibility.fully_compatible);
        assert!(analysis.wasm_compatibility.needs_wasi);
        assert!(analysis.wasm_compatibility.needs_js_interop);
    }

    #[test]
    fn test_generate_cargo_toml() {
        let analyzer = ImportAnalyzer::new();
        let code = "import json\nfrom datetime import datetime";

        let analysis = analyzer.analyze(code);
        let cargo_toml = analyzer.generate_cargo_toml_deps(&analysis);

        assert!(cargo_toml.contains("[dependencies]"));
    }
}
