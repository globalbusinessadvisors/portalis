//! Cargo.toml Generator - Auto-generates Cargo.toml from Python code
//!
//! Generates complete Cargo.toml with:
//! - Project metadata (name, version, edition)
//! - Dependencies (from import analysis)
//! - WASM-specific configuration
//! - Build profiles (dev, release, wasm)
//! - Features for optional functionality

use crate::import_analyzer::{ImportAnalysis, RustDependency};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Cargo.toml generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoConfig {
    /// Package name (defaults to "transpiled_project")
    pub package_name: String,

    /// Package version (defaults to "0.1.0")
    pub version: String,

    /// Rust edition (defaults to "2021")
    pub edition: String,

    /// Authors list
    pub authors: Vec<String>,

    /// Package description
    pub description: Option<String>,

    /// License (e.g., "MIT", "Apache-2.0")
    pub license: Option<String>,

    /// Enable WASM optimizations
    pub wasm_optimized: bool,

    /// Enable WASI support
    pub wasi_support: bool,

    /// Additional features to include
    pub features: Vec<String>,
}

impl Default for CargoConfig {
    fn default() -> Self {
        Self {
            package_name: "transpiled_project".to_string(),
            version: "0.1.0".to_string(),
            edition: "2021".to_string(),
            authors: vec!["Portalis Transpiler <noreply@portalis.dev>".to_string()],
            description: Some("Generated from Python by Portalis".to_string()),
            license: Some("MIT".to_string()),
            wasm_optimized: true,
            wasi_support: false,
            features: vec![],
        }
    }
}

/// Cargo.toml generator
pub struct CargoGenerator {
    config: CargoConfig,
}

impl CargoGenerator {
    /// Create new generator with default config
    pub fn new() -> Self {
        Self {
            config: CargoConfig::default(),
        }
    }

    /// Create generator with custom config
    pub fn with_config(config: CargoConfig) -> Self {
        Self { config }
    }

    /// Generate complete Cargo.toml from import analysis
    pub fn generate(&self, analysis: &ImportAnalysis) -> String {
        let mut output = String::new();

        // Package section
        output.push_str(&self.generate_package_section());
        output.push('\n');

        // Dependencies section
        output.push_str(&self.generate_dependencies_section(analysis));
        output.push('\n');

        // WASM-specific dependencies
        if analysis.wasm_compatibility.needs_js_interop || analysis.wasm_compatibility.needs_wasi {
            output.push_str(&self.generate_wasm_dependencies(analysis));
            output.push('\n');
        }

        // Features section
        if !self.config.features.is_empty() || analysis.wasm_compatibility.needs_wasi {
            output.push_str(&self.generate_features_section(analysis));
            output.push('\n');
        }

        // Profile sections
        output.push_str(&self.generate_profiles_section());
        output.push('\n');

        // Lib section for WASM
        output.push_str(&self.generate_lib_section(analysis));

        output
    }

    /// Generate [package] section
    fn generate_package_section(&self) -> String {
        let mut section = String::new();

        section.push_str("[package]\n");
        section.push_str(&format!("name = \"{}\"\n", self.config.package_name));
        section.push_str(&format!("version = \"{}\"\n", self.config.version));
        section.push_str(&format!("edition = \"{}\"\n", self.config.edition));

        if !self.config.authors.is_empty() {
            section.push_str(&format!("authors = [{}]\n",
                self.config.authors.iter()
                    .map(|a| format!("\"{}\"", a))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        if let Some(ref desc) = self.config.description {
            section.push_str(&format!("description = \"{}\"\n", desc));
        }

        if let Some(ref license) = self.config.license {
            section.push_str(&format!("license = \"{}\"\n", license));
        }

        section
    }

    /// Generate [dependencies] section
    fn generate_dependencies_section(&self, analysis: &ImportAnalysis) -> String {
        let mut section = String::new();
        section.push_str("[dependencies]\n");

        // Collect unique dependencies
        let mut deps: Vec<&RustDependency> = analysis.rust_dependencies.iter()
            .filter(|d| d.target.is_none())
            .collect();

        deps.sort_by(|a, b| a.crate_name.cmp(&b.crate_name));

        for dep in deps {
            if dep.features.is_empty() {
                section.push_str(&format!("{} = \"{}\"\n", dep.crate_name, dep.version));
            } else {
                section.push_str(&format!(
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

        // Add common WASM dependencies if needed
        if analysis.wasm_compatibility.needs_js_interop {
            section.push_str("serde = { version = \"1\", features = [\"derive\"] }\n");
            section.push_str("serde-wasm-bindgen = \"0.6\"\n");
        }

        section
    }

    /// Generate WASM-specific dependencies
    fn generate_wasm_dependencies(&self, analysis: &ImportAnalysis) -> String {
        let mut section = String::new();

        section.push_str("[target.'cfg(target_arch = \"wasm32\")'.dependencies]\n");

        if analysis.wasm_compatibility.needs_js_interop {
            section.push_str("wasm-bindgen = \"0.2\"\n");
            section.push_str("wasm-bindgen-futures = \"0.4\"\n");
            section.push_str("js-sys = \"0.3\"\n");
            section.push_str("web-sys = { version = \"0.3\", features = [\"console\"] }\n");

            // Add getrandom for random number generation
            if self.needs_random(analysis) {
                section.push_str("getrandom = { version = \"0.2\", features = [\"js\"] }\n");
            }

            // Add wasm-timer for time operations
            if self.needs_timers(analysis) {
                section.push_str("wasm-timer = \"0.2\"\n");
            }
        }

        if analysis.wasm_compatibility.needs_wasi || self.config.wasi_support {
            section.push_str("wasi = { version = \"0.11\", optional = true }\n");
        }

        section
    }

    /// Generate [features] section
    fn generate_features_section(&self, analysis: &ImportAnalysis) -> String {
        let mut section = String::new();
        section.push_str("[features]\n");

        // Default features
        let mut default_features = vec![];

        if analysis.wasm_compatibility.needs_wasi || self.config.wasi_support {
            section.push_str("wasi = [\"dep:wasi\"]\n");
            default_features.push("wasi");
        }

        if !default_features.is_empty() {
            section.push_str(&format!("default = [{}]\n",
                default_features.iter()
                    .map(|f| format!("\"{}\"", f))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        // User-defined features
        for feature in &self.config.features {
            section.push_str(&format!("{} = []\n", feature));
        }

        section
    }

    /// Generate build profiles
    fn generate_profiles_section(&self) -> String {
        let mut section = String::new();

        if self.config.wasm_optimized {
            section.push_str("# Optimized release profile for WASM\n");
            section.push_str("[profile.release]\n");
            section.push_str("opt-level = \"z\"     # Optimize for size\n");
            section.push_str("lto = true           # Enable Link Time Optimization\n");
            section.push_str("codegen-units = 1    # Reduce parallel codegen for better optimization\n");
            section.push_str("panic = \"abort\"      # Abort on panic (smaller binary)\n");
            section.push_str("strip = true         # Strip symbols from binary\n");
            section.push('\n');

            section.push_str("# Development profile\n");
            section.push_str("[profile.dev]\n");
            section.push_str("opt-level = 0\n");
            section.push('\n');

            section.push_str("# WASM-specific profile\n");
            section.push_str("[profile.wasm]\n");
            section.push_str("inherits = \"release\"\n");
            section.push_str("opt-level = \"z\"\n");
            section.push_str("lto = true\n");
            section.push_str("codegen-units = 1\n");
        } else {
            section.push_str("[profile.release]\n");
            section.push_str("opt-level = 3\n");
        }

        section
    }

    /// Generate [lib] section for WASM
    fn generate_lib_section(&self, analysis: &ImportAnalysis) -> String {
        let mut section = String::new();

        // Only add lib section if targeting WASM
        if analysis.wasm_compatibility.needs_js_interop || !analysis.wasm_compatibility.fully_compatible {
            section.push_str("[lib]\n");
            section.push_str("crate-type = [\"cdylib\", \"rlib\"]\n");
        }

        section
    }

    /// Check if project needs random number generation
    fn needs_random(&self, analysis: &ImportAnalysis) -> bool {
        analysis.python_imports.iter().any(|imp| {
            matches!(imp.module.as_str(), "random" | "secrets" | "uuid")
        })
    }

    /// Check if project needs timers
    fn needs_timers(&self, analysis: &ImportAnalysis) -> bool {
        analysis.python_imports.iter().any(|imp| {
            matches!(imp.module.as_str(), "time" | "datetime" | "asyncio")
        })
    }

    /// Generate complete Cargo workspace configuration
    pub fn generate_workspace(&self, projects: Vec<(&str, &ImportAnalysis)>) -> String {
        let mut output = String::new();

        output.push_str("[workspace]\n");
        output.push_str("members = [\n");
        for (name, _) in &projects {
            output.push_str(&format!("    \"{}\",\n", name));
        }
        output.push_str("]\n\n");

        output.push_str("# Shared workspace dependencies\n");
        output.push_str("[workspace.dependencies]\n");

        // Collect common dependencies across all projects
        let mut all_deps: HashSet<String> = HashSet::new();
        for (_, analysis) in &projects {
            for dep in &analysis.rust_dependencies {
                all_deps.insert(dep.crate_name.clone());
            }
        }

        let mut sorted_deps: Vec<_> = all_deps.into_iter().collect();
        sorted_deps.sort();

        for dep in sorted_deps {
            output.push_str(&format!("{} = \"*\"\n", dep));
        }

        output
    }

    /// Generate .cargo/config.toml for WASM build configuration
    pub fn generate_cargo_config(&self, analysis: &ImportAnalysis) -> String {
        let mut output = String::new();

        output.push_str("# Cargo build configuration for WASM\n\n");

        output.push_str("[build]\n");
        output.push_str("target = \"wasm32-unknown-unknown\"\n\n");

        if analysis.wasm_compatibility.needs_wasi {
            output.push_str("# Alternative WASI target\n");
            output.push_str("# target = \"wasm32-wasi\"\n\n");
        }

        output.push_str("[target.wasm32-unknown-unknown]\n");
        output.push_str("rustflags = [\n");
        output.push_str("    \"-C\", \"link-arg=-s\",  # Strip debug symbols\n");
        output.push_str("]\n\n");

        if self.config.wasm_optimized {
            output.push_str("# WASM optimization flags\n");
            output.push_str("[profile.release.package.\"*\"]\n");
            output.push_str("opt-level = \"z\"\n");
            output.push_str("lto = true\n");
        }

        output
    }

    /// Update config with detected WASI needs
    pub fn with_wasi_support(mut self, needs_wasi: bool) -> Self {
        self.config.wasi_support = needs_wasi;
        self
    }

    /// Set package name
    pub fn with_package_name(mut self, name: String) -> Self {
        self.config.package_name = name;
        self
    }

    /// Set package version
    pub fn with_version(mut self, version: String) -> Self {
        self.config.version = version;
        self
    }
}

impl Default for CargoGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import_analyzer::{ImportAnalyzer, PythonImport, ImportType};

    #[test]
    fn test_generate_basic_cargo_toml() {
        let generator = CargoGenerator::new();
        let analyzer = ImportAnalyzer::new();

        let code = "import json";
        let analysis = analyzer.analyze(code);

        let cargo_toml = generator.generate(&analysis);

        assert!(cargo_toml.contains("[package]"));
        assert!(cargo_toml.contains("name = \"transpiled_project\""));
        assert!(cargo_toml.contains("version = \"0.1.0\""));
        assert!(cargo_toml.contains("edition = \"2021\""));
    }

    #[test]
    fn test_generate_with_dependencies() {
        let generator = CargoGenerator::new();
        let analyzer = ImportAnalyzer::new();

        let code = r#"
import json
from datetime import datetime
"#;
        let analysis = analyzer.analyze(code);

        let cargo_toml = generator.generate(&analysis);

        assert!(cargo_toml.contains("[dependencies]"));
        assert!(cargo_toml.contains("serde_json"));
        assert!(cargo_toml.contains("chrono"));
    }

    #[test]
    fn test_generate_wasm_dependencies() {
        let generator = CargoGenerator::new();
        let analyzer = ImportAnalyzer::new();

        let code = "import asyncio\nimport uuid";
        let analysis = analyzer.analyze(code);

        let cargo_toml = generator.generate(&analysis);

        assert!(cargo_toml.contains("wasm-bindgen"));
        assert!(cargo_toml.contains("getrandom"));
    }

    #[test]
    fn test_generate_with_wasi() {
        let generator = CargoGenerator::new().with_wasi_support(true);
        let analyzer = ImportAnalyzer::new();

        let code = "from pathlib import Path";
        let analysis = analyzer.analyze(code);

        let cargo_toml = generator.generate(&analysis);

        assert!(cargo_toml.contains("wasi"));
        assert!(cargo_toml.contains("[features]"));
    }

    #[test]
    fn test_generate_optimized_profile() {
        let generator = CargoGenerator::new();
        let analyzer = ImportAnalyzer::new();

        let code = "import json";
        let analysis = analyzer.analyze(code);

        let cargo_toml = generator.generate(&analysis);

        assert!(cargo_toml.contains("[profile.release]"));
        assert!(cargo_toml.contains("opt-level = \"z\""));
        assert!(cargo_toml.contains("lto = true"));
    }

    #[test]
    fn test_generate_lib_section() {
        let generator = CargoGenerator::new();
        let analyzer = ImportAnalyzer::new();

        let code = "import asyncio";
        let analysis = analyzer.analyze(code);

        let cargo_toml = generator.generate(&analysis);

        assert!(cargo_toml.contains("[lib]"));
        assert!(cargo_toml.contains("crate-type = [\"cdylib\", \"rlib\"]"));
    }

    #[test]
    fn test_custom_config() {
        let config = CargoConfig {
            package_name: "my_project".to_string(),
            version: "1.0.0".to_string(),
            edition: "2021".to_string(),
            authors: vec!["Alice <alice@example.com>".to_string()],
            description: Some("Custom project".to_string()),
            license: Some("Apache-2.0".to_string()),
            wasm_optimized: true,
            wasi_support: false,
            features: vec!["custom".to_string()],
        };

        let generator = CargoGenerator::with_config(config);
        let analyzer = ImportAnalyzer::new();

        let code = "import json";
        let analysis = analyzer.analyze(code);

        let cargo_toml = generator.generate(&analysis);

        assert!(cargo_toml.contains("name = \"my_project\""));
        assert!(cargo_toml.contains("version = \"1.0.0\""));
        assert!(cargo_toml.contains("Alice <alice@example.com>"));
        assert!(cargo_toml.contains("Apache-2.0"));
    }

    #[test]
    fn test_generate_cargo_config() {
        let generator = CargoGenerator::new();
        let analyzer = ImportAnalyzer::new();

        let code = "import json";
        let analysis = analyzer.analyze(code);

        let cargo_config = generator.generate_cargo_config(&analysis);

        assert!(cargo_config.contains("[build]"));
        assert!(cargo_config.contains("target = \"wasm32-unknown-unknown\""));
        assert!(cargo_config.contains("rustflags"));
    }
}
