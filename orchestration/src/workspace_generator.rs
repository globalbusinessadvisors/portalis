//! Workspace Generator Module
//!
//! Generates Cargo workspace structure for multi-crate Rust projects.

use portalis_core::{Error, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Workspace generator configuration
pub struct WorkspaceGenerator {
    root_path: PathBuf,
}

/// A single crate in the workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateInfo {
    pub name: String,
    pub path: String,
    pub rust_code: String,
    pub dependencies: Vec<String>,
    pub external_deps: Vec<ExternalDependency>,
}

/// External dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalDependency {
    pub name: String,
    pub version: String,
    pub features: Vec<String>,
}

/// Workspace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    pub name: String,
    pub crates: Vec<CrateInfo>,
}

impl WorkspaceGenerator {
    pub fn new(root_path: PathBuf) -> Self {
        Self { root_path }
    }

    /// Generate complete workspace structure
    pub fn generate(&self, config: &WorkspaceConfig) -> Result<()> {
        // Create root directory
        fs::create_dir_all(&self.root_path)
            .map_err(|e| Error::Pipeline(format!("Failed to create workspace root: {}", e)))?;

        // Generate root Cargo.toml
        self.generate_workspace_toml(config)?;

        // Generate each crate
        for crate_info in &config.crates {
            self.generate_crate(crate_info)?;
        }

        // Generate README
        self.generate_readme(config)?;

        Ok(())
    }

    /// Generate workspace root Cargo.toml
    fn generate_workspace_toml(&self, config: &WorkspaceConfig) -> Result<()> {
        let mut content = String::new();

        content.push_str("[workspace]\n");
        content.push_str("resolver = \"2\"\n");
        content.push_str("members = [\n");

        for crate_info in &config.crates {
            content.push_str(&format!("    \"{}\",\n", crate_info.path));
        }

        content.push_str("]\n\n");

        // Workspace-level dependencies (shared)
        content.push_str("[workspace.dependencies]\n");

        // Collect all unique external dependencies
        let mut all_deps: HashMap<String, &ExternalDependency> = HashMap::new();
        for crate_info in &config.crates {
            for dep in &crate_info.external_deps {
                all_deps.entry(dep.name.clone()).or_insert(dep);
            }
        }

        for (name, dep) in &all_deps {
            if dep.features.is_empty() {
                content.push_str(&format!("{} = \"{}\"\n", name, dep.version));
            } else {
                content.push_str(&format!(
                    "{} = {{ version = \"{}\", features = {:?} }}\n",
                    name, dep.version, dep.features
                ));
            }
        }

        let toml_path = self.root_path.join("Cargo.toml");
        fs::write(&toml_path, content)
            .map_err(|e| Error::Pipeline(format!("Failed to write workspace Cargo.toml: {}", e)))?;

        Ok(())
    }

    /// Generate a single crate
    fn generate_crate(&self, crate_info: &CrateInfo) -> Result<()> {
        let crate_path = self.root_path.join(&crate_info.path);
        let src_path = crate_path.join("src");

        // Create directories
        fs::create_dir_all(&src_path)
            .map_err(|e| Error::Pipeline(format!("Failed to create crate directory: {}", e)))?;

        // Generate crate Cargo.toml
        self.generate_crate_toml(crate_info)?;

        // Generate lib.rs
        let lib_path = src_path.join("lib.rs");
        fs::write(&lib_path, &crate_info.rust_code)
            .map_err(|e| Error::Pipeline(format!("Failed to write lib.rs: {}", e)))?;

        Ok(())
    }

    /// Generate crate-level Cargo.toml
    fn generate_crate_toml(&self, crate_info: &CrateInfo) -> Result<()> {
        let mut content = String::new();

        content.push_str("[package]\n");
        content.push_str(&format!("name = \"{}\"\n", crate_info.name));
        content.push_str("version = \"0.1.0\"\n");
        content.push_str("edition = \"2021\"\n\n");

        // Dependencies
        if !crate_info.dependencies.is_empty() || !crate_info.external_deps.is_empty() {
            content.push_str("[dependencies]\n");

            // Internal dependencies
            for dep in &crate_info.dependencies {
                content.push_str(&format!("{} = {{ path = \"../{}\" }}\n", dep, dep));
            }

            // External dependencies (from workspace)
            for dep in &crate_info.external_deps {
                content.push_str(&format!("{} = {{ workspace = true }}\n", dep.name));
            }
        }

        let toml_path = self.root_path.join(&crate_info.path).join("Cargo.toml");
        fs::write(&toml_path, content)
            .map_err(|e| Error::Pipeline(format!("Failed to write crate Cargo.toml: {}", e)))?;

        Ok(())
    }

    /// Generate workspace README
    fn generate_readme(&self, config: &WorkspaceConfig) -> Result<()> {
        let mut content = String::new();

        content.push_str(&format!("# {}\n\n", config.name));
        content.push_str("Generated by Portalis - Python to Rust transpiler\n\n");
        content.push_str("## Project Structure\n\n");

        content.push_str("```\n");
        content.push_str(&format!("{}/\n", config.name));
        content.push_str("├── Cargo.toml\n");
        for crate_info in &config.crates {
            content.push_str(&format!("├── {}/\n", crate_info.path));
            content.push_str(&format!("│   ├── Cargo.toml\n"));
            content.push_str(&format!("│   └── src/lib.rs\n"));
        }
        content.push_str("```\n\n");

        content.push_str("## Building\n\n");
        content.push_str("```bash\n");
        content.push_str("cargo build --workspace\n");
        content.push_str("```\n\n");

        content.push_str("## Testing\n\n");
        content.push_str("```bash\n");
        content.push_str("cargo test --workspace\n");
        content.push_str("```\n\n");

        content.push_str("## Crates\n\n");
        for crate_info in &config.crates {
            content.push_str(&format!("- **{}**: {}\n", crate_info.name, crate_info.path));
        }

        let readme_path = self.root_path.join("README.md");
        fs::write(&readme_path, content)
            .map_err(|e| Error::Pipeline(format!("Failed to write README: {}", e)))?;

        Ok(())
    }

    /// Clean/remove workspace directory
    pub fn clean(&self) -> Result<()> {
        if self.root_path.exists() {
            fs::remove_dir_all(&self.root_path)
                .map_err(|e| Error::Pipeline(format!("Failed to clean workspace: {}", e)))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_simple_workspace() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().join("test_workspace");

        let generator = WorkspaceGenerator::new(workspace_path.clone());

        let config = WorkspaceConfig {
            name: "my_project".to_string(),
            crates: vec![
                CrateInfo {
                    name: "core".to_string(),
                    path: "core".to_string(),
                    rust_code: "pub fn hello() -> &'static str { \"Hello\" }".to_string(),
                    dependencies: vec![],
                    external_deps: vec![],
                },
            ],
        };

        generator.generate(&config).unwrap();

        // Verify structure
        assert!(workspace_path.exists());
        assert!(workspace_path.join("Cargo.toml").exists());
        assert!(workspace_path.join("core").exists());
        assert!(workspace_path.join("core/Cargo.toml").exists());
        assert!(workspace_path.join("core/src/lib.rs").exists());
        assert!(workspace_path.join("README.md").exists());
    }

    #[test]
    fn test_generate_multi_crate_workspace() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().join("multi_crate");

        let generator = WorkspaceGenerator::new(workspace_path.clone());

        let config = WorkspaceConfig {
            name: "multi_crate_project".to_string(),
            crates: vec![
                CrateInfo {
                    name: "core".to_string(),
                    path: "core".to_string(),
                    rust_code: "pub struct Data {}".to_string(),
                    dependencies: vec![],
                    external_deps: vec![],
                },
                CrateInfo {
                    name: "utils".to_string(),
                    path: "utils".to_string(),
                    rust_code: "pub fn util() {}".to_string(),
                    dependencies: vec!["core".to_string()],
                    external_deps: vec![],
                },
            ],
        };

        generator.generate(&config).unwrap();

        // Verify both crates exist
        assert!(workspace_path.join("core").exists());
        assert!(workspace_path.join("utils").exists());

        // Verify utils depends on core
        let utils_toml = fs::read_to_string(workspace_path.join("utils/Cargo.toml")).unwrap();
        assert!(utils_toml.contains("core"));
    }

    #[test]
    fn test_workspace_with_external_deps() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().join("with_deps");

        let generator = WorkspaceGenerator::new(workspace_path.clone());

        let config = WorkspaceConfig {
            name: "project_with_deps".to_string(),
            crates: vec![
                CrateInfo {
                    name: "app".to_string(),
                    path: "app".to_string(),
                    rust_code: "use serde_json::Value;".to_string(),
                    dependencies: vec![],
                    external_deps: vec![
                        ExternalDependency {
                            name: "serde_json".to_string(),
                            version: "1.0".to_string(),
                            features: vec![],
                        },
                    ],
                },
            ],
        };

        generator.generate(&config).unwrap();

        // Verify workspace Cargo.toml has dependencies
        let workspace_toml = fs::read_to_string(workspace_path.join("Cargo.toml")).unwrap();
        assert!(workspace_toml.contains("serde_json"));
        assert!(workspace_toml.contains("1.0"));

        // Verify crate uses workspace dependency
        let app_toml = fs::read_to_string(workspace_path.join("app/Cargo.toml")).unwrap();
        assert!(app_toml.contains("workspace = true"));
    }

    #[test]
    fn test_clean_workspace() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().join("to_clean");

        let generator = WorkspaceGenerator::new(workspace_path.clone());

        let config = WorkspaceConfig {
            name: "temp_project".to_string(),
            crates: vec![
                CrateInfo {
                    name: "temp".to_string(),
                    path: "temp".to_string(),
                    rust_code: "".to_string(),
                    dependencies: vec![],
                    external_deps: vec![],
                },
            ],
        };

        generator.generate(&config).unwrap();
        assert!(workspace_path.exists());

        generator.clean().unwrap();
        assert!(!workspace_path.exists());
    }
}
