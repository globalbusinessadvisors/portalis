//! Python Standard Library to Rust Crate Mapping
//!
//! Maps Python imports to equivalent Rust crates and provides translation rules

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMapping {
    /// Python module name
    pub python_module: String,
    /// Rust crate name (for Cargo.toml)
    pub rust_crate: Option<String>,
    /// Rust use statement
    pub rust_use: String,
    /// Additional dependencies needed
    pub dependencies: Vec<String>,
    /// Version constraint
    pub version: String,
    /// WASM compatibility status
    pub wasm_compatible: WasmCompatibility,
    /// Notes about limitations or special handling
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WasmCompatibility {
    /// Fully compatible with WASM
    Full,
    /// Partially compatible (some features may not work)
    Partial,
    /// Requires WASI support
    RequiresWasi,
    /// Requires JS interop
    RequiresJsInterop,
    /// Not compatible with WASM
    Incompatible,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMapping {
    /// Python function name
    pub python_name: String,
    /// Rust equivalent
    pub rust_equiv: String,
    /// Requires module import
    pub requires_use: Option<String>,
    /// WASM compatibility
    pub wasm_compatible: WasmCompatibility,
    /// Transformation notes (e.g., parameter changes)
    pub transform_notes: Option<String>,
}

pub struct StdlibMapper {
    modules: HashMap<String, ModuleMapping>,
    functions: HashMap<String, HashMap<String, FunctionMapping>>,
}

impl StdlibMapper {
    pub fn new() -> Self {
        let mut mapper = Self {
            modules: HashMap::new(),
            functions: HashMap::new(),
        };
        mapper.init_mappings();
        mapper
    }

    fn init_mappings(&mut self) {
        // Use comprehensive mappings from the new module
        let comprehensive_mappings = crate::stdlib_mappings_comprehensive::init_critical_mappings();

        for (module_mapping, function_mappings) in comprehensive_mappings {
            self.add_module(module_mapping.clone());
            for func_mapping in function_mappings {
                self.add_function_mapping(&module_mapping.python_module, func_mapping);
            }
        }

        // Add legacy basic mappings that might not be in comprehensive yet
        self.add_legacy_mappings();
    }

    fn add_legacy_mappings(&mut self) {
        // Random module (if not already added)
        if !self.modules.contains_key("random") {
            self.add_module(ModuleMapping {
                python_module: "random".to_string(),
                rust_crate: Some("rand".to_string()),
                rust_use: "rand".to_string(),
                dependencies: vec![],
                version: "0.8".to_string(),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                notes: Some("Requires getrandom with js feature for WASM".to_string()),
            });

            self.add_function_mapping("random", FunctionMapping {
                python_name: "random".to_string(),
                rust_equiv: "rand::random::<f64>".to_string(),
                requires_use: Some("rand".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: None,
            });

            self.add_function_mapping("random", FunctionMapping {
                python_name: "randint".to_string(),
                rust_equiv: "rand::thread_rng().gen_range".to_string(),
                requires_use: Some("rand::Rng".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Takes range as (start..=end)".to_string()),
            });
        }

        // Datetime module (if not already added)
        if !self.modules.contains_key("datetime") {
            self.add_module(ModuleMapping {
                python_module: "datetime".to_string(),
                rust_crate: Some("chrono".to_string()),
                rust_use: "chrono".to_string(),
                dependencies: vec![],
                version: "0.4".to_string(),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                notes: Some("Uses JS Date in browser, native in Node.js".to_string()),
            });

            self.add_function_mapping("datetime", FunctionMapping {
                python_name: "datetime.now".to_string(),
                rust_equiv: "chrono::Utc::now".to_string(),
                requires_use: Some("chrono::Utc".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: None,
            });
        }

        // Re (regex) module (if not already added)
        if !self.modules.contains_key("re") {
            self.add_module(ModuleMapping {
                python_module: "re".to_string(),
                rust_crate: Some("regex".to_string()),
                rust_use: "regex::Regex".to_string(),
                dependencies: vec![],
                version: "1".to_string(),
                wasm_compatible: WasmCompatibility::Full,
                notes: None,
            });

            self.add_function_mapping("re", FunctionMapping {
                python_name: "compile".to_string(),
                rust_equiv: "Regex::new".to_string(),
                requires_use: Some("regex::Regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            });

            self.add_function_mapping("re", FunctionMapping {
                python_name: "match".to_string(),
                rust_equiv: "Regex::is_match".to_string(),
                requires_use: Some("regex::Regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            });
        }

        // Sys module
        if !self.modules.contains_key("sys") {
            self.add_module(ModuleMapping {
                python_module: "sys".to_string(),
                rust_crate: None,
                rust_use: "std::env".to_string(),
                dependencies: vec![],
                version: "*".to_string(),
                wasm_compatible: WasmCompatibility::Partial,
                notes: Some("Limited functionality in WASM".to_string()),
            });

            self.add_function_mapping("sys", FunctionMapping {
                python_name: "argv".to_string(),
                rust_equiv: "std::env::args".to_string(),
                requires_use: Some("std::env".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Not available in browser WASM".to_string()),
            });
        }

        // OS module (enhanced)
        if !self.modules.contains_key("os") {
            self.add_module(ModuleMapping {
                python_module: "os".to_string(),
                rust_crate: None,
                rust_use: "std::env".to_string(),
                dependencies: vec![],
                version: "*".to_string(),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                notes: Some("Most functions require WASI".to_string()),
            });

            self.add_function_mapping("os", FunctionMapping {
                python_name: "getcwd".to_string(),
                rust_equiv: "std::env::current_dir".to_string(),
                requires_use: Some("std::env".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            });

            self.add_function_mapping("os", FunctionMapping {
                python_name: "getenv".to_string(),
                rust_equiv: "std::env::var".to_string(),
                requires_use: Some("std::env".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            });
        }

        // JSON module
        if !self.modules.contains_key("json") {
            self.add_module(ModuleMapping {
                python_module: "json".to_string(),
                rust_crate: Some("serde_json".to_string()),
                rust_use: "serde_json".to_string(),
                dependencies: vec!["serde".to_string()],
                version: "1.0".to_string(),
                wasm_compatible: WasmCompatibility::Full,
                notes: None,
            });

            self.add_function_mapping("json", FunctionMapping {
                python_name: "loads".to_string(),
                rust_equiv: "serde_json::from_str".to_string(),
                requires_use: Some("serde_json".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            });

            self.add_function_mapping("json", FunctionMapping {
                python_name: "dumps".to_string(),
                rust_equiv: "serde_json::to_string".to_string(),
                requires_use: Some("serde_json".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            });
        }
    }

    fn add_module(&mut self, mapping: ModuleMapping) {
        self.modules.insert(mapping.python_module.clone(), mapping);
    }

    fn add_function_mapping(&mut self, module: &str, mapping: FunctionMapping) {
        self.functions
            .entry(module.to_string())
            .or_insert_with(HashMap::new)
            .insert(mapping.python_name.clone(), mapping);
    }

    pub fn get_module_mapping(&self, module: &str) -> Option<&ModuleMapping> {
        self.modules.get(module)
    }

    pub fn get_function_mapping(&self, module: &str, function: &str) -> Option<&FunctionMapping> {
        self.functions.get(module)?.get(function)
    }

    pub fn generate_use_statements(&self, modules: &[String]) -> Vec<String> {
        modules
            .iter()
            .filter_map(|module| {
                self.get_module_mapping(module)
                    .map(|mapping| format!("use {};", mapping.rust_use))
            })
            .collect()
    }

    pub fn generate_cargo_dependencies(&self, modules: &[String]) -> HashMap<String, String> {
        let mut deps = HashMap::new();

        for module in modules {
            if let Some(mapping) = self.get_module_mapping(module) {
                if let Some(crate_name) = &mapping.rust_crate {
                    deps.insert(crate_name.clone(), mapping.version.clone());
                }

                for dep in &mapping.dependencies {
                    deps.insert(dep.clone(), "*".to_string());
                }
            }
        }

        deps
    }

    /// Get WASM compatibility info for a module
    pub fn get_wasm_compatibility(&self, module: &str) -> Option<WasmCompatibility> {
        self.get_module_mapping(module)
            .map(|m| m.wasm_compatible.clone())
    }

    /// Get all mapped modules
    pub fn get_all_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    // Compatibility aliases for existing code
    pub fn get_module(&self, module: &str) -> Option<&ModuleMapping> {
        self.get_module_mapping(module)
    }

    pub fn get_function(&self, module: &str, function: &str) -> Option<String> {
        self.get_function_mapping(module, function)
            .map(|f| f.rust_equiv.clone())
    }

    pub fn collect_use_statements(&self, modules: &[String]) -> Vec<String> {
        self.generate_use_statements(modules)
    }

    /// Get statistics
    pub fn get_stats(&self) -> StdlibStats {
        let total = self.modules.len();
        let full_compat = self.modules.values()
            .filter(|m| m.wasm_compatible == WasmCompatibility::Full)
            .count();
        let partial_compat = self.modules.values()
            .filter(|m| m.wasm_compatible == WasmCompatibility::Partial)
            .count();
        let requires_wasi = self.modules.values()
            .filter(|m| m.wasm_compatible == WasmCompatibility::RequiresWasi)
            .count();
        let requires_js = self.modules.values()
            .filter(|m| m.wasm_compatible == WasmCompatibility::RequiresJsInterop)
            .count();
        let incompatible = self.modules.values()
            .filter(|m| m.wasm_compatible == WasmCompatibility::Incompatible)
            .count();

        StdlibStats {
            total_mapped: total,
            full_wasm_compat: full_compat,
            partial_wasm_compat: partial_compat,
            requires_wasi,
            requires_js_interop: requires_js,
            incompatible,
        }
    }
}

#[derive(Debug)]
pub struct StdlibStats {
    pub total_mapped: usize,
    pub full_wasm_compat: usize,
    pub partial_wasm_compat: usize,
    pub requires_wasi: usize,
    pub requires_js_interop: usize,
    pub incompatible: usize,
}

impl Default for StdlibMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_module_mapping() {
        let mapper = StdlibMapper::new();
        let math_mod = mapper.get_module_mapping("math");
        assert!(math_mod.is_some());
        assert_eq!(math_mod.unwrap().wasm_compatible, WasmCompatibility::Full);
    }

    #[test]
    fn test_json_module_mapping() {
        let mapper = StdlibMapper::new();
        let json_mod = mapper.get_module_mapping("json");
        assert!(json_mod.is_some());
        assert_eq!(json_mod.unwrap().rust_crate, Some("serde_json".to_string()));
    }

    #[test]
    fn test_function_mapping() {
        let mapper = StdlibMapper::new();
        let sqrt_func = mapper.get_function_mapping("math", "sqrt");
        assert!(sqrt_func.is_some());
        assert_eq!(sqrt_func.unwrap().rust_equiv, "f64::sqrt");
    }

    #[test]
    fn test_cargo_dependencies() {
        let mapper = StdlibMapper::new();
        let deps = mapper.generate_cargo_dependencies(&vec!["json".to_string()]);
        assert!(deps.contains_key("serde_json"));
    }

    #[test]
    fn test_stats() {
        let mapper = StdlibMapper::new();
        let stats = mapper.get_stats();
        assert!(stats.total_mapped > 0);
        println!("Stdlib mapping stats: {:?}", stats);
    }
}
