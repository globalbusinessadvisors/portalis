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

// Helper macros for cleaner mapping definitions
macro_rules! module_mapping {
    ($module:expr, $crate:expr, $use:expr, $deps:expr, $version:expr, $wasm:expr) => {
        ModuleMapping {
            python_module: $module.to_string(),
            rust_crate: $crate.map(|s: &str| s.to_string()),
            rust_use: $use.to_string(),
            dependencies: $deps.iter().map(|s: &&str| s.to_string()).collect(),
            version: $version.to_string(),
            wasm_compatible: $wasm,
            notes: None,
        }
    };
    ($module:expr, $crate:expr, $use:expr, $deps:expr, $version:expr, $wasm:expr, $notes:expr) => {
        ModuleMapping {
            python_module: $module.to_string(),
            rust_crate: $crate.map(|s: &str| s.to_string()),
            rust_use: $use.to_string(),
            dependencies: $deps.iter().map(|s: &&str| s.to_string()).collect(),
            version: $version.to_string(),
            wasm_compatible: $wasm,
            notes: Some($notes.to_string()),
        }
    };
}

macro_rules! func_mapping {
    ($py:expr, $rust:expr) => {
        FunctionMapping {
            python_name: $py.to_string(),
            rust_equiv: $rust.to_string(),
            requires_use: None,
            wasm_compatible: WasmCompatibility::Full,
            transform_notes: None,
        }
    };
    ($py:expr, $rust:expr, $use:expr) => {
        FunctionMapping {
            python_name: $py.to_string(),
            rust_equiv: $rust.to_string(),
            requires_use: Some($use.to_string()),
            wasm_compatible: WasmCompatibility::Full,
            transform_notes: None,
        }
    };
    ($py:expr, $rust:expr, $use:expr, $wasm:expr) => {
        FunctionMapping {
            python_name: $py.to_string(),
            rust_equiv: $rust.to_string(),
            requires_use: Some($use.to_string()),
            wasm_compatible: $wasm,
            transform_notes: None,
        }
    };
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
        // Math module - Pure computation, WASM compatible
        self.add_module(module_mapping!(
            "math", None::<&str>, "std::f64::consts", &[], "*", WasmCompatibility::Full
        ));

        self.add_function_mapping("math", func_mapping!("sqrt", "f64::sqrt"));
        self.add_function_mapping("math", func_mapping!("pow", "f64::powf"));
        self.add_function_mapping("math", func_mapping!("floor", "f64::floor"));
        self.add_function_mapping("math", func_mapping!("ceil", "f64::ceil"));
        self.add_function_mapping("math", func_mapping!("pi", "std::f64::consts::PI", "std::f64::consts::PI"));
        self.add_function_mapping("math", func_mapping!("e", "std::f64::consts::E", "std::f64::consts::E"));

        // JSON module
        self.add_module(ModuleMapping {
            python_module: "json".to_string(),
            rust_crate: Some("serde_json".to_string()),
            rust_use: "serde_json".to_string(),
            dependencies: vec!["serde".to_string()],
            version: "1.0".to_string(),
        });

        self.add_function_mapping("json", FunctionMapping {
            python_name: "loads".to_string(),
            rust_equiv: "serde_json::from_str".to_string(),
            requires_use: Some("serde_json".to_string()),
        });

        self.add_function_mapping("json", FunctionMapping {
            python_name: "dumps".to_string(),
            rust_equiv: "serde_json::to_string".to_string(),
            requires_use: Some("serde_json".to_string()),
        });

        // OS module
        self.add_module(ModuleMapping {
            python_module: "os".to_string(),
            rust_crate: None, // Uses std::env, std::fs
            rust_use: "std::env".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("os", FunctionMapping {
            python_name: "getcwd".to_string(),
            rust_equiv: "std::env::current_dir".to_string(),
            requires_use: Some("std::env".to_string()),
        });

        self.add_function_mapping("os", FunctionMapping {
            python_name: "getenv".to_string(),
            rust_equiv: "std::env::var".to_string(),
            requires_use: Some("std::env".to_string()),
        });

        // OS.path module
        self.add_module(ModuleMapping {
            python_module: "os.path".to_string(),
            rust_crate: None, // Uses std::path
            rust_use: "std::path::Path".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("os.path", FunctionMapping {
            python_name: "exists".to_string(),
            rust_equiv: "Path::new(&path).exists".to_string(),
            requires_use: Some("std::path::Path".to_string()),
        });

        self.add_function_mapping("os.path", FunctionMapping {
            python_name: "join".to_string(),
            rust_equiv: "Path::new(&a).join(b)".to_string(),
            requires_use: Some("std::path::Path".to_string()),
        });

        // Sys module
        self.add_module(ModuleMapping {
            python_module: "sys".to_string(),
            rust_crate: None, // Uses std::env
            rust_use: "std::env".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("sys", FunctionMapping {
            python_name: "argv".to_string(),
            rust_equiv: "std::env::args".to_string(),
            requires_use: Some("std::env".to_string()),
        });

        self.add_function_mapping("sys", FunctionMapping {
            python_name: "exit".to_string(),
            rust_equiv: "std::process::exit".to_string(),
            requires_use: Some("std::process".to_string()),
        });

        // Time module
        self.add_module(ModuleMapping {
            python_module: "time".to_string(),
            rust_crate: Some("std::time".to_string()),
            rust_use: "std::time".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("time", FunctionMapping {
            python_name: "sleep".to_string(),
            rust_equiv: "std::thread::sleep".to_string(),
            requires_use: Some("std::thread".to_string()),
        });

        self.add_function_mapping("time", FunctionMapping {
            python_name: "time".to_string(),
            rust_equiv: "std::time::SystemTime::now".to_string(),
            requires_use: Some("std::time::SystemTime".to_string()),
        });

        // Collections module
        self.add_module(ModuleMapping {
            python_module: "collections".to_string(),
            rust_crate: None, // Uses std::collections
            rust_use: "std::collections".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("collections", FunctionMapping {
            python_name: "defaultdict".to_string(),
            rust_equiv: "HashMap::new".to_string(),
            requires_use: Some("std::collections::HashMap".to_string()),
        });

        self.add_function_mapping("collections", FunctionMapping {
            python_name: "Counter".to_string(),
            rust_equiv: "HashMap::new".to_string(),
            requires_use: Some("std::collections::HashMap".to_string()),
        });

        // Random module
        self.add_module(ModuleMapping {
            python_module: "random".to_string(),
            rust_crate: Some("rand".to_string()),
            rust_use: "rand".to_string(),
            dependencies: vec![],
            version: "0.8".to_string(),
        });

        self.add_function_mapping("random", FunctionMapping {
            python_name: "random".to_string(),
            rust_equiv: "rand::random::<f64>".to_string(),
            requires_use: Some("rand".to_string()),
        });

        self.add_function_mapping("random", FunctionMapping {
            python_name: "randint".to_string(),
            rust_equiv: "rand::thread_rng().gen_range".to_string(),
            requires_use: Some("rand::Rng".to_string()),
        });

        // Datetime module
        self.add_module(ModuleMapping {
            python_module: "datetime".to_string(),
            rust_crate: Some("chrono".to_string()),
            rust_use: "chrono".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
        });

        self.add_function_mapping("datetime", FunctionMapping {
            python_name: "datetime.now".to_string(),
            rust_equiv: "chrono::Utc::now".to_string(),
            requires_use: Some("chrono::Utc".to_string()),
        });

        // Re (regex) module
        self.add_module(ModuleMapping {
            python_module: "re".to_string(),
            rust_crate: Some("regex".to_string()),
            rust_use: "regex::Regex".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
        });

        self.add_function_mapping("re", FunctionMapping {
            python_name: "compile".to_string(),
            rust_equiv: "Regex::new".to_string(),
            requires_use: Some("regex::Regex".to_string()),
        });

        self.add_function_mapping("re", FunctionMapping {
            python_name: "match".to_string(),
            rust_equiv: "Regex::is_match".to_string(),
            requires_use: Some("regex::Regex".to_string()),
        });

        // Itertools module
        self.add_module(ModuleMapping {
            python_module: "itertools".to_string(),
            rust_crate: Some("itertools".to_string()),
            rust_use: "itertools".to_string(),
            dependencies: vec![],
            version: "0.12".to_string(),
        });

        self.add_function_mapping("itertools", FunctionMapping {
            python_name: "chain".to_string(),
            rust_equiv: "itertools::chain".to_string(),
            requires_use: Some("itertools".to_string()),
        });

        self.add_function_mapping("itertools", FunctionMapping {
            python_name: "zip_longest".to_string(),
            rust_equiv: "itertools::zip_longest".to_string(),
            requires_use: Some("itertools".to_string()),
        });

        // Pathlib module
        self.add_module(ModuleMapping {
            python_module: "pathlib".to_string(),
            rust_crate: None,
            rust_use: "std::path::Path".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("pathlib", FunctionMapping {
            python_name: "Path".to_string(),
            rust_equiv: "Path::new".to_string(),
            requires_use: Some("std::path::Path".to_string()),
        });

        // String module
        self.add_module(ModuleMapping {
            python_module: "string".to_string(),
            rust_crate: None,
            rust_use: "std::string::String".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        // Subprocess module
        self.add_module(ModuleMapping {
            python_module: "subprocess".to_string(),
            rust_crate: None,
            rust_use: "std::process::Command".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("subprocess", FunctionMapping {
            python_name: "run".to_string(),
            rust_equiv: "Command::new".to_string(),
            requires_use: Some("std::process::Command".to_string()),
        });

        // Shutil module
        self.add_module(ModuleMapping {
            python_module: "shutil".to_string(),
            rust_crate: Some("fs_extra".to_string()),
            rust_use: "fs_extra".to_string(),
            dependencies: vec![],
            version: "1.3".to_string(),
        });

        self.add_function_mapping("shutil", FunctionMapping {
            python_name: "copy".to_string(),
            rust_equiv: "std::fs::copy".to_string(),
            requires_use: Some("std::fs".to_string()),
        });

        self.add_function_mapping("shutil", FunctionMapping {
            python_name: "rmtree".to_string(),
            rust_equiv: "std::fs::remove_dir_all".to_string(),
            requires_use: Some("std::fs".to_string()),
        });

        // Glob module
        self.add_module(ModuleMapping {
            python_module: "glob".to_string(),
            rust_crate: Some("glob".to_string()),
            rust_use: "glob::glob".to_string(),
            dependencies: vec![],
            version: "0.3".to_string(),
        });

        self.add_function_mapping("glob", FunctionMapping {
            python_name: "glob".to_string(),
            rust_equiv: "glob::glob".to_string(),
            requires_use: Some("glob::glob".to_string()),
        });

        // CSV module
        self.add_module(ModuleMapping {
            python_module: "csv".to_string(),
            rust_crate: Some("csv".to_string()),
            rust_use: "csv".to_string(),
            dependencies: vec![],
            version: "1.3".to_string(),
        });

        self.add_function_mapping("csv", FunctionMapping {
            python_name: "reader".to_string(),
            rust_equiv: "csv::Reader::from_path".to_string(),
            requires_use: Some("csv".to_string()),
        });

        self.add_function_mapping("csv", FunctionMapping {
            python_name: "writer".to_string(),
            rust_equiv: "csv::Writer::from_path".to_string(),
            requires_use: Some("csv".to_string()),
        });

        // Hashlib module
        self.add_module(ModuleMapping {
            python_module: "hashlib".to_string(),
            rust_crate: Some("sha2".to_string()),
            rust_use: "sha2".to_string(),
            dependencies: vec![],
            version: "0.10".to_string(),
        });

        self.add_function_mapping("hashlib", FunctionMapping {
            python_name: "sha256".to_string(),
            rust_equiv: "sha2::Sha256::new".to_string(),
            requires_use: Some("sha2::Sha256".to_string()),
        });

        self.add_function_mapping("hashlib", FunctionMapping {
            python_name: "md5".to_string(),
            rust_equiv: "md5::Md5::new".to_string(),
            requires_use: Some("md5::Md5".to_string()),
        });

        // Base64 module
        self.add_module(ModuleMapping {
            python_module: "base64".to_string(),
            rust_crate: Some("base64".to_string()),
            rust_use: "base64".to_string(),
            dependencies: vec![],
            version: "0.21".to_string(),
        });

        self.add_function_mapping("base64", FunctionMapping {
            python_name: "b64encode".to_string(),
            rust_equiv: "base64::encode".to_string(),
            requires_use: Some("base64".to_string()),
        });

        self.add_function_mapping("base64", FunctionMapping {
            python_name: "b64decode".to_string(),
            rust_equiv: "base64::decode".to_string(),
            requires_use: Some("base64".to_string()),
        });

        // UUID module
        self.add_module(ModuleMapping {
            python_module: "uuid".to_string(),
            rust_crate: Some("uuid".to_string()),
            rust_use: "uuid::Uuid".to_string(),
            dependencies: vec![],
            version: "1.6".to_string(),
        });

        self.add_function_mapping("uuid", FunctionMapping {
            python_name: "uuid4".to_string(),
            rust_equiv: "Uuid::new_v4".to_string(),
            requires_use: Some("uuid::Uuid".to_string()),
        });

        // Functools module
        self.add_module(ModuleMapping {
            python_module: "functools".to_string(),
            rust_crate: None,
            rust_use: "std::iter".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        // Typing module (mostly for type hints)
        self.add_module(ModuleMapping {
            python_module: "typing".to_string(),
            rust_crate: None,
            rust_use: "std".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        // Copy module
        self.add_module(ModuleMapping {
            python_module: "copy".to_string(),
            rust_crate: None,
            rust_use: "std::clone::Clone".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("copy", FunctionMapping {
            python_name: "copy".to_string(),
            rust_equiv: "clone".to_string(),
            requires_use: None,
        });

        self.add_function_mapping("copy", FunctionMapping {
            python_name: "deepcopy".to_string(),
            rust_equiv: "clone".to_string(),
            requires_use: None,
        });

        // Pickle module
        self.add_module(ModuleMapping {
            python_module: "pickle".to_string(),
            rust_crate: Some("bincode".to_string()),
            rust_use: "bincode".to_string(),
            dependencies: vec!["serde".to_string()],
            version: "1.3".to_string(),
        });

        self.add_function_mapping("pickle", FunctionMapping {
            python_name: "dumps".to_string(),
            rust_equiv: "bincode::serialize".to_string(),
            requires_use: Some("bincode".to_string()),
        });

        self.add_function_mapping("pickle", FunctionMapping {
            python_name: "loads".to_string(),
            rust_equiv: "bincode::deserialize".to_string(),
            requires_use: Some("bincode".to_string()),
        });

        // Threading module
        self.add_module(ModuleMapping {
            python_module: "threading".to_string(),
            rust_crate: None,
            rust_use: "std::thread".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("threading", FunctionMapping {
            python_name: "Thread".to_string(),
            rust_equiv: "std::thread::spawn".to_string(),
            requires_use: Some("std::thread".to_string()),
        });

        // Queue module
        self.add_module(ModuleMapping {
            python_module: "queue".to_string(),
            rust_crate: Some("crossbeam".to_string()),
            rust_use: "crossbeam::channel".to_string(),
            dependencies: vec![],
            version: "0.5".to_string(),
        });

        self.add_function_mapping("queue", FunctionMapping {
            python_name: "Queue".to_string(),
            rust_equiv: "crossbeam::channel::unbounded".to_string(),
            requires_use: Some("crossbeam::channel".to_string()),
        });

        // Logging module
        self.add_module(ModuleMapping {
            python_module: "logging".to_string(),
            rust_crate: Some("log".to_string()),
            rust_use: "log".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
        });

        self.add_function_mapping("logging", FunctionMapping {
            python_name: "info".to_string(),
            rust_equiv: "log::info!".to_string(),
            requires_use: Some("log".to_string()),
        });

        self.add_function_mapping("logging", FunctionMapping {
            python_name: "error".to_string(),
            rust_equiv: "log::error!".to_string(),
            requires_use: Some("log".to_string()),
        });

        // Argparse module
        self.add_module(ModuleMapping {
            python_module: "argparse".to_string(),
            rust_crate: Some("clap".to_string()),
            rust_use: "clap::Parser".to_string(),
            dependencies: vec![],
            version: "4.4".to_string(),
        });

        // Requests equivalent
        self.add_module(ModuleMapping {
            python_module: "requests".to_string(),
            rust_crate: Some("reqwest".to_string()),
            rust_use: "reqwest".to_string(),
            dependencies: vec!["tokio".to_string()],
            version: "0.11".to_string(),
        });

        self.add_function_mapping("requests", FunctionMapping {
            python_name: "get".to_string(),
            rust_equiv: "reqwest::blocking::get".to_string(),
            requires_use: Some("reqwest".to_string()),
        });

        self.add_function_mapping("requests", FunctionMapping {
            python_name: "post".to_string(),
            rust_equiv: "reqwest::blocking::Client::new().post".to_string(),
            requires_use: Some("reqwest".to_string()),
        });

        // XML module
        self.add_module(ModuleMapping {
            python_module: "xml.etree".to_string(),
            rust_crate: Some("quick-xml".to_string()),
            rust_use: "quick_xml".to_string(),
            dependencies: vec![],
            version: "0.31".to_string(),
        });

        // Configparser module
        self.add_module(ModuleMapping {
            python_module: "configparser".to_string(),
            rust_crate: Some("ini".to_string()),
            rust_use: "ini::Ini".to_string(),
            dependencies: vec![],
            version: "1.3".to_string(),
        });

        // Unittest module
        self.add_module(ModuleMapping {
            python_module: "unittest".to_string(),
            rust_crate: None,
            rust_use: "std".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        // Enum module
        self.add_module(ModuleMapping {
            python_module: "enum".to_string(),
            rust_crate: None,
            rust_use: "std".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        // Array module
        self.add_module(ModuleMapping {
            python_module: "array".to_string(),
            rust_crate: None,
            rust_use: "std::vec::Vec".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        // Heapq module
        self.add_module(ModuleMapping {
            python_module: "heapq".to_string(),
            rust_crate: None,
            rust_use: "std::collections::BinaryHeap".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("heapq", FunctionMapping {
            python_name: "heappush".to_string(),
            rust_equiv: "BinaryHeap::push".to_string(),
            requires_use: Some("std::collections::BinaryHeap".to_string()),
        });

        self.add_function_mapping("heapq", FunctionMapping {
            python_name: "heappop".to_string(),
            rust_equiv: "BinaryHeap::pop".to_string(),
            requires_use: Some("std::collections::BinaryHeap".to_string()),
        });

        // Bisect module
        self.add_module(ModuleMapping {
            python_module: "bisect".to_string(),
            rust_crate: None,
            rust_use: "std::vec::Vec".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("bisect", FunctionMapping {
            python_name: "bisect_left".to_string(),
            rust_equiv: "binary_search".to_string(),
            requires_use: None,
        });

        // Contextlib module
        self.add_module(ModuleMapping {
            python_module: "contextlib".to_string(),
            rust_crate: None,
            rust_use: "std".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        // Dataclasses module
        self.add_module(ModuleMapping {
            python_module: "dataclasses".to_string(),
            rust_crate: None,
            rust_use: "std".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        // Asyncio module
        self.add_module(ModuleMapping {
            python_module: "asyncio".to_string(),
            rust_crate: Some("tokio".to_string()),
            rust_use: "tokio".to_string(),
            dependencies: vec![],
            version: "1.35".to_string(),
        });

        self.add_function_mapping("asyncio", FunctionMapping {
            python_name: "run".to_string(),
            rust_equiv: "tokio::runtime::Runtime::new().block_on".to_string(),
            requires_use: Some("tokio".to_string()),
        });

        self.add_function_mapping("asyncio", FunctionMapping {
            python_name: "sleep".to_string(),
            rust_equiv: "tokio::time::sleep".to_string(),
            requires_use: Some("tokio::time".to_string()),
        });

        // Decimal module
        self.add_module(ModuleMapping {
            python_module: "decimal".to_string(),
            rust_crate: Some("rust_decimal".to_string()),
            rust_use: "rust_decimal::Decimal".to_string(),
            dependencies: vec![],
            version: "1.33".to_string(),
        });

        // Fractions module
        self.add_module(ModuleMapping {
            python_module: "fractions".to_string(),
            rust_crate: Some("num-rational".to_string()),
            rust_use: "num_rational::Ratio".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
        });

        // Statistics module
        self.add_module(ModuleMapping {
            python_module: "statistics".to_string(),
            rust_crate: Some("statrs".to_string()),
            rust_use: "statrs".to_string(),
            dependencies: vec![],
            version: "0.16".to_string(),
        });

        self.add_function_mapping("statistics", FunctionMapping {
            python_name: "mean".to_string(),
            rust_equiv: "statrs::statistics::Statistics::mean".to_string(),
            requires_use: Some("statrs::statistics::Statistics".to_string()),
        });

        self.add_function_mapping("statistics", FunctionMapping {
            python_name: "median".to_string(),
            rust_equiv: "statrs::statistics::Statistics::median".to_string(),
            requires_use: Some("statrs::statistics::Statistics".to_string()),
        });

        // Secrets module
        self.add_module(ModuleMapping {
            python_module: "secrets".to_string(),
            rust_crate: Some("rand".to_string()),
            rust_use: "rand".to_string(),
            dependencies: vec![],
            version: "0.8".to_string(),
        });

        self.add_function_mapping("secrets", FunctionMapping {
            python_name: "token_bytes".to_string(),
            rust_equiv: "rand::random".to_string(),
            requires_use: Some("rand".to_string()),
        });

        // Tempfile module
        self.add_module(ModuleMapping {
            python_module: "tempfile".to_string(),
            rust_crate: Some("tempfile".to_string()),
            rust_use: "tempfile".to_string(),
            dependencies: vec![],
            version: "3.8".to_string(),
        });

        self.add_function_mapping("tempfile", FunctionMapping {
            python_name: "TemporaryFile".to_string(),
            rust_equiv: "tempfile::tempfile".to_string(),
            requires_use: Some("tempfile".to_string()),
        });

        self.add_function_mapping("tempfile", FunctionMapping {
            python_name: "NamedTemporaryFile".to_string(),
            rust_equiv: "tempfile::NamedTempFile::new".to_string(),
            requires_use: Some("tempfile".to_string()),
        });

        // getpass module
        self.add_module(ModuleMapping {
            python_module: "getpass".to_string(),
            rust_crate: Some("rpassword".to_string()),
            rust_use: "rpassword".to_string(),
            dependencies: vec![],
            version: "7.3".to_string(),
        });

        self.add_function_mapping("getpass", FunctionMapping {
            python_name: "getpass".to_string(),
            rust_equiv: "rpassword::read_password".to_string(),
            requires_use: Some("rpassword".to_string()),
        });

        // Platform module
        self.add_module(ModuleMapping {
            python_module: "platform".to_string(),
            rust_crate: None,
            rust_use: "std::env".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("platform", FunctionMapping {
            python_name: "system".to_string(),
            rust_equiv: "std::env::consts::OS".to_string(),
            requires_use: Some("std::env".to_string()),
        });

        // Warnings module
        self.add_module(ModuleMapping {
            python_module: "warnings".to_string(),
            rust_crate: None,
            rust_use: "std".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        // IO module
        self.add_module(ModuleMapping {
            python_module: "io".to_string(),
            rust_crate: None,
            rust_use: "std::io".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
        });

        self.add_function_mapping("io", FunctionMapping {
            python_name: "BytesIO".to_string(),
            rust_equiv: "std::io::Cursor::new".to_string(),
            requires_use: Some("std::io::Cursor".to_string()),
        });

        self.add_function_mapping("io", FunctionMapping {
            python_name: "StringIO".to_string(),
            rust_equiv: "String::new".to_string(),
            requires_use: None,
        });

        // Zlib module
        self.add_module(ModuleMapping {
            python_module: "zlib".to_string(),
            rust_crate: Some("flate2".to_string()),
            rust_use: "flate2".to_string(),
            dependencies: vec![],
            version: "1.0".to_string(),
        });

        self.add_function_mapping("zlib", FunctionMapping {
            python_name: "compress".to_string(),
            rust_equiv: "flate2::write::ZlibEncoder::new".to_string(),
            requires_use: Some("flate2::write::ZlibEncoder".to_string()),
        });

        self.add_function_mapping("zlib", FunctionMapping {
            python_name: "decompress".to_string(),
            rust_equiv: "flate2::read::ZlibDecoder::new".to_string(),
            requires_use: Some("flate2::read::ZlibDecoder".to_string()),
        });

        // Gzip module
        self.add_module(ModuleMapping {
            python_module: "gzip".to_string(),
            rust_crate: Some("flate2".to_string()),
            rust_use: "flate2".to_string(),
            dependencies: vec![],
            version: "1.0".to_string(),
        });

        self.add_function_mapping("gzip", FunctionMapping {
            python_name: "open".to_string(),
            rust_equiv: "flate2::read::GzDecoder::new".to_string(),
            requires_use: Some("flate2::read::GzDecoder".to_string()),
        });

        // Bz2 module
        self.add_module(ModuleMapping {
            python_module: "bz2".to_string(),
            rust_crate: Some("bzip2".to_string()),
            rust_use: "bzip2".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
        });

        // Zipfile module
        self.add_module(ModuleMapping {
            python_module: "zipfile".to_string(),
            rust_crate: Some("zip".to_string()),
            rust_use: "zip".to_string(),
            dependencies: vec![],
            version: "0.6".to_string(),
        });

        self.add_function_mapping("zipfile", FunctionMapping {
            python_name: "ZipFile".to_string(),
            rust_equiv: "zip::ZipArchive::new".to_string(),
            requires_use: Some("zip::ZipArchive".to_string()),
        });

        // Tarfile module
        self.add_module(ModuleMapping {
            python_module: "tarfile".to_string(),
            rust_crate: Some("tar".to_string()),
            rust_use: "tar".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
        });

        self.add_function_mapping("tarfile", FunctionMapping {
            python_name: "open".to_string(),
            rust_equiv: "tar::Archive::new".to_string(),
            requires_use: Some("tar::Archive".to_string()),
        });

        // SQLite module
        self.add_module(ModuleMapping {
            python_module: "sqlite3".to_string(),
            rust_crate: Some("rusqlite".to_string()),
            rust_use: "rusqlite".to_string(),
            dependencies: vec![],
            version: "0.30".to_string(),
        });

        self.add_function_mapping("sqlite3", FunctionMapping {
            python_name: "connect".to_string(),
            rust_equiv: "rusqlite::Connection::open".to_string(),
            requires_use: Some("rusqlite::Connection".to_string()),
        });
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

    /// Get module mapping for a Python import
    pub fn get_module(&self, python_module: &str) -> Option<&ModuleMapping> {
        self.modules.get(python_module)
    }

    /// Get function mapping
    pub fn get_function(&self, module: &str, function: &str) -> Option<&FunctionMapping> {
        self.functions
            .get(module)
            .and_then(|funcs| funcs.get(function))
    }

    /// Generate Rust use statements for a Python import
    pub fn generate_use_statements(&self, python_module: &str) -> Vec<String> {
        let mut uses = Vec::new();

        if let Some(mapping) = self.get_module(python_module) {
            uses.push(format!("use {};", mapping.rust_use));
        }

        uses
    }

    /// Generate Cargo.toml dependencies
    pub fn generate_cargo_deps(&self, python_module: &str) -> Vec<String> {
        let mut deps = Vec::new();

        if let Some(mapping) = self.get_module(python_module) {
            if let Some(crate_name) = &mapping.rust_crate {
                deps.push(format!("{} = \"{}\"", crate_name, mapping.version));
            }

            for dep in &mapping.dependencies {
                deps.push(format!("{} = \"1.0\"", dep));
            }
        }

        deps
    }

    /// Translate a Python function call to Rust
    pub fn translate_call(&self, module: &str, function: &str) -> Option<String> {
        self.get_function(module, function)
            .map(|m| m.rust_equiv.clone())
    }

    /// Get all required use statements for imported modules
    pub fn collect_use_statements(&self, imports: &[String]) -> Vec<String> {
        let mut all_uses = Vec::new();

        for import in imports {
            all_uses.extend(self.generate_use_statements(import));
        }

        // Deduplicate
        all_uses.sort();
        all_uses.dedup();
        all_uses
    }

    /// Get all Cargo dependencies for imported modules
    pub fn collect_cargo_deps(&self, imports: &[String]) -> Vec<String> {
        let mut all_deps = Vec::new();

        for import in imports {
            all_deps.extend(self.generate_cargo_deps(import));
        }

        // Deduplicate
        all_deps.sort();
        all_deps.dedup();
        all_deps
    }
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
        let mapping = mapper.get_module("math").unwrap();
        assert_eq!(mapping.rust_use, "std::f64::consts");
    }

    #[test]
    fn test_json_module_mapping() {
        let mapper = StdlibMapper::new();
        let mapping = mapper.get_module("json").unwrap();
        assert_eq!(mapping.rust_crate, Some("serde_json".to_string()));
    }

    #[test]
    fn test_function_translation() {
        let mapper = StdlibMapper::new();
        let rust_call = mapper.translate_call("math", "sqrt").unwrap();
        assert_eq!(rust_call, "f64::sqrt");
    }

    #[test]
    fn test_use_statements_generation() {
        let mapper = StdlibMapper::new();
        let uses = mapper.generate_use_statements("json");
        assert!(!uses.is_empty());
        assert!(uses[0].contains("serde_json"));
    }

    #[test]
    fn test_cargo_deps_generation() {
        let mapper = StdlibMapper::new();
        let deps = mapper.generate_cargo_deps("json");
        assert!(!deps.is_empty());
        assert!(deps.iter().any(|d| d.contains("serde_json")));
    }
}
