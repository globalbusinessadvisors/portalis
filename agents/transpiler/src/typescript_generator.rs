//! TypeScript Definition Generator - Generates .d.ts files from Rust/WASM exports
//!
//! This module provides:
//! 1. TypeScript type mapping for Rust types
//! 2. Function signature extraction and conversion
//! 3. Struct/Class type definitions
//! 4. Enum type definitions
//! 5. JSDoc comment generation
//! 6. Generic type support
//! 7. Import/export statement generation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// TypeScript type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TsType {
    Number,
    String,
    Boolean,
    Void,
    Any,
    Unknown,
    Null,
    Undefined,
    BigInt,
    Array(Box<TsType>),
    Tuple(Vec<TsType>),
    Object(HashMap<String, TsType>),
    Union(Vec<TsType>),
    Literal(String),
    Function {
        params: Vec<(String, TsType)>,
        return_type: Box<TsType>,
    },
    Promise(Box<TsType>),
    Generic {
        name: String,
        params: Vec<TsType>,
    },
    Custom(String),
}

impl TsType {
    /// Convert to TypeScript string representation
    pub fn to_ts_string(&self) -> String {
        match self {
            TsType::Number => "number".to_string(),
            TsType::String => "string".to_string(),
            TsType::Boolean => "boolean".to_string(),
            TsType::Void => "void".to_string(),
            TsType::Any => "any".to_string(),
            TsType::Unknown => "unknown".to_string(),
            TsType::Null => "null".to_string(),
            TsType::Undefined => "undefined".to_string(),
            TsType::BigInt => "bigint".to_string(),
            TsType::Array(inner) => format!("{}[]", inner.to_ts_string()),
            TsType::Tuple(types) => {
                let types_str = types.iter()
                    .map(|t| t.to_ts_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{}]", types_str)
            }
            TsType::Object(fields) => {
                if fields.is_empty() {
                    "{}".to_string()
                } else {
                    let fields_str = fields.iter()
                        .map(|(k, v)| format!("{}: {}", k, v.to_ts_string()))
                        .collect::<Vec<_>>()
                        .join("; ");
                    format!("{{ {} }}", fields_str)
                }
            }
            TsType::Union(types) => {
                types.iter()
                    .map(|t| t.to_ts_string())
                    .collect::<Vec<_>>()
                    .join(" | ")
            }
            TsType::Literal(val) => format!("\"{}\"", val),
            TsType::Function { params, return_type } => {
                let params_str = params.iter()
                    .map(|(name, ty)| format!("{}: {}", name, ty.to_ts_string()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({}) => {}", params_str, return_type.to_ts_string())
            }
            TsType::Promise(inner) => format!("Promise<{}>", inner.to_ts_string()),
            TsType::Generic { name, params } => {
                if params.is_empty() {
                    name.clone()
                } else {
                    let params_str = params.iter()
                        .map(|p| p.to_ts_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{}<{}>", name, params_str)
                }
            }
            TsType::Custom(name) => name.clone(),
        }
    }

    /// Map Rust type to TypeScript type
    pub fn from_rust_type(rust_type: &str) -> Self {
        match rust_type {
            // Numeric types
            "i8" | "i16" | "i32" | "u8" | "u16" | "u32" | "f32" | "f64" | "isize" | "usize" => {
                TsType::Number
            }
            "i64" | "u64" | "i128" | "u128" => TsType::BigInt,

            // String types
            "String" | "&str" | "str" => TsType::String,

            // Boolean
            "bool" => TsType::Boolean,

            // Unit type
            "()" => TsType::Void,

            // Option<T>
            s if s.starts_with("Option<") => {
                let inner = s.strip_prefix("Option<").unwrap().strip_suffix(">").unwrap();
                TsType::Union(vec![
                    TsType::from_rust_type(inner),
                    TsType::Null,
                ])
            }

            // Result<T, E>
            s if s.starts_with("Result<") => {
                let inner = s.strip_prefix("Result<").unwrap().strip_suffix(">").unwrap();
                let parts: Vec<&str> = inner.splitn(2, ',').collect();
                if !parts.is_empty() {
                    TsType::from_rust_type(parts[0].trim())
                } else {
                    TsType::Any
                }
            }

            // Vec<T>
            s if s.starts_with("Vec<") => {
                let inner = s.strip_prefix("Vec<").unwrap().strip_suffix(">").unwrap();
                TsType::Array(Box::new(TsType::from_rust_type(inner)))
            }

            // Array types
            s if s.starts_with("[") && s.contains(";") => {
                // [T; N] -> T[]
                let inner = s.strip_prefix("[").unwrap().split(';').next().unwrap();
                TsType::Array(Box::new(TsType::from_rust_type(inner)))
            }

            // HashMap, BTreeMap -> Record
            s if s.starts_with("HashMap<") || s.starts_with("BTreeMap<") => {
                let inner = if s.starts_with("HashMap<") {
                    s.strip_prefix("HashMap<").unwrap()
                } else {
                    s.strip_prefix("BTreeMap<").unwrap()
                }.strip_suffix(">").unwrap();

                let parts: Vec<&str> = inner.splitn(2, ',').collect();
                if parts.len() == 2 {
                    let key_type = TsType::from_rust_type(parts[0].trim());
                    let val_type = TsType::from_rust_type(parts[1].trim());
                    TsType::Generic {
                        name: "Record".to_string(),
                        params: vec![key_type, val_type],
                    }
                } else {
                    TsType::Any
                }
            }

            // Generic types
            s if s.contains("<") => {
                let name = s.split('<').next().unwrap();
                TsType::Custom(name.to_string())
            }

            // Custom types
            _ => TsType::Custom(rust_type.to_string()),
        }
    }
}

/// TypeScript function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsFunction {
    pub name: String,
    pub params: Vec<TsParameter>,
    pub return_type: TsType,
    pub is_async: bool,
    pub doc_comment: Option<String>,
    pub is_export: bool,
}

/// TypeScript parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsParameter {
    pub name: String,
    pub ty: TsType,
    pub optional: bool,
    pub default_value: Option<String>,
}

/// TypeScript interface/type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsInterface {
    pub name: String,
    pub fields: Vec<TsField>,
    pub extends: Vec<String>,
    pub doc_comment: Option<String>,
    pub is_export: bool,
}

/// TypeScript field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsField {
    pub name: String,
    pub ty: TsType,
    pub optional: bool,
    pub readonly: bool,
    pub doc_comment: Option<String>,
}

/// TypeScript enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsEnum {
    pub name: String,
    pub variants: Vec<TsEnumVariant>,
    pub doc_comment: Option<String>,
    pub is_export: bool,
}

/// TypeScript enum variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsEnumVariant {
    pub name: String,
    pub value: Option<String>,
    pub doc_comment: Option<String>,
}

/// TypeScript module definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsModule {
    pub functions: Vec<TsFunction>,
    pub interfaces: Vec<TsInterface>,
    pub enums: Vec<TsEnum>,
    pub type_aliases: Vec<TsTypeAlias>,
    pub imports: Vec<TsImport>,
}

/// TypeScript type alias
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsTypeAlias {
    pub name: String,
    pub ty: TsType,
    pub doc_comment: Option<String>,
    pub is_export: bool,
}

/// TypeScript import statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsImport {
    pub items: Vec<String>,
    pub from: String,
}

/// TypeScript definition generator
pub struct TypeScriptGenerator {
    module: TsModule,
}

impl TypeScriptGenerator {
    /// Create new TypeScript generator
    pub fn new() -> Self {
        Self {
            module: TsModule {
                functions: Vec::new(),
                interfaces: Vec::new(),
                enums: Vec::new(),
                type_aliases: Vec::new(),
                imports: Vec::new(),
            },
        }
    }

    /// Add function definition
    pub fn add_function(&mut self, func: TsFunction) {
        self.module.functions.push(func);
    }

    /// Add interface definition
    pub fn add_interface(&mut self, interface: TsInterface) {
        self.module.interfaces.push(interface);
    }

    /// Add enum definition
    pub fn add_enum(&mut self, enum_def: TsEnum) {
        self.module.enums.push(enum_def);
    }

    /// Add type alias
    pub fn add_type_alias(&mut self, alias: TsTypeAlias) {
        self.module.type_aliases.push(alias);
    }

    /// Add import statement
    pub fn add_import(&mut self, import: TsImport) {
        self.module.imports.push(import);
    }

    /// Generate TypeScript definitions from Rust function
    pub fn from_rust_function(
        &mut self,
        name: &str,
        params: &[(String, String)],
        return_type: &str,
        doc: Option<&str>,
    ) {
        let ts_params = params.iter().map(|(name, ty)| {
            TsParameter {
                name: name.clone(),
                ty: TsType::from_rust_type(ty),
                optional: ty.starts_with("Option<"),
                default_value: None,
            }
        }).collect();

        let ts_return = TsType::from_rust_type(return_type);

        self.add_function(TsFunction {
            name: name.to_string(),
            params: ts_params,
            return_type: ts_return,
            is_async: return_type.contains("Future") || return_type.contains("Promise"),
            doc_comment: doc.map(|s| s.to_string()),
            is_export: true,
        });
    }

    /// Generate TypeScript definitions from Rust struct
    pub fn from_rust_struct(
        &mut self,
        name: &str,
        fields: &[(String, String, bool)], // (name, type, optional)
        doc: Option<&str>,
    ) {
        let ts_fields = fields.iter().map(|(name, ty, optional)| {
            TsField {
                name: name.clone(),
                ty: TsType::from_rust_type(ty),
                optional: *optional,
                readonly: false,
                doc_comment: None,
            }
        }).collect();

        self.add_interface(TsInterface {
            name: name.to_string(),
            fields: ts_fields,
            extends: Vec::new(),
            doc_comment: doc.map(|s| s.to_string()),
            is_export: true,
        });
    }

    /// Generate TypeScript definitions from Rust enum
    pub fn from_rust_enum(
        &mut self,
        name: &str,
        variants: &[String],
        doc: Option<&str>,
    ) {
        let ts_variants = variants.iter().map(|v| {
            TsEnumVariant {
                name: v.clone(),
                value: None,
                doc_comment: None,
            }
        }).collect();

        self.add_enum(TsEnum {
            name: name.to_string(),
            variants: ts_variants,
            doc_comment: doc.map(|s| s.to_string()),
            is_export: true,
        });
    }

    /// Generate complete .d.ts file
    pub fn generate(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str("// TypeScript definitions for WASM module\n");
        output.push_str("// Generated by Portalis Transpiler\n\n");

        // Imports
        for import in &self.module.imports {
            output.push_str(&format!(
                "import {{ {} }} from '{}';\n",
                import.items.join(", "),
                import.from
            ));
        }
        if !self.module.imports.is_empty() {
            output.push('\n');
        }

        // Type aliases
        for alias in &self.module.type_aliases {
            if let Some(ref doc) = alias.doc_comment {
                output.push_str(&format!("/**\n * {}\n */\n", doc));
            }
            output.push_str(&format!(
                "{}type {} = {};\n\n",
                if alias.is_export { "export " } else { "" },
                alias.name,
                alias.ty.to_ts_string()
            ));
        }

        // Enums
        for enum_def in &self.module.enums {
            if let Some(ref doc) = enum_def.doc_comment {
                output.push_str(&format!("/**\n * {}\n */\n", doc));
            }
            output.push_str(&format!(
                "{}enum {} {{\n",
                if enum_def.is_export { "export " } else { "" },
                enum_def.name
            ));
            for variant in &enum_def.variants {
                if let Some(ref doc) = variant.doc_comment {
                    output.push_str(&format!("  /** {} */\n", doc));
                }
                if let Some(ref value) = variant.value {
                    output.push_str(&format!("  {} = {},\n", variant.name, value));
                } else {
                    output.push_str(&format!("  {},\n", variant.name));
                }
            }
            output.push_str("}\n\n");
        }

        // Interfaces
        for interface in &self.module.interfaces {
            if let Some(ref doc) = interface.doc_comment {
                output.push_str(&format!("/**\n * {}\n */\n", doc));
            }

            let extends = if interface.extends.is_empty() {
                String::new()
            } else {
                format!(" extends {}", interface.extends.join(", "))
            };

            output.push_str(&format!(
                "{}interface {}{} {{\n",
                if interface.is_export { "export " } else { "" },
                interface.name,
                extends
            ));

            for field in &interface.fields {
                if let Some(ref doc) = field.doc_comment {
                    output.push_str(&format!("  /** {} */\n", doc));
                }
                output.push_str(&format!(
                    "  {}{}{}: {};\n",
                    if field.readonly { "readonly " } else { "" },
                    field.name,
                    if field.optional { "?" } else { "" },
                    field.ty.to_ts_string()
                ));
            }
            output.push_str("}\n\n");
        }

        // Functions
        for func in &self.module.functions {
            if let Some(ref doc) = func.doc_comment {
                output.push_str(&format!("/**\n * {}\n", doc));

                // Add param docs
                for param in &func.params {
                    output.push_str(&format!(
                        " * @param {} - {}\n",
                        param.name,
                        param.ty.to_ts_string()
                    ));
                }

                // Add return doc
                output.push_str(&format!(
                    " * @returns {}\n",
                    func.return_type.to_ts_string()
                ));
                output.push_str(" */\n");
            }

            let params = func.params.iter().map(|p| {
                format!(
                    "{}{}: {}{}",
                    p.name,
                    if p.optional { "?" } else { "" },
                    p.ty.to_ts_string(),
                    if let Some(ref default) = p.default_value {
                        format!(" = {}", default)
                    } else {
                        String::new()
                    }
                )
            }).collect::<Vec<_>>().join(", ");

            let return_type = if func.is_async {
                TsType::Promise(Box::new(func.return_type.clone())).to_ts_string()
            } else {
                func.return_type.to_ts_string()
            };

            output.push_str(&format!(
                "{}{}function {}({}): {};\n\n",
                if func.is_export { "export " } else { "" },
                if func.is_async { "async " } else { "" },
                func.name,
                params,
                return_type
            ));
        }

        output
    }

    /// Generate WASM-specific definitions
    pub fn generate_wasm_definitions(module_name: &str) -> String {
        format!(r#"// TypeScript definitions for WASM module: {}
// Generated by Portalis Transpiler

/**
 * Initialize the WASM module
 * @param module_or_path - Optional WebAssembly.Module, path to .wasm file, or URL
 * @returns Promise that resolves to the WebAssembly.Instance
 */
export default function init(
  module_or_path?: WebAssembly.Module | string | URL | Request | Response | BufferSource | Promise<any>
): Promise<WebAssembly.Instance>;

/**
 * Synchronously initialize the WASM module (Node.js only)
 * @param module - WebAssembly.Module or path to .wasm file
 * @returns WebAssembly.Instance
 */
export function initSync(module: WebAssembly.Module | BufferSource): WebAssembly.Instance;

/**
 * Check if the WASM module is initialized
 * @returns true if initialized, false otherwise
 */
export function isInitialized(): boolean;

/**
 * Get the WebAssembly.Instance
 * @returns The WASM instance or null if not initialized
 */
export function getInstance(): WebAssembly.Instance | null;

/**
 * Get the WebAssembly.Memory
 * @returns The WASM memory object
 */
export function getMemory(): WebAssembly.Memory;

/**
 * WASM module exports
 */
export interface WasmExports {{
  memory: WebAssembly.Memory;
  [key: string]: any;
}}

// Your custom WASM exports will be added below
// Example:
// export function add(a: number, b: number): number;
// export function process_data(data: string): string;
"#, module_name)
    }
}

impl Default for TypeScriptGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_to_ts_type_conversion() {
        assert_eq!(TsType::from_rust_type("i32").to_ts_string(), "number");
        assert_eq!(TsType::from_rust_type("String").to_ts_string(), "string");
        assert_eq!(TsType::from_rust_type("bool").to_ts_string(), "boolean");
        assert_eq!(TsType::from_rust_type("()").to_ts_string(), "void");
        assert_eq!(TsType::from_rust_type("i64").to_ts_string(), "bigint");
    }

    #[test]
    fn test_option_type() {
        let ts_type = TsType::from_rust_type("Option<String>");
        assert_eq!(ts_type.to_ts_string(), "string | null");
    }

    #[test]
    fn test_vec_type() {
        let ts_type = TsType::from_rust_type("Vec<i32>");
        assert_eq!(ts_type.to_ts_string(), "number[]");
    }

    #[test]
    fn test_hashmap_type() {
        let ts_type = TsType::from_rust_type("HashMap<String, i32>");
        assert_eq!(ts_type.to_ts_string(), "Record<string, number>");
    }

    #[test]
    fn test_function_generation() {
        let mut gen = TypeScriptGenerator::new();
        gen.from_rust_function(
            "add",
            &[("a".to_string(), "i32".to_string()), ("b".to_string(), "i32".to_string())],
            "i32",
            Some("Adds two numbers"),
        );

        let output = gen.generate();
        assert!(output.contains("export async function add(a: number, b: number): Promise<number>;") ||
                output.contains("export function add(a: number, b: number): number;"));
    }

    #[test]
    fn test_interface_generation() {
        let mut gen = TypeScriptGenerator::new();
        gen.from_rust_struct(
            "User",
            &[
                ("name".to_string(), "String".to_string(), false),
                ("age".to_string(), "i32".to_string(), false),
                ("email".to_string(), "Option<String>".to_string(), true),
            ],
            Some("User data structure"),
        );

        let output = gen.generate();
        assert!(output.contains("export interface User"));
        assert!(output.contains("name: string"));
        assert!(output.contains("age: number"));
        assert!(output.contains("email?: string | null"));
    }

    #[test]
    fn test_enum_generation() {
        let mut gen = TypeScriptGenerator::new();
        gen.from_rust_enum(
            "Status",
            &["Pending".to_string(), "Active".to_string(), "Completed".to_string()],
            Some("Status enum"),
        );

        let output = gen.generate();
        assert!(output.contains("export enum Status"));
        assert!(output.contains("Pending"));
        assert!(output.contains("Active"));
        assert!(output.contains("Completed"));
    }

    #[test]
    fn test_wasm_definitions() {
        let defs = TypeScriptGenerator::generate_wasm_definitions("my_module");
        assert!(defs.contains("export default function init"));
        assert!(defs.contains("initSync"));
        assert!(defs.contains("isInitialized"));
        assert!(defs.contains("getInstance"));
        assert!(defs.contains("getMemory"));
    }

    #[test]
    fn test_complex_types() {
        let ts_type = TsType::from_rust_type("Result<Vec<String>, String>");
        assert_eq!(ts_type.to_ts_string(), "string[]");

        let ts_type = TsType::from_rust_type("Option<Vec<i32>>");
        assert_eq!(ts_type.to_ts_string(), "number[] | null");
    }
}
