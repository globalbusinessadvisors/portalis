// TypeScript Definitions Demo
// Demonstrates comprehensive TypeScript .d.ts generation from Rust code

use portalis_transpiler::typescript_generator::{
    TypeScriptGenerator, TsFunction, TsParameter, TsInterface, TsField,
    TsEnum, TsEnumVariant, TsType, TsTypeAlias, TsImport,
};

fn main() {
    println!("=== TypeScript Definitions Generation Demo ===\n");
    println!("Demonstrates: Rust → TypeScript type conversion, .d.ts generation\n");
    println!("{}", "=".repeat(80));

    // Demo 1: Basic Type Conversions
    demo_type_conversions();
    println!("\n{}", "=".repeat(80));

    // Demo 2: Function Definitions
    demo_function_definitions();
    println!("\n{}", "=".repeat(80));

    // Demo 3: Interface/Struct Definitions
    demo_interface_definitions();
    println!("\n{}", "=".repeat(80));

    // Demo 4: Enum Definitions
    demo_enum_definitions();
    println!("\n{}", "=".repeat(80));

    // Demo 5: Complex Types
    demo_complex_types();
    println!("\n{}", "=".repeat(80));

    // Demo 6: Complete WASM Module
    demo_complete_wasm_module();
    println!("\n{}", "=".repeat(80));

    // Demo 7: Real-World Example
    demo_real_world_example();
    println!("\n{}", "=".repeat(80));

    println!("\n🎉 TypeScript definitions demonstration complete!");
}

fn demo_type_conversions() {
    println!("\n=== Demo 1: Basic Type Conversions ===\n");
    println!("Rust Type → TypeScript Type:\n");

    let conversions = vec![
        ("i32", "number"),
        ("i64", "bigint"),
        ("f64", "number"),
        ("String", "string"),
        ("&str", "string"),
        ("bool", "boolean"),
        ("()", "void"),
        ("Vec<i32>", "number[]"),
        ("Vec<String>", "string[]"),
        ("Option<String>", "string | null"),
        ("Option<i32>", "number | null"),
        ("HashMap<String, i32>", "Record<string, number>"),
        ("Result<String, String>", "string"),
        ("[u8; 32]", "number[]"),
    ];

    for (rust_type, expected_ts) in conversions {
        let ts_type = TsType::from_rust_type(rust_type);
        let ts_string = ts_type.to_ts_string();
        let check = if ts_string == expected_ts { "✓" } else { "✗" };
        println!("  {} {:30} → {}", check, rust_type, ts_string);
    }
}

fn demo_function_definitions() {
    println!("\n=== Demo 2: Function Definitions ===\n");

    let mut gen = TypeScriptGenerator::new();

    // Simple function
    gen.from_rust_function(
        "add",
        &[
            ("a".to_string(), "i32".to_string()),
            ("b".to_string(), "i32".to_string()),
        ],
        "i32",
        Some("Adds two numbers together"),
    );

    // Async function
    gen.add_function(TsFunction {
        name: "fetchData".to_string(),
        params: vec![
            TsParameter {
                name: "url".to_string(),
                ty: TsType::String,
                optional: false,
                default_value: None,
            },
        ],
        return_type: TsType::String,
        is_async: true,
        doc_comment: Some("Fetches data from a URL".to_string()),
        is_export: true,
    });

    // Function with optional parameters
    gen.add_function(TsFunction {
        name: "greet".to_string(),
        params: vec![
            TsParameter {
                name: "name".to_string(),
                ty: TsType::String,
                optional: false,
                default_value: None,
            },
            TsParameter {
                name: "title".to_string(),
                ty: TsType::String,
                optional: true,
                default_value: Some("\"Mr.\"".to_string()),
            },
        ],
        return_type: TsType::String,
        is_async: false,
        doc_comment: Some("Greets a person with optional title".to_string()),
        is_export: true,
    });

    println!("Generated Function Definitions:");
    println!("{}", "-".repeat(80));
    println!("{}", gen.generate());
    println!("{}", "-".repeat(80));
}

fn demo_interface_definitions() {
    println!("\n=== Demo 3: Interface/Struct Definitions ===\n");

    let mut gen = TypeScriptGenerator::new();

    // User struct
    gen.from_rust_struct(
        "User",
        &[
            ("id".to_string(), "u64".to_string(), false),
            ("name".to_string(), "String".to_string(), false),
            ("email".to_string(), "String".to_string(), false),
            ("age".to_string(), "Option<i32>".to_string(), true),
            ("tags".to_string(), "Vec<String>".to_string(), false),
        ],
        Some("Represents a user in the system"),
    );

    // Config struct with complex types
    gen.add_interface(TsInterface {
        name: "Config".to_string(),
        fields: vec![
            TsField {
                name: "api_key".to_string(),
                ty: TsType::String,
                optional: false,
                readonly: true,
                doc_comment: Some("API authentication key".to_string()),
            },
            TsField {
                name: "timeout".to_string(),
                ty: TsType::Number,
                optional: true,
                readonly: false,
                doc_comment: Some("Request timeout in milliseconds".to_string()),
            },
            TsField {
                name: "retry_count".to_string(),
                ty: TsType::Number,
                optional: true,
                readonly: false,
                doc_comment: None,
            },
        ],
        extends: Vec::new(),
        doc_comment: Some("Configuration options".to_string()),
        is_export: true,
    });

    println!("Generated Interface Definitions:");
    println!("{}", "-".repeat(80));
    println!("{}", gen.generate());
    println!("{}", "-".repeat(80));
}

fn demo_enum_definitions() {
    println!("\n=== Demo 4: Enum Definitions ===\n");

    let mut gen = TypeScriptGenerator::new();

    // Simple enum
    gen.from_rust_enum(
        "Status",
        &["Pending".to_string(), "Active".to_string(), "Completed".to_string(), "Failed".to_string()],
        Some("Represents the status of an operation"),
    );

    // Enum with explicit values
    gen.add_enum(TsEnum {
        name: "LogLevel".to_string(),
        variants: vec![
            TsEnumVariant {
                name: "Error".to_string(),
                value: Some("0".to_string()),
                doc_comment: Some("Error level logging".to_string()),
            },
            TsEnumVariant {
                name: "Warn".to_string(),
                value: Some("1".to_string()),
                doc_comment: Some("Warning level logging".to_string()),
            },
            TsEnumVariant {
                name: "Info".to_string(),
                value: Some("2".to_string()),
                doc_comment: Some("Info level logging".to_string()),
            },
            TsEnumVariant {
                name: "Debug".to_string(),
                value: Some("3".to_string()),
                doc_comment: Some("Debug level logging".to_string()),
            },
        ],
        doc_comment: Some("Logging levels".to_string()),
        is_export: true,
    });

    println!("Generated Enum Definitions:");
    println!("{}", "-".repeat(80));
    println!("{}", gen.generate());
    println!("{}", "-".repeat(80));
}

fn demo_complex_types() {
    println!("\n=== Demo 5: Complex Types ===\n");

    let mut gen = TypeScriptGenerator::new();

    // Type aliases
    gen.add_type_alias(TsTypeAlias {
        name: "UserId".to_string(),
        ty: TsType::Number,
        doc_comment: Some("Unique user identifier".to_string()),
        is_export: true,
    });

    gen.add_type_alias(TsTypeAlias {
        name: "Result".to_string(),
        ty: TsType::Union(vec![
            TsType::Custom("Success".to_string()),
            TsType::Custom("Error".to_string()),
        ]),
        doc_comment: Some("Operation result".to_string()),
        is_export: true,
    });

    gen.add_type_alias(TsTypeAlias {
        name: "Callback".to_string(),
        ty: TsType::Function {
            params: vec![("data".to_string(), TsType::String)],
            return_type: Box::new(TsType::Void),
        },
        doc_comment: Some("Callback function type".to_string()),
        is_export: true,
    });

    // Generic types
    gen.add_type_alias(TsTypeAlias {
        name: "ApiResponse".to_string(),
        ty: TsType::Generic {
            name: "Promise".to_string(),
            params: vec![
                TsType::Object({
                    let mut fields = std::collections::HashMap::new();
                    fields.insert("data".to_string(), TsType::Any);
                    fields.insert("status".to_string(), TsType::Number);
                    fields
                }),
            ],
        },
        doc_comment: Some("API response wrapper".to_string()),
        is_export: true,
    });

    println!("Generated Complex Type Definitions:");
    println!("{}", "-".repeat(80));
    println!("{}", gen.generate());
    println!("{}", "-".repeat(80));
}

fn demo_complete_wasm_module() {
    println!("\n=== Demo 6: Complete WASM Module ===\n");

    let wasm_defs = TypeScriptGenerator::generate_wasm_definitions("calculator");

    println!("WASM Module Definitions (calculator.d.ts):");
    println!("{}", "=".repeat(80));
    println!("{}", wasm_defs);
    println!("{}", "=".repeat(80));

    // Add custom exports
    let mut gen = TypeScriptGenerator::new();

    gen.from_rust_function(
        "add",
        &[
            ("a".to_string(), "i32".to_string()),
            ("b".to_string(), "i32".to_string()),
        ],
        "i32",
        Some("Adds two integers"),
    );

    gen.from_rust_function(
        "multiply",
        &[
            ("a".to_string(), "f64".to_string()),
            ("b".to_string(), "f64".to_string()),
        ],
        "f64",
        Some("Multiplies two floating point numbers"),
    );

    gen.from_rust_function(
        "factorial",
        &[("n".to_string(), "i32".to_string())],
        "i64",
        Some("Calculates factorial of a number"),
    );

    println!("\nCustom WASM Exports:");
    println!("{}", "-".repeat(80));
    println!("{}", gen.generate());
    println!("{}", "-".repeat(80));
}

fn demo_real_world_example() {
    println!("\n=== Demo 7: Real-World Example - Data Processing Library ===\n");

    let mut gen = TypeScriptGenerator::new();

    // Enums
    gen.from_rust_enum(
        "DataFormat",
        &["JSON".to_string(), "CSV".to_string(), "XML".to_string(), "Binary".to_string()],
        Some("Supported data formats"),
    );

    gen.from_rust_enum(
        "ProcessingStatus",
        &["Queued".to_string(), "Processing".to_string(), "Completed".to_string(), "Failed".to_string()],
        Some("Processing pipeline status"),
    );

    // Interfaces
    gen.from_rust_struct(
        "DataSource",
        &[
            ("id".to_string(), "String".to_string(), false),
            ("url".to_string(), "String".to_string(), false),
            ("format".to_string(), "DataFormat".to_string(), false),
            ("headers".to_string(), "HashMap<String, String>".to_string(), true),
        ],
        Some("Data source configuration"),
    );

    gen.from_rust_struct(
        "ProcessingResult",
        &[
            ("status".to_string(), "ProcessingStatus".to_string(), false),
            ("data".to_string(), "Vec<u8>".to_string(), true),
            ("error_message".to_string(), "Option<String>".to_string(), true),
            ("duration_ms".to_string(), "u64".to_string(), false),
        ],
        Some("Result of data processing"),
    );

    // Functions
    gen.add_function(TsFunction {
        name: "processData".to_string(),
        params: vec![
            TsParameter {
                name: "source".to_string(),
                ty: TsType::Custom("DataSource".to_string()),
                optional: false,
                default_value: None,
            },
        ],
        return_type: TsType::Custom("ProcessingResult".to_string()),
        is_async: true,
        doc_comment: Some("Process data from a source".to_string()),
        is_export: true,
    });

    gen.add_function(TsFunction {
        name: "transformData".to_string(),
        params: vec![
            TsParameter {
                name: "data".to_string(),
                ty: TsType::Array(Box::new(TsType::Number)),
                optional: false,
                default_value: None,
            },
            TsParameter {
                name: "format".to_string(),
                ty: TsType::Custom("DataFormat".to_string()),
                optional: false,
                default_value: None,
            },
        ],
        return_type: TsType::String,
        is_async: false,
        doc_comment: Some("Transform data to specified format".to_string()),
        is_export: true,
    });

    gen.add_function(TsFunction {
        name: "validateData".to_string(),
        params: vec![
            TsParameter {
                name: "data".to_string(),
                ty: TsType::String,
                optional: false,
                default_value: None,
            },
        ],
        return_type: TsType::Boolean,
        is_async: false,
        doc_comment: Some("Validate data format and structure".to_string()),
        is_export: true,
    });

    // Type aliases
    gen.add_type_alias(TsTypeAlias {
        name: "DataProcessor".to_string(),
        ty: TsType::Function {
            params: vec![
                ("input".to_string(), TsType::String),
            ],
            return_type: Box::new(TsType::Promise(Box::new(TsType::String))),
        },
        doc_comment: Some("Data processor function type".to_string()),
        is_export: true,
    });

    println!("Complete Data Processing Library (data_processor.d.ts):");
    println!("{}", "=".repeat(80));
    println!("{}", gen.generate());
    println!("{}", "=".repeat(80));

    println!("\nUsage Example:");
    println!("{}", "-".repeat(80));
    println!(r#"import {{ processData, DataSource, DataFormat, ProcessingStatus }} from './data_processor';

async function main() {{
  const source: DataSource = {{
    id: "source-1",
    url: "https://api.example.com/data",
    format: DataFormat.JSON,
    headers: {{ "Authorization": "Bearer token" }}
  }};

  const result = await processData(source);

  if (result.status === ProcessingStatus.Completed) {{
    console.log('Processing completed in', result.duration_ms, 'ms');
  }} else {{
    console.error('Processing failed:', result.error_message);
  }}
}}
"#);
    println!("{}", "-".repeat(80));
}
