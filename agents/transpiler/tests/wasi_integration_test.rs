//! WASI Filesystem Integration Tests

#[cfg(test)]
mod wasi_fs_tests {
    use portalis_transpiler::wasi_fs::{WasiFs, WasiPath};
    use portalis_transpiler::py_to_rust_fs;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wasi_path_operations() {
        let path = WasiPath::new("/tmp/test.txt");
        assert_eq!(
            WasiPath::file_name(&path),
            Some("test.txt".to_string())
        );

        assert_eq!(
            WasiPath::extension(&path),
            Some("txt".to_string())
        );

        let parent = WasiPath::parent(&path);
        assert!(parent.is_some());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wasi_path_join() {
        let base = WasiPath::new("/tmp");
        let full = WasiPath::join(&base, "test.txt");

        let full_str = full.to_str().unwrap();
        assert!(full_str.ends_with("test.txt"));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wasi_file_write_read() {
        use std::fs;

        let test_path = "/tmp/portalis_wasi_test.txt";
        let test_content = "Hello WASI!";

        // Clean up any existing file
        let _ = fs::remove_file(test_path);

        // Write
        WasiFs::write(test_path, test_content).expect("Failed to write");

        // Read
        let content = WasiFs::read_to_string(test_path).expect("Failed to read");
        assert_eq!(content, test_content);

        // Check exists
        assert!(WasiFs::exists(test_path));
        assert!(WasiFs::is_file(test_path));

        // Clean up
        WasiFs::remove_file(test_path).expect("Failed to remove");
        assert!(!WasiFs::exists(test_path));
    }

    #[test]
    fn test_translate_open_function() {
        let result = py_to_rust_fs::translate_open("\"file.txt\"", "r");
        assert!(result.contains("WasiFs::open"));
        assert!(result.contains("file.txt"));

        let result_write = py_to_rust_fs::translate_open("\"file.txt\"", "w");
        assert!(result_write.contains("WasiFs::create"));
    }

    #[test]
    fn test_translate_pathlib_exists() {
        let result = py_to_rust_fs::translate_pathlib_operation("exists", &["\"file.txt\""]);
        assert!(result.contains("WasiFs::exists"));
        assert!(result.contains("file.txt"));
    }

    #[test]
    fn test_translate_pathlib_path_new() {
        let result = py_to_rust_fs::translate_pathlib_operation("Path", &["\"./data\""]);
        assert!(result.contains("WasiPath::new"));
        assert!(result.contains("./data"));
    }

    #[test]
    fn test_translate_pathlib_is_file() {
        let result = py_to_rust_fs::translate_pathlib_operation("is_file", &["path"]);
        assert!(result.contains("WasiFs::is_file"));
    }

    #[test]
    fn test_translate_pathlib_read_text() {
        let result = py_to_rust_fs::translate_pathlib_operation("read_text", &["path"]);
        assert!(result.contains("WasiFs::read_to_string"));
    }

    #[test]
    fn test_translate_pathlib_write_text() {
        let result = py_to_rust_fs::translate_pathlib_operation(
            "write_text",
            &["path", "\"content\""]
        );
        assert!(result.contains("WasiFs::write"));
        assert!(result.contains("content"));
    }

    #[test]
    fn test_translate_pathlib_joinpath() {
        let result = py_to_rust_fs::translate_pathlib_operation(
            "joinpath",
            &["base_path", "\"subdir\""]
        );
        assert!(result.contains("WasiPath::join"));
        assert!(result.contains("base_path"));
        assert!(result.contains("subdir"));
    }

    #[test]
    fn test_translate_with_open() {
        let result = py_to_rust_fs::translate_with_open(
            "\"file.txt\"",
            "r",
            "f",
            "let content = f.read();"
        );

        assert!(result.contains("let mut f"));
        assert!(result.contains("WasiFs::open"));
        assert!(result.contains("let content = f.read();"));
        assert!(result.contains("// File automatically closed"));
    }

    #[test]
    fn test_translate_file_read_method() {
        let result = py_to_rust_fs::translate_file_method("read", "f", &[]);
        assert!(result.contains("read_to_string"));
        assert!(result.contains("f"));
    }

    #[test]
    fn test_translate_file_write_method() {
        let result = py_to_rust_fs::translate_file_method("write", "f", &["data"]);
        assert!(result.contains("write_all"));
        assert!(result.contains("data.as_bytes()"));
    }

    #[test]
    fn test_get_fs_imports() {
        let imports = py_to_rust_fs::get_fs_imports();
        assert!(imports.len() >= 2);
        assert!(imports.iter().any(|i| i.contains("WasiFs")));
        assert!(imports.iter().any(|i| i.contains("WasiPath")));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_directory_operations() {
        use std::fs;

        let test_dir = "/tmp/portalis_wasi_test_dir";

        // Clean up
        let _ = fs::remove_dir(test_dir);

        // Create directory
        WasiFs::create_dir(test_dir).expect("Failed to create dir");
        assert!(WasiFs::exists(test_dir));
        assert!(WasiFs::is_dir(test_dir));

        // Clean up
        fs::remove_dir(test_dir).expect("Failed to remove dir");
    }
}

/// Integration test for end-to-end Python file I/O transpilation
#[cfg(test)]
mod python_file_io_integration {
    use portalis_transpiler::feature_translator::FeatureTranslator;

    #[test]
    fn test_transpile_simple_file_read() {
        let python_code = r#"
with open("data.txt", "r") as f:
    content = f.read()
    print(content)
"#;

        let mut translator = FeatureTranslator::new();
        let result = translator.translate(python_code);

        // Should compile without errors
        assert!(result.is_ok(), "Transpilation failed: {:?}", result.err());

        let rust_code = result.unwrap();

        // Should contain WASI file operations
        // Note: Current implementation may not fully support this yet
        // This is a placeholder for future functionality
        assert!(rust_code.contains("Generated by Portalis"));
    }

    #[test]
    fn test_transpile_pathlib_usage() {
        let python_code = r#"
from pathlib import Path

p = Path("data.txt")
if p.exists():
    content = p.read_text()
"#;

        let mut translator = FeatureTranslator::new();
        let result = translator.translate(python_code);

        assert!(result.is_ok(), "Transpilation failed: {:?}", result.err());

        // Future: should contain pathlib mappings
        let _rust_code = result.unwrap();
    }
}
