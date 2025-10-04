//! Python Filesystem Operations → Rust/WASI Translation
//!
//! Translates Python file I/O to WASM-compatible Rust code using WASI

use crate::wasi_fs::{WasiFs, WasiPath, WasiDirectory};

/// Translate Python's open() to Rust
pub fn translate_open(filename: &str, mode: &str) -> String {
    match mode {
        "r" | "" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::open({})", filename)
        }
        "w" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::create({})", filename)
        }
        "a" => {
            // Append mode
            format!("portalis_transpiler::wasi_fs::WasiFs::open({}).and_then(|mut f| {{ /* seek to end */ Ok(f) }})", filename)
        }
        "rb" => {
            // Binary read
            format!("portalis_transpiler::wasi_fs::WasiFs::open({})", filename)
        }
        "wb" => {
            // Binary write
            format!("portalis_transpiler::wasi_fs::WasiFs::create({})", filename)
        }
        _ => {
            format!("portalis_transpiler::wasi_fs::WasiFs::open({}) /* mode: {} */", filename, mode)
        }
    }
}

/// Translate Python's pathlib.Path operations
pub fn translate_pathlib_operation(operation: &str, args: &[&str]) -> String {
    match operation {
        "Path" => {
            if args.is_empty() {
                "portalis_transpiler::wasi_fs::WasiPath::new(\".\")".to_string()
            } else {
                format!("portalis_transpiler::wasi_fs::WasiPath::new({})", args[0])
            }
        }
        "exists" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::exists({})", args.get(0).unwrap_or(&"path"))
        }
        "is_file" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::is_file({})", args.get(0).unwrap_or(&"path"))
        }
        "is_dir" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::is_dir({})", args.get(0).unwrap_or(&"path"))
        }
        "read_text" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::read_to_string({})", args.get(0).unwrap_or(&"path"))
        }
        "write_text" => {
            let path = args.get(0).unwrap_or(&"path");
            let content = args.get(1).unwrap_or(&"content");
            format!("portalis_transpiler::wasi_fs::WasiFs::write({}, {})", path, content)
        }
        "mkdir" => {
            let parents = args.get(1).unwrap_or(&"false");
            if *parents == "true" || *parents == "parents=True" {
                format!("portalis_transpiler::wasi_fs::WasiFs::create_dir_all({})", args.get(0).unwrap_or(&"path"))
            } else {
                format!("portalis_transpiler::wasi_fs::WasiFs::create_dir({})", args.get(0).unwrap_or(&"path"))
            }
        }
        "rmdir" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::remove_dir({})", args.get(0).unwrap_or(&"path"))
        }
        "unlink" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::remove_file({})", args.get(0).unwrap_or(&"path"))
        }
        "iterdir" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::read_dir({}).unwrap_or_default()", args.get(0).unwrap_or(&"path"))
        }
        "glob" | "rglob" => {
            // Basic glob support - would need more sophisticated pattern matching
            format!("portalis_transpiler::wasi_fs::WasiFs::list_dir({}).unwrap_or_default()", args.get(0).unwrap_or(&"path"))
        }
        "stat" => {
            format!("portalis_transpiler::wasi_fs::WasiFs::metadata({})", args.get(0).unwrap_or(&"path"))
        }
        "joinpath" | "join" => {
            let base = args.get(0).unwrap_or(&"path");
            let other = args.get(1).unwrap_or(&"\"\"");
            format!("portalis_transpiler::wasi_fs::WasiPath::join({}, {})", base, other)
        }
        "name" => {
            format!("portalis_transpiler::wasi_fs::WasiPath::file_name({}).unwrap_or_default()", args.get(0).unwrap_or(&"path"))
        }
        "parent" => {
            format!("portalis_transpiler::wasi_fs::WasiPath::parent({}).unwrap_or_else(|| std::path::PathBuf::from(\".\"))", args.get(0).unwrap_or(&"path"))
        }
        "suffix" => {
            format!("portalis_transpiler::wasi_fs::WasiPath::extension({}).map(|s| format!(\".{{}}\", s)).unwrap_or_default()", args.get(0).unwrap_or(&"path"))
        }
        _ => {
            format!("/* Unsupported pathlib operation: {} */", operation)
        }
    }
}

/// Generate context manager code for file operations
pub fn translate_with_open(filename: &str, mode: &str, var_name: &str, body: &str) -> String {
    format!(
        r#"{{
    let mut {} = {};
    // File operations
    {}
    // File automatically closed when scope ends (RAII)
}}"#,
        var_name,
        translate_open(filename, mode),
        body
    )
}

/// Translate Python file methods to Rust
pub fn translate_file_method(method: &str, file_var: &str, args: &[&str]) -> String {
    match method {
        "read" => {
            if args.is_empty() {
                format!("{{ let mut content = String::new(); {}.read_to_string(&mut content)?; content }}", file_var)
            } else {
                format!("{}.read({})", file_var, args[0])
            }
        }
        "readline" => {
            format!("{}.read_line()", file_var)
        }
        "readlines" => {
            format!("{}.lines().collect::<Vec<_>>()", file_var)
        }
        "write" => {
            format!("{}.write_all({}.as_bytes())?", file_var, args.get(0).unwrap_or(&"data"))
        }
        "writelines" => {
            format!(
                "for line in {} {{ {}.write_all(line.as_bytes())?; }}",
                args.get(0).unwrap_or(&"lines"),
                file_var
            )
        }
        "close" => {
            format!("drop({})", file_var)
        }
        "flush" => {
            format!("{}.flush()", file_var)
        }
        _ => {
            format!("/* Unsupported file method: {} */", method)
        }
    }
}

/// Generate imports needed for filesystem operations
pub fn get_fs_imports() -> Vec<String> {
    vec![
        "use portalis_transpiler::wasi_fs::{WasiFs, WasiPath};".to_string(),
        "use std::path::PathBuf;".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_open() {
        let result = translate_open("\"file.txt\"", "r");
        assert!(result.contains("WasiFs::open"));
    }

    #[test]
    fn test_translate_pathlib() {
        let result = translate_pathlib_operation("exists", &["path"]);
        assert!(result.contains("WasiFs::exists"));
    }

    #[test]
    fn test_translate_with_open() {
        let result = translate_with_open("\"file.txt\"", "r", "f", "let content = f.read();");
        assert!(result.contains("let mut f"));
        assert!(result.contains("WasiFs::open"));
    }
}
