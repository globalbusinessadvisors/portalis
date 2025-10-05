//! Common Python Libraries Translation
//!
//! Translates popular Python libraries to their Rust equivalents:
//! - Requests → reqwest (HTTP client)
//! - pytest → Rust testing framework
//! - pydantic → serde with validator
//! - logging → tracing
//! - argparse → clap
//! - and more

use std::collections::HashMap;

/// Python library to translate
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PythonLibrary {
    Requests,
    Pytest,
    Pydantic,
    Logging,
    Argparse,
    Json,
    Datetime,
    Pathlib,
    Regex,
    Os,
    Sys,
    Collections,
}

/// Requests HTTP operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestsOp {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
    Session,
}

/// Pytest testing operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PytestOp {
    Test,
    Fixture,
    Parametrize,
    Assert,
    Raises,
    Skip,
    Mark,
}

/// Pydantic validation operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PydanticOp {
    BaseModel,
    Field,
    Validator,
    RootValidator,
    Config,
}

/// Logging operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoggingOp {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
    Logger,
}

/// Argparse operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgparseOp {
    ArgumentParser,
    AddArgument,
    ParseArgs,
}

/// Common libraries translator
pub struct CommonLibrariesTranslator {
    /// Required imports
    imports: HashMap<PythonLibrary, Vec<String>>,
    /// Cargo dependencies
    dependencies: HashMap<PythonLibrary, Vec<(&'static str, &'static str)>>,
}

impl CommonLibrariesTranslator {
    pub fn new() -> Self {
        let mut translator = Self {
            imports: HashMap::new(),
            dependencies: HashMap::new(),
        };

        translator.initialize_mappings();
        translator
    }

    fn initialize_mappings(&mut self) {
        // Requests → reqwest
        self.dependencies.insert(PythonLibrary::Requests, vec![
            ("reqwest", "0.11"),
            ("tokio", "1.0"),
        ]);

        // Pytest → built-in testing
        self.dependencies.insert(PythonLibrary::Pytest, vec![]);

        // Pydantic → serde + validator
        self.dependencies.insert(PythonLibrary::Pydantic, vec![
            ("serde", "1.0"),
            ("serde_json", "1.0"),
            ("validator", "0.16"),
        ]);

        // Logging → tracing
        self.dependencies.insert(PythonLibrary::Logging, vec![
            ("tracing", "0.1"),
            ("tracing-subscriber", "0.3"),
        ]);

        // Argparse → clap
        self.dependencies.insert(PythonLibrary::Argparse, vec![
            ("clap", "4.0"),
        ]);

        // Json → serde_json
        self.dependencies.insert(PythonLibrary::Json, vec![
            ("serde_json", "1.0"),
        ]);

        // Datetime → chrono
        self.dependencies.insert(PythonLibrary::Datetime, vec![
            ("chrono", "0.4"),
        ]);

        // Re → regex
        self.dependencies.insert(PythonLibrary::Regex, vec![
            ("regex", "1.0"),
        ]);
    }

    /// Translate Requests HTTP operation
    pub fn translate_requests(&mut self, op: &RequestsOp, args: &[String]) -> String {
        self.add_import(PythonLibrary::Requests, "use reqwest");

        match op {
            RequestsOp::Get => {
                // requests.get(url) → reqwest::get(url).await?
                if args.is_empty() {
                    return "reqwest::get(\"url\").await?.text().await?".to_string();
                }
                format!("reqwest::get({}).await?.text().await?", args[0])
            }
            RequestsOp::Post => {
                // requests.post(url, json=data) → reqwest::Client::new().post(url).json(&data).send().await?
                if args.len() >= 2 {
                    format!("reqwest::Client::new().post({}).json(&{}).send().await?", args[0], args[1])
                } else {
                    "reqwest::Client::new().post(url).send().await?".to_string()
                }
            }
            RequestsOp::Put => {
                if args.is_empty() {
                    return "reqwest::Client::new().put(url).send().await?".to_string();
                }
                format!("reqwest::Client::new().put({}).send().await?", args[0])
            }
            RequestsOp::Delete => {
                if args.is_empty() {
                    return "reqwest::Client::new().delete(url).send().await?".to_string();
                }
                format!("reqwest::Client::new().delete({}).send().await?", args[0])
            }
            RequestsOp::Patch => {
                if args.is_empty() {
                    return "reqwest::Client::new().patch(url).send().await?".to_string();
                }
                format!("reqwest::Client::new().patch({}).send().await?", args[0])
            }
            RequestsOp::Head => {
                if args.is_empty() {
                    return "reqwest::Client::new().head(url).send().await?".to_string();
                }
                format!("reqwest::Client::new().head({}).send().await?", args[0])
            }
            RequestsOp::Options => {
                "reqwest::Client::new().request(Method::OPTIONS, url).send().await?".to_string()
            }
            RequestsOp::Session => {
                // requests.Session() → reqwest::Client::new()
                "reqwest::Client::new()".to_string()
            }
        }
    }

    /// Translate pytest test
    pub fn translate_pytest(&mut self, op: &PytestOp, args: &[String]) -> String {
        match op {
            PytestOp::Test => {
                // def test_foo(): → #[test] fn test_foo()
                if args.is_empty() {
                    return "#[test]\nfn test_foo() {\n    // test body\n}".to_string();
                }
                format!("#[test]\nfn {}() {{\n    // test body\n}}", args[0])
            }
            PytestOp::Fixture => {
                // @pytest.fixture → setup function
                "// Use setup/teardown or lazy_static for fixtures".to_string()
            }
            PytestOp::Parametrize => {
                // @pytest.mark.parametrize → use test cases macro or loop
                "#[test_case::test_case(/* cases */)]".to_string()
            }
            PytestOp::Assert => {
                // assert x == y → assert_eq!(x, y)
                if args.len() >= 2 {
                    format!("assert_eq!({}, {});", args[0], args[1])
                } else {
                    "assert!(condition);".to_string()
                }
            }
            PytestOp::Raises => {
                // with pytest.raises(Exception): → #[should_panic]
                "#[should_panic(expected = \"...\")]".to_string()
            }
            PytestOp::Skip => {
                // @pytest.mark.skip → #[ignore]
                "#[ignore]".to_string()
            }
            PytestOp::Mark => {
                // @pytest.mark.slow → #[cfg_attr(...)]
                "#[cfg_attr(not(feature = \"slow_tests\"), ignore)]".to_string()
            }
        }
    }

    /// Translate pydantic model
    pub fn translate_pydantic(&mut self, op: &PydanticOp, args: &[String]) -> String {
        self.add_import(PythonLibrary::Pydantic, "use serde::{Serialize, Deserialize}");
        self.add_import(PythonLibrary::Pydantic, "use validator::Validate");

        match op {
            PydanticOp::BaseModel => {
                // class Model(BaseModel): → #[derive(Serialize, Deserialize, Validate)]
                if args.is_empty() {
                    return "#[derive(Debug, Serialize, Deserialize, Validate)]\nstruct Model {\n    // fields\n}".to_string();
                }
                format!("#[derive(Debug, Serialize, Deserialize, Validate)]\nstruct {} {{\n    // fields\n}}", args[0])
            }
            PydanticOp::Field => {
                // field: int = Field(gt=0) → #[validate(range(min = 0))]
                if args.len() >= 2 {
                    format!("#[validate({})]\npub {}: {},", args[1], args[0], "T")
                } else {
                    "pub field: T,".to_string()
                }
            }
            PydanticOp::Validator => {
                // @validator → custom validation function
                "// Implement custom validation in separate function".to_string()
            }
            PydanticOp::RootValidator => {
                // @root_validator → validate entire struct
                "// Implement validation in struct impl".to_string()
            }
            PydanticOp::Config => {
                // class Config: → derive attributes
                "#[serde(rename_all = \"camelCase\")]".to_string()
            }
        }
    }

    /// Translate logging
    pub fn translate_logging(&mut self, op: &LoggingOp, args: &[String]) -> String {
        self.add_import(PythonLibrary::Logging, "use tracing::{debug, info, warn, error}");

        match op {
            LoggingOp::Debug => {
                if args.is_empty() {
                    return "debug!(\"message\");".to_string();
                }
                format!("debug!({});", args.join(", "))
            }
            LoggingOp::Info => {
                if args.is_empty() {
                    return "info!(\"message\");".to_string();
                }
                format!("info!({});", args.join(", "))
            }
            LoggingOp::Warning => {
                if args.is_empty() {
                    return "warn!(\"message\");".to_string();
                }
                format!("warn!({});", args.join(", "))
            }
            LoggingOp::Error => {
                if args.is_empty() {
                    return "error!(\"message\");".to_string();
                }
                format!("error!({});", args.join(", "))
            }
            LoggingOp::Critical => {
                if args.is_empty() {
                    return "error!(\"CRITICAL: message\");".to_string();
                }
                format!("error!(\"CRITICAL: {{}}\", {});", args.join(", "))
            }
            LoggingOp::Logger => {
                // logging.getLogger(__name__) → tracing setup
                "// Use tracing_subscriber::fmt::init()".to_string()
            }
        }
    }

    /// Translate argparse
    pub fn translate_argparse(&mut self, op: &ArgparseOp, args: &[String]) -> String {
        self.add_import(PythonLibrary::Argparse, "use clap::Parser");

        match op {
            ArgparseOp::ArgumentParser => {
                // ArgumentParser() → #[derive(Parser)] struct
                "#[derive(Parser, Debug)]\n#[command(author, version, about)]\nstruct Args {\n    // fields\n}".to_string()
            }
            ArgparseOp::AddArgument => {
                // parser.add_argument('--name') → struct field with attribute
                if args.is_empty() {
                    return "#[arg(short, long)]\nfield: String,".to_string();
                }
                format!("#[arg({})]\n{}: String,", args[0], args.get(1).unwrap_or(&"field".to_string()))
            }
            ArgparseOp::ParseArgs => {
                // parser.parse_args() → Args::parse()
                "Args::parse()".to_string()
            }
        }
    }

    /// Translate datetime operations
    pub fn translate_datetime(&self, operation: &str, args: &[String]) -> String {
        match operation {
            "now" => {
                // datetime.now() → Utc::now() or Local::now()
                "chrono::Utc::now()".to_string()
            }
            "date" => {
                // datetime.date(2024, 1, 1) → NaiveDate::from_ymd_opt(2024, 1, 1)
                if args.len() >= 3 {
                    format!("chrono::NaiveDate::from_ymd_opt({}, {}, {}).unwrap()", args[0], args[1], args[2])
                } else {
                    "chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()".to_string()
                }
            }
            "strftime" => {
                // dt.strftime('%Y-%m-%d') → dt.format("%Y-%m-%d").to_string()
                if args.is_empty() {
                    return "dt.format(\"%Y-%m-%d\").to_string()".to_string();
                }
                format!("dt.format({}).to_string()", args[0])
            }
            "strptime" => {
                // datetime.strptime(s, '%Y-%m-%d') → NaiveDate::parse_from_str(s, "%Y-%m-%d")
                if args.len() >= 2 {
                    format!("chrono::NaiveDateTime::parse_from_str({}, {}).unwrap()", args[0], args[1])
                } else {
                    "chrono::NaiveDateTime::parse_from_str(s, \"%Y-%m-%d\").unwrap()".to_string()
                }
            }
            "timedelta" => {
                // timedelta(days=1) → Duration::days(1)
                "chrono::Duration::days(1)".to_string()
            }
            _ => format!("/* datetime.{} */", operation),
        }
    }

    /// Translate pathlib operations
    pub fn translate_pathlib(&self, operation: &str, args: &[String]) -> String {
        match operation {
            "Path" => {
                // Path('file.txt') → Path::new("file.txt")
                if args.is_empty() {
                    return "std::path::Path::new(\"file.txt\")".to_string();
                }
                format!("std::path::Path::new({})", args[0])
            }
            "exists" => {
                // path.exists() → path.exists()
                "path.exists()".to_string()
            }
            "is_file" => {
                "path.is_file()".to_string()
            }
            "is_dir" => {
                "path.is_dir()".to_string()
            }
            "read_text" => {
                // path.read_text() → std::fs::read_to_string(path)?
                "std::fs::read_to_string(path)?".to_string()
            }
            "write_text" => {
                // path.write_text(s) → std::fs::write(path, s)?
                if args.is_empty() {
                    return "std::fs::write(path, content)?".to_string();
                }
                format!("std::fs::write(path, {})?", args[0])
            }
            "mkdir" => {
                // path.mkdir() → std::fs::create_dir(path)?
                "std::fs::create_dir(path)?".to_string()
            }
            "rmdir" => {
                "std::fs::remove_dir(path)?".to_string()
            }
            "glob" => {
                // path.glob('*.txt') → use glob crate
                if args.is_empty() {
                    return "glob::glob(\"*.txt\")?".to_string();
                }
                format!("glob::glob({})?", args[0])
            }
            _ => format!("/* pathlib.{} */", operation),
        }
    }

    /// Translate regex operations
    pub fn translate_regex(&self, operation: &str, args: &[String]) -> String {
        match operation {
            "compile" => {
                // re.compile(r'pattern') → Regex::new(r"pattern")?
                if args.is_empty() {
                    return "regex::Regex::new(r\"pattern\")?".to_string();
                }
                format!("regex::Regex::new({})?", args[0])
            }
            "match" | "search" => {
                // re.match(pattern, text) → regex.is_match(text)
                if args.len() >= 2 {
                    format!("regex::Regex::new({})?.is_match({})", args[0], args[1])
                } else {
                    "regex.is_match(text)".to_string()
                }
            }
            "findall" => {
                // re.findall(pattern, text) → regex.find_iter(text).collect()
                "regex.find_iter(text).map(|m| m.as_str()).collect::<Vec<_>>()".to_string()
            }
            "sub" => {
                // re.sub(pattern, repl, text) → regex.replace_all(text, repl)
                if args.len() >= 2 {
                    format!("regex.replace_all({}, {})", args[1], args[0])
                } else {
                    "regex.replace_all(text, replacement)".to_string()
                }
            }
            _ => format!("/* re.{} */", operation),
        }
    }

    /// Add import for library
    fn add_import(&mut self, lib: PythonLibrary, import: &str) {
        self.imports.entry(lib)
            .or_default()
            .push(import.to_string());
    }

    /// Get imports for specific library
    pub fn get_imports(&self, lib: &PythonLibrary) -> Vec<String> {
        self.imports.get(lib).cloned().unwrap_or_default()
    }

    /// Get all cargo dependencies
    pub fn get_cargo_dependencies(&self) -> Vec<(&'static str, &'static str)> {
        let mut deps = Vec::new();
        for lib_deps in self.dependencies.values() {
            deps.extend(lib_deps.iter().copied());
        }
        deps.sort();
        deps.dedup();
        deps
    }

    /// Get dependencies for specific library
    pub fn get_library_dependencies(&self, lib: &PythonLibrary) -> Vec<(&'static str, &'static str)> {
        self.dependencies.get(lib).cloned().unwrap_or_default()
    }
}

impl Default for CommonLibrariesTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_requests_get() {
        let mut translator = CommonLibrariesTranslator::new();
        let result = translator.translate_requests(&RequestsOp::Get, &["\"https://api.example.com\"".to_string()]);
        assert!(result.contains("reqwest::get"));
        assert!(result.contains("await"));
    }

    #[test]
    fn test_pytest_test() {
        let mut translator = CommonLibrariesTranslator::new();
        let result = translator.translate_pytest(&PytestOp::Test, &["test_example".to_string()]);
        assert!(result.contains("#[test]"));
        assert!(result.contains("test_example"));
    }

    #[test]
    fn test_pydantic_model() {
        let mut translator = CommonLibrariesTranslator::new();
        let result = translator.translate_pydantic(&PydanticOp::BaseModel, &["User".to_string()]);
        assert!(result.contains("Serialize"));
        assert!(result.contains("Deserialize"));
        assert!(result.contains("User"));
    }

    #[test]
    fn test_logging() {
        let mut translator = CommonLibrariesTranslator::new();
        let result = translator.translate_logging(&LoggingOp::Info, &["\"message\"".to_string()]);
        assert!(result.contains("info!"));
    }

    #[test]
    fn test_datetime() {
        let translator = CommonLibrariesTranslator::new();
        let result = translator.translate_datetime("now", &[]);
        assert!(result.contains("chrono"));
    }
}
