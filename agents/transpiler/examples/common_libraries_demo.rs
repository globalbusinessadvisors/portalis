//! Common Python Libraries Translation Examples
//!
//! Demonstrates translation of popular Python libraries to Rust equivalents:
//! - Requests → reqwest
//! - pytest → #[test] + assert macros
//! - pydantic → serde + validator
//! - logging → tracing
//! - argparse → clap
//! - datetime → chrono
//! - pathlib → std::path
//! - regex → regex crate

use portalis_transpiler::common_libraries_translator::*;

fn main() {
    println!("=== Common Python Libraries Translation Examples ===\n");

    // Example 1: Requests HTTP library
    example_requests();

    // Example 2: pytest testing framework
    example_pytest();

    // Example 3: pydantic data validation
    example_pydantic();

    // Example 4: logging
    example_logging();

    // Example 5: argparse CLI
    example_argparse();

    // Example 6: datetime
    example_datetime();

    // Example 7: pathlib
    example_pathlib();

    // Example 8: regex
    example_regex();
}

fn example_requests() {
    println!("## Example 1: Requests → reqwest\n");

    let mut translator = CommonLibrariesTranslator::new();

    println!("Python Requests                    →  Rust reqwest");
    println!("{}", "-".repeat(80));

    let examples = vec![
        (RequestsOp::Get, vec!["\"https://api.example.com\"".to_string()], "GET request"),
        (RequestsOp::Post, vec!["\"https://api.example.com\"".to_string(), "data".to_string()], "POST with JSON"),
        (RequestsOp::Put, vec!["\"https://api.example.com/1\"".to_string(), "data".to_string()], "PUT update"),
        (RequestsOp::Delete, vec!["\"https://api.example.com/1\"".to_string()], "DELETE request"),
        (RequestsOp::Patch, vec!["\"https://api.example.com/1\"".to_string(), "data".to_string()], "PATCH update"),
    ];

    for (op, args, desc) in examples {
        let rust = translator.translate_requests(&op, &args);
        println!("{}", desc);
        println!("  {}\n", rust);
    }

    println!("Example: Fetching JSON data\n");
    println!("Python:");
    println!(r#"
import requests

response = requests.get('https://api.github.com/users/octocat')
data = response.json()
print(data['name'])
"#);

    println!("Rust:");
    println!(r#"
use reqwest;
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let response = reqwest::get("https://api.github.com/users/octocat")
        .await?
        .json::<Value>()
        .await?;

    println!("{{}}", response["name"]);
    Ok(())
}}
"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_pytest() {
    println!("## Example 2: pytest → Rust testing\n");

    let mut translator = CommonLibrariesTranslator::new();

    println!("Python pytest                      →  Rust #[test]");
    println!("{}", "-".repeat(80));

    println!("Test function:");
    let rust = translator.translate_pytest(&PytestOp::Test, &["test_addition".to_string()]);
    println!("{}\n", rust);

    println!("Assertions:");
    let assertions = vec![
        (PytestOp::Assert, vec!["result".to_string(), "expected".to_string()], "assert result == expected"),
        (PytestOp::Raises, vec![], "assert raises exception"),
        (PytestOp::Skip, vec!["reason".to_string()], "skip test"),
        (PytestOp::Mark, vec!["slow".to_string()], "mark test"),
    ];

    for (op, args, python) in assertions {
        let rust = translator.translate_pytest(&op, &args);
        println!("  {} → {}", python, rust);
    }

    println!("\nExample: Unit test\n");
    println!("Python:");
    println!(r#"
import pytest

def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0
"#);

    println!("Rust:");
    println!(r#"
fn add(a: i32, b: i32) -> i32 {{
    a + b
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_add() {{
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }}

    #[test]
    #[should_panic]
    fn test_divide_by_zero() {{
        let _ = 1 / 0;
    }}
}}
"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_pydantic() {
    println!("## Example 3: pydantic → serde + validator\n");

    let mut translator = CommonLibrariesTranslator::new();

    println!("Python pydantic                    →  Rust serde + validator");
    println!("{}", "-".repeat(80));

    println!("Base model:");
    let rust = translator.translate_pydantic(&PydanticOp::BaseModel, &["User".to_string()]);
    println!("{}\n", rust);

    println!("Field with validation:");
    let rust = translator.translate_pydantic(&PydanticOp::Field, &["email".to_string(), "String".to_string()]);
    println!("{}\n", rust);

    println!("Example: Data validation\n");
    println!("Python:");
    println!(r#"
from pydantic import BaseModel, Field, validator

class User(BaseModel):
    name: str
    email: str
    age: int = Field(..., ge=0, le=150)

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v
"#);

    println!("Rust:");
    println!(r#"
use serde::{{Deserialize, Serialize}};
use validator::{{Validate, ValidationError}};

#[derive(Debug, Serialize, Deserialize, Validate)]
struct User {{
    name: String,

    #[validate(email)]
    email: String,

    #[validate(range(min = 0, max = 150))]
    age: i32,
}}

impl User {{
    fn validate_email(email: &str) -> Result<(), ValidationError> {{
        if !email.contains('@') {{
            return Err(ValidationError::new("Invalid email"));
        }}
        Ok(())
    }}
}}
"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_logging() {
    println!("## Example 4: logging → tracing\n");

    let mut translator = CommonLibrariesTranslator::new();

    println!("Python logging                     →  Rust tracing");
    println!("{}", "-".repeat(80));

    let levels = vec![
        (LoggingOp::Debug, vec!["Debug message".to_string()], "DEBUG"),
        (LoggingOp::Info, vec!["Info message".to_string()], "INFO"),
        (LoggingOp::Warning, vec!["Warning message".to_string()], "WARNING"),
        (LoggingOp::Error, vec!["Error message".to_string()], "ERROR"),
    ];

    for (op, args, level) in levels {
        let rust = translator.translate_logging(&op, &args);
        println!("{:<10} → {}", level, rust);
    }

    println!("\nExample: Application logging\n");
    println!("Python:");
    println!(r#"
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(data):
    logger.info(f"Processing {{len(data)}} items")
    try:
        # process data
        logger.debug("Data processed successfully")
    except Exception as e:
        logger.error(f"Error: {{e}}")
"#);

    println!("Rust:");
    println!(r#"
use tracing::{{info, debug, error}};

fn process_data(data: &[Data]) {{
    info!("Processing {{}} items", data.len());

    match process(data) {{
        Ok(_) => debug!("Data processed successfully"),
        Err(e) => error!("Error: {{}}", e),
    }}
}}

// In main:
// tracing_subscriber::fmt::init();
"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_argparse() {
    println!("## Example 5: argparse → clap\n");

    let mut translator = CommonLibrariesTranslator::new();

    println!("Python argparse                    →  Rust clap");
    println!("{}", "-".repeat(80));

    println!("Parser creation:");
    let rust = translator.translate_argparse(&ArgparseOp::ArgumentParser, &["My App".to_string()]);
    println!("{}\n", rust);

    println!("Add arguments:");
    let rust = translator.translate_argparse(&ArgparseOp::AddArgument,
        &["--verbose".to_string(), "bool".to_string()]);
    println!("{}\n", rust);

    println!("Example: CLI application\n");
    println!("Python:");
    println!(r#"
import argparse

parser = argparse.ArgumentParser(description='Process files')
parser.add_argument('input', help='Input file')
parser.add_argument('--output', '-o', help='Output file')
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()
print(f"Input: {{args.input}}")
"#);

    println!("Rust:");
    println!(r#"
use clap::Parser;

#[derive(Parser)]
#[command(about = "Process files")]
struct Args {{
    /// Input file
    input: String,

    /// Output file
    #[arg(short, long)]
    output: Option<String>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}}

fn main() {{
    let args = Args::parse();
    println!("Input: {{}}", args.input);
}}
"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_datetime() {
    println!("## Example 6: datetime → chrono\n");

    let translator = CommonLibrariesTranslator::new();

    println!("Python datetime                    →  Rust chrono");
    println!("{}", "-".repeat(80));

    println!("Current time:");
    let rust = translator.translate_datetime("now", &[]);
    println!("  datetime.now() → {}\n", rust);

    println!("Format:");
    let rust = translator.translate_datetime("strftime", &["dt".to_string(), "\"%Y-%m-%d\"".to_string()]);
    println!("  dt.strftime() → {}\n", rust);

    println!("Parse:");
    let rust = translator.translate_datetime("strptime", &["\"2024-01-01\"".to_string(), "\"%Y-%m-%d\"".to_string()]);
    println!("  datetime.strptime() → {}\n", rust);

    println!("Example: Working with dates\n");
    println!("Python:");
    println!(r#"
from datetime import datetime, timedelta

now = datetime.now()
tomorrow = now + timedelta(days=1)
formatted = tomorrow.strftime('%Y-%m-%d')
print(formatted)
"#);

    println!("Rust:");
    println!(r#"
use chrono::{{Local, Duration}};

let now = Local::now();
let tomorrow = now + Duration::days(1);
let formatted = tomorrow.format("%Y-%m-%d");
println!("{{}}", formatted);
"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_pathlib() {
    println!("## Example 7: pathlib → std::path\n");

    let translator = CommonLibrariesTranslator::new();

    println!("Python pathlib                     →  Rust std::path");
    println!("{}", "-".repeat(80));

    println!("Path operations:");
    let ops = vec![
        ("path", vec!["\"file.txt\"".to_string()], "Create path"),
        ("exists", vec!["path".to_string()], "Check exists"),
        ("is_file", vec!["path".to_string()], "Is file?"),
        ("is_dir", vec!["path".to_string()], "Is directory?"),
        ("parent", vec!["path".to_string()], "Parent directory"),
        ("stem", vec!["path".to_string()], "File stem"),
    ];

    for (method, args, desc) in ops {
        let rust = translator.translate_pathlib(method, &args);
        println!("{:<20} → {}", desc, rust);
    }

    println!("\nExample: File operations\n");
    println!("Python:");
    println!(r#"
from pathlib import Path

path = Path('data/file.txt')
if path.exists():
    print(f"Size: {{path.stat().st_size}}")
    parent = path.parent
    name = path.stem
"#);

    println!("Rust:");
    println!(r#"
use std::path::Path;

let path = Path::new("data/file.txt");
if path.exists() {{
    if let Ok(metadata) = path.metadata() {{
        println!("Size: {{}}", metadata.len());
    }}
    let parent = path.parent();
    let name = path.file_stem();
}}
"#);

    println!("\n{}\n", "=".repeat(80));
}

fn example_regex() {
    println!("## Example 8: regex → regex crate\n");

    let translator = CommonLibrariesTranslator::new();

    println!("Python re                          →  Rust regex");
    println!("{}", "-".repeat(80));

    println!("Pattern matching:");
    let ops = vec![
        ("compile", vec!["r\"\\d+\"".to_string()], "Compile pattern"),
        ("match", vec!["pattern".to_string(), "\"text\"".to_string()], "Match at start"),
        ("search", vec!["pattern".to_string(), "\"text\"".to_string()], "Search anywhere"),
        ("findall", vec!["pattern".to_string(), "\"text\"".to_string()], "Find all matches"),
        ("sub", vec!["pattern".to_string(), "\"replacement\"".to_string(), "\"text\"".to_string()], "Replace"),
    ];

    for (method, args, desc) in ops {
        let rust = translator.translate_regex(method, &args);
        println!("{:<20} → {}", desc, rust);
    }

    println!("\nExample: Email validation\n");
    println!("Python:");
    println!(r#"
import re

pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
email = 'user@example.com'

if pattern.match(email):
    print('Valid email')
"#);

    println!("Rust:");
    println!(r#"
use regex::Regex;

let pattern = Regex::new(r"^[\w\.-]+@[\w\.-]+\.\w+$").unwrap();
let email = "user@example.com";

if pattern.is_match(email) {{
    println!("Valid email");
}}
"#);

    println!("\nRequired dependencies:");
    for (name, version) in translator.get_cargo_dependencies() {
        println!("  {} = \"{}\"", name, version);
    }

    println!("\n{}\n", "=".repeat(80));
}
