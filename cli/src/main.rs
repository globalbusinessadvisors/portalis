//! Portalis CLI
//!
//! Command-line interface for the Portalis translation platform.

mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand};
use commands::{AssessCommand, PlanCommand};
use portalis_core::assessment::ReportFormat;
use portalis_orchestration::Pipeline;
use std::fs;
use std::path::PathBuf;
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "portalis")]
#[command(about = "Python → Rust → WASM Translation Platform", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Translate a Python file to WASM
    Translate {
        /// Input Python file
        #[arg(short, long)]
        input: PathBuf,

        /// Output WASM file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Show generated Rust code
        #[arg(long)]
        show_rust: bool,
    },

    /// Assess Python codebase for compatibility
    Assess {
        /// Python project directory
        #[arg(short, long)]
        project: PathBuf,

        /// Output report file
        #[arg(short, long)]
        report: Option<PathBuf>,

        /// Report format (html, json, markdown)
        #[arg(short, long, default_value = "html")]
        format: String,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Generate migration plan
    Plan {
        /// Python project directory
        #[arg(short, long)]
        project: PathBuf,

        /// Migration strategy (full, incremental, bottom-up, top-down, critical-path)
        #[arg(short, long, default_value = "bottom-up")]
        strategy: String,

        /// Output plan file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show version information
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Translate { input, output, show_rust } => {
            translate_file(input, output, show_rust).await?;
        }
        Commands::Assess { project, report, format, verbose } => {
            let fmt = parse_report_format(&format)?;
            let cmd = AssessCommand::new(project, report, fmt, verbose);
            cmd.execute().await?;
        }
        Commands::Plan { project, strategy, output, verbose } => {
            let strat = parse_migration_strategy(&strategy)?;
            let cmd = PlanCommand::new(project, strat, output, verbose);
            cmd.execute().await?;
        }
        Commands::Version => {
            println!("Portalis v{}", env!("CARGO_PKG_VERSION"));
            println!("Python → Rust → WASM Translation Platform");
        }
    }

    Ok(())
}

async fn translate_file(input: PathBuf, output: Option<PathBuf>, show_rust: bool) -> Result<()> {
    println!("🔄 Translating {:?}", input);

    // Read input file
    let source_code = fs::read_to_string(&input)?;

    // Create pipeline and execute
    let mut pipeline = Pipeline::new();
    let result = pipeline.translate(input.clone(), source_code).await?;

    // Show Rust code if requested
    if show_rust {
        println!("\n📝 Generated Rust Code:");
        println!("{}", "=".repeat(80));
        println!("{}", result.rust_code);
        println!("{}", "=".repeat(80));
    }

    // Write WASM output
    let output_path = output.unwrap_or_else(|| {
        let mut path = input.clone();
        path.set_extension("wasm");
        path
    });

    fs::write(&output_path, &result.wasm_bytes)?;

    println!("\n✅ Translation complete!");
    println!("   Rust code: {} lines", result.rust_code.lines().count());
    println!("   WASM size: {} bytes", result.wasm_bytes.len());
    println!("   Tests: {} passed, {} failed", result.test_passed, result.test_failed);
    println!("   Output: {:?}", output_path);

    Ok(())
}

fn parse_report_format(format: &str) -> Result<ReportFormat> {
    match format.to_lowercase().as_str() {
        "html" => Ok(ReportFormat::Html),
        "json" => Ok(ReportFormat::Json),
        "markdown" | "md" => Ok(ReportFormat::Markdown),
        "pdf" => Ok(ReportFormat::Pdf),
        _ => Err(anyhow::anyhow!("Invalid format: {}. Use: html, json, markdown, or pdf", format)),
    }
}

fn parse_migration_strategy(strategy: &str) -> Result<commands::plan::MigrationStrategy> {
    use commands::plan::MigrationStrategy;

    match strategy.to_lowercase().as_str() {
        "full" => Ok(MigrationStrategy::FullMigration),
        "incremental" => Ok(MigrationStrategy::Incremental),
        "bottom-up" => Ok(MigrationStrategy::BottomUp),
        "top-down" => Ok(MigrationStrategy::TopDown),
        "critical-path" => Ok(MigrationStrategy::CriticalPath),
        _ => Err(anyhow::anyhow!("Invalid strategy: {}. Use: full, incremental, bottom-up, top-down, or critical-path", strategy)),
    }
}
