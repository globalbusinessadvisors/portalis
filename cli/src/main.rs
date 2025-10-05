//! Portalis CLI
//!
//! Command-line interface for the Portalis Python → Rust → WASM translation platform.

mod commands;
mod config;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use commands::{
    assess::AssessCommand,
    batch::BatchCommand,
    doctor::DoctorCommand,
    package::PackageCommand,
    plan::PlanCommand,
    serve::ServeCommand,
    test::TestCommand,
    translate::TranslateCommand,
};
use std::io;
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

#[derive(Parser)]
#[command(name = "portalis")]
#[command(version)]
#[command(about = "Python → Rust → WASM Translation Platform", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Path to configuration file
    #[arg(long, global = true, env = "PORTALIS_CONFIG")]
    config: Option<std::path::PathBuf>,

    /// Increase logging verbosity (-v, -vv, -vvv)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Suppress non-error output
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Color output
    #[arg(long, global = true, value_enum, default_value = "auto")]
    color: ColorMode,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, ValueEnum)]
enum ColorMode {
    Always,
    Auto,
    Never,
}

#[derive(Subcommand)]
enum Commands {
    /// Translate a Python file to WASM
    Translate {
        #[command(flatten)]
        args: TranslateCommand,
    },

    /// Batch translate multiple Python files
    Batch {
        #[command(flatten)]
        args: BatchCommand,
    },

    /// Run conformance tests
    Test {
        #[command(flatten)]
        args: TestCommand,
    },

    /// Assess Python codebase for compatibility
    Assess {
        #[command(flatten)]
        args: AssessCommand,
    },

    /// Generate migration plan
    Plan {
        #[command(flatten)]
        args: PlanCommand,
    },

    /// Package WASM for deployment
    Package {
        #[command(flatten)]
        args: PackageCommand,
    },

    /// Run translation service
    Serve {
        #[command(flatten)]
        args: ServeCommand,
    },

    /// System diagnostics
    Doctor {
        #[command(flatten)]
        args: DoctorCommand,
    },

    /// Generate shell completion scripts
    Completion {
        /// The shell to generate completion for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Configure color mode
    match cli.color {
        ColorMode::Always => {
            colored::control::set_override(true);
        }
        ColorMode::Never => {
            colored::control::set_override(false);
        }
        ColorMode::Auto => {
            // Auto-detect based on terminal
        }
    }

    // Configure logging
    if !cli.quiet {
        let log_level = match cli.verbose {
            0 => "info",
            1 => "debug",
            _ => "trace",
        };

        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new(log_level))
            )
            .with_span_events(FmtSpan::CLOSE)
            .with_target(cli.verbose >= 2)
            .with_thread_ids(cli.verbose >= 3)
            .init();
    }

    // Load configuration if specified
    let _config = if let Some(config_path) = cli.config {
        Some(config::Config::load(config_path)?)
    } else {
        config::Config::load_default().ok()
    };

    // Execute command
    match cli.command {
        Commands::Translate { args } => args.execute().await?,
        Commands::Batch { args } => args.execute().await?,
        Commands::Test { args } => args.execute().await?,
        Commands::Assess { args } => args.execute().await?,
        Commands::Plan { args } => args.execute().await?,
        Commands::Package { args } => args.execute().await?,
        Commands::Serve { args } => args.execute().await?,
        Commands::Doctor { args } => args.execute().await?,
        Commands::Completion { shell } => {
            generate_completion(shell);
        }
    }

    Ok(())
}

fn generate_completion(shell: Shell) {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "portalis", &mut io::stdout());
}
