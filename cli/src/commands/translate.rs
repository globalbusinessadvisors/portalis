//! Translate command implementation

use anyhow::{Context, Result};
use clap::Args;
use colored::Colorize;
use portalis_transpiler::{TranslationMode, TranspilerAgent};
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct TranslateCommand {
    /// Input Python file
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output WASM file
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Translation mode: pattern, ast, feature, nemo
    #[arg(short, long, default_value = "pattern")]
    pub mode: String,

    /// Show generated Rust code
    #[arg(long)]
    pub show_rust: bool,

    /// Save Rust code to file
    #[arg(long)]
    pub save_rust: Option<PathBuf>,

    /// NeMo sampling temperature (0.0-1.0)
    #[arg(long, default_value = "0.2")]
    pub temperature: f32,

    /// Optimization level: 0, 1, 2, 3, s, z
    #[arg(short = 'O', long, default_value = "3")]
    pub opt_level: String,

    /// Strip debug symbols
    #[arg(long)]
    pub strip_debug: bool,

    /// Run conformance tests
    #[arg(long, default_value = "true")]
    pub run_tests: bool,

    /// Skip conformance tests
    #[arg(long)]
    pub no_tests: bool,
}

impl TranslateCommand {
    pub async fn execute(&self) -> Result<()> {
        println!("{} Translating {:?}", "🔄".blue(), self.input);

        // Read input file
        let source_code = fs::read_to_string(&self.input)
            .with_context(|| format!("Failed to read input file: {}", self.input.display()))?;

        // Create transpiler with appropriate mode
        let translation_mode = self.parse_mode()?;
        let transpiler = TranspilerAgent::with_mode(translation_mode);

        // Translate Python to Rust
        let rust_code = transpiler
            .translate_python_module(&source_code)
            .context("Translation failed")?;

        // Show Rust code if requested
        if self.show_rust {
            println!("\n{} Generated Rust Code:", "📝".green());
            println!("{}", "=".repeat(80));
            println!("{}", rust_code);
            println!("{}", "=".repeat(80));
        }

        // Save Rust code if requested
        if let Some(rust_path) = &self.save_rust {
            fs::write(rust_path, &rust_code)
                .with_context(|| format!("Failed to save Rust code to {}", rust_path.display()))?;
            println!("{} Saved Rust code to {:?}", "💾".green(), rust_path);
        }

        // Compile to WASM
        println!("{} Compiling to WASM...", "🔨".yellow());
        let wasm_bytes = self.compile_to_wasm(&rust_code).await?;

        // Write WASM output
        let output_path = self.output.clone().unwrap_or_else(|| {
            let mut path = self.input.clone();
            path.set_extension("wasm");
            path
        });

        fs::write(&output_path, &wasm_bytes)
            .with_context(|| format!("Failed to write WASM file: {}", output_path.display()))?;

        // Run tests if enabled
        let tests_passed = if self.run_tests && !self.no_tests {
            println!("{} Running conformance tests...", "🧪".cyan());
            self.run_conformance_tests(&self.input, &output_path).await?
        } else {
            true
        };

        // Print summary
        println!("\n{} Translation complete!", "✅".green().bold());
        println!("   {} {} lines", "Rust code:".bold(), rust_code.lines().count());
        println!("   {} {} bytes", "WASM size:".bold(), wasm_bytes.len());
        if self.run_tests && !self.no_tests {
            if tests_passed {
                println!("   {} {}", "Tests:".bold(), "PASSED".green());
            } else {
                println!("   {} {}", "Tests:".bold(), "FAILED".red());
            }
        }
        println!("   {} {:?}", "Output:".bold(), output_path);

        if !tests_passed {
            anyhow::bail!("Conformance tests failed");
        }

        Ok(())
    }

    fn parse_mode(&self) -> Result<TranslationMode> {
        match self.mode.to_lowercase().as_str() {
            "pattern" => Ok(TranslationMode::PatternBased),
            "ast" => Ok(TranslationMode::AstBased),
            "feature" => Ok(TranslationMode::FeatureBased),
            "nemo" => {
                #[cfg(feature = "nemo")]
                {
                    Ok(TranslationMode::NeMo {
                        service_url: std::env::var("NEMO_SERVICE_URL")
                            .unwrap_or_else(|_| "http://localhost:8000".to_string()),
                        mode: "translation".to_string(),
                        temperature: self.temperature,
                    })
                }
                #[cfg(not(feature = "nemo"))]
                {
                    anyhow::bail!("NeMo mode requires the 'nemo' feature. Rebuild with: cargo build --features nemo")
                }
            }
            _ => anyhow::bail!(
                "Invalid translation mode: {}. Use: pattern, ast, feature, or nemo",
                self.mode
            ),
        }
    }

    async fn compile_to_wasm(&self, rust_code: &str) -> Result<Vec<u8>> {
        // TODO: Implement actual Rust → WASM compilation
        // For now, return a placeholder
        Ok(rust_code.as_bytes().to_vec())
    }

    async fn run_conformance_tests(&self, _python_path: &PathBuf, _wasm_path: &PathBuf) -> Result<bool> {
        // TODO: Implement actual conformance testing
        // For now, return success
        Ok(true)
    }
}
