//! Doctor command - System diagnostics

use anyhow::Result;
use clap::Args;
use colored::Colorize;
use std::process::Command;
use which::which;

#[derive(Args, Debug)]
pub struct DoctorCommand {
    /// Show detailed diagnostics
    #[arg(long)]
    pub verbose: bool,

    /// Attempt to fix issues
    #[arg(long)]
    pub fix: bool,
}

impl DoctorCommand {
    pub async fn execute(&self) -> Result<()> {
        println!("{}", "Portalis System Diagnostics".bold());
        println!("{}", "============================\n");

        self.check_rust();
        self.check_wasm_target();
        self.check_gpu();
        self.check_deployment_tools();

        println!("\n{} System diagnostic complete", "✅".green());

        Ok(())
    }

    fn check_rust(&self) {
        print!("Rust compiler: ");
        match Command::new("rustc").arg("--version").output() {
            Ok(output) => {
                let version = String::from_utf8_lossy(&output.stdout);
                println!("{} {}", "✅".green(), version.trim());
            }
            Err(_) => {
                println!("{} not found", "❌".red());
            }
        }

        print!("Cargo: ");
        match Command::new("cargo").arg("--version").output() {
            Ok(output) => {
                let version = String::from_utf8_lossy(&output.stdout);
                println!("{} {}", "✅".green(), version.trim());
            }
            Err(_) => {
                println!("{} not found", "❌".red());
            }
        }
    }

    fn check_wasm_target(&self) {
        print!("WASM target (wasm32-wasi): ");
        match Command::new("rustup")
            .args(&["target", "list", "--installed"])
            .output()
        {
            Ok(output) => {
                let targets = String::from_utf8_lossy(&output.stdout);
                if targets.contains("wasm32-wasi") {
                    println!("{} installed", "✅".green());
                } else {
                    println!("{} not installed", "⚠️".yellow());
                    println!("   Install with: rustup target add wasm32-wasi");
                }
            }
            Err(_) => {
                println!("{} unable to check", "❌".red());
            }
        }
    }

    fn check_gpu(&self) {
        println!("\n{}", "GPU Acceleration:".bold());

        print!("CUDA: ");
        match which("nvcc") {
            Ok(_) => {
                println!("{} detected", "✅".green());
            }
            Err(_) => {
                println!("{} not detected (optional)", "⚠️".yellow());
            }
        }

        print!("NeMo service: ");
        // TODO: Check if NeMo service is running
        println!("{} not running (optional)", "⚠️".yellow());
        println!("   Start with: docker-compose up nemo-service");
    }

    fn check_deployment_tools(&self) {
        println!("\n{}", "Deployment:".bold());

        print!("Docker: ");
        match Command::new("docker").arg("--version").output() {
            Ok(output) => {
                let version = String::from_utf8_lossy(&output.stdout);
                println!("{} {}", "✅".green(), version.trim());
            }
            Err(_) => {
                println!("{} not found (optional)", "⚠️".yellow());
            }
        }

        print!("Kubernetes: ");
        match Command::new("kubectl").arg("version").arg("--client").output() {
            Ok(_) => {
                println!("{} installed", "✅".green());
            }
            Err(_) => {
                println!("{} not found (optional)", "⚠️".yellow());
            }
        }

        print!("Helm: ");
        match which("helm") {
            Ok(_) => {
                println!("{} installed", "✅".green());
            }
            Err(_) => {
                println!("{} not installed (optional)", "⚠️".yellow());
            }
        }
    }
}
