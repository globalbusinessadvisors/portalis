//! Batch translation command

use anyhow::{Context, Result};
use clap::Args;
use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::JoinSet;
use walkdir::WalkDir;

#[derive(Args, Debug)]
pub struct BatchCommand {
    /// Input directory with Python files
    #[arg(short, long)]
    pub input_dir: PathBuf,

    /// Output directory for WASM files
    #[arg(short, long, default_value = "./dist")]
    pub output_dir: PathBuf,

    /// File pattern to match
    #[arg(short, long, default_value = "**/*.py")]
    pub pattern: String,

    /// Number of parallel workers
    #[arg(short = 'j', long)]
    pub parallel: Option<usize>,

    /// Translation mode
    #[arg(short, long, default_value = "pattern")]
    pub mode: String,

    /// Recursively search subdirectories
    #[arg(short, long, default_value = "true")]
    pub recursive: bool,

    /// Maintain directory structure in output
    #[arg(long, default_value = "true")]
    pub preserve_structure: bool,

    /// Stop on first error
    #[arg(long)]
    pub fail_fast: bool,

    /// Continue after errors
    #[arg(long, default_value = "true")]
    pub continue_on_error: bool,
}

impl BatchCommand {
    pub async fn execute(&self) -> Result<()> {
        println!("{} Batch translating Python files", "🚀".blue().bold());
        println!("   {} {:?}", "Input:".bold(), self.input_dir);
        println!("   {} {:?}", "Output:".bold(), self.output_dir);

        // Find all Python files
        let files = self.find_python_files()?;
        println!("   {} {} files", "Found:".bold(), files.len());

        if files.is_empty() {
            println!("{} No Python files found", "⚠️".yellow());
            return Ok(());
        }

        // Determine number of workers
        let workers = self.parallel.unwrap_or_else(|| num_cpus::get());
        println!("   {} {}\n", "Workers:".bold(), workers);

        // Create output directory
        std::fs::create_dir_all(&self.output_dir)
            .context("Failed to create output directory")?;

        // Setup progress bars
        let multi = Arc::new(MultiProgress::new());
        let main_pb = multi.add(ProgressBar::new(files.len() as u64));
        main_pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        // Process files in parallel
        let mut tasks = JoinSet::new();
        let mut success_count = 0usize;
        let mut failure_count = 0usize;

        for file in files {
            let translate_cmd = super::translate::TranslateCommand {
                input: file.clone(),
                output: Some(self.get_output_path(&file)?),
                mode: self.mode.clone(),
                show_rust: false,
                save_rust: None,
                temperature: 0.2,
                opt_level: "3".to_string(),
                strip_debug: false,
                run_tests: false,
                no_tests: true,
            };

            let file_pb = multi.add(ProgressBar::new_spinner());
            file_pb.set_message(format!("Translating {:?}", file.file_name().unwrap()));

            tasks.spawn(async move {
                let result = translate_cmd.execute().await;
                file_pb.finish_and_clear();
                (file, result)
            });

            // Limit concurrent tasks
            while tasks.len() >= workers {
                if let Some(result) = tasks.join_next().await {
                    match result {
                        Ok((_file, Ok(_))) => {
                            success_count += 1;
                            main_pb.inc(1);
                            main_pb.set_message(format!("{} succeeded", success_count));
                        }
                        Ok((file, Err(e))) => {
                            failure_count += 1;
                            main_pb.inc(1);
                            eprintln!("{} Failed to translate {:?}: {}", "❌".red(), file, e);

                            if self.fail_fast {
                                return Err(anyhow::anyhow!("Translation failed (fail-fast enabled)"));
                            }
                        }
                        Err(e) => {
                            eprintln!("{} Task error: {}", "❌".red(), e);
                        }
                    }
                }
            }
        }

        // Wait for remaining tasks
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok((_, Ok(_))) => {
                    success_count += 1;
                    main_pb.inc(1);
                    main_pb.set_message(format!("{} succeeded", success_count));
                }
                Ok((file, Err(e))) => {
                    failure_count += 1;
                    main_pb.inc(1);
                    eprintln!("{} Failed to translate {:?}: {}", "❌".red(), file, e);
                }
                Err(e) => {
                    eprintln!("{} Task error: {}", "❌".red(), e);
                }
            }
        }

        main_pb.finish_with_message("Batch translation complete");

        // Print summary
        println!("\n{} Batch translation complete!", "✅".green().bold());
        println!("   {} {}", "Succeeded:".bold(), success_count.to_string().green());
        if failure_count > 0 {
            println!("   {} {}", "Failed:".bold(), failure_count.to_string().red());
        }

        if failure_count > 0 && !self.continue_on_error {
            anyhow::bail!("{} files failed to translate", failure_count);
        }

        Ok(())
    }

    fn find_python_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        let walker = if self.recursive {
            WalkDir::new(&self.input_dir)
        } else {
            WalkDir::new(&self.input_dir).max_depth(1)
        };

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("py") {
                files.push(path.to_path_buf());
            }
        }

        Ok(files)
    }

    fn get_output_path(&self, input: &PathBuf) -> Result<PathBuf> {
        let mut output = self.output_dir.clone();

        if self.preserve_structure {
            // Preserve directory structure
            let rel_path = input
                .strip_prefix(&self.input_dir)
                .context("Failed to compute relative path")?;
            output = output.join(rel_path);
        } else {
            // Flat structure - just filename
            if let Some(filename) = input.file_name() {
                output = output.join(filename);
            }
        }

        output.set_extension("wasm");

        // Create parent directories if needed
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }

        Ok(output)
    }
}
