//! Migration Planning Command
//!
//! Analyzes dependency graph and generates migration plan.

use anyhow::{Context, Result};
use clap::Args;
use portalis_ingest::{ProjectParser, PythonProject};
use std::collections::HashMap;
use std::path::PathBuf;

/// Migration plan
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    pub strategy: MigrationStrategy,
    pub phases: Vec<MigrationPhase>,
    pub total_modules: usize,
    pub estimated_duration_days: u32,
}

/// Migration strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationStrategy {
    FullMigration,      // Translate everything at once
    Incremental,        // Translate module by module
    BottomUp,           // Start with dependencies
    TopDown,            // Start with top-level modules
    CriticalPath,       // Start with most important modules
}

/// Migration phase
#[derive(Debug, Clone)]
pub struct MigrationPhase {
    pub phase_number: usize,
    pub name: String,
    pub modules: Vec<String>,
    pub estimated_days: u32,
    pub dependencies_satisfied: bool,
}

/// Migration planning command
#[derive(Args, Debug)]
pub struct PlanCommand {
    /// Python project directory
    #[arg(short, long)]
    pub project: PathBuf,

    /// Migration strategy (full, incremental, bottom-up, top-down, critical-path)
    #[arg(short, long, default_value = "bottom-up")]
    pub strategy: String,

    /// Output plan file
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

impl PlanCommand {

    /// Execute the planning command
    pub async fn execute(&self) -> Result<()> {
        let strategy = self.parse_strategy()?;

        println!("🗺️  Creating migration plan for: {}", self.project.display());
        println!("   Strategy: {:?}\n", strategy);

        // Parse the project
        let project = self.parse_project()?;

        println!("📊 Analyzed {} modules", project.modules.len());
        println!("   Dependencies: {} edges\n", project.dependency_graph.edges.len());

        // Generate migration plan
        let plan = self.generate_plan(&project, strategy)?;

        // Print plan
        self.print_plan(&plan);

        // Save plan if output specified
        if let Some(output_path) = &self.output {
            self.save_plan(&plan, output_path)?;
            println!("\n📄 Plan saved to: {}", output_path.display());
        }

        Ok(())
    }

    /// Parse the Python project
    fn parse_project(&self) -> Result<PythonProject> {
        let parser = ProjectParser::new();
        parser.parse_project(&self.project)
            .context("Failed to parse Python project")
    }

    /// Generate migration plan
    fn generate_plan(&self, project: &PythonProject, strategy: MigrationStrategy) -> Result<MigrationPlan> {
        let plan = match strategy {
            MigrationStrategy::FullMigration => self.plan_full_migration(project)?,
            MigrationStrategy::Incremental => self.plan_incremental(project)?,
            MigrationStrategy::BottomUp => self.plan_bottom_up(project)?,
            MigrationStrategy::TopDown => self.plan_top_down(project)?,
            MigrationStrategy::CriticalPath => self.plan_critical_path(project)?,
        };

        Ok(plan)
    }

    /// Plan full migration (all at once)
    fn plan_full_migration(&self, project: &PythonProject) -> Result<MigrationPlan> {
        let all_modules: Vec<String> = project.modules.keys().cloned().collect();

        let phases = vec![MigrationPhase {
            phase_number: 1,
            name: "Full Translation".to_string(),
            modules: all_modules.clone(),
            estimated_days: (all_modules.len() as f64 * 2.0).ceil() as u32,
            dependencies_satisfied: true,
        }];

        Ok(MigrationPlan {
            strategy: MigrationStrategy::FullMigration,
            phases,
            total_modules: all_modules.len(),
            estimated_duration_days: (all_modules.len() as f64 * 2.0).ceil() as u32,
        })
    }

    /// Plan incremental migration (5 modules at a time)
    fn plan_incremental(&self, project: &PythonProject) -> Result<MigrationPlan> {
        let modules: Vec<String> = project.modules.keys().cloned().collect();
        let batch_size = 5;
        let mut phases = Vec::new();

        for (i, chunk) in modules.chunks(batch_size).enumerate() {
            phases.push(MigrationPhase {
                phase_number: i + 1,
                name: format!("Batch {} (modules {}-{})", i + 1, i * batch_size + 1, i * batch_size + chunk.len()),
                modules: chunk.to_vec(),
                estimated_days: (chunk.len() as f64 * 2.0).ceil() as u32,
                dependencies_satisfied: true,
            });
        }

        let total_days = phases.iter().map(|p| p.estimated_days).sum();

        Ok(MigrationPlan {
            strategy: MigrationStrategy::Incremental,
            phases,
            total_modules: modules.len(),
            estimated_duration_days: total_days,
        })
    }

    /// Plan bottom-up migration (dependencies first)
    fn plan_bottom_up(&self, project: &PythonProject) -> Result<MigrationPlan> {
        let parser = ProjectParser::new();
        let sorted = parser.topological_sort(&project.dependency_graph)
            .context("Failed to sort dependencies (circular dependency?)")?;

        let mut phases = Vec::new();
        let batch_size = 5;

        for (i, chunk) in sorted.chunks(batch_size).enumerate() {
            phases.push(MigrationPhase {
                phase_number: i + 1,
                name: format!("Layer {} (dependencies satisfied)", i + 1),
                modules: chunk.to_vec(),
                estimated_days: (chunk.len() as f64 * 2.0).ceil() as u32,
                dependencies_satisfied: true,
            });
        }

        let total_days = phases.iter().map(|p| p.estimated_days).sum();

        Ok(MigrationPlan {
            strategy: MigrationStrategy::BottomUp,
            phases,
            total_modules: sorted.len(),
            estimated_duration_days: total_days,
        })
    }

    /// Plan top-down migration (top-level first)
    fn plan_top_down(&self, project: &PythonProject) -> Result<MigrationPlan> {
        let parser = ProjectParser::new();
        let mut sorted = parser.topological_sort(&project.dependency_graph)
            .context("Failed to sort dependencies")?;

        // Reverse for top-down
        sorted.reverse();

        let mut phases = Vec::new();
        let batch_size = 5;

        for (i, chunk) in sorted.chunks(batch_size).enumerate() {
            phases.push(MigrationPhase {
                phase_number: i + 1,
                name: format!("Layer {} (top-level first)", i + 1),
                modules: chunk.to_vec(),
                estimated_days: (chunk.len() as f64 * 2.5).ceil() as u32, // Slightly longer (dependencies may not be ready)
                dependencies_satisfied: false,
            });
        }

        let total_days = phases.iter().map(|p| p.estimated_days).sum();

        Ok(MigrationPlan {
            strategy: MigrationStrategy::TopDown,
            phases,
            total_modules: sorted.len(),
            estimated_duration_days: total_days,
        })
    }

    /// Plan critical path migration
    fn plan_critical_path(&self, project: &PythonProject) -> Result<MigrationPlan> {
        // Identify modules with most dependents (most "important")
        let mut module_importance: Vec<(String, usize)> = project.dependency_graph.nodes
            .iter()
            .map(|(name, node)| (name.clone(), node.dependents.len()))
            .collect();

        module_importance.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by importance (descending)

        let mut phases = Vec::new();
        let batch_size = 5;

        for (i, chunk) in module_importance.chunks(batch_size).enumerate() {
            let modules: Vec<String> = chunk.iter().map(|(name, _)| name.clone()).collect();

            phases.push(MigrationPhase {
                phase_number: i + 1,
                name: format!("Priority {} (high impact modules)", i + 1),
                modules,
                estimated_days: (chunk.len() as f64 * 2.0).ceil() as u32,
                dependencies_satisfied: false,
            });
        }

        let total_days = phases.iter().map(|p| p.estimated_days).sum();

        Ok(MigrationPlan {
            strategy: MigrationStrategy::CriticalPath,
            phases,
            total_modules: module_importance.len(),
            estimated_duration_days: total_days,
        })
    }

    /// Print migration plan
    fn print_plan(&self, plan: &MigrationPlan) {
        println!("{}", "=".repeat(80));
        println!("MIGRATION PLAN");
        println!("{}", "=".repeat(80));

        println!("\n📋 Strategy: {:?}", plan.strategy);
        println!("📦 Total Modules: {}", plan.total_modules);
        println!("⏱️  Estimated Duration: {} days ({:.1} weeks)",
            plan.estimated_duration_days,
            plan.estimated_duration_days as f64 / 5.0);
        println!("🔢 Phases: {}", plan.phases.len());

        println!("\n📅 PHASES:\n");

        for phase in &plan.phases {
            println!("Phase {} - {}", phase.phase_number, phase.name);
            println!("  Duration: {} days", phase.estimated_days);
            println!("  Modules: {}", phase.modules.len());

            if self.verbose {
                for module in &phase.modules {
                    println!("    • {}", module);
                }
            } else {
                let preview: Vec<_> = phase.modules.iter().take(3).collect();
                for module in &preview {
                    println!("    • {}", module);
                }
                if phase.modules.len() > 3 {
                    println!("    ... and {} more", phase.modules.len() - 3);
                }
            }

            if !phase.dependencies_satisfied {
                println!("  ⚠️  Warning: Dependencies may not be ready");
            }

            println!();
        }

        println!("{}", "=".repeat(80));

        // Print recommendations
        println!("\n💡 RECOMMENDATIONS:\n");

        match plan.strategy {
            MigrationStrategy::FullMigration => {
                println!("  • Ensure comprehensive test coverage before starting");
                println!("  • Plan for a complete system freeze during migration");
                println!("  • Have rollback plan ready");
            }
            MigrationStrategy::Incremental => {
                println!("  • Test each batch thoroughly before moving to next");
                println!("  • Maintain Python fallbacks during transition");
                println!("  • Monitor performance after each batch");
            }
            MigrationStrategy::BottomUp => {
                println!("  • Safest approach - dependencies always available");
                println!("  • Can run both versions in parallel during transition");
                println!("  • Benefits visible incrementally");
            }
            MigrationStrategy::TopDown => {
                println!("  • May require temporary shims for missing dependencies");
                println!("  • Higher risk but faster time to value");
                println!("  • Ensure critical modules are tested thoroughly");
            }
            MigrationStrategy::CriticalPath => {
                println!("  • Focus on high-impact modules first");
                println!("  • Balance risk with reward");
                println!("  • May need to refactor dependencies");
            }
        }
    }

    /// Save plan to file
    fn save_plan(&self, plan: &MigrationPlan, path: &PathBuf) -> Result<()> {
        let mut content = String::new();

        content.push_str(&format!("# Migration Plan - {:?}\n\n", plan.strategy));
        content.push_str(&format!("**Total Modules:** {}\n", plan.total_modules));
        content.push_str(&format!("**Estimated Duration:** {} days\n", plan.estimated_duration_days));
        content.push_str(&format!("**Phases:** {}\n\n", plan.phases.len()));

        content.push_str("## Phases\n\n");

        for phase in &plan.phases {
            content.push_str(&format!("### Phase {} - {}\n\n", phase.phase_number, phase.name));
            content.push_str(&format!("- **Duration:** {} days\n", phase.estimated_days));
            content.push_str(&format!("- **Modules:** {}\n\n", phase.modules.len()));

            content.push_str("**Module List:**\n\n");
            for module in &phase.modules {
                content.push_str(&format!("- {}\n", module));
            }
            content.push_str("\n");
        }

        std::fs::write(path, content)?;
        Ok(())
    }

    fn parse_strategy(&self) -> Result<MigrationStrategy> {
        match self.strategy.to_lowercase().as_str() {
            "full" => Ok(MigrationStrategy::FullMigration),
            "incremental" => Ok(MigrationStrategy::Incremental),
            "bottom-up" => Ok(MigrationStrategy::BottomUp),
            "top-down" => Ok(MigrationStrategy::TopDown),
            "critical-path" => Ok(MigrationStrategy::CriticalPath),
            _ => anyhow::bail!("Invalid strategy: {}. Use: full, incremental, bottom-up, top-down, or critical-path", self.strategy),
        }
    }
}
