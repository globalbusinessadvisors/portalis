//! Assessment Command
//!
//! Scans Python codebase and generates compatibility assessment report.

use anyhow::{Context, Result};
use clap::Args;
use portalis_core::assessment::{
    FeatureDetector, CompatibilityAnalyzer, EffortEstimator, ReportGenerator, ReportFormat,
    feature_detector::PythonAst,
};
use portalis_ingest::{ProjectParser, PythonProject};
use std::collections::HashMap;
use std::path::PathBuf;

/// Assessment command
#[derive(Args, Debug)]
pub struct AssessCommand {
    /// Python project directory
    #[arg(short, long)]
    pub project: PathBuf,

    /// Output report file
    #[arg(short = 'o', long)]
    pub report: Option<PathBuf>,

    /// Report format (html, json, markdown, pdf)
    #[arg(short, long, default_value = "html")]
    pub format: String,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

impl AssessCommand {
    /// Execute the assessment command
    pub async fn execute(&self) -> Result<()> {
        let format = self.parse_report_format()?;

        println!("🔍 Assessing Python project at: {}", self.project.display());
        println!();

        // Parse the project
        let project = self.parse_project()?;

        println!("📊 Analyzing {} Python modules...", project.modules.len());
        println!();

        // Detect features
        let feature_detector = FeatureDetector::new();
        let mut file_features = HashMap::new();

        for (module_name, module) in &project.modules {
            // Convert portalis_ingest::PythonAst to assessment::PythonAst
            let ast = Self::convert_ast(&module.ast);
            let features = feature_detector.detect(&ast, &module.path.display().to_string());
            file_features.insert(module.path.display().to_string(), features);

            if self.verbose {
                println!("  ✓ {} - {} features", module_name,
                    file_features[&module.path.display().to_string()].summary.total_features);
            }
        }

        // Analyze compatibility
        println!("\n🔬 Analyzing compatibility...");
        let analyzer = CompatibilityAnalyzer::new();
        let compatibility = analyzer.analyze_files(&file_features);

        // Calculate metrics and estimate effort
        println!("📈 Estimating effort...");
        let estimator = EffortEstimator::new();

        let total_loc = self.count_lines_of_code(&project)?;
        let combined_features = self.combine_features(&file_features);
        let metrics = estimator.calculate_metrics(
            &combined_features,
            total_loc,
            compatibility.score.overall,
        );

        let effort = estimator.estimate(&combined_features, &compatibility, metrics);

        // Generate report
        println!("\n📝 Generating report...");
        let report_gen = ReportGenerator::new();
        let report = report_gen.generate(
            &self.get_project_name()?,
            &self.project,
            combined_features,
            compatibility,
            effort,
        );

        // Print summary to console
        self.print_summary(&report);

        // Save report to file
        let output_path = self.get_output_path();
        report_gen.save_report(&report, &output_path, format)
            .context("Failed to save report")?;

        println!("\n✅ Assessment complete!");
        println!("📄 Report saved to: {}", output_path.display());

        Ok(())
    }

    /// Parse the Python project
    fn parse_project(&self) -> Result<PythonProject> {
        let parser = ProjectParser::new();
        parser.parse_project(&self.project)
            .context("Failed to parse Python project")
    }

    /// Count total lines of code
    fn count_lines_of_code(&self, project: &PythonProject) -> Result<usize> {
        let mut total = 0;

        for module in project.modules.values() {
            if let Ok(content) = std::fs::read_to_string(&module.path) {
                total += content.lines()
                    .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
                    .count();
            }
        }

        Ok(total)
    }

    /// Combine all feature sets
    fn combine_features(&self, file_features: &HashMap<String, portalis_core::assessment::FeatureSet>)
        -> portalis_core::assessment::FeatureSet
    {
        use portalis_core::assessment::FeatureSet;
        use portalis_core::assessment::feature_detector::FeatureSummary;

        let mut all_features = Vec::new();

        for features in file_features.values() {
            all_features.extend(features.features.clone());
        }

        let total_features = all_features.len();
        let fully_supported = all_features.iter()
            .filter(|f| f.support == portalis_core::assessment::feature_detector::FeatureSupport::Full)
            .count();
        let partially_supported = all_features.iter()
            .filter(|f| f.support == portalis_core::assessment::feature_detector::FeatureSupport::Partial)
            .count();
        let unsupported = all_features.iter()
            .filter(|f| f.support == portalis_core::assessment::feature_detector::FeatureSupport::None)
            .count();

        let mut by_category = HashMap::new();
        for feature in &all_features {
            *by_category.entry(feature.category.clone()).or_insert(0) += 1;
        }

        FeatureSet {
            features: all_features,
            summary: FeatureSummary {
                total_features,
                fully_supported,
                partially_supported,
                unsupported,
                by_category,
            },
        }
    }

    /// Get project name from path
    fn get_project_name(&self) -> Result<String> {
        Ok(self.project
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string())
    }

    /// Get output path
    fn get_output_path(&self) -> PathBuf {
        if let Some(path) = &self.report {
            path.clone()
        } else {
            let extension = match self.format.as_str() {
                "html" => "html",
                "json" => "json",
                "markdown" | "md" => "md",
                "pdf" => "pdf",
                _ => "html",
            };

            PathBuf::from(format!("portalis-assessment.{}", extension))
        }
    }

    /// Convert portalis_ingest::PythonAst to assessment::PythonAst
    fn convert_ast(ingest_ast: &portalis_ingest::PythonAst) -> PythonAst {
        use portalis_core::assessment::feature_detector::{
            PythonFunction, PythonClass, PythonImport, PythonParameter, PythonAttribute
        };

        let functions = ingest_ast.functions.iter().map(|f| PythonFunction {
            name: f.name.clone(),
            params: f.params.iter().map(|p| PythonParameter {
                name: p.name.clone(),
                type_hint: p.type_hint.clone(),
                default: p.default.clone(),
            }).collect(),
            return_type: f.return_type.clone(),
            body: f.body.clone(),
            decorators: f.decorators.clone(),
        }).collect();

        let classes = ingest_ast.classes.iter().map(|c| PythonClass {
            name: c.name.clone(),
            bases: c.bases.clone(),
            methods: c.methods.iter().map(|m| PythonFunction {
                name: m.name.clone(),
                params: m.params.iter().map(|p| PythonParameter {
                    name: p.name.clone(),
                    type_hint: p.type_hint.clone(),
                    default: p.default.clone(),
                }).collect(),
                return_type: m.return_type.clone(),
                body: m.body.clone(),
                decorators: m.decorators.clone(),
            }).collect(),
            attributes: c.attributes.iter().map(|a| PythonAttribute {
                name: a.name.clone(),
                type_hint: a.type_hint.clone(),
            }).collect(),
        }).collect();

        let imports = ingest_ast.imports.iter().map(|i| PythonImport {
            module: i.module.clone(),
            items: i.items.clone(),
            alias: i.alias.clone(),
        }).collect();

        PythonAst {
            functions,
            classes,
            imports,
        }
    }

    /// Print summary to console
    fn print_summary(&self, report: &portalis_core::assessment::AssessmentReport) {
        println!("\n{}", "=".repeat(80));
        println!("ASSESSMENT SUMMARY");
        println!("{}", "=".repeat(80));

        println!("\n📊 Translatability Score: {:.0}%", report.compatibility.score.overall);

        println!("\n📈 Features:");
        println!("  • Total: {}", report.features.summary.total_features);
        println!("  • Fully Supported: {} ({:.0}%)",
            report.features.summary.fully_supported,
            (report.features.summary.fully_supported as f64 / report.features.summary.total_features as f64 * 100.0));
        println!("  • Partially Supported: {}", report.features.summary.partially_supported);
        println!("  • Unsupported: {}", report.features.summary.unsupported);

        if !report.compatibility.blockers.is_empty() {
            println!("\n⚠️  Critical Blockers: {}", report.compatibility.blockers.len());
            for (i, blocker) in report.compatibility.blockers.iter().take(5).enumerate() {
                println!("  {}. {} ({} occurrences)", i + 1, blocker.feature, blocker.count);
            }
            if report.compatibility.blockers.len() > 5 {
                println!("  ... and {} more", report.compatibility.blockers.len() - 5);
            }
        }

        println!("\n⏱️  Effort Estimate:");
        println!("  • Total: {:.0} hours ({:.1} weeks)",
            report.effort.total_hours,
            report.effort.total_hours / 40.0);
        println!("  • Range: {:.0}-{:.0} hours",
            report.effort.min_hours,
            report.effort.max_hours);
        println!("  • Timeline: {}-{} days",
            report.effort.timeline.days_min,
            report.effort.timeline.days_max);

        println!("\n💡 Recommendation:");
        println!("  {}", report.executive_summary.recommendation);

        println!("\n{}", "=".repeat(80));
    }

    fn parse_report_format(&self) -> Result<ReportFormat> {
        match self.format.to_lowercase().as_str() {
            "html" => Ok(ReportFormat::Html),
            "json" => Ok(ReportFormat::Json),
            "markdown" | "md" => Ok(ReportFormat::Markdown),
            "pdf" => Ok(ReportFormat::Pdf),
            _ => anyhow::bail!("Invalid format: {}. Use: html, json, markdown, or pdf", self.format),
        }
    }
}
