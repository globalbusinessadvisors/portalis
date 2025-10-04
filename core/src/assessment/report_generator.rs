//! Report Generator
//!
//! Generates assessment reports in various formats (HTML, JSON, Markdown, PDF).

use super::compatibility_analyzer::{CompatibilityReport, Blocker, Warning, Recommendation};
use super::effort_estimator::EffortEstimate;
use super::feature_detector::FeatureSet;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Report format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Html,
    Json,
    Markdown,
    Pdf, // Generated from HTML
}

/// Complete assessment report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentReport {
    pub project_name: String,
    pub project_path: String,
    pub timestamp: String,
    pub features: FeatureSet,
    pub compatibility: CompatibilityReport,
    pub effort: EffortEstimate,
    pub executive_summary: ExecutiveSummary,
}

/// Executive summary for non-technical stakeholders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    pub recommendation: String,
    pub translatability: String,
    pub estimated_time: String,
    pub estimated_cost: String,
    pub key_risks: Vec<String>,
    pub key_benefits: Vec<String>,
}

/// Report generator
pub struct ReportGenerator;

impl ReportGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate a complete assessment report
    pub fn generate(
        &self,
        project_name: &str,
        project_path: &Path,
        features: FeatureSet,
        compatibility: CompatibilityReport,
        effort: EffortEstimate,
    ) -> AssessmentReport {
        let timestamp = chrono::Utc::now().to_rfc3339();

        let executive_summary = self.generate_executive_summary(
            &compatibility,
            &effort,
        );

        AssessmentReport {
            project_name: project_name.to_string(),
            project_path: project_path.display().to_string(),
            timestamp,
            features,
            compatibility,
            effort,
            executive_summary,
        }
    }

    /// Generate executive summary
    fn generate_executive_summary(
        &self,
        compatibility: &CompatibilityReport,
        effort: &EffortEstimate,
    ) -> ExecutiveSummary {
        let score = compatibility.score.overall;

        let recommendation = if score >= 90.0 {
            "Highly Recommended - Excellent compatibility"
        } else if score >= 70.0 {
            "Recommended - Good compatibility with minor adjustments"
        } else if score >= 50.0 {
            "Conditional - Requires refactoring before migration"
        } else {
            "Not Recommended - Significant incompatibilities present"
        }.to_string();

        let translatability = format!("{:.0}% of code is translatable", score);

        let weeks = effort.total_hours / 40.0;
        let estimated_time = if weeks >= 4.0 {
            format!("{:.1} months", weeks / 4.0)
        } else {
            format!("{:.1} weeks", weeks)
        };

        // Estimate cost at $150/hour (typical contractor rate)
        let cost = effort.total_hours * 150.0;
        let estimated_cost = if cost >= 10000.0 {
            format!("${:.0}K - ${:.0}K", effort.min_hours * 150.0 / 1000.0, effort.max_hours * 150.0 / 1000.0)
        } else {
            format!("${:.0} - ${:.0}", effort.min_hours * 150.0, effort.max_hours * 150.0)
        };

        let mut key_risks = Vec::new();
        for blocker in compatibility.blockers.iter().take(3) {
            key_risks.push(format!("{} ({})", blocker.feature, blocker.description));
        }
        if key_risks.is_empty() {
            key_risks.push("No significant risks identified".to_string());
        }

        let key_benefits = vec![
            format!("Performance: 2-10x faster execution in WASM"),
            format!("Portability: Run anywhere with WASM support"),
            format!("Security: Sandboxed execution environment"),
            format!("Type Safety: Rust's strong type system"),
        ];

        ExecutiveSummary {
            recommendation,
            translatability,
            estimated_time,
            estimated_cost,
            key_risks,
            key_benefits,
        }
    }

    /// Export report to HTML format
    pub fn to_html(&self, report: &AssessmentReport) -> String {
        let html = format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portalis Assessment Report - {}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin: 30px 0 15px 0;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
        h3 {{
            color: #7f8c8d;
            margin: 20px 0 10px 0;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px 8px 0 0;
            margin: -40px -40px 30px -40px;
        }}
        .header h1 {{
            color: white;
            border: none;
            margin: 0;
        }}
        .meta {{
            color: rgba(255,255,255,0.9);
            margin-top: 10px;
            font-size: 14px;
        }}
        .score-card {{
            display: flex;
            gap: 20px;
            margin: 30px 0;
            flex-wrap: wrap;
        }}
        .score-item {{
            flex: 1;
            min-width: 200px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .score-value {{
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .score-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .exec-summary {{
            background: #ecf0f1;
            padding: 25px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .recommendation {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .blocker {{
            background: #fee;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .warning {{
            background: #fef9e7;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .recommendation-item {{
            background: #e8f5e9;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .priority-high {{
            border-left-color: #e74c3c;
        }}
        .priority-medium {{
            border-left-color: #f39c12;
        }}
        .priority-low {{
            border-left-color: #95a5a6;
        }}
        ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}
        li {{
            margin: 5px 0;
        }}
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .feature-table th, .feature-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        .feature-table th {{
            background: #34495e;
            color: white;
            font-weight: 600;
        }}
        .feature-table tr:hover {{
            background: #f8f9fa;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-full {{
            background: #d4edda;
            color: #155724;
        }}
        .badge-partial {{
            background: #fff3cd;
            color: #856404;
        }}
        .badge-none {{
            background: #f8d7da;
            color: #721c24;
        }}
        .phase {{
            background: white;
            border: 1px solid #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .phase-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Portalis Assessment Report</h1>
            <div class="meta">
                <div>Project: {}</div>
                <div>Path: {}</div>
                <div>Generated: {}</div>
            </div>
        </div>

        <h2>Executive Summary</h2>
        <div class="exec-summary">
            <div class="recommendation">{}</div>
            <p><strong>Translatability:</strong> {}</p>
            <p><strong>Estimated Time:</strong> {}</p>
            <p><strong>Estimated Cost:</strong> {}</p>

            <h3>Key Risks</h3>
            <ul>
                {}
            </ul>

            <h3>Key Benefits</h3>
            <ul>
                {}
            </ul>
        </div>

        <h2>Compatibility Score</h2>
        <div class="score-card">
            <div class="score-item">
                <div class="score-label">Overall Translatability</div>
                <div class="score-value">{:.0}%</div>
            </div>
            <div class="score-item">
                <div class="score-label">Fully Supported Features</div>
                <div class="score-value">{}</div>
            </div>
            <div class="score-item">
                <div class="score-label">Blockers</div>
                <div class="score-value">{}</div>
            </div>
            <div class="score-item">
                <div class="score-label">Estimated Effort</div>
                <div class="score-value">{:.0}h</div>
            </div>
        </div>

        {}

        {}

        {}

        {}

        <div class="footer">
            <p>Generated by Portalis Assessment Tool v1.0</p>
            <p>Python → Rust → WASM Translation Platform</p>
        </div>
    </div>
</body>
</html>"#,
            report.project_name,
            report.project_name,
            report.project_path,
            report.timestamp,
            report.executive_summary.recommendation,
            report.executive_summary.translatability,
            report.executive_summary.estimated_time,
            report.executive_summary.estimated_cost,
            report.executive_summary.key_risks.iter()
                .map(|r| format!("<li>{}</li>", r))
                .collect::<Vec<_>>()
                .join("\n                "),
            report.executive_summary.key_benefits.iter()
                .map(|b| format!("<li>{}</li>", b))
                .collect::<Vec<_>>()
                .join("\n                "),
            report.compatibility.score.overall,
            report.features.summary.fully_supported,
            report.compatibility.blockers.len(),
            report.effort.total_hours,
            self.generate_blockers_html(&report.compatibility.blockers),
            self.generate_warnings_html(&report.compatibility.warnings),
            self.generate_recommendations_html(&report.compatibility.recommendations),
            self.generate_timeline_html(&report.effort),
        );

        html
    }

    fn generate_blockers_html(&self, blockers: &[Blocker]) -> String {
        if blockers.is_empty() {
            return r#"<h2>Translation Blockers</h2>
        <div class="blocker">
            <strong>No blockers found!</strong>
            <p>All detected features are supported or partially supported.</p>
        </div>"#.to_string();
        }

        let mut html = String::from("<h2>Translation Blockers</h2>");
        html.push_str(&format!("<p>Found {} critical issues that must be addressed:</p>", blockers.len()));

        for blocker in blockers {
            html.push_str(&format!(
                r#"<div class="blocker">
            <strong>{}</strong> ({:?} Impact) - {} occurrences
            <p>{}</p>
            {}
        </div>"#,
                blocker.feature,
                blocker.impact,
                blocker.count,
                blocker.description,
                if let Some(workaround) = &blocker.workaround {
                    format!("<p><strong>Workaround:</strong> {}</p>", workaround)
                } else {
                    String::new()
                }
            ));
        }

        html
    }

    fn generate_warnings_html(&self, warnings: &[Warning]) -> String {
        if warnings.is_empty() {
            return String::new();
        }

        let mut html = String::from("<h2>Warnings (Partial Support)</h2>");
        html.push_str(&format!("<p>Found {} features with partial support:</p>", warnings.len()));

        for warning in warnings {
            html.push_str(&format!(
                r#"<div class="warning">
            <strong>{}</strong> - {} occurrences
            <p>{}</p>
            <p><strong>Limitation:</strong> {}</p>
        </div>"#,
                warning.feature,
                warning.count,
                warning.description,
                warning.limitation
            ));
        }

        html
    }

    fn generate_recommendations_html(&self, recommendations: &[Recommendation]) -> String {
        let mut html = String::from("<h2>Recommendations</h2>");

        for rec in recommendations {
            let priority_class = match rec.priority {
                super::compatibility_analyzer::RecommendationPriority::High => "priority-high",
                super::compatibility_analyzer::RecommendationPriority::Medium => "priority-medium",
                super::compatibility_analyzer::RecommendationPriority::Low => "priority-low",
            };

            html.push_str(&format!(
                r#"<div class="recommendation-item {}">
            <strong>{}</strong> ({:?} Priority)
            <p>{}</p>
            <ul>
                {}
            </ul>
        </div>"#,
                priority_class,
                rec.title,
                rec.priority,
                rec.description,
                rec.action_items.iter()
                    .map(|item| format!("<li>{}</li>", item))
                    .collect::<Vec<_>>()
                    .join("\n                ")
            ));
        }

        html
    }

    fn generate_timeline_html(&self, effort: &EffortEstimate) -> String {
        let mut html = String::from("<h2>Migration Timeline</h2>");
        html.push_str(&format!(
            "<p>Expected duration: {} days ({}-{} days range)</p>",
            effort.timeline.days_expected,
            effort.timeline.days_min,
            effort.timeline.days_max
        ));

        for phase in &effort.timeline.phases {
            html.push_str(&format!(
                r#"<div class="phase">
            <div class="phase-header">{} ({} days)</div>
            <p>{}</p>
            <ul>
                {}
            </ul>
        </div>"#,
                phase.name,
                phase.duration_days,
                phase.description,
                phase.deliverables.iter()
                    .map(|d| format!("<li>{}</li>", d))
                    .collect::<Vec<_>>()
                    .join("\n                ")
            ));
        }

        html
    }

    /// Export report to JSON format
    pub fn to_json(&self, report: &AssessmentReport) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(report)
    }

    /// Export report to Markdown format
    pub fn to_markdown(&self, report: &AssessmentReport) -> String {
        let mut md = String::new();

        md.push_str(&format!("# Portalis Assessment Report\n\n"));
        md.push_str(&format!("**Project:** {}\n", report.project_name));
        md.push_str(&format!("**Path:** {}\n", report.project_path));
        md.push_str(&format!("**Generated:** {}\n\n", report.timestamp));

        md.push_str("## Executive Summary\n\n");
        md.push_str(&format!("**Recommendation:** {}\n\n", report.executive_summary.recommendation));
        md.push_str(&format!("- **Translatability:** {}\n", report.executive_summary.translatability));
        md.push_str(&format!("- **Estimated Time:** {}\n", report.executive_summary.estimated_time));
        md.push_str(&format!("- **Estimated Cost:** {}\n\n", report.executive_summary.estimated_cost));

        md.push_str("### Key Risks\n\n");
        for risk in &report.executive_summary.key_risks {
            md.push_str(&format!("- {}\n", risk));
        }

        md.push_str("\n### Key Benefits\n\n");
        for benefit in &report.executive_summary.key_benefits {
            md.push_str(&format!("- {}\n", benefit));
        }

        md.push_str(&format!("\n## Compatibility Score\n\n"));
        md.push_str(&format!("- **Overall:** {:.0}%\n", report.compatibility.score.overall));
        md.push_str(&format!("- **Fully Supported:** {}\n", report.features.summary.fully_supported));
        md.push_str(&format!("- **Partially Supported:** {}\n", report.features.summary.partially_supported));
        md.push_str(&format!("- **Unsupported (Blockers):** {}\n\n", report.features.summary.unsupported));

        if !report.compatibility.blockers.is_empty() {
            md.push_str("## Translation Blockers\n\n");
            for blocker in &report.compatibility.blockers {
                md.push_str(&format!("### {} ({:?} Impact)\n\n", blocker.feature, blocker.impact));
                md.push_str(&format!("- **Count:** {}\n", blocker.count));
                md.push_str(&format!("- **Description:** {}\n", blocker.description));
                if let Some(workaround) = &blocker.workaround {
                    md.push_str(&format!("- **Workaround:** {}\n", workaround));
                }
                md.push_str("\n");
            }
        }

        md.push_str(&format!("## Effort Estimate\n\n"));
        md.push_str(&format!("- **Total:** {:.0} hours\n", report.effort.total_hours));
        md.push_str(&format!("- **Range:** {:.0}-{:.0} hours\n", report.effort.min_hours, report.effort.max_hours));
        md.push_str(&format!("- **Timeline:** {}-{} days\n\n",
            report.effort.timeline.days_min, report.effort.timeline.days_max));

        md.push_str("### Breakdown\n\n");
        md.push_str(&format!("- Analysis: {:.0}h\n", report.effort.breakdown.analysis_hours));
        md.push_str(&format!("- Refactoring: {:.0}h\n", report.effort.breakdown.refactoring_hours));
        md.push_str(&format!("- Translation: {:.0}h\n", report.effort.breakdown.translation_hours));
        md.push_str(&format!("- Testing: {:.0}h\n", report.effort.breakdown.testing_hours));
        md.push_str(&format!("- Integration: {:.0}h\n", report.effort.breakdown.integration_hours));
        md.push_str(&format!("- Documentation: {:.0}h\n\n", report.effort.breakdown.documentation_hours));

        md.push_str("---\n\n");
        md.push_str("*Generated by Portalis Assessment Tool - Python → Rust → WASM Translation Platform*\n");

        md
    }

    /// Save report to file
    pub fn save_report(
        &self,
        report: &AssessmentReport,
        output_path: &Path,
        format: ReportFormat,
    ) -> Result<(), std::io::Error> {
        let content = match format {
            ReportFormat::Html => self.to_html(report),
            ReportFormat::Json => self.to_json(report).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, e)
            })?,
            ReportFormat::Markdown => self.to_markdown(report),
            ReportFormat::Pdf => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "PDF generation requires external tool (wkhtmltopdf). Generate HTML first, then convert."
                ));
            }
        };

        std::fs::write(output_path, content)?;
        Ok(())
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::feature_detector::FeatureSummary;
    use std::collections::HashMap;

    #[test]
    fn test_generate_report() {
        let generator = ReportGenerator::new();
        let features = FeatureSet {
            features: vec![],
            summary: super::super::feature_detector::FeatureSummary {
                total_features: 10,
                fully_supported: 8,
                partially_supported: 2,
                unsupported: 0,
                by_category: HashMap::new(),
            },
        };

        let compatibility = CompatibilityReport {
            score: super::super::compatibility_analyzer::TranslatabilityScore {
                overall: 90.0,
                by_category: HashMap::new(),
                confidence: super::super::compatibility_analyzer::ConfidenceLevel::High,
            },
            blockers: vec![],
            warnings: vec![],
            recommendations: vec![],
            file_analysis: HashMap::new(),
        };

        let effort = EffortEstimate {
            total_hours: 100.0,
            min_hours: 70.0,
            max_hours: 130.0,
            confidence: super::super::effort_estimator::EstimateConfidence::Medium,
            breakdown: super::super::effort_estimator::EffortBreakdown {
                analysis_hours: 20.0,
                refactoring_hours: 10.0,
                translation_hours: 40.0,
                testing_hours: 20.0,
                integration_hours: 5.0,
                documentation_hours: 5.0,
            },
            timeline: super::super::effort_estimator::Timeline {
                days_min: 10,
                days_expected: 15,
                days_max: 20,
                parallel_work: true,
                phases: vec![],
            },
            complexity_metrics: super::super::effort_estimator::ComplexityMetrics {
                total_loc: 1000,
                translatable_loc: 900,
                functions_count: 50,
                classes_count: 10,
                average_function_complexity: 5.0,
                cyclomatic_complexity: None,
                dependency_depth: 3,
            },
        };

        let report = generator.generate(
            "Test Project",
            Path::new("/test/project"),
            features,
            compatibility,
            effort,
        );

        assert_eq!(report.project_name, "Test Project");
        assert!(report.executive_summary.recommendation.contains("Recommended"));
    }

    #[test]
    fn test_html_generation() {
        let generator = ReportGenerator::new();
        let report = create_test_report();
        let html = generator.to_html(&report);

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Project"));
        assert!(html.contains("Portalis Assessment Report"));
    }

    #[test]
    fn test_markdown_generation() {
        let generator = ReportGenerator::new();
        let report = create_test_report();
        let md = generator.to_markdown(&report);

        assert!(md.contains("# Portalis Assessment Report"));
        assert!(md.contains("Test Project"));
    }

    fn create_test_report() -> AssessmentReport {
        AssessmentReport {
            project_name: "Test Project".to_string(),
            project_path: "/test/project".to_string(),
            timestamp: "2025-10-03T00:00:00Z".to_string(),
            features: FeatureSet {
                features: vec![],
                summary: FeatureSummary {
                    total_features: 10,
                    fully_supported: 10,
                    partially_supported: 0,
                    unsupported: 0,
                    by_category: HashMap::new(),
                },
            },
            compatibility: CompatibilityReport {
                score: super::super::compatibility_analyzer::TranslatabilityScore {
                    overall: 100.0,
                    by_category: HashMap::new(),
                    confidence: super::super::compatibility_analyzer::ConfidenceLevel::High,
                },
                blockers: vec![],
                warnings: vec![],
                recommendations: vec![],
                file_analysis: HashMap::new(),
            },
            effort: EffortEstimate {
                total_hours: 100.0,
                min_hours: 70.0,
                max_hours: 130.0,
                confidence: super::super::effort_estimator::EstimateConfidence::High,
                breakdown: super::super::effort_estimator::EffortBreakdown {
                    analysis_hours: 20.0,
                    refactoring_hours: 0.0,
                    translation_hours: 50.0,
                    testing_hours: 20.0,
                    integration_hours: 5.0,
                    documentation_hours: 5.0,
                },
                timeline: super::super::effort_estimator::Timeline {
                    days_min: 10,
                    days_expected: 15,
                    days_max: 20,
                    parallel_work: false,
                    phases: vec![],
                },
                complexity_metrics: super::super::effort_estimator::ComplexityMetrics {
                    total_loc: 1000,
                    translatable_loc: 1000,
                    functions_count: 50,
                    classes_count: 10,
                    average_function_complexity: 5.0,
                    cyclomatic_complexity: None,
                    dependency_depth: 3,
                },
            },
            executive_summary: ExecutiveSummary {
                recommendation: "Highly Recommended".to_string(),
                translatability: "100% translatable".to_string(),
                estimated_time: "2.5 weeks".to_string(),
                estimated_cost: "$10K - $19K".to_string(),
                key_risks: vec!["No significant risks".to_string()],
                key_benefits: vec!["Performance boost".to_string()],
            },
        }
    }
}
