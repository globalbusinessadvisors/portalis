//! Effort Estimation Engine
//!
//! Estimates migration effort based on code complexity and features.

use super::compatibility_analyzer::{CompatibilityReport, BlockerImpact};
use super::feature_detector::{FeatureSet, FeatureCategory};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Effort estimate for migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffortEstimate {
    pub total_hours: f64,
    pub min_hours: f64,
    pub max_hours: f64,
    pub confidence: EstimateConfidence,
    pub breakdown: EffortBreakdown,
    pub timeline: Timeline,
    pub complexity_metrics: ComplexityMetrics,
}

/// Confidence in the estimate
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimateConfidence {
    High,    // Based on comprehensive analysis
    Medium,  // Based on partial information
    Low,     // Rough estimate only
}

/// Breakdown of effort by phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffortBreakdown {
    pub analysis_hours: f64,
    pub refactoring_hours: f64,
    pub translation_hours: f64,
    pub testing_hours: f64,
    pub integration_hours: f64,
    pub documentation_hours: f64,
}

/// Timeline estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    pub days_min: u32,
    pub days_expected: u32,
    pub days_max: u32,
    pub parallel_work: bool,
    pub phases: Vec<Phase>,
}

/// Migration phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase {
    pub name: String,
    pub duration_days: u32,
    pub description: String,
    pub deliverables: Vec<String>,
}

/// Code complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub total_loc: usize,
    pub translatable_loc: usize,
    pub functions_count: usize,
    pub classes_count: usize,
    pub average_function_complexity: f64,
    pub cyclomatic_complexity: Option<u32>,
    pub dependency_depth: usize,
}

/// Effort estimator
pub struct EffortEstimator {
    /// Hours per line of code (base rate)
    hours_per_loc: f64,
    /// Multiplier for complex code
    complexity_multiplier: f64,
    /// Base hours for setup and teardown
    base_overhead_hours: f64,
}

impl EffortEstimator {
    pub fn new() -> Self {
        Self {
            hours_per_loc: 0.05,           // 3 minutes per LOC on average
            complexity_multiplier: 1.5,     // 50% more for complex code
            base_overhead_hours: 40.0,      // 1 week overhead
        }
    }

    /// Estimate effort for a migration
    pub fn estimate(
        &self,
        features: &FeatureSet,
        compatibility: &CompatibilityReport,
        metrics: ComplexityMetrics,
    ) -> EffortEstimate {
        // Calculate base translation effort
        let translation_hours = self.estimate_translation_effort(&metrics, compatibility);

        // Calculate refactoring effort (for blockers)
        let refactoring_hours = self.estimate_refactoring_effort(&compatibility.blockers);

        // Calculate testing effort (proportional to code size)
        let testing_hours = translation_hours * 0.5; // 50% of translation time

        // Calculate integration effort
        let integration_hours = self.base_overhead_hours + (metrics.dependency_depth as f64 * 8.0);

        // Calculate analysis and documentation
        let analysis_hours = 20.0 + (metrics.total_loc as f64 / 1000.0 * 2.0);
        let documentation_hours = 16.0 + (metrics.classes_count as f64 * 0.5);

        let breakdown = EffortBreakdown {
            analysis_hours,
            refactoring_hours,
            translation_hours,
            testing_hours,
            integration_hours,
            documentation_hours,
        };

        let total_hours = analysis_hours
            + refactoring_hours
            + translation_hours
            + testing_hours
            + integration_hours
            + documentation_hours;

        // Add uncertainty bounds (±30%)
        let min_hours = total_hours * 0.7;
        let max_hours = total_hours * 1.3;

        // Determine confidence
        let confidence = if metrics.total_loc > 1000 && features.summary.total_features > 100 {
            EstimateConfidence::High
        } else if metrics.total_loc > 100 {
            EstimateConfidence::Medium
        } else {
            EstimateConfidence::Low
        };

        // Generate timeline
        let timeline = self.generate_timeline(total_hours, &breakdown);

        EffortEstimate {
            total_hours,
            min_hours,
            max_hours,
            confidence,
            breakdown,
            timeline,
            complexity_metrics: metrics,
        }
    }

    /// Estimate translation effort
    fn estimate_translation_effort(
        &self,
        metrics: &ComplexityMetrics,
        compatibility: &CompatibilityReport,
    ) -> f64 {
        let base_hours = metrics.translatable_loc as f64 * self.hours_per_loc;

        // Adjust for complexity
        let complexity_factor = if metrics.average_function_complexity > 10.0 {
            self.complexity_multiplier
        } else if metrics.average_function_complexity > 5.0 {
            1.2
        } else {
            1.0
        };

        // Adjust for compatibility score
        let compatibility_factor = if compatibility.score.overall < 70.0 {
            1.3 // 30% more time for low compatibility
        } else if compatibility.score.overall < 85.0 {
            1.1
        } else {
            1.0
        };

        base_hours * complexity_factor * compatibility_factor
    }

    /// Estimate refactoring effort
    fn estimate_refactoring_effort(&self, blockers: &[super::compatibility_analyzer::Blocker]) -> f64 {
        let mut total_hours = 0.0;

        for blocker in blockers {
            let hours_per_instance = match blocker.impact {
                BlockerImpact::Critical => 8.0,  // 1 day per instance
                BlockerImpact::High => 4.0,      // Half day
                BlockerImpact::Medium => 2.0,    // 2 hours
                BlockerImpact::Low => 0.5,       // 30 minutes
            };

            total_hours += hours_per_instance * blocker.count as f64;
        }

        total_hours
    }

    /// Generate timeline from effort estimate
    fn generate_timeline(&self, total_hours: f64, breakdown: &EffortBreakdown) -> Timeline {
        // Assume 6 productive hours per day, single developer
        let hours_per_day = 6.0;

        // Sequential phases
        let days_expected = (total_hours / hours_per_day).ceil() as u32;
        let days_min = ((total_hours * 0.7) / hours_per_day).ceil() as u32;
        let days_max = ((total_hours * 1.3) / hours_per_day).ceil() as u32;

        // With parallel work (testing while translating), can reduce by 20%
        let parallel_work = total_hours > 100.0; // Only beneficial for larger projects

        let mut phases = Vec::new();

        // Phase 1: Analysis & Planning
        phases.push(Phase {
            name: "Analysis & Planning".to_string(),
            duration_days: (breakdown.analysis_hours / hours_per_day).ceil() as u32,
            description: "Analyze codebase, identify blockers, create migration plan".to_string(),
            deliverables: vec![
                "Compatibility assessment report".to_string(),
                "Migration strategy document".to_string(),
                "Risk assessment".to_string(),
            ],
        });

        // Phase 2: Refactoring (if needed)
        if breakdown.refactoring_hours > 0.0 {
            phases.push(Phase {
                name: "Refactoring".to_string(),
                duration_days: (breakdown.refactoring_hours / hours_per_day).ceil() as u32,
                description: "Remove blockers and refactor incompatible code patterns".to_string(),
                deliverables: vec![
                    "Refactored Python codebase".to_string(),
                    "Updated tests".to_string(),
                ],
            });
        }

        // Phase 3: Translation
        phases.push(Phase {
            name: "Translation".to_string(),
            duration_days: (breakdown.translation_hours / hours_per_day).ceil() as u32,
            description: "Translate Python code to Rust and compile to WASM".to_string(),
            deliverables: vec![
                "Translated Rust code".to_string(),
                "WASM modules".to_string(),
                "Build system".to_string(),
            ],
        });

        // Phase 4: Testing & Validation
        phases.push(Phase {
            name: "Testing & Validation".to_string(),
            duration_days: (breakdown.testing_hours / hours_per_day).ceil() as u32,
            description: "Validate correctness and performance of translated code".to_string(),
            deliverables: vec![
                "Test suite results".to_string(),
                "Performance benchmarks".to_string(),
                "Bug fixes".to_string(),
            ],
        });

        // Phase 5: Integration
        phases.push(Phase {
            name: "Integration".to_string(),
            duration_days: (breakdown.integration_hours / hours_per_day).ceil() as u32,
            description: "Integrate WASM modules into production environment".to_string(),
            deliverables: vec![
                "Deployment scripts".to_string(),
                "Integration tests".to_string(),
                "Monitoring setup".to_string(),
            ],
        });

        // Phase 6: Documentation
        phases.push(Phase {
            name: "Documentation".to_string(),
            duration_days: (breakdown.documentation_hours / hours_per_day).ceil() as u32,
            description: "Create comprehensive documentation for migrated system".to_string(),
            deliverables: vec![
                "API documentation".to_string(),
                "Migration guide".to_string(),
                "Troubleshooting guide".to_string(),
            ],
        });

        Timeline {
            days_min,
            days_expected,
            days_max,
            parallel_work,
            phases,
        }
    }

    /// Calculate complexity metrics from features
    pub fn calculate_metrics(
        &self,
        features: &FeatureSet,
        total_loc: usize,
        translatable_percentage: f64,
    ) -> ComplexityMetrics {
        let functions_count = features.features.iter()
            .filter(|f| f.category == FeatureCategory::Function)
            .map(|f| f.count)
            .sum();

        let classes_count = features.features.iter()
            .filter(|f| f.category == FeatureCategory::Class)
            .map(|f| f.count)
            .sum();

        // Estimate average function complexity (simplified)
        let average_function_complexity = if functions_count > 0 {
            (total_loc as f64) / (functions_count as f64)
        } else {
            5.0
        };

        // Estimate dependency depth (simplified - would need actual graph analysis)
        let dependency_depth = (classes_count as f64).sqrt().ceil() as usize;

        ComplexityMetrics {
            total_loc,
            translatable_loc: (total_loc as f64 * translatable_percentage / 100.0) as usize,
            functions_count,
            classes_count,
            average_function_complexity,
            cyclomatic_complexity: None, // Would need deeper AST analysis
            dependency_depth,
        }
    }

    /// Format estimate as human-readable string
    pub fn format_estimate(&self, estimate: &EffortEstimate) -> String {
        let weeks = estimate.total_hours / 40.0;
        let months = weeks / 4.0;

        let time_str = if months >= 1.0 {
            format!("{:.1} months ({:.0} hours)", months, estimate.total_hours)
        } else if weeks >= 1.0 {
            format!("{:.1} weeks ({:.0} hours)", weeks, estimate.total_hours)
        } else {
            format!("{:.0} hours", estimate.total_hours)
        };

        format!(
            "Estimated effort: {} (range: {:.0}-{:.0} hours)\nConfidence: {:?}\nTimeline: {}-{} days",
            time_str,
            estimate.min_hours,
            estimate.max_hours,
            estimate.confidence,
            estimate.timeline.days_min,
            estimate.timeline.days_max
        )
    }
}

impl Default for EffortEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::compatibility_analyzer::{TranslatabilityScore, ConfidenceLevel};
    use crate::assessment::feature_detector::FeatureSummary;

    #[test]
    fn test_estimate_simple_project() {
        let estimator = EffortEstimator::new();

        let features = FeatureSet {
            features: vec![],
            summary: FeatureSummary {
                total_features: 10,
                fully_supported: 10,
                partially_supported: 0,
                unsupported: 0,
                by_category: HashMap::new(),
            },
        };

        let compatibility = CompatibilityReport {
            score: TranslatabilityScore {
                overall: 100.0,
                by_category: HashMap::new(),
                confidence: ConfidenceLevel::High,
            },
            blockers: vec![],
            warnings: vec![],
            recommendations: vec![],
            file_analysis: HashMap::new(),
        };

        let metrics = ComplexityMetrics {
            total_loc: 1000,
            translatable_loc: 1000,
            functions_count: 50,
            classes_count: 10,
            average_function_complexity: 5.0,
            cyclomatic_complexity: None,
            dependency_depth: 3,
        };

        let estimate = estimator.estimate(&features, &compatibility, metrics);

        assert!(estimate.total_hours > 0.0);
        assert!(estimate.min_hours < estimate.total_hours);
        assert!(estimate.max_hours > estimate.total_hours);
        assert!(estimate.timeline.days_expected > 0);
    }

    #[test]
    fn test_complexity_metrics_calculation() {
        let estimator = EffortEstimator::new();

        let features = FeatureSet {
            features: vec![],
            summary: FeatureSummary {
                total_features: 20,
                fully_supported: 15,
                partially_supported: 5,
                unsupported: 0,
                by_category: {
                    let mut map = HashMap::new();
                    map.insert(FeatureCategory::Function, 15);
                    map.insert(FeatureCategory::Class, 5);
                    map
                },
            },
        };

        let metrics = estimator.calculate_metrics(&features, 1000, 90.0);

        assert_eq!(metrics.total_loc, 1000);
        assert_eq!(metrics.translatable_loc, 900);
    }
}
