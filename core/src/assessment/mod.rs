//! Assessment Module
//!
//! Tools for assessing Python codebases for Portalis compatibility.

pub mod feature_detector;
pub mod compatibility_analyzer;
pub mod effort_estimator;
pub mod report_generator;

pub use feature_detector::{FeatureDetector, FeatureSet, FeatureSupport, DetectedFeature};
pub use compatibility_analyzer::{CompatibilityAnalyzer, CompatibilityReport, TranslatabilityScore};
pub use effort_estimator::{EffortEstimator, EffortEstimate, ComplexityMetrics};
pub use report_generator::{ReportGenerator, ReportFormat, AssessmentReport};
