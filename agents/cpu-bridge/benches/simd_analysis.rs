//! SIMD Performance Analysis and Speedup Calculator
//!
//! This module provides utilities for analyzing SIMD performance benchmarks,
//! calculating speedup ratios, and generating performance reports.

use std::collections::HashMap;
use std::fmt;

/// Platform-specific performance data
#[derive(Debug, Clone)]
pub struct PlatformPerformance {
    pub platform: String,
    pub simd_type: String,
    pub vector_width_bytes: usize,
    pub operation: String,
    pub throughput_ops_per_sec: f64,
    pub latency_ns: f64,
    pub speedup_vs_scalar: f64,
}

impl PlatformPerformance {
    pub fn new(
        platform: String,
        simd_type: String,
        vector_width_bytes: usize,
        operation: String,
        throughput_ops_per_sec: f64,
        latency_ns: f64,
        speedup_vs_scalar: f64,
    ) -> Self {
        Self {
            platform,
            simd_type,
            vector_width_bytes,
            operation,
            throughput_ops_per_sec,
            latency_ns,
            speedup_vs_scalar,
        }
    }

    /// Calculate efficiency (speedup / ideal_speedup)
    /// Ideal speedup is based on vector width
    pub fn efficiency(&self) -> f64 {
        let ideal_speedup = (self.vector_width_bytes / 8) as f64;
        self.speedup_vs_scalar / ideal_speedup
    }
}

impl fmt::Display for PlatformPerformance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Platform: {} ({})", self.platform, self.simd_type)?;
        writeln!(f, "  Operation: {}", self.operation)?;
        writeln!(f, "  Vector Width: {} bytes", self.vector_width_bytes)?;
        writeln!(f, "  Throughput: {:.2} ops/sec", self.throughput_ops_per_sec)?;
        writeln!(f, "  Latency: {:.2} ns", self.latency_ns)?;
        writeln!(f, "  Speedup: {:.2}x vs scalar", self.speedup_vs_scalar)?;
        writeln!(f, "  Efficiency: {:.1}%", self.efficiency() * 100.0)?;
        Ok(())
    }
}

/// Performance matrix comparing different platforms and operations
#[derive(Debug, Clone)]
pub struct PerformanceMatrix {
    pub results: Vec<PlatformPerformance>,
}

impl PerformanceMatrix {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    pub fn add(&mut self, result: PlatformPerformance) {
        self.results.push(result);
    }

    /// Get all results for a specific operation
    pub fn by_operation(&self, operation: &str) -> Vec<&PlatformPerformance> {
        self.results
            .iter()
            .filter(|r| r.operation == operation)
            .collect()
    }

    /// Get all results for a specific platform
    pub fn by_platform(&self, platform: &str) -> Vec<&PlatformPerformance> {
        self.results
            .iter()
            .filter(|r| r.platform == platform)
            .collect()
    }

    /// Calculate average speedup across all operations
    pub fn average_speedup(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(|r| r.speedup_vs_scalar).sum();
        sum / self.results.len() as f64
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str("=== SIMD Performance Matrix ===\n\n");

        // Group by platform
        let mut platforms: HashMap<String, Vec<&PlatformPerformance>> = HashMap::new();
        for result in &self.results {
            platforms
                .entry(result.platform.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (platform, results) in platforms.iter() {
            report.push_str(&format!("\n{}\n", "=".repeat(60)));
            report.push_str(&format!("Platform: {}\n", platform));
            report.push_str(&format!("{}\n\n", "=".repeat(60)));

            for result in results {
                report.push_str(&format!("{}\n", result));
            }

            // Calculate platform averages
            let avg_speedup: f64 = results.iter().map(|r| r.speedup_vs_scalar).sum::<f64>()
                / results.len() as f64;
            let avg_efficiency: f64 = results.iter().map(|r| r.efficiency()).sum::<f64>()
                / results.len() as f64;

            report.push_str(&format!("Platform Average:\n"));
            report.push_str(&format!("  Speedup: {:.2}x\n", avg_speedup));
            report.push_str(&format!("  Efficiency: {:.1}%\n\n", avg_efficiency * 100.0));
        }

        // Overall summary
        report.push_str(&format!("\n{}\n", "=".repeat(60)));
        report.push_str("Overall Summary\n");
        report.push_str(&format!("{}\n\n", "=".repeat(60)));
        report.push_str(&format!("Total Benchmarks: {}\n", self.results.len()));
        report.push_str(&format!("Average Speedup: {:.2}x\n", self.average_speedup()));

        report
    }

    /// Generate expected vs actual speedup comparison
    pub fn speedup_comparison(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Expected vs Actual Speedup ===\n\n");

        for result in &self.results {
            let ideal = (result.vector_width_bytes / 8) as f64;
            let actual = result.speedup_vs_scalar;
            let efficiency = (actual / ideal) * 100.0;

            report.push_str(&format!(
                "{:<40} | Ideal: {:.1}x | Actual: {:.2}x | Efficiency: {:.1}%\n",
                format!("{} ({})", result.operation, result.simd_type),
                ideal,
                actual,
                efficiency
            ));
        }

        report
    }
}

impl Default for PerformanceMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// Expected performance targets for different SIMD instruction sets
pub struct PerformanceTargets;

impl PerformanceTargets {
    /// Get expected speedup range for AVX2
    pub fn avx2_string_matching() -> (f64, f64) {
        (2.0, 4.0) // Min 2x, Max 4x
    }

    /// Get expected speedup range for AVX2 batch operations
    pub fn avx2_batch_ops() -> (f64, f64) {
        (2.5, 4.0)
    }

    /// Get expected speedup range for NEON
    pub fn neon_string_matching() -> (f64, f64) {
        (2.0, 3.0) // Min 2x, Max 3x
    }

    /// Get expected speedup range for NEON batch operations
    pub fn neon_batch_ops() -> (f64, f64) {
        (2.0, 3.0)
    }

    /// Get expected speedup range for SSE4.2
    pub fn sse42_string_matching() -> (f64, f64) {
        (1.5, 2.5)
    }

    /// Validate if actual speedup meets expectations
    pub fn validate_speedup(simd_type: &str, operation: &str, actual_speedup: f64) -> bool {
        let (min, max) = match (simd_type, operation) {
            ("AVX2", op) if op.contains("string") => Self::avx2_string_matching(),
            ("AVX2", op) if op.contains("batch") => Self::avx2_batch_ops(),
            ("NEON", op) if op.contains("string") => Self::neon_string_matching(),
            ("NEON", op) if op.contains("batch") => Self::neon_batch_ops(),
            ("SSE4.2", op) if op.contains("string") => Self::sse42_string_matching(),
            _ => (1.0, 8.0), // Default range
        };

        actual_speedup >= min && actual_speedup <= max
    }
}

/// Optimization recommendations based on benchmark results
pub struct OptimizationRecommendations {
    recommendations: Vec<String>,
}

impl OptimizationRecommendations {
    pub fn new() -> Self {
        Self {
            recommendations: Vec::new(),
        }
    }

    /// Analyze performance matrix and generate recommendations
    pub fn analyze(matrix: &PerformanceMatrix) -> Self {
        let mut recs = Self::new();

        for result in &matrix.results {
            let efficiency = result.efficiency();

            // Check if efficiency is low
            if efficiency < 0.3 {
                recs.add(format!(
                    "LOW EFFICIENCY: {} on {} has {:.1}% efficiency. Consider optimizing memory alignment or reducing scalar fallback paths.",
                    result.operation, result.platform, efficiency * 100.0
                ));
            }

            // Check if speedup is below expected
            if result.speedup_vs_scalar < 1.5 {
                recs.add(format!(
                    "LOW SPEEDUP: {} on {} shows only {:.2}x speedup. Investigate if SIMD is being fully utilized.",
                    result.operation, result.platform, result.speedup_vs_scalar
                ));
            }

            // Check for excellent performance to identify optimization patterns
            if efficiency > 0.7 && result.speedup_vs_scalar > 3.0 {
                recs.add(format!(
                    "EXCELLENT: {} on {} achieves {:.2}x speedup with {:.1}% efficiency. This pattern should be applied to similar operations.",
                    result.operation, result.platform, result.speedup_vs_scalar, efficiency * 100.0
                ));
            }
        }

        if recs.recommendations.is_empty() {
            recs.add("All operations meet expected performance targets.".to_string());
        }

        recs
    }

    fn add(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }

    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Optimization Recommendations ===\n\n");

        for (i, rec) in self.recommendations.iter().enumerate() {
            report.push_str(&format!("{}. {}\n\n", i + 1, rec));
        }

        report
    }
}

impl Default for OptimizationRecommendations {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate sample performance data for demonstration
pub fn generate_sample_data() -> PerformanceMatrix {
    let mut matrix = PerformanceMatrix::new();

    // x86_64 with AVX2
    matrix.add(PlatformPerformance::new(
        "x86_64 Linux".to_string(),
        "AVX2".to_string(),
        32,
        "String Matching".to_string(),
        1_500_000.0,
        650.0,
        3.2,
    ));

    matrix.add(PlatformPerformance::new(
        "x86_64 Linux".to_string(),
        "AVX2".to_string(),
        32,
        "Batch String Operations".to_string(),
        2_000_000.0,
        500.0,
        3.8,
    ));

    matrix.add(PlatformPerformance::new(
        "x86_64 Linux".to_string(),
        "AVX2".to_string(),
        32,
        "Character Counting".to_string(),
        3_500_000.0,
        285.0,
        4.2,
    ));

    // x86_64 with SSE4.2
    matrix.add(PlatformPerformance::new(
        "x86_64 Linux".to_string(),
        "SSE4.2".to_string(),
        16,
        "String Matching".to_string(),
        850_000.0,
        1175.0,
        1.8,
    ));

    // ARM64 with NEON
    matrix.add(PlatformPerformance::new(
        "ARM64 Linux".to_string(),
        "NEON".to_string(),
        16,
        "String Matching".to_string(),
        1_200_000.0,
        830.0,
        2.5,
    ));

    matrix.add(PlatformPerformance::new(
        "ARM64 Linux".to_string(),
        "NEON".to_string(),
        16,
        "Batch String Operations".to_string(),
        1_500_000.0,
        665.0,
        2.8,
    ));

    // Scalar baseline
    matrix.add(PlatformPerformance::new(
        "Any Platform".to_string(),
        "Scalar".to_string(),
        1,
        "String Matching".to_string(),
        470_000.0,
        2125.0,
        1.0,
    ));

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_performance() {
        let perf = PlatformPerformance::new(
            "x86_64".to_string(),
            "AVX2".to_string(),
            32,
            "String Match".to_string(),
            1_000_000.0,
            1000.0,
            3.2,
        );

        assert_eq!(perf.platform, "x86_64");
        assert_eq!(perf.speedup_vs_scalar, 3.2);
        assert!(perf.efficiency() > 0.0);
    }

    #[test]
    fn test_performance_matrix() {
        let mut matrix = PerformanceMatrix::new();
        matrix.add(PlatformPerformance::new(
            "x86_64".to_string(),
            "AVX2".to_string(),
            32,
            "Op1".to_string(),
            1000.0,
            100.0,
            2.0,
        ));
        matrix.add(PlatformPerformance::new(
            "x86_64".to_string(),
            "AVX2".to_string(),
            32,
            "Op2".to_string(),
            2000.0,
            50.0,
            4.0,
        ));

        assert_eq!(matrix.results.len(), 2);
        assert_eq!(matrix.average_speedup(), 3.0);
    }

    #[test]
    fn test_performance_targets() {
        assert!(PerformanceTargets::validate_speedup("AVX2", "string_matching", 3.0));
        assert!(!PerformanceTargets::validate_speedup("AVX2", "string_matching", 1.0));
    }

    #[test]
    fn test_optimization_recommendations() {
        let matrix = generate_sample_data();
        let recs = OptimizationRecommendations::analyze(&matrix);
        assert!(!recs.recommendations.is_empty());
    }

    #[test]
    fn test_generate_reports() {
        let matrix = generate_sample_data();
        let summary = matrix.summary();
        assert!(summary.contains("SIMD Performance Matrix"));

        let comparison = matrix.speedup_comparison();
        assert!(comparison.contains("Expected vs Actual"));
    }
}
