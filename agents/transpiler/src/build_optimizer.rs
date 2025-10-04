//! Build Optimizer - WASM binary size optimization and analysis
//!
//! This module provides:
//! 1. Cargo.toml profile generation for size optimization
//! 2. Build configuration analysis and recommendations
//! 3. Size estimation and tracking
//! 4. wasm-opt integration commands
//! 5. Dependency impact analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Build optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization (debug builds)
    None,
    /// Basic optimization (opt-level = 1)
    Basic,
    /// Standard optimization (opt-level = 2)
    Standard,
    /// Performance optimization (opt-level = 3)
    Performance,
    /// Size optimization (opt-level = "s")
    Size,
    /// Aggressive size optimization (opt-level = "z")
    AggressiveSize,
}

impl OptimizationLevel {
    /// Get the cargo opt-level string
    pub fn as_opt_level(&self) -> &str {
        match self {
            Self::None => "0",
            Self::Basic => "1",
            Self::Standard => "2",
            Self::Performance => "3",
            Self::Size => "\"s\"",
            Self::AggressiveSize => "\"z\"",
        }
    }

    /// Get expected size reduction percentage
    pub fn size_reduction_estimate(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Basic => 10.0,
            Self::Standard => 20.0,
            Self::Performance => 15.0,
            Self::Size => 40.0,
            Self::AggressiveSize => 50.0,
        }
    }
}

/// Build optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildOptimization {
    /// Optimization level
    pub opt_level: OptimizationLevel,
    /// Enable Link Time Optimization
    pub lto: LtoSetting,
    /// Number of codegen units (1 = best optimization, slower build)
    pub codegen_units: Option<u32>,
    /// Panic strategy (abort = smaller binary)
    pub panic: PanicStrategy,
    /// Strip debug symbols
    pub strip: StripSetting,
    /// Enable incremental compilation
    pub incremental: bool,
    /// Additional rustflags
    pub rustflags: Vec<String>,
}

/// LTO (Link Time Optimization) setting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LtoSetting {
    /// No LTO
    Off,
    /// Thin LTO (faster, good optimization)
    Thin,
    /// Fat LTO (slower, best optimization)
    Fat,
}

impl LtoSetting {
    pub fn as_toml_value(&self) -> &str {
        match self {
            Self::Off => "false",
            Self::Thin => "\"thin\"",
            Self::Fat => "true",
        }
    }

    pub fn size_reduction_estimate(&self) -> f64 {
        match self {
            Self::Off => 0.0,
            Self::Thin => 15.0,
            Self::Fat => 25.0,
        }
    }
}

/// Panic strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanicStrategy {
    /// Unwind (default, larger binary)
    Unwind,
    /// Abort (smaller binary)
    Abort,
}

impl PanicStrategy {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Unwind => "unwind",
            Self::Abort => "abort",
        }
    }

    pub fn size_reduction_estimate(&self) -> f64 {
        match self {
            Self::Unwind => 0.0,
            Self::Abort => 10.0,
        }
    }
}

/// Strip setting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StripSetting {
    /// No stripping
    None,
    /// Strip debug symbols
    Debuginfo,
    /// Strip everything
    Symbols,
}

impl StripSetting {
    pub fn as_toml_value(&self) -> &str {
        match self {
            Self::None => "false",
            Self::Debuginfo => "\"debuginfo\"",
            Self::Symbols => "true",
        }
    }

    pub fn size_reduction_estimate(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Debuginfo => 20.0,
            Self::Symbols => 30.0,
        }
    }
}

impl Default for BuildOptimization {
    fn default() -> Self {
        Self {
            opt_level: OptimizationLevel::Standard,
            lto: LtoSetting::Off,
            codegen_units: None,
            panic: PanicStrategy::Unwind,
            strip: StripSetting::None,
            incremental: true,
            rustflags: vec![],
        }
    }
}

impl BuildOptimization {
    /// Create size-optimized configuration
    pub fn for_size() -> Self {
        Self {
            opt_level: OptimizationLevel::AggressiveSize,
            lto: LtoSetting::Fat,
            codegen_units: Some(1),
            panic: PanicStrategy::Abort,
            strip: StripSetting::Symbols,
            incremental: false,
            rustflags: vec![
                "-C".to_string(),
                "link-arg=-s".to_string(),
            ],
        }
    }

    /// Create performance-optimized configuration
    pub fn for_performance() -> Self {
        Self {
            opt_level: OptimizationLevel::Performance,
            lto: LtoSetting::Thin,
            codegen_units: Some(16),
            panic: PanicStrategy::Unwind,
            strip: StripSetting::Debuginfo,
            incremental: true,
            rustflags: vec![],
        }
    }

    /// Generate Cargo.toml profile section
    pub fn to_cargo_profile(&self, profile_name: &str) -> String {
        let mut output = String::new();

        output.push_str(&format!("[profile.{}]\n", profile_name));
        output.push_str(&format!("opt-level = {}\n", self.opt_level.as_opt_level()));
        output.push_str(&format!("lto = {}\n", self.lto.as_toml_value()));

        if let Some(units) = self.codegen_units {
            output.push_str(&format!("codegen-units = {}\n", units));
        }

        output.push_str(&format!("panic = \"{}\"\n", self.panic.as_str()));
        output.push_str(&format!("strip = {}\n", self.strip.as_toml_value()));
        output.push_str(&format!("incremental = {}\n", self.incremental));

        output
    }

    /// Generate .cargo/config.toml rustflags section
    pub fn to_cargo_config(&self) -> String {
        if self.rustflags.is_empty() {
            return String::new();
        }

        let mut output = String::new();
        output.push_str("[target.wasm32-unknown-unknown]\n");
        output.push_str("rustflags = [\n");

        for flag in &self.rustflags {
            output.push_str(&format!("    \"{}\",\n", flag));
        }

        output.push_str("]\n");
        output
    }

    /// Estimate total size reduction
    pub fn estimate_size_reduction(&self) -> f64 {
        let mut reduction = 0.0;

        reduction += self.opt_level.size_reduction_estimate();
        reduction += self.lto.size_reduction_estimate();
        reduction += self.panic.size_reduction_estimate();
        reduction += self.strip.size_reduction_estimate();

        // Cap at 85% reduction (realistic maximum)
        reduction.min(85.0)
    }
}

/// WASM optimization tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmOptConfig {
    /// Optimization level (0-4)
    pub level: u8,
    /// Enable size optimizations
    pub shrink_level: u8,
    /// Enable all optimizations
    pub all_features: bool,
    /// Additional passes
    pub passes: Vec<String>,
}

impl Default for WasmOptConfig {
    fn default() -> Self {
        Self {
            level: 3,
            shrink_level: 2,
            all_features: false,
            passes: vec![],
        }
    }
}

impl WasmOptConfig {
    /// Create aggressive size optimization config
    pub fn for_size() -> Self {
        Self {
            level: 4,
            shrink_level: 2,
            all_features: true,
            passes: vec![
                "strip-debug".to_string(),
                "strip-producers".to_string(),
                "strip-target-features".to_string(),
            ],
        }
    }

    /// Generate wasm-opt command
    pub fn to_command(&self, input: &str, output: &str) -> String {
        let mut cmd = String::from("wasm-opt");

        cmd.push_str(&format!(" -O{}", self.level));

        if self.shrink_level > 0 {
            cmd.push_str(&format!(" -O{}", "z".repeat(self.shrink_level as usize)));
        }

        if self.all_features {
            cmd.push_str(" --all-features");
        }

        for pass in &self.passes {
            cmd.push_str(&format!(" --{}", pass));
        }

        cmd.push_str(&format!(" {} -o {}", input, output));
        cmd
    }

    /// Estimate size reduction
    pub fn estimate_size_reduction(&self) -> f64 {
        let base = match self.level {
            0 => 0.0,
            1 => 10.0,
            2 => 20.0,
            3 => 30.0,
            4 => 40.0,
            _ => 40.0,
        };

        let shrink = self.shrink_level as f64 * 5.0;
        let passes = self.passes.len() as f64 * 2.0;

        (base + shrink + passes).min(50.0)
    }
}

/// Build size analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildSizeAnalysis {
    /// Original size in bytes
    pub original_size: u64,
    /// After cargo optimization
    pub after_cargo_opt: u64,
    /// After wasm-opt
    pub after_wasm_opt: Option<u64>,
    /// After gzip compression
    pub after_gzip: Option<u64>,
    /// Dependency contributions
    pub dependency_sizes: HashMap<String, u64>,
}

impl BuildSizeAnalysis {
    /// Calculate total reduction
    pub fn total_reduction_percent(&self) -> f64 {
        let final_size = self.final_size();
        if self.original_size == 0 {
            return 0.0;
        }

        ((self.original_size - final_size) as f64 / self.original_size as f64) * 100.0
    }

    /// Get final size
    pub fn final_size(&self) -> u64 {
        self.after_gzip
            .or(self.after_wasm_opt)
            .unwrap_or(self.after_cargo_opt)
    }

    /// Format size as human-readable
    pub fn format_size(bytes: u64) -> String {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        }
    }

    /// Generate size report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Build Size Analysis ===\n\n");
        report.push_str(&format!("Original size:        {}\n", Self::format_size(self.original_size)));
        report.push_str(&format!("After cargo opt:      {} ({:.1}% reduction)\n",
            Self::format_size(self.after_cargo_opt),
            self.cargo_reduction_percent()
        ));

        if let Some(wasm_opt_size) = self.after_wasm_opt {
            report.push_str(&format!("After wasm-opt:       {} ({:.1}% reduction)\n",
                Self::format_size(wasm_opt_size),
                self.wasm_opt_reduction_percent()
            ));
        }

        if let Some(gzip_size) = self.after_gzip {
            report.push_str(&format!("After gzip:           {} ({:.1}% reduction)\n",
                Self::format_size(gzip_size),
                self.gzip_reduction_percent()
            ));
        }

        report.push_str(&format!("\nTotal reduction:      {:.1}%\n", self.total_reduction_percent()));
        report.push_str(&format!("Final size:           {}\n", Self::format_size(self.final_size())));

        if !self.dependency_sizes.is_empty() {
            report.push_str("\n=== Dependency Size Contributions ===\n\n");
            let mut deps: Vec<_> = self.dependency_sizes.iter().collect();
            deps.sort_by(|a, b| b.1.cmp(a.1));

            for (dep, size) in deps.iter().take(10) {
                report.push_str(&format!("  {} - {}\n", dep, Self::format_size(**size)));
            }
        }

        report
    }

    fn cargo_reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        ((self.original_size - self.after_cargo_opt) as f64 / self.original_size as f64) * 100.0
    }

    fn wasm_opt_reduction_percent(&self) -> f64 {
        if let Some(wasm_opt_size) = self.after_wasm_opt {
            if self.after_cargo_opt == 0 {
                return 0.0;
            }
            ((self.after_cargo_opt - wasm_opt_size) as f64 / self.after_cargo_opt as f64) * 100.0
        } else {
            0.0
        }
    }

    fn gzip_reduction_percent(&self) -> f64 {
        if let Some(gzip_size) = self.after_gzip {
            let base = self.after_wasm_opt.unwrap_or(self.after_cargo_opt);
            if base == 0 {
                return 0.0;
            }
            ((base - gzip_size) as f64 / base as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Build size estimator
pub struct BuildSizeEstimator;

impl BuildSizeEstimator {
    /// Estimate WASM binary size based on dependencies
    pub fn estimate_size(num_dependencies: usize, has_async: bool, has_networking: bool) -> u64 {
        // Base size: ~50 KB
        let mut size: u64 = 50 * 1024;

        // Each dependency: ~20 KB average
        size += (num_dependencies * 20 * 1024) as u64;

        // Async runtime: +100 KB
        if has_async {
            size += 100 * 1024;
        }

        // Networking: +150 KB
        if has_networking {
            size += 150 * 1024;
        }

        size
    }

    /// Estimate size after optimizations
    pub fn estimate_optimized_size(
        original: u64,
        cargo_opt: &BuildOptimization,
        wasm_opt: Option<&WasmOptConfig>,
        use_gzip: bool,
    ) -> BuildSizeAnalysis {
        let cargo_reduction = cargo_opt.estimate_size_reduction() / 100.0;
        let after_cargo = (original as f64 * (1.0 - cargo_reduction)) as u64;

        let after_wasm = if let Some(wasm_config) = wasm_opt {
            let wasm_reduction = wasm_config.estimate_size_reduction() / 100.0;
            Some((after_cargo as f64 * (1.0 - wasm_reduction)) as u64)
        } else {
            None
        };

        let after_gzip = if use_gzip {
            let base = after_wasm.unwrap_or(after_cargo);
            // Gzip typically achieves 60-75% compression on WASM
            Some((base as f64 * 0.30) as u64)
        } else {
            None
        };

        BuildSizeAnalysis {
            original_size: original,
            after_cargo_opt: after_cargo,
            after_wasm_opt: after_wasm,
            after_gzip,
            dependency_sizes: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_levels() {
        assert_eq!(OptimizationLevel::None.as_opt_level(), "0");
        assert_eq!(OptimizationLevel::AggressiveSize.as_opt_level(), "\"z\"");
    }

    #[test]
    fn test_size_optimization_profile() {
        let opt = BuildOptimization::for_size();
        let profile = opt.to_cargo_profile("release");

        assert!(profile.contains("opt-level = \"z\""));
        assert!(profile.contains("lto = true"));
        assert!(profile.contains("codegen-units = 1"));
        assert!(profile.contains("panic = \"abort\""));
        assert!(profile.contains("strip = true"));
    }

    #[test]
    fn test_wasm_opt_command() {
        let config = WasmOptConfig::for_size();
        let cmd = config.to_command("input.wasm", "output.wasm");

        assert!(cmd.contains("wasm-opt"));
        assert!(cmd.contains("-O4"));
        assert!(cmd.contains("-Ozz"));
        assert!(cmd.contains("input.wasm"));
        assert!(cmd.contains("output.wasm"));
    }

    #[test]
    fn test_size_estimation() {
        let original = 10 * 1024 * 1024; // 10 MB
        let cargo_opt = BuildOptimization::for_size();
        let wasm_opt = WasmOptConfig::for_size();

        let analysis = BuildSizeEstimator::estimate_optimized_size(
            original,
            &cargo_opt,
            Some(&wasm_opt),
            true,
        );

        // Should be significantly reduced
        assert!(analysis.final_size() < original / 2);
        assert!(analysis.total_reduction_percent() > 50.0);
    }

    #[test]
    fn test_size_formatting() {
        assert_eq!(BuildSizeAnalysis::format_size(512), "512 B");
        assert_eq!(BuildSizeAnalysis::format_size(1024), "1.0 KB");
        assert_eq!(BuildSizeAnalysis::format_size(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_build_size_estimator() {
        let size = BuildSizeEstimator::estimate_size(10, true, true);

        // Base (50KB) + 10 deps (200KB) + async (100KB) + network (150KB) = 500KB
        assert_eq!(size, 500 * 1024);
    }

    #[test]
    fn test_optimization_reduction_estimates() {
        let opt = BuildOptimization::for_size();
        let reduction = opt.estimate_size_reduction();

        // Should estimate significant reduction
        assert!(reduction > 50.0);
        assert!(reduction <= 85.0); // Capped at 85%
    }
}
