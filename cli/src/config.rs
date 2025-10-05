//! Configuration file support for Portalis CLI

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub translation: TranslationConfig,

    #[serde(default)]
    pub optimization: OptimizationConfig,

    #[serde(default)]
    pub gpu: GpuConfig,

    #[serde(default)]
    pub services: ServicesConfig,

    #[serde(default)]
    pub output: OutputConfig,

    #[serde(default)]
    pub logging: LoggingConfig,

    #[serde(default)]
    pub cache: CacheConfig,

    #[serde(default)]
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    #[serde(default = "default_mode")]
    pub mode: String,

    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[serde(default = "default_true")]
    pub include_metrics: bool,

    #[serde(default = "default_true")]
    pub run_tests: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    #[serde(default = "default_opt_level")]
    pub opt_level: String,

    #[serde(default = "default_true")]
    pub strip_debug: bool,

    #[serde(default = "default_true")]
    pub lto: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default)]
    pub cuda_device: u32,

    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[serde(default = "default_memory_limit")]
    pub memory_limit_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicesConfig {
    pub nemo_url: Option<String>,
    pub triton_url: Option<String>,
    pub dgx_cloud_url: Option<String>,

    #[serde(default)]
    pub nemo: NemoConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NemoConfig {
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    #[serde(default = "default_retry")]
    pub retry_attempts: u32,

    #[serde(default = "default_retry_delay")]
    pub retry_delay_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "default_wasm_dir")]
    pub wasm_dir: PathBuf,

    #[serde(default = "default_rust_dir")]
    pub rust_dir: PathBuf,

    #[serde(default)]
    pub preserve_rust: bool,

    #[serde(default = "default_true")]
    pub preserve_artifacts: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,

    #[serde(default = "default_log_format")]
    pub format: String,

    #[serde(default = "default_log_output")]
    pub output: String,

    pub file: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_cache_dir")]
    pub directory: PathBuf,

    #[serde(default = "default_cache_size")]
    pub max_size_mb: usize,

    #[serde(default = "default_ttl")]
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    #[serde(default = "default_workers")]
    pub parallel_workers: usize,

    #[serde(default = "default_max_requests")]
    pub max_concurrent_requests: usize,

    #[serde(default = "default_timeout")]
    pub request_timeout_seconds: u64,
}

// Default values
fn default_mode() -> String { "pattern".to_string() }
fn default_temperature() -> f32 { 0.2 }
fn default_true() -> bool { true }
fn default_opt_level() -> String { "3".to_string() }
fn default_batch_size() -> usize { 32 }
fn default_memory_limit() -> usize { 8192 }
fn default_timeout() -> u64 { 30 }
fn default_retry() -> u32 { 3 }
fn default_retry_delay() -> u64 { 1000 }
fn default_wasm_dir() -> PathBuf { PathBuf::from("./dist/wasm") }
fn default_rust_dir() -> PathBuf { PathBuf::from("./dist/rust") }
fn default_log_level() -> String { "info".to_string() }
fn default_log_format() -> String { "pretty".to_string() }
fn default_log_output() -> String { "stdout".to_string() }
fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("portalis")
}
fn default_cache_size() -> usize { 1024 }
fn default_ttl() -> u64 { 86400 }
fn default_workers() -> usize { num_cpus::get() }
fn default_max_requests() -> usize { 100 }

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            mode: default_mode(),
            temperature: default_temperature(),
            include_metrics: default_true(),
            run_tests: default_true(),
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            opt_level: default_opt_level(),
            strip_debug: default_true(),
            lto: default_true(),
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cuda_device: 0,
            batch_size: default_batch_size(),
            memory_limit_mb: default_memory_limit(),
        }
    }
}

impl Default for ServicesConfig {
    fn default() -> Self {
        Self {
            nemo_url: None,
            triton_url: None,
            dgx_cloud_url: None,
            nemo: NemoConfig::default(),
        }
    }
}

impl Default for NemoConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: default_timeout(),
            retry_attempts: default_retry(),
            retry_delay_ms: default_retry_delay(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            wasm_dir: default_wasm_dir(),
            rust_dir: default_rust_dir(),
            preserve_rust: false,
            preserve_artifacts: default_true(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            output: default_log_output(),
            file: None,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            directory: default_cache_dir(),
            max_size_mb: default_cache_size(),
            ttl_seconds: default_ttl(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            parallel_workers: default_workers(),
            max_concurrent_requests: default_max_requests(),
            request_timeout_seconds: default_timeout(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            translation: TranslationConfig::default(),
            optimization: OptimizationConfig::default(),
            gpu: GpuConfig::default(),
            services: ServicesConfig::default(),
            output: OutputConfig::default(),
            logging: LoggingConfig::default(),
            cache: CacheConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {}", path.as_ref().display()))?;

        let config: Config = toml::from_str(&contents)
            .with_context(|| "Failed to parse config file")?;

        Ok(config)
    }

    pub fn load_default() -> Result<Self> {
        let mut config_paths = vec![
            PathBuf::from("portalis.toml"),
            PathBuf::from(".portalis.toml"),
        ];

        if let Some(config_dir) = dirs::config_dir() {
            config_paths.push(config_dir.join("portalis/config.toml"));
        }

        for path in config_paths {
            if path.exists() {
                return Self::load(path);
            }
        }

        Ok(Self::default())
    }
}
