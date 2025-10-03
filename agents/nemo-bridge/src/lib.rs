//! NeMo Bridge - Rust client for NVIDIA NeMo translation service
//!
//! Provides HTTP/gRPC bridge between Rust transpiler agents and Python-based
//! NeMo translation service for GPU-accelerated code translation.

use portalis_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// NeMo translation service client
pub struct NeMoClient {
    base_url: String,
    client: reqwest::Client,
}

/// Translation request to NeMo service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateRequest {
    /// Python source code to translate
    pub python_code: String,

    /// Translation mode: "fast" | "quality" | "streaming"
    #[serde(default = "default_mode")]
    pub mode: String,

    /// Temperature for sampling (0.0-1.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Include confidence scores
    #[serde(default)]
    pub include_metrics: bool,
}

/// Translation response from NeMo service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateResponse {
    /// Generated Rust code
    pub rust_code: String,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,

    /// Translation metrics
    #[serde(default)]
    pub metrics: TranslationMetrics,
}

/// Translation performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranslationMetrics {
    /// Total translation time (ms)
    pub total_time_ms: f32,

    /// GPU utilization percentage
    pub gpu_utilization: f32,

    /// Number of tokens processed
    pub tokens_processed: usize,

    /// Model inference time (ms)
    pub inference_time_ms: f32,
}

fn default_mode() -> String {
    "quality".to_string()
}

fn default_temperature() -> f32 {
    0.2
}

impl NeMoClient {
    /// Create new NeMo client
    ///
    /// # Arguments
    /// * `base_url` - Base URL of NeMo service (e.g., "http://localhost:8000")
    ///
    /// # Example
    /// ```no_run
    /// use portalis_nemo_bridge::NeMoClient;
    ///
    /// let client = NeMoClient::new("http://localhost:8000").unwrap();
    /// ```
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| Error::Pipeline(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            base_url: base_url.into(),
            client,
        })
    }

    /// Translate Python code to Rust using NeMo service
    ///
    /// # Arguments
    /// * `request` - Translation request with Python code and options
    ///
    /// # Returns
    /// Translation response with generated Rust code and metrics
    ///
    /// # Example
    /// ```no_run
    /// use portalis_nemo_bridge::{NeMoClient, TranslateRequest};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = NeMoClient::new("http://localhost:8000")?;
    /// let request = TranslateRequest {
    ///     python_code: "def add(a: int, b: int) -> int:\n    return a + b".to_string(),
    ///     mode: "quality".to_string(),
    ///     temperature: 0.2,
    ///     include_metrics: true,
    /// };
    ///
    /// let response = client.translate(request).await?;
    /// println!("Rust code: {}", response.rust_code);
    /// println!("Confidence: {:.2}%", response.confidence * 100.0);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn translate(&self, request: TranslateRequest) -> Result<TranslateResponse> {
        let url = format!("{}/api/v1/translation/translate", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Pipeline(format!("Failed to send translation request: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::Pipeline(format!(
                "NeMo service returned error {}: {}",
                status, error_text
            )));
        }

        let translate_response: TranslateResponse = response
            .json()
            .await
            .map_err(|e| Error::Pipeline(format!("Failed to parse translation response: {}", e)))?;

        Ok(translate_response)
    }

    /// Check if NeMo service is healthy and available
    ///
    /// # Returns
    /// `Ok(true)` if service is healthy, error otherwise
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Pipeline(format!("Health check failed: {}", e)))?;

        Ok(response.status().is_success())
    }

    /// Get service information and capabilities
    pub async fn get_info(&self) -> Result<ServiceInfo> {
        let url = format!("{}/api/v1/info", self.base_url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Pipeline(format!("Failed to get service info: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Pipeline(format!(
                "Service info request failed with status: {}",
                response.status()
            )));
        }

        let info: ServiceInfo = response
            .json()
            .await
            .map_err(|e| Error::Pipeline(format!("Failed to parse service info: {}", e)))?;

        Ok(info)
    }
}

/// NeMo service information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    /// Service name
    pub name: String,

    /// Service version
    pub version: String,

    /// Whether CUDA/GPU is enabled
    pub cuda_enabled: bool,

    /// Available translation modes
    pub modes: Vec<String>,

    /// Maximum code length
    pub max_code_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = NeMoClient::new("http://localhost:8000");
        assert!(client.is_ok());
    }

    #[test]
    fn test_translate_request_serialization() {
        let request = TranslateRequest {
            python_code: "def test(): pass".to_string(),
            mode: "fast".to_string(),
            temperature: 0.5,
            include_metrics: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("def test"));
        assert!(json.contains("fast"));
    }

    #[test]
    fn test_default_values() {
        let request = serde_json::from_str::<TranslateRequest>(
            r#"{"python_code": "def test(): pass"}"#
        ).unwrap();

        assert_eq!(request.mode, "quality");
        assert_eq!(request.temperature, 0.2);
        assert_eq!(request.include_metrics, false);
    }

    #[tokio::test]
    async fn test_translate_response_deserialization() {
        let json = r#"{
            "rust_code": "pub fn test() {}",
            "confidence": 0.95,
            "metrics": {
                "total_time_ms": 150.5,
                "gpu_utilization": 0.85,
                "tokens_processed": 42,
                "inference_time_ms": 120.0
            }
        }"#;

        let response: TranslateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.rust_code, "pub fn test() {}");
        assert_eq!(response.confidence, 0.95);
        assert_eq!(response.metrics.total_time_ms, 150.5);
        assert_eq!(response.metrics.tokens_processed, 42);
    }

    // Integration test - requires running NeMo service
    #[tokio::test]
    #[ignore] // Only run when NeMo service is available
    async fn test_translate_integration() {
        let client = NeMoClient::new("http://localhost:8000").unwrap();

        let request = TranslateRequest {
            python_code: "def add(a: int, b: int) -> int:\n    return a + b".to_string(),
            mode: "fast".to_string(),
            temperature: 0.2,
            include_metrics: true,
        };

        let result = client.translate(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(!response.rust_code.is_empty());
        assert!(response.confidence > 0.0);
    }

    #[tokio::test]
    #[ignore] // Only run when NeMo service is available
    async fn test_health_check_integration() {
        let client = NeMoClient::new("http://localhost:8000").unwrap();
        let health = client.health_check().await;
        assert!(health.is_ok());
        assert!(health.unwrap());
    }
}
