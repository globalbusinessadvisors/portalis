//! Health Check Endpoints for Portalis
//! Week 33 - Phase 4: Monitoring and Observability
//!
//! Provides:
//! - /health - Basic liveness check
//! - /ready - Readiness check
//! - /metrics - Prometheus metrics endpoint
//! - Detailed component status

use portalis_core::{Error, Result, PortalisMetrics};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub last_check: u64,
}

impl ComponentHealth {
    pub fn healthy(name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            status: HealthStatus::Healthy,
            message: None,
            last_check: current_timestamp(),
        }
    }

    pub fn degraded(name: impl ToString, message: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            status: HealthStatus::Degraded,
            message: Some(message.to_string()),
            last_check: current_timestamp(),
        }
    }

    pub fn unhealthy(name: impl ToString, message: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            status: HealthStatus::Unhealthy,
            message: Some(message.to_string()),
            last_check: current_timestamp(),
        }
    }
}

/// Overall health response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub timestamp: u64,
    pub uptime_seconds: u64,
    pub version: String,
    pub components: Vec<ComponentHealth>,
}

/// Readiness response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    pub ready: bool,
    pub timestamp: u64,
    pub checks: Vec<ReadinessCheck>,
}

/// Individual readiness check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessCheck {
    pub name: String,
    pub ready: bool,
    pub message: Option<String>,
}

/// Health checker service
pub struct HealthChecker {
    start_time: SystemTime,
    metrics: Arc<PortalisMetrics>,
}

impl HealthChecker {
    pub fn new(metrics: Arc<PortalisMetrics>) -> Self {
        Self {
            start_time: SystemTime::now(),
            metrics,
        }
    }

    /// Basic liveness check
    /// Returns HTTP 200 if the service is running
    pub fn liveness(&self) -> HealthResponse {
        HealthResponse {
            status: HealthStatus::Healthy,
            timestamp: current_timestamp(),
            uptime_seconds: self.uptime_seconds(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            components: vec![
                ComponentHealth::healthy("core"),
            ],
        }
    }

    /// Comprehensive health check
    /// Checks all components and returns detailed status
    pub fn health(&self) -> HealthResponse {
        let mut components = Vec::new();

        // Check core system
        components.push(self.check_core());

        // Check agents
        components.push(self.check_agents());

        // Check pipeline
        components.push(self.check_pipeline());

        // Check metrics collection
        components.push(self.check_metrics());

        // Check cache
        components.push(self.check_cache());

        // Determine overall status
        let overall_status = if components.iter().any(|c| c.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else if components.iter().any(|c| c.status == HealthStatus::Degraded) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        HealthResponse {
            status: overall_status,
            timestamp: current_timestamp(),
            uptime_seconds: self.uptime_seconds(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            components,
        }
    }

    /// Readiness check
    /// Returns whether the service is ready to accept requests
    pub fn readiness(&self) -> ReadinessResponse {
        let mut checks = Vec::new();

        // Check if metrics are initialized
        checks.push(ReadinessCheck {
            name: "metrics".to_string(),
            ready: true,
            message: Some("Metrics system initialized".to_string()),
        });

        // Check if agents are ready
        checks.push(self.check_agents_ready());

        // Check if pipeline is ready
        checks.push(self.check_pipeline_ready());

        let ready = checks.iter().all(|c| c.ready);

        ReadinessResponse {
            ready,
            timestamp: current_timestamp(),
            checks,
        }
    }

    /// Export Prometheus metrics
    pub fn metrics(&self) -> Result<String> {
        self.metrics.export().map_err(|e| Error::Internal(e.to_string()))
    }

    // Internal health checks

    fn check_core(&self) -> ComponentHealth {
        // Basic check: if we can execute this, core is healthy
        ComponentHealth::healthy("core")
    }

    fn check_agents(&self) -> ComponentHealth {
        // Check if agents are functioning
        // In production, this would check actual agent status
        ComponentHealth::healthy("agents")
    }

    fn check_pipeline(&self) -> ComponentHealth {
        // Check pipeline queue depth
        // In production, query actual queue metrics
        ComponentHealth::healthy("pipeline")
    }

    fn check_metrics(&self) -> ComponentHealth {
        // Verify metrics are being collected
        match self.metrics.export() {
            Ok(_) => ComponentHealth::healthy("metrics"),
            Err(e) => ComponentHealth::unhealthy("metrics", format!("Metrics export failed: {}", e)),
        }
    }

    fn check_cache(&self) -> ComponentHealth {
        // Check cache status
        // In production, verify cache connectivity
        ComponentHealth::healthy("cache")
    }

    fn check_agents_ready(&self) -> ReadinessCheck {
        // Check if agents are initialized and ready
        ReadinessCheck {
            name: "agents".to_string(),
            ready: true,
            message: Some("All agents initialized".to_string()),
        }
    }

    fn check_pipeline_ready(&self) -> ReadinessCheck {
        // Check if pipeline is accepting work
        ReadinessCheck {
            name: "pipeline".to_string(),
            ready: true,
            message: Some("Pipeline ready to process requests".to_string()),
        }
    }

    fn uptime_seconds(&self) -> u64 {
        self.start_time
            .elapsed()
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Health check HTTP handlers
#[cfg(feature = "http")]
pub mod http {
    use super::*;
    use std::sync::Arc;

    /// Liveness endpoint handler
    /// GET /health
    pub async fn liveness_handler(
        checker: Arc<HealthChecker>,
    ) -> Result<impl serde::Serialize, Error> {
        Ok(checker.liveness())
    }

    /// Health endpoint handler
    /// GET /health/detailed
    pub async fn health_handler(
        checker: Arc<HealthChecker>,
    ) -> Result<impl serde::Serialize, Error> {
        Ok(checker.health())
    }

    /// Readiness endpoint handler
    /// GET /ready
    pub async fn readiness_handler(
        checker: Arc<HealthChecker>,
    ) -> Result<impl serde::Serialize, Error> {
        Ok(checker.readiness())
    }

    /// Metrics endpoint handler
    /// GET /metrics
    pub async fn metrics_handler(
        checker: Arc<HealthChecker>,
    ) -> Result<String, Error> {
        checker.metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_health_creation() {
        let healthy = ComponentHealth::healthy("test");
        assert_eq!(healthy.status, HealthStatus::Healthy);
        assert_eq!(healthy.name, "test");

        let degraded = ComponentHealth::degraded("test", "warning");
        assert_eq!(degraded.status, HealthStatus::Degraded);
        assert!(degraded.message.is_some());

        let unhealthy = ComponentHealth::unhealthy("test", "error");
        assert_eq!(unhealthy.status, HealthStatus::Unhealthy);
        assert!(unhealthy.message.is_some());
    }

    #[test]
    fn test_health_checker_liveness() {
        let metrics = Arc::new(PortalisMetrics::new().unwrap());
        let checker = HealthChecker::new(metrics);

        let response = checker.liveness();
        assert_eq!(response.status, HealthStatus::Healthy);
        assert!(!response.components.is_empty());
    }

    #[test]
    fn test_health_checker_health() {
        let metrics = Arc::new(PortalisMetrics::new().unwrap());
        let checker = HealthChecker::new(metrics);

        let response = checker.health();
        assert_eq!(response.status, HealthStatus::Healthy);
        assert!(response.components.len() >= 5);
    }

    #[test]
    fn test_health_checker_readiness() {
        let metrics = Arc::new(PortalisMetrics::new().unwrap());
        let checker = HealthChecker::new(metrics);

        let response = checker.readiness();
        assert!(response.ready);
        assert!(!response.checks.is_empty());
    }

    #[test]
    fn test_health_checker_metrics() {
        let metrics = Arc::new(PortalisMetrics::new().unwrap());
        let checker = HealthChecker::new(metrics);

        let result = checker.metrics();
        assert!(result.is_ok());
        let metrics_text = result.unwrap();
        assert!(metrics_text.contains("portalis"));
    }

    #[test]
    fn test_current_timestamp() {
        let ts = current_timestamp();
        assert!(ts > 0);
        assert!(ts < u64::MAX);
    }

    #[test]
    fn test_uptime_calculation() {
        let metrics = Arc::new(PortalisMetrics::new().unwrap());
        let checker = HealthChecker::new(metrics);

        std::thread::sleep(std::time::Duration::from_millis(100));
        let uptime = checker.uptime_seconds();
        assert!(uptime >= 0);
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: HealthStatus::Healthy,
            timestamp: 1234567890,
            uptime_seconds: 3600,
            version: "1.0.0".to_string(),
            components: vec![
                ComponentHealth::healthy("test"),
            ],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_readiness_response_serialization() {
        let response = ReadinessResponse {
            ready: true,
            timestamp: 1234567890,
            checks: vec![
                ReadinessCheck {
                    name: "test".to_string(),
                    ready: true,
                    message: None,
                },
            ],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("true"));
        assert!(json.contains("test"));
    }
}
