//! Integration tests for TranspilerAgent with NeMo bridge
//!
//! These tests require the 'nemo' feature flag to be enabled.

#![cfg(feature = "nemo")]

use portalis_transpiler::{TranslationMode, TranspilerAgent};
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test]
async fn test_transpiler_with_nemo_mode() {
    // Start mock NeMo server
    let mock_server = MockServer::start().await;

    // Configure mock response
    let rust_code = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}";
    let mock_response = serde_json::json!({
        "rust_code": rust_code,
        "confidence": 0.95,
        "metrics": {
            "total_time_ms": 150.5,
            "gpu_utilization": 0.85,
            "tokens_processed": 42,
            "inference_time_ms": 120.0
        }
    });

    Mock::given(method("POST"))
        .and(path("/api/v1/translation/translate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&mock_response))
        .mount(&mock_server)
        .await;

    // Create transpiler with NeMo mode
    let mode = TranslationMode::NeMo {
        service_url: mock_server.uri(),
        mode: "quality".to_string(),
        temperature: 0.2,
    };

    let agent = TranspilerAgent::with_mode(mode);

    // Verify mode is set correctly
    match agent.translation_mode() {
        TranslationMode::NeMo { service_url, .. } => {
            assert_eq!(service_url, &mock_server.uri());
        }
        _ => panic!("Expected NeMo mode"),
    }
}

#[tokio::test]
async fn test_transpiler_fallback_to_pattern_based() {
    // Create transpiler with default (pattern-based) mode
    let agent = TranspilerAgent::new();

    match agent.translation_mode() {
        TranslationMode::PatternBased => {
            // Expected default
        }
        _ => panic!("Expected PatternBased mode"),
    }
}

#[tokio::test]
async fn test_translation_mode_switch() {
    let mock_server = MockServer::start().await;

    // Start with pattern-based
    let agent1 = TranspilerAgent::new();
    assert!(matches!(
        agent1.translation_mode(),
        TranslationMode::PatternBased
    ));

    // Switch to NeMo
    let mode = TranslationMode::NeMo {
        service_url: mock_server.uri(),
        mode: "fast".to_string(),
        temperature: 0.5,
    };
    let agent2 = TranspilerAgent::with_mode(mode);

    match agent2.translation_mode() {
        TranslationMode::NeMo {
            service_url,
            mode,
            temperature,
        } => {
            assert_eq!(service_url, &mock_server.uri());
            assert_eq!(mode, "fast");
            assert_eq!(*temperature, 0.5);
        }
        _ => panic!("Expected NeMo mode"),
    }
}
