//! Integration tests for NeMo bridge with mock server

use portalis_nemo_bridge::{NeMoClient, TranslateRequest, TranslateResponse};
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test]
async fn test_translate_with_mock_server() {
    // Start mock server
    let mock_server = MockServer::start().await;

    // Configure mock response
    let mock_response = serde_json::json!({
        "rust_code": "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}",
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

    // Create client pointing to mock server
    let client = NeMoClient::new(&mock_server.uri()).unwrap();

    // Make request
    let request = TranslateRequest {
        python_code: "def add(a: int, b: int) -> int:\n    return a + b".to_string(),
        mode: "quality".to_string(),
        temperature: 0.2,
        include_metrics: true,
    };

    let response = client.translate(request).await.unwrap();

    // Verify response
    assert!(response.rust_code.contains("pub fn add"));
    assert_eq!(response.confidence, 0.95);
    assert_eq!(response.metrics.total_time_ms, 150.5);
    assert_eq!(response.metrics.gpu_utilization, 0.85);
    assert_eq!(response.metrics.tokens_processed, 42);
}

#[tokio::test]
async fn test_translate_error_handling() {
    let mock_server = MockServer::start().await;

    // Configure mock to return error
    Mock::given(method("POST"))
        .and(path("/api/v1/translation/translate"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let client = NeMoClient::new(&mock_server.uri()).unwrap();

    let request = TranslateRequest {
        python_code: "invalid python code!!!".to_string(),
        mode: "fast".to_string(),
        temperature: 0.0,
        include_metrics: false,
    };

    let result = client.translate(request).await;
    assert!(result.is_err());

    let error_msg = format!("{:?}", result.unwrap_err());
    assert!(error_msg.contains("500"));
}

#[tokio::test]
async fn test_health_check_success() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "healthy"
        })))
        .mount(&mock_server)
        .await;

    let client = NeMoClient::new(&mock_server.uri()).unwrap();
    let health = client.health_check().await.unwrap();
    assert!(health);
}

#[tokio::test]
async fn test_health_check_failure() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&mock_server)
        .await;

    let client = NeMoClient::new(&mock_server.uri()).unwrap();
    let health = client.health_check().await.unwrap();
    assert!(!health);
}

#[tokio::test]
async fn test_get_service_info() {
    let mock_server = MockServer::start().await;

    let info_response = serde_json::json!({
        "name": "portalis-nemo",
        "version": "0.1.0",
        "cuda_enabled": true,
        "modes": ["fast", "quality", "streaming"],
        "max_code_length": 100000
    });

    Mock::given(method("GET"))
        .and(path("/api/v1/info"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&info_response))
        .mount(&mock_server)
        .await;

    let client = NeMoClient::new(&mock_server.uri()).unwrap();
    let info = client.get_info().await.unwrap();

    assert_eq!(info.name, "portalis-nemo");
    assert_eq!(info.version, "0.1.0");
    assert!(info.cuda_enabled);
    assert_eq!(info.modes.len(), 3);
    assert_eq!(info.max_code_length, 100000);
}

#[tokio::test]
async fn test_translate_with_different_modes() {
    let mock_server = MockServer::start().await;

    // Test "fast" mode
    let fast_response = serde_json::json!({
        "rust_code": "// Fast translation\npub fn test() {}",
        "confidence": 0.80,
        "metrics": {
            "total_time_ms": 50.0,
            "gpu_utilization": 0.60,
            "tokens_processed": 20,
            "inference_time_ms": 40.0
        }
    });

    Mock::given(method("POST"))
        .and(path("/api/v1/translation/translate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&fast_response))
        .mount(&mock_server)
        .await;

    let client = NeMoClient::new(&mock_server.uri()).unwrap();

    let request = TranslateRequest {
        python_code: "def test(): pass".to_string(),
        mode: "fast".to_string(),
        temperature: 0.5,
        include_metrics: true,
    };

    let response = client.translate(request).await.unwrap();
    assert!(response.rust_code.contains("Fast translation"));
    assert_eq!(response.confidence, 0.80);
}

#[tokio::test]
async fn test_translate_timeout() {
    let mock_server = MockServer::start().await;

    // Don't mount any mock - let it timeout
    let client = NeMoClient::new(&mock_server.uri()).unwrap();

    let request = TranslateRequest {
        python_code: "def test(): pass".to_string(),
        mode: "quality".to_string(),
        temperature: 0.2,
        include_metrics: false,
    };

    // This should timeout or fail since no mock is mounted
    let result = client.translate(request).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_translate_with_class_code() {
    let mock_server = MockServer::start().await;

    let class_response = serde_json::json!({
        "rust_code": "pub struct Calculator {\n    pub precision: i32,\n}\n\nimpl Calculator {\n    pub fn new(precision: i32) -> Self {\n        Self { precision }\n    }\n}",
        "confidence": 0.92,
        "metrics": {
            "total_time_ms": 220.0,
            "gpu_utilization": 0.88,
            "tokens_processed": 65,
            "inference_time_ms": 180.0
        }
    });

    Mock::given(method("POST"))
        .and(path("/api/v1/translation/translate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&class_response))
        .mount(&mock_server)
        .await;

    let client = NeMoClient::new(&mock_server.uri()).unwrap();

    let request = TranslateRequest {
        python_code: r#"
class Calculator:
    def __init__(self, precision: int):
        self.precision = precision
"#
        .to_string(),
        mode: "quality".to_string(),
        temperature: 0.2,
        include_metrics: true,
    };

    let response = client.translate(request).await.unwrap();
    assert!(response.rust_code.contains("struct Calculator"));
    assert!(response.rust_code.contains("impl Calculator"));
    assert_eq!(response.confidence, 0.92);
}
