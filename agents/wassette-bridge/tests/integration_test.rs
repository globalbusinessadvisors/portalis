//! Integration tests for Wassette bridge

use portalis_wassette_bridge::{
    ComponentPermissions, ExecutionResult, ValidationReport, WassetteClient, WassetteConfig,
};
use std::path::Path;
use tempfile::TempDir;

#[tokio::test]
async fn test_client_creation_default() {
    let client = WassetteClient::default();
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_client_creation_with_config() {
    let config = WassetteConfig {
        enable_sandbox: true,
        max_memory_mb: 256,
        max_execution_time_secs: 60,
        permissions: ComponentPermissions::default(),
    };

    let client = WassetteClient::new(config);
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_mock_component_validation() {
    let client = WassetteClient::default().unwrap();
    let temp_dir = TempDir::new().unwrap();
    let wasm_path = temp_dir.path().join("test.wasm");

    // Create a minimal WASM file for testing
    std::fs::write(&wasm_path, b"\x00asm\x01\x00\x00\x00").unwrap();

    let report = client.validate_component(&wasm_path);

    assert!(report.is_ok());
    let report = report.unwrap();

    // In mock mode, validation succeeds
    // With runtime, this minimal WASM module will fail (not a component)
    if !cfg!(feature = "runtime") {
        assert!(report.is_valid);
    } else {
        assert!(!report.is_valid, "Invalid component should fail validation");
        assert!(!report.errors.is_empty(), "Should have error messages");
    }
}

#[tokio::test]
async fn test_mock_component_load() {
    let client = WassetteClient::default().unwrap();
    let temp_dir = TempDir::new().unwrap();
    let wasm_path = temp_dir.path().join("test.wasm");

    std::fs::write(&wasm_path, b"\x00asm\x01\x00\x00\x00").unwrap();

    let handle = client.load_component(&wasm_path);

    // In mock mode, loading should succeed (no validation)
    // With runtime, it will fail because this isn't a valid component
    if !cfg!(feature = "runtime") {
        assert!(handle.is_ok());
        let handle = handle.unwrap();
        assert!(!handle.id().is_empty());
        assert_eq!(handle.path(), wasm_path.as_path());
    } else {
        // With runtime, invalid components should fail
        assert!(handle.is_err());
    }
}

#[tokio::test]
async fn test_mock_component_execution() {
    let client = WassetteClient::default().unwrap();
    let temp_dir = TempDir::new().unwrap();
    let wasm_path = temp_dir.path().join("test.wasm");

    std::fs::write(&wasm_path, b"\x00asm\x01\x00\x00\x00").unwrap();

    // In mock mode, loading and execution should succeed
    // With runtime, this will fail because it's not a valid component
    if !cfg!(feature = "runtime") {
        let handle = client.load_component(&wasm_path).unwrap();
        let result = client.execute_component(&handle, vec!["arg1".to_string()]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
    } else {
        // With runtime enabled, this should fail
        let handle = client.load_component(&wasm_path);
        assert!(handle.is_err(), "Invalid WASM should fail with runtime enabled");
    }
}

#[tokio::test]
async fn test_restrictive_permissions() {
    let mut config = WassetteConfig::default();
    config.permissions.allow_fs = false;
    config.permissions.allow_network = false;
    config.permissions.allow_env = false;

    let client = WassetteClient::new(config);
    assert!(client.is_ok());

    // Verify default restrictive permissions
    let client = client.unwrap();
    // Client should be created successfully even with restrictive permissions
}

#[tokio::test]
async fn test_permissive_permissions() {
    let mut config = WassetteConfig::default();
    config.permissions.allow_fs = true;
    config.permissions.allow_network = true;
    config.permissions.allow_env = true;
    config.permissions.allowed_paths = vec!["/tmp".to_string()];
    config.permissions.allowed_hosts = vec!["api.example.com".to_string()];
    config.permissions.allowed_env_vars = vec!["PATH".to_string()];

    let client = WassetteClient::new(config);
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_client_availability() {
    let client = WassetteClient::default().unwrap();

    // Without runtime feature, should report as unavailable
    #[cfg(not(feature = "runtime"))]
    assert!(!client.is_available());

    // With runtime feature, should report as available
    #[cfg(feature = "runtime")]
    assert!(client.is_available());
}
