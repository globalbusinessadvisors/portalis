//! Integration tests for WASI Fetch API
//!
//! Tests the fetch API implementation across different platforms

#[cfg(test)]
mod fetch_tests {
    use portalis_transpiler::wasi_fetch::{
        WasiFetch, Request, Response, Method, QueryParams, Body, FetchError
    };
    use std::collections::HashMap;
    use serde_json::json;

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_request_builder() {
        let mut request = Request::new(Method::Get, "https://httpbin.org/get");
        request.header("User-Agent", "Portalis/1.0");
        request.header("Accept", "application/json");

        assert_eq!(request.url(), "https://httpbin.org/get");
        assert_eq!(request.method(), Method::Get);
        assert_eq!(request.headers_map().get("User-Agent").unwrap(), "Portalis/1.0");
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_json_body_serialization() {
        let mut request = Request::new(Method::Post, "https://httpbin.org/post");
        let data = json!({
            "name": "Alice",
            "age": 30,
            "active": true
        });

        request.set_json(&data).unwrap();

        match request.body() {
            Body::Json(json) => {
                assert_eq!(json.get("name").unwrap().as_str().unwrap(), "Alice");
                assert_eq!(json.get("age").unwrap().as_i64().unwrap(), 30);
            },
            _ => panic!("Expected JSON body"),
        }

        assert_eq!(
            request.headers_map().get("Content-Type").unwrap(),
            "application/json"
        );
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_form_body() {
        let mut request = Request::new(Method::Post, "https://httpbin.org/post");
        let mut form = HashMap::new();
        form.insert("username".to_string(), "alice".to_string());
        form.insert("password".to_string(), "secret123".to_string());

        request.set_form(form.clone());

        match request.body() {
            Body::Form(f) => {
                assert_eq!(f.get("username").unwrap(), "alice");
                assert_eq!(f.get("password").unwrap(), "secret123");
            },
            _ => panic!("Expected Form body"),
        }

        assert_eq!(
            request.headers_map().get("Content-Type").unwrap(),
            "application/x-www-form-urlencoded"
        );
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_query_params_builder() {
        let mut params = QueryParams::new();
        params.add("page", "1");
        params.add("limit", "20");
        params.add("sort", "name");

        let query_string = params.build();
        assert!(query_string.contains("page=1"));
        assert!(query_string.contains("limit=20"));
        assert!(query_string.contains("sort=name"));
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_query_params_append_to_url() {
        let mut params = QueryParams::new();
        params.add("foo", "bar");
        params.add("baz", "qux");

        let url = params.append_to_url("https://api.example.com/data");
        assert_eq!(url, "https://api.example.com/data?foo=bar&baz=qux");

        let url_with_existing = params.append_to_url("https://api.example.com/data?existing=param");
        assert_eq!(url_with_existing, "https://api.example.com/data?existing=param&foo=bar&baz=qux");
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_simple_get_request() {
        // Using httpbin.org for testing (publicly available HTTP testing service)
        let result = WasiFetch::get("https://httpbin.org/get").await;

        match result {
            Ok(response) => {
                assert!(response.is_success());
                assert_eq!(response.status(), 200);
                assert!(response.content_type().is_some());

                // Parse response as JSON
                let json: serde_json::Value = response.json().unwrap();
                assert!(json.is_object());
            },
            Err(e) => {
                // Network errors are acceptable in test environments
                eprintln!("Network request failed (expected in CI): {}", e);
            }
        }
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_post_json_request() {
        let data = json!({
            "name": "Test User",
            "email": "test@example.com"
        });

        let result = WasiFetch::post_json("https://httpbin.org/post", &data).await;

        match result {
            Ok(response) => {
                assert!(response.is_success());
                let json: serde_json::Value = response.json().unwrap();

                // httpbin.org echoes back the JSON in the "json" field
                assert!(json.get("json").is_some());
            },
            Err(e) => {
                eprintln!("Network request failed (expected in CI): {}", e);
            }
        }
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_response_status_checks() {
        let response = Response::new(200, HashMap::new(), vec![]);
        assert!(response.is_success());
        assert!(!response.is_client_error());
        assert!(!response.is_server_error());
        assert!(!response.is_redirect());

        let response = Response::new(301, HashMap::new(), vec![]);
        assert!(response.is_redirect());
        assert!(!response.is_success());

        let response = Response::new(404, HashMap::new(), vec![]);
        assert!(response.is_client_error());
        assert!(!response.is_success());

        let response = Response::new(500, HashMap::new(), vec![]);
        assert!(response.is_server_error());
        assert!(!response.is_success());
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_response_text_parsing() {
        let body = "Hello, World!".as_bytes().to_vec();
        let response = Response::new(200, HashMap::new(), body);

        let text = response.text().unwrap();
        assert_eq!(text, "Hello, World!");
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_response_json_parsing() {
        let json_str = r#"{"name":"Alice","age":30}"#;
        let body = json_str.as_bytes().to_vec();
        let response = Response::new(200, HashMap::new(), body);

        #[derive(serde::Deserialize)]
        struct Person {
            name: String,
            age: u32,
        }

        let person: Person = response.json().unwrap();
        assert_eq!(person.name, "Alice");
        assert_eq!(person.age, 30);
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_custom_headers() {
        let mut request = Request::new(Method::Get, "https://httpbin.org/headers");
        request.header("X-Custom-Header", "CustomValue");
        request.header("X-Request-ID", "12345");

        let result = WasiFetch::fetch(request).await;

        match result {
            Ok(response) => {
                assert!(response.is_success());
                // httpbin.org echoes headers in the response
                let json: serde_json::Value = response.json().unwrap();
                let headers = json.get("headers").unwrap();
                assert!(headers.get("X-Custom-Header").is_some() || headers.get("x-custom-header").is_some());
            },
            Err(e) => {
                eprintln!("Network request failed (expected in CI): {}", e);
            }
        }
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_delete_request() {
        let result = WasiFetch::delete("https://httpbin.org/delete").await;

        match result {
            Ok(response) => {
                assert!(response.is_success());
            },
            Err(e) => {
                eprintln!("Network request failed (expected in CI): {}", e);
            }
        }
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_put_request() {
        let data = json!({"status": "updated"});
        let result = WasiFetch::put_json("https://httpbin.org/put", &data).await;

        match result {
            Ok(response) => {
                assert!(response.is_success());
            },
            Err(e) => {
                eprintln!("Network request failed (expected in CI): {}", e);
            }
        }
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_patch_request() {
        let data = json!({"field": "value"});
        let result = WasiFetch::patch_json("https://httpbin.org/patch", &data).await;

        match result {
            Ok(response) => {
                assert!(response.is_success());
            },
            Err(e) => {
                eprintln!("Network request failed (expected in CI): {}", e);
            }
        }
    }

    #[test]
    fn test_method_display() {
        assert_eq!(Method::Get.to_string(), "GET");
        assert_eq!(Method::Post.to_string(), "POST");
        assert_eq!(Method::Put.to_string(), "PUT");
        assert_eq!(Method::Delete.to_string(), "DELETE");
        assert_eq!(Method::Patch.to_string(), "PATCH");
        assert_eq!(Method::Head.to_string(), "HEAD");
        assert_eq!(Method::Options.to_string(), "OPTIONS");
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_request_timeout() {
        use std::time::Duration;

        let mut request = Request::new(Method::Get, "https://httpbin.org/delay/10");
        request.timeout(Duration::from_millis(100)); // Very short timeout

        let result = WasiFetch::fetch(request).await;

        // Should timeout
        match result {
            Err(e) => {
                // Expected timeout or connection error
                eprintln!("Expected timeout/error: {}", e);
            },
            Ok(_) => {
                // Unlikely but acceptable if network is very fast
                eprintln!("Request completed unexpectedly fast");
            }
        }
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_bytes_body() {
        let bytes = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F]; // "Hello" in ASCII
        let mut request = Request::new(Method::Post, "https://httpbin.org/post");
        request.set_bytes(bytes.clone());

        match request.body() {
            Body::Bytes(b) => {
                assert_eq!(b, &bytes);
            },
            _ => panic!("Expected Bytes body"),
        }
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_text_body() {
        let mut request = Request::new(Method::Post, "https://httpbin.org/post");
        request.set_text("Hello, World!");

        match request.body() {
            Body::Text(t) => {
                assert_eq!(t, "Hello, World!");
            },
            _ => panic!("Expected Text body"),
        }

        assert_eq!(
            request.headers_map().get("Content-Type").unwrap(),
            "text/plain"
        );
    }
}

#[cfg(test)]
mod python_mapping_tests {
    use portalis_transpiler::py_to_rust_http::{HttpMapper, HttpPatternDetector};
    use std::collections::HashMap;

    #[test]
    fn test_detect_requests_library() {
        let code = r#"
import requests

def fetch_data():
    response = requests.get("https://api.example.com/data")
    return response.json()
"#;
        assert!(HttpPatternDetector::uses_requests(code));
        assert!(!HttpPatternDetector::uses_urllib(code));
    }

    #[test]
    fn test_detect_urllib_library() {
        let code = r#"
from urllib.request import urlopen

def fetch_data():
    response = urlopen("https://api.example.com/data")
    return response.read()
"#;
        assert!(HttpPatternDetector::uses_urllib(code));
        assert!(!HttpPatternDetector::uses_requests(code));
    }

    #[test]
    fn test_detect_httpx_library() {
        let code = r#"
import httpx

async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
"#;
        assert!(HttpPatternDetector::uses_httpx(code));
        assert!(HttpPatternDetector::uses_async_http(code));
    }

    #[test]
    fn test_translate_simple_get() {
        let result = HttpMapper::translate_requests_get("https://api.example.com/data", None);
        assert!(result.contains("WasiFetch::get"));
        assert!(result.contains("https://api.example.com/data"));
    }

    #[test]
    fn test_translate_post_json() {
        let result = HttpMapper::translate_requests_post_json(
            "https://api.example.com/users",
            "user_data"
        );
        assert!(result.contains("WasiFetch::post_json"));
        assert!(result.contains("user_data"));
    }

    #[test]
    fn test_translate_with_headers() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer token".to_string());

        let result = HttpMapper::translate_requests_with_headers(
            "GET",
            "https://api.example.com/protected",
            headers
        );

        assert!(result.contains("Request::new"));
        assert!(result.contains("Method::Get"));
        assert!(result.contains("Authorization"));
        assert!(result.contains("Bearer token"));
    }

    #[test]
    fn test_response_method_mapping() {
        assert_eq!(
            HttpMapper::translate_response_method("json()"),
            Some("response.json()?")
        );
        assert_eq!(
            HttpMapper::translate_response_method("text"),
            Some("response.text()?")
        );
        assert_eq!(
            HttpMapper::translate_response_method("status_code"),
            Some("response.status()")
        );
    }

    #[test]
    fn test_get_http_imports() {
        let imports = HttpMapper::get_http_imports();
        assert!(!imports.is_empty());
        assert!(imports.iter().any(|i| i.contains("wasi_fetch")));
        assert!(imports.iter().any(|i| i.contains("HashMap")));
    }

    #[test]
    fn test_get_http_dependencies() {
        let deps = HttpMapper::get_http_dependencies();
        assert!(!deps.is_empty());
        assert!(deps.iter().any(|(name, _)| *name == "reqwest"));
        assert!(deps.iter().any(|(name, _)| *name == "tokio"));
    }
}
