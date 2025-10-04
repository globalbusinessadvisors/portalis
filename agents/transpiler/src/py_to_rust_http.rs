//! Python to Rust HTTP API Mapping
//!
//! Maps Python HTTP libraries to WASI-compatible Rust fetch API:
//! - requests -> wasi_fetch
//! - urllib.request -> wasi_fetch
//! - httpx -> wasi_fetch
//! - aiohttp -> wasi_fetch (async)
//!
//! This module provides translation patterns for common HTTP operations.

use std::collections::HashMap;

/// HTTP library translation patterns
pub struct HttpMapper;

impl HttpMapper {
    /// Map Python requests.get() to Rust
    ///
    /// Python:
    /// ```python
    /// response = requests.get("https://api.example.com/data")
    /// data = response.json()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let response = WasiFetch::get("https://api.example.com/data").await?;
    /// let data: serde_json::Value = response.json()?;
    /// ```
    pub fn translate_requests_get(url: &str, params: Option<HashMap<String, String>>) -> String {
        let url_with_params = if let Some(params) = params {
            format!("{{
    let mut query = QueryParams::new();
    {}
    query.append_to_url(\"{}\")
}}",
                params.iter()
                    .map(|(k, v)| format!("    query.add(\"{}\", \"{}\");", k, v))
                    .collect::<Vec<_>>()
                    .join("\n"),
                url
            )
        } else {
            format!("\"{}\"", url)
        };

        format!("let response = WasiFetch::get({}).await?;", url_with_params)
    }

    /// Map Python requests.post() with JSON to Rust
    ///
    /// Python:
    /// ```python
    /// response = requests.post("https://api.example.com/users", json={"name": "Alice"})
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let data = serde_json::json!({"name": "Alice"});
    /// let response = WasiFetch::post_json("https://api.example.com/users", &data).await?;
    /// ```
    pub fn translate_requests_post_json(url: &str, json_var: &str) -> String {
        format!("let response = WasiFetch::post_json(\"{}\", &{}).await?;", url, json_var)
    }

    /// Map Python requests.post() with form data to Rust
    ///
    /// Python:
    /// ```python
    /// response = requests.post("https://api.example.com/login", data={"user": "alice", "pass": "secret"})
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let mut form = HashMap::new();
    /// form.insert("user".to_string(), "alice".to_string());
    /// form.insert("pass".to_string(), "secret".to_string());
    /// let response = WasiFetch::post_form("https://api.example.com/login", form).await?;
    /// ```
    pub fn translate_requests_post_form(url: &str, data: HashMap<String, String>) -> String {
        let mut code = String::new();
        code.push_str("let mut form = HashMap::new();\n");

        for (key, value) in data {
            code.push_str(&format!("form.insert(\"{}\".to_string(), \"{}\".to_string());\n", key, value));
        }

        code.push_str(&format!("let response = WasiFetch::post_form(\"{}\", form).await?;", url));
        code
    }

    /// Map Python requests with headers to Rust
    ///
    /// Python:
    /// ```python
    /// headers = {"Authorization": "Bearer token123"}
    /// response = requests.get("https://api.example.com/protected", headers=headers)
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let mut request = Request::new(Method::Get, "https://api.example.com/protected");
    /// request.header("Authorization", "Bearer token123");
    /// let response = WasiFetch::fetch(request).await?;
    /// ```
    pub fn translate_requests_with_headers(
        method: &str,
        url: &str,
        headers: HashMap<String, String>
    ) -> String {
        let rust_method = method.to_uppercase();
        let method_enum = match rust_method.as_str() {
            "GET" => "Method::Get",
            "POST" => "Method::Post",
            "PUT" => "Method::Put",
            "DELETE" => "Method::Delete",
            "PATCH" => "Method::Patch",
            _ => "Method::Get",
        };

        let mut code = format!("let mut request = Request::new({}, \"{}\");\n", method_enum, url);

        for (key, value) in headers {
            code.push_str(&format!("request.header(\"{}\", \"{}\");\n", key, value));
        }

        code.push_str("let response = WasiFetch::fetch(request).await?;");
        code
    }

    /// Map Python urllib.request.urlopen() to Rust
    ///
    /// Python:
    /// ```python
    /// from urllib.request import urlopen
    /// response = urlopen("https://api.example.com/data")
    /// data = response.read()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let response = WasiFetch::get("https://api.example.com/data").await?;
    /// let data = response.bytes();
    /// ```
    pub fn translate_urllib_urlopen(url: &str) -> String {
        format!(
            "let response = WasiFetch::get(\"{}\").await?;\nlet data = response.bytes();",
            url
        )
    }

    /// Map Python httpx async requests to Rust
    ///
    /// Python:
    /// ```python
    /// async with httpx.AsyncClient() as client:
    ///     response = await client.get("https://api.example.com/data")
    ///     data = response.json()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// let response = WasiFetch::get("https://api.example.com/data").await?;
    /// let data: serde_json::Value = response.json()?;
    /// ```
    pub fn translate_httpx_async_get(url: &str) -> String {
        format!("let response = WasiFetch::get(\"{}\").await?;", url)
    }

    /// Generate complete Rust function from Python requests code
    ///
    /// Python:
    /// ```python
    /// def fetch_user(user_id):
    ///     response = requests.get(f"https://api.example.com/users/{user_id}")
    ///     return response.json()
    /// ```
    ///
    /// Rust:
    /// ```rust,no_run
    /// async fn fetch_user(user_id: i32) -> Result<serde_json::Value> {
    ///     let url = format!("https://api.example.com/users/{}", user_id);
    ///     let response = WasiFetch::get(&url).await?;
    ///     let data = response.json()?;
    ///     Ok(data)
    /// }
    /// ```
    pub fn generate_async_http_function(
        func_name: &str,
        params: Vec<(&str, &str)>,
        return_type: &str,
        http_call: &str,
    ) -> String {
        let param_str = params.iter()
            .map(|(name, typ)| format!("{}: {}", name, typ))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "async fn {}({}) -> Result<{}> {{\n    {}\n    Ok(data)\n}}",
            func_name, param_str, return_type, http_call
        )
    }

    /// Map response methods from Python to Rust
    pub fn translate_response_method(method: &str) -> Option<&'static str> {
        match method {
            // requests library methods
            "json()" => Some("response.json()?"),
            ".json()" => Some("response.json()?"),
            "text" => Some("response.text()?"),
            ".text" => Some("response.text()?"),
            "content" => Some("response.bytes()"),
            ".content" => Some("response.bytes()"),
            "status_code" => Some("response.status()"),
            ".status_code" => Some("response.status()"),
            "headers" => Some("response.headers()"),
            ".headers" => Some("response.headers()"),

            // urllib methods
            "read()" => Some("response.bytes()"),
            ".read()" => Some("response.bytes()"),
            "getcode()" => Some("response.status()"),
            ".getcode()" => Some("response.status()"),

            _ => None,
        }
    }

    /// Get required imports for HTTP functionality
    pub fn get_http_imports() -> Vec<&'static str> {
        vec![
            "use crate::wasi_fetch::{WasiFetch, Request, Response, Method, QueryParams};",
            "use std::collections::HashMap;",
            "use anyhow::Result;",
        ]
    }

    /// Get required Cargo dependencies for HTTP functionality
    pub fn get_http_dependencies() -> Vec<(&'static str, &'static str)> {
        vec![
            ("reqwest", "0.11"),
            ("tokio", "1.35"),
            ("serde_json", "1.0"),
        ]
    }
}

/// Python HTTP pattern detection
pub struct HttpPatternDetector;

impl HttpPatternDetector {
    /// Detect if Python code uses requests library
    pub fn uses_requests(python_code: &str) -> bool {
        python_code.contains("import requests")
            || python_code.contains("from requests import")
            || python_code.contains("requests.get")
            || python_code.contains("requests.post")
    }

    /// Detect if Python code uses urllib
    pub fn uses_urllib(python_code: &str) -> bool {
        python_code.contains("import urllib")
            || python_code.contains("from urllib")
            || python_code.contains("urlopen")
    }

    /// Detect if Python code uses httpx
    pub fn uses_httpx(python_code: &str) -> bool {
        python_code.contains("import httpx")
            || python_code.contains("from httpx import")
            || python_code.contains("httpx.get")
            || python_code.contains("httpx.AsyncClient")
    }

    /// Detect if Python code uses aiohttp
    pub fn uses_aiohttp(python_code: &str) -> bool {
        python_code.contains("import aiohttp")
            || python_code.contains("from aiohttp import")
            || python_code.contains("aiohttp.ClientSession")
    }

    /// Detect if async HTTP calls are used
    pub fn uses_async_http(python_code: &str) -> bool {
        (Self::uses_httpx(python_code) && python_code.contains("async"))
            || Self::uses_aiohttp(python_code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_requests_get() {
        let result = HttpMapper::translate_requests_get("https://api.example.com/data", None);
        assert!(result.contains("WasiFetch::get"));
        assert!(result.contains("https://api.example.com/data"));
    }

    #[test]
    fn test_translate_requests_get_with_params() {
        let mut params = HashMap::new();
        params.insert("page".to_string(), "1".to_string());
        params.insert("limit".to_string(), "10".to_string());

        let result = HttpMapper::translate_requests_get("https://api.example.com/data", Some(params));
        assert!(result.contains("QueryParams"));
        assert!(result.contains("query.add"));
    }

    #[test]
    fn test_translate_requests_post_json() {
        let result = HttpMapper::translate_requests_post_json("https://api.example.com/users", "user_data");
        assert!(result.contains("WasiFetch::post_json"));
        assert!(result.contains("user_data"));
    }

    #[test]
    fn test_translate_requests_with_headers() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer token123".to_string());

        let result = HttpMapper::translate_requests_with_headers("GET", "https://api.example.com", headers);
        assert!(result.contains("Request::new"));
        assert!(result.contains("Method::Get"));
        assert!(result.contains("Authorization"));
    }

    #[test]
    fn test_response_method_translation() {
        assert_eq!(HttpMapper::translate_response_method("json()"), Some("response.json()?"));
        assert_eq!(HttpMapper::translate_response_method("text"), Some("response.text()?"));
        assert_eq!(HttpMapper::translate_response_method("status_code"), Some("response.status()"));
    }

    #[test]
    fn test_pattern_detection() {
        assert!(HttpPatternDetector::uses_requests("import requests\nrequests.get('url')"));
        assert!(HttpPatternDetector::uses_urllib("from urllib.request import urlopen"));
        assert!(HttpPatternDetector::uses_httpx("import httpx\nawait httpx.get('url')"));
        assert!(HttpPatternDetector::uses_async_http("import httpx\nasync def foo():\n    await httpx.get('url')"));
    }

    #[test]
    fn test_get_imports() {
        let imports = HttpMapper::get_http_imports();
        assert!(!imports.is_empty());
        assert!(imports[0].contains("wasi_fetch"));
    }

    #[test]
    fn test_generate_async_function() {
        let result = HttpMapper::generate_async_http_function(
            "fetch_user",
            vec![("user_id", "i32")],
            "serde_json::Value",
            "let response = WasiFetch::get(&url).await?;\n    let data = response.json()?;"
        );

        assert!(result.contains("async fn fetch_user"));
        assert!(result.contains("user_id: i32"));
        assert!(result.contains("Result<serde_json::Value>"));
    }
}
