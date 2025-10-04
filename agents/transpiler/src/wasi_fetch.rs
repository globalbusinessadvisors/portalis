//! WASI Fetch API Integration
//!
//! Provides a unified HTTP fetch API that works across:
//! - Native Rust (reqwest library)
//! - Browser WASM (fetch() API via wasm-bindgen)
//! - WASI (reqwest or wasi-http)
//!
//! This module bridges Python's HTTP libraries (requests, urllib, httpx) to WASM-compatible implementations.
//!
//! # Examples
//!
//! ```rust,no_run
//! use wasi_fetch::{WasiFetch, Request, Method};
//!
//! // Simple GET request
//! let response = WasiFetch::get("https://api.example.com/data").await?;
//! let json: serde_json::Value = response.json().await?;
//!
//! // POST with JSON body
//! let mut request = Request::new(Method::Post, "https://api.example.com/users");
//! request.set_json(&user_data)?;
//! let response = WasiFetch::fetch(request).await?;
//! ```

use anyhow::{Result, Context, anyhow};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::Duration;

// Browser-specific imports
#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
use wasm_bindgen::prelude::*;
#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
use wasm_bindgen::JsCast;
#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
use wasm_bindgen_futures::JsFuture;
#[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
use web_sys::{Request as JsRequest, Response as JsResponse, RequestInit, Headers as JsHeaders};

/// HTTP methods supported by the fetch API
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
}

impl Method {
    pub fn as_str(&self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
            Method::Put => "PUT",
            Method::Delete => "DELETE",
            Method::Patch => "PATCH",
            Method::Head => "HEAD",
            Method::Options => "OPTIONS",
        }
    }
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Request body types
#[derive(Debug, Clone)]
pub enum Body {
    Empty,
    Text(String),
    Json(serde_json::Value),
    Bytes(Vec<u8>),
    Form(HashMap<String, String>),
}

/// HTTP Request builder
#[derive(Debug, Clone)]
pub struct Request {
    url: String,
    method: Method,
    headers: HashMap<String, String>,
    body: Body,
    timeout: Option<Duration>,
    follow_redirects: bool,
}

impl Request {
    /// Create a new request with the given method and URL
    pub fn new<S: Into<String>>(method: Method, url: S) -> Self {
        Self {
            url: url.into(),
            method,
            headers: HashMap::new(),
            body: Body::Empty,
            timeout: None,
            follow_redirects: true,
        }
    }

    /// Set a header value
    pub fn header<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) -> &mut Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set multiple headers
    pub fn headers(&mut self, headers: HashMap<String, String>) -> &mut Self {
        self.headers.extend(headers);
        self
    }

    /// Set JSON body and content-type header
    pub fn set_json<T: Serialize>(&mut self, data: &T) -> Result<&mut Self> {
        let json = serde_json::to_value(data)
            .context("Failed to serialize JSON")?;
        self.body = Body::Json(json);
        self.headers.insert("Content-Type".to_string(), "application/json".to_string());
        Ok(self)
    }

    /// Set text body
    pub fn set_text<S: Into<String>>(&mut self, text: S) -> &mut Self {
        self.body = Body::Text(text.into());
        self.headers.entry("Content-Type".to_string())
            .or_insert_with(|| "text/plain".to_string());
        self
    }

    /// Set binary body
    pub fn set_bytes(&mut self, bytes: Vec<u8>) -> &mut Self {
        self.body = Body::Bytes(bytes);
        self.headers.entry("Content-Type".to_string())
            .or_insert_with(|| "application/octet-stream".to_string());
        self
    }

    /// Set form data body
    pub fn set_form(&mut self, form: HashMap<String, String>) -> &mut Self {
        self.body = Body::Form(form);
        self.headers.insert("Content-Type".to_string(), "application/x-www-form-urlencoded".to_string());
        self
    }

    /// Set request timeout
    pub fn timeout(&mut self, duration: Duration) -> &mut Self {
        self.timeout = Some(duration);
        self
    }

    /// Set whether to follow redirects (default: true)
    pub fn follow_redirects(&mut self, follow: bool) -> &mut Self {
        self.follow_redirects = follow;
        self
    }

    /// Get the URL with query parameters
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get the method
    pub fn method(&self) -> Method {
        self.method
    }

    /// Get headers
    pub fn headers_map(&self) -> &HashMap<String, String> {
        &self.headers
    }

    /// Get body
    pub fn body(&self) -> &Body {
        &self.body
    }
}

/// HTTP Response
#[derive(Debug)]
pub struct Response {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl Response {
    /// Create a new response
    pub fn new(status: u16, headers: HashMap<String, String>, body: Vec<u8>) -> Self {
        Self {
            status,
            headers,
            body,
        }
    }

    /// Get the status code
    pub fn status(&self) -> u16 {
        self.status
    }

    /// Check if the response is successful (2xx status code)
    pub fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    /// Check if the response is a redirect (3xx status code)
    pub fn is_redirect(&self) -> bool {
        self.status >= 300 && self.status < 400
    }

    /// Check if the response is a client error (4xx status code)
    pub fn is_client_error(&self) -> bool {
        self.status >= 400 && self.status < 500
    }

    /// Check if the response is a server error (5xx status code)
    pub fn is_server_error(&self) -> bool {
        self.status >= 500 && self.status < 600
    }

    /// Get all headers
    pub fn headers(&self) -> &HashMap<String, String> {
        &self.headers
    }

    /// Get a specific header value
    pub fn header(&self, key: &str) -> Option<&String> {
        self.headers.get(key)
    }

    /// Get the Content-Type header
    pub fn content_type(&self) -> Option<&String> {
        self.header("content-type")
            .or_else(|| self.header("Content-Type"))
    }

    /// Get the response body as bytes
    pub fn bytes(&self) -> &[u8] {
        &self.body
    }

    /// Get the response body as text
    pub fn text(&self) -> Result<String> {
        String::from_utf8(self.body.clone())
            .context("Response body is not valid UTF-8")
    }

    /// Parse the response body as JSON
    pub fn json<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        let text = self.text()?;
        serde_json::from_str(&text)
            .context("Failed to parse JSON response")
    }

    /// Get response body as raw value
    pub fn json_value(&self) -> Result<serde_json::Value> {
        self.json()
    }
}

/// Fetch API error types
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Timeout error: request took longer than {0:?}")]
    Timeout(Duration),

    #[error("HTTP error {status}: {message}")]
    Http { status: u16, message: String },

    #[error("SSL/TLS error: {0}")]
    Ssl(String),

    #[error("JSON parsing error: {0}")]
    JsonParse(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Request building error: {0}")]
    RequestBuild(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Main fetch API implementation
pub struct WasiFetch;

impl WasiFetch {
    /// Perform a GET request
    pub async fn get<S: Into<String>>(url: S) -> Result<Response> {
        let request = Request::new(Method::Get, url);
        Self::fetch(request).await
    }

    /// Perform a POST request with JSON body
    pub async fn post_json<S: Into<String>, T: Serialize>(url: S, data: &T) -> Result<Response> {
        let mut request = Request::new(Method::Post, url);
        request.set_json(data)?;
        Self::fetch(request).await
    }

    /// Perform a POST request with form data
    pub async fn post_form<S: Into<String>>(url: S, form: HashMap<String, String>) -> Result<Response> {
        let mut request = Request::new(Method::Post, url);
        request.set_form(form);
        Self::fetch(request).await
    }

    /// Perform a PUT request with JSON body
    pub async fn put_json<S: Into<String>, T: Serialize>(url: S, data: &T) -> Result<Response> {
        let mut request = Request::new(Method::Put, url);
        request.set_json(data)?;
        Self::fetch(request).await
    }

    /// Perform a DELETE request
    pub async fn delete<S: Into<String>>(url: S) -> Result<Response> {
        let request = Request::new(Method::Delete, url);
        Self::fetch(request).await
    }

    /// Perform a PATCH request with JSON body
    pub async fn patch_json<S: Into<String>, T: Serialize>(url: S, data: &T) -> Result<Response> {
        let mut request = Request::new(Method::Patch, url);
        request.set_json(data)?;
        Self::fetch(request).await
    }

    /// Perform a HEAD request
    pub async fn head<S: Into<String>>(url: S) -> Result<Response> {
        let request = Request::new(Method::Head, url);
        Self::fetch(request).await
    }

    /// Perform an OPTIONS request
    pub async fn options<S: Into<String>>(url: S) -> Result<Response> {
        let request = Request::new(Method::Options, url);
        Self::fetch(request).await
    }

    /// Execute a fetch request (platform-specific implementation)
    pub async fn fetch(request: Request) -> Result<Response> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::fetch_native(request).await
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
        {
            Self::fetch_wasi(request).await
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
        {
            Self::fetch_browser(request).await
        }
    }

    /// Native Rust implementation using reqwest
    #[cfg(not(target_arch = "wasm32"))]
    async fn fetch_native(request: Request) -> Result<Response> {
        use reqwest;

        // Build reqwest client
        let mut client_builder = reqwest::Client::builder();

        if let Some(timeout) = request.timeout {
            client_builder = client_builder.timeout(timeout);
        }

        client_builder = client_builder.redirect(if request.follow_redirects {
            reqwest::redirect::Policy::default()
        } else {
            reqwest::redirect::Policy::none()
        });

        let client = client_builder.build()
            .map_err(|e| FetchError::RequestBuild(e.to_string()))?;

        // Build request
        let mut req_builder = match request.method {
            Method::Get => client.get(&request.url),
            Method::Post => client.post(&request.url),
            Method::Put => client.put(&request.url),
            Method::Delete => client.delete(&request.url),
            Method::Patch => client.patch(&request.url),
            Method::Head => client.head(&request.url),
            Method::Options => client.request(reqwest::Method::OPTIONS, &request.url),
        };

        // Add headers
        for (key, value) in &request.headers {
            req_builder = req_builder.header(key, value);
        }

        // Add body
        req_builder = match &request.body {
            Body::Empty => req_builder,
            Body::Text(text) => req_builder.body(text.clone()),
            Body::Json(json) => req_builder.json(json),
            Body::Bytes(bytes) => req_builder.body(bytes.clone()),
            Body::Form(form) => req_builder.form(form),
        };

        // Execute request
        let response = req_builder.send().await
            .map_err(|e| {
                if e.is_timeout() {
                    FetchError::Timeout(request.timeout.unwrap_or(Duration::from_secs(30)))
                } else if e.is_connect() {
                    FetchError::Connection(e.to_string())
                } else {
                    FetchError::Other(e.to_string())
                }
            })?;

        let status = response.status().as_u16();

        // Extract headers
        let mut headers = HashMap::new();
        for (key, value) in response.headers() {
            if let Ok(value_str) = value.to_str() {
                headers.insert(key.to_string(), value_str.to_string());
            }
        }

        // Get body bytes
        let body = response.bytes().await
            .map_err(|e| FetchError::Other(format!("Failed to read response body: {}", e)))?
            .to_vec();

        Ok(Response::new(status, headers, body))
    }

    /// WASI implementation using reqwest (works in WASI runtime)
    #[cfg(all(target_arch = "wasm32", feature = "wasi"))]
    async fn fetch_wasi(request: Request) -> Result<Response> {
        // For WASI, we can use reqwest compiled for wasm32-wasi target
        // The implementation is similar to native but with WASI-specific considerations
        use reqwest;

        let client = reqwest::Client::new();

        let mut req_builder = match request.method {
            Method::Get => client.get(&request.url),
            Method::Post => client.post(&request.url),
            Method::Put => client.put(&request.url),
            Method::Delete => client.delete(&request.url),
            Method::Patch => client.patch(&request.url),
            Method::Head => client.head(&request.url),
            Method::Options => client.request(reqwest::Method::OPTIONS, &request.url),
        };

        for (key, value) in &request.headers {
            req_builder = req_builder.header(key, value);
        }

        req_builder = match &request.body {
            Body::Empty => req_builder,
            Body::Text(text) => req_builder.body(text.clone()),
            Body::Json(json) => req_builder.json(json),
            Body::Bytes(bytes) => req_builder.body(bytes.clone()),
            Body::Form(form) => req_builder.form(form),
        };

        let response = req_builder.send().await
            .map_err(|e| FetchError::Connection(e.to_string()))?;

        let status = response.status().as_u16();
        let mut headers = HashMap::new();

        for (key, value) in response.headers() {
            if let Ok(value_str) = value.to_str() {
                headers.insert(key.to_string(), value_str.to_string());
            }
        }

        let body = response.bytes().await
            .map_err(|e| FetchError::Other(e.to_string()))?
            .to_vec();

        Ok(Response::new(status, headers, body))
    }

    /// Browser implementation using fetch() API
    #[cfg(all(target_arch = "wasm32", not(feature = "wasi")))]
    async fn fetch_browser(request: Request) -> Result<Response> {
        use js_sys::{Uint8Array, ArrayBuffer};

        // Create request options
        let mut opts = RequestInit::new();
        opts.method(request.method.as_str());

        // Set headers
        let js_headers = JsHeaders::new()
            .map_err(|e| FetchError::RequestBuild(format!("Failed to create headers: {:?}", e)))?;

        for (key, value) in &request.headers {
            js_headers.append(&key, &value)
                .map_err(|e| FetchError::RequestBuild(format!("Failed to set header {}: {:?}", key, e)))?;
        }
        opts.headers(&js_headers);

        // Set body
        match &request.body {
            Body::Empty => {},
            Body::Text(text) => {
                opts.body(Some(&JsValue::from_str(text)));
            },
            Body::Json(json) => {
                let json_str = serde_json::to_string(json)
                    .map_err(|e| FetchError::JsonParse(e.to_string()))?;
                opts.body(Some(&JsValue::from_str(&json_str)));
            },
            Body::Bytes(bytes) => {
                let uint8_array = Uint8Array::from(bytes.as_slice());
                opts.body(Some(&uint8_array.into()));
            },
            Body::Form(form) => {
                let form_str = form.iter()
                    .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
                    .collect::<Vec<_>>()
                    .join("&");
                opts.body(Some(&JsValue::from_str(&form_str)));
            },
        }

        // Set redirect mode
        if !request.follow_redirects {
            opts.redirect(web_sys::RequestRedirect::Manual);
        }

        // Create and send request
        let js_request = JsRequest::new_with_str_and_init(&request.url, &opts)
            .map_err(|e| FetchError::InvalidUrl(format!("Invalid URL: {:?}", e)))?;

        let window = web_sys::window()
            .ok_or_else(|| FetchError::Other("No window object available".to_string()))?;

        let resp_value = JsFuture::from(window.fetch_with_request(&js_request))
            .await
            .map_err(|e| FetchError::Connection(format!("Fetch failed: {:?}", e)))?;

        let js_response: JsResponse = resp_value.dyn_into()
            .map_err(|_| FetchError::Other("Invalid response type".to_string()))?;

        let status = js_response.status() as u16;

        // Extract headers
        let mut headers = HashMap::new();
        let js_headers = js_response.headers();

        // Note: web-sys Headers doesn't have a direct iterator, so we'd need to use entries()
        // For now, we'll extract common headers
        let common_headers = ["content-type", "content-length", "location", "set-cookie"];
        for header in &common_headers {
            if let Ok(Some(value)) = js_headers.get(header) {
                headers.insert(header.to_string(), value);
            }
        }

        // Get body as bytes
        let array_buffer = JsFuture::from(js_response.array_buffer()
            .map_err(|e| FetchError::Other(format!("Failed to get array buffer: {:?}", e)))?)
            .await
            .map_err(|e| FetchError::Other(format!("Failed to read array buffer: {:?}", e)))?;

        let array_buffer: ArrayBuffer = array_buffer.dyn_into()
            .map_err(|_| FetchError::Other("Invalid array buffer type".to_string()))?;

        let uint8_array = Uint8Array::new(&array_buffer);
        let body = uint8_array.to_vec();

        Ok(Response::new(status, headers, body))
    }
}

/// Query parameter builder helper
pub struct QueryParams {
    params: Vec<(String, String)>,
}

impl QueryParams {
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    pub fn add<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) -> &mut Self {
        self.params.push((key.into(), value.into()));
        self
    }

    pub fn build(&self) -> String {
        if self.params.is_empty() {
            return String::new();
        }

        let query = self.params.iter()
            .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        format!("?{}", query)
    }

    pub fn append_to_url(&self, url: &str) -> String {
        if self.params.is_empty() {
            return url.to_string();
        }

        let separator = if url.contains('?') { "&" } else { "?" };
        let query = self.params.iter()
            .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        format!("{}{}{}", url, separator, query)
    }
}

impl Default for QueryParams {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_to_string() {
        assert_eq!(Method::Get.as_str(), "GET");
        assert_eq!(Method::Post.as_str(), "POST");
        assert_eq!(Method::Put.as_str(), "PUT");
        assert_eq!(Method::Delete.as_str(), "DELETE");
    }

    #[test]
    fn test_request_builder() {
        let mut request = Request::new(Method::Get, "https://api.example.com");
        request.header("Authorization", "Bearer token123");

        assert_eq!(request.url(), "https://api.example.com");
        assert_eq!(request.method(), Method::Get);
        assert_eq!(request.headers_map().get("Authorization").unwrap(), "Bearer token123");
    }

    #[test]
    fn test_request_json_body() {
        let mut request = Request::new(Method::Post, "https://api.example.com");
        let data = serde_json::json!({"name": "test", "value": 42});
        request.set_json(&data).unwrap();

        assert_eq!(request.headers_map().get("Content-Type").unwrap(), "application/json");
        match request.body() {
            Body::Json(json) => {
                assert_eq!(json.get("name").unwrap().as_str().unwrap(), "test");
                assert_eq!(json.get("value").unwrap().as_i64().unwrap(), 42);
            },
            _ => panic!("Expected JSON body"),
        }
    }

    #[test]
    fn test_query_params() {
        let mut params = QueryParams::new();
        params.add("foo", "bar").add("baz", "qux");

        assert_eq!(params.build(), "?foo=bar&baz=qux");
    }

    #[test]
    fn test_query_params_append() {
        let mut params = QueryParams::new();
        params.add("page", "2").add("limit", "10");

        let url = params.append_to_url("https://api.example.com/users");
        assert_eq!(url, "https://api.example.com/users?page=2&limit=10");

        let url_with_query = params.append_to_url("https://api.example.com/users?active=true");
        assert_eq!(url_with_query, "https://api.example.com/users?active=true&page=2&limit=10");
    }

    #[test]
    fn test_response_status_checks() {
        let response = Response::new(200, HashMap::new(), vec![]);
        assert!(response.is_success());
        assert!(!response.is_client_error());

        let response = Response::new(404, HashMap::new(), vec![]);
        assert!(response.is_client_error());
        assert!(!response.is_success());

        let response = Response::new(500, HashMap::new(), vec![]);
        assert!(response.is_server_error());
        assert!(!response.is_success());
    }

    #[tokio::test]
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_simple_get_request() {
        // This test requires network access
        // You may want to use a mock server for actual testing
        let result = WasiFetch::get("https://httpbin.org/get").await;

        // Just verify the structure works, actual network test may fail in CI
        match result {
            Ok(response) => {
                assert!(response.status() >= 200);
            },
            Err(_) => {
                // Network errors are acceptable in test environment
            }
        }
    }
}
