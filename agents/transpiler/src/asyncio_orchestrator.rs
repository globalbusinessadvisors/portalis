//! Full Asyncio Support - Orchestrator Module
//!
//! This module provides a unified interface for translating Python asyncio patterns
//! to Rust async/await, integrating all asyncio components and adding advanced patterns.

use crate::py_to_rust_asyncio::{
    AsyncContextMapper, AsyncFunctionGenerator, AsyncImportGenerator, AsyncIteratorMapper,
    AsyncSyncMapper, AsyncioMapper, AsyncioPatternDetector,
};
use crate::python_ast::{FunctionParam, PyExpr, PyStmt, TypeAnnotation};
use std::collections::HashMap;

/// Comprehensive asyncio orchestrator
pub struct AsyncioOrchestrator {
    /// Track async context (are we inside an async function?)
    in_async_context: bool,
    /// Track spawned tasks
    spawned_tasks: Vec<String>,
    /// Track required imports
    required_imports: HashMap<String, Vec<String>>,
    /// Configuration
    config: AsyncioConfig,
}

/// Configuration for asyncio translation
#[derive(Debug, Clone)]
pub struct AsyncioConfig {
    /// Target runtime (tokio, async-std, smol)
    pub runtime: AsyncRuntime,
    /// Enable automatic error propagation
    pub auto_error_handling: bool,
    /// Use tracing for async debugging
    pub enable_tracing: bool,
    /// Generate cancellation support
    pub cancellation_support: bool,
    /// WASM compatibility mode
    pub wasm_compatible: bool,
}

#[derive(Debug, Clone)]
pub enum AsyncRuntime {
    Tokio,
    AsyncStd,
    Smol,
    Wasm,
}

impl Default for AsyncioConfig {
    fn default() -> Self {
        Self {
            runtime: AsyncRuntime::Tokio,
            auto_error_handling: true,
            enable_tracing: false,
            cancellation_support: false,
            wasm_compatible: false,
        }
    }
}

impl AsyncioOrchestrator {
    pub fn new(config: AsyncioConfig) -> Self {
        Self {
            in_async_context: false,
            spawned_tasks: Vec::new(),
            required_imports: HashMap::new(),
            config,
        }
    }

    pub fn with_default() -> Self {
        Self::new(AsyncioConfig::default())
    }

    /// Translate a complete async module
    pub fn translate_async_module(&mut self, statements: &[PyStmt]) -> String {
        let mut output = String::new();

        // Generate imports
        output.push_str(&self.generate_imports());
        output.push_str("\n\n");

        // Translate each statement
        for stmt in statements {
            output.push_str(&self.translate_statement(stmt));
            output.push_str("\n\n");
        }

        output
    }

    /// Translate a single statement
    pub fn translate_statement(&mut self, stmt: &PyStmt) -> String {
        match stmt {
            PyStmt::FunctionDef {
                name,
                params,
                body,
                return_type,
                is_async,
                ..
            } => self.translate_async_function(name, params, body, return_type, *is_async),
            PyStmt::Expr(expr) => self.translate_expression(expr),
            _ => format!("// TODO: Translate {:?}", stmt),
        }
    }

    /// Translate async function definition
    fn translate_async_function(
        &mut self,
        name: &str,
        params: &[FunctionParam],
        body: &[PyStmt],
        return_type: &Option<TypeAnnotation>,
        is_async: bool,
    ) -> String {
        // Track async context
        let previous_context = self.in_async_context;
        self.in_async_context = is_async;

        let mut code = String::new();

        // Add tracing if enabled
        if self.config.enable_tracing && is_async {
            code.push_str("#[tracing::instrument]\n");
        }

        // Function signature
        let async_keyword = if is_async { "async " } else { "" };
        let param_str = self.format_params(params);
        let return_str = self.format_return_type(return_type, is_async);

        code.push_str(&format!(
            "pub {}fn {}({}) {} {{\n",
            async_keyword, name, param_str, return_str
        ));

        // Add cancellation token if enabled
        if self.config.cancellation_support && is_async {
            code.push_str("    let cancel_token = tokio_util::sync::CancellationToken::new();\n");
        }

        // Translate body
        for stmt in body {
            let translated = self.translate_statement(stmt);
            for line in translated.lines() {
                code.push_str(&format!("    {}\n", line));
            }
        }

        code.push_str("}\n");

        // Restore context
        self.in_async_context = previous_context;

        code
    }

    /// Translate expression with async awareness
    fn translate_expression(&mut self, expr: &PyExpr) -> String {
        match expr {
            PyExpr::Await(inner) => self.translate_await_expr(inner),
            PyExpr::Call { func, args, kwargs } => {
                self.translate_call_expr(func, args, kwargs)
            }
            _ => format!("// TODO: Translate expr {:?}", expr),
        }
    }

    /// Translate await expression
    fn translate_await_expr(&mut self, expr: &PyExpr) -> String {
        let inner = self.translate_expression(expr);

        if self.config.auto_error_handling {
            format!("{}.await?", inner)
        } else {
            format!("{}.await", inner)
        }
    }

    /// Translate async function calls
    fn translate_call_expr(
        &mut self,
        func: &PyExpr,
        args: &[PyExpr],
        _kwargs: &HashMap<String, PyExpr>,
    ) -> String {
        // Check if it's an asyncio call
        if let PyExpr::Attribute { value, attr } = func {
            if let PyExpr::Name(module) = value.as_ref() {
                if module == "asyncio" {
                    return self.translate_asyncio_call(attr, args);
                }
            }
        }

        format!("// TODO: Translate call")
    }

    /// Translate asyncio-specific calls
    fn translate_asyncio_call(&mut self, method: &str, args: &[PyExpr]) -> String {
        match method {
            "run" => self.translate_asyncio_run(args),
            "create_task" | "ensure_future" => self.translate_create_task(args),
            "gather" => self.translate_gather(args),
            "sleep" => self.translate_sleep(args),
            "wait_for" => self.translate_wait_for(args),
            "wait" => self.translate_wait(args),
            "as_completed" => self.translate_as_completed(args),
            "shield" => self.translate_shield(args),
            _ => format!("// asyncio.{}", method),
        }
    }

    fn translate_asyncio_run(&mut self, _args: &[PyExpr]) -> String {
        self.add_import("tokio", vec!["tokio::main".to_string()]);
        "#[tokio::main]\nasync fn main() -> Result<()> {\n    // Main logic\n    Ok(())\n}".to_string()
    }

    fn translate_create_task(&mut self, args: &[PyExpr]) -> String {
        self.add_import("tokio", vec!["tokio::spawn".to_string()]);
        let task_expr = if !args.is_empty() {
            format!("{:?}", args[0])
        } else {
            "task()".to_string()
        };

        let task_name = format!("task_{}", self.spawned_tasks.len());
        self.spawned_tasks.push(task_name.clone());

        format!("let {} = tokio::spawn({})", task_name, task_expr)
    }

    fn translate_gather(&mut self, args: &[PyExpr]) -> String {
        self.add_import("tokio", vec!["tokio::join".to_string()]);

        let tasks = args
            .iter()
            .map(|arg| format!("{:?}", arg))
            .collect::<Vec<_>>()
            .join(", ");

        format!("let results = tokio::join!({});", tasks)
    }

    fn translate_sleep(&mut self, args: &[PyExpr]) -> String {
        self.add_import("tokio::time", vec!["tokio::time::sleep".to_string()]);

        let duration = if !args.is_empty() {
            format!("{:?}", args[0])
        } else {
            "1.0".to_string()
        };

        format!(
            "tokio::time::sleep(tokio::time::Duration::from_secs_f64({})).await",
            duration
        )
    }

    fn translate_wait_for(&mut self, args: &[PyExpr]) -> String {
        self.add_import("tokio::time", vec!["tokio::time::timeout".to_string()]);

        let (future, timeout) = if args.len() >= 2 {
            (format!("{:?}", args[0]), format!("{:?}", args[1]))
        } else {
            ("future()".to_string(), "10.0".to_string())
        };

        format!(
            "tokio::time::timeout(tokio::time::Duration::from_secs_f64({}), {})",
            timeout, future
        )
    }

    fn translate_wait(&mut self, args: &[PyExpr]) -> String {
        self.add_import("futures", vec!["futures::future::select_all".to_string()]);

        let tasks = if !args.is_empty() {
            format!("{:?}", args[0])
        } else {
            "tasks".to_string()
        };

        format!("let (result, _index, remaining) = futures::future::select_all({}).await;", tasks)
    }

    fn translate_as_completed(&mut self, args: &[PyExpr]) -> String {
        self.add_import("futures", vec![
            "futures::stream::FuturesUnordered".to_string(),
            "futures::StreamExt".to_string(),
        ]);

        let tasks = if !args.is_empty() {
            format!("{:?}", args[0])
        } else {
            "tasks".to_string()
        };

        format!(
            "let mut stream = futures::stream::FuturesUnordered::from_iter({});\nwhile let Some(result) = stream.next().await {{ /* process */ }}",
            tasks
        )
    }

    fn translate_shield(&mut self, args: &[PyExpr]) -> String {
        // Shield prevents cancellation - in Rust, we can use spawn + abort handle
        self.add_import("tokio", vec!["tokio::spawn".to_string()]);

        let task = if !args.is_empty() {
            format!("{:?}", args[0])
        } else {
            "task()".to_string()
        };

        format!(
            "let handle = tokio::spawn({});\n// Handle cannot be cancelled from parent",
            task
        )
    }

    // Helper methods

    fn add_import(&mut self, crate_name: &str, items: Vec<String>) {
        self.required_imports
            .entry(crate_name.to_string())
            .or_insert_with(Vec::new)
            .extend(items);
    }

    fn generate_imports(&self) -> String {
        let mut imports = Vec::new();

        for (crate_name, items) in &self.required_imports {
            for item in items {
                imports.push(format!("use {};", item));
            }
        }

        // Add standard async imports based on runtime
        match self.config.runtime {
            AsyncRuntime::Tokio => {
                imports.insert(0, "use tokio;".to_string());
            }
            AsyncRuntime::AsyncStd => {
                imports.insert(0, "use async_std;".to_string());
            }
            AsyncRuntime::Smol => {
                imports.insert(0, "use smol;".to_string());
            }
            AsyncRuntime::Wasm => {
                imports.insert(0, "use wasm_bindgen_futures;".to_string());
            }
        }

        imports.join("\n")
    }

    fn format_params(&self, params: &[FunctionParam]) -> String {
        params
            .iter()
            .map(|param| {
                let rust_type = self.python_type_to_rust(param.type_annotation.as_ref());
                format!("{}: {}", param.name, rust_type)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn format_return_type(&self, return_type: &Option<TypeAnnotation>, is_async: bool) -> String {
        let base_type = match return_type {
            Some(t) => self.python_type_to_rust(Some(t)),
            None => "()".to_string(),
        };

        if is_async && self.config.auto_error_handling {
            format!("-> Result<{}>", base_type)
        } else if is_async {
            format!("-> {}", base_type)
        } else {
            format!("-> {}", base_type)
        }
    }

    fn python_type_to_rust(&self, type_ann: Option<&TypeAnnotation>) -> String {
        match type_ann {
            Some(TypeAnnotation::Name(name)) => match name.as_str() {
                "int" => "i32",
                "float" => "f64",
                "str" => "String",
                "bool" => "bool",
                "None" => "()",
                _ => name,
            }
            .to_string(),
            Some(TypeAnnotation::Generic { base, args }) => {
                let base_type = self.python_type_to_rust(Some(base));
                let arg_types = args
                    .iter()
                    .map(|arg| self.python_type_to_rust(Some(arg)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}<{}>", base_type, arg_types)
            }
            None => "()".to_string(),
        }
    }
}

/// Generate comprehensive asyncio patterns
pub struct AsyncioPatterns;

impl AsyncioPatterns {
    /// Generate error handling wrapper for async functions
    pub fn generate_error_wrapper(function_name: &str, body: &str) -> String {
        format!(
            r#"async fn {}() -> Result<()> {{
    match async {{
{}
    }}.await {{
        Ok(result) => Ok(result),
        Err(e) => {{
            tracing::error!("Error in {}: {{:?}}", e);
            Err(e)
        }}
    }}
}}"#,
            function_name, body, function_name
        )
    }

    /// Generate retry pattern for async operations
    pub fn generate_retry_pattern(operation: &str, max_retries: u32) -> String {
        format!(
            r#"async fn retry_operation() -> Result<Response> {{
    let mut attempts = 0;
    loop {{
        attempts += 1;
        match {} {{
            Ok(result) => return Ok(result),
            Err(e) if attempts < {} => {{
                let backoff = std::time::Duration::from_millis(100 * 2u64.pow(attempts - 1));
                tokio::time::sleep(backoff).await;
                continue;
            }}
            Err(e) => return Err(e),
        }}
    }}
}}"#,
            operation, max_retries
        )
    }

    /// Generate timeout pattern with graceful degradation
    pub fn generate_timeout_with_fallback(
        operation: &str,
        timeout_secs: f64,
        fallback: &str,
    ) -> String {
        format!(
            r#"match tokio::time::timeout(
    tokio::time::Duration::from_secs_f64({}),
    {}
).await {{
    Ok(result) => result,
    Err(_timeout) => {{
        tracing::warn!("Operation timed out, using fallback");
        {}
    }}
}}"#,
            timeout_secs, operation, fallback
        )
    }

    /// Generate async select pattern (race between futures)
    pub fn generate_select_pattern(futures: Vec<&str>) -> String {
        let branches = futures
            .iter()
            .enumerate()
            .map(|(i, f)| format!("    result{} = {} => {{ result{} }}", i, f, i))
            .collect::<Vec<_>>()
            .join(",\n");

        format!(
            r#"tokio::select! {{
{}
}}"#,
            branches
        )
    }

    /// Generate async stream pattern
    pub fn generate_stream_pattern(source: &str, transform: &str) -> String {
        format!(
            r#"use futures::{{stream, StreamExt}};

let stream = {}
    .map(|item| {})
    .buffer_unordered(10); // Process up to 10 items concurrently

tokio::pin!(stream);
while let Some(result) = stream.next().await {{
    // Process result
}}"#,
            source, transform
        )
    }

    /// Generate broadcast channel pattern
    pub fn generate_broadcast_pattern() -> String {
        r#"use tokio::sync::broadcast;

let (tx, _rx) = broadcast::channel(100);

// Sender
tx.send(event).unwrap();

// Receiver
let mut rx = tx.subscribe();
while let Ok(event) = rx.recv().await {
    // Process event
}"#
        .to_string()
    }

    /// Generate graceful shutdown pattern
    pub fn generate_shutdown_pattern() -> String {
        r#"use tokio::signal;

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, starting graceful shutdown");
}"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = AsyncioOrchestrator::with_default();
        assert!(!orchestrator.in_async_context);
        assert_eq!(orchestrator.spawned_tasks.len(), 0);
    }

    #[test]
    fn test_import_tracking() {
        let mut orchestrator = AsyncioOrchestrator::with_default();
        orchestrator.add_import("tokio", vec!["tokio::spawn".to_string()]);

        assert!(orchestrator.required_imports.contains_key("tokio"));
    }

    #[test]
    fn test_error_wrapper_generation() {
        let wrapper = AsyncioPatterns::generate_error_wrapper("test_func", "    // body");
        assert!(wrapper.contains("async fn test_func"));
        assert!(wrapper.contains("Result<()>"));
    }

    #[test]
    fn test_retry_pattern() {
        let retry = AsyncioPatterns::generate_retry_pattern("fetch_data()", 3);
        assert!(retry.contains("max_retries"));
        assert!(retry.contains("backoff"));
    }
}
