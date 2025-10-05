//! Python Decorator Translation to Rust
//!
//! This module provides comprehensive translation of Python decorators to Rust equivalents.
//! Python decorators are syntactic sugar for higher-order functions, while Rust uses
//! various patterns including macros, wrapper functions, and traits.

use crate::python_ast::{PyExpr, TypeAnnotation};
use std::collections::HashMap;

/// Information about a decorator
#[derive(Debug, Clone)]
pub struct DecoratorInfo {
    pub name: String,
    pub args: Vec<PyExpr>,
    pub kwargs: HashMap<String, PyExpr>,
}

/// Represents different decorator translation strategies
#[derive(Debug, Clone)]
pub enum DecoratorStrategy {
    /// Translates to a Rust attribute (e.g., #[inline])
    Attribute(String),
    /// Translates to a wrapper function
    WrapperFunction(String),
    /// Translates to a macro
    Macro(String),
    /// Translates to a trait implementation
    Trait(String),
    /// Custom code generation
    Custom(String),
    /// Comment only (unsupported)
    Comment(String),
}

/// Main decorator translator
pub struct DecoratorTranslator {
    /// Known decorator mappings
    mappings: HashMap<String, DecoratorStrategy>,
    /// Whether to generate wrapper functions
    generate_wrappers: bool,
}

impl Default for DecoratorTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl DecoratorTranslator {
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // Built-in decorators
        mappings.insert(
            "property".to_string(),
            DecoratorStrategy::Trait("getter implementation".to_string()),
        );
        mappings.insert(
            "staticmethod".to_string(),
            DecoratorStrategy::Comment("static method - no self parameter".to_string()),
        );
        mappings.insert(
            "classmethod".to_string(),
            DecoratorStrategy::Comment("class method - use associated function".to_string()),
        );

        // Functools decorators
        mappings.insert(
            "lru_cache".to_string(),
            DecoratorStrategy::WrapperFunction("cached".to_string()),
        );
        mappings.insert(
            "cache".to_string(),
            DecoratorStrategy::WrapperFunction("cached".to_string()),
        );
        mappings.insert(
            "wraps".to_string(),
            DecoratorStrategy::Attribute("#[inline]".to_string()),
        );
        mappings.insert(
            "total_ordering".to_string(),
            DecoratorStrategy::Attribute("#[derive(PartialOrd, Ord)]".to_string()),
        );

        // Timing decorators
        mappings.insert(
            "timeit".to_string(),
            DecoratorStrategy::WrapperFunction("timed".to_string()),
        );
        mappings.insert(
            "time_it".to_string(),
            DecoratorStrategy::WrapperFunction("timed".to_string()),
        );

        // Logging decorators
        mappings.insert(
            "log".to_string(),
            DecoratorStrategy::WrapperFunction("logged".to_string()),
        );
        mappings.insert(
            "debug".to_string(),
            DecoratorStrategy::WrapperFunction("debug_logged".to_string()),
        );

        // Validation decorators
        mappings.insert(
            "validate".to_string(),
            DecoratorStrategy::WrapperFunction("validated".to_string()),
        );
        mappings.insert(
            "require".to_string(),
            DecoratorStrategy::WrapperFunction("require".to_string()),
        );

        // Async decorators
        mappings.insert(
            "async_timeout".to_string(),
            DecoratorStrategy::WrapperFunction("timeout".to_string()),
        );
        mappings.insert(
            "retry".to_string(),
            DecoratorStrategy::WrapperFunction("retry".to_string()),
        );

        // Testing decorators
        mappings.insert(
            "pytest.fixture".to_string(),
            DecoratorStrategy::Attribute("#[test]".to_string()),
        );
        mappings.insert(
            "unittest.mock.patch".to_string(),
            DecoratorStrategy::Comment("mock decorator".to_string()),
        );

        // Performance decorators
        mappings.insert(
            "profile".to_string(),
            DecoratorStrategy::WrapperFunction("profiled".to_string()),
        );
        mappings.insert(
            "memoize".to_string(),
            DecoratorStrategy::WrapperFunction("memoized".to_string()),
        );

        // Deprecation
        mappings.insert(
            "deprecated".to_string(),
            DecoratorStrategy::Attribute("#[deprecated]".to_string()),
        );

        Self {
            mappings,
            generate_wrappers: true,
        }
    }

    /// Parse a decorator expression into DecoratorInfo
    pub fn parse_decorator(&self, decorator: &PyExpr) -> DecoratorInfo {
        match decorator {
            PyExpr::Name(name) => DecoratorInfo {
                name: name.clone(),
                args: vec![],
                kwargs: HashMap::new(),
            },
            PyExpr::Call { func, args, kwargs } => {
                let name = match func.as_ref() {
                    PyExpr::Name(n) => n.clone(),
                    PyExpr::Attribute { value, attr } => {
                        // Handle module.decorator (e.g., pytest.fixture)
                        if let PyExpr::Name(module) = value.as_ref() {
                            format!("{}.{}", module, attr)
                        } else {
                            attr.clone()
                        }
                    }
                    _ => "unknown".to_string(),
                };

                DecoratorInfo {
                    name,
                    args: args.clone(),
                    kwargs: kwargs.clone(),
                }
            }
            PyExpr::Attribute { value, attr } => {
                if let PyExpr::Name(module) = value.as_ref() {
                    DecoratorInfo {
                        name: format!("{}.{}", module, attr),
                        args: vec![],
                        kwargs: HashMap::new(),
                    }
                } else {
                    DecoratorInfo {
                        name: attr.clone(),
                        args: vec![],
                        kwargs: HashMap::new(),
                    }
                }
            }
            _ => DecoratorInfo {
                name: "unknown".to_string(),
                args: vec![],
                kwargs: HashMap::new(),
            },
        }
    }

    /// Translate a decorator to Rust code
    pub fn translate_decorator(&self, decorator: &PyExpr) -> String {
        let info = self.parse_decorator(decorator);

        if let Some(strategy) = self.mappings.get(&info.name) {
            match strategy {
                DecoratorStrategy::Attribute(attr) => attr.clone(),
                DecoratorStrategy::WrapperFunction(wrapper) => {
                    format!("// @{} -> use {} wrapper", info.name, wrapper)
                }
                DecoratorStrategy::Macro(mac) => {
                    format!("{}!", mac)
                }
                DecoratorStrategy::Trait(trait_impl) => {
                    format!("// @{} -> {}", info.name, trait_impl)
                }
                DecoratorStrategy::Custom(code) => code.clone(),
                DecoratorStrategy::Comment(comment) => {
                    format!("// @{}: {}", info.name, comment)
                }
            }
        } else {
            // Unknown decorator - preserve as comment
            if !info.args.is_empty() || !info.kwargs.is_empty() {
                format!("// @{}(...)", info.name)
            } else {
                format!("// @{}", info.name)
            }
        }
    }

    /// Generate wrapper function for decorator pattern
    pub fn generate_wrapper_function(
        &self,
        decorator_name: &str,
        function_name: &str,
        params: &[(String, Option<TypeAnnotation>)],
        return_type: &Option<TypeAnnotation>,
        original_body: &str,
    ) -> String {
        match decorator_name {
            "timeit" | "time_it" => {
                self.generate_timing_wrapper(function_name, params, return_type, original_body)
            }
            "log" | "debug" => {
                self.generate_logging_wrapper(decorator_name, function_name, params, return_type, original_body)
            }
            "lru_cache" | "cache" | "memoize" => {
                self.generate_cache_wrapper(function_name, params, return_type, original_body)
            }
            "retry" => {
                self.generate_retry_wrapper(function_name, params, return_type, original_body)
            }
            _ => String::new(),
        }
    }

    fn generate_timing_wrapper(
        &self,
        function_name: &str,
        params: &[(String, Option<TypeAnnotation>)],
        return_type: &Option<TypeAnnotation>,
        body: &str,
    ) -> String {
        let param_list = self.format_params(params);
        let return_type_str = self.format_return_type(return_type);

        format!(
            r#"pub fn {}<{}>({}) {} {{
    let _start = std::time::Instant::now();
    let result = {{
{}
    }};
    let _duration = _start.elapsed();
    tracing::debug!("{} executed in {{:?}}", _duration);
    result
}}"#,
            function_name,
            self.extract_generic_params(params),
            param_list,
            return_type_str,
            self.indent_body(body, 2),
            function_name
        )
    }

    fn generate_logging_wrapper(
        &self,
        decorator_name: &str,
        function_name: &str,
        params: &[(String, Option<TypeAnnotation>)],
        return_type: &Option<TypeAnnotation>,
        body: &str,
    ) -> String {
        let param_list = self.format_params(params);
        let return_type_str = self.format_return_type(return_type);
        let log_level = if decorator_name == "debug" { "debug" } else { "info" };

        format!(
            r#"pub fn {}<{}>({}) {} {{
    tracing::{}!("Entering {}", {});
    let result = {{
{}
    }};
    tracing::{}!("Exiting {}", {{:?}}", result);
    result
}}"#,
            function_name,
            self.extract_generic_params(params),
            param_list,
            return_type_str,
            log_level,
            function_name,
            self.format_param_names(params),
            self.indent_body(body, 2),
            log_level,
            function_name
        )
    }

    fn generate_cache_wrapper(
        &self,
        function_name: &str,
        params: &[(String, Option<TypeAnnotation>)],
        return_type: &Option<TypeAnnotation>,
        body: &str,
    ) -> String {
        let param_list = self.format_params(params);
        let return_type_str = self.format_return_type(return_type);

        format!(
            r#"// Note: Use lazy_static and HashMap for caching
use lazy_static::lazy_static;
use std::sync::Mutex;
use std::collections::HashMap;

lazy_static! {{
    static ref {}_CACHE: Mutex<HashMap<String, {}>> = Mutex::new(HashMap::new());
}}

pub fn {}<{}>({}) {} {{
    let cache_key = format!("{{:?}}", ({},));

    // Check cache
    if let Ok(cache) = {}_CACHE.lock() {{
        if let Some(cached) = cache.get(&cache_key) {{
            return cached.clone();
        }}
    }}

    // Compute result
    let result = {{
{}
    }};

    // Store in cache
    if let Ok(mut cache) = {}_CACHE.lock() {{
        cache.insert(cache_key, result.clone());
    }}

    result
}}"#,
            function_name.to_uppercase(),
            self.extract_return_type_name(return_type),
            function_name,
            self.extract_generic_params(params),
            param_list,
            return_type_str,
            self.format_param_names(params),
            function_name.to_uppercase(),
            self.indent_body(body, 2),
            function_name.to_uppercase()
        )
    }

    fn generate_retry_wrapper(
        &self,
        function_name: &str,
        params: &[(String, Option<TypeAnnotation>)],
        return_type: &Option<TypeAnnotation>,
        body: &str,
    ) -> String {
        let param_list = self.format_params(params);
        let return_type_str = self.format_return_type(return_type);

        format!(
            r#"pub fn {}<{}>({}) {} {{
    const MAX_RETRIES: u32 = 3;
    let mut attempts = 0;

    loop {{
        attempts += 1;
        match (|| {{
{}
        }})() {{
            Ok(result) => return Ok(result),
            Err(e) if attempts < MAX_RETRIES => {{
                tracing::warn!("{} failed (attempt {{}}): {{:?}}", attempts, e);
                std::thread::sleep(std::time::Duration::from_millis(100 * attempts as u64));
                continue;
            }}
            Err(e) => return Err(e),
        }}
    }}
}}"#,
            function_name,
            self.extract_generic_params(params),
            param_list,
            return_type_str,
            self.indent_body(body, 3),
            function_name
        )
    }

    // Helper functions

    fn format_params(&self, params: &[(String, Option<TypeAnnotation>)]) -> String {
        params
            .iter()
            .map(|(name, type_ann)| {
                let rust_type = self.python_type_to_rust(type_ann.as_ref());
                format!("{}: {}", name, rust_type)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn format_param_names(&self, params: &[(String, Option<TypeAnnotation>)]) -> String {
        params
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn format_return_type(&self, return_type: &Option<TypeAnnotation>) -> String {
        match return_type {
            Some(t) => format!("-> {}", self.python_type_to_rust(Some(t))),
            None => "-> ()".to_string(),
        }
    }

    fn extract_generic_params(&self, _params: &[(String, Option<TypeAnnotation>)]) -> String {
        // For now, no generics - can be enhanced
        String::new()
    }

    fn extract_return_type_name(&self, return_type: &Option<TypeAnnotation>) -> String {
        match return_type {
            Some(TypeAnnotation::Name(name)) => self.python_type_to_rust_name(name),
            _ => "()".to_string(),
        }
    }

    fn python_type_to_rust(&self, type_ann: Option<&TypeAnnotation>) -> String {
        match type_ann {
            Some(TypeAnnotation::Name(name)) => self.python_type_to_rust_name(name),
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

    fn python_type_to_rust_name(&self, name: &str) -> String {
        match name {
            "int" => "i32",
            "float" => "f64",
            "str" => "String",
            "bool" => "bool",
            "list" => "Vec",
            "dict" => "HashMap",
            "set" => "HashSet",
            "tuple" => "tuple",
            "None" => "()",
            _ => name,
        }
        .to_string()
    }

    fn indent_body(&self, body: &str, level: usize) -> String {
        let indent = "    ".repeat(level);
        body.lines()
            .map(|line| format!("{}{}", indent, line))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Generate decorator helper functions for common patterns
pub fn generate_decorator_helpers() -> String {
    r#"// Decorator helper functions

/// Timing decorator helper
pub fn time_function<F, R>(name: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    tracing::debug!("{} executed in {:?}", name, duration);
    result
}

/// Logging decorator helper
pub fn log_function<F, R>(name: &str, f: F) -> R
where
    F: FnOnce() -> R,
    R: std::fmt::Debug,
{
    tracing::info!("Entering {}", name);
    let result = f();
    tracing::info!("Exiting {} with {:?}", name, result);
    result
}

/// Retry decorator helper
pub fn retry_function<F, R, E>(max_retries: u32, f: F) -> Result<R, E>
where
    F: Fn() -> Result<R, E>,
    E: std::fmt::Debug,
{
    let mut attempts = 0;
    loop {
        attempts += 1;
        match f() {
            Ok(result) => return Ok(result),
            Err(e) if attempts < max_retries => {
                tracing::warn!("Attempt {} failed: {:?}", attempts, e);
                std::thread::sleep(std::time::Duration::from_millis(100 * attempts as u64));
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_decorator() {
        let translator = DecoratorTranslator::new();
        let decorator = PyExpr::Name("staticmethod".to_string());
        let info = translator.parse_decorator(&decorator);

        assert_eq!(info.name, "staticmethod");
        assert!(info.args.is_empty());
        assert!(info.kwargs.is_empty());
    }

    #[test]
    fn test_translate_property_decorator() {
        let translator = DecoratorTranslator::new();
        let decorator = PyExpr::Name("property".to_string());
        let result = translator.translate_decorator(&decorator);

        assert!(result.contains("getter"));
    }

    #[test]
    fn test_translate_lru_cache() {
        let translator = DecoratorTranslator::new();
        let decorator = PyExpr::Name("lru_cache".to_string());
        let result = translator.translate_decorator(&decorator);

        assert!(result.contains("cached") || result.contains("lru_cache"));
    }
}
