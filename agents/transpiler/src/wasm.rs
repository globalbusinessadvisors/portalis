//! WASM bindings for Python-to-Rust transpiler
//! Provides JavaScript-compatible interface for browser and Node.js

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use crate::feature_translator::FeatureTranslator;

#[wasm_bindgen]
pub struct TranspilerWasm {
    translator: FeatureTranslator,
}

#[wasm_bindgen]
impl TranspilerWasm {
    /// Create a new transpiler instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Set panic hook for better error messages in browser console
        console_error_panic_hook::set_once();

        Self {
            translator: FeatureTranslator::new(),
        }
    }

    /// Translate Python source code to Rust
    ///
    /// # Arguments
    /// * `python_source` - Python source code as string
    ///
    /// # Returns
    /// * Rust source code as string, or error
    #[wasm_bindgen]
    pub fn translate(&mut self, python_source: &str) -> Result<String, JsValue> {
        self.translator
            .translate(python_source)
            .map_err(|e| JsValue::from_str(&format!("Translation error: {:?}", e)))
    }

    /// Translate with detailed metadata
    ///
    /// Returns JSON object with rust_code, line count, and success status
    #[wasm_bindgen]
    pub fn translate_detailed(&mut self, python_source: &str) -> Result<JsValue, JsValue> {
        let rust_code = self.translator
            .translate(python_source)
            .map_err(|e| JsValue::from_str(&format!("Translation error: {:?}", e)))?;

        let result = TranslationResult {
            rust_code: rust_code.clone(),
            lines: rust_code.lines().count(),
            success: true,
        };

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {:?}", e)))
    }

    /// Get transpiler version
    #[wasm_bindgen]
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

#[derive(Serialize, Deserialize)]
pub struct TranslationResult {
    pub rust_code: String,
    pub lines: usize,
    pub success: bool,
}

// Ensure proper initialization for WASM
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}
