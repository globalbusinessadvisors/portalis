//! Feature Detection Engine
//!
//! Detects Python language features and categorizes them by support level.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// Re-export types from ingest (to avoid circular dependency, we define minimal types here)
// In production, these would be shared through a common types crate

/// Simplified Python AST (matches portalis_ingest::PythonAst)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonAst {
    pub functions: Vec<PythonFunction>,
    pub classes: Vec<PythonClass>,
    pub imports: Vec<PythonImport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonFunction {
    pub name: String,
    pub params: Vec<PythonParameter>,
    pub return_type: Option<String>,
    pub body: String,
    pub decorators: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonParameter {
    pub name: String,
    pub type_hint: Option<String>,
    pub default: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonClass {
    pub name: String,
    pub bases: Vec<String>,
    pub methods: Vec<PythonFunction>,
    pub attributes: Vec<PythonAttribute>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonAttribute {
    pub name: String,
    pub type_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonImport {
    pub module: String,
    pub items: Vec<String>,
    pub alias: Option<String>,
}

/// Support level for detected features
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureSupport {
    /// Fully supported with complete translation
    Full,
    /// Partially supported with limitations
    Partial,
    /// Not supported (blocker)
    None,
}

/// Detected feature in Python code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedFeature {
    pub category: FeatureCategory,
    pub name: String,
    pub support: FeatureSupport,
    pub count: usize,
    pub locations: Vec<FeatureLocation>,
    pub details: Option<String>,
}

/// Feature category
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureCategory {
    Function,
    Class,
    Decorator,
    TypeHint,
    Import,
    AsyncAwait,
    Metaclass,
    DynamicFeature,
    MagicMethod,
    Comprehension,
    Generator,
    ContextManager,
    Other,
}

/// Location of a detected feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureLocation {
    pub file: String,
    pub line: Option<usize>,
    pub context: String,
}

/// Set of all detected features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub features: Vec<DetectedFeature>,
    pub summary: FeatureSummary,
}

/// Summary of detected features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSummary {
    pub total_features: usize,
    pub fully_supported: usize,
    pub partially_supported: usize,
    pub unsupported: usize,
    pub by_category: HashMap<FeatureCategory, usize>,
}

/// Feature detection engine
pub struct FeatureDetector {
    /// Known decorators and their support levels
    decorator_support: HashMap<String, FeatureSupport>,
    /// Known magic methods and their support levels
    magic_method_support: HashMap<String, FeatureSupport>,
}

impl FeatureDetector {
    pub fn new() -> Self {
        let mut decorator_support = HashMap::new();
        // Fully supported decorators
        decorator_support.insert("property".to_string(), FeatureSupport::Full);
        decorator_support.insert("staticmethod".to_string(), FeatureSupport::Full);
        decorator_support.insert("classmethod".to_string(), FeatureSupport::Full);
        // Partially supported
        decorator_support.insert("dataclass".to_string(), FeatureSupport::Partial);
        decorator_support.insert("lru_cache".to_string(), FeatureSupport::Partial);
        // Unsupported
        decorator_support.insert("abstractmethod".to_string(), FeatureSupport::None);

        let mut magic_method_support = HashMap::new();
        // Fully supported magic methods
        magic_method_support.insert("__init__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__str__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__repr__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__eq__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__ne__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__add__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__sub__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__mul__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__len__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__getitem__".to_string(), FeatureSupport::Full);
        magic_method_support.insert("__setitem__".to_string(), FeatureSupport::Full);
        // Partially supported
        magic_method_support.insert("__enter__".to_string(), FeatureSupport::Partial);
        magic_method_support.insert("__exit__".to_string(), FeatureSupport::Partial);
        magic_method_support.insert("__call__".to_string(), FeatureSupport::Partial);
        // Unsupported
        magic_method_support.insert("__metaclass__".to_string(), FeatureSupport::None);
        magic_method_support.insert("__getattr__".to_string(), FeatureSupport::None);
        magic_method_support.insert("__setattr__".to_string(), FeatureSupport::None);

        Self {
            decorator_support,
            magic_method_support,
        }
    }

    /// Detect all features in a Python AST
    pub fn detect(&self, ast: &PythonAst, file_path: &str) -> FeatureSet {
        let mut features = Vec::new();

        // Detect function features
        for func in &ast.functions {
            features.extend(self.detect_function_features(func, file_path));
        }

        // Detect class features
        for class in &ast.classes {
            features.extend(self.detect_class_features(class, file_path));
        }

        // Detect import features
        for import in &ast.imports {
            features.extend(self.detect_import_features(import, file_path));
        }

        // Generate summary
        let summary = self.generate_summary(&features);

        FeatureSet { features, summary }
    }

    /// Detect features in a function
    fn detect_function_features(&self, func: &PythonFunction, file_path: &str) -> Vec<DetectedFeature> {
        let mut features = Vec::new();

        // Basic function detection
        let is_magic = func.name.starts_with("__") && func.name.ends_with("__");
        let is_async = func.name.contains("async"); // Simplified detection

        if is_magic {
            let support = self.magic_method_support
                .get(&func.name)
                .cloned()
                .unwrap_or(FeatureSupport::Partial);

            features.push(DetectedFeature {
                category: FeatureCategory::MagicMethod,
                name: func.name.clone(),
                support,
                count: 1,
                locations: vec![FeatureLocation {
                    file: file_path.to_string(),
                    line: None,
                    context: format!("def {}", func.name),
                }],
                details: Some("Magic method detected".to_string()),
            });
        } else {
            features.push(DetectedFeature {
                category: FeatureCategory::Function,
                name: func.name.clone(),
                support: FeatureSupport::Full,
                count: 1,
                locations: vec![FeatureLocation {
                    file: file_path.to_string(),
                    line: None,
                    context: format!("def {}", func.name),
                }],
                details: None,
            });
        }

        // Detect decorators
        for decorator in &func.decorators {
            let support = self.decorator_support
                .get(decorator)
                .cloned()
                .unwrap_or(FeatureSupport::Partial);

            features.push(DetectedFeature {
                category: FeatureCategory::Decorator,
                name: decorator.clone(),
                support,
                count: 1,
                locations: vec![FeatureLocation {
                    file: file_path.to_string(),
                    line: None,
                    context: format!("@{} on {}", decorator, func.name),
                }],
                details: None,
            });
        }

        // Detect type hints
        if func.return_type.is_some() || func.params.iter().any(|p| p.type_hint.is_some()) {
            features.push(DetectedFeature {
                category: FeatureCategory::TypeHint,
                name: format!("{} type hints", func.name),
                support: FeatureSupport::Full,
                count: 1,
                locations: vec![FeatureLocation {
                    file: file_path.to_string(),
                    line: None,
                    context: format!("Type hints in {}", func.name),
                }],
                details: None,
            });
        }

        // Detect async functions (simplified)
        if is_async {
            features.push(DetectedFeature {
                category: FeatureCategory::AsyncAwait,
                name: "async function".to_string(),
                support: FeatureSupport::Partial,
                count: 1,
                locations: vec![FeatureLocation {
                    file: file_path.to_string(),
                    line: None,
                    context: format!("async def {}", func.name),
                }],
                details: Some("Async/await support is partial".to_string()),
            });
        }

        features
    }

    /// Detect features in a class
    fn detect_class_features(&self, class: &PythonClass, file_path: &str) -> Vec<DetectedFeature> {
        let mut features = Vec::new();

        // Basic class detection
        features.push(DetectedFeature {
            category: FeatureCategory::Class,
            name: class.name.clone(),
            support: FeatureSupport::Full,
            count: 1,
            locations: vec![FeatureLocation {
                file: file_path.to_string(),
                line: None,
                context: format!("class {}", class.name),
            }],
            details: None,
        });

        // Detect metaclasses (unsupported)
        for base in &class.bases {
            if base.contains("metaclass") || base == "type" {
                features.push(DetectedFeature {
                    category: FeatureCategory::Metaclass,
                    name: format!("Metaclass in {}", class.name),
                    support: FeatureSupport::None,
                    count: 1,
                    locations: vec![FeatureLocation {
                        file: file_path.to_string(),
                        line: None,
                        context: format!("class {}({})", class.name, base),
                    }],
                    details: Some("Metaclasses are not supported".to_string()),
                });
            }
        }

        // Detect methods
        for method in &class.methods {
            features.extend(self.detect_function_features(method, file_path));
        }

        features
    }

    /// Detect features in imports
    fn detect_import_features(&self, import: &PythonImport, file_path: &str) -> Vec<DetectedFeature> {
        let mut features = Vec::new();

        // Check for known problematic imports
        let unsupported_modules = ["eval", "exec", "compile", "inspect"];
        let partial_modules = ["asyncio", "typing", "dataclasses"];

        let support = if unsupported_modules.contains(&import.module.as_str()) {
            FeatureSupport::None
        } else if partial_modules.contains(&import.module.as_str()) {
            FeatureSupport::Partial
        } else {
            FeatureSupport::Full
        };

        let details = if support != FeatureSupport::Full {
            Some(format!("Module {} has {:?} support", import.module, support))
        } else {
            None
        };

        features.push(DetectedFeature {
            category: FeatureCategory::Import,
            name: import.module.clone(),
            support,
            count: 1,
            locations: vec![FeatureLocation {
                file: file_path.to_string(),
                line: None,
                context: format!("import {}", import.module),
            }],
            details,
        });

        features
    }

    /// Generate summary from detected features
    fn generate_summary(&self, features: &[DetectedFeature]) -> FeatureSummary {
        let total_features = features.len();
        let fully_supported = features.iter().filter(|f| f.support == FeatureSupport::Full).count();
        let partially_supported = features.iter().filter(|f| f.support == FeatureSupport::Partial).count();
        let unsupported = features.iter().filter(|f| f.support == FeatureSupport::None).count();

        let mut by_category = HashMap::new();
        for feature in features {
            *by_category.entry(feature.category.clone()).or_insert(0) += 1;
        }

        FeatureSummary {
            total_features,
            fully_supported,
            partially_supported,
            unsupported,
            by_category,
        }
    }
}

impl Default for FeatureDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_simple_function() {
        let detector = FeatureDetector::new();
        let ast = PythonAst {
            functions: vec![PythonFunction {
                name: "add".to_string(),
                params: vec![],
                return_type: Some("int".to_string()),
                body: String::new(),
                decorators: vec![],
            }],
            classes: vec![],
            imports: vec![],
        };

        let features = detector.detect(&ast, "test.py");
        assert!(features.summary.total_features > 0);
        assert!(features.summary.fully_supported > 0);
    }

    #[test]
    fn test_detect_magic_method() {
        let detector = FeatureDetector::new();
        let ast = PythonAst {
            functions: vec![PythonFunction {
                name: "__init__".to_string(),
                params: vec![],
                return_type: None,
                body: String::new(),
                decorators: vec![],
            }],
            classes: vec![],
            imports: vec![],
        };

        let features = detector.detect(&ast, "test.py");
        assert!(features.features.iter().any(|f| f.category == FeatureCategory::MagicMethod));
    }

    #[test]
    fn test_detect_unsupported_decorator() {
        let detector = FeatureDetector::new();
        let ast = PythonAst {
            functions: vec![PythonFunction {
                name: "test".to_string(),
                params: vec![],
                return_type: None,
                body: String::new(),
                decorators: vec!["abstractmethod".to_string()],
            }],
            classes: vec![],
            imports: vec![],
        };

        let features = detector.detect(&ast, "test.py");
        assert!(features.summary.unsupported > 0);
    }

    #[test]
    fn test_detect_metaclass() {
        let detector = FeatureDetector::new();
        let ast = PythonAst {
            functions: vec![],
            classes: vec![PythonClass {
                name: "Meta".to_string(),
                bases: vec!["type".to_string()],
                methods: vec![],
                attributes: vec![],
            }],
            imports: vec![],
        };

        let features = detector.detect(&ast, "test.py");
        assert!(features.features.iter().any(|f| f.category == FeatureCategory::Metaclass));
        assert!(features.summary.unsupported > 0);
    }
}
