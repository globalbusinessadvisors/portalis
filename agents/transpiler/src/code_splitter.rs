//! Code Splitter - Splits WASM modules for lazy loading and optimization
//!
//! This module provides:
//! 1. Module splitting analysis and recommendations
//! 2. Lazy loading strategy for WASM modules
//! 3. Dynamic import detection and optimization
//! 4. Chunk size analysis and balancing
//! 5. Split point detection based on usage patterns

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Code splitting strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplittingStrategy {
    /// No splitting - single monolithic bundle
    None,
    /// Split by route/page (for web apps)
    ByRoute,
    /// Split by feature (feature-based chunking)
    ByFeature,
    /// Split by lazy loading boundaries
    ByLazyLoad,
    /// Split by size threshold
    BySize,
    /// Automatic splitting based on analysis
    Automatic,
}

/// Split point in the code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitPoint {
    /// Split point identifier
    pub id: String,
    /// Module name
    pub module: String,
    /// Function or boundary name
    pub boundary: String,
    /// Estimated size of chunk (bytes)
    pub estimated_size: u64,
    /// Loading priority (0 = critical, 1 = high, 2 = medium, 3 = low)
    pub priority: u8,
    /// Dependencies needed by this chunk
    pub dependencies: Vec<String>,
    /// Can be lazy loaded
    pub lazy_loadable: bool,
}

/// Code chunk after splitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// Chunk identifier
    pub id: String,
    /// Chunk name
    pub name: String,
    /// Modules included in this chunk
    pub modules: Vec<String>,
    /// Functions included
    pub functions: Vec<String>,
    /// Estimated size (bytes)
    pub size: u64,
    /// Loading priority
    pub priority: u8,
    /// Chunks this depends on
    pub dependencies: Vec<String>,
    /// Is this an entry point chunk
    pub is_entry: bool,
}

impl CodeChunk {
    /// Check if chunk should be eagerly loaded
    pub fn should_preload(&self) -> bool {
        self.is_entry || self.priority == 0
    }

    /// Get loading strategy
    pub fn loading_strategy(&self) -> &str {
        if self.is_entry {
            "eager"
        } else if self.priority == 0 {
            "preload"
        } else if self.priority == 1 {
            "prefetch"
        } else {
            "lazy"
        }
    }
}

/// Code splitting analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplittingAnalysis {
    /// Original bundle size
    pub original_size: u64,
    /// Recommended chunks
    pub chunks: Vec<CodeChunk>,
    /// Split points identified
    pub split_points: Vec<SplitPoint>,
    /// Estimated initial load size (critical chunks only)
    pub initial_load_size: u64,
    /// Total size after splitting (including overhead)
    pub total_split_size: u64,
    /// Splitting strategy used
    pub strategy: SplittingStrategy,
}

impl SplittingAnalysis {
    /// Calculate size reduction percentage
    pub fn size_reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        ((self.original_size - self.initial_load_size) as f64 / self.original_size as f64) * 100.0
    }

    /// Calculate overhead from splitting
    pub fn splitting_overhead(&self) -> u64 {
        if self.total_split_size > self.original_size {
            self.total_split_size - self.original_size
        } else {
            0
        }
    }

    /// Generate detailed report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Code Splitting Analysis ===\n\n");
        report.push_str(&format!("Strategy: {:?}\n", self.strategy));
        report.push_str(&format!("Original Size: {}\n", Self::format_size(self.original_size)));
        report.push_str(&format!("Initial Load Size: {} ({:.1}% of original)\n",
            Self::format_size(self.initial_load_size),
            (self.initial_load_size as f64 / self.original_size as f64) * 100.0
        ));
        report.push_str(&format!("Total Split Size: {}\n", Self::format_size(self.total_split_size)));

        let overhead = self.splitting_overhead();
        if overhead > 0 {
            report.push_str(&format!("Splitting Overhead: {} ({:.1}%)\n",
                Self::format_size(overhead),
                (overhead as f64 / self.original_size as f64) * 100.0
            ));
        }

        report.push_str(&format!("\nChunks Generated: {}\n", self.chunks.len()));

        // Categorize chunks by loading strategy
        let mut eager = 0;
        let mut preload = 0;
        let mut prefetch = 0;
        let mut lazy = 0;

        for chunk in &self.chunks {
            match chunk.loading_strategy() {
                "eager" => eager += 1,
                "preload" => preload += 1,
                "prefetch" => prefetch += 1,
                "lazy" => lazy += 1,
                _ => {}
            }
        }

        report.push_str(&format!("  - Eager:    {} chunks\n", eager));
        report.push_str(&format!("  - Preload:  {} chunks\n", preload));
        report.push_str(&format!("  - Prefetch: {} chunks\n", prefetch));
        report.push_str(&format!("  - Lazy:     {} chunks\n", lazy));

        report.push_str("\nChunk Details:\n");
        for chunk in &self.chunks {
            report.push_str(&format!("  {} ({}) - {} - {} modules, {}\n",
                chunk.name,
                chunk.loading_strategy(),
                Self::format_size(chunk.size),
                chunk.modules.len(),
                if chunk.is_entry { "ENTRY" } else { "" }
            ));
        }

        report.push_str(&format!("\nSplit Points Identified: {}\n", self.split_points.len()));
        for sp in self.split_points.iter().take(5) {
            report.push_str(&format!("  - {} ({})\n", sp.boundary, sp.module));
        }

        report
    }

    fn format_size(bytes: u64) -> String {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        }
    }
}

/// Lazy loading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LazyLoadConfig {
    /// Enable lazy loading
    pub enabled: bool,
    /// Size threshold for creating separate chunk (bytes)
    pub size_threshold: u64,
    /// Maximum number of chunks
    pub max_chunks: usize,
    /// Minimum chunk size (bytes)
    pub min_chunk_size: u64,
    /// Enable route-based splitting
    pub route_based: bool,
}

impl Default for LazyLoadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            size_threshold: 50 * 1024, // 50 KB
            max_chunks: 20,
            min_chunk_size: 10 * 1024, // 10 KB
            route_based: false,
        }
    }
}

impl LazyLoadConfig {
    /// Create aggressive splitting config
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            size_threshold: 30 * 1024, // 30 KB
            max_chunks: 50,
            min_chunk_size: 5 * 1024, // 5 KB
            route_based: true,
        }
    }

    /// Create conservative splitting config
    pub fn conservative() -> Self {
        Self {
            enabled: true,
            size_threshold: 100 * 1024, // 100 KB
            max_chunks: 10,
            min_chunk_size: 20 * 1024, // 20 KB
            route_based: false,
        }
    }
}

/// Code splitter
pub struct CodeSplitter {
    config: LazyLoadConfig,
    strategy: SplittingStrategy,
}

impl CodeSplitter {
    /// Create new code splitter
    pub fn new(strategy: SplittingStrategy) -> Self {
        Self {
            config: LazyLoadConfig::default(),
            strategy,
        }
    }

    /// Create with custom config
    pub fn with_config(strategy: SplittingStrategy, config: LazyLoadConfig) -> Self {
        Self { config, strategy }
    }

    /// Analyze code for splitting opportunities
    pub fn analyze(&self, modules: &HashMap<String, u64>) -> SplittingAnalysis {
        let original_size: u64 = modules.values().sum();

        match self.strategy {
            SplittingStrategy::None => self.no_splitting(original_size),
            SplittingStrategy::BySize => self.split_by_size(modules),
            SplittingStrategy::ByFeature => self.split_by_feature(modules),
            SplittingStrategy::Automatic => self.auto_split(modules),
            _ => self.split_by_size(modules),
        }
    }

    /// No splitting - return single chunk
    fn no_splitting(&self, size: u64) -> SplittingAnalysis {
        SplittingAnalysis {
            original_size: size,
            chunks: vec![CodeChunk {
                id: "main".to_string(),
                name: "main".to_string(),
                modules: vec!["*".to_string()],
                functions: vec![],
                size,
                priority: 0,
                dependencies: vec![],
                is_entry: true,
            }],
            split_points: vec![],
            initial_load_size: size,
            total_split_size: size,
            strategy: SplittingStrategy::None,
        }
    }

    /// Split by size threshold
    fn split_by_size(&self, modules: &HashMap<String, u64>) -> SplittingAnalysis {
        let original_size: u64 = modules.values().sum();
        let mut chunks = Vec::new();
        let mut split_points = Vec::new();

        // Entry chunk (always loaded)
        let mut entry_modules = Vec::new();
        let mut entry_size = 0u64;

        // Lazy chunks
        let mut current_chunk_modules = Vec::new();
        let mut current_chunk_size = 0u64;
        let mut chunk_counter = 1;

        for (module, &size) in modules {
            // Critical modules go in entry chunk
            if Self::is_critical_module(module) {
                entry_modules.push(module.clone());
                entry_size += size;
            } else if current_chunk_size + size > self.config.size_threshold {
                // Create new chunk
                if !current_chunk_modules.is_empty() {
                    chunks.push(CodeChunk {
                        id: format!("chunk_{}", chunk_counter),
                        name: format!("lazy_{}", chunk_counter),
                        modules: current_chunk_modules.clone(),
                        functions: vec![],
                        size: current_chunk_size,
                        priority: 2,
                        dependencies: vec!["main".to_string()],
                        is_entry: false,
                    });

                    split_points.push(SplitPoint {
                        id: format!("split_{}", chunk_counter),
                        module: module.clone(),
                        boundary: format!("chunk_{}_boundary", chunk_counter),
                        estimated_size: current_chunk_size,
                        priority: 2,
                        dependencies: vec!["main".to_string()],
                        lazy_loadable: true,
                    });

                    chunk_counter += 1;
                }

                current_chunk_modules = vec![module.clone()];
                current_chunk_size = size;
            } else {
                current_chunk_modules.push(module.clone());
                current_chunk_size += size;
            }
        }

        // Add remaining chunk
        if !current_chunk_modules.is_empty() {
            chunks.push(CodeChunk {
                id: format!("chunk_{}", chunk_counter),
                name: format!("lazy_{}", chunk_counter),
                modules: current_chunk_modules,
                functions: vec![],
                size: current_chunk_size,
                priority: 2,
                dependencies: vec!["main".to_string()],
                is_entry: false,
            });
        }

        // Add entry chunk at the beginning
        chunks.insert(0, CodeChunk {
            id: "main".to_string(),
            name: "main".to_string(),
            modules: entry_modules,
            functions: vec![],
            size: entry_size,
            priority: 0,
            dependencies: vec![],
            is_entry: true,
        });

        let total_split_size = chunks.iter().map(|c| c.size).sum::<u64>() +
            (chunks.len() as u64 * 100); // Overhead per chunk

        SplittingAnalysis {
            original_size,
            chunks,
            split_points,
            initial_load_size: entry_size,
            total_split_size,
            strategy: SplittingStrategy::BySize,
        }
    }

    /// Split by feature
    fn split_by_feature(&self, modules: &HashMap<String, u64>) -> SplittingAnalysis {
        let original_size: u64 = modules.values().sum();
        let mut chunks = Vec::new();
        let mut split_points = Vec::new();

        // Categorize modules by feature
        let mut core_modules = Vec::new();
        let mut core_size = 0u64;
        let mut ui_modules = Vec::new();
        let mut ui_size = 0u64;
        let mut data_modules = Vec::new();
        let mut data_size = 0u64;

        for (module, &size) in modules {
            if module.contains("ui") || module.contains("component") {
                ui_modules.push(module.clone());
                ui_size += size;
            } else if module.contains("data") || module.contains("api") {
                data_modules.push(module.clone());
                data_size += size;
            } else {
                core_modules.push(module.clone());
                core_size += size;
            }
        }

        // Core chunk (entry)
        chunks.push(CodeChunk {
            id: "core".to_string(),
            name: "core".to_string(),
            modules: core_modules,
            functions: vec![],
            size: core_size,
            priority: 0,
            dependencies: vec![],
            is_entry: true,
        });

        // UI chunk (preload)
        if !ui_modules.is_empty() {
            chunks.push(CodeChunk {
                id: "ui".to_string(),
                name: "ui".to_string(),
                modules: ui_modules,
                functions: vec![],
                size: ui_size,
                priority: 1,
                dependencies: vec!["core".to_string()],
                is_entry: false,
            });

            split_points.push(SplitPoint {
                id: "ui_split".to_string(),
                module: "ui".to_string(),
                boundary: "ui_boundary".to_string(),
                estimated_size: ui_size,
                priority: 1,
                dependencies: vec!["core".to_string()],
                lazy_loadable: true,
            });
        }

        // Data chunk (lazy)
        if !data_modules.is_empty() {
            chunks.push(CodeChunk {
                id: "data".to_string(),
                name: "data".to_string(),
                modules: data_modules,
                functions: vec![],
                size: data_size,
                priority: 2,
                dependencies: vec!["core".to_string()],
                is_entry: false,
            });

            split_points.push(SplitPoint {
                id: "data_split".to_string(),
                module: "data".to_string(),
                boundary: "data_boundary".to_string(),
                estimated_size: data_size,
                priority: 2,
                dependencies: vec!["core".to_string()],
                lazy_loadable: true,
            });
        }

        let total_split_size = chunks.iter().map(|c| c.size).sum::<u64>() +
            (chunks.len() as u64 * 100);

        SplittingAnalysis {
            original_size,
            chunks,
            split_points,
            initial_load_size: core_size,
            total_split_size,
            strategy: SplittingStrategy::ByFeature,
        }
    }

    /// Automatic splitting with intelligent analysis
    fn auto_split(&self, modules: &HashMap<String, u64>) -> SplittingAnalysis {
        let original_size: u64 = modules.values().sum();

        // Use size-based splitting as the automatic strategy
        if original_size > 500 * 1024 {
            // Large bundle - use aggressive splitting
            let mut splitter = Self::with_config(
                SplittingStrategy::BySize,
                LazyLoadConfig::aggressive(),
            );
            splitter.split_by_size(modules)
        } else if original_size > 200 * 1024 {
            // Medium bundle - use default splitting
            self.split_by_size(modules)
        } else {
            // Small bundle - no splitting needed
            self.no_splitting(original_size)
        }
    }

    /// Check if module is critical (must be in entry chunk)
    fn is_critical_module(module: &str) -> bool {
        module == "main" ||
        module.contains("init") ||
        module.contains("bootstrap") ||
        module.contains("core")
    }

    /// Generate dynamic import code
    pub fn generate_dynamic_import(&self, chunk: &CodeChunk) -> String {
        format!(
            r#"
// Dynamic import for chunk: {}
async fn load_{}() -> Result<(), JsValue> {{
    wasm_bindgen_futures::JsFuture::from(
        js_sys::eval(&format!(
            "import('./pkg/{}.js')",
        )).unwrap()
    ).await?;
    Ok(())
}}
"#,
            chunk.name, chunk.id, chunk.name
        )
    }

    /// Generate webpack config for code splitting
    pub fn generate_webpack_config(&self, analysis: &SplittingAnalysis) -> String {
        let mut config = String::new();

        config.push_str("module.exports = {\n");
        config.push_str("  optimization: {\n");
        config.push_str("    splitChunks: {\n");
        config.push_str("      chunks: 'all',\n");
        config.push_str(&format!("      maxSize: {},\n", self.config.size_threshold));
        config.push_str(&format!("      minSize: {},\n", self.config.min_chunk_size));
        config.push_str("      cacheGroups: {\n");

        for chunk in &analysis.chunks {
            if !chunk.is_entry {
                config.push_str(&format!("        {}: {{\n", chunk.id));
                config.push_str(&format!("          name: '{}',\n", chunk.name));
                config.push_str(&format!("          priority: {},\n", chunk.priority));
                config.push_str("          reuseExistingChunk: true,\n");
                config.push_str("        },\n");
            }
        }

        config.push_str("      },\n");
        config.push_str("    },\n");
        config.push_str("  },\n");
        config.push_str("};\n");

        config
    }
}

impl Default for CodeSplitter {
    fn default() -> Self {
        Self::new(SplittingStrategy::Automatic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_splitting() {
        let splitter = CodeSplitter::new(SplittingStrategy::None);
        let mut modules = HashMap::new();
        modules.insert("main".to_string(), 100_000);

        let analysis = splitter.analyze(&modules);

        assert_eq!(analysis.chunks.len(), 1);
        assert_eq!(analysis.initial_load_size, 100_000);
    }

    #[test]
    fn test_split_by_size() {
        let splitter = CodeSplitter::new(SplittingStrategy::BySize);
        let mut modules = HashMap::new();
        modules.insert("main".to_string(), 40_000);
        modules.insert("module_a".to_string(), 60_000);
        modules.insert("module_b".to_string(), 80_000);

        let analysis = splitter.analyze(&modules);

        assert!(analysis.chunks.len() > 1);
        assert!(analysis.initial_load_size < analysis.original_size);
    }

    #[test]
    fn test_split_by_feature() {
        let splitter = CodeSplitter::new(SplittingStrategy::ByFeature);
        let mut modules = HashMap::new();
        modules.insert("core".to_string(), 50_000);
        modules.insert("ui_component".to_string(), 30_000);
        modules.insert("data_api".to_string(), 40_000);

        let analysis = splitter.analyze(&modules);

        assert!(analysis.chunks.len() >= 2);
        // Core should be in entry chunk
        assert!(analysis.chunks.iter().any(|c| c.is_entry));
    }

    #[test]
    fn test_lazy_load_config() {
        let aggressive = LazyLoadConfig::aggressive();
        let conservative = LazyLoadConfig::conservative();

        assert!(aggressive.size_threshold < conservative.size_threshold);
        assert!(aggressive.max_chunks > conservative.max_chunks);
    }

    #[test]
    fn test_chunk_loading_strategy() {
        let entry_chunk = CodeChunk {
            id: "main".to_string(),
            name: "main".to_string(),
            modules: vec![],
            functions: vec![],
            size: 1000,
            priority: 0,
            dependencies: vec![],
            is_entry: true,
        };

        assert_eq!(entry_chunk.loading_strategy(), "eager");
        assert!(entry_chunk.should_preload());
    }
}
