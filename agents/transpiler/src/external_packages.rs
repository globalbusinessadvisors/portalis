//! External Package Mapping - Maps popular PyPI packages to Rust crates
//!
//! Provides mappings for the top 100 PyPI packages to their Rust equivalents,
//! enabling transpilation of real-world Python applications.

use crate::stdlib_mapper::WasmCompatibility;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// External package mapping (PyPI package → Rust crate)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalPackageMapping {
    /// Python package name (e.g., "numpy", "pandas")
    pub python_package: String,

    /// Rust crate name
    pub rust_crate: String,

    /// Crate version
    pub version: String,

    /// Required features
    pub features: Vec<String>,

    /// WASM compatibility
    pub wasm_compatible: WasmCompatibility,

    /// Function/API mappings
    pub api_mappings: Vec<ApiMapping>,

    /// Installation notes
    pub notes: Option<String>,

    /// Alternative crates (fallbacks)
    pub alternatives: Vec<String>,
}

/// Individual API/function mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMapping {
    /// Python API (e.g., "numpy.array", "pd.DataFrame")
    pub python_api: String,

    /// Rust equivalent (e.g., "ndarray::Array::from_vec")
    pub rust_equiv: String,

    /// Required use statement
    pub requires_use: Option<String>,

    /// Transformation notes
    pub transform_notes: Option<String>,

    /// WASM compatibility for this specific API
    pub wasm_compatible: WasmCompatibility,
}

/// External package registry
pub struct ExternalPackageRegistry {
    packages: HashMap<String, ExternalPackageMapping>,
}

impl ExternalPackageRegistry {
    /// Create new registry with top 100 PyPI packages
    pub fn new() -> Self {
        let mut registry = Self {
            packages: HashMap::new(),
        };

        // Initialize with priority packages
        registry.init_priority_packages();
        registry.init_data_science_packages();
        registry.init_web_packages();
        registry.init_ml_packages();
        registry.init_utility_packages();

        // Phase 2: Expand to 200+ packages
        registry.init_expanded_data_science();
        registry.init_expanded_web_frameworks();
        registry.init_async_networking();
        registry.init_testing_packages();
        registry.init_cli_tui_packages();
        registry.init_devops_packages();
        registry.init_database_packages();
        registry.init_data_format_packages();
        registry.init_security_packages();
        registry.init_messaging_packages();

        registry
    }

    /// Get package mapping by name
    pub fn get_package(&self, package_name: &str) -> Option<&ExternalPackageMapping> {
        self.packages.get(package_name)
    }

    /// Get API mapping
    pub fn get_api_mapping(&self, package: &str, api: &str) -> Option<String> {
        self.packages.get(package)
            .and_then(|pkg| {
                pkg.api_mappings.iter()
                    .find(|m| m.python_api == api)
                    .map(|m| m.rust_equiv.clone())
            })
    }

    /// Get all package names
    pub fn package_names(&self) -> Vec<String> {
        self.packages.keys().cloned().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> PackageStats {
        let total = self.packages.len();
        let full_wasm = self.packages.values()
            .filter(|p| p.wasm_compatible == WasmCompatibility::Full)
            .count();
        let partial_wasm = self.packages.values()
            .filter(|p| p.wasm_compatible == WasmCompatibility::Partial)
            .count();
        let requires_js = self.packages.values()
            .filter(|p| p.wasm_compatible == WasmCompatibility::RequiresJsInterop)
            .count();
        let incompatible = self.packages.values()
            .filter(|p| p.wasm_compatible == WasmCompatibility::Incompatible)
            .count();

        PackageStats {
            total_packages: total,
            full_wasm_compat: full_wasm,
            partial_wasm_compat: partial_wasm,
            requires_js_interop: requires_js,
            incompatible,
        }
    }

    /// Initialize priority packages (NumPy, Pandas, Requests, Pillow, Scikit-learn)
    fn init_priority_packages(&mut self) {
        // NumPy → ndarray
        self.packages.insert(
            "numpy".to_string(),
            ExternalPackageMapping {
                python_package: "numpy".to_string(),
                rust_crate: "ndarray".to_string(),
                version: "0.15".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![
                    ApiMapping {
                        python_api: "numpy.array".to_string(),
                        rust_equiv: "ndarray::arr1".to_string(),
                        requires_use: Some("ndarray".to_string()),
                        transform_notes: Some("Use arr1 for 1D, arr2 for 2D, Array::from_vec for general".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                    ApiMapping {
                        python_api: "numpy.zeros".to_string(),
                        rust_equiv: "ndarray::Array::zeros".to_string(),
                        requires_use: Some("ndarray::Array".to_string()),
                        transform_notes: Some("Specify shape as tuple".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                    ApiMapping {
                        python_api: "numpy.ones".to_string(),
                        rust_equiv: "ndarray::Array::ones".to_string(),
                        requires_use: Some("ndarray::Array".to_string()),
                        transform_notes: None,
                        wasm_compatible: WasmCompatibility::Full,
                    },
                    ApiMapping {
                        python_api: "numpy.arange".to_string(),
                        rust_equiv: "ndarray::Array::range".to_string(),
                        requires_use: Some("ndarray::Array".to_string()),
                        transform_notes: Some("Use Array::range or Array::linspace".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                    ApiMapping {
                        python_api: "numpy.dot".to_string(),
                        rust_equiv: ".dot()".to_string(),
                        requires_use: None,
                        transform_notes: Some("Method on Array".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                ],
                notes: Some("Pure computation, fully WASM compatible. Use ndarray-rand for random arrays.".to_string()),
                alternatives: vec!["nalgebra".to_string()],
            },
        );

        // Pandas → Polars
        self.packages.insert(
            "pandas".to_string(),
            ExternalPackageMapping {
                python_package: "pandas".to_string(),
                rust_crate: "polars".to_string(),
                version: "0.35".to_string(),
                features: vec!["lazy".to_string()],
                wasm_compatible: WasmCompatibility::Partial,
                api_mappings: vec![
                    ApiMapping {
                        python_api: "pandas.DataFrame".to_string(),
                        rust_equiv: "polars::prelude::DataFrame::new".to_string(),
                        requires_use: Some("polars::prelude::*".to_string()),
                        transform_notes: Some("✅ WASM: In-memory DataFrame operations fully supported".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                    ApiMapping {
                        python_api: "pandas.read_csv".to_string(),
                        rust_equiv: "polars::prelude::CsvReader::from_path".to_string(),
                        requires_use: Some("polars::prelude::*".to_string()),
                        transform_notes: Some("❌ WASM: File I/O requires WASI runtime. Alternative: embed data or use IndexedDB in browser".to_string()),
                        wasm_compatible: WasmCompatibility::RequiresWasi,
                    },
                    ApiMapping {
                        python_api: "df.head".to_string(),
                        rust_equiv: ".head(Some(n))".to_string(),
                        requires_use: None,
                        transform_notes: Some("✅ WASM: Operations work everywhere".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                    ApiMapping {
                        python_api: "df.describe".to_string(),
                        rust_equiv: ".describe(None)".to_string(),
                        requires_use: None,
                        transform_notes: None,
                        wasm_compatible: WasmCompatibility::Full,
                    },
                ],
                notes: Some("Polars is a fast DataFrame library. ✅ WASM: In-memory operations (DataFrame creation, transformations, aggregations) work everywhere. ❌ WASM: File I/O (read_csv, read_parquet, write_*) requires WASI. Browser alternative: embed data as JSON/arrays or use IndexedDB.".to_string()),
                alternatives: vec![],
            },
        );

        // Requests → reqwest
        self.packages.insert(
            "requests".to_string(),
            ExternalPackageMapping {
                python_package: "requests".to_string(),
                rust_crate: "reqwest".to_string(),
                version: "0.11".to_string(),
                features: vec!["json".to_string(), "blocking".to_string()],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![
                    ApiMapping {
                        python_api: "requests.get".to_string(),
                        rust_equiv: "reqwest::blocking::get".to_string(),
                        requires_use: Some("reqwest".to_string()),
                        transform_notes: Some("Use async version in WASM with wasm-bindgen-futures".to_string()),
                        wasm_compatible: WasmCompatibility::RequiresJsInterop,
                    },
                    ApiMapping {
                        python_api: "requests.post".to_string(),
                        rust_equiv: "reqwest::blocking::Client::new().post".to_string(),
                        requires_use: Some("reqwest".to_string()),
                        transform_notes: Some("Async in WASM".to_string()),
                        wasm_compatible: WasmCompatibility::RequiresJsInterop,
                    },
                ],
                notes: Some("Uses fetch() API in browser WASM".to_string()),
                alternatives: vec!["ureq".to_string()],
            },
        );

        // Pillow → image
        self.packages.insert(
            "pillow".to_string(),
            ExternalPackageMapping {
                python_package: "pillow".to_string(),
                rust_crate: "image".to_string(),
                version: "0.24".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![
                    ApiMapping {
                        python_api: "Image.open".to_string(),
                        rust_equiv: "image::open".to_string(),
                        requires_use: Some("image".to_string()),
                        transform_notes: Some("Returns ImageBuffer".to_string()),
                        wasm_compatible: WasmCompatibility::RequiresWasi,
                    },
                    ApiMapping {
                        python_api: "Image.new".to_string(),
                        rust_equiv: "image::ImageBuffer::new".to_string(),
                        requires_use: Some("image::ImageBuffer".to_string()),
                        transform_notes: None,
                        wasm_compatible: WasmCompatibility::Full,
                    },
                ],
                notes: Some("Image processing fully works in WASM, I/O needs WASI".to_string()),
                alternatives: vec!["imageproc".to_string()],
            },
        );

        // Scikit-learn → linfa
        self.packages.insert(
            "sklearn".to_string(),
            ExternalPackageMapping {
                python_package: "sklearn".to_string(),
                rust_crate: "linfa".to_string(),
                version: "0.7".to_string(),
                features: vec!["linfa-linear".to_string(), "linfa-clustering".to_string()],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![
                    ApiMapping {
                        python_api: "sklearn.linear_model.LinearRegression".to_string(),
                        rust_equiv: "linfa_linear::LinearRegression::new".to_string(),
                        requires_use: Some("linfa_linear::LinearRegression".to_string()),
                        transform_notes: Some("Use with linfa Dataset".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                    ApiMapping {
                        python_api: "sklearn.cluster.KMeans".to_string(),
                        rust_equiv: "linfa_clustering::KMeans::params".to_string(),
                        requires_use: Some("linfa_clustering::KMeans".to_string()),
                        transform_notes: None,
                        wasm_compatible: WasmCompatibility::Full,
                    },
                ],
                notes: Some("ML algorithms work in WASM. Dataset loading may need WASI.".to_string()),
                alternatives: vec!["smartcore".to_string()],
            },
        );
    }

    /// Initialize data science packages
    fn init_data_science_packages(&mut self) {
        // SciPy → nalgebra + statrs
        self.packages.insert(
            "scipy".to_string(),
            ExternalPackageMapping {
                python_package: "scipy".to_string(),
                rust_crate: "nalgebra".to_string(),
                version: "0.32".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![
                    ApiMapping {
                        python_api: "scipy.linalg.inv".to_string(),
                        rust_equiv: "matrix.try_inverse()".to_string(),
                        requires_use: Some("nalgebra::Matrix".to_string()),
                        transform_notes: Some("Use nalgebra for linear algebra".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                    ApiMapping {
                        python_api: "scipy.stats.norm".to_string(),
                        rust_equiv: "Normal::new(mean, std_dev)".to_string(),
                        requires_use: Some("statrs::distribution::Normal".to_string()),
                        transform_notes: Some("Use statrs for statistical distributions".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                    ApiMapping {
                        python_api: "scipy.optimize.minimize".to_string(),
                        rust_equiv: "argmin optimization crate".to_string(),
                        requires_use: Some("argmin".to_string()),
                        transform_notes: Some("Use argmin or optim crates for optimization".to_string()),
                        wasm_compatible: WasmCompatibility::Full,
                    },
                ],
                notes: Some("Linear algebra via nalgebra, stats via statrs, optimization via argmin".to_string()),
                alternatives: vec!["statrs".to_string(), "ndarray-linalg".to_string()],
            },
        );

        // Matplotlib → plotters
        self.packages.insert(
            "matplotlib".to_string(),
            ExternalPackageMapping {
                python_package: "matplotlib".to_string(),
                rust_crate: "plotters".to_string(),
                version: "0.3".to_string(),
                features: vec!["all_elements".to_string()],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![
                    ApiMapping {
                        python_api: "plt.plot".to_string(),
                        rust_equiv: "chart.draw_series(LineSeries::new(...))".to_string(),
                        requires_use: Some("plotters::prelude::*".to_string()),
                        transform_notes: Some("Use plotters-canvas for WASM rendering".to_string()),
                        wasm_compatible: WasmCompatibility::RequiresJsInterop,
                    },
                    ApiMapping {
                        python_api: "plt.scatter".to_string(),
                        rust_equiv: "chart.draw_series(PointSeries::of_element(...))".to_string(),
                        requires_use: Some("plotters::prelude::*".to_string()),
                        transform_notes: Some("Use scatter plot series".to_string()),
                        wasm_compatible: WasmCompatibility::RequiresJsInterop,
                    },
                ],
                notes: Some("Use plotters-canvas for WASM rendering to HTML canvas".to_string()),
                alternatives: vec!["plotly".to_string()],
            },
        );
    }

    /// Initialize web packages
    fn init_web_packages(&mut self) {
        // Flask → actix-web
        self.packages.insert(
            "flask".to_string(),
            ExternalPackageMapping {
                python_package: "flask".to_string(),
                rust_crate: "actix-web".to_string(),
                version: "4".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Incompatible,
                api_mappings: vec![],
                notes: Some("Use warp or axum for server. Not applicable in browser WASM.".to_string()),
                alternatives: vec!["warp".to_string(), "axum".to_string()],
            },
        );

        // Django → (no direct equivalent)
        self.packages.insert(
            "django".to_string(),
            ExternalPackageMapping {
                python_package: "django".to_string(),
                rust_crate: "actix-web".to_string(),
                version: "4".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Incompatible,
                api_mappings: vec![],
                notes: Some("Full framework - consider actix-web + diesel for similar functionality".to_string()),
                alternatives: vec!["rocket".to_string()],
            },
        );

        // aiohttp → reqwest async
        self.packages.insert(
            "aiohttp".to_string(),
            ExternalPackageMapping {
                python_package: "aiohttp".to_string(),
                rust_crate: "reqwest".to_string(),
                version: "0.11".to_string(),
                features: vec!["json".to_string()],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("Async HTTP client/server".to_string()),
                alternatives: vec!["hyper".to_string()],
            },
        );
    }

    /// Initialize ML packages
    fn init_ml_packages(&mut self) {
        // TensorFlow → (limited Rust support)
        self.packages.insert(
            "tensorflow".to_string(),
            ExternalPackageMapping {
                python_package: "tensorflow".to_string(),
                rust_crate: "tensorflow".to_string(),
                version: "0.20".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Incompatible,
                api_mappings: vec![],
                notes: Some("Limited Rust bindings, not WASM compatible. Consider tract or burn.".to_string()),
                alternatives: vec!["tract".to_string(), "burn".to_string()],
            },
        );

        // PyTorch → tch-rs / burn
        self.packages.insert(
            "torch".to_string(),
            ExternalPackageMapping {
                python_package: "torch".to_string(),
                rust_crate: "burn".to_string(),
                version: "0.11".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Partial,
                api_mappings: vec![],
                notes: Some("Burn is a pure Rust deep learning framework. ✅ WASM: Inference works with pre-trained models. Training works for small models. ❌ WASM: Large model training limited by memory. Use burn with wasm-bindgen backend or tract for ONNX inference.".to_string()),
                alternatives: vec!["tch".to_string(), "tract".to_string()],
            },
        );
    }

    /// Initialize utility packages
    fn init_utility_packages(&mut self) {
        // pydantic → serde
        self.packages.insert(
            "pydantic".to_string(),
            ExternalPackageMapping {
                python_package: "pydantic".to_string(),
                rust_crate: "serde".to_string(),
                version: "1".to_string(),
                features: vec!["derive".to_string()],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Use serde with derive for validation".to_string()),
                alternatives: vec!["validator".to_string()],
            },
        );

        // pytest → (use Rust testing)
        self.packages.insert(
            "pytest".to_string(),
            ExternalPackageMapping {
                python_package: "pytest".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Use Rust's built-in test framework (#[test])".to_string()),
                alternatives: vec![],
            },
        );

        // click → clap
        self.packages.insert(
            "click".to_string(),
            ExternalPackageMapping {
                python_package: "click".to_string(),
                rust_crate: "clap".to_string(),
                version: "4".to_string(),
                features: vec!["derive".to_string()],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("CLI framework".to_string()),
                alternatives: vec!["structopt".to_string()],
            },
        );

        // Add more packages for 16-100
        self.init_packages_16_35();
        self.init_packages_36_55();
        self.init_packages_56_75();
        self.init_packages_76_100();
    }

    /// Initialize packages 16-35
    fn init_packages_16_35(&mut self) {
        // 16. beautifulsoup4 → scraper
        self.packages.insert(
            "bs4".to_string(),
            ExternalPackageMapping {
                python_package: "bs4".to_string(),
                rust_crate: "scraper".to_string(),
                version: "0.18".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("HTML parsing and web scraping".to_string()),
                alternatives: vec!["select".to_string()],
            },
        );

        // 17. lxml → quick-xml
        self.packages.insert(
            "lxml".to_string(),
            ExternalPackageMapping {
                python_package: "lxml".to_string(),
                rust_crate: "quick-xml".to_string(),
                version: "0.31".to_string(),
                features: vec!["serialize".to_string()],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("XML/HTML processing".to_string()),
                alternatives: vec!["roxmltree".to_string()],
            },
        );

        // 18. cryptography → ring
        self.packages.insert(
            "cryptography".to_string(),
            ExternalPackageMapping {
                python_package: "cryptography".to_string(),
                rust_crate: "ring".to_string(),
                version: "0.17".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Cryptographic primitives".to_string()),
                alternatives: vec!["rustls".to_string(), "openssl".to_string()],
            },
        );

        // 19. sqlalchemy → diesel
        self.packages.insert(
            "sqlalchemy".to_string(),
            ExternalPackageMapping {
                python_package: "sqlalchemy".to_string(),
                rust_crate: "diesel".to_string(),
                version: "2".to_string(),
                features: vec!["sqlite".to_string()],
                wasm_compatible: WasmCompatibility::Partial,
                api_mappings: vec![],
                notes: Some("ORM for databases. ✅ WASM: SQLite works with sql.js in browser (compile diesel with wasm32-unknown-unknown, sqlite feature). ❌ WASM: PostgreSQL/MySQL require network access via JS interop or server proxy. Use sqlx for async or sea-orm for more flexible async ORM.".to_string()),
                alternatives: vec!["sqlx".to_string(), "sea-orm".to_string()],
            },
        );

        // 20. jinja2 → tera
        self.packages.insert(
            "jinja2".to_string(),
            ExternalPackageMapping {
                python_package: "jinja2".to_string(),
                rust_crate: "tera".to_string(),
                version: "1".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Template engine".to_string()),
                alternatives: vec!["handlebars".to_string(), "askama".to_string()],
            },
        );

        // 21. redis → redis-rs
        self.packages.insert(
            "redis".to_string(),
            ExternalPackageMapping {
                python_package: "redis".to_string(),
                rust_crate: "redis".to_string(),
                version: "0.24".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("Redis client - needs network, use WebSocket in browser".to_string()),
                alternatives: vec![],
            },
        );

        // 22. celery → (no direct equivalent)
        self.packages.insert(
            "celery".to_string(),
            ExternalPackageMapping {
                python_package: "celery".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Incompatible,
                api_mappings: vec![],
                notes: Some("Distributed task queue - not applicable in WASM. Use message queues on server.".to_string()),
                alternatives: vec![],
            },
        );

        // 23. boto3 → rusoto / aws-sdk-rust
        self.packages.insert(
            "boto3".to_string(),
            ExternalPackageMapping {
                python_package: "boto3".to_string(),
                rust_crate: "aws-sdk-s3".to_string(),
                version: "1".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("AWS SDK - works in WASM with JS interop for HTTP".to_string()),
                alternatives: vec!["rusoto_core".to_string()],
            },
        );

        // 24. pymongo → mongodb
        self.packages.insert(
            "pymongo".to_string(),
            ExternalPackageMapping {
                python_package: "pymongo".to_string(),
                rust_crate: "mongodb".to_string(),
                version: "2".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("MongoDB driver - async works in WASM with JS interop".to_string()),
                alternatives: vec![],
            },
        );

        // 25. marshmallow → serde
        self.packages.insert(
            "marshmallow".to_string(),
            ExternalPackageMapping {
                python_package: "marshmallow".to_string(),
                rust_crate: "serde".to_string(),
                version: "1".to_string(),
                features: vec!["derive".to_string()],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Serialization/deserialization".to_string()),
                alternatives: vec![],
            },
        );

        // 26. alembic → (use diesel migrations)
        self.packages.insert(
            "alembic".to_string(),
            ExternalPackageMapping {
                python_package: "alembic".to_string(),
                rust_crate: "diesel_migrations".to_string(),
                version: "2".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Partial,
                api_mappings: vec![],
                notes: Some("Database migrations. ✅ WASM: SQLite migrations work with sql.js. Apply migrations programmatically with diesel_migrations. ❌ WASM: PostgreSQL/MySQL migrations require database connectivity. Typically run migrations server-side, not in WASM.".to_string()),
                alternatives: vec!["sqlx".to_string()],
            },
        );

        // 27. httpx → reqwest
        self.packages.insert(
            "httpx".to_string(),
            ExternalPackageMapping {
                python_package: "httpx".to_string(),
                rust_crate: "reqwest".to_string(),
                version: "0.11".to_string(),
                features: vec!["json".to_string()],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("Modern HTTP client".to_string()),
                alternatives: vec![],
            },
        );

        // 28. fastapi → actix-web / axum
        self.packages.insert(
            "fastapi".to_string(),
            ExternalPackageMapping {
                python_package: "fastapi".to_string(),
                rust_crate: "axum".to_string(),
                version: "0.7".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Incompatible,
                api_mappings: vec![],
                notes: Some("Web framework - server-side only".to_string()),
                alternatives: vec!["actix-web".to_string()],
            },
        );

        // 29. pytest-cov → (use tarpaulin)
        self.packages.insert(
            "pytest-cov".to_string(),
            ExternalPackageMapping {
                python_package: "pytest-cov".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Use tarpaulin or llvm-cov for Rust coverage".to_string()),
                alternatives: vec![],
            },
        );

        // 30. PyYAML → serde_yaml
        self.packages.insert(
            "yaml".to_string(),
            ExternalPackageMapping {
                python_package: "yaml".to_string(),
                rust_crate: "serde_yaml".to_string(),
                version: "0.9".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("YAML parsing and serialization".to_string()),
                alternatives: vec![],
            },
        );

        // 31. python-dotenv → dotenvy
        self.packages.insert(
            "dotenv".to_string(),
            ExternalPackageMapping {
                python_package: "dotenv".to_string(),
                rust_crate: "dotenvy".to_string(),
                version: "0.15".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresWasi,
                api_mappings: vec![],
                notes: Some(".env file loading - needs filesystem".to_string()),
                alternatives: vec![],
            },
        );

        // 32. python-jose → jsonwebtoken
        self.packages.insert(
            "jose".to_string(),
            ExternalPackageMapping {
                python_package: "jose".to_string(),
                rust_crate: "jsonwebtoken".to_string(),
                version: "9".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("JWT tokens".to_string()),
                alternatives: vec![],
            },
        );

        // 33. passlib → argon2
        self.packages.insert(
            "passlib".to_string(),
            ExternalPackageMapping {
                python_package: "passlib".to_string(),
                rust_crate: "argon2".to_string(),
                version: "0.5".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Password hashing".to_string()),
                alternatives: vec!["bcrypt".to_string()],
            },
        );

        // 34. werkzeug → (web utilities)
        self.packages.insert(
            "werkzeug".to_string(),
            ExternalPackageMapping {
                python_package: "werkzeug".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Incompatible,
                api_mappings: vec![],
                notes: Some("WSGI utilities - use axum/actix-web utilities instead".to_string()),
                alternatives: vec![],
            },
        );

        // 35. black → rustfmt
        self.packages.insert(
            "black".to_string(),
            ExternalPackageMapping {
                python_package: "black".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Code formatter - use rustfmt".to_string()),
                alternatives: vec![],
            },
        );
    }

    /// Initialize packages 36-55
    fn init_packages_36_55(&mut self) {
        // 36. mypy → (use Rust's type system)
        self.packages.insert(
            "mypy".to_string(),
            ExternalPackageMapping {
                python_package: "mypy".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Type checker - Rust has built-in type checking".to_string()),
                alternatives: vec![],
            },
        );

        // 37. greenlet → tokio
        self.packages.insert(
            "greenlet".to_string(),
            ExternalPackageMapping {
                python_package: "greenlet".to_string(),
                rust_crate: "tokio".to_string(),
                version: "1".to_string(),
                features: vec!["rt".to_string()],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("Async runtime".to_string()),
                alternatives: vec!["async-std".to_string()],
            },
        );

        // 38. protobuf → prost
        self.packages.insert(
            "protobuf".to_string(),
            ExternalPackageMapping {
                python_package: "protobuf".to_string(),
                rust_crate: "prost".to_string(),
                version: "0.12".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Protocol Buffers".to_string()),
                alternatives: vec!["protobuf".to_string()],
            },
        );

        // 39. grpcio → tonic
        self.packages.insert(
            "grpcio".to_string(),
            ExternalPackageMapping {
                python_package: "grpcio".to_string(),
                rust_crate: "tonic".to_string(),
                version: "0.10".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("gRPC - works with gRPC-Web in browser".to_string()),
                alternatives: vec![],
            },
        );

        // 40. prometheus_client → prometheus
        self.packages.insert(
            "prometheus_client".to_string(),
            ExternalPackageMapping {
                python_package: "prometheus_client".to_string(),
                rust_crate: "prometheus".to_string(),
                version: "0.13".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Partial,
                api_mappings: vec![],
                notes: Some("Metrics collection. ✅ WASM: Counter, Gauge, Histogram metric types work. Exporting metrics requires HTTP endpoint (use JS fetch to POST metrics). ❌ WASM: Can't run HTTP server directly. Send metrics to collector via JS interop or batch to IndexedDB.".to_string()),
                alternatives: vec![],
            },
        );

        // 41. hypothesis → proptest / quickcheck
        self.packages.insert(
            "hypothesis".to_string(),
            ExternalPackageMapping {
                python_package: "hypothesis".to_string(),
                rust_crate: "proptest".to_string(),
                version: "1".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Property-based testing".to_string()),
                alternatives: vec!["quickcheck".to_string()],
            },
        );

        // 42. factory_boy → (build test fixtures manually)
        self.packages.insert(
            "factory_boy".to_string(),
            ExternalPackageMapping {
                python_package: "factory_boy".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Test fixtures - build manually or use fake-rs".to_string()),
                alternatives: vec!["fake".to_string()],
            },
        );

        // 43. faker → fake
        self.packages.insert(
            "faker".to_string(),
            ExternalPackageMapping {
                python_package: "faker".to_string(),
                rust_crate: "fake".to_string(),
                version: "2".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Fake data generation".to_string()),
                alternatives: vec![],
            },
        );

        // 44. arrow → arrow-rs
        self.packages.insert(
            "arrow".to_string(),
            ExternalPackageMapping {
                python_package: "arrow".to_string(),
                rust_crate: "arrow".to_string(),
                version: "50".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Apache Arrow columnar format".to_string()),
                alternatives: vec![],
            },
        );

        // 45. s3fs → rusoto_s3
        self.packages.insert(
            "s3fs".to_string(),
            ExternalPackageMapping {
                python_package: "s3fs".to_string(),
                rust_crate: "aws-sdk-s3".to_string(),
                version: "1".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("S3 filesystem interface".to_string()),
                alternatives: vec![],
            },
        );

        // 46. pyarrow → arrow-rs
        self.packages.insert(
            "pyarrow".to_string(),
            ExternalPackageMapping {
                python_package: "pyarrow".to_string(),
                rust_crate: "arrow".to_string(),
                version: "50".to_string(),
                features: vec!["ipc".to_string()],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Arrow with Parquet support".to_string()),
                alternatives: vec![],
            },
        );

        // 47. scrapy → (use reqwest + scraper)
        self.packages.insert(
            "scrapy".to_string(),
            ExternalPackageMapping {
                python_package: "scrapy".to_string(),
                rust_crate: "scraper".to_string(),
                version: "0.18".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("Web scraping framework - combine reqwest + scraper".to_string()),
                alternatives: vec![],
            },
        );

        // 48. tweepy → (use Twitter API directly)
        self.packages.insert(
            "tweepy".to_string(),
            ExternalPackageMapping {
                python_package: "tweepy".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("Twitter API - use reqwest with API directly".to_string()),
                alternatives: vec![],
            },
        );

        // 49. stripe → (use Stripe API)
        self.packages.insert(
            "stripe".to_string(),
            ExternalPackageMapping {
                python_package: "stripe".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("Stripe payments - use reqwest with Stripe API".to_string()),
                alternatives: vec![],
            },
        );

        // 50. google-api-python-client → (use APIs directly)
        self.packages.insert(
            "google-api-python-client".to_string(),
            ExternalPackageMapping {
                python_package: "google-api-python-client".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                api_mappings: vec![],
                notes: Some("Google APIs - use reqwest with API directly".to_string()),
                alternatives: vec![],
            },
        );

        // 51-55: Additional packages
        // 51. tabulate → comfy-table
        self.packages.insert(
            "tabulate".to_string(),
            ExternalPackageMapping {
                python_package: "tabulate".to_string(),
                rust_crate: "comfy-table".to_string(),
                version: "7".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Pretty-print tables".to_string()),
                alternatives: vec!["tabled".to_string()],
            },
        );

        // 52. colorama → colored
        self.packages.insert(
            "colorama".to_string(),
            ExternalPackageMapping {
                python_package: "colorama".to_string(),
                rust_crate: "colored".to_string(),
                version: "2".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Full,
                api_mappings: vec![],
                notes: Some("Terminal colors".to_string()),
                alternatives: vec!["termcolor".to_string()],
            },
        );

        // 53. tqdm → indicatif
        self.packages.insert(
            "tqdm".to_string(),
            ExternalPackageMapping {
                python_package: "tqdm".to_string(),
                rust_crate: "indicatif".to_string(),
                version: "0.17".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Partial,
                api_mappings: vec![],
                notes: Some("Progress bars. ✅ WASM: Logic works (tracking progress state). ❌ WASM: Terminal rendering not available. Browser alternative: use HTML progress elements via wasm-bindgen, or log progress to console.log, or update DOM directly.".to_string()),
                alternatives: vec!["pbr".to_string()],
            },
        );

        // 54. psutil → sysinfo
        self.packages.insert(
            "psutil".to_string(),
            ExternalPackageMapping {
                python_package: "psutil".to_string(),
                rust_crate: "sysinfo".to_string(),
                version: "0.30".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Incompatible,
                api_mappings: vec![],
                notes: Some("System/process utilities - not available in WASM".to_string()),
                alternatives: vec![],
            },
        );

        // 55. paramiko → (SSH not in WASM)
        self.packages.insert(
            "paramiko".to_string(),
            ExternalPackageMapping {
                python_package: "paramiko".to_string(),
                rust_crate: "".to_string(),
                version: "*".to_string(),
                features: vec![],
                wasm_compatible: WasmCompatibility::Incompatible,
                api_mappings: vec![],
                notes: Some("SSH - not available in WASM sandbox".to_string()),
                alternatives: vec![],
            },
        );
    }

    /// Initialize packages 56-75
    fn init_packages_56_75(&mut self) {
        // 56. openpyxl → calamine
        self.packages.insert("openpyxl".to_string(), ExternalPackageMapping {
            python_package: "openpyxl".to_string(), rust_crate: "calamine".to_string(),
            version: "0.23".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Excel file reading".to_string()), alternatives: vec![],
        });

        // 57. xlrd → calamine
        self.packages.insert("xlrd".to_string(), ExternalPackageMapping {
            python_package: "xlrd".to_string(), rust_crate: "calamine".to_string(),
            version: "0.23".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Excel reading".to_string()), alternatives: vec![],
        });

        // 58. networkx → petgraph
        self.packages.insert("networkx".to_string(), ExternalPackageMapping {
            python_package: "networkx".to_string(), rust_crate: "petgraph".to_string(),
            version: "0.6".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Graph algorithms".to_string()), alternatives: vec![],
        });

        // 59. pytz → chrono-tz
        self.packages.insert("pytz".to_string(), ExternalPackageMapping {
            python_package: "pytz".to_string(), rust_crate: "chrono-tz".to_string(),
            version: "0.8".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Timezone database".to_string()), alternatives: vec![],
        });

        // 60. kafka-python → rdkafka
        self.packages.insert("kafka-python".to_string(), ExternalPackageMapping {
            python_package: "kafka-python".to_string(), rust_crate: "rdkafka".to_string(),
            version: "0.36".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("Kafka client".to_string()), alternatives: vec![],
        });

        // 61. pika → lapin
        self.packages.insert("pika".to_string(), ExternalPackageMapping {
            python_package: "pika".to_string(), rust_crate: "lapin".to_string(),
            version: "2".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("RabbitMQ client".to_string()), alternatives: vec![],
        });

        // 62. selenium → (WebDriver not in WASM)
        self.packages.insert("selenium".to_string(), ExternalPackageMapping {
            python_package: "selenium".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("Browser automation - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // 63. playwright → (Browser automation not in WASM)
        self.packages.insert("playwright".to_string(), ExternalPackageMapping {
            python_package: "playwright".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("Browser automation - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // 64. opencv-python → (Computer vision limited)
        self.packages.insert("cv2".to_string(), ExternalPackageMapping {
            python_package: "cv2".to_string(), rust_crate: "imageproc".to_string(),
            version: "0.23".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![], notes: Some("Computer vision. ✅ WASM: Basic image processing (filters, transformations, edge detection) via imageproc + image crates. ❌ WASM: Advanced CV algorithms, video processing, camera access (use browser APIs via wasm-bindgen). Use image + imageproc for basic ops, wasm-opencv bindings for advanced features.".to_string()),
            alternatives: vec!["image".to_string()],
        });

        // 65. spacy → (NLP model loading incompatible)
        self.packages.insert("spacy".to_string(), ExternalPackageMapping {
            python_package: "spacy".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("NLP - large models incompatible with WASM".to_string()),
            alternatives: vec![],
        });

        // 66. nltk → (Text processing)
        self.packages.insert("nltk".to_string(), ExternalPackageMapping {
            python_package: "nltk".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![], notes: Some("Natural language processing. ✅ WASM: Basic tokenization, stemming, string operations work (implement with regex or rust-tokenizers). ❌ WASM: Large corpus/model downloads not feasible. Embed small datasets or use lightweight alternatives like rust-tokenizers, unicode-segmentation.".to_string()),
            alternatives: vec!["rust-tokenizers".to_string()],
        });

        // 67. pypdf2 → pdf
        self.packages.insert("pypdf2".to_string(), ExternalPackageMapping {
            python_package: "pypdf2".to_string(), rust_crate: "pdf".to_string(),
            version: "0.9".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("PDF parsing".to_string()), alternatives: vec![],
        });

        // 68. reportlab → (PDF generation)
        self.packages.insert("reportlab".to_string(), ExternalPackageMapping {
            python_package: "reportlab".to_string(), rust_crate: "printpdf".to_string(),
            version: "0.7".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("PDF generation".to_string()), alternatives: vec![],
        });

        // 69. docx → (Word documents)
        self.packages.insert("docx".to_string(), ExternalPackageMapping {
            python_package: "docx".to_string(), rust_crate: "docx-rs".to_string(),
            version: "0.4".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Word document generation".to_string()), alternatives: vec![],
        });

        // 70. pillow (alternative name PIL)
        self.packages.insert("PIL".to_string(), ExternalPackageMapping {
            python_package: "PIL".to_string(), rust_crate: "image".to_string(),
            version: "0.24".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Image processing (alias for pillow)".to_string()),
            alternatives: vec![],
        });

        // 71. schedule → (Task scheduling)
        self.packages.insert("schedule".to_string(), ExternalPackageMapping {
            python_package: "schedule".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("Task scheduling - use JS setInterval/setTimeout in browser".to_string()),
            alternatives: vec![],
        });

        // 72. dateutil → chrono
        self.packages.insert("dateutil".to_string(), ExternalPackageMapping {
            python_package: "dateutil".to_string(), rust_crate: "chrono".to_string(),
            version: "0.4".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Date utilities".to_string()), alternatives: vec![],
        });

        // 73. humanize → (Human-readable formatting)
        self.packages.insert("humanize".to_string(), ExternalPackageMapping {
            python_package: "humanize".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Format numbers/dates - implement manually".to_string()),
            alternatives: vec![],
        });

        // 74. validators → (Data validation)
        self.packages.insert("validators".to_string(), ExternalPackageMapping {
            python_package: "validators".to_string(), rust_crate: "validator".to_string(),
            version: "0.16".to_string(), features: vec!["derive".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Data validation".to_string()), alternatives: vec![],
        });

        // 75. email-validator → (Email validation)
        self.packages.insert("email-validator".to_string(), ExternalPackageMapping {
            python_package: "email-validator".to_string(), rust_crate: "validator".to_string(),
            version: "0.16".to_string(), features: vec!["derive".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Email validation with validator crate".to_string()),
            alternatives: vec!["email-address".to_string()],
        });
    }

    /// Initialize packages 76-100
    fn init_packages_76_100(&mut self) {
        // 76. elasticsearch → elasticsearch
        self.packages.insert("elasticsearch".to_string(), ExternalPackageMapping {
            python_package: "elasticsearch".to_string(), rust_crate: "elasticsearch".to_string(),
            version: "8".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("Elasticsearch client".to_string()), alternatives: vec![],
        });

        // 77. sqlparse → sqlparser
        self.packages.insert("sqlparse".to_string(), ExternalPackageMapping {
            python_package: "sqlparse".to_string(), rust_crate: "sqlparser".to_string(),
            version: "0.43".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("SQL parser".to_string()), alternatives: vec![],
        });

        // 78. psycopg2 → tokio-postgres
        self.packages.insert("psycopg2".to_string(), ExternalPackageMapping {
            python_package: "psycopg2".to_string(), rust_crate: "tokio-postgres".to_string(),
            version: "0.7".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("PostgreSQL driver".to_string()), alternatives: vec![],
        });

        // 79. mysql-connector → mysql_async
        self.packages.insert("mysql-connector".to_string(), ExternalPackageMapping {
            python_package: "mysql-connector".to_string(), rust_crate: "mysql_async".to_string(),
            version: "0.33".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("MySQL driver".to_string()), alternatives: vec![],
        });

        // 80. cx_Oracle → (Oracle not in WASM)
        self.packages.insert("cx_Oracle".to_string(), ExternalPackageMapping {
            python_package: "cx_Oracle".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("Oracle database - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // 81. pyodbc → (ODBC not in WASM)
        self.packages.insert("pyodbc".to_string(), ExternalPackageMapping {
            python_package: "pyodbc".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("ODBC - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // 82. coverage → (Code coverage)
        self.packages.insert("coverage".to_string(), ExternalPackageMapping {
            python_package: "coverage".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Code coverage - use tarpaulin or cargo-llvm-cov".to_string()),
            alternatives: vec![],
        });

        // 83. mock → (Mocking)
        self.packages.insert("mock".to_string(), ExternalPackageMapping {
            python_package: "mock".to_string(), rust_crate: "mockall".to_string(),
            version: "0.12".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Mocking library".to_string()), alternatives: vec!["mockito".to_string()],
        });

        // 84. nose → (Testing framework)
        self.packages.insert("nose".to_string(), ExternalPackageMapping {
            python_package: "nose".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Testing - use Rust #[test]".to_string()),
            alternatives: vec![],
        });

        // 85. luigi → (Workflow orchestration)
        self.packages.insert("luigi".to_string(), ExternalPackageMapping {
            python_package: "luigi".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("Workflow orchestration - not WASM compatible".to_string()),
            alternatives: vec![],
        });

        // 86. airflow → (Workflow orchestration)
        self.packages.insert("airflow".to_string(), ExternalPackageMapping {
            python_package: "airflow".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("Workflow orchestration - not WASM compatible".to_string()),
            alternatives: vec![],
        });

        // 87. docker → (Container management)
        self.packages.insert("docker".to_string(), ExternalPackageMapping {
            python_package: "docker".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("Docker API - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // 88. kubernetes → (K8s management)
        self.packages.insert("kubernetes".to_string(), ExternalPackageMapping {
            python_package: "kubernetes".to_string(), rust_crate: "kube".to_string(),
            version: "0.87".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("Kubernetes API - requires HTTP client".to_string()),
            alternatives: vec![],
        });

        // 89. ansible → (Configuration management)
        self.packages.insert("ansible".to_string(), ExternalPackageMapping {
            python_package: "ansible".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("Configuration management - not WASM compatible".to_string()),
            alternatives: vec![],
        });

        // 90. fabric → (SSH deployment)
        self.packages.insert("fabric".to_string(), ExternalPackageMapping {
            python_package: "fabric".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![], notes: Some("SSH deployment - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // 91. bcrypt → bcrypt
        self.packages.insert("bcrypt".to_string(), ExternalPackageMapping {
            python_package: "bcrypt".to_string(), rust_crate: "bcrypt".to_string(),
            version: "0.15".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Password hashing".to_string()), alternatives: vec!["argon2".to_string()],
        });

        // 92. argon2 → argon2
        self.packages.insert("argon2".to_string(), ExternalPackageMapping {
            python_package: "argon2".to_string(), rust_crate: "argon2".to_string(),
            version: "0.5".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Password hashing".to_string()), alternatives: vec!["bcrypt".to_string()],
        });

        // 93. jwt → jsonwebtoken
        self.packages.insert("jwt".to_string(), ExternalPackageMapping {
            python_package: "jwt".to_string(), rust_crate: "jsonwebtoken".to_string(),
            version: "9".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("JWT tokens".to_string()), alternatives: vec![],
        });

        // 94. oauth → (OAuth)
        self.packages.insert("oauth".to_string(), ExternalPackageMapping {
            python_package: "oauth".to_string(), rust_crate: "oauth2".to_string(),
            version: "4".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("OAuth - works with JS interop".to_string()),
            alternatives: vec![],
        });

        // 95. stripe → (Stripe API)
        self.packages.insert("stripe".to_string(), ExternalPackageMapping {
            python_package: "stripe".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("Stripe payments - use stripe-js in browser".to_string()),
            alternatives: vec![],
        });

        // 96. twilio → (Twilio API)
        self.packages.insert("twilio".to_string(), ExternalPackageMapping {
            python_package: "twilio".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("Twilio API - use HTTP client".to_string()),
            alternatives: vec![],
        });

        // 97. sendgrid → (SendGrid API)
        self.packages.insert("sendgrid".to_string(), ExternalPackageMapping {
            python_package: "sendgrid".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("SendGrid email - use HTTP client".to_string()),
            alternatives: vec![],
        });

        // 98. mailgun → (Mailgun API)
        self.packages.insert("mailgun".to_string(), ExternalPackageMapping {
            python_package: "mailgun".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("Mailgun email - use HTTP client".to_string()),
            alternatives: vec![],
        });

        // 99. sentry-sdk → sentry
        self.packages.insert("sentry-sdk".to_string(), ExternalPackageMapping {
            python_package: "sentry-sdk".to_string(), rust_crate: "sentry".to_string(),
            version: "0.32".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("Error tracking - works in WASM with JS interop".to_string()),
            alternatives: vec![],
        });

        // 100. newrelic → (New Relic APM)
        self.packages.insert("newrelic".to_string(), ExternalPackageMapping {
            python_package: "newrelic".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![], notes: Some("APM - use New Relic browser agent in WASM".to_string()),
            alternatives: vec![],
        });

        // 101. poetry → (Package management) - Bonus
        self.packages.insert("poetry".to_string(), ExternalPackageMapping {
            python_package: "poetry".to_string(), rust_crate: "".to_string(),
            version: "*".to_string(), features: vec![], wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![], notes: Some("Package management - use Cargo".to_string()),
            alternatives: vec![],
        });
    }

    /// Add package mapping
    pub fn add_package(&mut self, mapping: ExternalPackageMapping) {
        self.packages.insert(mapping.python_package.clone(), mapping);
    }

    /// Initialize expanded data science packages (packages 101-120)
    fn init_expanded_data_science(&mut self) {
        // scikit-learn expanded mappings
        self.packages.insert("scikit-learn".to_string(), ExternalPackageMapping {
            python_package: "scikit-learn".to_string(),
            rust_crate: "linfa".to_string(),
            version: "0.7".to_string(),
            features: vec!["linfa-linear".to_string(), "linfa-clustering".to_string(), "linfa-trees".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![
                ApiMapping {
                    python_api: "sklearn.tree.DecisionTreeClassifier".to_string(),
                    rust_equiv: "linfa_trees::DecisionTree::params".to_string(),
                    requires_use: Some("linfa_trees::DecisionTree".to_string()),
                    transform_notes: Some("Use linfa decision trees".to_string()),
                    wasm_compatible: WasmCompatibility::Full,
                },
                ApiMapping {
                    python_api: "sklearn.ensemble.RandomForestClassifier".to_string(),
                    rust_equiv: "smartcore random forest".to_string(),
                    requires_use: Some("smartcore::ensemble::random_forest_classifier".to_string()),
                    transform_notes: Some("Use smartcore for ensemble methods".to_string()),
                    wasm_compatible: WasmCompatibility::Full,
                },
            ],
            notes: Some("ML algorithms fully work in WASM".to_string()),
            alternatives: vec!["smartcore".to_string()],
        });

        // smartcore (alternative to scikit-learn)
        self.packages.insert("smartcore".to_string(), ExternalPackageMapping {
            python_package: "smartcore".to_string(),
            rust_crate: "smartcore".to_string(),
            version: "0.3".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Comprehensive ML library, alternative to linfa".to_string()),
            alternatives: vec!["linfa".to_string()],
        });

        // seaborn → plotters
        self.packages.insert("seaborn".to_string(), ExternalPackageMapping {
            python_package: "seaborn".to_string(),
            rust_crate: "plotters".to_string(),
            version: "0.3".to_string(),
            features: vec!["all_elements".to_string()],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![
                ApiMapping {
                    python_api: "sns.heatmap".to_string(),
                    rust_equiv: "custom heatmap with plotters".to_string(),
                    requires_use: Some("plotters::prelude::*".to_string()),
                    transform_notes: Some("Build heatmap using Rectangle series".to_string()),
                    wasm_compatible: WasmCompatibility::RequiresJsInterop,
                },
            ],
            notes: Some("Statistical visualization using plotters with custom styling".to_string()),
            alternatives: vec![],
        });

        // statsmodels → statrs
        self.packages.insert("statsmodels".to_string(), ExternalPackageMapping {
            python_package: "statsmodels".to_string(),
            rust_crate: "statrs".to_string(),
            version: "0.16".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![
                ApiMapping {
                    python_api: "statsmodels.api.OLS".to_string(),
                    rust_equiv: "linfa_linear::LinearRegression".to_string(),
                    requires_use: Some("linfa_linear::LinearRegression".to_string()),
                    transform_notes: Some("Use linfa for regression, statrs for distributions".to_string()),
                    wasm_compatible: WasmCompatibility::Full,
                },
            ],
            notes: Some("Statistical modeling via statrs and linfa".to_string()),
            alternatives: vec!["linfa".to_string()],
        });

        // xarray → ndarray
        self.packages.insert("xarray".to_string(), ExternalPackageMapping {
            python_package: "xarray".to_string(),
            rust_crate: "ndarray".to_string(),
            version: "0.15".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![
                ApiMapping {
                    python_api: "xr.DataArray".to_string(),
                    rust_equiv: "Array with custom dimension tracking".to_string(),
                    requires_use: Some("ndarray::Array".to_string()),
                    transform_notes: Some("Use ndarray with HashMap for dimension labels".to_string()),
                    wasm_compatible: WasmCompatibility::Full,
                },
            ],
            notes: Some("N-dimensional labeled arrays using ndarray with custom indexing".to_string()),
            alternatives: vec![],
        });

        // sympy → symbolica (symbolic math)
        self.packages.insert("sympy".to_string(), ExternalPackageMapping {
            python_package: "sympy".to_string(),
            rust_crate: "symbolica".to_string(),
            version: "0.6".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Symbolic mathematics".to_string()),
            alternatives: vec![],
        });

        // dask → rayon
        self.packages.insert("dask".to_string(), ExternalPackageMapping {
            python_package: "dask".to_string(),
            rust_crate: "rayon".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Parallel computing. WASM: Limited thread support, use wasm-bindgen-rayon".to_string()),
            alternatives: vec!["tokio".to_string()],
        });

        // numba → (Rust is already compiled)
        self.packages.insert("numba".to_string(), ExternalPackageMapping {
            python_package: "numba".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("JIT compilation - Rust is already compiled to native/WASM".to_string()),
            alternatives: vec![],
        });

        // h5py → hdf5-rust
        self.packages.insert("h5py".to_string(), ExternalPackageMapping {
            python_package: "h5py".to_string(),
            rust_crate: "hdf5".to_string(),
            version: "0.8".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresWasi,
            api_mappings: vec![],
            notes: Some("HDF5 file format - requires filesystem access".to_string()),
            alternatives: vec![],
        });

        // zarr → (custom implementation needed)
        self.packages.insert("zarr".to_string(), ExternalPackageMapping {
            python_package: "zarr".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Chunked, compressed array storage - implement using ndarray + compression crates".to_string()),
            alternatives: vec!["ndarray".to_string()],
        });
    }

    /// Initialize expanded web frameworks (packages 121-135)
    fn init_expanded_web_frameworks(&mut self) {
        // bottle → warp
        self.packages.insert("bottle".to_string(), ExternalPackageMapping {
            python_package: "bottle".to_string(),
            rust_crate: "warp".to_string(),
            version: "0.3".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Lightweight web framework - server-side only".to_string()),
            alternatives: vec!["axum".to_string()],
        });

        // tornado → tokio + warp
        self.packages.insert("tornado".to_string(), ExternalPackageMapping {
            python_package: "tornado".to_string(),
            rust_crate: "tokio".to_string(),
            version: "1".to_string(),
            features: vec!["full".to_string()],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Async web framework - use tokio + warp/axum".to_string()),
            alternatives: vec!["warp".to_string(), "axum".to_string()],
        });

        // sanic → actix-web
        self.packages.insert("sanic".to_string(), ExternalPackageMapping {
            python_package: "sanic".to_string(),
            rust_crate: "actix-web".to_string(),
            version: "4".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Async web framework - server-side only".to_string()),
            alternatives: vec![],
        });

        // starlette → axum
        self.packages.insert("starlette".to_string(), ExternalPackageMapping {
            python_package: "starlette".to_string(),
            rust_crate: "axum".to_string(),
            version: "0.7".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("ASGI framework - use axum for similar functionality".to_string()),
            alternatives: vec!["actix-web".to_string()],
        });

        // uvicorn → (ASGI server)
        self.packages.insert("uvicorn".to_string(), ExternalPackageMapping {
            python_package: "uvicorn".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("ASGI server - use actix-web or warp server".to_string()),
            alternatives: vec![],
        });

        // gunicorn → (WSGI server)
        self.packages.insert("gunicorn".to_string(), ExternalPackageMapping {
            python_package: "gunicorn".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("WSGI server - built into Rust web frameworks".to_string()),
            alternatives: vec![],
        });

        // cherrypy → actix-web
        self.packages.insert("cherrypy".to_string(), ExternalPackageMapping {
            python_package: "cherrypy".to_string(),
            rust_crate: "actix-web".to_string(),
            version: "4".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Web framework - server-side only".to_string()),
            alternatives: vec!["warp".to_string()],
        });

        // pyramid → actix-web
        self.packages.insert("pyramid".to_string(), ExternalPackageMapping {
            python_package: "pyramid".to_string(),
            rust_crate: "actix-web".to_string(),
            version: "4".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Web framework - server-side only".to_string()),
            alternatives: vec![],
        });

        // graphene → async-graphql
        self.packages.insert("graphene".to_string(), ExternalPackageMapping {
            python_package: "graphene".to_string(),
            rust_crate: "async-graphql".to_string(),
            version: "6".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("GraphQL - server works, client needs JS interop in WASM".to_string()),
            alternatives: vec!["juniper".to_string()],
        });

        // strawberry → async-graphql
        self.packages.insert("strawberry".to_string(), ExternalPackageMapping {
            python_package: "strawberry".to_string(),
            rust_crate: "async-graphql".to_string(),
            version: "6".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("GraphQL framework".to_string()),
            alternatives: vec![],
        });
    }

    /// Initialize async/networking packages (packages 136-145)
    fn init_async_networking(&mut self) {
        // websockets → tokio-tungstenite
        self.packages.insert("websockets".to_string(), ExternalPackageMapping {
            python_package: "websockets".to_string(),
            rust_crate: "tokio-tungstenite".to_string(),
            version: "0.21".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![
                ApiMapping {
                    python_api: "websockets.connect".to_string(),
                    rust_equiv: "connect_async(url)".to_string(),
                    requires_use: Some("tokio_tungstenite::connect_async".to_string()),
                    transform_notes: Some("In WASM, use web_sys::WebSocket with wasm-bindgen".to_string()),
                    wasm_compatible: WasmCompatibility::RequiresJsInterop,
                },
            ],
            notes: Some("WebSocket - use browser WebSocket API in WASM via wasm-bindgen".to_string()),
            alternatives: vec!["ws".to_string()],
        });

        // trio → tokio
        self.packages.insert("trio".to_string(), ExternalPackageMapping {
            python_package: "trio".to_string(),
            rust_crate: "tokio".to_string(),
            version: "1".to_string(),
            features: vec!["full".to_string()],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Async I/O - use tokio with wasm-bindgen-futures in WASM".to_string()),
            alternatives: vec!["async-std".to_string()],
        });

        // anyio → async-std
        self.packages.insert("anyio".to_string(), ExternalPackageMapping {
            python_package: "anyio".to_string(),
            rust_crate: "async-std".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Async abstraction layer".to_string()),
            alternatives: vec!["tokio".to_string()],
        });

        // asyncio → tokio
        self.packages.insert("asyncio".to_string(), ExternalPackageMapping {
            python_package: "asyncio".to_string(),
            rust_crate: "tokio".to_string(),
            version: "1".to_string(),
            features: vec!["rt".to_string(), "macros".to_string()],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![
                ApiMapping {
                    python_api: "asyncio.run".to_string(),
                    rust_equiv: "tokio::runtime::Runtime::new().block_on".to_string(),
                    requires_use: Some("tokio::runtime::Runtime".to_string()),
                    transform_notes: Some("In WASM use wasm-bindgen-futures::spawn_local".to_string()),
                    wasm_compatible: WasmCompatibility::RequiresJsInterop,
                },
            ],
            notes: Some("Async runtime".to_string()),
            alternatives: vec!["async-std".to_string()],
        });

        // uvloop → tokio
        self.packages.insert("uvloop".to_string(), ExternalPackageMapping {
            python_package: "uvloop".to_string(),
            rust_crate: "tokio".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Fast event loop - Tokio is already optimized".to_string()),
            alternatives: vec![],
        });

        // urllib3 → reqwest
        self.packages.insert("urllib3".to_string(), ExternalPackageMapping {
            python_package: "urllib3".to_string(),
            rust_crate: "reqwest".to_string(),
            version: "0.11".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("HTTP client".to_string()),
            alternatives: vec!["ureq".to_string()],
        });

        // curl → curl-rust / reqwest
        self.packages.insert("pycurl".to_string(), ExternalPackageMapping {
            python_package: "pycurl".to_string(),
            rust_crate: "reqwest".to_string(),
            version: "0.11".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("HTTP client - use reqwest".to_string()),
            alternatives: vec!["curl".to_string()],
        });

        // certifi → (TLS certificates)
        self.packages.insert("certifi".to_string(), ExternalPackageMapping {
            python_package: "certifi".to_string(),
            rust_crate: "rustls".to_string(),
            version: "0.22".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("TLS certificates - rustls includes root certs".to_string()),
            alternatives: vec!["webpki-roots".to_string()],
        });

        // charset-normalizer → encoding_rs
        self.packages.insert("charset-normalizer".to_string(), ExternalPackageMapping {
            python_package: "charset-normalizer".to_string(),
            rust_crate: "encoding_rs".to_string(),
            version: "0.8".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Character encoding detection and conversion".to_string()),
            alternatives: vec![],
        });

        // idna → idna
        self.packages.insert("idna".to_string(), ExternalPackageMapping {
            python_package: "idna".to_string(),
            rust_crate: "idna".to_string(),
            version: "0.5".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Internationalized domain names".to_string()),
            alternatives: vec![],
        });
    }

    /// Initialize testing packages (packages 146-155)
    fn init_testing_packages(&mut self) {
        // tox → cargo workspace
        self.packages.insert("tox".to_string(), ExternalPackageMapping {
            python_package: "tox".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Test automation - use Cargo workspaces and profiles".to_string()),
            alternatives: vec![],
        });

        // behave → cucumber-rust
        self.packages.insert("behave".to_string(), ExternalPackageMapping {
            python_package: "behave".to_string(),
            rust_crate: "cucumber".to_string(),
            version: "0.20".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("BDD testing framework".to_string()),
            alternatives: vec![],
        });

        // locust → (load testing)
        self.packages.insert("locust".to_string(), ExternalPackageMapping {
            python_package: "locust".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Load testing - not WASM compatible".to_string()),
            alternatives: vec![],
        });

        // responses → mockito
        self.packages.insert("responses".to_string(), ExternalPackageMapping {
            python_package: "responses".to_string(),
            rust_crate: "mockito".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("HTTP mocking for tests".to_string()),
            alternatives: vec!["wiremock".to_string()],
        });

        // freezegun → (time mocking)
        self.packages.insert("freezegun".to_string(), ExternalPackageMapping {
            python_package: "freezegun".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Time mocking - use dependency injection or mock time providers".to_string()),
            alternatives: vec![],
        });

        // bandit → cargo-audit
        self.packages.insert("bandit".to_string(), ExternalPackageMapping {
            python_package: "bandit".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Security linter - use cargo-audit and cargo-deny".to_string()),
            alternatives: vec![],
        });

        // safety → cargo-audit
        self.packages.insert("safety".to_string(), ExternalPackageMapping {
            python_package: "safety".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Dependency security scanner - use cargo-audit".to_string()),
            alternatives: vec![],
        });

        // pytest-asyncio → tokio::test
        self.packages.insert("pytest-asyncio".to_string(), ExternalPackageMapping {
            python_package: "pytest-asyncio".to_string(),
            rust_crate: "tokio".to_string(),
            version: "1".to_string(),
            features: vec!["test-util".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Async testing - use #[tokio::test]".to_string()),
            alternatives: vec![],
        });

        // pytest-mock → mockall
        self.packages.insert("pytest-mock".to_string(), ExternalPackageMapping {
            python_package: "pytest-mock".to_string(),
            rust_crate: "mockall".to_string(),
            version: "0.12".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Mocking framework".to_string()),
            alternatives: vec!["mockito".to_string()],
        });

        // pytest-benchmark → criterion
        self.packages.insert("pytest-benchmark".to_string(), ExternalPackageMapping {
            python_package: "pytest-benchmark".to_string(),
            rust_crate: "criterion".to_string(),
            version: "0.5".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Benchmarking - use criterion or built-in bencher".to_string()),
            alternatives: vec![],
        });
    }

    /// Initialize CLI/TUI packages (packages 156-165)
    fn init_cli_tui_packages(&mut self) {
        // typer → clap
        self.packages.insert("typer".to_string(), ExternalPackageMapping {
            python_package: "typer".to_string(),
            rust_crate: "clap".to_string(),
            version: "4".to_string(),
            features: vec!["derive".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![
                ApiMapping {
                    python_api: "typer.Option".to_string(),
                    rust_equiv: "#[arg(short, long)]".to_string(),
                    requires_use: Some("clap::Parser".to_string()),
                    transform_notes: Some("Use clap derive macros".to_string()),
                    wasm_compatible: WasmCompatibility::Full,
                },
            ],
            notes: Some("CLI framework with type hints".to_string()),
            alternatives: vec![],
        });

        // rich → console / termion
        self.packages.insert("rich".to_string(), ExternalPackageMapping {
            python_package: "rich".to_string(),
            rust_crate: "console".to_string(),
            version: "0.15".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Rich terminal output - limited in WASM, use for native CLIs".to_string()),
            alternatives: vec!["termion".to_string(), "crossterm".to_string()],
        });

        // prompt_toolkit → rustyline
        self.packages.insert("prompt_toolkit".to_string(), ExternalPackageMapping {
            python_package: "prompt_toolkit".to_string(),
            rust_crate: "rustyline".to_string(),
            version: "13".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Interactive prompts - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // blessed → crossterm
        self.packages.insert("blessed".to_string(), ExternalPackageMapping {
            python_package: "blessed".to_string(),
            rust_crate: "crossterm".to_string(),
            version: "0.27".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Terminal manipulation".to_string()),
            alternatives: vec!["termion".to_string()],
        });

        // questionary → dialoguer
        self.packages.insert("questionary".to_string(), ExternalPackageMapping {
            python_package: "questionary".to_string(),
            rust_crate: "dialoguer".to_string(),
            version: "0.11".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Interactive prompts".to_string()),
            alternatives: vec!["inquire".to_string()],
        });

        // inquirer → inquire
        self.packages.insert("inquirer".to_string(), ExternalPackageMapping {
            python_package: "inquirer".to_string(),
            rust_crate: "inquire".to_string(),
            version: "0.6".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Interactive prompts".to_string()),
            alternatives: vec!["dialoguer".to_string()],
        });

        // textual → tui-rs
        self.packages.insert("textual".to_string(), ExternalPackageMapping {
            python_package: "textual".to_string(),
            rust_crate: "ratatui".to_string(),
            version: "0.25".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("TUI framework - use ratatui (formerly tui-rs)".to_string()),
            alternatives: vec![],
        });

        // argparse → clap
        self.packages.insert("argparse".to_string(), ExternalPackageMapping {
            python_package: "argparse".to_string(),
            rust_crate: "clap".to_string(),
            version: "4".to_string(),
            features: vec!["derive".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("CLI argument parsing".to_string()),
            alternatives: vec![],
        });

        // docopt → docopt
        self.packages.insert("docopt".to_string(), ExternalPackageMapping {
            python_package: "docopt".to_string(),
            rust_crate: "docopt".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("CLI from docstrings".to_string()),
            alternatives: vec!["clap".to_string()],
        });

        // fire → clap
        self.packages.insert("fire".to_string(), ExternalPackageMapping {
            python_package: "fire".to_string(),
            rust_crate: "clap".to_string(),
            version: "4".to_string(),
            features: vec!["derive".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Auto CLI generation - use clap derive".to_string()),
            alternatives: vec![],
        });
    }

    /// Initialize DevOps packages (packages 166-180)
    fn init_devops_packages(&mut self) {
        // docker (bollard)
        self.packages.insert("docker".to_string(), ExternalPackageMapping {
            python_package: "docker".to_string(),
            rust_crate: "bollard".to_string(),
            version: "0.15".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![
                ApiMapping {
                    python_api: "docker.from_env".to_string(),
                    rust_equiv: "Docker::connect_with_socket_defaults".to_string(),
                    requires_use: Some("bollard::Docker".to_string()),
                    transform_notes: Some("Async Docker client".to_string()),
                    wasm_compatible: WasmCompatibility::Incompatible,
                },
            ],
            notes: Some("Docker API client - server-side only".to_string()),
            alternatives: vec![],
        });

        // kubernetes (kube-rs)
        self.packages.insert("kubernetes".to_string(), ExternalPackageMapping {
            python_package: "kubernetes".to_string(),
            rust_crate: "kube".to_string(),
            version: "0.87".to_string(),
            features: vec!["runtime".to_string(), "client".to_string()],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![
                ApiMapping {
                    python_api: "kubernetes.client.CoreV1Api".to_string(),
                    rust_equiv: "Api::<Pod>::all".to_string(),
                    requires_use: Some("kube::Api".to_string()),
                    transform_notes: Some("Type-safe Kubernetes API".to_string()),
                    wasm_compatible: WasmCompatibility::RequiresJsInterop,
                },
            ],
            notes: Some("Kubernetes client - requires HTTP access".to_string()),
            alternatives: vec![],
        });

        // invoke → (task runner)
        self.packages.insert("invoke".to_string(), ExternalPackageMapping {
            python_package: "invoke".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Task runner - use Makefile, just, or cargo-make".to_string()),
            alternatives: vec![],
        });

        // virtualenv → (Rust uses Cargo)
        self.packages.insert("virtualenv".to_string(), ExternalPackageMapping {
            python_package: "virtualenv".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Virtual environments - Rust uses Cargo workspaces".to_string()),
            alternatives: vec![],
        });

        // pipenv → (Rust uses Cargo)
        self.packages.insert("pipenv".to_string(), ExternalPackageMapping {
            python_package: "pipenv".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Package management - use Cargo".to_string()),
            alternatives: vec![],
        });

        // conda → (package manager)
        self.packages.insert("conda".to_string(), ExternalPackageMapping {
            python_package: "conda".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Package/environment manager - use Cargo".to_string()),
            alternatives: vec![],
        });

        // git → git2-rs
        self.packages.insert("git".to_string(), ExternalPackageMapping {
            python_package: "git".to_string(),
            rust_crate: "git2".to_string(),
            version: "0.18".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Git operations - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // subprocess → std::process
        self.packages.insert("subprocess".to_string(), ExternalPackageMapping {
            python_package: "subprocess".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Process execution - use std::process::Command (not in WASM)".to_string()),
            alternatives: vec![],
        });

        // sh → (shell commands)
        self.packages.insert("sh".to_string(), ExternalPackageMapping {
            python_package: "sh".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Shell commands - use std::process::Command".to_string()),
            alternatives: vec![],
        });

        // systemd → (system service)
        self.packages.insert("systemd".to_string(), ExternalPackageMapping {
            python_package: "systemd".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Systemd - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // supervisor → (process control)
        self.packages.insert("supervisor".to_string(), ExternalPackageMapping {
            python_package: "supervisor".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Process control - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // terraform → (IaC)
        self.packages.insert("terraform".to_string(), ExternalPackageMapping {
            python_package: "terraform".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Infrastructure as Code - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // pulumi → (IaC)
        self.packages.insert("pulumi".to_string(), ExternalPackageMapping {
            python_package: "pulumi".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Infrastructure as Code - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // watchdog → notify
        self.packages.insert("watchdog".to_string(), ExternalPackageMapping {
            python_package: "watchdog".to_string(),
            rust_crate: "notify".to_string(),
            version: "6".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Filesystem monitoring - requires OS access".to_string()),
            alternatives: vec![],
        });

        // filelock → fs2
        self.packages.insert("filelock".to_string(), ExternalPackageMapping {
            python_package: "filelock".to_string(),
            rust_crate: "fs2".to_string(),
            version: "0.4".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresWasi,
            api_mappings: vec![],
            notes: Some("File locking - requires filesystem".to_string()),
            alternatives: vec![],
        });
    }

    /// Initialize database packages (packages 181-195)
    fn init_database_packages(&mut self) {
        // asyncpg → tokio-postgres
        self.packages.insert("asyncpg".to_string(), ExternalPackageMapping {
            python_package: "asyncpg".to_string(),
            rust_crate: "tokio-postgres".to_string(),
            version: "0.7".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Async PostgreSQL driver".to_string()),
            alternatives: vec!["sqlx".to_string()],
        });

        // psycopg3 → tokio-postgres
        self.packages.insert("psycopg3".to_string(), ExternalPackageMapping {
            python_package: "psycopg3".to_string(),
            rust_crate: "tokio-postgres".to_string(),
            version: "0.7".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("PostgreSQL driver".to_string()),
            alternatives: vec![],
        });

        // aiomysql → mysql_async
        self.packages.insert("aiomysql".to_string(), ExternalPackageMapping {
            python_package: "aiomysql".to_string(),
            rust_crate: "mysql_async".to_string(),
            version: "0.33".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Async MySQL driver".to_string()),
            alternatives: vec![],
        });

        // motor → mongodb (async)
        self.packages.insert("motor".to_string(), ExternalPackageMapping {
            python_package: "motor".to_string(),
            rust_crate: "mongodb".to_string(),
            version: "2".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Async MongoDB driver".to_string()),
            alternatives: vec![],
        });

        // cassandra-driver → cassandra-cpp / scylla
        self.packages.insert("cassandra-driver".to_string(), ExternalPackageMapping {
            python_package: "cassandra-driver".to_string(),
            rust_crate: "scylla".to_string(),
            version: "0.12".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Cassandra/ScyllaDB driver".to_string()),
            alternatives: vec![],
        });

        // neo4j → neo4rs
        self.packages.insert("neo4j".to_string(), ExternalPackageMapping {
            python_package: "neo4j".to_string(),
            rust_crate: "neo4rs".to_string(),
            version: "0.7".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Neo4j graph database driver".to_string()),
            alternatives: vec![],
        });

        // influxdb → influxdb
        self.packages.insert("influxdb".to_string(), ExternalPackageMapping {
            python_package: "influxdb".to_string(),
            rust_crate: "influxdb".to_string(),
            version: "0.7".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("InfluxDB time-series database client".to_string()),
            alternatives: vec![],
        });

        // clickhouse-driver → clickhouse-rs
        self.packages.insert("clickhouse-driver".to_string(), ExternalPackageMapping {
            python_package: "clickhouse-driver".to_string(),
            rust_crate: "clickhouse".to_string(),
            version: "0.11".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("ClickHouse database client".to_string()),
            alternatives: vec![],
        });

        // aiosqlite → rusqlite with async
        self.packages.insert("aiosqlite".to_string(), ExternalPackageMapping {
            python_package: "aiosqlite".to_string(),
            rust_crate: "sqlx".to_string(),
            version: "0.7".to_string(),
            features: vec!["sqlite".to_string()],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Async SQLite - works with sql.js in WASM".to_string()),
            alternatives: vec!["rusqlite".to_string()],
        });

        // peewee → diesel
        self.packages.insert("peewee".to_string(), ExternalPackageMapping {
            python_package: "peewee".to_string(),
            rust_crate: "diesel".to_string(),
            version: "2".to_string(),
            features: vec!["sqlite".to_string()],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Lightweight ORM".to_string()),
            alternatives: vec!["sea-orm".to_string()],
        });

        // tortoise-orm → sea-orm
        self.packages.insert("tortoise-orm".to_string(), ExternalPackageMapping {
            python_package: "tortoise-orm".to_string(),
            rust_crate: "sea-orm".to_string(),
            version: "0.12".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Async ORM".to_string()),
            alternatives: vec!["diesel".to_string()],
        });

        // orator → diesel
        self.packages.insert("orator".to_string(), ExternalPackageMapping {
            python_package: "orator".to_string(),
            rust_crate: "diesel".to_string(),
            version: "2".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("ORM".to_string()),
            alternatives: vec![],
        });

        // databases → sqlx
        self.packages.insert("databases".to_string(), ExternalPackageMapping {
            python_package: "databases".to_string(),
            rust_crate: "sqlx".to_string(),
            version: "0.7".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Async database support".to_string()),
            alternatives: vec![],
        });

        // sqlite3 → rusqlite
        self.packages.insert("sqlite3".to_string(), ExternalPackageMapping {
            python_package: "sqlite3".to_string(),
            rust_crate: "rusqlite".to_string(),
            version: "0.30".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("SQLite - works with sql.js in WASM".to_string()),
            alternatives: vec!["sqlx".to_string()],
        });

        // duckdb → duckdb-rs
        self.packages.insert("duckdb".to_string(), ExternalPackageMapping {
            python_package: "duckdb".to_string(),
            rust_crate: "duckdb".to_string(),
            version: "0.10".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("DuckDB analytics database - WASM support experimental".to_string()),
            alternatives: vec![],
        });
    }

    /// Initialize data format packages (packages 196-205)
    fn init_data_format_packages(&mut self) {
        // msgpack → rmp-serde
        self.packages.insert("msgpack".to_string(), ExternalPackageMapping {
            python_package: "msgpack".to_string(),
            rust_crate: "rmp-serde".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("MessagePack serialization".to_string()),
            alternatives: vec![],
        });

        // cbor2 → ciborium
        self.packages.insert("cbor2".to_string(), ExternalPackageMapping {
            python_package: "cbor2".to_string(),
            rust_crate: "ciborium".to_string(),
            version: "0.2".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("CBOR serialization".to_string()),
            alternatives: vec!["serde_cbor".to_string()],
        });

        // toml → toml
        self.packages.insert("toml".to_string(), ExternalPackageMapping {
            python_package: "toml".to_string(),
            rust_crate: "toml".to_string(),
            version: "0.8".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("TOML parsing and serialization".to_string()),
            alternatives: vec![],
        });

        // configparser → ini
        self.packages.insert("configparser".to_string(), ExternalPackageMapping {
            python_package: "configparser".to_string(),
            rust_crate: "ini".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("INI file parsing".to_string()),
            alternatives: vec![],
        });

        // parquet → parquet
        self.packages.insert("parquet".to_string(), ExternalPackageMapping {
            python_package: "parquet".to_string(),
            rust_crate: "parquet".to_string(),
            version: "50".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Parquet columnar format".to_string()),
            alternatives: vec![],
        });

        // avro → apache-avro
        self.packages.insert("avro".to_string(), ExternalPackageMapping {
            python_package: "avro".to_string(),
            rust_crate: "apache-avro".to_string(),
            version: "0.16".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Apache Avro serialization".to_string()),
            alternatives: vec![],
        });

        // orjson → serde_json
        self.packages.insert("orjson".to_string(), ExternalPackageMapping {
            python_package: "orjson".to_string(),
            rust_crate: "serde_json".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Fast JSON - use serde_json or simd-json".to_string()),
            alternatives: vec!["simd-json".to_string()],
        });

        // ujson → serde_json
        self.packages.insert("ujson".to_string(), ExternalPackageMapping {
            python_package: "ujson".to_string(),
            rust_crate: "serde_json".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Ultra-fast JSON".to_string()),
            alternatives: vec![],
        });

        // jsonschema → jsonschema
        self.packages.insert("jsonschema".to_string(), ExternalPackageMapping {
            python_package: "jsonschema".to_string(),
            rust_crate: "jsonschema".to_string(),
            version: "0.17".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("JSON Schema validation".to_string()),
            alternatives: vec![],
        });

        // xmltodict → quick-xml + serde
        self.packages.insert("xmltodict".to_string(), ExternalPackageMapping {
            python_package: "xmltodict".to_string(),
            rust_crate: "quick-xml".to_string(),
            version: "0.31".to_string(),
            features: vec!["serialize".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("XML to dict conversion".to_string()),
            alternatives: vec!["serde-xml-rs".to_string()],
        });
    }

    /// Initialize security packages (packages 206-215)
    fn init_security_packages(&mut self) {
        // pyjwt → jsonwebtoken
        self.packages.insert("pyjwt".to_string(), ExternalPackageMapping {
            python_package: "pyjwt".to_string(),
            rust_crate: "jsonwebtoken".to_string(),
            version: "9".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("JWT encoding/decoding".to_string()),
            alternatives: vec![],
        });

        // oauthlib → oauth2
        self.packages.insert("oauthlib".to_string(), ExternalPackageMapping {
            python_package: "oauthlib".to_string(),
            rust_crate: "oauth2".to_string(),
            version: "4".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("OAuth 2.0".to_string()),
            alternatives: vec![],
        });

        // authlib → oauth2
        self.packages.insert("authlib".to_string(), ExternalPackageMapping {
            python_package: "authlib".to_string(),
            rust_crate: "oauth2".to_string(),
            version: "4".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Authentication library".to_string()),
            alternatives: vec![],
        });

        // itsdangerous → (session signing)
        self.packages.insert("itsdangerous".to_string(), ExternalPackageMapping {
            python_package: "itsdangerous".to_string(),
            rust_crate: "ring".to_string(),
            version: "0.17".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Cryptographic signing - use ring or ed25519-dalek".to_string()),
            alternatives: vec!["ed25519-dalek".to_string()],
        });

        // scrypt → scrypt
        self.packages.insert("scrypt".to_string(), ExternalPackageMapping {
            python_package: "scrypt".to_string(),
            rust_crate: "scrypt".to_string(),
            version: "0.11".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Scrypt password hashing".to_string()),
            alternatives: vec!["argon2".to_string()],
        });

        // nacl → sodiumoxide
        self.packages.insert("nacl".to_string(), ExternalPackageMapping {
            python_package: "nacl".to_string(),
            rust_crate: "sodiumoxide".to_string(),
            version: "0.2".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("NaCl cryptography - limited WASM support".to_string()),
            alternatives: vec!["ed25519-dalek".to_string(), "x25519-dalek".to_string()],
        });

        // paramiko → (SSH library)
        self.packages.insert("paramiko".to_string(), ExternalPackageMapping {
            python_package: "paramiko".to_string(),
            rust_crate: "ssh2".to_string(),
            version: "0.9".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("SSH client - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // keyring → (credential storage)
        self.packages.insert("keyring".to_string(), ExternalPackageMapping {
            python_package: "keyring".to_string(),
            rust_crate: "keyring".to_string(),
            version: "2".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Credential storage - OS specific, not in WASM".to_string()),
            alternatives: vec![],
        });

        // secrets → getrandom
        self.packages.insert("secrets".to_string(), ExternalPackageMapping {
            python_package: "secrets".to_string(),
            rust_crate: "getrandom".to_string(),
            version: "0.2".to_string(),
            features: vec!["js".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Cryptographically strong random numbers - works in WASM".to_string()),
            alternatives: vec!["rand".to_string()],
        });

        // hashlib → sha2 / blake3
        self.packages.insert("hashlib".to_string(), ExternalPackageMapping {
            python_package: "hashlib".to_string(),
            rust_crate: "sha2".to_string(),
            version: "0.10".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Hashing algorithms".to_string()),
            alternatives: vec!["blake3".to_string(), "md-5".to_string()],
        });
    }

    /// Initialize messaging packages (packages 216-225)
    fn init_messaging_packages(&mut self) {
        // aio-pika → lapin
        self.packages.insert("aio-pika".to_string(), ExternalPackageMapping {
            python_package: "aio-pika".to_string(),
            rust_crate: "lapin".to_string(),
            version: "2".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Async RabbitMQ client".to_string()),
            alternatives: vec![],
        });

        // aiokafka → rdkafka
        self.packages.insert("aiokafka".to_string(), ExternalPackageMapping {
            python_package: "aiokafka".to_string(),
            rust_crate: "rdkafka".to_string(),
            version: "0.36".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("Async Kafka client".to_string()),
            alternatives: vec![],
        });

        // nats → nats
        self.packages.insert("nats".to_string(), ExternalPackageMapping {
            python_package: "nats".to_string(),
            rust_crate: "async-nats".to_string(),
            version: "0.33".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("NATS messaging client".to_string()),
            alternatives: vec![],
        });

        // paho-mqtt → rumqttc
        self.packages.insert("paho-mqtt".to_string(), ExternalPackageMapping {
            python_package: "paho-mqtt".to_string(),
            rust_crate: "rumqttc".to_string(),
            version: "0.23".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("MQTT client".to_string()),
            alternatives: vec!["paho-mqtt".to_string()],
        });

        // zeromq → zmq
        self.packages.insert("zeromq".to_string(), ExternalPackageMapping {
            python_package: "zeromq".to_string(),
            rust_crate: "zmq".to_string(),
            version: "0.10".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("ZeroMQ messaging - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // Additional popular packages (221-240)

        // regex → regex
        self.packages.insert("regex".to_string(), ExternalPackageMapping {
            python_package: "regex".to_string(),
            rust_crate: "regex".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Regular expressions - fully compatible".to_string()),
            alternatives: vec![],
        });

        // uuid → uuid
        self.packages.insert("uuid".to_string(), ExternalPackageMapping {
            python_package: "uuid".to_string(),
            rust_crate: "uuid".to_string(),
            version: "1".to_string(),
            features: vec!["v4".to_string(), "serde".to_string()],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("UUID generation".to_string()),
            alternatives: vec![],
        });

        // pillow-simd → image (same as pillow)
        self.packages.insert("pillow-simd".to_string(), ExternalPackageMapping {
            python_package: "pillow-simd".to_string(),
            rust_crate: "image".to_string(),
            version: "0.24".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("SIMD-optimized image processing".to_string()),
            alternatives: vec![],
        });

        // multiprocessing → rayon
        self.packages.insert("multiprocessing".to_string(), ExternalPackageMapping {
            python_package: "multiprocessing".to_string(),
            rust_crate: "rayon".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Parallel processing - limited WASM threading".to_string()),
            alternatives: vec!["tokio".to_string()],
        });

        // threading → std::thread
        self.packages.insert("threading".to_string(), ExternalPackageMapping {
            python_package: "threading".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Threading - use std::thread (limited WASM support)".to_string()),
            alternatives: vec![],
        });

        // queue → crossbeam-channel
        self.packages.insert("queue".to_string(), ExternalPackageMapping {
            python_package: "queue".to_string(),
            rust_crate: "crossbeam-channel".to_string(),
            version: "0.5".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Thread-safe queues".to_string()),
            alternatives: vec!["flume".to_string()],
        });

        // ftplib → suppaftp
        self.packages.insert("ftplib".to_string(), ExternalPackageMapping {
            python_package: "ftplib".to_string(),
            rust_crate: "suppaftp".to_string(),
            version: "5".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("FTP client - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // smtplib → lettre
        self.packages.insert("smtplib".to_string(), ExternalPackageMapping {
            python_package: "smtplib".to_string(),
            rust_crate: "lettre".to_string(),
            version: "0.11".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("SMTP email client".to_string()),
            alternatives: vec![],
        });

        // imaplib → async-imap
        self.packages.insert("imaplib".to_string(), ExternalPackageMapping {
            python_package: "imaplib".to_string(),
            rust_crate: "async-imap".to_string(),
            version: "0.9".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            api_mappings: vec![],
            notes: Some("IMAP email client".to_string()),
            alternatives: vec![],
        });

        // telnetlib → (no direct equivalent)
        self.packages.insert("telnetlib".to_string(), ExternalPackageMapping {
            python_package: "telnetlib".to_string(),
            rust_crate: "".to_string(),
            version: "*".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Incompatible,
            api_mappings: vec![],
            notes: Some("Telnet client - not available in WASM".to_string()),
            alternatives: vec![],
        });

        // zipfile → zip
        self.packages.insert("zipfile".to_string(), ExternalPackageMapping {
            python_package: "zipfile".to_string(),
            rust_crate: "zip".to_string(),
            version: "0.6".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("ZIP archive handling".to_string()),
            alternatives: vec![],
        });

        // tarfile → tar
        self.packages.insert("tarfile".to_string(), ExternalPackageMapping {
            python_package: "tarfile".to_string(),
            rust_crate: "tar".to_string(),
            version: "0.4".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("TAR archive handling".to_string()),
            alternatives: vec![],
        });

        // gzip → flate2
        self.packages.insert("gzip".to_string(), ExternalPackageMapping {
            python_package: "gzip".to_string(),
            rust_crate: "flate2".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("GZIP compression".to_string()),
            alternatives: vec![],
        });

        // bz2 → bzip2
        self.packages.insert("bz2".to_string(), ExternalPackageMapping {
            python_package: "bz2".to_string(),
            rust_crate: "bzip2".to_string(),
            version: "0.4".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("BZIP2 compression".to_string()),
            alternatives: vec![],
        });

        // lzma → xz2
        self.packages.insert("lzma".to_string(), ExternalPackageMapping {
            python_package: "lzma".to_string(),
            rust_crate: "xz2".to_string(),
            version: "0.1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("LZMA compression".to_string()),
            alternatives: vec![],
        });

        // zlib → flate2
        self.packages.insert("zlib".to_string(), ExternalPackageMapping {
            python_package: "zlib".to_string(),
            rust_crate: "flate2".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Zlib compression".to_string()),
            alternatives: vec![],
        });

        // base64 → base64
        self.packages.insert("base64".to_string(), ExternalPackageMapping {
            python_package: "base64".to_string(),
            rust_crate: "base64".to_string(),
            version: "0.21".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Base64 encoding/decoding".to_string()),
            alternatives: vec![],
        });

        // urllib → url
        self.packages.insert("urllib".to_string(), ExternalPackageMapping {
            python_package: "urllib".to_string(),
            rust_crate: "url".to_string(),
            version: "2".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("URL parsing and manipulation".to_string()),
            alternatives: vec![],
        });

        // tempfile → tempfile
        self.packages.insert("tempfile".to_string(), ExternalPackageMapping {
            python_package: "tempfile".to_string(),
            rust_crate: "tempfile".to_string(),
            version: "3".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::RequiresWasi,
            api_mappings: vec![],
            notes: Some("Temporary files - requires filesystem".to_string()),
            alternatives: vec![],
        });

        // pickle → bincode
        self.packages.insert("pickle".to_string(), ExternalPackageMapping {
            python_package: "pickle".to_string(),
            rust_crate: "bincode".to_string(),
            version: "1".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Full,
            api_mappings: vec![],
            notes: Some("Binary serialization - use bincode or serde".to_string()),
            alternatives: vec!["serde".to_string()],
        });

        // shelve → (no direct equivalent)
        self.packages.insert("shelve".to_string(), ExternalPackageMapping {
            python_package: "shelve".to_string(),
            rust_crate: "sled".to_string(),
            version: "0.34".to_string(),
            features: vec![],
            wasm_compatible: WasmCompatibility::Partial,
            api_mappings: vec![],
            notes: Some("Persistent dictionary - use sled or rocksdb".to_string()),
            alternatives: vec!["rocksdb".to_string()],
        });
    }
}

impl Default for ExternalPackageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Package statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageStats {
    pub total_packages: usize,
    pub full_wasm_compat: usize,
    pub partial_wasm_compat: usize,
    pub requires_js_interop: usize,
    pub incompatible: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ExternalPackageRegistry::new();
        assert!(!registry.packages.is_empty());
    }

    #[test]
    fn test_get_numpy_mapping() {
        let registry = ExternalPackageRegistry::new();
        let numpy = registry.get_package("numpy");

        assert!(numpy.is_some());
        let numpy = numpy.unwrap();
        assert_eq!(numpy.rust_crate, "ndarray");
        assert_eq!(numpy.wasm_compatible, WasmCompatibility::Full);
    }

    #[test]
    fn test_get_pandas_mapping() {
        let registry = ExternalPackageRegistry::new();
        let pandas = registry.get_package("pandas");

        assert!(pandas.is_some());
        let pandas = pandas.unwrap();
        assert_eq!(pandas.rust_crate, "polars");
    }

    #[test]
    fn test_api_mapping() {
        let registry = ExternalPackageRegistry::new();
        let numpy_array = registry.get_api_mapping("numpy", "numpy.array");

        assert!(numpy_array.is_some());
        assert!(numpy_array.unwrap().contains("ndarray"));
    }

    #[test]
    fn test_stats() {
        let registry = ExternalPackageRegistry::new();
        let stats = registry.stats();

        println!("Total packages: {}", stats.total_packages);
        println!("Full WASM: {}", stats.full_wasm_compat);
        println!("Partial WASM: {}", stats.partial_wasm_compat);
        println!("Requires JS: {}", stats.requires_js_interop);
        println!("Incompatible: {}", stats.incompatible);

        assert!(stats.total_packages >= 220, "Should have at least 220 packages, got {}", stats.total_packages);
        assert!(stats.full_wasm_compat > 0);
    }

    #[test]
    fn test_expanded_packages() {
        let registry = ExternalPackageRegistry::new();

        // Test data science expansions
        assert!(registry.get_package("scikit-learn").is_some());
        assert!(registry.get_package("seaborn").is_some());
        assert!(registry.get_package("statsmodels").is_some());
        assert!(registry.get_package("xarray").is_some());

        // Test web frameworks
        assert!(registry.get_package("bottle").is_some());
        assert!(registry.get_package("tornado").is_some());
        assert!(registry.get_package("sanic").is_some());

        // Test async/networking
        assert!(registry.get_package("websockets").is_some());
        assert!(registry.get_package("trio").is_some());
        assert!(registry.get_package("anyio").is_some());

        // Test testing packages
        assert!(registry.get_package("behave").is_some());
        assert!(registry.get_package("tox").is_some());

        // Test CLI/TUI
        assert!(registry.get_package("typer").is_some());
        assert!(registry.get_package("rich").is_some());

        // Test DevOps
        assert!(registry.get_package("docker").is_some());
        assert!(registry.get_package("kubernetes").is_some());

        // Test databases
        assert!(registry.get_package("asyncpg").is_some());
        assert!(registry.get_package("motor").is_some());

        // Test data formats
        assert!(registry.get_package("msgpack").is_some());
        assert!(registry.get_package("parquet").is_some());
    }

    #[test]
    fn test_api_mappings_expanded() {
        let registry = ExternalPackageRegistry::new();

        // Test scipy API mappings
        let scipy = registry.get_package("scipy").unwrap();
        assert!(!scipy.api_mappings.is_empty());
        assert!(scipy.api_mappings.iter().any(|m| m.python_api.contains("linalg")));

        // Test matplotlib API mappings
        let matplotlib = registry.get_package("matplotlib").unwrap();
        assert!(!matplotlib.api_mappings.is_empty());

        // Test websockets API mappings
        let websockets = registry.get_package("websockets").unwrap();
        assert!(!websockets.api_mappings.is_empty());
    }
}
