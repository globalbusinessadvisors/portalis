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
                api_mappings: vec![],
                notes: Some("Linear algebra via nalgebra, stats via statrs".to_string()),
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
                api_mappings: vec![],
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

        assert_eq!(stats.total_packages, 100, "Should have exactly 100 packages");
        assert!(stats.full_wasm_compat > 0);
    }
}
