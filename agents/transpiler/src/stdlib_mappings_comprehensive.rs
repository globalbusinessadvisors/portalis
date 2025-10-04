

//! Comprehensive Python Standard Library to Rust Mappings
//!
//! This module provides extensive mappings for Python stdlib modules to Rust equivalents
//! with WASM compatibility information.

use super::stdlib_mapper::{ModuleMapping, FunctionMapping, WasmCompatibility};

/// Initialize all critical stdlib module mappings (Priority: Critical - 50 modules)
pub fn init_critical_mappings() -> Vec<(ModuleMapping, Vec<FunctionMapping>)> {
    let mut mappings = Vec::new();

    // ========== Math & Numbers ==========

    // math - Mathematical functions
    mappings.push((
        ModuleMapping {
            python_module: "math".to_string(),
            rust_crate: None,
            rust_use: "std::f64::consts".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Pure computation, fully WASM compatible".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "sqrt".to_string(),
                rust_equiv: "f64::sqrt".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "pow".to_string(),
                rust_equiv: "f64::powf".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "floor".to_string(),
                rust_equiv: "f64::floor".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "ceil".to_string(),
                rust_equiv: "f64::ceil".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "pi".to_string(),
                rust_equiv: "std::f64::consts::PI".to_string(),
                requires_use: Some("std::f64::consts::PI".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "e".to_string(),
                rust_equiv: "std::f64::consts::E".to_string(),
                requires_use: Some("std::f64::consts::E".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // ========== I/O & File System (WASI required) ==========

    // pathlib - Path operations
    mappings.push((
        ModuleMapping {
            python_module: "pathlib".to_string(),
            rust_crate: None,
            rust_use: "std::path::Path".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Requires WASI for filesystem access in browser".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "Path".to_string(),
                rust_equiv: "Path::new".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "exists".to_string(),
                rust_equiv: "path.exists()".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "is_file".to_string(),
                rust_equiv: "path.is_file()".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "is_dir".to_string(),
                rust_equiv: "path.is_dir()".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
        ],
    ));

    // io - Core I/O operations
    mappings.push((
        ModuleMapping {
            python_module: "io".to_string(),
            rust_crate: None,
            rust_use: "std::io".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Basic I/O works, file operations need WASI".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "StringIO".to_string(),
                rust_equiv: "String::new".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("In-memory string buffer, no I/O needed".to_string()),
            },
            FunctionMapping {
                python_name: "BytesIO".to_string(),
                rust_equiv: "Vec::<u8>::new".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("In-memory bytes buffer".to_string()),
            },
        ],
    ));

    // tempfile - Temporary files
    mappings.push((
        ModuleMapping {
            python_module: "tempfile".to_string(),
            rust_crate: Some("tempfile".to_string()),
            rust_use: "tempfile".to_string(),
            dependencies: vec![],
            version: "3".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Requires WASI filesystem support".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "TemporaryFile".to_string(),
                rust_equiv: "tempfile::tempfile".to_string(),
                requires_use: Some("tempfile".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "NamedTemporaryFile".to_string(),
                rust_equiv: "tempfile::NamedTempFile::new".to_string(),
                requires_use: Some("tempfile::NamedTempFile".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
        ],
    ));

    // glob - Filename pattern matching
    mappings.push((
        ModuleMapping {
            python_module: "glob".to_string(),
            rust_crate: Some("glob".to_string()),
            rust_use: "glob".to_string(),
            dependencies: vec![],
            version: "0.3".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Needs filesystem access".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "glob".to_string(),
                rust_equiv: "glob::glob".to_string(),
                requires_use: Some("glob".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns iterator instead of list".to_string()),
            },
        ],
    ));

    // ========== Data Structures & Algorithms ==========

    // collections - Advanced collections
    mappings.push((
        ModuleMapping {
            python_module: "collections".to_string(),
            rust_crate: Some("indexmap".to_string()),
            rust_use: "std::collections".to_string(),
            dependencies: vec!["indexmap".to_string()],
            version: "2".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("All collection types work in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "defaultdict".to_string(),
                rust_equiv: "HashMap::new".to_string(),
                requires_use: Some("std::collections::HashMap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use .entry().or_insert() pattern for defaults".to_string()),
            },
            FunctionMapping {
                python_name: "Counter".to_string(),
                rust_equiv: "HashMap::new".to_string(),
                requires_use: Some("std::collections::HashMap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("HashMap<T, usize> with counting logic".to_string()),
            },
            FunctionMapping {
                python_name: "deque".to_string(),
                rust_equiv: "VecDeque::new".to_string(),
                requires_use: Some("std::collections::VecDeque".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "OrderedDict".to_string(),
                rust_equiv: "IndexMap::new".to_string(),
                requires_use: Some("indexmap::IndexMap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Requires indexmap crate".to_string()),
            },
        ],
    ));

    // itertools - Iterator tools (already mapped but expanded)
    mappings.push((
        ModuleMapping {
            python_module: "itertools".to_string(),
            rust_crate: Some("itertools".to_string()),
            rust_use: "itertools".to_string(),
            dependencies: vec![],
            version: "0.12".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Pure computation, fully WASM compatible".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "chain".to_string(),
                rust_equiv: "itertools::chain".to_string(),
                requires_use: Some("itertools".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "product".to_string(),
                rust_equiv: "itertools::iproduct".to_string(),
                requires_use: Some("itertools::iproduct".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "permutations".to_string(),
                rust_equiv: "itertools::permutations".to_string(),
                requires_use: Some("itertools::Itertools".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Call .permutations() on iterator".to_string()),
            },
            FunctionMapping {
                python_name: "combinations".to_string(),
                rust_equiv: "itertools::combinations".to_string(),
                requires_use: Some("itertools::Itertools".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // heapq - Heap queue
    mappings.push((
        ModuleMapping {
            python_module: "heapq".to_string(),
            rust_crate: None,
            rust_use: "std::collections::BinaryHeap".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: None,
        },
        vec![
            FunctionMapping {
                python_name: "heappush".to_string(),
                rust_equiv: "heap.push".to_string(),
                requires_use: Some("std::collections::BinaryHeap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("BinaryHeap is max-heap, use Reverse for min-heap".to_string()),
            },
            FunctionMapping {
                python_name: "heappop".to_string(),
                rust_equiv: "heap.pop".to_string(),
                requires_use: Some("std::collections::BinaryHeap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // functools - Functional programming tools
    mappings.push((
        ModuleMapping {
            python_module: "functools".to_string(),
            rust_crate: Some("cached".to_string()),
            rust_use: "std::cmp".to_string(),
            dependencies: vec!["cached".to_string()],
            version: "0.46".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Most functions implementable with Rust patterns".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "reduce".to_string(),
                rust_equiv: "iterator.fold".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use .fold() or .reduce() on iterators".to_string()),
            },
            FunctionMapping {
                python_name: "lru_cache".to_string(),
                rust_equiv: "#[cached]".to_string(),
                requires_use: Some("cached::proc_macro::cached".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use cached crate's macro".to_string()),
            },
            FunctionMapping {
                python_name: "partial".to_string(),
                rust_equiv: "closure_with_captures".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use closures to capture arguments".to_string()),
            },
        ],
    ));

    // ========== Text Processing ==========

    // csv - CSV file handling
    mappings.push((
        ModuleMapping {
            python_module: "csv".to_string(),
            rust_crate: Some("csv".to_string()),
            rust_use: "csv".to_string(),
            dependencies: vec!["serde".to_string()],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Works with strings, file I/O requires WASI".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "reader".to_string(),
                rust_equiv: "csv::Reader::from_reader".to_string(),
                requires_use: Some("csv".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Can read from string slices".to_string()),
            },
            FunctionMapping {
                python_name: "writer".to_string(),
                rust_equiv: "csv::Writer::from_writer".to_string(),
                requires_use: Some("csv".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Can write to Vec<u8>".to_string()),
            },
            FunctionMapping {
                python_name: "DictReader".to_string(),
                rust_equiv: "csv::Reader::from_reader".to_string(),
                requires_use: Some("csv".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use serde to deserialize to structs".to_string()),
            },
        ],
    ));

    // xml.etree.ElementTree - XML parsing
    mappings.push((
        ModuleMapping {
            python_module: "xml.etree.ElementTree".to_string(),
            rust_crate: Some("quick-xml".to_string()),
            rust_use: "quick_xml".to_string(),
            dependencies: vec!["serde".to_string()],
            version: "0.31".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Fully WASM compatible for parsing".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "parse".to_string(),
                rust_equiv: "quick_xml::Reader::from_str".to_string(),
                requires_use: Some("quick_xml::Reader".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "fromstring".to_string(),
                rust_equiv: "quick_xml::Reader::from_str".to_string(),
                requires_use: Some("quick_xml::Reader".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // textwrap - Text wrapping
    mappings.push((
        ModuleMapping {
            python_module: "textwrap".to_string(),
            rust_crate: Some("textwrap".to_string()),
            rust_use: "textwrap".to_string(),
            dependencies: vec![],
            version: "0.16".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Pure text processing".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "wrap".to_string(),
                rust_equiv: "textwrap::wrap".to_string(),
                requires_use: Some("textwrap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "fill".to_string(),
                rust_equiv: "textwrap::fill".to_string(),
                requires_use: Some("textwrap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // ========== Binary Data ==========

    // struct - Binary data packing
    mappings.push((
        ModuleMapping {
            python_module: "struct".to_string(),
            rust_crate: Some("byteorder".to_string()),
            rust_use: "byteorder".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Binary operations fully supported".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "pack".to_string(),
                rust_equiv: "write_endian".to_string(),
                requires_use: Some("byteorder::{WriteBytesExt, LittleEndian}".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use WriteBytesExt trait methods".to_string()),
            },
            FunctionMapping {
                python_name: "unpack".to_string(),
                rust_equiv: "read_endian".to_string(),
                requires_use: Some("byteorder::{ReadBytesExt, LittleEndian}".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use ReadBytesExt trait methods".to_string()),
            },
        ],
    ));

    // base64 - Base64 encoding
    mappings.push((
        ModuleMapping {
            python_module: "base64".to_string(),
            rust_crate: Some("base64".to_string()),
            rust_use: "base64".to_string(),
            dependencies: vec![],
            version: "0.21".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: None,
        },
        vec![
            FunctionMapping {
                python_name: "b64encode".to_string(),
                rust_equiv: "base64::encode".to_string(),
                requires_use: Some("base64".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "b64decode".to_string(),
                rust_equiv: "base64::decode".to_string(),
                requires_use: Some("base64".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // ========== Date & Time ==========

    // time - Time access
    mappings.push((
        ModuleMapping {
            python_module: "time".to_string(),
            rust_crate: None,
            rust_use: "std::time".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Basic time works, some functions need JS interop in browser".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "time".to_string(),
                rust_equiv: "SystemTime::now".to_string(),
                requires_use: Some("std::time::SystemTime".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("In browser, uses JS Date.now()".to_string()),
            },
            FunctionMapping {
                python_name: "sleep".to_string(),
                rust_equiv: "std::thread::sleep".to_string(),
                requires_use: Some("std::time::Duration".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Use async sleep in WASM: wasm_timer::Delay".to_string()),
            },
        ],
    ));

    // ========== Networking (Requires JS interop in browser) ==========

    // urllib.request - URL handling
    mappings.push((
        ModuleMapping {
            python_module: "urllib.request".to_string(),
            rust_crate: Some("reqwest".to_string()),
            rust_use: "reqwest".to_string(),
            dependencies: vec![],
            version: "0.11".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Uses fetch API in browser, native HTTP in Node.js".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "urlopen".to_string(),
                rust_equiv: "reqwest::get".to_string(),
                requires_use: Some("reqwest".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Async function, returns Future".to_string()),
            },
            FunctionMapping {
                python_name: "Request".to_string(),
                rust_equiv: "reqwest::Client::new".to_string(),
                requires_use: Some("reqwest".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: None,
            },
        ],
    ));

    // socket - Low-level networking
    mappings.push((
        ModuleMapping {
            python_module: "socket".to_string(),
            rust_crate: None,
            rust_use: "std::net".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Raw sockets not available in WASM, use WebSocket or fetch instead".to_string()),
        },
        vec![],
    ));

    // ========== Compression & Archives ==========

    // gzip - Gzip compression
    mappings.push((
        ModuleMapping {
            python_module: "gzip".to_string(),
            rust_crate: Some("flate2".to_string()),
            rust_use: "flate2".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Pure compression, works in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "compress".to_string(),
                rust_equiv: "flate2::write::GzEncoder::new".to_string(),
                requires_use: Some("flate2::write::GzEncoder".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "decompress".to_string(),
                rust_equiv: "flate2::read::GzDecoder::new".to_string(),
                requires_use: Some("flate2::read::GzDecoder".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // zipfile - ZIP archives
    mappings.push((
        ModuleMapping {
            python_module: "zipfile".to_string(),
            rust_crate: Some("zip".to_string()),
            rust_use: "zip".to_string(),
            dependencies: vec![],
            version: "0.6".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Works with in-memory data, file I/O needs WASI".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "ZipFile".to_string(),
                rust_equiv: "zip::ZipArchive::new".to_string(),
                requires_use: Some("zip::ZipArchive".to_string()),
                wasm_compatible: WasmCompatibility::Partial,
                transform_notes: Some("Can work with Cursor<Vec<u8>> for in-memory".to_string()),
            },
        ],
    ));

    // ========== Cryptography & Hashing ==========

    // hashlib - Secure hashes
    mappings.push((
        ModuleMapping {
            python_module: "hashlib".to_string(),
            rust_crate: Some("sha2".to_string()),
            rust_use: "sha2".to_string(),
            dependencies: vec!["md5".to_string()],
            version: "0.10".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("All hash algorithms work in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "sha256".to_string(),
                rust_equiv: "Sha256::new".to_string(),
                requires_use: Some("sha2::{Sha256, Digest}".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "md5".to_string(),
                rust_equiv: "Md5::new".to_string(),
                requires_use: Some("md5::Md5".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // secrets - Cryptographically strong random numbers
    mappings.push((
        ModuleMapping {
            python_module: "secrets".to_string(),
            rust_crate: Some("rand".to_string()),
            rust_use: "rand".to_string(),
            dependencies: vec!["getrandom".to_string()],
            version: "0.8".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Uses crypto.getRandomValues() in browser".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "token_bytes".to_string(),
                rust_equiv: "rand::random::<[u8; N]>".to_string(),
                requires_use: Some("rand".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Requires getrandom with js feature".to_string()),
            },
            FunctionMapping {
                python_name: "token_hex".to_string(),
                rust_equiv: "hex::encode(rand::random::<[u8; N]>())".to_string(),
                requires_use: Some("rand".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: None,
            },
        ],
    ));

    // ========== Email & Internet Protocols ==========

    // email.message - Email message handling
    mappings.push((
        ModuleMapping {
            python_module: "email.message".to_string(),
            rust_crate: Some("lettre".to_string()),
            rust_use: "lettre::message".to_string(),
            dependencies: vec![],
            version: "0.11".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Message parsing/creation works, sending needs JS interop in browser".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "EmailMessage".to_string(),
                rust_equiv: "lettre::Message::builder".to_string(),
                requires_use: Some("lettre::Message".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use builder pattern for message construction".to_string()),
            },
        ],
    ));

    // smtplib - SMTP client
    mappings.push((
        ModuleMapping {
            python_module: "smtplib".to_string(),
            rust_crate: Some("lettre".to_string()),
            rust_use: "lettre".to_string(),
            dependencies: vec![],
            version: "0.11".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Requires network access - JS interop in browser, WASI sockets in server".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "SMTP".to_string(),
                rust_equiv: "lettre::SmtpTransport::relay".to_string(),
                requires_use: Some("lettre::SmtpTransport".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Use SmtpTransport builder".to_string()),
            },
        ],
    ));

    // http.client - HTTP client
    mappings.push((
        ModuleMapping {
            python_module: "http.client".to_string(),
            rust_crate: Some("reqwest".to_string()),
            rust_use: "reqwest".to_string(),
            dependencies: vec![],
            version: "0.11".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Uses fetch API in browser, native HTTP in server WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "HTTPConnection".to_string(),
                rust_equiv: "reqwest::Client::new".to_string(),
                requires_use: Some("reqwest".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Async client, returns Future".to_string()),
            },
            FunctionMapping {
                python_name: "HTTPSConnection".to_string(),
                rust_equiv: "reqwest::Client::builder().https_only(true).build".to_string(),
                requires_use: Some("reqwest".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: None,
            },
        ],
    ));

    // ========== Testing & Logging ==========

    // unittest - Unit testing
    mappings.push((
        ModuleMapping {
            python_module: "unittest".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Maps to Rust's built-in test framework".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "TestCase".to_string(),
                rust_equiv: "#[cfg(test)]".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use #[test] attribute on functions".to_string()),
            },
            FunctionMapping {
                python_name: "assertEqual".to_string(),
                rust_equiv: "assert_eq!".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "assertTrue".to_string(),
                rust_equiv: "assert!".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "assertFalse".to_string(),
                rust_equiv: "assert!".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use assert!(!condition)".to_string()),
            },
        ],
    ));

    // logging - Logging facility
    mappings.push((
        ModuleMapping {
            python_module: "logging".to_string(),
            rust_crate: Some("tracing".to_string()),
            rust_use: "tracing".to_string(),
            dependencies: vec!["tracing-subscriber".to_string()],
            version: "0.1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("tracing works in WASM with wasm-compatible subscriber".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "debug".to_string(),
                rust_equiv: "tracing::debug!".to_string(),
                requires_use: Some("tracing::debug".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "info".to_string(),
                rust_equiv: "tracing::info!".to_string(),
                requires_use: Some("tracing::info".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "warning".to_string(),
                rust_equiv: "tracing::warn!".to_string(),
                requires_use: Some("tracing::warn".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "error".to_string(),
                rust_equiv: "tracing::error!".to_string(),
                requires_use: Some("tracing::error".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // ========== Command Line & Configuration ==========

    // argparse - Command-line parsing
    mappings.push((
        ModuleMapping {
            python_module: "argparse".to_string(),
            rust_crate: Some("clap".to_string()),
            rust_use: "clap".to_string(),
            dependencies: vec![],
            version: "4".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Parsing works, but CLI args not available in browser WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "ArgumentParser".to_string(),
                rust_equiv: "clap::Command::new".to_string(),
                requires_use: Some("clap::Command".to_string()),
                wasm_compatible: WasmCompatibility::Partial,
                transform_notes: Some("Use clap's builder or derive API".to_string()),
            },
            FunctionMapping {
                python_name: "add_argument".to_string(),
                rust_equiv: ".arg(clap::Arg::new)".to_string(),
                requires_use: Some("clap::Arg".to_string()),
                wasm_compatible: WasmCompatibility::Partial,
                transform_notes: None,
            },
        ],
    ));

    // configparser - Configuration file parser
    mappings.push((
        ModuleMapping {
            python_module: "configparser".to_string(),
            rust_crate: Some("ini".to_string()),
            rust_use: "ini".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Pure parsing, works everywhere".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "ConfigParser".to_string(),
                rust_equiv: "ini::Ini::new".to_string(),
                requires_use: Some("ini::Ini".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // ========== System & Process ==========

    // subprocess - Subprocess management
    mappings.push((
        ModuleMapping {
            python_module: "subprocess".to_string(),
            rust_crate: None,
            rust_use: "std::process".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Process spawning not available in WASM - use JS interop for browser, WASI for server (limited)".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "run".to_string(),
                rust_equiv: "std::process::Command::new".to_string(),
                requires_use: Some("std::process::Command".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Only works in native Rust, not WASM".to_string()),
            },
        ],
    ));

    // signal - Signal handling
    mappings.push((
        ModuleMapping {
            python_module: "signal".to_string(),
            rust_crate: Some("signal-hook".to_string()),
            rust_use: "signal_hook".to_string(),
            dependencies: vec![],
            version: "0.3".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Signals not available in WASM".to_string()),
        },
        vec![],
    ));

    // ========== Concurrency ==========

    // threading - Thread-based parallelism
    mappings.push((
        ModuleMapping {
            python_module: "threading".to_string(),
            rust_crate: None,
            rust_use: "std::thread".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Use Web Workers in browser, WASM threads experimental".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "Thread".to_string(),
                rust_equiv: "std::thread::spawn".to_string(),
                requires_use: Some("std::thread".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Use wasm-bindgen-rayon or Web Workers".to_string()),
            },
        ],
    ));

    // asyncio - Asynchronous I/O (enhanced)
    mappings.push((
        ModuleMapping {
            python_module: "asyncio".to_string(),
            rust_crate: Some("tokio".to_string()),
            rust_use: "tokio".to_string(),
            dependencies: vec!["wasm-bindgen-futures".to_string()],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Use tokio with wasm-bindgen-futures in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "run".to_string(),
                rust_equiv: "tokio::runtime::Runtime::new().unwrap().block_on".to_string(),
                requires_use: Some("tokio::runtime::Runtime".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Use wasm_bindgen_futures::spawn_local in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "create_task".to_string(),
                rust_equiv: "tokio::spawn".to_string(),
                requires_use: Some("tokio".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "sleep".to_string(),
                rust_equiv: "tokio::time::sleep".to_string(),
                requires_use: Some("tokio::time".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Use wasm_timer::Delay in WASM".to_string()),
            },
        ],
    ));

    // queue - Synchronized queue
    mappings.push((
        ModuleMapping {
            python_module: "queue".to_string(),
            rust_crate: Some("crossbeam-channel".to_string()),
            rust_use: "crossbeam_channel".to_string(),
            dependencies: vec![],
            version: "0.5".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Works in single-threaded WASM, limited in multi-threaded".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "Queue".to_string(),
                rust_equiv: "crossbeam_channel::unbounded".to_string(),
                requires_use: Some("crossbeam_channel".to_string()),
                wasm_compatible: WasmCompatibility::Partial,
                transform_notes: Some("Returns (Sender, Receiver) tuple".to_string()),
            },
        ],
    ));

    // ========== Data Serialization ==========

    // pickle - Python object serialization
    mappings.push((
        ModuleMapping {
            python_module: "pickle".to_string(),
            rust_crate: Some("serde_pickle".to_string()),
            rust_use: "serde_pickle".to_string(),
            dependencies: vec!["serde".to_string()],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Limited Python pickle compatibility, use serde for Rust types".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "dumps".to_string(),
                rust_equiv: "serde_pickle::to_vec".to_string(),
                requires_use: Some("serde_pickle".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Requires Serialize trait".to_string()),
            },
            FunctionMapping {
                python_name: "loads".to_string(),
                rust_equiv: "serde_pickle::from_slice".to_string(),
                requires_use: Some("serde_pickle".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Requires Deserialize trait".to_string()),
            },
        ],
    ));

    // ========== String & Data Processing ==========

    // difflib - Helpers for computing deltas
    mappings.push((
        ModuleMapping {
            python_module: "difflib".to_string(),
            rust_crate: Some("similar".to_string()),
            rust_use: "similar".to_string(),
            dependencies: vec![],
            version: "2".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Text diffing, fully WASM compatible".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "unified_diff".to_string(),
                rust_equiv: "similar::TextDiff::from_lines".to_string(),
                requires_use: Some("similar::TextDiff".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use .unified_diff() method".to_string()),
            },
        ],
    ));

    // shlex - Simple lexical analysis
    mappings.push((
        ModuleMapping {
            python_module: "shlex".to_string(),
            rust_crate: Some("shlex".to_string()),
            rust_use: "shlex".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Shell-like syntax parsing".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "split".to_string(),
                rust_equiv: "shlex::split".to_string(),
                requires_use: Some("shlex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // fnmatch - Unix filename pattern matching
    mappings.push((
        ModuleMapping {
            python_module: "fnmatch".to_string(),
            rust_crate: Some("globset".to_string()),
            rust_use: "globset".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Pattern matching works everywhere".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "fnmatch".to_string(),
                rust_equiv: "globset::Glob::new().unwrap().compile_matcher().is_match".to_string(),
                requires_use: Some("globset::Glob".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Build glob matcher first".to_string()),
            },
        ],
    ));

    // ========== Advanced Data Types ==========

    // dataclasses - Data classes
    mappings.push((
        ModuleMapping {
            python_module: "dataclasses".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Maps to Rust structs with derive macros".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "dataclass".to_string(),
                rust_equiv: "#[derive(Debug, Clone)]".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use struct with derive attributes".to_string()),
            },
        ],
    ));

    // enum - Enumerations
    mappings.push((
        ModuleMapping {
            python_module: "enum".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Maps to Rust enum types".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "Enum".to_string(),
                rust_equiv: "enum".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use Rust enum with variants".to_string()),
            },
        ],
    ));

    // typing - Type hints
    mappings.push((
        ModuleMapping {
            python_module: "typing".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Type hints map to Rust type system".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "List".to_string(),
                rust_equiv: "Vec".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "Dict".to_string(),
                rust_equiv: "HashMap".to_string(),
                requires_use: Some("std::collections::HashMap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "Optional".to_string(),
                rust_equiv: "Option".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // ========== Utilities ==========

    // uuid - UUID objects
    mappings.push((
        ModuleMapping {
            python_module: "uuid".to_string(),
            rust_crate: Some("uuid".to_string()),
            rust_use: "uuid".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("UUID v4 needs random source - use getrandom with js feature".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "uuid4".to_string(),
                rust_equiv: "uuid::Uuid::new_v4".to_string(),
                requires_use: Some("uuid::Uuid".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Requires getrandom with wasm feature".to_string()),
            },
        ],
    ));

    // copy - Shallow and deep copy
    mappings.push((
        ModuleMapping {
            python_module: "copy".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use .clone() for copying in Rust".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "copy".to_string(),
                rust_equiv: ".clone()".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Shallow copy via Clone trait".to_string()),
            },
            FunctionMapping {
                python_name: "deepcopy".to_string(),
                rust_equiv: ".clone()".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Clone is deep by default in Rust".to_string()),
            },
        ],
    ));

    // datetime - Date and time handling
    mappings.push((
        ModuleMapping {
            python_module: "datetime".to_string(),
            rust_crate: Some("chrono".to_string()),
            rust_use: "chrono".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Use chrono with wasm-bindgen for browser, js-sys for current time".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "datetime.now".to_string(),
                rust_equiv: "chrono::Local::now".to_string(),
                requires_use: Some("chrono::Local".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("In WASM, use js_sys::Date::now()".to_string()),
            },
            FunctionMapping {
                python_name: "datetime.utcnow".to_string(),
                rust_equiv: "chrono::Utc::now".to_string(),
                requires_use: Some("chrono::Utc".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("In WASM, use js_sys::Date::now()".to_string()),
            },
            FunctionMapping {
                python_name: "date.today".to_string(),
                rust_equiv: "chrono::Local::now().date_naive".to_string(),
                requires_use: Some("chrono::Local".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "timedelta".to_string(),
                rust_equiv: "chrono::Duration".to_string(),
                requires_use: Some("chrono::Duration".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Pure computation, fully WASM compatible".to_string()),
            },
        ],
    ));

    // decimal - Decimal fixed point arithmetic
    mappings.push((
        ModuleMapping {
            python_module: "decimal".to_string(),
            rust_crate: Some("rust_decimal".to_string()),
            rust_use: "rust_decimal".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Exact decimal arithmetic, fully WASM compatible".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "Decimal".to_string(),
                rust_equiv: "rust_decimal::Decimal::from_str".to_string(),
                requires_use: Some("rust_decimal::Decimal".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use Decimal::from_str() or Decimal::new()".to_string()),
            },
            FunctionMapping {
                python_name: "getcontext".to_string(),
                rust_equiv: "rust_decimal::RoundingStrategy".to_string(),
                requires_use: Some("rust_decimal::RoundingStrategy".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Rounding is set per operation in Rust".to_string()),
            },
        ],
    ));

    // fractions - Rational numbers
    mappings.push((
        ModuleMapping {
            python_module: "fractions".to_string(),
            rust_crate: Some("num-rational".to_string()),
            rust_use: "num_rational".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Arbitrary precision rational numbers, fully WASM compatible".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "Fraction".to_string(),
                rust_equiv: "num_rational::Ratio::new".to_string(),
                requires_use: Some("num_rational::Ratio".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Ratio::new(numerator, denominator)".to_string()),
            },
            FunctionMapping {
                python_name: "gcd".to_string(),
                rust_equiv: "num_integer::gcd".to_string(),
                requires_use: Some("num_integer::gcd".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use num-integer crate for GCD".to_string()),
            },
        ],
    ));

    mappings
}

/// Get statistics about stdlib coverage
pub fn get_coverage_stats() -> StdlibCoverageStats {
    let critical_mappings = init_critical_mappings();

    StdlibCoverageStats {
        total_python_stdlib_modules: 278,
        mapped_modules: critical_mappings.len(),
        full_wasm_compat: critical_mappings.iter()
            .filter(|(m, _)| m.wasm_compatible == WasmCompatibility::Full)
            .count(),
        partial_wasm_compat: critical_mappings.iter()
            .filter(|(m, _)| m.wasm_compatible == WasmCompatibility::Partial)
            .count(),
        requires_wasi: critical_mappings.iter()
            .filter(|(m, _)| m.wasm_compatible == WasmCompatibility::RequiresWasi)
            .count(),
        requires_js_interop: critical_mappings.iter()
            .filter(|(m, _)| m.wasm_compatible == WasmCompatibility::RequiresJsInterop)
            .count(),
        incompatible: critical_mappings.iter()
            .filter(|(m, _)| m.wasm_compatible == WasmCompatibility::Incompatible)
            .count(),
    }
}

#[derive(Debug)]
pub struct StdlibCoverageStats {
    pub total_python_stdlib_modules: usize,
    pub mapped_modules: usize,
    pub full_wasm_compat: usize,
    pub partial_wasm_compat: usize,
    pub requires_wasi: usize,
    pub requires_js_interop: usize,
    pub incompatible: usize,
}
