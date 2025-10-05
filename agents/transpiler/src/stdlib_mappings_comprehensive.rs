

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
            FunctionMapping {
                python_name: "joinpath".to_string(),
                rust_equiv: "path.join".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "parent".to_string(),
                rust_equiv: "path.parent()".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns Option<&Path>".to_string()),
            },
            FunctionMapping {
                python_name: "name".to_string(),
                rust_equiv: "path.file_name()".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns Option<&OsStr>".to_string()),
            },
            FunctionMapping {
                python_name: "suffix".to_string(),
                rust_equiv: "path.extension()".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns Option<&OsStr>".to_string()),
            },
            FunctionMapping {
                python_name: "stem".to_string(),
                rust_equiv: "path.file_stem()".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns Option<&OsStr>".to_string()),
            },
            FunctionMapping {
                python_name: "is_absolute".to_string(),
                rust_equiv: "path.is_absolute()".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "is_relative".to_string(),
                rust_equiv: "path.is_relative()".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "glob".to_string(),
                rust_equiv: "glob::glob".to_string(),
                requires_use: Some("glob".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Requires glob crate, returns iterator".to_string()),
            },
            FunctionMapping {
                python_name: "rglob".to_string(),
                rust_equiv: "glob::glob with **".to_string(),
                requires_use: Some("glob".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Recursive glob pattern".to_string()),
            },
            FunctionMapping {
                python_name: "mkdir".to_string(),
                rust_equiv: "std::fs::create_dir".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "rmdir".to_string(),
                rust_equiv: "std::fs::remove_dir".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "unlink".to_string(),
                rust_equiv: "std::fs::remove_file".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "rename".to_string(),
                rust_equiv: "std::fs::rename".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "touch".to_string(),
                rust_equiv: "std::fs::OpenOptions::new().create(true).write(true).open".to_string(),
                requires_use: Some("std::fs::OpenOptions".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Create or update file timestamp".to_string()),
            },
            FunctionMapping {
                python_name: "read_text".to_string(),
                rust_equiv: "std::fs::read_to_string".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "read_bytes".to_string(),
                rust_equiv: "std::fs::read".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "write_text".to_string(),
                rust_equiv: "std::fs::write".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "write_bytes".to_string(),
                rust_equiv: "std::fs::write".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "resolve".to_string(),
                rust_equiv: "std::fs::canonicalize".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns absolute path, resolves symlinks".to_string()),
            },
            FunctionMapping {
                python_name: "stat".to_string(),
                rust_equiv: "std::fs::metadata".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns Metadata struct".to_string()),
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
            FunctionMapping {
                python_name: "namedtuple".to_string(),
                rust_equiv: "struct with #[derive()]".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Define a struct with named fields".to_string()),
            },
            FunctionMapping {
                python_name: "ChainMap".to_string(),
                rust_equiv: "multiple HashMap with fallback".to_string(),
                requires_use: Some("std::collections::HashMap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Implement lookup chain manually".to_string()),
            },
            FunctionMapping {
                python_name: "deque.append".to_string(),
                rust_equiv: "VecDeque::push_back".to_string(),
                requires_use: Some("std::collections::VecDeque".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "deque.appendleft".to_string(),
                rust_equiv: "VecDeque::push_front".to_string(),
                requires_use: Some("std::collections::VecDeque".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "deque.pop".to_string(),
                rust_equiv: "VecDeque::pop_back".to_string(),
                requires_use: Some("std::collections::VecDeque".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns Option<T>".to_string()),
            },
            FunctionMapping {
                python_name: "deque.popleft".to_string(),
                rust_equiv: "VecDeque::pop_front".to_string(),
                requires_use: Some("std::collections::VecDeque".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns Option<T>".to_string()),
            },
            FunctionMapping {
                python_name: "deque.rotate".to_string(),
                rust_equiv: "VecDeque::rotate_left/rotate_right".to_string(),
                requires_use: Some("std::collections::VecDeque".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use rotate_left for positive, rotate_right for negative".to_string()),
            },
            FunctionMapping {
                python_name: "Counter.most_common".to_string(),
                rust_equiv: "BTreeMap or sort by value".to_string(),
                requires_use: Some("std::collections::BTreeMap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Sort HashMap by value descending".to_string()),
            },
            FunctionMapping {
                python_name: "Counter.elements".to_string(),
                rust_equiv: "flat_map over counts".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Expand HashMap<T, usize> to iterator".to_string()),
            },
            FunctionMapping {
                python_name: "Counter.update".to_string(),
                rust_equiv: "*map.entry(k).or_insert(0) += 1".to_string(),
                requires_use: Some("std::collections::HashMap".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use entry API to update counts".to_string()),
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
            FunctionMapping {
                python_name: "wraps".to_string(),
                rust_equiv: "#[inline] or manual wrapper".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Preserve function metadata manually".to_string()),
            },
            FunctionMapping {
                python_name: "total_ordering".to_string(),
                rust_equiv: "#[derive(Ord, PartialOrd)]".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Derive comparison traits".to_string()),
            },
            FunctionMapping {
                python_name: "cmp_to_key".to_string(),
                rust_equiv: "Ord trait implementation".to_string(),
                requires_use: Some("std::cmp::Ordering".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Implement Ord trait".to_string()),
            },
            FunctionMapping {
                python_name: "cache".to_string(),
                rust_equiv: "#[cached]".to_string(),
                requires_use: Some("cached::proc_macro::cached".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use cached crate's macro".to_string()),
            },
            FunctionMapping {
                python_name: "singledispatch".to_string(),
                rust_equiv: "trait with impl per type".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use trait dispatch".to_string()),
            },
            FunctionMapping {
                python_name: "update_wrapper".to_string(),
                rust_equiv: "manual wrapper implementation".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Copy metadata manually".to_string()),
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
        vec![
            FunctionMapping {
                python_name: "socket".to_string(),
                rust_equiv: "TcpStream or UdpSocket".to_string(),
                requires_use: Some("std::net::{TcpStream, UdpSocket}".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Native only - use WebSocket in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "connect".to_string(),
                rust_equiv: "TcpStream::connect".to_string(),
                requires_use: Some("std::net::TcpStream".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Native only".to_string()),
            },
            FunctionMapping {
                python_name: "bind".to_string(),
                rust_equiv: "TcpListener::bind".to_string(),
                requires_use: Some("std::net::TcpListener".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Server sockets not available in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "listen".to_string(),
                rust_equiv: "TcpListener::accept in loop".to_string(),
                requires_use: Some("std::net::TcpListener".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Server sockets not available in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "accept".to_string(),
                rust_equiv: "TcpListener::accept".to_string(),
                requires_use: Some("std::net::TcpListener".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Returns (TcpStream, SocketAddr)".to_string()),
            },
            FunctionMapping {
                python_name: "send".to_string(),
                rust_equiv: "TcpStream::write".to_string(),
                requires_use: Some("std::io::Write".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Use Write trait".to_string()),
            },
            FunctionMapping {
                python_name: "recv".to_string(),
                rust_equiv: "TcpStream::read".to_string(),
                requires_use: Some("std::io::Read".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Use Read trait".to_string()),
            },
            FunctionMapping {
                python_name: "sendto".to_string(),
                rust_equiv: "UdpSocket::send_to".to_string(),
                requires_use: Some("std::net::UdpSocket".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("UDP sockets not available in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "recvfrom".to_string(),
                rust_equiv: "UdpSocket::recv_from".to_string(),
                requires_use: Some("std::net::UdpSocket".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Returns (usize, SocketAddr)".to_string()),
            },
            FunctionMapping {
                python_name: "close".to_string(),
                rust_equiv: "drop(socket)".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Automatic on scope exit".to_string()),
            },
            FunctionMapping {
                python_name: "gethostname".to_string(),
                rust_equiv: "hostname::get".to_string(),
                requires_use: Some("hostname".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Requires hostname crate".to_string()),
            },
            FunctionMapping {
                python_name: "gethostbyname".to_string(),
                rust_equiv: "ToSocketAddrs trait".to_string(),
                requires_use: Some("std::net::ToSocketAddrs".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("DNS resolution not available in WASM".to_string()),
            },
        ],
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
            FunctionMapping {
                python_name: "critical".to_string(),
                rust_equiv: "tracing::error!".to_string(),
                requires_use: Some("tracing::error".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use error! for critical messages".to_string()),
            },
            FunctionMapping {
                python_name: "getLogger".to_string(),
                rust_equiv: "tracing::Span::current".to_string(),
                requires_use: Some("tracing::Span".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use spans for structured logging".to_string()),
            },
            FunctionMapping {
                python_name: "basicConfig".to_string(),
                rust_equiv: "tracing_subscriber::fmt::init".to_string(),
                requires_use: Some("tracing_subscriber::fmt".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Initialize subscriber".to_string()),
            },
            FunctionMapping {
                python_name: "FileHandler".to_string(),
                rust_equiv: "tracing_appender::rolling".to_string(),
                requires_use: Some("tracing_appender".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Use tracing-appender for file logging".to_string()),
            },
            FunctionMapping {
                python_name: "StreamHandler".to_string(),
                rust_equiv: "tracing_subscriber::fmt::Layer".to_string(),
                requires_use: Some("tracing_subscriber::fmt".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Default output layer".to_string()),
            },
            FunctionMapping {
                python_name: "Formatter".to_string(),
                rust_equiv: "tracing_subscriber::fmt::format".to_string(),
                requires_use: Some("tracing_subscriber::fmt".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Custom format configuration".to_string()),
            },
            FunctionMapping {
                python_name: "setLevel".to_string(),
                rust_equiv: "EnvFilter::new".to_string(),
                requires_use: Some("tracing_subscriber::EnvFilter".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use environment filter".to_string()),
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
                rust_equiv: "Command::new().output()".to_string(),
                requires_use: Some("std::process::Command".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Only works in native Rust, not WASM".to_string()),
            },
            FunctionMapping {
                python_name: "Popen".to_string(),
                rust_equiv: "Command::new().spawn()".to_string(),
                requires_use: Some("std::process::Command".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Returns Child process handle".to_string()),
            },
            FunctionMapping {
                python_name: "call".to_string(),
                rust_equiv: "Command::new().status()".to_string(),
                requires_use: Some("std::process::Command".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Returns exit status only".to_string()),
            },
            FunctionMapping {
                python_name: "check_call".to_string(),
                rust_equiv: "Command::new().status()?.success()".to_string(),
                requires_use: Some("std::process::Command".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Check if exit status is success".to_string()),
            },
            FunctionMapping {
                python_name: "check_output".to_string(),
                rust_equiv: "Command::new().output()".to_string(),
                requires_use: Some("std::process::Command".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Returns Output with stdout/stderr".to_string()),
            },
            FunctionMapping {
                python_name: "communicate".to_string(),
                rust_equiv: "child.wait_with_output()".to_string(),
                requires_use: Some("std::process::Child".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Wait and get output".to_string()),
            },
            FunctionMapping {
                python_name: "wait".to_string(),
                rust_equiv: "child.wait()".to_string(),
                requires_use: Some("std::process::Child".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Returns ExitStatus".to_string()),
            },
            FunctionMapping {
                python_name: "poll".to_string(),
                rust_equiv: "child.try_wait()".to_string(),
                requires_use: Some("std::process::Child".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Non-blocking status check".to_string()),
            },
            FunctionMapping {
                python_name: "kill".to_string(),
                rust_equiv: "child.kill()".to_string(),
                requires_use: Some("std::process::Child".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Force kill process".to_string()),
            },
            FunctionMapping {
                python_name: "terminate".to_string(),
                rust_equiv: "child.kill()".to_string(),
                requires_use: Some("std::process::Child".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Same as kill() in Rust".to_string()),
            },
            FunctionMapping {
                python_name: "PIPE".to_string(),
                rust_equiv: "Stdio::piped()".to_string(),
                requires_use: Some("std::process::Stdio".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Create pipe for stdin/stdout/stderr".to_string()),
            },
            FunctionMapping {
                python_name: "DEVNULL".to_string(),
                rust_equiv: "Stdio::null()".to_string(),
                requires_use: Some("std::process::Stdio".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Discard output".to_string()),
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
            FunctionMapping {
                python_name: "Tuple".to_string(),
                rust_equiv: "(T1, T2, ...)".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use Rust tuple types".to_string()),
            },
            FunctionMapping {
                python_name: "Set".to_string(),
                rust_equiv: "HashSet".to_string(),
                requires_use: Some("std::collections::HashSet".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "Union".to_string(),
                rust_equiv: "enum or trait object".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use enum for closed unions, trait objects for open".to_string()),
            },
            FunctionMapping {
                python_name: "Any".to_string(),
                rust_equiv: "Box<dyn std::any::Any>".to_string(),
                requires_use: Some("std::any::Any".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Dynamic typing via Any trait".to_string()),
            },
            FunctionMapping {
                python_name: "Callable".to_string(),
                rust_equiv: "Fn/FnMut/FnOnce trait".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use appropriate closure trait".to_string()),
            },
            FunctionMapping {
                python_name: "TypeVar".to_string(),
                rust_equiv: "generic parameter <T>".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use Rust generics".to_string()),
            },
            FunctionMapping {
                python_name: "Generic".to_string(),
                rust_equiv: "struct/enum with generics".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Define generic types directly".to_string()),
            },
            FunctionMapping {
                python_name: "Protocol".to_string(),
                rust_equiv: "trait definition".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use Rust traits for structural typing".to_string()),
            },
            FunctionMapping {
                python_name: "Literal".to_string(),
                rust_equiv: "const generic or enum".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use const generics or enums for literal types".to_string()),
            },
            FunctionMapping {
                python_name: "Final".to_string(),
                rust_equiv: "const or private field".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use const or make field private".to_string()),
            },
            FunctionMapping {
                python_name: "ClassVar".to_string(),
                rust_equiv: "const or static".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Class variables as const or static".to_string()),
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
            FunctionMapping {
                python_name: "datetime".to_string(),
                rust_equiv: "chrono::NaiveDateTime::new".to_string(),
                requires_use: Some("chrono::NaiveDateTime".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Construct from date and time".to_string()),
            },
            FunctionMapping {
                python_name: "strftime".to_string(),
                rust_equiv: "datetime.format()".to_string(),
                requires_use: Some("chrono".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Format datetime to string".to_string()),
            },
            FunctionMapping {
                python_name: "strptime".to_string(),
                rust_equiv: "NaiveDateTime::parse_from_str".to_string(),
                requires_use: Some("chrono::NaiveDateTime".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Parse string to datetime".to_string()),
            },
            FunctionMapping {
                python_name: "fromtimestamp".to_string(),
                rust_equiv: "DateTime::from_timestamp".to_string(),
                requires_use: Some("chrono::DateTime".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Create from Unix timestamp".to_string()),
            },
            FunctionMapping {
                python_name: "isoformat".to_string(),
                rust_equiv: "datetime.to_rfc3339()".to_string(),
                requires_use: Some("chrono".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Convert to ISO 8601 string".to_string()),
            },
            FunctionMapping {
                python_name: "replace".to_string(),
                rust_equiv: "datetime.with_*()".to_string(),
                requires_use: Some("chrono".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use with_year, with_month, etc.".to_string()),
            },
            FunctionMapping {
                python_name: "weekday".to_string(),
                rust_equiv: "datetime.weekday().num_days_from_monday()".to_string(),
                requires_use: Some("chrono::Datelike".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns 0-6 for Monday-Sunday".to_string()),
            },
            FunctionMapping {
                python_name: "timestamp".to_string(),
                rust_equiv: "datetime.timestamp()".to_string(),
                requires_use: Some("chrono".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Convert to Unix timestamp".to_string()),
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

    // ========== Additional High-Priority Modules (50+ new mappings) ==========

    // abc - Abstract base classes
    mappings.push((
        ModuleMapping {
            python_module: "abc".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Maps to Rust trait system".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "ABC".to_string(),
                rust_equiv: "trait".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Define trait with abstract methods".to_string()),
            },
            FunctionMapping {
                python_name: "abstractmethod".to_string(),
                rust_equiv: "trait_method".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Define method without default implementation in trait".to_string()),
            },
        ],
    ));

    // contextlib - Context managers
    mappings.push((
        ModuleMapping {
            python_module: "contextlib".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Maps to RAII pattern with Drop trait".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "contextmanager".to_string(),
                rust_equiv: "Drop::drop".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use RAII pattern - cleanup in Drop implementation".to_string()),
            },
            FunctionMapping {
                python_name: "closing".to_string(),
                rust_equiv: "Drop::drop".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Automatic via Drop trait".to_string()),
            },
        ],
    ));

    // concurrent.futures - Concurrent execution
    mappings.push((
        ModuleMapping {
            python_module: "concurrent.futures".to_string(),
            rust_crate: Some("rayon".to_string()),
            rust_use: "rayon".to_string(),
            dependencies: vec!["tokio".to_string()],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Use rayon for thread pools, tokio for async futures".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "ThreadPoolExecutor".to_string(),
                rust_equiv: "rayon::ThreadPool::new".to_string(),
                requires_use: Some("rayon::ThreadPoolBuilder".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Use wasm-bindgen-rayon in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "ProcessPoolExecutor".to_string(),
                rust_equiv: "rayon::ThreadPool::new".to_string(),
                requires_use: Some("rayon::ThreadPoolBuilder".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("No process support in WASM, use threads".to_string()),
            },
            FunctionMapping {
                python_name: "Future".to_string(),
                rust_equiv: "tokio::task::JoinHandle".to_string(),
                requires_use: Some("tokio::task".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Use tokio futures".to_string()),
            },
        ],
    ));

    // multiprocessing - Process-based parallelism
    mappings.push((
        ModuleMapping {
            python_module: "multiprocessing".to_string(),
            rust_crate: Some("rayon".to_string()),
            rust_use: "rayon".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Use rayon for thread-based parallelism, no process support in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "Pool".to_string(),
                rust_equiv: "rayon::ThreadPool::new".to_string(),
                requires_use: Some("rayon::ThreadPoolBuilder".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Thread-based only, not multiprocess".to_string()),
            },
            FunctionMapping {
                python_name: "Process".to_string(),
                rust_equiv: "std::thread::spawn".to_string(),
                requires_use: Some("std::thread".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Use threads instead of processes".to_string()),
            },
        ],
    ));

    // random - Random number generation (comprehensive)
    mappings.push((
        ModuleMapping {
            python_module: "random".to_string(),
            rust_crate: Some("rand".to_string()),
            rust_use: "rand".to_string(),
            dependencies: vec!["getrandom".to_string()],
            version: "0.8".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Requires getrandom with js feature in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "random".to_string(),
                rust_equiv: "rand::random::<f64>".to_string(),
                requires_use: Some("rand".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "randint".to_string(),
                rust_equiv: "rand::thread_rng().gen_range".to_string(),
                requires_use: Some("rand::Rng".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("gen_range(start..=end)".to_string()),
            },
            FunctionMapping {
                python_name: "choice".to_string(),
                rust_equiv: "rand::seq::SliceRandom::choose".to_string(),
                requires_use: Some("rand::seq::SliceRandom".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Call .choose(&mut rng) on slice".to_string()),
            },
            FunctionMapping {
                python_name: "shuffle".to_string(),
                rust_equiv: "rand::seq::SliceRandom::shuffle".to_string(),
                requires_use: Some("rand::seq::SliceRandom".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Call .shuffle(&mut rng) on slice".to_string()),
            },
            FunctionMapping {
                python_name: "uniform".to_string(),
                rust_equiv: "rand::thread_rng().gen_range".to_string(),
                requires_use: Some("rand::Rng".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("gen_range(a..b) for float range".to_string()),
            },
            FunctionMapping {
                python_name: "sample".to_string(),
                rust_equiv: "rand::seq::index::sample".to_string(),
                requires_use: Some("rand::seq::index".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Sample k items from population".to_string()),
            },
            FunctionMapping {
                python_name: "choices".to_string(),
                rust_equiv: "rand::seq::SliceRandom::choose_multiple".to_string(),
                requires_use: Some("rand::seq::SliceRandom".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Choose with replacement".to_string()),
            },
            FunctionMapping {
                python_name: "seed".to_string(),
                rust_equiv: "SeedableRng::seed_from_u64".to_string(),
                requires_use: Some("rand::SeedableRng".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Use StdRng::seed_from_u64".to_string()),
            },
            FunctionMapping {
                python_name: "getstate".to_string(),
                rust_equiv: "RNG state is opaque".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Cannot get RNG state in Rust".to_string()),
            },
            FunctionMapping {
                python_name: "setstate".to_string(),
                rust_equiv: "Create new RNG with seed".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Use SeedableRng instead".to_string()),
            },
            FunctionMapping {
                python_name: "randbytes".to_string(),
                rust_equiv: "rand::thread_rng().gen::<[u8; N]>".to_string(),
                requires_use: Some("rand::Rng".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Generate random bytes".to_string()),
            },
            FunctionMapping {
                python_name: "gauss".to_string(),
                rust_equiv: "rand_distr::Normal::new".to_string(),
                requires_use: Some("rand_distr::Normal".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Requires rand_distr crate".to_string()),
            },
        ],
    ));

    // re - Regular expressions (comprehensive)
    mappings.push((
        ModuleMapping {
            python_module: "re".to_string(),
            rust_crate: Some("regex".to_string()),
            rust_use: "regex".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Full regex support in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "compile".to_string(),
                rust_equiv: "Regex::new".to_string(),
                requires_use: Some("regex::Regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "match".to_string(),
                rust_equiv: "Regex::is_match".to_string(),
                requires_use: Some("regex::Regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "search".to_string(),
                rust_equiv: "Regex::find".to_string(),
                requires_use: Some("regex::Regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "findall".to_string(),
                rust_equiv: "Regex::find_iter".to_string(),
                requires_use: Some("regex::Regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns iterator, collect to vector".to_string()),
            },
            FunctionMapping {
                python_name: "sub".to_string(),
                rust_equiv: "Regex::replace_all".to_string(),
                requires_use: Some("regex::Regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "split".to_string(),
                rust_equiv: "Regex::split".to_string(),
                requires_use: Some("regex::Regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns iterator, collect to vector".to_string()),
            },
            FunctionMapping {
                python_name: "finditer".to_string(),
                rust_equiv: "Regex::find_iter".to_string(),
                requires_use: Some("regex::Regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns Match iterator".to_string()),
            },
            FunctionMapping {
                python_name: "escape".to_string(),
                rust_equiv: "regex::escape".to_string(),
                requires_use: Some("regex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Escapes special regex characters".to_string()),
            },
        ],
    ));

    // os - Operating system interface (comprehensive)
    mappings.push((
        ModuleMapping {
            python_module: "os".to_string(),
            rust_crate: None,
            rust_use: "std::env, std::fs".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Most filesystem operations require WASI".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "getcwd".to_string(),
                rust_equiv: "std::env::current_dir".to_string(),
                requires_use: Some("std::env".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "getenv".to_string(),
                rust_equiv: "std::env::var".to_string(),
                requires_use: Some("std::env".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns Result, not Option".to_string()),
            },
            FunctionMapping {
                python_name: "listdir".to_string(),
                rust_equiv: "std::fs::read_dir".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns iterator of DirEntry".to_string()),
            },
            FunctionMapping {
                python_name: "mkdir".to_string(),
                rust_equiv: "std::fs::create_dir".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "remove".to_string(),
                rust_equiv: "std::fs::remove_file".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "rename".to_string(),
                rust_equiv: "std::fs::rename".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "chdir".to_string(),
                rust_equiv: "std::env::set_current_dir".to_string(),
                requires_use: Some("std::env".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "makedirs".to_string(),
                rust_equiv: "std::fs::create_dir_all".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "rmdir".to_string(),
                rust_equiv: "std::fs::remove_dir".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "removedirs".to_string(),
                rust_equiv: "std::fs::remove_dir_all".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Removes directory and all parents".to_string()),
            },
            FunctionMapping {
                python_name: "walk".to_string(),
                rust_equiv: "walkdir::WalkDir".to_string(),
                requires_use: Some("walkdir::WalkDir".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Requires walkdir crate, recursive directory iterator".to_string()),
            },
            FunctionMapping {
                python_name: "scandir".to_string(),
                rust_equiv: "std::fs::read_dir".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns DirEntry iterator with metadata".to_string()),
            },
            FunctionMapping {
                python_name: "environ".to_string(),
                rust_equiv: "std::env::vars".to_string(),
                requires_use: Some("std::env".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns iterator of (String, String) tuples".to_string()),
            },
            FunctionMapping {
                python_name: "getpid".to_string(),
                rust_equiv: "std::process::id".to_string(),
                requires_use: Some("std::process".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Not available in browser WASM".to_string()),
            },
            FunctionMapping {
                python_name: "unlink".to_string(),
                rust_equiv: "std::fs::remove_file".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Alias for remove".to_string()),
            },
        ],
    ));

    // os.path - Path operations
    mappings.push((
        ModuleMapping {
            python_module: "os.path".to_string(),
            rust_crate: None,
            rust_use: "std::path::Path".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Path manipulation works, filesystem access needs WASI".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "join".to_string(),
                rust_equiv: "Path::join".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "exists".to_string(),
                rust_equiv: "Path::exists".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "basename".to_string(),
                rust_equiv: "Path::file_name".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "dirname".to_string(),
                rust_equiv: "Path::parent".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "split".to_string(),
                rust_equiv: "Path::file_name + Path::parent".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns tuple (parent, filename)".to_string()),
            },
            FunctionMapping {
                python_name: "splitext".to_string(),
                rust_equiv: "Path::file_stem + Path::extension".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns tuple (name, extension)".to_string()),
            },
            FunctionMapping {
                python_name: "isfile".to_string(),
                rust_equiv: "Path::is_file".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "isdir".to_string(),
                rust_equiv: "Path::is_dir".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "isabs".to_string(),
                rust_equiv: "Path::is_absolute".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "abspath".to_string(),
                rust_equiv: "std::fs::canonicalize".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns absolute path, resolves symlinks".to_string()),
            },
            FunctionMapping {
                python_name: "normpath".to_string(),
                rust_equiv: "Path::canonicalize".to_string(),
                requires_use: Some("std::path::Path".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Normalizes path, resolves .. and .".to_string()),
            },
            FunctionMapping {
                python_name: "getsize".to_string(),
                rust_equiv: "std::fs::metadata(path)?.len()".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns file size in bytes".to_string()),
            },
            FunctionMapping {
                python_name: "getmtime".to_string(),
                rust_equiv: "std::fs::metadata(path)?.modified()".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns SystemTime, need to convert to timestamp".to_string()),
            },
            FunctionMapping {
                python_name: "getctime".to_string(),
                rust_equiv: "std::fs::metadata(path)?.created()".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns SystemTime, need to convert to timestamp".to_string()),
            },
            FunctionMapping {
                python_name: "getatime".to_string(),
                rust_equiv: "std::fs::metadata(path)?.accessed()".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: Some("Returns SystemTime, need to convert to timestamp".to_string()),
            },
        ],
    ));

    // sys - System-specific parameters
    mappings.push((
        ModuleMapping {
            python_module: "sys".to_string(),
            rust_crate: None,
            rust_use: "std::env".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Limited in WASM, some features unavailable".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "argv".to_string(),
                rust_equiv: "std::env::args".to_string(),
                requires_use: Some("std::env".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Not available in browser WASM".to_string()),
            },
            FunctionMapping {
                python_name: "exit".to_string(),
                rust_equiv: "std::process::exit".to_string(),
                requires_use: Some("std::process".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Use panic! or return from main in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "platform".to_string(),
                rust_equiv: "std::env::consts::OS".to_string(),
                requires_use: Some("std::env::consts".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns 'wasm32' in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "stdin".to_string(),
                rust_equiv: "std::io::stdin".to_string(),
                requires_use: Some("std::io".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Not available in browser WASM".to_string()),
            },
            FunctionMapping {
                python_name: "stdout".to_string(),
                rust_equiv: "std::io::stdout".to_string(),
                requires_use: Some("std::io".to_string()),
                wasm_compatible: WasmCompatibility::Partial,
                transform_notes: Some("Limited in browser, use console.log via wasm-bindgen".to_string()),
            },
            FunctionMapping {
                python_name: "stderr".to_string(),
                rust_equiv: "std::io::stderr".to_string(),
                requires_use: Some("std::io".to_string()),
                wasm_compatible: WasmCompatibility::Partial,
                transform_notes: Some("Limited in browser, use console.error via wasm-bindgen".to_string()),
            },
            FunctionMapping {
                python_name: "version".to_string(),
                rust_equiv: "env!(\"CARGO_PKG_VERSION\")".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use compile-time version string".to_string()),
            },
            FunctionMapping {
                python_name: "version_info".to_string(),
                rust_equiv: "env!(\"CARGO_PKG_VERSION\")".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Parse version string into major.minor.patch".to_string()),
            },
            FunctionMapping {
                python_name: "maxsize".to_string(),
                rust_equiv: "isize::MAX".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "maxunicode".to_string(),
                rust_equiv: "char::MAX as u32".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Rust supports full Unicode range".to_string()),
            },
            FunctionMapping {
                python_name: "executable".to_string(),
                rust_equiv: "std::env::current_exe".to_string(),
                requires_use: Some("std::env".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Not available in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "byteorder".to_string(),
                rust_equiv: "cfg!(target_endian = \"little\")".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use cfg! macro to check endianness".to_string()),
            },
        ],
    ));

    // json - JSON encoder/decoder (comprehensive)
    mappings.push((
        ModuleMapping {
            python_module: "json".to_string(),
            rust_crate: Some("serde_json".to_string()),
            rust_use: "serde_json".to_string(),
            dependencies: vec!["serde".to_string()],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Full JSON support in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "dumps".to_string(),
                rust_equiv: "serde_json::to_string".to_string(),
                requires_use: Some("serde_json".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Requires Serialize trait".to_string()),
            },
            FunctionMapping {
                python_name: "loads".to_string(),
                rust_equiv: "serde_json::from_str".to_string(),
                requires_use: Some("serde_json".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Requires Deserialize trait".to_string()),
            },
            FunctionMapping {
                python_name: "dump".to_string(),
                rust_equiv: "serde_json::to_writer".to_string(),
                requires_use: Some("serde_json".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Writes to any writer".to_string()),
            },
            FunctionMapping {
                python_name: "load".to_string(),
                rust_equiv: "serde_json::from_reader".to_string(),
                requires_use: Some("serde_json".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Reads from any reader".to_string()),
            },
        ],
    ));

    // string - String operations
    mappings.push((
        ModuleMapping {
            python_module: "string".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("String methods are built into Rust String type".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "ascii_letters".to_string(),
                rust_equiv: "('a'..='z').chain('A'..='Z')".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Build from char ranges".to_string()),
            },
            FunctionMapping {
                python_name: "digits".to_string(),
                rust_equiv: "('0'..='9')".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Char range for digits".to_string()),
            },
        ],
    ));

    // traceback - Print or retrieve stack traceback
    mappings.push((
        ModuleMapping {
            python_module: "traceback".to_string(),
            rust_crate: Some("backtrace".to_string()),
            rust_use: "backtrace".to_string(),
            dependencies: vec![],
            version: "0.3".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Limited stack trace support in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "print_exc".to_string(),
                rust_equiv: "eprintln!(\"{:?}\", err)".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use Debug formatting for errors".to_string()),
            },
            FunctionMapping {
                python_name: "format_exc".to_string(),
                rust_equiv: "format!(\"{:?}\", err)".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // warnings - Warning control
    mappings.push((
        ModuleMapping {
            python_module: "warnings".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use tracing or log crate for warnings".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "warn".to_string(),
                rust_equiv: "tracing::warn!".to_string(),
                requires_use: Some("tracing".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use logging/tracing macros".to_string()),
            },
        ],
    ));

    // weakref - Weak references
    mappings.push((
        ModuleMapping {
            python_module: "weakref".to_string(),
            rust_crate: None,
            rust_use: "std::rc::Weak, std::sync::Weak".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use Weak<T> from std::rc or std::sync".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "ref".to_string(),
                rust_equiv: "Rc::downgrade".to_string(),
                requires_use: Some("std::rc::{Rc, Weak}".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Creates weak reference from Rc".to_string()),
            },
        ],
    ));

    // http - HTTP modules
    mappings.push((
        ModuleMapping {
            python_module: "http".to_string(),
            rust_crate: Some("http".to_string()),
            rust_use: "http".to_string(),
            dependencies: vec![],
            version: "0.2".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("HTTP types, use reqwest for actual HTTP requests".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "HTTPStatus".to_string(),
                rust_equiv: "http::StatusCode".to_string(),
                requires_use: Some("http::StatusCode".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "HTTPConnection".to_string(),
                rust_equiv: "reqwest::blocking::Client".to_string(),
                requires_use: Some("reqwest::blocking::Client".to_string()),
                wasm_compatible: WasmCompatibility::Partial,
                transform_notes: Some("Use reqwest::Client (async) in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "HTTPSConnection".to_string(),
                rust_equiv: "reqwest::blocking::Client".to_string(),
                requires_use: Some("reqwest::blocking::Client".to_string()),
                wasm_compatible: WasmCompatibility::Partial,
                transform_notes: Some("HTTPS handled automatically by reqwest".to_string()),
            },
            FunctionMapping {
                python_name: "request".to_string(),
                rust_equiv: "reqwest::Client::request".to_string(),
                requires_use: Some("reqwest::Client".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Async in WASM via fetch API".to_string()),
            },
            FunctionMapping {
                python_name: "getresponse".to_string(),
                rust_equiv: "response.await".to_string(),
                requires_use: Some("reqwest::Response".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Await the response future".to_string()),
            },
            FunctionMapping {
                python_name: "read".to_string(),
                rust_equiv: "response.text().await".to_string(),
                requires_use: Some("reqwest::Response".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: Some("Get response body as text".to_string()),
            },
            FunctionMapping {
                python_name: "HTTPServer".to_string(),
                rust_equiv: "axum::Server or hyper::Server".to_string(),
                requires_use: Some("axum or hyper".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("HTTP servers not available in WASM".to_string()),
            },
            FunctionMapping {
                python_name: "BaseHTTPRequestHandler".to_string(),
                rust_equiv: "axum::handler or hyper::service".to_string(),
                requires_use: Some("axum or hyper".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Use Axum handlers or Hyper services".to_string()),
            },
            FunctionMapping {
                python_name: "do_GET".to_string(),
                rust_equiv: "axum::get handler".to_string(),
                requires_use: Some("axum::routing::get".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Define GET route handler".to_string()),
            },
            FunctionMapping {
                python_name: "do_POST".to_string(),
                rust_equiv: "axum::post handler".to_string(),
                requires_use: Some("axum::routing::post".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Define POST route handler".to_string()),
            },
            FunctionMapping {
                python_name: "send_response".to_string(),
                rust_equiv: "Response::builder().status()".to_string(),
                requires_use: Some("http::Response".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Build HTTP response".to_string()),
            },
            FunctionMapping {
                python_name: "send_header".to_string(),
                rust_equiv: "response.headers_mut().insert()".to_string(),
                requires_use: Some("http::header".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Add header to response".to_string()),
            },
        ],
    ));

    // urllib.parse - URL parsing
    mappings.push((
        ModuleMapping {
            python_module: "urllib.parse".to_string(),
            rust_crate: Some("url".to_string()),
            rust_use: "url".to_string(),
            dependencies: vec![],
            version: "2".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Full URL parsing in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "urlparse".to_string(),
                rust_equiv: "Url::parse".to_string(),
                requires_use: Some("url::Url".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "urlencode".to_string(),
                rust_equiv: "url::form_urlencoded::Serializer".to_string(),
                requires_use: Some("url::form_urlencoded".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "quote".to_string(),
                rust_equiv: "urlencoding::encode".to_string(),
                requires_use: Some("urlencoding".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Requires urlencoding crate".to_string()),
            },
        ],
    ));

    // html - HTML manipulation
    mappings.push((
        ModuleMapping {
            python_module: "html".to_string(),
            rust_crate: Some("html-escape".to_string()),
            rust_use: "html_escape".to_string(),
            dependencies: vec![],
            version: "0.2".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("HTML escaping/unescaping".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "escape".to_string(),
                rust_equiv: "html_escape::encode_text".to_string(),
                requires_use: Some("html_escape".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "unescape".to_string(),
                rust_equiv: "html_escape::decode_html_entities".to_string(),
                requires_use: Some("html_escape".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // html.parser - HTML/XHTML parser
    mappings.push((
        ModuleMapping {
            python_module: "html.parser".to_string(),
            rust_crate: Some("scraper".to_string()),
            rust_use: "scraper".to_string(),
            dependencies: vec![],
            version: "0.17".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("HTML parsing and selection".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "HTMLParser".to_string(),
                rust_equiv: "scraper::Html::parse_document".to_string(),
                requires_use: Some("scraper::Html".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // statistics - Mathematical statistics
    mappings.push((
        ModuleMapping {
            python_module: "statistics".to_string(),
            rust_crate: Some("statistical".to_string()),
            rust_use: "statistical".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Statistical functions, fully WASM compatible".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "mean".to_string(),
                rust_equiv: "statistical::mean".to_string(),
                requires_use: Some("statistical".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "median".to_string(),
                rust_equiv: "statistical::median".to_string(),
                requires_use: Some("statistical".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "stdev".to_string(),
                rust_equiv: "statistical::standard_deviation".to_string(),
                requires_use: Some("statistical".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // binascii - Binary/ASCII conversions
    mappings.push((
        ModuleMapping {
            python_module: "binascii".to_string(),
            rust_crate: Some("hex".to_string()),
            rust_use: "hex".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Binary to ASCII conversions".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "hexlify".to_string(),
                rust_equiv: "hex::encode".to_string(),
                requires_use: Some("hex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "unhexlify".to_string(),
                rust_equiv: "hex::decode".to_string(),
                requires_use: Some("hex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // hmac - Keyed-hashing for message authentication
    mappings.push((
        ModuleMapping {
            python_module: "hmac".to_string(),
            rust_crate: Some("hmac".to_string()),
            rust_use: "hmac".to_string(),
            dependencies: vec!["sha2".to_string()],
            version: "0.12".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("HMAC implementation".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "new".to_string(),
                rust_equiv: "Hmac::<Sha256>::new_from_slice".to_string(),
                requires_use: Some("hmac::{Hmac, Mac}".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use with sha2::Sha256".to_string()),
            },
        ],
    ));

    // crypt - Password hashing
    mappings.push((
        ModuleMapping {
            python_module: "crypt".to_string(),
            rust_crate: Some("bcrypt".to_string()),
            rust_use: "bcrypt".to_string(),
            dependencies: vec![],
            version: "0.15".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use bcrypt or argon2 for password hashing".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "crypt".to_string(),
                rust_equiv: "bcrypt::hash".to_string(),
                requires_use: Some("bcrypt".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // getpass - Portable password input
    mappings.push((
        ModuleMapping {
            python_module: "getpass".to_string(),
            rust_crate: Some("rpassword".to_string()),
            rust_use: "rpassword".to_string(),
            dependencies: vec![],
            version: "7".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Terminal input not available in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "getpass".to_string(),
                rust_equiv: "rpassword::read_password".to_string(),
                requires_use: Some("rpassword".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: Some("Only works in native terminal".to_string()),
            },
        ],
    ));

    // platform - Access to platform identifying data
    mappings.push((
        ModuleMapping {
            python_module: "platform".to_string(),
            rust_crate: Some("platforms".to_string()),
            rust_use: "platforms".to_string(),
            dependencies: vec![],
            version: "3".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Platform detection, limited in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "system".to_string(),
                rust_equiv: "std::env::consts::OS".to_string(),
                requires_use: Some("std::env::consts".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "machine".to_string(),
                rust_equiv: "std::env::consts::ARCH".to_string(),
                requires_use: Some("std::env::consts".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // locale - Internationalization services
    mappings.push((
        ModuleMapping {
            python_module: "locale".to_string(),
            rust_crate: Some("sys-locale".to_string()),
            rust_use: "sys_locale".to_string(),
            dependencies: vec![],
            version: "0.3".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Locale detection, limited in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "getlocale".to_string(),
                rust_equiv: "sys_locale::get_locale".to_string(),
                requires_use: Some("sys_locale".to_string()),
                wasm_compatible: WasmCompatibility::RequiresJsInterop,
                transform_notes: None,
            },
        ],
    ));

    // gettext - Multilingual internationalization
    mappings.push((
        ModuleMapping {
            python_module: "gettext".to_string(),
            rust_crate: Some("gettext".to_string()),
            rust_use: "gettext".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("i18n support".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "gettext".to_string(),
                rust_equiv: "gettext::gettext".to_string(),
                requires_use: Some("gettext".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // pprint - Data pretty printer
    mappings.push((
        ModuleMapping {
            python_module: "pprint".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use Debug formatting with {:#?}".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "pprint".to_string(),
                rust_equiv: "println!(\"{:#?}\", value)".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Pretty-print with Debug trait".to_string()),
            },
        ],
    ));

    // reprlib - Alternate repr() implementation
    mappings.push((
        ModuleMapping {
            python_module: "reprlib".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use Debug or Display traits".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "repr".to_string(),
                rust_equiv: "format!(\"{:?}\", value)".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // tokenize - Tokenizer for Python source
    mappings.push((
        ModuleMapping {
            python_module: "tokenize".to_string(),
            rust_crate: Some("logos".to_string()),
            rust_use: "logos".to_string(),
            dependencies: vec![],
            version: "0.13".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use logos for tokenization".to_string()),
        },
        vec![],
    ));

    // ast - Abstract Syntax Trees
    mappings.push((
        ModuleMapping {
            python_module: "ast".to_string(),
            rust_crate: Some("syn".to_string()),
            rust_use: "syn".to_string(),
            dependencies: vec![],
            version: "2".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use syn for Rust AST parsing".to_string()),
        },
        vec![],
    ));

    // dis - Disassembler
    mappings.push((
        ModuleMapping {
            python_module: "dis".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("No direct equivalent in Rust".to_string()),
        },
        vec![],
    ));

    // inspect - Inspect live objects
    mappings.push((
        ModuleMapping {
            python_module: "inspect".to_string(),
            rust_crate: None,
            rust_use: "std::any".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Limited reflection via Any trait".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "getmembers".to_string(),
                rust_equiv: "std::any::type_name".to_string(),
                requires_use: Some("std::any".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Limited introspection in Rust".to_string()),
            },
        ],
    ));

    // code - Interpreter base classes
    mappings.push((
        ModuleMapping {
            python_module: "code".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("No interpreter in compiled Rust".to_string()),
        },
        vec![],
    ));

    // operator - Standard operators as functions
    mappings.push((
        ModuleMapping {
            python_module: "operator".to_string(),
            rust_crate: None,
            rust_use: "std::ops".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use operator traits from std::ops".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "add".to_string(),
                rust_equiv: "std::ops::Add::add".to_string(),
                requires_use: Some("std::ops::Add".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use + operator directly".to_string()),
            },
            FunctionMapping {
                python_name: "mul".to_string(),
                rust_equiv: "std::ops::Mul::mul".to_string(),
                requires_use: Some("std::ops::Mul".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use * operator directly".to_string()),
            },
            FunctionMapping {
                python_name: "sub".to_string(),
                rust_equiv: "std::ops::Sub::sub".to_string(),
                requires_use: Some("std::ops::Sub".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use - operator".to_string()),
            },
            FunctionMapping {
                python_name: "truediv".to_string(),
                rust_equiv: "std::ops::Div::div".to_string(),
                requires_use: Some("std::ops::Div".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use / operator".to_string()),
            },
            FunctionMapping {
                python_name: "floordiv".to_string(),
                rust_equiv: "a / b for integers".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Integer division already floors".to_string()),
            },
            FunctionMapping {
                python_name: "mod".to_string(),
                rust_equiv: "std::ops::Rem::rem".to_string(),
                requires_use: Some("std::ops::Rem".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use % operator".to_string()),
            },
            FunctionMapping {
                python_name: "pow".to_string(),
                rust_equiv: "f64::powf or i32::pow".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use appropriate power method".to_string()),
            },
            FunctionMapping {
                python_name: "eq".to_string(),
                rust_equiv: "std::cmp::PartialEq::eq".to_string(),
                requires_use: Some("std::cmp::PartialEq".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use == operator".to_string()),
            },
            FunctionMapping {
                python_name: "ne".to_string(),
                rust_equiv: "std::cmp::PartialEq::ne".to_string(),
                requires_use: Some("std::cmp::PartialEq".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use != operator".to_string()),
            },
            FunctionMapping {
                python_name: "lt".to_string(),
                rust_equiv: "std::cmp::PartialOrd::lt".to_string(),
                requires_use: Some("std::cmp::PartialOrd".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use < operator".to_string()),
            },
            FunctionMapping {
                python_name: "le".to_string(),
                rust_equiv: "std::cmp::PartialOrd::le".to_string(),
                requires_use: Some("std::cmp::PartialOrd".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use <= operator".to_string()),
            },
            FunctionMapping {
                python_name: "gt".to_string(),
                rust_equiv: "std::cmp::PartialOrd::gt".to_string(),
                requires_use: Some("std::cmp::PartialOrd".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use > operator".to_string()),
            },
            FunctionMapping {
                python_name: "ge".to_string(),
                rust_equiv: "std::cmp::PartialOrd::ge".to_string(),
                requires_use: Some("std::cmp::PartialOrd".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Or use >= operator".to_string()),
            },
            FunctionMapping {
                python_name: "itemgetter".to_string(),
                rust_equiv: "closure |x| x[index]".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use closure for indexing".to_string()),
            },
            FunctionMapping {
                python_name: "attrgetter".to_string(),
                rust_equiv: "closure |x| x.attr".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use closure for field access".to_string()),
            },
        ],
    ));

    // pdb - Python debugger
    mappings.push((
        ModuleMapping {
            python_module: "pdb".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Use rust-gdb or lldb for debugging".to_string()),
        },
        vec![],
    ));

    // profile/cProfile - Profiling
    mappings.push((
        ModuleMapping {
            python_module: "cProfile".to_string(),
            rust_crate: Some("pprof".to_string()),
            rust_use: "pprof".to_string(),
            dependencies: vec![],
            version: "0.13".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Profiling support, limited in WASM".to_string()),
        },
        vec![],
    ));

    // timeit - Measure execution time
    mappings.push((
        ModuleMapping {
            python_module: "timeit".to_string(),
            rust_crate: Some("criterion".to_string()),
            rust_use: "criterion".to_string(),
            dependencies: vec![],
            version: "0.5".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Benchmarking framework".to_string()),
        },
        vec![],
    ));

    // mmap - Memory-mapped file support
    mappings.push((
        ModuleMapping {
            python_module: "mmap".to_string(),
            rust_crate: Some("memmap2".to_string()),
            rust_use: "memmap2".to_string(),
            dependencies: vec![],
            version: "0.9".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Memory mapping not available in WASM".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "mmap".to_string(),
                rust_equiv: "memmap2::Mmap::map".to_string(),
                requires_use: Some("memmap2::Mmap".to_string()),
                wasm_compatible: WasmCompatibility::Incompatible,
                transform_notes: None,
            },
        ],
    ));

    // readline - GNU readline interface
    mappings.push((
        ModuleMapping {
            python_module: "readline".to_string(),
            rust_crate: Some("rustyline".to_string()),
            rust_use: "rustyline".to_string(),
            dependencies: vec![],
            version: "13".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Terminal readline not available in WASM".to_string()),
        },
        vec![],
    ));

    // shelve - Python object persistence
    mappings.push((
        ModuleMapping {
            python_module: "shelve".to_string(),
            rust_crate: Some("sled".to_string()),
            rust_use: "sled".to_string(),
            dependencies: vec!["serde".to_string()],
            version: "0.34".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Key-value store, needs filesystem".to_string()),
        },
        vec![],
    ));

    // dbm - Database interfaces
    mappings.push((
        ModuleMapping {
            python_module: "dbm".to_string(),
            rust_crate: Some("sled".to_string()),
            rust_use: "sled".to_string(),
            dependencies: vec![],
            version: "0.34".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Embedded database".to_string()),
        },
        vec![],
    ));

    // sqlite3 - DB-API 2.0 interface
    mappings.push((
        ModuleMapping {
            python_module: "sqlite3".to_string(),
            rust_crate: Some("rusqlite".to_string()),
            rust_use: "rusqlite".to_string(),
            dependencies: vec![],
            version: "0.30".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("SQLite embedded database, needs WASI filesystem".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "connect".to_string(),
                rust_equiv: "rusqlite::Connection::open".to_string(),
                requires_use: Some("rusqlite::Connection".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
        ],
    ));

    // bz2 - Bzip2 compression
    mappings.push((
        ModuleMapping {
            python_module: "bz2".to_string(),
            rust_crate: Some("bzip2".to_string()),
            rust_use: "bzip2".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Compression/decompression".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "compress".to_string(),
                rust_equiv: "bzip2::write::BzEncoder::new".to_string(),
                requires_use: Some("bzip2::write::BzEncoder".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // lzma - LZMA compression
    mappings.push((
        ModuleMapping {
            python_module: "lzma".to_string(),
            rust_crate: Some("xz2".to_string()),
            rust_use: "xz2".to_string(),
            dependencies: vec![],
            version: "0.1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("LZMA/XZ compression".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "compress".to_string(),
                rust_equiv: "xz2::write::XzEncoder::new".to_string(),
                requires_use: Some("xz2::write::XzEncoder".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // tarfile - Tar archive reading/writing
    mappings.push((
        ModuleMapping {
            python_module: "tarfile".to_string(),
            rust_crate: Some("tar".to_string()),
            rust_use: "tar".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
            wasm_compatible: WasmCompatibility::Partial,
            notes: Some("Works with in-memory data, file I/O needs WASI".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "TarFile".to_string(),
                rust_equiv: "tar::Archive::new".to_string(),
                requires_use: Some("tar::Archive".to_string()),
                wasm_compatible: WasmCompatibility::Partial,
                transform_notes: None,
            },
        ],
    ));

    // filecmp - File and directory comparisons
    mappings.push((
        ModuleMapping {
            python_module: "filecmp".to_string(),
            rust_crate: None,
            rust_use: "std::fs".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Use std::fs to read and compare".to_string()),
        },
        vec![],
    ));

    // fileinput - Iterate over lines from multiple input streams
    mappings.push((
        ModuleMapping {
            python_module: "fileinput".to_string(),
            rust_crate: None,
            rust_use: "std::io::BufRead".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Use BufReader with chained readers".to_string()),
        },
        vec![],
    ));

    // linecache - Random access to text lines
    mappings.push((
        ModuleMapping {
            python_module: "linecache".to_string(),
            rust_crate: None,
            rust_use: "std::fs".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Read file and index lines manually".to_string()),
        },
        vec![],
    ));

    // shutil - High-level file operations
    mappings.push((
        ModuleMapping {
            python_module: "shutil".to_string(),
            rust_crate: Some("fs_extra".to_string()),
            rust_use: "fs_extra".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::RequiresWasi,
            notes: Some("Extended filesystem operations".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "copy".to_string(),
                rust_equiv: "std::fs::copy".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "rmtree".to_string(),
                rust_equiv: "std::fs::remove_dir_all".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
            FunctionMapping {
                python_name: "move".to_string(),
                rust_equiv: "std::fs::rename".to_string(),
                requires_use: Some("std::fs".to_string()),
                wasm_compatible: WasmCompatibility::RequiresWasi,
                transform_notes: None,
            },
        ],
    ));

    // pty - Pseudo-terminal utilities
    mappings.push((
        ModuleMapping {
            python_module: "pty".to_string(),
            rust_crate: Some("nix".to_string()),
            rust_use: "nix::pty".to_string(),
            dependencies: vec![],
            version: "0.27".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("PTY not available in WASM".to_string()),
        },
        vec![],
    ));

    // select - I/O multiplexing
    mappings.push((
        ModuleMapping {
            python_module: "select".to_string(),
            rust_crate: Some("mio".to_string()),
            rust_use: "mio".to_string(),
            dependencies: vec![],
            version: "0.8".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Use async I/O instead in WASM".to_string()),
        },
        vec![],
    ));

    // selectors - High-level I/O multiplexing
    mappings.push((
        ModuleMapping {
            python_module: "selectors".to_string(),
            rust_crate: Some("mio".to_string()),
            rust_use: "mio".to_string(),
            dependencies: vec![],
            version: "0.8".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Use async I/O instead".to_string()),
        },
        vec![],
    ));

    // sched - Event scheduler
    mappings.push((
        ModuleMapping {
            python_module: "sched".to_string(),
            rust_crate: Some("tokio".to_string()),
            rust_use: "tokio::time".to_string(),
            dependencies: vec![],
            version: "1".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Use tokio timers for scheduling".to_string()),
        },
        vec![],
    ));

    // ctypes - Foreign function library
    mappings.push((
        ModuleMapping {
            python_module: "ctypes".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Incompatible,
            notes: Some("Native code is Rust, use FFI crate for C interop".to_string()),
        },
        vec![],
    ));

    // numbers - Numeric abstract base classes
    mappings.push((
        ModuleMapping {
            python_module: "numbers".to_string(),
            rust_crate: Some("num-traits".to_string()),
            rust_use: "num_traits".to_string(),
            dependencies: vec![],
            version: "0.2".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Numeric traits".to_string()),
        },
        vec![],
    ));

    // cmath - Mathematical functions for complex numbers
    mappings.push((
        ModuleMapping {
            python_module: "cmath".to_string(),
            rust_crate: Some("num-complex".to_string()),
            rust_use: "num_complex".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Complex number arithmetic".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "sqrt".to_string(),
                rust_equiv: "Complex::sqrt".to_string(),
                requires_use: Some("num_complex::Complex".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: None,
            },
        ],
    ));

    // array - Efficient arrays of numeric values
    mappings.push((
        ModuleMapping {
            python_module: "array".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use Vec<T> or arrays [T; N]".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "array".to_string(),
                rust_equiv: "Vec::new".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Vec<T> for dynamic, [T; N] for fixed size".to_string()),
            },
        ],
    ));

    // bisect - Array bisection algorithm
    mappings.push((
        ModuleMapping {
            python_module: "bisect".to_string(),
            rust_crate: None,
            rust_use: "".to_string(),
            dependencies: vec![],
            version: "*".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Use binary_search on slices".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "bisect_left".to_string(),
                rust_equiv: "slice::binary_search".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Returns Result with Ok(index) or Err(insert_index)".to_string()),
            },
        ],
    ));

    // calendar - General calendar-related functions
    mappings.push((
        ModuleMapping {
            python_module: "calendar".to_string(),
            rust_crate: Some("chrono".to_string()),
            rust_use: "chrono".to_string(),
            dependencies: vec![],
            version: "0.4".to_string(),
            wasm_compatible: WasmCompatibility::RequiresJsInterop,
            notes: Some("Use chrono for date/time calculations".to_string()),
        },
        vec![],
    ));

    // codecs - Codec registry and base classes
    mappings.push((
        ModuleMapping {
            python_module: "codecs".to_string(),
            rust_crate: Some("encoding_rs".to_string()),
            rust_use: "encoding_rs".to_string(),
            dependencies: vec![],
            version: "0.8".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Character encoding/decoding".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "encode".to_string(),
                rust_equiv: "String::as_bytes".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("UTF-8 by default, use encoding_rs for others".to_string()),
            },
            FunctionMapping {
                python_name: "decode".to_string(),
                rust_equiv: "String::from_utf8".to_string(),
                requires_use: None,
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use encoding_rs for non-UTF8".to_string()),
            },
        ],
    ));

    // unicodedata - Unicode database
    mappings.push((
        ModuleMapping {
            python_module: "unicodedata".to_string(),
            rust_crate: Some("unicode-normalization".to_string()),
            rust_use: "unicode_normalization".to_string(),
            dependencies: vec!["unicode-segmentation".to_string()],
            version: "0.1".to_string(),
            wasm_compatible: WasmCompatibility::Full,
            notes: Some("Unicode normalization and properties".to_string()),
        },
        vec![
            FunctionMapping {
                python_name: "normalize".to_string(),
                rust_equiv: "unicode_normalization::UnicodeNormalization::nfc".to_string(),
                requires_use: Some("unicode_normalization::UnicodeNormalization".to_string()),
                wasm_compatible: WasmCompatibility::Full,
                transform_notes: Some("Use .nfc(), .nfd(), .nfkc(), .nfkd()".to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_mappings_count() {
        let mappings = init_critical_mappings();
        assert!(mappings.len() >= 100, "Should have at least 100 stdlib mappings, got {}", mappings.len());
    }

    #[test]
    fn test_coverage_stats() {
        let stats = get_coverage_stats();
        assert!(stats.mapped_modules >= 100, "Should have mapped at least 100 modules");
        assert!(stats.full_wasm_compat > 0, "Should have some fully WASM compatible modules");
    }

    // Test new module mappings
    #[test]
    fn test_abc_module() {
        let mappings = init_critical_mappings();
        let abc = mappings.iter().find(|(m, _)| m.python_module == "abc");
        assert!(abc.is_some(), "abc module should be mapped");
        let (module, funcs) = abc.unwrap();
        assert_eq!(module.wasm_compatible, WasmCompatibility::Full);
        assert!(funcs.len() > 0);
    }

    #[test]
    fn test_contextlib_module() {
        let mappings = init_critical_mappings();
        let contextlib = mappings.iter().find(|(m, _)| m.python_module == "contextlib");
        assert!(contextlib.is_some(), "contextlib module should be mapped");
        assert_eq!(contextlib.unwrap().0.wasm_compatible, WasmCompatibility::Full);
    }

    #[test]
    fn test_concurrent_futures_module() {
        let mappings = init_critical_mappings();
        let concurrent = mappings.iter().find(|(m, _)| m.python_module == "concurrent.futures");
        assert!(concurrent.is_some(), "concurrent.futures module should be mapped");
        let (module, funcs) = concurrent.unwrap();
        assert_eq!(module.rust_crate, Some("rayon".to_string()));
        assert!(funcs.len() >= 3, "Should have ThreadPoolExecutor, ProcessPoolExecutor, Future mappings");
    }

    #[test]
    fn test_multiprocessing_module() {
        let mappings = init_critical_mappings();
        let mp = mappings.iter().find(|(m, _)| m.python_module == "multiprocessing");
        assert!(mp.is_some(), "multiprocessing module should be mapped");
        assert_eq!(mp.unwrap().0.wasm_compatible, WasmCompatibility::Incompatible);
    }

    #[test]
    fn test_random_module_comprehensive() {
        let mappings = init_critical_mappings();
        let random = mappings.iter().find(|(m, _)| m.python_module == "random");
        assert!(random.is_some(), "random module should be mapped");
        let (module, funcs) = random.unwrap();
        assert_eq!(module.rust_crate, Some("rand".to_string()));
        assert!(funcs.len() >= 4, "Should have random, randint, choice, shuffle");

        // Check specific functions
        let randint = funcs.iter().find(|f| f.python_name == "randint");
        assert!(randint.is_some());
        assert_eq!(randint.unwrap().requires_use, Some("rand::Rng".to_string()));
    }

    #[test]
    fn test_re_module_comprehensive() {
        let mappings = init_critical_mappings();
        let re = mappings.iter().find(|(m, _)| m.python_module == "re");
        assert!(re.is_some(), "re module should be mapped");
        let (module, funcs) = re.unwrap();
        assert_eq!(module.rust_crate, Some("regex".to_string()));
        assert!(funcs.len() >= 5, "Should have compile, match, search, findall, sub");
        assert_eq!(module.wasm_compatible, WasmCompatibility::Full);
    }

    #[test]
    fn test_os_module_comprehensive() {
        let mappings = init_critical_mappings();
        let os = mappings.iter().find(|(m, _)| m.python_module == "os");
        assert!(os.is_some(), "os module should be mapped");
        let (module, funcs) = os.unwrap();
        assert_eq!(module.wasm_compatible, WasmCompatibility::RequiresWasi);
        assert!(funcs.len() >= 6, "Should have getcwd, getenv, listdir, mkdir, remove, rename");
    }

    #[test]
    fn test_os_path_module() {
        let mappings = init_critical_mappings();
        let os_path = mappings.iter().find(|(m, _)| m.python_module == "os.path");
        assert!(os_path.is_some(), "os.path module should be mapped");
        let (_, funcs) = os_path.unwrap();
        assert!(funcs.len() >= 4, "Should have join, exists, basename, dirname");
    }

    #[test]
    fn test_sys_module_comprehensive() {
        let mappings = init_critical_mappings();
        let sys = mappings.iter().find(|(m, _)| m.python_module == "sys");
        assert!(sys.is_some(), "sys module should be mapped");
        let (module, funcs) = sys.unwrap();
        assert_eq!(module.wasm_compatible, WasmCompatibility::Partial);
        assert!(funcs.len() >= 3, "Should have argv, exit, platform");
    }

    #[test]
    fn test_json_module_comprehensive() {
        let mappings = init_critical_mappings();
        let json = mappings.iter().find(|(m, _)| m.python_module == "json");
        assert!(json.is_some(), "json module should be mapped");
        let (module, funcs) = json.unwrap();
        assert_eq!(module.rust_crate, Some("serde_json".to_string()));
        assert!(funcs.len() >= 4, "Should have dumps, loads, dump, load");
        assert_eq!(module.wasm_compatible, WasmCompatibility::Full);
    }

    #[test]
    fn test_statistics_module() {
        let mappings = init_critical_mappings();
        let stats = mappings.iter().find(|(m, _)| m.python_module == "statistics");
        assert!(stats.is_some(), "statistics module should be mapped");
        let (module, funcs) = stats.unwrap();
        assert_eq!(module.rust_crate, Some("statistical".to_string()));
        assert_eq!(module.wasm_compatible, WasmCompatibility::Full);
        assert!(funcs.len() >= 3, "Should have mean, median, stdev");
    }

    #[test]
    fn test_urllib_parse_module() {
        let mappings = init_critical_mappings();
        let url_parse = mappings.iter().find(|(m, _)| m.python_module == "urllib.parse");
        assert!(url_parse.is_some(), "urllib.parse module should be mapped");
        let (module, funcs) = url_parse.unwrap();
        assert_eq!(module.rust_crate, Some("url".to_string()));
        assert!(funcs.len() >= 3, "Should have urlparse, urlencode, quote");
    }

    #[test]
    fn test_html_modules() {
        let mappings = init_critical_mappings();

        let html = mappings.iter().find(|(m, _)| m.python_module == "html");
        assert!(html.is_some(), "html module should be mapped");

        let html_parser = mappings.iter().find(|(m, _)| m.python_module == "html.parser");
        assert!(html_parser.is_some(), "html.parser module should be mapped");
        assert_eq!(html_parser.unwrap().0.rust_crate, Some("scraper".to_string()));
    }

    #[test]
    fn test_crypto_modules() {
        let mappings = init_critical_mappings();

        let hmac = mappings.iter().find(|(m, _)| m.python_module == "hmac");
        assert!(hmac.is_some(), "hmac module should be mapped");
        assert_eq!(hmac.unwrap().0.wasm_compatible, WasmCompatibility::Full);

        let crypt = mappings.iter().find(|(m, _)| m.python_module == "crypt");
        assert!(crypt.is_some(), "crypt module should be mapped");
        assert_eq!(crypt.unwrap().0.rust_crate, Some("bcrypt".to_string()));
    }

    #[test]
    fn test_compression_modules() {
        let mappings = init_critical_mappings();

        let bz2 = mappings.iter().find(|(m, _)| m.python_module == "bz2");
        assert!(bz2.is_some(), "bz2 module should be mapped");
        assert_eq!(bz2.unwrap().0.wasm_compatible, WasmCompatibility::Full);

        let lzma = mappings.iter().find(|(m, _)| m.python_module == "lzma");
        assert!(lzma.is_some(), "lzma module should be mapped");

        let tarfile = mappings.iter().find(|(m, _)| m.python_module == "tarfile");
        assert!(tarfile.is_some(), "tarfile module should be mapped");
    }

    #[test]
    fn test_database_modules() {
        let mappings = init_critical_mappings();

        let sqlite = mappings.iter().find(|(m, _)| m.python_module == "sqlite3");
        assert!(sqlite.is_some(), "sqlite3 module should be mapped");
        assert_eq!(sqlite.unwrap().0.rust_crate, Some("rusqlite".to_string()));

        let shelve = mappings.iter().find(|(m, _)| m.python_module == "shelve");
        assert!(shelve.is_some(), "shelve module should be mapped");

        let dbm = mappings.iter().find(|(m, _)| m.python_module == "dbm");
        assert!(dbm.is_some(), "dbm module should be mapped");
    }

    #[test]
    fn test_unicode_modules() {
        let mappings = init_critical_mappings();

        let unicodedata = mappings.iter().find(|(m, _)| m.python_module == "unicodedata");
        assert!(unicodedata.is_some(), "unicodedata module should be mapped");
        assert_eq!(unicodedata.unwrap().0.wasm_compatible, WasmCompatibility::Full);

        let codecs = mappings.iter().find(|(m, _)| m.python_module == "codecs");
        assert!(codecs.is_some(), "codecs module should be mapped");
    }

    #[test]
    fn test_system_modules() {
        let mappings = init_critical_mappings();

        let platform = mappings.iter().find(|(m, _)| m.python_module == "platform");
        assert!(platform.is_some(), "platform module should be mapped");

        let locale = mappings.iter().find(|(m, _)| m.python_module == "locale");
        assert!(locale.is_some(), "locale module should be mapped");

        let gettext = mappings.iter().find(|(m, _)| m.python_module == "gettext");
        assert!(gettext.is_some(), "gettext module should be mapped");
    }

    #[test]
    fn test_debugging_profiling_modules() {
        let mappings = init_critical_mappings();

        let traceback = mappings.iter().find(|(m, _)| m.python_module == "traceback");
        assert!(traceback.is_some(), "traceback module should be mapped");

        let warnings = mappings.iter().find(|(m, _)| m.python_module == "warnings");
        assert!(warnings.is_some(), "warnings module should be mapped");

        let pdb = mappings.iter().find(|(m, _)| m.python_module == "pdb");
        assert!(pdb.is_some(), "pdb module should be mapped");

        let cprofile = mappings.iter().find(|(m, _)| m.python_module == "cProfile");
        assert!(cprofile.is_some(), "cProfile module should be mapped");

        let timeit = mappings.iter().find(|(m, _)| m.python_module == "timeit");
        assert!(timeit.is_some(), "timeit module should be mapped");
    }

    #[test]
    fn test_ast_parsing_modules() {
        let mappings = init_critical_mappings();

        let tokenize = mappings.iter().find(|(m, _)| m.python_module == "tokenize");
        assert!(tokenize.is_some(), "tokenize module should be mapped");

        let ast = mappings.iter().find(|(m, _)| m.python_module == "ast");
        assert!(ast.is_some(), "ast module should be mapped");

        let inspect = mappings.iter().find(|(m, _)| m.python_module == "inspect");
        assert!(inspect.is_some(), "inspect module should be mapped");
    }

    #[test]
    fn test_number_modules() {
        let mappings = init_critical_mappings();

        let cmath = mappings.iter().find(|(m, _)| m.python_module == "cmath");
        assert!(cmath.is_some(), "cmath module should be mapped");
        assert_eq!(cmath.unwrap().0.rust_crate, Some("num-complex".to_string()));

        let numbers = mappings.iter().find(|(m, _)| m.python_module == "numbers");
        assert!(numbers.is_some(), "numbers module should be mapped");

        let array = mappings.iter().find(|(m, _)| m.python_module == "array");
        assert!(array.is_some(), "array module should be mapped");

        let bisect = mappings.iter().find(|(m, _)| m.python_module == "bisect");
        assert!(bisect.is_some(), "bisect module should be mapped");
    }

    #[test]
    fn test_file_operations() {
        let mappings = init_critical_mappings();

        let shutil = mappings.iter().find(|(m, _)| m.python_module == "shutil");
        assert!(shutil.is_some(), "shutil module should be mapped");
        let (_, funcs) = shutil.unwrap();
        assert!(funcs.len() >= 3, "Should have copy, rmtree, move");

        let filecmp = mappings.iter().find(|(m, _)| m.python_module == "filecmp");
        assert!(filecmp.is_some(), "filecmp module should be mapped");
    }

    #[test]
    fn test_wasm_compatibility_categories() {
        let stats = get_coverage_stats();

        // Verify we have modules in each category
        assert!(stats.full_wasm_compat > 0, "Should have fully compatible modules");
        assert!(stats.partial_wasm_compat > 0, "Should have partially compatible modules");
        assert!(stats.requires_wasi > 0, "Should have WASI-dependent modules");
        assert!(stats.requires_js_interop > 0, "Should have JS interop modules");
        assert!(stats.incompatible > 0, "Should have incompatible modules");

        // Total should equal sum of categories
        let total_categories = stats.full_wasm_compat + stats.partial_wasm_compat +
                               stats.requires_wasi + stats.requires_js_interop +
                               stats.incompatible;
        assert_eq!(total_categories, stats.mapped_modules,
                   "All modules should be categorized by WASM compatibility");
    }

    #[test]
    fn test_all_mappings_have_rust_use() {
        let mappings = init_critical_mappings();
        for (module, _) in mappings {
            // rust_use can be empty for some modules (like abc, contextlib) that map to Rust concepts
            // but we should have either rust_use or a note explaining why not
            if module.rust_use.is_empty() {
                assert!(module.notes.is_some(),
                       "Module {} has no rust_use and no notes explaining why",
                       module.python_module);
            }
        }
    }

    #[test]
    fn test_function_mappings_completeness() {
        let mappings = init_critical_mappings();

        // Modules that should have function mappings
        let modules_with_funcs = vec!["math", "random", "re", "os", "json", "collections"];

        for module_name in modules_with_funcs {
            let module = mappings.iter().find(|(m, _)| m.python_module == module_name);
            assert!(module.is_some(), "Module {} should exist", module_name);
            let (_, funcs) = module.unwrap();
            assert!(!funcs.is_empty(), "Module {} should have function mappings", module_name);
        }
    }
}
