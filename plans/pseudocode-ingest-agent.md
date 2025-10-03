# PORTALIS: Ingest Agent Pseudocode
## SPARC Phase 2: Pseudocode - Ingest Agent

**Version:** 1.0
**Date:** 2025-10-03
**Agent:** Ingest Agent (Agent #1)
**Testing Approach:** London School TDD (Outside-In, Mockist)

---

## Table of Contents

1. [Agent Overview](#1-agent-overview)
2. [Data Structures](#2-data-structures)
3. [Core Algorithms](#3-core-algorithms)
4. [Input/Output Contracts](#4-inputoutput-contracts)
5. [Error Handling Strategy](#5-error-handling-strategy)
6. [London School TDD Test Points](#6-london-school-tdd-test-points)

---

## 1. Agent Overview

### 1.1 Purpose

The **Ingest Agent** is the entry point of the Portalis pipeline. It is responsible for:

- Accepting Python input (single file or package directory)
- Validating Python syntax and structure
- Detecting circular dependencies
- Processing configuration inputs
- Cataloging external dependencies (stdlib, third-party, internal)
- Producing a validated, normalized representation of the input for downstream agents

### 1.2 Responsibilities (from Specification)

**Functional Requirements Addressed:**

- **FR-2.1.1**: Python Input Acceptance
  - Accept single Python script files (.py) in Script Mode
  - Accept Python package directories with setup.py/pyproject.toml in Library Mode
  - Support Python 3.8+ syntax and semantics
  - Handle relative and absolute imports
  - Detect and catalog external dependencies

- **FR-2.1.2**: Source Code Validation
  - Validate Python syntax before processing
  - Identify and report syntax errors with line numbers
  - Detect circular dependencies within the input codebase
  - Warn about unsupported Python features

- **FR-2.1.3**: Configuration Input
  - Accept configuration specifying target mode (Script/Library)
  - Accept optional optimization level flags
  - Accept optional test generation strategies
  - Accept optional GPU acceleration preferences

### 1.3 Design Principles

1. **Fail Fast**: Validate all inputs immediately; do not proceed with invalid inputs
2. **Comprehensive Validation**: Detect errors early to prevent downstream failures
3. **Separation of Concerns**: Separate file I/O, parsing, validation, and dependency analysis
4. **Testability**: All components use dependency injection for easy mocking (London School TDD)
5. **Graceful Degradation**: Provide partial results where possible, with clear error reporting

### 1.4 Agent Boundaries

**Inputs:**
- File paths (script or package directory)
- Configuration object

**Outputs:**
- Validated Python source tree
- Dependency graph
- Module catalog
- Validation report

**NOT Responsible For:**
- AST analysis (Analysis Agent's responsibility)
- Type inference (Analysis Agent's responsibility)
- Translation decisions (Specification Generator's responsibility)

---

## 2. Data Structures

### 2.1 Core Enums

```pseudocode
// Defines the mode of operation
ENUM OperationMode:
    Script      // Single .py file
    Library     // Full package directory

// Defines optimization level for compilation
ENUM OptimizationLevel:
    Debug       // Debug builds with symbols
    Release     // Optimized production builds

// Defines test generation strategy
ENUM TestStrategy:
    Conformance     // Golden vector testing
    Property        // Property-based testing
    Both            // Both conformance and property

// Defines dependency type classification
ENUM DependencyType:
    StandardLibrary     // Python stdlib (e.g., os, sys, json)
    ThirdParty          // PyPI packages (e.g., numpy, requests)
    Internal            // Internal project modules
    Unknown             // Cannot determine (requires further analysis)

// Defines Python feature support level
ENUM FeatureSupportLevel:
    Supported           // Full translation support
    PartialSupport      // Limited/conditional support
    Unsupported         // No translation support
    Warning             // Supported but risky

// Defines severity of validation issues
ENUM Severity:
    Error       // Blocking issue - cannot proceed
    Warning     // Non-blocking but important
    Info        // Informational message
```

### 2.2 Configuration Structures

```pseudocode
STRUCT IngestConfiguration:
    // Required fields
    mode: OperationMode
    source_path: Path
    output_directory: Path

    // Optional fields (with defaults)
    optimization_level: OptimizationLevel = Debug
    test_strategy: TestStrategy = Both
    gpu_enabled: Boolean = true
    target_features: List<String> = ["wasm", "nim", "omniverse"]

    // Validation parameters
    python_version_min: String = "3.8"
    python_version_max: String = "3.12"
    max_dependency_depth: Integer = 10
    allow_dynamic_imports: Boolean = false
    strict_mode: Boolean = true  // Fail on warnings if true

    // Logging and reporting
    log_level: String = "INFO"
    verbose: Boolean = false
```

### 2.3 Python Source Structures

```pseudocode
STRUCT PythonModule:
    // Module identification
    module_name: String                      // e.g., "mypackage.submodule"
    file_path: Path                          // Absolute path to .py file
    relative_path: Path                      // Path relative to package root
    is_package: Boolean                      // True if __init__.py

    // Source content
    source_code: String                      // Raw Python source
    encoding: String = "utf-8"               // File encoding
    line_count: Integer                      // Number of lines

    // Syntax validation
    is_valid: Boolean                        // Syntax check passed
    syntax_errors: List<SyntaxError>         // Empty if valid

    // Imports (detected but not resolved yet)
    imports: List<ImportStatement>           // All import statements

    // Metadata
    docstring: Optional<String>              // Module-level docstring
    checksum: String                         // SHA-256 of source
    last_modified: Timestamp                 // File modification time


STRUCT ImportStatement:
    // Import information
    import_type: String                      // "import" or "from_import"
    module_path: String                      // e.g., "os.path" or "numpy"
    imported_names: List<String>             // e.g., ["join", "exists"] or ["*"]
    alias: Optional<String>                  // e.g., "np" in "import numpy as np"

    // Location information
    line_number: Integer                     // Line in source file
    is_relative: Boolean                     // True for relative imports (e.g., "from . import")
    relative_level: Integer                  // Depth of relative import (0 for absolute)

    // Analysis flags
    is_conditional: Boolean                  // True if inside if/try/except
    is_dynamic: Boolean                      // True if uses __import__ or importlib


STRUCT SyntaxError:
    file_path: Path                          // File where error occurred
    line_number: Integer                     // Line number (1-indexed)
    column_number: Integer                   // Column number (1-indexed)
    message: String                          // Error message
    context: String                          // Surrounding source code lines
    error_type: String                       // E.g., "SyntaxError", "IndentationError"
```

### 2.4 Dependency Structures

```pseudocode
STRUCT Dependency:
    // Dependency identification
    name: String                             // e.g., "numpy", "os", "mypackage.utils"
    dependency_type: DependencyType          // stdlib, third-party, internal, unknown

    // Version information (for third-party)
    version_spec: Optional<String>           // e.g., ">=1.20.0,<2.0.0"
    detected_version: Optional<String>       // Version found in environment (if any)

    // Usage information
    imported_by: List<String>                // List of module names that import this
    import_count: Integer                    // Number of import statements

    // Metadata
    is_optional: Boolean                     // True if in try/except import block
    support_level: FeatureSupportLevel       // Translation support level
    notes: List<String>                      // Additional information/warnings


STRUCT DependencyGraph:
    // Graph structure
    nodes: Map<String, DependencyNode>       // Key: module name
    edges: List<DependencyEdge>              // Import relationships

    // Analysis results
    circular_dependencies: List<CircularDependencyChain>
    dependency_layers: List<List<String>>    // Topologically sorted layers
    root_modules: List<String>               // Entry points (no incoming edges)
    leaf_modules: List<String>               // No outgoing edges

    // Statistics
    total_modules: Integer
    max_depth: Integer                       // Longest dependency chain
    average_fanout: Float                    // Average number of dependencies per module


STRUCT DependencyNode:
    module_name: String
    module: PythonModule                     // Reference to the module
    dependencies: List<String>               // Names of modules this depends on
    dependents: List<String>                 // Names of modules that depend on this
    depth: Integer                           // Distance from root (0 for roots)


STRUCT DependencyEdge:
    from_module: String                      // Importing module
    to_module: String                        // Imported module
    import_statement: ImportStatement        // The actual import
    is_circular: Boolean                     // True if part of circular chain


STRUCT CircularDependencyChain:
    modules: List<String>                    // Modules in the cycle (e.g., [A, B, C, A])
    edges: List<DependencyEdge>              // The import statements forming the cycle
    severity: Severity                       // Error or Warning
    can_break: Boolean                       // True if breakable (e.g., via lazy import)
    suggestion: String                       // How to resolve the cycle
```

### 2.5 Validation Structures

```pseudocode
STRUCT ValidationIssue:
    severity: Severity                       // Error, Warning, Info
    category: String                         // E.g., "syntax", "circular_dependency", "unsupported_feature"
    message: String                          // Human-readable description
    file_path: Optional<Path>                // File where issue occurred
    line_number: Optional<Integer>           // Line number if applicable
    context: Optional<String>                // Code snippet for context
    suggestion: Optional<String>             // Remediation suggestion
    blocking: Boolean                        // True if prevents pipeline from continuing


STRUCT UnsupportedFeature:
    feature_name: String                     // E.g., "metaclass", "exec", "dynamic_import"
    locations: List<FeatureLocation>         // Where the feature is used
    support_level: FeatureSupportLevel       // Supported, PartialSupport, Unsupported
    workaround: Optional<String>             // Suggested alternative approach


STRUCT FeatureLocation:
    file_path: Path
    line_number: Integer
    context: String                          // Code snippet showing usage
```

### 2.6 Output Structures

```pseudocode
STRUCT IngestResult:
    // Status
    success: Boolean                         // True if ingest completed successfully
    mode: OperationMode                      // Script or Library

    // Validated source tree
    modules: List<PythonModule>              // All Python modules found
    entry_point: Optional<PythonModule>      // Main module (for Script mode)
    package_name: Optional<String>           // Package name (for Library mode)
    package_version: Optional<String>        // Package version (for Library mode)

    // Dependency analysis
    dependency_graph: DependencyGraph        // Complete dependency graph
    external_dependencies: List<Dependency>  // All external dependencies

    // Validation results
    validation_issues: List<ValidationIssue> // All issues found
    unsupported_features: List<UnsupportedFeature>  // Unsupported Python features

    // Statistics
    total_files: Integer                     // Number of .py files processed
    total_lines: Integer                     // Total LOC across all files
    valid_modules: Integer                   // Modules that passed validation
    failed_modules: Integer                  // Modules with syntax errors

    // Metadata
    processing_time_ms: Integer              // Time taken for ingestion
    timestamp: Timestamp                     // When ingestion completed
    configuration: IngestConfiguration       // Configuration used


STRUCT IngestReport:
    result: IngestResult

    // Formatted report sections
    summary: String                          // Executive summary
    validation_section: String               // Detailed validation results
    dependency_section: String               // Dependency analysis
    warnings_section: String                 // All warnings
    errors_section: String                   // All errors

    // Export formats
    to_json(): String                        // JSON representation
    to_yaml(): String                        // YAML representation
    to_text(): String                        // Human-readable text report
```

---

## 3. Core Algorithms

### 3.1 Main Ingest Algorithm

```pseudocode
FUNCTION ingest(config: IngestConfiguration) -> Result<IngestResult, IngestError>:
    """
    Main entry point for the Ingest Agent.

    Design Notes:
    - Follows fail-fast principle: validates inputs before processing
    - Uses dependency injection for all external services (FileSystem, PythonParser, etc.)
    - Returns comprehensive result object with all findings
    - All errors are collected and reported together (no silent failures)

    TDD Note: This is the primary test point for acceptance tests.
    Mock all dependencies (FileSystem, PythonParser, DependencyAnalyzer).
    """

    // Step 1: Validate configuration
    validation_result = validate_configuration(config)
    IF validation_result IS Error:
        RETURN Error(validation_result)

    // Step 2: Initialize result structure
    result = IngestResult {
        success: false,
        mode: config.mode,
        modules: [],
        validation_issues: [],
        unsupported_features: [],
        timestamp: current_timestamp()
    }

    start_time = current_time_ms()

    TRY:
        // Step 3: Discover Python files
        LOG.info("Discovering Python files from: {}", config.source_path)
        discovered_files = discover_python_files(config.source_path, config.mode)

        IF discovered_files.is_empty():
            RETURN Error(NoPythonFilesFound(config.source_path))

        LOG.info("Found {} Python files", discovered_files.length)
        result.total_files = discovered_files.length

        // Step 4: Load and parse all Python files
        LOG.info("Loading and parsing Python files...")
        FOR file_path IN discovered_files:
            module_result = load_and_parse_module(file_path, config)

            MATCH module_result:
                CASE Success(module):
                    result.modules.append(module)
                    result.total_lines += module.line_count

                    IF module.is_valid:
                        result.valid_modules += 1
                    ELSE:
                        result.failed_modules += 1
                        // Add syntax errors to validation issues
                        FOR syntax_error IN module.syntax_errors:
                            issue = ValidationIssue {
                                severity: Severity.Error,
                                category: "syntax",
                                message: syntax_error.message,
                                file_path: syntax_error.file_path,
                                line_number: syntax_error.line_number,
                                context: syntax_error.context,
                                blocking: true
                            }
                            result.validation_issues.append(issue)

                CASE Error(error):
                    // File couldn't be loaded (I/O error, encoding error, etc.)
                    issue = ValidationIssue {
                        severity: Severity.Error,
                        category: "file_load",
                        message: error.message,
                        file_path: file_path,
                        blocking: true
                    }
                    result.validation_issues.append(issue)

        // Step 5: Early exit if syntax errors in strict mode
        IF config.strict_mode AND result.failed_modules > 0:
            LOG.error("{} modules failed syntax validation", result.failed_modules)
            result.success = false
            result.processing_time_ms = current_time_ms() - start_time
            RETURN Success(result)  // Return result with errors

        // Step 6: Detect entry point (Script mode) or package info (Library mode)
        IF config.mode == OperationMode.Script:
            result.entry_point = detect_entry_point_script(result.modules, config)
        ELSE:
            package_info = detect_package_info(config.source_path, result.modules)
            result.package_name = package_info.name
            result.package_version = package_info.version

        // Step 7: Build dependency graph
        LOG.info("Building dependency graph...")
        graph_result = build_dependency_graph(result.modules, config.source_path)

        MATCH graph_result:
            CASE Success(graph):
                result.dependency_graph = graph

                // Check for circular dependencies
                IF NOT graph.circular_dependencies.is_empty():
                    LOG.warning("Found {} circular dependency chains",
                               graph.circular_dependencies.length)

                    FOR cycle IN graph.circular_dependencies:
                        issue = ValidationIssue {
                            severity: Severity.Warning,  // Warning, not error (Python allows this)
                            category: "circular_dependency",
                            message: "Circular dependency detected: {}".format(
                                " -> ".join(cycle.modules)
                            ),
                            blocking: false,
                            suggestion: cycle.suggestion
                        }
                        result.validation_issues.append(issue)

            CASE Error(error):
                // Dependency analysis failed - non-blocking but important
                issue = ValidationIssue {
                    severity: Severity.Warning,
                    category: "dependency_analysis",
                    message: "Failed to build complete dependency graph: {}".format(error),
                    blocking: false
                }
                result.validation_issues.append(issue)

        // Step 8: Catalog external dependencies
        LOG.info("Cataloging external dependencies...")
        deps_result = catalog_external_dependencies(
            result.modules,
            result.dependency_graph,
            config
        )

        MATCH deps_result:
            CASE Success(dependencies):
                result.external_dependencies = dependencies

                // Check for unsupported dependencies
                FOR dep IN dependencies:
                    IF dep.support_level == FeatureSupportLevel.Unsupported:
                        issue = ValidationIssue {
                            severity: Severity.Error,
                            category: "unsupported_dependency",
                            message: "Unsupported dependency: {} ({})".format(
                                dep.name,
                                dep.dependency_type
                            ),
                            blocking: true,
                            suggestion: "This dependency cannot be translated to Rust/WASM"
                        }
                        result.validation_issues.append(issue)

                    ELSE IF dep.support_level == FeatureSupportLevel.PartialSupport:
                        issue = ValidationIssue {
                            severity: Severity.Warning,
                            category: "partial_dependency_support",
                            message: "Partial support for dependency: {}".format(dep.name),
                            blocking: false,
                            suggestion: "Some features of this dependency may not work"
                        }
                        result.validation_issues.append(issue)

            CASE Error(error):
                LOG.warning("Failed to catalog all dependencies: {}", error)

        // Step 9: Detect unsupported Python features
        LOG.info("Scanning for unsupported Python features...")
        unsupported = detect_unsupported_features(result.modules, config)
        result.unsupported_features = unsupported

        FOR feature IN unsupported:
            severity = MATCH feature.support_level:
                FeatureSupportLevel.Unsupported -> Severity.Error
                FeatureSupportLevel.PartialSupport -> Severity.Warning
                _ -> Severity.Info

            FOR location IN feature.locations:
                issue = ValidationIssue {
                    severity: severity,
                    category: "unsupported_feature",
                    message: "Unsupported feature '{}' at {}:{}".format(
                        feature.feature_name,
                        location.file_path,
                        location.line_number
                    ),
                    file_path: location.file_path,
                    line_number: location.line_number,
                    context: location.context,
                    blocking: (severity == Severity.Error),
                    suggestion: feature.workaround
                }
                result.validation_issues.append(issue)

        // Step 10: Determine overall success
        has_blocking_issues = result.validation_issues.any(|issue| issue.blocking)
        result.success = NOT has_blocking_issues

        // Step 11: Finalize result
        result.processing_time_ms = current_time_ms() - start_time

        LOG.info("Ingestion completed in {}ms", result.processing_time_ms)
        LOG.info("Success: {}, Modules: {}, Issues: {}",
                 result.success, result.modules.length, result.validation_issues.length)

        RETURN Success(result)

    CATCH error AS IngestError:
        // Unexpected error during ingestion
        LOG.error("Ingestion failed with error: {}", error)
        result.success = false
        result.processing_time_ms = current_time_ms() - start_time
        RETURN Error(error)
```

### 3.2 Configuration Validation

```pseudocode
FUNCTION validate_configuration(config: IngestConfiguration) -> Result<Void, ValidationError>:
    """
    Validates the ingest configuration before processing.

    Design Notes:
    - Checks all required fields are present and valid
    - Validates file paths exist and are accessible
    - Ensures mode-specific requirements are met
    - Returns first error encountered (fail-fast)

    TDD Note: Unit test with various invalid configurations.
    No external dependencies to mock.
    """

    errors = []

    // Validate source path
    IF NOT exists(config.source_path):
        errors.append("Source path does not exist: {}".format(config.source_path))

    ELSE IF config.mode == OperationMode.Script:
        // Script mode: source must be a .py file
        IF NOT is_file(config.source_path):
            errors.append("Script mode requires a file, got directory: {}".format(
                config.source_path
            ))
        ELSE IF NOT config.source_path.ends_with(".py"):
            errors.append("Script mode requires .py file, got: {}".format(
                config.source_path
            ))

    ELSE IF config.mode == OperationMode.Library:
        // Library mode: source must be a directory
        IF NOT is_directory(config.source_path):
            errors.append("Library mode requires a directory, got file: {}".format(
                config.source_path
            ))

    // Validate output directory
    IF exists(config.output_directory):
        IF NOT is_directory(config.output_directory):
            errors.append("Output directory exists but is not a directory: {}".format(
                config.output_directory
            ))
        IF NOT is_writable(config.output_directory):
            errors.append("Output directory is not writable: {}".format(
                config.output_directory
            ))
    ELSE:
        // Try to create output directory
        TRY:
            create_directory(config.output_directory)
        CATCH error:
            errors.append("Cannot create output directory: {}".format(error))

    // Validate Python version constraints
    TRY:
        min_version = parse_version(config.python_version_min)
        max_version = parse_version(config.python_version_max)

        IF min_version >= max_version:
            errors.append("Invalid Python version range: {} >= {}".format(
                config.python_version_min,
                config.python_version_max
            ))
    CATCH error:
        errors.append("Invalid Python version format: {}".format(error))

    // Validate max_dependency_depth
    IF config.max_dependency_depth < 1 OR config.max_dependency_depth > 100:
        errors.append("max_dependency_depth must be between 1 and 100, got: {}".format(
            config.max_dependency_depth
        ))

    // Validate target_features
    valid_features = ["wasm", "nim", "omniverse"]
    FOR feature IN config.target_features:
        IF feature NOT IN valid_features:
            errors.append("Unknown target feature: {} (valid: {})".format(
                feature,
                ", ".join(valid_features)
            ))

    // Return result
    IF NOT errors.is_empty():
        RETURN Error(ValidationError(errors))

    RETURN Success(Void)
```

### 3.3 Python File Discovery

```pseudocode
FUNCTION discover_python_files(source_path: Path, mode: OperationMode) -> List<Path>:
    """
    Discovers all Python files in the input source.

    Design Notes:
    - Script mode: returns single file
    - Library mode: recursively finds all .py files
    - Excludes common non-source directories (venv, __pycache__, .git, etc.)
    - Follows symlinks (with cycle detection)

    TDD Note: Mock filesystem operations (FileSystem interface).
    Test with various directory structures.
    """

    IF mode == OperationMode.Script:
        // Simple case: single file
        RETURN [source_path]

    // Library mode: recursive discovery
    files = []
    excluded_dirs = [
        "__pycache__",
        ".git",
        ".svn",
        "venv",
        "env",
        ".env",
        "node_modules",
        "build",
        "dist",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        "htmlcov",
        ".eggs"
    ]

    visited_inodes = Set()  // For symlink cycle detection

    FUNCTION recurse(directory: Path):
        """Helper function for recursive traversal"""

        // Check for symlink cycles
        inode = get_inode(directory)
        IF inode IN visited_inodes:
            LOG.warning("Symlink cycle detected at: {}", directory)
            RETURN
        visited_inodes.add(inode)

        TRY:
            entries = list_directory(directory)

            FOR entry IN entries:
                full_path = directory.join(entry)

                IF is_directory(full_path):
                    // Skip excluded directories
                    IF entry NOT IN excluded_dirs:
                        recurse(full_path)

                ELSE IF is_file(full_path) AND entry.ends_with(".py"):
                    files.append(full_path)

        CATCH error AS FileSystemError:
            LOG.warning("Failed to read directory {}: {}", directory, error)

    recurse(source_path)

    // Sort files for deterministic ordering
    files.sort()

    RETURN files
```

### 3.4 Module Loading and Parsing

```pseudocode
FUNCTION load_and_parse_module(file_path: Path, config: IngestConfiguration)
    -> Result<PythonModule, LoadError>:
    """
    Loads a Python file and parses it into a PythonModule structure.

    Design Notes:
    - Handles encoding detection and conversion
    - Parses Python source to detect syntax errors
    - Extracts imports and docstrings
    - Does NOT build full AST (that's Analysis Agent's job)

    TDD Note: Mock FileSystem and PythonParser.
    Test with valid/invalid Python, various encodings, import styles.
    """

    // Step 1: Read file contents
    TRY:
        source_code = read_file(file_path)
    CATCH error AS FileSystemError:
        RETURN Error(LoadError("Failed to read file: {}".format(error)))

    // Step 2: Detect encoding (look for encoding declaration in first 2 lines)
    encoding = detect_encoding(source_code)

    // Step 3: Compute checksum
    checksum = sha256(source_code)

    // Step 4: Get file metadata
    metadata = get_file_metadata(file_path)
    last_modified = metadata.modified_time

    // Step 5: Parse Python syntax
    parse_result = parse_python_syntax(source_code, file_path)

    syntax_errors = []
    is_valid = true

    MATCH parse_result:
        CASE Success(parse_tree):
            // Syntax is valid - extract information
            imports = extract_imports(parse_tree)
            docstring = extract_module_docstring(parse_tree)

        CASE Error(errors):
            // Syntax errors found
            is_valid = false
            syntax_errors = errors
            imports = []  // Cannot extract imports from invalid syntax
            docstring = None

    // Step 6: Determine module name and package status
    // For now, we don't compute this (will be done when building dep graph)
    // Just use filename as placeholder
    module_name = file_path.filename_without_extension()
    is_package = (file_path.filename() == "__init__.py")

    // Step 7: Count lines
    line_count = source_code.split("\n").length

    // Step 8: Construct module structure
    module = PythonModule {
        module_name: module_name,
        file_path: file_path,
        relative_path: file_path,  // Will be updated later
        is_package: is_package,
        source_code: source_code,
        encoding: encoding,
        line_count: line_count,
        is_valid: is_valid,
        syntax_errors: syntax_errors,
        imports: imports,
        docstring: docstring,
        checksum: checksum,
        last_modified: last_modified
    }

    RETURN Success(module)


FUNCTION detect_encoding(source: String) -> String:
    """
    Detects Python source encoding from encoding declaration or BOM.

    Design Notes:
    - Checks for BOM markers (UTF-8, UTF-16, UTF-32)
    - Looks for coding declaration in first 2 lines
    - Defaults to UTF-8 if no declaration found

    Reference: PEP 263
    """

    // Check for BOM
    IF source.starts_with("\ufeff"):
        RETURN "utf-8-sig"
    IF source.starts_with("\xff\xfe"):
        RETURN "utf-16-le"
    IF source.starts_with("\xfe\xff"):
        RETURN "utf-16-be"

    // Check first two lines for encoding declaration
    lines = source.split("\n")
    FOR i IN range(0, min(2, lines.length)):
        line = lines[i]

        // Pattern: # -*- coding: <encoding> -*-
        // or: # coding: <encoding>
        // or: # coding=<encoding>
        match = regex_match(line, r"coding[=:]\s*([\w-]+)")
        IF match IS NOT None:
            RETURN match.group(1)

    // Default to UTF-8
    RETURN "utf-8"


FUNCTION extract_imports(parse_tree: ParseTree) -> List<ImportStatement>:
    """
    Extracts all import statements from a parsed Python module.

    Design Notes:
    - Handles both 'import' and 'from...import' statements
    - Detects relative imports (from . import foo)
    - Identifies conditional imports (inside if/try blocks)
    - Detects dynamic imports (__import__, importlib)

    TDD Note: Test with various import styles, relative imports, star imports.
    """

    imports = []

    FOR node IN parse_tree.walk():
        MATCH node.type:
            CASE "import_statement":
                // Pattern: import foo, bar.baz as qux
                FOR module_spec IN node.modules:
                    import_stmt = ImportStatement {
                        import_type: "import",
                        module_path: module_spec.name,
                        imported_names: [],  // Empty for 'import' style
                        alias: module_spec.alias,
                        line_number: node.line_number,
                        is_relative: false,
                        relative_level: 0,
                        is_conditional: is_inside_conditional(node),
                        is_dynamic: false
                    }
                    imports.append(import_stmt)

            CASE "from_import_statement":
                // Pattern: from foo.bar import baz, qux as quux
                module_path = node.module_path
                relative_level = count_leading_dots(module_path)
                is_relative = (relative_level > 0)

                // Remove leading dots from module path
                IF is_relative:
                    module_path = module_path.lstrip(".")

                // Extract imported names
                imported_names = []
                FOR name_spec IN node.names:
                    imported_names.append(name_spec.name)

                import_stmt = ImportStatement {
                    import_type: "from_import",
                    module_path: module_path,
                    imported_names: imported_names,
                    alias: None,  // Aliases are per-name in from imports
                    line_number: node.line_number,
                    is_relative: is_relative,
                    relative_level: relative_level,
                    is_conditional: is_inside_conditional(node),
                    is_dynamic: false
                }
                imports.append(import_stmt)

            CASE "call_expression":
                // Check for dynamic imports: __import__("module")
                IF node.function_name == "__import__":
                    IF node.arguments.length > 0 AND node.arguments[0].is_string_literal:
                        module_name = node.arguments[0].value
                        import_stmt = ImportStatement {
                            import_type: "import",
                            module_path: module_name,
                            imported_names: [],
                            alias: None,
                            line_number: node.line_number,
                            is_relative: false,
                            relative_level: 0,
                            is_conditional: true,  // Dynamic imports are always conditional
                            is_dynamic: true
                        }
                        imports.append(import_stmt)

                // Check for importlib.import_module("module")
                ELSE IF node.function_name == "import_module":
                    IF node.arguments.length > 0 AND node.arguments[0].is_string_literal:
                        module_name = node.arguments[0].value
                        import_stmt = ImportStatement {
                            import_type: "import",
                            module_path: module_name,
                            imported_names: [],
                            alias: None,
                            line_number: node.line_number,
                            is_relative: false,
                            relative_level: 0,
                            is_conditional: true,
                            is_dynamic: true
                        }
                        imports.append(import_stmt)

    RETURN imports


FUNCTION is_inside_conditional(node: ParseNode) -> Boolean:
    """
    Checks if a node is inside a conditional block (if/try/except).
    """
    current = node.parent
    WHILE current IS NOT None:
        IF current.type IN ["if_statement", "try_statement", "except_clause"]:
            RETURN true
        current = current.parent
    RETURN false
```

### 3.5 Dependency Graph Construction

```pseudocode
FUNCTION build_dependency_graph(modules: List<PythonModule>, package_root: Path)
    -> Result<DependencyGraph, GraphError>:
    """
    Builds a complete dependency graph from the parsed modules.

    Design Notes:
    - Creates nodes for all internal modules
    - Resolves relative imports using package structure
    - Identifies external dependencies (excluded from internal graph)
    - Detects circular dependencies using cycle detection algorithm
    - Performs topological sort to create dependency layers

    TDD Note: Mock module resolution.
    Test with various import patterns, circular deps, relative imports.
    """

    graph = DependencyGraph {
        nodes: {},
        edges: [],
        circular_dependencies: [],
        dependency_layers: [],
        root_modules: [],
        leaf_modules: [],
        total_modules: modules.length,
        max_depth: 0,
        average_fanout: 0.0
    }

    // Step 1: Create a map of module names to modules
    module_map = {}
    FOR module IN modules:
        // Compute full module name based on file path relative to package root
        full_name = compute_module_name(module.file_path, package_root)
        module.module_name = full_name
        module.relative_path = module.file_path.relative_to(package_root)
        module_map[full_name] = module

    // Step 2: Create nodes for all modules
    FOR module_name, module IN module_map.items():
        node = DependencyNode {
            module_name: module_name,
            module: module,
            dependencies: [],
            dependents: [],
            depth: 0  // Will be computed later
        }
        graph.nodes[module_name] = node

    // Step 3: Resolve imports and create edges
    FOR module_name, node IN graph.nodes.items():
        module = node.module

        FOR import_stmt IN module.imports:
            // Resolve import to full module name
            resolved = resolve_import(
                import_stmt,
                module_name,
                module_map,
                package_root
            )

            MATCH resolved:
                CASE Success(target_module_name):
                    // Check if target is an internal module
                    IF target_module_name IN graph.nodes:
                        // Internal dependency
                        node.dependencies.append(target_module_name)
                        graph.nodes[target_module_name].dependents.append(module_name)

                        edge = DependencyEdge {
                            from_module: module_name,
                            to_module: target_module_name,
                            import_statement: import_stmt,
                            is_circular: false  // Will be updated in step 4
                        }
                        graph.edges.append(edge)

                    // Else: external dependency (handled in catalog_external_dependencies)

                CASE Error(error):
                    LOG.warning("Failed to resolve import '{}' in module '{}': {}",
                               import_stmt.module_path,
                               module_name,
                               error)

    // Step 4: Detect circular dependencies
    cycles = detect_cycles(graph)
    graph.circular_dependencies = cycles

    // Mark edges that are part of cycles
    cycle_edges = Set()
    FOR cycle IN cycles:
        FOR edge IN cycle.edges:
            cycle_edges.add((edge.from_module, edge.to_module))

    FOR edge IN graph.edges:
        IF (edge.from_module, edge.to_module) IN cycle_edges:
            edge.is_circular = true

    // Step 5: Perform topological sort (for acyclic portions)
    IF cycles.is_empty():
        layers = topological_sort_layers(graph)
        graph.dependency_layers = layers

        // Compute depths
        FOR layer_idx, layer IN enumerate(layers):
            FOR module_name IN layer:
                graph.nodes[module_name].depth = layer_idx

        graph.max_depth = layers.length - 1
    ELSE:
        // With cycles, we can't do pure topological sort
        // But we can still compute approximate depths ignoring cycle edges
        graph.dependency_layers = approximate_layers(graph, cycle_edges)
        graph.max_depth = graph.dependency_layers.length - 1

    // Step 6: Identify root and leaf modules
    FOR module_name, node IN graph.nodes.items():
        IF node.dependents.is_empty():
            graph.root_modules.append(module_name)
        IF node.dependencies.is_empty():
            graph.leaf_modules.append(module_name)

    // Step 7: Compute average fanout
    total_fanout = sum(node.dependencies.length FOR node IN graph.nodes.values())
    graph.average_fanout = total_fanout / graph.nodes.length IF graph.nodes.length > 0 ELSE 0.0

    RETURN Success(graph)


FUNCTION compute_module_name(file_path: Path, package_root: Path) -> String:
    """
    Computes the full Python module name from a file path.

    Examples:
    - package_root/foo/bar.py -> foo.bar
    - package_root/foo/__init__.py -> foo
    - package_root/main.py -> main
    """

    relative = file_path.relative_to(package_root)

    // Remove .py extension
    relative = relative.with_suffix("")

    // Convert path separators to dots
    parts = relative.parts

    // If last part is __init__, remove it (package name is parent dir)
    IF parts[-1] == "__init__":
        parts = parts[:-1]

    module_name = ".".join(parts)

    RETURN module_name


FUNCTION resolve_import(
    import_stmt: ImportStatement,
    importer_module: String,
    module_map: Map<String, PythonModule>,
    package_root: Path
) -> Result<String, ResolveError>:
    """
    Resolves an import statement to a full module name.

    Design Notes:
    - Handles absolute imports (import os, import foo.bar)
    - Handles relative imports (from . import foo, from ..bar import baz)
    - Returns external module names unchanged (for dependency cataloging)

    TDD Note: Test with various import patterns, edge cases.
    """

    module_path = import_stmt.module_path

    // Handle relative imports
    IF import_stmt.is_relative:
        // Compute base module by going up the package hierarchy
        importer_parts = importer_module.split(".")

        // Relative level determines how many levels up to go
        level = import_stmt.relative_level

        IF level > importer_parts.length:
            RETURN Error("Relative import goes beyond package root")

        // Go up 'level' levels (level=1 is current package, level=2 is parent, etc.)
        base_parts = importer_parts[:-(level - 1)] IF level > 1 ELSE importer_parts

        // Append the module path if present (may be empty for "from . import foo")
        IF NOT module_path.is_empty():
            base_parts.append(module_path)

        resolved_name = ".".join(base_parts)
        RETURN Success(resolved_name)

    // Absolute import - return as-is
    RETURN Success(module_path)


FUNCTION detect_cycles(graph: DependencyGraph) -> List<CircularDependencyChain>:
    """
    Detects all circular dependency chains in the graph.

    Design Notes:
    - Uses Tarjan's algorithm or similar to find strongly connected components
    - Each SCC with >1 node represents a circular dependency
    - Provides suggestions for breaking cycles (lazy imports, restructuring)

    Algorithm: DFS-based cycle detection with backtracking
    """

    cycles = []
    visited = Set()
    in_progress = Set()
    path_stack = []

    FUNCTION dfs(module_name: String):
        IF module_name IN visited:
            RETURN

        IF module_name IN in_progress:
            // Found a cycle - extract it from path_stack
            cycle_start_idx = path_stack.index_of(module_name)
            cycle_modules = path_stack[cycle_start_idx:] + [module_name]

            // Build list of edges in cycle
            cycle_edges = []
            FOR i IN range(0, cycle_modules.length - 1):
                from_mod = cycle_modules[i]
                to_mod = cycle_modules[i + 1]

                // Find the edge
                FOR edge IN graph.edges:
                    IF edge.from_module == from_mod AND edge.to_module == to_mod:
                        cycle_edges.append(edge)
                        BREAK

            // Determine if cycle can be broken
            can_break = any(edge.import_statement.is_conditional FOR edge IN cycle_edges)

            // Generate suggestion
            suggestion = IF can_break:
                "Consider moving conditional imports to function level (lazy import)"
            ELSE:
                "Restructure modules to eliminate circular dependency"

            cycle = CircularDependencyChain {
                modules: cycle_modules,
                edges: cycle_edges,
                severity: Severity.Warning,
                can_break: can_break,
                suggestion: suggestion
            }
            cycles.append(cycle)
            RETURN

        in_progress.add(module_name)
        path_stack.append(module_name)

        // Visit dependencies
        node = graph.nodes[module_name]
        FOR dep IN node.dependencies:
            dfs(dep)

        path_stack.pop()
        in_progress.remove(module_name)
        visited.add(module_name)

    // Run DFS from each node
    FOR module_name IN graph.nodes.keys():
        IF module_name NOT IN visited:
            dfs(module_name)

    RETURN cycles


FUNCTION topological_sort_layers(graph: DependencyGraph) -> List<List<String>>:
    """
    Performs topological sort and groups modules into layers.

    Design Notes:
    - Layer 0: modules with no dependencies (leaf nodes)
    - Layer N: modules whose dependencies are all in layers < N
    - Returns list of layers, where each layer can be processed in parallel

    Algorithm: Kahn's algorithm for topological sorting
    """

    // Compute in-degree for each node
    in_degree = {}
    FOR module_name IN graph.nodes.keys():
        in_degree[module_name] = graph.nodes[module_name].dependencies.length

    layers = []
    remaining = Set(graph.nodes.keys())

    WHILE NOT remaining.is_empty():
        // Find all nodes with in-degree 0
        current_layer = [name FOR name IN remaining IF in_degree[name] == 0]

        IF current_layer.is_empty():
            // Should never happen if graph is acyclic
            BREAK

        layers.append(current_layer)

        // Remove current layer nodes and update in-degrees
        FOR name IN current_layer:
            remaining.remove(name)

            // Decrease in-degree of dependents
            FOR dependent IN graph.nodes[name].dependents:
                IF dependent IN remaining:
                    in_degree[dependent] -= 1

    RETURN layers
```

### 3.6 External Dependency Cataloging

```pseudocode
FUNCTION catalog_external_dependencies(
    modules: List<PythonModule>,
    dependency_graph: DependencyGraph,
    config: IngestConfiguration
) -> Result<List<Dependency>, CatalogError>:
    """
    Catalogs all external dependencies (stdlib and third-party).

    Design Notes:
    - Identifies stdlib modules (using known stdlib list)
    - Identifies third-party modules (anything not stdlib or internal)
    - Attempts to determine versions from installed packages
    - Checks support level for translation (maintained allowlist/blocklist)
    - Aggregates import counts and locations

    TDD Note: Mock PackageInspector (for detecting installed packages).
    Test with various dependency types.
    """

    dependencies = {}  // Map: module_name -> Dependency

    // Step 1: Collect all imported module names
    all_imports = Set()
    internal_modules = Set(dependency_graph.nodes.keys())

    FOR module IN modules:
        FOR import_stmt IN module.imports:
            // Get top-level module name (e.g., "os.path" -> "os")
            top_level = import_stmt.module_path.split(".")[0]

            // Skip internal modules
            IF top_level NOT IN internal_modules:
                all_imports.add(top_level)

    // Step 2: Classify each external dependency
    FOR module_name IN all_imports:
        dep_type = classify_dependency(module_name)
        support_level = check_support_level(module_name, dep_type)

        // Try to get version information (for third-party packages)
        detected_version = None
        IF dep_type == DependencyType.ThirdParty:
            detected_version = get_installed_version(module_name)

        // Find all modules that import this dependency
        imported_by = []
        import_count = 0
        is_optional = false

        FOR module IN modules:
            module_imports_dep = false

            FOR import_stmt IN module.imports:
                top_level = import_stmt.module_path.split(".")[0]

                IF top_level == module_name:
                    module_imports_dep = true
                    import_count += 1

                    // Check if any import is conditional
                    IF import_stmt.is_conditional:
                        is_optional = true

            IF module_imports_dep:
                imported_by.append(module.module_name)

        // Get support notes
        notes = get_support_notes(module_name, dep_type, support_level)

        // Create dependency entry
        dependency = Dependency {
            name: module_name,
            dependency_type: dep_type,
            version_spec: None,  // Would need to parse requirements.txt/pyproject.toml
            detected_version: detected_version,
            imported_by: imported_by,
            import_count: import_count,
            is_optional: is_optional,
            support_level: support_level,
            notes: notes
        }

        dependencies[module_name] = dependency

    // Step 3: Try to extract version constraints from package metadata
    IF config.mode == OperationMode.Library:
        version_specs = extract_version_specs(config.source_path)

        FOR module_name, version_spec IN version_specs.items():
            IF module_name IN dependencies:
                dependencies[module_name].version_spec = version_spec

    RETURN Success(list(dependencies.values()))


FUNCTION classify_dependency(module_name: String) -> DependencyType:
    """
    Classifies a module as stdlib, third-party, or unknown.

    Design Notes:
    - Uses hardcoded list of stdlib modules (per Python version)
    - Anything not in stdlib is assumed to be third-party
    - Special handling for builtin modules (sys, os, etc.)
    """

    // Known stdlib modules (Python 3.8+)
    stdlib_modules = [
        // Builtins and core modules
        "sys", "os", "io", "time", "re", "math", "random", "collections",
        "itertools", "functools", "operator", "copy", "pickle", "json",

        // Common stdlib modules
        "pathlib", "datetime", "argparse", "logging", "typing", "enum",
        "dataclasses", "abc", "contextlib", "warnings", "weakref",

        // Data structures and algorithms
        "heapq", "bisect", "array", "queue", "sched", "struct", "codecs",

        // String and text processing
        "string", "textwrap", "unicodedata", "stringprep", "difflib",

        // File and directory access
        "tempfile", "glob", "fnmatch", "shutil", "zipfile", "tarfile",

        // Data persistence
        "sqlite3", "dbm", "shelve", "csv",

        // Networking
        "socket", "ssl", "select", "selectors", "asyncio", "email",
        "http", "urllib", "ftplib", "smtplib", "imaplib",

        // Concurrency
        "threading", "multiprocessing", "subprocess", "concurrent",

        // Crypto and hashing
        "hashlib", "hmac", "secrets",

        // Compression
        "zlib", "gzip", "bz2", "lzma",

        // Testing and development
        "unittest", "doctest", "pdb", "trace", "traceback", "inspect",

        // OS and system
        "platform", "errno", "ctypes", "fcntl", "resource", "signal",

        // Others
        "ast", "dis", "importlib", "pkgutil", "runpy", "timeit", "uuid"
    ]

    IF module_name IN stdlib_modules:
        RETURN DependencyType.StandardLibrary

    // Could enhance with actual package inspection
    RETURN DependencyType.ThirdParty


FUNCTION check_support_level(module_name: String, dep_type: DependencyType)
    -> FeatureSupportLevel:
    """
    Checks the support level for translating a dependency.

    Design Notes:
    - Maintains allowlist of fully supported dependencies
    - Maintains blocklist of unsupported dependencies
    - Defaults to PartialSupport for unknown third-party packages
    - Most stdlib modules are Supported (with exceptions)
    """

    // Unsupported dependencies (cannot translate to WASM)
    unsupported = [
        // GUI frameworks
        "tkinter", "wx", "pyqt5", "pyside2", "kivy",

        // C extension heavy packages (without Rust equivalents)
        "numpy",  // Would need special handling
        "scipy",
        "pandas",
        "matplotlib",

        // System-specific
        "winreg", "msvcrt", "fcntl", "termios",

        // Multi-threading (WASM limitation)
        "multiprocessing",  // Not supported in WASM

        // Others
        "ctypes",  // Direct C FFI not available in WASM
    ]

    // Partially supported (subset of functionality)
    partial_support = [
        "socket",      // Limited WASI socket support
        "ssl",         // Limited crypto support
        "subprocess",  // No process spawning in WASM
        "threading",   // Limited thread support in WASM
    ]

    // Fully supported stdlib modules (non-exhaustive)
    supported = [
        "re", "json", "math", "random", "collections", "itertools",
        "functools", "operator", "copy", "datetime", "enum", "dataclasses",
        "typing", "abc", "string", "textwrap", "heapq", "bisect",
        "hashlib", "hmac", "base64", "uuid", "pathlib"
    ]

    IF module_name IN unsupported:
        RETURN FeatureSupportLevel.Unsupported

    IF module_name IN partial_support:
        RETURN FeatureSupportLevel.PartialSupport

    IF module_name IN supported:
        RETURN FeatureSupportLevel.Supported

    IF dep_type == DependencyType.StandardLibrary:
        // Unknown stdlib module - default to partial support
        RETURN FeatureSupportLevel.PartialSupport

    // Third-party packages - default to unsupported (conservative)
    RETURN FeatureSupportLevel.Unsupported


FUNCTION get_support_notes(
    module_name: String,
    dep_type: DependencyType,
    support_level: FeatureSupportLevel
) -> List<String>:
    """
    Returns human-readable notes about dependency support.
    """

    notes = []

    IF support_level == FeatureSupportLevel.Unsupported:
        notes.append("This dependency cannot be translated to Rust/WASM")
        notes.append("Consider finding a Rust alternative or implementing core functionality")

    ELSE IF support_level == FeatureSupportLevel.PartialSupport:
        notes.append("Only a subset of this module's functionality is supported")
        notes.append("Manual verification of translated code is recommended")

    // Module-specific notes
    MATCH module_name:
        "multiprocessing":
            notes.append("WASM does not support multi-processing")
            notes.append("Consider using threading or async patterns instead")

        "socket":
            notes.append("WASI provides limited socket support")
            notes.append("Only basic TCP/UDP operations may work")

        "numpy":
            notes.append("Consider using ndarray or nalgebra crates in Rust")

        "subprocess":
            notes.append("WASM sandbox prevents spawning processes")
            notes.append("Rearchitect to avoid subprocess usage")

    RETURN notes


FUNCTION extract_version_specs(package_root: Path) -> Map<String, String>:
    """
    Extracts dependency version specifications from package metadata.

    Design Notes:
    - Looks for requirements.txt, setup.py, pyproject.toml
    - Parses dependency specifications
    - Returns map of package name to version spec

    TDD Note: Mock file reading and parsing.
    """

    version_specs = {}

    // Try requirements.txt
    requirements_path = package_root.join("requirements.txt")
    IF exists(requirements_path):
        content = read_file(requirements_path)
        FOR line IN content.split("\n"):
            line = line.strip()

            // Skip comments and empty lines
            IF line.is_empty() OR line.starts_with("#"):
                CONTINUE

            // Parse requirement (e.g., "numpy>=1.20.0,<2.0.0")
            parsed = parse_requirement(line)
            IF parsed IS NOT None:
                version_specs[parsed.name] = parsed.version_spec

    // Try pyproject.toml (PEP 621)
    pyproject_path = package_root.join("pyproject.toml")
    IF exists(pyproject_path):
        content = read_file(pyproject_path)
        parsed_toml = parse_toml(content)

        IF "project" IN parsed_toml AND "dependencies" IN parsed_toml["project"]:
            FOR dep IN parsed_toml["project"]["dependencies"]:
                parsed = parse_requirement(dep)
                IF parsed IS NOT None:
                    version_specs[parsed.name] = parsed.version_spec

    // Note: setup.py parsing is more complex (requires AST analysis)
    // Skipping for now - can be added later

    RETURN version_specs
```

### 3.7 Unsupported Feature Detection

```pseudocode
FUNCTION detect_unsupported_features(
    modules: List<PythonModule>,
    config: IngestConfiguration
) -> List<UnsupportedFeature>:
    """
    Scans Python code for features that are unsupported or partially supported.

    Design Notes:
    - Uses pattern matching on source code and/or AST
    - Checks for: metaclasses, exec/eval, dynamic imports, reflection, etc.
    - Groups findings by feature type with all locations
    - Provides workarounds where applicable

    TDD Note: Mock source code scanner.
    Test with code samples containing various unsupported features.
    """

    features = {}  // Map: feature_name -> UnsupportedFeature

    FOR module IN modules:
        IF NOT module.is_valid:
            // Skip modules with syntax errors
            CONTINUE

        // Pattern 1: exec/eval
        exec_locations = find_pattern(module, r"\b(exec|eval)\s*\(")
        IF NOT exec_locations.is_empty():
            add_or_update_feature(
                features,
                "exec_eval",
                FeatureSupportLevel.Unsupported,
                exec_locations,
                "Avoid exec/eval - use explicit control flow instead"
            )

        // Pattern 2: metaclasses
        metaclass_locations = find_pattern(module, r"metaclass\s*=")
        IF NOT metaclass_locations.is_empty():
            add_or_update_feature(
                features,
                "metaclass",
                FeatureSupportLevel.Unsupported,
                metaclass_locations,
                "Refactor to use composition or procedural macros in Rust"
            )

        // Pattern 3: Dynamic attribute access (getattr, setattr, delattr with variables)
        dynamic_attr_locations = find_pattern(module, r"\b(getattr|setattr|delattr)\s*\([^,]+,\s*[^\"']")
        IF NOT dynamic_attr_locations.is_empty():
            add_or_update_feature(
                features,
                "dynamic_attributes",
                FeatureSupportLevel.PartialSupport,
                dynamic_attr_locations,
                "Limited support - prefer static attribute access"
            )

        // Pattern 4: __import__ and importlib.import_module
        FOR import_stmt IN module.imports:
            IF import_stmt.is_dynamic:
                location = FeatureLocation {
                    file_path: module.file_path,
                    line_number: import_stmt.line_number,
                    context: get_source_line(module, import_stmt.line_number)
                }
                add_or_update_feature(
                    features,
                    "dynamic_import",
                    FeatureSupportLevel.Unsupported,
                    [location],
                    "Use static imports - dynamic module loading not supported"
                )

        // Pattern 5: Global state modification
        global_locations = find_pattern(module, r"\bglobal\s+\w+")
        IF NOT global_locations.is_empty():
            add_or_update_feature(
                features,
                "global_state",
                FeatureSupportLevel.Warning,
                global_locations,
                "Global state can cause issues - consider using function parameters"
            )

        // Pattern 6: Function decorators (complex ones)
        // Note: Simple decorators like @property, @staticmethod are supported
        decorator_locations = find_complex_decorators(module)
        IF NOT decorator_locations.is_empty():
            add_or_update_feature(
                features,
                "complex_decorators",
                FeatureSupportLevel.PartialSupport,
                decorator_locations,
                "Simple decorators supported - complex/dynamic decorators may need refactoring"
            )

        // Pattern 7: Generator expressions in certain contexts
        // (This is a placeholder - actual detection would be more sophisticated)

        // Pattern 8: Context managers (__enter__/__exit__)
        context_mgr_locations = find_pattern(module, r"def\s+__(enter|exit)__")
        IF NOT context_mgr_locations.is_empty():
            add_or_update_feature(
                features,
                "context_managers",
                FeatureSupportLevel.PartialSupport,
                context_mgr_locations,
                "Basic context managers supported - complex ones may need adaptation"
            )

    RETURN list(features.values())


FUNCTION find_pattern(module: PythonModule, regex_pattern: String) -> List<FeatureLocation>:
    """
    Finds all occurrences of a regex pattern in module source.
    """

    locations = []
    lines = module.source_code.split("\n")

    FOR line_idx, line IN enumerate(lines):
        IF regex_match(line, regex_pattern):
            location = FeatureLocation {
                file_path: module.file_path,
                line_number: line_idx + 1,  // 1-indexed
                context: get_context_lines(lines, line_idx, before=1, after=1)
            }
            locations.append(location)

    RETURN locations


FUNCTION add_or_update_feature(
    features: Map<String, UnsupportedFeature>,
    feature_name: String,
    support_level: FeatureSupportLevel,
    locations: List<FeatureLocation>,
    workaround: String
):
    """
    Adds a new feature or updates existing one with additional locations.
    """

    IF feature_name NOT IN features:
        features[feature_name] = UnsupportedFeature {
            feature_name: feature_name,
            locations: [],
            support_level: support_level,
            workaround: workaround
        }

    features[feature_name].locations.extend(locations)
```

---

## 4. Input/Output Contracts

### 4.1 Agent Input Contract

```yaml
INPUT: IngestConfiguration
  REQUIRED:
    mode: OperationMode                    # Script or Library
    source_path: Path                      # File or directory to process
    output_directory: Path                 # Where to write results

  OPTIONAL:
    optimization_level: OptimizationLevel  # Default: Debug
    test_strategy: TestStrategy            # Default: Both
    gpu_enabled: Boolean                   # Default: true
    target_features: List<String>          # Default: ["wasm", "nim", "omniverse"]
    python_version_min: String             # Default: "3.8"
    python_version_max: String             # Default: "3.12"
    max_dependency_depth: Integer          # Default: 10
    allow_dynamic_imports: Boolean         # Default: false
    strict_mode: Boolean                   # Default: true
    log_level: String                      # Default: "INFO"
    verbose: Boolean                       # Default: false

PRECONDITIONS:
  - source_path must exist and be accessible
  - source_path must be a .py file (Script mode) or directory (Library mode)
  - output_directory must be writable (or creatable)
  - python_version_min < python_version_max
  - max_dependency_depth in range [1, 100]

VALIDATION:
  - All paths validated before processing
  - Configuration checked for internal consistency
  - Fail fast on invalid configuration
```

### 4.2 Agent Output Contract

```yaml
OUTPUT: IngestResult
  ALWAYS_PRESENT:
    success: Boolean                       # True if no blocking issues
    mode: OperationMode                    # Echo of input mode
    modules: List<PythonModule>            # All discovered modules (may be empty on error)
    validation_issues: List<ValidationIssue>  # All issues found
    unsupported_features: List<UnsupportedFeature>  # Unsupported Python features
    total_files: Integer                   # Number of .py files found
    total_lines: Integer                   # Total LOC
    valid_modules: Integer                 # Modules passing syntax check
    failed_modules: Integer                # Modules with syntax errors
    processing_time_ms: Integer            # Processing duration
    timestamp: Timestamp                   # Completion time
    configuration: IngestConfiguration     # Configuration used

  CONDITIONAL:
    entry_point: Optional<PythonModule>    # Present in Script mode (if valid)
    package_name: Optional<String>         # Present in Library mode
    package_version: Optional<String>      # Present in Library mode
    dependency_graph: DependencyGraph      # Present if graph construction succeeded
    external_dependencies: List<Dependency>  # Present if cataloging succeeded

POSTCONDITIONS:
  - success == true IFF no blocking validation issues
  - valid_modules + failed_modules == total_files
  - All syntax errors are included in validation_issues
  - All circular dependencies are reported in validation_issues
  - All unsupported dependencies are reported in validation_issues
  - dependency_graph.total_modules == modules.length (if present)

ERROR_CASES:
  - ConfigurationError: Invalid configuration (returned before processing)
  - NoFilesFoundError: No Python files found at source_path
  - FileSystemError: Cannot read files (permissions, I/O errors)
  - ParseError: Unexpected parser failures (should be caught as syntax errors)

SUCCESS_CRITERIA:
  - Script Mode: Single valid Python file loaded and parsed
  - Library Mode: At least one valid Python file in package
  - All imports extracted (even if not all resolved)
  - Dependency graph constructed (even if it contains cycles)
  - All validation issues collected and reported
```

### 4.3 Interface to Next Agent (Analysis Agent)

```yaml
HANDOFF_TO_ANALYSIS_AGENT:
  WHAT_IS_PASSED:
    - IngestResult (complete structure)
    - All validated PythonModule objects with source code
    - Complete DependencyGraph (if available)
    - List of external dependencies with classification

  WHAT_IS_NOT_PASSED:
    - Invalid/unparseable modules (these are filtered out)
    - Raw file system paths (converted to structured data)
    - Configuration details (unless needed by Analysis Agent)

  ASSUMPTIONS_FOR_NEXT_AGENT:
    - All modules have valid Python syntax (syntax_errors list is empty)
    - All imports are extracted (but not necessarily resolved)
    - Circular dependencies are identified (but not resolved)
    - External dependencies are cataloged (but not analyzed for API surface)

  ANALYSIS_AGENT_RESPONSIBILITIES:
    - Build full AST for each module
    - Extract API surface (functions, classes, methods)
    - Infer types where annotations are missing
    - Analyze data flow and side effects
    - Deep analysis of external dependency usage
```

---

## 5. Error Handling Strategy

### 5.1 Error Categories

```pseudocode
// Error hierarchy for Ingest Agent

ENUM IngestErrorCategory:
    Configuration       // Invalid configuration
    FileSystem          // File I/O errors
    Parsing             // Python syntax errors
    Validation          // Validation failures
    DependencyAnalysis  // Dependency graph errors
    Internal            // Unexpected internal errors


STRUCT IngestError:
    category: IngestErrorCategory
    message: String
    cause: Optional<Error>              // Underlying error (if any)
    context: Map<String, String>        // Additional context
    recoverable: Boolean                // Can processing continue?
    suggestion: Optional<String>        // Remediation suggestion


// Specific error types

STRUCT ConfigurationError EXTENDS IngestError:
    invalid_fields: List<String>        // Which fields are invalid

STRUCT FileSystemError EXTENDS IngestError:
    path: Path                          // Path that caused error
    operation: String                   // "read", "write", "stat", etc.

STRUCT ParseError EXTENDS IngestError:
    file_path: Path
    line_number: Integer
    column_number: Integer

STRUCT ValidationError EXTENDS IngestError:
    issues: List<ValidationIssue>       // All validation issues
```

### 5.2 Error Handling Principles

```pseudocode
PRINCIPLES:
  1. Fail Fast on Configuration Errors
     - Invalid configuration prevents all processing
     - User must fix configuration before retrying

  2. Collect and Continue for Validation Errors
     - Syntax errors in one file don't stop processing other files
     - All validation issues are collected and reported together
     - User can see all problems at once

  3. Graceful Degradation
     - If dependency graph construction fails, continue without it
     - If one module fails to parse, continue with others
     - Always produce a result (even if it indicates failure)

  4. Rich Error Context
     - Include file paths, line numbers, code context
     - Provide actionable suggestions for remediation
     - Link to relevant documentation

  5. Structured Error Reporting
     - Errors are machine-readable (JSON serializable)
     - Errors are human-readable (formatted text output)
     - Severity levels guide user response (Error vs Warning vs Info)


EXAMPLE_ERROR_HANDLING:

  // Example 1: Configuration error (fail fast)
  IF config.source_path does not exist:
    RETURN Error(ConfigurationError {
      category: Configuration,
      message: "Source path does not exist: {}".format(config.source_path),
      invalid_fields: ["source_path"],
      recoverable: false,
      suggestion: "Check that the path is correct and accessible"
    })

  // Example 2: Syntax error (collect and continue)
  MATCH parse_result:
    CASE Error(syntax_error):
      issue = ValidationIssue {
        severity: Error,
        category: "syntax",
        message: syntax_error.message,
        file_path: file_path,
        line_number: syntax_error.line,
        context: syntax_error.context,
        blocking: true,
        suggestion: "Fix Python syntax errors before translation"
      }
      result.validation_issues.append(issue)
      // Continue processing other files

  // Example 3: Dependency analysis failure (warn and continue)
  MATCH build_dependency_graph(modules):
    CASE Error(graph_error):
      LOG.warning("Dependency graph construction failed: {}", graph_error)
      issue = ValidationIssue {
        severity: Warning,
        category: "dependency_analysis",
        message: "Could not build complete dependency graph",
        blocking: false,
        suggestion: "Some import relationships may be missing"
      }
      result.validation_issues.append(issue)
      // Continue without full dependency graph
```

### 5.3 Logging Strategy

```pseudocode
LOGGING_LEVELS:
  ERROR:   Blocking issues that prevent success
  WARNING: Non-blocking issues that require attention
  INFO:    Progress updates and high-level status
  DEBUG:   Detailed execution flow (verbose mode only)

LOGGING_POINTS:
  - Agent start/end (INFO)
  - Each processing phase (INFO)
  - File discovery count (INFO)
  - Validation issues (WARNING/ERROR)
  - Dependency graph statistics (INFO)
  - Performance metrics (DEBUG)
  - Unexpected errors (ERROR)

STRUCTURED_LOGGING:
  // Use structured logging for machine-readable output
  LOG.info("phase", "discovery", "files_found", 42)
  LOG.error("error", "syntax_error", "file", "/path/to/file.py", "line", 15)

  // Format for human consumption
  INFO: [Ingest] Phase: discovery | Files found: 42
  ERROR: [Ingest] Syntax error in /path/to/file.py:15
```

---

## 6. London School TDD Test Points

### 6.1 Testing Strategy Overview

The Ingest Agent follows **London School TDD** (mockist, outside-in approach):

1. **Start with Acceptance Tests**: Test the `ingest()` function with mocked dependencies
2. **Work Inward**: Test each component in isolation with mocks
3. **Verify Behavior**: Focus on interactions (method calls) not state
4. **Mock External Dependencies**: FileSystem, PythonParser, etc.

### 6.2 Test Doubles (Mocks/Stubs)

```pseudocode
// Mock interfaces for external dependencies

INTERFACE FileSystemInterface:
    exists(path: Path) -> Boolean
    is_file(path: Path) -> Boolean
    is_directory(path: Path) -> Boolean
    is_writable(path: Path) -> Boolean
    read_file(path: Path) -> Result<String, Error>
    list_directory(path: Path) -> Result<List<String>, Error>
    get_file_metadata(path: Path) -> Result<FileMetadata, Error>
    create_directory(path: Path) -> Result<Void, Error>


INTERFACE PythonParserInterface:
    parse_syntax(source: String, filename: Path) -> Result<ParseTree, List<SyntaxError>>
    extract_imports(tree: ParseTree) -> List<ImportStatement>
    extract_docstring(tree: ParseTree) -> Optional<String>


INTERFACE PackageInspectorInterface:
    get_installed_version(package_name: String) -> Optional<String>
    parse_requirements_file(path: Path) -> Map<String, String>
    parse_pyproject_toml(path: Path) -> Map<String, Any>


// Example test doubles

CLASS MockFileSystem IMPLEMENTS FileSystemInterface:
    // Configurable mock that returns predefined responses
    files: Map<Path, String>  // Path -> content

    FUNCTION read_file(path: Path) -> Result<String, Error>:
        IF path IN self.files:
            RETURN Success(self.files[path])
        RETURN Error(FileNotFound(path))

    // ... other methods


CLASS StubPythonParser IMPLEMENTS PythonParserInterface:
    // Stub that returns fixed responses for testing

    FUNCTION parse_syntax(source: String, filename: Path) -> Result<ParseTree, List<SyntaxError>>:
        IF source.contains("syntax error"):
            RETURN Error([SyntaxError {...}])
        RETURN Success(ParseTree {...})
```

### 6.3 Test Hierarchy

```pseudocode
ACCEPTANCE_TESTS:
  // Test the main ingest() function with all dependencies mocked

  TEST "successful script mode ingestion":
    GIVEN:
      - MockFileSystem with valid Python script
      - StubPythonParser that returns valid parse tree
      - Configuration for script mode

    WHEN:
      - Call ingest(config)

    THEN:
      - Result.success == true
      - Result.modules.length == 1
      - Result.entry_point is set
      - No validation errors
      - FileSystem.read_file was called once
      - PythonParser.parse_syntax was called once


  TEST "library mode with circular dependency":
    GIVEN:
      - MockFileSystem with 3 Python files (A imports B, B imports C, C imports A)
      - StubPythonParser that returns valid parse trees
      - Configuration for library mode

    WHEN:
      - Call ingest(config)

    THEN:
      - Result.success == true (circular deps are warnings, not errors)
      - Result.modules.length == 3
      - Result.dependency_graph.circular_dependencies.length == 1
      - Circular dependency is reported in validation_issues with Warning severity


  TEST "syntax error handling":
    GIVEN:
      - MockFileSystem with invalid Python file
      - StubPythonParser that returns syntax errors
      - Configuration with strict_mode == true

    WHEN:
      - Call ingest(config)

    THEN:
      - Result.success == false
      - Result.failed_modules == 1
      - Result.validation_issues contains syntax error with Error severity
      - Error includes file path, line number, and context


COMPONENT_TESTS:
  // Test individual functions with mocked dependencies

  TEST "validate_configuration with invalid path":
    GIVEN:
      - Configuration with non-existent source_path
      - MockFileSystem.exists() returns false

    WHEN:
      - Call validate_configuration(config)

    THEN:
      - Returns Error(ValidationError)
      - Error message mentions source_path


  TEST "discover_python_files excludes __pycache__":
    GIVEN:
      - MockFileSystem with directory structure:
        /pkg/module.py
        /pkg/__pycache__/module.pyc
        /pkg/subpkg/other.py
      - Library mode

    WHEN:
      - Call discover_python_files("/pkg", Library)

    THEN:
      - Returns ["/pkg/module.py", "/pkg/subpkg/other.py"]
      - Does not include files in __pycache__


  TEST "build_dependency_graph detects cycle":
    GIVEN:
      - 3 PythonModule objects with imports forming a cycle
      - MockModuleResolver

    WHEN:
      - Call build_dependency_graph(modules)

    THEN:
      - Result.circular_dependencies.length == 1
      - Cycle includes all 3 modules
      - Suggestion for breaking cycle is provided


  TEST "catalog_external_dependencies classifies stdlib vs third-party":
    GIVEN:
      - Modules importing "os", "numpy", "requests"
      - StubPackageInspector

    WHEN:
      - Call catalog_external_dependencies(modules)

    THEN:
      - Returns 3 dependencies
      - "os" is classified as StandardLibrary
      - "numpy" and "requests" are classified as ThirdParty


  TEST "detect_unsupported_features finds exec usage":
    GIVEN:
      - Module with source code containing "exec(code)"

    WHEN:
      - Call detect_unsupported_features([module])

    THEN:
      - Returns 1 UnsupportedFeature
      - Feature name is "exec_eval"
      - Support level is Unsupported
      - Location includes correct line number


UNIT_TESTS:
  // Test pure functions with no dependencies

  TEST "compute_module_name converts path to module name":
    GIVEN:
      - file_path = "/pkg/foo/bar.py"
      - package_root = "/pkg"

    WHEN:
      - Call compute_module_name(file_path, package_root)

    THEN:
      - Returns "foo.bar"


  TEST "compute_module_name handles __init__.py":
    GIVEN:
      - file_path = "/pkg/foo/__init__.py"
      - package_root = "/pkg"

    WHEN:
      - Call compute_module_name(file_path, package_root)

    THEN:
      - Returns "foo"


  TEST "resolve_import handles relative imports":
    GIVEN:
      - import_stmt: "from ..utils import helper" (level=2)
      - importer_module: "pkg.subpkg.module"

    WHEN:
      - Call resolve_import(import_stmt, importer_module)

    THEN:
      - Returns "pkg.utils"


  TEST "classify_dependency recognizes stdlib":
    GIVEN:
      - module_name = "os"

    WHEN:
      - Call classify_dependency("os")

    THEN:
      - Returns DependencyType.StandardLibrary


  TEST "detect_encoding finds utf-8 BOM":
    GIVEN:
      - source = "\ufeff# -*- coding: utf-8 -*-\nprint('hello')"

    WHEN:
      - Call detect_encoding(source)

    THEN:
      - Returns "utf-8-sig"
```

### 6.4 Contract Tests

```pseudocode
CONTRACT_TESTS:
  // Tests that verify interfaces between agents

  TEST "IngestResult satisfies contract for Analysis Agent":
    GIVEN:
      - Successful IngestResult from Script mode

    VERIFY:
      - result.success == true
      - result.modules is not empty
      - All modules in result.modules have is_valid == true
      - result.entry_point is present
      - result.dependency_graph is present
      - All external dependencies are cataloged

    // These postconditions ensure Analysis Agent can proceed


  TEST "IngestResult with errors provides diagnostic information":
    GIVEN:
      - Failed IngestResult (syntax errors)

    VERIFY:
      - result.success == false
      - result.validation_issues is not empty
      - All Error severity issues have blocking == true
      - Each issue has file_path and line_number
      - Each issue has suggestion for remediation

    // These postconditions ensure user can fix the errors
```

### 6.5 Test Coverage Targets

```
Line Coverage:        >80%
Branch Coverage:      >75%
Function Coverage:    100% (all public functions tested)
Contract Coverage:    100% (input/output contracts verified)

Critical Paths:       100% coverage
  - Configuration validation
  - File discovery
  - Syntax validation
  - Dependency graph construction
  - Error reporting

Edge Cases:           Required tests
  - Empty directories
  - Single file packages
  - Circular dependencies
  - Invalid encodings
  - Symlink cycles
  - Permission errors
```

### 6.6 Mock Verification Examples

```pseudocode
EXAMPLE_MOCK_VERIFICATION:

TEST "ingest calls FileSystem.read_file for each discovered file":
    // Setup
    mock_fs = MockFileSystem()
    mock_fs.add_file("/pkg/a.py", "print('a')")
    mock_fs.add_file("/pkg/b.py", "print('b')")

    config = IngestConfiguration {
        mode: Library,
        source_path: "/pkg",
        ...
    }

    // Execute
    result = ingest(config, file_system=mock_fs)

    // Verify interactions (London School style)
    ASSERT mock_fs.read_file_calls.length == 2
    ASSERT "/pkg/a.py" IN mock_fs.read_file_calls
    ASSERT "/pkg/b.py" IN mock_fs.read_file_calls

    // Verify result
    ASSERT result.success == true
    ASSERT result.total_files == 2


TEST "ingest retries on transient filesystem errors":
    // Setup
    mock_fs = MockFileSystem()
    mock_fs.read_file_will_fail_once("/pkg/a.py")  // Fail first time, succeed second

    config = IngestConfiguration {...}

    // Execute
    result = ingest(config, file_system=mock_fs)

    // Verify retry happened
    ASSERT mock_fs.read_file_call_count("/pkg/a.py") == 2
    ASSERT result.success == true
```

---

## 7. Summary

This pseudocode specification for the **Ingest Agent** provides:

1. **Complete Data Structures**: All enums, structs, and types needed for ingestion
2. **Detailed Algorithms**: Step-by-step pseudocode for all core functions
3. **Clear Contracts**: Input/output specifications and interface to next agent
4. **Robust Error Handling**: Comprehensive error taxonomy and handling strategies
5. **TDD Test Points**: London School test structure with mocks and verification

### Key Design Decisions

1. **Fail-Fast Configuration Validation**: Invalid configuration stops processing immediately
2. **Collect-and-Continue Validation**: Syntax errors in one file don't prevent processing others
3. **Graceful Degradation**: Partial results returned when possible
4. **Separation of Concerns**: Ingest only validates and catalogs; Analysis Agent does deep analysis
5. **Testability First**: All external dependencies injected via interfaces for easy mocking

### Next Steps (SPARC Phase 3: Architecture)

The Architecture phase will:
- Define concrete Rust types and trait implementations
- Specify error handling with Result types and custom errors
- Design the file system abstraction layer
- Detail the Python parser integration (likely using RustPython or similar)
- Specify serialization formats for passing data to Analysis Agent
- Define the logging and observability infrastructure

---

**END OF INGEST AGENT PSEUDOCODE**
