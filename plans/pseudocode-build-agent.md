# PORTALIS: Build Agent Pseudocode
## SPARC Phase 2: Pseudocode

**Agent:** Build Agent (Agent 5 of 7)
**Version:** 1.0
**Date:** 2025-10-03
**Methodology:** SPARC (Specification, Pseudocode, Architecture, Refinement, Completion)
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

The Build Agent is responsible for transforming validated Rust source code into optimized, portable WASM binaries. It orchestrates the Rust compilation toolchain, manages dependencies, and produces both production artifacts (.wasm) and debugging artifacts (.wat).

### 1.2 Responsibilities

**Core Responsibilities (MUST):**
- Compile Rust code to wasm32-wasi target using cargo
- Manage and resolve Rust crate dependencies
- Validate WASM compatibility of all dependencies
- Generate optimized WASM binary modules
- Produce WebAssembly Text (.wat) files for debugging
- Handle both debug and release build profiles
- Report compilation errors with context and remediation suggestions
- Vendor dependencies for reproducible builds

**Extended Responsibilities (SHOULD):**
- Optimize dependency tree depth
- Perform WASM post-processing optimizations
- Generate build metadata and manifests
- Cache build artifacts for incremental compilation

### 1.3 Requirements Mapping

| Requirement | Description | Priority |
|-------------|-------------|----------|
| FR-2.5.1 | Rust Compilation (cargo, wasm32-wasi, error reporting) | MUST |
| FR-2.5.2 | WASM Binary Generation (.wasm, WASI, optimization, .wat) | MUST |
| FR-2.5.3 | Dependency Management (resolution, validation, vendoring) | MUST |
| NFR-3.2.1 | Fault tolerance, partial outputs, retry logic | MUST |
| NFR-3.3.2 | Output safety, unsafe block documentation | MUST |
| NFR-3.3.3 | Dependency security, checksum verification | MUST |

### 1.4 Integration Points

**Inputs:**
- Rust source code from Transpiler Agent
- Build configuration (debug/release, optimization levels)
- Dependency manifests (Cargo.toml files)

**Outputs:**
- WASM binary modules (.wasm files)
- Debug artifacts (.wat files)
- Build metadata (success/failure reports, timing, sizes)
- Dependency graph and lock files

**External Tools:**
- rustc (Rust compiler)
- cargo (Rust build system)
- wasm-tools (WASM processing utilities)
- cargo-vendor (dependency vendoring)

---

## 2. Data Structures

### 2.1 Build Configuration

```rust
STRUCTURE BuildConfiguration:
    // Build profile
    profile: BuildProfile              // Debug or Release
    target: TargetTriple               // "wasm32-wasi"
    optimization_level: OptLevel       // 0, 1, 2, 3, s, z

    // Source locations
    workspace_root: Path               // Root of Rust workspace
    crate_manifest: Path               // Path to Cargo.toml
    output_directory: Path             // Where to place artifacts

    // Feature flags
    enable_lto: bool                   // Link-Time Optimization
    enable_strip: bool                 // Strip debug symbols
    codegen_units: Option<u32>         // Parallel codegen units

    // Dependency configuration
    vendor_dependencies: bool          // Vendor deps for reproducibility
    offline_mode: bool                 // Build without network access
    locked_dependencies: bool          // Use Cargo.lock strictly

    // Output configuration
    generate_wat: bool                 // Generate .wat debug files
    generate_metadata: bool            // Generate build manifest

    // Error handling
    max_compilation_retries: u32       // Retry transient errors
    timeout_seconds: u64               // Maximum build time

    // Validation
    validate_wasi_compat: bool         // Check WASI compatibility
    audit_dependencies: bool           // Security audit deps

ENUM BuildProfile:
    Debug
    Release

ENUM OptLevel:
    O0      // No optimization
    O1      // Basic optimization
    O2      // Full optimization
    O3      // Aggressive optimization
    Os      // Optimize for size
    Oz      // Aggressively optimize for size

STRUCTURE TargetTriple:
    architecture: String               // "wasm32"
    vendor: String                     // "unknown"
    system: String                     // "wasi"

    FUNCTION to_string() -> String:
        RETURN "{architecture}-{vendor}-{system}"
```

### 2.2 Dependency Graph

```rust
STRUCTURE DependencyGraph:
    nodes: Map<PackageId, DependencyNode>
    edges: List<DependencyEdge>
    root_packages: Set<PackageId>

    // Metadata
    total_crates: u32
    depth: u32                         // Maximum depth of tree
    wasi_incompatible: Set<PackageId>  // Crates not supporting wasm32-wasi

    FUNCTION add_node(node: DependencyNode) -> Result<(), Error>
    FUNCTION add_edge(from: PackageId, to: PackageId, kind: DependencyKind) -> Result<(), Error>
    FUNCTION topological_sort() -> Result<List<PackageId>, CyclicDependencyError>
    FUNCTION find_incompatible_crates() -> Set<PackageId>
    FUNCTION calculate_depth() -> u32

STRUCTURE DependencyNode:
    id: PackageId
    name: String
    version: SemanticVersion
    source: DependencySource

    // Compatibility
    supports_wasm32_wasi: bool
    no_std_compatible: bool

    // Metadata
    checksum: String                   // SHA-256 of crate
    features_enabled: Set<String>
    build_script: Option<Path>         // Custom build.rs

    // Security
    vulnerability_scan: VulnerabilityScanResult
    license: String

STRUCTURE DependencyEdge:
    from: PackageId
    to: PackageId
    kind: DependencyKind
    optional: bool

ENUM DependencyKind:
    Normal      // Regular dependency
    Dev         // Development-only dependency
    Build       // Build-time dependency

ENUM DependencySource:
    CratesIo(CratesIoSource)
    Git(GitSource)
    Path(PathSource)
    Registry(RegistrySource)

STRUCTURE PackageId:
    name: String
    version: SemanticVersion
    source_id: String                  // Unique identifier for source

    FUNCTION unique_key() -> String:
        RETURN "{name}-{version}@{source_id}"
```

### 2.3 Compilation Artifacts

```rust
STRUCTURE CompilationArtifacts:
    // Primary outputs
    wasm_binaries: Map<CrateName, WasmBinary>
    wat_files: Map<CrateName, WatFile>

    // Metadata
    build_manifest: BuildManifest
    dependency_lock: DependencyLockFile

    // Diagnostics
    compilation_messages: List<CompilerMessage>
    warnings: List<CompilerWarning>
    errors: List<CompilerError>

    // Performance metrics
    build_time: Duration
    artifact_sizes: Map<CrateName, u64>

    FUNCTION total_wasm_size() -> u64
    FUNCTION has_errors() -> bool
    FUNCTION filter_warnings(level: WarningLevel) -> List<CompilerWarning>

STRUCTURE WasmBinary:
    crate_name: String
    file_path: Path

    // Binary properties
    size_bytes: u64
    size_optimized: u64                // After wasm-opt

    // Module metadata
    imports: List<WasmImport>          // Required WASI imports
    exports: List<WasmExport>          // Exported functions
    memory_pages: u32                  // Linear memory pages
    table_size: u32                    // Function table size

    // Validation
    wasi_compliant: bool
    validation_errors: List<String>

    // Checksums
    sha256: String

    FUNCTION validate_wasi() -> Result<(), ValidationError>
    FUNCTION extract_imports() -> Result<List<WasmImport>, Error>
    FUNCTION extract_exports() -> Result<List<WasmExport>, Error>

STRUCTURE WatFile:
    crate_name: String
    file_path: Path
    size_bytes: u64

    // For debugging assistance
    function_names: List<String>
    type_section: String

STRUCTURE WasmImport:
    module: String                     // e.g., "wasi_snapshot_preview1"
    name: String                       // e.g., "fd_write"
    kind: ImportKind

STRUCTURE WasmExport:
    name: String
    kind: ExportKind

ENUM ImportKind:
    Function
    Table
    Memory
    Global

ENUM ExportKind:
    Function
    Table
    Memory
    Global
```

### 2.4 Build Metadata

```rust
STRUCTURE BuildManifest:
    version: String                    // Manifest format version
    timestamp: DateTime
    build_profile: BuildProfile

    // Environment
    rustc_version: String
    cargo_version: String
    host_triple: String
    target_triple: String

    // Artifacts
    artifacts: List<ArtifactMetadata>

    // Build statistics
    total_crates_compiled: u32
    total_build_time: Duration
    compilation_units: u32

    // Dependency snapshot
    dependency_tree: DependencyGraph

    FUNCTION to_json() -> String
    FUNCTION validate() -> Result<(), Error>

STRUCTURE ArtifactMetadata:
    name: String
    kind: ArtifactKind
    path: Path
    size: u64
    checksum: String

ENUM ArtifactKind:
    WasmBinary
    WatDebugFile
    DependencyVendor
    LockFile
    BuildLog
```

### 2.5 Error Representations

```rust
STRUCTURE CompilationError:
    severity: ErrorSeverity
    message: String
    code: Option<String>               // Rust error code (e.g., "E0308")

    // Location
    file: Option<Path>
    line: Option<u32>
    column: Option<u32>
    span: Option<SourceSpan>

    // Context
    snippet: Option<String>            // Code snippet with error
    suggestion: Option<String>         // Compiler suggestion

    // Categorization
    category: ErrorCategory
    remediation: RemediationStrategy

    FUNCTION format_diagnostic() -> String
    FUNCTION to_json() -> String

ENUM ErrorSeverity:
    Error
    Warning
    Note
    Help

ENUM ErrorCategory:
    SyntaxError
    TypeError
    BorrowCheckError
    TraitError
    LifetimeError
    LinkError
    WasmCompatibilityError
    DependencyResolutionError
    BuildScriptError

STRUCTURE RemediationStrategy:
    description: String
    automated_fix_available: bool
    fix_script: Option<String>
    documentation_link: Option<String>

STRUCTURE DependencyConflict:
    package_name: String
    conflicting_versions: List<ConflictingVersion>
    root_cause: ConflictCause
    resolution_strategy: ConflictResolution

STRUCTURE ConflictingVersion:
    version: SemanticVersion
    required_by: List<PackageId>
    reason: String

ENUM ConflictCause:
    VersionMismatch
    FeatureConflict
    WasmIncompatibility
    PlatformIncompatibility

ENUM ConflictResolution:
    UseNewerVersion
    UseOlderVersion
    UnifyFeatures
    RemoveDependency
    ManualIntervention
```

---

## 3. Core Algorithms

### 3.1 Main Build Orchestration

```rust
ALGORITHM build_wasm_modules(
    rust_workspace: RustWorkspace,
    config: BuildConfiguration
) -> Result<CompilationArtifacts, BuildError>:

    LOG("Starting WASM build orchestration", INFO)
    start_time = current_timestamp()

    // Initialize artifact collection
    artifacts = CompilationArtifacts::new()

    TRY:
        // Step 1: Validate Rust workspace structure
        LOG("Validating Rust workspace structure", DEBUG)
        validation_result = validate_workspace_structure(rust_workspace)
        IF validation_result.is_error():
            RETURN Error(WorkspaceInvalid(validation_result.errors))

        // Step 2: Resolve and validate dependencies
        LOG("Resolving dependency graph", INFO)
        dependency_graph = resolve_dependencies(
            rust_workspace,
            config.locked_dependencies,
            config.offline_mode
        )

        // Step 3: Check WASM compatibility of dependencies
        LOG("Validating WASM compatibility", INFO)
        incompatible_deps = dependency_graph.find_incompatible_crates()
        IF NOT incompatible_deps.is_empty():
            LOG("Found WASM-incompatible dependencies: {incompatible_deps}", WARN)
            TRY:
                dependency_graph = resolve_incompatibilities(
                    dependency_graph,
                    incompatible_deps
                )
            CATCH ResolutionImpossible AS e:
                RETURN Error(WasmIncompatibleDependencies(incompatible_deps))

        // Step 4: Vendor dependencies if configured
        IF config.vendor_dependencies:
            LOG("Vendoring dependencies for reproducible build", INFO)
            vendor_result = vendor_dependencies(rust_workspace, dependency_graph)
            artifacts.add_vendor_archive(vendor_result.archive_path)

        // Step 5: Audit dependencies for security
        IF config.audit_dependencies:
            LOG("Auditing dependencies for vulnerabilities", INFO)
            audit_result = audit_dependency_security(dependency_graph)
            IF audit_result.has_critical_vulnerabilities():
                LOG("Critical vulnerabilities found", ERROR)
                RETURN Error(SecurityVulnerabilities(audit_result.critical))
            artifacts.add_audit_report(audit_result)

        // Step 6: Compile Rust to WASM
        LOG("Compiling Rust to WASM (target: wasm32-wasi)", INFO)
        compilation_result = compile_to_wasm(
            rust_workspace,
            config,
            dependency_graph
        )

        IF compilation_result.has_errors():
            LOG("Compilation failed with errors", ERROR)
            artifacts.errors = compilation_result.errors
            artifacts.warnings = compilation_result.warnings
            RETURN Error(CompilationFailed(compilation_result))

        artifacts.wasm_binaries = compilation_result.binaries
        artifacts.warnings = compilation_result.warnings

        // Step 7: Generate WAT debug files
        IF config.generate_wat:
            LOG("Generating WAT debug files", DEBUG)
            FOR EACH (crate_name, wasm_binary) IN artifacts.wasm_binaries:
                wat_file = generate_wat_file(wasm_binary)
                artifacts.wat_files[crate_name] = wat_file

        // Step 8: Optimize WASM binaries
        LOG("Optimizing WASM binaries", INFO)
        optimization_result = optimize_wasm_binaries(
            artifacts.wasm_binaries,
            config.optimization_level
        )
        artifacts.wasm_binaries = optimization_result.optimized_binaries

        // Step 9: Validate WASM outputs
        LOG("Validating WASM binaries", INFO)
        FOR EACH (crate_name, wasm_binary) IN artifacts.wasm_binaries:
            validation = validate_wasm_binary(wasm_binary)
            IF NOT validation.is_valid():
                LOG("WASM validation failed for {crate_name}", ERROR)
                RETURN Error(WasmValidationFailed(crate_name, validation.errors))

        // Step 10: Generate build metadata
        IF config.generate_metadata:
            LOG("Generating build manifest", DEBUG)
            build_time = current_timestamp() - start_time
            manifest = generate_build_manifest(
                config,
                dependency_graph,
                artifacts,
                build_time
            )
            artifacts.build_manifest = manifest
            write_manifest_file(manifest, config.output_directory)

        LOG("Build completed successfully", INFO)
        LOG("Total build time: {build_time}s", INFO)
        LOG("Total WASM size: {artifacts.total_wasm_size()} bytes", INFO)

        RETURN Ok(artifacts)

    CATCH DependencyResolutionError AS e:
        LOG("Dependency resolution failed: {e}", ERROR)
        RETURN Error(e)

    CATCH CompilationError AS e:
        LOG("Compilation failed: {e}", ERROR)
        RETURN Error(e)

    CATCH TimeoutError AS e:
        LOG("Build timeout exceeded: {config.timeout_seconds}s", ERROR)
        RETURN Error(BuildTimeout(config.timeout_seconds))

    CATCH Error AS e:
        LOG("Unexpected build error: {e}", ERROR)
        RETURN Error(BuildFailed(e))
```

### 3.2 Dependency Resolution

```rust
ALGORITHM resolve_dependencies(
    workspace: RustWorkspace,
    use_lockfile: bool,
    offline: bool
) -> Result<DependencyGraph, DependencyError>:

    LOG("Starting dependency resolution", DEBUG)
    graph = DependencyGraph::new()

    // Step 1: Parse Cargo.toml files
    manifests = collect_cargo_manifests(workspace)
    FOR EACH manifest IN manifests:
        root_package = parse_package_manifest(manifest)
        graph.add_node(root_package.to_node())
        graph.root_packages.insert(root_package.id)

    // Step 2: Check for existing Cargo.lock
    lockfile_path = workspace.root.join("Cargo.lock")
    IF use_lockfile AND lockfile_path.exists():
        LOG("Using existing Cargo.lock for exact versions", DEBUG)
        locked_dependencies = parse_cargo_lock(lockfile_path)
        RETURN build_graph_from_lock(locked_dependencies)

    // Step 3: Resolve transitive dependencies
    queue = Queue::from(graph.root_packages)
    visited = Set::new()

    WHILE NOT queue.is_empty():
        current_pkg = queue.dequeue()

        IF visited.contains(current_pkg):
            CONTINUE
        visited.insert(current_pkg)

        // Fetch dependency metadata
        dependencies = fetch_package_dependencies(current_pkg, offline)

        FOR EACH dep IN dependencies:
            // Resolve version
            resolved_version = resolve_version(
                dep.name,
                dep.version_req,
                graph,
                offline
            )

            dep_node = DependencyNode {
                id: PackageId::new(dep.name, resolved_version),
                name: dep.name,
                version: resolved_version,
                source: dep.source,
                supports_wasm32_wasi: UNKNOWN,  // Will validate later
                features_enabled: dep.features,
                checksum: fetch_checksum(dep, offline)
            }

            graph.add_node(dep_node)
            graph.add_edge(current_pkg, dep_node.id, dep.kind)

            IF dep.kind == DependencyKind::Normal:
                queue.enqueue(dep_node.id)

    // Step 4: Detect cycles
    IF graph.has_cycle():
        cycle = graph.find_cycle()
        LOG("Circular dependency detected: {cycle}", ERROR)
        RETURN Error(CircularDependency(cycle))

    // Step 5: Calculate graph metadata
    graph.depth = graph.calculate_depth()
    graph.total_crates = graph.nodes.len()

    LOG("Resolved {graph.total_crates} crates (depth: {graph.depth})", INFO)

    RETURN Ok(graph)


ALGORITHM resolve_version(
    package_name: String,
    version_req: VersionRequirement,
    current_graph: DependencyGraph,
    offline: bool
) -> Result<SemanticVersion, VersionResolutionError>:

    // Check if already resolved in graph
    existing_versions = current_graph.nodes
        .filter(|node| node.name == package_name)
        .map(|node| node.version)

    IF NOT existing_versions.is_empty():
        // Find compatible version
        FOR EACH version IN existing_versions:
            IF version_req.matches(version):
                LOG("Reusing existing version {version} for {package_name}", DEBUG)
                RETURN Ok(version)

        // Version conflict detected
        LOG("Version conflict for {package_name}: {version_req} vs {existing_versions}", WARN)
        conflict = DependencyConflict {
            package_name: package_name,
            conflicting_versions: existing_versions,
            root_cause: ConflictCause::VersionMismatch
        }
        RETURN Error(VersionConflict(conflict))

    // Fetch available versions from registry
    IF offline:
        RETURN Error(CannotResolveOffline(package_name))

    available_versions = fetch_available_versions(package_name)

    // Select highest version matching requirement
    selected = available_versions
        .filter(|v| version_req.matches(v))
        .max()

    IF selected.is_none():
        RETURN Error(NoMatchingVersion(package_name, version_req))

    RETURN Ok(selected)


ALGORITHM resolve_incompatibilities(
    graph: DependencyGraph,
    incompatible: Set<PackageId>
) -> Result<DependencyGraph, ResolutionError>:

    LOG("Attempting to resolve {incompatible.len()} incompatible dependencies", INFO)

    FOR EACH pkg_id IN incompatible:
        // Strategy 1: Check for WASM-compatible alternative version
        alternative_versions = find_wasm_compatible_versions(pkg_id.name)

        IF NOT alternative_versions.is_empty():
            LOG("Found WASM-compatible version for {pkg_id.name}", INFO)
            graph = replace_dependency_version(graph, pkg_id, alternative_versions[0])
            CONTINUE

        // Strategy 2: Check if dependency is optional
        dependents = graph.find_dependents(pkg_id)
        all_optional = dependents.all(|dep| dep.optional == true)

        IF all_optional:
            LOG("Removing optional incompatible dependency: {pkg_id}", INFO)
            graph = remove_dependency(graph, pkg_id)
            CONTINUE

        // Strategy 3: Check for feature flags that disable incompatibility
        wasm_features = find_wasm_enabling_features(pkg_id)
        IF NOT wasm_features.is_empty():
            LOG("Enabling WASM features for {pkg_id}: {wasm_features}", INFO)
            graph = enable_features(graph, pkg_id, wasm_features)
            CONTINUE

        // No resolution strategy worked
        LOG("Cannot resolve WASM incompatibility for {pkg_id}", ERROR)
        RETURN Error(ResolutionImpossible(pkg_id))

    RETURN Ok(graph)
```

### 3.3 Cargo Compilation

```rust
ALGORITHM compile_to_wasm(
    workspace: RustWorkspace,
    config: BuildConfiguration,
    dependencies: DependencyGraph
) -> Result<CompilationResult, CompilationError>:

    LOG("Starting Rust to WASM compilation", INFO)
    result = CompilationResult::new()

    // Step 1: Setup cargo environment
    cargo_env = setup_cargo_environment(config, dependencies)

    // Step 2: Build cargo command
    cargo_cmd = CargoCommand::new("build")
        .arg("--target", config.target.to_string())
        .arg("--target-dir", config.output_directory)

    // Add profile-specific flags
    MATCH config.profile:
        BuildProfile::Release:
            cargo_cmd.arg("--release")
        BuildProfile::Debug:
            // Debug is default, no flag needed
            PASS

    // Add optimization flags
    cargo_env.set("RUSTFLAGS", build_rustflags(config))

    // Add codegen options
    IF config.codegen_units.is_some():
        cargo_env.set("CARGO_CODEGEN_UNITS", config.codegen_units.unwrap())

    // Configure dependency mode
    IF config.offline_mode:
        cargo_cmd.arg("--offline")

    IF config.locked_dependencies:
        cargo_cmd.arg("--locked")

    // Step 3: Execute compilation
    LOG("Executing: {cargo_cmd.to_string()}", DEBUG)

    retry_count = 0
    max_retries = config.max_compilation_retries

    LOOP:
        process = execute_cargo_with_timeout(
            cargo_cmd,
            cargo_env,
            config.timeout_seconds
        )

        MATCH process.wait():
            ProcessResult::Success(output):
                // Parse compilation messages
                messages = parse_cargo_output(output.stderr)
                result.warnings = messages.filter(|m| m.severity == Warning)
                result.errors = messages.filter(|m| m.severity == Error)

                // Collect compiled binaries
                result.binaries = collect_wasm_binaries(
                    config.output_directory,
                    config.target
                )

                LOG("Compilation successful: {result.binaries.len()} binaries", INFO)
                RETURN Ok(result)

            ProcessResult::Failure(output):
                messages = parse_cargo_output(output.stderr)
                errors = messages.filter(|m| m.severity == Error)

                // Check if error is retryable
                IF is_transient_error(errors) AND retry_count < max_retries:
                    retry_count += 1
                    LOG("Retrying compilation ({retry_count}/{max_retries})", WARN)
                    wait_backoff(retry_count)
                    CONTINUE

                // Non-retryable error or max retries exceeded
                result.errors = errors
                result.warnings = messages.filter(|m| m.severity == Warning)

                LOG("Compilation failed with {errors.len()} errors", ERROR)
                RETURN Ok(result)  // Return result with errors

            ProcessResult::Timeout:
                LOG("Compilation timeout after {config.timeout_seconds}s", ERROR)
                RETURN Error(CompilationTimeout(config.timeout_seconds))

            ProcessResult::Killed(signal):
                RETURN Error(ProcessKilled(signal))


FUNCTION build_rustflags(config: BuildConfiguration) -> String:
    flags = List::new()

    // Target-specific flags
    flags.push("-C target-feature=+simd128")  // Enable WASM SIMD

    // Optimization flags
    MATCH config.optimization_level:
        OptLevel::O0:
            flags.push("-C opt-level=0")
        OptLevel::O1:
            flags.push("-C opt-level=1")
        OptLevel::O2:
            flags.push("-C opt-level=2")
        OptLevel::O3:
            flags.push("-C opt-level=3")
        OptLevel::Os:
            flags.push("-C opt-level=s")
        OptLevel::Oz:
            flags.push("-C opt-level=z")

    // Link-time optimization
    IF config.enable_lto:
        flags.push("-C lto=fat")

    // Strip symbols in release
    IF config.enable_strip AND config.profile == BuildProfile::Release:
        flags.push("-C strip=symbols")

    // Panic behavior for WASM
    flags.push("-C panic=abort")

    RETURN flags.join(" ")


FUNCTION parse_cargo_output(stderr: String) -> List<CompilerMessage>:
    messages = List::new()

    // Cargo outputs JSON messages when using --message-format=json
    // For simplicity, parse text output
    lines = stderr.split("\n")

    FOR EACH line IN lines:
        // Match patterns like "error[E0308]: ..." or "warning: ..."
        IF line.starts_with("error"):
            message = parse_error_message(line)
            messages.push(message)
        ELSE IF line.starts_with("warning"):
            message = parse_warning_message(line)
            messages.push(message)
        ELSE IF line.contains("help:") OR line.contains("note:"):
            // Attach to previous message if exists
            IF NOT messages.is_empty():
                messages.last_mut().add_note(line)

    RETURN messages


FUNCTION is_transient_error(errors: List<CompilerError>) -> bool:
    transient_patterns = [
        "could not download",
        "network error",
        "connection refused",
        "timeout",
        "temporary failure"
    ]

    FOR EACH error IN errors:
        FOR EACH pattern IN transient_patterns:
            IF error.message.contains(pattern):
                RETURN true

    RETURN false
```

### 3.4 Dependency Vendoring

```rust
ALGORITHM vendor_dependencies(
    workspace: RustWorkspace,
    dependency_graph: DependencyGraph
) -> Result<VendorResult, VendorError>:

    LOG("Vendoring dependencies for reproducible builds", INFO)

    vendor_dir = workspace.root.join("vendor")
    create_directory(vendor_dir)

    // Step 1: Download all dependencies
    downloaded_crates = List::new()

    FOR EACH node IN dependency_graph.nodes.values():
        IF node.source == DependencySource::CratesIo:
            LOG("Downloading {node.name}-{node.version}", DEBUG)

            crate_data = download_crate(node.name, node.version)

            // Verify checksum
            IF NOT verify_checksum(crate_data, node.checksum):
                LOG("Checksum mismatch for {node.name}-{node.version}", ERROR)
                RETURN Error(ChecksumMismatch(node.name))

            // Extract to vendor directory
            extract_path = vendor_dir.join("{node.name}-{node.version}")
            extract_crate(crate_data, extract_path)

            downloaded_crates.push(node)

    // Step 2: Generate vendor configuration
    vendor_config = generate_vendor_config(downloaded_crates, vendor_dir)
    vendor_config_path = workspace.root.join(".cargo/config.toml")
    write_file(vendor_config_path, vendor_config.to_toml())

    // Step 3: Create vendor archive
    archive_path = workspace.root.join("vendor.tar.gz")
    create_archive(vendor_dir, archive_path)

    LOG("Vendored {downloaded_crates.len()} crates to {vendor_dir}", INFO)

    RETURN Ok(VendorResult {
        vendor_directory: vendor_dir,
        archive_path: archive_path,
        crate_count: downloaded_crates.len(),
        total_size: calculate_directory_size(vendor_dir)
    })


FUNCTION verify_checksum(data: Bytes, expected_checksum: String) -> bool:
    actual_checksum = sha256_hash(data)
    RETURN actual_checksum == expected_checksum


FUNCTION generate_vendor_config(
    crates: List<DependencyNode>,
    vendor_dir: Path
) -> VendorConfig:

    config = VendorConfig::new()

    FOR EACH crate IN crates:
        source_replacement = SourceReplacement {
            replace_with: "vendored-sources",
            directory: vendor_dir.join("{crate.name}-{crate.version}")
        }
        config.add_source(crate.source_id, source_replacement)

    RETURN config
```

### 3.5 WASM Optimization

```rust
ALGORITHM optimize_wasm_binaries(
    binaries: Map<CrateName, WasmBinary>,
    opt_level: OptLevel
) -> Result<OptimizationResult, OptimizationError>:

    LOG("Optimizing WASM binaries", INFO)
    optimized = Map::new()

    FOR EACH (crate_name, binary) IN binaries:
        LOG("Optimizing {crate_name} ({binary.size_bytes} bytes)", DEBUG)

        original_size = binary.size_bytes

        // Step 1: Run wasm-opt
        opt_result = run_wasm_opt(binary.file_path, opt_level)

        // Step 2: Strip custom sections (debug info)
        stripped = strip_custom_sections(opt_result.output_path)

        // Step 3: Run wasm-gc (garbage collection)
        gc_result = run_wasm_gc(stripped)

        optimized_binary = WasmBinary {
            crate_name: crate_name,
            file_path: gc_result.output_path,
            size_bytes: file_size(gc_result.output_path),
            size_optimized: file_size(gc_result.output_path),
            imports: binary.imports,
            exports: binary.exports,
            memory_pages: binary.memory_pages,
            table_size: binary.table_size,
            wasi_compliant: binary.wasi_compliant,
            sha256: calculate_sha256(gc_result.output_path)
        }

        size_reduction = ((original_size - optimized_binary.size_bytes) as f64 / original_size as f64) * 100.0
        LOG("Optimized {crate_name}: {original_size} -> {optimized_binary.size_bytes} bytes ({size_reduction:.1}% reduction)", INFO)

        optimized[crate_name] = optimized_binary

    RETURN Ok(OptimizationResult {
        optimized_binaries: optimized
    })


FUNCTION run_wasm_opt(input: Path, opt_level: OptLevel) -> Result<WasmOptOutput, Error>:
    opt_flags = MATCH opt_level:
        OptLevel::O0 -> ["-O0"]
        OptLevel::O1 -> ["-O1"]
        OptLevel::O2 -> ["-O2"]
        OptLevel::O3 -> ["-O3"]
        OptLevel::Os -> ["-Os"]
        OptLevel::Oz -> ["-Oz"]

    output_path = input.with_extension("opt.wasm")

    cmd = Command::new("wasm-opt")
        .args(opt_flags)
        .arg(input)
        .arg("-o", output_path)

    result = cmd.execute()

    IF NOT result.success():
        RETURN Error(WasmOptFailed(result.stderr))

    RETURN Ok(WasmOptOutput {
        output_path: output_path
    })


FUNCTION strip_custom_sections(wasm_path: Path) -> Path:
    // Custom sections to remove for size optimization
    sections_to_remove = [
        "name",          // Function names (debug info)
        "producers",     // Toolchain info
        "target_features"
    ]

    output = wasm_path.with_extension("stripped.wasm")

    FOR EACH section IN sections_to_remove:
        cmd = Command::new("wasm-tools")
            .arg("strip")
            .arg("--delete", section)
            .arg(wasm_path)
            .arg("-o", output)

        cmd.execute()
        wasm_path = output

    RETURN output
```

### 3.6 WASM Validation

```rust
ALGORITHM validate_wasm_binary(binary: WasmBinary) -> ValidationResult:

    LOG("Validating WASM binary: {binary.crate_name}", DEBUG)
    result = ValidationResult::new()

    // Step 1: Validate WASM format
    format_validation = validate_wasm_format(binary.file_path)
    IF NOT format_validation.is_valid():
        result.add_error("Invalid WASM format: {format_validation.error}")
        RETURN result

    // Step 2: Extract and validate imports
    TRY:
        imports = extract_wasm_imports(binary.file_path)
        binary.imports = imports

        // Validate all imports are WASI
        FOR EACH import IN imports:
            IF NOT is_wasi_import(import):
                result.add_error("Non-WASI import found: {import.module}::{import.name}")
    CATCH Error AS e:
        result.add_error("Failed to extract imports: {e}")

    // Step 3: Extract exports
    TRY:
        exports = extract_wasm_exports(binary.file_path)
        binary.exports = exports

        IF exports.is_empty():
            result.add_warning("No exports found in WASM module")
    CATCH Error AS e:
        result.add_error("Failed to extract exports: {e}")

    // Step 4: Validate memory configuration
    memory_validation = validate_memory_layout(binary.file_path)
    IF NOT memory_validation.is_valid():
        result.add_error("Invalid memory layout: {memory_validation.error}")

    // Step 5: Check for unsupported features
    unsupported_features = check_unsupported_features(binary.file_path)
    FOR EACH feature IN unsupported_features:
        result.add_error("Unsupported WASM feature: {feature}")

    // Set validation status
    binary.wasi_compliant = result.is_valid()
    binary.validation_errors = result.errors

    RETURN result


FUNCTION is_wasi_import(import: WasmImport) -> bool:
    wasi_modules = [
        "wasi_snapshot_preview1",
        "wasi_unstable"
    ]

    RETURN wasi_modules.contains(import.module)


FUNCTION extract_wasm_imports(wasm_path: Path) -> Result<List<WasmImport>, Error>:
    // Use wasm-tools to inspect module
    cmd = Command::new("wasm-tools")
        .arg("print")
        .arg(wasm_path)

    output = cmd.execute()

    IF NOT output.success():
        RETURN Error(WasmToolsFailed(output.stderr))

    // Parse WAT output to extract imports
    imports = parse_wat_imports(output.stdout)

    RETURN Ok(imports)


FUNCTION extract_wasm_exports(wasm_path: Path) -> Result<List<WasmExport>, Error>:
    cmd = Command::new("wasm-tools")
        .arg("print")
        .arg(wasm_path)

    output = cmd.execute()

    IF NOT output.success():
        RETURN Error(WasmToolsFailed(output.stderr))

    exports = parse_wat_exports(output.stdout)

    RETURN Ok(exports)
```

### 3.7 WAT Generation

```rust
ALGORITHM generate_wat_file(wasm_binary: WasmBinary) -> Result<WatFile, Error>:

    LOG("Generating WAT file for {wasm_binary.crate_name}", DEBUG)

    // Convert WASM to WAT using wasm2wat
    wat_path = wasm_binary.file_path.with_extension("wat")

    cmd = Command::new("wasm2wat")
        .arg(wasm_binary.file_path)
        .arg("-o", wat_path)
        .arg("--generate-names")  // Generate human-readable names
        .arg("--fold-exprs")      // Fold expressions for readability

    result = cmd.execute()

    IF NOT result.success():
        RETURN Error(Wasm2WatFailed(result.stderr))

    // Parse WAT to extract metadata
    wat_content = read_file(wat_path)
    function_names = extract_function_names(wat_content)
    type_section = extract_type_section(wat_content)

    wat_file = WatFile {
        crate_name: wasm_binary.crate_name,
        file_path: wat_path,
        size_bytes: file_size(wat_path),
        function_names: function_names,
        type_section: type_section
    }

    LOG("Generated WAT file: {wat_path} ({wat_file.size_bytes} bytes)", DEBUG)

    RETURN Ok(wat_file)


FUNCTION extract_function_names(wat_content: String) -> List<String>:
    names = List::new()

    // Parse function declarations: (func $name ...)
    pattern = regex("\\(func \\$([a-zA-Z0-9_]+)")

    FOR EACH match IN pattern.find_all(wat_content):
        names.push(match.group(1))

    RETURN names
```

### 3.8 Dependency Auditing

```rust
ALGORITHM audit_dependency_security(
    dependency_graph: DependencyGraph
) -> AuditResult:

    LOG("Auditing {dependency_graph.total_crates} dependencies for security vulnerabilities", INFO)

    audit_result = AuditResult::new()

    // Use cargo-audit or similar tool
    FOR EACH node IN dependency_graph.nodes.values():
        // Check against vulnerability database
        vulnerabilities = query_vulnerability_database(node.name, node.version)

        IF NOT vulnerabilities.is_empty():
            FOR EACH vuln IN vulnerabilities:
                audit_result.add_vulnerability(node, vuln)

                MATCH vuln.severity:
                    Severity::Critical:
                        LOG("CRITICAL vulnerability in {node.name}-{node.version}: {vuln.id}", ERROR)
                    Severity::High:
                        LOG("HIGH vulnerability in {node.name}-{node.version}: {vuln.id}", WARN)
                    Severity::Medium:
                        LOG("MEDIUM vulnerability in {node.name}-{node.version}: {vuln.id}", WARN)
                    Severity::Low:
                        LOG("LOW vulnerability in {node.name}-{node.version}: {vuln.id}", INFO)

        // Check license compatibility
        IF NOT is_license_compatible(node.license):
            audit_result.add_license_issue(node)

    RETURN audit_result


FUNCTION query_vulnerability_database(
    package: String,
    version: SemanticVersion
) -> List<Vulnerability>:

    // Query RustSec Advisory Database
    // https://rustsec.org/

    advisories = fetch_advisories_for_package(package)

    vulnerabilities = List::new()

    FOR EACH advisory IN advisories:
        IF advisory.affected_versions.contains(version):
            vuln = Vulnerability {
                id: advisory.id,
                package: package,
                version: version,
                severity: advisory.severity,
                description: advisory.description,
                patched_versions: advisory.patched_versions,
                cvss_score: advisory.cvss_score
            }
            vulnerabilities.push(vuln)

    RETURN vulnerabilities
```

---

## 4. Input/Output Contracts

### 4.1 Agent Input Contract

```yaml
INPUT: BuildAgentInput
  rust_workspace:
    root_path: Path                    # Root of Rust workspace
    crates: List<CrateMetadata>       # Individual crates to build
    cargo_workspace_toml: Path         # Workspace Cargo.toml

  build_configuration:
    profile: "debug" | "release"
    optimization_level: "0" | "1" | "2" | "3" | "s" | "z"
    target: "wasm32-wasi"              # Fixed for this agent

    # Dependency options
    vendor_dependencies: bool          # Default: true
    offline_mode: bool                 # Default: false
    locked_dependencies: bool          # Default: true
    audit_dependencies: bool           # Default: true

    # Output options
    output_directory: Path
    generate_wat: bool                 # Default: true
    generate_metadata: bool            # Default: true

    # Compilation options
    enable_lto: bool                   # Default: true for release
    enable_strip: bool                 # Default: true for release
    codegen_units: Option<u32>         # Default: None (use rustc default)

    # Constraints
    max_compilation_retries: u32       # Default: 3
    timeout_seconds: u64               # Default: 1800 (30 min)

  context:
    pipeline_id: String                # For tracking/logging
    previous_agent: "TranspilerAgent"
    mode: "script" | "library"

PRECONDITIONS:
  - rust_workspace.root_path exists and is readable
  - All crates have valid Cargo.toml files
  - Rust toolchain is installed with wasm32-wasi target
  - wasm-tools and wasm-opt are available in PATH
  - output_directory is writable
  - If offline_mode=true, all dependencies are cached
  - If locked_dependencies=true, Cargo.lock exists

INVARIANTS:
  - target MUST be "wasm32-wasi"
  - profile MUST be either "debug" or "release"
  - timeout_seconds > 0
  - max_compilation_retries >= 0
```

### 4.2 Agent Output Contract

```yaml
OUTPUT: BuildAgentOutput
  status: "success" | "partial_success" | "failure"

  artifacts:
    wasm_binaries: List<WasmBinaryArtifact>
    wat_files: List<WatFileArtifact>
    build_manifest: BuildManifest
    vendor_archive: Option<Path>
    dependency_lock: Path              # Cargo.lock

  metadata:
    total_crates_built: u32
    successful_builds: u32
    failed_builds: u32
    total_build_time_seconds: f64
    total_wasm_size_bytes: u64

  dependency_info:
    dependency_graph: DependencyGraph
    total_dependencies: u32
    dependency_tree_depth: u32
    wasi_incompatible_deps: List<String>

  diagnostics:
    compilation_errors: List<CompilationError>
    compilation_warnings: List<CompilationWarning>
    validation_errors: List<ValidationError>
    security_audit: Option<AuditResult>

  performance_metrics:
    compilation_time: Duration
    optimization_time: Duration
    validation_time: Duration
    dependency_resolution_time: Duration

POSTCONDITIONS:
  - IF status == "success":
      - All crates compiled successfully
      - All WASM binaries are valid and WASI-compliant
      - wasm_binaries.len() == rust_workspace.crates.len()
      - compilation_errors.is_empty() == true
      - All artifacts exist at specified paths

  - IF status == "partial_success":
      - At least one crate compiled successfully
      - wasm_binaries.len() > 0
      - compilation_errors.is_empty() == false
      - Errors are documented in diagnostics

  - IF status == "failure":
      - No WASM binaries produced OR critical validation failures
      - compilation_errors.is_empty() == false
      - Error details provided in diagnostics

  - ALWAYS:
      - build_manifest is generated if requested
      - dependency_graph is complete
      - All file paths are absolute
      - All checksums are SHA-256

SIDE EFFECTS:
  - Creates WASM files in output_directory
  - Creates WAT files if generate_wat=true
  - Creates vendor directory if vendor_dependencies=true
  - Updates Cargo.lock with resolved dependencies
  - May download crates from crates.io unless offline=true
  - Logs all activities to logging system
```

### 4.3 Error Output Contract

```yaml
ERROR_OUTPUT: BuildError
  error_type: ErrorType
  message: String
  context: ErrorContext
  remediation: RemediationSuggestion

  # Specific error details
  details: ONEOF
    - CompilationFailure:
        failed_crates: List<CrateName>
        errors: List<CompilationError>
        partial_artifacts: List<WasmBinaryArtifact>

    - DependencyResolutionFailure:
        unresolved_packages: List<PackageName>
        conflicts: List<DependencyConflict>
        dependency_graph: PartialDependencyGraph

    - WasmValidationFailure:
        invalid_binaries: List<CrateName>
        validation_errors: List<ValidationError>

    - SecurityAuditFailure:
        critical_vulnerabilities: List<Vulnerability>
        affected_packages: List<PackageName>

    - TimeoutError:
        elapsed_seconds: u64
        timeout_seconds: u64
        stage: BuildStage

    - WasmIncompatibility:
        incompatible_dependencies: List<PackageName>
        reason: String

ENUM ErrorType:
  - CompilationError
  - DependencyError
  - ValidationError
  - SecurityError
  - TimeoutError
  - ConfigurationError
  - ToolchainError
  - IOError

STRUCTURE ErrorContext:
  pipeline_id: String
  agent: "BuildAgent"
  stage: BuildStage
  timestamp: DateTime
  rust_version: String
  cargo_version: String

ENUM BuildStage:
  - DependencyResolution
  - Vendoring
  - Compilation
  - Optimization
  - Validation
  - WatGeneration
```

---

## 5. Error Handling Strategy

### 5.1 Error Categories and Handling

```rust
ERROR HANDLING TAXONOMY:

1. RECOVERABLE ERRORS (Retry with backoff):
   - Network failures during dependency download
   - Transient cargo registry timeouts
   - Temporary file system issues
   - Race conditions in parallel builds

   STRATEGY:
     - Retry up to max_compilation_retries times
     - Exponential backoff: 2^retry_count seconds
     - Log each retry attempt
     - Return error after max retries exceeded

2. DEPENDENCY RESOLUTION ERRORS:
   - Version conflicts
   - Missing dependencies
   - WASM-incompatible dependencies
   - Circular dependencies

   STRATEGY:
     - Attempt automatic resolution (see resolve_incompatibilities)
     - Provide detailed conflict information
     - Suggest alternative versions
     - Allow partial compilation if some modules succeed
     - Document incompatible deps in output

3. COMPILATION ERRORS:
   - Rust syntax errors
   - Type errors
   - Borrow checker errors
   - Linking errors

   STRATEGY:
     - Capture full compiler output
     - Parse error messages with context
     - Provide file/line/column information
     - Include compiler suggestions
     - Allow parallel compilation of independent crates
     - Return partial artifacts for successful crates

4. VALIDATION ERRORS:
   - Invalid WASM format
   - Non-WASI imports
   - Memory layout issues
   - Unsupported WASM features

   STRATEGY:
     - Fail compilation for invalid WASM
     - Provide detailed validation report
     - Suggest fixes for common issues
     - Document WASI compatibility requirements

5. SECURITY ERRORS:
   - Checksum mismatches
   - Known vulnerabilities
   - License incompatibilities

   STRATEGY:
     - CRITICAL: Fail immediately on checksum mismatch
     - HIGH/CRITICAL vulns: Fail unless override flag set
     - MEDIUM/LOW vulns: Warn but allow compilation
     - Log all security findings

6. TIMEOUT ERRORS:
   - Compilation exceeds timeout
   - Dependency resolution timeout

   STRATEGY:
     - Kill process gracefully
     - Save partial state if possible
     - Report which stage timed out
     - Suggest increasing timeout

7. INFRASTRUCTURE ERRORS:
   - Missing tools (rustc, wasm-tools)
   - Insufficient disk space
   - Permission errors

   STRATEGY:
     - Validate prerequisites before starting
     - Provide clear error messages
     - Suggest installation/configuration steps
     - Fail fast on missing prerequisites
```

### 5.2 Error Recovery Pseudocode

```rust
ALGORITHM handle_compilation_error(
    error: CompilationError,
    config: BuildConfiguration,
    retry_count: u32
) -> ErrorHandlingDecision:

    LOG("Handling compilation error: {error.category}", WARN)

    MATCH error.category:
        ErrorCategory::DependencyResolutionError:
            // Try to resolve automatically
            IF retry_count == 0:  // Only try once
                resolution = attempt_dependency_resolution(error)
                IF resolution.is_success():
                    RETURN ErrorHandlingDecision::Retry
                ELSE:
                    RETURN ErrorHandlingDecision::FailWithPartial(resolution.partial_graph)
            ELSE:
                RETURN ErrorHandlingDecision::Fail

        ErrorCategory::WasmCompatibilityError:
            // Document incompatibility and suggest alternatives
            alternatives = find_wasm_alternatives(error.package)
            suggestion = RemediationSuggestion {
                description: "WASM-incompatible dependency detected",
                automated_fix_available: false,
                documentation_link: Some("https://portalis.docs/wasm-compat"),
                suggestions: alternatives
            }
            RETURN ErrorHandlingDecision::FailWithSuggestion(suggestion)

        ErrorCategory::TypeError, ErrorCategory::BorrowCheckError:
            // These are bugs in transpiler - cannot auto-fix
            // But provide detailed diagnostic
            diagnostic = format_detailed_diagnostic(error)
            suggestion = RemediationSuggestion {
                description: "Type error in generated Rust code - may indicate transpiler bug",
                automated_fix_available: false,
                documentation_link: Some("https://portalis.docs/transpiler-errors")
            }
            RETURN ErrorHandlingDecision::FailWithDiagnostic(diagnostic, suggestion)

        ErrorCategory::LinkError:
            // Try different linker flags
            IF retry_count < config.max_compilation_retries:
                LOG("Retrying with alternative linker configuration", INFO)
                RETURN ErrorHandlingDecision::RetryWithModifiedConfig(
                    alternative_linker_config()
                )
            ELSE:
                RETURN ErrorHandlingDecision::Fail

        _:
            // Unknown or unrecoverable error
            IF is_transient_error([error]) AND retry_count < config.max_compilation_retries:
                RETURN ErrorHandlingDecision::Retry
            ELSE:
                RETURN ErrorHandlingDecision::Fail


ENUM ErrorHandlingDecision:
    Retry                              // Retry same operation
    RetryWithModifiedConfig(Config)    // Retry with different settings
    FailWithPartial(PartialResult)     // Return partial artifacts
    FailWithSuggestion(Remediation)    // Fail with fix suggestion
    FailWithDiagnostic(Diagnostic, Remediation)
    Fail                               // Hard failure


ALGORITHM format_detailed_diagnostic(error: CompilationError) -> String:
    diagnostic = StringBuilder::new()

    diagnostic.append("COMPILATION ERROR\n")
    diagnostic.append("=================\n\n")

    // Error location
    IF error.file.is_some():
        diagnostic.append("File: {error.file.unwrap()}\n")
        IF error.line.is_some():
            diagnostic.append("Line: {error.line.unwrap()}\n")
        IF error.column.is_some():
            diagnostic.append("Column: {error.column.unwrap()}\n")

    // Error message
    diagnostic.append("\nError: {error.message}\n")

    // Code snippet
    IF error.snippet.is_some():
        diagnostic.append("\nCode:\n")
        diagnostic.append("-----\n")
        diagnostic.append("{error.snippet.unwrap()}\n")
        diagnostic.append("-----\n")

    // Compiler suggestion
    IF error.suggestion.is_some():
        diagnostic.append("\nSuggestion: {error.suggestion.unwrap()}\n")

    // Remediation
    IF error.remediation.automated_fix_available:
        diagnostic.append("\nAutomatic fix available\n")

    RETURN diagnostic.to_string()
```

### 5.3 Partial Success Handling

```rust
ALGORITHM handle_partial_compilation_success(
    successful_crates: List<WasmBinary>,
    failed_crates: List<CompilationError>,
    config: BuildConfiguration
) -> BuildAgentOutput:

    LOG("Partial compilation success: {successful_crates.len()} succeeded, {failed_crates.len()} failed", WARN)

    // Collect successful artifacts
    artifacts = CompilationArtifacts {
        wasm_binaries: successful_crates.into_map(),
        wat_files: Map::new(),
        build_manifest: generate_partial_manifest(successful_crates, failed_crates),
        errors: failed_crates,
        warnings: List::new()
    }

    // Generate WAT for successful builds
    IF config.generate_wat:
        FOR EACH binary IN successful_crates:
            wat = generate_wat_file(binary)
            artifacts.wat_files[binary.crate_name] = wat

    // Create output with partial status
    output = BuildAgentOutput {
        status: "partial_success",
        artifacts: artifacts,
        metadata: BuildMetadata {
            total_crates_built: successful_crates.len(),
            successful_builds: successful_crates.len(),
            failed_builds: failed_crates.len(),
            total_wasm_size_bytes: successful_crates.sum(|b| b.size_bytes)
        },
        diagnostics: Diagnostics {
            compilation_errors: failed_crates
        }
    }

    RETURN output
```

---

## 6. London School TDD Test Points

### 6.1 Test Hierarchy

```
ACCEPTANCE TESTS (Outside-In)
├─ Happy Path Tests
│  ├─ Test: Compile simple Rust crate to WASM
│  ├─ Test: Compile multi-crate workspace to WASM
│  ├─ Test: Generate WAT debug files
│  └─ Test: Vendor and build offline
│
├─ Error Path Tests
│  ├─ Test: Handle compilation errors gracefully
│  ├─ Test: Handle dependency conflicts
│  ├─ Test: Handle WASM validation failures
│  └─ Test: Handle timeout scenarios
│
└─ Edge Case Tests
   ├─ Test: Build with no dependencies
   ├─ Test: Build with circular dependencies (should fail)
   ├─ Test: Build with WASM-incompatible deps
   └─ Test: Partial compilation success

INTEGRATION TESTS (Agent Contracts)
├─ Test: BuildAgent receives correct input from TranspilerAgent
├─ Test: BuildAgent produces correct output for TestAgent
├─ Test: Cargo toolchain integration
├─ Test: wasm-tools integration
├─ Test: Dependency registry integration
└─ Test: File system operations

UNIT TESTS (Component Isolation)
├─ DependencyResolution
│  ├─ Test: Parse Cargo.toml correctly
│  ├─ Test: Resolve version requirements
│  ├─ Test: Detect version conflicts
│  ├─ Test: Build dependency graph
│  └─ Test: Topological sort
│
├─ Compilation
│  ├─ Test: Build cargo command correctly
│  ├─ Test: Parse compiler output
│  ├─ Test: Handle compilation retries
│  └─ Test: Collect WASM binaries
│
├─ WasmValidation
│  ├─ Test: Validate WASM format
│  ├─ Test: Extract imports
│  ├─ Test: Extract exports
│  ├─ Test: Validate WASI compliance
│  └─ Test: Detect unsupported features
│
├─ Optimization
│  ├─ Test: Run wasm-opt with correct flags
│  ├─ Test: Strip custom sections
│  ├─ Test: Calculate size reduction
│  └─ Test: Preserve functionality after optimization
│
├─ Vendoring
│  ├─ Test: Download crates correctly
│  ├─ Test: Verify checksums
│  ├─ Test: Generate vendor config
│  └─ Test: Create vendor archive
│
└─ ErrorHandling
   ├─ Test: Categorize errors correctly
   ├─ Test: Retry transient errors
   ├─ Test: Generate remediation suggestions
   └─ Test: Handle partial success
```

### 6.2 Mock/Stub Strategy

```rust
TEST DOUBLES FOR BUILD AGENT:

1. MOCKS (Behavior Verification):

   MOCK CargoExecutor:
     - Verifies cargo commands are constructed correctly
     - Verifies correct flags for debug/release
     - Verifies target is wasm32-wasi
     - Tracks number of invocations

     EXPECTATIONS:
       - expect_build_called_with(args: List<String>)
       - expect_vendor_called_once()
       - verify_all_expectations()

   MOCK DependencyRegistry:
     - Verifies dependency queries
     - Verifies checksum verification
     - Tracks download requests

     EXPECTATIONS:
       - expect_fetch_crate(name: String, version: Version)
       - expect_verify_checksum(name: String, checksum: String)

   MOCK FileSystem:
     - Verifies file writes
     - Verifies directory creation
     - Tracks file system operations

     EXPECTATIONS:
       - expect_write_file(path: Path, content: Bytes)
       - expect_create_directory(path: Path)

2. STUBS (State Verification):

   STUB CompilerOutput:
     - Returns pre-defined compiler messages
     - Simulates success/failure scenarios
     - Provides test fixtures

     CONFIGURATIONS:
       - with_success_output() -> CompilerOutput
       - with_error_output(errors: List<String>) -> CompilerOutput
       - with_warnings(warnings: List<String>) -> CompilerOutput

   STUB WasmBinaryReader:
     - Returns mock WASM module data
     - Provides test imports/exports

     CONFIGURATIONS:
       - with_valid_wasm() -> WasmBinary
       - with_invalid_format() -> WasmBinary
       - with_non_wasi_imports() -> WasmBinary

   STUB DependencyGraphBuilder:
     - Returns pre-built dependency graphs
     - Simulates various dependency scenarios

     CONFIGURATIONS:
       - with_simple_graph() -> DependencyGraph
       - with_conflict() -> DependencyGraph
       - with_circular_dependency() -> DependencyGraph

3. FAKES (Working Implementations):

   FAKE InMemoryFileSystem:
     - Fully functional in-memory file system
     - Allows testing without disk I/O
     - Supports all file operations

   FAKE MockRegistry:
     - In-memory crate registry
     - Pre-populated with test crates
     - Deterministic behavior

   FAKE TestWasmValidator:
     - Validates test WASM modules
     - Configurable validation rules
     - Deterministic results
```

### 6.3 Key Test Scenarios

```rust
ACCEPTANCE TEST SCENARIO 1: Successful Simple Build
GIVEN:
  - A valid single-crate Rust workspace
  - All dependencies are WASM-compatible
  - Build configuration set to release mode
  - wasm32-wasi target installed

WHEN:
  - BuildAgent.build_wasm_modules() is called

THEN:
  - EXPECT status == "success"
  - EXPECT wasm_binaries.len() == 1
  - EXPECT wasm_binary is valid and WASI-compliant
  - EXPECT wat_file is generated
  - EXPECT build_manifest is created
  - EXPECT no compilation errors
  - EXPECT build completes in < 60 seconds (for small crate)

MOCKS:
  - CargoExecutor: verify build called with correct args
  - FileSystem: verify output files written


ACCEPTANCE TEST SCENARIO 2: Dependency Conflict Resolution
GIVEN:
  - A Rust workspace with dependency conflicts
  - Package A requires serde 1.0.100
  - Package B requires serde 1.0.150
  - Both versions are compatible

WHEN:
  - BuildAgent.build_wasm_modules() is called

THEN:
  - EXPECT dependency resolution succeeds
  - EXPECT unified version (1.0.150) is used
  - EXPECT status == "success"
  - EXPECT dependency_graph shows single serde version

MOCKS:
  - DependencyRegistry: stub with conflicting versions
  - CargoExecutor: verify correct resolution


ACCEPTANCE TEST SCENARIO 3: WASM Validation Failure
GIVEN:
  - Rust code compiles successfully
  - Generated WASM has non-WASI imports

WHEN:
  - BuildAgent.build_wasm_modules() is called

THEN:
  - EXPECT status == "failure"
  - EXPECT validation_errors contains "Non-WASI import"
  - EXPECT specific import is identified
  - EXPECT remediation suggestion provided

MOCKS:
  - WasmValidator: stub to return non-WASI imports


ACCEPTANCE TEST SCENARIO 4: Partial Compilation Success
GIVEN:
  - Multi-crate workspace with 3 crates
  - Crate A and B compile successfully
  - Crate C has type errors

WHEN:
  - BuildAgent.build_wasm_modules() is called

THEN:
  - EXPECT status == "partial_success"
  - EXPECT wasm_binaries.len() == 2
  - EXPECT successful_builds == 2
  - EXPECT failed_builds == 1
  - EXPECT errors contain Crate C errors
  - EXPECT partial artifacts returned

MOCKS:
  - CargoExecutor: return success for A, B; failure for C


UNIT TEST SCENARIO 1: Version Conflict Detection
GIVEN:
  - DependencyGraph with package "tokio"
  - Version 1.0.0 required by package A
  - Version 2.0.0 required by package B

WHEN:
  - resolve_version("tokio", "^1.0", graph) is called

THEN:
  - EXPECT Error(VersionConflict)
  - EXPECT conflict details show both versions
  - EXPECT conflict resolution strategy suggested

MOCKS:
  - Mock DependencyGraph with pre-populated nodes


UNIT TEST SCENARIO 2: Transient Error Retry
GIVEN:
  - Compilation fails with "network error"
  - max_retries = 3
  - retry_count = 0

WHEN:
  - compile_to_wasm() is called

THEN:
  - EXPECT 3 retry attempts
  - EXPECT exponential backoff between retries
  - EXPECT Error(CompilationFailed) after max retries

MOCKS:
  - Mock CargoExecutor: return network error 3 times


UNIT TEST SCENARIO 3: Checksum Verification
GIVEN:
  - Downloaded crate data with SHA-256 hash
  - Expected checksum from registry

WHEN:
  - verify_checksum(data, expected) is called

THEN:
  - IF checksums match: EXPECT Ok(true)
  - IF checksums differ: EXPECT Err(ChecksumMismatch)

STUBS:
  - Stub crate data with known hash


INTEGRATION TEST SCENARIO 1: Cargo Integration
GIVEN:
  - Real Rust toolchain installed
  - Simple test crate

WHEN:
  - Actual cargo build is executed

THEN:
  - EXPECT WASM binary produced
  - EXPECT binary is valid WASM
  - EXPECT wasm32-wasi target used

NOTE: Uses real cargo, not mocked


INTEGRATION TEST SCENARIO 2: Contract with TranspilerAgent
GIVEN:
  - Mock TranspilerAgent output

WHEN:
  - BuildAgent receives input

THEN:
  - EXPECT all required fields present
  - EXPECT Rust workspace structure valid
  - EXPECT Cargo.toml files exist

MOCKS:
  - Mock TranspilerAgent with contract-compliant output
```

### 6.4 Contract Tests

```rust
CONTRACT TEST: Input from TranspilerAgent
TEST verify_input_contract_from_transpiler():
    // Setup
    mock_transpiler = MockTranspilerAgent()
    transpiler_output = mock_transpiler.generate_output()

    // Convert to BuildAgent input
    build_input = BuildAgentInput::from_transpiler_output(transpiler_output)

    // Verify contract
    ASSERT build_input.rust_workspace.root_path.exists()
    ASSERT build_input.rust_workspace.crates.len() > 0

    FOR EACH crate IN build_input.rust_workspace.crates:
        cargo_toml = crate.root.join("Cargo.toml")
        ASSERT cargo_toml.exists()
        ASSERT cargo_toml.is_valid_toml()

    ASSERT build_input.context.previous_agent == "TranspilerAgent"


CONTRACT TEST: Output to TestAgent
TEST verify_output_contract_to_test_agent():
    // Setup
    build_agent = BuildAgent::new()
    test_input = create_test_rust_workspace()

    // Execute
    build_output = build_agent.build_wasm_modules(test_input)

    // Verify contract
    ASSERT build_output.status IN ["success", "partial_success", "failure"]

    IF build_output.status == "success":
        ASSERT build_output.artifacts.wasm_binaries.len() > 0
        ASSERT build_output.diagnostics.compilation_errors.is_empty()

        FOR EACH (name, binary) IN build_output.artifacts.wasm_binaries:
            ASSERT binary.file_path.exists()
            ASSERT binary.wasi_compliant == true
            ASSERT binary.size_bytes > 0

    // Verify TestAgent can consume output
    mock_test_agent = MockTestAgent()
    ASSERT_NO_ERROR(
        mock_test_agent.receive_input(build_output)
    )


CONTRACT TEST: Error Output Contract
TEST verify_error_output_contract():
    // Setup with failure scenario
    build_agent = BuildAgent::new()
    invalid_input = create_invalid_rust_workspace()

    // Execute
    result = build_agent.build_wasm_modules(invalid_input)

    // Verify error structure
    ASSERT result.status == "failure"
    ASSERT result.diagnostics.compilation_errors.len() > 0

    FOR EACH error IN result.diagnostics.compilation_errors:
        ASSERT error.message.len() > 0
        ASSERT error.category IS ErrorCategory
        ASSERT error.remediation IS RemediationStrategy

        IF error.file.is_some():
            ASSERT error.file.unwrap().is_absolute()
```

### 6.5 Coverage Requirements

```
COVERAGE TARGETS FOR BUILD AGENT:

Line Coverage: >85%
  - All main algorithms covered
  - All error paths tested
  - All optimization paths tested

Branch Coverage: >80%
  - All conditional logic tested
  - All error handling branches covered
  - All retry logic paths tested

Contract Coverage: 100%
  - Input contract fully tested
  - Output contract fully tested
  - Error contract fully tested
  - Integration contracts fully tested

Mutation Testing: >75%
  - Critical logic mutations detected
  - Error handling mutations detected
  - Validation logic mutations detected

CRITICAL PATHS REQUIRING 100% COVERAGE:
  - Dependency resolution algorithm
  - WASM validation logic
  - Checksum verification
  - Error categorization and handling
  - Retry logic for transient errors
  - Contract validation
```

---

## 7. Performance Considerations

### 7.1 Compilation Performance

```rust
PERFORMANCE OPTIMIZATIONS:

1. PARALLEL COMPILATION:
   - Use cargo's parallel compilation (--jobs flag)
   - Build independent crates in parallel
   - Default: Use all available CPU cores

   PSEUDOCODE:
     IF workspace.crates.len() > 1:
         dependency_order = dependency_graph.topological_sort()
         parallel_groups = group_by_dependency_level(dependency_order)

         FOR EACH group IN parallel_groups:
             // Compile all crates in group in parallel
             futures = group.map(|crate| compile_crate_async(crate))
             results = await_all(futures)

2. INCREMENTAL COMPILATION:
   - Preserve target directory between builds
   - Use cargo's incremental compilation feature
   - Cache dependency compilation results

   PSEUDOCODE:
     cargo_env.set("CARGO_INCREMENTAL", "1")
     cargo_env.set("CARGO_TARGET_DIR", persistent_cache_dir)

3. DEPENDENCY CACHING:
   - Cache downloaded crates locally
   - Use sccache for compilation caching
   - Reuse vendored dependencies

   PSEUDOCODE:
     IF cache.contains(dependency):
         RETURN cache.get(dependency)
     ELSE:
         dep = download_dependency()
         cache.insert(dependency, dep)
         RETURN dep

4. OPTIMIZATION PIPELINE:
   - Run wasm-opt in parallel for multiple binaries
   - Pipeline optimization stages
   - Use faster optimization levels for debug builds

   PSEUDOCODE:
     IF config.profile == Debug:
         opt_level = O1  // Fast optimization
     ELSE:
         opt_level = Os  // Size optimization
```

### 7.2 Build Time Targets

```
PERFORMANCE TARGETS:

Script Mode (single crate, <500 LOC):
  - Dependency resolution: <5 seconds
  - Compilation: <15 seconds
  - Optimization: <5 seconds
  - Validation: <2 seconds
  - Total: <30 seconds

Library Mode (multi-crate, <5000 LOC):
  - Dependency resolution: <30 seconds
  - Compilation: <5 minutes
  - Optimization: <1 minute
  - Validation: <30 seconds
  - Total: <7 minutes

Large Library (50,000 LOC):
  - Total: <30 minutes (with parallelization)
```

---

## 8. Security Considerations

```rust
SECURITY MEASURES:

1. DEPENDENCY VERIFICATION:
   - ALWAYS verify checksums
   - NEVER skip checksum verification
   - Use SHA-256 for all checksums
   - Fail on checksum mismatch

2. VULNERABILITY SCANNING:
   - Query RustSec advisory database
   - Fail on CRITICAL vulnerabilities
   - Warn on HIGH vulnerabilities
   - Log MEDIUM/LOW vulnerabilities

3. SAFE CODE GENERATION:
   - Audit generated WASM for unsafe patterns
   - Document any unsafe blocks
   - Validate WASI sandbox compliance
   - Ensure no arbitrary code execution

4. SUPPLY CHAIN SECURITY:
   - Vendor dependencies for reproducibility
   - Lock dependency versions
   - Audit dependency licenses
   - Maintain allowlist of approved crates (SHOULD)

5. BUILD ISOLATION:
   - Run builds in isolated environment
   - Limit network access during build
   - Prevent path traversal in file operations
   - Sanitize all file paths
```

---

## Document Status

**SPARC Phase 2 (Pseudocode): BUILD AGENT COMPLETE**

This document provides comprehensive pseudocode for the Build Agent, covering:
- Complete data structures for build configuration, dependencies, and artifacts
- Detailed algorithms for compilation, optimization, and validation
- Comprehensive error handling strategies
- Full input/output contracts
- London School TDD test points with 85%+ coverage targets
- Performance and security considerations

### Next Steps
1. **Phase 3 (Architecture)**: Detailed component design and interfaces
2. **Phase 4 (Refinement)**: Implementation and iterative improvement
3. **Phase 5 (Completion)**: Final implementation and validation

---

**END OF BUILD AGENT PSEUDOCODE**
