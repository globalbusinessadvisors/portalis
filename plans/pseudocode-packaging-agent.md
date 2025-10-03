# PORTALIS: Packaging Agent Pseudocode
## SPARC Phase 2: Pseudocode

**Agent:** Packaging Agent (Agent 7 of 7)
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

The Packaging Agent is responsible for transforming validated WASM binaries into production-ready, enterprise-grade deployment artifacts. It creates NIM (NVIDIA Inference Microservices) containers, integrates with Triton Inference Server, ensures Omniverse compatibility, and generates comprehensive distribution packages.

### 1.2 Responsibilities

**Core Responsibilities (MUST):**
- Generate NIM service containers with Docker/containerization
- Create Triton model configuration files and register endpoints
- Validate Omniverse compatibility of WASM modules
- Package all artifacts into distributable archives
- Generate service API documentation (OpenAPI/gRPC schemas)
- Create health check endpoints for containerized services
- Generate comprehensive manifests and README files
- Publish artifacts to distribution repositories

**Extended Responsibilities (SHOULD):**
- Configure autoscaling parameters for Triton
- Generate Omniverse integration documentation
- Provide example Omniverse simulation scenarios
- Optimize container image layers
- Generate deployment scripts and Kubernetes manifests

### 1.3 Requirements Mapping

| Requirement | Description | Priority |
|-------------|-------------|----------|
| FR-2.7.1 | NIM Service Generation (containers, health checks, API docs) | MUST |
| FR-2.7.2 | Triton Integration (configs, endpoints, batch/interactive) | MUST |
| FR-2.7.3 | Omniverse Compatibility (WASM validation, integration docs) | MUST |
| FR-2.7.4 | Distribution Packaging (archives, manifests, README) | MUST |
| NFR-3.2.1 | Fault tolerance, partial outputs | MUST |
| NFR-3.3.2 | Output safety, container security | MUST |
| NFR-3.4.2 | Documentation generation | MUST |

### 1.4 Integration Points

**Inputs:**
- WASM binaries from Build Agent
- Test results and parity reports from Test Agent
- Build metadata and performance benchmarks
- Translation specifications from earlier pipeline stages

**Outputs:**
- NIM container images (Docker)
- Triton model configurations
- Distribution archives (tarballs)
- API documentation (OpenAPI/gRPC)
- Deployment manifests and scripts
- Omniverse integration packages

**External Tools:**
- Docker/Podman (container creation)
- Triton Inference Server
- NVIDIA Container Toolkit
- Omniverse Kit SDK
- tar/gzip (archive creation)

---

## 2. Data Structures

### 2.1 Package Configuration

```rust
STRUCTURE PackagingConfiguration:
    // Package identity
    package_name: String               // Name of the service/package
    package_version: SemanticVersion   // Version number
    description: String                // Package description

    // Source artifacts
    wasm_binaries: List<WasmBinaryPath>
    test_results: Path                 // Test results from Test Agent
    build_metadata: BuildManifest      // From Build Agent
    parity_report: Path                // Conformance report

    // Output configuration
    output_directory: Path             // Where to place packages

    // Target platforms
    enable_nim_packaging: bool         // Default: true
    enable_triton_integration: bool    // Default: true
    enable_omniverse_packaging: bool   // Default: true
    enable_distribution_archive: bool  // Default: true

    // Container configuration
    container_config: ContainerConfiguration

    // Triton configuration
    triton_config: TritonConfiguration

    // Omniverse configuration
    omniverse_config: OmniverseConfiguration

    // Distribution configuration
    distribution_config: DistributionConfiguration

    // Documentation generation
    generate_api_docs: bool            // Default: true
    api_doc_format: ApiDocFormat       // OpenAPI, gRPC, both

    // Publishing
    publish_to_registry: bool          // Default: false
    registry_url: Option<String>       // Container registry

    // Metadata
    pipeline_id: String                // For tracking
    mode: OperationMode                // Script or Library

ENUM OperationMode:
    Script
    Library

ENUM ApiDocFormat:
    OpenAPI
    GRPC
    Both
```

### 2.2 Container Configuration

```rust
STRUCTURE ContainerConfiguration:
    // Base image
    base_image: String                 // e.g., "nvcr.io/nvidia/nim:24.01"
    runtime: ContainerRuntime          // Docker, Podman

    // Image metadata
    image_name: String                 // Container image name
    image_tag: String                  // Container image tag
    labels: Map<String, String>        // Container labels

    // WASM runtime
    wasm_runtime: WasmRuntime          // wasmtime, wasmer
    wasm_runtime_version: String

    // Health checks
    health_check_endpoint: String      // "/health"
    health_check_interval: Duration    // Default: 30s
    health_check_timeout: Duration     // Default: 5s
    health_check_retries: u32          // Default: 3

    // Service ports
    service_port: u16                  // Default: 8000
    metrics_port: u16                  // Default: 9090

    // Resource limits
    memory_limit: Option<String>       // e.g., "2Gi"
    cpu_limit: Option<String>          // e.g., "2000m"
    gpu_required: bool                 // Default: false
    gpu_count: Option<u32>             // Number of GPUs

    // Environment variables
    environment_vars: Map<String, String>

    // Volume mounts
    volumes: List<VolumeMount>

    // Entrypoint
    entrypoint: List<String>           // Container entrypoint command
    command_args: List<String>         // Additional args

    // Security
    run_as_non_root: bool              // Default: true
    read_only_root_filesystem: bool    // Default: true
    drop_capabilities: List<String>    // Capabilities to drop

ENUM ContainerRuntime:
    Docker
    Podman

ENUM WasmRuntime:
    Wasmtime
    Wasmer
    WasmEdge

STRUCTURE VolumeMount:
    host_path: String
    container_path: String
    read_only: bool
```

### 2.3 Triton Configuration

```rust
STRUCTURE TritonConfiguration:
    // Model repository
    model_repository_path: Path        // Where Triton models live

    // Model metadata
    model_name: String
    model_version: u32                 // Model version number
    platform: TritonPlatform           // "wasm_abi" or custom

    // Instance configuration
    max_batch_size: u32                // 0 for no batching
    instance_groups: List<InstanceGroup>

    // Input/Output specification
    inputs: List<ModelInput>
    outputs: List<ModelOutput>

    // Scheduling
    dynamic_batching: Option<DynamicBatchingConfig>
    sequence_batching: Option<SequenceBatchingConfig>

    // Backend parameters
    backend_parameters: Map<String, String>

    // Optimization
    optimization: OptimizationConfig

    // Versioning policy
    version_policy: VersionPolicy

ENUM TritonPlatform:
    WasmABI                            // Custom WASM backend
    Python                             // Python backend (wrapping WASM)
    Custom(String)                     // Custom backend name

STRUCTURE InstanceGroup:
    name: String
    kind: InstanceKind                 // CPU, GPU
    count: u32                         // Number of instances
    gpus: Option<List<u32>>            // GPU IDs if GPU kind

ENUM InstanceKind:
    CPU
    GPU

STRUCTURE ModelInput:
    name: String
    data_type: DataType
    dims: List<i64>                    // -1 for dynamic dimensions
    format: Option<TensorFormat>
    optional: bool                     // Default: false

STRUCTURE ModelOutput:
    name: String
    data_type: DataType
    dims: List<i64>
    label_filename: Option<String>

ENUM DataType:
    BOOL
    UINT8
    UINT16
    UINT32
    UINT64
    INT8
    INT16
    INT32
    INT64
    FP16
    FP32
    FP64
    STRING
    BYTES

ENUM TensorFormat:
    Linear
    NHWC
    NCHW

STRUCTURE DynamicBatchingConfig:
    preferred_batch_size: List<u32>
    max_queue_delay_microseconds: u64
    preserve_ordering: bool
    priority_levels: u32
    default_priority_level: u32

STRUCTURE SequenceBatchingConfig:
    max_sequence_idle_microseconds: u64
    control_inputs: List<ControlInput>
    states: List<SequenceState>

STRUCTURE ControlInput:
    name: String
    control_type: ControlType

ENUM ControlType:
    Start
    Ready
    End
    CorrelationId

STRUCTURE SequenceState:
    name: String
    data_type: DataType
    dims: List<i64>

STRUCTURE OptimizationConfig:
    graph_optimization: bool
    cuda_graphs: bool                  // For GPU deployments
    gather_kernel_buffer_threshold: u64

STRUCTURE VersionPolicy:
    policy: VersionPolicyType
    versions: Option<List<u32>>        // Specific versions

ENUM VersionPolicyType:
    Latest(u32)                        // Latest N versions
    All
    Specific                           // Specific versions list
```

### 2.4 Omniverse Configuration

```rust
STRUCTURE OmniverseConfiguration:
    // Extension metadata
    extension_name: String
    extension_version: String
    extension_description: String

    // WASM module paths
    wasm_modules: List<WasmModulePath>

    // Omniverse Kit compatibility
    required_kit_version: String       // e.g., "105.0"
    compatible_kit_versions: List<String>

    // Extension dependencies
    dependencies: List<OmniExtensionDep>

    // Integration points
    integration_type: OmniverseIntegrationType

    // Documentation
    generate_integration_docs: bool
    example_scenarios: List<ExampleScenario>

    // Validation
    validate_wasm_compatibility: bool  // Default: true

    // Distribution
    package_as_extension: bool         // Default: true
    extension_icon: Option<Path>
    extension_preview: Option<Path>

STRUCTURE WasmModulePath:
    module_name: String
    file_path: Path
    entry_point: String                // Main function name
    description: String

STRUCTURE OmniExtensionDep:
    extension_name: String
    version_requirement: String        // e.g., "^1.0.0"

ENUM OmniverseIntegrationType:
    Simulation                         // Physics/simulation extension
    Renderer                           // Rendering extension
    Tool                               // Utility/tool extension
    DataProcessor                      // Data processing extension
    Custom(String)

STRUCTURE ExampleScenario:
    name: String
    description: String
    scenario_file: Option<Path>        // USD scene file
    documentation: String
```

### 2.5 Distribution Package

```rust
STRUCTURE DistributionConfiguration:
    // Archive settings
    archive_format: ArchiveFormat
    compression_level: u32             // 0-9 for gzip

    // Contents
    include_source_rust: bool          // Default: true
    include_test_reports: bool         // Default: true
    include_benchmarks: bool           // Default: true
    include_documentation: bool        // Default: true

    // Manifest
    generate_manifest: bool            // Default: true
    manifest_format: ManifestFormat

    // README
    generate_readme: bool              // Default: true
    readme_format: ReadmeFormat
    readme_template: Option<Path>

    // Licensing
    license_file: Option<Path>
    license_type: Option<String>       // e.g., "Apache-2.0"

    // Checksums
    generate_checksums: bool           // Default: true
    checksum_algorithms: List<ChecksumAlgorithm>

    // Publishing
    publish_metadata: PublishMetadata

ENUM ArchiveFormat:
    TarGz                              // .tar.gz
    TarBz2                             // .tar.bz2
    TarXz                              // .tar.xz
    Zip                                // .zip

ENUM ManifestFormat:
    JSON
    YAML
    TOML

ENUM ReadmeFormat:
    Markdown
    PlainText
    RestructuredText

ENUM ChecksumAlgorithm:
    SHA256
    SHA512
    Blake3

STRUCTURE PublishMetadata:
    author: String
    maintainer: String
    homepage: Option<String>
    repository: Option<String>
    documentation_url: Option<String>
    keywords: List<String>
    categories: List<String>
```

### 2.6 Packaging Artifacts

```rust
STRUCTURE PackagingArtifacts:
    // Container artifacts
    nim_containers: List<ContainerArtifact>

    // Triton artifacts
    triton_models: List<TritonModelArtifact>

    // Omniverse artifacts
    omniverse_packages: List<OmniversePackageArtifact>

    // Distribution artifacts
    distribution_archives: List<DistributionArchive>

    // Documentation
    api_documentation: List<ApiDocumentation>
    integration_guides: List<IntegrationGuide>

    // Deployment resources
    deployment_manifests: List<DeploymentManifest>

    // Metadata
    package_manifest: PackageManifest
    checksums: Map<Path, String>

    FUNCTION total_artifacts_count() -> u32
    FUNCTION total_size_bytes() -> u64
    FUNCTION validate_completeness() -> Result<(), ValidationError>

STRUCTURE ContainerArtifact:
    image_name: String
    image_tag: String
    image_id: String                   // Docker image ID
    image_size_bytes: u64
    dockerfile_path: Path
    image_archive_path: Option<Path>   // Exported tar archive

    // Container metadata
    created_at: DateTime
    base_image: String
    layers: List<ImageLayer>

    // Health check info
    health_check_config: HealthCheckConfig

    // Service endpoints
    exposed_ports: List<u16>
    environment_vars: Map<String, String>

    // Checksums
    image_digest: String               // SHA256 digest

STRUCTURE ImageLayer:
    layer_id: String
    size_bytes: u64
    created_by: String                 // Command that created layer
    comment: Option<String>

STRUCTURE HealthCheckConfig:
    endpoint: String
    interval_seconds: u64
    timeout_seconds: u64
    retries: u32

STRUCTURE TritonModelArtifact:
    model_name: String
    model_version: u32
    model_repository_path: Path

    // Configuration files
    config_pbtxt_path: Path            // config.pbtxt

    // Model files
    wasm_model_paths: List<Path>

    // Metadata
    platform: String
    max_batch_size: u32
    instance_count: u32

    // Validation
    validated: bool
    validation_report: Option<String>

STRUCTURE OmniversePackageArtifact:
    extension_name: String
    extension_version: String
    package_path: Path                 // .zip or directory

    // Extension manifest
    extension_toml_path: Path

    // WASM modules
    wasm_modules: List<WasmModulePath>

    // Documentation
    readme_path: Path
    examples_path: Option<Path>

    // Compatibility
    kit_version_compatibility: String
    validated: bool

STRUCTURE DistributionArchive:
    archive_name: String
    archive_path: Path
    archive_format: ArchiveFormat
    compressed_size_bytes: u64
    uncompressed_size_bytes: u64

    // Contents inventory
    contents: ArchiveInventory

    // Checksums
    checksums: Map<ChecksumAlgorithm, String>

    // Manifest
    manifest_path: Path

    // README
    readme_path: Path

STRUCTURE ArchiveInventory:
    total_files: u32
    total_directories: u32
    file_types: Map<String, u32>       // Extension -> count

    // Content organization
    directories: List<DirectoryEntry>

STRUCTURE DirectoryEntry:
    name: String
    path: String
    description: String
    file_count: u32

STRUCTURE ApiDocumentation:
    format: ApiDocFormat
    file_path: Path

    // For OpenAPI
    openapi_spec: Option<OpenApiSpec>

    // For gRPC
    grpc_proto: Option<Path>
    grpc_descriptor: Option<Path>

STRUCTURE OpenApiSpec:
    version: String                    // "3.0.0"
    title: String
    description: String
    servers: List<ServerConfig>
    paths: Map<String, PathItem>
    components: Components

STRUCTURE ServerConfig:
    url: String
    description: String

STRUCTURE PathItem:
    path: String
    methods: Map<HttpMethod, Operation>

ENUM HttpMethod:
    GET
    POST
    PUT
    DELETE
    PATCH

STRUCTURE Operation:
    summary: String
    description: String
    parameters: List<Parameter>
    request_body: Option<RequestBody>
    responses: Map<StatusCode, Response>

STRUCTURE Parameter:
    name: String
    location: ParameterLocation
    description: String
    required: bool
    schema: JsonSchema

ENUM ParameterLocation:
    Path
    Query
    Header
    Cookie

STRUCTURE RequestBody:
    description: String
    content: Map<MediaType, MediaTypeObject>
    required: bool

STRUCTURE Response:
    description: String
    content: Map<MediaType, MediaTypeObject>

STRUCTURE MediaTypeObject:
    schema: JsonSchema
    examples: Map<String, Example>

STRUCTURE Components:
    schemas: Map<String, JsonSchema>
    responses: Map<String, Response>
    parameters: Map<String, Parameter>

STRUCTURE JsonSchema:
    schema_type: String                // "object", "array", etc.
    properties: Map<String, JsonSchema>
    required: List<String>
    description: Option<String>

STRUCTURE IntegrationGuide:
    title: String
    format: DocumentFormat
    file_path: Path
    content_sections: List<String>

ENUM DocumentFormat:
    Markdown
    HTML
    PDF

STRUCTURE DeploymentManifest:
    manifest_type: DeploymentType
    file_path: Path

    // For Kubernetes
    kubernetes_resources: Option<List<K8sResource>>

    // For Docker Compose
    docker_compose_services: Option<Map<String, ComposeService>>

ENUM DeploymentType:
    Kubernetes
    DockerCompose
    Helm
    Terraform

STRUCTURE K8sResource:
    api_version: String
    kind: String                       // "Deployment", "Service", etc.
    metadata: Map<String, String>
    spec: Map<String, Value>

STRUCTURE ComposeService:
    image: String
    ports: List<String>
    environment: Map<String, String>
    volumes: List<String>
    depends_on: List<String>

STRUCTURE PackageManifest:
    version: String                    // Manifest format version
    package_name: String
    package_version: SemanticVersion
    created_at: DateTime

    // Pipeline metadata
    pipeline_id: String
    mode: OperationMode

    // Artifacts inventory
    artifacts: ArtifactsInventory

    // Input metadata
    source_info: SourceInfo

    // Build metadata
    build_info: BuildInfo

    // Test metadata
    test_info: TestInfo

    // Performance metrics
    performance_metrics: PerformanceMetrics

    FUNCTION to_json() -> String
    FUNCTION to_yaml() -> String
    FUNCTION validate() -> Result<(), Error>

STRUCTURE ArtifactsInventory:
    nim_containers: List<ArtifactEntry>
    triton_models: List<ArtifactEntry>
    omniverse_packages: List<ArtifactEntry>
    distribution_archives: List<ArtifactEntry>
    documentation: List<ArtifactEntry>

STRUCTURE ArtifactEntry:
    name: String
    path: String                       // Relative path in package
    size_bytes: u64
    checksum_sha256: String
    artifact_type: String

STRUCTURE SourceInfo:
    original_python_source: String
    python_version: String
    lines_of_code: u32
    input_mode: String                 // "script" or "library"

STRUCTURE BuildInfo:
    rust_version: String
    cargo_version: String
    wasm_binaries: List<WasmBinaryInfo>
    build_time_seconds: f64
    total_wasm_size_bytes: u64

STRUCTURE WasmBinaryInfo:
    name: String
    size_bytes: u64
    wasi_compliant: bool
    exports: List<String>

STRUCTURE TestInfo:
    total_tests: u32
    passed_tests: u32
    failed_tests: u32
    test_coverage_percent: f64
    parity_score_percent: f64

STRUCTURE PerformanceMetrics:
    benchmark_results: List<BenchmarkResult>
    speedup_factor: f64                // vs original Python
    memory_usage_bytes: u64

STRUCTURE BenchmarkResult:
    name: String
    python_time_ms: f64
    rust_wasm_time_ms: f64
    speedup_factor: f64
```

---

## 3. Core Algorithms

### 3.1 Main Packaging Orchestration

```rust
ALGORITHM package_artifacts(
    wasm_binaries: List<WasmBinary>,
    config: PackagingConfiguration
) -> Result<PackagingArtifacts, PackagingError>:

    LOG("Starting artifact packaging orchestration", INFO)
    start_time = current_timestamp()

    artifacts = PackagingArtifacts::new()

    TRY:
        // Step 1: Validate inputs
        LOG("Validating input artifacts", DEBUG)
        validation_result = validate_input_artifacts(wasm_binaries, config)
        IF validation_result.is_error():
            RETURN Error(InvalidInput(validation_result.errors))

        // Step 2: NIM Service Generation
        IF config.enable_nim_packaging:
            LOG("Generating NIM container services", INFO)
            nim_result = generate_nim_containers(
                wasm_binaries,
                config.container_config,
                config.package_name,
                config.package_version
            )

            IF nim_result.is_ok():
                artifacts.nim_containers = nim_result.containers
                LOG("Generated {nim_result.containers.len()} NIM containers", INFO)
            ELSE:
                LOG("NIM container generation failed: {nim_result.error}", ERROR)
                IF config.fail_on_partial_errors:
                    RETURN Error(NimGenerationFailed(nim_result.error))

        // Step 3: Triton Integration
        IF config.enable_triton_integration:
            LOG("Configuring Triton inference endpoints", INFO)
            triton_result = configure_triton_models(
                wasm_binaries,
                config.triton_config,
                config.package_name
            )

            IF triton_result.is_ok():
                artifacts.triton_models = triton_result.models
                LOG("Configured {triton_result.models.len()} Triton models", INFO)
            ELSE:
                LOG("Triton configuration failed: {triton_result.error}", WARN)
                IF config.fail_on_partial_errors:
                    RETURN Error(TritonConfigFailed(triton_result.error))

        // Step 4: Omniverse Packaging
        IF config.enable_omniverse_packaging:
            LOG("Creating Omniverse-compatible packages", INFO)
            omniverse_result = package_for_omniverse(
                wasm_binaries,
                config.omniverse_config,
                config.package_name,
                config.package_version
            )

            IF omniverse_result.is_ok():
                artifacts.omniverse_packages = omniverse_result.packages
                LOG("Created {omniverse_result.packages.len()} Omniverse packages", INFO)
            ELSE:
                LOG("Omniverse packaging failed: {omniverse_result.error}", WARN)
                // Omniverse is optional, continue

        // Step 5: Generate API Documentation
        IF config.generate_api_docs:
            LOG("Generating API documentation", INFO)
            api_docs_result = generate_api_documentation(
                wasm_binaries,
                config.api_doc_format,
                config.container_config,
                config.build_metadata
            )

            artifacts.api_documentation = api_docs_result.documentation
            LOG("Generated {api_docs_result.documentation.len()} API docs", INFO)

        // Step 6: Generate Integration Guides
        LOG("Generating integration guides", DEBUG)
        integration_guides = generate_integration_guides(
            artifacts,
            config
        )
        artifacts.integration_guides = integration_guides

        // Step 7: Generate Deployment Manifests
        LOG("Generating deployment manifests", INFO)
        deployment_manifests = generate_deployment_manifests(
            artifacts.nim_containers,
            artifacts.triton_models,
            config
        )
        artifacts.deployment_manifests = deployment_manifests

        // Step 8: Create Distribution Archives
        IF config.enable_distribution_archive:
            LOG("Creating distribution archives", INFO)
            archive_result = create_distribution_archives(
                wasm_binaries,
                artifacts,
                config.distribution_config,
                config.build_metadata,
                config.test_results,
                config.parity_report
            )

            artifacts.distribution_archives = archive_result.archives
            artifacts.checksums = archive_result.checksums
            LOG("Created {archive_result.archives.len()} distribution archives", INFO)

        // Step 9: Generate Package Manifest
        LOG("Generating package manifest", DEBUG)
        package_manifest = generate_package_manifest(
            config,
            artifacts,
            current_timestamp() - start_time
        )
        artifacts.package_manifest = package_manifest

        // Step 10: Publish to Registry (if configured)
        IF config.publish_to_registry:
            LOG("Publishing artifacts to registry", INFO)
            publish_result = publish_artifacts(
                artifacts,
                config.registry_url.unwrap(),
                config.package_name,
                config.package_version
            )

            IF publish_result.is_error():
                LOG("Failed to publish: {publish_result.error}", WARN)
                // Continue, publish is optional

        // Step 11: Final Validation
        LOG("Validating packaging completeness", DEBUG)
        validation = artifacts.validate_completeness()
        IF validation.is_error():
            LOG("Packaging validation warnings: {validation.warnings}", WARN)

        packaging_time = current_timestamp() - start_time
        LOG("Packaging completed successfully in {packaging_time}s", INFO)
        LOG("Total artifacts: {artifacts.total_artifacts_count()}", INFO)
        LOG("Total size: {artifacts.total_size_bytes()} bytes", INFO)

        RETURN Ok(artifacts)

    CATCH NimGenerationError AS e:
        LOG("NIM generation error: {e}", ERROR)
        RETURN Error(e)

    CATCH TritonConfigError AS e:
        LOG("Triton configuration error: {e}", ERROR)
        RETURN Error(e)

    CATCH ArchiveCreationError AS e:
        LOG("Archive creation error: {e}", ERROR)
        RETURN Error(e)

    CATCH Error AS e:
        LOG("Unexpected packaging error: {e}", ERROR)
        RETURN Error(PackagingFailed(e))
```

### 3.2 NIM Container Generation

```rust
ALGORITHM generate_nim_containers(
    wasm_binaries: List<WasmBinary>,
    container_config: ContainerConfiguration,
    package_name: String,
    package_version: SemanticVersion
) -> Result<NimContainerResult, NimError>:

    LOG("Generating NIM containers for {wasm_binaries.len()} WASM binaries", INFO)
    result = NimContainerResult::new()

    FOR EACH wasm_binary IN wasm_binaries:
        LOG("Creating NIM container for {wasm_binary.crate_name}", DEBUG)

        TRY:
            // Step 1: Generate Dockerfile
            dockerfile = generate_dockerfile(
                wasm_binary,
                container_config,
                package_name,
                package_version
            )

            dockerfile_path = write_dockerfile(
                dockerfile,
                container_config.output_dir,
                wasm_binary.crate_name
            )

            // Step 2: Build container image
            image_name = "{package_name}-{wasm_binary.crate_name}"
            image_tag = "{package_version}"

            build_result = build_container_image(
                dockerfile_path,
                image_name,
                image_tag,
                container_config
            )

            IF build_result.is_error():
                LOG("Failed to build container for {wasm_binary.crate_name}", ERROR)
                CONTINUE  // Skip this binary, continue with others

            // Step 3: Generate health check service
            health_check_script = generate_health_check_service(
                wasm_binary,
                container_config.health_check_endpoint
            )

            // Step 4: Extract image metadata
            image_metadata = inspect_container_image(
                build_result.image_id
            )

            // Step 5: Export container image (optional)
            image_archive_path = NONE
            IF container_config.export_image:
                archive_path = export_container_image(
                    build_result.image_id,
                    "{image_name}_{image_tag}.tar"
                )
                image_archive_path = Some(archive_path)

            // Step 6: Create artifact entry
            container_artifact = ContainerArtifact {
                image_name: image_name,
                image_tag: image_tag,
                image_id: build_result.image_id,
                image_size_bytes: image_metadata.size,
                dockerfile_path: dockerfile_path,
                image_archive_path: image_archive_path,
                created_at: current_timestamp(),
                base_image: container_config.base_image,
                layers: image_metadata.layers,
                health_check_config: HealthCheckConfig {
                    endpoint: container_config.health_check_endpoint,
                    interval_seconds: container_config.health_check_interval.as_secs(),
                    timeout_seconds: container_config.health_check_timeout.as_secs(),
                    retries: container_config.health_check_retries
                },
                exposed_ports: [container_config.service_port, container_config.metrics_port],
                environment_vars: container_config.environment_vars.clone(),
                image_digest: calculate_image_digest(build_result.image_id)
            }

            result.containers.push(container_artifact)
            LOG("Successfully created container: {image_name}:{image_tag}", INFO)

        CATCH ContainerBuildError AS e:
            LOG("Container build failed for {wasm_binary.crate_name}: {e}", ERROR)
            result.errors.push(e)
            CONTINUE

    IF result.containers.is_empty():
        RETURN Error(NoContainersGenerated)

    RETURN Ok(result)


ALGORITHM generate_dockerfile(
    wasm_binary: WasmBinary,
    config: ContainerConfiguration,
    package_name: String,
    version: SemanticVersion
) -> Dockerfile:

    dockerfile = Dockerfile::new()

    // Layer 1: Base image
    dockerfile.from(config.base_image, "base")

    // Layer 2: Labels
    dockerfile.label("maintainer", "Portalis Platform")
    dockerfile.label("version", version.to_string())
    dockerfile.label("package", package_name)
    dockerfile.label("wasm.module", wasm_binary.crate_name)

    FOR EACH (key, value) IN config.labels:
        dockerfile.label(key, value)

    // Layer 3: Install WASM runtime
    MATCH config.wasm_runtime:
        WasmRuntime::Wasmtime:
            dockerfile.run([
                "apt-get update",
                "apt-get install -y curl",
                "curl -fsSL https://wasmtime.dev/install.sh | bash",
                "ln -s ~/.wasmtime/bin/wasmtime /usr/local/bin/"
            ])

        WasmRuntime::Wasmer:
            dockerfile.run([
                "curl https://get.wasmer.io -sSfL | sh",
                "ln -s ~/.wasmer/bin/wasmer /usr/local/bin/"
            ])

        WasmRuntime::WasmEdge:
            dockerfile.run([
                "curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash"
            ])

    // Layer 4: Create service user (security)
    IF config.run_as_non_root:
        dockerfile.run([
            "useradd -r -u 1001 -g root wasmuser",
            "mkdir -p /app",
            "chown -R wasmuser:root /app"
        ])

    // Layer 5: Copy WASM binary
    dockerfile.workdir("/app")
    dockerfile.copy(wasm_binary.file_path, "/app/{wasm_binary.crate_name}.wasm")

    // Layer 6: Copy health check script
    dockerfile.copy("health_check.py", "/app/health_check.py")
    dockerfile.run("chmod +x /app/health_check.py")

    // Layer 7: Install Python for health checks and HTTP serving
    dockerfile.run([
        "apt-get install -y python3 python3-pip",
        "pip3 install fastapi uvicorn"
    ])

    // Layer 8: Copy service wrapper
    dockerfile.copy("service_wrapper.py", "/app/service_wrapper.py")

    // Layer 9: Environment variables
    FOR EACH (key, value) IN config.environment_vars:
        dockerfile.env(key, value)

    dockerfile.env("WASM_MODULE", "/app/{wasm_binary.crate_name}.wasm")
    dockerfile.env("SERVICE_PORT", config.service_port.to_string())
    dockerfile.env("METRICS_PORT", config.metrics_port.to_string())

    // Layer 10: Expose ports
    dockerfile.expose(config.service_port)
    dockerfile.expose(config.metrics_port)

    // Layer 11: Health check
    dockerfile.healthcheck(
        interval: config.health_check_interval,
        timeout: config.health_check_timeout,
        retries: config.health_check_retries,
        command: "curl -f http://localhost:{config.service_port}/health || exit 1"
    )

    // Layer 12: Switch to non-root user
    IF config.run_as_non_root:
        dockerfile.user("wasmuser")

    // Layer 13: Entrypoint
    IF config.entrypoint.is_empty():
        dockerfile.entrypoint([
            "python3",
            "/app/service_wrapper.py"
        ])
    ELSE:
        dockerfile.entrypoint(config.entrypoint)

    IF NOT config.command_args.is_empty():
        dockerfile.cmd(config.command_args)

    RETURN dockerfile


FUNCTION generate_health_check_service(
    wasm_binary: WasmBinary,
    health_endpoint: String
) -> String:

    // Generate a simple Python FastAPI health check service
    service_code = """
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import subprocess
import time
import os

app = FastAPI()

wasm_module = os.environ.get('WASM_MODULE')
start_time = time.time()

@app.get('/health')
async def health_check():
    '''Health check endpoint for container orchestration'''
    uptime = time.time() - start_time

    # Basic health: check if WASM module exists
    if not os.path.exists(wasm_module):
        return JSONResponse(
            status_code=503,
            content={{
                'status': 'unhealthy',
                'reason': 'WASM module not found',
                'module': wasm_module
            }}
        )

    return JSONResponse(
        status_code=200,
        content={{
            'status': 'healthy',
            'uptime_seconds': uptime,
            'module': wasm_module,
            'exports': {wasm_binary.exports}
        }}
    )

@app.get('/ready')
async def readiness_check():
    '''Readiness check for Kubernetes'''
    return {{'ready': True}}

@app.get('/metrics')
async def metrics():
    '''Basic Prometheus-compatible metrics'''
    uptime = time.time() - start_time

    metrics_text = f'''
# HELP wasm_service_uptime_seconds Service uptime in seconds
# TYPE wasm_service_uptime_seconds gauge
wasm_service_uptime_seconds {uptime}

# HELP wasm_service_info Service information
# TYPE wasm_service_info gauge
wasm_service_info{{module="{wasm_binary.crate_name}"}} 1
'''

    return Response(content=metrics_text, media_type='text/plain')

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('SERVICE_PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
"""

    RETURN service_code


FUNCTION build_container_image(
    dockerfile_path: Path,
    image_name: String,
    image_tag: String,
    config: ContainerConfiguration
) -> Result<ImageBuildResult, BuildError>:

    LOG("Building container image: {image_name}:{image_tag}", DEBUG)

    // Construct build command
    MATCH config.runtime:
        ContainerRuntime::Docker:
            build_cmd = ["docker", "build"]
        ContainerRuntime::Podman:
            build_cmd = ["podman", "build"]

    build_cmd.push("-t", "{image_name}:{image_tag}")
    build_cmd.push("-f", dockerfile_path.to_string())

    // Add build args
    IF config.base_image.is_some():
        build_cmd.push("--build-arg", "BASE_IMAGE={config.base_image}")

    // Add context directory
    context_dir = dockerfile_path.parent().unwrap()
    build_cmd.push(context_dir.to_string())

    // Execute build
    process = execute_command_with_timeout(
        build_cmd,
        timeout: Duration::from_secs(600)  // 10 minute timeout
    )

    MATCH process.wait():
        ProcessResult::Success(output):
            // Extract image ID from output
            image_id = extract_image_id_from_output(output.stdout)

            LOG("Container image built successfully: {image_id}", INFO)

            RETURN Ok(ImageBuildResult {
                image_id: image_id,
                image_name: image_name,
                image_tag: image_tag,
                build_log: output.stdout
            })

        ProcessResult::Failure(output):
            LOG("Container build failed: {output.stderr}", ERROR)
            RETURN Error(BuildFailed(output.stderr))

        ProcessResult::Timeout:
            RETURN Error(BuildTimeout)


FUNCTION export_container_image(
    image_id: String,
    output_filename: String
) -> Result<Path, ExportError>:

    LOG("Exporting container image {image_id} to {output_filename}", DEBUG)

    export_cmd = ["docker", "save", "-o", output_filename, image_id]

    process = execute_command(export_cmd)

    IF process.is_success():
        output_path = Path::new(output_filename)
        LOG("Image exported successfully: {output_path}", INFO)
        RETURN Ok(output_path)
    ELSE:
        RETURN Error(ExportFailed(process.stderr))
```

### 3.3 Triton Model Configuration

```rust
ALGORITHM configure_triton_models(
    wasm_binaries: List<WasmBinary>,
    triton_config: TritonConfiguration,
    package_name: String
) -> Result<TritonModelResult, TritonError>:

    LOG("Configuring Triton models for {wasm_binaries.len()} WASM binaries", INFO)
    result = TritonModelResult::new()

    FOR EACH wasm_binary IN wasm_binaries:
        LOG("Creating Triton configuration for {wasm_binary.crate_name}", DEBUG)

        TRY:
            // Step 1: Create model repository structure
            model_name = "{package_name}_{wasm_binary.crate_name}"
            model_version = triton_config.model_version

            model_repo_path = create_triton_model_repository(
                triton_config.model_repository_path,
                model_name,
                model_version
            )

            // Step 2: Copy WASM binary to model repository
            wasm_model_path = model_repo_path.join("{model_version}/model.wasm")
            copy_file(wasm_binary.file_path, wasm_model_path)

            // Step 3: Generate config.pbtxt
            config_pbtxt = generate_triton_config_pbtxt(
                wasm_binary,
                triton_config,
                model_name
            )

            config_path = model_repo_path.join("config.pbtxt")
            write_file(config_path, config_pbtxt)

            // Step 4: Create custom backend script (if using Python backend)
            IF triton_config.platform == TritonPlatform::Python:
                backend_script = generate_python_wasm_backend(
                    wasm_binary,
                    triton_config
                )

                backend_path = model_repo_path.join("{model_version}/model.py")
                write_file(backend_path, backend_script)

            // Step 5: Validate Triton configuration
            validation_result = validate_triton_config(
                config_path,
                model_repo_path
            )

            // Step 6: Create artifact entry
            triton_artifact = TritonModelArtifact {
                model_name: model_name,
                model_version: model_version,
                model_repository_path: model_repo_path,
                config_pbtxt_path: config_path,
                wasm_model_paths: [wasm_model_path],
                platform: triton_config.platform.to_string(),
                max_batch_size: triton_config.max_batch_size,
                instance_count: triton_config.instance_groups.iter().sum(|g| g.count),
                validated: validation_result.is_valid,
                validation_report: validation_result.report
            }

            result.models.push(triton_artifact)
            LOG("Triton model configured: {model_name} v{model_version}", INFO)

        CATCH TritonConfigError AS e:
            LOG("Triton config failed for {wasm_binary.crate_name}: {e}", ERROR)
            result.errors.push(e)
            CONTINUE

    IF result.models.is_empty():
        RETURN Error(NoModelsConfigured)

    RETURN Ok(result)


FUNCTION generate_triton_config_pbtxt(
    wasm_binary: WasmBinary,
    config: TritonConfiguration,
    model_name: String
) -> String:

    pbtxt = StringBuilder::new()

    // Basic model info
    pbtxt.append_line("name: \"{model_name}\"")
    pbtxt.append_line("platform: \"{config.platform}\"")
    pbtxt.append_line("max_batch_size: {config.max_batch_size}")
    pbtxt.append_line("")

    // Inputs
    FOR EACH input IN config.inputs:
        pbtxt.append_line("input {{")
        pbtxt.append_line("  name: \"{input.name}\"")
        pbtxt.append_line("  data_type: {input.data_type}")

        pbtxt.append("  dims: [")
        pbtxt.append(input.dims.join(", "))
        pbtxt.append_line("]")

        IF input.optional:
            pbtxt.append_line("  optional: true")

        pbtxt.append_line("}}")
        pbtxt.append_line("")

    // Outputs
    FOR EACH output IN config.outputs:
        pbtxt.append_line("output {{")
        pbtxt.append_line("  name: \"{output.name}\"")
        pbtxt.append_line("  data_type: {output.data_type}")

        pbtxt.append("  dims: [")
        pbtxt.append(output.dims.join(", "))
        pbtxt.append_line("]")

        pbtxt.append_line("}}")
        pbtxt.append_line("")

    // Instance groups
    FOR EACH instance_group IN config.instance_groups:
        pbtxt.append_line("instance_group {{")
        pbtxt.append_line("  name: \"{instance_group.name}\"")
        pbtxt.append_line("  kind: {instance_group.kind}")
        pbtxt.append_line("  count: {instance_group.count}")

        IF instance_group.kind == InstanceKind::GPU AND instance_group.gpus.is_some():
            pbtxt.append("  gpus: [")
            pbtx.append(instance_group.gpus.unwrap().join(", "))
            pbtxt.append_line("]")

        pbtxt.append_line("}}")
        pbtxt.append_line("")

    // Dynamic batching
    IF config.dynamic_batching.is_some():
        db = config.dynamic_batching.unwrap()
        pbtxt.append_line("dynamic_batching {{")

        IF NOT db.preferred_batch_size.is_empty():
            pbtxt.append("  preferred_batch_size: [")
            pbtxt.append(db.preferred_batch_size.join(", "))
            pbtxt.append_line("]")

        pbtxt.append_line("  max_queue_delay_microseconds: {db.max_queue_delay_microseconds}")
        pbtxt.append_line("  preserve_ordering: {db.preserve_ordering}")
        pbtxt.append_line("}}")
        pbtxt.append_line("")

    // Optimization
    IF config.optimization.graph_optimization:
        pbtxt.append_line("optimization {{")
        pbtxt.append_line("  graph {{")
        pbtxt.append_line("    level: 1")
        pbtxt.append_line("  }}")
        pbtxt.append_line("}}")
        pbtxt.append_line("")

    // Version policy
    pbtxt.append_line("version_policy {{")
    MATCH config.version_policy.policy:
        VersionPolicyType::Latest(n):
            pbtxt.append_line("  latest {{")
            pbtxt.append_line("    num_versions: {n}")
            pbtxt.append_line("  }}")

        VersionPolicyType::All:
            pbtxt.append_line("  all {{}}")

        VersionPolicyType::Specific:
            pbtxt.append_line("  specific {{")
            pbtxt.append("    versions: [")
            pbtxt.append(config.version_policy.versions.unwrap().join(", "))
            pbtxt.append_line("]")
            pbtxt.append_line("  }}")

    pbtxt.append_line("}}")

    RETURN pbtxt.to_string()


FUNCTION generate_python_wasm_backend(
    wasm_binary: WasmBinary,
    config: TritonConfiguration
) -> String:

    // Generate Python backend that wraps WASM module
    backend_code = """
import triton_python_backend_utils as pb_utils
import wasmtime
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        '''Initialize the WASM runtime and load module'''
        model_config = json.loads(args['model_config'])

        # Create WASM runtime
        self.engine = wasmtime.Engine()
        self.store = wasmtime.Store(self.engine)

        # Load WASM module
        model_path = args['model_repository'] + '/' + args['model_version'] + '/model.wasm'
        module = wasmtime.Module.from_file(self.engine, model_path)

        # Instantiate module
        self.instance = wasmtime.Instance(self.store, module, [])

        # Get exported functions
        self.exports = self.instance.exports(self.store)

        # Cache input/output configs
        self.input_configs = model_config['input']
        self.output_configs = model_config['output']

    def execute(self, requests):
        '''Execute inference requests'''
        responses = []

        for request in requests:
            # Extract inputs
            inputs = {}
            for input_config in self.input_configs:
                input_name = input_config['name']
                input_tensor = pb_utils.get_input_tensor_by_name(request, input_name)
                inputs[input_name] = input_tensor.as_numpy()

            # Call WASM function
            # This is simplified - actual implementation depends on ABI
            result = self._call_wasm_function(inputs)

            # Create output tensors
            output_tensors = []
            for output_config in self.output_configs:
                output_name = output_config['name']
                output_data = result[output_name]
                output_tensor = pb_utils.Tensor(output_name, output_data)
                output_tensors.append(output_tensor)

            # Create response
            inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(inference_response)

        return responses

    def _call_wasm_function(self, inputs):
        '''Call WASM exported function'''
        # Implementation depends on WASM ABI
        # This is a placeholder
        main_func = self.exports['main']
        result = main_func(self.store, inputs)
        return result

    def finalize(self):
        '''Cleanup'''
        pass
"""

    RETURN backend_code
```

### 3.4 Omniverse Package Creation

```rust
ALGORITHM package_for_omniverse(
    wasm_binaries: List<WasmBinary>,
    omniverse_config: OmniverseConfiguration,
    package_name: String,
    package_version: SemanticVersion
) -> Result<OmniversePackageResult, OmniverseError>:

    LOG("Creating Omniverse-compatible packages", INFO)
    result = OmniversePackageResult::new()

    // Step 1: Validate WASM compatibility with Omniverse
    IF omniverse_config.validate_wasm_compatibility:
        LOG("Validating WASM modules for Omniverse compatibility", DEBUG)

        FOR EACH wasm_binary IN wasm_binaries:
            validation = validate_omniverse_wasm_compatibility(wasm_binary)

            IF NOT validation.is_compatible:
                LOG("WASM module {wasm_binary.crate_name} not Omniverse compatible", WARN)
                result.incompatible_modules.push(wasm_binary.crate_name)
                CONTINUE

    // Step 2: Create extension directory structure
    extension_name = omniverse_config.extension_name
    extension_dir = create_omniverse_extension_structure(
        package_name,
        extension_name,
        omniverse_config
    )

    // Step 3: Generate extension.toml
    extension_toml = generate_extension_toml(
        extension_name,
        omniverse_config,
        package_version
    )

    extension_toml_path = extension_dir.join("extension.toml")
    write_file(extension_toml_path, extension_toml)

    // Step 4: Copy WASM modules
    wasm_modules_dir = extension_dir.join("wasm_modules")
    create_directory(wasm_modules_dir)

    wasm_module_paths = List::new()

    FOR EACH wasm_binary IN wasm_binaries:
        IF result.incompatible_modules.contains(wasm_binary.crate_name):
            CONTINUE

        wasm_dest = wasm_modules_dir.join("{wasm_binary.crate_name}.wasm")
        copy_file(wasm_binary.file_path, wasm_dest)

        wasm_module_paths.push(WasmModulePath {
            module_name: wasm_binary.crate_name,
            file_path: wasm_dest,
            entry_point: "main",  // Default entry point
            description: "Translated from Python via Portalis"
        })

    // Step 5: Generate Python bindings for Omniverse Kit
    python_bindings = generate_omniverse_python_bindings(
        wasm_module_paths,
        extension_name
    )

    bindings_path = extension_dir.join("python/{extension_name}/__init__.py")
    create_directories(bindings_path.parent())
    write_file(bindings_path, python_bindings)

    // Step 6: Generate integration documentation
    IF omniverse_config.generate_integration_docs:
        integration_docs = generate_omniverse_integration_docs(
            extension_name,
            wasm_module_paths,
            omniverse_config
        )

        docs_path = extension_dir.join("docs/integration.md")
        create_directories(docs_path.parent())
        write_file(docs_path, integration_docs)

    // Step 7: Create example scenarios
    IF NOT omniverse_config.example_scenarios.is_empty():
        examples_dir = extension_dir.join("examples")
        create_directory(examples_dir)

        FOR EACH scenario IN omniverse_config.example_scenarios:
            scenario_file = generate_example_scenario(scenario, wasm_module_paths)
            scenario_path = examples_dir.join("{scenario.name}.py")
            write_file(scenario_path, scenario_file)

    // Step 8: Generate README
    readme = generate_omniverse_readme(
        extension_name,
        omniverse_config,
        wasm_module_paths,
        package_version
    )

    readme_path = extension_dir.join("README.md")
    write_file(readme_path, readme)

    // Step 9: Package extension
    IF omniverse_config.package_as_extension:
        package_path = create_extension_package(
            extension_dir,
            extension_name,
            package_version
        )

        LOG("Created Omniverse extension package: {package_path}", INFO)
    ELSE:
        package_path = extension_dir

    // Step 10: Create artifact
    omniverse_artifact = OmniversePackageArtifact {
        extension_name: extension_name,
        extension_version: package_version.to_string(),
        package_path: package_path,
        extension_toml_path: extension_toml_path,
        wasm_modules: wasm_module_paths,
        readme_path: readme_path,
        examples_path: IF omniverse_config.example_scenarios.is_empty()
                       THEN None
                       ELSE Some(extension_dir.join("examples")),
        kit_version_compatibility: omniverse_config.required_kit_version,
        validated: true
    }

    result.packages.push(omniverse_artifact)

    RETURN Ok(result)


FUNCTION generate_extension_toml(
    extension_name: String,
    config: OmniverseConfiguration,
    version: SemanticVersion
) -> String:

    toml = StringBuilder::new()

    // Package metadata
    toml.append_line("[package]")
    toml.append_line("name = \"{extension_name}\"")
    toml.append_line("version = \"{version}\"")
    toml.append_line("description = \"{config.extension_description}\"")
    toml.append_line("category = \"Portalis\"")
    toml.append_line("")

    // Dependencies
    toml.append_line("[dependencies]")
    toml.append_line("\"omni.kit.uiapp\" = {{}}  # Basic Kit UI")

    FOR EACH dep IN config.dependencies:
        toml.append_line("\"{dep.extension_name}\" = {{ version = \"{dep.version_requirement}\" }}")

    toml.append_line("")

    // Python module
    toml.append_line("[[python.module]]")
    toml.append_line("name = \"{extension_name}\"")
    toml.append_line("")

    // Kit version compatibility
    toml.append_line("[package.target]")
    toml.append_line("kit = \"{config.required_kit_version}\"")
    toml.append_line("")

    RETURN toml.to_string()


FUNCTION generate_omniverse_python_bindings(
    wasm_modules: List<WasmModulePath>,
    extension_name: String
) -> String:

    bindings = """
'''
{extension_name} - Omniverse Extension
Generated by Portalis Platform
'''

import omni.ext
import omni.ui as ui
import wasmtime
from pathlib import Path

class {extension_name}Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        '''Called when extension starts'''
        print(f'[{extension_name}] Extension starting...')

        # Initialize WASM runtime
        self._engine = wasmtime.Engine()
        self._store = wasmtime.Store(self._engine)
        self._modules = {{}}

        # Load WASM modules
        extension_path = Path(__file__).parent.parent
        wasm_dir = extension_path / 'wasm_modules'

"""

    FOR EACH module IN wasm_modules:
        bindings.append("""
        # Load {module.module_name}
        module_path = wasm_dir / '{module.module_name}.wasm'
        wasm_module = wasmtime.Module.from_file(self._engine, str(module_path))
        instance = wasmtime.Instance(self._store, wasm_module, [])
        self._modules['{module.module_name}'] = instance
        print(f'[{extension_name}] Loaded WASM module: {module.module_name}')
""")

    bindings.append("""

        print(f'[{extension_name}] Extension started successfully')

    def on_shutdown(self):
        '''Called when extension shuts down'''
        print(f'[{extension_name}] Extension shutting down...')
        self._modules = None

    def get_module(self, module_name):
        '''Get a loaded WASM module instance'''
        return self._modules.get(module_name)

    def call_function(self, module_name, function_name, *args):
        '''Call a function in a WASM module'''
        module = self.get_module(module_name)
        if module is None:
            raise ValueError(f'Module not found: {{module_name}}')

        func = module.exports(self._store)[function_name]
        return func(self._store, *args)
""")

    RETURN bindings


FUNCTION validate_omniverse_wasm_compatibility(
    wasm_binary: WasmBinary
) -> OmniverseCompatibilityResult:

    result = OmniverseCompatibilityResult::new()

    // Check 1: WASI compliance (required)
    IF NOT wasm_binary.wasi_compliant:
        result.is_compatible = false
        result.issues.push("WASM module must be WASI compliant")
        RETURN result

    // Check 2: No threading (Omniverse limitation)
    IF uses_wasm_threads(wasm_binary):
        result.is_compatible = false
        result.issues.push("WASM threads not supported in Omniverse")
        RETURN result

    // Check 3: Size limitations (warn if too large)
    max_size = 100 * 1024 * 1024  // 100 MB
    IF wasm_binary.size_bytes > max_size:
        result.warnings.push("Large WASM module may impact performance")

    // Check 4: Memory requirements
    IF wasm_binary.memory_pages > 1024:  // >64 MB
        result.warnings.push("High memory usage may cause issues")

    result.is_compatible = true
    RETURN result
```

### 3.5 Distribution Archive Creation

```rust
ALGORITHM create_distribution_archives(
    wasm_binaries: List<WasmBinary>,
    artifacts: PackagingArtifacts,
    distribution_config: DistributionConfiguration,
    build_metadata: BuildManifest,
    test_results: Path,
    parity_report: Path
) -> Result<DistributionResult, ArchiveError>:

    LOG("Creating distribution archives", INFO)
    result = DistributionResult::new()

    // Step 1: Create staging directory
    staging_dir = create_temp_directory("portalis_dist_staging")

    TRY:
        // Step 2: Organize artifacts in staging directory
        organize_distribution_contents(
            staging_dir,
            wasm_binaries,
            artifacts,
            distribution_config,
            build_metadata,
            test_results,
            parity_report
        )

        // Step 3: Generate README
        IF distribution_config.generate_readme:
            readme_content = generate_distribution_readme(
                wasm_binaries,
                artifacts,
                build_metadata,
                distribution_config
            )

            readme_path = staging_dir.join(
                MATCH distribution_config.readme_format:
                    ReadmeFormat::Markdown => "README.md"
                    ReadmeFormat::PlainText => "README.txt"
                    ReadmeFormat::RestructuredText => "README.rst"
            )

            write_file(readme_path, readme_content)

        // Step 4: Generate manifest
        IF distribution_config.generate_manifest:
            manifest_content = generate_distribution_manifest(
                staging_dir,
                wasm_binaries,
                artifacts,
                build_metadata
            )

            manifest_path = staging_dir.join(
                MATCH distribution_config.manifest_format:
                    ManifestFormat::JSON => "manifest.json"
                    ManifestFormat::YAML => "manifest.yaml"
                    ManifestFormat::TOML => "manifest.toml"
            )

            write_file(manifest_path, manifest_content)

        // Step 5: Copy license file
        IF distribution_config.license_file.is_some():
            license_src = distribution_config.license_file.unwrap()
            license_dest = staging_dir.join("LICENSE")
            copy_file(license_src, license_dest)

        // Step 6: Create archive
        archive_name = generate_archive_filename(
            build_metadata.package_name,
            build_metadata.package_version,
            distribution_config.archive_format
        )

        archive_path = create_archive(
            staging_dir,
            archive_name,
            distribution_config.archive_format,
            distribution_config.compression_level
        )

        LOG("Created archive: {archive_path}", INFO)

        // Step 7: Calculate checksums
        checksums = Map::new()

        IF distribution_config.generate_checksums:
            FOR EACH algorithm IN distribution_config.checksum_algorithms:
                checksum = calculate_checksum(archive_path, algorithm)
                checksums.insert(algorithm, checksum)

                // Write checksum file
                checksum_file = "{archive_path}.{algorithm.extension()}"
                write_file(checksum_file, "{checksum}  {archive_name}")

        // Step 8: Get archive metadata
        archive_size = file_size(archive_path)
        uncompressed_size = calculate_uncompressed_size(staging_dir)

        // Step 9: Create inventory
        inventory = create_archive_inventory(staging_dir)

        // Step 10: Create distribution artifact
        distribution_archive = DistributionArchive {
            archive_name: archive_name,
            archive_path: archive_path,
            archive_format: distribution_config.archive_format,
            compressed_size_bytes: archive_size,
            uncompressed_size_bytes: uncompressed_size,
            contents: inventory,
            checksums: checksums,
            manifest_path: manifest_path,
            readme_path: readme_path
        }

        result.archives.push(distribution_archive)
        result.checksums = checksums

        LOG("Distribution archive created successfully", INFO)
        LOG("Compressed size: {archive_size} bytes", INFO)
        LOG("Compression ratio: {(uncompressed_size as f64 / archive_size as f64):.2}x", INFO)

        RETURN Ok(result)

    FINALLY:
        // Cleanup staging directory
        remove_directory_recursive(staging_dir)


FUNCTION organize_distribution_contents(
    staging_dir: Path,
    wasm_binaries: List<WasmBinary>,
    artifacts: PackagingArtifacts,
    config: DistributionConfiguration,
    build_metadata: BuildManifest,
    test_results: Path,
    parity_report: Path
) -> Result<(), Error>:

    // Directory structure:
    // /
    // ├── wasm/              (WASM binaries)
    // ├── containers/        (Container artifacts)
    // ├── triton/            (Triton configs)
    // ├── omniverse/         (Omniverse packages)
    // ├── src/               (Rust source - optional)
    // ├── tests/             (Test results - optional)
    // ├── docs/              (Documentation)
    // ├── deployment/        (Deployment manifests)
    // ├── README.md
    // ├── manifest.json
    // └── LICENSE

    // WASM binaries
    wasm_dir = staging_dir.join("wasm")
    create_directory(wasm_dir)

    FOR EACH wasm_binary IN wasm_binaries:
        dest = wasm_dir.join("{wasm_binary.crate_name}.wasm")
        copy_file(wasm_binary.file_path, dest)

    // Container artifacts
    IF NOT artifacts.nim_containers.is_empty():
        containers_dir = staging_dir.join("containers")
        create_directory(containers_dir)

        FOR EACH container IN artifacts.nim_containers:
            container_subdir = containers_dir.join(container.image_name)
            create_directory(container_subdir)

            // Copy Dockerfile
            copy_file(
                container.dockerfile_path,
                container_subdir.join("Dockerfile")
            )

            // Copy image archive if exists
            IF container.image_archive_path.is_some():
                copy_file(
                    container.image_archive_path.unwrap(),
                    container_subdir.join("image.tar")
                )

    // Triton models
    IF NOT artifacts.triton_models.is_empty():
        triton_dir = staging_dir.join("triton")
        create_directory(triton_dir)

        FOR EACH triton_model IN artifacts.triton_models:
            copy_directory_recursive(
                triton_model.model_repository_path,
                triton_dir.join(triton_model.model_name)
            )

    // Omniverse packages
    IF NOT artifacts.omniverse_packages.is_empty():
        omniverse_dir = staging_dir.join("omniverse")
        create_directory(omniverse_dir)

        FOR EACH omni_pkg IN artifacts.omniverse_packages:
            copy_directory_recursive(
                omni_pkg.package_path,
                omniverse_dir.join(omni_pkg.extension_name)
            )

    // Rust source (optional)
    IF config.include_source_rust:
        src_dir = staging_dir.join("src")
        create_directory(src_dir)

        // Copy from build metadata
        copy_directory_recursive(
            build_metadata.workspace_root,
            src_dir
        )

    // Test results (optional)
    IF config.include_test_reports:
        tests_dir = staging_dir.join("tests")
        create_directory(tests_dir)

        copy_file(test_results, tests_dir.join("test_results.json"))
        copy_file(parity_report, tests_dir.join("parity_report.json"))

    // Documentation
    docs_dir = staging_dir.join("docs")
    create_directory(docs_dir)

    FOR EACH api_doc IN artifacts.api_documentation:
        copy_file(api_doc.file_path, docs_dir.join(api_doc.file_path.file_name()))

    FOR EACH guide IN artifacts.integration_guides:
        copy_file(guide.file_path, docs_dir.join(guide.file_path.file_name()))

    // Deployment manifests
    IF NOT artifacts.deployment_manifests.is_empty():
        deployment_dir = staging_dir.join("deployment")
        create_directory(deployment_dir)

        FOR EACH manifest IN artifacts.deployment_manifests:
            copy_file(manifest.file_path, deployment_dir.join(manifest.file_path.file_name()))

    RETURN Ok(())


FUNCTION generate_distribution_readme(
    wasm_binaries: List<WasmBinary>,
    artifacts: PackagingArtifacts,
    build_metadata: BuildManifest,
    config: DistributionConfiguration
) -> String:

    readme = StringBuilder::new()

    readme.append_line("# {build_metadata.package_name}")
    readme.append_line("")
    readme.append_line("Version: {build_metadata.package_version}")
    readme.append_line("Generated: {current_timestamp()}")
    readme.append_line("Generated by: Portalis Platform")
    readme.append_line("")

    readme.append_line("## Overview")
    readme.append_line("")
    readme.append_line("This package contains Python code translated to Rust/WASM, packaged as")
    readme.append_line("enterprise-ready NIM microservices.")
    readme.append_line("")

    readme.append_line("## Contents")
    readme.append_line("")
    readme.append_line("- `wasm/` - WebAssembly binaries ({wasm_binaries.len()} modules)")

    IF NOT artifacts.nim_containers.is_empty():
        readme.append_line("- `containers/` - NIM container images and Dockerfiles")

    IF NOT artifacts.triton_models.is_empty():
        readme.append_line("- `triton/` - Triton Inference Server model configurations")

    IF NOT artifacts.omniverse_packages.is_empty():
        readme.append_line("- `omniverse/` - NVIDIA Omniverse extension packages")

    IF config.include_source_rust:
        readme.append_line("- `src/` - Generated Rust source code")

    IF config.include_test_reports:
        readme.append_line("- `tests/` - Test results and parity reports")

    readme.append_line("- `docs/` - API documentation and integration guides")
    readme.append_line("- `deployment/` - Deployment manifests (Kubernetes, Docker Compose)")
    readme.append_line("")

    readme.append_line("## Quick Start")
    readme.append_line("")
    readme.append_line("### Running with Docker")
    readme.append_line("")
    readme.append_line("```bash")
    readme.append_line("cd containers/<service-name>")
    readme.append_line("docker build -t <service-name> .")
    readme.append_line("docker run -p 8000:8000 <service-name>")
    readme.append_line("```")
    readme.append_line("")

    IF NOT artifacts.triton_models.is_empty():
        readme.append_line("### Deploying with Triton")
        readme.append_line("")
        readme.append_line("```bash")
        readme.append_line("# Copy model repository")
        readme.append_line("cp -r triton/* /path/to/triton/model_repository/")
        readme.append_line("")
        readme.append_line("# Start Triton server")
        readme.append_line("tritonserver --model-repository=/path/to/triton/model_repository")
        readme.append_line("```")
        readme.append_line("")

    readme.append_line("## WASM Modules")
    readme.append_line("")

    FOR EACH wasm_binary IN wasm_binaries:
        readme.append_line("### {wasm_binary.crate_name}")
        readme.append_line("")
        readme.append_line("- **Size:** {wasm_binary.size_bytes} bytes")
        readme.append_line("- **WASI Compliant:** {wasm_binary.wasi_compliant}")
        readme.append_line("- **Exports:** {wasm_binary.exports.join(', ')}")
        readme.append_line("")

    readme.append_line("## Performance")
    readme.append_line("")
    readme.append_line("Translation metrics:")
    readme.append_line("")
    readme.append_line("- **Build time:** {build_metadata.build_time_seconds}s")
    readme.append_line("- **Total WASM size:** {build_metadata.total_wasm_size_bytes} bytes")

    IF build_metadata.test_info.is_some():
        test_info = build_metadata.test_info.unwrap()
        readme.append_line("- **Test pass rate:** {test_info.passed_tests}/{test_info.total_tests}")
        readme.append_line("- **Parity score:** {test_info.parity_score_percent}%")

    readme.append_line("")
    readme.append_line("## Documentation")
    readme.append_line("")
    readme.append_line("See `docs/` directory for:")
    readme.append_line("")
    readme.append_line("- API documentation (OpenAPI/gRPC specs)")
    readme.append_line("- Integration guides")
    readme.append_line("- Deployment instructions")
    readme.append_line("")

    IF config.license_type.is_some():
        readme.append_line("## License")
        readme.append_line("")
        readme.append_line("{config.license_type.unwrap()}")
        readme.append_line("")

    readme.append_line("## Support")
    readme.append_line("")
    readme.append_line("Generated by Portalis Platform")

    IF config.publish_metadata.homepage.is_some():
        readme.append_line("Homepage: {config.publish_metadata.homepage.unwrap()}")

    IF config.publish_metadata.repository.is_some():
        readme.append_line("Repository: {config.publish_metadata.repository.unwrap()}")

    readme.append_line("")

    RETURN readme.to_string()


FUNCTION create_archive(
    source_dir: Path,
    archive_name: String,
    format: ArchiveFormat,
    compression_level: u32
) -> Result<Path, ArchiveError>:

    LOG("Creating archive: {archive_name}", DEBUG)

    MATCH format:
        ArchiveFormat::TarGz:
            archive_path = "{archive_name}.tar.gz"
            cmd = ["tar", "-czf", archive_path, "-C", source_dir.to_string(), "."]

            // Set compression level via env var
            env = Map::new()
            env.insert("GZIP", "-{compression_level}")

            execute_command_with_env(cmd, env)

        ArchiveFormat::TarBz2:
            archive_path = "{archive_name}.tar.bz2"
            cmd = ["tar", "-cjf", archive_path, "-C", source_dir.to_string(), "."]
            execute_command(cmd)

        ArchiveFormat::TarXz:
            archive_path = "{archive_name}.tar.xz"
            cmd = ["tar", "-cJf", archive_path, "-C", source_dir.to_string(), "."]
            execute_command(cmd)

        ArchiveFormat::Zip:
            archive_path = "{archive_name}.zip"
            cmd = ["zip", "-r", "-{compression_level}", archive_path, source_dir.to_string()]
            execute_command(cmd)

    IF NOT Path::new(archive_path).exists():
        RETURN Error(ArchiveCreationFailed)

    LOG("Archive created: {archive_path}", INFO)
    RETURN Ok(Path::new(archive_path))
```

### 3.6 API Documentation Generation

```rust
ALGORITHM generate_api_documentation(
    wasm_binaries: List<WasmBinary>,
    doc_format: ApiDocFormat,
    container_config: ContainerConfiguration,
    build_metadata: BuildManifest
) -> Result<ApiDocResult, DocGenerationError>:

    LOG("Generating API documentation", INFO)
    result = ApiDocResult::new()

    MATCH doc_format:
        ApiDocFormat::OpenAPI:
            openapi_doc = generate_openapi_spec(
                wasm_binaries,
                container_config,
                build_metadata
            )
            result.documentation.push(openapi_doc)

        ApiDocFormat::GRPC:
            grpc_doc = generate_grpc_spec(
                wasm_binaries,
                build_metadata
            )
            result.documentation.push(grpc_doc)

        ApiDocFormat::Both:
            openapi_doc = generate_openapi_spec(
                wasm_binaries,
                container_config,
                build_metadata
            )
            result.documentation.push(openapi_doc)

            grpc_doc = generate_grpc_spec(
                wasm_binaries,
                build_metadata
            )
            result.documentation.push(grpc_doc)

    RETURN Ok(result)


FUNCTION generate_openapi_spec(
    wasm_binaries: List<WasmBinary>,
    container_config: ContainerConfiguration,
    build_metadata: BuildManifest
) -> ApiDocumentation:

    openapi = OpenApiSpec {
        version: "3.0.0",
        title: "{build_metadata.package_name} API",
        description: "API for WASM microservices generated by Portalis",
        servers: [
            ServerConfig {
                url: "http://localhost:{container_config.service_port}",
                description: "Development server"
            }
        ],
        paths: Map::new(),
        components: Components {
            schemas: Map::new(),
            responses: Map::new(),
            parameters: Map::new()
        }
    }

    // Add health check endpoint
    openapi.paths.insert("/health", PathItem {
        path: "/health",
        methods: Map::from([
            (HttpMethod::GET, Operation {
                summary: "Health check",
                description: "Check service health status",
                parameters: [],
                request_body: None,
                responses: Map::from([
                    (200, Response {
                        description: "Service is healthy",
                        content: Map::from([
                            ("application/json", MediaTypeObject {
                                schema: JsonSchema {
                                    schema_type: "object",
                                    properties: Map::from([
                                        ("status", JsonSchema { schema_type: "string", ... }),
                                        ("uptime_seconds", JsonSchema { schema_type: "number", ... })
                                    ]),
                                    required: ["status"]
                                }
                            })
                        ])
                    })
                ])
            })
        ])
    })

    // Add inference endpoints for each WASM module
    FOR EACH wasm_binary IN wasm_binaries:
        endpoint_path = "/v1/models/{wasm_binary.crate_name}/infer"

        openapi.paths.insert(endpoint_path, PathItem {
            path: endpoint_path,
            methods: Map::from([
                (HttpMethod::POST, Operation {
                    summary: "Run inference on {wasm_binary.crate_name}",
                    description: "Execute WASM module with input data",
                    parameters: [],
                    request_body: Some(RequestBody {
                        description: "Input data for inference",
                        content: Map::from([
                            ("application/json", MediaTypeObject {
                                schema: JsonSchema {
                                    schema_type: "object",
                                    properties: Map::from([
                                        ("inputs", JsonSchema { schema_type: "array", ... })
                                    ]),
                                    required: ["inputs"]
                                }
                            })
                        ]),
                        required: true
                    }),
                    responses: Map::from([
                        (200, Response {
                            description: "Inference successful",
                            content: Map::from([
                                ("application/json", MediaTypeObject {
                                    schema: JsonSchema {
                                        schema_type: "object",
                                        properties: Map::from([
                                            ("outputs", JsonSchema { schema_type: "array", ... })
                                        ])
                                    }
                                })
                            ])
                        })
                    ])
                })
            ])
        })

    // Serialize to JSON
    openapi_json = serialize_openapi_to_json(openapi)

    // Write to file
    doc_path = "api/openapi.json"
    write_file(doc_path, openapi_json)

    RETURN ApiDocumentation {
        format: ApiDocFormat::OpenAPI,
        file_path: Path::new(doc_path),
        openapi_spec: Some(openapi),
        grpc_proto: None,
        grpc_descriptor: None
    }
```

---

## 4. Input/Output Contracts

### 4.1 Agent Input Contract

```yaml
INPUT: PackagingAgentInput
  wasm_binaries:
    binaries: List<WasmBinary>       # From Build Agent
    metadata: BuildManifest           # Build metadata

  test_results:
    results_path: Path                # Test results JSON
    parity_report_path: Path          # Parity analysis
    benchmarks_path: Path             # Performance benchmarks

  packaging_configuration:
    package_name: String
    package_version: String           # Semantic version
    description: String

    # Platform targets
    enable_nim_packaging: bool        # Default: true
    enable_triton_integration: bool   # Default: true
    enable_omniverse_packaging: bool  # Default: true (SHOULD)
    enable_distribution_archive: bool # Default: true

    # Container config
    container_config: ContainerConfiguration

    # Triton config
    triton_config: TritonConfiguration

    # Omniverse config
    omniverse_config: OmniverseConfiguration

    # Distribution config
    distribution_config: DistributionConfiguration

    # Output location
    output_directory: Path

  context:
    pipeline_id: String
    previous_agent: "TestAgent"
    mode: "script" | "library"

PRECONDITIONS:
  - wasm_binaries.binaries.len() > 0
  - All WASM binaries exist and are valid
  - test_results paths exist and are readable
  - output_directory is writable
  - If enable_nim_packaging=true, Docker/Podman is available
  - If enable_triton_integration=true, Triton model repository path is valid
  - package_version follows semantic versioning

INVARIANTS:
  - package_name.len() > 0
  - At least one packaging target is enabled
  - container_config.service_port != container_config.metrics_port
  - triton_config.model_version > 0
```

### 4.2 Agent Output Contract

```yaml
OUTPUT: PackagingAgentOutput
  status: "success" | "partial_success" | "failure"

  artifacts:
    nim_containers: List<ContainerArtifact>
    triton_models: List<TritonModelArtifact>
    omniverse_packages: List<OmniversePackageArtifact>
    distribution_archives: List<DistributionArchive>
    api_documentation: List<ApiDocumentation>
    integration_guides: List<IntegrationGuide>
    deployment_manifests: List<DeploymentManifest>

  package_metadata:
    package_manifest: PackageManifest
    checksums: Map<Path, String>      # SHA256 checksums
    total_artifacts: u32
    total_size_bytes: u64

  packaging_metrics:
    packaging_time_seconds: f64
    nim_generation_time: Duration
    triton_config_time: Duration
    archive_creation_time: Duration

  diagnostics:
    warnings: List<Warning>
    errors: List<PackagingError>
    validation_results: List<ValidationResult>

POSTCONDITIONS:
  - IF status == "success":
      - All requested packaging targets produced artifacts
      - All artifacts are valid and accessible
      - package_manifest is complete
      - All checksums are generated
      - distribution_archives.len() >= 1

  - IF status == "partial_success":
      - At least one packaging target succeeded
      - errors contains details of failed targets
      - Successful artifacts are available

  - IF status == "failure":
      - No usable artifacts produced
      - errors contains failure reasons

  - ALWAYS:
      - All file paths are absolute
      - All checksums are SHA-256
      - Package manifest is valid JSON/YAML
      - All artifacts have size > 0

SIDE EFFECTS:
  - Creates container images if NIM packaging enabled
  - Creates Triton model repository if Triton enabled
  - Creates archive files in output directory
  - May push images to registry if publishing enabled
  - Generates documentation files
  - Creates deployment manifests
```

### 4.3 Error Output Contract

```yaml
ERROR_OUTPUT: PackagingError
  error_type: ErrorType
  message: String
  context: ErrorContext
  remediation: RemediationSuggestion

  details: ONEOF
    - NimGenerationError:
        failed_containers: List<String>
        docker_errors: List<String>
        build_logs: List<String>

    - TritonConfigError:
        failed_models: List<String>
        config_errors: List<String>
        validation_errors: List<String>

    - OmniversePackagingError:
        incompatible_modules: List<String>
        validation_errors: List<String>

    - ArchiveCreationError:
        failed_archives: List<String>
        io_errors: List<String>

    - PublishingError:
        registry_url: String
        auth_error: Option<String>
        push_errors: List<String>

ENUM ErrorType:
  - ContainerBuildError
  - TritonConfigError
  - OmniverseCompatibilityError
  - ArchiveError
  - DocumentationError
  - PublishingError
  - ValidationError
  - IOError

STRUCTURE ErrorContext:
  pipeline_id: String
  agent: "PackagingAgent"
  stage: PackagingStage
  timestamp: DateTime

ENUM PackagingStage:
  - NimGeneration
  - TritonConfiguration
  - OmniversePackaging
  - DocumentationGeneration
  - ArchiveCreation
  - Publishing
```

---

## 5. Error Handling Strategy

### 5.1 Error Categories and Handling

```rust
ERROR HANDLING TAXONOMY:

1. CONTAINER BUILD ERRORS:
   - Docker daemon not available
   - Dockerfile generation errors
   - Image build failures
   - Image export failures

   STRATEGY:
     - Validate Docker availability before starting
     - Provide detailed build logs
     - Allow partial success (skip failed containers)
     - Suggest Dockerfile fixes for common issues

2. TRITON CONFIGURATION ERRORS:
   - Invalid model configuration
   - Incompatible backend
   - Version conflicts
   - Repository structure errors

   STRATEGY:
     - Validate config.pbtxt syntax
     - Check Triton compatibility
     - Provide example configurations
     - Allow partial success

3. OMNIVERSE COMPATIBILITY ERRORS:
   - WASM not WASI compliant
   - Threading usage detected
   - Size limitations exceeded
   - Kit version incompatibility

   STRATEGY:
     - Validate before packaging
     - Provide compatibility report
     - Warn on potential issues
     - Omniverse is optional, continue on failure

4. ARCHIVE CREATION ERRORS:
   - Insufficient disk space
   - Permission errors
   - Compression failures
   - Checksum generation errors

   STRATEGY:
     - Check disk space before starting
     - Validate permissions early
     - Retry with different compression
     - Provide partial archives if possible

5. DOCUMENTATION GENERATION ERRORS:
   - Missing metadata
   - Template errors
   - Serialization failures

   STRATEGY:
     - Use sensible defaults
     - Generate minimal docs on error
     - Log warnings but continue

6. PUBLISHING ERRORS:
   - Registry unavailable
   - Authentication failures
   - Network errors
   - Push failures

   STRATEGY:
     - Retry with backoff
     - Provide clear auth instructions
     - Publishing is optional, continue
     - Save artifacts locally regardless
```

### 5.2 Partial Success Handling

```rust
ALGORITHM handle_partial_packaging_success(
    successful_artifacts: PackagingArtifacts,
    failed_stages: List<PackagingStage>,
    errors: List<PackagingError>
) -> PackagingAgentOutput:

    LOG("Partial packaging success", WARN)
    LOG("Succeeded: {successful_artifacts.total_artifacts_count()} artifacts", INFO)
    LOG("Failed: {failed_stages.len()} stages", WARN)

    // Generate package manifest even for partial success
    package_manifest = generate_package_manifest(
        successful_artifacts,
        failed_stages,
        errors
    )

    // Note failed stages in manifest
    package_manifest.notes.push("Partial packaging - some stages failed")
    FOR EACH stage IN failed_stages:
        package_manifest.notes.push("Failed: {stage}")

    output = PackagingAgentOutput {
        status: "partial_success",
        artifacts: successful_artifacts,
        package_metadata: PackageMetadata {
            package_manifest: package_manifest,
            checksums: generate_checksums(successful_artifacts),
            total_artifacts: successful_artifacts.total_artifacts_count(),
            total_size_bytes: successful_artifacts.total_size_bytes()
        },
        diagnostics: Diagnostics {
            errors: errors,
            warnings: generate_warnings_for_partial_success(failed_stages)
        }
    }

    RETURN output


FUNCTION generate_warnings_for_partial_success(
    failed_stages: List<PackagingStage>
) -> List<Warning>:

    warnings = List::new()

    FOR EACH stage IN failed_stages:
        MATCH stage:
            PackagingStage::NimGeneration:
                warnings.push(Warning {
                    message: "NIM containers not generated - Docker deployment unavailable",
                    severity: WarningSeverity::High,
                    remediation: "Install Docker and retry packaging"
                })

            PackagingStage::TritonConfiguration:
                warnings.push(Warning {
                    message: "Triton models not configured - Triton deployment unavailable",
                    severity: WarningSeverity::High,
                    remediation: "Check Triton configuration and retry"
                })

            PackagingStage::OmniversePackaging:
                warnings.push(Warning {
                    message: "Omniverse packages not created",
                    severity: WarningSeverity::Medium,
                    remediation: "Review WASM compatibility requirements"
                })

            PackagingStage::Publishing:
                warnings.push(Warning {
                    message: "Artifacts not published to registry",
                    severity: WarningSeverity::Low,
                    remediation: "Manually push artifacts using provided scripts"
                })

    RETURN warnings
```

---

## 6. London School TDD Test Points

### 6.1 Test Hierarchy

```
ACCEPTANCE TESTS (Outside-In)
├─ Happy Path Tests
│  ├─ Test: Complete packaging pipeline (all targets)
│  ├─ Test: Generate NIM container from single WASM
│  ├─ Test: Configure Triton model for WASM
│  ├─ Test: Create Omniverse extension package
│  └─ Test: Generate distribution archive
│
├─ Error Path Tests
│  ├─ Test: Handle Docker unavailable
│  ├─ Test: Handle invalid Triton config
│  ├─ Test: Handle WASM incompatible with Omniverse
│  ├─ Test: Handle insufficient disk space
│  └─ Test: Handle publishing failures
│
└─ Edge Case Tests
   ├─ Test: Package with no NIM containers
   ├─ Test: Package with only distribution archive
   ├─ Test: Partial success (some stages fail)
   └─ Test: Large WASM binaries (>100MB)

INTEGRATION TESTS (Agent Contracts)
├─ Test: PackagingAgent receives correct input from TestAgent
├─ Test: PackagingAgent produces complete output
├─ Test: Docker integration
├─ Test: Triton model validation
├─ Test: Archive creation and extraction
└─ Test: Checksum verification

UNIT TESTS (Component Isolation)
├─ NIM Container Generation
│  ├─ Test: Generate Dockerfile correctly
│  ├─ Test: Build container image
│  ├─ Test: Generate health check service
│  ├─ Test: Export container image
│  └─ Test: Calculate image digest
│
├─ Triton Configuration
│  ├─ Test: Generate config.pbtxt
│  ├─ Test: Create model repository structure
│  ├─ Test: Generate Python WASM backend
│  ├─ Test: Validate Triton config
│  └─ Test: Handle dynamic batching config
│
├─ Omniverse Packaging
│  ├─ Test: Validate WASM compatibility
│  ├─ Test: Generate extension.toml
│  ├─ Test: Generate Python bindings
│  ├─ Test: Create extension package
│  └─ Test: Generate integration docs
│
├─ Distribution Archives
│  ├─ Test: Organize archive contents
│  ├─ Test: Generate README
│  ├─ Test: Generate manifest
│  ├─ Test: Create tar.gz archive
│  ├─ Test: Calculate checksums
│  └─ Test: Validate archive integrity
│
├─ API Documentation
│  ├─ Test: Generate OpenAPI spec
│  ├─ Test: Generate gRPC proto
│  ├─ Test: Add health check endpoints
│  └─ Test: Add inference endpoints
│
└─ Error Handling
   ├─ Test: Handle partial packaging success
   ├─ Test: Generate remediation suggestions
   ├─ Test: Categorize packaging errors
   └─ Test: Cleanup on failure
```

### 6.2 Mock/Stub Strategy

```rust
TEST DOUBLES FOR PACKAGING AGENT:

1. MOCKS (Behavior Verification):

   MOCK DockerClient:
     - Verifies Docker commands
     - Verifies image builds
     - Tracks container operations

     EXPECTATIONS:
       - expect_build_image(dockerfile: Path, tag: String)
       - expect_export_image(image_id: String, path: Path)
       - verify_healthcheck_configured()

   MOCK TritonValidator:
     - Verifies Triton config generation
     - Validates model repository structure
     - Tracks configuration operations

     EXPECTATIONS:
       - expect_config_generated(model_name: String)
       - expect_model_version(version: u32)
       - verify_inputs_outputs_defined()

   MOCK ArchiveCreator:
     - Verifies archive creation
     - Tracks compression operations
     - Validates checksums

     EXPECTATIONS:
       - expect_create_archive(source: Path, dest: Path)
       - expect_checksum_calculated(algorithm: ChecksumAlgorithm)

2. STUBS (State Verification):

   STUB WasmBinaryStub:
     - Returns mock WASM binaries
     - Simulates various sizes
     - Configurable metadata

     CONFIGURATIONS:
       - with_small_binary() -> WasmBinary  # <1MB
       - with_large_binary() -> WasmBinary  # >100MB
       - with_exports(exports: List<String>) -> WasmBinary

   STUB BuildMetadataStub:
     - Returns mock build metadata
     - Configurable test results
     - Simulates various scenarios

     CONFIGURATIONS:
       - with_successful_build() -> BuildManifest
       - with_partial_build() -> BuildManifest
       - with_benchmarks() -> BuildManifest

   STUB ContainerRuntimeStub:
     - Simulates Docker/Podman
     - Returns success/failure
     - Configurable delays

     CONFIGURATIONS:
       - with_docker_available() -> ContainerRuntime
       - with_docker_unavailable() -> ContainerRuntime
       - with_build_failure() -> ContainerRuntime

3. FAKES (Working Implementations):

   FAKE InMemoryArchive:
     - Fully functional in-memory archiving
     - Fast testing without disk I/O
     - Supports all archive formats

   FAKE MockDockerDaemon:
     - In-memory Docker simulation
     - Tracks all operations
     - Deterministic behavior

   FAKE TestFileSystem:
     - In-memory file system
     - Supports all file operations
     - Fast and deterministic
```

### 6.3 Key Test Scenarios

```rust
ACCEPTANCE TEST SCENARIO 1: Complete Packaging Pipeline
GIVEN:
  - Valid WASM binaries from Build Agent
  - Test results from Test Agent
  - All packaging targets enabled
  - Docker available
  - Sufficient disk space

WHEN:
  - PackagingAgent.package_artifacts() is called

THEN:
  - EXPECT status == "success"
  - EXPECT nim_containers.len() > 0
  - EXPECT triton_models.len() > 0
  - EXPECT omniverse_packages.len() > 0
  - EXPECT distribution_archives.len() > 0
  - EXPECT api_documentation.len() > 0
  - EXPECT package_manifest is valid
  - EXPECT all checksums generated

MOCKS:
  - DockerClient: verify build called
  - TritonValidator: verify config generated
  - ArchiveCreator: verify archive created


ACCEPTANCE TEST SCENARIO 2: NIM Container Generation
GIVEN:
  - Single WASM binary
  - Container configuration
  - Docker available

WHEN:
  - generate_nim_containers() is called

THEN:
  - EXPECT Dockerfile generated
  - EXPECT container image built
  - EXPECT health check configured
  - EXPECT service port exposed
  - EXPECT metrics port exposed
  - EXPECT image digest calculated

MOCKS:
  - DockerClient: verify commands
  - FileSystem: verify Dockerfile written


ACCEPTANCE TEST SCENARIO 3: Triton Model Configuration
GIVEN:
  - WASM binary with exports
  - Triton configuration with inputs/outputs
  - Model repository path

WHEN:
  - configure_triton_models() is called

THEN:
  - EXPECT model repository created
  - EXPECT config.pbtxt generated and valid
  - EXPECT WASM copied to version directory
  - EXPECT Python backend generated (if platform=Python)
  - EXPECT validation successful

MOCKS:
  - TritonValidator: verify config syntax


ACCEPTANCE TEST SCENARIO 4: Partial Success
GIVEN:
  - Valid WASM binaries
  - Docker not available
  - Triton and distribution enabled

WHEN:
  - PackagingAgent.package_artifacts() is called

THEN:
  - EXPECT status == "partial_success"
  - EXPECT nim_containers is empty
  - EXPECT triton_models.len() > 0
  - EXPECT distribution_archives.len() > 0
  - EXPECT errors contains Docker unavailable error
  - EXPECT warnings contains NIM generation warning

MOCKS:
  - DockerClient: return unavailable error


UNIT TEST SCENARIO 1: Dockerfile Generation
GIVEN:
  - WASM binary metadata
  - Container configuration

WHEN:
  - generate_dockerfile() is called

THEN:
  - EXPECT Dockerfile has FROM instruction
  - EXPECT WASM runtime installed
  - EXPECT non-root user created (if configured)
  - EXPECT health check defined
  - EXPECT ports exposed
  - EXPECT environment variables set

STUBS:
  - WasmBinaryStub


UNIT TEST SCENARIO 2: Triton config.pbtxt Generation
GIVEN:
  - Model name and version
  - Inputs and outputs specification
  - Instance group configuration

WHEN:
  - generate_triton_config_pbtxt() is called

THEN:
  - EXPECT valid protobuf text format
  - EXPECT all inputs defined
  - EXPECT all outputs defined
  - EXPECT instance groups configured
  - EXPECT version policy set

STUBS:
  - TritonConfiguration


UNIT TEST SCENARIO 3: Archive Creation
GIVEN:
  - Staging directory with artifacts
  - Archive configuration
  - Sufficient disk space

WHEN:
  - create_archive() is called

THEN:
  - EXPECT archive file created
  - EXPECT compressed size < uncompressed size
  - EXPECT archive contains all files
  - EXPECT checksums generated
  - EXPECT manifest included

MOCKS:
  - FileSystem: verify operations
  - ArchiveCreator: verify compression


UNIT TEST SCENARIO 4: OpenAPI Spec Generation
GIVEN:
  - WASM binaries with exports
  - Container configuration

WHEN:
  - generate_openapi_spec() is called

THEN:
  - EXPECT OpenAPI 3.0 compliant
  - EXPECT /health endpoint defined
  - EXPECT inference endpoints for each module
  - EXPECT request/response schemas defined
  - EXPECT server URL configured

STUBS:
  - WasmBinaryStub with known exports


INTEGRATION TEST SCENARIO 1: Docker Integration
GIVEN:
  - Real Docker daemon available
  - Simple test Dockerfile

WHEN:
  - Actual Docker build is executed

THEN:
  - EXPECT image built successfully
  - EXPECT image has correct tags
  - EXPECT container can be started
  - EXPECT health check works

NOTE: Uses real Docker, not mocked


INTEGRATION TEST SCENARIO 2: Contract with TestAgent
GIVEN:
  - Mock TestAgent output

WHEN:
  - PackagingAgent receives input

THEN:
  - EXPECT all required fields present
  - EXPECT WASM binaries valid
  - EXPECT test results accessible
  - EXPECT build metadata complete

MOCKS:
  - Mock TestAgent with contract-compliant output


CONTRACT TEST: Input from TestAgent
TEST verify_input_contract_from_test_agent():
    mock_test_agent = MockTestAgent()
    test_output = mock_test_agent.generate_output()

    packaging_input = PackagingAgentInput::from_test_output(test_output)

    ASSERT packaging_input.wasm_binaries.len() > 0
    ASSERT packaging_input.test_results.exists()
    ASSERT packaging_input.parity_report.exists()
    ASSERT packaging_input.context.previous_agent == "TestAgent"


CONTRACT TEST: Output Completeness
TEST verify_output_contract_completeness():
    packaging_agent = PackagingAgent::new()
    test_input = create_test_packaging_input()

    output = packaging_agent.package_artifacts(test_input)

    ASSERT output.status IN ["success", "partial_success", "failure"]

    IF output.status == "success":
        ASSERT output.artifacts.distribution_archives.len() > 0

        FOR EACH archive IN output.artifacts.distribution_archives:
            ASSERT archive.archive_path.exists()
            ASSERT archive.checksums.len() > 0
            ASSERT archive.manifest_path.exists()
            ASSERT archive.readme_path.exists()
```

### 6.4 Coverage Requirements

```
COVERAGE TARGETS FOR PACKAGING AGENT:

Line Coverage: >80%
  - All packaging algorithms covered
  - All error paths tested
  - All format generators tested

Branch Coverage: >75%
  - All conditional packaging logic
  - All error handling branches
  - All format variations

Contract Coverage: 100%
  - Input contract fully tested
  - Output contract fully tested
  - Error contract fully tested
  - Integration contracts tested

Mutation Testing: >70%
  - Critical packaging logic mutations detected
  - Error handling mutations detected
  - Validation logic mutations detected

CRITICAL PATHS REQUIRING 100% COVERAGE:
  - Dockerfile generation
  - Triton config.pbtxt generation
  - Archive creation and integrity
  - Checksum calculation
  - Manifest generation
  - Contract validation
```

---

## Document Status

**SPARC Phase 2 (Pseudocode): PACKAGING AGENT COMPLETE**

This document provides comprehensive pseudocode for the Packaging Agent, covering:
- Complete data structures for containers, Triton, Omniverse, and distribution
- Detailed algorithms for NIM generation, Triton integration, and archiving
- Docker image layering with multi-stage builds
- Triton model versioning and configuration
- Omniverse extension packaging with WASM validation
- Distribution manifest generation with inventory
- Comprehensive error handling for partial success
- London School TDD test points with 80%+ coverage targets

### Key Deliverables

1. **NIM Service Generation**: Docker containers with health checks, WASM runtime, service wrappers
2. **Triton Integration**: config.pbtxt generation, Python WASM backend, model repository structure
3. **Omniverse Compatibility**: Extension packaging, Python bindings, compatibility validation
4. **Distribution Packaging**: Organized archives, comprehensive manifests, checksums, README

### Next Steps
1. **Phase 3 (Architecture)**: Detailed component design and interfaces
2. **Phase 4 (Refinement)**: Implementation and iterative improvement
3. **Phase 5 (Completion)**: Final implementation and validation

---

**END OF PACKAGING AGENT PSEUDOCODE**
