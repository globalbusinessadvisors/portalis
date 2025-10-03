# PORTALIS: Orchestration Layer Pseudocode
## SPARC Phase 2: Pseudocode - Orchestration Layer (Layer 2)

**Version:** 1.0
**Date:** 2025-10-03
**Layer:** Orchestration Layer (Layer 2 of 5)
**Testing Approach:** London School TDD (Outside-In, Mockist)

---

## Table of Contents

1. [Layer Overview](#1-layer-overview)
2. [Data Structures](#2-data-structures)
3. [Core Algorithms](#3-core-algorithms)
4. [Input/Output Contracts](#4-inputoutput-contracts)
5. [Error Handling Strategy](#5-error-handling-strategy)
6. [London School TDD Test Points](#6-london-school-tdd-test-points)

---

## 1. Layer Overview

### 1.1 Purpose

The **Orchestration Layer** is the command-and-control center of the Portalis pipeline. It sits between the Presentation Layer (CLI, API, Dashboard) and the Agent Swarm Layer (7 specialized agents). Its responsibilities are:

- **Flow Control**: Manage the end-to-end pipeline execution from Python input to packaged WASM output
- **Agent Coordination**: Spawn, monitor, and coordinate the 7 specialized agents
- **Pipeline State Management**: Track pipeline state, support checkpointing and resumption
- **Error Recovery**: Implement retry logic, circuit breakers, and graceful degradation
- **Progress Reporting**: Provide real-time progress updates, ETA estimation, and observability

### 1.2 Components

The Orchestration Layer consists of three primary components:

#### **1. Flow Controller**
- Manages overall pipeline execution flow
- Implements pipeline state machine (stages, transitions, error states)
- Coordinates sequential, parallel, and conditional agent execution
- Handles mode-specific workflows (Script Mode vs Library Mode)

#### **2. Agent Coordinator**
- Manages agent lifecycle (spawn, monitor, shutdown)
- Routes messages between agents
- Implements agent supervision (health checks, timeouts, restarts)
- Provides agent isolation and resource management

#### **3. Pipeline Manager**
- Maintains pipeline state and metadata
- Implements checkpointing for long-running Library Mode executions
- Supports pipeline resumption from checkpoints
- Provides pipeline history and audit logging

### 1.3 Responsibilities (from Specification)

**Functional Requirements Addressed:**

- **FR-2.8.1**: Logging
  - Structured logs with JSON format
  - Error context and stack traces
  - Configurable log levels (DEBUG, INFO, WARN, ERROR)

- **FR-2.8.2**: Progress Reporting
  - Progress percentages for each pipeline stage
  - ETA estimation based on historical data
  - Current stage identification

- **FR-2.8.3**: Metrics Collection
  - Translation success rates
  - Compilation success/failure rates
  - GPU utilization during accelerated phases
  - End-to-end processing time

- **FR-2.8.4**: Report Generation
  - Summary reports with all metrics
  - Translation coverage statistics
  - Test results and parity analysis
  - Performance benchmarks

**Non-Functional Requirements Addressed:**

- **NFR-3.1.1**: Performance
  - Script Mode: <60 seconds end-to-end
  - Library Mode: <15 minutes for 5000 LOC

- **NFR-3.2.1**: Fault Tolerance
  - Retry logic for transient errors
  - Checkpointing for resumable operations
  - Graceful degradation on agent failures

### 1.4 Design Principles

1. **Pipeline as State Machine**: Clear stages, transitions, and error handling
2. **Agent Independence**: Agents are independent processes/services with well-defined contracts
3. **Observability First**: All operations logged, metrics collected, progress tracked
4. **Fault Tolerance**: Retry, circuit breaking, and graceful degradation built-in
5. **Testability**: All components use dependency injection for London School TDD

### 1.5 Layer Boundaries

**Inputs from Presentation Layer:**
- Pipeline execution requests (Script Mode or Library Mode)
- Configuration objects
- Control commands (pause, resume, cancel)

**Outputs to Presentation Layer:**
- Pipeline execution results
- Real-time progress updates
- Error reports and warnings
- Metrics and performance data

**Interactions with Agent Swarm Layer:**
- Agent spawn/shutdown commands
- Inter-agent message routing
- Agent health monitoring
- Resource allocation requests

**Interactions with Infrastructure Layer:**
- Checkpoint persistence (storage)
- Metrics collection (monitoring)
- Cache access (performance)

---

## 2. Data Structures

### 2.1 Core Enums

```pseudocode
// Defines the execution mode
ENUM ExecutionMode:
    Script      // Single Python file (<60s target)
    Library     // Full package (<15min target)

// Defines pipeline execution stages
ENUM PipelineStage:
    Initializing    // Setting up pipeline
    Ingesting       // Ingest Agent processing
    Analyzing       // Analysis Agent processing
    Specifying      // Specification Generator processing
    Transpiling     // Transpiler Agent processing
    Building        // Build Agent processing
    Testing         // Test Agent processing
    Packaging       // Packaging Agent processing
    Completed       // All stages complete
    Failed          // Pipeline failed
    Cancelled       // User cancelled

// Defines pipeline execution state
ENUM PipelineState:
    Pending         // Waiting to start
    Running         // Currently executing
    Paused          // Temporarily paused
    Checkpointed    // Saved for later resumption
    Completed       // Successfully completed
    Failed          // Failed with errors
    Cancelled       // User cancelled

// Defines agent execution status
ENUM AgentStatus:
    Idle            // Agent not assigned work
    Starting        // Agent initializing
    Running         // Agent processing
    Waiting         // Waiting for dependencies
    Retrying        // Retrying after transient error
    Completed       // Finished successfully
    Failed          // Failed with error
    TimedOut        // Exceeded timeout
    Crashed         // Unexpected termination

// Defines retry strategy
ENUM RetryStrategy:
    NoRetry         // Don't retry
    Immediate       // Retry immediately
    ExponentialBackoff  // Exponential backoff
    Fixed           // Fixed delay between retries

// Defines circuit breaker state
ENUM CircuitBreakerState:
    Closed          // Normal operation
    Open            // Blocking requests (too many failures)
    HalfOpen        // Testing if service recovered
```

### 2.2 Pipeline State Structures

```pseudocode
STRUCT PipelineConfiguration:
    // Execution mode and source
    mode: ExecutionMode
    source_path: Path
    output_directory: Path

    // Agent configuration
    agent_timeout_seconds: Map<String, Integer>  // Timeout per agent type
    max_retries_per_agent: Integer = 3
    retry_strategy: RetryStrategy = ExponentialBackoff

    // Performance settings
    enable_parallel_execution: Boolean = true
    max_concurrent_agents: Integer = 4
    enable_gpu_acceleration: Boolean = true

    // Checkpointing
    enable_checkpointing: Boolean = true  // Always true for Library Mode
    checkpoint_interval_seconds: Integer = 300  // Checkpoint every 5 minutes
    checkpoint_storage_path: Path

    // Observability
    log_level: String = "INFO"
    enable_metrics: Boolean = true
    enable_progress_reporting: Boolean = true
    enable_tracing: Boolean = true

    // Error handling
    fail_fast: Boolean = false  // Continue on agent failure if false
    circuit_breaker_enabled: Boolean = true
    circuit_breaker_threshold: Integer = 5  // Failures before opening circuit


STRUCT PipelineMetadata:
    // Identification
    pipeline_id: String                     // Unique pipeline execution ID (UUID)
    created_at: Timestamp
    updated_at: Timestamp

    // Configuration snapshot
    configuration: PipelineConfiguration

    // Execution context
    user_id: Optional<String>               // Who initiated the pipeline
    session_id: Optional<String>            // Session context
    parent_pipeline_id: Optional<String>    // If resumed from checkpoint

    // Tags and annotations
    tags: Map<String, String>               // User-defined tags
    annotations: Map<String, String>        // System-generated metadata


STRUCT PipelineExecutionState:
    // Current state
    pipeline_id: String
    current_stage: PipelineStage
    state: PipelineState
    metadata: PipelineMetadata

    // Stage tracking
    completed_stages: List<PipelineStage>
    current_agent: Optional<String>         // Currently executing agent
    pending_stages: List<PipelineStage>

    // Progress tracking
    overall_progress_percent: Float         // 0.0 to 100.0
    stage_progress_percent: Float           // Progress within current stage
    estimated_completion_time: Optional<Timestamp>

    // Timing
    started_at: Timestamp
    completed_at: Optional<Timestamp>
    stage_start_times: Map<PipelineStage, Timestamp>
    stage_durations: Map<PipelineStage, Integer>  // Duration in milliseconds

    // Results and artifacts
    stage_results: Map<PipelineStage, AgentResult>
    final_result: Optional<PipelineResult>

    // Error tracking
    errors: List<PipelineError>
    warnings: List<PipelineWarning>
    retry_count_per_stage: Map<PipelineStage, Integer>

    // Checkpointing
    last_checkpoint_time: Optional<Timestamp>
    checkpoint_count: Integer


STRUCT AgentRegistration:
    // Agent identification
    agent_id: String                        // Unique instance ID
    agent_type: String                      // "ingest", "analysis", etc.
    agent_name: String                      // Human-readable name

    // Agent connection info
    endpoint: String                        // URL or address for agent communication
    protocol: String                        // "http", "grpc", "zeromq"

    // Agent capabilities
    capabilities: List<String>              // What this agent can do
    required_resources: ResourceRequirements

    // Health monitoring
    status: AgentStatus
    last_heartbeat: Timestamp
    consecutive_failures: Integer

    // Performance tracking
    total_tasks_processed: Integer
    total_failures: Integer
    average_processing_time_ms: Float


STRUCT ResourceRequirements:
    cpu_cores: Integer                      // Number of CPU cores needed
    memory_mb: Integer                      // RAM in megabytes
    gpu_required: Boolean                   // Whether GPU is needed
    gpu_memory_mb: Integer                  // VRAM if GPU required
    disk_mb: Integer                        // Temporary disk space needed
    network_bandwidth_mbps: Integer         // Network bandwidth needed
```

### 2.3 Agent Coordination Structures

```pseudocode
STRUCT AgentTask:
    // Task identification
    task_id: String                         // Unique task ID
    agent_type: String                      // Which agent type should handle this
    pipeline_id: String                     // Parent pipeline

    // Task payload
    input_data: AgentInput                  // Input for agent (polymorphic)
    context: TaskContext                    // Execution context

    // Scheduling
    priority: Integer                       // Higher = more urgent (0-10)
    dependencies: List<String>              // Task IDs that must complete first
    timeout_seconds: Integer
    retry_policy: RetryPolicy

    // State tracking
    status: AgentStatus
    assigned_agent_id: Optional<String>     // Which agent instance is handling this
    submitted_at: Timestamp
    started_at: Optional<Timestamp>
    completed_at: Optional<Timestamp>

    // Results
    result: Optional<AgentResult>
    error: Optional<AgentError>


STRUCT TaskContext:
    // Pipeline context
    pipeline_id: String
    execution_mode: ExecutionMode

    // Resource context
    allocated_cpu_cores: Integer
    allocated_memory_mb: Integer
    allocated_gpu: Optional<GpuAllocation>

    // Tracing and observability
    trace_id: String                        // Distributed tracing
    span_id: String
    parent_span_id: Optional<String>

    // Configuration
    configuration: Map<String, Any>         // Agent-specific config


STRUCT AgentInput:
    // Base polymorphic type for agent inputs
    // Specific agents have their own input types:
    // - IngestInput
    // - AnalysisInput
    // - SpecificationInput
    // etc.

    input_type: String                      // "ingest", "analysis", etc.
    data: Map<String, Any>                  // Serializable data


STRUCT AgentResult:
    // Base polymorphic type for agent results
    task_id: String
    agent_id: String
    agent_type: String

    // Result status
    success: Boolean
    result_type: String                     // "ingest_result", "analysis_result", etc.
    data: Map<String, Any>                  // Serializable result data

    // Performance metrics
    processing_time_ms: Integer
    cpu_time_ms: Integer
    gpu_time_ms: Integer
    memory_peak_mb: Integer

    // Outputs
    artifacts: List<Artifact>               // Files, binaries, reports generated
    metrics: Map<String, Float>             // Agent-specific metrics

    // Issues
    warnings: List<String>
    info_messages: List<String>


STRUCT Artifact:
    // Represents a file or data artifact produced by an agent
    artifact_id: String
    artifact_type: String                   // "rust_source", "wasm_binary", "report"
    path: Path                              // Location on disk or in storage
    size_bytes: Integer
    checksum: String                        // SHA-256
    created_at: Timestamp
    metadata: Map<String, Any>              // Artifact-specific metadata


STRUCT RetryPolicy:
    max_retries: Integer                    // Maximum retry attempts
    strategy: RetryStrategy
    base_delay_ms: Integer                  // Base delay for backoff strategies
    max_delay_ms: Integer                   // Maximum delay cap
    timeout_multiplier: Float               // Increase timeout on each retry (e.g., 1.5x)
    retryable_errors: List<String>          // Which error types to retry
```

### 2.4 Checkpoint Structures

```pseudocode
STRUCT Checkpoint:
    // Checkpoint identification
    checkpoint_id: String                   // Unique checkpoint ID
    pipeline_id: String                     // Associated pipeline
    sequence_number: Integer                // Checkpoint sequence (1, 2, 3...)

    // State snapshot
    execution_state: PipelineExecutionState
    agent_states: Map<String, AgentState>   // State of each agent

    // Artifacts snapshot
    artifacts_manifest: List<Artifact>      // All artifacts at checkpoint time
    intermediate_results: Map<PipelineStage, AgentResult>

    // Metadata
    created_at: Timestamp
    storage_path: Path                      // Where checkpoint is stored
    size_bytes: Integer
    compressed: Boolean

    // Resumption info
    can_resume: Boolean                     // Whether this checkpoint is resumable
    resume_from_stage: PipelineStage        // Which stage to resume from
    required_artifacts: List<String>        // Artifacts needed for resumption


STRUCT AgentState:
    // Agent state at checkpoint time
    agent_id: String
    agent_type: String
    status: AgentStatus

    // Internal state (agent-specific)
    internal_state: Map<String, Any>        // Serialized agent state
    pending_work: List<String>              // Unfinished work items

    // Resources held
    allocated_resources: ResourceAllocation


STRUCT ResourceAllocation:
    cpu_cores: List<Integer>                // Which CPU cores allocated
    memory_pages: List<String>              // Memory regions allocated
    gpu_device_id: Optional<Integer>        // GPU device ID if allocated
    gpu_memory_handles: List<String>        // GPU memory allocations
```

### 2.5 Error Recovery Structures

```pseudocode
STRUCT CircuitBreaker:
    // Circuit breaker for an agent type or external service
    name: String                            // What this protects
    state: CircuitBreakerState
    failure_threshold: Integer              // Failures before opening
    success_threshold: Integer              // Successes to close from half-open
    timeout_ms: Integer                     // How long to stay open

    // State tracking
    consecutive_failures: Integer
    consecutive_successes: Integer
    last_state_change: Timestamp
    failure_count_window: List<Timestamp>   // Recent failures (sliding window)

    // Statistics
    total_requests: Integer
    total_failures: Integer
    total_successes: Integer
    total_rejected: Integer                 // Rejected while open


STRUCT PipelineError:
    // Represents an error during pipeline execution
    error_id: String
    pipeline_id: String
    stage: PipelineStage
    agent_type: Optional<String>

    // Error details
    error_type: String                      // "AgentTimeout", "CompilationFailed", etc.
    message: String
    details: Map<String, Any>               // Additional context
    stack_trace: Optional<String>

    // Recovery
    recoverable: Boolean                    // Can pipeline continue?
    retry_attempted: Boolean
    retry_count: Integer

    // Timing
    occurred_at: Timestamp


STRUCT PipelineWarning:
    warning_id: String
    pipeline_id: String
    stage: PipelineStage
    message: String
    category: String                        // "performance", "quality", "compatibility"
    occurred_at: Timestamp
```

### 2.6 Progress Tracking Structures

```pseudocode
STRUCT ProgressUpdate:
    // Real-time progress update event
    pipeline_id: String
    timestamp: Timestamp

    // Current status
    current_stage: PipelineStage
    stage_name: String                      // Human-readable stage name
    overall_progress_percent: Float         // 0.0 to 100.0
    stage_progress_percent: Float

    // Timing
    elapsed_time_ms: Integer                // Time since pipeline start
    estimated_remaining_ms: Optional<Integer>
    estimated_completion_time: Optional<Timestamp>

    // Current activity
    current_activity: String                // "Parsing module foo.py", "Compiling crate bar"
    items_processed: Integer                // Context-dependent (files, functions, tests)
    total_items: Integer

    // Performance
    throughput: Optional<Float>             // Items per second
    gpu_utilization_percent: Optional<Float>
    memory_usage_mb: Integer


STRUCT ProgressTracker:
    // Tracks progress for a pipeline stage
    stage: PipelineStage
    total_work_units: Integer               // Total work to do
    completed_work_units: Integer           // Completed so far
    failed_work_units: Integer              // Failed items

    // Timing
    started_at: Timestamp
    last_update_at: Timestamp

    // Rate estimation
    recent_completion_times: List<Integer>  // Recent work unit durations (sliding window)
    average_work_unit_time_ms: Float        // Moving average

    // ETA calculation
    estimated_remaining_work_units: Integer
    estimated_completion_time: Optional<Timestamp>
```

### 2.7 Output Structures

```pseudocode
STRUCT PipelineResult:
    // Final result of pipeline execution
    pipeline_id: String
    success: Boolean
    mode: ExecutionMode

    // Execution summary
    started_at: Timestamp
    completed_at: Timestamp
    total_duration_ms: Integer
    stage_durations: Map<PipelineStage, Integer>

    // Artifacts produced
    artifacts: List<Artifact>
    rust_workspace_path: Optional<Path>
    wasm_binaries: List<Path>
    nim_containers: List<Path>
    reports: List<Path>

    // Quality metrics
    translation_coverage: Float             // 0.0 to 1.0
    test_pass_rate: Float
    conformance_score: Float
    performance_improvement: Float          // Speedup factor

    // Resource usage
    peak_memory_mb: Integer
    total_cpu_time_ms: Integer
    total_gpu_time_ms: Integer
    gpu_utilization_avg: Optional<Float>

    // Issues
    errors: List<PipelineError>
    warnings: List<PipelineWarning>

    // Metadata
    metadata: PipelineMetadata
    stage_results: Map<PipelineStage, AgentResult>


STRUCT PipelineReport:
    // Comprehensive report for user
    result: PipelineResult

    // Formatted sections
    executive_summary: String
    stage_summaries: Map<PipelineStage, String>
    metrics_section: String
    errors_section: String
    warnings_section: String
    recommendations: List<String>

    // Visualizations (as data for rendering)
    timeline_data: TimelineVisualization
    metrics_charts: List<ChartData>

    // Export formats
    FUNCTION to_json() -> String
    FUNCTION to_yaml() -> String
    FUNCTION to_html() -> String
    FUNCTION to_markdown() -> String


STRUCT TimelineVisualization:
    // Data for rendering pipeline execution timeline
    stages: List<StageTimelineEntry>
    total_duration_ms: Integer
    critical_path: List<String>             // Stages on critical path


STRUCT StageTimelineEntry:
    stage: PipelineStage
    start_time_offset_ms: Integer           // Offset from pipeline start
    duration_ms: Integer
    status: String                          // "success", "failed", "skipped"
    parallel_with: List<PipelineStage>      // Stages that ran in parallel
```

---

## 3. Core Algorithms

### 3.1 Flow Controller - Main Pipeline Execution

```pseudocode
FUNCTION execute_pipeline(config: PipelineConfiguration) -> Result<PipelineResult, PipelineError>:
    """
    Main entry point for pipeline execution.

    Design Notes:
    - Implements the pipeline state machine
    - Coordinates all agents through Agent Coordinator
    - Manages checkpointing through Pipeline Manager
    - Handles mode-specific workflows (Script vs Library)
    - Collects metrics and generates final report

    TDD Note: This is the primary acceptance test point.
    Mock AgentCoordinator, PipelineManager, and ProgressReporter.
    """

    // Step 1: Initialize pipeline
    LOG.info("Initializing pipeline", mode=config.mode, source=config.source_path)

    pipeline_id = generate_uuid()
    metadata = PipelineMetadata {
        pipeline_id: pipeline_id,
        created_at: current_timestamp(),
        updated_at: current_timestamp(),
        configuration: config,
        tags: {},
        annotations: {}
    }

    execution_state = PipelineExecutionState {
        pipeline_id: pipeline_id,
        current_stage: PipelineStage.Initializing,
        state: PipelineState.Running,
        metadata: metadata,
        completed_stages: [],
        pending_stages: determine_pipeline_stages(config.mode),
        overall_progress_percent: 0.0,
        stage_progress_percent: 0.0,
        started_at: current_timestamp(),
        stage_start_times: {},
        stage_durations: {},
        stage_results: {},
        errors: [],
        warnings: [],
        retry_count_per_stage: {},
        checkpoint_count: 0
    }

    // Step 2: Initialize components
    agent_coordinator = AgentCoordinator.new(config)
    pipeline_manager = PipelineManager.new(execution_state, config)
    progress_reporter = ProgressReporter.new(pipeline_id)
    metrics_collector = MetricsCollector.new(pipeline_id)

    TRY:
        // Step 3: Check for resumable checkpoint
        IF config.enable_checkpointing:
            checkpoint = pipeline_manager.find_latest_checkpoint(config.source_path)

            IF checkpoint IS NOT None AND checkpoint.can_resume:
                LOG.info("Found resumable checkpoint", checkpoint_id=checkpoint.checkpoint_id)

                // Restore state from checkpoint
                execution_state = pipeline_manager.restore_from_checkpoint(checkpoint)

                LOG.info("Resuming from stage", stage=execution_state.current_stage)

        // Step 4: Execute pipeline stages
        FOR stage IN execution_state.pending_stages:
            execution_state.current_stage = stage
            execution_state.stage_start_times[stage] = current_timestamp()

            LOG.info("Starting stage", stage=stage)
            progress_reporter.update_stage(stage, 0.0)

            // Execute stage
            stage_result = execute_stage(
                stage,
                execution_state,
                agent_coordinator,
                pipeline_manager,
                progress_reporter,
                metrics_collector,
                config
            )

            MATCH stage_result:
                CASE Success(result):
                    // Stage completed successfully
                    execution_state.stage_results[stage] = result
                    execution_state.completed_stages.append(stage)

                    duration = current_timestamp() - execution_state.stage_start_times[stage]
                    execution_state.stage_durations[stage] = duration

                    LOG.info("Stage completed", stage=stage, duration_ms=duration)
                    progress_reporter.update_stage(stage, 100.0)

                    // Update overall progress
                    completed_count = execution_state.completed_stages.length
                    total_count = execution_state.pending_stages.length
                    execution_state.overall_progress_percent = (completed_count / total_count) * 100.0

                    // Checkpoint if enabled and interval passed
                    IF config.enable_checkpointing:
                        IF should_checkpoint(execution_state, config):
                            checkpoint_result = pipeline_manager.create_checkpoint(execution_state)

                            MATCH checkpoint_result:
                                CASE Success(checkpoint):
                                    execution_state.last_checkpoint_time = current_timestamp()
                                    execution_state.checkpoint_count += 1
                                    LOG.info("Checkpoint created", checkpoint_id=checkpoint.checkpoint_id)

                                CASE Error(error):
                                    LOG.warning("Checkpoint failed", error=error)
                                    // Continue execution despite checkpoint failure

                CASE Error(error):
                    // Stage failed
                    LOG.error("Stage failed", stage=stage, error=error)
                    execution_state.errors.append(PipelineError {
                        error_id: generate_uuid(),
                        pipeline_id: pipeline_id,
                        stage: stage,
                        error_type: error.error_type,
                        message: error.message,
                        details: error.details,
                        recoverable: error.recoverable,
                        occurred_at: current_timestamp()
                    })

                    // Determine if we should continue or abort
                    IF error.recoverable AND NOT config.fail_fast:
                        // Log warning and continue to next stage
                        warning = PipelineWarning {
                            warning_id: generate_uuid(),
                            pipeline_id: pipeline_id,
                            stage: stage,
                            message: "Stage failed but pipeline continuing: {}".format(error.message),
                            category: "error_recovery",
                            occurred_at: current_timestamp()
                        }
                        execution_state.warnings.append(warning)
                    ELSE:
                        // Fatal error - abort pipeline
                        execution_state.state = PipelineState.Failed

                        // Create final checkpoint for debugging
                        IF config.enable_checkpointing:
                            pipeline_manager.create_checkpoint(execution_state)

                        RETURN build_failed_result(execution_state, error)

        // Step 5: All stages completed successfully
        execution_state.current_stage = PipelineStage.Completed
        execution_state.state = PipelineState.Completed
        execution_state.completed_at = current_timestamp()
        execution_state.overall_progress_percent = 100.0

        LOG.info("Pipeline completed successfully",
                 duration_ms=execution_state.completed_at - execution_state.started_at)

        // Step 6: Build final result
        final_result = build_pipeline_result(execution_state, metrics_collector)
        execution_state.final_result = final_result

        // Step 7: Generate comprehensive report
        report = generate_pipeline_report(final_result)

        // Step 8: Cleanup
        agent_coordinator.shutdown_all_agents()

        RETURN Success(final_result)

    CATCH error AS PipelineError:
        // Unexpected error during pipeline execution
        LOG.error("Pipeline failed with unexpected error", error=error)

        execution_state.state = PipelineState.Failed
        execution_state.errors.append(PipelineError {
            error_id: generate_uuid(),
            pipeline_id: pipeline_id,
            stage: execution_state.current_stage,
            error_type: "UnexpectedError",
            message: error.message,
            details: {"stack_trace": error.stack_trace},
            recoverable: false,
            occurred_at: current_timestamp()
        })

        // Attempt emergency checkpoint
        IF config.enable_checkpointing:
            TRY:
                pipeline_manager.create_checkpoint(execution_state)
            CATCH checkpoint_error:
                LOG.error("Emergency checkpoint failed", error=checkpoint_error)

        // Cleanup
        agent_coordinator.shutdown_all_agents()

        RETURN Error(error)


FUNCTION determine_pipeline_stages(mode: ExecutionMode) -> List<PipelineStage>:
    """
    Determines the sequence of stages for the given execution mode.

    Design Notes:
    - Script Mode: Full pipeline but with single-file optimizations
    - Library Mode: Full pipeline with parallel processing support
    """

    // Common stages for both modes
    stages = [
        PipelineStage.Ingesting,
        PipelineStage.Analyzing,
        PipelineStage.Specifying,
        PipelineStage.Transpiling,
        PipelineStage.Building,
        PipelineStage.Testing,
        PipelineStage.Packaging
    ]

    RETURN stages


FUNCTION should_checkpoint(state: PipelineExecutionState, config: PipelineConfiguration) -> Boolean:
    """
    Determines if a checkpoint should be created at this point.
    """

    IF NOT config.enable_checkpointing:
        RETURN false

    IF state.last_checkpoint_time IS None:
        // First checkpoint after first stage completes
        RETURN state.completed_stages.length > 0

    elapsed_since_last = current_timestamp() - state.last_checkpoint_time

    RETURN elapsed_since_last >= config.checkpoint_interval_seconds * 1000
```

### 3.2 Stage Execution with Agent Coordination

```pseudocode
FUNCTION execute_stage(
    stage: PipelineStage,
    state: PipelineExecutionState,
    coordinator: AgentCoordinator,
    pipeline_manager: PipelineManager,
    progress_reporter: ProgressReporter,
    metrics_collector: MetricsCollector,
    config: PipelineConfiguration
) -> Result<AgentResult, StageError>:
    """
    Executes a single pipeline stage by coordinating the appropriate agent.

    Design Notes:
    - Maps pipeline stage to agent type
    - Handles retry logic with exponential backoff
    - Implements circuit breaker pattern for failing agents
    - Collects metrics and progress updates
    - Supports parallel execution for independent work items (Library Mode)

    TDD Note: Mock AgentCoordinator.execute_agent().
    Test retry logic, circuit breaker, and error handling.
    """

    // Step 1: Map stage to agent type
    agent_type = stage_to_agent_type(stage)

    // Step 2: Check circuit breaker
    circuit_breaker = coordinator.get_circuit_breaker(agent_type)

    IF circuit_breaker.state == CircuitBreakerState.Open:
        LOG.error("Circuit breaker open for agent", agent_type=agent_type)
        RETURN Error(StageError {
            stage: stage,
            error_type: "CircuitBreakerOpen",
            message: "Agent {} is unavailable (circuit breaker open)".format(agent_type),
            recoverable: false
        })

    // Step 3: Prepare agent input from previous stage results
    agent_input = prepare_agent_input(stage, state)

    // Step 4: Create task context
    task_context = TaskContext {
        pipeline_id: state.pipeline_id,
        execution_mode: config.mode,
        trace_id: state.pipeline_id,
        span_id: generate_uuid(),
        configuration: {}
    }

    // Step 5: Execute agent with retry logic
    retry_policy = RetryPolicy {
        max_retries: config.max_retries_per_agent,
        strategy: config.retry_strategy,
        base_delay_ms: 1000,
        max_delay_ms: 30000,
        timeout_multiplier: 1.5,
        retryable_errors: ["TransientError", "TimeoutError", "NetworkError"]
    }

    retry_count = 0
    last_error = None

    WHILE retry_count <= retry_policy.max_retries:
        TRY:
            LOG.info("Executing agent", agent_type=agent_type, attempt=retry_count + 1)

            // Execute agent
            start_time = current_timestamp()

            result = coordinator.execute_agent(
                agent_type,
                agent_input,
                task_context,
                timeout_seconds=calculate_timeout(config, agent_type, retry_count)
            )

            duration = current_timestamp() - start_time

            // Success - record metrics and return
            metrics_collector.record_agent_success(agent_type, duration)
            circuit_breaker.record_success()

            LOG.info("Agent completed successfully",
                     agent_type=agent_type,
                     duration_ms=duration,
                     retry_count=retry_count)

            RETURN Success(result)

        CATCH error AS AgentError:
            last_error = error
            retry_count += 1

            LOG.warning("Agent execution failed",
                       agent_type=agent_type,
                       attempt=retry_count,
                       error=error.message)

            // Record failure in circuit breaker
            circuit_breaker.record_failure()

            // Check if error is retryable
            IF error.error_type NOT IN retry_policy.retryable_errors:
                LOG.error("Non-retryable error encountered", error_type=error.error_type)
                BREAK

            // Check if we have retries left
            IF retry_count > retry_policy.max_retries:
                LOG.error("Max retries exceeded", max_retries=retry_policy.max_retries)
                BREAK

            // Calculate backoff delay
            delay_ms = calculate_backoff_delay(retry_policy, retry_count)

            LOG.info("Retrying after delay", delay_ms=delay_ms)
            sleep(delay_ms)

    // All retries exhausted
    metrics_collector.record_agent_failure(agent_type, last_error)

    RETURN Error(StageError {
        stage: stage,
        error_type: last_error.error_type,
        message: "Agent {} failed after {} retries: {}".format(
            agent_type,
            retry_count,
            last_error.message
        ),
        details: last_error.details,
        recoverable: last_error.recoverable
    })


FUNCTION stage_to_agent_type(stage: PipelineStage) -> String:
    """
    Maps a pipeline stage to the corresponding agent type.
    """

    MATCH stage:
        PipelineStage.Ingesting -> "ingest"
        PipelineStage.Analyzing -> "analysis"
        PipelineStage.Specifying -> "specification"
        PipelineStage.Transpiling -> "transpiler"
        PipelineStage.Building -> "build"
        PipelineStage.Testing -> "test"
        PipelineStage.Packaging -> "packaging"
        _ -> ERROR("Unknown stage: {}".format(stage))


FUNCTION prepare_agent_input(stage: PipelineStage, state: PipelineExecutionState) -> AgentInput:
    """
    Prepares input for an agent based on previous stage results.

    Design Notes:
    - For first stage (Ingest), input is from configuration
    - For subsequent stages, input is from previous stage's output
    - Handles data transformation between stages
    """

    MATCH stage:
        PipelineStage.Ingesting:
            // First stage - input from configuration
            RETURN AgentInput {
                input_type: "ingest",
                data: {
                    "source_path": state.metadata.configuration.source_path,
                    "mode": state.metadata.configuration.mode,
                    "output_directory": state.metadata.configuration.output_directory
                }
            }

        PipelineStage.Analyzing:
            // Input from Ingest stage
            ingest_result = state.stage_results[PipelineStage.Ingesting]
            RETURN AgentInput {
                input_type: "analysis",
                data: {
                    "ingest_result": ingest_result.data,
                    "modules": ingest_result.data["modules"],
                    "dependency_graph": ingest_result.data["dependency_graph"]
                }
            }

        PipelineStage.Specifying:
            // Input from Analysis stage
            analysis_result = state.stage_results[PipelineStage.Analyzing]
            RETURN AgentInput {
                input_type: "specification",
                data: {
                    "api_spec": analysis_result.data["api_spec"],
                    "dependency_graph": analysis_result.data["dependency_graph"],
                    "contracts": analysis_result.data["contracts"]
                }
            }

        PipelineStage.Transpiling:
            // Input from Specification stage
            spec_result = state.stage_results[PipelineStage.Specifying]
            analysis_result = state.stage_results[PipelineStage.Analyzing]
            RETURN AgentInput {
                input_type: "transpiler",
                data: {
                    "rust_spec": spec_result.data["rust_specification"],
                    "python_modules": analysis_result.data["modules"],
                    "api_spec": analysis_result.data["api_spec"]
                }
            }

        PipelineStage.Building:
            // Input from Transpiler stage
            transpiler_result = state.stage_results[PipelineStage.Transpiling]
            RETURN AgentInput {
                input_type: "build",
                data: {
                    "rust_workspace": transpiler_result.data["rust_workspace"],
                    "crate_structure": transpiler_result.data["crate_structure"]
                }
            }

        PipelineStage.Testing:
            // Input from Build stage and earlier stages
            build_result = state.stage_results[PipelineStage.Building]
            analysis_result = state.stage_results[PipelineStage.Analyzing]
            RETURN AgentInput {
                input_type: "test",
                data: {
                    "wasm_binaries": build_result.data["wasm_binaries"],
                    "python_tests": analysis_result.data["tests"],
                    "api_spec": analysis_result.data["api_spec"]
                }
            }

        PipelineStage.Packaging:
            // Input from Test stage and Build stage
            test_result = state.stage_results[PipelineStage.Testing]
            build_result = state.stage_results[PipelineStage.Building]
            RETURN AgentInput {
                input_type: "packaging",
                data: {
                    "wasm_binaries": build_result.data["wasm_binaries"],
                    "test_results": test_result.data["conformance_report"],
                    "metadata": state.metadata
                }
            }


FUNCTION calculate_timeout(config: PipelineConfiguration, agent_type: String, retry_count: Integer) -> Integer:
    """
    Calculates timeout for an agent execution, increasing on retries.
    """

    base_timeout = config.agent_timeout_seconds.get(agent_type, 300)  // Default 5 minutes

    // Increase timeout on retries
    multiplier = config.retry_strategy match {
        RetryStrategy.ExponentialBackoff -> 1.5 ** retry_count
        RetryStrategy.Fixed -> 1.0
        _ -> 1.0
    }

    RETURN Integer(base_timeout * multiplier)


FUNCTION calculate_backoff_delay(policy: RetryPolicy, retry_count: Integer) -> Integer:
    """
    Calculates delay before next retry attempt.
    """

    MATCH policy.strategy:
        RetryStrategy.NoRetry:
            RETURN 0

        RetryStrategy.Immediate:
            RETURN 0

        RetryStrategy.Fixed:
            RETURN policy.base_delay_ms

        RetryStrategy.ExponentialBackoff:
            // Exponential backoff: base_delay * 2^(retry_count - 1)
            delay = policy.base_delay_ms * (2 ** (retry_count - 1))

            // Cap at max_delay_ms
            RETURN min(delay, policy.max_delay_ms)
```

### 3.3 Agent Coordinator - Agent Lifecycle Management

```pseudocode
CLASS AgentCoordinator:
    """
    Coordinates agent lifecycle, execution, and supervision.

    Responsibilities:
    - Spawn and register agents
    - Route messages to agents
    - Monitor agent health
    - Implement circuit breaker pattern
    - Handle agent failures and restarts
    """

    // State
    registered_agents: Map<String, AgentRegistration>  // agent_id -> registration
    circuit_breakers: Map<String, CircuitBreaker>      // agent_type -> circuit breaker
    agent_pool: Map<String, List<String>>              // agent_type -> [agent_ids]
    health_monitor: HealthMonitor
    message_router: MessageRouter


    FUNCTION new(config: PipelineConfiguration) -> AgentCoordinator:
        """
        Initializes the Agent Coordinator.
        """

        self.registered_agents = {}
        self.circuit_breakers = {}
        self.agent_pool = {}

        // Initialize circuit breakers for each agent type
        agent_types = ["ingest", "analysis", "specification", "transpiler",
                      "build", "test", "packaging"]

        FOR agent_type IN agent_types:
            self.circuit_breakers[agent_type] = CircuitBreaker {
                name: agent_type,
                state: CircuitBreakerState.Closed,
                failure_threshold: config.circuit_breaker_threshold,
                success_threshold: 2,
                timeout_ms: 60000,  // 1 minute
                consecutive_failures: 0,
                consecutive_successes: 0,
                last_state_change: current_timestamp(),
                failure_count_window: [],
                total_requests: 0,
                total_failures: 0,
                total_successes: 0,
                total_rejected: 0
            }

            self.agent_pool[agent_type] = []

        // Start health monitor
        self.health_monitor = HealthMonitor.new()
        self.health_monitor.start()

        // Initialize message router
        self.message_router = MessageRouter.new()

        RETURN self


    FUNCTION spawn_agent(agent_type: String, config: Map<String, Any>) -> Result<String, Error>:
        """
        Spawns a new agent instance.

        Design Notes:
        - Agents are spawned as separate processes or containers
        - Registers agent with health monitor
        - Adds to agent pool

        TDD Note: Mock process spawning and agent initialization.
        """

        LOG.info("Spawning agent", agent_type=agent_type)

        TRY:
            // Generate unique agent ID
            agent_id = generate_uuid()

            // Spawn agent process (implementation-specific)
            endpoint = spawn_agent_process(agent_type, agent_id, config)

            // Wait for agent to be ready
            ready = wait_for_agent_ready(endpoint, timeout_seconds=30)

            IF NOT ready:
                RETURN Error("Agent failed to become ready within timeout")

            // Register agent
            registration = AgentRegistration {
                agent_id: agent_id,
                agent_type: agent_type,
                agent_name: "{}_{}".format(agent_type, agent_id[:8]),
                endpoint: endpoint,
                protocol: "http",
                capabilities: get_agent_capabilities(agent_type),
                required_resources: get_agent_resource_requirements(agent_type),
                status: AgentStatus.Idle,
                last_heartbeat: current_timestamp(),
                consecutive_failures: 0,
                total_tasks_processed: 0,
                total_failures: 0,
                average_processing_time_ms: 0.0
            }

            self.registered_agents[agent_id] = registration
            self.agent_pool[agent_type].append(agent_id)

            // Register with health monitor
            self.health_monitor.register(agent_id, endpoint)

            LOG.info("Agent spawned successfully",
                     agent_id=agent_id,
                     agent_type=agent_type,
                     endpoint=endpoint)

            RETURN Success(agent_id)

        CATCH error:
            LOG.error("Failed to spawn agent", agent_type=agent_type, error=error)
            RETURN Error(error)


    FUNCTION execute_agent(
        agent_type: String,
        input: AgentInput,
        context: TaskContext,
        timeout_seconds: Integer
    ) -> Result<AgentResult, AgentError>:
        """
        Executes an agent with the given input.

        Design Notes:
        - Selects available agent from pool (or spawns new one)
        - Sends task to agent via message router
        - Waits for result with timeout
        - Updates agent statistics

        TDD Note: Mock agent selection and message routing.
        """

        // Check circuit breaker
        circuit_breaker = self.circuit_breakers[agent_type]

        IF circuit_breaker.state == CircuitBreakerState.Open:
            // Check if timeout has passed to transition to half-open
            elapsed = current_timestamp() - circuit_breaker.last_state_change

            IF elapsed >= circuit_breaker.timeout_ms:
                // Transition to half-open
                circuit_breaker.state = CircuitBreakerState.HalfOpen
                circuit_breaker.last_state_change = current_timestamp()
                circuit_breaker.consecutive_successes = 0
                LOG.info("Circuit breaker half-open", agent_type=agent_type)
            ELSE:
                // Still open - reject request
                circuit_breaker.total_rejected += 1
                RETURN Error(AgentError {
                    error_type: "CircuitBreakerOpen",
                    message: "Agent {} unavailable (circuit breaker open)".format(agent_type),
                    recoverable: true
                })

        // Get or spawn agent
        agent_id_result = get_available_agent(agent_type)

        MATCH agent_id_result:
            CASE Error(error):
                RETURN Error(error)

            CASE Success(agent_id):
                agent = self.registered_agents[agent_id]

                // Update agent status
                agent.status = AgentStatus.Running

                // Create task
                task = AgentTask {
                    task_id: generate_uuid(),
                    agent_type: agent_type,
                    pipeline_id: context.pipeline_id,
                    input_data: input,
                    context: context,
                    priority: 5,
                    dependencies: [],
                    timeout_seconds: timeout_seconds,
                    status: AgentStatus.Running,
                    assigned_agent_id: agent_id,
                    submitted_at: current_timestamp()
                }

                LOG.info("Executing task on agent",
                         task_id=task.task_id,
                         agent_id=agent_id,
                         agent_type=agent_type)

                TRY:
                    // Send task to agent
                    start_time = current_timestamp()

                    result = self.message_router.send_and_wait(
                        agent.endpoint,
                        task,
                        timeout_seconds
                    )

                    duration = current_timestamp() - start_time

                    // Update agent statistics
                    agent.status = AgentStatus.Completed
                    agent.total_tasks_processed += 1
                    agent.average_processing_time_ms = (
                        (agent.average_processing_time_ms * (agent.total_tasks_processed - 1) + duration) /
                        agent.total_tasks_processed
                    )
                    agent.last_heartbeat = current_timestamp()

                    RETURN Success(result)

                CATCH error AS AgentError:
                    // Task failed
                    agent.status = AgentStatus.Failed
                    agent.total_failures += 1
                    agent.consecutive_failures += 1

                    LOG.error("Agent task failed",
                             task_id=task.task_id,
                             agent_id=agent_id,
                             error=error)

                    // Check if agent should be removed from pool
                    IF agent.consecutive_failures >= 3:
                        LOG.warning("Agent has too many consecutive failures, removing from pool",
                                   agent_id=agent_id)
                        self.remove_agent(agent_id)

                    RETURN Error(error)


    FUNCTION get_available_agent(agent_type: String) -> Result<String, Error>:
        """
        Gets an available agent from the pool or spawns a new one.
        """

        // Look for idle agent in pool
        FOR agent_id IN self.agent_pool[agent_type]:
            agent = self.registered_agents[agent_id]

            IF agent.status == AgentStatus.Idle:
                RETURN Success(agent_id)

        // No idle agents - spawn new one
        LOG.info("No idle agents available, spawning new agent", agent_type=agent_type)

        spawn_result = self.spawn_agent(agent_type, {})

        RETURN spawn_result


    FUNCTION get_circuit_breaker(agent_type: String) -> CircuitBreaker:
        """
        Gets the circuit breaker for an agent type.
        """
        RETURN self.circuit_breakers[agent_type]


    FUNCTION shutdown_all_agents():
        """
        Gracefully shuts down all agents.
        """

        LOG.info("Shutting down all agents", count=self.registered_agents.length)

        FOR agent_id, agent IN self.registered_agents.items():
            TRY:
                LOG.info("Shutting down agent", agent_id=agent_id)

                // Send shutdown command
                self.message_router.send_shutdown(agent.endpoint)

                // Wait for graceful shutdown (with timeout)
                wait_for_agent_shutdown(agent.endpoint, timeout_seconds=10)

            CATCH error:
                LOG.warning("Failed to gracefully shutdown agent",
                           agent_id=agent_id,
                           error=error)

                // Force kill if graceful shutdown fails
                force_kill_agent(agent_id)

        // Stop health monitor
        self.health_monitor.stop()

        LOG.info("All agents shut down")


    FUNCTION remove_agent(agent_id: String):
        """
        Removes a failed agent from the pool.
        """

        agent = self.registered_agents[agent_id]

        // Remove from pool
        self.agent_pool[agent.agent_type].remove(agent_id)

        // Deregister from health monitor
        self.health_monitor.deregister(agent_id)

        // Remove from registered agents
        self.registered_agents.remove(agent_id)

        LOG.info("Agent removed", agent_id=agent_id, agent_type=agent.agent_type)


CLASS CircuitBreaker:
    """
    Implements circuit breaker pattern for agent resilience.
    """

    FUNCTION record_success():
        """
        Records a successful operation.
        """

        self.total_requests += 1
        self.total_successes += 1
        self.consecutive_failures = 0

        IF self.state == CircuitBreakerState.HalfOpen:
            self.consecutive_successes += 1

            IF self.consecutive_successes >= self.success_threshold:
                // Close circuit
                self.state = CircuitBreakerState.Closed
                self.last_state_change = current_timestamp()
                LOG.info("Circuit breaker closed", name=self.name)


    FUNCTION record_failure():
        """
        Records a failed operation.
        """

        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0

        // Add to failure window
        self.failure_count_window.append(current_timestamp())

        // Remove old failures (older than 1 minute)
        cutoff_time = current_timestamp() - 60000
        self.failure_count_window = [
            t FOR t IN self.failure_count_window IF t > cutoff_time
        ]

        IF self.state == CircuitBreakerState.Closed:
            // Check if we should open
            IF self.consecutive_failures >= self.failure_threshold:
                // Open circuit
                self.state = CircuitBreakerState.Open
                self.last_state_change = current_timestamp()
                LOG.warning("Circuit breaker opened",
                           name=self.name,
                           failures=self.consecutive_failures)

        ELSE IF self.state == CircuitBreakerState.HalfOpen:
            // Failure in half-open - reopen circuit
            self.state = CircuitBreakerState.Open
            self.last_state_change = current_timestamp()
            LOG.warning("Circuit breaker reopened", name=self.name)
```

### 3.4 Pipeline Manager - Checkpointing and Resumption

```pseudocode
CLASS PipelineManager:
    """
    Manages pipeline state, checkpointing, and resumption.

    Responsibilities:
    - Maintain pipeline state
    - Create checkpoints at regular intervals
    - Restore pipeline from checkpoints
    - Manage checkpoint storage
    - Provide pipeline history
    """

    state: PipelineExecutionState
    config: PipelineConfiguration
    storage: CheckpointStorage


    FUNCTION new(state: PipelineExecutionState, config: PipelineConfiguration) -> PipelineManager:
        """
        Initializes the Pipeline Manager.
        """

        self.state = state
        self.config = config
        self.storage = CheckpointStorage.new(config.checkpoint_storage_path)

        RETURN self


    FUNCTION create_checkpoint(state: PipelineExecutionState) -> Result<Checkpoint, Error>:
        """
        Creates a checkpoint of the current pipeline state.

        Design Notes:
        - Serializes pipeline state
        - Captures artifact locations (not contents - too large)
        - Stores to configured storage backend
        - Returns checkpoint metadata

        TDD Note: Mock storage backend.
        """

        LOG.info("Creating checkpoint", pipeline_id=state.pipeline_id)

        TRY:
            checkpoint_id = generate_uuid()
            sequence_number = state.checkpoint_count + 1

            // Collect artifacts manifest
            artifacts_manifest = collect_artifacts_from_state(state)

            // Determine if checkpoint is resumable
            can_resume = determine_resumability(state)
            resume_from_stage = determine_resume_stage(state)

            // Create checkpoint object
            checkpoint = Checkpoint {
                checkpoint_id: checkpoint_id,
                pipeline_id: state.pipeline_id,
                sequence_number: sequence_number,
                execution_state: state,
                agent_states: {},  // Would capture agent-specific state if needed
                artifacts_manifest: artifacts_manifest,
                intermediate_results: state.stage_results,
                created_at: current_timestamp(),
                storage_path: Path(""),  // Will be set by storage
                size_bytes: 0,  // Will be calculated
                compressed: true,
                can_resume: can_resume,
                resume_from_stage: resume_from_stage,
                required_artifacts: [a.artifact_id FOR a IN artifacts_manifest]
            }

            // Serialize checkpoint
            serialized = serialize_checkpoint(checkpoint)

            // Compress
            compressed = compress_data(serialized)
            checkpoint.size_bytes = compressed.length

            // Store checkpoint
            storage_path = self.storage.save_checkpoint(
                checkpoint.checkpoint_id,
                compressed
            )
            checkpoint.storage_path = storage_path

            LOG.info("Checkpoint created",
                     checkpoint_id=checkpoint_id,
                     sequence=sequence_number,
                     size_bytes=checkpoint.size_bytes,
                     path=storage_path)

            RETURN Success(checkpoint)

        CATCH error:
            LOG.error("Failed to create checkpoint", error=error)
            RETURN Error(error)


    FUNCTION find_latest_checkpoint(source_path: Path) -> Optional<Checkpoint>:
        """
        Finds the latest resumable checkpoint for a given source path.

        Design Notes:
        - Searches checkpoint storage for checkpoints matching source path
        - Returns most recent resumable checkpoint
        - Validates checkpoint integrity
        """

        LOG.info("Searching for resumable checkpoint", source_path=source_path)

        TRY:
            // Query storage for checkpoints
            checkpoints = self.storage.list_checkpoints_for_source(source_path)

            IF checkpoints.is_empty():
                LOG.info("No checkpoints found for source", source_path=source_path)
                RETURN None

            // Sort by created_at descending
            checkpoints.sort_by(|c| c.created_at, descending=true)

            // Find first resumable checkpoint
            FOR checkpoint IN checkpoints:
                IF checkpoint.can_resume:
                    // Validate checkpoint integrity
                    IF validate_checkpoint(checkpoint):
                        LOG.info("Found resumable checkpoint",
                                checkpoint_id=checkpoint.checkpoint_id,
                                created_at=checkpoint.created_at,
                                resume_from=checkpoint.resume_from_stage)
                        RETURN checkpoint
                    ELSE:
                        LOG.warning("Checkpoint failed validation",
                                   checkpoint_id=checkpoint.checkpoint_id)

            LOG.info("No valid resumable checkpoints found")
            RETURN None

        CATCH error:
            LOG.error("Error searching for checkpoints", error=error)
            RETURN None


    FUNCTION restore_from_checkpoint(checkpoint: Checkpoint) -> PipelineExecutionState:
        """
        Restores pipeline state from a checkpoint.

        Design Notes:
        - Loads checkpoint data from storage
        - Deserializes pipeline state
        - Validates artifacts are still available
        - Updates state for resumption
        """

        LOG.info("Restoring from checkpoint", checkpoint_id=checkpoint.checkpoint_id)

        TRY:
            // Load checkpoint data
            compressed_data = self.storage.load_checkpoint(checkpoint.checkpoint_id)

            // Decompress
            serialized = decompress_data(compressed_data)

            // Deserialize
            restored_checkpoint = deserialize_checkpoint(serialized)

            // Validate artifacts are still available
            missing_artifacts = []
            FOR artifact IN restored_checkpoint.artifacts_manifest:
                IF NOT file_exists(artifact.path):
                    missing_artifacts.append(artifact.artifact_id)

            IF NOT missing_artifacts.is_empty():
                LOG.warning("Some artifacts missing from checkpoint",
                           missing=missing_artifacts.length)
                // Could fail here or continue with partial state

            // Get execution state
            restored_state = restored_checkpoint.execution_state

            // Update state for resumption
            restored_state.state = PipelineState.Running
            restored_state.metadata.parent_pipeline_id = restored_state.pipeline_id
            restored_state.pipeline_id = generate_uuid()  // New pipeline ID for resumed execution
            restored_state.metadata.updated_at = current_timestamp()

            // Update pending stages (skip completed ones)
            restored_state.pending_stages = [
                stage FOR stage IN restored_state.pending_stages
                IF stage NOT IN restored_state.completed_stages
            ]

            LOG.info("Pipeline state restored",
                     new_pipeline_id=restored_state.pipeline_id,
                     resume_from=checkpoint.resume_from_stage,
                     pending_stages=restored_state.pending_stages.length)

            RETURN restored_state

        CATCH error:
            LOG.error("Failed to restore from checkpoint", error=error)
            RAISE error


    FUNCTION determine_resumability(state: PipelineExecutionState) -> Boolean:
        """
        Determines if a checkpoint is resumable.

        A checkpoint is resumable if:
        - At least one stage has completed successfully
        - No unrecoverable errors have occurred
        - All artifacts from completed stages are available
        """

        IF state.completed_stages.is_empty():
            RETURN false

        IF state.state == PipelineState.Failed:
            // Check if failure was recoverable
            unrecoverable_errors = [e FOR e IN state.errors IF NOT e.recoverable]
            IF NOT unrecoverable_errors.is_empty():
                RETURN false

        RETURN true


    FUNCTION determine_resume_stage(state: PipelineExecutionState) -> PipelineStage:
        """
        Determines which stage to resume from.

        - If current stage failed, resume from current stage
        - If current stage completed, resume from next stage
        """

        IF state.current_stage IN state.completed_stages:
            // Current stage completed - resume from next
            all_stages = determine_pipeline_stages(state.metadata.configuration.mode)
            current_index = all_stages.index_of(state.current_stage)

            IF current_index + 1 < all_stages.length:
                RETURN all_stages[current_index + 1]
            ELSE:
                RETURN PipelineStage.Completed
        ELSE:
            // Current stage not completed - resume from current
            RETURN state.current_stage


FUNCTION validate_checkpoint(checkpoint: Checkpoint) -> Boolean:
    """
    Validates checkpoint integrity.
    """

    // Check if checkpoint file exists
    IF NOT file_exists(checkpoint.storage_path):
        LOG.warning("Checkpoint file missing", path=checkpoint.storage_path)
        RETURN false

    // Could add checksum validation here

    RETURN true


FUNCTION collect_artifacts_from_state(state: PipelineExecutionState) -> List<Artifact>:
    """
    Collects all artifacts from stage results.
    """

    artifacts = []

    FOR stage, result IN state.stage_results.items():
        artifacts.extend(result.artifacts)

    RETURN artifacts


FUNCTION serialize_checkpoint(checkpoint: Checkpoint) -> Bytes:
    """
    Serializes checkpoint to bytes (JSON or MessagePack).
    """

    // Convert to JSON
    json_str = to_json(checkpoint)

    RETURN json_str.encode("utf-8")


FUNCTION deserialize_checkpoint(data: Bytes) -> Checkpoint:
    """
    Deserializes checkpoint from bytes.
    """

    json_str = data.decode("utf-8")

    RETURN from_json(json_str, Checkpoint)


FUNCTION compress_data(data: Bytes) -> Bytes:
    """
    Compresses data using gzip or similar.
    """

    RETURN gzip_compress(data)


FUNCTION decompress_data(data: Bytes) -> Bytes:
    """
    Decompresses data.
    """

    RETURN gzip_decompress(data)
```

### 3.5 Progress Reporting and ETA Estimation

```pseudocode
CLASS ProgressReporter:
    """
    Provides real-time progress updates and ETA estimation.

    Responsibilities:
    - Track progress for each pipeline stage
    - Calculate overall pipeline progress
    - Estimate time to completion
    - Emit progress events for UI consumption
    """

    pipeline_id: String
    stage_trackers: Map<PipelineStage, ProgressTracker>
    progress_listeners: List<ProgressListener>
    historical_durations: Map<PipelineStage, List<Integer>>  // For ETA calculation


    FUNCTION new(pipeline_id: String) -> ProgressReporter:
        """
        Initializes the Progress Reporter.
        """

        self.pipeline_id = pipeline_id
        self.stage_trackers = {}
        self.progress_listeners = []
        self.historical_durations = load_historical_durations()

        RETURN self


    FUNCTION update_stage(stage: PipelineStage, progress_percent: Float):
        """
        Updates progress for a specific stage.

        Design Notes:
        - Updates stage tracker
        - Recalculates overall progress
        - Recalculates ETA
        - Emits progress event
        """

        // Get or create tracker for this stage
        IF stage NOT IN self.stage_trackers:
            self.stage_trackers[stage] = ProgressTracker {
                stage: stage,
                total_work_units: 100,  // Use percentage as work units
                completed_work_units: 0,
                failed_work_units: 0,
                started_at: current_timestamp(),
                last_update_at: current_timestamp(),
                recent_completion_times: [],
                average_work_unit_time_ms: 0.0,
                estimated_remaining_work_units: 100,
                estimated_completion_time: None
            }

        tracker = self.stage_trackers[stage]

        // Update tracker
        tracker.completed_work_units = progress_percent
        tracker.last_update_at = current_timestamp()

        // Calculate average work unit time
        elapsed = current_timestamp() - tracker.started_at
        IF tracker.completed_work_units > 0:
            tracker.average_work_unit_time_ms = elapsed / tracker.completed_work_units

        // Estimate remaining time for this stage
        remaining_work = tracker.total_work_units - tracker.completed_work_units
        IF tracker.average_work_unit_time_ms > 0:
            estimated_remaining_ms = remaining_work * tracker.average_work_unit_time_ms
            tracker.estimated_completion_time = current_timestamp() + estimated_remaining_ms

        // Calculate overall progress
        overall_progress = self.calculate_overall_progress()

        // Estimate overall completion time
        overall_eta = self.estimate_overall_completion_time()

        // Create progress update event
        update = ProgressUpdate {
            pipeline_id: self.pipeline_id,
            timestamp: current_timestamp(),
            current_stage: stage,
            stage_name: stage_to_human_readable(stage),
            overall_progress_percent: overall_progress,
            stage_progress_percent: progress_percent,
            elapsed_time_ms: elapsed,
            estimated_remaining_ms: overall_eta,
            estimated_completion_time: IF overall_eta IS NOT None:
                current_timestamp() + overall_eta
            ELSE:
                None,
            current_activity: "",  // Would be filled by specific stage
            items_processed: Integer(tracker.completed_work_units),
            total_items: Integer(tracker.total_work_units),
            throughput: None,
            gpu_utilization_percent: None,  // Would be filled from metrics
            memory_usage_mb: 0
        }

        // Emit event to listeners
        self.emit_progress_update(update)


    FUNCTION calculate_overall_progress() -> Float:
        """
        Calculates overall pipeline progress.

        Design Notes:
        - Weights each stage equally (could be weighted by expected duration)
        - Returns percentage from 0.0 to 100.0
        """

        all_stages = [
            PipelineStage.Ingesting,
            PipelineStage.Analyzing,
            PipelineStage.Specifying,
            PipelineStage.Transpiling,
            PipelineStage.Building,
            PipelineStage.Testing,
            PipelineStage.Packaging
        ]

        total_stages = all_stages.length
        completed_progress = 0.0

        FOR stage IN all_stages:
            IF stage IN self.stage_trackers:
                tracker = self.stage_trackers[stage]
                stage_progress = tracker.completed_work_units / tracker.total_work_units
                completed_progress += stage_progress
            // If stage not started, contributes 0

        overall = (completed_progress / total_stages) * 100.0

        RETURN min(overall, 100.0)


    FUNCTION estimate_overall_completion_time() -> Optional<Integer>:
        """
        Estimates time remaining for entire pipeline.

        Design Notes:
        - Uses historical data for stages not yet started
        - Uses current progress for in-progress stage
        - Returns None if not enough data
        """

        all_stages = [
            PipelineStage.Ingesting,
            PipelineStage.Analyzing,
            PipelineStage.Specifying,
            PipelineStage.Transpiling,
            PipelineStage.Building,
            PipelineStage.Testing,
            PipelineStage.Packaging
        ]

        total_estimated_remaining_ms = 0

        FOR stage IN all_stages:
            IF stage IN self.stage_trackers:
                tracker = self.stage_trackers[stage]

                // Stage in progress or completed
                IF tracker.completed_work_units < tracker.total_work_units:
                    // In progress - use current estimate
                    remaining = tracker.total_work_units - tracker.completed_work_units
                    estimated_ms = remaining * tracker.average_work_unit_time_ms
                    total_estimated_remaining_ms += estimated_ms
                // If completed, contributes 0
            ELSE:
                // Stage not started - use historical average
                IF stage IN self.historical_durations:
                    durations = self.historical_durations[stage]
                    IF NOT durations.is_empty():
                        avg_duration = sum(durations) / durations.length
                        total_estimated_remaining_ms += avg_duration
                    ELSE:
                        // No historical data - can't estimate
                        RETURN None
                ELSE:
                    // No historical data
                    RETURN None

        RETURN Integer(total_estimated_remaining_ms)


    FUNCTION emit_progress_update(update: ProgressUpdate):
        """
        Emits progress update to all registered listeners.
        """

        FOR listener IN self.progress_listeners:
            TRY:
                listener.on_progress_update(update)
            CATCH error:
                LOG.warning("Progress listener failed", error=error)


    FUNCTION add_listener(listener: ProgressListener):
        """
        Adds a progress listener.
        """
        self.progress_listeners.append(listener)


    FUNCTION record_stage_completion(stage: PipelineStage, duration_ms: Integer):
        """
        Records completion of a stage for future ETA estimation.
        """

        IF stage NOT IN self.historical_durations:
            self.historical_durations[stage] = []

        self.historical_durations[stage].append(duration_ms)

        // Keep only last 100 durations
        IF self.historical_durations[stage].length > 100:
            self.historical_durations[stage] = self.historical_durations[stage][-100:]

        // Persist to storage
        save_historical_durations(self.historical_durations)


FUNCTION load_historical_durations() -> Map<PipelineStage, List<Integer>>:
    """
    Loads historical stage durations from storage.
    """

    // Would load from file or database
    // For now, return empty map
    RETURN {}


FUNCTION save_historical_durations(durations: Map<PipelineStage, List<Integer>>):
    """
    Saves historical stage durations to storage.
    """

    // Would save to file or database
    PASS


FUNCTION stage_to_human_readable(stage: PipelineStage) -> String:
    """
    Converts stage enum to human-readable string.
    """

    MATCH stage:
        PipelineStage.Initializing -> "Initializing"
        PipelineStage.Ingesting -> "Ingesting Python Code"
        PipelineStage.Analyzing -> "Analyzing Code Structure"
        PipelineStage.Specifying -> "Generating Rust Specifications"
        PipelineStage.Transpiling -> "Transpiling to Rust"
        PipelineStage.Building -> "Building WASM Binaries"
        PipelineStage.Testing -> "Running Conformance Tests"
        PipelineStage.Packaging -> "Packaging NIM Services"
        PipelineStage.Completed -> "Completed"
        PipelineStage.Failed -> "Failed"
        PipelineStage.Cancelled -> "Cancelled"


INTERFACE ProgressListener:
    """
    Interface for progress update listeners.
    """

    FUNCTION on_progress_update(update: ProgressUpdate)
```

### 3.6 Result Building and Report Generation

```pseudocode
FUNCTION build_pipeline_result(
    state: PipelineExecutionState,
    metrics: MetricsCollector
) -> PipelineResult:
    """
    Builds the final pipeline result from execution state.

    Design Notes:
    - Aggregates all stage results
    - Collects artifacts
    - Calculates quality metrics
    - Includes resource usage statistics
    """

    // Calculate total duration
    total_duration_ms = state.completed_at - state.started_at

    // Collect all artifacts
    all_artifacts = []
    FOR stage, result IN state.stage_results.items():
        all_artifacts.extend(result.artifacts)

    // Extract specific artifact paths
    rust_workspace = find_artifact_path(all_artifacts, "rust_workspace")
    wasm_binaries = find_artifact_paths(all_artifacts, "wasm_binary")
    nim_containers = find_artifact_paths(all_artifacts, "nim_container")
    reports = find_artifact_paths(all_artifacts, "report")

    // Calculate quality metrics
    quality_metrics = calculate_quality_metrics(state)

    // Get resource usage from metrics collector
    resource_usage = metrics.get_resource_usage()

    // Build result
    result = PipelineResult {
        pipeline_id: state.pipeline_id,
        success: state.state == PipelineState.Completed,
        mode: state.metadata.configuration.mode,
        started_at: state.started_at,
        completed_at: state.completed_at,
        total_duration_ms: total_duration_ms,
        stage_durations: state.stage_durations,
        artifacts: all_artifacts,
        rust_workspace_path: rust_workspace,
        wasm_binaries: wasm_binaries,
        nim_containers: nim_containers,
        reports: reports,
        translation_coverage: quality_metrics.translation_coverage,
        test_pass_rate: quality_metrics.test_pass_rate,
        conformance_score: quality_metrics.conformance_score,
        performance_improvement: quality_metrics.performance_improvement,
        peak_memory_mb: resource_usage.peak_memory_mb,
        total_cpu_time_ms: resource_usage.total_cpu_time_ms,
        total_gpu_time_ms: resource_usage.total_gpu_time_ms,
        gpu_utilization_avg: resource_usage.gpu_utilization_avg,
        errors: state.errors,
        warnings: state.warnings,
        metadata: state.metadata,
        stage_results: state.stage_results
    }

    RETURN result


FUNCTION build_failed_result(
    state: PipelineExecutionState,
    error: StageError
) -> PipelineResult:
    """
    Builds a result for a failed pipeline.
    """

    // Similar to build_pipeline_result but marks as failed
    // and includes partial results

    result = PipelineResult {
        pipeline_id: state.pipeline_id,
        success: false,
        mode: state.metadata.configuration.mode,
        started_at: state.started_at,
        completed_at: current_timestamp(),
        total_duration_ms: current_timestamp() - state.started_at,
        stage_durations: state.stage_durations,
        artifacts: [],  // Collect partial artifacts
        errors: state.errors,
        warnings: state.warnings,
        metadata: state.metadata,
        stage_results: state.stage_results
    }

    RETURN result


FUNCTION calculate_quality_metrics(state: PipelineExecutionState) -> QualityMetrics:
    """
    Calculates quality metrics from stage results.
    """

    metrics = QualityMetrics {
        translation_coverage: 0.0,
        test_pass_rate: 0.0,
        conformance_score: 0.0,
        performance_improvement: 0.0
    }

    // Extract from Test Agent results
    IF PipelineStage.Testing IN state.stage_results:
        test_result = state.stage_results[PipelineStage.Testing]

        IF "conformance_report" IN test_result.data:
            report = test_result.data["conformance_report"]
            metrics.test_pass_rate = report.get("pass_rate", 0.0)
            metrics.conformance_score = report.get("conformance_score", 0.0)

        IF "performance_comparison" IN test_result.data:
            perf = test_result.data["performance_comparison"]
            metrics.performance_improvement = perf.get("speedup_factor", 0.0)

    // Extract from Transpiler Agent results
    IF PipelineStage.Transpiling IN state.stage_results:
        transpile_result = state.stage_results[PipelineStage.Transpiling]

        IF "translation_coverage" IN transpile_result.data:
            metrics.translation_coverage = transpile_result.data["translation_coverage"]

    RETURN metrics


FUNCTION generate_pipeline_report(result: PipelineResult) -> PipelineReport:
    """
    Generates comprehensive pipeline report.

    Design Notes:
    - Creates formatted sections for user consumption
    - Generates visualization data
    - Supports multiple export formats (JSON, YAML, HTML, Markdown)
    """

    // Executive Summary
    executive_summary = generate_executive_summary(result)

    // Stage Summaries
    stage_summaries = {}
    FOR stage, stage_result IN result.stage_results.items():
        stage_summaries[stage] = generate_stage_summary(stage, stage_result)

    // Metrics Section
    metrics_section = generate_metrics_section(result)

    // Errors Section
    errors_section = generate_errors_section(result.errors)

    // Warnings Section
    warnings_section = generate_warnings_section(result.warnings)

    // Recommendations
    recommendations = generate_recommendations(result)

    // Timeline Data
    timeline = generate_timeline_visualization(result)

    // Metrics Charts
    charts = generate_metrics_charts(result)

    report = PipelineReport {
        result: result,
        executive_summary: executive_summary,
        stage_summaries: stage_summaries,
        metrics_section: metrics_section,
        errors_section: errors_section,
        warnings_section: warnings_section,
        recommendations: recommendations,
        timeline_data: timeline,
        metrics_charts: charts
    }

    RETURN report


FUNCTION generate_executive_summary(result: PipelineResult) -> String:
    """
    Generates executive summary text.
    """

    summary = """
    Pipeline Execution Summary
    ==========================

    Pipeline ID: {pipeline_id}
    Mode: {mode}
    Status: {status}
    Duration: {duration}

    Results:
    - Translation Coverage: {coverage}%
    - Test Pass Rate: {pass_rate}%
    - Performance Improvement: {speedup}x

    Artifacts Generated:
    - Rust Workspace: {rust_path}
    - WASM Binaries: {wasm_count} file(s)
    - NIM Containers: {nim_count} container(s)
    """.format(
        pipeline_id=result.pipeline_id,
        mode=result.mode,
        status="Success" IF result.success ELSE "Failed",
        duration=format_duration(result.total_duration_ms),
        coverage=result.translation_coverage * 100,
        pass_rate=result.test_pass_rate * 100,
        speedup=result.performance_improvement,
        rust_path=result.rust_workspace_path,
        wasm_count=result.wasm_binaries.length,
        nim_count=result.nim_containers.length
    )

    RETURN summary


FUNCTION find_artifact_path(artifacts: List<Artifact], artifact_type: String) -> Optional<Path>:
    """
    Finds the first artifact of a given type.
    """

    FOR artifact IN artifacts:
        IF artifact.artifact_type == artifact_type:
            RETURN artifact.path

    RETURN None


FUNCTION find_artifact_paths(artifacts: List<Artifact>, artifact_type: String) -> List<Path>:
    """
    Finds all artifacts of a given type.
    """

    paths = []

    FOR artifact IN artifacts:
        IF artifact.artifact_type == artifact_type:
            paths.append(artifact.path)

    RETURN paths


STRUCT QualityMetrics:
    translation_coverage: Float
    test_pass_rate: Float
    conformance_score: Float
    performance_improvement: Float
```

---

## 4. Input/Output Contracts

### 4.1 Layer Input Contract (from Presentation Layer)

```yaml
INPUT: PipelineExecutionRequest
  REQUIRED:
    configuration: PipelineConfiguration
      mode: ExecutionMode                    # Script or Library
      source_path: Path                      # Python file or package
      output_directory: Path                 # Output location

  OPTIONAL:
    agent_timeout_seconds: Map<String, Integer>  # Per-agent timeouts
    max_retries_per_agent: Integer               # Default: 3
    enable_checkpointing: Boolean                # Default: true for Library
    checkpoint_interval_seconds: Integer         # Default: 300
    log_level: String                            # Default: "INFO"
    fail_fast: Boolean                           # Default: false

PRECONDITIONS:
  - Configuration must be valid (validated by Presentation Layer)
  - Source path must exist and be accessible
  - Output directory must be writable
  - Agent services must be available or spawnable

VALIDATION:
  - Configuration validated before pipeline execution
  - Resource availability checked (CPU, memory, GPU)
  - Storage capacity verified for checkpoints
```

### 4.2 Layer Output Contract (to Presentation Layer)

```yaml
OUTPUT: PipelineResult
  ALWAYS_PRESENT:
    pipeline_id: String                      # Unique execution ID
    success: Boolean                         # True if pipeline completed successfully
    mode: ExecutionMode                      # Script or Library
    started_at: Timestamp
    completed_at: Timestamp
    total_duration_ms: Integer
    errors: List<PipelineError>              # All errors (empty if success)
    warnings: List<PipelineWarning>          # All warnings
    metadata: PipelineMetadata

  CONDITIONAL (on success):
    artifacts: List<Artifact>                # All generated artifacts
    rust_workspace_path: Path                # Rust workspace location
    wasm_binaries: List<Path>                # WASM binary paths
    nim_containers: List<Path>               # NIM container paths
    translation_coverage: Float              # 0.0 to 1.0
    test_pass_rate: Float                    # 0.0 to 1.0
    conformance_score: Float
    performance_improvement: Float           # Speedup factor

  PERFORMANCE_METRICS:
    stage_durations: Map<PipelineStage, Integer>
    peak_memory_mb: Integer
    total_cpu_time_ms: Integer
    total_gpu_time_ms: Integer
    gpu_utilization_avg: Float

POSTCONDITIONS:
  - success == true IFF all pipeline stages completed without blocking errors
  - All artifacts listed in artifacts are accessible
  - All errors have context and suggestions
  - Stage durations sum to approximately total_duration_ms
  - If success, at least one WASM binary is produced

ERROR_CASES:
  - ConfigurationError: Invalid configuration
  - AgentSpawnError: Failed to spawn required agent
  - AgentTimeoutError: Agent exceeded timeout
  - CheckpointError: Failed to create or restore checkpoint
  - ResourceExhaustedError: Insufficient CPU/memory/GPU resources
```

### 4.3 Progress Update Event Contract

```yaml
EVENT: ProgressUpdate
  EMITTED_WHEN:
    - Pipeline stage starts
    - Pipeline stage progresses (every ~5% or 10 seconds)
    - Pipeline stage completes
    - Pipeline completes or fails

  FIELDS:
    pipeline_id: String
    timestamp: Timestamp
    current_stage: PipelineStage
    stage_name: String                       # Human-readable
    overall_progress_percent: Float          # 0.0 to 100.0
    stage_progress_percent: Float
    elapsed_time_ms: Integer
    estimated_remaining_ms: Optional<Integer>
    estimated_completion_time: Optional<Timestamp>
    current_activity: String                 # Current operation description
    items_processed: Integer
    total_items: Integer

  CONSUMERS:
    - Web Dashboard (WebSocket subscription)
    - CLI (real-time terminal updates)
    - Monitoring System (metrics collection)
```

### 4.4 Interface to Agent Swarm Layer

```yaml
AGENT_TASK_REQUEST:
  INPUT:
    task_id: String
    agent_type: String                       # "ingest", "analysis", etc.
    input_data: AgentInput                   # Agent-specific input
    context: TaskContext
    timeout_seconds: Integer

  OUTPUT:
    result: AgentResult
      success: Boolean
      result_type: String
      data: Map<String, Any>
      processing_time_ms: Integer
      artifacts: List<Artifact>
      warnings: List<String>

  ERROR_CASES:
    - AgentTimeoutError: Timeout exceeded
    - AgentCrashedError: Agent process crashed
    - InvalidInputError: Input validation failed
    - ResourceUnavailableError: Required resources not available

AGENT_HEALTH_CHECK:
  INPUT:
    agent_id: String

  OUTPUT:
    status: AgentStatus
    last_heartbeat: Timestamp
    resource_usage: ResourceUsage
    queue_depth: Integer                     # Pending tasks

AGENT_SHUTDOWN:
  INPUT:
    agent_id: String
    graceful: Boolean                        # Graceful vs force
    timeout_seconds: Integer

  OUTPUT:
    success: Boolean
```

---

## 5. Error Handling Strategy

### 5.1 Error Categories

```pseudocode
// Error hierarchy for Orchestration Layer

ENUM OrchestrationErrorCategory:
    Configuration       // Invalid configuration
    AgentManagement     // Agent spawn/shutdown errors
    Execution           // Pipeline execution errors
    Checkpoint          // Checkpoint/restore errors
    Resource            // Resource allocation errors
    Timeout             // Timeout errors
    Internal            // Unexpected internal errors


STRUCT StageError:
    stage: PipelineStage
    error_type: String                       // Specific error type
    message: String
    details: Map<String, Any>
    recoverable: Boolean                     // Can pipeline continue?
    caused_by: Optional<Error>               // Underlying error


STRUCT AgentError:
    agent_type: String
    agent_id: Optional<String>
    error_type: String
    message: String
    details: Map<String, Any>
    recoverable: Boolean
    stack_trace: Optional<String>
```

### 5.2 Error Handling Principles

```pseudocode
PRINCIPLES:
  1. Retry Transient Errors
     - Network timeouts
     - Temporary resource unavailability
     - Transient agent failures
     - Use exponential backoff

  2. Circuit Breaker for Repeated Failures
     - Open circuit after N consecutive failures
     - Prevent cascading failures
     - Allow automatic recovery (half-open state)

  3. Graceful Degradation
     - Continue pipeline if error is non-blocking
     - Provide partial results when possible
     - Always create checkpoint before fatal failure

  4. Fail Fast on Configuration Errors
     - Invalid configuration stops pipeline immediately
     - User must fix configuration before retry

  5. Comprehensive Error Context
     - Include pipeline stage, agent type, task ID
     - Provide stack traces for debugging
     - Suggest remediation actions
     - Link to relevant documentation


ERROR_RECOVERY_STRATEGIES:

  STRATEGY: Retry with Exponential Backoff
    APPLIES_TO:
      - AgentTimeoutError
      - NetworkError
      - TransientResourceError

    ALGORITHM:
      1. Wait: base_delay * 2^(retry_count - 1)
      2. Retry up to max_retries times
      3. Increase timeout on each retry
      4. Fail if max_retries exhausted

  STRATEGY: Circuit Breaker
    APPLIES_TO:
      - Repeated agent failures
      - External service failures (NeMo, Triton)

    STATES:
      Closed: Normal operation, requests pass through
      Open: Failures exceeded threshold, reject requests
      Half-Open: Testing if service recovered

    TRANSITIONS:
      Closed -> Open: After N consecutive failures
      Open -> Half-Open: After timeout period
      Half-Open -> Closed: After M consecutive successes
      Half-Open -> Open: On any failure

  STRATEGY: Checkpoint and Resume
    APPLIES_TO:
      - Long-running Library Mode pipelines
      - After each successful stage

    ALGORITHM:
      1. Create checkpoint after each stage completion
      2. On failure, checkpoint includes error state
      3. User can resume from last checkpoint
      4. Skips completed stages on resume

  STRATEGY: Graceful Degradation
    APPLIES_TO:
      - Non-critical stage failures (when fail_fast=false)
      - Partial test failures
      - Non-blocking validation warnings

    ALGORITHM:
      1. Log error as warning
      2. Continue to next stage
      3. Mark result as partial success
      4. Include limitations in final report


EXAMPLE_ERROR_HANDLING:

  // Example 1: Retry transient agent timeout
  TRY:
    result = execute_agent(agent_type, input, context, timeout=300)
  CATCH error AS AgentTimeoutError:
    IF retry_count < max_retries:
      delay = calculate_backoff_delay(retry_policy, retry_count)
      sleep(delay)
      retry_count += 1
      // Retry with increased timeout
      timeout = timeout * 1.5
      RETRY
    ELSE:
      // Max retries exhausted
      RETURN Error(StageError {
        stage: current_stage,
        error_type: "AgentTimeout",
        message: "Agent timed out after {} retries".format(max_retries),
        recoverable: false
      })

  // Example 2: Circuit breaker for failing agent
  circuit_breaker = get_circuit_breaker(agent_type)

  IF circuit_breaker.state == Open:
    RETURN Error(StageError {
      stage: current_stage,
      error_type: "CircuitBreakerOpen",
      message: "Agent {} is unavailable (too many failures)".format(agent_type),
      recoverable: false
    })

  TRY:
    result = execute_agent(...)
    circuit_breaker.record_success()
  CATCH error:
    circuit_breaker.record_failure()
    RAISE error

  // Example 3: Checkpoint before fatal error
  IF fatal_error:
    TRY:
      pipeline_manager.create_checkpoint(execution_state)
    CATCH checkpoint_error:
      LOG.error("Failed to create emergency checkpoint", error=checkpoint_error)

    RETURN Error(fatal_error)
```

### 5.3 Logging Strategy

```pseudocode
LOGGING_LEVELS:
  ERROR:   Blocking errors that cause stage/pipeline failure
  WARNING: Non-blocking issues, degraded performance, retries
  INFO:    Progress updates, stage transitions, key milestones
  DEBUG:   Detailed execution flow, agent communication, metrics

LOGGING_POINTS:
  - Pipeline start/end (INFO)
  - Stage start/completion (INFO)
  - Agent spawn/shutdown (INFO)
  - Checkpoint creation/restoration (INFO)
  - Errors and warnings (ERROR/WARNING)
  - Retry attempts (WARNING)
  - Circuit breaker state changes (WARNING)
  - Progress updates (DEBUG)
  - Agent communication (DEBUG)

STRUCTURED_LOGGING:
  // Use JSON format for machine-readable logs
  {
    "timestamp": "2025-10-03T12:34:56.789Z",
    "level": "INFO",
    "component": "OrchestrationLayer",
    "pipeline_id": "uuid-1234",
    "stage": "Transpiling",
    "message": "Stage completed successfully",
    "duration_ms": 15234,
    "agent_type": "transpiler",
    "context": {...}
  }

CORRELATION:
  - Use pipeline_id to correlate all logs for a pipeline
  - Use trace_id/span_id for distributed tracing
  - Include stage and agent_type for filtering
```

---

## 6. London School TDD Test Points

### 6.1 Testing Strategy Overview

The Orchestration Layer follows **London School TDD** (mockist, outside-in approach):

1. **Start with Acceptance Tests**: Test `execute_pipeline()` with mocked dependencies
2. **Work Inward**: Test each component (FlowController, AgentCoordinator, PipelineManager) in isolation
3. **Verify Behavior**: Focus on interactions (method calls) not state
4. **Mock External Dependencies**: Agents, storage, message router, etc.

### 6.2 Test Doubles (Mocks/Stubs)

```pseudocode
// Mock interfaces for external dependencies

INTERFACE AgentCoordinatorInterface:
    execute_agent(agent_type: String, input: AgentInput, context: TaskContext, timeout: Integer)
        -> Result<AgentResult, AgentError>
    spawn_agent(agent_type: String, config: Map<String, Any>) -> Result<String, Error>
    get_circuit_breaker(agent_type: String) -> CircuitBreaker
    shutdown_all_agents()


INTERFACE PipelineManagerInterface:
    create_checkpoint(state: PipelineExecutionState) -> Result<Checkpoint, Error>
    find_latest_checkpoint(source_path: Path) -> Optional<Checkpoint>
    restore_from_checkpoint(checkpoint: Checkpoint) -> PipelineExecutionState


INTERFACE ProgressReporterInterface:
    update_stage(stage: PipelineStage, progress: Float)
    add_listener(listener: ProgressListener)
    record_stage_completion(stage: PipelineStage, duration: Integer)


INTERFACE MetricsCollectorInterface:
    record_agent_success(agent_type: String, duration: Integer)
    record_agent_failure(agent_type: String, error: AgentError)
    get_resource_usage() -> ResourceUsage


INTERFACE CheckpointStorageInterface:
    save_checkpoint(checkpoint_id: String, data: Bytes) -> Path
    load_checkpoint(checkpoint_id: String) -> Bytes
    list_checkpoints_for_source(source_path: Path) -> List<Checkpoint>


INTERFACE MessageRouterInterface:
    send_and_wait(endpoint: String, task: AgentTask, timeout: Integer) -> Result<AgentResult, Error>
    send_shutdown(endpoint: String)


// Example test doubles

CLASS MockAgentCoordinator IMPLEMENTS AgentCoordinatorInterface:
    // Configurable mock that records method calls
    execute_agent_calls: List<(String, AgentInput, TaskContext, Integer)>
    execute_agent_results: Map<String, Result<AgentResult, AgentError>>

    FUNCTION execute_agent(agent_type, input, context, timeout) -> Result<AgentResult, AgentError>:
        self.execute_agent_calls.append((agent_type, input, context, timeout))

        IF agent_type IN self.execute_agent_results:
            RETURN self.execute_agent_results[agent_type]

        // Default success
        RETURN Success(AgentResult {...})


CLASS StubPipelineManager IMPLEMENTS PipelineManagerInterface:
    // Stub with predefined responses
    checkpoints: Map<Path, Checkpoint>

    FUNCTION find_latest_checkpoint(source_path) -> Optional<Checkpoint>:
        RETURN self.checkpoints.get(source_path, None)

    FUNCTION create_checkpoint(state) -> Result<Checkpoint, Error>:
        checkpoint = Checkpoint {...}
        RETURN Success(checkpoint)
```

### 6.3 Test Hierarchy

```pseudocode
ACCEPTANCE_TESTS:
  // Test execute_pipeline() with all dependencies mocked

  TEST "successful script mode pipeline execution":
    GIVEN:
      - MockAgentCoordinator with successful agent responses for all stages
      - StubPipelineManager with no existing checkpoints
      - MockProgressReporter
      - MockMetricsCollector
      - Valid PipelineConfiguration for Script Mode

    WHEN:
      - Call execute_pipeline(config)

    THEN:
      - Result.success == true
      - All 7 stages executed (Ingest -> Analyze -> Specify -> Transpile -> Build -> Test -> Package)
      - AgentCoordinator.execute_agent called 7 times (once per stage)
      - Each stage result captured in final result
      - Progress updates emitted for each stage
      - Metrics collected for all stages
      - Final result includes all artifacts


  TEST "library mode with checkpoint resumption":
    GIVEN:
      - Existing checkpoint after Transpiling stage
      - StubPipelineManager.find_latest_checkpoint returns checkpoint
      - MockAgentCoordinator configured to succeed for remaining stages
      - Valid PipelineConfiguration for Library Mode with checkpointing enabled

    WHEN:
      - Call execute_pipeline(config)

    THEN:
      - Pipeline restored from checkpoint
      - Only remaining stages executed (Building, Testing, Packaging)
      - AgentCoordinator.execute_agent called 3 times (not 7)
      - Completed stages from checkpoint not re-executed
      - New checkpoint created after Building stage completes


  TEST "stage failure with retry and recovery":
    GIVEN:
      - MockAgentCoordinator configured to fail Transpiling stage twice, succeed third time
      - PipelineConfiguration with max_retries=3, retry_strategy=ExponentialBackoff
      - Other stages succeed

    WHEN:
      - Call execute_pipeline(config)

    THEN:
      - Result.success == true (eventual success after retries)
      - AgentCoordinator.execute_agent called for Transpiling stage 3 times
      - Exponential backoff delays observed between retries
      - Timeout increased on each retry
      - Final result includes retry count in metadata


  TEST "fatal error creates emergency checkpoint":
    GIVEN:
      - MockAgentCoordinator configured to fail Building stage with non-recoverable error
      - StubPipelineManager
      - PipelineConfiguration with fail_fast=true

    WHEN:
      - Call execute_pipeline(config)

    THEN:
      - Result.success == false
      - Pipeline stopped at Building stage
      - PipelineManager.create_checkpoint called (emergency checkpoint)
      - Error details included in result
      - Partial results from completed stages included


COMPONENT_TESTS:
  // Test individual orchestration components

  TEST "FlowController.execute_stage with successful agent":
    GIVEN:
      - MockAgentCoordinator.execute_agent returns success
      - PipelineStage.Analyzing
      - Execution state with completed Ingesting stage

    WHEN:
      - Call execute_stage(Analyzing, state, coordinator, ...)

    THEN:
      - Returns Success(AgentResult)
      - AgentCoordinator.execute_agent called with agent_type="analysis"
      - Input prepared from Ingesting stage result
      - Metrics recorded
      - Progress updated


  TEST "FlowController.execute_stage with circuit breaker open":
    GIVEN:
      - MockAgentCoordinator with circuit breaker open for "transpiler"
      - PipelineStage.Transpiling

    WHEN:
      - Call execute_stage(Transpiling, state, coordinator, ...)

    THEN:
      - Returns Error(StageError)
      - Error type is "CircuitBreakerOpen"
      - AgentCoordinator.execute_agent NOT called (rejected immediately)


  TEST "AgentCoordinator.execute_agent records circuit breaker failures":
    GIVEN:
      - AgentCoordinator with closed circuit breaker
      - Agent execution fails 5 times consecutively
      - Circuit breaker threshold = 5

    WHEN:
      - Call execute_agent() 5 times with failures
      - Call execute_agent() 6th time

    THEN:
      - First 5 calls record failures
      - After 5th failure, circuit breaker state becomes Open
      - 6th call rejected with CircuitBreakerOpen error


  TEST "AgentCoordinator spawns new agent when pool is empty":
    GIVEN:
      - AgentCoordinator with empty agent pool for "ingest"
      - spawn_agent() succeeds

    WHEN:
      - Call execute_agent(agent_type="ingest", ...)

    THEN:
      - spawn_agent("ingest") called
      - New agent registered in pool
      - Task sent to new agent
      - Result returned


  TEST "PipelineManager.create_checkpoint serializes state":
    GIVEN:
      - PipelineExecutionState with 3 completed stages
      - MockCheckpointStorage

    WHEN:
      - Call create_checkpoint(state)

    THEN:
      - Checkpoint object created with state snapshot
      - Artifacts manifest collected
      - Checkpoint serialized to bytes
      - Checkpoint compressed
      - Storage.save_checkpoint called with compressed data
      - Returns Success(Checkpoint)


  TEST "PipelineManager.restore_from_checkpoint deserializes state":
    GIVEN:
      - Checkpoint with state after Analyzing stage
      - MockCheckpointStorage.load_checkpoint returns checkpoint data

    WHEN:
      - Call restore_from_checkpoint(checkpoint)

    THEN:
      - Checkpoint loaded from storage
      - Data decompressed
      - State deserialized
      - Pending stages updated (skip completed stages)
      - New pipeline_id generated
      - Returns updated PipelineExecutionState


  TEST "ProgressReporter.calculate_overall_progress":
    GIVEN:
      - ProgressReporter with 7 stages
      - 3 stages completed (Ingest, Analyze, Specify)
      - Current stage (Transpile) at 50%

    WHEN:
      - Call calculate_overall_progress()

    THEN:
      - Returns approximately 50% overall
        (3 complete + 0.5 in progress) / 7 = 0.5 = 50%


  TEST "ProgressReporter.estimate_overall_completion_time uses historical data":
    GIVEN:
      - ProgressReporter with historical durations for all stages
      - Current stage (Transpile) in progress
      - Remaining stages not started

    WHEN:
      - Call estimate_overall_completion_time()

    THEN:
      - Estimates based on current progress + historical averages
      - Returns estimated remaining time in milliseconds


  TEST "CircuitBreaker transitions from Closed to Open on failures":
    GIVEN:
      - CircuitBreaker in Closed state
      - failure_threshold = 5

    WHEN:
      - Call record_failure() 5 times

    THEN:
      - After 5th failure, state becomes Open
      - last_state_change updated
      - consecutive_failures = 5


  TEST "CircuitBreaker transitions from Open to HalfOpen after timeout":
    GIVEN:
      - CircuitBreaker in Open state
      - timeout_ms = 60000 (1 minute)
      - 61 seconds elapsed since last_state_change

    WHEN:
      - Request comes in (triggers timeout check)

    THEN:
      - State transitions to HalfOpen
      - Request is allowed to proceed (not rejected)


  TEST "CircuitBreaker transitions from HalfOpen to Closed on successes":
    GIVEN:
      - CircuitBreaker in HalfOpen state
      - success_threshold = 2

    WHEN:
      - Call record_success() 2 times

    THEN:
      - After 2nd success, state becomes Closed
      - consecutive_successes reset to 0


UNIT_TESTS:
  // Test pure functions with no dependencies

  TEST "determine_pipeline_stages returns correct stages for Script mode":
    WHEN:
      - Call determine_pipeline_stages(ExecutionMode.Script)

    THEN:
      - Returns [Ingesting, Analyzing, Specifying, Transpiling, Building, Testing, Packaging]


  TEST "stage_to_agent_type maps stages correctly":
    ASSERT stage_to_agent_type(PipelineStage.Ingesting) == "ingest"
    ASSERT stage_to_agent_type(PipelineStage.Analyzing) == "analysis"
    ASSERT stage_to_agent_type(PipelineStage.Transpiling) == "transpiler"


  TEST "calculate_backoff_delay with exponential backoff":
    GIVEN:
      - RetryPolicy with strategy=ExponentialBackoff, base_delay_ms=1000, max_delay_ms=30000

    ASSERT calculate_backoff_delay(policy, retry_count=1) == 1000   // 1000 * 2^0
    ASSERT calculate_backoff_delay(policy, retry_count=2) == 2000   // 1000 * 2^1
    ASSERT calculate_backoff_delay(policy, retry_count=3) == 4000   // 1000 * 2^2
    ASSERT calculate_backoff_delay(policy, retry_count=10) == 30000 // Capped at max


  TEST "should_checkpoint returns true after interval":
    GIVEN:
      - PipelineExecutionState with last_checkpoint_time = 5 minutes ago
      - PipelineConfiguration with checkpoint_interval_seconds = 300 (5 minutes)

    WHEN:
      - Call should_checkpoint(state, config)

    THEN:
      - Returns true


  TEST "should_checkpoint returns false before interval":
    GIVEN:
      - PipelineExecutionState with last_checkpoint_time = 2 minutes ago
      - PipelineConfiguration with checkpoint_interval_seconds = 300 (5 minutes)

    WHEN:
      - Call should_checkpoint(state, config)

    THEN:
      - Returns false


  TEST "prepare_agent_input for Analyzing stage uses Ingest result":
    GIVEN:
      - PipelineExecutionState with Ingesting stage completed
      - Ingest stage result contains modules and dependency_graph

    WHEN:
      - Call prepare_agent_input(PipelineStage.Analyzing, state)

    THEN:
      - Returns AgentInput with input_type="analysis"
      - Data includes "ingest_result", "modules", "dependency_graph" from Ingest stage
```

### 6.4 Contract Tests

```pseudocode
CONTRACT_TESTS:
  // Tests that verify interfaces between layers

  TEST "PipelineResult satisfies contract for Presentation Layer":
    GIVEN:
      - Successful PipelineResult from Script mode

    VERIFY:
      - result.success == true
      - result.pipeline_id is present
      - result.total_duration_ms > 0
      - result.artifacts is not empty
      - result.wasm_binaries contains at least one path
      - All paths in artifacts are absolute
      - All errors and warnings have context


  TEST "ProgressUpdate satisfies contract for consumers":
    GIVEN:
      - ProgressUpdate emitted during Transpiling stage

    VERIFY:
      - pipeline_id is present
      - current_stage == PipelineStage.Transpiling
      - overall_progress_percent in range [0.0, 100.0]
      - stage_progress_percent in range [0.0, 100.0]
      - timestamp is recent (within last 5 seconds)


  TEST "AgentTask satisfies contract for Agent Swarm Layer":
    GIVEN:
      - AgentTask created for Transpiler Agent

    VERIFY:
      - task_id is unique
      - agent_type == "transpiler"
      - input_data contains required fields for transpiler
      - context includes pipeline_id, trace_id
      - timeout_seconds > 0
```

### 6.5 Test Coverage Targets

```
Line Coverage:        >80%
Branch Coverage:      >75%
Function Coverage:    100% (all public functions tested)
Contract Coverage:    100% (input/output contracts verified)

Critical Paths:       100% coverage
  - Pipeline state machine transitions
  - Agent coordination and retry logic
  - Circuit breaker state transitions
  - Checkpoint creation and restoration
  - Error handling and recovery
  - Progress calculation and ETA estimation

Edge Cases:           Required tests
  - Circuit breaker state transitions (all combinations)
  - Retry with all strategies (immediate, fixed, exponential)
  - Checkpoint restoration with missing artifacts
  - Agent timeout during all stages
  - Concurrent agent failures
  - Rapid pipeline cancellation
```

### 6.6 Mock Verification Examples

```pseudocode
EXAMPLE_MOCK_VERIFICATION:

TEST "execute_pipeline calls agents in correct sequence":
    // Setup
    mock_coordinator = MockAgentCoordinator()
    mock_coordinator.execute_agent_results = {
        "ingest": Success(AgentResult {...}),
        "analysis": Success(AgentResult {...}),
        "specification": Success(AgentResult {...}),
        "transpiler": Success(AgentResult {...}),
        "build": Success(AgentResult {...}),
        "test": Success(AgentResult {...}),
        "packaging": Success(AgentResult {...})
    }

    config = PipelineConfiguration {mode: Script, ...}

    // Execute
    result = execute_pipeline(config, agent_coordinator=mock_coordinator)

    // Verify interactions (London School style)
    ASSERT mock_coordinator.execute_agent_calls.length == 7

    // Verify call sequence
    ASSERT mock_coordinator.execute_agent_calls[0][0] == "ingest"
    ASSERT mock_coordinator.execute_agent_calls[1][0] == "analysis"
    ASSERT mock_coordinator.execute_agent_calls[2][0] == "specification"
    ASSERT mock_coordinator.execute_agent_calls[3][0] == "transpiler"
    ASSERT mock_coordinator.execute_agent_calls[4][0] == "build"
    ASSERT mock_coordinator.execute_agent_calls[5][0] == "test"
    ASSERT mock_coordinator.execute_agent_calls[6][0] == "packaging"

    // Verify result
    ASSERT result.success == true
    ASSERT result.stage_results.length == 7


TEST "pipeline creates checkpoints at intervals":
    // Setup
    mock_manager = MockPipelineManager()
    mock_manager.find_latest_checkpoint_result = None  // No existing checkpoint

    config = PipelineConfiguration {
        mode: Library,
        enable_checkpointing: true,
        checkpoint_interval_seconds: 0,  // Checkpoint after every stage
        ...
    }

    // Execute (assume all stages succeed quickly)
    result = execute_pipeline(config, pipeline_manager=mock_manager)

    // Verify checkpoints created
    ASSERT mock_manager.create_checkpoint_calls.length >= 1

    // Verify checkpoint created after stages
    // (Exact count depends on timing and checkpoint interval)


TEST "circuit breaker prevents calls to failing agent":
    // Setup
    mock_coordinator = MockAgentCoordinator()

    // Configure transpiler to fail 5 times (threshold)
    FOR i IN range(5):
        mock_coordinator.add_failure_response("transpiler", AgentError {...})

    config = PipelineConfiguration {
        circuit_breaker_enabled: true,
        circuit_breaker_threshold: 5,
        ...
    }

    // Execute first pipeline - should fail and open circuit
    result1 = execute_pipeline(config, agent_coordinator=mock_coordinator)

    ASSERT result1.success == false
    ASSERT mock_coordinator.get_circuit_breaker("transpiler").state == Open

    // Execute second pipeline immediately - should be rejected
    result2 = execute_pipeline(config, agent_coordinator=mock_coordinator)

    ASSERT result2.success == false
    ASSERT "CircuitBreakerOpen" in result2.errors[0].error_type

    // Verify transpiler was NOT called in second pipeline
    // (Circuit breaker rejected before agent call)
```

---

## 7. Summary

This pseudocode specification for the **Orchestration Layer** provides:

1. **Complete Data Structures**: All enums, structs, and types for orchestration, state management, and coordination
2. **Detailed Algorithms**: Step-by-step pseudocode for pipeline execution, agent coordination, checkpointing, and progress tracking
3. **Clear Contracts**: Input/output specifications and interfaces to Presentation and Agent Swarm layers
4. **Robust Error Handling**: Comprehensive retry logic, circuit breaker pattern, and graceful degradation
5. **TDD Test Points**: London School test structure with mocks and verification for all components

### Key Design Decisions

1. **Pipeline State Machine**: Clear stages with defined transitions and error states
2. **Agent Independence**: Agents are separate processes with well-defined contracts
3. **Retry with Exponential Backoff**: Transient errors are retried with increasing delays
4. **Circuit Breaker Pattern**: Prevents cascading failures from repeatedly failing agents
5. **Checkpoint and Resume**: Long-running pipelines can be paused and resumed
6. **Progress Tracking with ETA**: Real-time progress updates with completion time estimates
7. **Observability First**: Structured logging, metrics collection, comprehensive reporting
8. **Testability**: All external dependencies injected via interfaces for easy mocking

### Component Responsibilities

#### Flow Controller
- Manages overall pipeline execution
- Implements pipeline state machine
- Coordinates sequential and parallel agent execution
- Handles mode-specific workflows (Script vs Library)

#### Agent Coordinator
- Manages agent lifecycle (spawn, monitor, shutdown)
- Routes messages to agents
- Implements circuit breaker pattern
- Handles agent health monitoring and failures

#### Pipeline Manager
- Maintains pipeline state
- Creates checkpoints at regular intervals
- Restores pipeline from checkpoints
- Provides pipeline history and audit trail

### Performance Targets

- **Script Mode**: <60 seconds end-to-end
- **Library Mode**: <15 minutes for 5000 LOC
- **Checkpoint Creation**: <5 seconds
- **Agent Spawn**: <10 seconds
- **Progress Update Frequency**: Every 5% or 10 seconds

### Next Steps (SPARC Phase 3: Architecture)

The Architecture phase will:
- Define concrete Rust types and trait implementations
- Specify async/await patterns for concurrent agent execution
- Design the message passing infrastructure (gRPC, ZeroMQ, etc.)
- Detail the checkpoint storage backend (filesystem, S3, database)
- Specify metrics collection and monitoring integration (Prometheus, OpenTelemetry)
- Define the distributed tracing implementation

---

**END OF ORCHESTRATION LAYER PSEUDOCODE**
