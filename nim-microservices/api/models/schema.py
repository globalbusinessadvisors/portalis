"""
API Schema Models for NIM Microservice

Pydantic models for request/response validation and OpenAPI documentation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class TranslationMode(str, Enum):
    """Translation mode options"""
    STANDARD = "standard"
    FAST = "fast"
    QUALITY = "quality"
    STREAMING = "streaming"


class OptimizationLevel(str, Enum):
    """Rust compilation optimization levels"""
    DEBUG = "debug"
    RELEASE = "release"
    RELEASE_WITH_DEBUG = "release-with-debug"


class TranslationRequest(BaseModel):
    """Request model for single code translation"""

    python_code: str = Field(
        ...,
        description="Python source code to translate",
        min_length=1,
        max_length=100000
    )

    mode: TranslationMode = Field(
        default=TranslationMode.STANDARD,
        description="Translation mode (affects speed/quality tradeoff)"
    )

    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context (imports, type hints, examples)"
    )

    temperature: Optional[float] = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Model temperature for generation (0.0-2.0)"
    )

    max_length: Optional[int] = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Maximum length of generated code"
    )

    include_alternatives: bool = Field(
        default=False,
        description="Include alternative translations"
    )

    @validator('python_code')
    def validate_python_code(cls, v):
        if not v.strip():
            raise ValueError("Python code cannot be empty")
        return v


class TranslationResponse(BaseModel):
    """Response model for code translation"""

    rust_code: str = Field(
        ...,
        description="Translated Rust code"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )

    alternatives: Optional[List[str]] = Field(
        default=None,
        description="Alternative translations if requested"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Translation metadata and statistics"
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Translation warnings"
    )

    suggestions: List[str] = Field(
        default_factory=list,
        description="Optimization suggestions"
    )

    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation"""

    source_files: List[str] = Field(
        ...,
        description="List of Python source files (as strings)",
        min_items=1,
        max_items=100
    )

    project_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Project configuration (dependencies, features)"
    )

    optimization_level: OptimizationLevel = Field(
        default=OptimizationLevel.RELEASE,
        description="Rust compilation optimization level"
    )

    compile_wasm: bool = Field(
        default=True,
        description="Compile to WebAssembly after translation"
    )

    run_tests: bool = Field(
        default=False,
        description="Run test suite after translation"
    )


class BatchTranslationResponse(BaseModel):
    """Response model for batch translation"""

    translated_files: List[str] = Field(
        ...,
        description="List of translated Rust files"
    )

    compilation_status: List[str] = Field(
        ...,
        description="Compilation status for each file"
    )

    performance_metrics: Dict[str, Any] = Field(
        ...,
        description="Performance metrics and benchmarks"
    )

    wasm_binaries: Optional[List[bytes]] = Field(
        default=None,
        description="Compiled WASM binaries if requested"
    )

    total_processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )

    success_count: int = Field(
        ...,
        description="Number of successfully translated files"
    )

    failure_count: int = Field(
        default=0,
        description="Number of failed translations"
    )


class StreamingChunk(BaseModel):
    """Streaming response chunk"""

    chunk_type: str = Field(
        ...,
        description="Type of chunk (code, metadata, complete)"
    )

    content: str = Field(
        ...,
        description="Chunk content"
    )

    is_final: bool = Field(
        default=False,
        description="Whether this is the final chunk"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Chunk metadata"
    )


class HealthCheckResponse(BaseModel):
    """Health check response"""

    status: str = Field(
        ...,
        description="Service status (healthy, degraded, unhealthy)"
    )

    version: str = Field(
        ...,
        description="Service version"
    )

    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds"
    )

    gpu_available: bool = Field(
        ...,
        description="Whether GPU is available"
    )

    model_loaded: bool = Field(
        ...,
        description="Whether translation model is loaded"
    )

    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of dependencies (triton, nemo, cuda)"
    )


class MetricsResponse(BaseModel):
    """Metrics response"""

    total_requests: int = Field(
        ...,
        description="Total number of requests processed"
    )

    successful_requests: int = Field(
        ...,
        description="Number of successful requests"
    )

    failed_requests: int = Field(
        ...,
        description="Number of failed requests"
    )

    avg_processing_time_ms: float = Field(
        ...,
        description="Average processing time in milliseconds"
    )

    p95_latency_ms: float = Field(
        ...,
        description="95th percentile latency in milliseconds"
    )

    p99_latency_ms: float = Field(
        ...,
        description="99th percentile latency in milliseconds"
    )

    gpu_utilization: Optional[float] = Field(
        default=None,
        description="GPU utilization percentage"
    )

    memory_usage_mb: float = Field(
        ...,
        description="Memory usage in megabytes"
    )


class ErrorResponse(BaseModel):
    """Error response"""

    error: str = Field(
        ...,
        description="Error type"
    )

    message: str = Field(
        ...,
        description="Error message"
    )

    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )

    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )


class ModelInfo(BaseModel):
    """Model information"""

    model_name: str = Field(
        ...,
        description="Model name"
    )

    version: str = Field(
        ...,
        description="Model version"
    )

    framework: str = Field(
        ...,
        description="Framework (NeMo, Triton, etc.)"
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model parameters and configuration"
    )

    capabilities: List[str] = Field(
        default_factory=list,
        description="Model capabilities"
    )
