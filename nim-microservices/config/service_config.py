"""
Service Configuration

Central configuration management for NIM microservice.
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from pathlib import Path


class ServiceConfig(BaseSettings):
    """Service configuration with environment variable support"""

    # Service info
    service_name: str = Field(
        default="portalis-nim",
        env="SERVICE_NAME"
    )
    service_version: str = Field(
        default="1.0.0",
        env="SERVICE_VERSION"
    )
    environment: str = Field(
        default="production",
        env="ENVIRONMENT"
    )

    # API Configuration
    host: str = Field(
        default="0.0.0.0",
        env="HOST"
    )
    port: int = Field(
        default=8000,
        env="PORT"
    )
    workers: int = Field(
        default=1,
        env="WORKERS"
    )
    log_level: str = Field(
        default="info",
        env="LOG_LEVEL"
    )

    # Model Configuration
    model_path: str = Field(
        default="/models/nemo_translation.nemo",
        env="MODEL_PATH"
    )
    model_version: str = Field(
        default="1.0.0",
        env="MODEL_VERSION"
    )

    # GPU Configuration
    enable_cuda: bool = Field(
        default=True,
        env="ENABLE_CUDA"
    )
    gpu_id: int = Field(
        default=0,
        env="GPU_ID"
    )
    gpu_memory_fraction: float = Field(
        default=0.9,
        env="GPU_MEMORY_FRACTION"
    )

    # Triton Configuration
    triton_url: str = Field(
        default="localhost:8000",
        env="TRITON_URL"
    )
    triton_protocol: str = Field(
        default="http",
        env="TRITON_PROTOCOL"
    )
    triton_model_name: str = Field(
        default="translation_model",
        env="TRITON_MODEL_NAME"
    )

    # Performance Configuration
    batch_size: int = Field(
        default=32,
        env="BATCH_SIZE"
    )
    max_queue_size: int = Field(
        default=100,
        env="MAX_QUEUE_SIZE"
    )
    request_timeout: int = Field(
        default=300,
        env="REQUEST_TIMEOUT"
    )

    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=60,
        env="RATE_LIMIT_PER_MINUTE"
    )
    rate_limit_burst: int = Field(
        default=10,
        env="RATE_LIMIT_BURST"
    )

    # Authentication
    api_keys: Optional[str] = Field(
        default=None,
        env="API_KEYS"
    )
    enable_auth: bool = Field(
        default=False,
        env="ENABLE_AUTH"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS"
    )
    metrics_port: int = Field(
        default=9090,
        env="METRICS_PORT"
    )
    enable_tracing: bool = Field(
        default=False,
        env="ENABLE_TRACING"
    )
    jaeger_agent_host: Optional[str] = Field(
        default=None,
        env="JAEGER_AGENT_HOST"
    )

    # CORS
    cors_origins: str = Field(
        default="*",
        env="CORS_ORIGINS"
    )

    # Feature Flags
    enable_streaming: bool = Field(
        default=True,
        env="ENABLE_STREAMING"
    )
    enable_batch: bool = Field(
        default=True,
        env="ENABLE_BATCH"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_api_keys_dict(self):
        """Parse API keys from comma-separated string"""
        if not self.api_keys:
            return {}

        keys = {}
        for item in self.api_keys.split(","):
            parts = item.split(":")
            if len(parts) == 2:
                keys[parts[0]] = parts[1]
        return keys

    def get_cors_origins_list(self):
        """Parse CORS origins from comma-separated string"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global config instance
_config: Optional[ServiceConfig] = None


def get_service_config() -> ServiceConfig:
    """Get or create service configuration"""
    global _config
    if _config is None:
        _config = ServiceConfig()
    return _config


def reload_config() -> ServiceConfig:
    """Reload service configuration"""
    global _config
    _config = ServiceConfig()
    return _config
