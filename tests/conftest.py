"""
Shared test fixtures and configuration for Portalis NVIDIA Stack testing

This module provides:
- Common fixtures for all test suites
- Test environment setup and teardown
- Mock services and data generators
- Shared utilities for testing
"""

import os
import sys
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, MagicMock
import torch
import numpy as np

# Add project directories to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "nemo-integration" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda-acceleration"))
sys.path.insert(0, str(PROJECT_ROOT / "deployment" / "triton"))
sys.path.insert(0, str(PROJECT_ROOT / "nim-microservices" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "dgx-cloud" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "omniverse-integration"))


# ============================================================================
# Session-scoped fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Global test configuration."""
    return {
        "triton_url": os.getenv("TRITON_URL", "localhost:8000"),
        "nim_api_url": os.getenv("NIM_API_URL", "http://localhost:8000"),
        "dgx_cloud_url": os.getenv("DGX_CLOUD_URL", "http://localhost:8080"),
        "test_timeout": int(os.getenv("TEST_TIMEOUT", "300")),
        "enable_gpu_tests": os.getenv("ENABLE_GPU_TESTS", "false").lower() == "true",
        "enable_network_tests": os.getenv("ENABLE_NETWORK_TESTS", "true").lower() == "true",
        "test_data_dir": PROJECT_ROOT / "tests" / "fixtures" / "data",
        "mock_mode": os.getenv("MOCK_MODE", "true").lower() == "true",
    }


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def gpu_device():
    """Get GPU device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ============================================================================
# Function-scoped fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for translation tests."""
    return """
def fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number.'''
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def main():
    result = fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def sample_python_class() -> str:
    """Sample Python class for translation tests."""
    return """
class Calculator:
    '''Simple calculator class.'''

    def __init__(self):
        self.result = 0

    def add(self, a: float, b: float) -> float:
        self.result = a + b
        return self.result

    def multiply(self, a: float, b: float) -> float:
        self.result = a * b
        return self.result

    def power(self, base: float, exponent: float) -> float:
        self.result = base ** exponent
        return self.result
"""


@pytest.fixture
def sample_batch_files() -> List[str]:
    """Sample batch of Python files for testing."""
    return [
        "def add(a, b): return a + b",
        "def subtract(a, b): return a - b",
        "def multiply(a, b): return a * b",
        "def divide(a, b): return a / b if b != 0 else 0",
        "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
    ]


@pytest.fixture
def sample_rust_code() -> str:
    """Sample Rust code for validation tests."""
    return """
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}

fn main() {
    let result = fibonacci(10);
    println!("Fibonacci(10) = {}", result);
}
"""


@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Sample numpy array for testing."""
    return np.random.rand(100, 100).astype(np.float32)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Sample embeddings for testing."""
    return np.random.randn(10, 768).astype(np.float32)


# ============================================================================
# NeMo-specific fixtures
# ============================================================================

@pytest.fixture
def mock_nemo_model():
    """Mock NeMo model for testing."""
    model = MagicMock()
    model.generate.return_value = [
        "fn test() -> i64 { 42 }",
        "fn example() -> String { String::from(\"test\") }"
    ]
    model.encode.return_value = np.random.randn(768).astype(np.float32)
    return model


@pytest.fixture
def nemo_inference_config():
    """NeMo inference configuration for testing."""
    return {
        "max_length": 512,
        "temperature": 0.2,
        "batch_size": 4,
        "use_gpu": False,
        "top_k": 50,
        "top_p": 0.95,
    }


# ============================================================================
# CUDA-specific fixtures
# ============================================================================

@pytest.fixture
def cuda_context(cuda_available):
    """CUDA context for testing."""
    if not cuda_available:
        pytest.skip("CUDA not available")

    # Setup CUDA context
    torch.cuda.init()
    device = torch.device("cuda:0")

    yield device

    # Cleanup
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.fixture
def sample_cuda_tensors(cuda_available):
    """Sample CUDA tensors for testing."""
    if cuda_available:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return {
        "input": torch.randn(100, 100, device=device),
        "weights": torch.randn(100, 100, device=device),
        "bias": torch.randn(100, device=device),
    }


# ============================================================================
# Triton-specific fixtures
# ============================================================================

@pytest.fixture
def triton_client_config(test_config):
    """Triton client configuration."""
    return {
        "url": test_config["triton_url"],
        "protocol": "http",
        "verbose": False,
        "max_retries": 3,
        "timeout": 30.0,
    }


@pytest.fixture
def mock_triton_client():
    """Mock Triton client for testing."""
    client = MagicMock()
    client.is_server_ready.return_value = True
    client.is_model_ready.return_value = True
    client.get_server_metadata.return_value = {
        "name": "triton",
        "version": "2.40.0"
    }
    return client


# ============================================================================
# NIM-specific fixtures
# ============================================================================

@pytest.fixture
async def nim_api_client(test_config):
    """Async HTTP client for NIM API testing."""
    import httpx

    async with httpx.AsyncClient(
        base_url=test_config["nim_api_url"],
        timeout=30.0
    ) as client:
        yield client


@pytest.fixture
def nim_auth_headers():
    """Authentication headers for NIM API."""
    return {
        "Authorization": "Bearer test_token",
        "X-API-Key": "test_api_key",
    }


@pytest.fixture
def nim_translation_request():
    """Sample NIM translation request."""
    return {
        "python_code": "def test(): return 42",
        "mode": "fast",
        "temperature": 0.2,
        "max_length": 512,
        "include_alternatives": False,
    }


# ============================================================================
# DGX Cloud-specific fixtures
# ============================================================================

@pytest.fixture
def dgx_cloud_config():
    """DGX Cloud configuration for testing."""
    return {
        "cluster_size": 4,
        "gpu_per_node": 8,
        "node_type": "DGX-A100",
        "storage_type": "nfs",
        "cost_limit": 1000.0,
    }


@pytest.fixture
def mock_dgx_scheduler():
    """Mock DGX scheduler for testing."""
    scheduler = MagicMock()
    scheduler.submit_job.return_value = {"job_id": "test-job-123", "status": "queued"}
    scheduler.get_job_status.return_value = {"status": "running", "progress": 0.5}
    return scheduler


# ============================================================================
# Omniverse-specific fixtures
# ============================================================================

@pytest.fixture
def wasm_binary() -> bytes:
    """Sample WASM binary for testing."""
    # Minimal valid WASM module
    return bytes([
        0x00, 0x61, 0x73, 0x6d,  # Magic number
        0x01, 0x00, 0x00, 0x00,  # Version
    ])


@pytest.fixture
def omniverse_stage_config():
    """Omniverse stage configuration."""
    return {
        "stage_url": "omniverse://localhost/test_stage.usd",
        "fps": 60,
        "enable_physics": True,
        "enable_rendering": False,
    }


@pytest.fixture
def mock_usd_stage():
    """Mock USD stage for testing."""
    stage = MagicMock()
    stage.GetPrimAtPath.return_value = MagicMock()
    return stage


# ============================================================================
# Performance testing fixtures
# ============================================================================

@pytest.fixture
def performance_metrics():
    """Container for collecting performance metrics."""
    return {
        "latencies": [],
        "throughputs": [],
        "memory_usage": [],
        "gpu_utilization": [],
    }


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "warmup_iterations": 5,
        "test_iterations": 100,
        "concurrent_requests": 10,
        "timeout": 300.0,
    }


# ============================================================================
# Test data generators
# ============================================================================

@pytest.fixture
def python_code_generator():
    """Generator for random Python code snippets."""
    def generate(complexity: str = "simple", count: int = 1) -> List[str]:
        templates = {
            "simple": [
                "def func_{i}(): return {i}",
                "x_{i} = {i} * 2",
                "result_{i} = [{i}, {i}+1, {i}+2]",
            ],
            "medium": [
                "def calculate_{i}(x):\n    return x * {i} + {i}",
                "class Class_{i}:\n    def __init__(self):\n        self.value = {i}",
            ],
            "complex": [
                """
def complex_{i}(n: int) -> int:
    if n <= 1:
        return n
    return complex_{i}(n-1) + complex_{i}(n-2)
""",
            ],
        }

        results = []
        template_list = templates.get(complexity, templates["simple"])

        for i in range(count):
            template = template_list[i % len(template_list)]
            results.append(template.format(i=i))

        return results

    return generate


# ============================================================================
# Cleanup and markers
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (moderate speed, local services)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (slow, full stack)"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU access"
    )
    config.addinivalue_line(
        "markers", "cuda: Tests requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests (>5 seconds)"
    )
    config.addinivalue_line(
        "markers", "benchmark: Performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_network = pytest.mark.skip(reason="Network tests disabled")
    skip_slow = pytest.mark.skip(reason="Slow tests disabled")

    enable_gpu = os.getenv("ENABLE_GPU_TESTS", "false").lower() == "true"
    enable_network = os.getenv("ENABLE_NETWORK_TESTS", "true").lower() == "true"
    enable_slow = os.getenv("ENABLE_SLOW_TESTS", "true").lower() == "true"

    for item in items:
        if "gpu" in item.keywords or "cuda" in item.keywords:
            if not enable_gpu or not torch.cuda.is_available():
                item.add_marker(skip_gpu)

        if "requires_network" in item.keywords:
            if not enable_network:
                item.add_marker(skip_network)

        if "slow" in item.keywords:
            if not enable_slow:
                item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    old_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Cleanup CUDA resources after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
