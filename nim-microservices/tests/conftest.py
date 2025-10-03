"""
Pytest configuration and fixtures for integration tests
"""

import pytest
import os
import sys


# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        "api_url": os.getenv("TEST_API_URL", "http://localhost:8000"),
        "grpc_url": os.getenv("TEST_GRPC_URL", "localhost:50051"),
        "timeout": int(os.getenv("TEST_TIMEOUT", "30")),
    }


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing"""
    return """
def factorial(n: int) -> int:
    '''Calculate factorial of n'''
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""


@pytest.fixture
def sample_batch_files():
    """Sample batch files for testing"""
    return [
        "def add(a, b): return a + b",
        "def multiply(a, b): return a * b",
        "def power(a, b): return a ** b"
    ]
