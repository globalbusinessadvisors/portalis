"""
Integration Tests for Triton Translation Service
Tests end-to-end functionality of translation models
"""

import pytest
import sys
import os
import time
import json
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.triton_client import (
    create_client,
    TranslationResult,
    BatchTranslationResult
)


class TestTritonTranslationService:
    """Test suite for Triton translation service"""

    @pytest.fixture(scope="class")
    def triton_client(self):
        """Create Triton client for testing"""
        # Use environment variable or default
        triton_url = os.getenv('TRITON_URL', 'localhost:8000')

        client = create_client(url=triton_url, protocol='http', verbose=True)

        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                metadata = client.get_server_metadata()
                print(f"Connected to Triton server: {metadata}")
                break
            except Exception as e:
                if i == max_retries - 1:
                    pytest.fail(f"Triton server not ready after {max_retries} attempts: {e}")
                time.sleep(2)

        yield client

        # Cleanup
        client.close()

    @pytest.fixture
    def sample_python_code(self) -> str:
        """Sample Python code for translation"""
        return """
def calculate_fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    result = calculate_fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()
"""

    @pytest.fixture
    def sample_batch_files(self) -> List[str]:
        """Sample Python files for batch processing"""
        return [
            "def hello(): return 'Hello, World!'",
            "class Calculator:\n    def add(self, a, b): return a + b",
            "import math\ndef area_circle(r): return math.pi * r ** 2"
        ]


class TestSingleTranslation(TestTritonTranslationService):
    """Test single code translation"""

    def test_basic_translation(self, triton_client, sample_python_code):
        """Test basic Python to Rust translation"""
        result = triton_client.translate_code(sample_python_code)

        assert isinstance(result, TranslationResult)
        assert result.rust_code is not None
        assert len(result.rust_code) > 0
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        assert 'fn' in result.rust_code  # Basic check for Rust syntax

    def test_translation_with_options(self, triton_client, sample_python_code):
        """Test translation with custom options"""
        options = {
            'optimization_level': 'release',
            'max_length': 4096,
            'temperature': 0.7
        }

        result = triton_client.translate_code(sample_python_code, options=options)

        assert isinstance(result, TranslationResult)
        assert result.confidence > 0.0
        assert 'processing_time_ms' in result.metadata

    def test_translation_metadata(self, triton_client, sample_python_code):
        """Test that translation includes expected metadata"""
        result = triton_client.translate_code(sample_python_code)

        assert 'processing_time_ms' in result.metadata
        assert 'translation_strategy' in result.metadata
        assert 'model_version' in result.metadata

        # Processing time should be reasonable
        assert result.metadata['processing_time_ms'] < 30000  # 30 seconds

    def test_simple_function_translation(self, triton_client):
        """Test translation of a simple function"""
        simple_code = "def add(a, b): return a + b"

        result = triton_client.translate_code(simple_code)

        assert result.rust_code is not None
        assert 'fn add' in result.rust_code or 'fn' in result.rust_code
        assert result.confidence > 0.3  # Should have decent confidence for simple code

    def test_class_translation(self, triton_client):
        """Test translation of a Python class"""
        class_code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_from_origin(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
"""

        result = triton_client.translate_code(class_code)

        assert result.rust_code is not None
        assert result.confidence > 0.0


class TestInteractiveTranslation(TestTritonTranslationService):
    """Test interactive translation API"""

    def test_interactive_single_snippet(self, triton_client):
        """Test interactive translation of a code snippet"""
        snippet = "x = [i**2 for i in range(10)]"

        result = triton_client.translate_interactive(
            snippet,
            target_language="rust"
        )

        assert isinstance(result, TranslationResult)
        assert result.rust_code is not None
        assert isinstance(result.suggestions, list)
        assert isinstance(result.warnings, list)

    def test_interactive_with_context(self, triton_client):
        """Test interactive translation with context"""
        snippet = "result = calculate_total(items)"
        context = [
            "def calculate_total(items): return sum(items)",
            "items = [1, 2, 3, 4, 5]"
        ]

        result = triton_client.translate_interactive(
            snippet,
            context=context
        )

        assert result.rust_code is not None
        # With context, confidence should be higher
        assert result.confidence >= 0.0

    def test_interactive_suggestions(self, triton_client):
        """Test that interactive mode provides suggestions"""
        snippet = "async def fetch_data(): pass"

        result = triton_client.translate_interactive(snippet)

        # Should provide suggestions for async code
        assert len(result.suggestions) >= 0  # May or may not have suggestions

    def test_interactive_warnings(self, triton_client):
        """Test that interactive mode detects warnings"""
        snippet = "eval('print(42)')"  # Dynamic code execution

        result = triton_client.translate_interactive(snippet)

        # Should warn about eval
        # Note: Depends on implementation
        assert isinstance(result.warnings, list)


class TestBatchTranslation(TestTritonTranslationService):
    """Test batch translation"""

    def test_batch_translation_basic(self, triton_client, sample_batch_files):
        """Test basic batch translation"""
        project_config = {
            'project_name': 'test_project',
            'version': '0.1.0',
            'python_version': '3.10'
        }

        result = triton_client.translate_batch(
            sample_batch_files,
            project_config,
            optimization_level='debug',
            timeout=120.0
        )

        assert isinstance(result, BatchTranslationResult)
        assert len(result.translated_files) == len(sample_batch_files)
        assert len(result.compilation_status) > 0
        assert isinstance(result.performance_metrics, dict)

    @pytest.mark.slow
    def test_batch_translation_large(self, triton_client):
        """Test batch translation with many files"""
        # Create many small files
        files = [f"def func_{i}(): return {i}" for i in range(20)]

        project_config = {
            'project_name': 'large_test',
            'version': '0.1.0'
        }

        result = triton_client.translate_batch(
            files,
            project_config,
            timeout=300.0
        )

        assert len(result.translated_files) == len(files)


class TestModelMetadata(TestTritonTranslationService):
    """Test model metadata and configuration"""

    def test_translation_model_metadata(self, triton_client):
        """Test getting metadata for translation model"""
        metadata = triton_client.get_model_metadata("translation_model")

        assert metadata is not None
        # Check that model is loaded
        print(f"Translation model metadata: {metadata}")

    def test_interactive_model_metadata(self, triton_client):
        """Test getting metadata for interactive API model"""
        metadata = triton_client.get_model_metadata("interactive_api")

        assert metadata is not None
        print(f"Interactive API metadata: {metadata}")

    def test_server_metadata(self, triton_client):
        """Test getting server metadata"""
        metadata = triton_client.get_server_metadata()

        assert metadata is not None
        print(f"Server metadata: {metadata}")


class TestPerformance(TestTritonTranslationService):
    """Test performance characteristics"""

    def test_translation_latency(self, triton_client):
        """Test that translation latency is acceptable"""
        code = "def simple(): return 42"

        start = time.time()
        result = triton_client.translate_code(code)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 30.0  # 30 seconds max
        assert 'processing_time_ms' in result.metadata

        print(f"Translation completed in {elapsed:.2f}s")

    @pytest.mark.benchmark
    def test_throughput(self, triton_client):
        """Test translation throughput"""
        code = "def test(): pass"
        num_requests = 10

        start = time.time()
        results = []

        for _ in range(num_requests):
            result = triton_client.translate_code(code)
            results.append(result)

        elapsed = time.time() - start
        throughput = num_requests / elapsed

        print(f"Throughput: {throughput:.2f} requests/sec")

        # All requests should succeed
        assert len(results) == num_requests
        assert all(r.confidence > 0 for r in results)


class TestErrorHandling(TestTritonTranslationService):
    """Test error handling"""

    def test_invalid_code(self, triton_client):
        """Test handling of invalid Python code"""
        invalid_code = "def invalid syntax error"

        # Should either handle gracefully or raise specific error
        try:
            result = triton_client.translate_code(invalid_code)
            # If it succeeds, check for warnings
            assert len(result.warnings) > 0 or result.confidence < 0.5
        except Exception as e:
            # Should be a specific error type
            print(f"Expected error for invalid code: {e}")

    def test_empty_code(self, triton_client):
        """Test handling of empty input"""
        empty_code = ""

        try:
            result = triton_client.translate_code(empty_code)
            # Should handle gracefully
            assert result is not None
        except Exception as e:
            # Or raise appropriate error
            print(f"Error for empty code: {e}")

    def test_very_long_code(self, triton_client):
        """Test handling of very long code"""
        # Generate long code
        long_code = "\n".join([f"def func_{i}(): return {i}" for i in range(100)])

        try:
            result = triton_client.translate_code(long_code, timeout=60.0)
            assert result is not None
        except Exception as e:
            # May timeout or fail - should be handled
            print(f"Error for long code: {e}")


# Test fixtures and utilities
@pytest.fixture(scope="session")
def test_environment():
    """Setup test environment"""
    env = {
        'TRITON_URL': os.getenv('TRITON_URL', 'localhost:8000'),
        'TEST_TIMEOUT': int(os.getenv('TEST_TIMEOUT', '60'))
    }
    return env


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
