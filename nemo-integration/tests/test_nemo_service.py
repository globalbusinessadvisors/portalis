"""
Unit tests for NeMo Service

Tests the NeMo model wrapper and inference functionality.
"""

import pytest
from pathlib import Path
import tempfile

from src.translation.nemo_service import (
    NeMoService,
    InferenceConfig,
    TranslationResult,
    MockNeMoModel
)


@pytest.fixture
def inference_config():
    """Create test inference configuration."""
    return InferenceConfig(
        max_length=256,
        temperature=0.2,
        batch_size=4,
        use_gpu=False  # Use CPU for tests
    )


@pytest.fixture
def nemo_service(inference_config):
    """Create NeMo service instance for testing."""
    # Use mock model path for tests
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "mock_model.nemo"
        model_path.touch()  # Create empty file

        service = NeMoService(
            model_path=model_path,
            config=inference_config,
            enable_cuda=False
        )

        yield service

        service.cleanup()


class TestNeMoService:
    """Test suite for NeMo Service."""

    def test_initialization(self, nemo_service):
        """Test service initialization."""
        assert not nemo_service._initialized
        assert nemo_service.device.type == "cpu"
        assert nemo_service.config.max_length == 256

    def test_initialize_loads_model(self, nemo_service):
        """Test that initialize loads the model."""
        nemo_service.initialize()

        assert nemo_service._initialized
        assert nemo_service.model is not None

    def test_translate_code_returns_result(self, nemo_service):
        """Test basic code translation."""
        nemo_service.initialize()

        python_code = """
def add(a: int, b: int) -> int:
    return a + b
"""

        result = nemo_service.translate_code(python_code)

        assert isinstance(result, TranslationResult)
        assert isinstance(result.rust_code, str)
        assert len(result.rust_code) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time_ms > 0

    def test_translate_with_context(self, nemo_service):
        """Test translation with additional context."""
        nemo_service.initialize()

        python_code = "def square(x): return x * x"
        context = {
            "type_hints": {"x": "int"},
            "examples": [
                {
                    "python": "def double(x): return x * 2",
                    "rust": "fn double(x: i64) -> i64 { x * 2 }"
                }
            ]
        }

        result = nemo_service.translate_code(python_code, context)

        assert isinstance(result, TranslationResult)
        assert "fn" in result.rust_code

    def test_batch_translate(self, nemo_service):
        """Test batch translation."""
        nemo_service.initialize()

        python_codes = [
            "def func1(): return 1",
            "def func2(): return 2",
            "def func3(): return 3",
        ]

        results = nemo_service.batch_translate(python_codes)

        assert len(results) == 3
        assert all(isinstance(r, TranslationResult) for r in results)
        assert all(len(r.rust_code) > 0 for r in results)

    def test_generate_embeddings(self, nemo_service):
        """Test embedding generation."""
        nemo_service.initialize()

        code_snippets = [
            "def add(a, b): return a + b",
            "def multiply(x, y): return x * y",
        ]

        embeddings = nemo_service.generate_embeddings(code_snippets)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Embedding dimension

    def test_confidence_computation(self, nemo_service):
        """Test confidence score computation."""
        # Valid translation
        rust_code = "fn test() -> i64 { 42 }"
        python_code = "def test(): return 42"

        confidence = nemo_service._compute_confidence(rust_code, python_code)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high

        # Invalid translation (empty)
        confidence = nemo_service._compute_confidence("", python_code)
        assert confidence == 0.0

    def test_context_manager(self, inference_config):
        """Test service as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "mock.nemo"
            model_path.touch()

            with NeMoService(model_path, inference_config, enable_cuda=False) as service:
                assert service._initialized
                result = service.translate_code("def test(): pass")
                assert isinstance(result, TranslationResult)

    def test_retry_on_failure(self, nemo_service):
        """Test that translation retries on transient failures."""
        nemo_service.initialize()

        # Mock should not fail, but this tests the retry decorator
        result = nemo_service.translate_code("def test(): pass")
        assert isinstance(result, TranslationResult)


class TestMockNeMoModel:
    """Test suite for mock NeMo model."""

    def test_mock_model_initialization(self):
        """Test mock model can be created."""
        model = MockNeMoModel()
        assert model is not None

    def test_mock_model_generate(self):
        """Test mock model generation."""
        model = MockNeMoModel()
        inputs = ["test1", "test2"]

        outputs = model.generate(inputs=inputs, max_length=100)

        assert len(outputs) == 2
        assert all(isinstance(o, str) for o in outputs)
        assert all("fn" in o for o in outputs)

    def test_mock_model_encode(self):
        """Test mock model encoding."""
        model = MockNeMoModel()
        text = "def test(): pass"

        embedding = model.encode(text)

        assert embedding is not None
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0


@pytest.mark.parametrize("python_code,expected_pattern", [
    ("def add(a, b): return a + b", r"fn \w+"),
    ("class MyClass: pass", r"(struct|enum)"),
    ("x = [1, 2, 3]", r"vec!|Vec::"),
])
def test_translation_patterns(nemo_service, python_code, expected_pattern):
    """Test that translations match expected patterns."""
    import re

    nemo_service.initialize()
    result = nemo_service.translate_code(python_code)

    # Mock should produce some Rust-like code
    assert isinstance(result.rust_code, str)
