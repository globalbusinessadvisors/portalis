"""
Unit Tests for NeMo Service Collaboration (London School TDD)

Tests the interaction between NeMo service and its collaborators.
Focus on behavior and interaction patterns, not implementation details.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np

from nemo_integration.src.translation.nemo_service import (
    NeMoService,
    InferenceConfig,
    TranslationResult,
)


@pytest.fixture
def mock_nemo_model():
    """Mock NeMo model for testing collaborations."""
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model
    model.generate.return_value = ["fn test() -> i32 { 0 }"]
    model.encode.return_value = np.random.randn(768)
    return model


@pytest.fixture
def inference_config():
    """Test inference configuration."""
    return InferenceConfig(
        max_length=512,
        temperature=0.2,
        batch_size=8,
        use_gpu=False
    )


class TestNeMoServiceInitialization:
    """Test NeMo service initialization and setup."""

    def test_service_initializes_model_on_first_call(self, inference_config):
        """
        GIVEN a NeMo service that hasn't been initialized
        WHEN initialize is called
        THEN it should load the model
        """
        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', False):
            service = NeMoService(
                model_path="/mock/path/model.nemo",
                config=inference_config,
                enable_cuda=False
            )

            assert not service._initialized

            service.initialize()

            assert service._initialized
            assert service.model is not None

    def test_service_uses_cuda_device_when_available(self, inference_config):
        """
        GIVEN CUDA is available and enabled
        WHEN service is initialized
        THEN it should use CUDA device
        """
        with patch('nemo_integration.src.translation.nemo_service.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device = Mock()

            service = NeMoService(
                model_path="/mock/path/model.nemo",
                config=inference_config,
                enable_cuda=True
            )

            # Should check for CUDA availability
            mock_torch.cuda.is_available.assert_called()

    def test_service_does_not_reinitialize_when_already_initialized(self, inference_config):
        """
        GIVEN a service that is already initialized
        WHEN initialize is called again
        THEN it should not reload the model
        """
        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', False):
            service = NeMoService(
                model_path="/mock/path/model.nemo",
                config=inference_config,
                enable_cuda=False
            )

            service.initialize()
            first_model = service.model

            service.initialize()

            # Should be the same model instance
            assert service.model is first_model


class TestTranslationWorkflow:
    """Test the translation workflow and collaborations."""

    def test_translate_code_builds_prompt_from_input(self, inference_config, mock_nemo_model):
        """
        GIVEN Python code to translate
        WHEN translate_code is called
        THEN it should build a proper prompt
        """
        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', True), \
             patch('nemo_integration.src.translation.nemo_service.TextGenerationModel') as mock_model_class:

            mock_model_class.restore_from.return_value = mock_nemo_model

            service = NeMoService(
                model_path="/mock/model.nemo",
                config=inference_config,
                enable_cuda=False
            )
            service.initialize()

            python_code = "def add(a, b): return a + b"
            result = service.translate_code(python_code)

            # Should have called model.generate with a prompt
            mock_nemo_model.generate.assert_called_once()
            call_args = mock_nemo_model.generate.call_args
            prompts = call_args[1]['inputs']

            assert len(prompts) == 1
            assert python_code in prompts[0]
            assert "Rust" in prompts[0]  # Prompt should mention target language

    def test_translate_code_includes_context_in_prompt(self, inference_config, mock_nemo_model):
        """
        GIVEN additional context is provided
        WHEN translate_code is called
        THEN context should be included in the prompt
        """
        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', True), \
             patch('nemo_integration.src.translation.nemo_service.TextGenerationModel') as mock_model_class:

            mock_model_class.restore_from.return_value = mock_nemo_model

            service = NeMoService(
                model_path="/mock/model.nemo",
                config=inference_config,
                enable_cuda=False
            )
            service.initialize()

            context = {
                'examples': [
                    {
                        'python': 'def double(x): return x * 2',
                        'rust': 'fn double(x: i64) -> i64 { x * 2 }'
                    }
                ]
            }

            result = service.translate_code("def test(): pass", context=context)

            # Prompt should include examples from context
            call_args = mock_nemo_model.generate.call_args
            prompt = call_args[1]['inputs'][0]
            assert 'double' in prompt  # Example should be in prompt

    def test_translate_code_uses_configured_parameters(self, inference_config, mock_nemo_model):
        """
        GIVEN specific inference configuration
        WHEN translate_code is called
        THEN it should use configured parameters
        """
        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', True), \
             patch('nemo_integration.src.translation.nemo_service.TextGenerationModel') as mock_model_class:

            mock_model_class.restore_from.return_value = mock_nemo_model

            service = NeMoService(
                model_path="/mock/model.nemo",
                config=inference_config,
                enable_cuda=False
            )
            service.initialize()

            result = service.translate_code("def test(): pass")

            # Should use configured parameters
            call_args = mock_nemo_model.generate.call_args
            assert call_args[1]['max_length'] == inference_config.max_length
            assert call_args[1]['temperature'] == inference_config.temperature
            assert call_args[1]['top_k'] == inference_config.top_k

    def test_translate_code_computes_confidence_from_result(self, inference_config, mock_nemo_model):
        """
        GIVEN a translation result
        WHEN translate_code completes
        THEN it should compute confidence score
        """
        mock_nemo_model.generate.return_value = ["fn add(a: i64, b: i64) -> i64 { a + b }"]

        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', True), \
             patch('nemo_integration.src.translation.nemo_service.TextGenerationModel') as mock_model_class:

            mock_model_class.restore_from.return_value = mock_nemo_model

            service = NeMoService(
                model_path="/mock/model.nemo",
                config=inference_config,
                enable_cuda=False
            )
            service.initialize()

            result = service.translate_code("def add(a, b): return a + b")

            assert isinstance(result, TranslationResult)
            assert 0.0 <= result.confidence <= 1.0


class TestBatchProcessing:
    """Test batch translation workflow."""

    def test_batch_translate_processes_multiple_codes_in_batches(self, inference_config, mock_nemo_model):
        """
        GIVEN multiple Python code snippets
        WHEN batch_translate is called
        THEN it should process them in configured batch sizes
        """
        mock_nemo_model.generate.return_value = ["fn test() -> i32 { 0 }"] * 10

        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', True), \
             patch('nemo_integration.src.translation.nemo_service.TextGenerationModel') as mock_model_class:

            mock_model_class.restore_from.return_value = mock_nemo_model

            # Set batch size to 3
            config = InferenceConfig(batch_size=3, use_gpu=False)

            service = NeMoService(
                model_path="/mock/model.nemo",
                config=config,
                enable_cuda=False
            )
            service.initialize()

            # Process 10 codes (should result in 4 batches: 3, 3, 3, 1)
            codes = [f"def func{i}(): pass" for i in range(10)]
            results = service.batch_translate(codes)

            assert len(results) == 10
            # Should have called generate 4 times (10 codes / 3 batch size = 4 batches)
            assert mock_nemo_model.generate.call_count == 4

    def test_batch_translate_maintains_order(self, inference_config, mock_nemo_model):
        """
        GIVEN a batch of codes
        WHEN batch_translate is called
        THEN results should maintain the same order
        """
        # Return different outputs for verification
        outputs = [f"fn func{i}() -> i32 {{ {i} }}" for i in range(5)]
        mock_nemo_model.generate.side_effect = [[o] for o in outputs]

        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', True), \
             patch('nemo_integration.src.translation.nemo_service.TextGenerationModel') as mock_model_class:

            mock_model_class.restore_from.return_value = mock_nemo_model

            config = InferenceConfig(batch_size=1, use_gpu=False)
            service = NeMoService(
                model_path="/mock/model.nemo",
                config=config,
                enable_cuda=False
            )
            service.initialize()

            codes = [f"def func{i}(): pass" for i in range(5)]
            results = service.batch_translate(codes)

            # Results should be in same order
            for i, result in enumerate(results):
                assert f"{i}" in result.rust_code


class TestRetryBehavior:
    """Test retry behavior on failures."""

    def test_translate_code_retries_on_transient_failure(self, inference_config):
        """
        GIVEN a model that fails transiently
        WHEN translate_code is called
        THEN it should retry up to configured attempts
        """
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Fail twice, then succeed
        mock_model.generate.side_effect = [
            RuntimeError("Temporary error"),
            RuntimeError("Temporary error"),
            ["fn test() -> i32 { 0 }"]
        ]

        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', True), \
             patch('nemo_integration.src.translation.nemo_service.TextGenerationModel') as mock_model_class:

            mock_model_class.restore_from.return_value = mock_model

            service = NeMoService(
                model_path="/mock/model.nemo",
                config=inference_config,
                enable_cuda=False
            )
            service.initialize()

            result = service.translate_code("def test(): pass")

            # Should have retried and eventually succeeded
            assert mock_model.generate.call_count == 3
            assert isinstance(result, TranslationResult)


class TestResourceManagement:
    """Test resource cleanup and management."""

    def test_cleanup_clears_cuda_cache_when_gpu_enabled(self, inference_config):
        """
        GIVEN service is using GPU
        WHEN cleanup is called
        THEN it should clear CUDA cache
        """
        with patch('nemo_integration.src.translation.nemo_service.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = MagicMock()

            service = NeMoService(
                model_path="/mock/model.nemo",
                config=inference_config,
                enable_cuda=True
            )

            service.cleanup()

            # Should call empty_cache
            mock_torch.cuda.empty_cache.assert_called_once()

    def test_context_manager_initializes_and_cleans_up(self, inference_config):
        """
        GIVEN service used as context manager
        WHEN entering and exiting context
        THEN it should initialize and cleanup properly
        """
        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', False):
            with NeMoService(
                model_path="/mock/model.nemo",
                config=inference_config,
                enable_cuda=False
            ) as service:
                # Should be initialized on entry
                assert service._initialized

            # Should be cleaned up on exit
            assert not service._initialized


class TestEmbeddingGeneration:
    """Test code embedding generation."""

    def test_generate_embeddings_returns_correct_shape(self, inference_config, mock_nemo_model):
        """
        GIVEN multiple code snippets
        WHEN generate_embeddings is called
        THEN it should return embeddings with correct shape
        """
        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', True), \
             patch('nemo_integration.src.translation.nemo_service.TextGenerationModel') as mock_model_class:

            mock_model_class.restore_from.return_value = mock_nemo_model

            service = NeMoService(
                model_path="/mock/model.nemo",
                config=inference_config,
                enable_cuda=False
            )
            service.initialize()

            codes = ["def add(): pass", "def sub(): pass", "def mul(): pass"]
            embeddings = service.generate_embeddings(codes)

            # Should return array with shape (num_codes, embedding_dim)
            assert embeddings.shape[0] == len(codes)
            assert embeddings.shape[1] > 0  # Has embedding dimension

    def test_generate_embeddings_uses_model_encoder(self, inference_config, mock_nemo_model):
        """
        GIVEN a model with encode method
        WHEN generate_embeddings is called
        THEN it should use model's encoder
        """
        with patch('nemo_integration.src.translation.nemo_service.HAS_NEMO', True), \
             patch('nemo_integration.src.translation.nemo_service.TextGenerationModel') as mock_model_class:

            mock_model_class.restore_from.return_value = mock_nemo_model

            service = NeMoService(
                model_path="/mock/model.nemo",
                config=inference_config,
                enable_cuda=False
            )
            service.initialize()

            codes = ["def test(): pass"]
            embeddings = service.generate_embeddings(codes)

            # Should call model's encode method
            assert mock_nemo_model.encode.called
