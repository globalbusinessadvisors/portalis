"""
NeMo Model Service Wrapper

Provides a high-level interface to NVIDIA NeMo models for code translation.
Handles model loading, inference, batching, and GPU memory management.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
from loguru import logger
import torch
import numpy as np

try:
    from nemo.collections.nlp.models import TextGenerationModel
    from nemo.collections.nlp.modules.common.transformer import TransformerEmbedding
    HAS_NEMO = True
except ImportError:
    logger.warning("NeMo not available, using mock implementation")
    HAS_NEMO = False

from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class InferenceConfig:
    """Configuration for NeMo inference."""

    max_length: int = 512
    temperature: float = 0.2
    top_k: int = 5
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    num_beams: int = 1
    batch_size: int = 32
    use_gpu: bool = True
    gpu_id: int = 0


@dataclass
class TranslationResult:
    """Result from NeMo translation."""

    rust_code: str
    confidence: float
    alternatives: List[str]
    metadata: Dict[str, Any]
    processing_time_ms: float


class NeMoService:
    """
    Service wrapper for NVIDIA NeMo language models.

    Provides:
    - Model loading and initialization
    - Batch inference with GPU acceleration
    - Embedding generation for code similarity
    - Model state management
    - Error handling and fallbacks
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[InferenceConfig] = None,
        enable_cuda: bool = True
    ):
        """
        Initialize NeMo service.

        Args:
            model_path: Path to NeMo model checkpoint or model name
            config: Inference configuration
            enable_cuda: Enable CUDA acceleration
        """
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.config = config or InferenceConfig()
        self.enable_cuda = enable_cuda and torch.cuda.is_available()

        self.model: Optional[Any] = None
        self.device = torch.device(f"cuda:{self.config.gpu_id}" if self.enable_cuda else "cpu")

        self._initialized = False
        self._model_lock = asyncio.Lock()

        logger.info(
            f"NeMo service initialized - Device: {self.device}, "
            f"Model: {self.model_path}"
        )

    def initialize(self) -> None:
        """Load and initialize NeMo model."""
        if self._initialized:
            logger.warning("NeMo service already initialized")
            return

        logger.info(f"Loading NeMo model from {self.model_path}")

        try:
            if HAS_NEMO:
                # Load NeMo model
                self.model = TextGenerationModel.restore_from(
                    str(self.model_path),
                    map_location=self.device
                )
                self.model.eval()

                if self.enable_cuda:
                    self.model = self.model.to(self.device)

                logger.info("NeMo model loaded successfully")
            else:
                logger.warning("Using mock NeMo implementation")
                self.model = MockNeMoModel()

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load NeMo model: {e}")
            raise RuntimeError(f"NeMo model initialization failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def translate_code(
        self,
        python_code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TranslationResult:
        """
        Translate Python code to Rust using NeMo model.

        Args:
            python_code: Python source code to translate
            context: Additional context (imports, type hints, etc.)

        Returns:
            TranslationResult with Rust code and metadata
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.perf_counter()

        # Build prompt with context
        prompt = self._build_translation_prompt(python_code, context or {})

        # Generate Rust code
        try:
            outputs = self._generate(
                prompts=[prompt],
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                num_return_sequences=self.config.num_beams
            )

            # Parse and rank outputs
            rust_code = outputs[0] if outputs else ""
            alternatives = outputs[1:] if len(outputs) > 1 else []

            # Compute confidence score
            confidence = self._compute_confidence(rust_code, python_code)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return TranslationResult(
                rust_code=rust_code,
                confidence=confidence,
                alternatives=alternatives,
                metadata={
                    "model": str(self.model_path),
                    "prompt_length": len(prompt),
                    "output_length": len(rust_code),
                },
                processing_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise RuntimeError(f"NeMo translation error: {e}") from e

    def batch_translate(
        self,
        python_codes: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[TranslationResult]:
        """
        Translate multiple Python code snippets in batch.

        Args:
            python_codes: List of Python source codes
            contexts: List of contexts for each code snippet

        Returns:
            List of TranslationResult objects
        """
        if not self._initialized:
            self.initialize()

        contexts = contexts or [{} for _ in python_codes]

        # Build prompts
        prompts = [
            self._build_translation_prompt(code, ctx)
            for code, ctx in zip(python_codes, contexts)
        ]

        # Batch inference
        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_codes = python_codes[i:i+batch_size]

            # Generate for batch
            outputs = self._generate(
                prompts=batch_prompts,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p
            )

            # Create results
            for j, (rust_code, python_code) in enumerate(zip(outputs, batch_codes)):
                confidence = self._compute_confidence(rust_code, python_code)
                results.append(
                    TranslationResult(
                        rust_code=rust_code,
                        confidence=confidence,
                        alternatives=[],
                        metadata={"batch_index": i + j},
                        processing_time_ms=0.0  # Batch processing
                    )
                )

        return results

    def generate_embeddings(
        self,
        code_snippets: List[str]
    ) -> np.ndarray:
        """
        Generate embeddings for code snippets.

        Args:
            code_snippets: List of code strings

        Returns:
            Numpy array of shape (num_snippets, embedding_dim)
        """
        if not self._initialized:
            self.initialize()

        # Tokenize and encode
        embeddings = []

        for snippet in code_snippets:
            # Use model's encoder to get embeddings
            if HAS_NEMO and hasattr(self.model, 'encode'):
                embedding = self.model.encode(snippet)
            else:
                # Fallback: use simple hash-based embedding
                embedding = self._mock_embedding(snippet)

            embeddings.append(embedding)

        return np.array(embeddings)

    def _build_translation_prompt(
        self,
        python_code: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for translation task."""
        # Include few-shot examples if provided
        examples = context.get('examples', [])
        example_text = ""

        if examples:
            for ex in examples[:3]:  # Limit to 3 examples
                example_text += f"\n\nPython:\n{ex['python']}\n\nRust:\n{ex['rust']}"

        prompt = f"""Translate the following Python code to idiomatic Rust.
Preserve semantics, handle errors with Result types, and maintain performance.

{example_text}

Python:
{python_code}

Rust:"""

        return prompt

    def _generate(
        self,
        prompts: List[str],
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate text using NeMo model."""
        if HAS_NEMO and self.model:
            # Use NeMo model for generation
            outputs = self.model.generate(
                inputs=prompts,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences
            )
            return outputs
        else:
            # Mock implementation
            return [self._mock_translate(p) for p in prompts]

    def _compute_confidence(self, rust_code: str, python_code: str) -> float:
        """Compute confidence score for translation."""
        # Simple heuristic based on code length ratio and syntax validity
        if not rust_code or not python_code:
            return 0.0

        # Length ratio check (Rust typically 1.2-1.8x longer than Python)
        length_ratio = len(rust_code) / len(python_code)
        length_score = 1.0 if 0.8 <= length_ratio <= 2.5 else 0.5

        # Syntax check (basic heuristics)
        has_fn = 'fn ' in rust_code
        has_braces = '{' in rust_code and '}' in rust_code
        syntax_score = 1.0 if (has_fn or has_braces) else 0.3

        # Combined confidence
        confidence = (length_score + syntax_score) / 2.0

        return min(1.0, confidence)

    def _mock_embedding(self, text: str) -> np.ndarray:
        """Generate mock embedding for testing."""
        # Simple hash-based embedding
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(768)  # Standard embedding dimension

    def _mock_translate(self, prompt: str) -> str:
        """Mock translation for testing without NeMo."""
        # Extract Python code from prompt
        if "Python:" in prompt and "Rust:" in prompt:
            python_start = prompt.rfind("Python:")
            python_code = prompt[python_start:].replace("Python:", "").strip()

            # Simple mock translation
            return f"""fn mock_translation() -> Result<i64, Error> {{
    // Generated from: {python_code[:50]}...
    Ok(42)
}}"""

        return "fn default() -> i32 { 0 }"

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.model and self.enable_cuda:
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("NeMo service cleaned up")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class MockNeMoModel:
    """Mock NeMo model for testing without GPU."""

    def eval(self):
        return self

    def to(self, device):
        return self

    def generate(self, inputs, **kwargs):
        return [f"fn mock_{i}() -> i32 {{ 0 }}" for i in range(len(inputs))]

    def encode(self, text):
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(768)
