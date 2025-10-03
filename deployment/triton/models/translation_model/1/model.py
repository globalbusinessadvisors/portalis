"""
Triton Python Backend Model for Translation Service
Handles Python to Rust translation using NeMo models with GPU acceleration
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from typing import List, Dict, Any, Optional
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """
    Translation Model for Triton Inference Server
    Implements Python to Rust code translation using NeMo
    """

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the model, load NeMo, and setup GPU resources

        Args:
            args: Dictionary containing initialization parameters
        """
        logger.info("Initializing Translation Model...")

        # Parse model configuration
        self.model_config = json.loads(args['model_config'])

        # Get model parameters
        self.nemo_model_path = self._get_parameter(
            'NeMo_MODEL_PATH',
            '/models/nemo/translation_model'
        )

        self.execution_env = self._get_parameter(
            'EXECUTION_ENV_PATH',
            '/opt/tritonserver/backends/python/envs/translation_env'
        )

        # GPU configuration
        cuda_devices = self._get_parameter('CUDA_VISIBLE_DEVICES', '0')
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

        # Initialize NeMo translation engine
        self._init_nemo_engine()

        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'total_processing_time': 0.0,
            'avg_confidence': 0.0
        }

        logger.info(f"Model initialized successfully on GPUs: {cuda_devices}")

    def _get_parameter(self, key: str, default: str) -> str:
        """Extract parameter from model config"""
        params = self.model_config.get('parameters', {})
        for param in params:
            if param.get('key') == key:
                return param.get('value', {}).get('string_value', default)
        return default

    def _init_nemo_engine(self) -> None:
        """Initialize NeMo translation engine with GPU support"""
        try:
            # Import NeMo modules
            import torch
            from nemo.collections.nlp.models import TextTranslationModel

            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")

            # Load pre-trained translation model
            if os.path.exists(self.nemo_model_path):
                self.nemo_model = TextTranslationModel.restore_from(
                    self.nemo_model_path
                ).to(self.device)
                logger.info("NeMo model loaded from checkpoint")
            else:
                # Fallback to mock model for testing
                logger.warning("NeMo model not found, using mock translator")
                self.nemo_model = None

            # Warm up model
            if self.nemo_model:
                self._warmup_model()

        except Exception as e:
            logger.error(f"Error initializing NeMo engine: {e}")
            self.nemo_model = None

    def _warmup_model(self) -> None:
        """Warm up the model with sample inputs"""
        logger.info("Warming up NeMo model...")
        sample_code = "def hello_world():\n    print('Hello, World!')"
        try:
            _ = self._translate_code(sample_code, {})
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def execute(self, requests: List) -> List:
        """
        Execute translation for a batch of requests

        Args:
            requests: List of pb_utils.InferenceRequest

        Returns:
            List of pb_utils.InferenceResponse
        """
        responses = []
        batch_start = time.time()

        for request in requests:
            try:
                # Extract input tensors
                python_code_tensor = pb_utils.get_input_tensor_by_name(
                    request, "python_code"
                )
                python_code = python_code_tensor.as_numpy()[0].decode('utf-8')

                # Extract optional translation options
                options = {}
                try:
                    options_tensor = pb_utils.get_input_tensor_by_name(
                        request, "translation_options"
                    )
                    if options_tensor is not None:
                        options_str = options_tensor.as_numpy()[0].decode('utf-8')
                        options = json.loads(options_str)
                except:
                    pass  # Options are optional

                # Perform translation
                start_time = time.time()
                result = self._translate_code(python_code, options)
                processing_time = time.time() - start_time

                # Update metrics
                self._update_metrics(result, processing_time)

                # Create output tensors
                rust_code_tensor = pb_utils.Tensor(
                    "rust_code",
                    np.array([result['rust_code'].encode('utf-8')], dtype=object)
                )

                confidence_tensor = pb_utils.Tensor(
                    "confidence_score",
                    np.array([result['confidence']], dtype=np.float32)
                )

                metadata = {
                    'processing_time_ms': processing_time * 1000,
                    'warnings': result.get('warnings', []),
                    'translation_strategy': result.get('strategy', 'default'),
                    'model_version': '1.0'
                }

                metadata_tensor = pb_utils.Tensor(
                    "metadata",
                    np.array([json.dumps(metadata).encode('utf-8')], dtype=object)
                )

                # Create response
                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        rust_code_tensor,
                        confidence_tensor,
                        metadata_tensor
                    ]
                )

                responses.append(response)

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(e))
                )
                responses.append(error_response)

        batch_time = time.time() - batch_start
        logger.info(
            f"Processed batch of {len(requests)} requests in {batch_time:.3f}s"
        )

        return responses

    def _translate_code(
        self,
        python_code: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Translate Python code to Rust

        Args:
            python_code: Source Python code
            options: Translation options

        Returns:
            Dictionary with translated code and metadata
        """
        if self.nemo_model:
            # Use NeMo for translation
            return self._nemo_translate(python_code, options)
        else:
            # Fallback to rule-based translator for testing
            return self._fallback_translate(python_code, options)

    def _nemo_translate(
        self,
        python_code: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """NeMo-based translation"""
        import torch

        # Prepare input
        with torch.no_grad():
            # Tokenize and encode
            inputs = self._prepare_nemo_input(python_code, options)

            # Generate translation
            outputs = self.nemo_model.generate(
                inputs,
                max_length=options.get('max_length', 4096),
                num_beams=options.get('num_beams', 4),
                temperature=options.get('temperature', 0.7),
                top_p=options.get('top_p', 0.9)
            )

            # Decode output
            rust_code = self._decode_nemo_output(outputs)

            # Calculate confidence based on model scores
            confidence = self._calculate_confidence(outputs)

        return {
            'rust_code': rust_code,
            'confidence': confidence,
            'warnings': [],
            'strategy': 'nemo_generation'
        }

    def _prepare_nemo_input(
        self,
        python_code: str,
        options: Dict[str, Any]
    ) -> Any:
        """Prepare input for NeMo model"""
        # Add context and formatting
        prompt = f"Translate the following Python code to Rust:\n\n{python_code}\n\nRust code:"
        return self.nemo_model.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

    def _decode_nemo_output(self, outputs: Any) -> str:
        """Decode NeMo model output to Rust code"""
        decoded = self.nemo_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the Rust code part
        if "Rust code:" in decoded:
            rust_code = decoded.split("Rust code:")[-1].strip()
        else:
            rust_code = decoded
        return rust_code

    def _calculate_confidence(self, outputs: Any) -> float:
        """Calculate translation confidence score"""
        # Simplified confidence calculation
        # In production, use model probabilities
        return 0.85  # Placeholder

    def _fallback_translate(
        self,
        python_code: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simple rule-based translation for testing"""
        logger.warning("Using fallback translator")

        # Basic translation rules (for demo purposes)
        rust_code = python_code
        rust_code = rust_code.replace('def ', 'fn ')
        rust_code = rust_code.replace(':', ' {')
        rust_code = rust_code.replace('print(', 'println!(')
        rust_code = rust_code.replace("'", '"')

        # Add closing braces
        lines = rust_code.split('\n')
        rust_lines = []
        for line in lines:
            rust_lines.append(line)
            if line.strip().endswith('{'):
                # Count indentation to add matching }
                indent = len(line) - len(line.lstrip())
                rust_lines.append(' ' * indent + '}')

        return {
            'rust_code': '\n'.join(rust_lines),
            'confidence': 0.65,
            'warnings': ['Using fallback translator - results may be incomplete'],
            'strategy': 'rule_based_fallback'
        }

    def _update_metrics(self, result: Dict[str, Any], processing_time: float) -> None:
        """Update performance metrics"""
        self.metrics['total_requests'] += 1
        self.metrics['total_processing_time'] += processing_time

        if result['confidence'] > 0.5:
            self.metrics['successful_translations'] += 1
        else:
            self.metrics['failed_translations'] += 1

        # Update average confidence
        total = self.metrics['total_requests']
        old_avg = self.metrics['avg_confidence']
        self.metrics['avg_confidence'] = (
            (old_avg * (total - 1) + result['confidence']) / total
        )

    def finalize(self) -> None:
        """Cleanup resources on model unload"""
        logger.info("Finalizing Translation Model...")
        logger.info(f"Final metrics: {json.dumps(self.metrics, indent=2)}")

        # Cleanup NeMo resources
        if hasattr(self, 'nemo_model') and self.nemo_model:
            del self.nemo_model

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        logger.info("Model finalized successfully")
