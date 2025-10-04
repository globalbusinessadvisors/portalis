"""
Triton Python Backend for Rust Transpiler

Integrates Rust transpiler agents with Triton Inference Server.
Provides batched translation with GPU acceleration.
"""

import json
import numpy as np
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any
import triton_python_backend_utils as pb_utils

logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Python backend model for Rust transpiler integration"""

    def initialize(self, args: Dict[str, str]):
        """
        Initialize model.

        Args:
            args: Model initialization arguments
        """
        self.model_config = json.loads(args['model_config'])

        # Get configuration parameters
        params = self.model_config.get('parameters', {})

        self.rust_cli = params.get('EXECUTION_ENV_PATH', {}).get('string_value', '/rust-bin/portalis-cli')
        self.enable_cuda = params.get('ENABLE_CUDA', {}).get('string_value', 'true').lower() == 'true'
        self.rust_workers = int(params.get('RUST_WORKERS', {}).get('string_value', '8'))
        self.cache_size_mb = int(params.get('CACHE_SIZE_MB', {}).get('string_value', '2048'))

        # Verify CLI exists
        if not Path(self.rust_cli).exists():
            logger.error(f"Rust CLI not found at {self.rust_cli}")
            raise RuntimeError(f"Rust CLI not found at {self.rust_cli}")

        logger.info(f"Rust transpiler initialized:")
        logger.info(f"  CLI: {self.rust_cli}")
        logger.info(f"  CUDA: {self.enable_cuda}")
        logger.info(f"  Workers: {self.rust_workers}")
        logger.info(f"  Cache: {self.cache_size_mb}MB")

        # Initialize cache
        self.translation_cache: Dict[str, Dict[str, Any]] = {}

    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """
        Execute batch of translation requests.

        Args:
            requests: Batch of inference requests

        Returns:
            Batch of inference responses
        """
        responses = []

        for request in requests:
            try:
                # Parse inputs
                python_code_tensor = pb_utils.get_input_tensor_by_name(request, "python_code")
                python_code = python_code_tensor.as_numpy()[0].decode('utf-8')

                # Get optional translation options
                options_tensor = pb_utils.get_input_tensor_by_name(request, "translation_options")
                if options_tensor is not None:
                    options_str = options_tensor.as_numpy()[0].decode('utf-8')
                    options = json.loads(options_str) if options_str else {}
                else:
                    options = {}

                # Check cache
                cache_key = self._get_cache_key(python_code, options)
                if cache_key in self.translation_cache:
                    logger.debug("Cache hit for translation")
                    result = self.translation_cache[cache_key]
                else:
                    # Translate using Rust CLI
                    result = self._translate_with_rust(python_code, options)

                    # Cache result
                    self.translation_cache[cache_key] = result

                # Create response tensors
                response = self._create_response(result)
                responses.append(response)

            except Exception as e:
                logger.error(f"Translation failed: {e}", exc_info=True)
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Translation failed: {str(e)}")
                )
                responses.append(error_response)

        return responses

    def _translate_with_rust(self, python_code: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate Python code using Rust CLI.

        Args:
            python_code: Python source code
            options: Translation options

        Returns:
            Translation result
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_file = tmpdir_path / "input.py"
            output_file = tmpdir_path / "output.rs"
            metadata_file = tmpdir_path / "metadata.json"

            # Write input
            input_file.write_text(python_code)

            # Build command
            cmd = [
                self.rust_cli,
                "transpile",
                str(input_file),
                "--output", str(output_file),
                "--metadata", str(metadata_file),
                "--mode", options.get("mode", "balanced"),
                "--optimization-level", str(options.get("optimization_level", 2)),
            ]

            if self.enable_cuda:
                cmd.append("--enable-cuda")

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False
            )

            if result.returncode != 0:
                logger.error(f"Rust CLI failed: {result.stderr}")
                raise RuntimeError(f"Translation failed: {result.stderr}")

            # Read outputs
            rust_code = output_file.read_text() if output_file.exists() else ""

            metadata = {}
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())

            return {
                "rust_code": rust_code,
                "confidence": metadata.get("confidence", 0.95),
                "warnings": metadata.get("warnings", []),
                "suggestions": metadata.get("suggestions", []),
                "metadata": metadata
            }

    def _create_response(self, result: Dict[str, Any]) -> pb_utils.InferenceResponse:
        """
        Create Triton response from translation result.

        Args:
            result: Translation result

        Returns:
            Triton inference response
        """
        # Rust code output
        rust_code_tensor = pb_utils.Tensor(
            "rust_code",
            np.array([result["rust_code"].encode('utf-8')], dtype=object)
        )

        # Confidence score
        confidence_tensor = pb_utils.Tensor(
            "confidence_score",
            np.array([result["confidence"]], dtype=np.float32)
        )

        # Metadata as JSON string
        metadata_tensor = pb_utils.Tensor(
            "metadata",
            np.array([json.dumps(result["metadata"]).encode('utf-8')], dtype=object)
        )

        # Warnings
        warnings_str = json.dumps(result.get("warnings", []))
        warnings_tensor = pb_utils.Tensor(
            "warnings",
            np.array([warnings_str.encode('utf-8')], dtype=object)
        )

        # Suggestions
        suggestions_str = json.dumps(result.get("suggestions", []))
        suggestions_tensor = pb_utils.Tensor(
            "suggestions",
            np.array([suggestions_str.encode('utf-8')], dtype=object)
        )

        return pb_utils.InferenceResponse(
            output_tensors=[
                rust_code_tensor,
                confidence_tensor,
                metadata_tensor,
                warnings_tensor,
                suggestions_tensor
            ]
        )

    def _get_cache_key(self, python_code: str, options: Dict[str, Any]) -> str:
        """Generate cache key for translation"""
        import hashlib

        content = python_code + json.dumps(options, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def finalize(self):
        """Cleanup resources"""
        logger.info("Rust transpiler model finalized")
        self.translation_cache.clear()
