"""
Triton Client Library for Portalis Translation Services
Provides Python client for both REST and gRPC APIs
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
except ImportError:
    print("Warning: tritonclient not installed. Install with: pip install tritonclient[all]")
    httpclient = None
    grpcclient = None
    InferenceServerException = Exception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result from translation service"""
    rust_code: str
    confidence: float
    metadata: Dict[str, Any]
    warnings: List[str] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class BatchTranslationResult:
    """Result from batch translation"""
    translated_files: List[str]
    compilation_status: List[str]
    performance_metrics: Dict[str, Any]
    wasm_binaries: List[bytes]


class TritonTranslationClient:
    """
    Client for Triton Translation Services
    Supports both REST and gRPC protocols
    """

    def __init__(
        self,
        url: str = "localhost:8000",
        protocol: str = "http",
        verbose: bool = False
    ):
        """
        Initialize Triton client

        Args:
            url: Triton server URL (host:port)
            protocol: 'http' or 'grpc'
            verbose: Enable verbose logging
        """
        self.url = url
        self.protocol = protocol
        self.verbose = verbose

        # Initialize appropriate client
        if protocol == "http":
            if httpclient is None:
                raise ImportError("tritonclient[http] not installed")
            self.client = httpclient.InferenceServerClient(
                url=url,
                verbose=verbose
            )
        elif protocol == "grpc":
            if grpcclient is None:
                raise ImportError("tritonclient[grpc] not installed")
            self.client = grpcclient.InferenceServerClient(
                url=url,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        # Verify server is live
        self._verify_server()

        logger.info(f"Connected to Triton server at {url} using {protocol}")

    def _verify_server(self) -> None:
        """Verify server is live and responsive"""
        try:
            if not self.client.is_server_live():
                raise ConnectionError("Triton server is not live")
            if not self.client.is_server_ready():
                raise ConnectionError("Triton server is not ready")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Triton server: {e}")

    def translate_code(
        self,
        python_code: str,
        options: Optional[Dict[str, Any]] = None,
        model_name: str = "translation_model",
        timeout: float = 60.0
    ) -> TranslationResult:
        """
        Translate Python code to Rust

        Args:
            python_code: Source Python code
            options: Translation options (optional)
            model_name: Model to use for translation
            timeout: Request timeout in seconds

        Returns:
            TranslationResult with translated code and metadata
        """
        # Prepare inputs
        inputs = []

        # Python code input
        python_code_data = np.array([python_code.encode('utf-8')], dtype=object)
        inputs.append(self._create_input("python_code", python_code_data))

        # Optional translation options
        if options:
            options_data = np.array([json.dumps(options).encode('utf-8')], dtype=object)
            inputs.append(self._create_input("translation_options", options_data))

        # Define outputs
        outputs = [
            self._create_output("rust_code"),
            self._create_output("confidence_score"),
            self._create_output("metadata")
        ]

        # Execute inference
        try:
            result = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                timeout=timeout
            )

            # Parse results
            rust_code = result.as_numpy("rust_code")[0].decode('utf-8')
            confidence = float(result.as_numpy("confidence_score")[0])
            metadata_str = result.as_numpy("metadata")[0].decode('utf-8')
            metadata = json.loads(metadata_str)

            return TranslationResult(
                rust_code=rust_code,
                confidence=confidence,
                metadata=metadata,
                warnings=metadata.get('warnings', [])
            )

        except InferenceServerException as e:
            logger.error(f"Translation inference failed: {e}")
            raise

    def translate_interactive(
        self,
        code_snippet: str,
        target_language: str = "rust",
        context: Optional[List[str]] = None,
        model_name: str = "interactive_api",
        timeout: float = 30.0
    ) -> TranslationResult:
        """
        Interactive translation with context awareness

        Args:
            code_snippet: Code snippet to translate
            target_language: Target language
            context: Optional context from previous interactions
            model_name: Model name
            timeout: Request timeout

        Returns:
            TranslationResult with suggestions and warnings
        """
        inputs = []

        # Code snippet
        snippet_data = np.array([code_snippet.encode('utf-8')], dtype=object)
        inputs.append(self._create_input("code_snippet", snippet_data))

        # Target language
        lang_data = np.array([target_language.encode('utf-8')], dtype=object)
        inputs.append(self._create_input("target_language", lang_data))

        # Optional context
        if context:
            context_data = np.array([c.encode('utf-8') for c in context], dtype=object)
            inputs.append(self._create_input("context", context_data))

        # Define outputs
        outputs = [
            self._create_output("translated_code"),
            self._create_output("confidence"),
            self._create_output("suggestions"),
            self._create_output("warnings")
        ]

        try:
            result = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                timeout=timeout
            )

            # Parse results
            translated_code = result.as_numpy("translated_code")[0].decode('utf-8')
            confidence = float(result.as_numpy("confidence")[0])

            suggestions_data = result.as_numpy("suggestions")
            suggestions = [s.decode('utf-8') for s in suggestions_data]

            warnings_data = result.as_numpy("warnings")
            warnings = [w.decode('utf-8') for w in warnings_data]

            return TranslationResult(
                rust_code=translated_code,
                confidence=confidence,
                metadata={},
                warnings=warnings,
                suggestions=suggestions
            )

        except InferenceServerException as e:
            logger.error(f"Interactive translation failed: {e}")
            raise

    def translate_batch(
        self,
        source_files: List[str],
        project_config: Dict[str, Any],
        optimization_level: str = "release",
        model_name: str = "batch_processor",
        timeout: float = 600.0
    ) -> BatchTranslationResult:
        """
        Batch translation for entire projects

        Args:
            source_files: List of Python source files (as strings)
            project_config: Project configuration dictionary
            optimization_level: "debug" or "release"
            model_name: Batch processor model name
            timeout: Request timeout (longer for batch)

        Returns:
            BatchTranslationResult with all artifacts
        """
        inputs = []

        # Source files
        files_data = np.array([f.encode('utf-8') for f in source_files], dtype=object)
        inputs.append(self._create_input("source_files", files_data))

        # Project config
        config_data = np.array([json.dumps(project_config).encode('utf-8')], dtype=object)
        inputs.append(self._create_input("project_config", config_data))

        # Optimization level
        opt_data = np.array([optimization_level.encode('utf-8')], dtype=object)
        inputs.append(self._create_input("optimization_level", opt_data))

        # Define outputs
        outputs = [
            self._create_output("translated_files"),
            self._create_output("compilation_status"),
            self._create_output("performance_metrics"),
            self._create_output("wasm_binaries")
        ]

        try:
            result = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                timeout=timeout
            )

            # Parse results
            translated_data = result.as_numpy("translated_files")
            translated_files = [t.decode('utf-8') for t in translated_data]

            status_data = result.as_numpy("compilation_status")
            compilation_status = [s.decode('utf-8') for s in status_data]

            metrics_str = result.as_numpy("performance_metrics")[0].decode('utf-8')
            performance_metrics = json.loads(metrics_str)

            wasm_data = result.as_numpy("wasm_binaries")
            wasm_binaries = [w for w in wasm_data]

            return BatchTranslationResult(
                translated_files=translated_files,
                compilation_status=compilation_status,
                performance_metrics=performance_metrics,
                wasm_binaries=wasm_binaries
            )

        except InferenceServerException as e:
            logger.error(f"Batch translation failed: {e}")
            raise

    def _create_input(self, name: str, data: np.ndarray):
        """Create input tensor for request"""
        if self.protocol == "http":
            return httpclient.InferInput(name, data.shape, "BYTES")
        else:
            return grpcclient.InferInput(name, data.shape, "BYTES")

    def _create_output(self, name: str):
        """Create output tensor request"""
        if self.protocol == "http":
            return httpclient.InferRequestedOutput(name)
        else:
            return grpcclient.InferRequestedOutput(name)

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get metadata for a model"""
        try:
            metadata = self.client.get_model_metadata(model_name)
            return metadata
        except InferenceServerException as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a model"""
        try:
            config = self.client.get_model_config(model_name)
            return config
        except InferenceServerException as e:
            logger.error(f"Failed to get model config: {e}")
            raise

    def get_server_metadata(self) -> Dict[str, Any]:
        """Get server metadata"""
        try:
            metadata = self.client.get_server_metadata()
            return metadata
        except InferenceServerException as e:
            logger.error(f"Failed to get server metadata: {e}")
            raise

    def close(self) -> None:
        """Close client connection"""
        if hasattr(self.client, 'close'):
            self.client.close()


class AsyncTritonClient:
    """
    Async client for high-throughput applications
    Supports concurrent requests with asyncio
    """

    def __init__(
        self,
        url: str = "localhost:8001",
        verbose: bool = False
    ):
        """
        Initialize async gRPC client

        Args:
            url: Triton server gRPC URL
            verbose: Enable verbose logging
        """
        if grpcclient is None:
            raise ImportError("tritonclient[grpc] not installed")

        self.url = url
        self.verbose = verbose
        self.client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose
        )

        logger.info(f"Async client connected to {url}")

    async def translate_code_async(
        self,
        python_code: str,
        options: Optional[Dict[str, Any]] = None,
        model_name: str = "translation_model"
    ) -> TranslationResult:
        """
        Async translation (placeholder for full async implementation)

        Note: Full async requires async version of tritonclient
        This is a synchronous wrapper for demonstration
        """
        # In production, use aio-based triton client
        # For now, delegate to sync version
        sync_client = TritonTranslationClient(
            url=self.url.replace(':8001', ':8000'),
            protocol='http',
            verbose=self.verbose
        )

        result = sync_client.translate_code(
            python_code,
            options,
            model_name
        )

        sync_client.close()
        return result


# Convenience functions
def create_client(
    url: str = "localhost:8000",
    protocol: str = "http",
    verbose: bool = False
) -> TritonTranslationClient:
    """Create a Triton translation client"""
    return TritonTranslationClient(url, protocol, verbose)


def create_async_client(
    url: str = "localhost:8001",
    verbose: bool = False
) -> AsyncTritonClient:
    """Create an async Triton client"""
    return AsyncTritonClient(url, verbose)
