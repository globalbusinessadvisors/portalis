"""
Rust Transpiler Integration Module

Provides Python bindings and wrappers for the Rust transpiler agents.
Enables seamless integration between FastAPI and Rust-native translation services.
"""

import asyncio
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


@dataclass
class RustTranslationRequest:
    """Request for Rust transpiler"""
    python_code: str
    mode: str = "balanced"
    enable_cuda: bool = True
    optimization_level: int = 2
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RustTranslationResult:
    """Result from Rust transpiler"""
    rust_code: str
    confidence: float
    warnings: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]
    processing_time_ms: float
    compilation_status: Optional[str] = None
    wasm_binary: Optional[bytes] = None


class RustTranspilerClient:
    """
    Client for interacting with Rust transpiler agents.

    Supports both CLI-based invocation and native library integration
    (if Rust agents are compiled as shared libraries).
    """

    def __init__(
        self,
        cli_path: str = "/rust-bin/portalis-cli",
        max_workers: int = 4,
        timeout: int = 300
    ):
        """
        Initialize Rust transpiler client.

        Args:
            cli_path: Path to Rust CLI binary
            max_workers: Maximum concurrent workers
            timeout: Request timeout in seconds
        """
        self.cli_path = Path(cli_path)
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Verify CLI exists
        if not self.cli_path.exists():
            logger.warning(f"Rust CLI not found at {cli_path}, will use fallback mode")
            self.cli_available = False
        else:
            self.cli_available = True
            logger.info(f"Rust CLI available at {cli_path}")

    async def translate_code(
        self,
        request: RustTranslationRequest
    ) -> RustTranslationResult:
        """
        Translate Python code to Rust using native Rust agents.

        Args:
            request: Translation request

        Returns:
            Translation result with Rust code and metadata
        """
        start_time = time.time()

        if not self.cli_available:
            raise RuntimeError("Rust CLI not available")

        # Create temporary files for input/output
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_file = tmpdir_path / "input.py"
            output_file = tmpdir_path / "output.rs"
            metadata_file = tmpdir_path / "metadata.json"

            # Write Python code to input file
            input_file.write_text(request.python_code)

            # Build CLI command
            cmd = [
                str(self.cli_path),
                "transpile",
                str(input_file),
                "--output", str(output_file),
                "--metadata", str(metadata_file),
                "--mode", request.mode,
                "--optimization-level", str(request.optimization_level),
            ]

            if request.enable_cuda:
                cmd.append("--enable-cuda")

            if request.context:
                context_file = tmpdir_path / "context.json"
                context_file.write_text(json.dumps(request.context))
                cmd.extend(["--context", str(context_file)])

            # Execute transpiler
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._execute_cli,
                    cmd
                )

                # Read outputs
                rust_code = output_file.read_text() if output_file.exists() else ""

                metadata = {}
                if metadata_file.exists():
                    metadata = json.loads(metadata_file.read_text())

                # Parse result
                processing_time = (time.time() - start_time) * 1000

                return RustTranslationResult(
                    rust_code=rust_code,
                    confidence=metadata.get("confidence", 0.95),
                    warnings=metadata.get("warnings", []),
                    suggestions=metadata.get("suggestions", []),
                    metadata=metadata,
                    processing_time_ms=processing_time,
                    compilation_status=metadata.get("compilation_status", "success")
                )

            except subprocess.TimeoutExpired:
                logger.error(f"Rust transpiler timed out after {self.timeout}s")
                raise TimeoutError(f"Translation timed out after {self.timeout}s")

            except Exception as e:
                logger.error(f"Rust transpiler failed: {e}", exc_info=True)
                raise

    def _execute_cli(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """
        Execute CLI command synchronously.

        Args:
            cmd: Command to execute

        Returns:
            Completed process
        """
        logger.debug(f"Executing Rust CLI: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"Rust CLI failed with code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Rust transpiler failed: {result.stderr}")

        return result

    async def translate_batch(
        self,
        requests: List[RustTranslationRequest]
    ) -> List[RustTranslationResult]:
        """
        Translate multiple code snippets concurrently.

        Args:
            requests: List of translation requests

        Returns:
            List of translation results
        """
        tasks = [self.translate_code(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def compile_to_wasm(
        self,
        rust_code: str,
        optimization_level: int = 3
    ) -> bytes:
        """
        Compile Rust code to WASM binary.

        Args:
            rust_code: Rust source code
            optimization_level: Optimization level (0-3)

        Returns:
            WASM binary data
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rust_file = tmpdir_path / "lib.rs"
            wasm_file = tmpdir_path / "output.wasm"

            rust_file.write_text(rust_code)

            cmd = [
                str(self.cli_path),
                "compile-wasm",
                str(rust_file),
                "--output", str(wasm_file),
                "--optimization", str(optimization_level),
            ]

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._execute_cli, cmd)

            if not wasm_file.exists():
                raise RuntimeError("WASM compilation failed - output file not found")

            return wasm_file.read_bytes()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of Rust transpiler.

        Returns:
            Health status information
        """
        if not self.cli_available:
            return {
                "status": "degraded",
                "cli_available": False,
                "message": "Rust CLI not available"
            }

        try:
            cmd = [str(self.cli_path), "--version"]
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                subprocess.run,
                cmd,
                subprocess.PIPE,
                subprocess.PIPE,
                True,
                5  # 5 second timeout
            )

            version = result.stdout.decode().strip()

            return {
                "status": "healthy",
                "cli_available": True,
                "version": version,
                "cli_path": str(self.cli_path),
                "max_workers": self.max_workers
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "cli_available": False,
                "error": str(e)
            }

    def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)


# Global client instance
_rust_client: Optional[RustTranspilerClient] = None


def get_rust_transpiler() -> RustTranspilerClient:
    """
    Get or create global Rust transpiler client.

    Returns:
        Rust transpiler client instance
    """
    global _rust_client

    if _rust_client is None:
        from ..config.service_config import get_service_config
        config = get_service_config()

        _rust_client = RustTranspilerClient(
            cli_path=config.get("rust_cli_path", "/rust-bin/portalis-cli"),
            max_workers=config.get("rust_max_workers", 4),
            timeout=config.get("rust_timeout", 300)
        )

    return _rust_client


async def translate_with_rust(
    python_code: str,
    mode: str = "balanced",
    enable_cuda: bool = True,
    optimization_level: int = 2,
    context: Optional[Dict[str, Any]] = None
) -> RustTranslationResult:
    """
    Convenience function for Rust translation.

    Args:
        python_code: Python source code
        mode: Translation mode
        enable_cuda: Enable CUDA acceleration
        optimization_level: Optimization level
        context: Additional context

    Returns:
        Translation result
    """
    client = get_rust_transpiler()

    request = RustTranslationRequest(
        python_code=python_code,
        mode=mode,
        enable_cuda=enable_cuda,
        optimization_level=optimization_level,
        context=context
    )

    return await client.translate_code(request)
