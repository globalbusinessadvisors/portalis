"""
gRPC Server Implementation for Portalis Translation Service

High-performance gRPC server with bi-directional streaming support.
"""

import asyncio
import logging
import time
from typing import AsyncIterator
import grpc
from concurrent import futures

# Import generated protobuf code (will be generated from .proto file)
# For now, we'll create a mock implementation
# In production, run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. translation.proto

logger = logging.getLogger(__name__)


class TranslationServicer:
    """
    gRPC service implementation for translation.

    Implements all RPC methods defined in translation.proto.
    """

    def __init__(self, nemo_service=None, triton_client=None):
        """
        Initialize servicer.

        Args:
            nemo_service: NeMo service instance
            triton_client: Triton client instance
        """
        self.nemo_service = nemo_service
        self.triton_client = triton_client
        self.start_time = time.time()

        logger.info("Translation servicer initialized")

    async def TranslateCode(self, request, context):
        """
        Translate single Python code to Rust.

        Args:
            request: TranslateRequest
            context: gRPC context

        Returns:
            TranslateResponse
        """
        try:
            logger.info(
                f"gRPC TranslateCode request",
                extra={
                    "code_length": len(request.python_code),
                    "mode": request.mode
                }
            )

            start_time = time.time()

            # Use NeMo service for translation
            if self.nemo_service:
                result = self.nemo_service.translate_code(
                    python_code=request.python_code,
                    context=dict(request.context) if request.context else None
                )

                # Build response (mock structure - will be replaced with generated code)
                response = {
                    'rust_code': result.rust_code,
                    'confidence': result.confidence,
                    'alternatives': result.alternatives if request.include_alternatives else [],
                    'metadata': result.metadata,
                    'warnings': [],
                    'suggestions': [],
                    'processing_time_ms': (time.time() - start_time) * 1000
                }

                logger.info("Translation completed successfully")
                return response

            else:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Translation service not available")
                return {}

        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Translation failed: {str(e)}")
            return {}

    async def TranslateBatch(self, request, context):
        """
        Translate multiple files in batch.

        Args:
            request: BatchTranslateRequest
            context: gRPC context

        Returns:
            BatchTranslateResponse
        """
        try:
            logger.info(
                f"gRPC TranslateBatch request",
                extra={"file_count": len(request.source_files)}
            )

            start_time = time.time()

            # Convert to list for batch processing
            source_files = list(request.source_files.values())

            # Use Triton for batch processing
            if self.triton_client:
                result = self.triton_client.translate_batch(
                    source_files=source_files,
                    project_config=dict(request.project_config) if request.project_config else {},
                    optimization_level=request.optimization_level
                )

                response = {
                    'translated_files': dict(zip(request.source_files.keys(), result.translated_files)),
                    'compilation_status': {},
                    'performance_metrics': result.performance_metrics,
                    'wasm_binaries': {},
                    'total_processing_time_ms': (time.time() - start_time) * 1000,
                    'success_count': len([s for s in result.compilation_status if "success" in s.lower()]),
                    'failure_count': 0
                }

                return response

            else:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Batch translation service not available")
                return {}

        except Exception as e:
            logger.error(f"Batch translation failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Batch translation failed: {str(e)}")
            return {}

    async def TranslateStream(self, request, context) -> AsyncIterator:
        """
        Stream translation results.

        Args:
            request: TranslateRequest
            context: gRPC context

        Yields:
            TranslationChunk
        """
        try:
            logger.info("gRPC TranslateStream request")

            # Send metadata chunk
            yield {
                'chunk_type': 'metadata',
                'content': '{"status": "started"}',
                'is_final': False,
                'metadata': {}
            }

            # Perform translation
            if self.nemo_service:
                result = self.nemo_service.translate_code(
                    python_code=request.python_code,
                    context=dict(request.context) if request.context else None
                )

                # Stream code in chunks
                code = result.rust_code
                chunk_size = 100
                for i in range(0, len(code), chunk_size):
                    chunk = code[i:i+chunk_size]
                    yield {
                        'chunk_type': 'code',
                        'content': chunk,
                        'is_final': False,
                        'metadata': {}
                    }
                    await asyncio.sleep(0.01)  # Simulate streaming

                # Send final chunk
                yield {
                    'chunk_type': 'complete',
                    'content': '',
                    'is_final': True,
                    'metadata': {
                        'confidence': str(result.confidence),
                        'processing_time_ms': str(result.processing_time_ms)
                    }
                }

        except Exception as e:
            logger.error(f"Streaming translation failed: {e}", exc_info=True)
            yield {
                'chunk_type': 'error',
                'content': str(e),
                'is_final': True,
                'metadata': {}
            }

    async def TranslateInteractive(self, request_iterator, context) -> AsyncIterator:
        """
        Bidirectional streaming for interactive translation.

        Args:
            request_iterator: Stream of InteractiveRequest
            context: gRPC context

        Yields:
            InteractiveResponse
        """
        try:
            logger.info("gRPC TranslateInteractive started")

            async for request in request_iterator:
                request_type = request.request_type
                content = request.content

                logger.debug(f"Interactive request: {request_type}")

                if request_type == "translate":
                    # Perform translation
                    if self.nemo_service:
                        result = self.nemo_service.translate_code(
                            python_code=content,
                            context={}
                        )

                        yield {
                            'response_type': 'translation',
                            'content': result.rust_code,
                            'confidence': result.confidence,
                            'suggestions': []
                        }

                elif request_type == "refine":
                    # Refine existing translation
                    yield {
                        'response_type': 'refinement',
                        'content': f"// Refined version\n{content}",
                        'confidence': 0.95,
                        'suggestions': ["Consider using Result type", "Add error handling"]
                    }

                elif request_type == "explain":
                    # Explain translation
                    yield {
                        'response_type': 'explanation',
                        'content': "This code translates Python to idiomatic Rust...",
                        'confidence': 1.0,
                        'suggestions': []
                    }

        except Exception as e:
            logger.error(f"Interactive translation failed: {e}", exc_info=True)

    async def ListModels(self, request, context):
        """
        List available models.

        Args:
            request: ListModelsRequest
            context: gRPC context

        Returns:
            ListModelsResponse
        """
        try:
            models = [
                {
                    'name': 'translation_model',
                    'version': '1.0.0',
                    'framework': 'NeMo + Triton',
                    'capabilities': ['translation', 'batch', 'streaming'],
                    'status': 'ready',
                    'metadata': {}
                }
            ]

            return {'models': models}

        except Exception as e:
            logger.error(f"Failed to list models: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to list models: {str(e)}")
            return {}

    async def HealthCheck(self, request, context):
        """
        Health check.

        Args:
            request: HealthCheckRequest
            context: gRPC context

        Returns:
            HealthCheckResponse
        """
        try:
            # Check GPU availability
            gpu_available = False
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except ImportError:
                pass

            # Determine status
            status = "healthy" if self.nemo_service else "degraded"

            return {
                'status': status,
                'version': '1.0.0',
                'uptime_seconds': time.time() - self.start_time,
                'gpu_available': gpu_available,
                'model_loaded': self.nemo_service is not None,
                'dependencies': {
                    'nemo': 'available' if self.nemo_service else 'unavailable',
                    'triton': 'available' if self.triton_client else 'unavailable',
                    'cuda': 'available' if gpu_available else 'unavailable'
                }
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                'status': 'unhealthy',
                'version': '1.0.0',
                'uptime_seconds': 0,
                'gpu_available': False,
                'model_loaded': False,
                'dependencies': {}
            }


async def serve(
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
    nemo_service=None,
    triton_client=None
):
    """
    Start gRPC server.

    Args:
        host: Host to bind to
        port: Port to bind to
        max_workers: Maximum number of worker threads
        nemo_service: NeMo service instance
        triton_client: Triton client instance
    """
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
        ]
    )

    # Add servicer to server
    servicer = TranslationServicer(
        nemo_service=nemo_service,
        triton_client=triton_client
    )

    # Note: In production, use generated code:
    # translation_pb2_grpc.add_TranslationServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f'{host}:{port}')

    logger.info(f"Starting gRPC server on {host}:{port}")
    await server.start()

    logger.info("gRPC server started successfully")

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(grace=5)
        logger.info("gRPC server stopped")


def main():
    """Main entry point for gRPC server"""
    import sys
    sys.path.insert(0, '/workspace/portalis')

    # Initialize services
    nemo_service = None
    triton_client = None

    try:
        from nemo_integration.src.translation.nemo_service import NeMoService
        nemo_service = NeMoService(
            model_path="/models/nemo_translation.nemo",
            enable_cuda=True
        )
        nemo_service.initialize()
    except Exception as e:
        logger.warning(f"Failed to initialize NeMo service: {e}")

    try:
        from deployment.triton.configs.triton_client import TritonTranslationClient
        triton_client = TritonTranslationClient(
            url="localhost:8000",
            protocol="http"
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Triton client: {e}")

    # Run server
    asyncio.run(serve(
        host="0.0.0.0",
        port=50051,
        nemo_service=nemo_service,
        triton_client=triton_client
    ))


if __name__ == "__main__":
    main()
