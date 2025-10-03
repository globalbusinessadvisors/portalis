"""
Translation API Routes

REST endpoints for code translation services.
"""

import asyncio
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
import logging

from ..models import (
    TranslationRequest,
    TranslationResponse,
    BatchTranslationRequest,
    BatchTranslationResponse,
    StreamingChunk,
    ErrorResponse,
)
from ..middleware import record_translation_metrics
from ...config.service_config import get_service_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/translation", tags=["translation"])


# Service instances (initialized on startup)
nemo_service = None
triton_client = None


def get_nemo_service():
    """Get NeMo service instance"""
    global nemo_service
    if nemo_service is None:
        from nemo_integration.src.translation.nemo_service import NeMoService
        config = get_service_config()
        nemo_service = NeMoService(
            model_path=config.model_path,
            enable_cuda=config.enable_cuda
        )
        nemo_service.initialize()
    return nemo_service


def get_triton_client():
    """Get Triton client instance"""
    global triton_client
    if triton_client is None:
        from deployment.triton.configs.triton_client import TritonTranslationClient
        config = get_service_config()
        triton_client = TritonTranslationClient(
            url=config.triton_url,
            protocol=config.triton_protocol
        )
    return triton_client


@router.post(
    "/translate",
    response_model=TranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate Python code to Rust",
    description="Translate a single Python code snippet to idiomatic Rust code",
)
async def translate_code(
    request: TranslationRequest,
    http_request: Request,
    background_tasks: BackgroundTasks
) -> TranslationResponse:
    """
    Translate Python code to Rust.

    Args:
        request: Translation request with Python code
        http_request: HTTP request context
        background_tasks: Background task handler

    Returns:
        TranslationResponse with Rust code and metadata
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', 'unknown')

    logger.info(
        f"Translation request received",
        extra={
            "request_id": request_id,
            "code_length": len(request.python_code),
            "mode": request.mode.value,
        }
    )

    try:
        # Get service based on mode
        if request.mode == "fast":
            # Use NeMo directly for fast mode
            service = get_nemo_service()
            result = service.translate_code(
                python_code=request.python_code,
                context=request.context
            )

            response = TranslationResponse(
                rust_code=result.rust_code,
                confidence=result.confidence,
                alternatives=result.alternatives if request.include_alternatives else None,
                metadata=result.metadata,
                warnings=[],
                suggestions=[],
                processing_time_ms=result.processing_time_ms
            )

        else:
            # Use Triton for standard/quality modes
            client = get_triton_client()
            options = {
                "temperature": request.temperature,
                "max_length": request.max_length,
                "mode": request.mode.value,
            }
            if request.context:
                options.update(request.context)

            result = client.translate_code(
                python_code=request.python_code,
                options=options
            )

            response = TranslationResponse(
                rust_code=result.rust_code,
                confidence=result.confidence,
                alternatives=None,
                metadata=result.metadata,
                warnings=result.warnings,
                suggestions=result.suggestions,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Record metrics in background
        background_tasks.add_task(
            record_translation_metrics,
            mode=request.mode.value,
            duration=time.time() - start_time,
            success=True
        )

        logger.info(
            f"Translation completed",
            extra={
                "request_id": request_id,
                "confidence": response.confidence,
                "output_length": len(response.rust_code),
            }
        )

        return response

    except Exception as e:
        logger.error(
            f"Translation failed",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )

        # Record failure metrics
        background_tasks.add_task(
            record_translation_metrics,
            mode=request.mode.value,
            duration=time.time() - start_time,
            success=False
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.post(
    "/translate/batch",
    response_model=BatchTranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch translate Python files to Rust",
    description="Translate multiple Python source files to Rust in a single request",
)
async def translate_batch(
    request: BatchTranslationRequest,
    http_request: Request,
    background_tasks: BackgroundTasks
) -> BatchTranslationResponse:
    """
    Batch translate Python files to Rust.

    Args:
        request: Batch translation request
        http_request: HTTP request context
        background_tasks: Background task handler

    Returns:
        BatchTranslationResponse with all translated files
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', 'unknown')

    logger.info(
        f"Batch translation request received",
        extra={
            "request_id": request_id,
            "file_count": len(request.source_files),
            "optimization": request.optimization_level.value,
        }
    )

    try:
        # Use Triton for batch processing
        client = get_triton_client()

        result = client.translate_batch(
            source_files=request.source_files,
            project_config=request.project_config,
            optimization_level=request.optimization_level.value
        )

        response = BatchTranslationResponse(
            translated_files=result.translated_files,
            compilation_status=result.compilation_status,
            performance_metrics=result.performance_metrics,
            wasm_binaries=result.wasm_binaries if request.compile_wasm else None,
            total_processing_time_ms=(time.time() - start_time) * 1000,
            success_count=len([s for s in result.compilation_status if "success" in s.lower()]),
            failure_count=len([s for s in result.compilation_status if "error" in s.lower()])
        )

        # Record metrics
        background_tasks.add_task(
            record_translation_metrics,
            mode="batch",
            duration=time.time() - start_time,
            success=response.failure_count == 0
        )

        logger.info(
            f"Batch translation completed",
            extra={
                "request_id": request_id,
                "success_count": response.success_count,
                "failure_count": response.failure_count,
            }
        )

        return response

    except Exception as e:
        logger.error(
            f"Batch translation failed",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch translation failed: {str(e)}"
        )


@router.post(
    "/translate/stream",
    summary="Stream translation results",
    description="Translate Python code with streaming response",
)
async def translate_stream(
    request: TranslationRequest,
    http_request: Request
):
    """
    Stream translation results as they are generated.

    Args:
        request: Translation request
        http_request: HTTP request context

    Returns:
        Streaming response with translation chunks
    """
    request_id = getattr(http_request.state, 'request_id', 'unknown')

    logger.info(
        f"Streaming translation request received",
        extra={"request_id": request_id}
    )

    async def generate_stream():
        """Generate streaming chunks"""
        try:
            # Get service
            service = get_nemo_service()

            # Send initial metadata
            yield json.dumps({
                "chunk_type": "metadata",
                "content": json.dumps({
                    "request_id": request_id,
                    "status": "started"
                }),
                "is_final": False
            }) + "\n"

            # Perform translation
            result = service.translate_code(
                python_code=request.python_code,
                context=request.context
            )

            # Stream code in chunks (simulate streaming for now)
            code = result.rust_code
            chunk_size = 100
            for i in range(0, len(code), chunk_size):
                chunk = code[i:i+chunk_size]
                yield json.dumps({
                    "chunk_type": "code",
                    "content": chunk,
                    "is_final": False
                }) + "\n"
                await asyncio.sleep(0.01)  # Small delay for realistic streaming

            # Send final chunk with metadata
            yield json.dumps({
                "chunk_type": "complete",
                "content": "",
                "is_final": True,
                "metadata": {
                    "confidence": result.confidence,
                    "processing_time_ms": result.processing_time_ms
                }
            }) + "\n"

        except Exception as e:
            logger.error(
                f"Streaming translation failed",
                extra={"request_id": request_id, "error": str(e)},
                exc_info=True
            )
            yield json.dumps({
                "chunk_type": "error",
                "content": str(e),
                "is_final": True
            }) + "\n"

    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson",
        headers={
            "X-Request-ID": request_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@router.get(
    "/models",
    summary="List available models",
    description="Get information about available translation models",
)
async def list_models():
    """
    List available translation models.

    Returns:
        List of model information
    """
    try:
        config = get_service_config()

        models = [
            {
                "name": "translation_model",
                "version": config.model_version,
                "framework": "NeMo + Triton",
                "capabilities": ["translation", "batch", "streaming"],
                "status": "ready"
            }
        ]

        return {"models": models}

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )
