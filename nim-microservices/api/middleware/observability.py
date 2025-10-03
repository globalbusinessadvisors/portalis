"""
Observability Middleware

Provides logging, metrics, and tracing for API requests.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge
import logging

logger = logging.getLogger(__name__)


# Prometheus metrics
REQUEST_COUNT = Counter(
    'nim_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'nim_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

REQUEST_IN_PROGRESS = Gauge(
    'nim_requests_in_progress',
    'Number of requests in progress',
    ['method', 'endpoint']
)

TRANSLATION_DURATION = Histogram(
    'nim_translation_duration_seconds',
    'Translation duration in seconds',
    ['mode'],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
)

TRANSLATION_COUNT = Counter(
    'nim_translations_total',
    'Total number of translations',
    ['mode', 'status']
)

GPU_MEMORY_USAGE = Gauge(
    'nim_gpu_memory_bytes',
    'GPU memory usage in bytes'
)

GPU_UTILIZATION = Gauge(
    'nim_gpu_utilization_percent',
    'GPU utilization percentage'
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request logging, metrics, and tracing.

    Adds:
    - Request ID tracking
    - Timing metrics
    - Prometheus metrics
    - Structured logging
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract endpoint info
        method = request.method
        path = request.url.path
        endpoint = self._normalize_path(path)

        # Start timing
        start_time = time.time()

        # Track in-progress requests
        REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()

        # Log request
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )

        # Process request
        response = None
        status_code = 500
        error = None

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            error = str(e)
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "error": error,
                },
                exc_info=True
            )
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Update metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).inc()

            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

            REQUEST_IN_PROGRESS.labels(
                method=method,
                endpoint=endpoint
            ).dec()

            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration * 1000,
                    "error": error,
                }
            )

        # Add request ID to response headers
        if response:
            response.headers["X-Request-ID"] = request_id

        return response

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path for metrics grouping"""
        # Remove trailing slash
        path = path.rstrip('/')

        # Replace UUIDs and IDs with placeholders
        import re
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{uuid}',
            path
        )
        path = re.sub(r'/\d+', '/{id}', path)

        return path or '/'


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Additional middleware for detailed request/response logging.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Log request body for debugging (be careful with sensitive data)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) < 10000:  # Only log small bodies
                    logger.debug(
                        f"Request body",
                        extra={
                            "request_id": getattr(request.state, 'request_id', None),
                            "body_length": len(body),
                        }
                    )
            except Exception:
                pass

        response = await call_next(request)
        return response


def record_translation_metrics(
    mode: str,
    duration: float,
    success: bool
) -> None:
    """
    Record translation-specific metrics.

    Args:
        mode: Translation mode
        duration: Translation duration in seconds
        success: Whether translation succeeded
    """
    TRANSLATION_DURATION.labels(mode=mode).observe(duration)
    TRANSLATION_COUNT.labels(
        mode=mode,
        status='success' if success else 'failure'
    ).inc()


def update_gpu_metrics(
    memory_bytes: float,
    utilization_percent: float
) -> None:
    """
    Update GPU metrics.

    Args:
        memory_bytes: GPU memory usage in bytes
        utilization_percent: GPU utilization percentage
    """
    GPU_MEMORY_USAGE.set(memory_bytes)
    GPU_UTILIZATION.set(utilization_percent)
