"""
Portalis NIM Microservice - Main Application

FastAPI application for NVIDIA NIM-based translation service.
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware as FastAPICORS
from prometheus_client import make_asgi_app
import uvicorn

from .routes import translation_router, health_router
from .middleware import (
    ObservabilityMiddleware,
    AuthenticationMiddleware,
    RateLimiter,
)
from ..config import get_service_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks.
    """
    # Startup
    config = get_service_config()
    logger.info(f"Starting {config.service_name} v{config.service_version}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"CUDA enabled: {config.enable_cuda}")

    # Initialize services
    try:
        logger.info("Initializing services...")

        # Pre-load models if needed
        if config.environment == "production":
            logger.info("Pre-loading models for production...")
            # Models will be lazy-loaded on first request for now

        logger.info("Services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down service...")
    logger.info("Service shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    config = get_service_config()

    # Create FastAPI app
    app = FastAPI(
        title="Portalis NIM Microservice",
        description=(
            "NVIDIA NIM-based microservice for Python-to-Rust code translation. "
            "Provides enterprise-ready REST and gRPC APIs with auto-scaling, "
            "metrics, and observability."
        ),
        version=config.service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Add CORS middleware
    app.add_middleware(
        FastAPICORS,
        allow_origins=config.get_cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add observability middleware
    app.add_middleware(ObservabilityMiddleware)

    # Add authentication middleware (if enabled)
    if config.enable_auth:
        api_keys = config.get_api_keys_dict()
        rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_per_minute,
            burst_size=config.rate_limit_burst
        )
        app.add_middleware(
            AuthenticationMiddleware,
            api_keys=api_keys,
            rate_limiter=rate_limiter
        )
        logger.info("Authentication middleware enabled")
    else:
        logger.warning("Authentication middleware disabled - not recommended for production")

    # Include routers
    app.include_router(health_router)
    app.include_router(translation_router)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        logger.error(
            f"Unhandled exception",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "error": str(exc)
            },
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": request_id
            }
        )

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": config.service_name,
            "version": config.service_version,
            "status": "operational",
            "docs": "/docs",
            "health": "/health"
        }

    return app


def create_metrics_app() -> FastAPI:
    """
    Create separate app for Prometheus metrics.

    Returns:
        Metrics application
    """
    metrics_app = FastAPI(title="Metrics")

    # Mount Prometheus metrics
    metrics_asgi_app = make_asgi_app()
    metrics_app.mount("/metrics", metrics_asgi_app)

    @metrics_app.get("/")
    async def metrics_root():
        return {"status": "ok", "endpoints": ["/metrics"]}

    return metrics_app


# Create application instance
app = create_app()
metrics_app = create_metrics_app()


def main():
    """
    Main entry point for running the service.
    """
    config = get_service_config()

    logger.info(f"Starting server on {config.host}:{config.port}")
    logger.info(f"Workers: {config.workers}")
    logger.info(f"Log level: {config.log_level}")

    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level=config.log_level.lower(),
        access_log=True,
        reload=config.environment == "development"
    )


if __name__ == "__main__":
    main()
