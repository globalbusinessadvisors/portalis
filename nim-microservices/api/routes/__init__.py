"""API Routes Package"""

from .translation import router as translation_router
from .health import router as health_router

__all__ = ["translation_router", "health_router"]
