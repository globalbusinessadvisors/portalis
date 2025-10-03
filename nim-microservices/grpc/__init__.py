"""gRPC Package"""

from .server import TranslationServicer, serve

__all__ = ["TranslationServicer", "serve"]
