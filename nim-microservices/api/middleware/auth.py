"""
Authentication and Authorization Middleware

Provides API key validation and rate limiting.
"""

import time
from typing import Optional, Dict
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)

# Security scheme for API keys
security = HTTPBearer()


class RateLimiter:
    """
    Token bucket rate limiter.

    Implements per-client rate limiting with configurable limits.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens_per_second = requests_per_minute / 60.0

        # Storage: client_id -> (tokens, last_update)
        self.buckets: Dict[str, tuple[float, float]] = defaultdict(
            lambda: (self.burst_size, time.time())
        )

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.

        Args:
            client_id: Client identifier

        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        tokens, last_update = self.buckets[client_id]

        # Refill tokens based on time passed
        time_passed = now - last_update
        new_tokens = min(
            self.burst_size,
            tokens + time_passed * self.tokens_per_second
        )

        # Check if we have tokens
        if new_tokens >= 1.0:
            self.buckets[client_id] = (new_tokens - 1.0, now)
            return True
        else:
            self.buckets[client_id] = (new_tokens, now)
            return False

    def get_retry_after(self, client_id: str) -> int:
        """
        Get seconds until next request is allowed.

        Args:
            client_id: Client identifier

        Returns:
            Seconds until next request
        """
        tokens, _ = self.buckets[client_id]
        tokens_needed = 1.0 - tokens
        seconds = tokens_needed / self.tokens_per_second
        return max(1, int(seconds))


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API key authentication.

    Validates API keys and manages rate limiting.
    """

    def __init__(
        self,
        app,
        api_keys: Optional[Dict[str, str]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        exempt_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.api_keys = api_keys or {}
        self.rate_limiter = rate_limiter or RateLimiter()
        self.exempt_paths = exempt_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc"
        ]

    async def dispatch(self, request: Request, call_next):
        # Skip auth for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Extract API key
        api_key = self._extract_api_key(request)

        # Validate API key
        if not self._validate_api_key(api_key):
            logger.warning(
                f"Invalid API key",
                extra={
                    "path": request.url.path,
                    "client": request.client.host if request.client else None,
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Get client ID
        client_id = self._get_client_id(api_key)
        request.state.client_id = client_id

        # Check rate limit
        if not self.rate_limiter.is_allowed(client_id):
            retry_after = self.rate_limiter.get_retry_after(client_id)
            logger.warning(
                f"Rate limit exceeded",
                extra={
                    "client_id": client_id,
                    "path": request.url.path,
                }
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        # Process request
        response = await call_next(request)
        return response

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request"""
        # Try Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]

        # Try X-API-Key header
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            return api_key_header

        # Try query parameter (less secure, for testing only)
        api_key_param = request.query_params.get("api_key")
        if api_key_param:
            return api_key_param

        return None

    def _validate_api_key(self, api_key: Optional[str]) -> bool:
        """Validate API key"""
        if not api_key:
            return False

        # If no API keys configured, allow all (development mode)
        if not self.api_keys:
            return True

        # Check against configured keys
        return api_key in self.api_keys.values()

    def _get_client_id(self, api_key: str) -> str:
        """Get client ID from API key"""
        # Hash API key to get consistent client ID
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]


class CORSMiddleware(BaseHTTPMiddleware):
    """
    Simple CORS middleware for cross-origin requests.
    """

    def __init__(
        self,
        app,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]

    async def dispatch(self, request: Request, call_next):
        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._build_preflight_response(request)

        # Process request
        response = await call_next(request)

        # Add CORS headers
        origin = request.headers.get("origin")
        if origin and (
            "*" in self.allow_origins or origin in self.allow_origins
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response

    def _build_preflight_response(self, request: Request):
        """Build response for preflight OPTIONS request"""
        from fastapi import Response

        response = Response()
        origin = request.headers.get("origin")

        if origin and (
            "*" in self.allow_origins or origin in self.allow_origins
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = ", ".join(
                self.allow_methods
            )
            response.headers["Access-Control-Allow-Headers"] = ", ".join(
                self.allow_headers
            )
            response.headers["Access-Control-Max-Age"] = "600"

        return response
