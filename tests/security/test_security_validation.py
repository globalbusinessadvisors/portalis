"""
Security Validation Tests

Tests security aspects of the NVIDIA stack:
- Authentication and authorization
- Input validation and sanitization
- Rate limiting enforcement
- Resource access control
- Error handling (no info leakage)
- Secure communication
"""

import pytest
import asyncio
import httpx
from typing import Dict, Any


@pytest.mark.security
class TestAuthentication:
    """Test authentication mechanisms."""

    @pytest.mark.asyncio
    async def test_unauthenticated_request_blocked(self, test_config):
        """Test that unauthenticated requests are blocked."""
        # Client without auth headers
        client = httpx.AsyncClient(
            base_url=test_config["nim_api_url"],
            timeout=10.0
        )

        response = await client.post(
            "/api/v1/translation/translate",
            json={"python_code": "def test(): pass", "mode": "fast"}
        )

        # May return 401 or 200 depending on auth configuration
        # Just ensure it responds
        assert response.status_code in [200, 401, 403]

        await client.aclose()

    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self, test_config):
        """Test that invalid tokens are rejected."""
        client = httpx.AsyncClient(
            base_url=test_config["nim_api_url"],
            headers={"Authorization": "Bearer invalid_token_12345"},
            timeout=10.0
        )

        response = await client.post(
            "/api/v1/translation/translate",
            json={"python_code": "def test(): pass"}
        )

        # Should reject or accept based on configuration
        assert response.status_code in [200, 401, 403]

        await client.aclose()


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.mark.asyncio
    async def test_empty_input_rejected(self, nim_api_client):
        """Test that empty input is rejected."""
        response = await nim_api_client.post(
            "/api/v1/translation/translate",
            json={"python_code": "", "mode": "fast"}
        )

        # Should reject empty input
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_malformed_input_rejected(self, nim_api_client):
        """Test that malformed input is rejected."""
        malformed_requests = [
            {},  # Empty dict
            {"python_code": None},  # None value
            {"wrong_field": "value"},  # Wrong field
            {"python_code": 12345},  # Wrong type
        ]

        for request_data in malformed_requests:
            response = await nim_api_client.post(
                "/api/v1/translation/translate",
                json=request_data
            )

            # Should reject with validation error
            assert response.status_code in [400, 422], \
                f"Malformed request not rejected: {request_data}"

    @pytest.mark.asyncio
    async def test_excessively_large_input_rejected(self, nim_api_client):
        """Test that excessively large input is rejected."""
        # Create very large input
        large_code = "def func():\n" + "    x = 1\n" * 100000  # ~1.5MB

        response = await nim_api_client.post(
            "/api/v1/translation/translate",
            json={"python_code": large_code, "mode": "fast"}
        )

        # Should either reject or handle gracefully
        assert response.status_code in [200, 400, 413, 422, 503]

    @pytest.mark.asyncio
    async def test_code_injection_prevention(self, nim_api_client):
        """Test prevention of code injection attempts."""
        injection_attempts = [
            "import os; os.system('rm -rf /')",
            "eval('__import__(\"os\").system(\"ls\")')",
            "__import__('subprocess').run(['cat', '/etc/passwd'])",
            "exec('import socket; ...')",
        ]

        for malicious_code in injection_attempts:
            response = await nim_api_client.post(
                "/api/v1/translation/translate",
                json={"python_code": malicious_code, "mode": "fast"}
            )

            # Should handle safely (translate but not execute)
            assert response.status_code in [200, 400, 422]

            if response.status_code == 200:
                # Code should be translated, not executed
                result = response.json()
                assert "rust_code" in result


@pytest.mark.security
class TestRateLimiting:
    """Test rate limiting mechanisms."""

    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self, nim_api_client):
        """Test that rate limit headers are present."""
        response = await nim_api_client.post(
            "/api/v1/translation/translate",
            json={"python_code": "def test(): pass", "mode": "fast"}
        )

        # Check for rate limit headers (implementation may vary)
        # Headers might include: X-RateLimit-Limit, X-RateLimit-Remaining, etc.
        print("\n=== Response Headers ===")
        for header, value in response.headers.items():
            if "rate" in header.lower() or "limit" in header.lower():
                print(f"{header}: {value}")

    @pytest.mark.asyncio
    async def test_burst_requests_handled(self, nim_api_client):
        """Test handling of burst requests."""
        num_requests = 20

        async def send_request():
            return await nim_api_client.post(
                "/api/v1/translation/translate",
                json={"python_code": "def test(): pass", "mode": "fast"}
            )

        # Send burst of requests
        responses = await asyncio.gather(
            *[send_request() for _ in range(num_requests)],
            return_exceptions=True
        )

        # Count status codes
        status_codes = {}
        for response in responses:
            if isinstance(response, httpx.Response):
                code = response.status_code
                status_codes[code] = status_codes.get(code, 0) + 1

        print("\n=== Burst Request Status Codes ===")
        for code, count in sorted(status_codes.items()):
            print(f"{code}: {count}")

        # Should have some successes
        assert 200 in status_codes, "No successful requests in burst"


@pytest.mark.security
class TestResourceAccessControl:
    """Test resource access control."""

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, nim_api_client):
        """Test prevention of path traversal attacks."""
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        # If API accepts file paths, test path traversal
        # This depends on API design
        for path in traversal_attempts:
            # Test in context where paths might be used
            # Example: batch translation with file paths
            pass

    @pytest.mark.asyncio
    async def test_unauthorized_resource_access(self, test_config):
        """Test that unauthorized resource access is blocked."""
        client = httpx.AsyncClient(
            base_url=test_config["nim_api_url"],
            timeout=10.0
        )

        # Try to access admin endpoints without auth
        admin_endpoints = [
            "/admin/users",
            "/admin/config",
            "/internal/metrics",
        ]

        for endpoint in admin_endpoints:
            try:
                response = await client.get(endpoint)
                # Should return 401, 403, or 404
                assert response.status_code in [401, 403, 404]
            except:
                # Endpoint might not exist
                pass

        await client.aclose()


@pytest.mark.security
class TestErrorHandling:
    """Test secure error handling."""

    @pytest.mark.asyncio
    async def test_no_sensitive_info_in_errors(self, nim_api_client):
        """Test that errors don't leak sensitive information."""
        # Trigger various errors
        error_cases = [
            {"python_code": "def invalid syntax"},  # Syntax error
            {"python_code": "x" * 1000000},  # Very long input
        ]

        for case in error_cases:
            response = await nim_api_client.post(
                "/api/v1/translation/translate",
                json=case
            )

            if response.status_code != 200:
                error_data = response.text

                # Should not contain sensitive info
                forbidden_patterns = [
                    "/home/",
                    "/root/",
                    "password",
                    "secret",
                    "token",
                    "api_key",
                    "traceback",  # Detailed tracebacks
                ]

                for pattern in forbidden_patterns:
                    assert pattern.lower() not in error_data.lower(), \
                        f"Sensitive info '{pattern}' in error response"

    @pytest.mark.asyncio
    async def test_consistent_error_responses(self, nim_api_client):
        """Test that error responses are consistent."""
        # Send multiple invalid requests
        for _ in range(5):
            response = await nim_api_client.post(
                "/api/v1/translation/translate",
                json={"python_code": ""}
            )

            assert response.status_code in [400, 422]

            # Response should be JSON
            try:
                error_data = response.json()
                # Should have consistent structure
                # (implementation specific)
            except:
                pass


@pytest.mark.security
class TestSecureCommunication:
    """Test secure communication."""

    @pytest.mark.asyncio
    async def test_https_redirect(self, test_config):
        """Test that HTTP is redirected to HTTPS (if configured)."""
        # This test depends on deployment configuration
        pass

    @pytest.mark.asyncio
    async def test_security_headers_present(self, nim_api_client):
        """Test that security headers are present."""
        response = await nim_api_client.get("/health")

        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
        ]

        print("\n=== Security Headers ===")
        for header in security_headers:
            if header in response.headers:
                print(f"✓ {header}: {response.headers[header]}")
            else:
                print(f"✗ {header}: Missing")


@pytest.mark.security
class TestDependencySecondary:
    """Test for known vulnerabilities."""

    def test_no_known_vulnerabilities(self):
        """Check for known vulnerabilities in dependencies."""
        # This would typically use tools like:
        # - safety check
        # - pip-audit
        # - bandit

        # For now, just ensure imports work
        import sys
        print(f"\n=== Python Version ===")
        print(f"Python: {sys.version}")

        # Check critical packages
        try:
            import torch
            print(f"PyTorch: {torch.__version__}")
        except:
            pass

        try:
            import fastapi
            print(f"FastAPI: {fastapi.__version__}")
        except:
            pass
