"""
Feature: 11.4.1 Async With
Category: Async/Await
Complexity: Very High
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
async with async_context_manager() as resource:
    await use(resource)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("very high")
@pytest.mark.status("not_implemented")
def test_1141_async_with():
    """Test translation of 11.4.1 Async With."""
    pytest.skip("Feature not yet implemented")
