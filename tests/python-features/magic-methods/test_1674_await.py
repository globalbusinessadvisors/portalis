"""
Feature: 16.7.4 __await__
Category: Magic Methods
Complexity: Very High
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
def __await__(self):
    return (yield from self._coro)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("very high")
@pytest.mark.status("not_implemented")
def test_1674_await():
    """Test translation of 16.7.4 __await__."""
    pytest.skip("Feature not yet implemented")
