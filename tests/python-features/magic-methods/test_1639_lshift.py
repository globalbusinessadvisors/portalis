"""
Feature: 16.3.9 __lshift__
Category: Magic Methods
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
def __lshift__(self, other):
    return self.value << other
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_1639_lshift():
    """Test translation of 16.3.9 __lshift__."""
    pytest.skip("Feature not yet implemented")
