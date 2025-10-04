"""
Feature: 16.3.10 __rshift__
Category: Magic Methods
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
def __rshift__(self, other):
    return self.value >> other
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_16310_rshift():
    """Test translation of 16.3.10 __rshift__."""
    pytest.skip("Feature not yet implemented")
