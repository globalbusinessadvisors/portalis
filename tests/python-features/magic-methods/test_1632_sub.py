"""
Feature: 16.3.2 __sub__
Category: Magic Methods
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
def __sub__(self, other):
    return Point(self.x - other.x, self.y - other.y)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_1632_sub():
    """Test translation of 16.3.2 __sub__."""
    pytest.skip("Feature not yet implemented")
