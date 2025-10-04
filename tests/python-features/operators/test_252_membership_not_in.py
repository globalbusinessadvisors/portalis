"""
Feature: 2.5.2 Membership (not in)
Category: Operators
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
result = 4 not in [1, 2, 3]  # True
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_252_membership_not_in():
    """Test translation of 2.5.2 Membership (not in)."""
    pytest.skip("Feature not yet implemented")
