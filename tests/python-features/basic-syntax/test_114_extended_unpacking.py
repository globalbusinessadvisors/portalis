"""
Feature: 1.1.4 Extended Unpacking
Category: Basic Syntax & Literals
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2,3,4], last=5
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_114_extended_unpacking():
    """Test translation of 1.1.4 Extended Unpacking."""
    pytest.skip("Feature not yet implemented")
