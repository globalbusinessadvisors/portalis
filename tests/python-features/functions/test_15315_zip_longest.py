"""
Feature: 15.3.15 zip() longest
Category: Functions
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
from itertools import zip_longest
result = zip_longest(list1, list2, fillvalue=None)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_15315_zip_longest():
    """Test translation of 15.3.15 zip() longest."""
    pytest.skip("Feature not yet implemented")
