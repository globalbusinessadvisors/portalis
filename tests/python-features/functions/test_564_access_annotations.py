"""
Feature: 5.6.4 Access __annotations__
Category: Functions
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
print(func.__annotations__)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_564_access_annotations():
    """Test translation of 5.6.4 Access __annotations__."""
    pytest.skip("Feature not yet implemented")
