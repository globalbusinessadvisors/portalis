"""
Feature: 2.8.8 Negative Indexing
Category: Operators
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
last = lst[-1]
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_288_negative_indexing():
    """Test translation of 2.8.8 Negative Indexing."""
    pytest.skip("Feature not yet implemented")
