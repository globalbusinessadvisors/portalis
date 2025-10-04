"""
Feature: 9.1.1 Basic With
Category: Context Managers
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
with open("file.txt") as f:
    data = f.read()
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_911_basic_with():
    """Test translation of 9.1.1 Basic With."""
    pytest.skip("Feature not yet implemented")
