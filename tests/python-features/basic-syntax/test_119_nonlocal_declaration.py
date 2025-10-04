"""
Feature: 1.1.9 Nonlocal Declaration
Category: Basic Syntax & Literals
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
def outer():
    x = 10
    def inner():
        nonlocal x
        x = 20
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_119_nonlocal_declaration():
    """Test translation of 1.1.9 Nonlocal Declaration."""
    pytest.skip("Feature not yet implemented")
