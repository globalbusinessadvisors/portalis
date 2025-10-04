"""
Feature: 1.1.8 Global Declaration
Category: Basic Syntax & Literals
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
global_var = 10

def modify():
    global global_var
    global_var = 20
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_118_global_declaration():
    """Test translation of 1.1.8 Global Declaration."""
    pytest.skip("Feature not yet implemented")
