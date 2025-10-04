"""
Feature: 3.4.6 Yield From Statement
Category: Control Flow
Complexity: High
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
def gen():
    yield from another_generator()
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("high")
@pytest.mark.status("not_implemented")
def test_346_yield_from_statement():
    """Test translation of 3.4.6 Yield From Statement."""
    pytest.skip("Feature not yet implemented")
