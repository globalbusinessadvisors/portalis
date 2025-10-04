"""
Feature: 6.2.4 Read-Only Property
Category: Classes & OOP
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
@property
def constant(self):
    return 42
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_624_read_only_property():
    """Test translation of 6.2.4 Read-Only Property."""
    pytest.skip("Feature not yet implemented")
