"""
Feature: 14.1.5 Non-Data Descriptor (only __get__)
Category: Descriptors
Complexity: Very High
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
class NonDataDescriptor:
    def __get__(self, obj, objtype=None):
        return obj._value
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("very high")
@pytest.mark.status("not_implemented")
def test_1415_non_data_descriptor_only_get():
    """Test translation of 14.1.5 Non-Data Descriptor (only __get__)."""
    pytest.skip("Feature not yet implemented")
