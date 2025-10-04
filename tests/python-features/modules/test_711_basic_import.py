"""
Feature: 7.1.1 Basic Import
Category: Modules & Imports
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
import math
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_711_basic_import():
    """Test translation of 7.1.1 Basic Import."""
    pytest.skip("Feature not yet implemented")
