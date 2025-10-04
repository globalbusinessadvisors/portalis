"""
Feature: 2.8.5 Subscription ([])
Category: Operators
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
item = lst[0]
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_285_subscription():
    """Test translation of 2.8.5 Subscription ([])."""
    pytest.skip("Feature not yet implemented")
