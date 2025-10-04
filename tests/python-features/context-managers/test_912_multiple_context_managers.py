"""
Feature: 9.1.2 Multiple Context Managers
Category: Context Managers
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
with open("in.txt") as fin, open("out.txt", "w") as fout:
    fout.write(fin.read())
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_912_multiple_context_managers():
    """Test translation of 9.1.2 Multiple Context Managers."""
    pytest.skip("Feature not yet implemented")
