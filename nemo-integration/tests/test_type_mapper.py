"""
Unit tests for Type Mapper

Tests Python to Rust type system mapping.
"""

import pytest

from src.mapping.type_mapper import (
    TypeMapper,
    TypeMappingRegistry,
    TypeCategory,
    RustType
)


@pytest.fixture
def registry():
    """Create type mapping registry."""
    return TypeMappingRegistry()


@pytest.fixture
def mapper():
    """Create type mapper."""
    return TypeMapper()


class TestTypeMappingRegistry:
    """Test suite for TypeMappingRegistry."""

    def test_primitive_mappings(self, registry):
        """Test primitive type mappings."""
        assert registry.get_rust_type("int").name == "i64"
        assert registry.get_rust_type("float").name == "f64"
        assert registry.get_rust_type("str").name == "String"
        assert registry.get_rust_type("bool").name == "bool"
        assert registry.get_rust_type("None").name == "()"

    def test_collection_mappings(self, registry):
        """Test collection type mappings."""
        # Simple collections
        assert "Vec<T>" in registry.get_rust_type("list").name
        assert "HashMap<K, V>" in registry.get_rust_type("dict").name
        assert "HashSet<T>" in registry.get_rust_type("set").name

    def test_generic_instantiation(self, registry):
        """Test generic type instantiation."""
        # List[int] -> Vec<i64>
        rust_type = registry.get_rust_type("List", ["int"])
        assert rust_type.name == "Vec<i64>"

        # Dict[str, int] -> HashMap<String, i64>
        rust_type = registry.get_rust_type("Dict", ["str", "int"])
        assert rust_type.name == "HashMap<String, i64>"

        # Optional[str] -> Option<String>
        rust_type = registry.get_rust_type("Optional", ["str"])
        assert rust_type.name == "Option<String>"

    def test_stdlib_mappings(self, registry):
        """Test standard library type mappings."""
        path_type = registry.get_rust_type("pathlib.Path")
        assert path_type.name == "PathBuf"
        assert "use std::path::PathBuf;" in path_type.imports

        datetime_type = registry.get_rust_type("datetime.datetime")
        assert "DateTime<Utc>" in datetime_type.name
        assert any("chrono" in imp for imp in datetime_type.imports)

    def test_custom_type_registration(self, registry):
        """Test registering custom types."""
        registry.register_custom_type(
            "MyCustomType",
            "CustomRustType",
            is_copy=True
        )

        rust_type = registry.get_rust_type("MyCustomType")
        assert rust_type.name == "CustomRustType"
        assert rust_type.is_copy is True
        assert rust_type.category == TypeCategory.CUSTOM

    def test_unknown_type_fallback(self, registry):
        """Test handling of unknown types."""
        unknown_type = registry.get_rust_type("UnknownType")

        assert "Any" in unknown_type.name
        assert unknown_type.category == TypeCategory.UNKNOWN


class TestTypeMapper:
    """Test suite for TypeMapper."""

    def test_map_simple_annotation(self, mapper):
        """Test mapping simple type annotations."""
        rust_type = mapper.map_annotation("int")
        assert rust_type.name == "i64"

        rust_type = mapper.map_annotation("str")
        assert rust_type.name == "String"

    def test_map_optional_annotation(self, mapper):
        """Test mapping Optional types."""
        rust_type = mapper.map_annotation("Optional[int]")
        assert rust_type.name == "Option<i64>"

        rust_type = mapper.map_annotation("Optional[str]")
        assert rust_type.name == "Option<String>"

    def test_map_list_annotation(self, mapper):
        """Test mapping List types."""
        rust_type = mapper.map_annotation("List[int]")
        assert rust_type.name == "Vec<i64>"

        rust_type = mapper.map_annotation("List[str]")
        assert rust_type.name == "Vec<String>"

    def test_map_dict_annotation(self, mapper):
        """Test mapping Dict types."""
        rust_type = mapper.map_annotation("Dict[str, int]")
        assert rust_type.name == "HashMap<String, i64>"

        # Check imports
        assert "HashMap" in str(rust_type.imports)

    def test_map_union_annotation(self, mapper):
        """Test mapping Union types."""
        rust_type = mapper.map_annotation("Union[int, str]")

        # Union should map to some enum-like type
        assert rust_type is not None
        assert rust_type.category == TypeCategory.GENERIC

    def test_map_function_signature(self, mapper):
        """Test mapping function signatures."""
        param_types = ["int", "str", "Optional[bool]"]
        return_type = "List[int]"

        signature = mapper.map_function_signature(param_types, return_type)

        assert len(signature["parameters"]) == 3
        assert signature["parameters"][0].name == "i64"
        assert signature["parameters"][1].name == "String"
        assert signature["parameters"][2].name == "Option<bool>"
        assert signature["return_type"].name == "Vec<i64>"

    def test_infer_type_from_value(self, mapper):
        """Test type inference from runtime values."""
        # Integer
        rust_type = mapper.infer_type_from_value(42)
        assert rust_type.name == "i64"

        # String
        rust_type = mapper.infer_type_from_value("hello")
        assert rust_type.name == "String"

        # List
        rust_type = mapper.infer_type_from_value([1, 2, 3])
        assert "Vec<" in rust_type.name
        assert "i64" in rust_type.name

        # Dict
        rust_type = mapper.infer_type_from_value({"key": "value"})
        assert "HashMap<" in rust_type.name

    def test_get_all_imports(self, mapper):
        """Test collecting all required imports."""
        rust_types = [
            mapper.map_annotation("Dict[str, int]"),
            mapper.map_annotation("Set[int]"),
            mapper.map_annotation("pathlib.Path"),
        ]

        imports = mapper.registry.get_all_imports(rust_types)

        assert any("HashMap" in imp for imp in imports)
        assert any("HashSet" in imp for imp in imports)
        assert any("PathBuf" in imp for imp in imports)


@pytest.mark.parametrize("python_type,expected_rust", [
    ("int", "i64"),
    ("float", "f64"),
    ("str", "String"),
    ("bool", "bool"),
    ("bytes", "Vec<u8>"),
    ("None", "()"),
])
def test_primitive_type_mappings(mapper, python_type, expected_rust):
    """Test all primitive type mappings."""
    rust_type = mapper.map_annotation(python_type)
    assert rust_type.name == expected_rust


@pytest.mark.parametrize("python_type,expected_pattern", [
    ("List[int]", r"Vec<i64>"),
    ("Dict[str, int]", r"HashMap<String, i64>"),
    ("Set[str]", r"HashSet<String>"),
    ("Optional[int]", r"Option<i64>"),
    ("Tuple[int, str]", r"\(i64, String\)"),
])
def test_generic_type_patterns(mapper, python_type, expected_pattern):
    """Test generic type mapping patterns."""
    import re

    rust_type = mapper.map_annotation(python_type)
    assert re.match(expected_pattern, rust_type.name)
