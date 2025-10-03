"""
Python to Rust Type Mapping System

Provides comprehensive mapping between Python and Rust type systems,
including primitives, collections, generics, and custom types.
"""

from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import libcst as cst
from loguru import logger


class TypeCategory(Enum):
    """Categories of Python types."""
    PRIMITIVE = "primitive"
    COLLECTION = "collection"
    GENERIC = "generic"
    CALLABLE = "callable"
    CUSTOM = "custom"
    STDLIB = "stdlib"
    UNKNOWN = "unknown"


@dataclass
class RustType:
    """Represents a Rust type with metadata."""

    name: str
    category: TypeCategory
    is_copy: bool = False
    is_clone: bool = True
    requires_lifetime: bool = False
    imports: Set[str] = field(default_factory=set)
    documentation: str = ""

    def __str__(self) -> str:
        return self.name


@dataclass
class TypeMapping:
    """Maps a Python type to a Rust type with conversion rules."""

    python_type: str
    rust_type: RustType
    conversion_fn: Optional[str] = None
    notes: str = ""


class TypeMappingRegistry:
    """
    Registry of Python to Rust type mappings.

    Provides:
    - Primitive type mappings
    - Collection type mappings
    - Generic type handling
    - Stdlib type equivalents
    - Custom type registration
    """

    def __init__(self):
        self._primitives: Dict[str, RustType] = {}
        self._collections: Dict[str, RustType] = {}
        self._generics: Dict[str, RustType] = {}
        self._stdlib: Dict[str, RustType] = {}
        self._custom: Dict[str, RustType] = {}

        self._initialize_default_mappings()

    def _initialize_default_mappings(self) -> None:
        """Initialize default type mappings."""

        # Primitive types
        self._primitives = {
            "int": RustType("i64", TypeCategory.PRIMITIVE, is_copy=True),
            "float": RustType("f64", TypeCategory.PRIMITIVE, is_copy=True),
            "str": RustType("String", TypeCategory.PRIMITIVE, is_clone=True),
            "bool": RustType("bool", TypeCategory.PRIMITIVE, is_copy=True),
            "bytes": RustType("Vec<u8>", TypeCategory.PRIMITIVE),
            "None": RustType("()", TypeCategory.PRIMITIVE, is_copy=True),
        }

        # Collection types
        self._collections = {
            "list": RustType("Vec<T>", TypeCategory.COLLECTION),
            "dict": RustType(
                "HashMap<K, V>",
                TypeCategory.COLLECTION,
                imports={"use std::collections::HashMap;"}
            ),
            "set": RustType(
                "HashSet<T>",
                TypeCategory.COLLECTION,
                imports={"use std::collections::HashSet;"}
            ),
            "tuple": RustType("(T1, T2, ...)", TypeCategory.COLLECTION, is_copy=True),
            "frozenset": RustType(
                "HashSet<T>",
                TypeCategory.COLLECTION,
                imports={"use std::collections::HashSet;"}
            ),
        }

        # Generic types (from typing module)
        self._generics = {
            "Optional": RustType("Option<T>", TypeCategory.GENERIC),
            "Union": RustType("enum { ... }", TypeCategory.GENERIC),
            "List": RustType("Vec<T>", TypeCategory.GENERIC),
            "Dict": RustType(
                "HashMap<K, V>",
                TypeCategory.GENERIC,
                imports={"use std::collections::HashMap;"}
            ),
            "Set": RustType(
                "HashSet<T>",
                TypeCategory.GENERIC,
                imports={"use std::collections::HashSet;"}
            ),
            "Tuple": RustType("(T1, T2, ...)", TypeCategory.GENERIC),
            "Callable": RustType("Box<dyn Fn(Args) -> Ret>", TypeCategory.CALLABLE),
            "Iterator": RustType("impl Iterator<Item = T>", TypeCategory.GENERIC),
            "Iterable": RustType("impl IntoIterator<Item = T>", TypeCategory.GENERIC),
            "Any": RustType(
                "Box<dyn Any>",
                TypeCategory.GENERIC,
                imports={"use std::any::Any;"}
            ),
        }

        # Stdlib types
        self._stdlib = {
            "pathlib.Path": RustType(
                "PathBuf",
                TypeCategory.STDLIB,
                imports={"use std::path::PathBuf;"}
            ),
            "datetime.datetime": RustType(
                "DateTime<Utc>",
                TypeCategory.STDLIB,
                imports={"use chrono::{DateTime, Utc};"}
            ),
            "datetime.date": RustType(
                "NaiveDate",
                TypeCategory.STDLIB,
                imports={"use chrono::NaiveDate;"}
            ),
            "datetime.time": RustType(
                "NaiveTime",
                TypeCategory.STDLIB,
                imports={"use chrono::NaiveTime;"}
            ),
            "re.Pattern": RustType(
                "Regex",
                TypeCategory.STDLIB,
                imports={"use regex::Regex;"}
            ),
            "io.TextIOWrapper": RustType(
                "BufReader<File>",
                TypeCategory.STDLIB,
                imports={"use std::io::BufReader;", "use std::fs::File;"}
            ),
            "io.BytesIO": RustType(
                "Cursor<Vec<u8>>",
                TypeCategory.STDLIB,
                imports={"use std::io::Cursor;"}
            ),
        }

    def get_rust_type(
        self,
        python_type: str,
        type_args: Optional[List[str]] = None
    ) -> RustType:
        """
        Get Rust type for a given Python type.

        Args:
            python_type: Python type name (e.g., 'int', 'List', 'Optional')
            type_args: Generic type arguments if applicable

        Returns:
            RustType object with mapping information
        """
        # Check primitives first
        if python_type in self._primitives:
            return self._primitives[python_type]

        # Check collections
        if python_type in self._collections:
            rust_type = self._collections[python_type]
            if type_args:
                # Instantiate generic collection
                return self._instantiate_generic(rust_type, type_args)
            return rust_type

        # Check generics
        if python_type in self._generics:
            rust_type = self._generics[python_type]
            if type_args:
                return self._instantiate_generic(rust_type, type_args)
            return rust_type

        # Check stdlib
        if python_type in self._stdlib:
            return self._stdlib[python_type]

        # Check custom types
        if python_type in self._custom:
            return self._custom[python_type]

        # Unknown type - return dynamic type
        logger.warning(f"Unknown type '{python_type}', using Box<dyn Any>")
        return RustType(
            "Box<dyn Any>",
            TypeCategory.UNKNOWN,
            imports={"use std::any::Any;"}
        )

    def _instantiate_generic(
        self,
        rust_type: RustType,
        type_args: List[str]
    ) -> RustType:
        """Instantiate a generic Rust type with concrete type arguments."""
        name = rust_type.name

        if "Vec<T>" in name:
            # List[int] -> Vec<i64>
            arg_type = self.get_rust_type(type_args[0])
            new_name = f"Vec<{arg_type.name}>"

        elif "HashMap<K, V>" in name:
            # Dict[str, int] -> HashMap<String, i64>
            key_type = self.get_rust_type(type_args[0])
            val_type = self.get_rust_type(type_args[1])
            new_name = f"HashMap<{key_type.name}, {val_type.name}>"

        elif "HashSet<T>" in name:
            # Set[str] -> HashSet<String>
            arg_type = self.get_rust_type(type_args[0])
            new_name = f"HashSet<{arg_type.name}>"

        elif "Option<T>" in name:
            # Optional[int] -> Option<i64>
            arg_type = self.get_rust_type(type_args[0])
            new_name = f"Option<{arg_type.name}>"

        elif "Tuple" in name or "(" in name:
            # Tuple[int, str, bool] -> (i64, String, bool)
            arg_types = [self.get_rust_type(arg) for arg in type_args]
            new_name = f"({', '.join(t.name for t in arg_types)})"

        else:
            # Generic instantiation
            new_name = name.replace("T", type_args[0] if type_args else "T")

        return RustType(
            new_name,
            rust_type.category,
            rust_type.is_copy,
            rust_type.is_clone,
            rust_type.requires_lifetime,
            rust_type.imports.copy()
        )

    def register_custom_type(
        self,
        python_name: str,
        rust_name: str,
        **kwargs
    ) -> None:
        """Register a custom type mapping."""
        rust_type = RustType(rust_name, TypeCategory.CUSTOM, **kwargs)
        self._custom[python_name] = rust_type
        logger.info(f"Registered custom type: {python_name} -> {rust_name}")

    def get_all_imports(self, rust_types: List[RustType]) -> Set[str]:
        """Get all required imports for a set of Rust types."""
        imports = set()
        for rt in rust_types:
            imports.update(rt.imports)
        return imports


class TypeMapper:
    """
    High-level type mapping interface.

    Analyzes Python type annotations and produces Rust type equivalents.
    """

    def __init__(self, registry: Optional[TypeMappingRegistry] = None):
        self.registry = registry or TypeMappingRegistry()

    def map_annotation(self, annotation: Any) -> RustType:
        """
        Map a Python type annotation to Rust type.

        Args:
            annotation: Python type annotation (from AST or typing module)

        Returns:
            RustType object
        """
        if annotation is None:
            return self.registry.get_rust_type("None")

        # Handle string annotations
        if isinstance(annotation, str):
            return self._parse_string_annotation(annotation)

        # Handle libcst annotations
        if isinstance(annotation, cst.Annotation):
            return self._map_cst_annotation(annotation)

        # Handle typing module types
        return self._map_typing_annotation(annotation)

    def _parse_string_annotation(self, annotation: str) -> RustType:
        """Parse string type annotation."""
        annotation = annotation.strip()

        # Handle Optional[T]
        if annotation.startswith("Optional["):
            inner = annotation[9:-1]
            inner_type = self._parse_string_annotation(inner)
            return RustType(
                f"Option<{inner_type.name}>",
                TypeCategory.GENERIC
            )

        # Handle List[T]
        if annotation.startswith("List["):
            inner = annotation[5:-1]
            inner_type = self._parse_string_annotation(inner)
            return RustType(
                f"Vec<{inner_type.name}>",
                TypeCategory.GENERIC
            )

        # Handle Dict[K, V]
        if annotation.startswith("Dict["):
            args = annotation[5:-1].split(",")
            key_type = self._parse_string_annotation(args[0].strip())
            val_type = self._parse_string_annotation(args[1].strip())
            return RustType(
                f"HashMap<{key_type.name}, {val_type.name}>",
                TypeCategory.GENERIC,
                imports={"use std::collections::HashMap;"}
            )

        # Handle Union types
        if annotation.startswith("Union["):
            # Return generic enum type (needs custom implementation)
            return RustType(
                "UnionType",
                TypeCategory.GENERIC,
                documentation=f"Union type: {annotation}"
            )

        # Simple type
        return self.registry.get_rust_type(annotation)

    def _map_cst_annotation(self, annotation: cst.Annotation) -> RustType:
        """Map libcst annotation node to Rust type."""
        # Extract type from annotation node
        type_node = annotation.annotation

        if isinstance(type_node, cst.Name):
            # Simple type like 'int', 'str'
            return self.registry.get_rust_type(type_node.value)

        elif isinstance(type_node, cst.Subscript):
            # Generic type like 'List[int]', 'Optional[str]'
            base = type_node.value
            if isinstance(base, cst.Name):
                base_name = base.value

                # Extract type arguments
                type_args = self._extract_subscript_args(type_node.slice)
                return self.registry.get_rust_type(base_name, type_args)

        # Fallback
        return self.registry.get_rust_type("Any")

    def _extract_subscript_args(self, slice_node) -> List[str]:
        """Extract type arguments from subscript slice."""
        args = []

        # Handle different slice structures
        if isinstance(slice_node, cst.Index):
            value = slice_node.value
            if isinstance(value, cst.Name):
                args.append(value.value)
            elif isinstance(value, cst.Tuple):
                for elem in value.elements:
                    if isinstance(elem.value, cst.Name):
                        args.append(elem.value.value)

        return args

    def _map_typing_annotation(self, annotation: Any) -> RustType:
        """Map Python typing module annotation."""
        # This handles runtime typing objects
        type_str = str(annotation)
        return self._parse_string_annotation(type_str)

    def map_function_signature(
        self,
        param_types: List[Any],
        return_type: Any
    ) -> Dict[str, Any]:
        """
        Map function signature to Rust.

        Args:
            param_types: List of parameter type annotations
            return_type: Return type annotation

        Returns:
            Dictionary with Rust signature information
        """
        rust_params = [self.map_annotation(pt) for pt in param_types]
        rust_return = self.map_annotation(return_type)

        # Collect all imports
        all_types = rust_params + [rust_return]
        imports = self.registry.get_all_imports(all_types)

        return {
            "parameters": rust_params,
            "return_type": rust_return,
            "imports": imports,
        }

    def infer_type_from_value(self, value: Any) -> RustType:
        """Infer Rust type from Python runtime value."""
        python_type = type(value).__name__

        # Handle collections with element type inference
        if isinstance(value, list):
            if value:
                elem_type = self.infer_type_from_value(value[0])
                return RustType(f"Vec<{elem_type.name}>", TypeCategory.COLLECTION)
            return RustType("Vec<T>", TypeCategory.COLLECTION)

        elif isinstance(value, dict):
            if value:
                first_key = next(iter(value.keys()))
                first_val = value[first_key]
                key_type = self.infer_type_from_value(first_key)
                val_type = self.infer_type_from_value(first_val)
                return RustType(
                    f"HashMap<{key_type.name}, {val_type.name}>",
                    TypeCategory.COLLECTION,
                    imports={"use std::collections::HashMap;"}
                )
            return self.registry.get_rust_type("dict")

        elif isinstance(value, set):
            if value:
                elem_type = self.infer_type_from_value(next(iter(value)))
                return RustType(
                    f"HashSet<{elem_type.name}>",
                    TypeCategory.COLLECTION,
                    imports={"use std::collections::HashSet;"}
                )
            return self.registry.get_rust_type("set")

        # Simple types
        return self.registry.get_rust_type(python_type)
