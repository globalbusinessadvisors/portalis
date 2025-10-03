"""
Main NeMo-based Translator

Orchestrates the translation of Python code to Rust using NeMo models,
type mapping, and semantic understanding.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
from loguru import logger

import libcst as cst

from .nemo_service import NeMoService, InferenceConfig, TranslationResult
from ..mapping.type_mapper import TypeMapper
from ..mapping.error_mapper import ErrorMapper


@dataclass
class TranslationConfig:
    """Configuration for translation process."""

    model_path: str
    gpu_enabled: bool = True
    batch_size: int = 32
    max_length: int = 512
    temperature: float = 0.2
    use_templates: bool = True
    validate_output: bool = True


@dataclass
class TranslatedCode:
    """Result of translating Python code to Rust."""

    rust_code: str
    confidence: float
    imports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


class NeMoTranslator:
    """
    High-level translator using NeMo for Python → Rust translation.

    Integrates:
    - NeMo language model for code generation
    - Type mapping for accurate type translations
    - Error mapping for exception handling
    - Template-based patterns for common idioms
    """

    def __init__(self, config: TranslationConfig):
        self.config = config

        # Initialize NeMo service
        inference_config = InferenceConfig(
            max_length=config.max_length,
            temperature=config.temperature,
            batch_size=config.batch_size,
            use_gpu=config.gpu_enabled
        )

        self.nemo_service = NeMoService(
            model_path=config.model_path,
            config=inference_config,
            enable_cuda=config.gpu_enabled
        )

        # Initialize mappers
        self.type_mapper = TypeMapper()
        self.error_mapper = ErrorMapper()

        # Statistics
        self.stats = {
            "translations": 0,
            "successes": 0,
            "failures": 0,
            "total_time_ms": 0.0
        }

        logger.info("NeMo translator initialized")

    def initialize(self) -> None:
        """Initialize the translator and load models."""
        self.nemo_service.initialize()
        logger.info("Translator ready")

    def translate_function(
        self,
        python_code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TranslatedCode:
        """
        Translate a Python function to Rust.

        Args:
            python_code: Python function source code
            context: Additional context (type hints, examples, etc.)

        Returns:
            TranslatedCode with Rust implementation
        """
        start_time = time.perf_counter()
        context = context or {}

        try:
            # Parse Python code to extract type information
            analysis = self._analyze_python_code(python_code)

            # Enhance context with type information
            enhanced_context = {
                **context,
                "type_hints": analysis["type_hints"],
                "exceptions": analysis["exceptions"],
                "imports": analysis["imports"],
            }

            # Use NeMo for translation
            result = self.nemo_service.translate_code(python_code, enhanced_context)

            # Post-process Rust code
            rust_code = self._post_process_rust(result.rust_code, analysis)

            # Collect imports
            imports = self._collect_imports(analysis)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update statistics
            self.stats["translations"] += 1
            self.stats["successes"] += 1
            self.stats["total_time_ms"] += elapsed_ms

            return TranslatedCode(
                rust_code=rust_code,
                confidence=result.confidence,
                imports=imports,
                metadata={
                    **result.metadata,
                    "analysis": analysis,
                },
                processing_time_ms=elapsed_ms
            )

        except Exception as e:
            self.stats["failures"] += 1
            logger.error(f"Translation failed: {e}")

            return TranslatedCode(
                rust_code="// Translation failed",
                confidence=0.0,
                warnings=[str(e)],
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )

    def translate_class(
        self,
        python_code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TranslatedCode:
        """
        Translate a Python class to Rust struct/enum.

        Args:
            python_code: Python class source code
            context: Additional context

        Returns:
            TranslatedCode with Rust implementation
        """
        # Similar to translate_function but handles class-specific patterns
        analysis = self._analyze_python_code(python_code)

        # Check if class should be enum or struct
        if self._is_enum_like(analysis):
            context = {**(context or {}), "target_type": "enum"}
        else:
            context = {**(context or {}), "target_type": "struct"}

        return self.translate_function(python_code, context)

    def translate_module(
        self,
        python_code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TranslatedCode:
        """
        Translate an entire Python module to Rust.

        Args:
            python_code: Python module source code
            context: Additional context

        Returns:
            TranslatedCode with complete Rust module
        """
        # Parse module structure
        try:
            module = cst.parse_module(python_code)

            # Extract components
            functions = []
            classes = []
            imports = []

            for stmt in module.body:
                if isinstance(stmt, cst.FunctionDef):
                    functions.append(stmt)
                elif isinstance(stmt, cst.ClassDef):
                    classes.append(stmt)
                elif isinstance(stmt, (cst.Import, cst.ImportFrom)):
                    imports.append(stmt)

            # Translate components
            rust_items = []

            for func in functions:
                func_code = module.code_for_node(func)
                result = self.translate_function(func_code, context)
                rust_items.append(result.rust_code)

            for cls in classes:
                cls_code = module.code_for_node(cls)
                result = self.translate_class(cls_code, context)
                rust_items.append(result.rust_code)

            # Combine into module
            rust_module = self._build_rust_module(rust_items, imports)

            return TranslatedCode(
                rust_code=rust_module,
                confidence=0.8,  # Average confidence
                imports=self._extract_module_imports(imports),
                metadata={"num_functions": len(functions), "num_classes": len(classes)}
            )

        except Exception as e:
            logger.error(f"Module translation failed: {e}")
            return TranslatedCode(
                rust_code="// Module translation failed",
                confidence=0.0,
                warnings=[str(e)]
            )

    def batch_translate(
        self,
        python_codes: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[TranslatedCode]:
        """
        Translate multiple Python code snippets in batch.

        Args:
            python_codes: List of Python source codes
            contexts: List of contexts for each code

        Returns:
            List of TranslatedCode objects
        """
        contexts = contexts or [{} for _ in python_codes]

        # Analyze all codes
        analyses = [self._analyze_python_code(code) for code in python_codes]

        # Enhance contexts
        enhanced_contexts = [
            {
                **ctx,
                "type_hints": analysis["type_hints"],
                "exceptions": analysis["exceptions"],
            }
            for ctx, analysis in zip(contexts, analyses)
        ]

        # Batch translate with NeMo
        results = self.nemo_service.batch_translate(python_codes, enhanced_contexts)

        # Post-process all results
        translated_codes = []
        for result, analysis in zip(results, analyses):
            rust_code = self._post_process_rust(result.rust_code, analysis)
            imports = self._collect_imports(analysis)

            translated_codes.append(
                TranslatedCode(
                    rust_code=rust_code,
                    confidence=result.confidence,
                    imports=imports,
                    metadata=result.metadata
                )
            )

        return translated_codes

    def _analyze_python_code(self, python_code: str) -> Dict[str, Any]:
        """
        Analyze Python code to extract type hints, exceptions, etc.

        Args:
            python_code: Python source code

        Returns:
            Analysis dictionary with extracted information
        """
        try:
            module = cst.parse_module(python_code)

            analysis = {
                "type_hints": {},
                "exceptions": [],
                "imports": [],
                "has_async": False,
                "has_generators": False,
            }

            # Extract type hints from function signatures
            for node in module.walk():
                if isinstance(node, cst.FunctionDef):
                    # Extract parameter types
                    for param in node.params.params:
                        if param.annotation:
                            param_name = param.name.value
                            analysis["type_hints"][param_name] = param.annotation

                    # Extract return type
                    if node.returns:
                        analysis["type_hints"]["__return__"] = node.returns

                    # Check for async
                    if node.asynchronous:
                        analysis["has_async"] = True

                # Extract raised exceptions
                elif isinstance(node, cst.Raise):
                    if node.exc:
                        if isinstance(node.exc, cst.Call):
                            if isinstance(node.exc.func, cst.Name):
                                analysis["exceptions"].append(node.exc.func.value)

                # Extract imports
                elif isinstance(node, (cst.Import, cst.ImportFrom)):
                    analysis["imports"].append(node)

            return analysis

        except Exception as e:
            logger.warning(f"Code analysis failed: {e}")
            return {
                "type_hints": {},
                "exceptions": [],
                "imports": [],
                "has_async": False,
                "has_generators": False,
            }

    def _post_process_rust(self, rust_code: str, analysis: Dict[str, Any]) -> str:
        """
        Post-process generated Rust code.

        - Add proper type annotations
        - Fix error handling
        - Format code
        """
        # Basic cleanup
        rust_code = rust_code.strip()

        # Wrap in Result if exceptions are raised
        if analysis.get("exceptions"):
            if "-> Result<" not in rust_code and "fn " in rust_code:
                # Add Result wrapper
                pass  # TODO: Implement sophisticated wrapping

        # Add async if needed
        if analysis.get("has_async") and "async fn" not in rust_code:
            rust_code = rust_code.replace("fn ", "async fn ", 1)

        return rust_code

    def _collect_imports(self, analysis: Dict[str, Any]) -> List[str]:
        """Collect required Rust imports based on analysis."""
        imports = set()

        # Add imports based on type hints
        for type_hint in analysis.get("type_hints", {}).values():
            rust_type = self.type_mapper.map_annotation(type_hint)
            imports.update(rust_type.imports)

        # Add imports based on exceptions
        for exception in analysis.get("exceptions", []):
            mapping = self.error_mapper.get_mapping(exception)
            if mapping:
                imports.update(mapping.requires_imports)

        return sorted(list(imports))

    def _is_enum_like(self, analysis: Dict[str, Any]) -> bool:
        """Determine if class should be translated to enum."""
        # Heuristic: if class has only class variables (no methods), it's enum-like
        # This is a simplified check - real implementation would be more sophisticated
        return False

    def _build_rust_module(
        self,
        rust_items: List[str],
        python_imports: List[Any]
    ) -> str:
        """Build complete Rust module from translated items."""
        # Generate module header
        header = "// Auto-generated by Portalis NeMo Translator\n\n"

        # Collect all imports
        all_imports = set()
        # TODO: Extract imports from rust_items

        imports_section = "\n".join(sorted(all_imports))

        # Combine items
        items_section = "\n\n".join(rust_items)

        return f"{header}{imports_section}\n\n{items_section}"

    def _extract_module_imports(self, python_imports: List[Any]) -> List[str]:
        """Extract Rust imports from Python imports."""
        # TODO: Map Python imports to Rust equivalents
        return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get translation statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successes"] / self.stats["translations"]
                if self.stats["translations"] > 0
                else 0.0
            ),
            "avg_time_ms": (
                self.stats["total_time_ms"] / self.stats["translations"]
                if self.stats["translations"] > 0
                else 0.0
            ),
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.nemo_service.cleanup()
        logger.info("Translator cleaned up")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
