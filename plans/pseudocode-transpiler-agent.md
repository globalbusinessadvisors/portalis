# Transpiler Agent - Pseudocode
## PORTALIS SPARC Phase 2: Pseudocode

**Agent**: Transpiler Agent
**Version**: 1.0
**Date**: 2025-10-03
**Responsibility**: Generate syntactically valid, idiomatic Rust code from Python source while preserving semantic equivalence

---

## Table of Contents

1. [Agent Overview](#1-agent-overview)
2. [Data Structures](#2-data-structures)
3. [Core Algorithms](#3-core-algorithms)
4. [Input/Output Contracts](#4-inputoutput-contracts)
5. [CUDA Integration Points](#5-cuda-integration-points)
6. [London School TDD Test Points](#6-london-school-tdd-test-points)

---

## 1. Agent Overview

### 1.1 Purpose

The Transpiler Agent is the core code generation component of the Portalis platform. It consumes analyzed Python ASTs and specification contracts to produce idiomatic, semantically equivalent Rust code optimized for WASM compilation.

### 1.2 Responsibilities

- **Rust Code Generation** (FR-2.4.1): Generate syntactically valid, idiomatic Rust preserving Python semantics
- **Multi-Crate Workspace Generation** (FR-2.4.2): Organize code into proper Rust workspace structure
- **Error Handling Translation** (FR-2.4.3): Map Python exceptions to Rust Result/Error types
- **Type System Mapping** (FR-2.4.4): Translate Python types to appropriate Rust equivalents
- **CUDA Acceleration** (FR-2.4.5): Leverage GPU for parallel translation at scale

### 1.3 Sub-Components

```
TranspilerAgent
├── SyntaxTransformer      # AST-level Python → Rust mapping
├── SemanticAdapter        # Python-specific semantics handling
├── WorkspaceOrganizer     # Multi-crate structure generation
├── ErrorTranslator        # Exception → Result translation
├── TypeMapper             # Python → Rust type system mapping
├── TemplateEngine         # Code generation from templates
└── CUDAAccelerator        # GPU-accelerated batch processing
```

---

## 2. Data Structures

### 2.1 Translation Context

```pseudocode
STRUCTURE TranslationContext:
    # Input data from previous agents
    python_ast: PythonAST
    api_spec: APISpec
    dependency_graph: DependencyGraph
    contract_map: ContractMap
    rust_spec: RustSpecification

    # Configuration
    mode: ExecutionMode  # Script | Library
    optimization_level: OptLevel  # Debug | Release
    target_features: Set<Feature>  # [WASM, NIM, Omniverse]

    # State tracking
    translated_modules: Map<ModulePath, RustModule>
    type_mappings: Map<PythonType, RustType>
    error_types: Map<ExceptionType, RustErrorEnum>
    import_graph: Map<ModulePath, Set<ImportPath>>

    # GPU resources
    cuda_context: CUDAContext
    embedding_cache: GPUEmbeddingCache
    translation_candidates: GPUCandidateStore

    # Output accumulation
    workspace: RustWorkspace
    cargo_configs: Map<CrateName, CargoToml>
    module_comments: Map<ModulePath, DocumentationBlock>
END STRUCTURE


STRUCTURE RustWorkspace:
    root_path: Path
    workspace_cargo_toml: CargoToml
    crates: Map<CrateName, RustCrate>
    shared_types: SharedTypeDefinitions
    workspace_members: List<CrateName>
END STRUCTURE


STRUCTURE RustCrate:
    name: CrateName
    path: Path
    cargo_toml: CargoToml
    lib_module: Option<RustModule>
    bin_modules: List<RustModule>
    sub_modules: Map<ModuleName, RustModule>
    dependencies: Map<CrateName, VersionSpec>
    dev_dependencies: Map<CrateName, VersionSpec>
END STRUCTURE


STRUCTURE RustModule:
    name: ModuleName
    path: Path
    items: List<RustItem>
    imports: List<UseStatement>
    module_attributes: List<Attribute>
    documentation: DocumentationBlock
    source_mapping: Map<RustLocation, PythonLocation>  # For debugging
END STRUCTURE


STRUCTURE RustItem:
    VARIANT:
        Function(RustFunction)
        Struct(RustStruct)
        Enum(RustEnum)
        Trait(RustTrait)
        Impl(RustImpl)
        Constant(RustConst)
        TypeAlias(RustTypeAlias)
        Module(RustModule)
END STRUCTURE
```

### 2.2 Translation Templates

```pseudocode
STRUCTURE TranslationTemplate:
    name: TemplateName
    pattern: PythonPattern  # AST pattern to match
    rust_generator: Fn(MatchedNodes, Context) -> RustCode
    confidence_score: f32
    required_imports: Set<RustImport>
    required_features: Set<CargoFeature>
    applicability_test: Fn(PythonAST) -> bool
END STRUCTURE


STRUCTURE CodeTemplate:
    # Pre-defined templates for common patterns

    # Exception handling
    try_except_template: Template
    raise_exception_template: Template
    custom_exception_template: Template

    # Async/await
    async_function_template: Template
    await_expression_template: Template
    async_context_manager_template: Template

    # Type conversions
    type_cast_template: Template
    type_check_template: Template
    union_type_template: Template

    # Collections
    list_comprehension_template: Template
    dict_comprehension_template: Template
    generator_expression_template: Template

    # Context managers
    with_statement_template: Template
    context_manager_protocol_template: Template

    # OOP patterns
    class_definition_template: Template
    inheritance_template: Template
    property_template: Template
    decorator_template: Template
END STRUCTURE
```

### 2.3 Type Mapping Registry

```pseudocode
STRUCTURE TypeMappingRegistry:
    # Primitive mappings
    primitives: Map<PythonPrimitive, RustPrimitive>
    # Examples:
    # int -> i64 (default) | i32 | i128 (based on range analysis)
    # float -> f64
    # str -> String | &str (based on ownership analysis)
    # bool -> bool
    # bytes -> Vec<u8> | &[u8]
    # None -> Option<T>::None

    # Collection mappings
    collections: Map<PythonCollection, RustCollection>
    # Examples:
    # list[T] -> Vec<T>
    # dict[K, V] -> HashMap<K, V> | BTreeMap<K, V>
    # set[T] -> HashSet<T> | BTreeSet<T>
    # tuple[T1, T2, ...] -> (T1, T2, ...)
    # frozenset[T] -> HashSet<T> (with immutable wrapper)

    # Special types
    special_types: Map<PythonSpecialType, RustType>
    # Examples:
    # typing.Optional[T] -> Option<T>
    # typing.Union[T1, T2] -> enum { Variant1(T1), Variant2(T2) }
    # typing.Callable[[Args], Ret] -> Fn(Args) -> Ret
    # typing.Any -> Box<dyn Any> (with type erasure)

    # User-defined class mappings
    class_mappings: Map<ClassName, StructOrEnumMapping>

    # Stdlib type mappings
    stdlib_mappings: Map<StdlibType, RustEquivalent>
    # Examples:
    # pathlib.Path -> std::path::PathBuf
    # datetime.datetime -> chrono::DateTime<Utc>
    # re.Pattern -> regex::Regex
    # io.TextIOWrapper -> std::io::BufReader<File>
END STRUCTURE
```

### 2.4 Error Translation Map

```pseudocode
STRUCTURE ErrorTranslationMap:
    # Standard Python exceptions -> Rust error types
    exception_hierarchy: Map<ExceptionType, ErrorEnumVariant>
    # Examples:
    # ValueError -> Error::ValueError(String)
    # TypeError -> Error::TypeError { expected: String, got: String }
    # KeyError -> Error::KeyError(String)
    # FileNotFoundError -> Error::IoError(std::io::Error)

    # Custom exception classes
    custom_exceptions: Map<CustomExceptionClass, CustomErrorEnum>

    # Exception chaining
    exception_context: ExceptionContextStrategy
    # Strategy for preserving Python's "raise ... from ..." semantics

    # Error propagation patterns
    propagation_rules: Map<ExceptionType, PropagationStrategy>
    # Examples:
    # Some exceptions map to panic! (assertion failures)
    # Most map to Result::Err propagation with ?
    # Some require custom handling
END STRUCTURE
```

### 2.5 GPU Translation Cache

```pseudocode
STRUCTURE GPUTranslationCache:
    # Embeddings for translation candidates
    python_embeddings: CUDAMatrix<f32>  # Shape: [N, embedding_dim]
    rust_embeddings: CUDAMatrix<f32>    # Shape: [M, embedding_dim]

    # Similarity index
    similarity_index: CUDAKDTree  # GPU-accelerated k-NN search

    # Pre-computed translation candidates
    candidate_store: Map<PythonASTHash, List<RustCandidate>>
    # Each candidate includes:
    #   - rust_code: String
    #   - confidence: f32
    #   - required_context: Set<Dependency>

    # Translation history (for learning)
    successful_translations: List<(PythonAST, RustCode, Metrics)>
    failed_translations: List<(PythonAST, ErrorReport)>

    # GPU memory pools
    embedding_pool: CUDAMemoryPool
    computation_stream: CUDAStream
END STRUCTURE
```

---

## 3. Core Algorithms

### 3.1 Main Transpilation Entry Point

```pseudocode
FUNCTION transpile_codebase(context: TranslationContext) -> Result<RustWorkspace, Error>:
    """
    Main entry point for transpiling Python codebase to Rust.
    Coordinates all sub-components and manages the translation pipeline.
    """

    # Step 1: Initialize GPU resources if enabled
    IF context.config.gpu_enabled THEN:
        initialize_cuda_context(context)
        preload_embeddings_to_gpu(context)
        warm_up_translation_cache(context)
    END IF

    # Step 2: Organize workspace structure
    workspace_layout = determine_workspace_layout(context)
    context.workspace = create_workspace_skeleton(workspace_layout)

    # Step 3: Translate modules in dependency order
    ordered_modules = topological_sort(context.dependency_graph)

    # Step 4: GPU-accelerated batch processing
    IF context.config.gpu_enabled AND length(ordered_modules) > BATCH_THRESHOLD THEN:
        translated_modules = gpu_batch_translate_modules(ordered_modules, context)
    ELSE:
        translated_modules = []
        FOR module IN ordered_modules DO:
            translated = translate_single_module(module, context)
            translated_modules.append(translated)
        END FOR
    END IF

    # Step 5: Organize translated modules into crates
    FOR translated_module IN translated_modules DO:
        crate = determine_target_crate(translated_module, workspace_layout)
        add_module_to_crate(context.workspace, crate, translated_module)
    END FOR

    # Step 6: Generate workspace-level configuration
    generate_workspace_cargo_toml(context.workspace, context)
    generate_crate_cargo_tomls(context.workspace, context)

    # Step 7: Add cross-module glue code
    generate_module_reexports(context.workspace)
    generate_workspace_prelude(context.workspace)

    # Step 8: Inject WASI compatibility layer
    IF "WASM" IN context.target_features THEN:
        inject_wasi_shims(context.workspace)
    END IF

    # Step 9: Validate generated code
    validation_result = validate_workspace(context.workspace)
    IF validation_result.has_errors() THEN:
        RETURN Error::ValidationFailed(validation_result.errors)
    END IF

    # Step 10: Cleanup GPU resources
    IF context.config.gpu_enabled THEN:
        cleanup_cuda_context(context)
    END IF

    RETURN Ok(context.workspace)
END FUNCTION
```

### 3.2 Module Translation

```pseudocode
FUNCTION translate_single_module(
    python_module: PythonModule,
    context: TranslationContext
) -> Result<RustModule, Error>:
    """
    Translate a single Python module to Rust.
    """

    # Initialize output module
    rust_module = RustModule::new(python_module.name)

    # Extract module-level documentation
    rust_module.documentation = extract_module_docs(python_module)

    # Translate module-level imports
    rust_module.imports = translate_imports(python_module.imports, context)

    # Translate module-level constants
    FOR constant IN python_module.constants DO:
        rust_constant = translate_constant(constant, context)
        rust_module.items.append(RustItem::Constant(rust_constant))
    END FOR

    # Translate type aliases and newtype definitions
    FOR type_alias IN python_module.type_aliases DO:
        rust_type_alias = translate_type_alias(type_alias, context)
        rust_module.items.append(RustItem::TypeAlias(rust_type_alias))
    END FOR

    # Translate classes (structs/enums/traits)
    FOR class_def IN python_module.classes DO:
        rust_items = translate_class(class_def, context)
        rust_module.items.extend(rust_items)
    END FOR

    # Translate top-level functions
    FOR function_def IN python_module.functions DO:
        rust_function = translate_function(function_def, context)
        rust_module.items.append(RustItem::Function(rust_function))
    END FOR

    # Add module attributes (e.g., #![allow(dead_code)] for generated code)
    rust_module.module_attributes = generate_module_attributes(python_module, context)

    # Build source location mapping for debugging
    rust_module.source_mapping = build_source_mapping(python_module, rust_module)

    RETURN Ok(rust_module)
END FUNCTION
```

### 3.3 Function Translation

```pseudocode
FUNCTION translate_function(
    py_func: PythonFunctionDef,
    context: TranslationContext
) -> RustFunction:
    """
    Translate a Python function to Rust, handling signatures, body, and special cases.
    """

    # Extract function metadata
    func_name = sanitize_identifier(py_func.name)
    is_async = py_func.is_async
    is_generator = py_func.is_generator

    # Translate function signature
    signature = translate_function_signature(py_func, context)

    # Special handling for async functions
    IF is_async THEN:
        signature.is_async = True
        signature.return_type = wrap_in_future(signature.return_type)
    END IF

    # Special handling for generators
    IF is_generator THEN:
        # Convert to iterator pattern
        RETURN translate_generator_to_iterator(py_func, context)
    END IF

    # Translate function body
    body = translate_function_body(py_func.body, context)

    # Extract and translate docstring
    documentation = translate_docstring(py_func.docstring, context)

    # Determine visibility
    visibility = determine_visibility(py_func, context)

    # Collect required attributes
    attributes = []
    IF needs_inline_hint(py_func) THEN:
        attributes.append("#[inline]")
    END IF
    IF is_async THEN:
        attributes.append("#[async_trait]")  # If trait method
    END IF

    # Build Rust function
    rust_function = RustFunction {
        name: func_name,
        signature: signature,
        body: body,
        documentation: documentation,
        visibility: visibility,
        attributes: attributes,
    }

    RETURN rust_function
END FUNCTION


FUNCTION translate_function_signature(
    py_func: PythonFunctionDef,
    context: TranslationContext
) -> RustSignature:
    """
    Translate Python function signature to Rust, including parameters and return type.
    """

    parameters = []

    # Handle 'self' parameter for methods
    IF py_func.is_method THEN:
        self_param = determine_self_type(py_func, context)
        # Options: self, &self, &mut self, self: Box<Self>, etc.
        parameters.append(self_param)
    END IF

    # Translate regular parameters
    FOR param IN py_func.parameters DO:
        rust_param = translate_parameter(param, context)
        parameters.append(rust_param)
    END FOR

    # Handle *args and **kwargs
    IF py_func.has_var_args THEN:
        # Convert to Vec<T> for *args
        varargs_param = RustParameter {
            name: param_name,
            param_type: "Vec<DynamicValue>",  # Or specific type if inferable
            default_value: None,
        }
        parameters.append(varargs_param)
    END IF

    IF py_func.has_var_kwargs THEN:
        # Convert to HashMap<String, T> for **kwargs
        kwargs_param = RustParameter {
            name: param_name,
            param_type: "HashMap<String, DynamicValue>",
            default_value: None,
        }
        parameters.append(kwargs_param)
    END IF

    # Translate return type
    return_type = translate_type_annotation(py_func.return_annotation, context)

    # Wrap in Result if function can raise exceptions
    IF py_func.can_raise_exceptions THEN:
        error_type = determine_error_type(py_func, context)
        return_type = "Result<{return_type}, {error_type}>"
    END IF

    signature = RustSignature {
        parameters: parameters,
        return_type: return_type,
        generics: extract_generic_parameters(py_func, context),
        where_clauses: generate_where_clauses(py_func, context),
    }

    RETURN signature
END FUNCTION
```

### 3.4 Class Translation

```pseudocode
FUNCTION translate_class(
    py_class: PythonClassDef,
    context: TranslationContext
) -> List<RustItem>:
    """
    Translate a Python class to Rust struct/enum + trait implementations.
    Returns multiple Rust items (struct, impl blocks, trait impls).
    """

    rust_items = []

    # Determine if this should be a struct or enum
    IF is_enum_like(py_class) THEN:
        # Translate to Rust enum (e.g., for sum types or state machines)
        rust_enum = translate_to_enum(py_class, context)
        rust_items.append(RustItem::Enum(rust_enum))
        base_type = rust_enum.name
    ELSE:
        # Translate to Rust struct
        rust_struct = translate_to_struct(py_class, context)
        rust_items.append(RustItem::Struct(rust_struct))
        base_type = rust_struct.name
    END IF

    # Translate methods to impl block
    impl_block = RustImpl {
        target_type: base_type,
        trait_impl: None,  # Inherent impl
        methods: [],
    }

    FOR method IN py_class.methods DO:
        # Skip special methods that become trait impls
        IF is_special_method(method) THEN:
            CONTINUE
        END IF

        rust_method = translate_function(method, context)
        impl_block.methods.append(rust_method)
    END FOR

    rust_items.append(RustItem::Impl(impl_block))

    # Translate special methods to trait implementations
    trait_impls = translate_special_methods_to_traits(py_class, context)
    rust_items.extend(trait_impls)

    # Handle inheritance via trait bounds
    IF py_class.has_base_classes THEN:
        inheritance_traits = translate_inheritance(py_class, context)
        rust_items.extend(inheritance_traits)
    END IF

    # Translate nested classes
    FOR nested_class IN py_class.nested_classes DO:
        nested_items = translate_class(nested_class, context)
        rust_items.extend(nested_items)
    END FOR

    RETURN rust_items
END FUNCTION


FUNCTION translate_to_struct(
    py_class: PythonClassDef,
    context: TranslationContext
) -> RustStruct:
    """
    Translate Python class to Rust struct.
    """

    struct_name = sanitize_type_name(py_class.name)
    fields = []

    # Extract fields from __init__ or class attributes
    FOR attribute IN py_class.instance_attributes DO:
        rust_field = RustField {
            name: sanitize_identifier(attribute.name),
            field_type: translate_type_annotation(attribute.type_hint, context),
            visibility: determine_field_visibility(attribute),
            documentation: extract_field_docs(attribute),
        }
        fields.append(rust_field)
    END FOR

    # Add derives based on class behavior
    derives = determine_derives(py_class, context)
    # Common derives: Debug, Clone, PartialEq, Serialize, Deserialize

    # Generate documentation from class docstring
    documentation = translate_docstring(py_class.docstring, context)

    rust_struct = RustStruct {
        name: struct_name,
        fields: fields,
        derives: derives,
        documentation: documentation,
        visibility: Visibility::Public,
        generics: extract_generic_parameters(py_class, context),
    }

    RETURN rust_struct
END FUNCTION
```

### 3.5 Statement Translation

```pseudocode
FUNCTION translate_function_body(
    statements: List<PythonStatement>,
    context: TranslationContext
) -> RustBlock:
    """
    Translate Python function body (list of statements) to Rust block.
    """

    rust_statements = []

    FOR py_stmt IN statements DO:
        rust_stmt = translate_statement(py_stmt, context)
        rust_statements.append(rust_stmt)
    END FOR

    RETURN RustBlock { statements: rust_statements }
END FUNCTION


FUNCTION translate_statement(
    py_stmt: PythonStatement,
    context: TranslationContext
) -> RustStatement:
    """
    Dispatch statement translation based on statement type.
    """

    MATCH py_stmt.type:
        CASE StatementType::Assignment:
            RETURN translate_assignment(py_stmt, context)

        CASE StatementType::AugmentedAssignment:
            RETURN translate_augmented_assignment(py_stmt, context)

        CASE StatementType::Return:
            RETURN translate_return(py_stmt, context)

        CASE StatementType::If:
            RETURN translate_if_statement(py_stmt, context)

        CASE StatementType::While:
            RETURN translate_while_loop(py_stmt, context)

        CASE StatementType::For:
            RETURN translate_for_loop(py_stmt, context)

        CASE StatementType::Try:
            RETURN translate_try_except(py_stmt, context)

        CASE StatementType::Raise:
            RETURN translate_raise(py_stmt, context)

        CASE StatementType::With:
            RETURN translate_with_statement(py_stmt, context)

        CASE StatementType::Assert:
            RETURN translate_assert(py_stmt, context)

        CASE StatementType::Expression:
            expr = translate_expression(py_stmt.expression, context)
            RETURN RustStatement::Expression(expr)

        CASE StatementType::Pass:
            RETURN RustStatement::Empty

        CASE StatementType::Break:
            RETURN RustStatement::Break

        CASE StatementType::Continue:
            RETURN RustStatement::Continue

        DEFAULT:
            THROW Error::UnsupportedStatement(py_stmt.type)
    END MATCH
END FUNCTION
```

### 3.6 Exception Handling Translation

```pseudocode
FUNCTION translate_try_except(
    py_try: PythonTryStatement,
    context: TranslationContext
) -> RustStatement:
    """
    Translate Python try/except to Rust Result-based error handling.

    Strategy:
    - Wrap try block in a closure that returns Result
    - Map except handlers to match arms on Result::Err
    - Translate finally block to proper cleanup
    """

    # Check if we can use simple ? operator propagation
    IF is_simple_propagation(py_try) THEN:
        RETURN translate_simple_error_propagation(py_try, context)
    END IF

    # Complex case: multiple except handlers, finally, else clauses

    # Step 1: Translate try block to Result-returning closure
    try_block_code = translate_function_body(py_try.try_body, context)

    # Wrap in closure
    try_closure = RustClosure {
        parameters: [],
        return_type: "Result<T, Error>",
        body: try_block_code,
    }

    # Step 2: Translate except handlers to match arms
    match_arms = []

    FOR except_handler IN py_try.except_handlers DO:
        # Get exception type(s)
        exception_types = except_handler.exception_types

        # Translate handler body
        handler_body = translate_function_body(except_handler.body, context)

        # Create match arm
        FOR exc_type IN exception_types DO:
            rust_error_pattern = map_exception_to_error_pattern(exc_type, context)

            match_arm = RustMatchArm {
                pattern: "Err({rust_error_pattern})",
                guard: None,
                body: handler_body,
            }
            match_arms.append(match_arm)
        END FOR
    END FOR

    # Add catch-all arm for unhandled exceptions
    IF NOT has_bare_except(py_try) THEN:
        match_arms.append(RustMatchArm {
            pattern: "Err(e)",
            guard: None,
            body: "return Err(e)",
        })
    ELSE:
        # Bare except catches everything
        bare_handler_body = translate_function_body(py_try.bare_except_body, context)
        match_arms.append(RustMatchArm {
            pattern: "Err(_)",
            guard: None,
            body: bare_handler_body,
        })
    END IF

    # Add Ok arm for else clause
    IF py_try.has_else_clause THEN:
        else_body = translate_function_body(py_try.else_body, context)
        match_arms.append(RustMatchArm {
            pattern: "Ok(val)",
            guard: None,
            body: "{else_body}; Ok(val)",
        })
    ELSE:
        match_arms.append(RustMatchArm {
            pattern: "Ok(val)",
            guard: None,
            body: "Ok(val)",
        })
    END IF

    # Step 3: Build match expression
    match_expr = RustMatch {
        scrutinee: "try_closure()",
        arms: match_arms,
    }

    # Step 4: Handle finally clause
    IF py_try.has_finally_clause THEN:
        finally_body = translate_function_body(py_try.finally_body, context)

        # Wrap in RAII guard or explicit try/finally pattern
        result = generate_finally_wrapper(match_expr, finally_body)
        RETURN result
    ELSE:
        RETURN RustStatement::Expression(match_expr)
    END IF
END FUNCTION


FUNCTION translate_raise(
    py_raise: PythonRaiseStatement,
    context: TranslationContext
) -> RustStatement:
    """
    Translate Python raise statement to Rust return Err(...).
    """

    # Handle bare 'raise' (re-raise current exception)
    IF py_raise.exception IS None THEN:
        RETURN RustStatement::Expression("return Err(current_error)")
    END IF

    # Translate exception expression
    exception_expr = translate_expression(py_raise.exception, context)

    # Map to Rust error type
    error_type = determine_error_type_from_exception(py_raise.exception, context)

    # Handle exception chaining (raise ... from ...)
    IF py_raise.has_cause THEN:
        cause_expr = translate_expression(py_raise.cause, context)
        error_construction = "{error_type}::new({exception_expr}).with_source({cause_expr})"
    ELSE:
        error_construction = "{error_type}::new({exception_expr})"
    END IF

    RETURN RustStatement::Expression("return Err({error_construction})")
END FUNCTION
```

### 3.7 Type System Mapping

```pseudocode
FUNCTION translate_type_annotation(
    py_type: PythonTypeAnnotation,
    context: TranslationContext
) -> RustType:
    """
    Translate Python type annotation to Rust type.
    """

    # Handle None type
    IF py_type IS None OR py_type == "None" THEN:
        RETURN "()"  # Unit type
    END IF

    # Handle primitives
    IF py_type IN context.type_registry.primitives THEN:
        RETURN context.type_registry.primitives[py_type]
    END IF

    # Handle generic types
    IF is_generic_type(py_type) THEN:
        RETURN translate_generic_type(py_type, context)
    END IF

    # Handle collections
    IF is_collection_type(py_type) THEN:
        RETURN translate_collection_type(py_type, context)
    END IF

    # Handle user-defined classes
    IF is_user_class(py_type, context) THEN:
        class_name = sanitize_type_name(py_type.name)
        RETURN class_name
    END IF

    # Handle stdlib types
    IF is_stdlib_type(py_type) THEN:
        RETURN context.type_registry.stdlib_mappings[py_type]
    END IF

    # Fallback to Any type (with warning)
    WARN "Unable to precisely map type {py_type}, using dynamic dispatch"
    RETURN "Box<dyn Any>"
END FUNCTION


FUNCTION translate_generic_type(
    py_generic: PythonGenericType,
    context: TranslationContext
) -> RustType:
    """
    Translate Python generic types (Optional, Union, List, etc.) to Rust.
    """

    origin = py_generic.origin
    args = py_generic.args

    MATCH origin:
        CASE "Optional":
            # Optional[T] -> Option<T>
            inner = translate_type_annotation(args[0], context)
            RETURN "Option<{inner}>"

        CASE "Union":
            # Union[T1, T2, ...] -> custom enum
            RETURN translate_union_type(args, context)

        CASE "List" OR "list":
            # List[T] -> Vec<T>
            inner = translate_type_annotation(args[0], context)
            RETURN "Vec<{inner}>"

        CASE "Dict" OR "dict":
            # Dict[K, V] -> HashMap<K, V>
            key_type = translate_type_annotation(args[0], context)
            value_type = translate_type_annotation(args[1], context)
            RETURN "HashMap<{key_type}, {value_type}>"

        CASE "Set" OR "set":
            # Set[T] -> HashSet<T>
            inner = translate_type_annotation(args[0], context)
            RETURN "HashSet<{inner}>"

        CASE "Tuple" OR "tuple":
            # Tuple[T1, T2, ...] -> (T1, T2, ...)
            element_types = [translate_type_annotation(arg, context) FOR arg IN args]
            RETURN "({join(element_types, ', ')})"

        CASE "Callable":
            # Callable[[Args], Ret] -> Fn(Args) -> Ret or Box<dyn Fn(Args) -> Ret>
            RETURN translate_callable_type(args, context)

        CASE "Iterator":
            # Iterator[T] -> impl Iterator<Item = T>
            inner = translate_type_annotation(args[0], context)
            RETURN "impl Iterator<Item = {inner}>"

        DEFAULT:
            # Generic user type
            base_type = sanitize_type_name(origin)
            type_params = [translate_type_annotation(arg, context) FOR arg IN args]
            RETURN "{base_type}<{join(type_params, ', ')}>"
    END MATCH
END FUNCTION


FUNCTION translate_union_type(
    type_args: List<PythonType>,
    context: TranslationContext
) -> RustType:
    """
    Translate Python Union type to Rust enum.
    """

    # Check if this is a common pattern we've seen before
    union_hash = compute_type_hash(type_args)
    IF union_hash IN context.type_mappings THEN:
        RETURN context.type_mappings[union_hash]
    END IF

    # Generate unique enum name
    enum_name = generate_union_enum_name(type_args)

    # Create enum variants
    variants = []
    FOR i, type_arg IN ENUMERATE(type_args) DO:
        variant_name = generate_variant_name(type_arg, i)
        variant_type = translate_type_annotation(type_arg, context)

        variants.append(RustEnumVariant {
            name: variant_name,
            data: variant_type,
        })
    END FOR

    # Create enum definition
    union_enum = RustEnum {
        name: enum_name,
        variants: variants,
        derives: ["Debug", "Clone"],
        visibility: Visibility::Public,
    }

    # Register this enum for later code generation
    context.workspace.shared_types.add_enum(union_enum)

    # Cache mapping
    context.type_mappings[union_hash] = enum_name

    RETURN enum_name
END FUNCTION
```

### 3.8 Workspace Organization

```pseudocode
FUNCTION determine_workspace_layout(
    context: TranslationContext
) -> WorkspaceLayout:
    """
    Determine how to organize Python modules into Rust crates.
    """

    IF context.mode == ExecutionMode::Script THEN:
        # Script mode: single binary crate
        RETURN WorkspaceLayout::SingleBinary {
            crate_name: sanitize_crate_name(context.python_ast.script_name),
        }
    ELSE:
        # Library mode: multi-crate workspace
        RETURN determine_library_layout(context)
    END IF
END FUNCTION


FUNCTION determine_library_layout(
    context: TranslationContext
) -> WorkspaceLayout:
    """
    Organize library into logical crates based on module structure.

    Strategy:
    - Core library crate (main public API)
    - Internal crates for large sub-modules
    - Test crate for integration tests
    - Example binaries for demonstrations
    """

    # Extract package metadata
    package_name = context.python_ast.package_name
    root_module = context.python_ast.root_module

    # Initialize workspace
    layout = WorkspaceLayout::MultiCrate {
        workspace_name: sanitize_crate_name(package_name),
        crates: [],
    }

    # Create main library crate
    main_crate = CrateSpec {
        name: sanitize_crate_name(package_name),
        crate_type: CrateType::Library,
        modules: [],
    }

    # Analyze module tree for crate boundaries
    module_clusters = cluster_modules_by_coupling(
        context.dependency_graph,
        context.api_spec
    )

    FOR cluster IN module_clusters DO:
        IF cluster.size > CRATE_SPLIT_THRESHOLD THEN:
            # Create separate crate for large cluster
            sub_crate = CrateSpec {
                name: "{package_name}_{cluster.name}",
                crate_type: CrateType::Library,
                modules: cluster.modules,
            }
            layout.crates.append(sub_crate)

            # Add dependency from main crate to sub-crate
            main_crate.dependencies[sub_crate.name] = "{ path = \"../{sub_crate.name}\" }"
        ELSE:
            # Include in main crate
            main_crate.modules.extend(cluster.modules)
        END IF
    END FOR

    layout.crates.append(main_crate)

    # Create test crate
    test_crate = CrateSpec {
        name: "{package_name}_tests",
        crate_type: CrateType::Binary,
        modules: extract_test_modules(context.python_ast),
    }
    layout.crates.append(test_crate)

    RETURN layout
END FUNCTION


FUNCTION generate_workspace_cargo_toml(
    workspace: RustWorkspace,
    context: TranslationContext
) -> VOID:
    """
    Generate root Cargo.toml for workspace.
    """

    cargo_toml = CargoToml::new()

    # Workspace section
    cargo_toml.add_section("workspace", {
        members: [crate.name FOR crate IN workspace.crates],
        resolver: "2",
    })

    # Workspace-wide dependencies (shared versions)
    shared_deps = extract_common_dependencies(workspace)
    cargo_toml.add_section("workspace.dependencies", shared_deps)

    # Workspace-wide profile settings
    cargo_toml.add_section("profile.release", {
        opt_level: 3,
        lto: "fat",
        codegen_units: 1,
        panic: "abort",
    })

    # WASM-specific profile
    IF "WASM" IN context.target_features THEN:
        cargo_toml.add_section("profile.wasm-release", {
            inherits: "release",
            opt_level: "z",  # Optimize for size
            strip: "symbols",
        })
    END IF

    # Write to file
    workspace.workspace_cargo_toml = cargo_toml
    write_file(workspace.root_path / "Cargo.toml", cargo_toml.to_string())
END FUNCTION


FUNCTION generate_crate_cargo_toml(
    crate: RustCrate,
    context: TranslationContext
) -> VOID:
    """
    Generate Cargo.toml for individual crate.
    """

    cargo_toml = CargoToml::new()

    # Package metadata
    cargo_toml.add_section("package", {
        name: crate.name,
        version: "0.1.0",
        edition: "2021",
        authors: ["Portalis Transpiler <generated@portalis.ai>"],
    })

    # Library or binary configuration
    IF crate.crate_type == CrateType::Library THEN:
        cargo_toml.add_section("lib", {
            name: crate.name,
            path: "src/lib.rs",
            crate_type: ["cdylib", "rlib"] IF "WASM" IN context.target_features ELSE ["rlib"],
        })
    END IF

    # Dependencies
    cargo_toml.add_section("dependencies", crate.dependencies)

    # Dev dependencies (for tests)
    cargo_toml.add_section("dev-dependencies", crate.dev_dependencies)

    # WASM-specific target configuration
    IF "WASM" IN context.target_features THEN:
        cargo_toml.add_section("target.'cfg(target_arch = \"wasm32\")'", {
            dependencies: {
                wasm_bindgen: "0.2",
                wasi: "0.11",
            }
        })
    END IF

    # Features
    features = determine_cargo_features(crate, context)
    IF NOT is_empty(features) THEN:
        cargo_toml.add_section("features", features)
    END IF

    # Write to file
    crate.cargo_toml = cargo_toml
    write_file(crate.path / "Cargo.toml", cargo_toml.to_string())
END FUNCTION
```

### 3.9 GPU-Accelerated Batch Translation

```pseudocode
FUNCTION gpu_batch_translate_modules(
    modules: List<PythonModule>,
    context: TranslationContext
) -> List<RustModule>:
    """
    Leverage GPU for parallel translation of multiple modules.

    Strategy:
    1. Compute embeddings for all Python functions on GPU
    2. Perform k-NN search to find similar translated examples
    3. Use NeMo for batch code generation
    4. Rank and select best translations in parallel
    """

    # Step 1: Extract all functions from all modules
    all_functions = []
    function_to_module = {}

    FOR module IN modules DO:
        FOR function IN module.functions DO:
            all_functions.append(function)
            function_to_module[function.id] = module
        END FOR
    END FOR

    # Step 2: Compute embeddings in batch on GPU
    function_embeddings = compute_embeddings_gpu(all_functions, context.cuda_context)
    # Shape: [num_functions, embedding_dim]

    # Step 3: Search for similar examples in translation cache
    k = 5  # Number of similar examples to retrieve
    similar_examples = gpu_knn_search(
        query_embeddings=function_embeddings,
        database_embeddings=context.embedding_cache.rust_embeddings,
        k=k,
        cuda_context=context.cuda_context
    )
    # Shape: [num_functions, k, (index, distance)]

    # Step 4: Build translation prompts with few-shot examples
    translation_prompts = []
    FOR i, function IN ENUMERATE(all_functions) DO:
        examples = get_examples_from_indices(similar_examples[i], context)
        prompt = build_translation_prompt(function, examples)
        translation_prompts.append(prompt)
    END FOR

    # Step 5: Batch generate Rust code using NeMo
    rust_codes = nemo_batch_generate(
        prompts=translation_prompts,
        model=context.nemo_model,
        batch_size=32,
        max_length=512,
        temperature=0.2,  # Low temperature for deterministic code
    )
    # Returns: List[List[RustCode]], multiple candidates per function

    # Step 6: Rank candidates using GPU-accelerated semantic similarity
    best_translations = []
    FOR i, candidates IN ENUMERATE(rust_codes) DO:
        # Compute embeddings for all candidates
        candidate_embeddings = compute_embeddings_gpu(candidates, context.cuda_context)

        # Compute similarity to known good translations
        similarity_scores = compute_similarity_matrix_gpu(
            candidate_embeddings,
            context.embedding_cache.rust_embeddings,
            context.cuda_context
        )

        # Select best candidate
        best_idx = argmax(mean(similarity_scores, axis=1))
        best_translations.append(candidates[best_idx])
    END FOR

    # Step 7: Assemble translated modules
    translated_modules = {}
    FOR i, function IN ENUMERATE(all_functions) DO:
        module = function_to_module[function.id]
        IF module NOT IN translated_modules THEN:
            translated_modules[module] = RustModule::new(module.name)
        END IF

        rust_function = parse_rust_function(best_translations[i])
        translated_modules[module].items.append(RustItem::Function(rust_function))
    END FOR

    RETURN list(translated_modules.values())
END FUNCTION


FUNCTION compute_embeddings_gpu(
    code_items: List<CodeItem>,
    cuda_context: CUDAContext
) -> CUDAMatrix<f32>:
    """
    Compute embeddings for code items using GPU-accelerated model.
    """

    # Tokenize code
    tokens = [tokenize(item.source_code) FOR item IN code_items]

    # Pad/truncate to fixed length
    max_length = 512
    padded_tokens = pad_sequences(tokens, max_length)

    # Transfer to GPU
    token_tensor = cuda_tensor_from_host(padded_tokens, cuda_context)

    # Run embedding model on GPU
    embedding_tensor = cuda_context.embedding_model.forward(token_tensor)
    # Shape: [batch_size, embedding_dim]

    RETURN embedding_tensor
END FUNCTION


FUNCTION gpu_knn_search(
    query_embeddings: CUDAMatrix<f32>,
    database_embeddings: CUDAMatrix<f32>,
    k: int,
    cuda_context: CUDAContext
) -> CUDAMatrix<(int, f32)>:
    """
    Perform k-nearest neighbor search on GPU.
    """

    # Compute pairwise distances using GPU matrix multiplication
    # distances[i, j] = ||query[i] - database[j]||^2
    distances = cuda_pairwise_euclidean_distance(
        query_embeddings,
        database_embeddings,
        cuda_context
    )
    # Shape: [num_queries, num_database]

    # Find k smallest distances per query using GPU sorting
    indices, scores = cuda_topk(distances, k, largest=False)
    # Shapes: [num_queries, k], [num_queries, k]

    # Combine indices and scores
    results = cuda_stack([indices, scores], dim=2)
    # Shape: [num_queries, k, 2]

    RETURN results
END FUNCTION
```

### 3.10 Template-Based Code Generation

```pseudocode
FUNCTION apply_template(
    template: TranslationTemplate,
    matched_nodes: PythonASTNodes,
    context: TranslationContext
) -> RustCode:
    """
    Apply a code generation template to matched Python AST nodes.
    """

    # Extract template variables from matched nodes
    template_vars = extract_template_variables(matched_nodes, template)

    # Recursively translate nested expressions
    FOR var_name, var_value IN template_vars DO:
        IF is_ast_node(var_value) THEN:
            template_vars[var_name] = translate_expression(var_value, context)
        ELSIF is_type_annotation(var_value) THEN:
            template_vars[var_name] = translate_type_annotation(var_value, context)
        END IF
    END FOR

    # Generate Rust code using template generator function
    rust_code = template.rust_generator(template_vars, context)

    # Add required imports
    FOR import IN template.required_imports DO:
        context.current_module.imports.add(import)
    END FOR

    # Add required cargo features
    FOR feature IN template.required_features DO:
        context.current_crate.cargo_features.add(feature)
    END FOR

    RETURN rust_code
END FUNCTION


# Example template for list comprehensions
TEMPLATE list_comprehension_template:
    PATTERN:
        # [expr(item) for item in iterable if condition(item)]
        ListComp {
            element: expr,
            generators: [
                Comprehension {
                    target: item,
                    iter: iterable,
                    ifs: [condition]
                }
            ]
        }

    GENERATOR FUNCTION:
        FUNCTION generate(vars, context):
            expr = vars["expr"]
            item = vars["item"]
            iterable = vars["iterable"]
            condition = vars["condition"]

            # Determine element type
            element_type = infer_type(expr, context)

            rust_code = """
                {iterable}
                    .iter()
                    .filter(|{item}| {condition})
                    .map(|{item}| {expr})
                    .collect::<Vec<{element_type}>>()
            """

            RETURN rust_code
        END FUNCTION

    REQUIRED_IMPORTS: []
    REQUIRED_FEATURES: []
END TEMPLATE


# Example template for async/await
TEMPLATE async_await_template:
    PATTERN:
        # await async_expr
        Await { value: async_expr }

    GENERATOR FUNCTION:
        FUNCTION generate(vars, context):
            async_expr = vars["async_expr"]

            # Check if we're in an async context
            IF NOT context.current_function.is_async THEN:
                THROW Error::AwaitOutsideAsyncContext
            END IF

            # Simple translation
            rust_code = "{async_expr}.await"

            RETURN rust_code
        END FUNCTION

    REQUIRED_IMPORTS: []
    REQUIRED_FEATURES: []
END TEMPLATE


# Example template for context managers
TEMPLATE with_statement_template:
    PATTERN:
        # with context_expr as var:
        #     body
        With {
            items: [WithItem { context_expr, optional_vars: var }],
            body: body_statements
        }

    GENERATOR FUNCTION:
        FUNCTION generate(vars, context):
            context_expr = vars["context_expr"]
            var = vars["var"]
            body = vars["body_statements"]

            # Determine if we need RAII guard or explicit drop
            IF supports_drop_trait(context_expr, context) THEN:
                # Use RAII pattern
                rust_code = """
                    {{
                        let {var} = {context_expr};
                        {body}
                    }}
                """
            ELSE:
                # Explicit try/finally pattern
                rust_code = """
                    {{
                        let mut {var} = {context_expr}.__enter__();
                        let result = (|| {{ {body} }})();
                        {var}.__exit__(None, None, None);
                        result
                    }}
                """
            END IF

            RETURN rust_code
        END FUNCTION

    REQUIRED_IMPORTS: []
    REQUIRED_FEATURES: []
END TEMPLATE
```

---

## 4. Input/Output Contracts

### 4.1 Agent Input Contract

```pseudocode
INTERFACE TranspilerAgentInput:
    # From Analysis Agent
    python_ast: PythonAST
    api_spec: APISpec
    dependency_graph: DependencyGraph
    contract_map: ContractMap

    # From Specification Generator Agent
    rust_spec: RustSpecification
    type_mappings: TypeMappingHints
    error_types: ErrorTypeDefinitions

    # Configuration
    mode: ExecutionMode  # Script | Library
    optimization_level: OptLevel  # Debug | Release
    gpu_enabled: bool
    target_features: Set<Feature>  # [WASM, NIM, Omniverse]
    output_dir: Path

    # Optional
    custom_templates: Option<List<TranslationTemplate>>
    type_mapping_overrides: Option<Map<PythonType, RustType>>
END INTERFACE
```

### 4.2 Agent Output Contract

```pseudocode
INTERFACE TranspilerAgentOutput:
    # Success case
    status: "success" | "partial_success" | "failure"

    # Generated artifacts
    workspace: RustWorkspace
    translation_report: TranslationReport

    # Metadata
    metrics: TranslationMetrics
    warnings: List<Warning>
    errors: List<Error>
END INTERFACE


STRUCTURE TranslationReport:
    total_modules: int
    translated_modules: int
    total_functions: int
    translated_functions: int
    total_classes: int
    translated_classes: int

    unsupported_features: List<UnsupportedFeature>
    translation_confidence: Map<ItemPath, f32>  # Confidence scores

    generated_files: List<Path>
    source_mappings: Map<RustPath, PythonPath>
END STRUCTURE


STRUCTURE TranslationMetrics:
    translation_time_ms: u64
    gpu_time_ms: u64
    cpu_time_ms: u64

    lines_of_python: u64
    lines_of_rust: u64

    gpu_utilization: f32  # 0.0 to 1.0
    memory_peak_mb: u64

    cache_hit_rate: f32
    template_coverage: f32  # % of code generated via templates
END STRUCTURE
```

### 4.3 Sub-Component Interfaces

```pseudocode
INTERFACE SyntaxTransformer:
    FUNCTION transform_ast(
        python_ast: PythonAST,
        context: TranslationContext
    ) -> Result<RustAST, Error>
END INTERFACE


INTERFACE SemanticAdapter:
    FUNCTION adapt_semantics(
        rust_ast: RustAST,
        python_semantics: SemanticModel,
        context: TranslationContext
    ) -> Result<RustAST, Error>
END INTERFACE


INTERFACE WorkspaceOrganizer:
    FUNCTION organize_workspace(
        modules: List<RustModule>,
        layout: WorkspaceLayout,
        context: TranslationContext
    ) -> Result<RustWorkspace, Error>
END INTERFACE


INTERFACE ErrorTranslator:
    FUNCTION translate_exceptions(
        python_exception_tree: ExceptionHierarchy,
        context: TranslationContext
    ) -> Result<RustErrorEnums, Error>
END INTERFACE


INTERFACE TypeMapper:
    FUNCTION map_type(
        python_type: PythonType,
        context: TranslationContext
    ) -> Result<RustType, Error>
END INTERFACE


INTERFACE TemplateEngine:
    FUNCTION match_template(
        ast_node: PythonASTNode
    ) -> Option<TranslationTemplate>

    FUNCTION apply_template(
        template: TranslationTemplate,
        ast_node: PythonASTNode,
        context: TranslationContext
    ) -> Result<RustCode, Error>
END INTERFACE


INTERFACE CUDAAccelerator:
    FUNCTION batch_translate(
        items: List<PythonItem>,
        context: TranslationContext
    ) -> Result<List<RustItem>, Error>

    FUNCTION compute_embeddings(
        code_items: List<CodeItem>
    ) -> Result<CUDAMatrix<f32>, Error>

    FUNCTION rank_candidates(
        candidates: List<List<RustCode>>,
        reference_embeddings: CUDAMatrix<f32>
    ) -> Result<List<RustCode>, Error>
END INTERFACE
```

---

## 5. CUDA Integration Points

### 5.1 GPU-Accelerated Operations

```pseudocode
# FR-2.4.5: CUDA Acceleration for Translation

CUDA_INTEGRATION_POINT_1: "Batch AST Parsing"
    LOCATION: Module processing loop
    OPERATION: Parse multiple Python ASTs in parallel
    EXPECTED_SPEEDUP: 5-10x for large codebases
    IMPLEMENTATION:
        - Transfer AST tokens to GPU memory
        - Launch parallel parsing kernels
        - Collect parsed ASTs back to CPU


CUDA_INTEGRATION_POINT_2: "Embedding Similarity Search"
    LOCATION: Template matching, candidate selection
    OPERATION: k-NN search for similar code patterns
    EXPECTED_SPEEDUP: 100-1000x for large databases
    IMPLEMENTATION:
        - Maintain embedding database in GPU memory
        - Use GPU-accelerated k-NN libraries (cuML, FAISS-GPU)
        - Return top-k matches for few-shot learning


CUDA_INTEGRATION_POINT_3: "Batch Code Generation"
    LOCATION: NeMo translation step
    OPERATION: Generate Rust code for multiple functions in parallel
    EXPECTED_SPEEDUP: 10-50x with batching
    IMPLEMENTATION:
        - Batch prompts for NeMo inference
        - Use tensor parallelism for large models
        - Stream results back as they complete


CUDA_INTEGRATION_POINT_4: "Translation Candidate Ranking"
    LOCATION: Post-generation validation
    OPERATION: Rank multiple Rust candidates by semantic similarity
    EXPECTED_SPEEDUP: 10-20x
    IMPLEMENTATION:
        - Compute embeddings for all candidates on GPU
        - Parallel similarity computation
        - GPU-based sorting/ranking


CUDA_INTEGRATION_POINT_5: "Type Inference Acceleration"
    LOCATION: Type mapping phase
    OPERATION: Parallel type constraint solving
    EXPECTED_SPEEDUP: 5-10x for complex type systems
    IMPLEMENTATION:
        - Represent type constraints as graph on GPU
        - Parallel graph algorithms for inference
        - Return inferred types
```

### 5.2 Memory Management

```pseudocode
STRUCTURE CUDAMemoryStrategy:
    # Embedding cache persistence
    persistent_embeddings: PinnedMemory  # Python/Rust code embeddings
    persistent_size_mb: 2048  # 2 GB reserved

    # Streaming buffers
    input_stream_buffer: CUDAStream  # For incoming Python ASTs
    output_stream_buffer: CUDAStream  # For generated Rust code
    buffer_size_mb: 512

    # Computation workspace
    workspace_memory: CUDAMemoryPool  # Temporary allocations
    workspace_size_mb: 4096  # 4 GB

    # Overflow strategy
    overflow_policy: "SpillToHost"  # Fallback to CPU if GPU memory full
END STRUCTURE
```

### 5.3 Error Handling and Fallback

```pseudocode
FUNCTION gpu_operation_with_fallback(
    operation: Fn() -> Result<T, CUDAError>,
    cpu_fallback: Fn() -> Result<T, Error>
) -> Result<T, Error>:
    """
    Execute GPU operation with CPU fallback on failure.
    """

    TRY:
        result = operation()
        RETURN Ok(result)
    CATCH error AS CUDAError:
        IF error.type == CUDAErrorType::OutOfMemory THEN:
            WARN "GPU out of memory, falling back to CPU"
            RETURN cpu_fallback()
        ELSIF error.type == CUDAErrorType::DeviceNotFound THEN:
            WARN "No GPU detected, using CPU"
            RETURN cpu_fallback()
        ELSE:
            # Critical GPU error, propagate
            RETURN Err(Error::CUDAFailed(error))
        END IF
    END TRY
END FUNCTION
```

---

## 6. London School TDD Test Points

### 6.1 Outside-In Test Strategy

```pseudocode
# Acceptance Test (Highest Level)
TEST "Transpiler Agent translates simple Python script to Rust":
    # Arrange
    mock_analysis_output = create_mock_analysis_output()
    mock_spec_output = create_mock_spec_output()
    expected_workspace = create_expected_workspace()

    transpiler = TranspilerAgent::new()

    # Act
    result = transpiler.transpile_codebase(mock_analysis_output, mock_spec_output)

    # Assert
    ASSERT result.is_ok()
    ASSERT result.workspace.crates.length == 1
    ASSERT workspace_matches(result.workspace, expected_workspace)

    # Verify mocks called correctly
    VERIFY mock_analysis_output.api_spec was accessed
    VERIFY mock_spec_output.rust_spec was accessed
END TEST


# Contract Test (Agent Interface)
TEST "Transpiler Agent adheres to input/output contract":
    # Arrange
    valid_input = create_valid_transpiler_input()
    transpiler = TranspilerAgent::new()

    # Act
    result = transpiler.transpile_codebase(valid_input)

    # Assert - verify output structure
    ASSERT result has field "status"
    ASSERT result has field "workspace"
    ASSERT result has field "metrics"
    ASSERT result has field "warnings"
    ASSERT result has field "errors"

    # Assert - verify field types
    ASSERT result.status IN ["success", "partial_success", "failure"]
    ASSERT result.workspace IS RustWorkspace
    ASSERT result.metrics IS TranslationMetrics
END TEST
```

### 6.2 Component-Level Tests (Test Doubles)

```pseudocode
# Mock for SyntaxTransformer
MOCK SyntaxTransformerMock IMPLEMENTS SyntaxTransformer:
    expected_calls: List<(PythonAST, RustAST)>
    call_count: int = 0

    FUNCTION transform_ast(python_ast, context):
        ASSERT call_count < length(expected_calls)
        expected_input, canned_output = expected_calls[call_count]
        ASSERT python_ast == expected_input
        call_count += 1
        RETURN Ok(canned_output)
    END FUNCTION

    FUNCTION verify():
        ASSERT call_count == length(expected_calls), "Not all expected calls made"
    END FUNCTION
END MOCK


# Stub for TypeMapper
STUB TypeMapperStub IMPLEMENTS TypeMapper:
    mapping: Map<PythonType, RustType>

    FUNCTION map_type(python_type, context):
        IF python_type IN mapping THEN:
            RETURN Ok(mapping[python_type])
        ELSE:
            RETURN Err(Error::UnknownType(python_type))
        END IF
    END FUNCTION
END STUB


# Fake for CUDAAccelerator (deterministic, no real GPU)
FAKE CUDAAcceleratorFake IMPLEMENTS CUDAAccelerator:
    translation_database: Map<PythonItem, RustItem>

    FUNCTION batch_translate(items, context):
        results = []
        FOR item IN items DO:
            IF item IN translation_database THEN:
                results.append(translation_database[item])
            ELSE:
                results.append(generate_dummy_translation(item))
            END IF
        END FOR
        RETURN Ok(results)
    END FUNCTION

    FUNCTION compute_embeddings(code_items):
        # Return dummy embeddings
        num_items = length(code_items)
        embedding_dim = 768
        dummy_matrix = zeros([num_items, embedding_dim])
        RETURN Ok(dummy_matrix)
    END FUNCTION
END FAKE
```

### 6.3 Unit Tests (Pure Logic)

```pseudocode
TEST "translate_type_annotation maps int to i64":
    # Arrange
    py_type = PythonPrimitive::Int
    context = create_empty_context()

    # Act
    rust_type = translate_type_annotation(py_type, context)

    # Assert
    ASSERT rust_type == "i64"
END TEST


TEST "translate_type_annotation maps Optional[str] to Option<String>":
    # Arrange
    py_type = PythonGenericType {
        origin: "Optional",
        args: [PythonPrimitive::Str]
    }
    context = create_empty_context()

    # Act
    rust_type = translate_type_annotation(py_type, context)

    # Assert
    ASSERT rust_type == "Option<String>"
END TEST


TEST "translate_try_except with single handler generates match expression":
    # Arrange
    py_try = PythonTryStatement {
        try_body: [PythonStatement::Return(PythonExpr::IntLiteral(42))],
        except_handlers: [
            ExceptHandler {
                exception_types: [PythonException::ValueError],
                body: [PythonStatement::Return(PythonExpr::IntLiteral(0))]
            }
        ],
        else_body: [],
        finally_body: []
    }
    context = create_empty_context()

    # Act
    rust_stmt = translate_try_except(py_try, context)

    # Assert
    ASSERT rust_stmt IS RustStatement::Expression
    ASSERT rust_stmt.expr IS RustMatch
    ASSERT length(rust_stmt.expr.arms) == 2  # Err(ValueError), Ok(val)
END TEST


TEST "generate_workspace_layout creates single binary for script mode":
    # Arrange
    context = create_context_with_mode(ExecutionMode::Script)
    context.python_ast.script_name = "hello.py"

    # Act
    layout = determine_workspace_layout(context)

    # Assert
    ASSERT layout IS WorkspaceLayout::SingleBinary
    ASSERT layout.crate_name == "hello"
END TEST


TEST "generate_workspace_layout creates multi-crate for library mode":
    # Arrange
    context = create_context_with_mode(ExecutionMode::Library)
    context.python_ast.package_name = "my_package"
    # Mock large dependency graph
    context.dependency_graph = create_large_dependency_graph()

    # Act
    layout = determine_workspace_layout(context)

    # Assert
    ASSERT layout IS WorkspaceLayout::MultiCrate
    ASSERT layout.workspace_name == "my_package"
    ASSERT length(layout.crates) > 1  # Should split into multiple crates
END TEST
```

### 6.4 Integration Tests

```pseudocode
TEST "Full translation pipeline for simple Python function":
    # Arrange
    python_code = """
    def add(a: int, b: int) -> int:
        return a + b
    """
    python_ast = parse_python(python_code)

    # Create real components (not mocks)
    syntax_transformer = SyntaxTransformer::new()
    type_mapper = TypeMapper::new()
    template_engine = TemplateEngine::new()

    context = TranslationContext {
        python_ast: python_ast,
        mode: ExecutionMode::Script,
        # ... other fields
    }

    # Act
    rust_function = translate_function(python_ast.functions[0], context)

    # Assert
    ASSERT rust_function.name == "add"
    ASSERT length(rust_function.signature.parameters) == 2
    ASSERT rust_function.signature.return_type == "i64"

    # Verify generated code compiles
    rust_code = generate_rust_code(rust_function)
    compilation_result = compile_rust_code(rust_code)
    ASSERT compilation_result.is_ok()
END TEST


TEST "Translation with CUDA acceleration produces same result as CPU":
    # Arrange
    python_modules = create_test_modules(num_modules=10)

    context_cpu = create_context(gpu_enabled=False)
    context_gpu = create_context(gpu_enabled=True)

    # Act
    result_cpu = gpu_batch_translate_modules(python_modules, context_cpu)
    result_gpu = gpu_batch_translate_modules(python_modules, context_gpu)

    # Assert - results should be functionally equivalent
    ASSERT length(result_cpu) == length(result_gpu)
    FOR i IN 0..length(result_cpu) DO:
        ASSERT modules_functionally_equal(result_cpu[i], result_gpu[i])
    END FOR
END TEST
```

### 6.5 Property-Based Tests

```pseudocode
TEST "Property: All generated Rust code is syntactically valid":
    # Use property-based testing framework (e.g., proptest)

    PROPERTY FOR ALL python_function IN arbitrary_python_functions():
        # Arrange
        context = create_test_context()

        # Act
        TRY:
            rust_function = translate_function(python_function, context)
            rust_code = generate_rust_code(rust_function)

            # Assert
            syntax_check_result = rust_syntax_checker(rust_code)
            ASSERT syntax_check_result.is_valid
        CATCH error:
            # Translation may fail for unsupported features, but should not crash
            ASSERT error IS Error::UnsupportedFeature
        END TRY
    END PROPERTY
END TEST


TEST "Property: Type mappings are consistent and bijective (where applicable)":
    PROPERTY FOR ALL python_type IN arbitrary_python_types():
        context = create_test_context()

        # Act
        rust_type = translate_type_annotation(python_type, context)

        # Assert - basic invariants
        ASSERT NOT is_empty(rust_type)
        ASSERT is_valid_rust_type(rust_type)

        # If we translate the same type twice, results should match
        rust_type2 = translate_type_annotation(python_type, context)
        ASSERT rust_type == rust_type2
    END PROPERTY
END TEST
```

### 6.6 Error Path Tests

```pseudocode
TEST "Transpiler handles unsupported Python feature gracefully":
    # Arrange
    python_code = """
    class Meta(type):
        pass  # Metaclass - unsupported
    """
    context = create_test_context()

    # Act
    result = transpile_codebase(python_code, context)

    # Assert
    ASSERT result.status == "partial_success" OR result.status == "failure"
    ASSERT length(result.errors) > 0
    ASSERT any(error.type == ErrorType::UnsupportedFeature FOR error IN result.errors)
    ASSERT result.errors[0].message CONTAINS "metaclass"
END TEST


TEST "Transpiler falls back to CPU when GPU unavailable":
    # Arrange
    python_modules = create_test_modules(num_modules=5)
    context = create_context(gpu_enabled=True)

    # Inject GPU failure
    inject_cuda_error(CUDAErrorType::DeviceNotFound)

    # Act
    result = gpu_batch_translate_modules(python_modules, context)

    # Assert - should succeed via CPU fallback
    ASSERT result.is_ok()
    ASSERT length(result.warnings) > 0
    ASSERT any(warning.message CONTAINS "falling back to CPU" FOR warning IN result.warnings)
END TEST


TEST "Transpiler reports circular dependencies":
    # Arrange
    python_modules = create_circular_dependency_modules()
    context = create_test_context()

    # Act
    result = transpile_codebase(python_modules, context)

    # Assert
    ASSERT result.status == "failure"
    ASSERT any(error.type == ErrorType::CircularDependency FOR error IN result.errors)
END TEST
```

---

## 7. Algorithm Complexity Analysis

### 7.1 Time Complexity

```pseudocode
OPERATION: transpile_codebase
    COMPLEXITY: O(N * M + E)
    WHERE:
        N = number of modules
        M = average number of items per module (functions, classes)
        E = number of edges in dependency graph

    BREAKDOWN:
        - Topological sort: O(N + E)
        - Module translation: O(N * M)
        - Workspace organization: O(N)
        - CUDA overhead (if enabled): O(M * log(D))
            WHERE D = size of translation database


OPERATION: translate_single_module
    COMPLEXITY: O(M)
    WHERE M = number of items in module

    BREAKDOWN:
        - Import translation: O(I) where I = number of imports
        - Item translation: O(M)
        - Documentation: O(M)


OPERATION: translate_function
    COMPLEXITY: O(S + P)
    WHERE:
        S = number of statements in function body
        P = number of parameters

    BREAKDOWN:
        - Signature translation: O(P)
        - Body translation: O(S)
        - Nested complexity for control flow: O(S * depth)


OPERATION: gpu_batch_translate_modules
    COMPLEXITY: O(F * E_dim + F * log(D) + B * L)
    WHERE:
        F = total number of functions
        E_dim = embedding dimensionality
        D = size of example database
        B = batch size for NeMo
        L = max sequence length

    BREAKDOWN:
        - Embedding computation: O(F * E_dim) on GPU
        - k-NN search: O(F * log(D)) on GPU
        - NeMo generation: O((F/B) * B * L) = O(F * L)
        - Candidate ranking: O(F * k * E_dim) on GPU
```

### 7.2 Space Complexity

```pseudocode
SPACE: TranslationContext
    COMPLEXITY: O(N * M + E + T + G)
    WHERE:
        N = number of modules
        M = average items per module
        E = edges in dependency graph
        T = size of type mapping registry
        G = GPU memory for embeddings (if enabled)

    BREAKDOWN:
        - AST storage: O(N * M)
        - Dependency graph: O(N + E)
        - Type mappings: O(T)
        - GPU embeddings: O((N * M) * E_dim) on GPU
        - Translation cache: O(D * E_dim) on GPU


SPACE: RustWorkspace
    COMPLEXITY: O(N * M)
    WHERE:
        N = number of modules
        M = average items per module

    BREAKDOWN:
        - Rust modules: O(N * M)
        - Cargo configs: O(C) where C = number of crates
        - Documentation: O(N * M)
```

---

## 8. Extensibility Points

### 8.1 Custom Translation Templates

```pseudocode
EXTENSION_POINT: "Custom Template Registration"

    INTERFACE:
        FUNCTION register_template(
            template: TranslationTemplate
        ) -> Result<(), Error>

    USE_CASE:
        - Domain-specific Python patterns
        - Project-specific idioms
        - Optimization for common patterns

    EXAMPLE:
        custom_template = TranslationTemplate {
            name: "NumPy array to ndarray crate",
            pattern: match_numpy_array_pattern,
            rust_generator: generate_ndarray_code,
            confidence_score: 0.95,
        }

        transpiler.register_template(custom_template)
END EXTENSION_POINT


EXTENSION_POINT: "Type Mapping Override"

    INTERFACE:
        FUNCTION override_type_mapping(
            python_type: PythonType,
            rust_type: RustType
        ) -> Result<(), Error>

    USE_CASE:
        - Map to specialized Rust crates
        - Performance-critical type choices
        - ABI compatibility requirements

    EXAMPLE:
        # Use smallvec instead of Vec for small lists
        transpiler.override_type_mapping(
            PythonType::List(PythonPrimitive::Int),
            "SmallVec<[i64; 8]>"
        )
END EXTENSION_POINT


EXTENSION_POINT: "Custom Error Type Generation"

    INTERFACE:
        FUNCTION customize_error_translation(
            exception_class: PythonClass,
            error_generator: Fn(PythonClass) -> RustEnum
        ) -> Result<(), Error>

    USE_CASE:
        - Preserve custom error hierarchies
        - Map to existing Rust error types
        - Integrate with error handling frameworks
END EXTENSION_POINT
```

---

## 9. Performance Optimization Strategies

### 9.1 Caching

```pseudocode
OPTIMIZATION: "Translation Cache"

    STRATEGY:
        - Cache AST hash -> RustCode mappings
        - Persist cache across runs
        - Invalidate on Python source changes

    IMPLEMENTATION:
        cache_key = hash(python_ast_node)
        IF cache_key IN translation_cache THEN:
            RETURN translation_cache[cache_key]
        ELSE:
            rust_code = perform_translation(python_ast_node)
            translation_cache[cache_key] = rust_code
            RETURN rust_code
        END IF

    EXPECTED_IMPROVEMENT: 2-5x for incremental builds
END OPTIMIZATION


OPTIMIZATION: "Embedding Precomputation"

    STRATEGY:
        - Precompute embeddings for standard library patterns
        - Load into GPU memory at startup
        - Amortize embedding cost across runs

    EXPECTED_IMPROVEMENT: 10-20% overall speedup
END OPTIMIZATION
```

### 9.2 Parallelization

```pseudocode
OPTIMIZATION: "Module-Level Parallelism"

    STRATEGY:
        - Translate independent modules in parallel
        - Use thread pool for CPU-bound work
        - Use GPU for data-parallel operations

    IMPLEMENTATION:
        independent_modules = find_independent_modules(dependency_graph)

        # CPU parallelism
        parallel_map(independent_modules, |module| {
            translate_single_module(module, context)
        })

        # GPU parallelism
        gpu_batch_translate_modules(independent_modules, context)

    EXPECTED_IMPROVEMENT: N-x where N = number of CPU cores
END OPTIMIZATION
```

---

## Document Status

**SPARC Phase 2 (Pseudocode): COMPLETE for Transpiler Agent**

This pseudocode document provides:
- Complete data structure definitions for translation context and outputs
- Detailed algorithms for all major translation operations
- GPU acceleration integration points with CUDA
- Comprehensive test strategy following London School TDD
- Input/output contracts at all levels
- Extensibility mechanisms for customization
- Performance optimization strategies

### Coverage

- FR-2.4.1: Rust Code Generation - COVERED
- FR-2.4.2: Multi-Crate Workspace Generation - COVERED
- FR-2.4.3: Error Handling Translation - COVERED
- FR-2.4.4: Type System Mapping - COVERED
- FR-2.4.5: CUDA Acceleration - COVERED

### Next Steps

1. Review and validate pseudocode with stakeholders
2. Proceed to SPARC Phase 3 (Architecture) for detailed component design
3. Begin London School TDD implementation starting with acceptance tests
4. Implement CUDA integration points with performance benchmarking

---

**END OF PSEUDOCODE DOCUMENT**
