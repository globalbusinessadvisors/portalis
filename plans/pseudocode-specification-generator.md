# SPECIFICATION GENERATOR AGENT (NeMo-Driven)
## SPARC Phase 2: Pseudocode

**Agent ID:** `SpecificationGenerator`
**Version:** 1.0
**Date:** 2025-10-03
**SPARC Phase:** 2 (Pseudocode)
**Dependencies:** Analysis Agent (upstream), Transpiler Agent (downstream)

---

## Table of Contents

1. [Agent Overview](#1-agent-overview)
2. [Data Structures](#2-data-structures)
3. [Core Algorithms](#3-core-algorithms)
4. [Input/Output Contracts](#4-inputoutput-contracts)
5. [NeMo Integration Strategy](#5-nemo-integration-strategy)
6. [London School TDD Test Points](#6-london-school-tdd-test-points)
7. [Error Handling](#7-error-handling)
8. [Performance Considerations](#8-performance-considerations)

---

## 1. Agent Overview

### 1.1 Purpose

The Specification Generator Agent is responsible for transforming analyzed Python API surfaces into comprehensive Rust interface specifications. It leverages NVIDIA NeMo language models to make intelligent translation decisions, generating Rust trait definitions, type mappings, ownership semantics, and WASI-compatible ABI contracts.

### 1.2 Responsibilities

**Primary Responsibilities (from FR-2.3.x):**
- **FR-2.3.1**: Rust Interface Synthesis
  - Generate Rust trait definitions from Python classes
  - Map Python types to Rust types
  - Specify ownership/borrowing semantics
  - Define error types using Result<T, E>
  - Specify lifetime annotations
- **FR-2.3.2**: ABI Contract Definition
  - Define WASI-compatible ABI for exported functions
  - Specify FFI-safe types for WASM boundaries
  - Document memory allocation/deallocation contracts
  - Define serialization formats for complex types
- **FR-2.3.3**: NeMo Translation Spec
  - Leverage NeMo for Python-to-Rust mapping
  - Generate structured translation specifications (JSON/YAML)
  - Include confidence scores
  - Provide ranked translation alternatives

### 1.3 Agent Lifecycle

```
INPUT: AnalysisResult
    ↓
[1] Initialize NeMo inference session
    ↓
[2] Process Python API surface (functions, classes, types)
    ↓
[3] For each Python construct:
    ├─> Query NeMo for translation strategies
    ├─> Rank alternatives by confidence
    ├─> Select optimal Rust mapping
    └─> Generate interface specification
    ↓
[4] Synthesize ABI contracts for WASM boundaries
    ↓
[5] Validate specifications for completeness
    ↓
[6] Serialize to structured format (JSON/YAML)
    ↓
OUTPUT: RustSpecification
```

---

## 2. Data Structures

### 2.1 Input Data Structures

```python
# From Analysis Agent
struct AnalysisResult:
    api_surface: ApiSurface
    dependency_graph: DependencyGraph
    contracts: ContractRegistry
    metadata: AnalysisMetadata

struct ApiSurface:
    functions: List[FunctionSignature]
    classes: List[ClassDefinition]
    modules: List[ModuleInfo]
    type_definitions: List[TypeDef]

struct FunctionSignature:
    name: String
    qualified_name: String  # module.Class.method
    parameters: List[Parameter]
    return_type: PythonType
    raises: List[ExceptionType]
    docstring: Optional[String]
    decorators: List[String]
    is_async: bool
    is_classmethod: bool
    is_staticmethod: bool
    purity: PurityClass  # Pure, Impure, Effectful
    side_effects: Set[SideEffect]  # IO, GlobalState, etc.

struct Parameter:
    name: String
    type_hint: Optional[PythonType]
    default_value: Optional[Any]
    is_optional: bool
    is_variadic: bool  # *args, **kwargs
    kind: ParameterKind  # Positional, Keyword, Both

struct ClassDefinition:
    name: String
    qualified_name: String
    base_classes: List[String]
    methods: List[FunctionSignature]
    properties: List[Property]
    class_variables: List[ClassVar]
    instance_variables: List[InstanceVar]
    docstring: Optional[String]
    is_abstract: bool
    is_dataclass: bool

struct PythonType:
    base_type: String  # int, str, list, dict, CustomClass
    type_params: List[PythonType]  # for generics: List[int]
    is_optional: bool
    is_union: bool
    union_members: List[PythonType]
    constraints: TypeConstraints  # value ranges, patterns

struct ContractRegistry:
    preconditions: Map[String, List[Condition]]  # func_name -> conditions
    postconditions: Map[String, List[Condition]]
    invariants: Map[String, List[Invariant]]
    ownership_hints: Map[String, OwnershipHint]  # inferred from usage
```

### 2.2 Output Data Structures

```python
struct RustSpecification:
    version: String  # spec format version
    source_metadata: SourceMetadata
    trait_definitions: List[TraitSpec]
    struct_definitions: List[StructSpec]
    enum_definitions: List[EnumSpec]
    type_mappings: TypeMappingRegistry
    abi_contracts: List[AbiContract]
    module_hierarchy: ModuleSpec
    confidence_report: ConfidenceReport

struct TraitSpec:
    name: String  # Rust trait name
    source_python_class: String  # qualified Python class name
    methods: List[TraitMethod]
    associated_types: List[AssociatedType]
    trait_bounds: List[TraitBound]
    documentation: String
    nemo_confidence: f64
    alternatives: List[AlternativeTraitSpec]

struct TraitMethod:
    name: String  # Rust method name (snake_case)
    signature: RustSignature
    self_kind: SelfKind  # None, ByValue, ByRef, ByMutRef
    is_async: bool
    documentation: String

struct RustSignature:
    parameters: List[RustParameter]
    return_type: RustType
    lifetime_params: List[LifetimeParam]
    type_params: List[TypeParam]
    where_clauses: List[WhereClause]

struct RustParameter:
    name: String
    rust_type: RustType
    ownership: OwnershipSemantics
    is_mutable: bool

struct RustType:
    base_type: String  # i32, String, Vec<T>, etc.
    type_args: List[RustType]
    is_reference: bool
    is_mutable_ref: bool
    lifetime: Optional[String]
    nullability: Nullability  # Option<T>, Result<T,E>, T

struct OwnershipSemantics:
    kind: OwnershipKind  # Owned, Borrowed, MutBorrowed, Copy
    lifetime: Optional[String]
    justification: String  # Why this ownership was chosen

enum OwnershipKind:
    Owned       # Takes ownership, move semantics
    Borrowed    # Immutable borrow &T
    MutBorrowed # Mutable borrow &mut T
    Copy        # Copy types (primitives)

struct StructSpec:
    name: String
    source_python_class: Optional[String]
    fields: List[StructField]
    derives: List[String]  # Debug, Clone, Serialize, etc.
    generics: List[GenericParam]
    documentation: String
    nemo_confidence: f64

struct EnumSpec:
    name: String
    source_python_union: Optional[String]
    variants: List[EnumVariant]
    derives: List[String]
    documentation: String
    representation: EnumRepr  # for ABI compatibility

enum EnumRepr:
    Default
    C           # #[repr(C)] for FFI
    Transparent # #[repr(transparent)]
    Packed      # #[repr(packed)]

struct AbiContract:
    function_name: String
    export_name: String  # for #[export_name = "..."]
    wasm_signature: WasmSignature
    memory_contract: MemoryContract
    error_handling: AbiErrorHandling
    documentation: String

struct WasmSignature:
    params: List[WasmType]
    results: List[WasmType]
    is_async: bool

enum WasmType:
    I32, I64, F32, F64  # WASM primitives
    Pointer             # i32 pointer into linear memory
    Handle              # opaque handle to host object

struct MemoryContract:
    allocator: AllocatorStrategy  # who allocates?
    ownership_transfer: bool      # does ownership cross boundary?
    deallocation: DeallocationStrategy
    alignment: usize
    documentation: String

enum AllocatorStrategy:
    CallerAllocates   # caller provides buffer
    CalleeAllocates   # callee allocates, returns pointer
    SharedArena       # both use shared arena

enum DeallocationStrategy:
    CallerDeallocates
    CalleeDeallocates
    ManualExplicit    # explicit free function
    GarbageCollected  # host GC handles it

struct ConfidenceReport:
    overall_confidence: f64
    per_construct_confidence: Map[String, f64]
    low_confidence_items: List[LowConfidenceItem]
    fallback_strategies: List[FallbackStrategy]
    warnings: List[String]

struct LowConfidenceItem:
    python_construct: String
    confidence: f64
    reason: String
    alternatives: List[AlternativeSpec]
    recommendation: String
```

### 2.3 NeMo-Specific Data Structures

```python
struct NeMoTranslationRequest:
    python_construct: PythonConstruct
    context: TranslationContext
    constraints: List[Constraint]
    preferences: TranslationPreferences

struct PythonConstruct:
    kind: ConstructKind  # Function, Class, Type, etc.
    source_code: String
    signature: Optional[FunctionSignature]
    type_info: Optional[PythonType]
    usage_patterns: List[UsagePattern]

enum ConstructKind:
    Function, Method, Class, Type, Module, Expression

struct TranslationContext:
    surrounding_code: String
    dependency_signatures: List[Signature]
    module_context: String
    stdlib_usage: Set[String]
    third_party_usage: Set[String]

struct Constraint:
    kind: ConstraintKind
    value: Any
    priority: Priority

enum ConstraintKind:
    MustBeWasmSafe     # no non-WASM types
    MustBeSendSync     # thread-safe
    PreferNoUnsafe     # avoid unsafe blocks
    PreferNoLifetimes  # simplify if possible
    MustBeSerializable # for ABI crossing

struct NeMoTranslationResponse:
    primary_translation: RustTranslation
    alternatives: List[RustTranslation]
    confidence_scores: Map[String, f64]
    reasoning: String
    warnings: List[String]

struct RustTranslation:
    rust_code: String
    type_mapping: TypeMapping
    ownership_strategy: OwnershipStrategy
    abi_compatibility: AbiCompatibility
    confidence: f64
    trade_offs: String

struct TypeMapping:
    python_type: PythonType
    rust_type: RustType
    conversion_required: bool
    conversion_code: Optional[String]
    lossy: bool
    rationale: String

struct OwnershipStrategy:
    parameter_ownership: Map[String, OwnershipKind]
    return_ownership: OwnershipKind
    lifetime_requirements: List[LifetimeRequirement]
    justification: String

struct AbiCompatibility:
    is_ffi_safe: bool
    is_wasm_safe: bool
    requires_wrapper: bool
    wrapper_strategy: Optional[WrapperStrategy]
```

---

## 3. Core Algorithms

### 3.1 Main Specification Generation Algorithm

```python
function generate_specification(analysis: AnalysisResult) -> RustSpecification:
    """
    Main entry point for specification generation.
    Orchestrates the entire translation process.
    """

    # Initialize
    nemo_session = initialize_nemo_session()
    spec = RustSpecification.new()
    type_registry = TypeMappingRegistry.new()

    # Process in dependency order (bottom-up)
    sorted_classes = topological_sort(analysis.api_surface.classes)

    # Phase 1: Type mapping
    for python_type in analysis.api_surface.type_definitions:
        mapping = generate_type_mapping(python_type, type_registry, nemo_session)
        type_registry.register(mapping)

    # Phase 2: Struct generation (for dataclasses, simple classes)
    for py_class in sorted_classes:
        if is_data_oriented(py_class):
            struct_spec = generate_struct_spec(py_class, type_registry, nemo_session)
            spec.struct_definitions.append(struct_spec)
            type_registry.register_composite(py_class.name, struct_spec)

    # Phase 3: Trait generation (for classes with behavior)
    for py_class in sorted_classes:
        if has_behavioral_methods(py_class):
            trait_spec = generate_trait_spec(py_class, type_registry, nemo_session)
            spec.trait_definitions.append(trait_spec)

    # Phase 4: Enum generation (for union types, exception hierarchies)
    for union_type in extract_union_types(analysis):
        enum_spec = generate_enum_spec(union_type, type_registry, nemo_session)
        spec.enum_definitions.append(enum_spec)

    # Phase 5: ABI contract generation
    for function in analysis.api_surface.functions:
        if is_public_api(function):
            abi_contract = generate_abi_contract(function, type_registry, nemo_session)
            spec.abi_contracts.append(abi_contract)

    # Phase 6: Module hierarchy
    spec.module_hierarchy = generate_module_spec(analysis.api_surface.modules)

    # Phase 7: Validation and confidence reporting
    spec.confidence_report = validate_and_score(spec, analysis)

    # Cleanup
    nemo_session.shutdown()

    return spec
```

### 3.2 Type Mapping Algorithm

```python
function generate_type_mapping(
    py_type: PythonType,
    registry: TypeMappingRegistry,
    nemo: NeMoSession
) -> TypeMapping:
    """
    Maps a Python type to a Rust type using NeMo-assisted inference.
    Handles primitives, collections, generics, and custom types.
    """

    # Check cache first
    if registry.has_mapping(py_type):
        return registry.get_mapping(py_type)

    # Handle built-in primitives
    if py_type.base_type in PRIMITIVE_TYPE_MAP:
        rust_type = map_primitive_type(py_type)
        return TypeMapping(
            python_type=py_type,
            rust_type=rust_type,
            conversion_required=false,
            lossy=false,
            rationale="Direct primitive mapping"
        )

    # Handle collections (List, Dict, Set, Tuple)
    if py_type.base_type in COLLECTION_TYPES:
        return map_collection_type(py_type, registry, nemo)

    # Handle Optional types
    if py_type.is_optional:
        inner_mapping = generate_type_mapping(
            unwrap_optional(py_type),
            registry,
            nemo
        )
        return wrap_in_option(inner_mapping)

    # Handle Union types
    if py_type.is_union:
        return map_union_type(py_type, registry, nemo)

    # Handle custom classes - query NeMo
    nemo_request = NeMoTranslationRequest(
        python_construct=PythonConstruct(
            kind=ConstructKind.Type,
            type_info=py_type
        ),
        context=build_context(py_type),
        constraints=[
            Constraint(ConstraintKind.MustBeWasmSafe, true, Priority.High)
        ]
    )

    response = nemo.query_type_mapping(nemo_request)

    # Select best translation
    best_translation = select_best_translation(
        response.alternatives,
        prefer_simple=true,
        prefer_no_lifetimes=true
    )

    mapping = TypeMapping(
        python_type=py_type,
        rust_type=parse_rust_type(best_translation.rust_code),
        conversion_required=best_translation.type_mapping.conversion_required,
        lossy=best_translation.type_mapping.lossy,
        rationale=best_translation.type_mapping.rationale
    )

    # Cache for future use
    registry.register(mapping)

    return mapping


function map_primitive_type(py_type: PythonType) -> RustType:
    """
    Maps Python primitive types to Rust equivalents.
    """
    PRIMITIVE_MAP = {
        "int": "i64",      # default to i64, may optimize later
        "float": "f64",
        "str": "String",
        "bool": "bool",
        "bytes": "Vec<u8>",
        "None": "()",
    }

    base_rust_type = PRIMITIVE_MAP[py_type.base_type]

    # Apply constraints if available
    if py_type.constraints:
        base_rust_type = optimize_numeric_type(base_rust_type, py_type.constraints)

    return RustType(base_type=base_rust_type)


function map_collection_type(
    py_type: PythonType,
    registry: TypeMappingRegistry,
    nemo: NeMoSession
) -> TypeMapping:
    """
    Maps Python collection types to Rust equivalents.
    """
    base = py_type.base_type

    # Recursively map type parameters
    rust_type_params = [
        generate_type_mapping(param, registry, nemo).rust_type
        for param in py_type.type_params
    ]

    if base == "list":
        rust_type = RustType(
            base_type="Vec",
            type_args=rust_type_params
        )
    elif base == "dict":
        rust_type = RustType(
            base_type="HashMap",
            type_args=rust_type_params
        )
    elif base == "set":
        rust_type = RustType(
            base_type="HashSet",
            type_args=rust_type_params
        )
    elif base == "tuple":
        # Tuples are heterogeneous in Python, homogeneous in Rust
        if all_same_type(rust_type_params):
            rust_type = RustType(
                base_type=f"({', '.join(str(t) for t in rust_type_params)})"
            )
        else:
            # Heterogeneous tuple - use actual Rust tuple
            rust_type = RustType(
                base_type=f"({', '.join(str(t) for t in rust_type_params)})"
            )

    return TypeMapping(
        python_type=py_type,
        rust_type=rust_type,
        conversion_required=false,
        lossy=false,
        rationale=f"Standard collection mapping: {base} -> {rust_type.base_type}"
    )


function map_union_type(
    py_type: PythonType,
    registry: TypeMappingRegistry,
    nemo: NeMoSession
) -> TypeMapping:
    """
    Maps Python Union types to Rust enums.
    """
    # Check if it's just Optional (Union[T, None])
    if is_optional_union(py_type):
        inner = get_non_none_type(py_type.union_members)
        inner_mapping = generate_type_mapping(inner, registry, nemo)
        return wrap_in_option(inner_mapping)

    # For complex unions, query NeMo for best enum structure
    nemo_request = NeMoTranslationRequest(
        python_construct=PythonConstruct(
            kind=ConstructKind.Type,
            type_info=py_type
        ),
        context=build_context(py_type),
        constraints=[
            Constraint(ConstraintKind.MustBeWasmSafe, true, Priority.High),
            Constraint(ConstraintKind.MustBeSerializable, true, Priority.High)
        ]
    )

    response = nemo.query_union_strategy(nemo_request)

    # NeMo should suggest enum name and variant structure
    return TypeMapping(
        python_type=py_type,
        rust_type=parse_rust_type(response.primary_translation.rust_code),
        conversion_required=true,
        lossy=false,
        rationale=response.reasoning
    )
```

### 3.3 Trait Generation Algorithm

```python
function generate_trait_spec(
    py_class: ClassDefinition,
    registry: TypeMappingRegistry,
    nemo: NeMoSession
) -> TraitSpec:
    """
    Generates a Rust trait specification from a Python class.
    """

    # Build NeMo request with full class context
    nemo_request = NeMoTranslationRequest(
        python_construct=PythonConstruct(
            kind=ConstructKind.Class,
            source_code=reconstruct_class_code(py_class),
            signature=None,
            usage_patterns=analyze_usage_patterns(py_class)
        ),
        context=TranslationContext(
            surrounding_code=get_module_context(py_class),
            dependency_signatures=get_dependency_signatures(py_class)
        ),
        constraints=[
            Constraint(ConstraintKind.PreferNoUnsafe, true, Priority.Medium)
        ],
        preferences=TranslationPreferences(
            idiomatic_rust=true,
            minimize_lifetimes=true
        )
    )

    # Query NeMo for trait structure
    response = nemo.query_trait_design(nemo_request)

    # Extract trait name (convert PascalCase to Rust conventions)
    trait_name = to_rust_trait_name(py_class.name)

    # Generate methods
    methods = []
    for py_method in py_class.methods:
        if not is_private(py_method):
            trait_method = generate_trait_method(
                py_method,
                py_class,
                registry,
                nemo,
                response.primary_translation.ownership_strategy
            )
            methods.append(trait_method)

    # Determine associated types (if class uses generics)
    associated_types = extract_associated_types(py_class, response)

    # Determine trait bounds
    trait_bounds = infer_trait_bounds(py_class, methods)

    # Select primary translation
    primary = response.primary_translation

    trait_spec = TraitSpec(
        name=trait_name,
        source_python_class=py_class.qualified_name,
        methods=methods,
        associated_types=associated_types,
        trait_bounds=trait_bounds,
        documentation=convert_docstring(py_class.docstring),
        nemo_confidence=primary.confidence,
        alternatives=[
            parse_alternative_trait(alt)
            for alt in response.alternatives[:3]  # top 3 alternatives
        ]
    )

    return trait_spec


function generate_trait_method(
    py_method: FunctionSignature,
    parent_class: ClassDefinition,
    registry: TypeMappingRegistry,
    nemo: NeMoSession,
    ownership_strategy: OwnershipStrategy
) -> TraitMethod:
    """
    Generates a single trait method from a Python method.
    """

    # Determine self kind
    self_kind = infer_self_kind(py_method, ownership_strategy)

    # Generate Rust signature
    rust_params = []
    for param in py_method.parameters:
        if param.name == "self":
            continue  # handled by self_kind

        rust_param = generate_rust_parameter(
            param,
            registry,
            nemo,
            ownership_strategy
        )
        rust_params.append(rust_param)

    # Map return type
    return_mapping = generate_type_mapping(
        py_method.return_type,
        registry,
        nemo
    )

    # Handle exceptions -> Result type
    return_type = wrap_in_result_if_needed(
        return_mapping.rust_type,
        py_method.raises
    )

    # Infer lifetime parameters
    lifetime_params = infer_lifetimes(
        rust_params,
        return_type,
        self_kind
    )

    # Build signature
    signature = RustSignature(
        parameters=rust_params,
        return_type=return_type,
        lifetime_params=lifetime_params,
        type_params=[],
        where_clauses=[]
    )

    # Convert name to snake_case
    rust_name = to_snake_case(py_method.name)

    # Handle async
    is_async = py_method.is_async

    return TraitMethod(
        name=rust_name,
        signature=signature,
        self_kind=self_kind,
        is_async=is_async,
        documentation=convert_docstring(py_method.docstring)
    )


function infer_self_kind(
    py_method: FunctionSignature,
    ownership_strategy: OwnershipStrategy
) -> SelfKind:
    """
    Infers how self should be taken in Rust trait method.
    """

    # Static methods
    if py_method.is_staticmethod:
        return SelfKind.None

    # Class methods -> typically &self (accessing class, not instance)
    if py_method.is_classmethod:
        return SelfKind.ByRef

    # Check if method mutates self
    if method_mutates_self(py_method):
        return SelfKind.ByMutRef

    # Check ownership strategy from NeMo
    if "self" in ownership_strategy.parameter_ownership:
        ownership = ownership_strategy.parameter_ownership["self"]
        if ownership == OwnershipKind.Owned:
            return SelfKind.ByValue
        elif ownership == OwnershipKind.MutBorrowed:
            return SelfKind.ByMutRef
        else:
            return SelfKind.ByRef

    # Default: immutable borrow
    return SelfKind.ByRef


function generate_rust_parameter(
    param: Parameter,
    registry: TypeMappingRegistry,
    nemo: NeMoSession,
    ownership_strategy: OwnershipStrategy
) -> RustParameter:
    """
    Generates a Rust parameter from a Python parameter.
    """

    # Map type
    type_mapping = generate_type_mapping(
        param.type_hint or infer_type(param),
        registry,
        nemo
    )

    rust_type = type_mapping.rust_type

    # Determine ownership
    ownership = infer_parameter_ownership(
        param,
        ownership_strategy,
        rust_type
    )

    # Apply ownership to type
    if ownership.kind == OwnershipKind.Borrowed:
        rust_type = make_reference(rust_type, mutable=false, ownership.lifetime)
    elif ownership.kind == OwnershipKind.MutBorrowed:
        rust_type = make_reference(rust_type, mutable=true, ownership.lifetime)
    # Owned and Copy don't modify the type

    # Handle default values -> Option<T> if optional
    if param.is_optional and not rust_type.is_option():
        rust_type = wrap_in_option_type(rust_type)

    # Convert name to snake_case
    rust_name = to_snake_case(param.name)

    return RustParameter(
        name=rust_name,
        rust_type=rust_type,
        ownership=ownership,
        is_mutable=ownership.kind == OwnershipKind.MutBorrowed
    )


function infer_parameter_ownership(
    param: Parameter,
    ownership_strategy: OwnershipStrategy,
    rust_type: RustType
) -> OwnershipSemantics:
    """
    Infers ownership semantics for a parameter.
    """

    # Check if NeMo provided strategy
    if param.name in ownership_strategy.parameter_ownership:
        kind = ownership_strategy.parameter_ownership[param.name]
        return OwnershipSemantics(
            kind=kind,
            lifetime=infer_lifetime(param, kind),
            justification=ownership_strategy.justification
        )

    # Apply heuristics

    # Copy types (primitives) -> pass by value
    if is_copy_type(rust_type):
        return OwnershipSemantics(
            kind=OwnershipKind.Copy,
            lifetime=None,
            justification="Primitive type implements Copy"
        )

    # Large types or collections -> borrow by default
    if is_large_type(rust_type) or is_collection(rust_type):
        return OwnershipSemantics(
            kind=OwnershipKind.Borrowed,
            lifetime=None,
            justification="Large or collection type, prefer borrowing"
        )

    # String types -> borrow as &str
    if rust_type.base_type == "String":
        return OwnershipSemantics(
            kind=OwnershipKind.Borrowed,
            lifetime=None,
            justification="String borrows as &str for flexibility"
        )

    # Default: owned
    return OwnershipSemantics(
        kind=OwnershipKind.Owned,
        lifetime=None,
        justification="Default ownership transfer"
    )
```

### 3.4 ABI Contract Generation Algorithm

```python
function generate_abi_contract(
    function: FunctionSignature,
    registry: TypeMappingRegistry,
    nemo: NeMoSession
) -> AbiContract:
    """
    Generates WASI-compatible ABI contract for a public function.
    This defines how the function will be exposed across WASM boundary.
    """

    # Generate Rust signature first
    rust_sig = generate_rust_signature(function, registry, nemo)

    # Convert to WASM-compatible signature
    wasm_sig = lower_to_wasm_signature(rust_sig, registry)

    # Determine memory contract
    memory_contract = infer_memory_contract(function, rust_sig, wasm_sig)

    # Determine error handling strategy
    error_handling = infer_abi_error_handling(function, rust_sig)

    # Generate export name (C-compatible)
    export_name = generate_export_name(function)

    contract = AbiContract(
        function_name=function.qualified_name,
        export_name=export_name,
        wasm_signature=wasm_sig,
        memory_contract=memory_contract,
        error_handling=error_handling,
        documentation=generate_abi_documentation(function, memory_contract)
    )

    return contract


function lower_to_wasm_signature(
    rust_sig: RustSignature,
    registry: TypeMappingRegistry
) -> WasmSignature:
    """
    Lowers a Rust signature to WASM primitives.
    Complex types become pointers into linear memory.
    """

    wasm_params = []
    wasm_results = []

    for param in rust_sig.parameters:
        wasm_type = lower_rust_type_to_wasm(param.rust_type)
        wasm_params.append(wasm_type)

    # WASM supports multi-value returns, but limit to 1 for compatibility
    if rust_sig.return_type.base_type != "()":
        wasm_ret = lower_rust_type_to_wasm(rust_sig.return_type)
        wasm_results.append(wasm_ret)

    return WasmSignature(
        params=wasm_params,
        results=wasm_results,
        is_async=false  # WASM doesn't support async natively
    )


function lower_rust_type_to_wasm(rust_type: RustType) -> WasmType:
    """
    Maps a Rust type to a WASM type.
    """

    base = rust_type.base_type

    # Direct mappings
    if base in ["i8", "u8", "i16", "u16", "i32", "u32", "bool"]:
        return WasmType.I32
    elif base in ["i64", "u64"]:
        return WasmType.I64
    elif base == "f32":
        return WasmType.F32
    elif base == "f64":
        return WasmType.F64

    # References, strings, collections -> pointers
    if rust_type.is_reference or base in ["String", "Vec", "HashMap", "HashSet"]:
        return WasmType.Pointer

    # Complex types -> serialize to bytes, pass pointer
    return WasmType.Pointer


function infer_memory_contract(
    function: FunctionSignature,
    rust_sig: RustSignature,
    wasm_sig: WasmSignature
) -> MemoryContract:
    """
    Infers memory allocation/deallocation contract for ABI crossing.
    """

    # Check if function returns complex types
    has_complex_return = any(
        param_type == WasmType.Pointer
        for param_type in wasm_sig.results
    )

    # Check if function takes complex parameters
    has_complex_params = any(
        param_type == WasmType.Pointer
        for param_type in wasm_sig.params
    )

    if has_complex_return:
        # Callee allocates return value, caller must deallocate
        allocator = AllocatorStrategy.CalleeAllocates
        dealloc = DeallocationStrategy.CallerDeallocates
        ownership_transfer = true
    elif has_complex_params:
        # Caller allocates parameters, callee borrows
        allocator = AllocatorStrategy.CallerAllocates
        dealloc = DeallocationStrategy.CallerDeallocates
        ownership_transfer = false
    else:
        # All primitives, no heap allocation
        allocator = AllocatorStrategy.CallerAllocates
        dealloc = DeallocationStrategy.CallerDeallocates
        ownership_transfer = false

    return MemoryContract(
        allocator=allocator,
        ownership_transfer=ownership_transfer,
        deallocation=dealloc,
        alignment=8,  # default alignment
        documentation=generate_memory_contract_docs(allocator, dealloc)
    )


function infer_abi_error_handling(
    function: FunctionSignature,
    rust_sig: RustSignature
) -> AbiErrorHandling:
    """
    Determines how errors are propagated across ABI boundary.
    """

    # Check if Rust signature returns Result
    is_result_type = (
        rust_sig.return_type.base_type.startswith("Result<")
    )

    if is_result_type:
        # Use discriminated union: {tag, ok_value, err_value}
        strategy = ErrorStrategy.DiscriminatedUnion
    elif len(function.raises) > 0:
        # Python function raises exceptions, but Rust doesn't return Result
        # Use error code + thread-local error storage
        strategy = ErrorStrategy.ErrorCodeWithLastError
    else:
        # No errors
        strategy = ErrorStrategy.None_

    return AbiErrorHandling(
        strategy=strategy,
        error_codes=generate_error_codes(function.raises),
        documentation=generate_error_handling_docs(strategy)
    )
```

### 3.5 NeMo Integration Algorithms

```python
function initialize_nemo_session() -> NeMoSession:
    """
    Initializes a NeMo inference session for translation queries.
    """

    # Load NeMo model (code generation or translation model)
    model_path = get_config("nemo.model_path")
    model = load_nemo_model(model_path)

    # Configure for code translation task
    config = NeMoConfig(
        task="python_to_rust_translation",
        temperature=0.2,  # low temperature for deterministic output
        top_k=5,          # return top 5 alternatives
        max_tokens=2048,
        use_gpu=true
    )

    session = NeMoSession(model, config)

    # Warm up with simple query
    session.warmup()

    return session


function query_nemo_for_translation(
    nemo: NeMoSession,
    request: NeMoTranslationRequest
) -> NeMoTranslationResponse:
    """
    Queries NeMo for translation suggestions.
    Implements caching and retry logic.
    """

    # Check cache
    cache_key = hash_request(request)
    if cache.has(cache_key):
        return cache.get(cache_key)

    # Build prompt for NeMo
    prompt = build_translation_prompt(request)

    # Query NeMo with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw_response = nemo.generate(prompt)
            response = parse_nemo_response(raw_response)

            # Validate response
            if validate_nemo_response(response, request):
                cache.put(cache_key, response)
                return response
            else:
                log_warning(f"Invalid NeMo response on attempt {attempt + 1}")
        except NeMoError as e:
            log_error(f"NeMo query failed: {e}")
            if attempt == max_retries - 1:
                raise
            sleep(backoff_delay(attempt))

    # Fallback if all retries failed
    return generate_fallback_response(request)


function build_translation_prompt(request: NeMoTranslationRequest) -> String:
    """
    Constructs a prompt for NeMo translation query.
    """

    prompt = f"""
You are an expert Python-to-Rust translator. Translate the following Python code to idiomatic Rust.

## Python Code:
```python
{request.python_construct.source_code}
```

## Context:
{request.context.surrounding_code}

## Constraints:
{format_constraints(request.constraints)}

## Requirements:
1. Generate idiomatic Rust code
2. Preserve semantic equivalence
3. Ensure WASM/WASI compatibility
4. Specify ownership semantics
5. Provide confidence score (0.0 - 1.0)
6. Provide 3-5 alternative translations ranked by confidence
7. Explain your reasoning

## Output Format (JSON):
{{
  "primary_translation": {{
    "rust_code": "...",
    "type_mapping": {{}},
    "ownership_strategy": {{}},
    "confidence": 0.95,
    "trade_offs": "..."
  }},
  "alternatives": [...],
  "reasoning": "...",
  "warnings": [...]
}}

Generate translation:
"""
    return prompt


function parse_nemo_response(raw: String) -> NeMoTranslationResponse:
    """
    Parses NeMo's JSON response into structured format.
    """

    # NeMo returns JSON
    data = parse_json(raw)

    primary = RustTranslation(
        rust_code=data["primary_translation"]["rust_code"],
        type_mapping=parse_type_mapping(data["primary_translation"]["type_mapping"]),
        ownership_strategy=parse_ownership_strategy(
            data["primary_translation"]["ownership_strategy"]
        ),
        abi_compatibility=parse_abi_compatibility(
            data["primary_translation"].get("abi_compatibility", {})
        ),
        confidence=data["primary_translation"]["confidence"],
        trade_offs=data["primary_translation"].get("trade_offs", "")
    )

    alternatives = [
        parse_rust_translation(alt)
        for alt in data.get("alternatives", [])
    ]

    # Extract confidence scores
    confidence_scores = {
        "primary": primary.confidence,
        **{f"alt_{i}": alt.confidence for i, alt in enumerate(alternatives)}
    }

    return NeMoTranslationResponse(
        primary_translation=primary,
        alternatives=alternatives,
        confidence_scores=confidence_scores,
        reasoning=data.get("reasoning", ""),
        warnings=data.get("warnings", [])
    )


function select_best_translation(
    alternatives: List[RustTranslation],
    **preferences
) -> RustTranslation:
    """
    Selects the best translation based on confidence and preferences.
    """

    # Score each alternative
    scores = []
    for alt in alternatives:
        score = alt.confidence

        # Apply preference weights
        if preferences.get("prefer_simple"):
            score *= 1.1 if is_simple_code(alt.rust_code) else 0.9

        if preferences.get("prefer_no_lifetimes"):
            lifetime_count = count_lifetime_annotations(alt.rust_code)
            score *= (1.0 - 0.05 * lifetime_count)

        if preferences.get("prefer_no_unsafe"):
            unsafe_count = count_unsafe_blocks(alt.rust_code)
            score *= (1.0 - 0.1 * unsafe_count)

        scores.append((score, alt))

    # Return highest scoring
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1]


function validate_and_score(
    spec: RustSpecification,
    analysis: AnalysisResult
) -> ConfidenceReport:
    """
    Validates the generated specification and produces confidence report.
    """

    # Calculate overall confidence (weighted average)
    total_constructs = (
        len(spec.trait_definitions) +
        len(spec.struct_definitions) +
        len(spec.enum_definitions) +
        len(spec.abi_contracts)
    )

    total_confidence = 0.0
    per_construct = {}

    for trait_spec in spec.trait_definitions:
        total_confidence += trait_spec.nemo_confidence
        per_construct[trait_spec.name] = trait_spec.nemo_confidence

    for struct_spec in spec.struct_definitions:
        total_confidence += struct_spec.nemo_confidence
        per_construct[struct_spec.name] = struct_spec.nemo_confidence

    # Similar for enums and ABIs...

    overall_confidence = total_confidence / total_constructs if total_constructs > 0 else 0.0

    # Identify low confidence items
    low_confidence_items = []
    for name, confidence in per_construct.items():
        if confidence < 0.7:
            item = LowConfidenceItem(
                python_construct=name,
                confidence=confidence,
                reason="NeMo confidence below threshold",
                alternatives=get_alternatives_for(name, spec),
                recommendation="Manual review recommended"
            )
            low_confidence_items.append(item)

    # Check completeness
    warnings = []
    if not validate_completeness(spec, analysis):
        warnings.append("Incomplete API coverage")

    if not validate_abi_safety(spec):
        warnings.append("Some ABI contracts may not be WASM-safe")

    return ConfidenceReport(
        overall_confidence=overall_confidence,
        per_construct_confidence=per_construct,
        low_confidence_items=low_confidence_items,
        fallback_strategies=generate_fallback_strategies(low_confidence_items),
        warnings=warnings
    )
```

---

## 4. Input/Output Contracts

### 4.1 Agent Input Contract

```yaml
INPUT: AnalysisResult
  FROM: Analysis Agent

  REQUIRED_FIELDS:
    api_surface: ApiSurface
      functions: List[FunctionSignature]  # all public functions
      classes: List[ClassDefinition]      # all classes
      modules: List[ModuleInfo]           # module structure
      type_definitions: List[TypeDef]     # custom types

    dependency_graph: DependencyGraph
      call_graph: Graph                   # function call relationships
      data_flow: Graph                    # data flow between functions
      type_dependencies: Graph            # type dependency relationships

    contracts: ContractRegistry
      preconditions: Map                  # input validation contracts
      postconditions: Map                 # output guarantees
      invariants: Map                     # class invariants
      ownership_hints: Map                # inferred ownership patterns

    metadata: AnalysisMetadata
      python_version: String
      stdlib_usage: Set[String]
      third_party_deps: Set[String]
      complexity_metrics: Dict

  VALIDATION:
    - api_surface must not be empty
    - all FunctionSignatures must have valid return types
    - all ClassDefinitions must have unique qualified names
    - dependency_graph must be acyclic for classes
    - type_definitions must be resolvable
```

### 4.2 Agent Output Contract

```yaml
OUTPUT: RustSpecification
  TO: Transpiler Agent

  REQUIRED_FIELDS:
    version: String                          # spec format version
    source_metadata: SourceMetadata          # source provenance

    trait_definitions: List[TraitSpec]       # Rust traits
      - Each trait must have valid Rust identifier
      - Methods must have valid signatures
      - Trait bounds must be satisfiable

    struct_definitions: List[StructSpec]     # Rust structs
      - Each struct must have valid fields
      - Generics must be well-formed
      - Derives must be applicable

    enum_definitions: List[EnumSpec]         # Rust enums
      - Variants must be unique
      - Must specify repr for FFI enums

    type_mappings: TypeMappingRegistry       # Python -> Rust type map
      - All types used must be mapped
      - No circular type dependencies

    abi_contracts: List[AbiContract]         # WASM ABI definitions
      - All exported functions must have ABI contract
      - WASM signatures must use only WASM types
      - Memory contracts must specify allocator

    module_hierarchy: ModuleSpec             # Rust module structure
      - Must preserve Python package structure
      - No circular module dependencies

    confidence_report: ConfidenceReport      # Quality metrics
      - overall_confidence: 0.0 - 1.0
      - Must identify low confidence items

  VALIDATION:
    - All trait/struct/enum names are valid Rust identifiers
    - All type references are resolvable
    - ABI contracts cover all public API functions
    - Confidence report is complete
    - No compilation-breaking constructs (e.g., invalid lifetimes)

  GUARANTEES:
    - All generated Rust constructs are syntactically valid
    - Type mappings are semantically sound
    - ABI contracts are WASM/WASI compatible
    - Ownership semantics are safe (no dangling references)
    - Confidence scores accurately reflect NeMo certainty
```

---

## 5. NeMo Integration Strategy

### 5.1 NeMo Model Selection

```python
# Model configuration for Python-to-Rust translation
NEMO_MODEL_CONFIG = {
    "model_family": "GPT-based or T5-based code model",
    "size": "7B - 70B parameters",
    "fine_tuning": "Python-to-Rust translation pairs",
    "context_window": "8K - 32K tokens",
    "precision": "FP16 or INT8 for inference"
}

# Inference configuration
INFERENCE_CONFIG = {
    "batch_size": 8,           # parallel translation queries
    "temperature": 0.2,        # low for deterministic output
    "top_p": 0.9,
    "top_k": 5,                # return top 5 alternatives
    "repetition_penalty": 1.2,
    "max_new_tokens": 2048
}
```

### 5.2 Prompt Engineering Strategy

```python
# Template for translation prompts
TRANSLATION_PROMPT_TEMPLATE = """
<|system|>
You are an expert systems programmer specializing in Python-to-Rust translation.
You understand:
- Python semantics: dynamic typing, duck typing, exception handling
- Rust semantics: ownership, borrowing, lifetimes, type safety
- WASM/WASI constraints: FFI-safe types, linear memory, no multi-threading
- Idiomatic Rust patterns: Result types, iterators, pattern matching

<|user|>
Translate this Python {construct_type} to idiomatic Rust:

## Python Source:
```python
{python_code}
```

## Type Context:
{type_annotations}

## Usage Context:
{call_patterns}

## Constraints:
{constraints_list}

## Requirements:
1. Preserve semantic equivalence
2. Use idiomatic Rust patterns
3. Ensure WASM/WASI compatibility
4. Minimize unsafe code
5. Specify ownership semantics clearly
6. Provide confidence score and alternatives

<|assistant|>
"""

# Few-shot examples for better accuracy
FEW_SHOT_EXAMPLES = [
    {
        "python": "def add(a: int, b: int) -> int: return a + b",
        "rust": "fn add(a: i64, b: i64) -> i64 { a + b }",
        "confidence": 1.0,
        "reasoning": "Simple pure function, direct mapping"
    },
    {
        "python": "class Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1",
        "rust": "struct Counter { count: i64 }\nimpl Counter {\n    fn new() -> Self { Self { count: 0 } }\n    fn increment(&mut self) { self.count += 1; }\n}",
        "confidence": 0.95,
        "reasoning": "Mutable state requires &mut self"
    }
]
```

### 5.3 Response Validation

```python
function validate_nemo_response(
    response: NeMoTranslationResponse,
    request: NeMoTranslationRequest
) -> bool:
    """
    Validates NeMo's response for correctness and safety.
    """

    primary = response.primary_translation

    # Check 1: Valid Rust syntax
    if not is_valid_rust_syntax(primary.rust_code):
        log_error("NeMo generated invalid Rust syntax")
        return false

    # Check 2: Confidence score in valid range
    if not (0.0 <= primary.confidence <= 1.0):
        log_error("Invalid confidence score")
        return false

    # Check 3: Required fields present
    if not all([
        primary.rust_code,
        primary.type_mapping,
        primary.ownership_strategy
    ]):
        log_error("Missing required fields in NeMo response")
        return false

    # Check 4: WASM safety if required
    if has_constraint(request, ConstraintKind.MustBeWasmSafe):
        if not is_wasm_safe(primary.rust_code):
            log_error("Generated code is not WASM-safe")
            return false

    # Check 5: No unsafe code if constrained
    if has_constraint(request, ConstraintKind.PreferNoUnsafe):
        unsafe_count = count_unsafe_blocks(primary.rust_code)
        if unsafe_count > 0:
            log_warning(f"Generated code has {unsafe_count} unsafe blocks")
            # Don't fail, just warn

    # Check 6: Alternatives are ranked
    if len(response.alternatives) > 1:
        confidences = [alt.confidence for alt in response.alternatives]
        if confidences != sorted(confidences, reverse=True):
            log_warning("Alternatives not properly ranked by confidence")

    return true


function is_valid_rust_syntax(code: String) -> bool:
    """
    Validates Rust syntax using rustc --parse-only.
    """

    # Write code to temporary file
    temp_file = write_temp_file(code)

    # Run rustc parser
    result = run_command([
        "rustc",
        "--crate-type=lib",
        "--parse-only",
        temp_file
    ])

    return result.exit_code == 0
```

### 5.4 Caching Strategy

```python
class NeMoCache:
    """
    Caches NeMo translation responses to avoid redundant queries.
    """

    def __init__(self):
        self.memory_cache = LRUCache(capacity=1000)
        self.disk_cache = DiskCache(path="/tmp/portalis/nemo_cache")

    def hash_request(self, request: NeMoTranslationRequest) -> String:
        """
        Generates cache key from request.
        """
        # Hash based on code, context, and constraints
        components = [
            request.python_construct.source_code,
            str(request.context),
            str(sorted(request.constraints))
        ]
        return sha256("|".join(components))

    def get(self, key: String) -> Optional[NeMoTranslationResponse]:
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk cache
        if self.disk_cache.has(key):
            response = self.disk_cache.get(key)
            # Promote to memory cache
            self.memory_cache[key] = response
            return response

        return None

    def put(self, key: String, response: NeMoTranslationResponse):
        # Write to both caches
        self.memory_cache[key] = response
        self.disk_cache.put(key, response)
```

### 5.5 Fallback Strategy

```python
function generate_fallback_response(
    request: NeMoTranslationRequest
) -> NeMoTranslationResponse:
    """
    Generates a fallback response when NeMo fails or returns invalid results.
    Uses rule-based translation heuristics.
    """

    log_warning("Using fallback translation strategy")

    construct = request.python_construct

    # Use rule-based translation
    if construct.kind == ConstructKind.Function:
        rust_code = translate_function_heuristic(construct)
    elif construct.kind == ConstructKind.Class:
        rust_code = translate_class_heuristic(construct)
    elif construct.kind == ConstructKind.Type:
        rust_code = translate_type_heuristic(construct)
    else:
        raise UnsupportedConstruct(construct.kind)

    # Build conservative response
    return NeMoTranslationResponse(
        primary_translation=RustTranslation(
            rust_code=rust_code,
            type_mapping=infer_type_mapping_heuristic(construct),
            ownership_strategy=conservative_ownership_strategy(),
            abi_compatibility=default_abi_compatibility(),
            confidence=0.5,  # low confidence for heuristic approach
            trade_offs="Fallback heuristic translation - manual review recommended"
        ),
        alternatives=[],
        confidence_scores={"primary": 0.5},
        reasoning="NeMo query failed, using rule-based fallback",
        warnings=["Manual review required for this translation"]
    )
```

---

## 6. London School TDD Test Points

### 6.1 Test Hierarchy

```
Acceptance Tests (Outside-In)
  ├─> Test: End-to-end specification generation
  │   Given: Sample Python library with classes, functions, types
  │   When: Specification generator processes AnalysisResult
  │   Then: Valid RustSpecification produced with confidence > 0.8
  │
  └─> Test: NeMo integration produces ranked alternatives
      Given: Complex Python class with multiple methods
      When: NeMo is queried for trait design
      Then: Primary translation + 3+ alternatives returned, ranked by confidence

Integration Tests (Component Boundaries)
  ├─> Test: Type mapping registry consistency
  ├─> Test: NeMo session lifecycle (init, query, shutdown)
  ├─> Test: ABI contract generation for various function signatures
  └─> Test: Confidence report accuracy

Unit Tests (Individual Functions)
  ├─> Type Mapping Tests
  │   ├─> map_primitive_type: int -> i64, str -> String, etc.
  │   ├─> map_collection_type: List[int] -> Vec<i64>
  │   ├─> map_union_type: Union[int, str] -> Enum
  │   └─> map_optional_type: Optional[T] -> Option<T>
  │
  ├─> Trait Generation Tests
  │   ├─> generate_trait_spec: class -> trait with correct methods
  │   ├─> infer_self_kind: mutating methods -> &mut self
  │   ├─> generate_rust_parameter: parameter ownership inference
  │   └─> infer_lifetimes: lifetime annotation generation
  │
  ├─> ABI Contract Tests
  │   ├─> lower_to_wasm_signature: Rust types -> WASM primitives
  │   ├─> infer_memory_contract: allocation/deallocation strategy
  │   └─> infer_abi_error_handling: error propagation across boundary
  │
  └─> NeMo Integration Tests
      ├─> build_translation_prompt: correct prompt formatting
      ├─> parse_nemo_response: JSON parsing and validation
      ├─> validate_nemo_response: response validation logic
      └─> select_best_translation: ranking and selection logic
```

### 6.2 Key Test Scenarios

```python
# Test 1: Simple function translation
def test_simple_function_specification():
    """
    GIVEN a simple Python function with type hints
    WHEN specification generator processes it
    THEN a valid Rust function signature is generated with confidence > 0.9
    """

    # Arrange
    analysis = mock_analysis_result(
        functions=[
            FunctionSignature(
                name="add",
                parameters=[
                    Parameter("a", PythonType("int")),
                    Parameter("b", PythonType("int"))
                ],
                return_type=PythonType("int")
            )
        ]
    )

    nemo_mock = mock_nemo_session(
        responses={
            "add": NeMoResponse(
                rust_code="fn add(a: i64, b: i64) -> i64 { a + b }",
                confidence=0.95
            )
        }
    )

    # Act
    spec = generate_specification(analysis, nemo_session=nemo_mock)

    # Assert
    assert len(spec.abi_contracts) == 1
    assert spec.abi_contracts[0].function_name == "add"
    assert spec.confidence_report.overall_confidence > 0.9


# Test 2: Class with mutable methods
def test_class_with_mutable_state():
    """
    GIVEN a Python class with methods that mutate state
    WHEN trait specification is generated
    THEN mutating methods use &mut self
    """

    # Arrange
    py_class = ClassDefinition(
        name="Counter",
        methods=[
            FunctionSignature(name="increment", mutates_self=true),
            FunctionSignature(name="get_count", mutates_self=false)
        ]
    )

    registry = TypeMappingRegistry()
    nemo_mock = mock_nemo_session()

    # Act
    trait_spec = generate_trait_spec(py_class, registry, nemo_mock)

    # Assert
    increment_method = find_method(trait_spec, "increment")
    get_count_method = find_method(trait_spec, "get_count")

    assert increment_method.self_kind == SelfKind.ByMutRef
    assert get_count_method.self_kind == SelfKind.ByRef


# Test 3: Type mapping with generics
def test_generic_collection_mapping():
    """
    GIVEN a Python generic collection type
    WHEN type mapping is generated
    THEN correct Rust generic type is produced
    """

    # Arrange
    py_type = PythonType(
        base_type="list",
        type_params=[PythonType("int")]
    )

    registry = TypeMappingRegistry()
    nemo_mock = mock_nemo_session()

    # Act
    mapping = generate_type_mapping(py_type, registry, nemo_mock)

    # Assert
    assert mapping.rust_type.base_type == "Vec"
    assert len(mapping.rust_type.type_args) == 1
    assert mapping.rust_type.type_args[0].base_type == "i64"


# Test 4: ABI contract for complex return type
def test_abi_contract_complex_return():
    """
    GIVEN a function returning a complex type (e.g., List[Dict])
    WHEN ABI contract is generated
    THEN memory contract specifies callee allocation
    """

    # Arrange
    function = FunctionSignature(
        name="get_data",
        return_type=PythonType("list", [
            PythonType("dict", [PythonType("str"), PythonType("int")])
        ])
    )

    registry = TypeMappingRegistry()
    nemo_mock = mock_nemo_session()

    # Act
    contract = generate_abi_contract(function, registry, nemo_mock)

    # Assert
    assert contract.memory_contract.allocator == AllocatorStrategy.CalleeAllocates
    assert contract.memory_contract.deallocation == DeallocationStrategy.CallerDeallocates
    assert contract.wasm_signature.results[0] == WasmType.Pointer


# Test 5: NeMo response validation
def test_nemo_response_validation():
    """
    GIVEN a NeMo response with invalid Rust syntax
    WHEN response is validated
    THEN validation fails
    """

    # Arrange
    request = mock_translation_request()
    response = NeMoTranslationResponse(
        primary_translation=RustTranslation(
            rust_code="fn invalid syntax {",  # Invalid Rust
            confidence=0.9
        )
    )

    # Act
    is_valid = validate_nemo_response(response, request)

    # Assert
    assert is_valid == false


# Test 6: Confidence report generation
def test_low_confidence_detection():
    """
    GIVEN a specification with some low-confidence translations
    WHEN confidence report is generated
    THEN low-confidence items are identified
    """

    # Arrange
    spec = RustSpecification(
        trait_definitions=[
            TraitSpec(name="HighConf", nemo_confidence=0.95),
            TraitSpec(name="LowConf", nemo_confidence=0.6)
        ]
    )

    analysis = mock_analysis_result()

    # Act
    report = validate_and_score(spec, analysis)

    # Assert
    assert len(report.low_confidence_items) == 1
    assert report.low_confidence_items[0].python_construct == "LowConf"
    assert report.low_confidence_items[0].confidence == 0.6
```

### 6.3 Mock Strategies

```python
# Mock NeMo Session
class MockNeMoSession:
    """
    Mock NeMo session for testing without actual GPU inference.
    """

    def __init__(self, canned_responses: Dict):
        self.canned_responses = canned_responses
        self.query_count = 0

    def query_trait_design(self, request: NeMoTranslationRequest) -> NeMoTranslationResponse:
        self.query_count += 1

        # Return canned response or default
        key = extract_key(request)
        if key in self.canned_responses:
            return self.canned_responses[key]
        else:
            return default_response()

    def shutdown(self):
        pass


# Mock Analysis Result Builder
def mock_analysis_result(**kwargs) -> AnalysisResult:
    """
    Builds a mock AnalysisResult for testing.
    """
    return AnalysisResult(
        api_surface=ApiSurface(
            functions=kwargs.get("functions", []),
            classes=kwargs.get("classes", []),
            modules=kwargs.get("modules", []),
            type_definitions=kwargs.get("types", [])
        ),
        dependency_graph=mock_dependency_graph(),
        contracts=mock_contract_registry(),
        metadata=mock_metadata()
    )
```

---

## 7. Error Handling

### 7.1 Error Types

```python
# Agent-specific errors
class SpecificationGeneratorError(Exception):
    """Base exception for specification generator errors."""
    pass

class NeMoQueryError(SpecificationGeneratorError):
    """NeMo inference failed."""
    pass

class TypeMappingError(SpecificationGeneratorError):
    """Failed to map Python type to Rust type."""
    def __init__(self, python_type: PythonType, reason: String):
        self.python_type = python_type
        self.reason = reason

class AbiContractError(SpecificationGeneratorError):
    """Failed to generate ABI contract."""
    pass

class ValidationError(SpecificationGeneratorError):
    """Generated specification failed validation."""
    pass

class UnsupportedConstruct(SpecificationGeneratorError):
    """Python construct not supported for translation."""
    def __init__(self, construct: String, reason: String):
        self.construct = construct
        self.reason = reason
```

### 7.2 Error Handling Strategy

```python
function generate_specification_safe(analysis: AnalysisResult) -> Result[RustSpecification, Error]:
    """
    Safe version of generate_specification with comprehensive error handling.
    """

    try:
        # Initialize NeMo with error handling
        nemo_session = try_initialize_nemo_session()
        if nemo_session.is_err():
            log_error("Failed to initialize NeMo session")
            return Err(nemo_session.error())

        spec = RustSpecification.new()
        type_registry = TypeMappingRegistry.new()
        errors = []

        # Process types with partial failure support
        for python_type in analysis.api_surface.type_definitions:
            try:
                mapping = generate_type_mapping(python_type, type_registry, nemo_session)
                type_registry.register(mapping)
            except TypeMappingError as e:
                log_warning(f"Failed to map type {e.python_type}: {e.reason}")
                errors.append(e)
                # Continue with fallback type
                type_registry.register_fallback(python_type)

        # Process traits with error collection
        for py_class in analysis.api_surface.classes:
            try:
                trait_spec = generate_trait_spec(py_class, type_registry, nemo_session)
                spec.trait_definitions.append(trait_spec)
            except SpecificationGeneratorError as e:
                log_error(f"Failed to generate trait for {py_class.name}: {e}")
                errors.append(e)
                # Continue with remaining classes

        # If too many errors, fail entire process
        if len(errors) > len(analysis.api_surface.classes) * 0.5:
            return Err(TooManyErrors(errors))

        # Include errors in confidence report
        spec.confidence_report.warnings.extend([str(e) for e in errors])

        nemo_session.shutdown()

        return Ok(spec)

    except Exception as e:
        log_error(f"Unexpected error in specification generation: {e}")
        return Err(e)
```

---

## 8. Performance Considerations

### 8.1 GPU Acceleration Points

```python
# GPU-accelerated operations
GPU_ACCELERATED_OPS = [
    "nemo_batch_inference",     # Batch multiple translation queries
    "embedding_similarity",     # Find similar code patterns
    "type_inference_network",   # Neural type inference
]

# Batch processing for efficiency
function batch_nemo_queries(
    requests: List[NeMoTranslationRequest],
    nemo: NeMoSession
) -> List[NeMoTranslationResponse]:
    """
    Batch multiple NeMo queries for GPU efficiency.
    """

    # Group requests by similarity for cache efficiency
    batches = group_by_similarity(requests, batch_size=8)

    responses = []
    for batch in batches:
        # Parallel GPU inference
        batch_responses = nemo.batch_generate(batch)
        responses.extend(batch_responses)

    return responses
```

### 8.2 Caching Strategy

```python
# Multi-level cache
CACHE_STRATEGY = {
    "L1": "In-memory LRU (1000 entries)",
    "L2": "Disk cache (SSD, 10000 entries)",
    "L3": "Shared network cache (Redis, unlimited)"
}

# Cache hit rate target: >80%
```

### 8.3 Performance Targets

```
- Type mapping: <10ms per type (average)
- Trait generation: <100ms per class (average)
- ABI contract: <50ms per function (average)
- NeMo query (cached): <5ms
- NeMo query (uncached): <500ms with GPU
- End-to-end: <2 seconds for 100 functions
```

---

## Summary

This pseudocode defines the **Specification Generator Agent**, which:

1. **Transforms Python API surfaces into Rust specifications** using NeMo-driven translation
2. **Generates trait definitions, type mappings, and ownership semantics** with confidence scoring
3. **Creates WASI-compatible ABI contracts** for WASM boundary crossing
4. **Provides ranked translation alternatives** for human review when confidence is low
5. **Validates and reports confidence** for all generated specifications

**Key Design Decisions:**
- NeMo integration for AI-driven translation strategies
- Confidence scoring and alternative ranking for quality assurance
- Comprehensive type mapping covering primitives, collections, and custom types
- Ownership inference based on mutation analysis and NeMo recommendations
- ABI contract generation with explicit memory management semantics
- Multi-level caching for performance optimization
- Fallback strategies for NeMo failures

**Testing Strategy:**
- London School TDD with outside-in approach
- Mocked NeMo sessions for deterministic testing
- Contract tests for all data structure boundaries
- Property-based tests for type mapping correctness

This specification is ready for **SPARC Phase 3 (Architecture)** where we'll refine the component interactions and define detailed interfaces.

---

**END OF PSEUDOCODE DOCUMENT**
