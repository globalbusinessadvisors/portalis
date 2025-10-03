# PORTALIS: Analysis Agent Pseudocode
## SPARC Phase 2: Pseudocode

**Version:** 1.0
**Date:** 2025-10-03
**Component:** Analysis Agent (Agent #2)
**Testing Approach:** London School TDD (Outside-In, Mockist)

---

## Table of Contents

1. [Agent Overview](#1-agent-overview)
2. [Data Structures](#2-data-structures)
3. [Core Algorithms](#3-core-algorithms)
4. [Input/Output Contracts](#4-inputoutput-contracts)
5. [Error Handling Strategy](#5-error-handling-strategy)
6. [London School TDD Test Points](#6-london-school-tdd-test-points)

---

## 1. Agent Overview

### 1.1 Purpose

The Analysis Agent is responsible for deep semantic analysis of Python codebases. It extracts the complete API surface, constructs dependency graphs, discovers contracts from type hints and docstrings, and performs semantic analysis to identify mutability, side effects, and concurrency patterns.

### 1.2 Responsibilities

1. **API Surface Extraction** (FR-2.2.1)
   - Identify all public functions, classes, and methods
   - Extract function signatures with parameter types
   - Capture return types from annotations or inference
   - Extract docstrings and structured content
   - Detect exception types raised by each function

2. **Dependency Graph Construction** (FR-2.2.2)
   - Build complete call graphs for the codebase
   - Identify data flow between functions and modules
   - Detect external library dependencies
   - Categorize dependencies (stdlib, third-party, internal)
   - Identify and report circular dependencies

3. **Contract Discovery** (FR-2.2.3)
   - Extract input/output contracts from type hints
   - Infer contracts from docstrings (NumPy, Google, Sphinx formats)
   - Identify invariants from assertions and conditionals
   - Capture preconditions and postconditions

4. **Semantic Analysis** (FR-2.2.4)
   - Identify mutable vs immutable data structures
   - Detect side effects (I/O, global state, exceptions)
   - Classify functions as pure, impure, or effectful
   - Identify concurrency patterns (async/await, threading, multiprocessing)

### 1.3 Input Dependencies

- **AST Tree**: From Ingest Agent (parsed Python AST)
- **Source Code**: Original Python source files
- **Module Metadata**: Package structure, import resolution

### 1.4 Output Artifacts

- **API Catalog**: Complete public API surface with signatures
- **Dependency Graph**: Call graph, data flow graph, dependency tree
- **Contract Database**: Type contracts, invariants, pre/postconditions
- **Semantic Annotations**: Purity, mutability, side effects, concurrency markers

---

## 2. Data Structures

### 2.1 AST Representations

```pseudocode
// Enhanced AST Node with Analysis Metadata
STRUCTURE ASTNode:
    node_type: NodeType  // FunctionDef, ClassDef, Call, Assign, etc.
    source_location: SourceLocation
    children: List<ASTNode>
    attributes: Map<String, Value>
    analysis_metadata: AnalysisMetadata

STRUCTURE SourceLocation:
    file_path: String
    line_start: Integer
    line_end: Integer
    column_start: Integer
    column_end: Integer

STRUCTURE AnalysisMetadata:
    is_public: Boolean
    is_exported: Boolean
    scope: Scope  // module, class, function, block
    parent_scope: Optional<Scope>

ENUM NodeType:
    MODULE, CLASS_DEF, FUNCTION_DEF, METHOD_DEF,
    ASSIGN, CALL, IMPORT, ATTRIBUTE, NAME,
    IF, FOR, WHILE, WITH, TRY, RAISE, RETURN
```

### 2.2 API Surface Structures

```pseudocode
// API Catalog - Complete public interface
STRUCTURE APICatalog:
    functions: List<FunctionSignature>
    classes: List<ClassDefinition>
    constants: List<Constant>
    modules: List<ModuleInfo>

STRUCTURE FunctionSignature:
    name: String
    qualified_name: String  // module.class.function
    parameters: List<Parameter>
    return_type: TypeInfo
    exceptions: List<ExceptionType>
    docstring: Docstring
    decorators: List<Decorator>
    is_async: Boolean
    is_generator: Boolean
    source_location: SourceLocation
    visibility: Visibility  // PUBLIC, PRIVATE, PROTECTED

STRUCTURE Parameter:
    name: String
    type_annotation: Optional<TypeInfo>
    default_value: Optional<Value>
    kind: ParameterKind  // POSITIONAL, KEYWORD, VAR_POSITIONAL, VAR_KEYWORD

STRUCTURE TypeInfo:
    base_type: String  // int, str, List, Dict, etc.
    type_params: List<TypeInfo>  // for generics
    is_optional: Boolean
    is_union: Boolean
    union_types: List<TypeInfo>
    is_callable: Boolean
    callable_signature: Optional<FunctionSignature>

STRUCTURE ClassDefinition:
    name: String
    qualified_name: String
    base_classes: List<String>
    methods: List<FunctionSignature>
    attributes: List<ClassAttribute>
    docstring: Docstring
    decorators: List<Decorator>
    is_abstract: Boolean
    metaclass: Optional<String>

STRUCTURE ClassAttribute:
    name: String
    type_annotation: Optional<TypeInfo>
    default_value: Optional<Value>
    is_class_var: Boolean

STRUCTURE Docstring:
    raw_text: String
    format: DocFormat  // NUMPY, GOOGLE, SPHINX, PLAIN
    summary: String
    description: String
    parameters: List<ParamDoc>
    returns: ReturnDoc
    raises: List<ExceptionDoc>
    examples: List<Example>

ENUM DocFormat:
    NUMPY, GOOGLE, SPHINX, RESTRUCTURED_TEXT, PLAIN

STRUCTURE ParamDoc:
    name: String
    type_hint: Optional<String>
    description: String

STRUCTURE ReturnDoc:
    type_hint: Optional<String>
    description: String
```

### 2.3 Dependency Graph Structures

```pseudocode
// Call Graph - Function-level dependency tracking
STRUCTURE CallGraph:
    nodes: Map<FunctionID, CallGraphNode>
    edges: List<CallEdge>
    entry_points: Set<FunctionID>

STRUCTURE CallGraphNode:
    function_id: FunctionID
    function_signature: FunctionSignature
    calls_to: Set<FunctionID>
    called_by: Set<FunctionID>

STRUCTURE CallEdge:
    caller: FunctionID
    callee: FunctionID
    call_site: SourceLocation
    is_conditional: Boolean  // called within if/try/etc
    is_dynamic: Boolean  // dynamic dispatch, reflection

STRUCTURE FunctionID:
    qualified_name: String
    module_path: String

// Data Flow Graph
STRUCTURE DataFlowGraph:
    nodes: Map<VariableID, DataFlowNode>
    edges: List<DataFlowEdge>

STRUCTURE DataFlowNode:
    variable_id: VariableID
    definition_site: SourceLocation
    type_info: TypeInfo
    is_parameter: Boolean
    is_global: Boolean
    is_nonlocal: Boolean

STRUCTURE DataFlowEdge:
    from_var: VariableID
    to_var: VariableID
    edge_type: DataFlowType

ENUM DataFlowType:
    ASSIGNMENT, PARAMETER_PASS, RETURN, ATTRIBUTE_ACCESS, MUTATION

// Dependency Tree
STRUCTURE DependencyTree:
    root: ModuleNode
    external_deps: List<ExternalDependency>
    circular_deps: List<CircularDependency>

STRUCTURE ModuleNode:
    module_name: String
    module_path: String
    imports: List<Import>
    internal_deps: List<ModuleNode>

STRUCTURE Import:
    module_name: String
    imported_names: List<String>
    import_type: ImportType  // STDLIB, THIRD_PARTY, INTERNAL

ENUM ImportType:
    STDLIB, THIRD_PARTY, INTERNAL, UNKNOWN

STRUCTURE ExternalDependency:
    package_name: String
    version: Optional<String>
    import_type: ImportType
    used_symbols: Set<String>

STRUCTURE CircularDependency:
    cycle: List<String>  // module names in cycle
    severity: Severity  // WARNING, ERROR
```

### 2.4 Contract Structures

```pseudocode
// Contract Database
STRUCTURE ContractDatabase:
    function_contracts: Map<FunctionID, FunctionContract>
    class_contracts: Map<ClassID, ClassContract>
    invariants: List<Invariant>

STRUCTURE FunctionContract:
    function_id: FunctionID
    preconditions: List<Condition>
    postconditions: List<Condition>
    input_contract: InputContract
    output_contract: OutputContract
    exception_contract: ExceptionContract

STRUCTURE InputContract:
    parameters: List<ParameterConstraint>

STRUCTURE ParameterConstraint:
    param_name: String
    type_constraint: TypeInfo
    value_constraints: List<ValueConstraint>

STRUCTURE ValueConstraint:
    constraint_type: ConstraintType
    expression: String

ENUM ConstraintType:
    RANGE, NOT_NULL, POSITIVE, LENGTH, REGEX, CUSTOM

STRUCTURE OutputContract:
    return_type: TypeInfo
    return_constraints: List<ValueConstraint>

STRUCTURE ExceptionContract:
    exception_types: List<ExceptionType>
    conditions: List<Condition>  // when each exception is raised

STRUCTURE Condition:
    expression: String
    variables: Set<String>
    condition_type: ConditionType

ENUM ConditionType:
    ASSERT, IF_CHECK, RAISE_CONDITION, INVARIANT

STRUCTURE Invariant:
    scope: Scope  // function, class, module
    expression: String
    invariant_type: InvariantType

ENUM InvariantType:
    LOOP_INVARIANT, CLASS_INVARIANT, MODULE_INVARIANT
```

### 2.5 Semantic Analysis Structures

```pseudocode
// Semantic Annotations
STRUCTURE SemanticAnnotations:
    function_semantics: Map<FunctionID, FunctionSemantics>
    variable_semantics: Map<VariableID, VariableSemantics>

STRUCTURE FunctionSemantics:
    function_id: FunctionID
    purity: Purity
    side_effects: List<SideEffect>
    mutates: Set<VariableID>  // what this function mutates
    reads_global: Set<String>
    writes_global: Set<String>
    io_operations: List<IOOperation>
    concurrency_pattern: Optional<ConcurrencyPattern>

ENUM Purity:
    PURE,           // no side effects, deterministic
    IMPURE,         // has side effects
    CONDITIONALLY_PURE,  // pure under certain conditions
    UNKNOWN

STRUCTURE SideEffect:
    effect_type: SideEffectType
    target: Optional<String>
    location: SourceLocation

ENUM SideEffectType:
    IO_READ, IO_WRITE, GLOBAL_MUTATION,
    EXCEPTION_RAISED, SYSTEM_CALL,
    RANDOM_GENERATION, TIME_DEPENDENT

STRUCTURE IOOperation:
    operation_type: IOType
    resource: String  // file path, network address, etc.
    mode: String  // read, write, append

ENUM IOType:
    FILE_READ, FILE_WRITE, NETWORK, DATABASE, STDOUT, STDERR

STRUCTURE ConcurrencyPattern:
    pattern_type: ConcurrencyType
    primitives_used: Set<String>  // Lock, Queue, asyncio.gather, etc.
    synchronization: List<SyncPrimitive>

ENUM ConcurrencyType:
    ASYNC_AWAIT, THREADING, MULTIPROCESSING,
    CONCURRENT_FUTURES, NONE

STRUCTURE SyncPrimitive:
    primitive_type: String  // Lock, RLock, Semaphore, Event
    location: SourceLocation

STRUCTURE VariableSemantics:
    variable_id: VariableID
    mutability: Mutability
    lifetime: Lifetime

ENUM Mutability:
    IMMUTABLE,      // never mutated after creation
    MUTABLE,        // can be mutated
    CONDITIONALLY_MUTABLE,  // mutated under conditions
    UNKNOWN

STRUCTURE Lifetime:
    scope: Scope
    first_def: SourceLocation
    last_use: SourceLocation
```

---

## 3. Core Algorithms

### 3.1 API Surface Extraction

```pseudocode
// Main API extraction algorithm
FUNCTION extract_api_surface(ast_root: ASTNode, source_files: List<File>) -> APICatalog:
    catalog = NEW APICatalog()

    // Phase 1: Discover all definitions
    FOR EACH node IN ast_root.traverse(depth_first):
        IF node.type == FUNCTION_DEF:
            sig = extract_function_signature(node)
            IF is_public(sig):
                catalog.functions.append(sig)

        ELSE IF node.type == CLASS_DEF:
            class_def = extract_class_definition(node)
            IF is_public(class_def):
                catalog.classes.append(class_def)

        ELSE IF node.type == ASSIGN AND is_module_level(node):
            constant = extract_constant(node)
            IF is_public(constant):
                catalog.constants.append(constant)

    // Phase 2: Resolve forward references and type hints
    FOR EACH function IN catalog.functions:
        resolve_type_hints(function, catalog)

    FOR EACH class IN catalog.classes:
        resolve_base_classes(class, catalog)
        FOR EACH method IN class.methods:
            resolve_type_hints(method, catalog)

    RETURN catalog

// Extract function signature with all metadata
FUNCTION extract_function_signature(node: ASTNode) -> FunctionSignature:
    sig = NEW FunctionSignature()
    sig.name = node.attributes["name"]
    sig.qualified_name = build_qualified_name(node)
    sig.source_location = node.source_location

    // Extract parameters
    FOR EACH param IN node.children.filter(type=PARAMETER):
        p = NEW Parameter()
        p.name = param.attributes["name"]
        p.type_annotation = extract_type_annotation(param)
        p.default_value = extract_default_value(param)
        p.kind = determine_parameter_kind(param)
        sig.parameters.append(p)

    // Extract return type
    return_annotation = node.attributes.get("returns")
    IF return_annotation IS NOT NULL:
        sig.return_type = parse_type_annotation(return_annotation)
    ELSE:
        sig.return_type = infer_return_type(node)

    // Extract exceptions
    sig.exceptions = extract_raised_exceptions(node)

    // Extract docstring
    sig.docstring = parse_docstring(node.get_docstring())

    // Extract decorators
    sig.decorators = extract_decorators(node)

    // Determine async and generator flags
    sig.is_async = "async" IN node.attributes
    sig.is_generator = contains_yield(node)

    // Determine visibility
    sig.visibility = determine_visibility(sig.name)

    RETURN sig

// Parse and structure docstrings
FUNCTION parse_docstring(raw_text: String) -> Docstring:
    IF raw_text IS NULL OR raw_text.is_empty():
        RETURN empty_docstring()

    doc = NEW Docstring()
    doc.raw_text = raw_text
    doc.format = detect_docstring_format(raw_text)

    SWITCH doc.format:
        CASE NUMPY:
            parse_numpy_docstring(raw_text, doc)
        CASE GOOGLE:
            parse_google_docstring(raw_text, doc)
        CASE SPHINX:
            parse_sphinx_docstring(raw_text, doc)
        DEFAULT:
            doc.summary = raw_text.split("\n\n")[0]
            doc.description = raw_text

    RETURN doc

// Detect docstring format
FUNCTION detect_docstring_format(text: String) -> DocFormat:
    IF text.contains("Parameters\n----------"):
        RETURN NUMPY
    ELSE IF text.contains("Args:") AND text.contains("Returns:"):
        RETURN GOOGLE
    ELSE IF text.contains(":param ") OR text.contains(":type "):
        RETURN SPHINX
    ELSE:
        RETURN PLAIN

// Extract raised exceptions from function body
FUNCTION extract_raised_exceptions(node: ASTNode) -> List<ExceptionType>:
    exceptions = NEW Set<ExceptionType>()

    FOR EACH raise_node IN node.traverse().filter(type=RAISE):
        exception_type = get_exception_type(raise_node)
        IF exception_type IS NOT NULL:
            exceptions.add(exception_type)

    // Also check docstring for documented exceptions
    docstring = node.get_docstring()
    IF docstring IS NOT NULL:
        documented_exceptions = parse_raises_section(docstring)
        exceptions.add_all(documented_exceptions)

    RETURN exceptions.to_list()

// Infer return type through static analysis
FUNCTION infer_return_type(node: ASTNode) -> TypeInfo:
    return_nodes = node.traverse().filter(type=RETURN)

    IF return_nodes.is_empty():
        RETURN TypeInfo(base_type="None")

    inferred_types = NEW Set<TypeInfo>()

    FOR EACH return_node IN return_nodes:
        return_value = return_node.get_child("value")
        IF return_value IS NOT NULL:
            inferred_type = infer_expression_type(return_value)
            inferred_types.add(inferred_type)

    IF inferred_types.size() == 1:
        RETURN inferred_types.first()
    ELSE IF inferred_types.size() > 1:
        // Multiple return types - create union
        RETURN TypeInfo(
            is_union=true,
            union_types=inferred_types.to_list()
        )
    ELSE:
        RETURN TypeInfo(base_type="Unknown")

// Type inference for expressions
FUNCTION infer_expression_type(expr: ASTNode) -> TypeInfo:
    SWITCH expr.type:
        CASE CONSTANT:
            RETURN infer_from_constant(expr.attributes["value"])
        CASE NAME:
            RETURN lookup_variable_type(expr.attributes["id"])
        CASE CALL:
            callee = expr.get_child("func")
            func_sig = resolve_function(callee)
            IF func_sig IS NOT NULL:
                RETURN func_sig.return_type
            RETURN TypeInfo(base_type="Unknown")
        CASE LIST, TUPLE, SET:
            element_types = []
            FOR EACH element IN expr.get_children("elts"):
                element_types.append(infer_expression_type(element))
            RETURN TypeInfo(
                base_type=expr.type.to_string(),
                type_params=unify_types(element_types)
            )
        CASE DICT:
            key_types = []
            value_types = []
            FOR EACH key IN expr.get_children("keys"):
                key_types.append(infer_expression_type(key))
            FOR EACH value IN expr.get_children("values"):
                value_types.append(infer_expression_type(value))
            RETURN TypeInfo(
                base_type="Dict",
                type_params=[unify_types(key_types), unify_types(value_types)]
            )
        DEFAULT:
            RETURN TypeInfo(base_type="Unknown")

// Determine visibility based on naming conventions
FUNCTION determine_visibility(name: String) -> Visibility:
    IF name.starts_with("__") AND NOT name.ends_with("__"):
        RETURN PRIVATE  // name mangling
    ELSE IF name.starts_with("_"):
        RETURN PROTECTED
    ELSE:
        RETURN PUBLIC
```

### 3.2 Dependency Graph Construction

```pseudocode
// Build complete call graph
FUNCTION build_call_graph(ast_root: ASTNode, api_catalog: APICatalog) -> CallGraph:
    graph = NEW CallGraph()

    // Phase 1: Create nodes for all functions
    FOR EACH function IN api_catalog.functions:
        func_id = FunctionID(
            qualified_name=function.qualified_name,
            module_path=function.source_location.file_path
        )
        node = NEW CallGraphNode(
            function_id=func_id,
            function_signature=function
        )
        graph.nodes[func_id] = node

    // Phase 2: Extract call relationships
    FOR EACH function IN api_catalog.functions:
        func_node = find_function_ast_node(ast_root, function)
        calls = extract_function_calls(func_node)

        FOR EACH call IN calls:
            callee_id = resolve_call_target(call, function, api_catalog)
            IF callee_id IS NOT NULL:
                edge = NEW CallEdge(
                    caller=function.to_id(),
                    callee=callee_id,
                    call_site=call.source_location,
                    is_conditional=is_conditional_call(call),
                    is_dynamic=is_dynamic_call(call)
                )
                graph.edges.append(edge)

                // Update adjacency
                graph.nodes[function.to_id()].calls_to.add(callee_id)
                IF callee_id IN graph.nodes:
                    graph.nodes[callee_id].called_by.add(function.to_id())

    // Phase 3: Identify entry points
    FOR EACH (func_id, node) IN graph.nodes:
        IF node.called_by.is_empty():
            graph.entry_points.add(func_id)

    RETURN graph

// Extract all function calls from a function body
FUNCTION extract_function_calls(func_node: ASTNode) -> List<ASTNode>:
    calls = []

    FOR EACH node IN func_node.traverse(depth_first):
        IF node.type == CALL:
            calls.append(node)

    RETURN calls

// Resolve call target to function ID
FUNCTION resolve_call_target(call_node: ASTNode, caller: FunctionSignature,
                              catalog: APICatalog) -> Optional<FunctionID>:
    func_expr = call_node.get_child("func")

    IF func_expr.type == NAME:
        // Simple function call: foo()
        func_name = func_expr.attributes["id"]
        RETURN resolve_name_in_scope(func_name, caller, catalog)

    ELSE IF func_expr.type == ATTRIBUTE:
        // Method call: obj.method()
        obj_expr = func_expr.get_child("value")
        method_name = func_expr.attributes["attr"]

        obj_type = infer_expression_type(obj_expr)
        class_def = catalog.find_class(obj_type.base_type)

        IF class_def IS NOT NULL:
            method = class_def.find_method(method_name)
            IF method IS NOT NULL:
                RETURN method.to_id()

        RETURN NULL  // dynamic dispatch, cannot resolve statically

    ELSE:
        // Complex call (lambda, higher-order function)
        RETURN NULL

// Build data flow graph
FUNCTION build_data_flow_graph(ast_root: ASTNode) -> DataFlowGraph:
    graph = NEW DataFlowGraph()
    scopes = NEW ScopeStack()

    FUNCTION visit_node(node: ASTNode, scope: Scope):
        SWITCH node.type:
            CASE ASSIGN:
                targets = node.get_children("targets")
                value = node.get_child("value")

                FOR EACH target IN targets:
                    var_id = create_variable_id(target, scope)
                    df_node = NEW DataFlowNode(
                        variable_id=var_id,
                        definition_site=node.source_location,
                        type_info=infer_expression_type(value)
                    )
                    graph.nodes[var_id] = df_node

                    // Add data flow edges from value to target
                    value_vars = extract_variables(value)
                    FOR EACH v IN value_vars:
                        edge = NEW DataFlowEdge(
                            from_var=v,
                            to_var=var_id,
                            edge_type=ASSIGNMENT
                        )
                        graph.edges.append(edge)

            CASE FUNCTION_DEF:
                new_scope = create_function_scope(node, scope)
                scopes.push(new_scope)

                // Process parameters
                FOR EACH param IN node.get_children("args"):
                    var_id = create_variable_id(param, new_scope)
                    df_node = NEW DataFlowNode(
                        variable_id=var_id,
                        definition_site=param.source_location,
                        type_info=extract_type_annotation(param),
                        is_parameter=true
                    )
                    graph.nodes[var_id] = df_node

                // Process body
                FOR EACH child IN node.get_children("body"):
                    visit_node(child, new_scope)

                scopes.pop()

            CASE RETURN:
                value = node.get_child("value")
                IF value IS NOT NULL:
                    return_vars = extract_variables(value)
                    // Mark these as return flow
                    FOR EACH var IN return_vars:
                        // Create edge to special RETURN node
                        edge = NEW DataFlowEdge(
                            from_var=var,
                            to_var=VariableID("__return__", scope),
                            edge_type=RETURN
                        )
                        graph.edges.append(edge)

            DEFAULT:
                FOR EACH child IN node.children:
                    visit_node(child, scope)

    visit_node(ast_root, scopes.global_scope)
    RETURN graph

// Build dependency tree
FUNCTION build_dependency_tree(ast_root: ASTNode, package_root: Path) -> DependencyTree:
    tree = NEW DependencyTree()
    visited = NEW Set<String>()

    FUNCTION process_module(module_node: ASTNode, parent: Optional<ModuleNode>) -> ModuleNode:
        module_name = get_module_name(module_node)

        IF module_name IN visited:
            RETURN NULL  // avoid infinite recursion

        visited.add(module_name)

        node = NEW ModuleNode(
            module_name=module_name,
            module_path=module_node.source_location.file_path
        )

        // Extract imports
        FOR EACH import_node IN module_node.children.filter(type IN [IMPORT, IMPORT_FROM]):
            import_info = extract_import(import_node)
            node.imports.append(import_info)

            // Categorize import
            import_type = categorize_import(import_info.module_name)
            import_info.import_type = import_type

            IF import_type == STDLIB:
                // Track stdlib usage
                tree.stdlib_deps.add(import_info.module_name)
            ELSE IF import_type == THIRD_PARTY:
                // Track external dependency
                ext_dep = NEW ExternalDependency(
                    package_name=get_package_name(import_info.module_name),
                    used_symbols=import_info.imported_names.to_set()
                )
                tree.external_deps.append(ext_dep)
            ELSE IF import_type == INTERNAL:
                // Recursively process internal module
                internal_module = load_module(import_info.module_name, package_root)
                IF internal_module IS NOT NULL:
                    child_node = process_module(internal_module, node)
                    IF child_node IS NOT NULL:
                        node.internal_deps.append(child_node)

        RETURN node

    tree.root = process_module(ast_root, NULL)

    // Detect circular dependencies
    tree.circular_deps = detect_cycles(tree.root)

    RETURN tree

// Detect circular dependencies using DFS
FUNCTION detect_cycles(root: ModuleNode) -> List<CircularDependency>:
    cycles = []
    visited = NEW Set<String>()
    rec_stack = NEW Stack<String>()

    FUNCTION dfs(node: ModuleNode):
        IF node.module_name IN rec_stack:
            // Found a cycle
            cycle_start = rec_stack.index_of(node.module_name)
            cycle = rec_stack.slice(cycle_start).append(node.module_name)
            cycles.append(NEW CircularDependency(
                cycle=cycle,
                severity=ERROR
            ))
            RETURN

        IF node.module_name IN visited:
            RETURN

        visited.add(node.module_name)
        rec_stack.push(node.module_name)

        FOR EACH dep IN node.internal_deps:
            dfs(dep)

        rec_stack.pop()

    dfs(root)
    RETURN cycles

// Categorize import as stdlib, third-party, or internal
FUNCTION categorize_import(module_name: String) -> ImportType:
    // Check if stdlib
    IF module_name IN PYTHON_STDLIB_MODULES:
        RETURN STDLIB

    // Check if relative import or local package
    IF module_name.starts_with("."):
        RETURN INTERNAL

    // Check if in site-packages
    module_file = locate_module(module_name)
    IF module_file IS NOT NULL:
        IF "site-packages" IN module_file.path:
            RETURN THIRD_PARTY
        ELSE IF is_in_project_root(module_file):
            RETURN INTERNAL

    RETURN UNKNOWN
```

### 3.3 Contract Discovery

```pseudocode
// Extract contracts from type hints and docstrings
FUNCTION discover_contracts(api_catalog: APICatalog, ast_root: ASTNode) -> ContractDatabase:
    db = NEW ContractDatabase()

    FOR EACH function IN api_catalog.functions:
        contract = extract_function_contract(function, ast_root)
        db.function_contracts[function.to_id()] = contract

    FOR EACH class IN api_catalog.classes:
        class_contract = extract_class_contract(class, ast_root)
        db.class_contracts[class.to_id()] = class_contract

    // Extract module-level invariants
    db.invariants = extract_module_invariants(ast_root)

    RETURN db

// Extract complete contract for a function
FUNCTION extract_function_contract(function: FunctionSignature, ast_root: ASTNode) -> FunctionContract:
    contract = NEW FunctionContract(function_id=function.to_id())

    func_node = find_function_ast_node(ast_root, function)

    // Extract input contract from type hints and docstring
    contract.input_contract = extract_input_contract(function)

    // Extract preconditions from assertions and docstring
    contract.preconditions = extract_preconditions(func_node, function.docstring)

    // Extract postconditions
    contract.postconditions = extract_postconditions(func_node, function.docstring)

    // Extract output contract
    contract.output_contract = extract_output_contract(function)

    // Extract exception contract
    contract.exception_contract = extract_exception_contract(function, func_node)

    RETURN contract

// Extract input contract from parameters
FUNCTION extract_input_contract(function: FunctionSignature) -> InputContract:
    input_contract = NEW InputContract()

    FOR EACH param IN function.parameters:
        constraint = NEW ParameterConstraint(
            param_name=param.name,
            type_constraint=param.type_annotation
        )

        // Extract value constraints from docstring
        IF function.docstring IS NOT NULL:
            param_doc = function.docstring.find_parameter(param.name)
            IF param_doc IS NOT NULL:
                value_constraints = parse_constraints_from_doc(param_doc.description)
                constraint.value_constraints = value_constraints

        input_contract.parameters.append(constraint)

    RETURN input_contract

// Extract preconditions from assertions and docstring
FUNCTION extract_preconditions(func_node: ASTNode, docstring: Docstring) -> List<Condition>:
    preconditions = []

    // Extract from assertions at function start
    first_statements = get_first_statements(func_node, limit=10)

    FOR EACH stmt IN first_statements:
        IF stmt.type == ASSERT:
            test_expr = stmt.get_child("test")
            condition = NEW Condition(
                expression=ast_to_string(test_expr),
                variables=extract_variable_names(test_expr),
                condition_type=ASSERT
            )
            preconditions.append(condition)
        ELSE IF stmt.type != EXPR:  // Skip docstrings
            BREAK  // Stop at first non-assertion

    // Extract from docstring (e.g., "Requires: x > 0")
    IF docstring IS NOT NULL:
        doc_preconditions = parse_preconditions_from_docstring(docstring)
        preconditions.extend(doc_preconditions)

    RETURN preconditions

// Extract postconditions
FUNCTION extract_postconditions(func_node: ASTNode, docstring: Docstring) -> List<Condition>:
    postconditions = []

    // Extract from return statements with conditions
    FOR EACH return_node IN func_node.traverse().filter(type=RETURN):
        parent = return_node.parent
        IF parent.type == IF:
            condition_expr = parent.get_child("test")
            condition = NEW Condition(
                expression=ast_to_string(condition_expr),
                variables=extract_variable_names(condition_expr),
                condition_type=IF_CHECK
            )
            postconditions.append(condition)

    // Extract from docstring
    IF docstring IS NOT NULL:
        doc_postconditions = parse_postconditions_from_docstring(docstring)
        postconditions.extend(doc_postconditions)

    RETURN postconditions

// Extract output contract
FUNCTION extract_output_contract(function: FunctionSignature) -> OutputContract:
    output_contract = NEW OutputContract(
        return_type=function.return_type
    )

    // Extract constraints from docstring
    IF function.docstring IS NOT NULL AND function.docstring.returns IS NOT NULL:
        return_desc = function.docstring.returns.description
        constraints = parse_constraints_from_doc(return_desc)
        output_contract.return_constraints = constraints

    RETURN output_contract

// Parse constraints from natural language descriptions
FUNCTION parse_constraints_from_doc(description: String) -> List<ValueConstraint>:
    constraints = []

    // Pattern matching for common constraint patterns
    patterns = [
        (REGEX("must be positive"), NEW ValueConstraint(POSITIVE, "x > 0")),
        (REGEX("between (\\d+) and (\\d+)"), NEW ValueConstraint(RANGE, "min <= x <= max")),
        (REGEX("non-null|not null|cannot be null"), NEW ValueConstraint(NOT_NULL, "x is not None")),
        (REGEX("length (<=?|>=?|==) (\\d+)"), NEW ValueConstraint(LENGTH, "len(x) op value")),
    ]

    FOR EACH (pattern, constraint_template) IN patterns:
        matches = pattern.find_all(description.lower())
        IF matches IS NOT EMPTY:
            FOR EACH match IN matches:
                constraint = instantiate_constraint(constraint_template, match)
                constraints.append(constraint)

    RETURN constraints

// Extract exception contract
FUNCTION extract_exception_contract(function: FunctionSignature, func_node: ASTNode) -> ExceptionContract:
    contract = NEW ExceptionContract()

    // Get exception types (already extracted in API surface)
    contract.exception_types = function.exceptions

    // Extract conditions under which exceptions are raised
    FOR EACH raise_node IN func_node.traverse().filter(type=RAISE):
        exc_type = get_exception_type(raise_node)
        condition = extract_raise_condition(raise_node)

        IF condition IS NOT NULL:
            contract.conditions.append(condition)

    RETURN contract

// Extract condition under which exception is raised
FUNCTION extract_raise_condition(raise_node: ASTNode) -> Optional<Condition>:
    parent = raise_node.parent

    // Check if raise is inside an if statement
    WHILE parent IS NOT NULL:
        IF parent.type == IF:
            test_expr = parent.get_child("test")
            RETURN NEW Condition(
                expression=ast_to_string(test_expr),
                variables=extract_variable_names(test_expr),
                condition_type=RAISE_CONDITION
            )
        ELSE IF parent.type IN [FUNCTION_DEF, CLASS_DEF]:
            BREAK
        parent = parent.parent

    RETURN NULL

// Extract class-level invariants
FUNCTION extract_class_contract(class_def: ClassDefinition, ast_root: ASTNode) -> ClassContract:
    contract = NEW ClassContract()

    class_node = find_class_ast_node(ast_root, class_def)

    // Look for __init__ and extract initialization invariants
    init_method = class_def.find_method("__init__")
    IF init_method IS NOT NULL:
        contract.init_contract = extract_function_contract(init_method, class_node)

    // Extract class invariants from docstring
    IF class_def.docstring IS NOT NULL:
        contract.invariants = parse_invariants_from_docstring(class_def.docstring)

    // Look for validation methods (often indicate invariants)
    FOR EACH method IN class_def.methods:
        IF method.name.starts_with("validate") OR method.name.starts_with("check"):
            validation_invariants = extract_validation_invariants(method, class_node)
            contract.invariants.extend(validation_invariants)

    RETURN contract
```

### 3.4 Semantic Analysis

```pseudocode
// Perform semantic analysis on all functions
FUNCTION analyze_semantics(api_catalog: APICatalog, call_graph: CallGraph,
                           data_flow_graph: DataFlowGraph) -> SemanticAnnotations:
    annotations = NEW SemanticAnnotations()

    // Phase 1: Analyze each function
    FOR EACH function IN api_catalog.functions:
        semantics = analyze_function_semantics(function, call_graph, data_flow_graph)
        annotations.function_semantics[function.to_id()] = semantics

    // Phase 2: Propagate purity information through call graph
    propagate_purity(annotations, call_graph)

    // Phase 3: Analyze variable semantics
    annotations.variable_semantics = analyze_variable_semantics(data_flow_graph)

    RETURN annotations

// Analyze semantic properties of a single function
FUNCTION analyze_function_semantics(function: FunctionSignature, call_graph: CallGraph,
                                    data_flow_graph: DataFlowGraph) -> FunctionSemantics:
    semantics = NEW FunctionSemantics(function_id=function.to_id())

    func_node = find_function_ast_node(function)

    // Detect side effects
    semantics.side_effects = detect_side_effects(func_node)

    // Detect global variable access
    (reads, writes) = analyze_global_access(func_node)
    semantics.reads_global = reads
    semantics.writes_global = writes

    // Detect mutations
    semantics.mutates = detect_mutations(func_node, data_flow_graph)

    // Detect I/O operations
    semantics.io_operations = detect_io_operations(func_node)

    // Classify purity
    semantics.purity = classify_purity(semantics)

    // Detect concurrency patterns
    semantics.concurrency_pattern = detect_concurrency_pattern(func_node)

    RETURN semantics

// Detect all side effects in a function
FUNCTION detect_side_effects(func_node: ASTNode) -> List<SideEffect>:
    side_effects = []

    FOR EACH node IN func_node.traverse():
        SWITCH node.type:
            CASE CALL:
                callee = get_callee_name(node)

                // Check for known side-effecting functions
                IF callee IN ["print", "input", "open"]:
                    side_effects.append(NEW SideEffect(
                        effect_type=IO_WRITE,
                        location=node.source_location
                    ))
                ELSE IF callee IN ["random.random", "random.randint"]:
                    side_effects.append(NEW SideEffect(
                        effect_type=RANDOM_GENERATION,
                        location=node.source_location
                    ))
                ELSE IF callee IN ["time.time", "datetime.now"]:
                    side_effects.append(NEW SideEffect(
                        effect_type=TIME_DEPENDENT,
                        location=node.source_location
                    ))

            CASE ASSIGN:
                target = node.get_child("targets")[0]
                IF is_global_variable(target) OR is_nonlocal_variable(target):
                    side_effects.append(NEW SideEffect(
                        effect_type=GLOBAL_MUTATION,
                        target=get_variable_name(target),
                        location=node.source_location
                    ))

            CASE RAISE:
                side_effects.append(NEW SideEffect(
                    effect_type=EXCEPTION_RAISED,
                    target=get_exception_type(node),
                    location=node.source_location
                ))

    RETURN side_effects

// Analyze global variable access
FUNCTION analyze_global_access(func_node: ASTNode) -> (Set<String>, Set<String>):
    reads = NEW Set<String>()
    writes = NEW Set<String>()

    global_vars = extract_global_declarations(func_node)
    nonlocal_vars = extract_nonlocal_declarations(func_node)

    FOR EACH node IN func_node.traverse():
        IF node.type == NAME:
            var_name = node.attributes["id"]
            context = node.attributes["ctx"]

            IF var_name IN global_vars OR is_implicitly_global(var_name, func_node):
                IF context == LOAD:
                    reads.add(var_name)
                ELSE IF context IN [STORE, DEL]:
                    writes.add(var_name)

        ELSE IF node.type == ASSIGN:
            target = node.get_child("targets")[0]
            IF target.type == NAME AND target.attributes["id"] IN global_vars:
                writes.add(target.attributes["id"])

    RETURN (reads, writes)

// Detect mutations to parameters or variables
FUNCTION detect_mutations(func_node: ASTNode, data_flow_graph: DataFlowGraph) -> Set<VariableID>:
    mutations = NEW Set<VariableID>()

    // Get function parameters
    params = func_node.get_child("args").get_children()
    param_names = [p.attributes["name"] FOR p IN params]

    FOR EACH node IN func_node.traverse():
        IF node.type == ASSIGN AND is_mutation(node):
            target = node.get_child("targets")[0]
            IF target.type == ATTRIBUTE:
                // obj.attr = value (mutating obj)
                obj = target.get_child("value")
                IF obj.type == NAME AND obj.attributes["id"] IN param_names:
                    var_id = VariableID(obj.attributes["id"], get_scope(func_node))
                    mutations.add(var_id)

        ELSE IF node.type == CALL:
            callee = get_callee_name(node)

            // Check for in-place operations
            IF callee IN ["list.append", "list.extend", "dict.update", "set.add"]:
                obj_node = node.get_child("func").get_child("value")
                IF obj_node.type == NAME AND obj_node.attributes["id"] IN param_names:
                    var_id = VariableID(obj_node.attributes["id"], get_scope(func_node))
                    mutations.add(var_id)

    RETURN mutations

// Classify function purity
FUNCTION classify_purity(semantics: FunctionSemantics) -> Purity:
    // Pure function: no side effects, deterministic
    IF semantics.side_effects.is_empty() AND
       semantics.writes_global.is_empty() AND
       semantics.mutates.is_empty() AND
       semantics.io_operations.is_empty():
        RETURN PURE

    // Impure: has side effects
    IF NOT semantics.side_effects.is_empty():
        // Check if side effects are conditional
        all_conditional = true
        FOR EACH effect IN semantics.side_effects:
            IF NOT is_inside_conditional(effect.location):
                all_conditional = false
                BREAK

        IF all_conditional:
            RETURN CONDITIONALLY_PURE
        ELSE:
            RETURN IMPURE

    RETURN IMPURE

// Detect I/O operations
FUNCTION detect_io_operations(func_node: ASTNode) -> List<IOOperation>:
    io_ops = []

    FOR EACH node IN func_node.traverse().filter(type=CALL):
        callee = get_callee_name(node)

        IF callee == "open":
            // File I/O
            args = node.get_children("args")
            file_path = "unknown"
            mode = "r"

            IF args.length >= 1:
                file_path = try_evaluate_constant(args[0])
            IF args.length >= 2:
                mode = try_evaluate_constant(args[1])

            io_type = FILE_READ IF "r" IN mode ELSE FILE_WRITE
            io_ops.append(NEW IOOperation(
                operation_type=io_type,
                resource=file_path,
                mode=mode
            ))

        ELSE IF callee IN ["print", "sys.stdout.write"]:
            io_ops.append(NEW IOOperation(
                operation_type=STDOUT,
                resource="stdout",
                mode="w"
            ))

        ELSE IF callee.starts_with("requests."):
            io_ops.append(NEW IOOperation(
                operation_type=NETWORK,
                resource=extract_url_from_call(node),
                mode=extract_http_method(callee)
            ))

    RETURN io_ops

// Detect concurrency patterns
FUNCTION detect_concurrency_pattern(func_node: ASTNode) -> Optional<ConcurrencyPattern>:
    // Check for async/await
    IF func_node.attributes.get("is_async"):
        await_nodes = func_node.traverse().filter(type=AWAIT)
        RETURN NEW ConcurrencyPattern(
            pattern_type=ASYNC_AWAIT,
            primitives_used=extract_async_primitives(func_node)
        )

    // Check for threading
    threading_imports = find_imports(func_node, "threading")
    IF NOT threading_imports.is_empty():
        sync_primitives = detect_sync_primitives(func_node)
        RETURN NEW ConcurrencyPattern(
            pattern_type=THREADING,
            primitives_used=["threading"],
            synchronization=sync_primitives
        )

    // Check for multiprocessing
    mp_imports = find_imports(func_node, "multiprocessing")
    IF NOT mp_imports.is_empty():
        RETURN NEW ConcurrencyPattern(
            pattern_type=MULTIPROCESSING,
            primitives_used=["multiprocessing"]
        )

    // Check for concurrent.futures
    futures_calls = find_calls_matching(func_node, "concurrent.futures.*")
    IF NOT futures_calls.is_empty():
        RETURN NEW ConcurrencyPattern(
            pattern_type=CONCURRENT_FUTURES,
            primitives_used=extract_executor_types(futures_calls)
        )

    RETURN NULL

// Detect synchronization primitives
FUNCTION detect_sync_primitives(func_node: ASTNode) -> List<SyncPrimitive>:
    primitives = []

    primitive_types = ["Lock", "RLock", "Semaphore", "Event", "Condition", "Barrier"]

    FOR EACH node IN func_node.traverse().filter(type=CALL):
        callee = get_callee_name(node)

        FOR EACH prim_type IN primitive_types:
            IF callee.ends_with(prim_type):
                primitives.append(NEW SyncPrimitive(
                    primitive_type=prim_type,
                    location=node.source_location
                ))

    RETURN primitives

// Propagate purity information through call graph
FUNCTION propagate_purity(annotations: SemanticAnnotations, call_graph: CallGraph):
    changed = true
    max_iterations = 100
    iteration = 0

    WHILE changed AND iteration < max_iterations:
        changed = false
        iteration += 1

        FOR EACH (func_id, node) IN call_graph.nodes:
            semantics = annotations.function_semantics[func_id]

            IF semantics.purity == PURE:
                // Check if any called function is impure
                FOR EACH callee_id IN node.calls_to:
                    IF callee_id IN annotations.function_semantics:
                        callee_semantics = annotations.function_semantics[callee_id]

                        IF callee_semantics.purity != PURE:
                            // Calling impure function makes this impure
                            semantics.purity = IMPURE
                            changed = true
                            BREAK

// Analyze variable semantics (mutability and lifetime)
FUNCTION analyze_variable_semantics(data_flow_graph: DataFlowGraph) -> Map<VariableID, VariableSemantics>:
    var_semantics = NEW Map<VariableID, VariableSemantics>()

    FOR EACH (var_id, df_node) IN data_flow_graph.nodes:
        semantics = NEW VariableSemantics(variable_id=var_id)

        // Determine mutability
        semantics.mutability = determine_mutability(var_id, data_flow_graph)

        // Determine lifetime
        semantics.lifetime = determine_lifetime(var_id, data_flow_graph)

        var_semantics[var_id] = semantics

    RETURN var_semantics

// Determine if a variable is mutable
FUNCTION determine_mutability(var_id: VariableID, graph: DataFlowGraph) -> Mutability:
    df_node = graph.nodes[var_id]

    // Check type - some types are always immutable
    IF df_node.type_info.base_type IN ["int", "float", "str", "tuple", "frozenset"]:
        RETURN IMMUTABLE

    // Find all assignments to this variable
    mutations = graph.edges.filter(edge =>
        edge.to_var == var_id AND edge.edge_type == MUTATION
    )

    IF mutations.is_empty():
        // Never mutated after creation
        RETURN IMMUTABLE

    // Check if mutations are conditional
    all_conditional = true
    FOR EACH mutation IN mutations:
        IF NOT is_inside_conditional(mutation.location):
            all_conditional = false
            BREAK

    IF all_conditional:
        RETURN CONDITIONALLY_MUTABLE
    ELSE:
        RETURN MUTABLE

// Determine variable lifetime
FUNCTION determine_lifetime(var_id: VariableID, graph: DataFlowGraph) -> Lifetime:
    df_node = graph.nodes[var_id]

    // Find first definition
    first_def = df_node.definition_site

    // Find last use by traversing outgoing edges
    last_use = first_def

    visited = NEW Set<VariableID>()
    queue = NEW Queue<VariableID>()
    queue.enqueue(var_id)

    WHILE NOT queue.is_empty():
        current = queue.dequeue()
        IF current IN visited:
            CONTINUE
        visited.add(current)

        // Find all uses of this variable
        outgoing = graph.edges.filter(edge => edge.from_var == current)

        FOR EACH edge IN outgoing:
            use_location = get_edge_location(edge)
            IF use_location.line_end > last_use.line_end:
                last_use = use_location

            queue.enqueue(edge.to_var)

    RETURN NEW Lifetime(
        scope=df_node.scope,
        first_def=first_def,
        last_use=last_use
    )
```

---

## 4. Input/Output Contracts

### 4.1 Agent Input Contract

```pseudocode
STRUCTURE AnalysisAgentInput:
    ast_tree: ASTNode              // Parsed Python AST from Ingest Agent
    source_files: List<SourceFile>  // Original Python source code
    module_metadata: ModuleMetadata // Package structure and metadata
    config: AnalysisConfig          // Analysis configuration

STRUCTURE SourceFile:
    file_path: Path
    content: String
    encoding: String

STRUCTURE ModuleMetadata:
    package_name: String
    package_root: Path
    module_structure: ModuleTree
    entry_points: List<String>

STRUCTURE AnalysisConfig:
    enable_type_inference: Boolean
    enable_contract_discovery: Boolean
    enable_semantic_analysis: Boolean
    docstring_formats: List<DocFormat>
    max_inference_depth: Integer
    enable_runtime_tracing: Boolean  // For dynamic analysis

VALIDATION:
    REQUIRE ast_tree IS NOT NULL
    REQUIRE source_files IS NOT EMPTY
    REQUIRE module_metadata.package_root IS VALID_DIRECTORY
    REQUIRE config.max_inference_depth > 0
```

### 4.2 Agent Output Contract

```pseudocode
STRUCTURE AnalysisAgentOutput:
    api_catalog: APICatalog           // Complete public API surface
    call_graph: CallGraph             // Function call dependencies
    data_flow_graph: DataFlowGraph    // Variable data flow
    dependency_tree: DependencyTree   // Module dependencies
    contracts: ContractDatabase       // Type and behavioral contracts
    semantics: SemanticAnnotations    // Purity, mutability, effects
    analysis_report: AnalysisReport   // Summary and metrics
    errors: List<AnalysisError>
    warnings: List<AnalysisWarning>

STRUCTURE AnalysisReport:
    total_functions: Integer
    total_classes: Integer
    public_api_count: Integer
    private_api_count: Integer
    type_coverage: Float  // % of APIs with type annotations
    contract_coverage: Float  // % of APIs with contracts
    circular_dependencies: Integer
    pure_functions: Integer
    impure_functions: Integer
    analysis_duration_ms: Integer

STRUCTURE AnalysisError:
    error_type: AnalysisErrorType
    message: String
    location: Optional<SourceLocation>
    context: Map<String, String>

ENUM AnalysisErrorType:
    PARSE_ERROR, TYPE_RESOLUTION_ERROR,
    CIRCULAR_DEPENDENCY_ERROR, UNKNOWN_IMPORT_ERROR,
    CONTRACT_INFERENCE_ERROR

STRUCTURE AnalysisWarning:
    warning_type: AnalysisWarningType
    message: String
    location: Optional<SourceLocation>

ENUM AnalysisWarningType:
    MISSING_TYPE_HINT, MISSING_DOCSTRING,
    DYNAMIC_TYPING, UNRESOLVED_IMPORT,
    COMPLEX_CONTROL_FLOW, DEEP_NESTING

GUARANTEES:
    api_catalog.functions CONTAINS ONLY PUBLIC functions
    call_graph IS ACYCLIC OR circular_deps ARE REPORTED
    ALL type_info IN api_catalog ARE RESOLVED OR MARKED AS "Unknown"
    contracts.function_contracts.size() == api_catalog.functions.size()
    semantics.function_semantics.size() == api_catalog.functions.size()
```

### 4.3 Inter-Agent Contracts

```pseudocode
// Contract with Ingest Agent (upstream)
REQUIRES FROM IngestAgent:
    valid_ast: ASTNode WITH all nodes properly typed
    source_map: Map<ASTNode, SourceLocation>
    module_imports: List<Import> WITH resolution metadata

// Contract with Specification Generator (downstream)
PROVIDES TO SpecificationGenerator:
    complete_api_surface: APICatalog
    dependency_information: DependencyTree
    type_contracts: ContractDatabase
    semantic_metadata: SemanticAnnotations

// The Specification Generator expects:
//   - All public APIs identified and typed
//   - Type inference completed with confidence scores
//   - Contract specifications in structured format
//   - Purity and mutability annotations for translation
```

---

## 5. Error Handling Strategy

### 5.1 Error Categories and Recovery

```pseudocode
// Error handling framework
STRUCTURE ErrorHandler:
    errors: List<AnalysisError>
    warnings: List<AnalysisWarning>
    recovery_strategy: RecoveryStrategy

ENUM RecoveryStrategy:
    FAIL_FAST,           // Stop on first error
    BEST_EFFORT,         // Continue with partial results
    STRICT_VALIDATION    // Validate everything, collect all errors

// Main error handling wrapper
FUNCTION safe_analyze(input: AnalysisAgentInput, strategy: RecoveryStrategy) -> Result<AnalysisAgentOutput, Error>:
    handler = NEW ErrorHandler(recovery_strategy=strategy)

    TRY:
        // Phase 1: API Surface Extraction
        api_catalog = TRY extract_api_surface(input.ast_tree, input.source_files)
        CATCH error:
            handler.errors.append(create_error(PARSE_ERROR, error))
            IF strategy == FAIL_FAST:
                RETURN Error(handler.errors)
            api_catalog = NEW APICatalog()  // Empty catalog

        // Phase 2: Dependency Graph Construction
        call_graph = TRY build_call_graph(input.ast_tree, api_catalog)
        CATCH error:
            handler.errors.append(create_error(CIRCULAR_DEPENDENCY_ERROR, error))
            IF strategy == FAIL_FAST:
                RETURN Error(handler.errors)
            call_graph = NEW CallGraph()  // Empty graph

        dependency_tree = TRY build_dependency_tree(input.ast_tree, input.module_metadata.package_root)
        CATCH error:
            handler.errors.append(create_error(UNKNOWN_IMPORT_ERROR, error))
            IF strategy == FAIL_FAST:
                RETURN Error(handler.errors)
            dependency_tree = NEW DependencyTree()

        data_flow_graph = TRY build_data_flow_graph(input.ast_tree)
        CATCH error:
            handler.errors.append(create_error(TYPE_RESOLUTION_ERROR, error))
            IF strategy == FAIL_FAST:
                RETURN Error(handler.errors)
            data_flow_graph = NEW DataFlowGraph()

        // Phase 3: Contract Discovery
        contracts = TRY discover_contracts(api_catalog, input.ast_tree)
        CATCH error:
            handler.errors.append(create_error(CONTRACT_INFERENCE_ERROR, error))
            IF strategy == FAIL_FAST:
                RETURN Error(handler.errors)
            contracts = NEW ContractDatabase()

        // Phase 4: Semantic Analysis
        semantics = TRY analyze_semantics(api_catalog, call_graph, data_flow_graph)
        CATCH error:
            handler.errors.append(create_error(CONTRACT_INFERENCE_ERROR, error))
            IF strategy == FAIL_FAST:
                RETURN Error(handler.errors)
            semantics = NEW SemanticAnnotations()

        // Generate report
        report = generate_report(api_catalog, call_graph, contracts, semantics, handler)

        output = NEW AnalysisAgentOutput(
            api_catalog=api_catalog,
            call_graph=call_graph,
            data_flow_graph=data_flow_graph,
            dependency_tree=dependency_tree,
            contracts=contracts,
            semantics=semantics,
            analysis_report=report,
            errors=handler.errors,
            warnings=handler.warnings
        )

        RETURN Ok(output)

    CATCH fatal_error:
        RETURN Error([create_error(UNKNOWN_ERROR, fatal_error)])
```

### 5.2 Specific Error Handlers

```pseudocode
// Handle type resolution errors
FUNCTION handle_type_resolution_error(node: ASTNode, context: String) -> TypeInfo:
    LOG.warning("Cannot resolve type for {} at {}", context, node.source_location)

    // Try to infer from usage
    usage_type = try_infer_from_usage(node)
    IF usage_type IS NOT NULL:
        RETURN usage_type

    // Fall back to Any/Unknown
    RETURN TypeInfo(base_type="Unknown")

// Handle circular dependency detection
FUNCTION handle_circular_dependency(cycle: List<String>) -> RecoveryAction:
    LOG.error("Circular dependency detected: {}", " -> ".join(cycle))

    // Analyze if cycle can be broken
    weakest_link = find_weakest_dependency_link(cycle)

    IF weakest_link IS NOT NULL:
        LOG.info("Suggestion: Consider breaking cycle at {}", weakest_link)

    // Still include in dependency tree with marker
    RETURN RecoveryAction.MARK_AND_CONTINUE

// Handle missing docstrings
FUNCTION handle_missing_docstring(function: FunctionSignature):
    LOG.warning("Missing docstring for {}", function.qualified_name)

    // Try to generate minimal documentation from signature
    minimal_doc = NEW Docstring(
        raw_text="",
        format=PLAIN,
        summary=f"Function {function.name}",
        description=f"Auto-generated: Function {function.name} with {function.parameters.length} parameters"
    )

    RETURN minimal_doc

// Handle unresolvable imports
FUNCTION handle_unresolved_import(import_stmt: Import) -> ExternalDependency:
    LOG.warning("Cannot resolve import: {}", import_stmt.module_name)

    // Mark as unknown external dependency
    RETURN NEW ExternalDependency(
        package_name=import_stmt.module_name,
        version=NULL,
        import_type=UNKNOWN,
        used_symbols=import_stmt.imported_names.to_set()
    )
```

### 5.3 Validation and Invariants

```pseudocode
// Validate output before returning
FUNCTION validate_output(output: AnalysisAgentOutput) -> List<ValidationError>:
    errors = []

    // Invariant 1: All functions in call graph must be in API catalog
    FOR EACH (func_id, node) IN output.call_graph.nodes:
        IF NOT output.api_catalog.contains_function(func_id):
            errors.append(ValidationError(
                "Function in call graph not in API catalog: " + func_id.qualified_name
            ))

    // Invariant 2: All functions must have contracts
    FOR EACH function IN output.api_catalog.functions:
        IF function.to_id() NOT IN output.contracts.function_contracts:
            errors.append(ValidationError(
                "Missing contract for function: " + function.qualified_name
            ))

    // Invariant 3: All functions must have semantic annotations
    FOR EACH function IN output.api_catalog.functions:
        IF function.to_id() NOT IN output.semantics.function_semantics:
            errors.append(ValidationError(
                "Missing semantics for function: " + function.qualified_name
            ))

    // Invariant 4: Type info must be resolved or marked as Unknown
    FOR EACH function IN output.api_catalog.functions:
        IF function.return_type IS NULL:
            errors.append(ValidationError(
                "NULL return type for function: " + function.qualified_name
            ))

        FOR EACH param IN function.parameters:
            IF param.type_annotation IS NULL:
                errors.append(ValidationError(
                    "NULL type annotation for parameter: " + param.name
                ))

    RETURN errors
```

---

## 6. London School TDD Test Points

### 6.1 Test Hierarchy

```pseudocode
// Acceptance Tests (Outside-In)
TEST_SUITE AnalysisAgentAcceptanceTests:

    TEST "analyzes simple Python script with public API":
        // Arrange
        python_code = """
        def greet(name: str) -> str:
            '''Greet a person by name.'''
            return f"Hello, {name}!"
        """

        ast = MOCK(ASTNode)
        input = create_test_input(ast, [python_code])
        agent = NEW AnalysisAgent()

        // Act
        output = agent.analyze(input)

        // Assert
        EXPECT(output.api_catalog.functions.length).to_equal(1)
        EXPECT(output.api_catalog.functions[0].name).to_equal("greet")
        EXPECT(output.api_catalog.functions[0].parameters[0].name).to_equal("name")
        EXPECT(output.api_catalog.functions[0].return_type.base_type).to_equal("str")
        EXPECT(output.errors).to_be_empty()

    TEST "detects circular dependencies in modules":
        // Arrange
        module_a = create_module("a", imports=["b"])
        module_b = create_module("b", imports=["c"])
        module_c = create_module("c", imports=["a"])

        ast = build_ast([module_a, module_b, module_c])
        input = create_test_input(ast)
        agent = NEW AnalysisAgent()

        // Act
        output = agent.analyze(input)

        // Assert
        EXPECT(output.dependency_tree.circular_deps.length).to_equal(1)
        EXPECT(output.dependency_tree.circular_deps[0].cycle).to_contain(["a", "b", "c"])

    TEST "extracts contracts from NumPy-style docstrings":
        // Arrange
        python_code = """
        def divide(a: float, b: float) -> float:
            '''
            Divide two numbers.

            Parameters
            ----------
            a : float
                The dividend
            b : float
                The divisor (must be non-zero)

            Returns
            -------
            float
                The quotient

            Raises
            ------
            ValueError
                If b is zero
            '''
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        """

        ast = parse_python(python_code)
        input = create_test_input(ast, [python_code])
        agent = NEW AnalysisAgent()

        // Act
        output = agent.analyze(input)

        // Assert
        func_id = FunctionID("divide", "module")
        contract = output.contracts.function_contracts[func_id]

        EXPECT(contract).to_not_be_null()
        EXPECT(contract.input_contract.parameters.length).to_equal(2)
        EXPECT(contract.exception_contract.exception_types).to_contain("ValueError")
        EXPECT(contract.preconditions).to_not_be_empty()

// Integration Tests (Component Interaction)
TEST_SUITE AnalysisAgentIntegrationTests:

    TEST "API extraction integrates with type inference":
        // Arrange
        api_extractor = MOCK(APIExtractor)
        type_inferrer = MOCK(TypeInferrer)

        function_sig = NEW FunctionSignature(name="foo", return_type=NULL)
        WHEN(api_extractor.extract()).THEN_RETURN([function_sig])

        inferred_type = TypeInfo(base_type="int")
        WHEN(type_inferrer.infer(function_sig)).THEN_RETURN(inferred_type)

        agent = NEW AnalysisAgent(api_extractor, type_inferrer)

        // Act
        output = agent.analyze(create_test_input())

        // Assert
        VERIFY(api_extractor.extract).was_called()
        VERIFY(type_inferrer.infer).was_called_with(function_sig)
        EXPECT(output.api_catalog.functions[0].return_type).to_equal(inferred_type)

    TEST "call graph construction uses API catalog":
        // Arrange
        api_catalog = create_api_catalog_with_functions(["foo", "bar", "baz"])
        call_graph_builder = NEW CallGraphBuilder(api_catalog)

        ast = create_ast_with_calls(
            function="foo",
            calls=["bar", "baz"]
        )

        // Act
        call_graph = call_graph_builder.build(ast)

        // Assert
        foo_id = FunctionID("foo", "module")
        EXPECT(call_graph.nodes[foo_id].calls_to).to_contain([
            FunctionID("bar", "module"),
            FunctionID("baz", "module")
        ])

// Unit Tests (Individual Components)
TEST_SUITE APIExtractionUnitTests:

    TEST "extract_function_signature extracts name and parameters":
        // Arrange
        node = create_function_ast_node(
            name="calculate",
            params=[("x", "int"), ("y", "int")]
        )

        // Act
        signature = extract_function_signature(node)

        // Assert
        EXPECT(signature.name).to_equal("calculate")
        EXPECT(signature.parameters.length).to_equal(2)
        EXPECT(signature.parameters[0].name).to_equal("x")
        EXPECT(signature.parameters[0].type_annotation.base_type).to_equal("int")

    TEST "determine_visibility returns PUBLIC for normal names":
        EXPECT(determine_visibility("foo")).to_equal(PUBLIC)

    TEST "determine_visibility returns PROTECTED for underscore prefix":
        EXPECT(determine_visibility("_foo")).to_equal(PROTECTED)

    TEST "determine_visibility returns PRIVATE for double underscore":
        EXPECT(determine_visibility("__foo")).to_equal(PRIVATE)

    TEST "infer_return_type infers from literal":
        // Arrange
        return_node = create_return_node(value=42)

        // Act
        type_info = infer_expression_type(return_node.get_child("value"))

        // Assert
        EXPECT(type_info.base_type).to_equal("int")

TEST_SUITE CallGraphUnitTests:

    TEST "build_call_graph creates nodes for all functions":
        // Arrange
        api_catalog = create_api_catalog_with_functions(["foo", "bar"])
        ast = create_empty_ast()

        // Act
        graph = build_call_graph(ast, api_catalog)

        // Assert
        EXPECT(graph.nodes.size()).to_equal(2)
        EXPECT(graph.nodes).to_have_key(FunctionID("foo", "module"))
        EXPECT(graph.nodes).to_have_key(FunctionID("foo", "module"))

    TEST "extract_function_calls finds all CALL nodes":
        // Arrange
        func_node = create_ast_with_calls(["foo", "bar", "foo"])  // duplicate call

        // Act
        calls = extract_function_calls(func_node)

        // Assert
        EXPECT(calls.length).to_equal(3)

    TEST "detect_cycles identifies circular dependencies":
        // Arrange
        root = create_module_node("a", deps=[
            create_module_node("b", deps=[
                create_module_node("c", deps=[
                    create_module_node("a", deps=[])  // cycle back to a
                ])
            ])
        ])

        // Act
        cycles = detect_cycles(root)

        // Assert
        EXPECT(cycles.length).to_equal(1)
        EXPECT(cycles[0].cycle).to_equal(["a", "b", "c", "a"])

TEST_SUITE ContractDiscoveryUnitTests:

    TEST "parse_docstring detects NumPy format":
        // Arrange
        text = """
        Summary line.

        Parameters
        ----------
        x : int
            Description
        """

        // Act
        doc = parse_docstring(text)

        // Assert
        EXPECT(doc.format).to_equal(NUMPY)
        EXPECT(doc.parameters.length).to_equal(1)

    TEST "extract_preconditions finds assert statements":
        // Arrange
        func_node = create_function_with_body([
            create_assert_node("x > 0"),
            create_assert_node("y < 100")
        ])

        // Act
        preconditions = extract_preconditions(func_node, NULL)

        // Assert
        EXPECT(preconditions.length).to_equal(2)
        EXPECT(preconditions[0].expression).to_equal("x > 0")

    TEST "parse_constraints_from_doc recognizes positive constraint":
        // Arrange
        description = "The value must be positive"

        // Act
        constraints = parse_constraints_from_doc(description)

        // Assert
        EXPECT(constraints.length).to_equal(1)
        EXPECT(constraints[0].constraint_type).to_equal(POSITIVE)

TEST_SUITE SemanticAnalysisUnitTests:

    TEST "detect_side_effects identifies print statement":
        // Arrange
        func_node = create_function_with_call("print", ["hello"])

        // Act
        side_effects = detect_side_effects(func_node)

        // Assert
        EXPECT(side_effects.length).to_equal(1)
        EXPECT(side_effects[0].effect_type).to_equal(IO_WRITE)

    TEST "classify_purity returns PURE for no side effects":
        // Arrange
        semantics = NEW FunctionSemantics(
            side_effects=[],
            writes_global=new Set(),
            mutates=new Set(),
            io_operations=[]
        )

        // Act
        purity = classify_purity(semantics)

        // Assert
        EXPECT(purity).to_equal(PURE)

    TEST "classify_purity returns IMPURE for I/O operations":
        // Arrange
        semantics = NEW FunctionSemantics(
            side_effects=[NEW SideEffect(IO_WRITE, NULL)],
            writes_global=new Set(),
            mutates=new Set(),
            io_operations=[NEW IOOperation(FILE_WRITE, "out.txt", "w")]
        )

        // Act
        purity = classify_purity(semantics)

        // Assert
        EXPECT(purity).to_equal(IMPURE)

    TEST "detect_mutations identifies list.append":
        // Arrange
        func_node = create_function_with_params(["items"])
        add_method_call(func_node, "items", "append", [42])
        data_flow_graph = build_data_flow_graph(func_node)

        // Act
        mutations = detect_mutations(func_node, data_flow_graph)

        // Assert
        EXPECT(mutations.size()).to_equal(1)
        EXPECT(mutations).to_contain(VariableID("items", func_scope))

// Mock Definitions
TEST_SUITE MockDefinitions:

    MOCK APIExtractor:
        METHOD extract(ast: ASTNode) -> List<FunctionSignature>

    MOCK TypeInferrer:
        METHOD infer(function: FunctionSignature) -> TypeInfo

    MOCK DependencyResolver:
        METHOD resolve(import: Import) -> Optional<ModuleNode>

    MOCK ContractParser:
        METHOD parse(docstring: Docstring) -> Contract
```

### 6.2 Test Doubles Strategy

```pseudocode
// Stubs - Provide minimal responses
STUB TypeStub IMPLEMENTS TypeInferrer:
    METHOD infer(function: FunctionSignature) -> TypeInfo:
        RETURN TypeInfo(base_type="Unknown")

// Spies - Record interactions
SPY CallGraphSpy IMPLEMENTS CallGraphBuilder:
    calls_recorded: List<CallRecord>

    METHOD build(ast: ASTNode) -> CallGraph:
        calls_recorded.append(NEW CallRecord(ast))
        RETURN real_implementation.build(ast)

// Mocks - Verify behavior
MOCK ContractDiscoveryMock:
    expected_calls: List<ExpectedCall>

    METHOD discover(api: APICatalog) -> ContractDatabase:
        VERIFY(expected_calls.contains(api))
        RETURN predefined_response

// Fakes - Working implementations
FAKE InMemoryASTFake IMPLEMENTS ASTProvider:
    ast_store: Map<String, ASTNode>

    METHOD get_ast(module: String) -> ASTNode:
        RETURN ast_store.get(module, default=empty_ast())
```

### 6.3 Test Coverage Requirements

```pseudocode
COVERAGE_REQUIREMENTS:
    line_coverage: >= 85%
    branch_coverage: >= 80%

    critical_paths:
        - extract_api_surface: 100%
        - build_call_graph: 100%
        - discover_contracts: 95%
        - analyze_semantics: 95%
        - error_handling: 100%

    edge_cases:
        - empty_input: MUST_TEST
        - circular_dependencies: MUST_TEST
        - missing_type_hints: MUST_TEST
        - malformed_docstrings: MUST_TEST
        - deeply_nested_code: MUST_TEST
        - dynamic_imports: MUST_TEST
```

---

## 7. Performance Considerations

```pseudocode
// Optimization strategies for large codebases
FUNCTION optimize_for_scale(config: AnalysisConfig) -> AnalysisConfig:
    IF config.codebase_size > 10000:  // LOC
        // Enable parallel processing
        config.parallel_analysis = true
        config.worker_threads = CPU_COUNT

        // Limit inference depth
        config.max_inference_depth = 3

        // Cache frequently accessed data
        config.enable_caching = true
        config.cache_size_mb = 512

    RETURN config

// Incremental analysis for large graphs
FUNCTION incremental_call_graph_update(old_graph: CallGraph,
                                       changed_functions: Set<FunctionID>,
                                       api_catalog: APICatalog) -> CallGraph:
    // Only re-analyze affected functions
    affected = compute_transitive_callers(old_graph, changed_functions)

    FOR EACH func_id IN affected:
        // Remove old edges
        remove_edges_from(old_graph, func_id)

        // Re-extract calls
        func_node = find_function_ast_node(func_id)
        new_calls = extract_function_calls(func_node)

        // Add new edges
        FOR EACH call IN new_calls:
            add_edge_to(old_graph, func_id, call)

    RETURN old_graph
```

---

## Document Status

**SPARC Phase 2 (Pseudocode) for Analysis Agent: COMPLETE**

This pseudocode specification provides:
- Complete data structure definitions for AST, API catalog, graphs, contracts, and semantics
- Detailed algorithms for extraction, graph construction, contract discovery, and semantic analysis
- Clear input/output contracts with validation guarantees
- Comprehensive error handling strategy with recovery mechanisms
- London School TDD test points covering acceptance, integration, and unit levels

**Next Steps:**
- Phase 3 (Architecture): Detailed component design and interface specifications
- Phase 4 (Refinement): Iterative improvement based on implementation feedback
- Phase 5 (Completion): Final implementation in Rust

---

**END OF PSEUDOCODE DOCUMENT**
