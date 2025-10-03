# Implementation Roadmap: Python → Rust → WASM Platform

## Executive Summary

This roadmap outlines a phased approach to building the Python→Rust→WASM platform with NVIDIA integrations. The strategy emphasizes **incremental value delivery**, **risk mitigation through early validation**, and **TDD outside-in methodology** (working features first, then optimization).

**Key Principles:**
- Each phase delivers demonstrable, end-to-end value
- Critical technical risks are addressed early
- Foundation is built incrementally, not all upfront
- MVP focuses on Script Mode with real-world validation
- GPU acceleration added after core pipeline proves viable

---

## Phase 0: Foundation and Infrastructure
**Duration:** 2-3 weeks | **Team Size:** 2-3 engineers

### Objectives
Establish minimal scaffolding needed to begin TDD development. Focus on toolchain setup, not exhaustive infrastructure.

### Deliverables

#### 0.1 Development Environment Setup
- **Rust/WASM Toolchain**
  - Install Rust stable + nightly (for experimental WASM features)
  - Configure `wasm32-wasi` and `wasm32-unknown-unknown` targets
  - Setup `wasm-pack`, `wasmtime`, and `wasmer` runtimes
  - Validate basic Rust→WASM compilation with hello-world

- **Python Analysis Tools**
  - Install AST analysis libraries (`ast`, `astroid`, `rope`)
  - Setup type inference tools (`mypy`, `pyre`, `pyright`)
  - Configure dependency analysis (`pipdeptree`, `modulegraph`)

- **Testing Framework**
  - Rust: `cargo test`, `proptest` for property-based testing
  - Python: `pytest` with golden test framework
  - WASM: `wasm-bindgen-test` and runtime validation harness

#### 0.2 Minimal Agent Framework
- **Simple Task Queue**
  - Basic job definition (Python file → Rust output)
  - Single-threaded executor (parallelism in Phase 2)
  - File-based state persistence (JSON logs)

- **Agent Base Classes**
  - `Agent` trait: `execute(input) -> Result<output>`
  - Error handling with structured logging
  - Context passing (shared state between agents)

- **Communication Protocol**
  - File-based message passing (simplest viable approach)
  - Shared workspace directory structure
  - Agent output validation schemas

#### 0.3 Repository Structure
```
portalis/
├── agents/           # Agent implementations
│   ├── analyzer/     # Python code analysis
│   ├── translator/   # Python→Rust translation
│   ├── builder/      # Rust compilation
│   └── validator/    # Test execution
├── core/             # Shared libraries
│   ├── types/        # Common data structures
│   ├── utils/        # Helpers
│   └── runtime/      # WASM runtime integration
├── examples/         # Test cases (start small)
│   ├── hello_world.py
│   ├── fibonacci.py
│   └── simple_math.py
├── tests/            # Integration tests
└── docs/             # Architecture decisions
```

### Success Criteria
- [ ] Can compile and run a trivial Rust program as WASM
- [ ] Basic agent can read a Python file and output JSON analysis
- [ ] End-to-end test harness executes a dummy pipeline
- [ ] CI/CD pipeline runs tests on every commit

### Critical Dependencies
- None (bootstrap phase)

### Risk Mitigation
- **Risk:** Over-engineering infrastructure delays actual work
  - **Mitigation:** Timebox to 3 weeks max; use simplest tools (files, not databases)

---

## Phase 1: MVP - Script Mode End-to-End
**Duration:** 6-8 weeks | **Team Size:** 3-4 engineers

### Objectives
Deliver a **working, demonstrable** Script Mode pipeline that converts simple Python scripts to validated WASM modules. Proves core feasibility without GPU acceleration or complex dependencies.

### Scope: Supported Python Subset (MVP)
**Target:** Single-file scripts with these features:
- Basic types: `int`, `float`, `str`, `bool`, `list`, `dict`
- Control flow: `if/elif/else`, `for`, `while`, `break`, `continue`
- Functions: Pure functions with explicit signatures (no decorators)
- Arithmetic and string operations
- Simple error handling: `try/except/finally` (basic cases)

**Explicitly Out of Scope for MVP:**
- Classes and OOP (Phase 2)
- External dependencies (Phase 2)
- Advanced types (`dataclasses`, `TypedDict`, generics)
- Async/await
- File I/O (added in Phase 2)
- Numpy/numerical libraries (Phase 3)

### Deliverables

#### 1.1 Analyzer Agent (Weeks 1-2)
**Input:** Python script path
**Output:** JSON analysis document

**Capabilities:**
- Parse Python AST using `ast` module
- Extract function signatures with inferred types
- Build control flow graph (CFG) for each function
- Identify used stdlib features
- Detect unsupported constructs (emit warnings)

**Validation:**
- Test suite with 20+ scripts (simple → moderate complexity)
- Golden JSON outputs for regression testing

**Example Analysis Output:**
```json
{
  "functions": [
    {
      "name": "fibonacci",
      "params": [{"name": "n", "type": "int"}],
      "return_type": "int",
      "calls": [],
      "uses_stdlib": []
    }
  ],
  "unsupported": [],
  "complexity_score": 12
}
```

#### 1.2 Spec Generator Agent (Weeks 2-3)
**Input:** Analysis JSON
**Output:** Rust interface specification

**Capabilities:**
- Map Python types → Rust types (`int` → `i64`, `str` → `String`)
- Generate Rust function signatures
- Create struct definitions for data containers
- Define error types for Python exceptions
- Produce WASM ABI specification (extern "C" interfaces)

**Validation:**
- Generated Rust code compiles (even if unimplemented)
- Type mappings cover MVP Python subset
- WASM exports match expected function names

**Example Spec Output:**
```rust
// Generated specification
#[wasm_bindgen]
pub fn fibonacci(n: i64) -> i64 {
    unimplemented!() // Filled by Translator Agent
}
```

#### 1.3 Translator Agent (Weeks 3-5)
**Input:** Python AST + Rust spec
**Output:** Implemented Rust code

**Core Translation Rules:**
- **Functions:** Direct mapping with type conversions
  ```python
  def add(x: int, y: int) -> int:
      return x + y
  ```
  →
  ```rust
  pub fn add(x: i64, y: i64) -> i64 {
      x + y
  }
  ```

- **Control Flow:** Pattern-match translation
  - Python `if/else` → Rust `if/else`
  - Python `for x in range(n)` → Rust `for x in 0..n`
  - Python `while` → Rust `while`

- **Collections:**
  - Python `list` → Rust `Vec<T>`
  - Python `dict` → Rust `HashMap<K, V>` (simple cases)

- **Error Handling:**
  - Python `raise Exception` → Rust `panic!` (MVP; refined in Phase 2)

**Validation:**
- Unit tests: Each Python construct → Rust equivalent
- Translation consistency: Same input → same output
- Compile success rate: 95%+ on MVP test suite

#### 1.4 Builder Agent (Weeks 4-5)
**Input:** Rust code
**Output:** WASM module

**Capabilities:**
- Generate `Cargo.toml` with dependencies
- Invoke `wasm-pack build --target web`
- Run `cargo test` for Rust unit tests
- Produce `.wasm` artifact with metadata

**Validation:**
- WASM modules load in `wasmtime` runtime
- Exports are callable from JavaScript/Rust host
- Binary size benchmarks (track bloat)

#### 1.5 Validator Agent (Weeks 5-6)
**Input:** Original Python script + WASM module
**Output:** Test report (pass/fail + metrics)

**Testing Strategy:**
- **Golden Tests:** Capture Python execution results, replay in WASM
  ```python
  # test_fibonacci.py
  assert fibonacci(0) == 0
  assert fibonacci(10) == 55
  ```
  → Execute both Python and WASM, compare outputs

- **Property Tests:** Random inputs, check invariants
  ```rust
  proptest! {
      #[test]
      fn test_add_commutative(x in 0..100i64, y in 0..100i64) {
          assert_eq!(add(x, y), add(y, x));
      }
  }
  ```

- **Performance Benchmarks:**
  - Execution time (Python vs WASM)
  - Memory usage
  - WASM binary size

**Validation:**
- Report generation with pass/fail status
- Performance regression detection

#### 1.6 End-to-End Integration (Weeks 6-8)
**Pipeline Orchestration:**
```
Python Script → Analyzer → Spec Generator → Translator → Builder → Validator → Report
```

**CLI Tool:**
```bash
portalis convert script.py --output ./out --validate
```

**Integration Tests:**
- 10+ real-world Python scripts (fibonacci, sorting, math functions)
- Automated regression suite
- Performance benchmarks tracked over time

### Success Criteria
- [ ] Convert 10+ Python scripts to WASM with 100% test pass rate
- [ ] WASM execution produces identical results to Python (golden tests)
- [ ] End-to-end pipeline runs in <5 minutes for typical scripts
- [ ] Generated Rust code is readable and idiomatic (manual review)
- [ ] Public demo: Live script conversion with before/after comparison

### Critical Dependencies
- Phase 0 completion (toolchain + basic framework)

### Phase Gate Criteria (GO/NO-GO Decision)
**PASS Requirements:**
- At least 8/10 test scripts convert successfully
- Zero correctness failures (behavioral parity with Python)
- WASM modules run in standard runtimes (wasmtime, wasmer)
- Team confidence in architecture for scaling to Library Mode

**If FAIL:**
- Reassess translation strategy (may need IR layer)
- Adjust Phase 2 scope or extend Phase 1 timeline

### Risk Mitigation
- **Risk:** Python semantics too complex to translate directly
  - **Mitigation:** MVP subset explicitly excludes hard features; validate early
- **Risk:** WASM performance worse than Python
  - **Mitigation:** Focus on correctness first; optimize in Phase 3

---

## Phase 2: Library Mode and Advanced Features
**Duration:** 8-10 weeks | **Team Size:** 4-5 engineers

### Objectives
Extend pipeline to handle multi-file Python libraries with dependencies, classes, and realistic codebases. Introduce incremental translation (partial library support).

### Scope Additions
**New Python Features:**
- **Classes and OOP:** Inheritance, methods, properties
- **Modules:** Multi-file imports, package structure
- **File I/O:** Read/write with WASI filesystem support
- **External Dependencies:** Pure-Python stdlib (e.g., `json`, `re`, `datetime`)
- **Advanced Types:** `dataclasses`, `NamedTuple`, `Optional`, `Union`

**Still Out of Scope:**
- C-extension dependencies (numpy, pandas)
- Async/await
- Meta-programming (decorators, metaclasses) — limited support only

### Deliverables

#### 2.1 Multi-File Analyzer (Weeks 1-2)
**Enhancements:**
- Traverse Python package directory structure
- Resolve imports and build dependency graph
- Detect external vs internal dependencies
- Generate module-level analysis (per-file + cross-file)

**Output:**
```json
{
  "modules": [
    {"name": "utils.py", "functions": [...], "classes": [...]},
    {"name": "main.py", "imports": ["utils"], "functions": [...]}
  ],
  "dependency_graph": {
    "main": ["utils"],
    "utils": []
  },
  "external_deps": ["json", "re"]
}
```

#### 2.2 Rust Workspace Generator (Weeks 2-3)
**Map Python package → Rust workspace:**
```
my_library/
├── utils.py
└── main.py
```
→
```
my_library_rs/
├── Cargo.toml (workspace)
├── utils/
│   ├── Cargo.toml
│   └── src/lib.rs
└── main/
    ├── Cargo.toml
    └── src/lib.rs
```

**Features:**
- Generate workspace `Cargo.toml` with member crates
- Configure inter-crate dependencies
- Setup WASM target per module

#### 2.3 Class Translation (Weeks 3-5)
**Python Classes → Rust Structs + Impl Blocks**

**Example:**
```python
class Calculator:
    def __init__(self, name: str):
        self.name = name

    def add(self, x: int, y: int) -> int:
        return x + y
```
→
```rust
pub struct Calculator {
    name: String,
}

impl Calculator {
    pub fn new(name: String) -> Self {
        Calculator { name }
    }

    pub fn add(&self, x: i64, y: i64) -> i64 {
        x + y
    }
}
```

**Challenges:**
- Inheritance → Trait composition
- Magic methods (`__str__`, `__eq__`) → Rust trait implementations (`Display`, `PartialEq`)
- Polymorphism → Enum dispatch or trait objects

**Validation:**
- Class-based test suite (20+ classes with varying complexity)
- OOP pattern tests (inheritance, composition)

#### 2.4 Dependency Handling (Weeks 4-6)
**Strategy: Gradual Dependency Support**

**Tier 1 (Phase 2):** Pure-Python stdlib
- `json` → `serde_json`
- `re` → `regex`
- `datetime` → `chrono`
- Manual mapping table: Python module → Rust crate

**Tier 2 (Phase 3):** Numerical libraries
- `numpy` → Custom WASM-compatible implementations or bindings

**Tier 3 (Future):** C-extensions
- Require Rust rewrites or FFI (out of scope)

**Implementation:**
- Dependency detection in Analyzer
- Cargo.toml generation with mapped dependencies
- Automatic import translation (e.g., `import json` → `use serde_json`)

#### 2.5 WASI Integration (Weeks 5-7)
**Enable File I/O and Environment Access**

**Capabilities:**
- File read/write via WASI `wasi::filesystem`
- Environment variables via `wasi::env`
- Command-line arguments
- Sandboxed filesystem (preopened directories)

**Example:**
```python
with open("data.txt", "r") as f:
    content = f.read()
```
→
```rust
use std::fs;
let content = fs::read_to_string("data.txt").unwrap();
```

**Testing:**
- WASI runtime tests with real file operations
- Sandboxing validation (security)

#### 2.6 Incremental Translation Mode (Weeks 6-8)
**Support Partial Library Conversion**

**Use Case:** Large library with 100+ files
- Translate priority modules first
- Generate "stub" implementations for untranslated modules
- Track coverage metrics: "45% of library translated"

**Features:**
- `--modules` flag to select specific files
- Parity report: Translated vs. untranslated APIs
- Stub generator: Compiles but returns errors for unimplemented functions

**Example CLI:**
```bash
portalis convert my_lib/ --modules utils,core --partial
```

#### 2.7 Enhanced Validator (Weeks 7-8)
**Library-Scale Testing:**
- Import translated modules into Python test suite (via WASM-Python bridge)
- Cross-language API parity checks
- Performance comparison: Python vs. WASM for realistic workflows

**Parity Report:**
```
Module: utils
  Functions: 25/25 (100%)
  Classes: 10/10 (100%)
  Tests Passing: 142/150 (94.7%)
  Performance: 2.3x faster than Python
```

#### 2.8 Integration (Weeks 8-10)
**Library Mode Pipeline:**
```
Python Library → Multi-File Analyzer → Workspace Generator → Translator (per module)
→ Builder (workspace) → Validator (library tests) → Parity Report
```

**CLI:**
```bash
portalis convert my_library/ --mode library --output my_library_rs
```

### Success Criteria
- [ ] Convert at least 1 real-world Python library (e.g., subset of `requests` or custom internal library)
- [ ] Achieve 80%+ function coverage on target library
- [ ] Pass 90%+ of original Python tests
- [ ] Generate readable, maintainable Rust code (manual review)
- [ ] Public demo: Library conversion with parity report

### Critical Dependencies
- Phase 1 completion (Script Mode proven)
- Rust workspace tooling understanding

### Phase Gate Criteria
**PASS:**
- Successfully translate a library with 50+ functions/10+ classes
- <10% test failures (acceptable for MVP library support)
- Incremental mode works (can translate subsets)

**If FAIL:**
- Re-evaluate class translation strategy
- Consider limiting Library Mode scope

### Risk Mitigation
- **Risk:** Class inheritance too complex
  - **Mitigation:** Start with simple classes, add inheritance incrementally
- **Risk:** Dependency mapping incomplete
  - **Mitigation:** Focus on top 10 most-used stdlib modules first

---

## Phase 3: NVIDIA Integration and Optimization
**Duration:** 6-8 weeks | **Team Size:** 3-4 engineers (with GPU expertise)

### Objectives
Integrate NVIDIA stack to accelerate pipeline and demonstrate enterprise-scale performance. Add GPU-powered features: embeddings for code similarity, LLM-based translation refinement, and batch processing.

### NVIDIA Components

#### 3.1 NeMo Integration (Weeks 1-3)
**Use Case: LLM-Assisted Translation**

**Implementation:**
- Deploy NeMo LLM (e.g., CodeLlama, StarCoder) on DGX Cloud
- Fine-tune on Python→Rust translation pairs
- Use LLM for:
  - **Fallback Translation:** Complex Python idioms that rule-based translator struggles with
  - **Code Suggestion:** Offer multiple Rust implementations, rank by quality
  - **Comment Translation:** Preserve docstrings and inline comments

**Pipeline Integration:**
```
Translator Agent → [Rule-Based Translation] → [LLM Refinement (optional)] → Rust Code
```

**Validation:**
- A/B test: Rule-based vs. LLM-assisted translation quality
- Measure compilation success rate improvement
- Human evaluation of code readability

**Deliverable:**
- NeMo endpoint integrated into Translator Agent
- Configuration flag: `--use-llm` to enable LLM assistance

#### 3.2 CUDA-Accelerated Analysis (Weeks 2-4)
**Use Cases:**
- **Embedding Generation:** Vectorize Python functions for similarity search
- **Batch AST Parsing:** Parallel parsing of 1000+ files
- **Test Case Prioritization:** Rank tests by likelihood of catching regressions

**Implementation:**
- Use cuDF for data processing (dependency graphs, test coverage data)
- Implement embedding model (CodeBERT) with TensorRT acceleration
- Batch API: Process entire library in single GPU pass

**Performance Target:**
- Analyze 1000-file library in <2 minutes (vs. 30+ minutes CPU-only)

**Deliverable:**
- CUDA-enabled Analyzer Agent variant
- Benchmarks showing speedup

#### 3.3 Triton Inference Server (Weeks 3-5)
**Deployment: Translation as a Service**

**Architecture:**
```
Client → Triton Endpoint → NeMo (Translation LLM) → Rust Code
                        ↘ CUDA (Embedding/Analysis) ↗
```

**Features:**
- REST/gRPC API for translation requests
- Batch processing: Submit 100 scripts, get results asynchronously
- Model versioning: A/B test different translation models
- Metrics: Latency, throughput, GPU utilization

**Example API:**
```bash
curl -X POST https://triton.example.com/v2/translate \
  -d '{"python_code": "def add(x, y): return x + y"}'
```

**Deliverable:**
- Triton deployment scripts (Docker + Kubernetes)
- Client SDK (Python) for interacting with service
- Load testing results (requests/sec)

#### 3.4 NIM Packaging (Weeks 4-6)
**Microservice Containers**

**Package WASM Modules as NIM Services:**
- Input: WASM module + metadata
- Output: NIM container with embedded WASM runtime

**Features:**
- Portable deployment (cloud, edge, on-prem)
- Health checks and monitoring
- Auto-scaling based on request load

**Example:**
```bash
portalis package my_library.wasm --nim --output my_library_nim/
docker run -p 8000:8000 my_library_nim
```

**Deliverable:**
- NIM packaging agent
- Reference deployment (GKE or AWS EKS)
- Performance benchmarks vs. native Python service

#### 3.5 Performance Optimization (Weeks 5-7)
**Rust Code Optimization:**
- Profile generated WASM (identify bottlenecks)
- Apply Rust optimizations:
  - Replace `Vec` with arrays where possible
  - Use `&str` instead of `String` for read-only data
  - Enable LLVM optimizations (`opt-level = 3`, LTO)

**WASM Size Reduction:**
- Strip debug symbols
- Use `wasm-opt` (Binaryen) for aggressive minification
- Tree-shaking unused code

**Target:**
- 2-5x performance improvement over naive translation
- 30-50% reduction in WASM binary size

**Deliverable:**
- Optimization guide (automated + manual)
- Before/after benchmarks

#### 3.6 Omniverse Integration (Weeks 6-8)
**Demonstrate Portability**

**Use Case:** Run WASM modules inside Omniverse simulation
- Example: Physics calculation module (Python → Rust → WASM)
- Load WASM into Omniverse Kit extension
- Execute in real-time simulation loop

**Deliverable:**
- Omniverse demo scene with WASM module
- Tutorial: "Deploying WASM to Omniverse"
- Video demo for marketing

### Success Criteria
- [ ] LLM-assisted translation improves success rate by 10%+
- [ ] CUDA acceleration provides 10x+ speedup on large libraries
- [ ] Triton service handles 100+ req/sec with <500ms latency
- [ ] NIM containers deploy successfully on major cloud platforms
- [ ] Omniverse demo runs smoothly in public presentation

### Critical Dependencies
- Phase 2 completion (Library Mode working)
- Access to DGX Cloud or equivalent GPU resources
- NVIDIA partnership/licensing agreements

### Phase Gate Criteria
**PASS:**
- At least 2 NVIDIA technologies integrated and demonstrable
- Performance improvements measurable and significant
- Enterprise deployment path validated

**If FAIL:**
- Descope Omniverse or NIM (focus on Triton + NeMo only)

### Risk Mitigation
- **Risk:** GPU resources unavailable or expensive
  - **Mitigation:** Start with CPU baselines, add GPU as optional enhancement
- **Risk:** LLM translation quality poor
  - **Mitigation:** Keep rule-based translator as primary, LLM as fallback

---

## Phase 4: Enterprise Packaging and Deployment
**Duration:** 4-6 weeks | **Team Size:** 2-3 engineers

### Objectives
Productionize the platform for enterprise adoption: documentation, CI/CD, monitoring, and customer onboarding tools.

### Deliverables

#### 4.1 Documentation (Weeks 1-2)
**User Documentation:**
- Getting Started Guide
- CLI reference (`portalis --help` docs)
- Python compatibility matrix (supported features)
- Troubleshooting guide (common errors)

**Developer Documentation:**
- Architecture overview (agent design)
- Contribution guide (how to add new Python features)
- API reference (if offering SaaS)

**Enterprise Documentation:**
- Deployment guide (Kubernetes, Docker Compose)
- Security considerations (WASM sandboxing)
- Compliance (SOC2, GDPR considerations)

#### 4.2 CI/CD Pipelines (Weeks 2-3)
**Automated Testing:**
- PR validation: Run full test suite on every commit
- Nightly builds: Test against latest Rust/WASM toolchains
- Performance regression detection

**Release Automation:**
- Semantic versioning
- Automated changelogs
- Docker image builds for Triton/NIM

#### 4.3 Monitoring and Observability (Weeks 3-4)
**Metrics:**
- Translation success rate (per Python feature)
- Pipeline execution time (per phase)
- WASM performance vs. Python (latency, throughput)
- Error rates and failure modes

**Tools:**
- Prometheus + Grafana dashboards
- Distributed tracing (OpenTelemetry) for agent pipeline
- Alerting for critical failures

#### 4.4 Customer Onboarding Tools (Weeks 4-5)
**Assessment Tool:**
```bash
portalis assess my_library/ --report compatibility.html
```
- Analyzes Python codebase
- Generates compatibility report (% translatable)
- Identifies blockers (unsupported features)

**Migration Planner:**
- Recommends translation strategy (full vs. incremental)
- Estimates effort (LOC, complexity)
- Prioritizes modules for translation

#### 4.5 SaaS Platform (Optional, Week 5-6)
**If offering as a service:**
- Web UI for uploading Python code
- Dashboard for monitoring translation jobs
- API for programmatic access
- Billing and usage tracking

#### 4.6 Enterprise Case Studies (Week 6)
**Partner with Early Customers:**
- Translate a real enterprise library
- Document results (performance, cost savings)
- Create case study for marketing

### Success Criteria
- [ ] Complete documentation published
- [ ] CI/CD pipeline running reliably
- [ ] Monitoring dashboards deployed
- [ ] At least 1 enterprise case study completed
- [ ] Platform ready for beta customer onboarding

### Critical Dependencies
- Phase 3 completion (core features + NVIDIA integration)

---

## Critical Dependencies and Sequencing

### Dependency Graph
```
Phase 0 (Foundation)
   ↓
Phase 1 (Script Mode MVP) ← CRITICAL PATH
   ↓
Phase 2 (Library Mode)
   ↓
Phase 3 (NVIDIA Integration) ← Can partially parallelize with Phase 2
   ↓
Phase 4 (Enterprise Packaging)
```

### Critical Path Items
1. **Phase 1.3 (Translator Agent):** Core IP, highest technical risk
2. **Phase 1.5 (Validator Agent):** Proves correctness, essential for trust
3. **Phase 2.3 (Class Translation):** Unlocks realistic library support
4. **Phase 3.1 (NeMo Integration):** Key differentiator vs. competitors

### Parallelization Opportunities
- **Phase 2 + Early Phase 3:** While Library Mode stabilizes, start NeMo fine-tuning
- **Phase 3.6 (Omniverse) + Phase 4:** Omniverse demo can happen alongside documentation

---

## Phase Gate Criteria Summary

| Phase | GO Criteria | NO-GO Trigger | Escalation Action |
|-------|-------------|---------------|-------------------|
| **Phase 0** | Toolchain operational, agent framework running | Can't compile Rust→WASM after 3 weeks | Reassess WASM target viability |
| **Phase 1** | 8/10 scripts convert successfully, zero correctness bugs | <50% success rate | Pivot to interpreter-based approach instead of transpilation |
| **Phase 2** | 1 library with 80%+ coverage, 90%+ test pass rate | Can't handle classes reliably | Limit to functional programming subset only |
| **Phase 3** | 2+ NVIDIA integrations working, measurable performance gains | GPU integration fails or shows no benefit | Ship without GPU (CPU-only mode) |
| **Phase 4** | Documentation complete, 1 case study | No customer willing to pilot | Delay GA launch, extend beta period |

---

## Timeline Estimates and Resource Needs

### Overall Timeline
- **Optimistic:** 20 weeks (5 months)
- **Realistic:** 26 weeks (6.5 months)
- **Pessimistic:** 34 weeks (8.5 months)

### Phase Breakdown (Realistic Scenario)

| Phase | Duration | Start Week | End Week | Key Milestones |
|-------|----------|------------|----------|----------------|
| **Phase 0** | 3 weeks | Week 0 | Week 3 | Toolchain ready, first agent executes |
| **Phase 1** | 8 weeks | Week 3 | Week 11 | Script Mode MVP, public demo |
| **Phase 2** | 10 weeks | Week 11 | Week 21 | Library Mode, parity reports |
| **Phase 3** | 8 weeks | Week 18* | Week 26 | NVIDIA integration, Triton deployment |
| **Phase 4** | 6 weeks | Week 21* | Week 27 | Enterprise-ready, first customer |

*Phase 3 and 4 partially overlap with Phase 2

### Resource Requirements

#### Team Composition
- **Phase 0-1:** 3 engineers (2 backend, 1 DevOps)
- **Phase 2:** 4-5 engineers (3 backend, 1 frontend for tooling, 1 DevOps)
- **Phase 3:** +2 GPU/ML engineers (temp contractors okay)
- **Phase 4:** 2-3 engineers (1 technical writer, 1 DevOps, 1 backend)

#### Skill Requirements
- **Rust expertise:** 2-3 senior engineers (mandatory)
- **WASM/WASI knowledge:** 1 specialist (can train others)
- **Python internals:** 1 expert (AST, type inference)
- **NVIDIA stack:** 1-2 ML engineers (Phase 3 only)
- **DevOps/SRE:** 1 engineer (full-time Phases 0-4)

#### Infrastructure Costs (Monthly Estimates)
- **Phase 0-2:** $2,000-5,000 (CI/CD, dev environments)
- **Phase 3:** $15,000-30,000 (DGX Cloud, Triton hosting)
- **Phase 4:** $5,000-10,000 (production hosting, monitoring)

#### External Dependencies
- **NVIDIA Partnership:** Required for Phase 3 (DGX Cloud access, NIM licensing)
- **Beta Customers:** Needed for Phase 4 (ideally lined up by Week 15)

---

## Risk Management Strategy

### High-Impact Risks

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| Python semantics too complex to translate | High | Critical | MVP subset, incremental expansion | Pivot to Python→Rust bindings (FFI) instead of transpilation |
| WASM performance worse than Python | Medium | High | Focus on I/O-bound tasks first, optimize later | Market as "portability" not "performance" |
| NVIDIA integration fails/delays | Medium | Medium | Make GPU optional, CPU fallback | Ship without NVIDIA features, add in v2.0 |
| No customer demand for product | Low | Critical | Early customer discovery, pilot programs | Pivot to internal tooling or open-source community project |
| Key engineer leaves mid-project | Medium | High | Knowledge sharing, pair programming | Cross-train team, maintain documentation |

### Mitigation Strategies
1. **Weekly Risk Review:** Team discusses blockers, escalates early
2. **Prototype Early:** Build risky components first (e.g., class translation)
3. **Customer Validation:** Monthly check-ins with target users
4. **Plan B for GPU:** Ensure CPU-only mode always works

---

## Success Metrics (KPIs)

### Technical Metrics
- **Translation Coverage:** % of Python features supported
  - Phase 1 Target: 60% (basic features)
  - Phase 2 Target: 85% (+ classes, modules)
  - Phase 3 Target: 90% (+ optimizations)

- **Test Pass Rate:** % of Python tests passing on WASM
  - Phase 1: 95%+ (simple scripts)
  - Phase 2: 90%+ (libraries)
  - Phase 3: 92%+ (with LLM assist)

- **Performance:** WASM execution time vs. Python
  - Phase 1: 0.5-2x (acceptable overhead)
  - Phase 3: 2-5x faster (optimized)

### Business Metrics
- **Time to MVP:** Weeks to first public demo
  - Target: Week 11 (Phase 1 completion)

- **Customer Adoption:** # of beta customers
  - Target: 3 by Week 27 (Phase 4 completion)

- **Community Engagement:** GitHub stars, contributors (if open-source)
  - Target: 500+ stars by Week 30

---

## Iterative Release Strategy

### Release Cadence
- **v0.1 (Week 3):** Internal alpha - Foundation only
- **v0.2 (Week 11):** Public beta - Script Mode MVP
- **v0.5 (Week 21):** Library Mode beta
- **v1.0 (Week 27):** General Availability - Enterprise-ready

### Feature Flags
Enable gradual rollout of risky features:
- `--experimental-classes` (Phase 2)
- `--enable-llm` (Phase 3)
- `--gpu-accelerated` (Phase 3)

### Backward Compatibility
- Maintain compatibility for generated Rust code (API stability)
- Version CLI commands (allow `portalis v1 convert` and `portalis v2 convert` coexistence)

---

## Conclusion

This roadmap balances **aggressive execution** (6-8 months to v1.0) with **risk management** (clear phase gates, fallback plans). The TDD outside-in approach ensures:

1. **Early Validation:** Phase 1 proves core feasibility with real scripts
2. **Incremental Value:** Each phase delivers usable features
3. **Flexibility:** Phase 3 (GPU) can be descoped if needed
4. **Customer Focus:** Beta customers involved from Week 15+

**Next Steps:**
1. Get stakeholder approval on Phase 0-1 scope
2. Assemble initial team (3 engineers)
3. Kick off Phase 0 (Week 0)
4. Set up weekly progress reviews

**Key Decision Points:**
- **Week 11:** Phase 1 gate - GO/NO-GO for Library Mode
- **Week 21:** Phase 2 gate - Commit to NVIDIA integration?
- **Week 26:** Launch readiness review - GA or extended beta?
