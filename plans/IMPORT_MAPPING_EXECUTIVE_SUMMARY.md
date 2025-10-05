# Import Mapping Requirements - Executive Summary

**Date**: 2025-10-04
**Status**: Requirements Analysis Complete
**Full Report**: See `IMPORT_MAPPING_REQUIREMENTS_ANALYSIS.md`

---

## Overview

Analysis of requirements to expand Python-to-Rust import mappings from current 150 modules/packages to comprehensive coverage of 1000+ mappings for production-ready Python→WASM transpilation.

---

## Current State ✅

### Infrastructure
- **Framework**: Production-ready with WASM tracking
- **Architecture**: Module, Function, Class, and API-level mapping support
- **Integration**: Fully integrated with transpiler

### Coverage
| Category | Mapped | Total | Coverage |
|----------|--------|-------|----------|
| **Python Stdlib** | 50 | 301 | 16.6% |
| **External Packages** | 100 | 1000+ | 10% |
| **Total Mappings** | 150 | 1300+ | 11.5% |

### WASM Compatibility (Current Mappings)
- **Full WASM**: 66 mappings (44%)
- **Partial WASM**: 14 mappings (9%)
- **Requires WASI**: 15 mappings (10%)
- **Requires JS Interop**: 39 mappings (26%)
- **Incompatible**: 16 mappings (11%)

---

## Gap Analysis

### Unmapped Modules

**Python Standard Library** (251 unmapped):
- Critical priority: 30 modules (email, http, sqlite3, html, concurrent.futures, etc.)
- Medium priority: 50 modules (pdb, inspect, importlib, statistics, ssl, etc.)
- Low priority: 171 modules (platform-specific, deprecated, specialized)

**External Packages** (900+ unmapped):
- High priority: 50 packages (statsmodels, seaborn, plotly, transformers, lightgbm, etc.)
- Medium priority: 150 packages (scientific, async, testing, geospatial, etc.)
- Long tail: 700+ packages (domain-specific, SDKs, niche tools)

### Missing Infrastructure

**API Granularity**:
- ❌ Class-to-struct mappings
- ❌ Method-to-method mappings (only 5-10 functions per module currently)
- ❌ Parameter-level mappings
- ❌ Type mappings (Python types → Rust types)
- ❌ Default value handling

**WASM Support**:
- ❌ WASI filesystem polyfills
- ❌ JS interop layer (fetch, crypto, storage)
- ❌ Conditional compilation support
- ❌ Browser vs WASI vs Edge target handling

**Tooling**:
- ❌ Auto Cargo.toml generation from imports
- ❌ Auto use statement generation
- ❌ WASM setup script generation
- ❌ Migration guide generation

---

## Target State (18 Months)

### Coverage Goals

| Phase | Timeframe | Stdlib | External | Total | Coverage |
|-------|-----------|--------|----------|-------|----------|
| **Phase 1** ✅ | Month 0 | 50 | 100 | 150 | 11.5% |
| **Phase 2** | Months 1-6 | 150 | 300 | 450 | 34.6% |
| **Phase 3** | Months 7-12 | 250 | 600 | 850 | 65.4% |
| **Phase 4** | Months 13-18 | 280 | 1000 | 1280 | 98.5% |

### Quality Goals

| Metric | Current | Month 6 | Month 12 | Month 18 |
|--------|---------|---------|----------|----------|
| **APIs per module** | 6 | 15 | 25 | 40 |
| **Class mappings** | 10% | 60% | 90% | 100% |
| **WASI polyfills** | 0% | 100% | 100% | 100% |
| **JS interop** | 0% | 100% | 100% | 100% |
| **WASM tested** | 50% | 70% | 90% | 95% |

---

## Implementation Roadmap

### Phase 2: Expansion (Months 1-6)

**Deliverables**:
1. ✅ Map 100 stdlib modules (50 → 150)
2. ✅ Map 200 external packages (100 → 300)
3. ✅ Class-level mapping framework
4. ✅ Enhanced API granularity (15+ per module)
5. ✅ WASI integration layer
6. ✅ JS interop layer
7. ✅ Auto Cargo.toml generation

**Critical Modules (Month 1-2)**:
- `email` - Email handling
- `http.client` - HTTP client
- `html.parser` - HTML parsing
- `sqlite3` - Database
- `concurrent.futures` - Async tasks
- `multiprocessing` - Parallelism
- `shutil` - File operations
- `platform` - Platform info
- `urllib.parse` - URL parsing
- `mimetypes` - MIME types

**Key Packages (Month 1-2)**:
- Data Science: `statsmodels`, `seaborn`, `plotly`, `bokeh`, `xarray`
- ML/AI: `transformers`, `lightgbm`, `xgboost`, `onnx`
- Testing: `tox`, `pytest-asyncio`, `pytest-mock`, `flake8`
- Database: `asyncpg`, `aiomysql`, `motor`, `aioredis`

### Phase 3: Comprehensive (Months 7-12)

**Deliverables**:
1. ✅ Map 100 more stdlib modules (150 → 250)
2. ✅ Map 300 more external packages (300 → 600)
3. ✅ Complete type mapping system
4. ✅ Auto-generation tools
5. ✅ Documentation generator
6. ✅ WASM test infrastructure

### Phase 4: Complete Platform (Months 13-18)

**Deliverables**:
1. ✅ Remaining stdlib modules (250 → 280)
2. ✅ Complete external packages (600 → 1000)
3. ✅ Production deployment pipeline
4. ✅ One-command transpilation
5. ✅ Full WASM optimization

---

## Resource Requirements

### Team Composition

| Role | FTE | Responsibilities |
|------|-----|------------------|
| **Mapping Engineers** | 3-4 | Research, write mappings, test |
| **WASM Engineers** | 2 | Polyfills, browser compat, optimization |
| **Type System Expert** | 1 | Type mappings, generics, traits |
| **Documentation** | 1 | Migration guides, API docs, examples |
| **QA/Testing** | 2 | Coverage, WASM testing, integration |
| **Total** | **9-10** | |

### Effort Estimates

| Phase | Duration | Hours | FTE-Months |
|-------|----------|-------|------------|
| Phase 2 | 6 months | 600 | 3 |
| Phase 3 | 6 months | 1200 | 6 |
| Phase 4 | 6 months | 800 | 4 |
| **Total** | **18 months** | **2600** | **13** |

**Note**: With 9-10 FTE team, can complete in 18 months

---

## Key Achievements Required

### Infrastructure (Months 1-4)

1. **Class Mapping Framework**
   - `ClassMapping` struct with constructors
   - `MethodMapping` with parameters
   - Property mappings
   - Inheritance handling

2. **WASI Polyfill Layer**
   - Filesystem operations
   - Environment variables
   - Process information
   - Integration with `pathlib`, `tempfile`, `glob`

3. **JS Interop Layer**
   - Fetch API wrapper (HTTP)
   - Crypto API wrapper (random, secrets)
   - Storage API wrapper (persistence)
   - Performance API wrapper (timing)

4. **Type System**
   - Python → Rust type mappings
   - Generic type handling
   - Union types
   - Callable/function types
   - Custom class → struct

### Tooling (Months 5-8)

1. **Auto-generation**
   - Cargo.toml from imports
   - Use statements from mappings
   - WASM polyfill injection
   - Target-specific features

2. **Documentation**
   - Per-module WASM guides
   - Migration examples
   - API compatibility matrix
   - Type conversion reference

3. **Testing**
   - WASM test harness
   - Browser automation
   - WASI test suite
   - Performance benchmarks

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **WASM Limitations** | High | Document clearly, provide server alternatives |
| **Rust Crate Quality** | Medium | Contribute upstream, build wrappers, maintain alternatives |
| **API Mismatch** | Medium | Translation layers, clear docs, dual APIs |
| **Performance** | Medium | Profile, optimize, WASM SIMD, native fallbacks |

### Process Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Scope Creep** | High | Strict prioritization, regular review, phase gates |
| **Python Evolution** | Medium | Design for extensibility, monitor development |
| **Team Availability** | High | Stagger hiring, cross-train, knowledge base |

---

## Success Metrics

### Coverage Targets

| Milestone | Stdlib | External | Total | When |
|-----------|--------|----------|-------|------|
| **Baseline** ✅ | 50 (16.6%) | 100 (10%) | 150 (11.5%) | Complete |
| **Phase 2** | 150 (50%) | 300 (30%) | 450 (34.6%) | Month 6 |
| **Phase 3** | 250 (83%) | 600 (60%) | 850 (65.4%) | Month 12 |
| **Phase 4** | 280 (93%) | 1000 (100%) | 1280 (98.5%) | Month 18 |

### Quality Targets

- [ ] 15+ APIs per module by Month 6
- [ ] 25+ APIs per module by Month 12
- [ ] 40+ APIs per module by Month 18
- [ ] 100% WASI polyfills by Month 6
- [ ] 100% JS interop by Month 6
- [ ] 95% WASM tested by Month 18

### Integration Targets

- [ ] Auto Cargo.toml by Month 3
- [ ] WASI layer by Month 4
- [ ] JS interop by Month 6
- [ ] WASM pipeline by Month 9
- [ ] One-click transpile by Month 12

---

## Validation Strategy

### Real-World Test Projects

1. **Data Science**: Jupyter notebook with NumPy/Pandas analysis
2. **Web Application**: FastAPI REST API with SQLAlchemy
3. **CLI Tool**: Click-based command-line application
4. **ML Pipeline**: Scikit-learn training and inference
5. **Async Service**: aiohttp web scraper with async processing

### Success Criteria (per project)

- [ ] 100% imports resolve
- [ ] 95%+ APIs work correctly
- [ ] 90%+ performance vs Python
- [ ] Works in target WASM environment
- [ ] Clear migration guide

---

## Immediate Next Steps (Month 1)

### Week 1-2: Infrastructure

1. ✅ Design `ClassMapping` structure
2. ✅ Design `MethodMapping` structure
3. ✅ Design `ParamMapping` structure
4. ✅ Implement in mapping framework
5. ✅ Add tests for class mappings

### Week 2-3: Critical Stdlib

1. ✅ Map `email` package (20+ APIs)
2. ✅ Map `http.client` (15+ APIs)
3. ✅ Map `html.parser` (15+ APIs)
4. ✅ Map `sqlite3` (25+ APIs)
5. ✅ Map `concurrent.futures` (10+ APIs)

### Week 3-4: WASI Foundation

1. ✅ Design `WasiPolyfill` structure
2. ✅ Implement filesystem operations
3. ✅ Implement environment access
4. ✅ Test with `pathlib`, `tempfile`
5. ✅ Document WASI usage

### Week 4: External Packages

1. ✅ Map `statsmodels` (20+ APIs)
2. ✅ Map `seaborn` (15+ APIs)
3. ✅ Map `plotly` (20+ APIs)
4. ✅ Map `transformers` (partial, 10+ APIs)
5. ✅ Map `lightgbm` (15+ APIs)

---

## Dependencies & Prerequisites

### Technical Dependencies

- ✅ Rust toolchain (stable + nightly)
- ✅ wasm-pack, wasm-bindgen
- ⏳ Wasmtime, Wasmer (WASI runtimes)
- ⏳ Python 3.12+ (for testing)
- ⏳ Browser testing framework

### Infrastructure Dependencies

- ⏳ CI/CD for WASM builds
- ⏳ Browser automation (Playwright)
- ⏳ WASI test runners
- ⏳ Performance benchmarking setup
- ⏳ Documentation hosting (mdBook)

### Knowledge Prerequisites

- ✅ Rust expertise (ownership, traits, generics)
- ✅ Python stdlib deep knowledge
- ✅ WASM/WASI understanding
- ⏳ wasm-bindgen mastery
- ⏳ Browser API knowledge (fetch, crypto, storage)

---

## Key Decisions & Tradeoffs

### Architecture Decisions

1. **Mapping Granularity**: Class/method level (not just module)
   - **Pro**: Accurate transpilation
   - **Con**: More work per mapping
   - **Decision**: Worth it for quality

2. **WASM Compatibility Levels**: Five levels (Full, Partial, WASI, JS, Incompatible)
   - **Pro**: Clear expectations
   - **Con**: Complex to maintain
   - **Decision**: Essential for WASM success

3. **Type Mapping**: Explicit Python→Rust type table
   - **Pro**: Correctness, performance
   - **Con**: Maintenance overhead
   - **Decision**: Required for safety

4. **Multiple Rust Alternatives**: Maintain alternatives list
   - **Pro**: Flexibility, quality options
   - **Con**: More research, testing
   - **Decision**: Better user experience

### Scope Decisions

1. **Target 1000 packages** (not all PyPI)
   - Rationale: Covers 99%+ of real-world Python code
   - Tradeoff: Some niche packages unmapped

2. **93% stdlib coverage** (not 100%)
   - Rationale: Platform-specific and deprecated excluded
   - Tradeoff: Some edge cases unsupported

3. **API-level mapping** (not line-by-line)
   - Rationale: Idiomatic Rust is different from Python
   - Tradeoff: Manual mapping work vs auto-translation

---

## Conclusion

**Summary**: Comprehensive roadmap to expand Python→Rust import mappings from 150 to 1280+ over 18 months with 9-10 FTE team.

**Strengths**:
- ✅ Solid foundation (50 stdlib + 100 packages mapped)
- ✅ Production-ready infrastructure
- ✅ Clear WASM strategy
- ✅ Phased, risk-aware approach

**Next Steps**:
1. **Month 1**: Build class mapping framework + WASI foundation
2. **Months 2-3**: Map 20 critical stdlib modules + 50 packages
3. **Months 4-6**: Scale to 150 stdlib + 300 packages
4. **Months 7-12**: Comprehensive coverage + tooling
5. **Months 13-18**: Complete platform + production pipeline

**Success Criteria**: Can transpile 99%+ of real-world Python code to Rust/WASM with:
- 100% imports resolved
- 95%+ APIs working
- 90%+ performance
- Clear migration path

**Recommendation**: ✅ Proceed with phased implementation

---

**Document**: Executive Summary v1.0
**Full Analysis**: `IMPORT_MAPPING_REQUIREMENTS_ANALYSIS.md` (90+ pages)
**Last Updated**: 2025-10-04
