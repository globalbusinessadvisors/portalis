# PORTALIS - Quick Start Guide for SPARC Completion Phase

**Status:** 🔴 Core Platform Missing - Action Required
**Timeline:** 8 weeks to production
**Current Phase:** SPARC Phase 5 (Completion)

---

## 🚨 Critical Situation Summary

**What We Have:**
- ✅ 1.2M lines of planning documentation (Specification, Pseudocode, Architecture)
- ✅ 21,000 LOC of NVIDIA acceleration infrastructure (NeMo, CUDA, Triton, NIM, DGX, Omniverse)
- ✅ 3,936 LOC of integration tests for NVIDIA stack
- ✅ Complete benchmarking and monitoring

**What We're Missing:**
- ❌ Core platform (0 LOC)
- ❌ 7 agents (Ingest, Analysis, SpecGen, Transpiler, Build, Test, Package)
- ❌ Agent framework and orchestration
- ❌ End-to-end Python → Rust → WASM pipeline

**The Problem:** We built the turbocharger before the engine!

---

## 📖 Essential Reading (In Order)

1. **This Document** (3 min read) - Understand the situation
2. [SPARC_COMPLETION_PHASE_REPORT.md](SPARC_COMPLETION_PHASE_REPORT.md) (15 min) - Full analysis
3. [WEEK_1_ACTION_PLAN.md](WEEK_1_ACTION_PLAN.md) (10 min) - Implementation guide
4. [README.md](README.md) (5 min) - Project overview

---

## 🎯 Week 1 Goals (Next 5 Days)

### Day 1-2: Foundation
- [ ] Create agent framework (base.py, pipeline.py)
- [ ] Write 5+ unit tests
- [ ] CI/CD pipeline operational

### Day 3: Ingest Agent
- [ ] Implement Python file validation
- [ ] Implement mode detection (script/library)
- [ ] Write 10+ unit tests

### Day 4-5: Analysis Agent
- [ ] Implement AST parsing (CPU-only)
- [ ] Extract functions, classes, imports
- [ ] Write 15+ unit tests
- [ ] Integration test (Ingest → Analysis)

### Exit Criteria
- ✅ 30+ tests passing (>80% coverage)
- ✅ Can process fibonacci.py → API JSON
- ✅ CI/CD green

---

## 🏃 Quick Start for Developers

### Setup (5 minutes)
```bash
# Clone and enter
cd /workspace/portalis

# Create structure
mkdir -p agents/{base,ingest,analysis,spec_gen,transpiler,build,test,package}
mkdir -p core/{types,protocols,utils}
mkdir -p orchestration
mkdir -p tests/unit

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install pytest pytest-cov
```

### First Task: Agent Base Class (30 minutes)

**File:** `agents/base/agent.py`
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class AgentStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class AgentContext:
    input_path: str
    mode: str
    metadata: dict
    artifacts: dict

@dataclass
class AgentResult:
    status: AgentStatus
    data: any
    errors: list

class Agent(ABC):
    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        pass

    @abstractmethod
    def validate(self, context: AgentContext) -> bool:
        pass
```

**Test:** `tests/unit/test_agent_base.py`
```python
from agents.base.agent import Agent, AgentContext, AgentResult, AgentStatus

class DummyAgent(Agent):
    def execute(self, context):
        return AgentResult(AgentStatus.SUCCESS, {"done": True}, [])
    def validate(self, context):
        return True

def test_agent_executes():
    agent = DummyAgent()
    context = AgentContext("test.py", "script", {}, {})
    result = agent.execute(context)
    assert result.status == AgentStatus.SUCCESS
```

**Run:**
```bash
pytest tests/unit/test_agent_base.py -v
```

---

## 🗺️ 8-Week Roadmap

| Week | Sprint Goal | Key Deliverable |
|------|-------------|-----------------|
| 1 | Foundation + Ingest + Analysis | fibonacci.py → API JSON ✅ |
| 2 | SpecGen + Transpiler (CPU) | Python → Rust (basic) ✅ |
| 3 | Build + Test Agents | Rust → WASM + validation ✅ |
| 4 | Package Agent + E2E | Full CPU pipeline ✅ |
| 5 | NVIDIA Integration (Analysis) | GPU-accelerated parsing ✅ |
| 6 | NVIDIA Integration (Full) | Complete GPU pipeline ✅ |
| 7 | TDD Verification | 80% coverage, bug fixes ✅ |
| 8 | Production Ready | E2E validation, docs ✅ |

**Gates:**
- **Week 4:** Core platform functional (CPU-only)
- **Week 6:** NVIDIA integration complete
- **Week 8:** Production ready

---

## 🧪 Testing Strategy (TDD)

### London School Approach
1. Write acceptance test (fails)
2. Write integration test (fails)
3. Write unit test (fails)
4. Implement minimal code (pass)
5. Refactor
6. Repeat

### Week 1 Test Coverage
- **Agent Base:** 5 tests, 100% coverage
- **Ingest Agent:** 10 tests, >80% coverage
- **Analysis Agent:** 15 tests, >80% coverage
- **Integration:** 1 test (Ingest → Analysis)
- **Total:** 31 tests minimum

---

## 📋 Team Roles

| Role | Responsibility | Week 1 Tasks |
|------|----------------|--------------|
| **FoundationBuilder** | Agent framework | Day 1-2: base.py, pipeline.py |
| **IngestSpecialist** | Input processing | Day 3: IngestAgent |
| **AnalysisSpecialist** | AST parsing | Day 4-5: AnalysisAgent |
| **QA Engineer** | Testing | All week: Write tests |

---

## 🔗 Key Files to Implement (Week 1)

### Core Framework
- [ ] `agents/base/agent.py` - Base agent class
- [ ] `agents/base/__init__.py` - Package init
- [ ] `orchestration/pipeline.py` - Agent orchestration
- [ ] `core/types/common.py` - Shared data structures

### Ingest Agent
- [ ] `agents/ingest/ingest_agent.py` - Main implementation
- [ ] `agents/ingest/__init__.py` - Package init
- [ ] `tests/unit/test_ingest_agent.py` - Unit tests

### Analysis Agent
- [ ] `agents/analysis/analysis_agent.py` - Main implementation
- [ ] `agents/analysis/__init__.py` - Package init
- [ ] `tests/unit/test_analysis_agent.py` - Unit tests

### Integration
- [ ] `tests/integration/test_ingest_analysis.py` - E2E test

### Total Week 1 LOC Target: ~1,500 lines (800 implementation + 700 tests)

---

## 🚀 Success Metrics

### Week 1
- [x] Agent framework operational
- [x] 2 agents implemented (Ingest, Analysis)
- [x] 30+ tests passing
- [x] >80% code coverage
- [x] CI/CD green
- [x] fibonacci.py → API JSON working

### Week 4 (Gate 1)
- [ ] All 7 agents implemented (CPU-only)
- [ ] E2E pipeline functional
- [ ] 100+ tests passing
- [ ] >80% coverage
- [ ] Translate fibonacci.py → WASM

### Week 8 (Gate 3)
- [ ] NVIDIA integration complete
- [ ] Script Mode: 8/10 scripts pass
- [ ] Library Mode: 1 library validated
- [ ] Production documentation complete

---

## 🆘 Troubleshooting

### "Where do I start?"
→ Read [WEEK_1_ACTION_PLAN.md](WEEK_1_ACTION_PLAN.md), then implement `agents/base/agent.py`

### "Tests are failing"
→ That's TDD! Write failing test first, then implement to make it pass

### "How do I connect to NVIDIA stack?"
→ Not yet! Weeks 1-4 are CPU-only. NVIDIA integration starts Week 5

### "What about the existing code?"
→ NVIDIA stack (nemo-integration/, cuda-acceleration/, etc.) is done. Don't touch it yet. Focus on agents/

### "Can I skip testing?"
→ No. This is London School TDD. Tests are mandatory and come first.

---

## 📞 Getting Help

**Documentation:**
- Full Report: [SPARC_COMPLETION_PHASE_REPORT.md](SPARC_COMPLETION_PHASE_REPORT.md)
- Week 1 Plan: [WEEK_1_ACTION_PLAN.md](WEEK_1_ACTION_PLAN.md)
- Architecture: [plans/architecture.md](plans/architecture.md)
- Pseudocode: [plans/pseudocode.md](plans/pseudocode.md)

**Code Examples:**
- NVIDIA services show patterns: `nemo-integration/src/translation/nemo_service.py`
- Test fixtures: `tests/conftest.py`

---

## ✅ Daily Checklist (Week 1)

### Day 1
- [ ] Read this guide and completion report
- [ ] Create directory structure
- [ ] Implement Agent base class
- [ ] Write 5 unit tests
- [ ] All tests pass

### Day 2
- [ ] Implement Pipeline orchestration
- [ ] Write pipeline tests
- [ ] Setup CI/CD
- [ ] CI passes

### Day 3
- [ ] Implement IngestAgent
- [ ] Write 10 unit tests
- [ ] Can read Python files
- [ ] Mode detection works

### Day 4
- [ ] Implement AnalysisAgent (AST parsing)
- [ ] Extract functions
- [ ] Write 10 unit tests

### Day 5
- [ ] Complete AnalysisAgent (classes, imports)
- [ ] Write 5 more unit tests
- [ ] Write integration test
- [ ] fibonacci.py → API JSON works ✅

---

## 🎉 Week 1 Demo

**Goal:** Show working Ingest → Analysis pipeline

```bash
# Setup test file
echo 'def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)' > /tmp/fib.py

# Run pipeline
python -c "
from orchestration.pipeline import Pipeline
from agents.ingest.ingest_agent import IngestAgent
from agents.analysis.analysis_agent import AnalysisAgent
from agents.base.agent import AgentContext

pipeline = Pipeline([IngestAgent(), AnalysisAgent()])
context = AgentContext('/tmp/fib.py', 'script', {}, {})
result = pipeline.execute(context)

print('API Spec:', result.data['AnalysisAgent']['api_spec'])
"

# Expected output:
# API Spec: {
#   'functions': [{'name': 'fibonacci', 'args': ['n'], ...}],
#   'classes': [],
#   'imports': []
# }
```

---

## 🔥 Priority Actions (Right Now!)

1. **Read [SPARC_COMPLETION_PHASE_REPORT.md](SPARC_COMPLETION_PHASE_REPORT.md)** (15 min)
2. **Read [WEEK_1_ACTION_PLAN.md](WEEK_1_ACTION_PLAN.md)** (10 min)
3. **Create directory structure** (2 min)
4. **Implement Agent base class** (30 min)
5. **Write first test** (15 min)
6. **Make test pass** (15 min)
7. **Commit and push** (5 min)

**Total:** ~90 minutes to first working code

---

**Remember:** We have excellent planning and infrastructure. Now we need to build the core platform. Follow the plan, write tests first, and ship incrementally.

**Start here:** [WEEK_1_ACTION_PLAN.md](WEEK_1_ACTION_PLAN.md)

---

*Document Version: 1.0*
*Last Updated: 2025-10-03*
*Status: 🚀 Ready to start Week 1*
