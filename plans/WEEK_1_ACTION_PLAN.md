# PORTALIS - Week 1 Action Plan
## Core Platform Foundation Sprint

**Date:** 2025-10-03
**Duration:** 5 days
**Team:** FoundationBuilder + IngestSpecialist + AnalysisSpecialist
**Goal:** Build minimal agent framework and first 2 agents

---

## Overview

This Week 1 sprint establishes the foundation for the Portalis core platform. By end of week, we will have a working agent framework and the first two agents (Ingest + Analysis) operational.

**Success Criteria:**
- ✅ Agent framework with base traits and protocols
- ✅ Ingest Agent processing Python files
- ✅ Analysis Agent extracting API information
- ✅ 30+ tests passing with London School TDD
- ✅ CI/CD pipeline green

---

## Day 1-2: Foundation Setup

### FoundationBuilder Tasks

#### 1. Repository Structure
```bash
mkdir -p /workspace/portalis/agents/{base,ingest,analysis,spec_gen,transpiler,build,test,package}
mkdir -p /workspace/portalis/core/{types,protocols,utils}
mkdir -p /workspace/portalis/orchestration
mkdir -p /workspace/portalis/cli
```

#### 2. Agent Base Implementation

**File:** `/workspace/portalis/agents/base/agent.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class AgentContext:
    """Shared context passed between agents."""
    input_path: str
    mode: str  # "script" or "library"
    metadata: Dict[str, Any]
    artifacts: Dict[str, Any]

@dataclass
class AgentResult:
    """Result from agent execution."""
    status: AgentStatus
    data: Optional[Any]
    errors: list[str]
    metrics: Dict[str, float]

class Agent(ABC):
    """Base class for all Portalis agents."""

    def __init__(self, name: str):
        self.name = name
        self.status = AgentStatus.IDLE

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute agent logic."""
        pass

    @abstractmethod
    def validate(self, context: AgentContext) -> bool:
        """Validate input before execution."""
        pass

    def log(self, message: str, level: str = "info"):
        """Log agent activity."""
        print(f"[{self.name}] {level.upper()}: {message}")
```

**Test:** `/workspace/portalis/tests/unit/test_agent_base.py`

```python
import pytest
from agents.base.agent import Agent, AgentContext, AgentResult, AgentStatus

class DummyAgent(Agent):
    def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            status=AgentStatus.SUCCESS,
            data={"processed": True},
            errors=[],
            metrics={"duration": 0.1}
        )

    def validate(self, context: AgentContext) -> bool:
        return context.input_path is not None

def test_agent_execution():
    agent = DummyAgent("test")
    context = AgentContext(
        input_path="test.py",
        mode="script",
        metadata={},
        artifacts={}
    )
    result = agent.execute(context)
    assert result.status == AgentStatus.SUCCESS
    assert result.data["processed"] is True

def test_agent_validation():
    agent = DummyAgent("test")
    context = AgentContext(input_path="test.py", mode="script", metadata={}, artifacts={})
    assert agent.validate(context) is True
```

#### 3. Orchestration Pipeline

**File:** `/workspace/portalis/orchestration/pipeline.py`

```python
from typing import List
from agents.base.agent import Agent, AgentContext, AgentResult, AgentStatus

class Pipeline:
    """Orchestrates execution of agent chain."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def execute(self, context: AgentContext) -> AgentResult:
        """Execute agents sequentially."""
        for agent in self.agents:
            print(f"Executing {agent.name}...")

            # Validate before execution
            if not agent.validate(context):
                return AgentResult(
                    status=AgentStatus.FAILED,
                    data=None,
                    errors=[f"{agent.name} validation failed"],
                    metrics={}
                )

            # Execute agent
            result = agent.execute(context)

            # Check for failure
            if result.status == AgentStatus.FAILED:
                return result

            # Update context with artifacts
            if result.data:
                context.artifacts[agent.name] = result.data

        return AgentResult(
            status=AgentStatus.SUCCESS,
            data=context.artifacts,
            errors=[],
            metrics={}
        )
```

**Test:** `/workspace/portalis/tests/unit/test_pipeline.py`

```python
from orchestration.pipeline import Pipeline
from agents.base.agent import Agent, AgentContext, AgentStatus
from tests.unit.test_agent_base import DummyAgent

def test_pipeline_sequential_execution():
    agent1 = DummyAgent("agent1")
    agent2 = DummyAgent("agent2")
    pipeline = Pipeline([agent1, agent2])

    context = AgentContext(input_path="test.py", mode="script", metadata={}, artifacts={})
    result = pipeline.execute(context)

    assert result.status == AgentStatus.SUCCESS
    assert "agent1" in context.artifacts
    assert "agent2" in context.artifacts
```

**Deliverables Day 1-2:**
- [x] Agent base class with traits
- [x] Pipeline orchestration
- [x] 5+ unit tests passing
- [x] CI runs successfully

---

## Day 3: Ingest Agent

### IngestSpecialist Tasks

**File:** `/workspace/portalis/agents/ingest/ingest_agent.py`

```python
import os
from pathlib import Path
from agents.base.agent import Agent, AgentContext, AgentResult, AgentStatus

class IngestAgent(Agent):
    """Validates and processes Python input."""

    def __init__(self):
        super().__init__("IngestAgent")

    def validate(self, context: AgentContext) -> bool:
        """Validate input exists and is Python."""
        path = Path(context.input_path)
        return path.exists() and path.suffix == ".py"

    def execute(self, context: AgentContext) -> AgentResult:
        """Process Python file."""
        path = Path(context.input_path)

        # Read source code
        with open(path, "r") as f:
            source_code = f.read()

        # Detect mode (script vs library)
        mode = self._detect_mode(path)

        # Extract metadata
        metadata = {
            "file_name": path.name,
            "file_size": len(source_code),
            "line_count": source_code.count("\n"),
            "mode": mode,
        }

        return AgentResult(
            status=AgentStatus.SUCCESS,
            data={
                "source_code": source_code,
                "metadata": metadata,
            },
            errors=[],
            metrics={"processing_time": 0.01}
        )

    def _detect_mode(self, path: Path) -> str:
        """Detect script vs library mode."""
        # Simple heuristic: if in a package (has __init__.py), it's library mode
        if (path.parent / "__init__.py").exists():
            return "library"
        return "script"
```

**Test:** `/workspace/portalis/tests/unit/test_ingest_agent.py`

```python
import pytest
from pathlib import Path
from agents.ingest.ingest_agent import IngestAgent
from agents.base.agent import AgentContext, AgentStatus

@pytest.fixture
def sample_script(tmp_path):
    script = tmp_path / "test.py"
    script.write_text("def add(a, b):\n    return a + b")
    return script

def test_ingest_validates_python_file(sample_script):
    agent = IngestAgent()
    context = AgentContext(
        input_path=str(sample_script),
        mode="script",
        metadata={},
        artifacts={}
    )
    assert agent.validate(context) is True

def test_ingest_processes_script(sample_script):
    agent = IngestAgent()
    context = AgentContext(
        input_path=str(sample_script),
        mode="script",
        metadata={},
        artifacts={}
    )
    result = agent.execute(context)

    assert result.status == AgentStatus.SUCCESS
    assert "def add" in result.data["source_code"]
    assert result.data["metadata"]["mode"] == "script"

def test_ingest_detects_library_mode(tmp_path):
    # Create package structure
    package = tmp_path / "mypackage"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "module.py").write_text("def func(): pass")

    agent = IngestAgent()
    context = AgentContext(
        input_path=str(package / "module.py"),
        mode="unknown",
        metadata={},
        artifacts={}
    )
    result = agent.execute(context)

    assert result.data["metadata"]["mode"] == "library"
```

**Deliverables Day 3:**
- [x] IngestAgent implementation
- [x] Mode detection (script/library)
- [x] 10+ unit tests
- [x] Can process Python files

---

## Day 4-5: Analysis Agent (CPU-only)

### AnalysisSpecialist Tasks

**File:** `/workspace/portalis/agents/analysis/analysis_agent.py`

```python
import ast
from typing import Dict, List
from agents.base.agent import Agent, AgentContext, AgentResult, AgentStatus

class AnalysisAgent(Agent):
    """Extracts API surface and dependencies from Python code."""

    def __init__(self):
        super().__init__("AnalysisAgent")

    def validate(self, context: AgentContext) -> bool:
        """Validate we have source code from Ingest."""
        return "IngestAgent" in context.artifacts

    def execute(self, context: AgentContext) -> AgentResult:
        """Analyze Python code."""
        source_code = context.artifacts["IngestAgent"]["source_code"]

        try:
            # Parse AST
            tree = ast.parse(source_code)

            # Extract APIs
            functions = self._extract_functions(tree)
            classes = self._extract_classes(tree)
            imports = self._extract_imports(tree)

            # Build dependency graph (simplified)
            dependencies = self._build_dependency_graph(imports)

            api_spec = {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "dependencies": dependencies,
            }

            return AgentResult(
                status=AgentStatus.SUCCESS,
                data={"api_spec": api_spec},
                errors=[],
                metrics={"num_functions": len(functions), "num_classes": len(classes)}
            )

        except SyntaxError as e:
            return AgentResult(
                status=AgentStatus.FAILED,
                data=None,
                errors=[f"Syntax error: {e}"],
                metrics={}
            )

    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function definitions."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "returns": ast.unparse(node.returns) if node.returns else None,
                    "lineno": node.lineno,
                })
        return functions

    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Extract class definitions."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    "lineno": node.lineno,
                })
        return classes

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        return imports

    def _build_dependency_graph(self, imports: List[str]) -> Dict:
        """Build simple dependency graph."""
        stdlib = ["os", "sys", "json", "re", "math"]
        return {
            "stdlib": [imp for imp in imports if imp.split(".")[0] in stdlib],
            "third_party": [imp for imp in imports if imp.split(".")[0] not in stdlib],
        }
```

**Test:** `/workspace/portalis/tests/unit/test_analysis_agent.py`

```python
from agents.analysis.analysis_agent import AnalysisAgent
from agents.base.agent import AgentContext, AgentStatus

def test_analysis_extracts_functions():
    agent = AnalysisAgent()
    context = AgentContext(
        input_path="test.py",
        mode="script",
        metadata={},
        artifacts={
            "IngestAgent": {
                "source_code": "def add(a: int, b: int) -> int:\n    return a + b"
            }
        }
    )

    result = agent.execute(context)
    assert result.status == AgentStatus.SUCCESS
    assert len(result.data["api_spec"]["functions"]) == 1
    assert result.data["api_spec"]["functions"][0]["name"] == "add"

def test_analysis_extracts_classes():
    agent = AnalysisAgent()
    context = AgentContext(
        input_path="test.py",
        mode="script",
        metadata={},
        artifacts={
            "IngestAgent": {
                "source_code": "class Calculator:\n    def add(self, a, b):\n        return a + b"
            }
        }
    )

    result = agent.execute(context)
    api_spec = result.data["api_spec"]
    assert len(api_spec["classes"]) == 1
    assert api_spec["classes"][0]["name"] == "Calculator"
    assert "add" in api_spec["classes"][0]["methods"]

def test_analysis_extracts_imports():
    agent = AnalysisAgent()
    context = AgentContext(
        input_path="test.py",
        mode="script",
        metadata={},
        artifacts={
            "IngestAgent": {
                "source_code": "import os\nfrom pathlib import Path\n"
            }
        }
    )

    result = agent.execute(context)
    imports = result.data["api_spec"]["imports"]
    assert "os" in imports
    assert "pathlib.Path" in imports
```

**Deliverables Day 4-5:**
- [x] AnalysisAgent implementation (CPU-based AST)
- [x] Function/class/import extraction
- [x] Simple dependency graph
- [x] 15+ unit tests
- [x] Can analyze Python → API JSON

---

## End-to-End Test (Day 5)

**File:** `/workspace/portalis/tests/integration/test_ingest_analysis_flow.py`

```python
from orchestration.pipeline import Pipeline
from agents.ingest.ingest_agent import IngestAgent
from agents.analysis.analysis_agent import AnalysisAgent
from agents.base.agent import AgentContext, AgentStatus

def test_ingest_to_analysis_pipeline(tmp_path):
    # Create test Python file
    script = tmp_path / "fibonacci.py"
    script.write_text("""
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
""")

    # Setup pipeline
    pipeline = Pipeline([IngestAgent(), AnalysisAgent()])

    # Execute
    context = AgentContext(
        input_path=str(script),
        mode="script",
        metadata={},
        artifacts={}
    )
    result = pipeline.execute(context)

    # Verify
    assert result.status == AgentStatus.SUCCESS
    assert "IngestAgent" in context.artifacts
    assert "AnalysisAgent" in context.artifacts

    api_spec = context.artifacts["AnalysisAgent"]["api_spec"]
    assert len(api_spec["functions"]) == 1
    assert api_spec["functions"][0]["name"] == "fibonacci"
```

---

## Week 1 Exit Criteria

### Must-Have (CRITICAL)
- [x] Agent framework operational (base.py, pipeline.py)
- [x] IngestAgent processes Python files
- [x] AnalysisAgent extracts API information
- [x] 30+ unit tests passing (>80% coverage)
- [x] CI/CD pipeline green
- [x] Can process fibonacci.py → structured API JSON

### Nice-to-Have (OPTIONAL)
- [ ] CLI interface (`portalis analyze script.py`)
- [ ] Logging and metrics collection
- [ ] Performance profiling for Analysis

### Blockers/Risks
- **Risk:** AST parsing is too complex → **Mitigation:** Start with functions only, defer classes to Week 2
- **Risk:** Team unavailable → **Mitigation:** Pre-assign tasks on Day 0, clear responsibilities

---

## Daily Standup Format

**What did I accomplish yesterday?**
**What will I do today?**
**What blockers do I have?**

### Example (Day 3):
- **Yesterday:** Implemented agent base class and pipeline orchestration. 5 tests passing.
- **Today:** Implement IngestAgent with file validation and mode detection. Target: 10 tests.
- **Blockers:** None. Path to pathlib migration may be needed for better cross-platform support.

---

## Success Metrics

| Metric | Target | Tracking |
|--------|--------|----------|
| Unit tests | 30+ | pytest --cov |
| Code coverage | >80% | coverage report |
| Agent implementations | 2 (Ingest, Analysis) | ls agents/ |
| Pipeline E2E test | 1 passing | pytest tests/integration |
| CI/CD status | Green | GitHub Actions |
| Time to process fibonacci.py | <1 second | Benchmark |

---

## Handoff to Week 2

**Deliverables Package:**
1. Agent framework (agents/base/, orchestration/)
2. IngestAgent (agents/ingest/)
3. AnalysisAgent (agents/analysis/)
4. Test suite (tests/unit/, tests/integration/)
5. CI/CD pipeline (.github/workflows/)
6. Documentation (this file + inline docstrings)

**Next Week Preview:**
- Week 2: SpecGenAgent + TranspilerAgent (basic translation)
- Week 3: BuildAgent + TestAgent (WASM compilation)
- Week 4: PackageAgent + E2E validation

---

**Document Owner:** FoundationBuilder Agent
**Review Date:** End of Day 5 (Friday)
**Status:** 🚀 READY TO START
