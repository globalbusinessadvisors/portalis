# London School TDD Quick Reference Guide

## For the Portalis Development Team

---

## What is London School TDD?

London School TDD (Mockist TDD) focuses on:
1. **Outside-In Development**: Start with acceptance tests
2. **Interaction Testing**: Mock collaborators, test behavior
3. **Tell Don't Ask**: Objects tell others what to do
4. **Dependency Injection**: Easy mocking of dependencies

---

## The London School Workflow

```
1. Write Acceptance Test (RED)
   ├─ Describes user story
   └─ Tests from user perspective

2. Write Unit Test (RED)
   ├─ Mock all collaborators
   ├─ Test interactions
   └─ Verify behavior

3. Write Minimal Code (GREEN)
   ├─ Just enough to pass
   └─ No extra features

4. Refactor (REFACTOR)
   ├─ Improve design
   ├─ Extract collaborators
   └─ Keep tests passing

5. Verify Acceptance Test (GREEN)
   └─ End-to-end check
```

---

## Quick Start Examples

### Example 1: Testing an API Endpoint

**DON'T** (Classic School - No Mocks):
```python
def test_translate_endpoint():
    # Requires real NeMo service, Triton, GPU, etc.
    response = client.post("/translate", json={"code": "def test(): pass"})
    assert response.status_code == 200
    # Slow, brittle, hard to debug
```

**DO** (London School - Mock Collaborators):
```python
def test_translate_endpoint_delegates_to_nemo_service(mock_nemo_service):
    """
    GIVEN a translation request
    WHEN the endpoint is called
    THEN it should delegate to NeMo service
    """
    with patch('routes.get_nemo_service', return_value=mock_nemo_service):
        response = client.post("/translate", json={"code": "def test(): pass"})

    # Verify interaction
    mock_nemo_service.translate_code.assert_called_once()
    call_args = mock_nemo_service.translate_code.call_args
    assert call_args[1]['python_code'] == "def test(): pass"
```

### Example 2: Testing a Service

**DON'T** (Testing Implementation):
```python
def test_nemo_service_translate():
    service = NeMoService(model_path="...")
    result = service.translate_code("def add(): pass")

    # Tests what it does, not how it collaborates
    assert len(result.rust_code) > 0
```

**DO** (Testing Collaboration):
```python
def test_nemo_service_delegates_to_model(mock_model):
    """
    GIVEN a translation request
    WHEN translate_code is called
    THEN it should build prompt and call model.generate
    """
    service = NeMoService(model_path="...", model=mock_model)
    service.translate_code("def add(): pass")

    # Verify interaction
    mock_model.generate.assert_called_once()
    prompt = mock_model.generate.call_args[1]['inputs'][0]
    assert "def add(): pass" in prompt
    assert "Rust" in prompt
```

### Example 3: Acceptance Test (BDD Style)

```python
class TestUserTranslatesCode:
    """
    Feature: Code Translation
    As a developer
    I want to translate Python to Rust
    So that I can use high-performance Rust code
    """

    def test_user_translates_simple_function(self, client):
        """
        Scenario: Translate a simple function
        Given I have a simple Python function
        When I submit it for translation
        Then I receive valid Rust code
        And the confidence is high
        """
        # Given
        python_code = "def add(a, b): return a + b"

        # When
        response = client.post("/translate", json={"python_code": python_code})

        # Then
        assert response.status_code == 200
        data = response.json()
        assert "rust_code" in data
        assert data["confidence"] > 0.7  # And
```

---

## Mocking Cheat Sheet

### Creating Mocks

```python
from unittest.mock import Mock, MagicMock, patch

# Simple mock
mock_service = Mock()

# Mock with spec (enforces interface)
mock_service = Mock(spec=NeMoService)

# Magic mock (supports magic methods)
mock_model = MagicMock()

# Patch in context
with patch('module.ClassName') as mock_class:
    # Use mock_class
    pass

# Patch as decorator
@patch('module.ClassName')
def test_something(mock_class):
    # Use mock_class
    pass
```

### Configuring Mock Behavior

```python
# Return value
mock_service.translate.return_value = "fn test() {}"

# Return different values on successive calls
mock_service.translate.side_effect = ["result1", "result2", "result3"]

# Raise exception
mock_service.translate.side_effect = RuntimeError("Error")

# Configure attributes
mock_obj.some_attr = 42
mock_obj.some_method.return_value = "value"
```

### Verifying Interactions

```python
# Called once
mock_service.translate.assert_called_once()

# Called with specific arguments
mock_service.translate.assert_called_once_with(code="def test(): pass")

# Called with any arguments
mock_service.translate.assert_called()

# Not called
mock_service.translate.assert_not_called()

# Call count
assert mock_service.translate.call_count == 3

# Inspect call arguments
call_args = mock_service.translate.call_args
args, kwargs = call_args
# or
assert call_args[0][0] == "first positional arg"
assert call_args[1]['code'] == "keyword arg value"
```

---

## Fixture Patterns

### Service Mocks

```python
@pytest.fixture
def mock_nemo_service():
    """Mock NeMo service with configured behavior."""
    service = MagicMock()

    # Configure default behavior
    result = MagicMock()
    result.rust_code = "fn default() -> i32 { 0 }"
    result.confidence = 0.9
    result.processing_time_ms = 100.0

    service.translate_code.return_value = result

    return service
```

### Test Client with Mocks

```python
@pytest.fixture
def client_with_mocks(mock_nemo_service, mock_triton_client):
    """Test client with all services mocked."""
    from main import create_app

    with patch('routes.get_nemo_service', return_value=mock_nemo_service), \
         patch('routes.get_triton_client', return_value=mock_triton_client):
        app = create_app()
        return TestClient(app)
```

---

## Common Patterns

### Pattern 1: Route Handler Delegation

```python
def test_route_delegates_to_service(client, mock_service):
    """Route should delegate to service, not perform logic."""
    response = client.post("/endpoint", json={"data": "test"})

    # Verify delegation
    mock_service.process.assert_called_once_with(data="test")

    # Verify no business logic in route
    assert response.status_code == 200
```

### Pattern 2: Service Collaboration

```python
def test_service_collaborates_with_repository(mock_repo):
    """Service should use repository to fetch data."""
    service = MyService(repository=mock_repo)
    mock_repo.get.return_value = SomeData()

    result = service.process(id=123)

    # Verify collaboration
    mock_repo.get.assert_called_once_with(id=123)
    # Verify transformation
    assert isinstance(result, ProcessedData)
```

### Pattern 3: Error Handling

```python
def test_handles_service_error_gracefully(client, mock_service):
    """Should return 500 when service fails."""
    mock_service.process.side_effect = RuntimeError("Service down")

    response = client.post("/endpoint", json={"data": "test"})

    assert response.status_code == 500
    assert "error" in response.json()
```

---

## Test Organization

### File Structure

```
tests/
├── acceptance/              # User story tests (BDD)
│   └── test_user_workflow.py
├── unit/                    # Fast unit tests with mocks
│   ├── routes/
│   │   └── test_api_routes.py
│   ├── services/
│   │   └── test_business_service.py
│   └── models/
│       └── test_domain_models.py
├── integration/             # Service integration tests
│   └── test_external_apis.py
└── conftest.py             # Shared fixtures
```

### Test Class Organization

```python
class TestFeatureName:
    """Test a specific feature."""

    @pytest.fixture
    def setup_data(self):
        """Feature-specific fixture."""
        return {...}

    def test_happy_path(self, setup_data):
        """Test successful case."""
        pass

    def test_error_case(self, setup_data):
        """Test error handling."""
        pass

    def test_edge_case(self, setup_data):
        """Test boundary conditions."""
        pass
```

---

## Running Tests

### Run Specific Test Categories

```bash
# Fast unit tests only (TDD workflow)
pytest tests/unit/ -v

# Acceptance tests
pytest tests/acceptance/ -v

# Integration tests
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

### Run with Coverage

```bash
# Coverage report
pytest tests/unit/ --cov=nim_microservices --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Single test file
pytest tests/unit/test_translation_routes.py -v

# Single test class
pytest tests/unit/test_translation_routes.py::TestTranslateEndpoint -v

# Single test method
pytest tests/unit/test_translation_routes.py::TestTranslateEndpoint::test_fast_mode -v

# By marker
pytest -m unit -v
```

### Debug Tests

```bash
# Stop on first failure
pytest tests/unit/ -x

# Show print statements
pytest tests/unit/ -s

# Show locals in traceback
pytest tests/unit/ -l

# Enter debugger on failure
pytest tests/unit/ --pdb
```

---

## Best Practices

### DO ✓

1. **Mock All External Dependencies**
   ```python
   @patch('module.ExternalAPI')
   @patch('module.Database')
   def test_service(mock_db, mock_api):
       # Fast, isolated test
       pass
   ```

2. **Test Behavior, Not Implementation**
   ```python
   # Good: Tests what happens
   mock_service.process.assert_called_once()

   # Bad: Tests internal state
   assert service._internal_cache == {...}
   ```

3. **Use Descriptive Test Names**
   ```python
   def test_translate_code_delegates_to_nemo_when_fast_mode():
       """Clear what is being tested."""
       pass
   ```

4. **Follow Given-When-Then**
   ```python
   def test_example():
       # Given: Setup
       mock_service = Mock()

       # When: Action
       result = my_function(mock_service)

       # Then: Verification
       assert result == expected
   ```

### DON'T ✗

1. **Don't Test Implementation Details**
   ```python
   # Bad
   assert service._build_prompt("code") == "some prompt"

   # Good
   mock_model.generate.assert_called()
   assert "code" in mock_model.generate.call_args[0][0]
   ```

2. **Don't Use Real External Services**
   ```python
   # Bad
   response = requests.get("https://real-api.com")

   # Good
   with patch('requests.get') as mock_get:
       mock_get.return_value = MockResponse()
   ```

3. **Don't Write Slow Tests in Unit Tests**
   ```python
   # Bad
   time.sleep(5)  # In unit test

   # Good
   with patch('time.sleep'):  # Mock delays
       pass
   ```

4. **Don't Mix Concerns**
   ```python
   # Bad: Tests multiple things
   def test_everything():
       test_api_and_database_and_cache()

   # Good: Focused tests
   def test_api_interaction():
       # Just API
   ```

---

## Troubleshooting

### Mock Not Being Called

```python
# Problem: Mock not called
mock_service.method.assert_called_once()  # Fails

# Solution: Check patch path
# Make sure patch path matches import location
@patch('module.where.used.Service')  # Not where defined
```

### Mock Returns None

```python
# Problem: Mock returns None
result = mock_service.method()  # None

# Solution: Configure return value
mock_service.method.return_value = "expected"
```

### Can't Mock Static/Class Methods

```python
# Use patch.object
with patch.object(MyClass, 'static_method', return_value=42):
    result = MyClass.static_method()
    assert result == 42
```

### Async Tests

```python
# Use pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    mock_service = AsyncMock()
    mock_service.async_method.return_value = "result"

    result = await my_async_function(mock_service)

    mock_service.async_method.assert_awaited_once()
```

---

## Resources

### Documentation
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)

### Books
- "Growing Object-Oriented Software, Guided by Tests" by Freeman & Pryce
- "Test Driven Development: By Example" by Kent Beck

### Articles
- Martin Fowler: "Mocks Aren't Stubs"
- Uncle Bob: "The Three Laws of TDD"

---

## Team Workflow

### Daily TDD Cycle

1. **Morning: Review Stories**
   - Identify acceptance criteria
   - Write acceptance tests

2. **Development: Red-Green-Refactor**
   ```bash
   # Red: Write failing test
   pytest tests/unit/test_new_feature.py -x

   # Green: Make it pass
   # ... write minimal code ...

   # Refactor: Improve design
   # ... extract, rename, optimize ...

   # Verify: Run all tests
   pytest tests/unit/ -v
   ```

3. **Before Commit: Full Test Run**
   ```bash
   pytest tests/ --tb=short
   ```

4. **Code Review: Review Tests First**
   - Check test quality
   - Verify mocking strategy
   - Ensure coverage

---

**Remember:**
- Tests are documentation
- Mock collaborators, not data
- Fast tests enable TDD
- When in doubt, mock it out!
