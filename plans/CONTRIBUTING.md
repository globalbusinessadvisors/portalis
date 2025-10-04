# Contributing to Portalis

Thank you for your interest in contributing to Portalis! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Adding New Features](#adding-new-features)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:

- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

Report any unacceptable behavior to conduct@portalis.dev.

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Rust 1.75+ (`rustup` recommended)
- Git
- Familiarity with async Rust (Tokio)
- Understanding of Python (for translation features)

### Find an Issue

1. Browse [open issues](https://github.com/portalis/portalis/issues)
2. Look for `good-first-issue` or `help-wanted` labels
3. Comment on the issue to claim it
4. Wait for maintainer approval before starting work

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR-USERNAME/portalis.git
cd portalis

# Add upstream remote
git remote add upstream https://github.com/portalis/portalis.git

# Verify remotes
git remote -v
```

---

## Development Setup

### Install Dependencies

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WASM target
rustup target add wasm32-wasi

# Install development tools
cargo install cargo-watch cargo-nextest cargo-tarpaulin
```

### Build the Project

```bash
# Build all workspace members
cargo build --all

# Build with GPU features (requires CUDA)
cargo build --all --features gpu

# Run tests
cargo test --all

# Run specific agent tests
cargo test -p portalis-ingest
```

### Run Locally

```bash
# Run CLI
cargo run --bin portalis -- translate --input examples/fibonacci.py

# Run with logging
RUST_LOG=debug cargo run --bin portalis -- translate --input test.py

# Watch for changes and rebuild
cargo watch -x "test --all"
```

---

## Code Style Guidelines

### Rust Code Style

We follow standard Rust conventions:

#### Formatting

```bash
# Format code with rustfmt
cargo fmt --all

# Check formatting
cargo fmt --all -- --check
```

#### Linting

```bash
# Run clippy for linting
cargo clippy --all --all-targets --all-features

# Fix automatically (when possible)
cargo clippy --fix --all
```

#### Naming Conventions

```rust
// Types: PascalCase
struct TranslationResult { }
enum MessageType { }

// Functions and methods: snake_case
fn translate_python() { }
async fn process_message(&self) -> Result<()> { }

// Constants: SCREAMING_SNAKE_CASE
const MAX_FILE_SIZE: usize = 10_000_000;

// Module names: snake_case
mod type_inference;
```

#### Documentation

All public items must be documented:

```rust
/// Translates Python source code to Rust.
///
/// # Arguments
///
/// * `source` - Python source code to translate
/// * `mode` - Translation mode (Pattern or NeMo)
///
/// # Returns
///
/// Translated Rust code as a string
///
/// # Errors
///
/// Returns error if translation fails or unsupported features are encountered
///
/// # Example
///
/// ```
/// let rust_code = translator.translate(python_code, TranslationMode::Pattern)?;
/// ```
pub async fn translate(&self, source: &str, mode: TranslationMode) -> Result<String> {
    // Implementation
}
```

### Python Code Style (NVIDIA Stack)

For Python code in the NVIDIA integration:

```bash
# Use Black for formatting
black nemo-integration/ nim-microservices/

# Use isort for imports
isort nemo-integration/ nim-microservices/

# Use pylint for linting
pylint nemo-integration/ nim-microservices/
```

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples**:

```
feat(transpiler): add support for Python match statements

Implement translation for Python 3.10+ match/case statements to Rust
match expressions. Includes pattern matching for literals, variables,
and guards.

Fixes #123
```

```
fix(ingest): handle edge case in AST parsing

Fixed crash when parsing files with Unicode characters in comments.

Closes #456
```

---

## Adding New Features

### Python Language Features

To add support for a new Python feature:

1. **Update Analysis Agent**:
   - Add pattern detection in `/agents/analysis/`
   - Update type inference if needed

2. **Update Transpiler**:
   - Add translation rule in `/agents/transpiler/`
   - Map Python construct to Rust equivalent

3. **Add Tests**:
   - Unit tests for the translation rule
   - Integration tests for the full pipeline
   - Golden tests for expected output

4. **Update Documentation**:
   - Update `docs/python-compatibility.md`
   - Add examples to documentation

**Example** - Adding list comprehensions:

```rust
// In transpiler/src/expressions.rs

/// Translate Python list comprehension to Rust iterator chain
fn translate_list_comprehension(&self, comp: &ListComp) -> Result<String> {
    // [x**2 for x in range(10) if x % 2 == 0]
    // →
    // (0..10).filter(|x| x % 2 == 0).map(|x| x.pow(2)).collect()

    let iter_expr = self.translate_expr(&comp.iter)?;
    let map_expr = self.translate_expr(&comp.elt)?;
    let filter_expr = comp.ifs.first()
        .map(|f| self.translate_expr(f))
        .transpose()?;

    Ok(format!(
        "{}.{}map(|{}| {}).collect()",
        iter_expr,
        filter_expr.map(|f| format!("filter(|{}| {})", comp.target, f))
            .unwrap_or_default(),
        comp.target,
        map_expr
    ))
}
```

### New Agents

To add a new specialized agent:

1. **Create Agent Module**: `/agents/my-agent/`
2. **Implement Agent Trait**:

```rust
use portalis_core::{Agent, Message, Result};
use async_trait::async_trait;

pub struct MyAgent {
    // State
}

#[async_trait]
impl Agent for MyAgent {
    async fn process(&self, message: Message) -> Result<Message> {
        // Implementation
        todo!()
    }

    fn name(&self) -> &str {
        "MyAgent"
    }

    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
```

3. **Add Tests**:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_my_agent_processes_message() {
        let agent = MyAgent::new();
        let message = create_test_message();

        let result = agent.process(message).await;
        assert!(result.is_ok());
    }
}
```

4. **Integrate with Pipeline**:
   - Update `orchestration/src/pipeline.rs`
   - Add to message routing

---

## Testing Requirements

### Test Categories

All features must include:

1. **Unit Tests**: Test individual functions/modules
2. **Integration Tests**: Test agent interactions
3. **End-to-End Tests**: Test full translation pipeline

### London School TDD

We follow London School TDD (mockist approach):

```rust
use mockall::mock;

// Define mock for dependencies
mock! {
    NeMoClient {}

    #[async_trait]
    impl NeMoService for NeMoClient {
        async fn translate(&self, request: TranslateRequest) -> Result<TranslateResponse>;
    }
}

#[tokio::test]
async fn test_transpiler_with_mock_nemo() {
    // Arrange
    let mut mock_nemo = MockNeMoClient::new();
    mock_nemo.expect_translate()
        .times(1)
        .returning(|_| Ok(TranslateResponse {
            rust_code: "fn test() {}".to_string(),
            confidence: 0.95,
            metrics: Default::default(),
        }));

    let transpiler = Transpiler::new(Box::new(mock_nemo));

    // Act
    let result = transpiler.translate_with_nemo("def test(): pass").await;

    // Assert
    assert!(result.is_ok());
    assert!(result.unwrap().contains("fn test"));
}
```

### Coverage Requirements

- **Minimum Coverage**: 80% for new code
- **Critical Paths**: 95%+ coverage

```bash
# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage

# Open coverage/index.html in browser
```

### Running Tests

```bash
# Run all tests
cargo test --all

# Run with output
cargo test --all -- --nocapture

# Run specific test
cargo test test_name

# Run tests for specific package
cargo test -p portalis-transpiler

# Run with nextest (faster)
cargo nextest run --all
```

---

## Pull Request Process

### Before Submitting

1. **Update from upstream**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run checks**:
```bash
# Format code
cargo fmt --all

# Lint
cargo clippy --all -- -D warnings

# Test
cargo test --all

# Check documentation
cargo doc --all --no-deps
```

3. **Update CHANGELOG.md**:
```markdown
## [Unreleased]

### Added
- Support for Python match statements (#123)
```

### Submitting PR

1. **Push to your fork**:
```bash
git push origin feature/my-feature
```

2. **Create Pull Request**:
   - Go to GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out PR template

3. **PR Template**:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing locally

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added with good coverage
- [ ] CHANGELOG.md updated
```

### Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least 1 maintainer approval required
3. **Testing**: All tests must pass
4. **Documentation**: Docs must be updated

### Addressing Feedback

```bash
# Make requested changes
git add .
git commit -m "Address review feedback"

# Push updates
git push origin feature/my-feature

# PR automatically updates
```

---

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create git tag
5. Push to GitHub
6. CI builds and publishes to crates.io
7. Create GitHub release

---

## Community

### Communication Channels

- **GitHub Discussions**: General questions, ideas
- **Discord**: Real-time chat ([discord.gg/portalis](https://discord.gg/portalis))
- **GitHub Issues**: Bug reports, feature requests
- **Email**: dev@portalis.dev

### Getting Help

- Read [documentation](https://docs.portalis.dev)
- Search existing issues
- Ask in Discord
- Create a new issue with details

---

## Recognition

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Annual contributor awards

Thank you for contributing to Portalis!
