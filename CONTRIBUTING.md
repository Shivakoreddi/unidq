# Contributing to UNIDQ

Thank you for your interest in contributing to UNIDQ! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, PyTorch version)
- **Code sample** demonstrating the issue

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a detailed description of the proposed functionality
- Explain why this enhancement would be useful
- Include code examples if applicable

### Types of Contributions Needed

We welcome the following types of contributions:

1. **Bug fixes** - Fix issues in existing code
2. **New features** - Add new data quality tasks or capabilities
3. **Documentation** - Improve or add documentation
4. **Tests** - Increase test coverage
5. **Performance improvements** - Optimize existing code
6. **Examples** - Add usage examples or tutorials
7. **Pre-trained models** - Contribute pre-trained weights

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- Git

### Setup Instructions

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/unidq.git
cd unidq
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies**

```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks** (optional but recommended)

```bash
pip install pre-commit
pre-commit install
```

5. **Run tests to verify setup**

```bash
pytest tests/ -v
```

## Pull Request Process

1. **Create a new branch** for your feature or bugfix

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

2. **Make your changes** following our [coding standards](#coding-standards)

3. **Add tests** for your changes
   - All new features must include tests
   - Bug fixes should include regression tests

4. **Run the test suite**

```bash
pytest tests/ -v
```

5. **Update documentation** if needed
   - Update README.md if API changes
   - Add docstrings to new functions/classes
   - Update USER_GUIDE.md for new features

6. **Commit your changes**

```bash
git add .
git commit -m "Add feature: your feature description"
```

Follow conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `perf:` for performance improvements

7. **Push to your fork**

```bash
git push origin feature/your-feature-name
```

8. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template with:
     - Description of changes
     - Related issue numbers
     - Testing performed
     - Screenshots (if UI changes)

9. **Code Review**
   - Maintainers will review your PR
   - Address any requested changes
   - Once approved, your PR will be merged

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://github.com/psf/black) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting

```bash
# Format code
black unidq/

# Check linting
flake8 unidq/
```

### Code Organization

- Keep functions focused and small
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Use type hints where appropriate

### Documentation Style

Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function.
    
    Longer description if needed, explaining the purpose
    and behavior of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Aim for high test coverage (>80%)

### Test Structure

```python
import pytest
from unidq import UNIDQ

class TestFeatureName:
    """Test suite for FeatureName."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_sample_data()
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        result = function_under_test(sample_data)
        assert result == expected_value
    
    def test_edge_case(self):
        """Test edge case behavior."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=unidq --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestUNIDQ::test_forward_pass
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Run full test suite
4. Build and test package locally
5. Create git tag: `git tag -a v0.1.X -m "Release v0.1.X"`
6. Push tag: `git push origin v0.1.X`
7. Create GitHub release
8. Deploy to PyPI

### Release Cadence

- **Patch releases**: As needed for bug fixes
- **Minor releases**: Monthly or when significant features are ready
- **Major releases**: When breaking changes are necessary

### Release Criteria

A release is made when:
- All tests pass on supported platforms (Ubuntu, macOS, Windows)
- Documentation is updated
- CHANGELOG.md is current
- No critical bugs in the current version
- Compatible with latest two PyTorch releases

### Supported Versions

We maintain compatibility with:
- Python 3.8, 3.9, 3.10, 3.11
- PyTorch 1.9+ (latest two major versions)
- All major operating systems (Linux, macOS, Windows)

## Project Governance

### Maintainers

Current maintainers:
- @Shivakoreddi - Core maintainer
- @sravanisowrupilli - Core maintainer

### Decision Making

- **Minor changes**: Single maintainer approval
- **Major changes**: Consensus from both maintainers
- **Breaking changes**: Community discussion + maintainer consensus

### Becoming a Maintainer

Active contributors who demonstrate:
- Consistent high-quality contributions
- Deep understanding of the codebase
- Commitment to the project's goals

may be invited to become maintainers.

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: shivacse14@gmail.com, sravani.sowrupilli@gmail.com

### Recognition

We value all contributions! Contributors will be:
- Listed in the README.md
- Acknowledged in release notes
- Credited in academic citations when applicable

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to UNIDQ! 🎉
