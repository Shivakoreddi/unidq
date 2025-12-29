# UNIDQ Deployment Guide

This guide covers how to build and publish UNIDQ to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both [TestPyPI](https://test.pypi.org) and [PyPI](https://pypi.org)
2. **API Tokens**: Generate API tokens for authentication
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

3. **Configure credentials** in `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-testpypi-token-here
```

## Installation

Install build tools:
```bash
pip install build twine
```

## Build Process

### 1. Prepare for Release

Update version in `pyproject.toml`:
```toml
[project]
name = "unidq"
version = "0.1.0"  # Update this
```

Update `CHANGELOG.md` with release notes.

### 2. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info
```

### 3. Build the Package

```bash
python -m build
```

This creates:
```
dist/
├── unidq-0.1.0-py3-none-any.whl
└── unidq-0.1.0.tar.gz
```

### 4. Verify the Build

Check the contents:
```bash
tar -tzf dist/unidq-0.1.0.tar.gz
unzip -l dist/unidq-0.1.0-py3-none-any.whl
```

Run twine check:
```bash
twine check dist/*
```

## Testing on TestPyPI

### 1. Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

Or with explicit repository URL:
```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

### 2. Test Installation from TestPyPI

Create a test environment:
```bash
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate
```

Install from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ unidq
```

Note: Dependencies might not be available on TestPyPI. Install them separately:
```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    unidq
```

### 3. Verify Installation

```bash
python -c "from unidq import UNIDQ; print('Import successful!')"
python -c "import unidq; print(f'Version: {unidq.__version__}')"
```

Run a quick test:
```python
from unidq import UNIDQ, UNIDQConfig

config = UNIDQConfig()
model = UNIDQ(config)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
```

## Publishing to PyPI

### 1. Upload to PyPI

Once testing is successful:
```bash
twine upload dist/*
```

### 2. Verify on PyPI

Check the package page: https://pypi.org/project/unidq/

### 3. Test Installation from PyPI

```bash
pip install unidq
```

## Post-Release

### 1. Create Git Tag

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

### 2. Create GitHub Release

Go to: https://github.com/yourusername/unidq/releases/new
- Tag: v0.1.0
- Title: UNIDQ v0.1.0
- Description: Copy from CHANGELOG.md
- Attach dist files

### 3. Update Documentation

Update README badges if needed:
```markdown
[![PyPI version](https://badge.fury.io/py/unidq.svg)](https://pypi.org/project/unidq/)
```

## Automation Script

For convenience, use the provided script:

```bash
# Build and test
./scripts/build.sh

# Deploy to TestPyPI
./scripts/deploy.sh test

# Deploy to PyPI
./scripts/deploy.sh prod
```

## Troubleshooting

### Issue: "File already exists"

If you get this error, you need to:
1. Increment the version number in `pyproject.toml`
2. Rebuild: `python -m build`
3. Upload again

### Issue: "Invalid distribution"

Run `twine check dist/*` to see specific errors.

### Issue: Dependencies not found on TestPyPI

Use both TestPyPI and PyPI indexes:
```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    unidq
```

## Security Best Practices

1. **Never commit tokens** to git
2. **Use API tokens** instead of passwords
3. **Limit token scope** to upload only
4. **Rotate tokens** regularly
5. **Use 2FA** on PyPI accounts

## Version Management

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes (1.0.0 → 2.0.0)
- **MINOR**: New backwards-compatible features (1.0.0 → 1.1.0)
- **PATCH**: Backwards-compatible bug fixes (1.0.0 → 1.0.1)

Pre-release versions:
- Alpha: `0.1.0a1`
- Beta: `0.1.0b1`
- Release Candidate: `0.1.0rc1`

## Checklist

Before releasing:
- [ ] All tests pass (`pytest tests/`)
- [ ] Code is formatted (`black unidq/`)
- [ ] Linting passes (`flake8 unidq/`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version number is bumped
- [ ] Git is clean (no uncommitted changes)
- [ ] Build succeeds (`python -m build`)
- [ ] Twine check passes (`twine check dist/*`)
- [ ] TestPyPI upload works
- [ ] TestPyPI installation works
- [ ] Quick smoke test passes

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Build Documentation](https://pypa-build.readthedocs.io/)
