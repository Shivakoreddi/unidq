#!/bin/bash

# UNIDQ Build Script
# This script builds the package and runs checks

set -e  # Exit on error

echo "========================================="
echo "UNIDQ Package Build Script"
echo "========================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# Step 1: Clean previous builds
echo ""
echo "Step 1: Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
echo "✓ Cleaned"

# Step 2: Install/upgrade build tools
echo ""
echo "Step 2: Installing build tools..."
pip install --upgrade build twine
echo "✓ Build tools ready"

# Step 3: Run tests (optional, comment out if not needed)
echo ""
echo "Step 3: Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v || {
        echo "⚠ Tests failed. Continue anyway? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    }
    echo "✓ Tests passed"
else
    echo "⚠ pytest not found, skipping tests"
fi

# Step 4: Build the package
echo ""
echo "Step 4: Building package..."
python -m build
echo "✓ Build complete"

# Step 5: List created files
echo ""
echo "Created files:"
ls -lh dist/

# Step 6: Check the package
echo ""
echo "Step 5: Checking package..."
twine check dist/*
echo "✓ Package check passed"

# Step 7: Show package info
echo ""
echo "========================================="
echo "Build Summary"
echo "========================================="
echo "Package contents:"
tar -tzf dist/unidq-*.tar.gz | head -20
echo "..."

echo ""
echo "========================================="
echo "Build complete! 🎉"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Test on TestPyPI:"
echo "   ./scripts/deploy.sh test"
echo ""
echo "2. Deploy to PyPI:"
echo "   ./scripts/deploy.sh prod"
echo ""
echo "Or manually:"
echo "   twine upload --repository testpypi dist/*"
echo "   twine upload dist/*"
echo "========================================="
