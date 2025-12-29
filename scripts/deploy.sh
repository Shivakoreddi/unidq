#!/bin/bash

# UNIDQ Deployment Script
# Usage: ./scripts/deploy.sh [test|prod]

set -e  # Exit on error

ENVIRONMENT=${1:-test}

echo "========================================="
echo "UNIDQ Deployment Script"
echo "========================================="

if [ "$ENVIRONMENT" != "test" ] && [ "$ENVIRONMENT" != "prod" ]; then
    echo "Usage: $0 [test|prod]"
    echo "  test - Deploy to TestPyPI"
    echo "  prod - Deploy to PyPI"
    exit 1
fi

# Check if dist/ exists
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo "Error: dist/ directory is empty or doesn't exist."
    echo "Run './scripts/build.sh' first to build the package."
    exit 1
fi

# Verify twine is installed
if ! command -v twine &> /dev/null; then
    echo "Error: twine is not installed."
    echo "Run: pip install twine"
    exit 1
fi

# Check the package first
echo ""
echo "Checking package..."
twine check dist/*
echo "✓ Package check passed"

if [ "$ENVIRONMENT" = "test" ]; then
    echo ""
    echo "========================================="
    echo "Deploying to TestPyPI"
    echo "========================================="
    echo ""
    echo "This will upload to: https://test.pypi.org"
    echo "Continue? (y/n)"
    read -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
    
    echo ""
    echo "Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    
    echo ""
    echo "========================================="
    echo "Upload to TestPyPI complete! 🎉"
    echo "========================================="
    echo ""
    echo "View at: https://test.pypi.org/project/unidq/"
    echo ""
    echo "Test installation:"
    echo "  pip install --index-url https://test.pypi.org/simple/ \\"
    echo "      --extra-index-url https://pypi.org/simple/ \\"
    echo "      unidq"
    echo ""
    echo "After testing, deploy to production:"
    echo "  ./scripts/deploy.sh prod"
    echo "========================================="
    
elif [ "$ENVIRONMENT" = "prod" ]; then
    echo ""
    echo "========================================="
    echo "⚠️  PRODUCTION DEPLOYMENT ⚠️"
    echo "========================================="
    echo ""
    echo "This will upload to: https://pypi.org"
    echo ""
    echo "Have you:"
    echo "  - Tested on TestPyPI?"
    echo "  - Updated version number?"
    echo "  - Updated CHANGELOG.md?"
    echo "  - Committed all changes?"
    echo ""
    echo "⚠️  This action CANNOT be undone! ⚠️"
    echo ""
    echo "Type 'DEPLOY' to continue, or anything else to cancel:"
    read -r response
    
    if [ "$response" != "DEPLOY" ]; then
        echo "Deployment cancelled."
        exit 0
    fi
    
    echo ""
    echo "Uploading to PyPI..."
    twine upload dist/*
    
    # Get version from pyproject.toml
    VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
    
    echo ""
    echo "========================================="
    echo "Upload to PyPI complete! 🚀"
    echo "========================================="
    echo ""
    echo "View at: https://pypi.org/project/unidq/"
    echo ""
    echo "Installation:"
    echo "  pip install unidq"
    echo ""
    echo "Next steps:"
    echo "1. Create git tag:"
    echo "   git tag -a v${VERSION} -m 'Release version ${VERSION}'"
    echo "   git push origin v${VERSION}"
    echo ""
    echo "2. Create GitHub release:"
    echo "   https://github.com/yourusername/unidq/releases/new"
    echo ""
    echo "3. Announce the release!"
    echo "========================================="
fi
