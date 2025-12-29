#!/bin/bash

# Quick Deploy Script - One command to build and deploy
# Usage: ./scripts/quick_deploy.sh [test|prod]

set -e

ENVIRONMENT=${1:-test}

echo "========================================="
echo "UNIDQ Quick Deploy"
echo "========================================="

# Step 1: Install build tools
echo ""
echo "Installing build tools..."
pip install build twine

# Step 2: Clean and build
echo ""
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

echo ""
echo "Building package..."
python -m build

echo ""
echo "Package contents:"
ls -lh dist/

# Step 3: Check package
echo ""
echo "Checking package..."
twine check dist/*

# Step 4: Deploy
if [ "$ENVIRONMENT" = "test" ]; then
    echo ""
    echo "Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    
    echo ""
    echo "========================================="
    echo "Deployed to TestPyPI! 🎉"
    echo "========================================="
    echo ""
    echo "Test installation:"
    echo "  pip install --index-url https://test.pypi.org/simple/ \\"
    echo "      --extra-index-url https://pypi.org/simple/ \\"
    echo "      unidq"
    
elif [ "$ENVIRONMENT" = "prod" ]; then
    echo ""
    echo "⚠️  Uploading to PRODUCTION PyPI..."
    echo "Continue? Type 'yes' to confirm:"
    read -r response
    
    if [ "$response" = "yes" ]; then
        twine upload dist/*
        
        echo ""
        echo "========================================="
        echo "Deployed to PyPI! 🚀"
        echo "========================================="
        echo ""
        echo "Installation: pip install unidq"
        echo "View at: https://pypi.org/project/unidq/"
    else
        echo "Deployment cancelled."
    fi
else
    echo "Usage: $0 [test|prod]"
    exit 1
fi
