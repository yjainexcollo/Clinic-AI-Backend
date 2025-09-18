#!/bin/bash

# Clinic-AI Pre-commit Script

set -e

echo "🔍 Running pre-commit checks..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Get staged files
staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -z "$staged_files" ]; then
    echo "✅ No Python files staged for commit"
    exit 0
fi

echo "📝 Checking staged Python files:"
echo "$staged_files"

# Run isort to check import sorting
echo "📚 Checking import sorting..."
isort --check-only --diff $staged_files

# Run black to check code formatting
echo "🎨 Checking code formatting..."
black --check --diff $staged_files

# Run flake8 for linting
echo "🔍 Running linting checks..."
flake8 $staged_files

# Run mypy for type checking (if available)
if command -v mypy &> /dev/null; then
    echo "🔍 Running type checks..."
    mypy $staged_files
else
    echo "⚠️  mypy not found, skipping type checks"
fi

# Run tests for changed files
echo "🧪 Running tests for changed files..."
python -m pytest tests/ -v --tb=short

echo "✅ All pre-commit checks passed!"
