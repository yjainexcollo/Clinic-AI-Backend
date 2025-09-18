#!/bin/bash

# Clinic-AI Bootstrap Script

set -e

echo "🚀 Bootstrapping Clinic-AI project..."

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -e .

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install pytest pytest-cov black isort flake8 mypy

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env file with your configuration values"
fi

# Run initial tests
echo "🧪 Running initial tests..."
python -m pytest tests/ -v

echo "🎉 Bootstrap completed successfully!"
echo "📖 Run 'make help' to see available commands"
echo "🚀 Run 'make dev' to start the development server"
