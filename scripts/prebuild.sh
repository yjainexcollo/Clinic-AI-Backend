#!/bin/bash
# Pre-build script for Azure deployment

echo "🔧 Running pre-build script..."

# Ensure Python path is set correctly
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Create necessary directories
mkdir -p /tmp/clinicai
mkdir -p /tmp/clinicai_audio

# Set permissions
chmod +x startup.py
chmod +x startup-gunicorn.py

echo "✅ Pre-build script completed"
