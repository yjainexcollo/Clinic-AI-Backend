# Clinic-AI Backend Dockerfile
# Optimized for Azure App Service Container deployment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package in editable mode
RUN pip install -e .

# Default port (can be overridden by PORT environment variable)
ENV PORT=8000

# Expose port for Azure App Service
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Run the application
# Azure App Service will set PORT env var, but we default to 8000
CMD ["sh", "-c", "echo 'Starting Clinic-AI backend on port ${PORT:-8000}...' && PYTHONPATH=./src uvicorn clinicai.app:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 75 --proxy-headers --forwarded-allow-ips='*' --log-level info"]
