# Clinic-AI Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Whisper prefetch disabled to reduce image size and memory during build

# Expose port (Render provides $PORT at runtime)
EXPOSE 10000

# Run the application (bind to Render $PORT)
ENV PORT=10000
CMD ["sh", "-c", "echo 'Starting Clinic-AI backend...' && PYTHONPATH=./src uvicorn clinicai.app:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 75 --proxy-headers --forwarded-allow-ips='*' --log-level info"]
