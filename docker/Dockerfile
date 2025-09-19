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

# Prefetch Whisper model at build time to avoid large downloads on first boot
# You can override WHISPER_MODEL via build-arg (tiny|base|small|medium|large)
ARG WHISPER_MODEL=base
ENV WHISPER_MODEL=${WHISPER_MODEL}
ENV XDG_CACHE_HOME=/root/.cache
ENV WHISPER_CACHE_DIR=/root/.cache/whisper
RUN python - <<'PYCODE'
import os
import whisper
model = os.environ.get('WHISPER_MODEL','base')
download_root = os.environ.get('WHISPER_CACHE_DIR')
print(f"Prefetching Whisper model: {model} → {download_root}")
whisper.load_model(model, download_root=download_root)
print("Whisper model cached.")
PYCODE

# Expose port (Render provides $PORT at runtime)
EXPOSE 10000

# Run the application (bind to Render $PORT)
ENV PORT=10000
CMD ["sh", "-c", "PYTHONPATH=./src uvicorn clinicai.app:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 75 --proxy-headers --forwarded-allow-ips='*'"]
