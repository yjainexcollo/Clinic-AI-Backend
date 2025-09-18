# Clinic-AI Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.clinicai.app:app", "--host", "0.0.0.0", "--port", "8000"]
