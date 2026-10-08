# Multi-stage Dockerfile for ParseAnything Universal Ingestion Platform
FROM python:3.11-slim

# System dependencies for PyMuPDF, OpenCV, ONNXRuntime, and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . .

# Expose default HTTP API port
EXPOSE 8585

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8585
ENV MAX_COST_TARGET_PER_1K=10.00

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8585/api/v1/health || exit 1

# Launch production server
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8585"]
