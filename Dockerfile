# ==============================================================================
# GRACE AI Network Threat Detection System - Production Dockerfile
# Base: Python 3.12 Slim Linux Container
# Runtime: Gunicorn WSGI Server + LiteRT (TFLite) Edge Inference Engine
# ==============================================================================

FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    PYTHONPATH=/app \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1

# Install system dependencies (curl for container healthcheck probe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user for edge security hardening
RUN groupadd -r grace && useradd -r -g grace -u 1000 -d /app grace

WORKDIR /app

# Copy dependency specification and install via pip
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy entrypoint script and make executable
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application source tree
COPY . /app/

# Create persistent storage directories and assign ownership to grace user
RUN mkdir -p /app/data /app/logs /app/models/checkpoints && \
    chown -R grace:grace /app

# Switch to non-root detection service user
USER grace

# Warm the SQLite schema and reference model registry before serving traffic
RUN python scripts/seed_demo_data.py || \
    echo "[GRACE Warning] Demo seeding skipped during image build."

EXPOSE 8000

# Automated container healthcheck probing the /health/ endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health/ || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

# Production Gunicorn worker execution dynamically bound to $PORT
# Single worker + 2 threads keeps the embedded stream-simulation thread and the
# LiteRT interpreter resident in one process (edge memory ceiling NFR3).
CMD ["sh", "-c", "exec gunicorn src.dashboard.app:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 2 --timeout 120"]
