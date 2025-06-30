# Use Python 3.11 slim image as base
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements/ requirements/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/production.txt

# Copy application code
COPY src/ src/
COPY pyproject.toml ./

# Install the application
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/uploads

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Default command
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]


# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements/development.txt

# Switch back to root for development
USER root

# Create development directories
RUN mkdir -p /app/tests /app/docs

# Copy development files
COPY tests/ /app/tests/
COPY docs/ /app/docs/
COPY .pre-commit-config.yaml /app/
COPY Makefile /app/

# Switch back to appuser
USER appuser

# Development command
CMD ["uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]


# Production stage with additional security
FROM base as production

# Install additional security packages
USER root
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create security directory
RUN mkdir -p /app/security && chown -R appuser:appuser /app/security

# Switch back to appuser
USER appuser

# Production command with additional security headers
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "--max-requests", "1000", "--max-requests-jitter", "100"] 