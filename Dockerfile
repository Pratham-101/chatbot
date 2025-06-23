# Stage 1: Build stage with dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set cache directories for models
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

# Install dependencies
COPY requirements-prod.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Install google-cloud-storage
RUN pip install --no-cache-dir google-cloud-storage

# Run the data ingestion process to create the vector store
# COPY ./ingestion /app/ingestion
# COPY ./data /app/data
# COPY ./populate_vector_store.py /app/
# RUN python populate_vector_store.py

# Stage 2: Final production image
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user
RUN addgroup --system app && adduser --system --group app

# Set cache directories for models
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

# Copy Python executable and installed dependencies from the builder stage
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy pre-downloaded models
COPY --from=builder /app/.cache /app/.cache

# Copy application code
COPY ./main.py /app/main.py
COPY ./chatbot /app/chatbot
COPY download_vector_store.py /app/download_vector_store.py

# Ensure the app user owns the files
RUN chown -R app:app /app

# Switch to the non-root user
USER app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the download script before starting the API server
CMD ["sh", "-c", "python download_vector_store.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
