# ─── Go Code Reviewer — Multi-stage Docker Build ───────────────────────────
#
# Usage:
#   docker build -t go-code-reviewer .
#   docker run -p 8000:8000 --gpus all \
#     -v ./models:/app/models \
#     -v ./rag/qdrant_db:/app/rag/qdrant_db \
#     go-code-reviewer
#
# Environment variables:
#   MODEL_PATH     — Path to fine-tuned model (default: ./models/go-reviewer)
#   BASE_MODEL     — HF model ID (default: deepseek-ai/deepseek-coder-6.7b-instruct)
#   RAG_DB_PATH    — Path to Qdrant DB (default: ./rag/qdrant_db)
#   REVIEW_MODE    — hybrid | rag-only | fine-tune-only (default: hybrid)
#   OLLAMA_MODEL   — Ollama model name (optional, for Ollama backend)
#   OLLAMA_URL     — Ollama API URL (default: http://localhost:11434)
#   PORT           — API port (default: 8000)
# ────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Base with CUDA ──────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    golang-go \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# ── Stage 2: Dependencies ───────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Application ────────────────────────────────────────────────────
FROM deps AS app

# Copy application code
COPY standards/ ./standards/
COPY dataset/ ./dataset/
COPY training/ ./training/
COPY rag/ ./rag/
COPY pipeline/ ./pipeline/
COPY serving/ ./serving/
COPY cli/ ./cli/

# Create directories for models and data
RUN mkdir -p /app/models /app/rag/qdrant_db /app/output

# Default environment
ENV MODEL_PATH=/app/models/go-reviewer
ENV BASE_MODEL=deepseek-ai/deepseek-coder-6.7b-instruct
ENV RAG_DB_PATH=/app/rag/qdrant_db
ENV REVIEW_MODE=hybrid
ENV PORT=8000
ENV HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

# Run API server
CMD ["python", "-m", "uvicorn", "serving.api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
