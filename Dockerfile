# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: install Python dependencies into an isolated virtual environment
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# CPU-only PyTorch keeps the image roughly 2 GB smaller than the default CUDA build.
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the embedding model into the image: faster first upload, no runtime download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ---------------------------------------------------------------------------
# Runtime: slim image, non-root user (UID 1000 also matches Hugging Face Spaces)
# ---------------------------------------------------------------------------
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app \
 && mkdir -p /data && chown app:app /data

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False \
    DATA_DIR=/tmp/docugenius \
    PORT=7860

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/hf-cache /opt/hf-cache

WORKDIR /app
COPY --chown=app:app docugenius ./docugenius
COPY --chown=app:app frontend ./frontend

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "exec uvicorn docugenius.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
