# syntax=docker/dockerfile:1

# --- Stage 1: build a virtualenv with CPU-only torch -----------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the CPU-only torch wheel first so sentence-transformers doesn't pull
# the multi-gigabyte CUDA build, then the rest of the requirements.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# --- Stage 2: runtime ------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    PORT=8000

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . .

# Bake the offline corpus and both indices into the image so the container is
# queryable on first boot with no runtime downloads. This also caches the
# embedding model weights into the image layer.
RUN python build_index.py --offline

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health')"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
