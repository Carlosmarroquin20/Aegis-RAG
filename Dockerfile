# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# Aegis-RAG — Multi-stage production Dockerfile
#
# Stage layout:
#   base     → minimal Python 3.11 slim image with system deps
#   builder  → installs Python packages into an isolated virtualenv using uv
#   runtime  → copies only the virtualenv and application code
#
# Security practices applied:
#   - Non-root user in runtime stage (UID 1001)
#   - No build tools (gcc, pip, uv) in the final image
#   - Explicit COPY instead of ADD to avoid tar extraction vulnerabilities
#   - Read-only filesystem compatible (all writes go to /tmp or mounted volumes)
# ─────────────────────────────────────────────────────────────────────────────

# UV_VERSION must be recent enough to read the committed uv.lock format
# (lockfile version 1 / revision 3). Keep it aligned with the uv used to
# regenerate the lock; an older uv will reject the lockfile under --frozen.
ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.12.9

# ── Stage 1: Base ─────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Install only OS-level runtime deps (no build tools).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── uv provider ───────────────────────────────────────────────────────────────
# Pin the uv binary via a named stage. BuildKit does not expand build ARGs inside
# a COPY --from image reference, but it does in a FROM (where global ARGs are in
# scope), so alias the image here and COPY from the alias below.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ── Stage 2: Builder ──────────────────────────────────────────────────────────
FROM base AS builder

# Install uv for fast, reproducible dependency resolution.
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock* ./

# Install production dependencies only into an isolated venv.
# --frozen ensures the lockfile is respected; --no-dev excludes test tools; the
# redis + otel extras bundle the optional Redis rate-limiter backend and the
# OpenTelemetry stack so both can be enabled at runtime (RATE_LIMIT_BACKEND /
# TRACING_ENABLED) without rebuilding.
RUN uv sync --frozen --no-dev --extra redis --extra otel --no-install-project

# Copy application source after deps to maximize layer cache hits.
# README.md is required because pyproject sets `readme = "README.md"`; hatchling
# reads it while building the project wheel, so the build fails without it.
COPY README.md ./README.md
COPY src/ ./src/

# Install the project itself (editable=false for production).
RUN uv sync --frozen --no-dev --extra redis --extra otel

# ── Pre-bake the embedding model ──────────────────────────────────────────────
# The API loads a SentenceTransformers model at startup. Fetching it from the
# HuggingFace Hub on first boot breaks the project's air-gap guarantee, makes
# startup slow and non-deterministic, and fails whenever the Hub — or its Xet CAS
# backend, which is prone to stalling — is unreachable. Materialize it into the
# image now, via the exact code path used at runtime, so the container boots
# fully offline. HF_HUB_DISABLE_XET forces plain HTTPS and sidesteps the stall.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV HF_HOME=/app/hf-cache \
    HF_HUB_DISABLE_XET=1
RUN /app/.venv/bin/python -c "from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as F; F(model_name='${EMBEDDING_MODEL}', device='cpu')"

# ── Stage 3: Runtime ──────────────────────────────────────────────────────────
FROM base AS runtime

# Non-root user: prevents container escape escalation.
RUN groupadd --gid 1001 aegis && \
    useradd --uid 1001 --gid aegis --shell /bin/bash --create-home aegis

WORKDIR /app

# Copy the pre-built virtualenv, source, and the pre-baked model cache; do NOT
# copy build tools.
COPY --from=builder --chown=aegis:aegis /app/.venv /app/.venv
COPY --from=builder --chown=aegis:aegis /app/src /app/src
COPY --from=builder --chown=aegis:aegis /app/hf-cache /app/hf-cache

USER aegis

# HF_HOME points at the baked cache; the OFFLINE flags make huggingface_hub and
# transformers load the embedding model straight from it and never touch the
# network — the air-gap guarantee holds and startup is fast and deterministic.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    LOG_FORMAT="json" \
    LOG_LEVEL="INFO" \
    HF_HOME="/app/hf-cache" \
    HF_HUB_OFFLINE="1" \
    TRANSFORMERS_OFFLINE="1"

EXPOSE 8000

# Healthcheck for container orchestrators.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# structlog owns application logging (configured at startup); the AccessLog
# middleware emits the structured per-request line, so uvicorn's own access log
# is suppressed. Do NOT pass `--log-config /dev/null`: uvicorn feeds it to
# logging.config.fileConfig, which raises "is an empty file" and crash-loops.
CMD ["uvicorn", "aegis.interface.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log"]
