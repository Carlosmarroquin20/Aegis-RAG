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

# Base images are pinned by digest as well as tag: the tag says what it is, the
# digest makes the build reproducible and immune to a re-pushed tag. Dependabot
# updates both together, which it cannot do through a build ARG, so the
# references are spelled out in each FROM.

# ── Stage 1: Base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim@sha256:65bd3d9f1d7fa268dd1909973ba7eb4bdf41f6dcf2cfbe76ba9528b170f3fd79 AS base

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
# The uv binary comes from a named stage, copied from below. Its version must be
# recent enough to read the committed uv.lock format (lockfile version 1 /
# revision 3) and match UV_VERSION in the CI workflow; an older uv rejects the
# lockfile under --frozen.
FROM ghcr.io/astral-sh/uv:0.12.9@sha256:8b940d3a9d65bed080436972241af2e21c84b5e8c9193f7014ed71479ee795ff AS uv

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
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ENV HF_HOME=/app/hf-cache \
    HF_HUB_DISABLE_XET=1
RUN /app/.venv/bin/python -c "from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as F; F(model_name='${EMBEDDING_MODEL}', device='cpu')"

# The cross-encoder for the RETRIEVAL_MODE=*+rerank modes, baked for the same
# reason when one is named. Off by default, so the image stays about 470 MB
# smaller: build with
#   --build-arg RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
# to rerank (docs/adr/0016-hybrid-retrieval-by-default-reranker-opt-in.md).
ARG RERANKER_MODEL=
RUN if [ -n "${RERANKER_MODEL}" ]; then \
      /app/.venv/bin/python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${RERANKER_MODEL}', device='cpu')"; \
    fi

# The NLI model behind FAITHFULNESS_CHECK, likewise baked only when named
# (docs/adr/0017-answer-faithfulness-judged-by-nli.md):
#   --build-arg FAITHFULNESS_MODEL=MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli
ARG FAITHFULNESS_MODEL=
RUN if [ -n "${FAITHFULNESS_MODEL}" ]; then \
      /app/.venv/bin/python -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; T.from_pretrained('${FAITHFULNESS_MODEL}'); M.from_pretrained('${FAITHFULNESS_MODEL}')"; \
    fi

# The injection classifier for INJECTION_CLASSIFIER_MODE=shadow, likewise baked
# only when named (docs/adr/0020-injection-classifier-in-shadow-mode.md):
#   --build-arg INJECTION_CLASSIFIER_MODEL=protectai/deberta-v3-base-prompt-injection-v2
ARG INJECTION_CLASSIFIER_MODEL=
RUN if [ -n "${INJECTION_CLASSIFIER_MODEL}" ]; then \
      /app/.venv/bin/python -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; T.from_pretrained('${INJECTION_CLASSIFIER_MODEL}'); M.from_pretrained('${INJECTION_CLASSIFIER_MODEL}')"; \
    fi

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

# Healthcheck for container orchestrators. Startup loads the embedding model
# into memory: about 45 s warm and close to two minutes cold on a laptop's
# Docker VM with the multilingual model. Failures inside the start period do
# not count, and the first success ends it, so a generous period costs nothing
# when the container starts quickly.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
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
