# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# Aegis-RAG — public demo image (one container: nginx + API + embedded Chroma).
#
# Free hosting platforms such as Hugging Face Spaces run a single container, so
# this image folds the compose stack into one: the console is served by nginx,
# which forwards API paths to uvicorn on loopback, and Chroma runs in-process
# (CHROMA_MODE=embedded). The corpus is indexed at build time, so the container
# boots with its documents and never needs to write them.
#
# What changes compared with the compose deployment, and why:
#   - LLM: a hosted OpenAI-compatible provider (Groq by default) instead of
#     Ollama, because free CPU hardware cannot serve a local model at an
#     acceptable speed. Questions therefore leave the host; the console says so.
#   - Read-only corpus and one public key with a request limit and a daily
#     token budget (tenants.toml).
#   - No Prometheus, Grafana or Jaeger: /metrics is not reachable from outside.
#   - Retrieval is the default hybrid mode; the reranker, the faithfulness check
#     and the shadow injection classifier stay off, so no model beyond the
#     embedding one is baked and each answer costs no extra CPU on free hardware.
#
# Build from the repository root:
#   docker build -f deploy/huggingface/Dockerfile -t aegis-rag-demo .
# Run (needs an API key for the LLM provider):
#   docker run --rm -p 7860:7860 -e LLM_API_KEY=... aegis-rag-demo
# ─────────────────────────────────────────────────────────────────────────────

# Public on purpose: it ends up in the browser bundle. tenants.toml holds its
# digest, and tests/unit/test_demo_deployment.py checks that the two match.
ARG DEMO_API_KEY=aegis-public-demo-key
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small

# ── Stage 1: console bundle ───────────────────────────────────────────────────
FROM node:26-alpine@sha256:40cd011b55415bc894e45b02fdd5a5dfde056921b7d23a64c74f80497af48797 AS console

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/vite.config.ts frontend/tsconfig.json \
     frontend/tailwind.config.js frontend/postcss.config.js ./
COPY frontend/src ./src
ARG DEMO_API_KEY
RUN VITE_DEMO_API_KEY="${DEMO_API_KEY}" npm run build

# ── Stage 2: Python base ──────────────────────────────────────────────────────
FROM python:3.11-slim@sha256:65bd3d9f1d7fa268dd1909973ba7eb4bdf41f6dcf2cfbe76ba9528b170f3fd79 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM ghcr.io/astral-sh/uv:0.12.9@sha256:8b940d3a9d65bed080436972241af2e21c84b5e8c9193f7014ed71479ee795ff AS uv

# ── Stage 3: dependencies, embedding model and the indexed corpus ─────────────
FROM base AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./README.md
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Baked like the production image: the container never fetches the model.
ARG EMBEDDING_MODEL
ENV HF_HOME=/app/hf-cache \
    HF_HUB_DISABLE_XET=1
RUN /app/.venv/bin/python -c "from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as F; F(model_name='${EMBEDDING_MODEL}', device='cpu')"

# Index the corpus through the same pipeline the upload endpoint uses, into the
# collection of the demo tenant. Doing it here, not at startup, keeps boot fast
# and makes the shipped index part of the reviewed, reproducible build.
COPY scripts/seed_documents.py ./scripts/seed_documents.py
COPY examples/atencion-cliente ./corpus
COPY deploy/huggingface/tenants.toml ./deploy/tenants.toml
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    CHROMA_MODE=embedded CHROMA_PATH=/app/chroma-data \
    TENANTS_FILE=/app/deploy/tenants.toml EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
    PYTHONPATH=/app/src /app/.venv/bin/python scripts/seed_documents.py \
        --source-dir ./corpus --tenant demo

# ── Stage 4: runtime ──────────────────────────────────────────────────────────
FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled /etc/nginx/conf.d

# Hugging Face Spaces runs the container as UID 1000; the user is created with
# that UID so the files below are owned by whoever the platform runs as.
RUN groupadd --gid 1000 aegis && \
    useradd --uid 1000 --gid aegis --shell /bin/bash --create-home aegis

WORKDIR /app

COPY --from=builder --chown=aegis:aegis /app/.venv /app/.venv
COPY --from=builder --chown=aegis:aegis /app/src /app/src
COPY --from=builder --chown=aegis:aegis /app/hf-cache /app/hf-cache
# Chroma opens its SQLite file read-write even to serve queries.
COPY --from=builder --chown=aegis:aegis /app/chroma-data /app/chroma-data
COPY --from=builder --chown=aegis:aegis /app/deploy/tenants.toml /app/deploy/tenants.toml
COPY --from=console /app/dist /usr/share/nginx/html
COPY deploy/huggingface/nginx.conf /etc/nginx/nginx.conf
COPY deploy/huggingface/aegis-proxy.conf /etc/nginx/aegis-proxy.conf
COPY --chmod=0755 deploy/huggingface/start.sh /app/start.sh

USER aegis

# Every setting below can be overridden by the platform (Space variables and
# secrets are passed as environment variables). LLM_API_KEY has no default:
# the API refuses to start without it, which is the intended failure.
ARG EMBEDDING_MODEL
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    HF_HOME="/app/hf-cache" \
    HF_HUB_OFFLINE="1" \
    TRANSFORMERS_OFFLINE="1" \
    EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
    LOG_FORMAT="json" \
    LOG_LEVEL="INFO" \
    CHROMA_MODE="embedded" \
    CHROMA_PATH="/app/chroma-data" \
    TENANTS_FILE="/app/deploy/tenants.toml" \
    DOCUMENT_WRITES_ENABLED="false" \
    AUTH_FAILURE_LIMIT="0" \
    RATE_LIMIT_BACKEND="memory" \
    LLM_PROVIDER="openai_compatible" \
    LLM_BASE_URL="https://api.groq.com/openai/v1" \
    LLM_MODEL="openai/gpt-oss-20b"

# AUTH_FAILURE_LIMIT=0: the only valid key is public, so the throttle would
# protect nothing, and every visitor reaches the API from nginx on loopback,
# which would let one caller sending bad keys lock everyone else out.

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS -o /dev/null http://127.0.0.1:7860/health || exit 1

CMD ["/app/start.sh"]
