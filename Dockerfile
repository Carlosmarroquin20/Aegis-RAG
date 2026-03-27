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

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.5.0

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

# ── Stage 2: Builder ──────────────────────────────────────────────────────────
FROM base AS builder

# Install uv for fast, reproducible dependency resolution.
COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock* ./

# Install production dependencies only into an isolated venv.
# --frozen ensures the lockfile is respected; --no-dev excludes test tools.
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source after deps to maximize layer cache hits.
COPY src/ ./src/

# Install the project itself (editable=false for production).
RUN uv sync --frozen --no-dev

# ── Stage 3: Runtime ──────────────────────────────────────────────────────────
FROM base AS runtime

# Non-root user: prevents container escape escalation.
RUN groupadd --gid 1001 aegis && \
    useradd --uid 1001 --gid aegis --shell /bin/bash --create-home aegis

WORKDIR /app

# Copy the pre-built virtualenv and source; do NOT copy build tools.
COPY --from=builder --chown=aegis:aegis /app/.venv /app/.venv
COPY --from=builder --chown=aegis:aegis /app/src /app/src

USER aegis

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    LOG_FORMAT="json" \
    LOG_LEVEL="INFO"

EXPOSE 8000

# Healthcheck for container orchestrators.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "aegis.interface.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-config", "/dev/null"]
