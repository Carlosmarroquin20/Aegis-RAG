# ─────────────────────────────────────────────────────────────────────────────
# Aegis-RAG frontend — multi-stage build.
#   builder → installs deps and produces the static bundle with Vite
#   runtime → serves the bundle with nginx and proxies /api to the backend
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first for better layer caching.
# `npm ci` only: it installs exactly what package-lock.json pins and fails if the
# lockfile is missing or out of sync. Falling back to `npm install` would quietly
# resolve fresh versions and ship a bundle that was never tested.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── Stage 2: Serve ────────────────────────────────────────────────────────────
FROM nginx:1.31-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

# Probe over IPv4 explicitly: nginx only `listen 80;` (IPv4), but "localhost"
# resolves to IPv6 [::1] first in this image, so wget to localhost is refused
# and the container would never report healthy.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null 2>&1 || exit 1
