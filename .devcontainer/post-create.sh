#!/bin/bash
# Post-create hook for the Dev Container.
# Installs dev dependencies and sets up pre-commit hooks.
set -euo pipefail

echo "→ Installing dev dependencies with uv..."
uv sync --dev

echo "→ Installing pre-commit hooks..."
uv run pre-commit install

echo "→ Dev environment ready. Run: uv run pytest"
