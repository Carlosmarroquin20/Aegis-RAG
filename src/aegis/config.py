"""
Application-wide settings loaded from environment variables or .env file.
All security-relevant defaults are conservative (strict mode on, low rate limits).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "Aegis-RAG"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Security ───────────────────────────────────────────────────────────────
    # Strict mode raises the block threshold: SUSPICIOUS queries are rejected.
    # Set False only in controlled evaluation environments.
    security_strict_mode: bool = True
    api_key_header: str = "X-API-Key"
    # Comma-separated list of valid API keys; in production, load from a secrets manager.
    valid_api_keys: str = Field(default="", description="Comma-separated API keys")

    # ── Rate Limiting ──────────────────────────────────────────────────────────
    rate_limit_requests: int = 60  # requests per window
    rate_limit_window_seconds: int = 60
    rate_limit_burst: int = 10  # burst headroom above baseline

    # ── Vector Store (ChromaDB) ────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "aegis_documents"

    # ── LLM Backend (Ollama) ───────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: int = 120

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # "cuda" for GPU inference

    # ── Observability ──────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @property
    def api_keys_set(self) -> frozenset[str]:
        """Returns the valid API keys as an immutable set for O(1) lookup."""
        return frozenset(k.strip() for k in self.valid_api_keys.split(",") if k.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor — cached after first call to avoid repeated env parsing."""
    return Settings()
