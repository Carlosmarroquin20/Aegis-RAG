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

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins for production (e.g., "https://app.example.com").
    # Ignored when DEBUG=true (all origins are allowed in debug mode).
    cors_allowed_origins_str: str = Field(
        default="",
        alias="CORS_ALLOWED_ORIGINS",
        description="Comma-separated allowed origins for CORS",
    )

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Returns the allowed origins as a list for CORSMiddleware."""
        return [o.strip() for o in self.cors_allowed_origins_str.split(",") if o.strip()]

    # ── Rate Limiting ──────────────────────────────────────────────────────────
    rate_limit_requests: int = 60  # requests per window
    rate_limit_window_seconds: int = 60
    rate_limit_burst: int = 10  # burst headroom above baseline
    # Backend for the sliding-window store. "memory" is per-process (single worker
    # only); "redis" shares the window across workers/replicas. The redis backend
    # requires the optional dependency: pip install "aegis-rag[redis]".
    rate_limit_backend: str = "memory"  # "memory" | "redis"
    redis_url: str = "redis://localhost:6379/0"

    @field_validator("rate_limit_backend")
    @classmethod
    def validate_rate_limit_backend(cls, v: str) -> str:
        allowed = {"memory", "redis"}
        lowered = v.lower()
        if lowered not in allowed:
            raise ValueError(f"rate_limit_backend must be one of {allowed}")
        return lowered

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

    # ── Tracing (OpenTelemetry) ─────────────────────────────────────────────────
    # Off by default (air-gap friendly). When enabled, the SDK + exporters +
    # instrumentation are required: pip install "aegis-rag[otel]".
    tracing_enabled: bool = False
    # Where spans go: "console" prints them (dev, no backend needed); "otlp" ships
    # them over OTLP/HTTP to a collector/Jaeger/Tempo at otlp_endpoint.
    otel_exporter: str = "console"  # "console" | "otlp"
    otlp_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "aegis-rag"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @field_validator("otel_exporter")
    @classmethod
    def validate_otel_exporter(cls, v: str) -> str:
        allowed = {"console", "otlp"}
        lowered = v.lower()
        if lowered not in allowed:
            raise ValueError(f"otel_exporter must be one of {allowed}")
        return lowered

    @property
    def api_keys_set(self) -> frozenset[str]:
        """Returns the valid API keys as an immutable set for O(1) lookup."""
        return frozenset(k.strip() for k in self.valid_api_keys.split(",") if k.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor — cached after first call to avoid repeated env parsing."""
    return Settings()
