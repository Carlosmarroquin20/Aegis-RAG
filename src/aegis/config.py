"""
Application-wide settings loaded from environment variables or .env file.
All security-relevant defaults are conservative (strict mode on, low rate limits).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import metadata

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _installed_version() -> str:
    """
    The version declared in pyproject.toml, read from the installed package.

    pyproject.toml is the single place the version is written. Reading it back
    from the distribution metadata means /health, the OpenAPI document and the
    tracing resource can never report a number that differs from what was
    actually built and shipped.
    """
    try:
        return metadata.version("aegis-rag")
    except metadata.PackageNotFoundError:
        # Source tree used without installing the project (PYTHONPATH=src).
        return "0.0.0+unknown"


# A key this long, drawn from a URL-safe alphabet, carries about 190 bits. The
# failed-auth throttle slows guessing down; key length is what makes guessing
# pointless, so short keys are reported at startup.
MIN_RECOMMENDED_API_KEY_CHARS = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # The same .env also feeds docker compose's variable interpolation, so it
        # legitimately holds keys this class does not model (AEGIS_PUBLIC_BIND).
        # pydantic-settings rejects unknown dotenv keys by default, which would
        # stop the app from starting over a variable it has no use for.
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "Aegis-RAG"
    app_version: str = Field(default_factory=_installed_version)
    debug: bool = False

    # ── Security ───────────────────────────────────────────────────────────────
    # Strict mode raises the block threshold: SUSPICIOUS queries are rejected.
    # Set False only in controlled evaluation environments.
    security_strict_mode: bool = True
    api_key_header: str = "X-API-Key"
    # Comma-separated list of valid API keys; in production, load from a secrets manager.
    valid_api_keys: str = Field(default="", description="Comma-separated API keys")
    # Path to a TOML file declaring tenants: each with its own key digests,
    # collection, rate limit and daily token budget (see
    # infrastructure/tenancy/registry.py). When set, VALID_API_KEYS is ignored;
    # when empty, every key in VALID_API_KEYS is a tenant on CHROMA_COLLECTION.
    tenants_file: str = ""

    # ── Failed-authentication throttling ──────────────────────────────────────
    # After this many rejected API keys from one client address inside the
    # window, every request from that address is answered 429 until the oldest
    # failure ages out. Valid keys are refused too during the lockout: letting
    # them through would leave the attacker a working right/wrong oracle.
    # 0 disables the throttle.
    auth_failure_limit: int = Field(default=10, ge=0)
    auth_failure_window_seconds: int = Field(default=60, ge=1)
    # How many reverse proxies in front of the API are trusted to report the
    # client address in X-Forwarded-For.
    #
    #   0  the API is reached directly: use the TCP peer address and ignore the
    #      header entirely, since any client can send whatever it likes in it.
    #   N  take the N-th address from the right, i.e. the one appended by the
    #      outermost trusted proxy. Everything to its left is client-supplied.
    #
    # The default is 0 because the two ways of getting this wrong are not equal.
    # Trusting the header with no proxy lets an attacker rotate a fake address
    # and never be throttled — silently. Not trusting it behind a proxy puts all
    # clients in one bucket, which is noisy, obvious and fixed by this setting.
    trusted_proxy_hops: int = Field(default=0, ge=0)

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
    # What to do when the rate-limit store cannot be reached. True (fail open)
    # serves the request unthrottled; False (fail closed) answers 503. Either way
    # the request has already been authenticated, and every occurrence is logged
    # and counted in aegis_rate_limit_backend_errors_total.
    #
    # Open is the default because the limiter is an abuse control, not an
    # authorization one: failing closed turns an outage of an auxiliary cache into
    # a full outage of the API for legitimate, authenticated callers. Choose
    # closed where unmetered access is the worse outcome (e.g. per-request cost).
    rate_limit_fail_open: bool = True

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

    # ── LLM Backend ────────────────────────────────────────────────────────────
    # "ollama" runs the model locally and keeps the deployment air-gapped.
    # "openai_compatible" calls a hosted provider (Groq, OpenRouter, Mistral, a
    # vLLM/LiteLLM gateway, ...). Choosing it means the query and the retrieved
    # document excerpts are sent to that provider; nothing else changes.
    llm_provider: str = "ollama"  # "ollama" | "openai_compatible"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: int = 120

    # OpenAI-compatible provider. All three are required when it is selected.
    # e.g. https://api.groq.com/openai/v1 — the adapter appends /chat/completions.
    llm_base_url: str = ""
    llm_model: str = ""
    # SecretStr keeps the value out of repr(), logs and validation error dumps.
    llm_api_key: SecretStr = SecretStr("")
    llm_timeout_seconds: int = Field(default=60, ge=1)
    # US dollars per million tokens charged by the hosted (openai_compatible)
    # provider, used only to estimate spend in aegis_llm_cost_usd_total.
    # Answers from the local Ollama backend are always costed at zero.
    llm_input_price_per_mtok: float = Field(default=0.0, ge=0)
    llm_output_price_per_mtok: float = Field(default=0.0, ge=0)

    # Fallback: a second backend that answers when the first is rate limited,
    # down or too slow. "" disables it. A hosted fallback behind the local
    # model sends data off the host whenever the local model fails.
    llm_fallback_provider: str = ""  # "" | "ollama" | "openai_compatible"
    # Consecutive failures that open a backend's circuit, and how long it is
    # then skipped before one trial request may close it again.
    llm_circuit_failure_threshold: int = Field(default=3, ge=1)
    llm_circuit_reset_seconds: float = Field(default=30.0, gt=0)

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"ollama", "openai_compatible"}
        lowered = v.lower()
        if lowered not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}")
        return lowered

    @field_validator("llm_fallback_provider")
    @classmethod
    def validate_llm_fallback_provider(cls, v: str) -> str:
        allowed = {"", "ollama", "openai_compatible"}
        lowered = v.strip().lower()
        if lowered not in allowed:
            raise ValueError(f"llm_fallback_provider must be one of {allowed}")
        return lowered

    @model_validator(mode="after")
    def require_hosted_llm_settings(self) -> Settings:
        """
        Fails at startup, not on the first query, when the hosted provider is
        selected, as the main backend or as the fallback, without the settings
        it cannot work without.
        """
        if self.llm_fallback_provider and self.llm_fallback_provider == self.llm_provider:
            raise ValueError("LLM_FALLBACK_PROVIDER must differ from LLM_PROVIDER")
        if "openai_compatible" not in (self.llm_provider, self.llm_fallback_provider):
            return self
        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", self.llm_base_url),
                ("LLM_MODEL", self.llm_model),
                ("LLM_API_KEY", self.llm_api_key.get_secret_value()),
            )
            if not value.strip()
        ]
        if missing:
            raise ValueError(
                f"the openai_compatible LLM provider requires {', '.join(missing)} to be set"
            )
        if not self.llm_base_url.lower().startswith("https://") and not self.debug:
            # The API key travels in a header on every request.
            raise ValueError("LLM_BASE_URL must use https:// outside debug mode")
        return self

    # ── Embeddings ─────────────────────────────────────────────────────────────
    # Multilingual: the English-only all-MiniLM-L6-v2 ranked Spanish documents
    # barely better than chance (scripts/evaluate_retrieval.py). Same 384
    # dimensions, but vectors from different models are not comparable: changing
    # this requires re-ingesting, and the adapter refuses a mismatched collection.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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
    def weak_api_key_count(self) -> int:
        """Number of configured keys shorter than MIN_RECOMMENDED_API_KEY_CHARS."""
        return sum(len(key) < MIN_RECOMMENDED_API_KEY_CHARS for key in self.api_keys_set)

    @property
    def api_keys_set(self) -> frozenset[str]:
        """Returns the valid API keys as an immutable set for O(1) lookup."""
        return frozenset(k.strip() for k in self.valid_api_keys.split(",") if k.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor — cached after first call to avoid repeated env parsing."""
    return Settings()
