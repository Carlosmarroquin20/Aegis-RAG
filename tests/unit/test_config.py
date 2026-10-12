"""
Unit tests for application settings.

Settings are the first thing evaluated at import time; a mistake here stops the
process before any log line exists, so the failure modes are worth pinning down.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aegis.config import Settings


class TestDotenvFile:
    def test_keys_meant_for_docker_compose_do_not_break_startup(self, tmp_path: Path) -> None:
        """.env is shared with compose, which interpolates variables the app never reads."""
        env_file = tmp_path / ".env"
        env_file.write_text("AEGIS_PUBLIC_BIND=0.0.0.0\nVALID_API_KEYS=from-dotenv\n")

        settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

        assert settings.api_keys_set == frozenset({"from-dotenv"})


class TestValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rate_limit_backend", "memcached"),
            ("log_level", "VERBOSE"),
            ("otel_exporter", "zipkin"),
        ],
    )
    def test_unknown_choice_is_rejected_at_startup(self, field: str, value: str) -> None:
        with pytest.raises(ValidationError):
            Settings(**{field: value})

    def test_choices_are_normalized(self) -> None:
        settings = Settings(rate_limit_backend="REDIS", log_level="debug", otel_exporter="OTLP")
        assert settings.rate_limit_backend == "redis"
        assert settings.log_level == "DEBUG"
        assert settings.otel_exporter == "otlp"


class TestSecureDefaults:
    def test_defaults_are_conservative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("DEBUG", "VALID_API_KEYS", "SECURITY_STRICT_MODE", "TRACING_ENABLED"):
            monkeypatch.delenv(name, raising=False)
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.debug is False
        assert settings.security_strict_mode is True
        assert settings.tracing_enabled is False
        # No keys configured means nobody authenticates: closed, not open.
        assert settings.api_keys_set == frozenset()

    def test_api_keys_are_split_and_trimmed(self) -> None:
        settings = Settings(valid_api_keys=" alpha , beta,,gamma ")
        assert settings.api_keys_set == frozenset({"alpha", "beta", "gamma"})

    def test_cors_origins_are_split_and_trimmed(self) -> None:
        settings = Settings(CORS_ALLOWED_ORIGINS=" https://a.example ,https://b.example ")
        assert settings.cors_allowed_origins == ["https://a.example", "https://b.example"]


class TestVersion:
    def test_reported_version_is_the_one_declared_in_pyproject(self) -> None:
        """/health, the OpenAPI document and traces must all report what was built."""
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]

        assert Settings(_env_file=None).app_version == declared  # type: ignore[call-arg]

    def test_version_can_still_be_overridden_explicitly(self) -> None:
        assert Settings(app_version="9.9.9").app_version == "9.9.9"
