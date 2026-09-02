"""
OpenTelemetry tracing setup — the third pillar (traces) alongside logs and metrics.

Design:
  - ``opentelemetry-api`` is a lightweight base dependency, so instrumentation code
    (spans in the use cases, the structlog trace-context processor) can be written
    unconditionally. Without a configured SDK provider those calls are cheap no-ops,
    which keeps tracing genuinely opt-in and air-gap friendly.
  - The heavy pieces — the SDK, the OTLP/HTTP exporter, and the FastAPI/HTTPX
    auto-instrumentation — live in the optional ``otel`` extra and are imported
    lazily inside ``configure_tracing`` / ``instrument_app``. They run only when
    ``tracing_enabled`` is set.

Correlation:
  - ``add_trace_context`` injects ``trace_id``/``span_id`` into every structlog
    line, so a JSON log can be pivoted to its trace and vice versa.
  - The FastAPI server span wraps the whole request; the query use case adds child
    spans for each pipeline stage; HTTPX instrumentation traces the Ollama call.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

if TYPE_CHECKING:
    from fastapi import FastAPI

    from aegis.config import Settings

# Single import point for span creation elsewhere in the app.
tracer = trace.get_tracer("aegis")


def add_trace_context(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """
    structlog processor that adds the active trace/span IDs to the log event.

    No-op safe: when no span is recording (tracing disabled, or outside a request)
    the current span context is invalid and nothing is added.
    """
    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        event_dict["trace_id"] = format(span_context.trace_id, "032x")
        event_dict["span_id"] = format(span_context.span_id, "016x")
    return event_dict


def configure_tracing(settings: Settings) -> None:
    """
    Install a global TracerProvider with the configured exporter.

    Imported lazily: requires the ``otel`` extra
    (``pip install "aegis-rag[otel]"``). Call once, at startup, only when
    ``settings.tracing_enabled`` is true.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": settings.app_version,
        }
    )
    provider = TracerProvider(resource=resource)

    exporter: Any
    if settings.otel_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=f"{settings.otlp_endpoint.rstrip('/')}/v1/traces")
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def instrument_app(app: FastAPI) -> None:
    """
    Auto-instrument the FastAPI app (server spans) and the HTTPX client (the
    Ollama call). Lazily imported; requires the ``otel`` extra.
    """
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


def shutdown_tracing() -> None:
    """Flush and shut down the provider so batched spans are not lost on exit."""
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()
