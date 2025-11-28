"""
Observability module for logging, tracing, and metrics.

Provides:
- Custom tracing spans for operations
- Custom metrics for AI, transcription, and HTTP requests
- Integration with Azure Application Insights via OpenTelemetry
"""

from .tracing import (
    trace_operation,
    set_span_status,
    add_span_attribute,
    TRACING_AVAILABLE,
)

from .metrics import (
    record_ai_request,
    record_transcription_request,
    record_http_request,
    record_error,
    METRICS_AVAILABLE,
)

__all__ = [
    # Tracing
    "trace_operation",
    "set_span_status",
    "add_span_attribute",
    "TRACING_AVAILABLE",
    # Metrics
    "record_ai_request",
    "record_transcription_request",
    "record_http_request",
    "record_error",
    "METRICS_AVAILABLE",
]
