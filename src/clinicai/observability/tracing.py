"""
OpenTelemetry tracing configuration for Clinic-AI.

Provides custom tracing spans for key operations like AI calls, transcription, and database operations.
"""
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry tracing
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    
    tracer = trace.get_tracer(__name__)
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    tracer = None
    logger.warning("OpenTelemetry tracing not available. Install 'opentelemetry-api' for tracing support.")


@contextmanager
def trace_operation(operation_name: str, attributes: Optional[dict] = None):
    """
    Context manager for creating custom tracing spans.
    
    Args:
        operation_name: Name of the operation being traced
        attributes: Optional dictionary of attributes to add to the span
    
    Example:
        with trace_operation("ai_chat_completion", {"model": "gpt-4o-mini"}):
            response = await client.chat(...)
    """
    if not TRACING_AVAILABLE:
        yield
        return
    
    try:
        with tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span
    except Exception as e:
        logger.warning(f"Tracing error for {operation_name}: {e}")
        yield None


def set_span_status(span, success: bool, error_message: Optional[str] = None):
    """
    Set the status of a tracing span.
    
    Args:
        span: OpenTelemetry span object
        success: Whether the operation succeeded
        error_message: Optional error message if operation failed
    """
    if not TRACING_AVAILABLE or not span:
        return
    
    try:
        if success:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR, error_message or "Operation failed"))
    except Exception as e:
        logger.warning(f"Failed to set span status: {e}")


def add_span_attribute(span, key: str, value: any):
    """
    Add an attribute to a tracing span.
    
    Args:
        span: OpenTelemetry span object
        key: Attribute key
        value: Attribute value (will be converted to string)
    """
    if not TRACING_AVAILABLE or not span:
        return
    
    try:
        span.set_attribute(key, str(value))
    except Exception as e:
        logger.warning(f"Failed to add span attribute {key}: {e}")
