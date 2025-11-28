"""
Custom metrics for Clinic-AI using OpenTelemetry.

Provides custom metrics for AI operations, transcription, and application performance.
"""
import logging
from typing import Optional, Dict, Any
from time import time

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry metrics
try:
    from opentelemetry import metrics
    from opentelemetry.metrics import Counter, Histogram, UpDownCounter
    
    meter = metrics.get_meter(__name__)
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    meter = None
    logger.warning("OpenTelemetry metrics not available. Install 'opentelemetry-api' for metrics support.")


# Initialize custom metrics (lazy initialization)
_metrics_initialized = False
_ai_request_counter: Optional[Counter] = None
_ai_latency_histogram: Optional[Histogram] = None
_ai_token_counter: Optional[Counter] = None
_transcription_counter: Optional[Counter] = None
_transcription_latency_histogram: Optional[Histogram] = None
_request_counter: Optional[Counter] = None
_request_latency_histogram: Optional[Histogram] = None
_error_counter: Optional[Counter] = None


def _initialize_metrics():
    """Initialize custom metrics instruments."""
    global _metrics_initialized, _ai_request_counter, _ai_latency_histogram
    global _ai_token_counter, _transcription_counter, _transcription_latency_histogram
    global _request_counter, _request_latency_histogram, _error_counter
    
    if not METRICS_AVAILABLE or _metrics_initialized:
        return
    
    try:
        _ai_request_counter = meter.create_counter(
            name="clinicai.ai.requests",
            description="Total number of AI API requests",
            unit="1"
        )
        
        _ai_latency_histogram = meter.create_histogram(
            name="clinicai.ai.latency",
            description="AI API request latency in milliseconds",
            unit="ms"
        )
        
        _ai_token_counter = meter.create_counter(
            name="clinicai.ai.tokens",
            description="Total tokens used in AI requests",
            unit="1"
        )
        
        _transcription_counter = meter.create_counter(
            name="clinicai.transcription.requests",
            description="Total number of transcription requests",
            unit="1"
        )
        
        _transcription_latency_histogram = meter.create_histogram(
            name="clinicai.transcription.latency",
            description="Transcription request latency in seconds",
            unit="s"
        )
        
        _request_counter = meter.create_counter(
            name="clinicai.http.requests",
            description="Total HTTP requests",
            unit="1"
        )
        
        _request_latency_histogram = meter.create_histogram(
            name="clinicai.http.latency",
            description="HTTP request latency in milliseconds",
            unit="ms"
        )
        
        _error_counter = meter.create_counter(
            name="clinicai.errors",
            description="Total application errors",
            unit="1"
        )
        
        _metrics_initialized = True
        logger.info("Custom metrics initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize custom metrics: {e}")


def record_ai_request(model: str, latency_ms: float, tokens: int, success: bool = True):
    """
    Record an AI API request metric.
    
    Args:
        model: Model name (e.g., "gpt-4o-mini")
        latency_ms: Request latency in milliseconds
        tokens: Total tokens used
        success: Whether the request succeeded
    """
    if not METRICS_AVAILABLE:
        return
    
    try:
        _initialize_metrics()
        if _ai_request_counter:
            _ai_request_counter.add(1, {"model": model, "status": "success" if success else "error"})
        if _ai_latency_histogram:
            _ai_latency_histogram.record(latency_ms, {"model": model})
        if _ai_token_counter:
            _ai_token_counter.add(tokens, {"model": model})
        if not success and _error_counter:
            _error_counter.add(1, {"type": "ai_request", "model": model})
    except Exception as e:
        logger.warning(f"Failed to record AI request metric: {e}")


def record_transcription_request(latency_seconds: float, success: bool = True):
    """
    Record a transcription request metric.
    
    Args:
        latency_seconds: Transcription latency in seconds
        success: Whether the transcription succeeded
    """
    if not METRICS_AVAILABLE:
        return
    
    try:
        _initialize_metrics()
        if _transcription_counter:
            _transcription_counter.add(1, {"status": "success" if success else "error"})
        if _transcription_latency_histogram:
            _transcription_latency_histogram.record(latency_seconds)
        if not success and _error_counter:
            _error_counter.add(1, {"type": "transcription"})
    except Exception as e:
        logger.warning(f"Failed to record transcription metric: {e}")


def record_http_request(method: str, path: str, status_code: int, latency_ms: float):
    """
    Record an HTTP request metric.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        status_code: HTTP status code
        latency_ms: Request latency in milliseconds
    """
    if not METRICS_AVAILABLE:
        return
    
    try:
        _initialize_metrics()
        if _request_counter:
            _request_counter.add(1, {
                "method": method,
                "path": path,
                "status_code": str(status_code),
                "status": "success" if 200 <= status_code < 400 else "error"
            })
        if _request_latency_histogram:
            _request_latency_histogram.record(latency_ms, {
                "method": method,
                "path": path
            })
        if status_code >= 500 and _error_counter:
            _error_counter.add(1, {"type": "http_error", "status_code": str(status_code)})
    except Exception as e:
        logger.warning(f"Failed to record HTTP request metric: {e}")


def record_error(error_type: str, error_message: Optional[str] = None):
    """
    Record an application error.
    
    Args:
        error_type: Type of error (e.g., "database", "ai", "transcription")
        error_message: Optional error message
    """
    if not METRICS_AVAILABLE:
        return
    
    try:
        _initialize_metrics()
        if _error_counter:
            attributes = {"type": error_type}
            if error_message:
                attributes["message"] = error_message[:100]  # Truncate long messages
            _error_counter.add(1, attributes)
    except Exception as e:
        logger.warning(f"Failed to record error metric: {e}")
