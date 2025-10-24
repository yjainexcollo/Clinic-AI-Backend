"""
Performance tracking middleware for monitoring request/response metrics
"""
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track performance metrics for all requests
    """
    
    async def dispatch(self, request: Request, call_next):
        # Track start time
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        process_time_ms = round(process_time * 1000, 2)
        
        # Log performance metrics
        logger.info(
            f"PERFORMANCE: method={request.method} path={request.url.path} "
            f"status={response.status_code} latency={process_time_ms}ms "
            f"request_id={getattr(request.state, 'request_id', 'unknown')}"
        )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time_ms)
        
        # Log slow requests (> 1 second)
        if process_time > 1.0:
            logger.warning(
                f"SLOW_REQUEST: method={request.method} path={request.url.path} "
                f"latency={process_time_ms}ms"
            )
        
        return response

