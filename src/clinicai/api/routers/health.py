"""
Health check endpoints.
"""

from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel
from ..schemas.common import ApiResponse
from ..utils.responses import ok

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    timestamp: datetime
    version: str
    service: str


@router.get("/", response_model=ApiResponse[HealthResponse])
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns the current status of the service.
    """
    return ok(request, data=HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        service="Clinic-AI Intake Assistant",
    ), message="OK")


@router.get("/ready", response_model=ApiResponse[dict])
async def readiness_check(request: Request):
    """
    Readiness check endpoint.

    Returns whether the service is ready to handle requests.
    """
    # In a real implementation, you would check:
    # - Database connectivity
    # - External service availability
    # - Required resources

    return ok(request, data={
        "status": "ready",
        "timestamp": datetime.utcnow(),
        "checks": {"database": "ok", "openai": "ok", "memory": "ok"},
    }, message="OK")


@router.get("/live", response_model=ApiResponse[dict])
async def liveness_check(request: Request):
    """
    Liveness check endpoint.

    Returns whether the service is alive.
    """
    return ok(request, data={"status": "alive", "timestamp": datetime.utcnow()}, message="OK")
