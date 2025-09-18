"""
Health check endpoints.
"""

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    timestamp: datetime
    version: str
    service: str


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the current status of the service.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        service="Clinic-AI Intake Assistant",
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.

    Returns whether the service is ready to handle requests.
    """
    # In a real implementation, you would check:
    # - Database connectivity
    # - External service availability
    # - Required resources

    return {
        "status": "ready",
        "timestamp": datetime.utcnow(),
        "checks": {"database": "ok", "openai": "ok", "memory": "ok"},
    }


@router.get("/live")
async def liveness_check():
    """
    Liveness check endpoint.

    Returns whether the service is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow()}
