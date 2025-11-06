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
    Checks database, Azure services, and external dependencies.
    """
    import os
    from ...core.config import get_settings
    
    checks = {}
    all_ok = True
    
    # Check database connectivity
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        settings = get_settings()
        client = AsyncIOMotorClient(settings.database.uri, serverSelectionTimeoutMS=5000)
        await client.admin.command('ping')
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)[:50]}"
        all_ok = False
    
    # Check Azure Blob Storage
    try:
        from ...adapters.storage.azure_blob_service import get_azure_blob_service
        blob_service = get_azure_blob_service()
        # Quick check if client can be created
        _ = blob_service.client
        checks["azure_blob_storage"] = "ok"
    except Exception as e:
        checks["azure_blob_storage"] = f"error: {str(e)[:50]}"
        all_ok = False
    
    # Check Azure Key Vault
    try:
        from ...core.key_vault import get_key_vault_service
        key_vault = get_key_vault_service()
        if key_vault and key_vault.is_available:
            checks["azure_key_vault"] = "ok"
        else:
            checks["azure_key_vault"] = "not_configured"
    except Exception as e:
        checks["azure_key_vault"] = f"error: {str(e)[:50]}"
    
    # Check Application Insights
    app_insights_conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if app_insights_conn:
        checks["application_insights"] = "configured"
    else:
        checks["application_insights"] = "not_configured"
    
    # Check OpenAI (basic check - just verify key exists)
    settings = get_settings()
    if settings.openai.api_key:
        checks["openai"] = "configured"
    else:
        checks["openai"] = "not_configured"
        all_ok = False
    
    status = "ready" if all_ok else "degraded"

    return ok(request, data={
        "status": status,
        "timestamp": datetime.utcnow(),
        "checks": checks,
    }, message="OK" if all_ok else "Some services unavailable")


@router.get("/live", response_model=ApiResponse[dict])
async def liveness_check(request: Request):
    """
    Liveness check endpoint.

    Returns whether the service is alive.
    """
    return ok(request, data={"status": "alive", "timestamp": datetime.utcnow()}, message="OK")
