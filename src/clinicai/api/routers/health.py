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
    
    # Check Azure Blob Storage connectivity
    try:
        from ...adapters.storage.azure_blob_service import get_azure_blob_service
        import asyncio
        blob_service = get_azure_blob_service()
        # Quick check if client can be created
        client = blob_service.client
        
        # Test actual connectivity by getting account info (with timeout)
        async def test_connectivity():
            try:
                from ...adapters.storage.azure_blob_service import run_blocking
                # Try to get account properties (lightweight operation)
                await asyncio.wait_for(
                    run_blocking(
                        lambda: client.get_account_information()
                    ),
                    timeout=10.0  # 10 second timeout for health check
                )
                return True
            except Exception as e:
                import logging
                logger = logging.getLogger("clinicai")
                logger.error(f"Blob storage connectivity test failed: {e}")
                return False
        
        connectivity_ok = await test_connectivity()
        if connectivity_ok:
            checks["azure_blob_storage"] = "ok"
        else:
            checks["azure_blob_storage"] = "connectivity_failed"
            all_ok = False
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
    
    # Check Azure OpenAI (required - no fallback)
    settings = get_settings()
    azure_openai_configured = (
        settings.azure_openai.endpoint and 
        settings.azure_openai.api_key
    )
    
    if azure_openai_configured:
        # Check Azure OpenAI configuration
        try:
            # Verify deployment names are configured
            if settings.azure_openai.deployment_name:
                checks["azure_openai"] = "configured"
                checks["azure_openai_chat_deployment"] = settings.azure_openai.deployment_name
                checks["azure_openai_api_version"] = settings.azure_openai.api_version
            else:
                checks["azure_openai"] = "partially_configured"
                all_ok = False
        except Exception as e:
            checks["azure_openai"] = f"error: {str(e)[:50]}"
            all_ok = False
    else:
        # Azure OpenAI is required - no fallback
        checks["azure_openai"] = "not_configured"
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


@router.get("/audit", response_model=ApiResponse[dict])
async def audit_health_check(request: Request):
    """
    HIPAA audit log system health check endpoint.

    Tests audit log write capability and integrity verification.
    Returns the health status of the HIPAA audit logging system.
    """
    from ...core.hipaa_audit import get_audit_logger
    
    try:
        audit_logger = get_audit_logger()
        
        # Check if audit logger is initialized (use hasattr to avoid AttributeError)
        if not hasattr(audit_logger, '_initialized') or not audit_logger._initialized:
            return ok(request, data={
                "status": "unhealthy",
                "error": "Audit logger not initialized",
                "timestamp": datetime.utcnow()
            }, message="Audit log system not initialized")
        
        # Test audit log write
        test_audit_id = await audit_logger.log_phi_access(
            user_id="health_check",
            action="GET",
            resource_type="health_check",
            resource_id="test",
            patient_id=None,
            ip_address="127.0.0.1",
            user_agent="health-check",
            phi_fields=[],
            phi_accessed=False,
            success=True,
            details={"purpose": "health_check"}
        )
        
        # Verify integrity
        integrity_ok = await audit_logger.verify_audit_integrity(test_audit_id)
        
        # Get audit trail to verify read capability
        audit_trail = await audit_logger.get_audit_trail(
            user_id="health_check",
            limit=1
        )
        
        read_ok = len(audit_trail) > 0
        
        status = "healthy" if (integrity_ok and read_ok) else "degraded"
        
        return ok(request, data={
            "status": status,
            "integrity_check": integrity_ok,
            "read_check": read_ok,
            "test_audit_id": test_audit_id,
            "timestamp": datetime.utcnow()
        }, message="OK" if status == "healthy" else "Audit log system degraded")
        
    except Exception as e:
        return ok(request, data={
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }, message="Audit log system unavailable")
