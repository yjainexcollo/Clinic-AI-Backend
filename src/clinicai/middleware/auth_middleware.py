"""
Authentication middleware - validates authentication before request processing.

This middleware enforces authentication on all PHI endpoints for HIPAA compliance.
Public endpoints (health checks, docs) are excluded from authentication requirements.
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ..core.auth import get_auth_service
import logging

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce authentication on PHI endpoints.
    
    Public endpoints (health checks, API docs) are excluded.
    All other endpoints require valid authentication.
    """
    
    PUBLIC_PATHS = {
        "/",
        "/favicon.ico",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/health/live",
        "/health/ready",
    }
    
    PUBLIC_PATH_PREFIXES = {
        "/docs",
        "/redoc",
    }
    
    def is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public and doesn't require authentication."""
        normalized_path = path.rstrip("/") or "/"
        
        if normalized_path in self.PUBLIC_PATHS:
            return True
        
        for prefix in self.PUBLIC_PATH_PREFIXES:
            if path.startswith(prefix + "/"):
                return True
        
        return False
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and enforce authentication for non-public endpoints.
        
        For public endpoints: Allow access without authentication
        For PHI endpoints: Require valid authentication
        """
        # Skip authentication for public endpoints
        if self.is_public_endpoint(request.url.path):
            logger.debug(f"Public endpoint accessed: {request.url.path}")
            return await call_next(request)
        
        # Require authentication for all other endpoints (PHI access)
        auth_service = get_auth_service()
        
        # Extract authentication credentials from headers
        api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
        auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
        
        try:
            # Validate authentication and get user ID
            user_id = auth_service.get_user_from_request(
                api_key=api_key,
                auth_header=auth_header
            )
            
            # Store user_id in request state for audit logging and downstream use
            request.state.user_id = user_id
            logger.debug(f"✅ Authenticated user: {user_id} accessing {request.url.path}")
            
        except HTTPException as e:
            # Authentication failed - reject request
            logger.warning(
                f"❌ Authentication failed for {request.method} {request.url.path}: {e.detail} "
                f"(IP: {request.client.host if request.client else 'unknown'})"
            )
            
            return JSONResponse(
                status_code=401,
                content={
                    "error": "UNAUTHORIZED",
                    "message": "Authentication required for this endpoint",
                    "details": {
                        "path": request.url.path,
                        "method": request.method,
                        "hint": "Provide X-API-Key header or Authorization Bearer token"
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            # Unexpected error during authentication
            logger.error(f"❌ Authentication error for {request.url.path}: {e}", exc_info=True)
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "AUTHENTICATION_ERROR",
                    "message": "An error occurred during authentication",
                    "details": {"path": request.url.path}
                }
            )
        
        # Authentication successful - proceed with request
        return await call_next(request)