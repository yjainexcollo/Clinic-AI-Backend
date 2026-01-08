"""
HIPAA audit middleware - comprehensive request/response logging

This middleware logs all PHI access for HIPAA compliance. It extracts patient_id and visit_id
from URL paths where possible. For endpoints with IDs in the request body, those endpoints
should set request.state.audit_patient_id and request.state.audit_visit_id for accurate logging.
"""

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.hipaa_audit import get_audit_logger

logger = logging.getLogger(__name__)


def extract_user_id_from_request(request: Request) -> str:
    """
    Extract user ID from request headers or state.

    Priority:
    1. request.state.user_id (set by auth middleware) - PRIMARY SOURCE
    2. X-User-ID header (for trusted frontend scenarios)
    3. Authorization header (fallback if auth middleware didn't set user_id)

    Note: "anonymous" fallback removed for HIPAA compliance.
    All PHI access must be authenticated.
    """
    # Check if auth middleware already set user_id (this is the primary source)
    if hasattr(request.state, "user_id") and request.state.user_id:
        return request.state.user_id

    # Fallback: Try to extract from headers (for cases where auth middleware might not have run)
    # This should rarely be needed if authentication middleware is properly configured
    user_id_header = request.headers.get("X-User-ID") or request.headers.get("x-user-id")
    if user_id_header:
        logger.warning(
            f"Using X-User-ID header for user identification on {request.url.path}. "
            "This should be set by authentication middleware."
        )
        return user_id_header

    # Last resort: Try to get from Authorization header
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth_header:
        if auth_header.startswith("Bearer "):
            # Extract token and use first 20 chars as identifier
            token = auth_header[7:]
            logger.warning(
                f"Extracting user ID from Bearer token on {request.url.path}. "
                "Authentication middleware should have set request.state.user_id."
            )
            return f"bearer_{token[:20]}"
        elif auth_header.startswith("Basic "):
            # Basic auth - extract username
            import base64

            try:
                credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
                username, _ = credentials.split(":", 1)
                return username
            except Exception:
                pass

    # No authentication found - this should not happen if authentication middleware is working
    # Log as warning (not error) since this might be expected in some edge cases
    # The HIPAA middleware will skip audit logging if user_id is "unauthenticated"
    logger.warning(
        f"⚠️  No user ID found for request to {request.url.path}. "
        "Authentication middleware should have set request.state.user_id or rejected the request. "
        "Skipping HIPAA audit logging for this request."
    )
    return "unauthenticated"


class HIPAAAuditMiddleware(BaseHTTPMiddleware):
    """Middleware for HIPAA-compliant audit logging"""

    # Map endpoints to resource types and PHI fields
    PHI_MAPPINGS = {
        "/patients": {
            "resource_type": "patient",
            "phi_fields": ["name", "email", "phone", "date_of_birth", "address", "mrn"],
        },
        "/notes": {
            "resource_type": "clinical_note",
            "phi_fields": [
                "transcript",
                "soap_note",
                "diagnosis",
                "medications",
                "assessment",
            ],
        },
        "/intake": {
            "resource_type": "intake",
            "phi_fields": [
                "responses",
                "chief_complaint",
                "symptoms",
                "medical_history",
            ],
        },
        "/audio": {
            "resource_type": "audio_recording",
            "phi_fields": ["audio_file", "transcription", "recording"],
        },
        "/vitals": {
            "resource_type": "vital_signs",
            "phi_fields": ["blood_pressure", "heart_rate", "temperature", "weight"],
        },
        "/doctor": {
            "resource_type": "doctor_preference",
            "phi_fields": ["preferences", "templates", "settings"],
        },
        "/workflow": {
            "resource_type": "workflow_visit",
            "phi_fields": ["name", "mobile", "age", "gender", "visit_status"],
        },
    }

    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Determine if endpoint accesses PHI
        phi_info = self._get_phi_info(request.url.path)
        phi_accessed = phi_info is not None

        # Extract IDs from URL path (before processing request)
        resource_id, patient_id, visit_id = self._extract_ids_from_path(request.url.path)

        # Process request
        response = await call_next(request)

        # After processing, check if endpoint set IDs in request.state
        # This allows endpoints with body-based IDs to provide them
        if hasattr(request.state, "audit_patient_id"):
            patient_id = request.state.audit_patient_id
            if not resource_id:
                resource_id = patient_id
        if hasattr(request.state, "audit_visit_id"):
            visit_id = request.state.audit_visit_id

        # Calculate metrics
        process_time = time.time() - start_time

        # Log to HIPAA audit if PHI was accessed
        # Skip logging if authentication failed (401) or auth error (500) - auth middleware already logged it
        if phi_accessed and response.status_code not in [401, 500]:
            audit_logger = get_audit_logger()

            try:
                # Extract user ID from request
                # Check if user_id exists in request state (set by auth middleware)
                # If not, try fallback methods but log warning
                user_id = None
                if hasattr(request.state, "user_id") and request.state.user_id:
                    user_id = request.state.user_id
                else:
                    # Fallback: try to extract from headers (for edge cases)
                    user_id = extract_user_id_from_request(request)

                    # If still "unauthenticated", skip HIPAA logging (auth middleware should have rejected)
                    if user_id == "unauthenticated":
                        logger.warning(
                            f"⚠️  Skipping HIPAA audit log for {request.url.path}: "
                            "Request reached HIPAA middleware without authentication. "
                            "This should have been rejected by authentication middleware."
                        )
                        # Skip audit logging but continue to add headers below
                        user_id = None

                # Only log if we have a valid user_id
                if user_id and user_id != "unauthenticated":
                    await audit_logger.log_phi_access(
                        user_id=user_id,
                        action=request.method,
                        resource_type=phi_info["resource_type"],
                        resource_id=resource_id,
                        patient_id=patient_id,
                        ip_address=self._get_client_ip(request),
                        user_agent=request.headers.get("user-agent", "unknown"),
                        phi_fields=phi_info["phi_fields"],
                        phi_accessed=True,
                        success=response.status_code < 400,
                        details={
                            "endpoint": request.url.path,
                            "method": request.method,
                            "status_code": response.status_code,
                            "process_time_ms": round(process_time * 1000, 2),
                            "query_params": (dict(request.query_params) if request.query_params else {}),
                            "response_size": response.headers.get("content-length", "0"),
                            "visit_id": visit_id,
                            "resource_creation": request.method == "POST" and request.url.path == "/patients/",
                        },
                        request_id=request_id,
                        session_id=getattr(request.state, "session_id", None),
                    )
            except Exception as e:
                logger.error(f"Failed to log HIPAA audit: {e}")

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))

        return response

    def _get_phi_info(self, path: str) -> dict:
        """Get PHI information for endpoint"""
        for endpoint_prefix, phi_info in self.PHI_MAPPINGS.items():
            if path.startswith(endpoint_prefix):
                return phi_info
        return None

    def _extract_ids_from_path(self, path: str) -> tuple:
        """Extract resource ID, patient ID, and visit ID from URL path

        Returns: (resource_id, patient_id, visit_id)

        Note: For endpoints with IDs in request body (like /patients/consultations/answer),
        the endpoint should set request.state.audit_patient_id and request.state.audit_visit_id
        """
        parts = path.split("/")
        resource_id = None
        patient_id = None
        visit_id = None

        # Pattern: /patients/{patient_id}/...
        if len(parts) > 2 and parts[1] == "patients" and parts[2] not in ["consultations", "webhook", "summary", ""]:
            patient_id = parts[2]
            resource_id = patient_id

            # Pattern: /patients/{patient_id}/visits/{visit_id}/...
            if len(parts) > 4 and parts[3] == "visits":
                visit_id = parts[4]

        # Pattern: /notes/{patient_id}/visits/{visit_id}/...
        elif len(parts) > 2 and parts[1] == "notes":
            patient_id = parts[2]
            if len(parts) > 4 and parts[3] == "visits":
                visit_id = parts[4]
                resource_id = visit_id

        return resource_id, patient_id, visit_id

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address (considering proxies)"""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
