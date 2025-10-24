"""
HIPAA audit middleware - comprehensive request/response logging
"""
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from ..core.hipaa_audit import get_audit_logger
import logging

logger = logging.getLogger(__name__)


class HIPAAAuditMiddleware(BaseHTTPMiddleware):
    """Middleware for HIPAA-compliant audit logging"""
    
    # Map endpoints to resource types and PHI fields
    PHI_MAPPINGS = {
        "/patients": {
            "resource_type": "patient",
            "phi_fields": ["name", "email", "phone", "date_of_birth", "address", "mrn"]
        },
        "/notes": {
            "resource_type": "clinical_note",
            "phi_fields": ["transcript", "soap_note", "diagnosis", "medications", "assessment"]
        },
        "/intake": {
            "resource_type": "intake",
            "phi_fields": ["responses", "chief_complaint", "symptoms", "medical_history"]
        },
        "/audio": {
            "resource_type": "audio_recording",
            "phi_fields": ["audio_file", "transcription", "recording"]
        },
        "/vitals": {
            "resource_type": "vital_signs",
            "phi_fields": ["blood_pressure", "heart_rate", "temperature", "weight"]
        },
        "/doctor": {
            "resource_type": "doctor_preference",
            "phi_fields": ["preferences", "templates", "settings"]
        }
    }
    
    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Determine if endpoint accesses PHI
        phi_info = self._get_phi_info(request.url.path)
        phi_accessed = phi_info is not None
        
        # Extract IDs from path
        resource_id, patient_id = self._extract_ids(request.url.path)
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        
        # Log to HIPAA audit if PHI was accessed
        if phi_accessed:
            audit_logger = get_audit_logger()
            
            try:
                await audit_logger.log_phi_access(
                    user_id=getattr(request.state, "user_id", None),
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
                        "query_params": dict(request.query_params) if request.query_params else {},
                        "response_size": response.headers.get("content-length", "0")
                    },
                    request_id=request_id,
                    session_id=getattr(request.state, "session_id", None)
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
    
    def _extract_ids(self, path: str) -> tuple:
        """Extract resource ID and patient ID from path"""
        parts = path.split("/")
        resource_id = None
        patient_id = None
        
        # Try to extract IDs from common patterns
        if len(parts) > 2:
            # /patients/{patient_id}/...
            if parts[1] == "patients" and len(parts) > 2:
                patient_id = parts[2]
                resource_id = patient_id
            # /notes/{patient_id}/visits/{visit_id}/...
            elif parts[1] == "notes" and len(parts) > 2:
                patient_id = parts[2]
                if len(parts) > 4:
                    resource_id = parts[4]
        
        return resource_id, patient_id
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address (considering proxies)"""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

