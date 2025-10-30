from typing import Any, Optional
from fastapi import Request
from ..schemas.common import ApiResponse, ErrorResponse

def ok(request: Request, data: Any = None, message: str = "") -> ApiResponse[Any]:
    req_id = getattr(request.state, "request_id", None)
    return ApiResponse(success=True, message=message, request_id=req_id or "", data=data)

def fail(request: Request, error: str, message: str, details: Optional[dict] = None, status_message: str = "") -> ErrorResponse:
    req_id = getattr(request.state, "request_id", None)
    return ErrorResponse(error=error, message=message or status_message, request_id=req_id or "", details=details or {})
