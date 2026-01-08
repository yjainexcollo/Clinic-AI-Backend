from typing import Any, Optional

from fastapi import Request
from fastapi import status as http_status
from fastapi.responses import JSONResponse

from ..schemas.common import ApiResponse, ErrorResponse


def ok(request: Request, data: Any = None, message: str = "") -> ApiResponse[Any]:
    req_id = getattr(request.state, "request_id", None)
    return ApiResponse(success=True, message=message, request_id=req_id or "", data=data)


def fail(
    request: Request,
    error: str,
    message: str,
    details: Optional[dict] = None,
    status_message: str = "",
    status_code: int = http_status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> JSONResponse:
    """
    Return an error response with the specified status code.

    This function returns a JSONResponse (not a Pydantic model) to ensure
    the correct HTTP status code is set, regardless of the route decorator's status_code.

    Args:
        request: FastAPI Request object
        error: Error code/type (e.g., "DUPLICATE_PATIENT", "INTERNAL_ERROR")
        message: Human-readable error message
        details: Optional additional error details
        status_message: Deprecated, use message instead
        status_code: HTTP status code (default: 500)

    Returns:
        JSONResponse with ErrorResponse content and correct status code
    """
    req_id = getattr(request.state, "request_id", None)
    error_response = ErrorResponse(
        error=error,
        message=message or status_message,
        request_id=req_id or "",
        details=details or {},
    )
    return JSONResponse(status_code=status_code, content=error_response.dict())
