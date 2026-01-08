class APIError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400, details: dict = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details


class ValidationError(APIError):
    def __init__(self, message: str, details: dict = None):
        super().__init__("INVALID_INPUT", message, 422, details)


class NotFoundError(APIError):
    def __init__(self, message: str, details: dict = None):
        super().__init__("NOT_FOUND", message, 404, details)


class ConflictError(APIError):
    def __init__(self, message: str, details: dict = None):
        super().__init__("CONFLICT", message, 409, details)


class UnauthorizedError(APIError):
    def __init__(self, message: str = "Unauthorized", details: dict = None):
        super().__init__("UNAUTHORIZED", message, 401, details)


class RateLimitError(APIError):
    def __init__(self, message: str, details: dict = None):
        super().__init__("RATE_LIMITED", message, 429, details)


class DownstreamError(APIError):
    def __init__(self, message: str, details: dict = None):
        super().__init__("DOWNSTREAM_ERROR", message, 502, details)


# Domain-specific
class PatientNotFoundError(NotFoundError):
    def __init__(self, patient_id: str):
        super().__init__(f"Patient not found ({patient_id})", {"patient_id": patient_id})


class VisitNotFoundError(NotFoundError):
    def __init__(self, visit_id: str):
        super().__init__(f"Visit not found ({visit_id})", {"visit_id": visit_id})
