"""
Domain-specific error types for business rule violations.
"""

from typing import Any, Dict, Optional

from clinicai.api.errors import APIError, ConflictError, NotFoundError


class DomainError(APIError):
    pass


# All potentially imported errors for cross-router compatibility
class DuplicatePatientError(ConflictError):
    pass


class DuplicateQuestionError(ConflictError):
    pass


class IntakeAlreadyCompletedError(ConflictError):
    pass


class InvalidDiseaseError(APIError):
    pass


class PatientNotFoundError(NotFoundError):
    pass


class QuestionLimitExceededError(ConflictError):
    pass


class VisitNotFoundError(NotFoundError):
    pass


class PatientAlreadyExistsError(ConflictError):
    pass


class VisitAlreadyExistsError(ConflictError):
    pass


class InvalidPatientDataError(APIError):
    pass
