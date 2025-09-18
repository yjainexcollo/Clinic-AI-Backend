"""
Domain-specific error types for business rule violations.
"""

from typing import Any, Dict, Optional


class DomainError(Exception):
    """Base domain error."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class PatientNotFoundError(DomainError):
    """Patient not found."""

    def __init__(self, patient_id: str) -> None:
        message = f"Patient with ID '{patient_id}' not found"
        super().__init__(message, "PATIENT_NOT_FOUND", {"patient_id": patient_id})


class DuplicatePatientError(DomainError):
    """Patient already exists."""

    def __init__(self, patient_id: str) -> None:
        message = f"Patient with ID '{patient_id}' already exists"
        super().__init__(message, "DUPLICATE_PATIENT", {"patient_id": patient_id})


class InvalidPatientDataError(DomainError):
    """Invalid patient data."""

    def __init__(self, field: str, value: Any) -> None:
        message = f"Invalid patient data. Field: {field}, Value: {value}"
        super().__init__(
            message, "INVALID_PATIENT_DATA", {"field": field, "value": value}
        )


class QuestionLimitExceededError(DomainError):
    """Maximum questions limit exceeded."""

    def __init__(self, current_count: int, max_count: int) -> None:
        message = f"Question limit exceeded. Current: {current_count}, Max: {max_count}"
        super().__init__(
            message,
            "QUESTION_LIMIT_EXCEEDED",
            {"current_count": current_count, "max_count": max_count},
        )


class DuplicateQuestionError(DomainError):
    """Attempted to ask a duplicate question."""

    def __init__(self, question: str) -> None:
        message = f"Duplicate question detected: {question}"
        super().__init__(message, "DUPLICATE_QUESTION", {"question": question})


class InvalidDiseaseError(DomainError):
    """Invalid disease/complaint specified."""

    def __init__(self, disease: str) -> None:
        message = f"Invalid disease/complaint: {disease}"
        super().__init__(message, "INVALID_DISEASE", {"disease": disease})


class IntakeAlreadyCompletedError(DomainError):
    """Intake process already completed."""

    def __init__(self, visit_id: str) -> None:
        message = f"Intake already completed for visit: {visit_id}"
        super().__init__(message, "INTAKE_ALREADY_COMPLETED", {"visit_id": visit_id})


class VisitNotFoundError(DomainError):
    """Visit not found."""

    def __init__(self, visit_id: str) -> None:
        message = f"Visit with ID '{visit_id}' not found"
        super().__init__(message, "VISIT_NOT_FOUND", {"visit_id": visit_id})
