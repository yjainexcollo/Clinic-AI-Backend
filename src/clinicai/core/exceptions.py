"""
Exception handling for Clinic-AI application.

This module provides custom exception classes for different layers
of the application following Clean Architecture principles.
"""

from typing import Any, Dict, Optional


class ClinicAIException(Exception):
    """Base exception class for Clinic-AI application."""

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


class ConfigurationError(ClinicAIException):
    """Raised when there's a configuration error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "CONFIG_ERROR", details)


class DatabaseError(ClinicAIException):
    """Raised when there's a database operation error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "DATABASE_ERROR", details)


class ValidationError(ClinicAIException):
    """Raised when there's a validation error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "VALIDATION_ERROR", details)


class AuthenticationError(ClinicAIException):
    """Raised when there's an authentication error."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, "AUTH_ERROR", details)


class AuthorizationError(ClinicAIException):
    """Raised when there's an authorization error."""

    def __init__(
        self, message: str = "Access denied", details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, "AUTHORIZATION_ERROR", details)


class ExternalServiceError(ClinicAIException):
    """Raised when there's an external service error."""

    def __init__(
        self, service: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.service = service
        full_message = f"{service} service error: {message}"
        super().__init__(full_message, "EXTERNAL_SERVICE_ERROR", details)


class OpenAIError(ExternalServiceError):
    """Raised when there's an OpenAI API error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("OpenAI", message, details)


class DeepgramError(ExternalServiceError):
    """Raised when there's a Deepgram API error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("Deepgram", message, details)


class PatientNotFoundError(ClinicAIException):
    """Raised when a patient is not found."""

    def __init__(
        self, patient_id: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Patient with ID '{patient_id}' not found"
        super().__init__(message, "PATIENT_NOT_FOUND", details)


class ConsultationNotFoundError(ClinicAIException):
    """Raised when a consultation is not found."""

    def __init__(
        self, consultation_id: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Consultation with ID '{consultation_id}' not found"
        super().__init__(message, "CONSULTATION_NOT_FOUND", details)


class DuplicatePatientError(ClinicAIException):
    """Raised when trying to create a duplicate patient."""

    def __init__(
        self, patient_id: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Patient with ID '{patient_id}' already exists"
        super().__init__(message, "DUPLICATE_PATIENT", details)


class InvalidAudioFormatError(ClinicAIException):
    """Raised when audio format is invalid."""

    def __init__(self, format: str, details: Optional[Dict[str, Any]] = None) -> None:
        message = f"Invalid audio format: {format}"
        super().__init__(message, "INVALID_AUDIO_FORMAT", details)


class TranscriptionError(ClinicAIException):
    """Raised when audio transcription fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "TRANSCRIPTION_ERROR", details)


class SOAPGenerationError(ClinicAIException):
    """Raised when SOAP note generation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "SOAP_GENERATION_ERROR", details)


class CacheError(ClinicAIException):
    """Raised when there's a cache operation error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "CACHE_ERROR", details)


class EventBusError(ClinicAIException):
    """Raised when there's an event bus error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "EVENT_BUS_ERROR", details)
