"""Doctor domain entity representing a doctor/owner of patients and visits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..errors import InvalidPatientDataError


@dataclass
class Doctor:
    """Doctor domain entity.

    Note: We intentionally keep validation light here – most constraints
    are business/persistence level. The primary invariant is that
    doctor_id is non-empty.
    """

    doctor_id: str
    name: str
    email: Optional[str] = None
    status: str = "active"  # active, inactive, suspended, etc.
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        self._validate_doctor_data()

    def _validate_doctor_data(self) -> None:
        """Basic validation for doctor data."""
        # doctor_id must be non-empty and reasonably sized
        if not self.doctor_id or not self.doctor_id.strip():
            raise InvalidPatientDataError(  # reuse structured API-style error
                "INVALID_DOCTOR_DATA",
                "doctor_id must be a non-empty string",
                422,
                {"field": "doctor_id", "value": self.doctor_id},
            )

        if len(self.doctor_id) > 100:
            raise InvalidPatientDataError(
                "INVALID_DOCTOR_DATA",
                f"doctor_id too long (max 100 characters), got {len(self.doctor_id)}",
                422,
                {"field": "doctor_id", "value": self.doctor_id[:50]},
            )

        # Name should be at least 2 characters
        if not self.name or len(self.name.strip()) < 2:
            raise InvalidPatientDataError(
                "INVALID_DOCTOR_DATA",
                f"Name must be at least 2 characters, got {len(self.name.strip()) if self.name else 0}",
                422,
                {"field": "name", "value": self.name},
            )

        if len(self.name) > 120:
            raise InvalidPatientDataError(
                "INVALID_DOCTOR_DATA",
                f"Name too long (max 120 characters), got {len(self.name)}",
                422,
                {"field": "name", "value": self.name[:80]},
            )

        # Optional email: basic length check; detailed email validation can live at API layer
        if self.email and len(self.email) > 254:
            raise InvalidPatientDataError(
                "INVALID_DOCTOR_DATA",
                f"Email too long (max 254 characters), got {len(self.email)}",
                422,
                {"field": "email", "value": self.email[:80]},
            )


