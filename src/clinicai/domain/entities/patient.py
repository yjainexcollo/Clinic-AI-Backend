"""Patient domain entity representing a patient in the clinic system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from ..errors import InvalidPatientDataError
from ..value_objects.patient_id import PatientId

if TYPE_CHECKING:
    # Avoid circular import at runtime while keeping type hints
    from .visit import Visit


@dataclass
class Patient:
    """Patient domain entity."""

    patient_id: PatientId
    name: str
    mobile: str
    age: int
    gender: Optional[str] = None
    # recently_travelled removed - now stored on Visit (travel is visit-specific, not lifetime patient attribute)
    language: str = "en"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate patient data."""
        self._validate_patient_data()

    def _validate_patient_data(self) -> None:
        """Validate patient data according to business rules."""
        # Validate name
        if not self.name or len(self.name.strip()) < 2:
            raise InvalidPatientDataError(
                "INVALID_PATIENT_DATA",
                f"Name must be at least 2 characters, got {len(self.name.strip()) if self.name else 0}",
                422,
                {"field": "name", "value": self.name}
            )

        if len(self.name) > 80:
            raise InvalidPatientDataError(
                "INVALID_PATIENT_DATA",
                f"Name too long (max 80 characters), got {len(self.name)}",
                422,
                {"field": "name", "value": self.name[:50]}  # Truncate for details
            )

        # Validate mobile
        if not self.mobile or len(self.mobile.strip()) < 10:
            raise InvalidPatientDataError(
                "INVALID_PATIENT_DATA",
                f"Mobile number must be at least 10 digits, got {len(self.mobile.strip()) if self.mobile else 0}",
                422,
                {"field": "mobile", "value": self.mobile}
            )

        # Clean mobile number and validate
        clean_mobile = "".join(filter(str.isdigit, self.mobile))
        if len(clean_mobile) < 10 or len(clean_mobile) > 15:
            raise InvalidPatientDataError(
                "INVALID_PATIENT_DATA",
                f"Mobile number must be between 10 and 15 digits, got {len(clean_mobile)} digits",
                422,
                {"field": "mobile", "value": self.mobile, "cleaned_value": clean_mobile}
            )

        # Validate age
        if not isinstance(self.age, int) or self.age < 0 or self.age > 120:
            raise InvalidPatientDataError(
                "INVALID_PATIENT_DATA",
                f"Age must be between 0 and 120, got {self.age}",
                422,
                {"field": "age", "value": self.age}
            )
        
        # Validate language - accept both "en", "es", and "sp" (normalize "es" to "sp" for consistency)
        if self.language and self.language.lower() in ["es", "sp"]:
            self.language = "sp"  # Normalize to "sp" for consistency
        elif self.language not in ["en", "sp"]:
            raise InvalidPatientDataError(
                "INVALID_PATIENT_DATA",
                f"Invalid language: {self.language}. Must be 'en' or 'sp'",
                422,
                {"field": "language", "value": self.language}
            )

    def update_contact_info(self, name: str, mobile: str) -> None:
        """Update patient contact information."""
        # Validate new data
        if not name or len(name.strip()) < 2:
            raise InvalidPatientDataError(
                "INVALID_PATIENT_DATA",
                f"Name must be at least 2 characters, got {len(name.strip()) if name else 0}",
                422,
                {"field": "name", "value": name}
            )

        clean_mobile = "".join(filter(str.isdigit, mobile))
        if len(clean_mobile) < 10 or len(clean_mobile) > 15:
            raise InvalidPatientDataError(
                "INVALID_PATIENT_DATA",
                f"Mobile number must be between 10 and 15 digits, got {len(clean_mobile)} digits",
                422,
                {"field": "mobile", "value": mobile, "cleaned_value": clean_mobile}
            )

        self.name = name
        self.mobile = mobile
        self.updated_at = datetime.utcnow()

    def is_valid_for_consultation(self) -> bool:
        """Check if patient can start a new consultation."""
        return bool(self.name and self.mobile and self.age >= 0)
