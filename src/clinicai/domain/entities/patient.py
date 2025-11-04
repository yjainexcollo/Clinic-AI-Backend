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
    recently_travelled: bool = False
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
            raise InvalidPatientDataError("name", self.name)

        if len(self.name) > 80:
            raise InvalidPatientDataError("name", "Name too long (max 80 characters)")

        # Validate mobile
        if not self.mobile or len(self.mobile.strip()) < 10:
            raise InvalidPatientDataError("mobile", self.mobile)

        # Clean mobile number and validate
        clean_mobile = "".join(filter(str.isdigit, self.mobile))
        if len(clean_mobile) < 10 or len(clean_mobile) > 15:
            raise InvalidPatientDataError(
                "mobile", f"Invalid mobile number length: {len(clean_mobile)}"
            )

        # Validate age
        if not isinstance(self.age, int) or self.age < 0 or self.age > 120:
            raise InvalidPatientDataError("age", self.age)
        
        # Validate language - accept both "en", "es", and "sp" (normalize "es" to "sp" for consistency)
        if self.language and self.language.lower() in ["es", "sp"]:
            self.language = "sp"  # Normalize to "sp" for consistency
        elif self.language not in ["en", "sp"]:
            raise InvalidPatientDataError("language", f"Invalid language: {self.language}")

    def update_contact_info(self, name: str, mobile: str) -> None:
        """Update patient contact information."""
        # Validate new data
        if not name or len(name.strip()) < 2:
            raise InvalidPatientDataError("name", name)

        clean_mobile = "".join(filter(str.isdigit, mobile))
        if len(clean_mobile) < 10 or len(clean_mobile) > 15:
            raise InvalidPatientDataError(
                "mobile", f"Invalid mobile number length: {len(clean_mobile)}"
            )

        self.name = name
        self.mobile = mobile
        self.updated_at = datetime.utcnow()

    def is_valid_for_consultation(self) -> bool:
        """Check if patient can start a new consultation."""
        return bool(self.name and self.mobile and self.age >= 0)
