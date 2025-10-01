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
    visits: List["Visit"] = field(default_factory=list)
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
        
        # Validate language
        if self.language not in ["en", "sp"]:
            raise InvalidPatientDataError("language", f"Invalid language: {self.language}")

    def add_visit(self, visit: "Visit") -> None:
        """Add a new visit to the patient's history."""
        self.visits.append(visit)
        self.updated_at = datetime.utcnow()

    def get_latest_visit(self) -> Optional["Visit"]:
        """Get the most recent visit."""
        if not self.visits:
            return None
        return max(self.visits, key=lambda v: v.created_at)

    def get_visit_by_id(self, visit_id: str) -> Optional["Visit"]:
        """Get visit by visit ID."""
        for visit in self.visits:
            if visit.visit_id.value == visit_id:
                return visit
        return None

    def has_active_intake(self) -> bool:
        """Check if patient has an active intake session."""
        latest_visit = self.get_latest_visit()
        if not latest_visit:
            return False
        return latest_visit.intake_session.status == "in_progress"

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
