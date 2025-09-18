"""
Patient ID value object for type-safe patient identification.
Format: {PATIENT_NAME}_{PATIENT_PHONE_NUMBER}
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatientId:
    """Immutable patient identifier value object."""

    value: str

    def __post_init__(self) -> None:
        """Validate patient ID format."""
        if not self.value:
            raise ValueError("Patient ID cannot be empty")

        if not isinstance(self.value, str):
            raise ValueError("Patient ID must be a string")

        # Validate format: {patient_name}_{patient_phone_number}
        pattern = r"^[a-zA-Z0-9_]+_\d+$"
        if not re.match(pattern, self.value):
            raise ValueError(
                "Patient ID must follow format: {PATIENT_NAME}_{PATIENT_PHONE_NUMBER}"
            )

    def __str__(self) -> str:
        """String representation."""
        return self.value

    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if not isinstance(other, PatientId):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)

    @classmethod
    def generate(cls, patient_name: str, phone_number: str) -> "PatientId":
        """Generate a new patient ID from name and phone."""
        # Clean and format the name (remove spaces, special chars, convert to lowercase)
        clean_name = re.sub(r"[^a-zA-Z0-9]", "", patient_name).lower()
        if not clean_name:
            raise ValueError(
                "Patient name must contain at least one alphanumeric character"
            )

        # Clean phone number (remove all non-digits)
        clean_phone = re.sub(r"\D", "", phone_number)
        if not clean_phone:
            raise ValueError("Phone number must contain at least one digit")

        return cls(f"{clean_name}_{clean_phone}")
