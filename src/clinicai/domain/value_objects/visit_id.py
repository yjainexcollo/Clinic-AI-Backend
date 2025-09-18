"""
Visit ID value object for type-safe visit identification.
Format: CONSULT-YYYYMMDD-XXX
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class VisitId:
    """Immutable visit identifier value object."""

    value: str

    def __post_init__(self) -> None:
        """Validate visit ID format."""
        if not self.value:
            raise ValueError("Visit ID cannot be empty")

        if not isinstance(self.value, str):
            raise ValueError("Visit ID must be a string")

        # Validate format: CONSULT-YYYYMMDD-XXX
        pattern = r"^CONSULT-\d{8}-\d{3}$"
        if not re.match(pattern, self.value):
            raise ValueError("Visit ID must follow format: CONSULT-YYYYMMDD-XXX")

    def __str__(self) -> str:
        """String representation."""
        return self.value

    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if not isinstance(other, VisitId):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)

    @classmethod
    def generate(cls, date: datetime = None) -> "VisitId":
        """Generate a new visit ID."""
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%Y%m%d")

        # Generate 3-digit sequence number (in real app, this would be from DB)
        import random

        sequence = str(random.randint(1, 999)).zfill(3)

        return cls(f"CONSULT-{date_str}-{sequence}")
