"""
Question ID value object for tracking individual questions.
"""

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuestionId:
    """Immutable question identifier value object."""

    value: str

    def __post_init__(self) -> None:
        """Validate question ID format."""
        if not self.value:
            raise ValueError("Question ID cannot be empty")

    def __str__(self) -> str:
        """String representation."""
        return self.value

    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if not isinstance(other, QuestionId):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)

    @classmethod
    def generate(cls) -> "QuestionId":
        """Generate a new question ID."""
        return cls(f"Q{str(uuid.uuid4())[:8]}")
