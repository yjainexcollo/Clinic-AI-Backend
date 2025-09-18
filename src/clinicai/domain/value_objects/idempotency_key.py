"""Idempotency Key value object for repeat intake logic."""

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IdempotencyKey:
    """Immutable idempotency key value object."""

    value: str

    def __post_init__(self) -> None:
        """Validate idempotency key format."""
        if not self.value:
            raise ValueError("Idempotency key cannot be empty")

        if not isinstance(self.value, str):
            raise ValueError("Idempotency key must be a string")

        if len(self.value) < 8:
            raise ValueError("Idempotency key must be at least 8 characters")

    def __str__(self) -> str:
        """String representation."""
        return self.value

    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if not isinstance(other, IdempotencyKey):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)

    @classmethod
    def generate(cls) -> "IdempotencyKey":
        """Generate a new idempotency key using UUID4."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def from_string(cls, value: str) -> "IdempotencyKey":
        """Create from string value."""
        return cls(value)
