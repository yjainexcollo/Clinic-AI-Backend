"""
String utility functions for Clinic-AI application.
"""

import re
import uuid


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_id = str(uuid.uuid4()).replace("-", "")
    return f"{prefix}{unique_id}" if prefix else unique_id


def sanitize_string(text: str) -> str:
    """Sanitize string by removing special characters and normalizing whitespace."""
    # Remove special characters except alphanumeric, spaces, and basic punctuation
    sanitized = re.sub(r"[^\w\s\-.,!?]", "", text)
    # Normalize whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_phone_number(phone: str) -> bool:
    """Validate phone number format."""
    # Remove all non-digit characters
    digits_only = re.sub(r"\D", "", phone)
    # Check if it's a valid length (7-15 digits)
    return 7 <= len(digits_only) <= 15


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    # Convert to lowercase
    slug = text.lower()
    # Replace spaces and special characters with hyphens
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    # Remove leading/trailing hyphens
    return slug.strip("-")


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
