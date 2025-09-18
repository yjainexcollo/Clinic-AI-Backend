"""
Date and time utility functions for Clinic-AI application.
"""

from datetime import date, datetime, timezone
from typing import Optional, Union


def get_current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def format_timestamp(timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp to string."""
    return timestamp.strftime(format_str)


def parse_timestamp(
    timestamp_str: str, format_str: str = "%Y-%m-%d %H:%M:%S"
) -> Optional[datetime]:
    """Parse timestamp string to datetime object."""
    try:
        return datetime.strptime(timestamp_str, format_str)
    except ValueError:
        return None


def is_valid_date(date_str: str, format_str: str = "%Y-%m-%d") -> bool:
    """Check if date string is valid."""
    try:
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        return False


def get_age_from_birthdate(birthdate: Union[str, date, datetime]) -> int:
    """Calculate age from birthdate."""
    if isinstance(birthdate, str):
        try:
            birthdate = datetime.strptime(birthdate, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

    if isinstance(birthdate, datetime):
        birthdate = birthdate.date()

    today = date.today()
    age = today.year - birthdate.year

    # Adjust if birthday hasn't occurred this year
    if today.month < birthdate.month or (
        today.month == birthdate.month and today.day < birthdate.day
    ):
        age -= 1

    return age
