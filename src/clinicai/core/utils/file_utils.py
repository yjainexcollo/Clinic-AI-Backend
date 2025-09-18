"""
File utility functions for Clinic-AI application.
"""

import os
from pathlib import Path
from typing import List


def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return Path(filename).suffix.lower()


def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate if file type is allowed."""
    extension = get_file_extension(filename)
    return extension in allowed_extensions


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0


def create_directory(directory_path: str) -> bool:
    """Create directory if it doesn't exist."""
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def ensure_file_exists(filepath: str) -> bool:
    """Ensure file exists, create if it doesn't."""
    try:
        Path(filepath).touch()
        return True
    except OSError:
        return False
