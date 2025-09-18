"""
Utility functions for Clinic-AI application.

This module provides common utility functions used throughou
the application for various operations.
"""

from .crypto_utils import (
    decrypt_data,
    encrypt_data,
    generate_random_string,
    hash_password,
    verify_password,
)
from .datetime_utils import (
    format_timestamp,
    get_age_from_birthdate,
    get_current_timestamp,
    is_valid_date,
    parse_timestamp,
)
from .file_utils import (
    create_directory,
    ensure_file_exists,
    get_file_extension,
    get_file_size,
    validate_file_type,
)
from .string_utils import (
    generate_id,
    sanitize_string,
    slugify,
    truncate_string,
    validate_email,
    validate_phone_number,
)

__all__ = [
    # Datetime utilities
    "get_current_timestamp",
    "format_timestamp",
    "parse_timestamp",
    "is_valid_date",
    "get_age_from_birthdate",
    # String utilities
    "generate_id",
    "sanitize_string",
    "validate_email",
    "validate_phone_number",
    "slugify",
    "truncate_string",
    # File utilities
    "get_file_extension",
    "validate_file_type",
    "get_file_size",
    "create_directory",
    "ensure_file_exists",
    # Crypto utilities
    "hash_password",
    "verify_password",
    "generate_random_string",
    "encrypt_data",
    "decrypt_data",
]
