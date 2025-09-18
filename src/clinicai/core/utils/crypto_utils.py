"""
Cryptographic utility functions for Clinic-AI application.
"""

import hashlib
import secrets


def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == hashed_password


def generate_random_string(length: int = 32) -> str:
    """Generate cryptographically secure random string."""
    return secrets.token_urlsafe(length)


def encrypt_data(data: str, key: str) -> str:
    """Simple encryption for non-sensitive data."""
    # Note: This is a simple implementation. For production, use proper encryption
    import base64

    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data


def decrypt_data(encrypted_data: str, key: str) -> str:
    """Simple decryption for non-sensitive data."""
    # Note: This is a simple implementation. For production, use proper decryption
    import base64

    try:
        decoded_data = base64.b64decode(encrypted_data.encode()).decode()
        return decoded_data
    except Exception:
        return ""
