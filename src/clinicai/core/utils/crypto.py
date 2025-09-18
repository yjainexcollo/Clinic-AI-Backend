"""
Crypto utilities for PHI-safe operations:
- Fernet-based symmetric encryption/decryption
- Opaque ID encode/decode helpers for patient_id exposure

Env vars used:
- ENCRYPTION_KEY: base64-encoded 32-byte key for Fernet
"""

from __future__ import annotations

import base64
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken  # type: ignore
import logging


_fernet_singleton: Optional[Fernet] = None


def get_fernet() -> Fernet:
    global _fernet_singleton
    if _fernet_singleton is not None:
        return _fernet_singleton

    key = os.getenv("ENCRYPTION_KEY", "").strip()
    # If key is not provided, generate an ephemeral key for development usage
    # Note: This will invalidate previously issued opaque IDs on restart
    if not key:
        _fernet_singleton = Fernet(Fernet.generate_key())
        logging.getLogger("clinicai").warning(
            "ENCRYPTION_KEY not set. Using ephemeral key (dev only). Opaque IDs will change on restart."
        )
        return _fernet_singleton
    # If provided, it must be a valid urlsafe base64-encoded 32-byte key as required by Fernet
    try:
        _fernet_singleton = Fernet(key.encode("utf-8"))
        return _fernet_singleton
    except Exception as exc:
        # Fallback to ephemeral in case of malformed env key to avoid hard 500s during dev
        logging.getLogger("clinicai").warning(
            "Invalid ENCRYPTION_KEY provided. Falling back to ephemeral key (dev only). Error=%s",
            str(exc),
        )
        _fernet_singleton = Fernet(Fernet.generate_key())
        return _fernet_singleton


def encrypt_text(plaintext: str) -> str:
    if plaintext is None:
        raise ValueError("encrypt_text: plaintext cannot be None")
    f = get_fernet()
    token = f.encrypt(plaintext.encode("utf-8"))
    # Return urlsafe base64 string
    return token.decode("utf-8")


def decrypt_text(token_str: str) -> str:
    if not token_str:
        raise ValueError("decrypt_text: token cannot be empty")
    f = get_fernet()
    try:
        plaintext = f.decrypt(token_str.encode("utf-8"))
        return plaintext.decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Invalid opaque token") from exc


def encode_patient_id(internal_patient_id: str) -> str:
    """Produce an opaque, reversible ID for exposure to clients."""
    return encrypt_text(internal_patient_id)


def decode_patient_id(opaque_patient_id: str) -> str:
    """Recover internal patient_id from opaque client-facing token."""
    return decrypt_text(opaque_patient_id)


