"""Patient matching and normalization utilities for repeat intake logic."""

import re
from typing import Optional, Tuple


def normalize_name(name: str) -> str:
    """
    Normalize patient name for consistent matching.

    Rules:
    - Trim whitespace
    - Convert to lowercase
    - Remove extra spaces
    - Remove special characters except spaces and hyphens
    """
    if not name:
        return ""

    # Trim and convert to lowercase
    normalized = name.strip().lower()

    # Remove special characters except spaces and hyphens
    normalized = re.sub(r"[^\w\s\-]", "", normalized)

    # Replace multiple spaces with single space
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def normalize_phone(phone: str) -> str:
    """
    Normalize phone number to E.164 format.

    Rules:
    - Remove all non-digit characters
    - Add country code if missing (assume +1 for US)
    - Return in E.164 format (+1234567890)
    """
    if not phone:
        return ""

    # Remove all non-digit characters
    digits_only = re.sub(r"\D", "", phone)

    if not digits_only:
        return ""

    # Handle different phone number formats
    if len(digits_only) == 10:
        # US number without country code
        return f"+1{digits_only}"
    elif len(digits_only) == 11 and digits_only.startswith("1"):
        # US number with country code
        return f"+{digits_only}"
    elif len(digits_only) > 11:
        # International number
        return f"+{digits_only}"
    else:
        # Assume it's a valid number and add +1
        return f"+1{digits_only}"


def normalize_patient_data(name: str, phone: str) -> Tuple[str, str]:
    """
    Normalize both name and phone for patient matching.

    Returns:
        Tuple of (normalized_name, normalized_phone_e164)
    """
    return normalize_name(name), normalize_phone(phone)


def normalize_phone_digits_only(phone: str) -> str:
    """
    Normalize phone number to digits-only for matching.

    Rules:
    - Remove all non-digit characters
    - Return only digits string (no leading +)
    """
    if not phone:
        return ""

    digits_only = re.sub(r"\D", "", phone)
    return digits_only


def is_strong_match(name1: str, phone1: str, name2: str, phone2: str) -> bool:
    """
    Check if two patient records represent the same person.

    Strong match condition:
    - Normalized names are exactly equal
    - Normalized phone numbers are exactly equal
    """
    norm_name1, norm_phone1 = normalize_patient_data(name1, phone1)
    norm_name2, norm_phone2 = normalize_patient_data(name2, phone2)

    return norm_name1 == norm_name2 and norm_phone1 == norm_phone2 and norm_name1 != "" and norm_phone1 != ""


def generate_patient_id(name: str, phone: str) -> str:
    """
    Generate patient ID in format {patientname}_{patientmobilenumber}.

    - Patient name: trimmed, lowercase, spaces collapsed and removed of non-word chars
    - Mobile number: digits-only
    """
    norm_name = normalize_name(name)
    # Remove spaces from normalized name for ID token
    name_token = re.sub(r"\s+", "", norm_name)
    phone_digits = normalize_phone_digits_only(phone)
    return f"{name_token}_{phone_digits}"


def validate_phone_otp_verified(phone: str, otp_verified: bool) -> bool:
    """
    Validate that phone number is OTP verified.

    This is a placeholder for OTP verification logic.
    In a real implementation, this would check against an OTP service.
    """
    if not phone:
        return False

    # For now, assume all phone numbers are verified
    # In production, this would integrate with your OTP service
    return otp_verified


def should_prevent_rapid_repeat(patient_id: str, last_visit_time: Optional[str]) -> bool:
    """
    Check if rapid repeat visits should be prevented.

    Prevents multiple open visits within 30-60 minutes unless forced.
    """
    if not last_visit_time:
        return False

    # This would check the time difference in a real implementation
    # For now, return False to allow all visits
    return False
