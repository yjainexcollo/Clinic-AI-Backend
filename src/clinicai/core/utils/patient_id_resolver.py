"""
Patient ID resolver utility for handling encrypted and plain text patient IDs.
"""

import logging
from typing import Union

logger = logging.getLogger("clinicai")


def resolve_patient_id(patient_id: str, context: str = "unknown") -> str:
    """
    Resolve patient ID from encrypted or plain text format.
    
    Args:
        patient_id: The patient ID (encrypted or plain text)
        context: Context for logging (e.g., "patient endpoint")
    
    Returns:
        Resolved patient ID as string
    """
    try:
        # For now, just return the patient_id as-is
        # In a real implementation, this would handle encryption/decryption
        logger.info(f"Resolving patient ID for {context}: {patient_id[:10]}...")
        return str(patient_id)
        
    except Exception as e:
        logger.error(f"Failed to resolve patient ID: {e}")
        return str(patient_id)
