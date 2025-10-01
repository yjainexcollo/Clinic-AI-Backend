"""
File utility functions for Clinic-AI application.
"""

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List


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


def save_audio_file(
    temp_file_path: str, 
    storage_directory: str, 
    original_filename: Optional[str] = None
) -> Optional[str]:
    """
    Save audio file from temporary location to permanent storage.
    
    Args:
        temp_file_path: Path to temporary audio file
        storage_directory: Directory to save the audio file
        original_filename: Original filename (optional)
    
    Returns:
        Path to saved audio file, or None if failed
    """
    try:
        # Ensure storage directory exists
        if not create_directory(storage_directory):
            return None
        
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        if original_filename:
            # Preserve original extension
            extension = get_file_extension(original_filename)
            filename = f"adhoc_{timestamp}_{unique_id}{extension}"
        else:
            # Use temp file extension
            extension = get_file_extension(temp_file_path)
            filename = f"adhoc_{timestamp}_{unique_id}{extension}"
        
        # Create full path
        permanent_path = os.path.join(storage_directory, filename)
        
        # Copy file from temp to permanent location
        shutil.copy2(temp_file_path, permanent_path)
        
        return permanent_path
        
    except Exception:
        return None


