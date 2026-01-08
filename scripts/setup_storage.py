#!/usr/bin/env python3
"""
Setup script to create storage directories for audio files.
Audio files are stored locally and referenced in the database.
"""

import os
from pathlib import Path


def setup_storage_directories():
    """Create storage directories if they don't exist."""
    storage_path = Path("./storage")
    audio_storage_path = storage_path / "audio"

    # Create directories
    storage_path.mkdir(exist_ok=True)
    audio_storage_path.mkdir(exist_ok=True)

    # Create .gitkeep files to ensure directories are tracked
    (storage_path / ".gitkeep").touch()
    (audio_storage_path / ".gitkeep").touch()

    print(f"✅ Audio storage directories created:")
    print(f"   - {storage_path.absolute()}")
    print(f"   - {audio_storage_path.absolute()}")
    print(f"📝 Audio files will be stored locally and referenced in the database")


if __name__ == "__main__":
    setup_storage_directories()
