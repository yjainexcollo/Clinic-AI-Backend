"""
Transcription service interface for audio-to-text conversion.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class TranscriptionService(ABC):
    """Abstract service for audio transcription."""

    @abstractmethod
    async def transcribe_audio(
        self, 
        audio_file_path: str,
        language: str = "en",
        medical_context: bool = True,
        sas_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (default: "en")
            medical_context: Whether to optimize for medical terminology
            
        Returns:
            Dict containing transcript, confidence, duration, word_count
        """
        pass

    @abstractmethod
    async def validate_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Validate audio file format and quality.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dict containing validation results and metadata
        """
        pass
