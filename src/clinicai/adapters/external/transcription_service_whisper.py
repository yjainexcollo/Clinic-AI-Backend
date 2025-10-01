"""
Whisper-based transcription service implementation.
"""

import asyncio
import os
import tempfile
from typing import Dict, Any, Optional
import whisper
from pathlib import Path

from clinicai.application.ports.services.transcription_service import TranscriptionService
from clinicai.core.config import get_settings


class WhisperTranscriptionService(TranscriptionService):
    """Whisper implementation of TranscriptionService."""

    def __init__(self):
        self._settings = get_settings()
        self._model = None
        self._download_root = None  # cache disabled per deployment request

    def _get_model(self):
        """Lazy load the Whisper model only when needed."""
        if self._model is None:
            print(f"Loading Whisper model: {self._settings.whisper.model}")
            # Load without explicit download_root (let whisper handle temp cache)
            self._model = whisper.load_model(self._settings.whisper.model)
            print("Whisper model loaded successfully")
        return self._model

    async def transcribe_audio(
        self, 
        audio_file_path: str,
        language: str = None,
        medical_context: bool = None
    ) -> Dict[str, Any]:
        """Transcribe audio file using Whisper."""
        try:
            # Use configuration defaults if not provided
            if language is None:
                language = self._settings.whisper.language
            if medical_context is None:
                medical_context = self._settings.whisper.medical_context
                
            # Map our language codes to Whisper language codes
            language_map = {
                "en": "en",
                "sp": "es",  # Spanish
            }
            whisper_language = language_map.get(language, "en")
            
            # Debug logging
            import logging
            logger = logging.getLogger("clinicai")
            logger.info(f"WhisperTranscriptionService: Input language='{language}', mapped to whisper_language='{whisper_language}'")
                
            # Run Whisper transcription in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._transcribe_sync,
                audio_file_path,
                whisper_language,
                medical_context
            )
            
            return {
                "transcript": result["text"],
                "confidence": result.get("confidence", 0.0),
                "duration": result.get("duration", 0.0),
                "word_count": len(result["text"].split()),
                "language": language,
                "model": "whisper-base"
            }
            
        except Exception as e:
            raise ValueError(f"Transcription failed: {str(e)}")

    def _transcribe_sync(
        self, 
        audio_file_path: str, 
        language: str, 
        medical_context: bool
    ) -> Dict[str, Any]:
        """Synchronous transcription method."""
        # Configure Whisper options
        options = {
            "language": language,
            "fp16": False,  # Use fp32 for better accuracy
            "verbose": False
        }
        
        # Add medical context if requested
        if medical_context:
            # You can add medical-specific prompts or fine-tuning here
            pass
        
        # Transcribe using lazy-loaded model
        model = self._get_model()
        result = model.transcribe(audio_file_path, **options)
        
        return {
            "text": result["text"].strip(),
            "confidence": 0.9,  # Whisper doesn't provide confidence scores directly
            "duration": result.get("duration", 0.0)
        }

    async def validate_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """Validate audio file format and quality."""
        try:
            if not os.path.exists(audio_file_path):
                return {
                    "is_valid": False,
                    "error": "Audio file not found",
                    "file_size": 0,
                    "duration": 0
                }

            file_size = os.path.getsize(audio_file_path)
            
            # Check file size (max 25MB for Whisper)
            if file_size > 25 * 1024 * 1024:
                return {
                    "is_valid": False,
                    "error": f"Audio file too large ({file_size / (1024*1024):.1f}MB, max 25MB)",
                    "file_size": file_size,
                    "duration": 0
                }

            # Check file extension (include browser-recorded WebM)
            valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mpeg', '.mpg', '.webm']
            file_ext = Path(audio_file_path).suffix.lower()
            
            if file_ext not in valid_extensions:
                return {
                    "is_valid": False,
                    "error": f"Unsupported audio format: {file_ext}",
                    "file_size": file_size,
                    "duration": 0
                }

            # Basic validation passed
            return {
                "is_valid": True,
                "error": None,
                "file_size": file_size,
                "duration": 0,  # Will be determined during transcription
                "format": file_ext
            }

        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Validation error: {str(e)}",
                "file_size": 0,
                "duration": 0
            }
