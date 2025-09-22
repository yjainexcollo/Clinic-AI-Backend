"""
OpenAI Whisper API-based transcription service.
Fits low-memory Cloud Run by offloading STT to OpenAI.
"""

import os
from typing import Dict, Any
from pathlib import Path

from openai import OpenAI

from clinicai.application.ports.services.transcription_service import TranscriptionService
from clinicai.core.config import get_settings


class OpenAITranscriptionService(TranscriptionService):
    def __init__(self) -> None:
        self._settings = get_settings()
        api_key = self._settings.openai.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)

    async def transcribe_audio(
        self,
        audio_file_path: str,
        language: str = "en",
        medical_context: bool = True,
    ) -> Dict[str, Any]:
        # OpenAI v1 SDK
        # Whisper models: "whisper-1" or newer transcription-capable models
        # Note: duration/word_count aren't returned; provide best-effort fields
        try:
            with open(audio_file_path, "rb") as f:
                resp = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language or "en",
                    # prompt could be set based on medical_context
                )
            text = (resp.text or "").strip()
            return {
                "transcript": text,
                "confidence": 0.0,  # not provided by API
                "duration": None,    # unknown
                "word_count": len(text.split()) if text else 0,
                "language": language,
                "model": "openai-whisper-1",
            }
        except Exception as e:
            raise ValueError(f"Transcription failed: {e}")

    async def validate_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        try:
            p = Path(audio_file_path)
            if not p.exists() or not p.is_file():
                return {"is_valid": False, "error": "Audio file not found", "file_size": 0, "duration": 0}
            file_size = p.stat().st_size
            if file_size <= 0:
                return {"is_valid": False, "error": "Empty file", "file_size": 0, "duration": 0}
            # Basic 25–50MB check; OpenAI supports larger but keep user-friendly constraints
            if file_size > 50 * 1024 * 1024:
                return {
                    "is_valid": False,
                    "error": f"Audio file too large ({file_size / (1024*1024):.1f}MB, max 50MB)",
                    "file_size": file_size,
                    "duration": 0,
                }
            return {
                "is_valid": True,
                "error": None,
                "file_size": file_size,
                "duration": 0,
                "format": p.suffix.lower().lstrip("."),
            }
        except Exception as e:
            return {"is_valid": False, "error": f"Validation error: {e}", "file_size": 0, "duration": 0}


