"""FastAPI dependency providers.

Formatting-only changes; behavior preserved.
"""

import os
from functools import lru_cache
from typing import Annotated, Optional

from fastapi import Depends

from ..adapters.db.mongo.repositories.patient_repository import (
    MongoPatientRepository,
)
from ..adapters.db.mongo.repositories.visit_repository import (
    MongoVisitRepository,
)
from ..adapters.db.mongo.repositories.audio_repository import (
    AudioRepository,
)
from ..adapters.external.question_service_openai import OpenAIQuestionService
# Note: Azure Speech Service is used for transcription (not Whisper)
from ..adapters.external.soap_service_openai import OpenAISoapService
from ..application.ports.repositories.patient_repo import PatientRepository
from ..application.ports.repositories.visit_repo import VisitRepository
from ..application.ports.services.question_service import QuestionService
from ..application.ports.services.transcription_service import TranscriptionService
from ..application.ports.services.soap_service import SoapService
from ..core.auth import get_auth_service
from fastapi import Request


@lru_cache()
def get_visit_repository() -> VisitRepository:
    """Get visit repository instance."""
    return MongoVisitRepository()


@lru_cache()
def get_patient_repository() -> PatientRepository:
    """Get patient repository instance."""
    # Patient repository no longer needs visit repository dependency
    return MongoPatientRepository()


@lru_cache()
def get_audio_repository() -> AudioRepository:
    """Get audio repository instance (internal use for transcription workflow)."""
    return AudioRepository()


@lru_cache()
def get_question_service() -> QuestionService:
    """Get question service instance."""
    # In a real implementation, this would come from the DI container
    # For now, we'll create it directly
    return OpenAIQuestionService()




# Maintain a single instance of the transcription service per process
_TRANSCRIPTION_SERVICE_SINGLETON: Optional[TranscriptionService] = None

def get_transcription_service() -> TranscriptionService:
    """Get transcription service instance (singleton)."""
    global _TRANSCRIPTION_SERVICE_SINGLETON
    if _TRANSCRIPTION_SERVICE_SINGLETON is None:
        from clinicai.core.config import get_settings
        settings = get_settings()
        
        # Check which transcription service to use
        transcription_service_type = os.getenv("TRANSCRIPTION_SERVICE", "azure_speech").lower()
        
        if transcription_service_type == "azure_speech":
            print("Using Azure Speech Service transcription (batch with speaker diarization)")
            try:
                from clinicai.adapters.external.transcription_service_azure_speech import AzureSpeechTranscriptionService
                _TRANSCRIPTION_SERVICE_SINGLETON = AzureSpeechTranscriptionService()
            except ImportError as e:
                print(f"⚠️  Azure Speech Service not available: {e}")
                raise ValueError(
                    "Azure Speech Service is required. Please install required packages and configure "
                    "AZURE_SPEECH_SUBSCRIPTION_KEY and AZURE_SPEECH_REGION."
                )
            except ValueError as e:
                print(f"⚠️  Azure Speech Service configuration error: {e}")
                raise
        else:
            raise ValueError(
                f"Invalid TRANSCRIPTION_SERVICE: {transcription_service_type}. "
                "Only 'azure_speech' is supported. Please set TRANSCRIPTION_SERVICE=azure_speech"
            )
    
    return _TRANSCRIPTION_SERVICE_SINGLETON


@lru_cache()
def get_soap_service() -> SoapService:
    """Get SOAP service instance."""
    # In a real implementation, this would come from the DI container
    # For now, we'll create it directly
    return OpenAISoapService()


def get_current_user(request: Request) -> str:
    """
    Get current authenticated user ID from request state.
    
    This dependency can be used in endpoints to access the authenticated user ID.
    The authentication middleware must have run first (which it does by default).
    
    Note: This is optional - most endpoints don't need this since user_id is
    automatically logged by HIPAA audit middleware. Use this if you need to
    access user_id in your endpoint logic.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        # This should not happen if authentication middleware is working
        from fastapi import HTTPException
        raise HTTPException(
            status_code=401,
            detail="User not authenticated"
        )
    return user_id


# Dependency annotations for FastAPI
VisitRepositoryDep = Annotated[VisitRepository, Depends(get_visit_repository)]
PatientRepositoryDep = Annotated[PatientRepository, Depends(get_patient_repository)]
AudioRepositoryDep = Annotated[AudioRepository, Depends(get_audio_repository)]  # Internal use for transcription
QuestionServiceDep = Annotated[QuestionService, Depends(get_question_service)]
TranscriptionServiceDep = Annotated[TranscriptionService, Depends(get_transcription_service)]
SoapServiceDep = Annotated[SoapService, Depends(get_soap_service)]
CurrentUserDep = Annotated[str, Depends(get_current_user)]
