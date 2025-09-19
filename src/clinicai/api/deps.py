"""FastAPI dependency providers.

Formatting-only changes; behavior preserved.
"""

from functools import lru_cache
from typing import Annotated, Optional

from fastapi import Depends

from clinicai.adapters.db.mongo.repositories.patient_repository import (
    MongoPatientRepository,
)
from clinicai.adapters.external.question_service_openai import OpenAIQuestionService
from clinicai.adapters.external.transcription_service_whisper import WhisperTranscriptionService
from clinicai.adapters.external.soap_service_openai import OpenAISoapService
from clinicai.application.ports.repositories.patient_repo import PatientRepository
from clinicai.application.ports.services.question_service import QuestionService
from clinicai.application.ports.services.transcription_service import TranscriptionService
from clinicai.application.ports.services.soap_service import SoapService


@lru_cache()
def get_patient_repository() -> PatientRepository:
    """Get patient repository instance."""
    # In a real implementation, this would come from the DI container
    # For now, we'll create it directly
    return MongoPatientRepository()


@lru_cache()
def get_question_service() -> QuestionService:
    """Get question service instance."""
    # In a real implementation, this would come from the DI container
    # For now, we'll create it directly
    return OpenAIQuestionService()




# Maintain a single instance of the transcription service per process so the
# Whisper model is loaded only once and reused across requests.
_TRANSCRIPTION_SERVICE_SINGLETON: Optional[TranscriptionService] = None

def get_transcription_service() -> TranscriptionService:
    """Get transcription service instance (singleton)."""
    global _TRANSCRIPTION_SERVICE_SINGLETON
    if _TRANSCRIPTION_SERVICE_SINGLETON is None:
        _TRANSCRIPTION_SERVICE_SINGLETON = WhisperTranscriptionService()
    return _TRANSCRIPTION_SERVICE_SINGLETON


@lru_cache()
def get_soap_service() -> SoapService:
    """Get SOAP service instance."""
    # In a real implementation, this would come from the DI container
    # For now, we'll create it directly
    return OpenAISoapService()


# Dependency annotations for FastAPI
PatientRepositoryDep = Annotated[PatientRepository, Depends(get_patient_repository)]
QuestionServiceDep = Annotated[QuestionService, Depends(get_question_service)]
# Removed StablePatientRepository dependency
TranscriptionServiceDep = Annotated[TranscriptionService, Depends(get_transcription_service)]
SoapServiceDep = Annotated[SoapService, Depends(get_soap_service)]
