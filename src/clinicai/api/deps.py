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
from ..adapters.external.transcription_service_openai import OpenAITranscriptionService
# Conditional import for local Whisper - only import if needed
# from ..adapters.external.transcription_service_whisper import WhisperTranscriptionService
from ..adapters.external.soap_service_openai import OpenAISoapService
from ..adapters.services.action_plan_service import OpenAIActionPlanService
from ..application.ports.repositories.patient_repo import PatientRepository
from ..application.ports.repositories.visit_repo import VisitRepository
from ..application.ports.services.question_service import QuestionService
from ..application.ports.services.transcription_service import TranscriptionService
from ..application.ports.services.soap_service import SoapService
from ..application.ports.services.action_plan_service import ActionPlanService


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
    """Get audio repository instance."""
    return AudioRepository()


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
        from clinicai.core.config import get_settings
        settings = get_settings()
        
        # Check if we should use local Whisper or OpenAI Whisper API
        transcription_service_type = os.getenv("TRANSCRIPTION_SERVICE", "openai").lower()
        
        if transcription_service_type == "local":
            print("Using local Whisper transcription service")
            # Lazy import to avoid importing whisper if not needed
            from clinicai.adapters.external.transcription_service_whisper import WhisperTranscriptionService
            _TRANSCRIPTION_SERVICE_SINGLETON = WhisperTranscriptionService()
        else:
            print("Using OpenAI Whisper API transcription service")
            _TRANSCRIPTION_SERVICE_SINGLETON = OpenAITranscriptionService()
    
    return _TRANSCRIPTION_SERVICE_SINGLETON


@lru_cache()
def get_soap_service() -> SoapService:
    """Get SOAP service instance."""
    # In a real implementation, this would come from the DI container
    # For now, we'll create it directly
    return OpenAISoapService()


@lru_cache()
def get_action_plan_service() -> ActionPlanService:
    """Get Action Plan service instance."""
    # In a real implementation, this would come from the DI container
    # For now, we'll create it directly
    return OpenAIActionPlanService()


# Dependency annotations for FastAPI
VisitRepositoryDep = Annotated[VisitRepository, Depends(get_visit_repository)]
PatientRepositoryDep = Annotated[PatientRepository, Depends(get_patient_repository)]
AudioRepositoryDep = Annotated[AudioRepository, Depends(get_audio_repository)]
QuestionServiceDep = Annotated[QuestionService, Depends(get_question_service)]
# Removed StablePatientRepository dependency
TranscriptionServiceDep = Annotated[TranscriptionService, Depends(get_transcription_service)]
SoapServiceDep = Annotated[SoapService, Depends(get_soap_service)]
ActionPlanServiceDep = Annotated[ActionPlanService, Depends(get_action_plan_service)]
