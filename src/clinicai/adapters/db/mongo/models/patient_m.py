"""
MongoDB Beanie models used by the persistence layer.

Note: These are persisted documents; structure left unchanged to preserve runtime behavior.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from beanie import Document
from pydantic import BaseModel, Field, validator

from clinicai.domain.enums.workflow import VisitWorkflowType
import json


class QuestionAnswerMongo(BaseModel):
    """MongoDB model for question-answer pair."""
    question_id: str = Field(..., description="Question ID")
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    question_number: int = Field(..., description="Question number in sequence")

    class Config:
        # Exclude revision_id and other MongoDB-specific fields
        exclude = {"revision_id"}


class IntakeSessionMongo(BaseModel):
    """MongoDB model for intake session."""
    questions_asked: List[QuestionAnswerMongo] = Field(default_factory=list)
    current_question_count: int = Field(default=0)
    max_questions: int = Field(default=12)
    status: str = Field(default="in_progress")  # in_progress, completed, cancelled
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    pending_question: Optional[str] = Field(None, description="Pending next question to ask")
    travel_questions_count: int = Field(default=0, description="Number of travel-related questions asked")
    asked_categories: List[str] = Field(default_factory=list, description="Categories asked in strict sequence")
    class Config:
        # Exclude revision_id and other MongoDB-specific fields
        exclude = {"revision_id"}


class TranscriptionSessionMongo(BaseModel):
    """Embedded model for transcription session (no revision_id)."""
    audio_file_path: Optional[str] = Field(None, description="Path to audio file")
    transcript: Optional[str] = Field(None, description="Transcribed text")
    transcription_status: str = Field(default="pending", description="Status: pending, queued, processing, completed, failed")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = Field(None, description="Error message if failed")
    worker_id: Optional[str] = Field(None, description="Worker that claimed/processed this job (format: hostname:pid)")
    audio_duration_seconds: Optional[float] = Field(None, description="Audio duration in seconds")
    word_count: Optional[int] = Field(None, description="Word count of transcript")
    structured_dialogue: Optional[list[dict]] = Field(None, description="Ordered Doctor/Patient turns")
    transcription_id: Optional[str] = Field(None, description="Azure Speech Service transcription job ID for tracking")
    last_poll_status: Optional[str] = Field(None, description="Last polled status from Azure Speech Service (Succeeded, Running, Failed, etc.)")
    last_poll_at: Optional[datetime] = Field(None, description="Timestamp of last status poll from Azure Speech Service")
    # Observability timestamps for latency analysis
    enqueued_at: Optional[datetime] = Field(None, description="When job was enqueued to Azure Queue")
    dequeued_at: Optional[datetime] = Field(None, description="When worker dequeued the job")
    azure_job_created_at: Optional[datetime] = Field(None, description="When Azure Speech job was created")
    first_poll_at: Optional[datetime] = Field(None, description="When first status poll was made")
    results_downloaded_at: Optional[datetime] = Field(None, description="When transcription results were downloaded")
    db_saved_at: Optional[datetime] = Field(None, description="When transcript was saved to database")
    # Audio normalization metadata
    normalized_audio: Optional[bool] = Field(None, description="Whether audio was normalized/converted")
    original_content_type: Optional[str] = Field(None, description="Original content type before normalization")
    normalized_format: Optional[str] = Field(None, description="Format after normalization (e.g., wav_16khz_mono_pcm)")
    file_content_type: Optional[str] = Field(None, description="Final content type used for transcription")


class SoapNoteMongo(BaseModel):
    """Embedded model for SOAP note (no revision_id)."""
    subjective: str = Field(..., description="Subjective section")
    objective: dict = Field(..., description="Objective section (structured object)")
    assessment: str = Field(..., description="Assessment section")
    plan: str = Field(..., description="Plan section")
    highlights: List[str] = Field(default_factory=list, description="Key highlights")
    red_flags: List[str] = Field(default_factory=list, description="Red flags")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_info: Optional[dict] = Field(None, description="Model information")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    
    @validator('objective', pre=True)
    def validate_objective(cls, v):
        """Convert string objective to dict if needed."""
        if isinstance(v, str):
            try:
                # Try to parse as JSON if it looks like a dict string
                if v.strip().startswith('{') and v.strip().endswith('}'):
                    return json.loads(v)
                else:
                    # If it's not JSON, create a basic structure
                    return {
                        "vital_signs": {},
                        "physical_exam": {"general_appearance": v or "Not discussed"}
                    }
            except:
                # If parsing fails, create a basic structure
                return {
                    "vital_signs": {},
                    "physical_exam": {"general_appearance": v or "Not discussed"}
                }
        elif not isinstance(v, dict):
            # If it's neither string nor dict, create a basic structure
            return {
                "vital_signs": {},
                "physical_exam": {"general_appearance": "Not discussed"}
            }
        return v


class VisitMongo(Document):
    """MongoDB model for visit."""
    visit_id: str = Field(..., description="Visit ID", unique=True)
    patient_id: str = Field(..., description="Patient ID reference")
    symptom: str = Field(default="", description="Primary symptom / chief complaint for this visit")
    workflow_type: str = Field(default=VisitWorkflowType.SCHEDULED, description="Workflow type: scheduled or walk_in")
    status: str = Field(
        default="intake"
    )  # intake, transcription, soap_generation, prescription_analysis, completed, walk_in_patient
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Visit-specific travel history (moved from Patient - travel is visit-specific, not lifetime patient attribute)
    recently_travelled: bool = Field(default=False, description="Has the patient travelled recently for this visit")

    # Step 1: Pre-Visit Intake
    intake_session: Optional[IntakeSessionMongo] = None
    
    # Step 2: Pre-Visit Summary (EHR Storage)
    pre_visit_summary: Optional[dict] = None
    
    # Step 3: Audio Transcription & SOAP Generation
    transcription_session: Optional[TranscriptionSessionMongo] = None
    soap_note: Optional[SoapNoteMongo] = None
    vitals: Optional[dict] = None
    
    # Step 4: Post-Visit Summary (Patient Sharing)
    post_visit_summary: Optional[dict] = None

    class Config:
        # Exclude revision_id and other MongoDB-specific fields when serializing
        exclude = {"revision_id"}

    class Settings:
        name = "visits"
        indexes = [
            "visit_id",
            "patient_id", 
            "status",
            "workflow_type",
            "created_at",
            [("patient_id", 1), ("created_at", -1)],  # Compound index for patient visits ordered by date
            [("status", 1), ("created_at", -1)],  # Compound index for visits by status
            [("workflow_type", 1), ("status", 1)]  # Compound index for workflow type and status
        ]



class PatientMongo(Document):
    """MongoDB model for Patient entity."""

    patient_id: str = Field(..., description="Patient ID", unique=True)
    name: str = Field(..., description="Patient name")
    mobile: str = Field(..., description="Mobile number")
    age: int = Field(..., description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    # recently_travelled removed - now stored on Visit (travel is visit-specific)
    language: str = Field(default="en", description="Patient preferred language (en for English, es for Spanish)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        # Exclude revision_id and other MongoDB-specific fields when serializing
        exclude = {"revision_id"}


    class Settings:
        name = "patients"
        indexes = [
            "patient_id",                           # Keep for direct ID lookups
            [("mobile", 1), ("name", 1)],          # Compound index: mobile first, then name (optimized for find_by_name_and_mobile)
            "created_at",                           # Keep for date-based queries
            [("mobile", 1), ("created_at", -1)]    # Optional: for mobile + date queries (family members by date)
        ]


class MedicationImageMongo(Document):
    """MongoDB model for uploaded medication images with blob storage reference."""
    patient_id: str = Field(..., description="Patient ID reference")
    visit_id: str = Field(..., description="Visit ID reference")
    content_type: str = Field(..., description="MIME type of the image")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    
    # Blob storage reference
    blob_reference_id: str = Field(..., description="Reference to blob file storage")
    
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "medication_images"
        indexes = ["patient_id", "visit_id", "uploaded_at"]


class AudioFileMongo(Document):
    """MongoDB model for audio files with blob storage reference (internal use for transcription)."""
    audio_id: str = Field(..., description="Unique audio file ID", unique=True)
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the audio file")
    file_size: int = Field(..., description="File size in bytes")
    duration_seconds: Optional[float] = Field(None, description="Audio duration in seconds")
    
    # Blob storage reference (optional for backward compatibility with legacy records)
    blob_reference_id: Optional[str] = Field(None, description="Reference to blob file storage")
    
    # Metadata
    patient_id: Optional[str] = Field(None, description="Patient ID if linked to a patient")
    visit_id: Optional[str] = Field(None, description="Visit ID if linked to a visit")
    adhoc_id: Optional[str] = Field(None, description="Adhoc transcript ID if linked to adhoc transcript")
    audio_type: str = Field(default="visit", description="Type: visit or other (adhoc deprecated)")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "audio_files"
        indexes = [
            "audio_id",
            "patient_id", 
            "visit_id",
            "audio_type",
            "created_at"
        ]


class DoctorPreferencesMongo(Document):
    """MongoDB model for storing doctor preferences (standalone; independent of intake)."""

    # Legacy intake preferences (must remain for backward compatibility)
    doctor_id: str = Field(..., description="Doctor ID", unique=True)
    global_categories: list[str] = Field(default_factory=list)
    selected_categories: list[str] = Field(default_factory=list)
    max_questions: int = Field(default=12)

    # New optional preference fields (stored as plain JSON-serializable structures for flexibility)
    # AI configs: simple dicts; validated and defaulted at service/API layer
    pre_visit_ai_config: Optional[dict] = None
    soap_ai_config: Optional[dict] = None
    # Frontend configuration: order + per-section config (stored as list[dict])
    soap_order: Optional[list[str]] = None
    pre_visit_config: Optional[list[dict]] = None

    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "doctor_preferences"
        indexes = ["doctor_id", "updated_at"]


class LLMInteractionMongo(Document):
    """MongoDB document for logging raw LLM interactions for debugging/QA."""

    agent_name: str = Field(..., description="Logical agent name (e.g., agent1_medical_context)")
    visit_id: Optional[str] = Field(None, description="Visit ID (if known at log time)")
    patient_id: Optional[str] = Field(None, description="Patient ID (if known at log time)")
    system_prompt: str = Field(..., description="System prompt sent to the LLM")
    user_prompt: str = Field(..., description="User prompt or payload sent to the LLM")
    response_text: str = Field(..., description="Raw LLM response text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Normalized structured metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Log creation timestamp")

    class Settings:
        name = "llm_interactions"


# ============================================================================
# LLM Interaction Logging Models (Per-Visit Structured Logging)
# ============================================================================

class AgentLog(BaseModel):
    """Model for individual agent interaction log within an intake question."""
    agent_name: str = Field(..., description="Agent name (e.g., agent1_medical_context, agent2_extractor, agent3_question_generator)")
    user_prompt: Any = Field(..., description="User prompt sent to the LLM")
    response_text: Any = Field(..., description="Response text from the LLM")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata (prompt_version, etc.)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this agent call was made")


class IntakeQuestionLog(BaseModel):
    """Model for logging all agents involved in generating one intake question."""
    question_number: int = Field(..., description="Question number in sequence")
    question_id: Optional[str] = Field(None, description="Question ID")
    question_text: Optional[str] = Field(None, description="The actual question text")
    asked_at: datetime = Field(default_factory=datetime.utcnow, description="When the question was asked")
    agents: List[AgentLog] = Field(default_factory=list, description="List of agent interactions for this question")


class LLMCallLog(BaseModel):
    """Model for LLM call logs in other phases (previsit, soap, postvisit)."""
    agent_name: str = Field(..., description="Agent name or phase identifier")
    user_prompt: Any = Field(..., description="User prompt sent to the LLM")
    response_text: Any = Field(..., description="Response text from the LLM")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this call was made")


class LLMInteractionVisit(Document):
    """MongoDB document for structured per-visit LLM interaction logging."""
    visit_id: str = Field(..., description="Visit ID", unique=True)
    patient_id: str = Field(..., description="Patient ID")
    
    # Intake phase: stores all agent interactions per question (direct list, no wrapper)
    intake: List[IntakeQuestionLog] = Field(default_factory=list, description="Intake phase logs - list of questions with agent logs")
    
    # Other phases: stores single LLM call (only generated once per visit)
    pre_visit_summary: Optional[LLMCallLog] = Field(None, description="Pre-visit summary phase log")
    soap: Optional[LLMCallLog] = Field(None, description="SOAP generation phase log")
    post_visit_summary: Optional[LLMCallLog] = Field(None, description="Post-visit summary phase log")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    class Settings:
        name = "llm_interaction"
        indexes = [
            "visit_id",
            "patient_id",
            "updated_at",
            [("visit_id", 1), ("updated_at", -1)],
        ]