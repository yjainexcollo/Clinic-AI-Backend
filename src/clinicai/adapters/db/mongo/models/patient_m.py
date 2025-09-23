"""
MongoDB Beanie models used by the persistence layer.

Note: These are persisted documents; structure left unchanged to preserve runtime behavior.
"""

from datetime import datetime
from typing import List, Optional

from beanie import Document
from pydantic import BaseModel, Field


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

    class Config:
        # Exclude revision_id and other MongoDB-specific fields
        exclude = {"revision_id"}


class TranscriptionSessionMongo(BaseModel):
    """Embedded model for transcription session (no revision_id)."""
    audio_file_path: Optional[str] = Field(None, description="Path to audio file")
    transcript: Optional[str] = Field(None, description="Transcribed text")
    transcription_status: str = Field(default="pending", description="Status: pending, processing, completed, failed")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = Field(None, description="Error message if failed")
    audio_duration_seconds: Optional[float] = Field(None, description="Audio duration in seconds")
    word_count: Optional[int] = Field(None, description="Word count of transcript")


class SoapNoteMongo(BaseModel):
    """Embedded model for SOAP note (no revision_id)."""
    subjective: str = Field(..., description="Subjective section")
    objective: str = Field(..., description="Objective section")
    assessment: str = Field(..., description="Assessment section")
    plan: str = Field(..., description="Plan section")
    highlights: List[str] = Field(default_factory=list, description="Key highlights")
    red_flags: List[str] = Field(default_factory=list, description="Red flags")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_info: Optional[dict] = Field(None, description="Model information")
    confidence_score: Optional[float] = Field(None, description="Confidence score")


class VisitMongo(Document):
    """MongoDB model for visit."""
    visit_id: str = Field(..., description="Visit ID")
    patient_id: str = Field(..., description="Patient ID reference")
    status: str = Field(
        default="intake"
    )  # intake, transcription, soap_generation, prescription_analysis, completed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Step 1: Pre-Visit Intake
    intake_session: Optional[IntakeSessionMongo] = None
    
    # Step 2: Pre-Visit Summary (EHR Storage)
    pre_visit_summary: Optional[dict] = None
    
    # Step 3: Audio Transcription & SOAP Generation
    transcription_session: Optional[TranscriptionSessionMongo] = None
    soap_note: Optional[SoapNoteMongo] = None

    class Config:
        # Exclude revision_id and other MongoDB-specific fields when serializing
        exclude = {"revision_id"}



class PatientMongo(Document):
    """MongoDB model for Patient entity."""

    patient_id: str = Field(..., description="Patient ID", unique=True)
    name: str = Field(..., description="Patient name")
    mobile: str = Field(..., description="Mobile number")
    age: int = Field(..., description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    recently_travelled: bool = Field(default=False, description="Has the patient travelled recently")
    visits: List[VisitMongo] = Field(default_factory=list, description="List of visits")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        # Exclude revision_id and other MongoDB-specific fields when serializing
        exclude = {"revision_id"}


    class Settings:
        name = "patients"
        indexes = [
            "patient_id",
            "name",
            "mobile",
            "created_at"
        ]


class MedicationImageMongo(Document):
    """MongoDB model for uploaded medication images linked to a visit."""
    patient_id: str = Field(..., description="Patient ID reference")
    visit_id: str = Field(..., description="Visit ID reference")
    image_data: bytes = Field(..., description="Base64 encoded image data")
    content_type: str = Field(..., description="MIME type of the image")
    filename: str = Field(..., description="Original filename")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "medication_images"
        indexes = ["patient_id", "visit_id", "uploaded_at"]