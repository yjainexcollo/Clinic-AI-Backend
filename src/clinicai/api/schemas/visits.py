"""
Visit-related API schemas.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from .common import QuestionAnswer


class TranscriptionSessionSchema(BaseModel):
    """Transcription session schema."""
    
    audio_file_path: Optional[str] = Field(None, description="Path to audio file")
    transcript: Optional[str] = Field(None, description="Full transcript text")
    transcription_status: str = Field(default="pending", description="Status: pending, processing, completed, failed")
    started_at: Optional[str] = Field(None, description="Transcription start timestamp")
    completed_at: Optional[str] = Field(None, description="Transcription completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if transcription failed")
    audio_duration_seconds: Optional[float] = Field(None, description="Audio duration in seconds")
    word_count: Optional[int] = Field(None, description="Word count in transcript")
    structured_dialogue: Optional[List[Dict[str, Any]]] = Field(None, description="Structured doctor/patient dialogue")


class SoapNoteSchema(BaseModel):
    """SOAP note schema."""
    
    subjective: str = Field(..., description="Subjective findings")
    objective: Dict[str, Any] = Field(..., description="Objective findings and vitals")
    assessment: str = Field(..., description="Assessment and diagnosis")
    plan: str = Field(..., description="Treatment plan")
    highlights: List[str] = Field(default_factory=list, description="Key highlights")
    red_flags: List[str] = Field(default_factory=list, description="Red flag symptoms")
    generated_at: str = Field(..., description="SOAP note generation timestamp")
    model_info: Optional[Dict[str, Any]] = Field(None, description="AI model information")
    confidence_score: Optional[float] = Field(None, description="Confidence score")


class VisitListItemSchema(BaseModel):
    """Schema for visit list item (summary)."""
    
    visit_id: str = Field(..., description="Visit ID")
    symptom: str = Field(..., description="Primary symptom")
    workflow_type: str = Field(..., description="Workflow type: scheduled or walk_in")
    status: str = Field(..., description="Visit status")
    created_at: datetime = Field(..., description="Visit creation date")
    updated_at: datetime = Field(..., description="Visit last update date")
    has_transcript: bool = Field(False, description="Whether transcript is available")
    has_soap: bool = Field(False, description="Whether SOAP note is available")
    has_vitals: bool = Field(False, description="Whether vitals are available")
    has_pre_visit_summary: bool = Field(False, description="Whether pre-visit summary is available")
    has_post_visit_summary: bool = Field(False, description="Whether post-visit summary is available")


class VisitDetailSchema(BaseModel):
    """Schema for full visit details."""
    
    visit_id: str = Field(..., description="Visit ID")
    patient_id: str = Field(..., description="Patient ID")
    symptom: str = Field(..., description="Primary symptom")
    workflow_type: str = Field(..., description="Workflow type: scheduled or walk_in")
    status: str = Field(..., description="Visit status")
    created_at: datetime = Field(..., description="Visit creation date")
    updated_at: datetime = Field(..., description="Visit last update date")
    
    # Intake session (if scheduled workflow)
    intake_session: Optional[Dict[str, Any]] = Field(None, description="Intake session data")
    
    # Pre-visit summary
    pre_visit_summary: Optional[Dict[str, Any]] = Field(None, description="Pre-visit summary")
    
    # Transcription
    transcription_session: Optional[TranscriptionSessionSchema] = Field(None, description="Transcription session data")
    
    # SOAP note
    soap_note: Optional[SoapNoteSchema] = Field(None, description="SOAP note")
    
    # Vitals
    vitals: Optional[Dict[str, Any]] = Field(None, description="Vitals data")
    
    # Post-visit summary
    post_visit_summary: Optional[Dict[str, Any]] = Field(None, description="Post-visit summary")
    
    # Audio files
    audio_files: List[Dict[str, Any]] = Field(default_factory=list, description="Associated audio files")


class VisitListResponse(BaseModel):
    """Response schema for visit list endpoint."""
    
    visits: List[VisitListItemSchema] = Field(..., description="List of visits")
    total: int = Field(..., description="Total number of visits")

