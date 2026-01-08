"""Patient DTOs for API communication.

Removed unused imports; behavior unchanged.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...domain.entities.patient import Patient


@dataclass
class RegisterPatientRequest:
    """Request DTO for patient registration."""

    first_name: str
    last_name: str
    mobile: str
    age: int
    gender: str
    recently_travelled: bool = False
    consent: bool = True
    country: str = "US"
    language: str = "en"


@dataclass
class RegisterPatientResponse:
    """Response DTO for patient registration."""

    patient_id: str
    visit_id: str
    first_question: str
    message: str


@dataclass
class AnswerIntakeRequest:
    """Request DTO for answering intake questions."""

    patient_id: str
    visit_id: str
    answer: str


@dataclass
class EditAnswerRequest:
    """Request DTO for editing an answer."""

    patient_id: str
    visit_id: str
    question_number: int
    new_answer: str


@dataclass
class EditAnswerResponse:
    """Response DTO for editing an answer."""

    success: bool
    message: str
    next_question: Optional[str] = None
    question_count: Optional[int] = None
    max_questions: Optional[int] = None
    completion_percent: Optional[int] = None
    allows_image_upload: Optional[bool] = None


@dataclass
class AnswerIntakeResponse:
    """Response DTO for answering intake questions."""

    next_question: Optional[str]
    is_complete: bool
    question_count: int
    max_questions: int
    completion_percent: int
    message: str
    allows_image_upload: bool = False


@dataclass
class QuestionAnswerDTO:
    """DTO for question-answer pair."""

    question_id: str
    question: str
    answer: str
    timestamp: str
    question_number: int


@dataclass
class IntakeSummaryDTO:
    """DTO for intake session summary."""

    visit_id: str
    symptom: str
    status: str
    questions_asked: List[QuestionAnswerDTO]
    total_questions: int
    max_questions: int
    intake_status: str
    started_at: str
    completed_at: Optional[str]


@dataclass
class PatientSummaryDTO:
    """DTO for patient summary."""

    patient_id: str
    name: str
    mobile: str
    age: int
    created_at: str
    total_visits: int
    latest_visit: Optional[IntakeSummaryDTO]


@dataclass
class ResolvePatientRequest:
    """Request DTO for patient resolution."""

    name: str
    mobile: str
    age: int
    symptom: str


@dataclass
class PatientResolutionResult:
    """Result DTO for patient resolution."""

    patient: Optional[Patient] = None
    candidates: Optional[List[Patient]] = None
    resolution_type: str = ""  # "exact_match", "mobile_match", "new_patient"
    action: str = ""  # "continue_existing", "select_or_create", "create_new"
    message: str = ""


@dataclass
class PatientCandidateDTO:
    """DTO for patient candidate in family member selection."""

    patient_id: str
    name: str
    age: int
    total_visits: int
    last_visit_date: Optional[str]


@dataclass
class FamilyMemberSelectionRequest:
    """Request DTO for selecting a family member."""

    selected_patient_id: str
    symptom: str


@dataclass
class FamilyMemberSelectionResponse:
    """Response DTO for family member selection."""

    patient_id: str
    visit_id: str
    first_question: str
    message: str


# Step-02: Pre-Visit Summary DTOs
@dataclass
class PreVisitSummaryRequest:
    """Request DTO for generating pre-visit summary."""

    patient_id: str
    visit_id: str


@dataclass
class PreVisitSummaryResponse:
    """Response DTO for pre-visit summary."""

    patient_id: str
    visit_id: str
    summary: str
    generated_at: str


# Step-03: Audio Transcription & SOAP Generation DTOs
@dataclass
class AudioTranscriptionRequest:
    """Request DTO for audio transcription."""

    patient_id: str
    visit_id: str
    audio_file_path: Optional[str] = None  # Optional when sas_url provided
    language: str = "en"
    audio_duration: Optional[float] = None
    # Optional: direct SAS URL to existing blob for Azure Speech (avoids re-upload)
    sas_url: Optional[str] = None
    # P1-5: Optional diarization toggle (defaults to service config if not provided)
    enable_diarization: Optional[bool] = None


# Post-Visit Summary DTOs
@dataclass
class PostVisitSummaryRequest:
    """Request DTO for generating post-visit summary."""

    patient_id: str
    visit_id: str


@dataclass
class PostVisitSummaryResponse:
    """Response DTO for post-visit summary."""

    patient_id: str
    visit_id: str
    summary: str
    generated_at: str


@dataclass
class AudioTranscriptionResponse:
    """Response DTO for audio transcription."""

    patient_id: str
    visit_id: str
    transcript: str
    word_count: int
    audio_duration: Optional[float]
    transcription_status: str
    message: str


@dataclass
class SoapGenerationRequest:
    """Request DTO for SOAP generation."""

    patient_id: str
    visit_id: str
    transcript: Optional[str] = None  # If not provided, will use stored transcript
    # Optional per-visit SOAP template guiding this generation only.
    # When None, generator falls back to default behavior.
    template: Optional[Dict[str, Any]] = None


@dataclass
class SoapGenerationResponse:
    """Response DTO for SOAP generation."""

    patient_id: str
    visit_id: str
    soap_note: Dict[str, Any]
    generated_at: str
    message: str


@dataclass
class SoapNoteDTO:
    """DTO for SOAP note data.

    soap_order reflects the doctor's preferred ordering of sections
    (e.g., ["subjective", "objective", "assessment", "plan"]).
    Frontend can use this to render sections in the desired order
    without relying on JSON key order.
    """

    subjective: str
    objective: Dict[str, Any]
    assessment: str
    plan: str
    highlights: List[str]
    red_flags: List[str]
    generated_at: str
    model_info: Optional[Dict[str, Any]]
    confidence_score: Optional[float]
    soap_order: Optional[List[str]] = None


@dataclass
class TranscriptionSessionDTO:
    """DTO for transcription session data."""

    audio_file_path: Optional[str]
    transcript: Optional[str]
    transcription_status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    audio_duration_seconds: Optional[float]
    word_count: Optional[int]
    # Optional: cached structured dialogue (ordered Doctor/Patient turns)
    structured_dialogue: Optional[List[Dict[str, Any]]] = None
