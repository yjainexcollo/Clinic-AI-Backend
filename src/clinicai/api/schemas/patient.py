"""
Pydantic schemas for patient-related API endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, validator


def exclude_revision_id(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove revision_id from data dictionary recursively."""
    if isinstance(data, dict):
        # Remove revision_id from current level
        data = {k: v for k, v in data.items() if k != "revision_id"}
        # Recursively process nested dictionaries and lists
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = exclude_revision_id(value)
            elif isinstance(value, list):
                data[key] = [exclude_revision_id(item) if isinstance(item, dict) else item for item in value]
    return data


class RegisterPatientRequest(BaseModel):
    """Request schema for patient registration."""

    first_name: str = Field(..., min_length=1, max_length=40, description="Patient first name")
    last_name: str = Field(..., min_length=1, max_length=40, description="Patient last name")
    mobile: str = Field(..., min_length=8, max_length=16, description="Phone in E.164 format: +[country][national number]")
    age: int = Field(..., ge=0, le=120, description="Patient age (0-120)")
    gender: str = Field(..., description="Patient gender (e.g., male, female, other)")
    recently_travelled: bool = Field(False, description="Has the patient travelled recently")
    consent: bool = Field(..., description="Patient consent for data processing (must be true)")
    country: str = Field("US", description="ISO 3166-1 alpha-2 country code (default US)")

    @validator("first_name", "last_name")
    def validate_names(cls, v):
        if not v or not v.strip():
            raise ValueError("Name fields cannot be empty")
        return v.strip()

    @validator("mobile")
    def validate_mobile(cls, v):
        # Accept generic E.164 phones: +[1-9][7-14 digits]
        # (overall 8-16 chars incl '+')
        import re
        s = (v or "").strip()
        if not re.fullmatch(r"^\+[1-9]\d{7,14}$", s):
            raise ValueError("Phone must be E.164 (+ country code followed by 7-14 digits), e.g., +14155552671")
        return s

    @validator("gender")
    def validate_gender(cls, v):
        if not v or not v.strip():
            raise ValueError("Gender cannot be empty")
        return v.strip()

    @validator("consent")
    def validate_consent(cls, v):
        if v is not True:
            raise ValueError("Consent must be provided to proceed")
        return v


class RegisterPatientResponse(BaseModel):
    """Response schema for patient registration."""

    patient_id: str = Field(..., description="Generated patient ID")
    visit_id: str = Field(..., description="Generated visit ID")
    first_question: str = Field(..., description="First question for intake")
    message: str = Field(..., description="Success message")


class AnswerIntakeRequest(BaseModel):
    """Request schema for answering intake questions."""

    patient_id: str = Field(..., description="Patient ID")
    visit_id: str = Field(..., description="Visit ID")
    answer: str = Field(
        ..., min_length=1, max_length=1000, description="Answer to the question"
    )

    @validator("answer")
    def validate_answer(cls, v):
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()


class OCRQualityInfo(BaseModel):
    """OCR quality assessment information."""
    quality: str = Field(..., description="OCR quality: excellent, good, poor, or failed")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score (0.0 to 1.0)")
    extracted_text: str = Field(..., description="Text extracted from the image")
    extracted_medications: List[str] = Field(default_factory=list, description="Potential medication names found")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for better image quality")
    word_count: int = Field(..., description="Number of words extracted")
    has_medication_keywords: bool = Field(..., description="Whether medication-related keywords were found")


class AnswerIntakeResponse(BaseModel):
    """Response schema for answering intake questions."""

    next_question: Optional[str] = Field(None, description="Next question (if any)")
    is_complete: bool = Field(..., description="Whether intake is complete")
    question_count: int = Field(..., description="Current question count")
    max_questions: int = Field(..., description="Maximum questions allowed")
    completion_percent: int = Field(..., ge=0, le=100, description="LLM-assessed completion percent")
    message: str = Field(..., description="Status message")
    allows_image_upload: bool = Field(False, description="Whether the next question allows image upload")
    ocr_quality: Optional[OCRQualityInfo] = Field(None, description="OCR quality information if image was uploaded")


class EditAnswerRequest(BaseModel):
    """Request schema for editing an existing answer."""

    patient_id: str = Field(..., description="Patient ID")
    visit_id: str = Field(..., description="Visit ID")
    question_number: int = Field(..., ge=1, description="Question number to edit (1-based)")
    new_answer: str = Field(..., min_length=1, max_length=1000, description="Replacement answer")


class EditAnswerResponse(BaseModel):
    success: bool = Field(...)
    message: str = Field(...)
    next_question: Optional[str] = Field(None, description="Regenerated next question after edit")
    question_count: Optional[int] = Field(None, description="Current question count after truncation")
    max_questions: Optional[int] = Field(None, description="Max questions allowed")
    completion_percent: Optional[int] = Field(None, description="Updated completion percent")
    allows_image_upload: Optional[bool] = Field(None, description="Whether next question allows image upload")


class QuestionAnswerSchema(BaseModel):
    """Schema for question-answer pair."""

    question_id: str = Field(..., description="Question ID")
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")
    timestamp: datetime = Field(..., description="Timestamp")
    question_number: int = Field(..., description="Question number in sequence")
    attachment_image_paths: Optional[List[str]] = Field(None, description="Paths to attached images")
    ocr_texts: Optional[List[str]] = Field(None, description="OCR extracted text from images")

    class Config:
        # Exclude revision_id and other MongoDB-specific fields
        exclude = {"revision_id"}


class IntakeSummarySchema(BaseModel):
    """Schema for intake session summary."""

    visit_id: str = Field(..., description="Visit ID")
    status: str = Field(..., description="Visit status")
    questions_asked: List[QuestionAnswerSchema] = Field(
        ..., description="List of questions and answers"
    )
    total_questions: int = Field(..., description="Total questions asked")
    max_questions: int = Field(..., description="Maximum questions allowed")
    intake_status: str = Field(..., description="Intake session status")
    started_at: datetime = Field(..., description="Intake start time")
    completed_at: Optional[datetime] = Field(None, description="Intake completion time")

    class Config:
        # Exclude revision_id and other MongoDB-specific fields
        exclude = {"revision_id"}


class PatientSummarySchema(BaseModel):
    """Schema for patient summary."""

    patient_id: str = Field(..., description="Patient ID")
    name: str = Field(..., description="Patient name")
    mobile: str = Field(..., description="Mobile number")
    age: int = Field(..., description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    created_at: datetime = Field(..., description="Registration date")
    total_visits: int = Field(..., description="Total number of visits")
    latest_visit: Optional[IntakeSummarySchema] = Field(
        None, description="Latest visit details"
    )

    class Config:
        # Exclude revision_id and other MongoDB-specific fields
        exclude = {"revision_id"}


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")


# Step-02: Pre-Visit Summary Schemas
class PreVisitSummaryRequest(BaseModel):
    """Request schema for generating pre-visit summary."""

    patient_id: str = Field(..., description="Opaque Patient ID")
    visit_id: str = Field(..., description="Visit ID")


class PreVisitSummaryResponse(BaseModel):
    """Response schema for pre-visit summary."""

    patient_id: str = Field(..., description="Patient ID")
    visit_id: str = Field(..., description="Visit ID")
    summary: str = Field(..., description="Clinical summary in markdown/plain text")
    generated_at: str = Field(..., description="Summary generation timestamp")
