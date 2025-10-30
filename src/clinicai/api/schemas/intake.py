"""
Intake session schemas for patient Q&A.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import re

from .common import QuestionAnswer


class AnswerIntakeRequest(BaseModel):
    """Request schema for answering intake questions."""
    
    patient_id: str = Field(..., min_length=10, max_length=50)
    visit_id: str = Field(..., min_length=10, max_length=50)
    answer: str = Field(..., min_length=1, max_length=1000)
    
    @validator("answer", pre=True)
    def sanitize_answer(cls, v):
        s = v.strip() if isinstance(v, str) else v
        if not s:
            raise ValueError("Answer cannot be blank")
        return re.sub(r"[\x00-\x1F]+", "", s)[:1000]
    
    @validator("patient_id")
    def validate_patient_id(cls, v):
        if not v or len(v) < 5:
            raise ValueError("Invalid patient ID format")
        return v
    
    @validator("visit_id")
    def validate_visit_id(cls, v):
        if not v or len(v) < 5:
            raise ValueError("Invalid visit ID format")
        return v


class AnswerIntakeResponse(BaseModel):
    """Response schema for answering intake questions."""
    
    next_question: Optional[str] = Field(None, description="Next question (if any)")
    is_complete: bool = Field(..., description="Whether intake is complete")
    question_count: int = Field(..., description="Current question count")
    max_questions: int = Field(..., description="Maximum questions allowed")
    completion_percent: int = Field(..., ge=0, le=100, description="LLM-assessed completion percent")
    message: str = Field(..., description="Status message")
    allows_image_upload: bool = Field(False, description="Whether the next question allows image upload")


class EditAnswerRequest(BaseModel):
    """Request schema for editing an existing answer."""
    
    patient_id: str = Field(..., min_length=10, max_length=50)
    visit_id: str = Field(..., min_length=10, max_length=50)
    question_number: int = Field(..., ge=1)
    new_answer: str = Field(..., min_length=1, max_length=1000)

    @validator("new_answer", pre=True)
    def sanitize_new_answer(cls, v):
        s = v.strip() if isinstance(v, str) else v
        if not s:
            raise ValueError("New answer cannot be blank")
        return re.sub(r"[\x00-\x1F]+", "", s)[:1000]
    
    @validator("patient_id")
    def validate_patient_id(cls, v):
        if not v or len(v) < 5:
            raise ValueError("Invalid patient ID format")
        return v
    
    @validator("visit_id")
    def validate_visit_id(cls, v):
        if not v or len(v) < 5:
            raise ValueError("Invalid visit ID format")
        return v


class EditAnswerResponse(BaseModel):
    """Response schema for editing answers."""
    
    success: bool = Field(...)
    message: str = Field(...)
    next_question: Optional[str] = Field(None, description="Regenerated next question after edit")
    question_count: Optional[int] = Field(None, description="Current question count after truncation")
    max_questions: Optional[int] = Field(None, description="Max questions allowed")
    completion_percent: Optional[int] = Field(None, description="Updated completion percent")
    allows_image_upload: Optional[bool] = Field(None, description="Whether next question allows image upload")


class IntakeSummarySchema(BaseModel):
    """Schema for intake session summary."""
    
    visit_id: str = Field(..., description="Visit ID")
    status: str = Field(..., description="Visit status")
    questions_asked: List[QuestionAnswer] = Field(
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
