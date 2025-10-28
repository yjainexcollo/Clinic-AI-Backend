"""
Patient registration schemas.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator

from .common import PersonalInfo, ContactInfo


class RegisterPatientRequest(PersonalInfo, ContactInfo):
    """Request schema for patient registration."""
    
    recently_travelled: bool = Field(False, description="Has the patient travelled recently")
    consent: bool = Field(..., description="Patient consent for data processing (must be true)")
    country: str = Field("US", description="ISO 3166-1 alpha-2 country code (default US)")
    language: str = Field("en", description="Preferred language (en for English, es for Spanish)")
    
    @validator("language")
    def validate_language(cls, v):
        if v not in ["en", "es"]:
            raise ValueError("Language must be 'en' (English) or 'es' (Spanish)")
        return v
    
    @validator("consent")
    def validate_consent(cls, v):
        if v is not True:
            raise ValueError("Consent must be provided to proceed")
        return v
    
    @validator("country")
    def validate_country(cls, v):
        if len(v) != 2 or not v.isalpha():
            raise ValueError("Country must be a 2-letter ISO code")
        return v.upper()


class RegisterPatientResponse(BaseModel):
    """Response schema for patient registration."""
    
    patient_id: str = Field(..., description="Generated patient ID")
    visit_id: str = Field(..., description="Generated visit ID")
    first_question: str = Field(..., description="First question for intake")
    message: str = Field(..., description="Success message")


class PatientSummarySchema(BaseModel):
    """Schema for patient summary."""
    
    patient_id: str = Field(..., description="Patient ID")
    name: str = Field(..., description="Patient name")
    mobile: str = Field(..., description="Mobile number")
    age: int = Field(..., description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    created_at: datetime = Field(..., description="Registration date")
    total_visits: int = Field(..., description="Total number of visits")
    latest_visit_status: Optional[str] = Field(None, description="Latest visit status")
    
    class Config:
        # Exclude revision_id and other MongoDB-specific fields
        exclude = {"revision_id"}
