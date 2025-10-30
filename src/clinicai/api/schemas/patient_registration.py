"""
Patient registration schemas.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator
import re

from .common import PersonalInfo, ContactInfo


class RegisterPatientRequest(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=40, description="First name")
    last_name: str = Field(..., min_length=1, max_length=40, description="Last name")
    mobile: str = Field(..., pattern=r"^(\\+|)[0-9]{8,16}$", description="Mobile phone number (E.164 or local)")
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(...)
    recently_travelled: bool = Field(False)
    consent: bool = Field(..., description="Must be true")
    country: str = Field("US", min_length=2, max_length=2)
    language: str = Field("en", pattern=r"^(en|es)$")

    @validator("first_name", "last_name", pre=True)
    def validate_names(cls, v):
        s = v.strip() if isinstance(v, str) else v
        if not s:
            raise ValueError("First and last names cannot be blank")
        return re.sub(r"[\x00-\x1F]+", "", s)[:40]

    @validator("country", pre=True)
    def validate_country(cls, v):
        s = v.strip().upper()
        if len(s) != 2 or not s.isalpha():
            raise ValueError("country must be ISO alpha-2 code")
        return s

    @validator("mobile")
    def validate_mobile(cls, v):
        if not re.match(r"^(\\+|)[0-9]{8,16}$", v):
            raise ValueError("Invalid mobile format")
        return v.strip()

    @validator("consent")
    def validate_consent(cls, v):
        if v is not True:
            raise ValueError("Consent must be True")
        return v


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
