"""
Patient registration schemas.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import re

from .common import PersonalInfo, ContactInfo


class RegisterPatientRequest(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=40, description="First name")
    last_name: str = Field(..., min_length=1, max_length=40, description="Last name")
    mobile: str = Field(..., description="Mobile phone number (E.164 or local format)")
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(...)
    recently_travelled: bool = Field(False)
    consent: bool = Field(..., description="Must be true")
    country: str = Field("US", min_length=2, max_length=2)
    language: str = Field("en", pattern=r"^(en|es|sp)$")
    
    @validator("language", pre=True)
    def normalize_language(cls, v):
        """Normalize language codes: 'es' -> 'sp' for consistency with frontend."""
        if v and isinstance(v, str):
            normalized = v.lower().strip()
            # Map 'es' to 'sp' for consistency with frontend LanguageContext
            if normalized == "es":
                return "sp"
            if normalized in ["en", "sp"]:
                return normalized
        return "en"  # Default to English

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
        """Validate mobile number - supports E.164 format (+country code) or local format (8-16 digits)."""
        s = (v or "").strip()
        # E.164 format: + followed by 1-3 digit country code, then 7-14 digits (total 8-15 digits after +)
        # Examples: +1234567890, +18983492384, +447911123456
        if re.fullmatch(r"^\+[1-9]\d{7,14}$", s):
            return s
        # Local format: 8-16 digits without country code
        if re.fullmatch(r"^\d{8,16}$", s):
            return s
        raise ValueError("Phone must be E.164 format (+country code followed by 7-14 digits) or 8-16 local digits")

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


class LatestVisitInfo(BaseModel):
    """Schema for latest visit information."""
    
    visit_id: str = Field(..., description="Visit ID")
    workflow_type: str = Field(..., description="Workflow type: scheduled or walk_in")
    status: str = Field(..., description="Visit status")
    created_at: datetime = Field(..., description="Visit creation date")


class PatientWithVisitsSchema(BaseModel):
    """Schema for patient with aggregated visit information."""
    
    patient_id: str = Field(..., description="Patient ID")
    name: str = Field(..., description="Patient name")
    mobile: str = Field(..., description="Mobile number")
    age: int = Field(..., description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    latest_visit: Optional[LatestVisitInfo] = Field(None, description="Latest visit information")
    total_visits: int = Field(..., description="Total number of visits")
    scheduled_visits_count: int = Field(..., description="Number of scheduled visits")
    walk_in_visits_count: int = Field(..., description="Number of walk-in visits")


class PatientListResponse(BaseModel):
    """Response schema for patient list endpoint."""
    
    patients: List[PatientWithVisitsSchema] = Field(..., description="List of patients with visit information")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
