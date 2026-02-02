"""
Patient registration schemas.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator, validator

from .common import ContactInfo, PersonalInfo


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


# ---------------------------------------------------------------------------
# Unified registration (scheduled + walk-in). One schema: first_name + last_name
# only (no separate "name" field).
# ---------------------------------------------------------------------------


def _normalize_language(v):
    """Normalize language codes: 'es' -> 'sp' for consistency with frontend."""
    if v and isinstance(v, str):
        normalized = v.lower().strip()
        if normalized == "es":
            return "sp"
        if normalized in ["en", "sp"]:
            return normalized
    return "en"


def _validate_mobile(v):
    """Validate mobile number - E.164 or local format."""
    s = (v or "").strip()
    if re.fullmatch(r"^\+[1-9]\d{7,14}$", s):
        return s
    if re.fullmatch(r"^\d{8,16}$", s):
        return s
    raise ValueError("Phone must be E.164 format (+country code followed by 7-14 digits) or 8-16 local digits")


class UnifiedRegisterPatientRequest(BaseModel):
    """
    Unified request for patient registration (scheduled or walk-in).
    Both flows use first_name and last_name only; there is no "name" field.
    """

    workflow_type: str = Field(
        ...,
        description="'scheduled' or 'walk_in'.",
    )
    first_name: str = Field(..., min_length=1, max_length=40, description="First name")
    last_name: str = Field(..., min_length=1, max_length=40, description="Last name")
    mobile: str = Field(..., description="Mobile phone number (E.164 or local format)")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    recently_travelled: bool = Field(False, description="Recently travelled (scheduled)")
    consent: bool = Field(True, description="Must be true for scheduled workflow; default true for convenience")
    country: str = Field("US", min_length=2, max_length=2, description="ISO alpha-2 country code")
    language: str = Field(
        "en",
        pattern=r"^(en|es|sp)$",
        description="Preferred language (en or sp)",
    )

    @validator("workflow_type", pre=True)
    def normalize_workflow_type(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("workflow_type is required")
        n = v.lower().strip()
        if n in ("walk-in", "walk_in"):
            return "walk_in"
        if n == "scheduled":
            return "scheduled"
        raise ValueError("workflow_type must be 'scheduled' or 'walk_in'")

    @validator("language", pre=True)
    def normalize_language_unified(cls, v):
        return _normalize_language(v)

    @validator("mobile")
    def validate_mobile_unified(cls, v):
        return _validate_mobile(v)

    @validator("country", pre=True)
    def validate_country_unified(cls, v):
        if v is None or v == "":
            return "US"
        s = (v or "").strip().upper()
        if len(s) != 2 or not s.isalpha():
            raise ValueError("country must be ISO alpha-2 code")
        return s

    @validator("first_name", "last_name", pre=True)
    def sanitize_names(cls, v):
        if v is None:
            return None
        s = (v.strip() if isinstance(v, str) else v) or None
        if s is None:
            return None
        return re.sub(r"[\x00-\x1F]+", "", s)[:40]

    @validator("consent")
    def validate_consent_unified(cls, v, values):
        workflow = values.get("workflow_type")
        if workflow == "scheduled" and v is not True:
            raise ValueError("Consent must be True for scheduled workflow")
        return v

    @model_validator(mode="after")
    def validate_workflow_required_fields(self):
        workflow = self.workflow_type
        if workflow == "scheduled":
            if not (self.first_name and self.first_name.strip()) or not (self.last_name and self.last_name.strip()):
                raise ValueError("first_name and last_name are required for scheduled workflow")
            if self.age is None:
                raise ValueError("age is required for scheduled workflow")
        elif workflow == "walk_in":
            if not (self.first_name and self.first_name.strip()) or not (self.last_name and self.last_name.strip()):
                raise ValueError("first_name and last_name are required for walk_in workflow")
        return self


class UnifiedRegisterPatientResponse(BaseModel):
    """Unified response for patient registration."""

    patient_id: str = Field(..., description="Generated patient ID")
    visit_id: str = Field(..., description="Generated visit ID")
    workflow_type: str = Field(..., description="'scheduled' or 'walk_in'")
    status: str = Field(..., description="Visit status")
    first_question: Optional[str] = Field(None, description="First intake question (scheduled only)")
    message: str = Field(..., description="Success message")
