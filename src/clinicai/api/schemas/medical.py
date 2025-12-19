"""
Medical content schemas for SOAP notes, vitals, and medical data.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class VitalsData(BaseModel):
    """Vitals data schema."""
    
    blood_pressure: Optional[str] = Field(None, description="Blood pressure reading")
    heart_rate: Optional[str] = Field(None, description="Heart rate")
    temperature: Optional[str] = Field(None, description="Body temperature")
    respiratory_rate: Optional[str] = Field(None, description="Respiratory rate")
    oxygen_saturation: Optional[str] = Field(None, description="Oxygen saturation")
    weight: Optional[str] = Field(None, description="Weight")
    height: Optional[str] = Field(None, description="Height")
    bmi: Optional[str] = Field(None, description="Body Mass Index")
    notes: Optional[str] = Field(None, description="Additional notes")
    recorded_at: str = Field(..., description="Vitals recording timestamp")


class PhysicalExam(BaseModel):
    """Physical examination schema."""
    
    general_appearance: Optional[str] = Field(None, description="General appearance")
    heent: Optional[str] = Field(None, description="Head, Eyes, Ears, Nose, Throat")
    cardiac: Optional[str] = Field(None, description="Cardiac examination")
    respiratory: Optional[str] = Field(None, description="Respiratory examination")
    abdominal: Optional[str] = Field(None, description="Abdominal examination")
    neurological: Optional[str] = Field(None, description="Neurological examination")
    extremities: Optional[str] = Field(None, description="Extremities examination")
    gait: Optional[str] = Field(None, description="Gait assessment")


class SoapTemplateSchema(BaseModel):
    """Optional per-visit SOAP template schema used to guide generation.

    This is not a global template library – it is sent (and optionally stored)
    per visit/soap generation. Frontend can mirror the fields from the doctor
    template UI (e.g., screenshot form).
    """

    template_name: Optional[str] = Field(
        None, description="Human-friendly template name (e.g., 'General Template')"
    )
    category: Optional[str] = Field(
        None, description="Template category (e.g., 'General / Primary Care')"
    )
    speciality: Optional[str] = Field(
        None, description="Clinical speciality for this template (e.g., 'Primary Care')"
    )
    description: Optional[str] = Field(
        None, description="Optional description or usage notes for this template"
    )

    # Core SOAP text templates – can include placeholders like [patient_name], [blood_pressure], etc.
    soap_content: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-section SOAP templates. Keys typically include "
            "'subjective', 'objective', 'assessment', 'plan'."
        ),
    )

    tags: Optional[List[str]] = Field(
        None, description="Free-form tags/classifications (e.g., ['Test', 'Follow-up'])"
    )
    appointment_types: Optional[List[str]] = Field(
        None, description="Applicable appointment types for this template"
    )

    uploaded_at: Optional[datetime] = Field(
        None, description="When this template was created on the client (optional)"
    )


class SOAPNoteRequest(BaseModel):
    """Request schema for SOAP note generation."""
    
    patient_id: str = Field(..., min_length=10, max_length=200, description="Opaque/encrypted Patient ID (can be longer when encoded)")
    visit_id: str = Field(..., min_length=10, max_length=200, description="Opaque/encrypted Visit ID (can be longer when encoded)")
    transcript: Optional[str] = Field(None, description="Optional transcript text. If not provided, will use stored transcript from visit.")
    template: Optional[SoapTemplateSchema] = Field(
        None,
        description=(
            "Optional per-visit SOAP template. If provided, the generator will "
            "follow this structure; if omitted, default behavior is used."
        ),
    )


class SOAPNoteResponse(BaseModel):
    """Response schema for SOAP note."""
    
    subjective: str = Field(..., description="Subjective findings from patient")
    objective: Dict[str, Any] = Field(..., description="Objective findings and vitals")
    assessment: str = Field(..., description="Assessment and diagnosis")
    plan: str = Field(..., description="Treatment plan")
    highlights: List[str] = Field(default_factory=list, description="Key highlights")
    red_flags: List[str] = Field(default_factory=list, description="Red flag symptoms")
    generated_at: str = Field(..., description="SOAP note generation timestamp")
    model_info: Optional[Dict[str, Any]] = Field(None, description="AI model information")
    confidence_score: Optional[float] = Field(None, description="Confidence score")


class MedicationInfo(BaseModel):
    """Medication information schema."""
    
    name: str = Field(..., description="Medication name")
    dosage: str = Field(..., description="Dosage information")
    frequency: str = Field(..., description="Frequency of administration")
    duration: Optional[str] = Field(None, description="Duration of treatment")
    instructions: Optional[str] = Field(None, description="Special instructions")
    side_effects: Optional[List[str]] = Field(default_factory=list, description="Potential side effects")


class TestOrderInfo(BaseModel):
    """Test order information schema."""
    
    test_name: str = Field(..., description="Name of the test")
    purpose: str = Field(..., description="Purpose of the test")
    instructions: str = Field(..., description="Instructions for the test")
    urgency: str = Field(default="routine", description="Urgency level (routine, urgent, stat)")
    lab_location: Optional[str] = Field(None, description="Laboratory location")
    preparation_required: Optional[str] = Field(None, description="Preparation instructions")


class MedicalRecommendation(BaseModel):
    """Medical recommendation schema."""
    
    type: str = Field(..., description="Type of recommendation (lifestyle, dietary, exercise, etc.)")
    description: str = Field(..., description="Description of the recommendation")
    priority: str = Field(default="medium", description="Priority level (low, medium, high)")
    duration: Optional[str] = Field(None, description="Duration of the recommendation")
    follow_up_required: bool = Field(default=False, description="Whether follow-up is required")


class ActionPlanRequest(BaseModel):
    """Request schema for action plan generation."""
    
    patient_id: str = Field(..., min_length=10, max_length=200, description="Opaque/encrypted Patient ID (can be longer when encoded)")
    visit_id: str = Field(..., min_length=10, max_length=200, description="Opaque/encrypted Visit ID (can be longer when encoded)")


class ActionPlanResponse(BaseModel):
    """Response schema for action plan."""
    
    immediate_actions: List[str] = Field(default_factory=list, description="Immediate actions to take")
    short_term_goals: List[str] = Field(default_factory=list, description="Short-term goals (1-2 weeks)")
    long_term_goals: List[str] = Field(default_factory=list, description="Long-term goals (1-3 months)")
    monitoring_plan: List[str] = Field(default_factory=list, description="Monitoring and follow-up plan")
    patient_education: List[str] = Field(default_factory=list, description="Patient education topics")
    generated_at: str = Field(..., description="Action plan generation timestamp")
