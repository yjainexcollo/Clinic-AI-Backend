"""
Medical summary schemas for pre-visit and post-visit summaries.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


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
    medication_images: Optional[List[Dict[str, Any]]] = Field(
        None, description="Uploaded medication images metadata for the visit, if any"
    )
    red_flags: Optional[List[Dict[str, str]]] = Field(
        None, description="Red flags detected in patient responses (abusive language, incomplete information)"
    )


class PostVisitSummaryRequest(BaseModel):
    """Request schema for generating post-visit summary."""
    
    patient_id: str = Field(..., description="Opaque Patient ID")
    visit_id: str = Field(..., description="Visit ID")


class PostVisitSummaryResponse(BaseModel):
    """Response schema for post-visit summary - MEDICAL CONTENT ONLY."""
    
    # Medical Content (Generated during visit)
    chief_complaint: str = Field(..., description="Reason for visit in plain language")
    key_findings: List[str] = Field(..., description="Key findings from exam/consultation")
    diagnosis: str = Field(..., description="Diagnosis in layman-friendly terms")
    
    # Treatment Plan
    medications: List[Dict[str, str]] = Field(default_factory=list, description="Prescribed medications")
    other_recommendations: List[str] = Field(default_factory=list, description="Lifestyle recommendations")
    tests_ordered: List[Dict[str, str]] = Field(default_factory=list, description="Ordered tests")
    
    # Follow-Up
    next_appointment: Optional[str] = Field(None, description="Next appointment details")
    red_flag_symptoms: List[str] = Field(default_factory=list, description="Warning signs")
    patient_instructions: List[str] = Field(..., description="Patient instructions")
    
    # Closing Note
    reassurance_note: str = Field(..., description="Reassurance and encouragement message")
    
    # Metadata
    generated_at: str = Field(..., description="Summary generation timestamp")
