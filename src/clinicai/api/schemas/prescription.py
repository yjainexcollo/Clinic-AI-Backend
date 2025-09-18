"""
Pydantic schemas for prescription-related API endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Medicine(BaseModel):
    """Schema for individual medicine information extracted from prescription."""
    
    name: str = Field(..., description="Name of the medicine")
    dose: Optional[str] = Field(None, description="Dosage information")
    frequency: Optional[str] = Field(None, description="Frequency of administration")
    duration: Optional[str] = Field(None, description="Duration of treatment")


class PrescriptionResponse(BaseModel):
    """Response schema for prescription upload and analysis."""
    
    patient_id: str = Field(..., description="Patient ID")
    visit_id: str = Field(..., description="Visit ID")
    medicines: List[Medicine] = Field(default_factory=list, description="List of extracted medicines")
    tests: List[str] = Field(default_factory=list, description="List of recommended tests")
    instructions: List[str] = Field(default_factory=list, description="List of medical instructions")
    raw_text: str = Field(..., description="Raw extracted text from prescription images")
    processing_status: str = Field(..., description="Status of prescription processing")
    message: str = Field(..., description="Status message")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information for troubleshooting")


class ErrorResponse(BaseModel):
    """Error response schema for prescription endpoints."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")