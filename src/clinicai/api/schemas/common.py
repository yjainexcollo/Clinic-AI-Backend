"""
Common schemas and reusable components for scalable API design.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel

T = TypeVar("T")

# ============================================================================
# BASE RESPONSE SCHEMAS
# ============================================================================


class BaseResponse(BaseModel):
    """Base response schema with common fields."""

    success: bool = Field(True, description="Operation success status")
    message: str = Field("", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID for tracking")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ApiResponse(GenericModel, Generic[T]):
    success: bool = Field(True, description="Operation success status")
    message: str = Field("", description="Response message")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Response timestamp",
    )
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID for tracking")
    data: Optional[T] = Field(None, description="Response payload")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    success: bool = Field(False, description="Operation success status")
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Error timestamp",
    )
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID for tracking")


# ============================================================================
# REUSABLE COMPONENT SCHEMAS
# ============================================================================


class ContactInfo(BaseModel):
    """Contact information schema."""

    mobile: str = Field(..., min_length=8, max_length=16, description="Phone number")

    @validator("mobile")
    def validate_mobile(cls, v):
        import re

        s = (v or "").strip()
        if re.fullmatch(r"^\+[1-9]\d{7,14}$", s):
            return s
        if re.fullmatch(r"^\d{8,16}$", s):
            return s
        raise ValueError("Phone must be E.164 format or 8-16 local digits")


class PersonalInfo(BaseModel):
    """Personal information schema."""

    first_name: str = Field(..., min_length=1, max_length=40, description="First name")
    last_name: str = Field(..., min_length=1, max_length=40, description="Last name")
    age: int = Field(..., ge=0, le=120, description="Age")
    gender: str = Field(..., description="Gender")

    @validator("first_name", "last_name")
    def validate_names(cls, v):
        if not v or not v.strip():
            raise ValueError("Name fields cannot be empty")
        return v.strip()


class QuestionAnswer(BaseModel):
    """Question-answer pair schema."""

    question_id: str = Field(..., description="Question ID")
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")
    timestamp: datetime = Field(..., description="Timestamp")
    question_number: int = Field(..., ge=1, description="Question number")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# BLOB STORAGE SCHEMAS
# ============================================================================


class BlobFileInfo(BaseModel):
    """Blob file information schema."""

    file_id: str = Field(..., description="Unique file ID")
    blob_path: str = Field(..., description="Full blob path in Azure Storage")
    container_name: str = Field(..., description="Azure Storage container name")
    original_filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the file")
    file_size: int = Field(..., description="File size in bytes")
    blob_url: str = Field(..., description="Public blob URL")
    file_type: str = Field(..., description="Type of file (audio, image, document)")
    category: str = Field(default="general", description="File category")
    patient_id: Optional[str] = Field(None, description="Patient ID if linked")
    visit_id: Optional[str] = Field(None, description="Visit ID if linked")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="File expiry date")
