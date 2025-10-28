"""
Blob file reference model for tracking files stored in Azure Blob Storage.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from beanie import Document
from pydantic import Field


class BlobFileReference(Document):
    """MongoDB model for tracking files stored in Azure Blob Storage."""
    
    file_id: str = Field(..., description="Unique file ID", unique=True)
    blob_path: str = Field(..., description="Full blob path in Azure Storage")
    container_name: str = Field(..., description="Azure Storage container name")
    original_filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the file")
    file_size: int = Field(..., description="File size in bytes")
    blob_url: str = Field(..., description="Public blob URL")
    
    # File categorization
    file_type: str = Field(..., description="Type of file (audio, image, document)")
    category: str = Field(default="general", description="File category (adhoc, visit, medication, etc.)")
    
    # References to related entities
    patient_id: Optional[str] = Field(None, description="Patient ID if linked to a patient")
    visit_id: Optional[str] = Field(None, description="Visit ID if linked to a visit")
    adhoc_id: Optional[str] = Field(None, description="Adhoc transcript ID if linked to adhoc transcript")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional file metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="File expiry date (for temporary files)")
    
    # Status
    is_active: bool = Field(default=True, description="Whether the file is active")
    is_deleted: bool = Field(default=False, description="Whether the file is marked for deletion")
    
    class Settings:
        name = "blob_file_references"
        indexes = [
            "file_id",
            "blob_path",
            "file_type",
            "category",
            "patient_id",
            "visit_id", 
            "adhoc_id",
            "created_at",
            "expires_at",
            "is_active",
            "is_deleted"
        ]
