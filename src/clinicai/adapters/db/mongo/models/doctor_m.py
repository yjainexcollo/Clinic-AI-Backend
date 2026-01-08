"""MongoDB Beanie model for Doctor documents."""

from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field


class DoctorMongo(Document):
    """MongoDB model for Doctor entity."""

    doctor_id: str = Field(..., description="Doctor ID", unique=True)
    name: str = Field(..., description="Doctor display name")
    email: Optional[str] = Field(None, description="Doctor email address")
    status: str = Field(default="active", description="Doctor status (active/inactive)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "doctors"
        indexes = [
            "doctor_id",
            "status",
            "created_at",
        ]
