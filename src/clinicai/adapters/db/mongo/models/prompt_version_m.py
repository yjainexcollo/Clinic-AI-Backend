"""
MongoDB Beanie model for storing prompt version history.
"""

from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field

from clinicai.adapters.external.prompt_registry import PromptScenario


class PromptVersionMongo(Document):
    """MongoDB model for prompt version tracking."""

    scenario: str = Field(..., description="Prompt scenario (e.g., 'soap_summary')")
    version: str = Field(..., description="Version string (e.g., 'SOAP_V1_2025-12-02')")
    template_hash: str = Field(..., description="SHA256 hash of the prompt template")
    template_content: str = Field(..., description="Full prompt template content")
    
    # Metadata
    is_current: bool = Field(default=True, description="Whether this is the current active version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this version was created")
    git_commit: Optional[str] = Field(None, description="Git commit hash (if available)")
    
    # Semantic Versioning
    major_version: int = Field(default=1, description="Major version (manually incremented)")
    minor_version: int = Field(default=0, description="Minor version (auto-incremented on change)")
    
    # Version number extraction (for sorting)
    version_number: int = Field(..., description="Numeric sorting value (major * 1000 + minor)")

    class Settings:
        name = "prompt_versions"
        indexes = [
            "scenario",
            "version",
            "template_hash",
            "is_current",
            [("scenario", 1), ("is_current", 1)],  # Compound index for finding current version by scenario
            [("scenario", 1), ("version_number", -1)],  # Compound index for version history queries
            [("scenario", 1), ("major_version", 1), ("minor_version", 1)],  # Semantic version index
        ]


