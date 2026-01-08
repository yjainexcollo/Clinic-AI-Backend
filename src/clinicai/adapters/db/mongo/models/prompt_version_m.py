"""
MongoDB Beanie model for storing prompt version history.

Structure: 4 separate documents, one per scenario (intake, previsit, soap, postvisit)
Each document contains an array of all versions for that scenario.
"""

from datetime import datetime
from typing import List, Optional

from beanie import Document
from pydantic import BaseModel, Field

from clinicai.adapters.external.prompt_registry import PromptScenario


class PromptVersionEntry(BaseModel):
    """Individual prompt version entry stored in the versions array."""

    version: str = Field(..., description="Version string (e.g., 'INTAKE_V_1.3', 'PREVISIT_V_1.0')")
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


class PromptVersionMongo(Document):
    """
    MongoDB model for prompt version tracking.

    One document per scenario, each containing a versions array.
    Documents are identified by the scenario field.
    """

    scenario: str = Field(..., description="Prompt scenario (intake, previsit, soap, postvisit)")
    versions: List[PromptVersionEntry] = Field(
        default_factory=list, description="Array of all versions for this scenario"
    )

    class Settings:
        name = "prompt_versions"
        # Note: Unique index on scenario must be created manually in MongoDB or via migration
        # Beanie doesn't support unique index syntax directly in indexes list
        indexes = [
            "scenario",  # Index for finding document by scenario
        ]
