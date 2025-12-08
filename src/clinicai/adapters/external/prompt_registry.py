"""
Prompt registry for LLM scenarios and version tracking.

This module defines all LLM scenarios used in the application and maintains
a runtime registry of prompt versions that is automatically updated on startup.
"""

from __future__ import annotations

from enum import Enum


class PromptScenario(str, Enum):
    """LLM scenarios for telemetry and prompt versioning."""

    INTAKE = "intake"
    PREVISIT_SUMMARY = "previsit_summary"
    RED_FLAG = "red_flag"
    SOAP = "soap_summary"
    POSTVISIT_SUMMARY = "postvisit_summary"

    # Add your scenarios here:
    # YOUR_SCENARIO = "your_scenario"


# Default/fallback versions (used before DB initialization)
_DEFAULT_VERSIONS: dict[PromptScenario, str] = {
    PromptScenario.INTAKE: "INTAKE_V1_2025-01-15",
    PromptScenario.PREVISIT_SUMMARY: "PREVISIT_V1_2025-01-15",
    PromptScenario.RED_FLAG: "RED_FLAG_V1_2025-01-15",
    PromptScenario.SOAP: "SOAP_V1_2025-01-15",
    PromptScenario.POSTVISIT_SUMMARY: "POSTVISIT_V1_2025-01-15",
    # Add defaults for your scenarios
}

# Runtime versions (updated by PromptVersionManager on startup)
PROMPT_VERSIONS: dict[PromptScenario, str] = _DEFAULT_VERSIONS.copy()


__all__ = ["PromptScenario", "PROMPT_VERSIONS"]


