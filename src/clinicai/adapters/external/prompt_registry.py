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
# Format: SCENARIO_V_X.Y (e.g., INTAKE_V_1.0)
_DEFAULT_VERSIONS: dict[PromptScenario, str] = {
    PromptScenario.INTAKE: "INTAKE_V_1.0",
    PromptScenario.PREVISIT_SUMMARY: "PREVISIT_V_1.0",
    PromptScenario.SOAP: "SOAP_V_1.0",
    PromptScenario.POSTVISIT_SUMMARY: "POSTVISIT_V_1.0",
    # RED_FLAG is excluded from version tracking
}

# Manual Major Version Control
# Developers: Increment these values to start a new major version series (e.g., 1 -> 2 starts 2.0)
MAJOR_VERSIONS: dict[PromptScenario, int] = {
    PromptScenario.INTAKE: 1,
    PromptScenario.PREVISIT_SUMMARY: 1,
    PromptScenario.SOAP: 1,
    PromptScenario.POSTVISIT_SUMMARY: 1,
}

# Runtime versions (updated by PromptVersionManager on startup)
PROMPT_VERSIONS: dict[PromptScenario, str] = _DEFAULT_VERSIONS.copy()


__all__ = ["PromptScenario", "PROMPT_VERSIONS"]
