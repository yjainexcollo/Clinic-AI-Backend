"""
AI client factory.

This module centralizes creation of the core AI client used by the backend.
It is intentionally simple:

- Returns a direct Azure OpenAI client (no Helicone, no proxy)
- Uses configuration from `Settings.azure_openai`
"""

from __future__ import annotations

from .ai_client import AzureAIClient


def get_ai_client() -> AzureAIClient:
    """
    Get the default AI client for the application.

    Currently this is a thin wrapper over `AsyncAzureOpenAI` configured
    via environment / settings.
    """
    return AzureAIClient()


__all__ = ["get_ai_client", "AzureAIClient"]


