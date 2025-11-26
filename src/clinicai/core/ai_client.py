"""
Simple Azure OpenAI client wrapper for core AI operations.

Design goals:
- Use Azure OpenAI only (via AsyncAzureOpenAI)
- No proxy usage
- No Helicone integration
- No custom observability headers

This class is intentionally minimal and focused on correctness.
Retries, validation, and advanced monitoring are handled elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from openai import AsyncAzureOpenAI

from .config import get_settings


class AzureAIClient:
    """
    Thin wrapper around AsyncAzureOpenAI for common AI operations.

    This client:
    - Connects directly to Azure OpenAI (no proxy)
    - Does not send any Helicone or other custom observability headers
    - Uses deployment names from configuration
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        whisper_deployment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize AzureAIClient.

        If arguments are omitted, values are loaded from application settings.
        """
        settings = get_settings()

        endpoint = endpoint or settings.azure_openai.endpoint
        api_key = api_key or settings.azure_openai.api_key
        api_version = api_version or settings.azure_openai.api_version
        deployment_name = deployment_name or settings.azure_openai.deployment_name
        whisper_deployment_name = (
            whisper_deployment_name or settings.azure_openai.whisper_deployment_name
        )

        if not endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI endpoint and API key must be configured. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
            )

        if not deployment_name:
            raise ValueError(
                "Azure OpenAI deployment name is required. "
                "Set AZURE_OPENAI_DEPLOYMENT_NAME."
            )

        if not whisper_deployment_name:
            raise ValueError(
                "Azure OpenAI Whisper deployment name is required. "
                "Set AZURE_OPENAI_WHISPER_DEPLOYMENT_NAME."
            )

        # Normalize endpoint: Azure SDK does not expect trailing slash
        normalized_endpoint = endpoint.rstrip("/") if endpoint else endpoint

        self._deployment_name = deployment_name
        self._whisper_deployment_name = whisper_deployment_name

        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=normalized_endpoint,
        )

    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Generic chat completion helper.

        Args:
            messages: OpenAI chat messages list.
            model: Optional deployment name override. Defaults to configured deployment.
            temperature: Sampling temperature.
            max_tokens: Optional max tokens for the response.
            **kwargs: Passed directly to Azure OpenAI SDK.
        """
        deployment = model or self._deployment_name
        return await self._client.chat.completions.create(
            model=deployment,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def summarize(
        self,
        text: str,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Convenience helper for simple text summarization using chat completions.
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": text})

        return await self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def embed(
        self,
        inputs: Union[str, Sequence[str]],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Generate embeddings.

        Notes:
        - `model` should be set to an embeddings deployment.
        - If omitted, the main deployment name is used; callers may override.
        """
        deployment = model or self._deployment_name
        return await self._client.embeddings.create(
            model=deployment,
            input=inputs,
            **kwargs,
        )

    async def transcribe_whisper(
        self,
        file: Any,
        *,
        language: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Transcribe audio using an Azure OpenAI Whisper deployment.

        Args:
            file: Binary file-like object opened in rb mode.
            language: Optional language code.
        """
        return await self._client.audio.transcriptions.create(
            model=self._whisper_deployment_name,
            file=file,
            language=language,
            **kwargs,
        )


__all__ = ["AzureAIClient"]


