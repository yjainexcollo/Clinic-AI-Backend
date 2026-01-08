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

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from openai import AsyncAzureOpenAI

from .config import get_settings

logger = logging.getLogger("clinicai.ai")

# Optional observability (OpenTelemetry + custom metrics)
try:  # pragma: no cover - optional dependency
    from clinicai.observability import (
        METRICS_AVAILABLE,
        add_span_attribute,
        record_ai_request,
        record_error,
        set_span_status,
        trace_operation,
    )

    OBSERVABILITY_AVAILABLE = True
except Exception:  # ImportError or other issues
    OBSERVABILITY_AVAILABLE = False
    METRICS_AVAILABLE = False


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

        if not endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI endpoint and API key must be configured. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
            )

        if not deployment_name:
            raise ValueError("Azure OpenAI deployment name is required. " "Set AZURE_OPENAI_DEPLOYMENT_NAME.")

        # Normalize endpoint: Azure SDK does not expect trailing slash
        normalized_endpoint = endpoint.rstrip("/") if endpoint else endpoint

        self._deployment_name = deployment_name

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

        # Basic latency measurement
        start_time = time.time()

        if OBSERVABILITY_AVAILABLE:
            # Create a tracing span for the AI call
            attrs: Dict[str, Any] = {"deployment": deployment}
            with trace_operation("ai.chat.completion", attrs) as span:
                try:
                    response = await self._client.chat.completions.create(
                        model=deployment,
                        messages=list(messages),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    latency_ms = (time.time() - start_time) * 1000.0

                    # Extract token usage if available
                    usage = getattr(response, "usage", None)
                    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

                    if METRICS_AVAILABLE:
                        try:
                            record_ai_request(
                                model=deployment,
                                latency_ms=latency_ms,
                                tokens=total_tokens,
                                success=True,
                            )
                        except Exception:
                            # Metrics should never break the request path
                            logger.debug("Failed to record AI metrics", exc_info=True)

                    if span:
                        try:
                            add_span_attribute(span, "deployment", deployment)
                            add_span_attribute(span, "latency_ms", latency_ms)
                            add_span_attribute(span, "tokens", total_tokens)
                            set_span_status(span, success=True)
                        except Exception:
                            logger.debug("Failed to enrich AI span", exc_info=True)

                    logger.info(
                        "AI_CALL: deployment=%s tokens=%s latency_ms=%.2f",
                        deployment,
                        total_tokens,
                        latency_ms,
                    )
                    return response
                except Exception as exc:  # pragma: no cover - defensive path
                    latency_ms = (time.time() - start_time) * 1000.0

                    if METRICS_AVAILABLE:
                        try:
                            record_ai_request(
                                model=deployment,
                                latency_ms=latency_ms,
                                tokens=0,
                                success=False,
                            )
                            record_error(
                                error_type="ai_request_error",
                                error_message=str(exc)[:200],
                            )
                        except Exception:
                            logger.debug("Failed to record AI error metrics", exc_info=True)

                    if span:
                        try:
                            add_span_attribute(span, "error", str(exc)[:200])
                            set_span_status(span, success=False, error_message=str(exc))
                        except Exception:
                            logger.debug("Failed to set AI span error", exc_info=True)

                    logger.error(
                        "AI_CALL_ERROR: deployment=%s error=%s latency_ms=%.2f",
                        deployment,
                        str(exc),
                        latency_ms,
                    )
                    raise
        else:
            # Fallback path without OpenTelemetry / metrics
            response = await self._client.chat.completions.create(
                model=deployment,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            latency_ms = (time.time() - start_time) * 1000.0

            usage = getattr(response, "usage", None)
            total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

            logger.info(
                "AI_CALL: deployment=%s tokens=%s latency_ms=%.2f",
                deployment,
                total_tokens,
                latency_ms,
            )
            return response

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


__all__ = ["AzureAIClient"]
