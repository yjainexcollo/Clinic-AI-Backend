"""
Centralized LLM gateway with telemetry and prompt version tracking.

This module provides a unified interface for making LLM calls with:
- Automatic prompt version tracking
- Application Insights telemetry integration
- Consistent error handling
"""

import logging
import time
from typing import Any, Dict, List, Optional

from clinicai.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS
from clinicai.core.ai_client import AzureAIClient
from clinicai.observability.tracing import (
    trace_operation,
    add_span_attribute,
    set_span_status,
)

logger = logging.getLogger(__name__)


async def call_llm_with_telemetry(
    ai_client: AzureAIClient,
    scenario: PromptScenario,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Central gateway for LLM calls with telemetry.
    
    Args:
        ai_client: AzureAIClient instance
        scenario: PromptScenario enum value for this LLM call
        messages: List of message dicts for the LLM
        model: Optional model/deployment name override
        temperature: Sampling temperature
        max_tokens: Optional max tokens for response
        **kwargs: Additional arguments passed to chat completion
    
    Returns:
        LLM response object
    """
    prompt_version = PROMPT_VERSIONS.get(scenario, "UNKNOWN")
    start_time = time.perf_counter()

    with trace_operation(
        "llm_call",
        {
            "llm.scenario": scenario.value,
            "llm.prompt_version": prompt_version,
            "llm.model": model or "default",
        },
    ) as span:
        try:
            # Make the LLM call
            response = await ai_client.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Add version to span attributes
            if span:
                add_span_attribute(span, "llm.prompt_version", prompt_version)
                add_span_attribute(span, "llm.scenario", scenario.value)
                add_span_attribute(span, "llm.latency_ms", latency_ms)
                
                # Extract token usage if available
                usage = getattr(response, "usage", None)
                if usage:
                    total_tokens = getattr(usage, "total_tokens", 0)
                    add_span_attribute(span, "llm.tokens", total_tokens)
                
                set_span_status(span, success=True)
            
            logger.info(
                f"LLM call completed: scenario={scenario.value} "
                f"version={prompt_version} latency_ms={latency_ms:.2f}"
            )
            
            return response
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            if span:
                add_span_attribute(span, "llm.prompt_version", prompt_version)
                add_span_attribute(span, "llm.scenario", scenario.value)
                add_span_attribute(span, "llm.latency_ms", latency_ms)
                add_span_attribute(span, "llm.error", str(e)[:200])
                set_span_status(span, success=False, error_message=str(e))
            
            logger.error(
                f"LLM call failed: scenario={scenario.value} "
                f"version={prompt_version} error={str(e)}"
            )
            raise

