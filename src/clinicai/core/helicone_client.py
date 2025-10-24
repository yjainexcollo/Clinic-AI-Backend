"""
Helicone-integrated OpenAI client for AI observability
Tracks all prompts, responses, tokens, latency, and costs
"""
import os
import time
from openai import AsyncOpenAI
from typing import Optional, Dict, Any, List
import logging
import json

logger = logging.getLogger(__name__)


class HeliconeOpenAIClient:
    """
    OpenAI client with Helicone monitoring and comprehensive tracking
    """
    
    def __init__(
        self,
        api_key: str,
        helicone_api_key: Optional[str] = None,
        environment: str = "production",
        enable_cache: bool = False
    ):
        self.helicone_enabled = helicone_api_key is not None
        self.environment = environment
        
        if self.helicone_enabled:
            # Use Helicone proxy for all OpenAI calls
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://oai.hconeai.com/v1",
                default_headers={
                    "Helicone-Auth": f"Bearer {helicone_api_key}",
                    "Helicone-Property-Environment": environment,
                    "Helicone-Property-App": "clinic-ai",
                    "Helicone-Cache-Enabled": "true" if enable_cache else "false",
                }
            )
            logger.info("✅ Helicone AI observability enabled")
        else:
            self.client = AsyncOpenAI(api_key=api_key)
            logger.warning("⚠️  Helicone disabled - no AI observability")
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        prompt_name: Optional[str] = None,
        custom_properties: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tuple:
        """
        Chat completion with comprehensive tracking
        
        Returns:
            tuple: (response, metrics)
                - response: OpenAI response object
                - metrics: Dict with latency, tokens, cost
        """
        start_time = time.time()
        
        # Build Helicone headers
        extra_headers = {}
        if self.helicone_enabled:
            # User tracking
            if user_id:
                extra_headers["Helicone-User-Id"] = user_id
            if patient_id:
                extra_headers["Helicone-Property-Patient-Id"] = patient_id
            if session_id:
                extra_headers["Helicone-Session-Id"] = session_id
            
            # Prompt tracking
            if prompt_name:
                extra_headers["Helicone-Prompt-Id"] = prompt_name
            
            # Custom properties for filtering/analytics
            if custom_properties:
                for key, value in custom_properties.items():
                    extra_headers[f"Helicone-Property-{key}"] = str(value)
            
            # Default properties
            extra_headers["Helicone-Property-Model"] = model
            extra_headers["Helicone-Property-Temperature"] = str(temperature)
        
        try:
            # Make API call
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers if extra_headers else None,
                **kwargs
            )
            
            # Calculate metrics
            latency = time.time() - start_time
            metrics = self._calculate_metrics(response, latency, model)
            
            # Log metrics
            logger.info(
                f"AI_CALL: model={model} prompt_name={prompt_name} "
                f"tokens={metrics['total_tokens']} latency={metrics['latency_ms']}ms "
                f"cost=${metrics['estimated_cost']:.4f} patient={patient_id}"
            )
            
            return response, metrics
            
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"OpenAI API error: {e} (latency: {latency:.2f}s)")
            raise
    
    async def transcription(
        self,
        file,
        model: str = "whisper-1",
        language: Optional[str] = None,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """
        Audio transcription with tracking
        
        Returns:
            tuple: (transcription, metrics)
        """
        start_time = time.time()
        
        extra_headers = {}
        if self.helicone_enabled:
            if user_id:
                extra_headers["Helicone-User-Id"] = user_id
            if patient_id:
                extra_headers["Helicone-Property-Patient-Id"] = patient_id
            if session_id:
                extra_headers["Helicone-Session-Id"] = session_id
            
            extra_headers["Helicone-Property-Model"] = model
            extra_headers["Helicone-Property-Type"] = "transcription"
        
        try:
            response = await self.client.audio.transcriptions.create(
                model=model,
                file=file,
                language=language,
                extra_headers=extra_headers if extra_headers else None,
                **kwargs
            )
            
            latency = time.time() - start_time
            metrics = {
                "latency_ms": round(latency * 1000, 2),
                "model": model,
                "type": "transcription",
                "text_length": len(response.text) if hasattr(response, 'text') else 0
            }
            
            logger.info(
                f"AI_TRANSCRIPTION: model={model} latency={metrics['latency_ms']}ms "
                f"text_length={metrics['text_length']} patient={patient_id}"
            )
            
            return response, metrics
            
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Transcription error: {e} (latency: {latency:.2f}s)")
            raise
    
    def _calculate_metrics(self, response, latency: float, model: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for AI call"""
        usage = response.usage if hasattr(response, 'usage') else None
        
        metrics = {
            "latency_ms": round(latency * 1000, 2),
            "model": model,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "estimated_cost": self._estimate_cost(model, usage) if usage else 0.0,
            "finish_reason": response.choices[0].finish_reason if response.choices else None
        }
        
        return metrics
    
    def _estimate_cost(self, model: str, usage) -> float:
        """Estimate cost based on model and token usage"""
        # Pricing as of 2024 (update as needed)
        pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        }
        
        model_pricing = pricing.get(model, {"prompt": 0.001, "completion": 0.002})
        
        prompt_cost = (usage.prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost


# Factory function
def create_helicone_client(enable_cache: bool = False) -> HeliconeOpenAIClient:
    """Create Helicone-enabled OpenAI client"""
    from .config import get_settings
    
    settings = get_settings()
    helicone_key = os.environ.get("HELICONE_API_KEY")
    
    return HeliconeOpenAIClient(
        api_key=settings.openai.api_key,
        helicone_api_key=helicone_key,
        environment=settings.app_env,
        enable_cache=enable_cache
    )

