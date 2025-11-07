"""
Azure OpenAI client with Helicone support for AI observability.
Tracks all prompts, responses, tokens, latency, and costs
"""
import os
import time
import asyncio
from openai import AsyncAzureOpenAI
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """
    Azure OpenAI client with Helicone monitoring and comprehensive tracking
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        whisper_deployment_name: Optional[str] = None,
        helicone_api_key: Optional[str] = None,
        environment: str = "production",
        enable_cache: bool = False
    ):
        self.deployment_name = deployment_name
        self.whisper_deployment_name = whisper_deployment_name or "whisper"
        self.helicone_enabled = helicone_api_key is not None
        self.environment = environment
        
        # Build headers for Helicone
        default_headers = {}
        if self.helicone_enabled:
            default_headers = {
                "Helicone-Auth": f"Bearer {helicone_api_key}",
                "Helicone-Property-Environment": environment,
                "Helicone-Property-App": "clinic-ai",
                "Helicone-Cache-Enabled": "true" if enable_cache else "false",
            }
            logger.info("✅ Azure OpenAI with Helicone AI observability enabled")
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            default_headers=default_headers if default_headers else None
        )
        
        if not self.helicone_enabled:
            logger.warning("⚠️  Helicone disabled - no AI observability")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,  # Ignored for Azure OpenAI, uses deployment_name
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
                - response: Azure OpenAI response object
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
            extra_headers["Helicone-Property-Model"] = self.deployment_name
            extra_headers["Helicone-Property-Temperature"] = str(temperature)
        
        # Retry logic for rate limits and transient errors
        max_retries = 3
        base_delay = 1.0  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                # Make API call - Azure OpenAI uses deployment_name instead of model
                response = await self.client.chat.completions.create(
                    model=self.deployment_name,  # Use deployment name, not model name
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers if extra_headers else None,
                    **kwargs
                )
                
                # Calculate metrics
                latency = time.time() - start_time
                metrics = self._calculate_metrics(response, latency)
                
                # Log metrics
                logger.info(
                    f"AI_CALL: deployment={self.deployment_name} prompt_name={prompt_name} "
                    f"tokens={metrics['total_tokens']} latency={metrics['latency_ms']}ms "
                    f"patient={patient_id}"
                )
                
                return response, metrics
                
            except Exception as e:
                latency = time.time() - start_time
                error_str = str(e).lower()
                
                # Check if it's a rate limit error (429)
                is_rate_limit = (
                    "429" in str(e) or 
                    "rate limit" in error_str or 
                    "too many requests" in error_str or
                    "quota" in error_str
                )
                
                # Check if it's a transient error (5xx)
                is_transient = (
                    "500" in str(e) or 
                    "502" in str(e) or 
                    "503" in str(e) or
                    "504" in str(e) or
                    "timeout" in error_str or
                    "connection" in error_str
                )
                
                # Retry on rate limits or transient errors
                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Azure OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s... (latency: {latency:.2f}s)"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                # Don't retry on other errors or if we've exhausted retries
                error_type = "rate_limit" if is_rate_limit else ("transient" if is_transient else "permanent")
                logger.error(
                    f"Azure OpenAI API error ({error_type}): {e} "
                    f"(deployment={self.deployment_name}, attempt={attempt + 1}, latency: {latency:.2f}s)"
                )
                
                # Provide specific error messages
                if is_rate_limit:
                    raise ValueError(
                        f"Azure OpenAI rate limit exceeded. Please try again later. "
                        f"Deployment: {self.deployment_name}, Error: {str(e)}"
                    ) from e
                elif is_transient:
                    raise ValueError(
                        f"Azure OpenAI service temporarily unavailable. Please try again. "
                        f"Deployment: {self.deployment_name}, Error: {str(e)}"
                    ) from e
                else:
                    raise ValueError(
                        f"Azure OpenAI API error: {str(e)}. "
                        f"Deployment: {self.deployment_name}, "
                        f"Please verify your Azure OpenAI configuration and deployment status."
                    ) from e
    
    async def transcription(
        self,
        file,
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
            
            extra_headers["Helicone-Property-Model"] = self.whisper_deployment_name
            extra_headers["Helicone-Property-Type"] = "transcription"
        
        # Retry logic for rate limits and transient errors
        max_retries = 3
        base_delay = 1.0  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                # Azure OpenAI Whisper API - use whisper deployment name, not chat deployment
                response = await self.client.audio.transcriptions.create(
                    model=self.whisper_deployment_name,  # Use whisper deployment for transcription
                    file=file,
                    language=language,
                    extra_headers=extra_headers if extra_headers else None,
                    **kwargs
                )
                
                latency = time.time() - start_time
                metrics = {
                    "latency_ms": round(latency * 1000, 2),
                    "deployment": self.whisper_deployment_name,
                    "type": "transcription",
                    "text_length": len(response.text) if hasattr(response, 'text') else 0
                }
                
                logger.info(
                    f"AI_TRANSCRIPTION: deployment={self.whisper_deployment_name} latency={metrics['latency_ms']}ms "
                    f"text_length={metrics['text_length']} patient={patient_id}"
                )
                
                return response, metrics
                
            except Exception as e:
                latency = time.time() - start_time
                error_str = str(e).lower()
                
                # Check if it's a rate limit error (429)
                is_rate_limit = (
                    "429" in str(e) or 
                    "rate limit" in error_str or 
                    "too many requests" in error_str or
                    "quota" in error_str
                )
                
                # Check if it's a transient error (5xx)
                is_transient = (
                    "500" in str(e) or 
                    "502" in str(e) or 
                    "503" in str(e) or
                    "504" in str(e) or
                    "timeout" in error_str or
                    "connection" in error_str
                )
                
                # Retry on rate limits or transient errors
                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Azure OpenAI transcription error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s... (latency: {latency:.2f}s)"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                # Don't retry on other errors or if we've exhausted retries
                error_type = "rate_limit" if is_rate_limit else ("transient" if is_transient else "permanent")
                logger.error(
                    f"Azure OpenAI transcription error ({error_type}): {e} "
                    f"(deployment={self.whisper_deployment_name}, attempt={attempt + 1}, latency: {latency:.2f}s)"
                )
                
                # Provide specific error messages
                if is_rate_limit:
                    raise ValueError(
                        f"Azure OpenAI Whisper rate limit exceeded. Please try again later. "
                        f"Deployment: {self.whisper_deployment_name}, Error: {str(e)}"
                    ) from e
                elif is_transient:
                    raise ValueError(
                        f"Azure OpenAI Whisper service temporarily unavailable. Please try again. "
                        f"Deployment: {self.whisper_deployment_name}, Error: {str(e)}"
                    ) from e
                else:
                    raise ValueError(
                        f"Azure OpenAI transcription error: {str(e)}. "
                        f"Deployment: {self.whisper_deployment_name}, "
                        f"Please verify your Azure OpenAI Whisper deployment configuration."
                    ) from e
    
    def _calculate_metrics(self, response, latency: float) -> Dict[str, Any]:
        """Calculate comprehensive metrics for AI call"""
        usage = response.usage if hasattr(response, 'usage') else None
        
        metrics = {
            "latency_ms": round(latency * 1000, 2),
            "deployment": self.deployment_name,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "finish_reason": response.choices[0].finish_reason if response.choices else None
        }
        
        return metrics


# Factory function
def create_azure_openai_client(enable_cache: bool = False) -> AzureOpenAIClient:
    """Create Azure OpenAI client with Helicone support"""
    from .config import get_settings
    
    settings = get_settings()
    helicone_key = os.environ.get("HELICONE_API_KEY")
    
    if not settings.azure_openai.endpoint or not settings.azure_openai.api_key:
        raise ValueError(
            "Azure OpenAI endpoint and API key must be configured. "
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables "
            "or add them to Azure Key Vault."
        )
    
    return AzureOpenAIClient(
        endpoint=settings.azure_openai.endpoint,
        api_key=settings.azure_openai.api_key,
        api_version=settings.azure_openai.api_version,
        deployment_name=settings.azure_openai.deployment_name,
        whisper_deployment_name=settings.azure_openai.whisper_deployment_name,
        helicone_api_key=helicone_key,
        environment=settings.app_env,
        enable_cache=enable_cache
    )

