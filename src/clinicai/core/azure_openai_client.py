"""
Azure OpenAI client with Helicone support for AI observability.
Tracks all prompts, responses, tokens, latency, and costs
"""
import os
import time
import asyncio
from openai import AsyncAzureOpenAI
from typing import Optional, Dict, Any, List, Tuple
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
        self.api_version = api_version  # Store API version
        self.api_key = api_key  # Store API key for recreating client if needed
        
        # Normalize endpoint: remove trailing slash (Azure OpenAI SDK doesn't expect it)
        normalized_endpoint = endpoint.rstrip("/") if endpoint else endpoint
        self.endpoint = normalized_endpoint  # Store normalized endpoint for error messages
        
        # Build headers for Helicone
        self.default_headers = {}
        if self.helicone_enabled:
            self.default_headers = {
                "Helicone-Auth": f"Bearer {helicone_api_key}",
                "Helicone-Property-Environment": environment,
                "Helicone-Property-App": "clinic-ai",
                "Helicone-Cache-Enabled": "true" if enable_cache else "false",
            }
            logger.info("✅ Azure OpenAI with Helicone AI observability enabled")
        
        # Initialize Azure OpenAI client (Azure SDK does not allow base_url with azure_endpoint)
        helicone_base_url = os.getenv("HELICONE_BASE_URL")
        if helicone_base_url:
            logger.info(
                "🔁 HELICONE_BASE_URL is set (%s); "
                "ensure your Azure OpenAI endpoint is configured to route through Helicone.",
                helicone_base_url,
            )
        
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=normalized_endpoint,
            default_headers=self.default_headers if self.default_headers else None,
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
                
                # Check if it's a 404 error (deployment not found)
                is_not_found = (
                    "404" in str(e) or 
                    "not found" in error_str or
                    "resource not found" in error_str
                )
                
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
                
                # Retry on rate limits or transient errors (but not 404 - that's permanent)
                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Azure OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s... (latency: {latency:.2f}s)"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                # If it's a 404, try alternative API versions before giving up
                if is_not_found and attempt == 0:  # Only try alternative versions on first attempt
                    logger.warning(
                        f"404 error with API version '{self.api_version}', trying alternative versions..."
                    )
                    # Try alternative API versions
                    alternative_versions = [
                        "2024-12-01-preview",
                        "2024-08-01-preview",
                        "2024-07-18",
                        "2024-06-01",
                        "2024-02-15-preview",
                        "2023-12-01-preview"
                    ]
                    
                    for alt_version in alternative_versions:
                        if alt_version == self.api_version:
                            continue  # Skip the version we already tried
                        
                        try:
                            # Create a temporary client with alternative API version
                            temp_client = AsyncAzureOpenAI(
                                api_key=self.api_key,
                                api_version=alt_version,
                                azure_endpoint=self.endpoint,
                                default_headers=self.default_headers if self.default_headers else None
                            )
                            
                            # Try the call with alternative version
                            response = await temp_client.chat.completions.create(
                                model=self.deployment_name,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                extra_headers=extra_headers if extra_headers else None,
                                **kwargs
                            )
                            
                            # Success! Update our client to use this version
                            logger.info(
                                f"✅ Found working API version: {alt_version} (original: {self.api_version}). "
                                f"Updating client and continuing..."
                            )
                            self.api_version = alt_version
                            self.client = temp_client
                            
                            # Calculate metrics and return
                            latency = time.time() - start_time
                            metrics = self._calculate_metrics(response, latency)
                            logger.info(
                                f"AI_CALL: deployment={self.deployment_name} prompt_name={prompt_name} "
                                f"tokens={metrics['total_tokens']} latency={metrics['latency_ms']}ms "
                                f"patient={patient_id} (using API version {alt_version})"
                            )
                            return response, metrics
                            
                        except Exception as alt_error:
                            # This version didn't work, try next
                            continue
                    
                    # If we get here, no alternative version worked
                    logger.error(
                        f"All API versions failed for deployment '{self.deployment_name}'"
                    )
                
                # Don't retry on other errors or if we've exhausted retries
                error_type = "not_found" if is_not_found else ("rate_limit" if is_rate_limit else ("transient" if is_transient else "permanent"))
                logger.error(
                    f"Azure OpenAI API error ({error_type}): {e} "
                    f"(deployment={self.deployment_name}, attempt={attempt + 1}, latency: {latency:.2f}s)"
                )
                
                # Provide specific error messages
                if is_not_found:
                    raise ValueError(
                        f"Azure OpenAI deployment not found. The deployment '{self.deployment_name}' does not exist in your Azure OpenAI resource. "
                        f"Tried API versions: {self.api_version} and alternatives.\n"
                        f"Please verify:\n"
                        f"1. The deployment name '{self.deployment_name}' is correct (check AZURE_OPENAI_DEPLOYMENT_NAME in your .env file)\n"
                        f"2. The deployment exists in your Azure OpenAI resource (check Azure Portal > Azure OpenAI > Deployments)\n"
                        f"3. The endpoint is correct (current: {self.endpoint})\n"
                        f"4. Update AZURE_OPENAI_API_VERSION in your .env file to a supported version\n"
                        f"Error details: {str(e)}"
                    ) from e
                elif is_rate_limit:
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


async def validate_azure_openai_deployment(
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment_name: str,
    timeout: float = 5.0,
    try_alternative_versions: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate that an Azure OpenAI deployment exists by making a test call.
    
    Returns:
        tuple: (is_valid, error_message)
            - is_valid: True if deployment exists and is accessible
            - error_message: Error message if validation failed, None if successful
    """
    from openai import AsyncAzureOpenAI
    
    # Normalize endpoint: remove trailing slash if present
    normalized_endpoint = endpoint.rstrip("/") if endpoint else endpoint
    
    # List of API versions to try (most common ones)
    api_versions_to_try = [api_version]
    if try_alternative_versions:
        alternative_versions = [
            "2024-12-01-preview",
            "2024-08-01-preview", 
            "2024-07-18",
            "2024-06-01",
            "2024-02-15-preview",
            "2023-12-01-preview"
        ]
        # Add alternatives that aren't already in the list
        for alt_version in alternative_versions:
            if alt_version not in api_versions_to_try:
                api_versions_to_try.append(alt_version)
    
    last_error = None
    for version_to_try in api_versions_to_try:
        try:
            client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=version_to_try,
                azure_endpoint=normalized_endpoint,
                timeout=timeout
            )
            
            # Make a minimal test call to verify the deployment exists
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=deployment_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                ),
                timeout=timeout
            )
            
            # If we get here, the deployment exists with this API version
            if version_to_try != api_version:
                return True, f"Deployment validated successfully with API version '{version_to_try}' (original was '{api_version}'). Consider updating AZURE_OPENAI_API_VERSION to '{version_to_try}'."
            return True, None
            
        except asyncio.TimeoutError:
            last_error = f"Timeout with API version '{version_to_try}'"
            continue
        except Exception as e:
            error_str = str(e).lower()
            # If it's a 404, try next version. If it's auth/permission error, fail immediately
            if "401" in str(e) or "unauthorized" in error_str or "403" in str(e) or "forbidden" in error_str:
                # Don't try other versions for auth errors
                last_error = str(e)
                break
            elif "404" in str(e) or "not found" in error_str:
                # Try next API version
                last_error = str(e)
                continue
            else:
                # Other errors, try next version
                last_error = str(e)
                continue
    
    # If we get here, all API versions failed
    error_str = str(last_error).lower() if last_error else ""
    full_error = f"{type(last_error).__name__}: {last_error}" if last_error else "Unknown error"
    
    if "401" in str(last_error) or "unauthorized" in error_str or "authentication" in error_str:
        return False, (
            f"Authentication failed for Azure OpenAI resource at {endpoint}. "
            f"Tried API versions: {', '.join(api_versions_to_try)}\n"
            f"Please verify:\n"
            f"1. The API key (AZURE_OPENAI_API_KEY) is correct\n"
            f"2. The API key has not expired\n"
            f"3. The API key has access to this Azure OpenAI resource\n"
            f"Error: {full_error}"
        )
    elif "403" in str(last_error) or "forbidden" in error_str:
        return False, (
            f"Access forbidden to Azure OpenAI resource at {endpoint}. "
            f"Tried API versions: {', '.join(api_versions_to_try)}\n"
            f"Please verify you have the necessary permissions. "
            f"Error: {full_error}"
        )
    elif "404" in str(last_error) or "not found" in error_str or "resource not found" in error_str:
        return False, (
            f"Deployment '{deployment_name}' not found in Azure OpenAI resource at {endpoint}. "
            f"Tried API versions: {', '.join(api_versions_to_try)}\n"
            f"Please verify:\n"
            f"1. The deployment name '{deployment_name}' is correct (check Azure Portal > Azure OpenAI > Deployments)\n"
            f"2. The deployment exists and is in 'Succeeded' state\n"
            f"3. The endpoint '{endpoint}' points to the correct Azure OpenAI resource\n"
            f"4. The API key belongs to the same resource as the endpoint\n"
            f"5. Check Azure Portal for the supported API version for this deployment\n"
            f"Full error: {full_error}"
        )
    else:
        return False, (
            f"Error validating deployment '{deployment_name}' with all API versions tried: {', '.join(api_versions_to_try)}\n"
            f"Endpoint: {endpoint}\n"
            f"Last error: {full_error}"
        )


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

