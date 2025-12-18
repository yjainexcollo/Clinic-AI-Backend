"""
Configuration management for Clinic-AI application.

This module provides centralized configuration using Pydantic Settings
for environment-based configuration management.
"""

from typing import List, Optional

import os
from pathlib import Path

from pydantic import Field, field_validator, model_validator, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    model_config = SettingsConfigDict(env_prefix="MONGO_")

    uri: str = Field(
        default="", description="MongoDB connection URI"
    )
    db_name: str = Field(default="clinicai", description="MongoDB database name")
    collection: str = Field(default="clinicAi", description="MongoDB collection name")

    @validator("uri")
    def validate_mongo_uri(cls, v: str) -> str:
        """Validate MongoDB URI format."""
        if not v:
            raise ValueError("MongoDB URI is required. Please set MONGO_URI environment variable.")
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError(
                "MongoDB URI must start with 'mongodb://' or 'mongodb+srv://'"
            )
        return v


class OpenAISettings(BaseSettings):
    """OpenAI API configuration settings."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    api_key: str = Field(default="", description="OpenAI API key (not used when Azure OpenAI is configured)")
    model: str = Field(default="gpt-4o-mini", description="Model name for token calculations (defaults to match Azure OpenAI deployment)")
    max_tokens: int = Field(default=4000, description="Maximum tokens for responses")
    temperature: float = Field(
        default=0.3, description="Temperature for model responses"
    )

    @validator("api_key")
    def validate_api_key(cls, v: str) -> str:
        """Validate OpenAI API key format (optional - not used when Azure OpenAI is configured)."""
        if v and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v


class SecuritySettings(BaseSettings):
    """Security configuration settings."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    secret_key: str = Field(
        default="your-secret-key-change-in-production", description="JWT secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration time"
    )

    @validator("secret_key")
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key strength."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class CORSSettings(BaseSettings):
    """CORS configuration settings."""

    model_config = SettingsConfigDict(env_prefix="CORS_")

    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods",
    )
    allowed_headers: List[str] = Field(
        default=["*"], description="Allowed HTTP headers"
    )
    allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS"
    )

    @validator("allowed_origins", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse allowed origins from string or list."""
        if isinstance(v, str):
            # Handle JSON-like string format
            if v.startswith("[") and v.endswith("]"):
                import json

                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return [v.strip()]
            return [v.strip()]
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log format (json or text)")
    file_path: Optional[str] = Field(default=None, description="Log file path")

    @validator("level")
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class AudioSettings(BaseSettings):
    """Audio file processing configuration settings."""

    model_config = SettingsConfigDict(env_prefix="AUDIO_")

    max_size_mb: int = Field(default=50, description="Maximum audio file size in MB")
    allowed_formats: List[str] = Field(
        default=["mp3", "wav", "m4a", "flac", "ogg", "mpeg", "mpg"], 
        description="Allowed audio formats"
    )
    temp_dir: str = Field(default="/tmp/clinicai_audio", description="Temporary directory for audio files")

    @validator("max_size_mb")
    def validate_max_size(cls, v: int) -> int:
        """Validate max file size."""
        if v <= 0 or v > 500:
            raise ValueError("Max file size must be between 1 and 500 MB")
        return v


class SoapSettings(BaseSettings):
    """SOAP note generation configuration settings."""

    model_config = SettingsConfigDict(env_prefix="SOAP_")

    model: str = Field(default="gpt-4o-mini", description="Model for SOAP generation")
    max_tokens: int = Field(default=4000, description="Maximum tokens for SOAP generation")
    temperature: float = Field(default=0.3, description="Temperature for SOAP generation")
    include_highlights: bool = Field(default=True, description="Include highlights in SOAP")
    include_red_flags: bool = Field(default=True, description="Include red flags in SOAP")

    @validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class FileStorageSettings(BaseSettings):
    """File storage configuration settings."""

    model_config = SettingsConfigDict(env_prefix="FILE_")

    storage_type: str = Field(default="local", description="Storage type (local, s3, azure)")
    storage_path: str = Field(default="./storage", description="Local storage path")
    audio_storage_path: str = Field(default="./storage/audio", description="Audio files storage path")
    cleanup_after_hours: int = Field(default=24, description="Cleanup files after hours")
    save_audio_files: bool = Field(default=False, description="Whether to save original audio files (deprecated - use database storage)")

    @validator("storage_type")
    def validate_storage_type(cls, v: str) -> str:
        """Validate storage type."""
        valid_types = ["local", "s3", "azure"]
        if v not in valid_types:
            raise ValueError(f"Storage type must be one of: {valid_types}")
        return v


class AzureBlobSettings(BaseSettings):
    """Azure Blob Storage configuration settings."""

    model_config = SettingsConfigDict(env_prefix="AZURE_BLOB_")

    account_name: str = Field(default="", description="Azure Storage Account Name")
    account_key: str = Field(default="", description="Azure Storage Account Key")
    connection_string: str = Field(default="", description="Azure Storage Connection String")
    container_name: str = Field(default="clinicaiblobstorage", description="Blob container name")
    enable_cdn: bool = Field(default=False, description="Enable CDN for faster access")
    cdn_endpoint: Optional[str] = Field(None, description="CDN endpoint URL")
    default_expiry_hours: int = Field(default=24, description="Default expiry for signed URLs in hours")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    
    @validator("connection_string")
    def validate_connection_string(cls, v: str) -> str:
        """Validate Azure Storage connection string."""
        if v and not v.startswith("DefaultEndpointsProtocol="):
            raise ValueError("Invalid Azure Storage connection string format")
        return v


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="AZURE_OPENAI_")
    
    endpoint: str = Field(default="", description="Azure OpenAI endpoint URL")
    api_key: str = Field(default="", description="Azure OpenAI API key")
    api_version: str = Field(default="2024-12-01-preview", description="Azure OpenAI API version")
    deployment_name: str = Field(default="gpt-4o-mini", description="Azure OpenAI chat deployment name")
    
    @validator("endpoint")
    def validate_endpoint(cls, v: str) -> str:
        """Validate Azure OpenAI endpoint format."""
        if v and not (v.startswith("https://") and ".openai.azure.com" in v):
            # Allow empty string for optional use
            if v == "":
                return v
            raise ValueError("Invalid Azure OpenAI endpoint format. Must be: https://xxx.openai.azure.com/")
        return v


class AzureQueueSettings(BaseSettings):
    """Azure Queue Storage configuration settings with backward-compatible fallbacks."""
    
    model_config = SettingsConfigDict(env_prefix="AZURE_QUEUE_")
    
    connection_string: str = Field(default="", description="Azure Storage Connection String (same as Blob Storage)")
    queue_name: str = Field(default="transcription-queue", description="Queue name for transcription jobs")
    visibility_timeout: int = Field(default=3600, description="Message visibility timeout in seconds (60 min default, aligned with stale threshold)")
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts for failed jobs")
    max_dequeue_count: int = Field(default=5, description="Maximum dequeue count before moving to poison queue")
    processing_stale_seconds: int = Field(default=4000, description="Seconds before a processing job is considered stale and can be claimed (must be >= visibility_timeout)")
    poll_interval: int = Field(default=5, description="Worker poll interval in seconds")
    
    @classmethod
    def _get_connection_string_fallback(cls) -> str:
        """Get connection string with backward-compatible fallbacks."""
        # Try AZURE_QUEUE_CONNECTION_STRING first (current standard)
        conn_str = os.getenv("AZURE_QUEUE_CONNECTION_STRING", "")
        if conn_str:
            return conn_str
        # Fallback to AZURE_STORAGE_CONNECTION_STRING (old naming)
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        if conn_str:
            return conn_str
        # Fallback to AZURE_BLOB_CONNECTION_STRING (same storage account)
        conn_str = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
        if conn_str:
            return conn_str
        return ""
    
    @classmethod
    def _get_queue_name_fallback(cls) -> str:
        """Get queue name with backward-compatible fallback."""
        # Try AZURE_QUEUE_QUEUE_NAME first (current standard)
        queue_name = os.getenv("AZURE_QUEUE_QUEUE_NAME", "")
        if queue_name:
            return queue_name
        # Fallback to AZURE_QUEUE_NAME (old naming)
        queue_name = os.getenv("AZURE_QUEUE_NAME", "")
        if queue_name:
            return queue_name
        # Default fallback
        return "transcription-queue"
    
    @model_validator(mode='before')
    @classmethod
    def apply_fallbacks(cls, data: any) -> any:
        """Apply fallbacks before validation."""
        if isinstance(data, dict):
            # Apply connection_string fallback if empty
            if not data.get("connection_string"):
                fallback = cls._get_connection_string_fallback()
                if fallback:
                    data["connection_string"] = fallback
            
            # Apply queue_name fallback if default
            if data.get("queue_name") == "transcription-queue":
                fallback = cls._get_queue_name_fallback()
                if fallback and fallback != "transcription-queue":
                    data["queue_name"] = fallback
        
        return data
    
    @field_validator("connection_string", mode='before')
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        """Validate Azure Storage connection string format."""
        # Allow empty string for validation (will be caught later if needed)
        if not v:
            return v
        if not v.startswith("DefaultEndpointsProtocol="):
            raise ValueError("Invalid Azure Storage connection string format. Must start with 'DefaultEndpointsProtocol='")
        return v
    
    def model_post_init(self, __context) -> None:
        """Apply fallbacks after model initialization if values are still empty."""
        # Apply connection_string fallback if still empty
        if not self.connection_string:
            fallback = self._get_connection_string_fallback()
            if fallback:
                # Validate the fallback
                if not fallback.startswith("DefaultEndpointsProtocol="):
                    raise ValueError("Fallback Azure Storage connection string has invalid format")
                object.__setattr__(self, "connection_string", fallback)
        
        # Apply queue_name fallback if still default
        if self.queue_name == "transcription-queue":
            fallback = self._get_queue_name_fallback()
            if fallback and fallback != "transcription-queue":
                object.__setattr__(self, "queue_name", fallback)
        
        # Validate stale_seconds >= visibility_timeout to prevent churn
        if self.processing_stale_seconds < self.visibility_timeout:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"⚠️  PROCESSING_STALE_SECONDS ({self.processing_stale_seconds}s) < VISIBILITY_TIMEOUT ({self.visibility_timeout}s). "
                f"This can cause message churn. Setting stale_seconds to {self.visibility_timeout + 400}s"
            )
            object.__setattr__(self, "processing_stale_seconds", self.visibility_timeout + 400)


class AzureSpeechSettings(BaseSettings):
    """Azure Speech Service configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="AZURE_SPEECH_")
    
    subscription_key: str = Field(default="", description="Azure Speech Service subscription key")
    region: str = Field(default="", description="Azure Speech Service region (e.g., 'eastus', 'westus2')")
    endpoint: str = Field(default="", description="Azure Speech Service endpoint (optional, auto-generated if not provided)")
    enable_speaker_diarization: bool = Field(default=True, description="Enable speaker diarization (identifies different speakers)")
    enable_word_level_timestamps: bool = Field(default=False, description="Enable word-level timestamps (slower, disable for faster transcription)")
    max_speakers: int = Field(default=2, description="Maximum number of speakers to identify (default: 2 for Doctor/Patient)")
    transcription_mode: str = Field(default="batch", description="Transcription mode: 'batch' (recommended) or 'realtime'")
    batch_polling_interval: int = Field(default=5, description="Polling interval in seconds for batch transcription status")
    batch_max_wait_time: int = Field(default=1800, description="Maximum wait time in seconds for batch transcription (30 min)")
    
    @validator("region")
    def validate_region(cls, v: str) -> str:
        """Validate Azure region format."""
        if v and not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid Azure region format")
        return v
    
    @validator("transcription_mode")
    def validate_transcription_mode(cls, v: str) -> str:
        """Validate transcription mode."""
        if v.lower() not in ["batch", "realtime"]:
            raise ValueError("Transcription mode must be 'batch' or 'realtime'")
        return v.lower()


class IntakeSettings(BaseSettings):
    """Intake session configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="INTAKE_")
    
    max_questions: int = Field(
        default=12,
        description="Maximum intake questions allowed (default: 12)"
    )
    
    @validator("max_questions")
    def validate_max_questions(cls, v: int) -> int:
        """Validate max questions is reasonable."""
        if not 1 <= v <= 20:
            raise ValueError("max_questions must be between 1 and 20")
        return v


class LLMInteractionSettings(BaseSettings):
    """LLM interaction logging configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="LLM_INTERACTION_")
    
    enabled: bool = Field(
        default=True,
        description="Enable LLM interaction logging to database (default: True)"
    )
    log_system_prompts: bool = Field(
        default=False,
        description="Include system prompts in logs (default: False for privacy/optimization)"
    )
    batch_mode: bool = Field(
        default=False,
        description="Use batch mode for logging (reduces database writes, default: False)"
    )
    batch_size: int = Field(
        default=10,
        description="Batch size for batch mode (default: 10)"
    )
    enable_debug_logging: bool = Field(
        default=False,
        description="Enable debug logging for LLM interaction tracking (default: False)"
    )
    
    @validator("batch_size")
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size."""
        if v < 1 or v > 100:
            raise ValueError("Batch size must be between 1 and 100")
        return v


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application settings
    app_name: str = Field(default="Clinic-AI", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    app_env: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    port: int = Field(default=8000, description="Application port")
    host: str = Field(default="0.0.0.0", description="Application host")

    # Transcription / LLM settings
    transcription_simple_max_chars: int = Field(
        default=20000,
        env="TRANSCRIPTION_SIMPLE_MAX_CHARS",
        description="Max transcript chars for simple LLM path before chunking",
    )
    # Stuck transcription sweeper (optional safety net)
    transcription_stuck_sweeper_enabled: bool = Field(
        default=False,
        env="TRANSCRIPTION_STUCK_SWEEPER_ENABLED",
        description="Enable periodic sweeper to re-enqueue stuck queued transcription jobs",
    )
    transcription_stuck_sweeper_interval_seconds: int = Field(
        default=300,
        env="TRANSCRIPTION_STUCK_SWEEPER_INTERVAL_SECONDS",
        description="Interval in seconds between sweeper runs (default: 5 minutes)",
    )
    transcription_stuck_threshold_seconds: int = Field(
        default=600,
        env="TRANSCRIPTION_STUCK_THRESHOLD_SECONDS",
        description="Threshold in seconds to consider a queued job stuck (default: 10 minutes)",
    )

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    soap: SoapSettings = Field(default_factory=SoapSettings)
    file_storage: FileStorageSettings = Field(default_factory=FileStorageSettings)
    azure_blob: AzureBlobSettings = Field(default_factory=AzureBlobSettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    azure_queue: AzureQueueSettings = Field(default_factory=AzureQueueSettings)
    azure_speech: AzureSpeechSettings = Field(default_factory=AzureSpeechSettings)
    intake: IntakeSettings = Field(default_factory=IntakeSettings)
    llm_interaction: LLMInteractionSettings = Field(default_factory=LLMInteractionSettings)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Try to load secrets from Azure Key Vault if available
        # This allows secure secret management in production
        key_vault_secrets = {}
        try:
            from .key_vault import get_key_vault_service
            key_vault = get_key_vault_service()
            
            if key_vault and key_vault.is_available:
                # Load secrets from Key Vault (with fallback to env vars)
                # Secret names in Key Vault should match these patterns
                key_vault_secrets = {
                    "OPENAI_API_KEY": key_vault.get_secret("OPENAI-API-KEY"),
                    "MONGO_URI": key_vault.get_secret("MONGO-URI"),
                    "ENCRYPTION_KEY": key_vault.get_secret("ENCRYPTION-KEY"),
                    "SECURITY_SECRET_KEY": key_vault.get_secret("SECURITY-SECRET-KEY"),
                    "AZURE_BLOB_CONNECTION_STRING": key_vault.get_secret("AZURE-BLOB-CONNECTION-STRING"),
                    "AZURE_BLOB_ACCOUNT_NAME": key_vault.get_secret("AZURE-BLOB-ACCOUNT-NAME"),
                    "AZURE_BLOB_ACCOUNT_KEY": key_vault.get_secret("AZURE-BLOB-ACCOUNT-KEY"),
                    # Azure OpenAI secrets
                    "AZURE_OPENAI_ENDPOINT": key_vault.get_secret("AZURE-OPENAI-ENDPOINT"),
                    "AZURE_OPENAI_API_KEY": key_vault.get_secret("AZURE-OPENAI-API-KEY"),
                    "AZURE_OPENAI_API_VERSION": key_vault.get_secret("AZURE-OPENAI-API-VERSION"),
                    "AZURE_OPENAI_DEPLOYMENT_NAME": key_vault.get_secret("AZURE-OPENAI-DEPLOYMENT-NAME"),
                }
                # Only use Key Vault values if they exist (don't override env vars that are already set)
                for key, value in key_vault_secrets.items():
                    if value and not os.getenv(key):
                        os.environ[key] = value
                
                import logging
                logger = logging.getLogger("clinicai")
                logger.info("✅ Loaded secrets from Azure Key Vault")
        except Exception as e:
            import logging
            logger = logging.getLogger("clinicai")
            logger.debug(f"Key Vault integration skipped (using environment variables): {e}")
        
        # Override sub-settings with environment variables
        self.database = DatabaseSettings()
        self.openai = OpenAISettings()
        self.security = SecuritySettings()
        self.cors = CORSSettings()
        self.logging = LoggingSettings()
        self.audio = AudioSettings()
        self.soap = SoapSettings()
        self.file_storage = FileStorageSettings()
        self.azure_blob = AzureBlobSettings()
        self.azure_openai = AzureOpenAISettings()
        self.azure_queue = AzureQueueSettings()
        self.azure_speech = AzureSpeechSettings()
        self.intake = IntakeSettings()

    @validator("app_env")
    def validate_app_env(cls, v: str) -> str:
        """Validate application environment."""
        valid_envs = ["development", "staging", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"App environment must be one of: {valid_envs}")
        return v.lower()

    @validator("port")
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.app_env == "testing"


# Global settings instance (loaded after attempting to read .env)
_settings: Optional[Settings] = None


def _load_env_file_if_available() -> None:
    """Best-effort load of .env by searching current and parent directories.

    This helps in environments where the working directory isn't the backend folder
    and pydantic's env_file doesn't get resolved as expected.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        load_dotenv = None

    if load_dotenv is None:
        return

    cwd = Path(os.getcwd()).resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            # Do not override already-set environment variables
            load_dotenv(dotenv_path=str(candidate), override=False)
            break


def get_settings() -> Settings:
    """Get application settings instance (lazy-init with .env discovery)."""
    global _settings
    if _settings is None:
        _load_env_file_if_available()
        _settings = Settings()
    return _settings