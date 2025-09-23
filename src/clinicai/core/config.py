"""
Configuration management for Clinic-AI application.

This module provides centralized configuration using Pydantic Settings
for environment-based configuration management.
"""

from typing import List, Optional

import os
from pathlib import Path

from pydantic import Field, validator
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

    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="gpt-4", description="Default OpenAI model")
    max_tokens: int = Field(default=8000, description="Maximum tokens for responses")
    temperature: float = Field(
        default=0.7, description="Temperature for model responses"
    )

    @validator("api_key")
    def validate_api_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
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


class WhisperSettings(BaseSettings):
    """Whisper transcription configuration settings."""

    model_config = SettingsConfigDict(env_prefix="WHISPER_")

    model: str = Field(default="base", description="Whisper model size")
    language: str = Field(default="en", description="Audio language")
    medical_context: bool = Field(default=True, description="Enable medical context processing")
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory to cache Whisper model weights (set to persistent disk on Render)"
    )

    @validator("model")
    def validate_model(cls, v: str) -> str:
        """Validate Whisper model."""
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if v not in valid_models:
            raise ValueError(f"Whisper model must be one of: {valid_models}")
        return v


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

    model: str = Field(default="gpt-4", description="Model for SOAP generation")
    max_tokens: int = Field(default=2000, description="Maximum tokens for SOAP generation")
    temperature: float = Field(default=0.3, description="Temperature for SOAP generation")
    include_highlights: bool = Field(default=True, description="Include highlights in SOAP")
    include_red_flags: bool = Field(default=True, description="Include red flags in SOAP")

    @validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class MistralSettings(BaseSettings):
    """Mistral AI configuration settings."""

    model_config = SettingsConfigDict(env_prefix="MISTRAL_")

    api_key: str = Field(default="", description="Mistral API key")
    vision_model: str = Field(default="pixtral-12b-2409", description="Vision model for image analysis")
    chat_model: str = Field(default="mistral-large-latest", description="Chat model for text parsing")
    max_tokens: int = Field(default=2000, description="Maximum tokens for responses")
    temperature: float = Field(default=0.1, description="Temperature for model responses")

    @validator("api_key")
    def validate_api_key(cls, v: str) -> str:
        """Validate Mistral API key format."""
        # Mistral API keys are alphanumeric strings, no specific prefix required
        if v and len(v) < 10:
            raise ValueError("Mistral API key appears to be too short")
        return v

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
    cleanup_after_hours: int = Field(default=24, description="Cleanup files after hours")

    @validator("storage_type")
    def validate_storage_type(cls, v: str) -> str:
        """Validate storage type."""
        valid_types = ["local", "s3", "azure"]
        if v not in valid_types:
            raise ValueError(f"Storage type must be one of: {valid_types}")
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

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    soap: SoapSettings = Field(default_factory=SoapSettings)
    mistral: MistralSettings = Field(default_factory=MistralSettings)
    file_storage: FileStorageSettings = Field(default_factory=FileStorageSettings)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override sub-settings with environment variables
        self.database = DatabaseSettings()
        self.openai = OpenAISettings()
        self.security = SecuritySettings()
        self.cors = CORSSettings()
        self.logging = LoggingSettings()
        self.whisper = WhisperSettings()
        self.audio = AudioSettings()
        self.soap = SoapSettings()
        self.mistral = MistralSettings()
        self.file_storage = FileStorageSettings()

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