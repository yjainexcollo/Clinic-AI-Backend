"""
Free tier optimizations for Azure App Service
"""
import os
from typing import Dict, Any


def get_free_tier_config() -> Dict[str, Any]:
    """Get configuration optimized for free tier"""
    return {
        "openai_model": "gpt-4o-mini",  # Cheaper model
        "max_file_size": 10 * 1024 * 1024,  # 10MB limit
        "cache_ttl": 300,  # 5 minutes cache
        "max_workers": 1,  # Single worker
        "timeout": 30,  # 30 second timeout
        "memory_limit": 1024 * 1024 * 1024,  # 1GB limit
        "enable_caching": True,
        "use_openai_api": True,  # Use OpenAI API for all AI services
        "transcription_service": "openai",  # Use OpenAI Whisper API for free tier
        "whisper_model": "tiny",  # Lightest model for free tier
    }


def is_free_tier() -> bool:
    """Check if running on free tier"""
    # Check for Azure App Service free tier indicators
    return (
        os.environ.get("WEBSITE_SKU", "").upper() == "FREE" or
        os.environ.get("WEBSITE_INSTANCE_ID", "").startswith("Free") or
        os.environ.get("APP_SERVICE_PLAN_TIER", "").upper() == "FREE"
    )


def get_optimized_settings() -> Dict[str, Any]:
    """Get optimized settings based on tier"""
    if is_free_tier():
        return get_free_tier_config()
    else:
        # Production tier settings
        return {
            "openai_model": "gpt-4o",
            "max_file_size": 50 * 1024 * 1024,  # 50MB
            "cache_ttl": 3600,  # 1 hour
            "max_workers": 4,
            "timeout": 120,
            "memory_limit": 4 * 1024 * 1024 * 1024,  # 4GB
            "enable_caching": True,
            "use_openai_api": True,
            "transcription_service": "local",  # Use local Whisper for production
            "whisper_model": "base",  # Better model for production
        }
