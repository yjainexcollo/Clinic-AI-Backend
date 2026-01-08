"""
Authentication service for HIPAA compliance.
Supports API key authentication (simple) and can be extended to Azure AD.

This service validates API keys and tokens to identify users accessing PHI.
All PHI access must be authenticated for HIPAA compliance.
"""

import logging
import os
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service for validating users and API keys"""

    def __init__(self):
        """Initialize authentication service with API keys from environment or Key Vault"""
        self.api_keys: dict[str, str] = {}
        self._load_api_keys()

    def _load_api_keys(self) -> None:
        """Load API keys from environment variables or Azure Key Vault"""
        # Try Azure Key Vault first (for production)
        try:
            from .key_vault import get_key_vault_service

            key_vault = get_key_vault_service()

            if key_vault and key_vault.is_available:
                api_keys_str = key_vault.get_secret("API-KEYS")
                if api_keys_str:
                    self._parse_api_keys(api_keys_str)
                    logger.info("✅ Loaded API keys from Azure Key Vault")
                    return
        except Exception as e:
            logger.debug(f"Key Vault not available or API-KEYS not found: {e}")

        # Fallback to environment variable
        api_keys_str = os.getenv("API_KEYS", "")
        if api_keys_str:
            self._parse_api_keys(api_keys_str)
            logger.info("✅ Loaded API keys from environment variables")
        else:
            logger.warning("⚠️  No API keys configured. Authentication will fail for all requests.")

    def _parse_api_keys(self, api_keys_str: str) -> None:
        """
        Parse API keys string.
        Format: "key1:user1,key2:user2" (comma-separated key:user pairs)
        """
        if not api_keys_str or not api_keys_str.strip():
            return

        for pair in api_keys_str.split(","):
            pair = pair.strip()
            if not pair:
                continue

            if ":" in pair:
                parts = pair.split(":", 1)
                key = parts[0].strip()
                user_id = parts[1].strip()
                if key and user_id:
                    self.api_keys[key] = user_id
            else:
                # If no colon, use the key itself as user identifier
                self.api_keys[pair] = pair

    def validate_api_key(self, api_key: Optional[str]) -> str:
        """
        Validate API key and return user ID.

        Args:
            api_key: API key to validate

        Returns:
            User ID associated with the API key

        Raises:
            HTTPException: If API key is invalid or missing
        """
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Provide X-API-Key header or Authorization Bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Remove "Bearer " prefix if present
        if api_key.startswith("Bearer "):
            api_key = api_key[7:].strip()

        # Check API keys dictionary
        if api_key in self.api_keys:
            user_id = self.api_keys[api_key]
            logger.debug(f"✅ API key validated for user: {user_id}")
            return user_id

        # If not found in API keys, check if it's a valid token format
        # For future Azure AD integration, this is where JWT validation would happen
        # For now, reject unknown keys
        logger.warning(f"❌ Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def get_user_from_request(self, api_key: Optional[str] = None, auth_header: Optional[str] = None) -> str:
        """
        Extract and validate user ID from request headers.

        Priority:
        1. X-API-Key header
        2. Authorization Bearer token
        3. Authorization Basic auth

        Args:
            api_key: API key from X-API-Key header
            auth_header: Authorization header value

        Returns:
            User ID string

        Raises:
            HTTPException: If authentication fails
        """
        # Try API key first
        if api_key:
            return self.validate_api_key(api_key)

        # Try Authorization header
        if auth_header:
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
                return self.validate_api_key(token)
            elif auth_header.startswith("Basic "):
                # Basic auth - for future implementation
                # For now, extract username from Basic auth
                import base64

                try:
                    credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
                    username, _ = credentials.split(":", 1)
                    return username
                except Exception:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid Basic authentication",
                        headers={"WWW-Authenticate": "Basic"},
                    )

        # No authentication provided
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide X-API-Key header or Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Global instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get global authentication service instance (singleton)"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
