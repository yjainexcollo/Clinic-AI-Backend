"""
Azure Key Vault integration for secrets management.

This module provides secure access to secrets stored in Azure Key Vault.
It supports both Managed Identity (for Azure App Service) and DefaultAzureCredential
(for local development).
"""

import logging
import os
from typing import Optional

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

logger = logging.getLogger("clinicai")

# Suppress Azure SDK warnings for local development (IMDS endpoint not available)
# These warnings are expected when running locally without Azure credentials
azure_logger = logging.getLogger("azure.identity")
azure_logger.setLevel(logging.ERROR)  # Only show errors, suppress warnings/info
azure_core_logger = logging.getLogger("azure.core")
azure_core_logger.setLevel(logging.ERROR)


class AzureKeyVaultService:
    """Azure Key Vault service for secrets management."""

    def __init__(self, vault_name: str):
        """
        Initialize Azure Key Vault service.

        Args:
            vault_name: Name of the Azure Key Vault (without .vault.azure.net)
        """
        self.vault_name = vault_name
        self.vault_url = f"https://{vault_name}.vault.azure.net/"
        self._client: Optional[SecretClient] = None
        self._available = False
        self._is_azure_env = self._check_if_azure_environment()

    @staticmethod
    def _check_if_azure_environment() -> bool:
        """Check if running on Azure App Service."""
        # Check for Azure App Service environment variables
        azure_indicators = [
            "WEBSITE_INSTANCE_ID",
            "WEBSITE_SITE_NAME",
            "WEBSITE_RESOURCE_GROUP",
            "APPSETTING_WEBSITE_SITE_NAME",
        ]
        return any(os.getenv(indicator) for indicator in azure_indicators)

    @property
    def client(self) -> Optional[SecretClient]:
        """Get or create SecretClient."""
        if self._client is None:
            credential = None
            
            # Only try Managed Identity if running on Azure
            if self._is_azure_env:
                try:
                    credential = ManagedIdentityCredential()
                    logger.info("Using Managed Identity for Key Vault authentication")
                except Exception as e:
                    logger.debug(f"Managed Identity not available: {e}, trying DefaultAzureCredential")
            
            # Try DefaultAzureCredential (works for Azure CLI, VS Code, Azure PowerShell, etc.)
            if credential is None:
                try:
                    credential = DefaultAzureCredential()
                    logger.debug("Using DefaultAzureCredential for Key Vault authentication")
                except Exception as e2:
                    logger.debug(f"DefaultAzureCredential not available: {e2}. Will use environment variables.")
                    self._available = False
                    return None

            try:
                self._client = SecretClient(vault_url=self.vault_url, credential=credential)
                # Mark as available - actual access will be tested when get_secret() is called
                # This avoids the expensive list_properties_of_secrets() call that causes slow startup
                self._available = True
                logger.info(f"✅ Azure Key Vault client initialized: {self.vault_name}")

            except Exception as e:
                logger.debug(f"Failed to create Key Vault client: {e}. Will use environment variables.")
                self._available = False
                return None

        return self._client

    @property
    def is_available(self) -> bool:
        """Check if Key Vault is available and accessible."""
        if self._client is None:
            self.client  # Try to initialize
        return self._available

    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret from Key Vault.

        Args:
            secret_name: Name of the secret in Key Vault
            default: Default value if secret not found (optional)

        Returns:
            Secret value, default value, or None
        """
        if not self.is_available or not self.client:
            logger.debug(f"Key Vault not available, falling back to environment variable for: {secret_name}")
            # Fallback to environment variable
            env_key = secret_name.replace("-", "_").upper()
            return os.getenv(env_key, default)

        try:
            secret = self.client.get_secret(secret_name)
            logger.info(f"✅ Retrieved secret from Key Vault: {secret_name}")
            return secret.value
        except AzureError as e:
            # Mark as unavailable if authentication fails (likely local dev)
            error_str = str(e).lower()
            if "managedidentitycredential" in error_str or "imds" in error_str or "authentication unavailable" in error_str:
                logger.debug(f"Key Vault authentication unavailable (likely local dev), falling back to environment variable for: {secret_name}")
                self._available = False  # Mark as unavailable to prevent future attempts
            else:
                logger.debug(f"Failed to get secret from Key Vault: {secret_name}, error: {e}")
            
            # Fallback to environment variable
            env_key = secret_name.replace("-", "_").upper()
            env_value = os.getenv(env_key)
            if env_value:
                logger.debug(f"Using environment variable for: {secret_name}")
                return env_value
            if default is not None:
                logger.debug(f"Using default value for: {secret_name}")
                return default
            return None

    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """
        Set secret in Key Vault.

        Args:
            secret_name: Name of the secret
            secret_value: Value to store

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available or not self.client:
            logger.error(f"Key Vault not available, cannot set secret: {secret_name}")
            return False

        try:
            self.client.set_secret(secret_name, secret_value)
            logger.info(f"✅ Set secret in Key Vault: {secret_name}")
            return True
        except AzureError as e:
            logger.error(f"❌ Failed to set secret in Key Vault: {secret_name}, error: {e}")
            return False

    def list_secrets(self) -> list:
        """
        List all secret names in Key Vault.

        Returns:
            List of secret names
        """
        if not self.is_available or not self.client:
            return []

        try:
            secrets = list(self.client.list_properties_of_secrets())
            return [secret.name for secret in secrets]
        except AzureError as e:
            logger.warning(f"⚠️  Failed to list secrets from Key Vault: {e}")
            return []


# Global instance
_key_vault_service: Optional[AzureKeyVaultService] = None


def get_key_vault_service() -> Optional[AzureKeyVaultService]:
    """
    Get Azure Key Vault service instance (singleton).

    Returns:
        AzureKeyVaultService instance or None if not configured
    """
    global _key_vault_service

    # Allow explicit disable via env (e.g., local/dev wants .env only)
    if os.getenv("DISABLE_KEY_VAULT", "").lower() == "true":
        logger.debug("Key Vault integration disabled via DISABLE_KEY_VAULT")
        return None

    if _key_vault_service is None:
        vault_name = os.getenv("AZURE_KEY_VAULT_NAME", "Clinic-ai-key-vault")

        # Only initialize if vault name is provided
        if vault_name:
            try:
                _key_vault_service = AzureKeyVaultService(vault_name)
                if not _key_vault_service.is_available:
                    logger.debug(f"Key Vault '{vault_name}' is not available. Using environment variables.")
            except Exception as e:
                logger.debug(f"Key Vault initialization failed: {e}. Using environment variables.")
                _key_vault_service = None
        else:
            logger.debug("AZURE_KEY_VAULT_NAME not set, Key Vault integration disabled")

    return _key_vault_service
