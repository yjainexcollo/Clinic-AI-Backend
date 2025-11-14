"""
Azure Key Vault integration for secrets management.

This module provides secure access to secrets stored in Azure Key Vault.
It supports both Managed Identity (for Azure App Service) and DefaultAzureCredential
(for local development).
"""
import os
import logging
from typing import Optional
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import AzureError

logger = logging.getLogger("clinicai")


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
        
    @property
    def client(self) -> Optional[SecretClient]:
        """Get or create SecretClient."""
        if self._client is None:
            try:
                # Try Managed Identity first (for Azure App Service)
                # This is the recommended approach for production
                credential = ManagedIdentityCredential()
                logger.info("Using Managed Identity for Key Vault authentication")
            except Exception as e:
                logger.debug(f"Managed Identity not available: {e}, trying DefaultAzureCredential")
                try:
                    # Fallback to DefaultAzureCredential (for local dev)
                    # This supports: Azure CLI, VS Code, Azure PowerShell, etc.
                    credential = DefaultAzureCredential()
                    logger.info("Using DefaultAzureCredential for Key Vault authentication")
                except Exception as e2:
                    logger.warning(f"Failed to initialize Azure credentials: {e2}")
                    return None
            
            try:
                self._client = SecretClient(
                    vault_url=self.vault_url,
                    credential=credential
                )
                # Mark as available - actual access will be tested when get_secret() is called
                # This avoids the expensive list_properties_of_secrets() call that causes slow startup
                self._available = True
                logger.info(f"✅ Azure Key Vault client initialized: {self.vault_name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create Key Vault client: {e}")
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
            logger.warning(f"⚠️  Failed to get secret from Key Vault: {secret_name}, error: {e}")
            # Fallback to environment variable
            env_key = secret_name.replace("-", "_").upper()
            env_value = os.getenv(env_key)
            if env_value:
                logger.info(f"   Using environment variable for: {secret_name}")
                return env_value
            if default is not None:
                logger.info(f"   Using default value for: {secret_name}")
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
    
    if _key_vault_service is None:
        vault_name = os.getenv("AZURE_KEY_VAULT_NAME", "Clinic-ai-key-vault")
        
        # Only initialize if vault name is provided
        if vault_name:
            try:
                _key_vault_service = AzureKeyVaultService(vault_name)
                if not _key_vault_service.is_available:
                    logger.warning(f"⚠️  Key Vault '{vault_name}' is not available. Using environment variables.")
            except Exception as e:
                logger.warning(f"⚠️  Key Vault initialization failed: {e}. Using environment variables.")
                _key_vault_service = None
        else:
            logger.info("AZURE_KEY_VAULT_NAME not set, Key Vault integration disabled")
    
    return _key_vault_service

