"""
Azure Blob Storage service for file management.
Handles upload, download, and management of files in Azure Blob Storage.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging
from pathlib import Path

from azure.storage.blob import BlobServiceClient, BlobClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azure.storage.blob import ContentSettings

from ...core.config import get_settings

logger = logging.getLogger(__name__)


class AzureBlobStorageService:
    """Azure Blob Storage service for file operations."""
    
    def __init__(self):
        self.settings = get_settings().azure_blob
        self._client: Optional[BlobServiceClient] = None
        self._container_client = None
        
    @property
    def client(self) -> BlobServiceClient:
        """Get or create BlobServiceClient."""
        if self._client is None:
            if not self.settings.connection_string:
                raise ValueError("Azure Blob Storage connection string is required")
            
            self._client = BlobServiceClient.from_connection_string(
                self.settings.connection_string
            )
            logger.info(f"✅ Azure Blob Storage client initialized for account: {self.settings.account_name}")
        
        return self._client
    
    @property
    def container_client(self):
        """Get or create container client."""
        if self._container_client is None:
            self._container_client = self.client.get_container_client(
                self.settings.container_name
            )
        return self._container_client
    
    async def ensure_container_exists(self) -> bool:
        """Ensure the blob container exists."""
        try:
            self.container_client.create_container()
            logger.info(f"✅ Created blob container: {self.settings.container_name}")
            return True
        except ResourceExistsError:
            logger.info(f"📁 Blob container already exists: {self.settings.container_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create blob container: {e}")
            return False
    
    def _generate_blob_path(self, file_type: str, filename: str, **kwargs) -> str:
        """Generate blob path based on file type and metadata."""
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        
        if file_type == "audio":
            if kwargs.get("adhoc_id"):
                return f"audio/adhoc/{kwargs['adhoc_id']}/{filename}"
            elif kwargs.get("patient_id") and kwargs.get("visit_id"):
                return f"audio/visit/{kwargs['patient_id']}/{kwargs['visit_id']}/{filename}"
            else:
                return f"audio/temp/{timestamp}/{filename}"
        
        elif file_type == "image":
            if kwargs.get("patient_id") and kwargs.get("visit_id"):
                return f"images/medication/{kwargs['patient_id']}/{kwargs['visit_id']}/{filename}"
            else:
                return f"images/temp/{timestamp}/{filename}"
        
        else:
            return f"files/{file_type}/{timestamp}/{filename}"
    
    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: str,
        file_type: str = "file",
        **metadata
    ) -> Dict[str, Any]:
        """
        Upload file to Azure Blob Storage.
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            content_type: MIME type
            file_type: Type of file (audio, image, etc.)
            **metadata: Additional metadata for path generation
        
        Returns:
            Dict with blob information
        """
        try:
            # Ensure container exists
            await self.ensure_container_exists()
            
            # Generate unique filename if needed
            file_ext = Path(filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            
            # Generate blob path
            blob_path = self._generate_blob_path(file_type, unique_filename, **metadata)
            
            # Upload file
            blob_client = self.client.get_blob_client(
                container=self.settings.container_name,
                blob=blob_path
            )
            
            # Set content settings
            content_settings = ContentSettings(content_type=content_type)
            
            # Upload with metadata
            upload_result = blob_client.upload_blob(
                file_data,
                content_settings=content_settings,
                metadata={
                    "original_filename": filename,
                    "file_type": file_type,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    **{k: str(v) for k, v in metadata.items()}
                },
                overwrite=True
            )
            
            # Generate public URL
            blob_url = blob_client.url
            
            logger.info(f"✅ Uploaded file to blob storage: {blob_path}")
            
            return {
                "file_id": str(uuid.uuid4()),
                "blob_path": blob_path,
                "container_name": self.settings.container_name,
                "original_filename": filename,
                "content_type": content_type,
                "file_size": len(file_data),
                "blob_url": blob_url,
                "metadata": metadata,
                "uploaded_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to upload file to blob storage: {e}")
            raise
    
    async def download_file(self, blob_path: str) -> bytes:
        """
        Download file from Azure Blob Storage.
        
        Args:
            blob_path: Path to the blob
            
        Returns:
            File content as bytes
        """
        try:
            blob_client = self.client.get_blob_client(
                container=self.settings.container_name,
                blob=blob_path
            )
            
            download_stream = blob_client.download_blob()
            file_data = download_stream.readall()
            
            logger.info(f"✅ Downloaded file from blob storage: {blob_path}")
            return file_data
            
        except ResourceNotFoundError:
            logger.error(f"❌ File not found in blob storage: {blob_path}")
            raise FileNotFoundError(f"File not found: {blob_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download file from blob storage: {e}")
            raise
    
    async def delete_file(self, blob_path: str) -> bool:
        """
        Delete file from Azure Blob Storage.
        
        Args:
            blob_path: Path to the blob
            
        Returns:
            True if deleted successfully
        """
        try:
            blob_client = self.client.get_blob_client(
                container=self.settings.container_name,
                blob=blob_path
            )
            
            blob_client.delete_blob()
            logger.info(f"✅ Deleted file from blob storage: {blob_path}")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"⚠️ File not found for deletion: {blob_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to delete file from blob storage: {e}")
            return False
    
    def generate_signed_url(
        self,
        blob_path: str,
        expires_in_hours: Optional[int] = None,
        permissions: str = "r"
    ) -> str:
        """
        Generate a signed URL for secure access to blob.
        
        Args:
            blob_path: Path to the blob
            expires_in_hours: URL expiry time in hours
            permissions: Blob permissions (r=read, w=write, etc.)
            
        Returns:
            Signed URL
        """
        try:
            if expires_in_hours is None:
                expires_in_hours = self.settings.default_expiry_hours
            
            # Extract account_key and account_name from connection_string if not provided directly
            account_key = self.settings.account_key
            account_name = self.settings.account_name
            
            if not account_key and self.settings.connection_string:
                # Parse connection string to extract account_key
                # Format: DefaultEndpointsProtocol=https;AccountName=xxx;AccountKey=xxx;EndpointSuffix=core.windows.net
                conn_parts = self.settings.connection_string.split(';')
                for part in conn_parts:
                    if part.startswith('AccountKey='):
                        account_key = part.split('=', 1)[1]
                    elif part.startswith('AccountName=') and not account_name:
                        account_name = part.split('=', 1)[1]
            
            if not account_key:
                raise ValueError(
                    "Azure Blob Storage account_key is required for generating signed URLs. "
                    "Either set AZURE_BLOB_ACCOUNT_KEY or ensure AZURE_BLOB_CONNECTION_STRING contains AccountKey."
                )
            
            if not account_name:
                raise ValueError(
                    "Azure Blob Storage account_name is required for generating signed URLs. "
                    "Either set AZURE_BLOB_ACCOUNT_NAME or ensure AZURE_BLOB_CONNECTION_STRING contains AccountName."
                )
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.settings.container_name,
                blob_name=blob_path,
                account_key=account_key,
                permission=BlobSasPermissions(read=True) if permissions == "r" else BlobSasPermissions(read=True, write=True),
                expiry=datetime.utcnow() + timedelta(hours=expires_in_hours)
            )
            
            # Generate signed URL
            blob_url = f"https://{account_name}.blob.core.windows.net/{self.settings.container_name}/{blob_path}"
            signed_url = f"{blob_url}?{sas_token}"
            
            logger.info(f"✅ Generated signed URL for: {blob_path}")
            return signed_url
            
        except Exception as e:
            logger.error(f"❌ Failed to generate signed URL: {e}")
            raise
    
    async def list_files(self, prefix: str = "", max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List files in blob storage.
        
        Args:
            prefix: Prefix to filter files
            max_results: Maximum number of results
            
        Returns:
            List of file information
        """
        try:
            files = []
            
            for blob in self.container_client.list_blobs(name_starts_with=prefix):
                files.append({
                    "name": blob.name,
                    "size": blob.size,
                    "content_type": blob.content_settings.content_type if blob.content_settings else None,
                    "last_modified": blob.last_modified,
                    "metadata": blob.metadata or {}
                })
                
                if len(files) >= max_results:
                    break
            
            logger.info(f"✅ Listed {len(files)} files with prefix: {prefix}")
            return files
            
        except Exception as e:
            logger.error(f"❌ Failed to list files: {e}")
            raise
    
    async def get_file_info(self, blob_path: str) -> Optional[Dict[str, Any]]:
        """
        Get file information from blob storage.
        
        Args:
            blob_path: Path to the blob
            
        Returns:
            File information or None if not found
        """
        try:
            blob_client = self.client.get_blob_client(
                container=self.settings.container_name,
                blob=blob_path
            )
            
            properties = blob_client.get_blob_properties()
            
            return {
                "name": properties.name,
                "size": properties.size,
                "content_type": properties.content_settings.content_type if properties.content_settings else None,
                "last_modified": properties.last_modified,
                "metadata": properties.metadata or {},
                "etag": properties.etag
            }
            
        except ResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get file info: {e}")
            raise


# Factory function
def get_azure_blob_service() -> AzureBlobStorageService:
    """Get Azure Blob Storage service instance."""
    return AzureBlobStorageService()
