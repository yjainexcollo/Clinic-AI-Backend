"""
Azure Blob Storage service for file management.
Handles upload, download, and management of files in Azure Blob Storage.
"""

import os
import uuid
import time
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from azure.storage.blob import BlobServiceClient, BlobClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError, AzureError
from azure.storage.blob import ContentSettings
from azure.core.pipeline.transport import RequestsTransport
from azure.core.pipeline.policies import RetryPolicy

from ...core.config import get_settings

logger = logging.getLogger(__name__)


async def run_blocking(func, *args, **kwargs):
    """
    Run a blocking function in an executor to avoid blocking the event loop.
    
    Args:
        func: The blocking function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function call
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class AzureBlobStorageService:
    """Azure Blob Storage service for file operations."""
    
    def __init__(self):
        self.settings = get_settings().azure_blob
        self._client: Optional[BlobServiceClient] = None
        self._container_client = None
        # Timeout configuration (30s connect, 300s read to allow for large file uploads)
        # For a 4.44MB file, 60s is too short if network is slow
        self._connection_timeout = 30
        self._read_timeout = 300  # Increased to 300s to match asyncio timeout
        
    @property
    def client(self) -> BlobServiceClient:
        """Get or create BlobServiceClient with connection timeouts."""
        if self._client is None:
            if not self.settings.connection_string:
                raise ValueError("Azure Blob Storage connection string is required")
            
            # Configure connection timeouts to prevent hanging
            # The Azure SDK uses requests library under the hood
            # We configure timeouts by setting session.timeout which applies to all requests
            
            # Create a session with timeout configuration
            # The Azure SDK's RequestsTransport uses requests.Session internally
            session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                connect=3,
                read=3,
                backoff_factor=0.8,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            
            # Create adapter with retry strategy
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=10
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Monkey-patch the session's request method to always include timeout
            # This ensures all requests made through this session have a timeout
            original_request = session.request
            def request_with_timeout(method, url, **kwargs):
                # Add timeout if not already specified
                if 'timeout' not in kwargs:
                    kwargs['timeout'] = (self._connection_timeout, self._read_timeout)
                return original_request(method, url, **kwargs)
            session.request = request_with_timeout
            
            # Also patch send method as a fallback (some code paths might use send directly)
            original_send = session.send
            def send_with_timeout(request, **kwargs):
                if 'timeout' not in kwargs:
                    kwargs['timeout'] = (self._connection_timeout, self._read_timeout)
                return original_send(request, **kwargs)
            session.send = send_with_timeout
            
            # Create transport with custom session
            # All requests through this transport will have timeouts enforced
            transport = RequestsTransport(session=session)
            
            self._client = BlobServiceClient.from_connection_string(
                self.settings.connection_string,
                transport=transport
            )
            logger.info(f"✅ Azure Blob Storage client initialized for account: {self.settings.account_name} with connection_timeout=30s, read_timeout=60s")
        
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
        """Ensure the blob container exists (non-blocking)."""
        try:
            await run_blocking(self.container_client.create_container)
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
    
    async def upload_file_from_path(
        self,
        file_path: str,
        filename: str,
        content_type: str,
        file_type: str = "file",
        **metadata
    ) -> Dict[str, Any]:
        """
        Upload file to Azure Blob Storage by streaming from a file path (no memory loading).
        
        Args:
            file_path: Path to the file to upload
            filename: Original filename
            content_type: MIME type
            file_type: Type of file (audio, image, etc.)
            **metadata: Additional metadata for path generation
        
        Returns:
            Dict with blob information
        """
        upload_start_time = time.time()
        file_size = os.path.getsize(file_path)
        max_retries = 3
        base_delay = 1.0
        
        logger.info(
            f"Starting blob upload from file: {file_path}, filename={filename}, size={file_size} bytes "
            f"({file_size / (1024*1024):.2f}MB), file_type={file_type}"
        )
        print(f"🔵 [BlobService] Starting upload: {filename}, size={file_size} bytes ({file_size / (1024*1024):.2f}MB)")
        
        for attempt in range(max_retries):
            try:
                print(f"🔵 [BlobService] Attempt {attempt + 1}/{max_retries}: Generating blob path...")
                # Generate unique filename if needed
                file_ext = Path(filename).suffix
                unique_filename = f"{uuid.uuid4()}{file_ext}"
                
                # Generate blob path
                blob_path = self._generate_blob_path(file_type, unique_filename, **metadata)
                print(f"🔵 [BlobService] Blob path: {blob_path}")
                
                # Upload file with timeout and retry
                print(f"🔵 [BlobService] Creating blob client...")
                blob_client = self.client.get_blob_client(
                    container=self.settings.container_name,
                    blob=blob_path
                )
                
                # Set content settings
                print(f"🔵 [BlobService] Setting content settings...")
                content_settings = ContentSettings(content_type=content_type)
                
                # Upload by streaming from file (no memory loading)
                # Azure SDK can accept a file-like object for streaming
                print(f"🔵 [BlobService] Starting upload_blob (streaming from file)...")
                def _upload_from_file():
                    print(f"🔵 [BlobService] Inside _upload_from_file, opening file: {file_path}")
                    upload_start = time.time()
                    with open(file_path, "rb") as f:
                        file_open_time = time.time() - upload_start
                        print(f"🔵 [BlobService] File opened in {file_open_time:.3f}s, calling upload_blob...")
                        # Note: Azure SDK's upload_blob doesn't accept timeout parameter directly
                        # Timeout is configured at the session level (session.timeout = (30, 60))
                        # The asyncio.wait_for wrapper provides additional 300s timeout protection
                        try:
                            blob_call_start = time.time()
                            # Azure SDK automatically handles chunking for large files
                            # The SDK will chunk files > 4MB automatically
                            result = blob_client.upload_blob(
                                f,
                                content_settings=content_settings,
                                metadata={
                                    "original_filename": filename,
                                    "file_type": file_type,
                                    "uploaded_at": datetime.utcnow().isoformat(),
                                    **{k: str(v) for k, v in metadata.items()}
                                },
                                overwrite=True
                            )
                            blob_call_duration = time.time() - blob_call_start
                            total_duration = time.time() - upload_start
                            print(f"🔵 [BlobService] upload_blob completed! blob_call={blob_call_duration:.3f}s, total={total_duration:.3f}s")
                            return result
                        except requests.exceptions.Timeout as e:
                            blob_call_duration = time.time() - blob_call_start
                            total_duration = time.time() - upload_start
                            print(f"🔵 [BlobService] upload_blob timed out after {blob_call_duration:.3f}s (total={total_duration:.3f}s): {type(e).__name__}: {e}")
                            raise TimeoutError(f"Blob upload timed out: {e}") from e
                        except Exception as e:
                            blob_call_duration = time.time() - blob_call_start if 'blob_call_start' in locals() else 0
                            total_duration = time.time() - upload_start
                            print(f"🔵 [BlobService] upload_blob raised exception after {blob_call_duration:.3f}s (total={total_duration:.3f}s): {type(e).__name__}: {e}")
                            raise
                
                print(f"🔵 [BlobService] Calling run_blocking with 300s timeout...")
                blocking_start = time.time()
                upload_result = await asyncio.wait_for(
                    run_blocking(_upload_from_file),
                    timeout=300.0  # 300 second timeout
                )
                blocking_duration = time.time() - blocking_start
                print(f"🔵 [BlobService] Upload completed successfully! run_blocking took {blocking_duration:.3f}s")
                
                upload_duration = time.time() - upload_start_time
                blob_url = blob_client.url
                
                logger.info(
                    f"✅ Uploaded file to blob storage: {blob_path}, "
                    f"size={file_size} bytes, duration={upload_duration:.2f}s, "
                    f"attempt={attempt + 1}"
                )
                
                return {
                    "file_id": str(uuid.uuid4()),
                    "blob_path": blob_path,
                    "container_name": self.settings.container_name,
                    "original_filename": filename,
                    "content_type": content_type,
                    "file_size": file_size,
                    "blob_url": blob_url,
                    "metadata": metadata,
                    "uploaded_at": datetime.utcnow()
                }
                
            except asyncio.TimeoutError:
                upload_duration = time.time() - upload_start_time
                logger.error(
                    f"❌ Blob upload timeout after {upload_duration:.2f}s "
                    f"(attempt {attempt + 1}/{max_retries}): {filename}"
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying blob upload in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise TimeoutError(f"Blob upload timed out after {max_retries} attempts")
                
            except AzureError as e:
                upload_duration = time.time() - upload_start_time
                error_str = str(e).lower()
                is_transient = (
                    "timeout" in error_str or
                    "connection" in error_str or
                    "503" in str(e) or
                    "500" in str(e) or
                    "502" in str(e) or
                    "504" in str(e)
                )
                
                if is_transient and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Transient error during blob upload (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"❌ Failed to upload file to blob storage (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    raise
                    
            except Exception as e:
                upload_duration = time.time() - upload_start_time
                logger.error(
                    f"❌ Unexpected error during blob upload (attempt {attempt + 1}/{max_retries}): {e}",
                    exc_info=True
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
    
    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: str,
        file_type: str = "file",
        **metadata
    ) -> Dict[str, Any]:
        """
        Upload file to Azure Blob Storage with streaming, retry logic, and timeout.
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            content_type: MIME type
            file_type: Type of file (audio, image, etc.)
            **metadata: Additional metadata for path generation
        
        Returns:
            Dict with blob information
        """
        upload_start_time = time.time()
        file_size = len(file_data)
        max_retries = 3
        base_delay = 1.0
        
        logger.info(
            f"Starting blob upload: filename={filename}, size={file_size} bytes "
            f"({file_size / (1024*1024):.2f}MB), file_type={file_type}"
        )
        
        for attempt in range(max_retries):
            try:
                # Note: Container existence is ensured at startup, not per request
                # Generate unique filename if needed
                file_ext = Path(filename).suffix
                unique_filename = f"{uuid.uuid4()}{file_ext}"
                
                # Generate blob path
                blob_path = self._generate_blob_path(file_type, unique_filename, **metadata)
                
                # Upload file with timeout and retry
                blob_client = self.client.get_blob_client(
                    container=self.settings.container_name,
                    blob=blob_path
                )
                
                # Set content settings
                content_settings = ContentSettings(content_type=content_type)
                
                # Upload with timeout (300 seconds for large files)
                # Run synchronous upload in executor to avoid blocking
                upload_result = await asyncio.wait_for(
                    run_blocking(
                        blob_client.upload_blob,
                        file_data,
                        content_settings=content_settings,
                        metadata={
                            "original_filename": filename,
                            "file_type": file_type,
                            "uploaded_at": datetime.utcnow().isoformat(),
                            **{k: str(v) for k, v in metadata.items()}
                        },
                        overwrite=True
                    ),
                    timeout=300.0  # 300 second timeout
                )
                
                upload_duration = time.time() - upload_start_time
                blob_url = blob_client.url
                
                logger.info(
                    f"✅ Uploaded file to blob storage: {blob_path}, "
                    f"size={file_size} bytes, duration={upload_duration:.2f}s, "
                    f"attempt={attempt + 1}"
                )
                
                return {
                    "file_id": str(uuid.uuid4()),
                    "blob_path": blob_path,
                    "container_name": self.settings.container_name,
                    "original_filename": filename,
                    "content_type": content_type,
                    "file_size": file_size,
                    "blob_url": blob_url,
                    "metadata": metadata,
                    "uploaded_at": datetime.utcnow()
                }
                
            except asyncio.TimeoutError:
                upload_duration = time.time() - upload_start_time
                logger.error(
                    f"❌ Blob upload timeout after {upload_duration:.2f}s "
                    f"(attempt {attempt + 1}/{max_retries}): {filename}"
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying blob upload in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise TimeoutError(f"Blob upload timed out after {max_retries} attempts")
                
            except AzureError as e:
                upload_duration = time.time() - upload_start_time
                error_str = str(e).lower()
                is_transient = (
                    "timeout" in error_str or
                    "connection" in error_str or
                    "503" in str(e) or
                    "500" in str(e) or
                    "502" in str(e) or
                    "504" in str(e)
                )
                
                if is_transient and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Transient error during blob upload (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"❌ Failed to upload file to blob storage (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    raise
                    
            except Exception as e:
                upload_duration = time.time() - upload_start_time
                logger.error(
                    f"❌ Unexpected error during blob upload (attempt {attempt + 1}/{max_retries}): {e}",
                    exc_info=True
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
    
    async def download_file(self, blob_path: str) -> bytes:
        """
        Download file from Azure Blob Storage with retry logic and timeout.
        
        Args:
            blob_path: Path to the blob
            
        Returns:
            File content as bytes
        """
        download_start_time = time.time()
        max_retries = 3
        base_delay = 1.0
        
        logger.info(f"Starting blob download: {blob_path}")
        
        for attempt in range(max_retries):
            try:
                blob_client = self.client.get_blob_client(
                    container=self.settings.container_name,
                    blob=blob_path
                )
                
                # Download with timeout (300 seconds for large files)
                download_stream = await asyncio.wait_for(
                    run_blocking(blob_client.download_blob),
                    timeout=300.0  # 300 second timeout
                )
                
                file_data = await asyncio.wait_for(
                    run_blocking(download_stream.readall),
                    timeout=300.0  # 300 second timeout for reading
                )
                
                download_duration = time.time() - download_start_time
                file_size = len(file_data)
                
                logger.info(
                    f"✅ Downloaded file from blob storage: {blob_path}, "
                    f"size={file_size} bytes ({file_size / (1024*1024):.2f}MB), "
                    f"duration={download_duration:.2f}s, attempt={attempt + 1}"
                )
                
                return file_data
                
            except asyncio.TimeoutError:
                download_duration = time.time() - download_start_time
                logger.error(
                    f"❌ Blob download timeout after {download_duration:.2f}s "
                    f"(attempt {attempt + 1}/{max_retries}): {blob_path}"
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying blob download in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise TimeoutError(f"Blob download timed out after {max_retries} attempts")
                
            except ResourceNotFoundError:
                logger.error(f"❌ File not found in blob storage: {blob_path}")
                raise FileNotFoundError(f"File not found: {blob_path}")
                
            except AzureError as e:
                download_duration = time.time() - download_start_time
                error_str = str(e).lower()
                is_transient = (
                    "timeout" in error_str or
                    "connection" in error_str or
                    "503" in str(e) or
                    "500" in str(e) or
                    "502" in str(e) or
                    "504" in str(e)
                )
                
                if is_transient and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Transient error during blob download (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"❌ Failed to download file from blob storage (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    raise
                    
            except Exception as e:
                download_duration = time.time() - download_start_time
                logger.error(
                    f"❌ Unexpected error during blob download (attempt {attempt + 1}/{max_retries}): {e}",
                    exc_info=True
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
    
    async def delete_file(self, blob_path: str) -> bool:
        """
        Delete file from Azure Blob Storage (non-blocking).
        
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
            
            await run_blocking(blob_client.delete_blob)
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
            
            shared_access_signature = None
            if self.settings.connection_string:
                # Parse connection string to extract account_key / SAS
                # Format: DefaultEndpointsProtocol=https;AccountName=xxx;AccountKey=xxx;EndpointSuffix=core.windows.net
                # Azure portal sometimes provides SharedAccessSignature instead of AccountKey
                conn_parts = self.settings.connection_string.split(';')
                for part in conn_parts:
                    if part.startswith('AccountKey='):
                        account_key = part.split('=', 1)[1]
                    elif part.startswith('AccountName=') and not account_name:
                        account_name = part.split('=', 1)[1]
                    elif part.startswith('SharedAccessSignature='):
                        shared_access_signature = part.split('=', 1)[1].lstrip('?')
            
            if not account_key:
                if shared_access_signature:
                    blob_url = f"https://{account_name}.blob.core.windows.net/{self.settings.container_name}/{blob_path}"
                    signed_url = f"{blob_url}?{shared_access_signature}"
                    logger.info("⚠️ Using shared access signature from connection string for blob: %s", blob_path)
                    return signed_url
                raise ValueError(
                    "Azure Blob Storage account_key is required for generating signed URLs. "
                    "Either set AZURE_BLOB_ACCOUNT_KEY or ensure AZURE_BLOB_CONNECTION_STRING contains "
                    "AccountKey or SharedAccessSignature."
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
        List files in blob storage (non-blocking).
        
        Args:
            prefix: Prefix to filter files
            max_results: Maximum number of results
            
        Returns:
            List of file information
        """
        try:
            files = []
            
            # list_blobs() returns an iterator, so we need to wrap it
            blobs = await run_blocking(
                lambda: list(self.container_client.list_blobs(name_starts_with=prefix))
            )
            
            for blob in blobs:
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
        Get file information from blob storage (non-blocking).
        
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
            
            properties = await run_blocking(blob_client.get_blob_properties)
            
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
