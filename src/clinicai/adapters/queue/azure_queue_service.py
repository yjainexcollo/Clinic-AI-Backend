"""
Azure Queue Storage service for background job processing.
"""
import json
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from azure.storage.queue import QueueServiceClient, QueueClient
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

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


class AzureQueueService:
    """Azure Queue Storage service for job queuing."""
    
    def __init__(self):
        self.settings = get_settings().azure_queue
        self._queue_client: Optional[QueueClient] = None
        
    @property
    def queue_client(self) -> QueueClient:
        """Get or create QueueClient."""
        if self._queue_client is None:
            # Use blob connection string if queue connection string not set
            connection_string = self.settings.connection_string
            if not connection_string:
                # Fallback to blob storage connection string
                blob_settings = get_settings().azure_blob
                connection_string = blob_settings.connection_string
                
            if not connection_string:
                raise ValueError("Azure Queue Storage connection string is required. Set AZURE_QUEUE_CONNECTION_STRING or AZURE_BLOB_CONNECTION_STRING")
            
            queue_service = QueueServiceClient.from_connection_string(
                connection_string
            )
            self._queue_client = queue_service.get_queue_client(
                self.settings.queue_name
            )
            logger.info(f"✅ Azure Queue Storage client initialized for queue: {self.settings.queue_name}")
        
        return self._queue_client
    
    async def ensure_queue_exists(self) -> bool:
        """Ensure the queue exists (non-blocking)."""
        try:
            await run_blocking(self.queue_client.create_queue)
            logger.info(f"✅ Created queue: {self.settings.queue_name}")
            return True
        except ResourceExistsError:
            logger.info(f"📁 Queue already exists: {self.settings.queue_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create queue: {e}")
            return False
    
    async def enqueue_transcription_job(
        self,
        patient_id: str,
        visit_id: str,
        audio_file_id: str,
        language: str = "en"
    ) -> str:
        """
        Enqueue a transcription job (non-blocking).
        
        Args:
            patient_id: Internal patient ID
            visit_id: Visit ID
            audio_file_id: Audio file ID from database
            language: Transcription language
            
        Returns:
            Message ID
        """
        message = {
            "job_type": "transcription",
            "patient_id": patient_id,
            "visit_id": visit_id,
            "audio_file_id": audio_file_id,
            "language": language,
            "created_at": datetime.utcnow().isoformat(),
            "retry_count": 0
        }
        
        try:
            # Enqueue message (non-blocking)
            response = await run_blocking(
                self.queue_client.send_message,
                json.dumps(message),
                visibility_timeout=self.settings.visibility_timeout
            )
            
            logger.info(
                f"✅ Transcription job enqueued: patient={patient_id}, "
                f"visit={visit_id}, message_id={response.id}"
            )
            
            return response.id
        except Exception as e:
            logger.error(f"❌ Failed to enqueue transcription job: {e}")
            raise
    
    async def dequeue_transcription_job(self) -> Optional[Dict[str, Any]]:
        """
        Dequeue a transcription job (non-blocking).
        
        Returns:
            Dict with 'data', 'message_id', 'pop_receipt' or None
        """
        try:
            messages = await run_blocking(
                lambda: list(self.queue_client.receive_messages(
                    messages_per_page=1,
                    visibility_timeout=self.settings.visibility_timeout
                ))
            )
            
            for message in messages:
                try:
                    message_data = json.loads(message.content)
                    return {
                        "data": message_data,
                        "message_id": message.id,
                        "pop_receipt": message.pop_receipt
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse queue message: {e}")
                    # Delete invalid message (non-blocking)
                    await run_blocking(
                        self.queue_client.delete_message,
                        message.id,
                        message.pop_receipt
                    )
                    continue
            
            return None
        except Exception as e:
            logger.error(f"❌ Failed to dequeue transcription job: {e}")
            return None
    
    async def delete_message(self, message_id: str, pop_receipt: str) -> None:
        """Delete a processed message from the queue (non-blocking)."""
        try:
            await run_blocking(
                self.queue_client.delete_message,
                message_id,
                pop_receipt
            )
            logger.debug(f"✅ Deleted message: {message_id}")
        except Exception as e:
            logger.error(f"❌ Failed to delete message: {e}")
    
    async def update_message_visibility(
        self,
        message_id: str,
        pop_receipt: str,
        visibility_timeout: int
    ) -> str:
        """
        Update message visibility timeout (extend processing time) (non-blocking).
        
        Returns:
            New pop_receipt
        """
        try:
            response = await run_blocking(
                self.queue_client.update_message,
                message_id,
                pop_receipt,
                visibility_timeout=visibility_timeout
            )
            return response.pop_receipt
        except Exception as e:
            logger.error(f"❌ Failed to update message visibility: {e}")
            raise
    
    async def get_queue_length(self) -> int:
        """Get approximate number of messages in queue (non-blocking)."""
        try:
            properties = await run_blocking(self.queue_client.get_queue_properties)
            return properties.approximate_message_count
        except Exception as e:
            logger.error(f"❌ Failed to get queue length: {e}")
            return 0


# Singleton instance
_queue_service: Optional[AzureQueueService] = None


def get_azure_queue_service() -> AzureQueueService:
    """Get singleton Azure Queue Service instance."""
    global _queue_service
    if _queue_service is None:
        _queue_service = AzureQueueService()
    return _queue_service

