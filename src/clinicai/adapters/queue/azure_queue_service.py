"""
Azure Queue Storage service for background job processing.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.queue import QueueClient, QueueServiceClient

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
        self._poison_queue_client: Optional[QueueClient] = None
        self._last_empty_poll_log: float = 0.0
        self._empty_poll_log_interval: float = 30.0  # Log empty polls every 30 seconds

    def _extract_storage_account_name(self, connection_string: str) -> str:
        """Extract storage account name from connection string for logging."""
        try:
            # Connection string format: DefaultEndpointsProtocol=https;AccountName=NAME;AccountKey=KEY;...
            for part in connection_string.split(";"):
                if part.startswith("AccountName="):
                    return part.split("=", 1)[1]
        except Exception:
            pass
        return "unknown"

    @property
    def queue_client(self) -> QueueClient:
        """Get or create QueueClient using Settings (not direct env vars)."""
        if self._queue_client is None:
            # Use settings connection string (already has fallbacks applied in config)
            connection_string = self.settings.connection_string
            if not connection_string:
                # Final fallback to blob storage connection string from settings
                blob_settings = get_settings().azure_blob
                connection_string = blob_settings.connection_string

            if not connection_string:
                raise ValueError(
                    "Azure Queue Storage connection string is required. "
                    "Set one of: AZURE_QUEUE_CONNECTION_STRING, AZURE_STORAGE_CONNECTION_STRING, or AZURE_BLOB_CONNECTION_STRING"
                )

            # Extract storage account name for logging (masked)
            storage_account = self._extract_storage_account_name(connection_string)

            queue_service = QueueServiceClient.from_connection_string(connection_string)
            self._queue_client = queue_service.get_queue_client(self.settings.queue_name)

            # Log startup info with masked connection details
            logger.info(
                f"✅ Azure Queue Storage client initialized: "
                f"queue_name={self.settings.queue_name}, "
                f"storage_account={storage_account}, "
                f"visibility_timeout={self.settings.visibility_timeout}s, "
                f"poll_interval={self.settings.poll_interval}s"
            )

        return self._queue_client

    @property
    def queue_name(self) -> str:
        """Backward-compatible property to access queue name from settings."""
        return self.settings.queue_name

    @property
    def poison_queue_name(self) -> str:
        """Backward-compatible property to access poison queue name."""
        return f"{self.settings.queue_name}-poison"

    @property
    def poison_queue_client(self) -> QueueClient:
        """Get or create poison queue client."""
        if self._poison_queue_client is None:
            poison_queue_name = f"{self.settings.queue_name}-poison"
            connection_string = self.settings.connection_string
            if not connection_string:
                blob_settings = get_settings().azure_blob
                connection_string = blob_settings.connection_string

            if not connection_string:
                raise ValueError("Azure Queue Storage connection string is required")

            queue_service = QueueServiceClient.from_connection_string(connection_string)
            self._poison_queue_client = queue_service.get_queue_client(poison_queue_name)
            logger.info(f"✅ Poison queue client initialized: {poison_queue_name}")

        return self._poison_queue_client

    async def _move_to_poison_queue(self, message_id: str, pop_receipt: str, content: str, reason: str) -> bool:
        """Move a message to the poison queue and delete from main queue."""
        try:
            # Ensure poison queue exists
            try:
                await run_blocking(self.poison_queue_client.create_queue)
                logger.info(f"✅ Created poison queue: {self.poison_queue_client.queue_name}")
            except ResourceExistsError:
                pass  # Queue already exists, which is fine

            # Add metadata to poison message
            poison_message = {
                "original_message_id": message_id,
                "original_content": content,
                "reason": reason,
                "moved_at": datetime.utcnow().isoformat(),
            }

            # Send to poison queue
            await run_blocking(self.poison_queue_client.send_message, json.dumps(poison_message))

            # Delete from main queue
            await run_blocking(self.queue_client.delete_message, message_id, pop_receipt)

            logger.warning(f"⚠️  Moved message to poison queue: message_id={message_id}, reason={reason}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to move message to poison queue: {e}", exc_info=True)
            return False

    async def move_to_poison_message(
        self,
        message_id: str,
        pop_receipt: str,
        content: str,
        reason: str,
    ) -> bool:
        """
        Public wrapper to move a message to the poison queue and delete it from the main queue.

        This is used by workers that have already updated the database to mark the job as failed.
        """
        return await self._move_to_poison_queue(message_id, pop_receipt, content, reason)

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
        language: str = "en",
        retry_count: int = 0,
        delay_seconds: int = 0,
        request_id: Optional[str] = None,
        doctor_id: Optional[str] = None,
    ) -> str:
        """
        Enqueue a transcription job (non-blocking).

        Args:
            patient_id: Internal patient ID
            visit_id: Visit ID
            audio_file_id: Audio file ID from database
            language: Transcription language
            retry_count: Number of retry attempts (for tracking)
            delay_seconds: Delay before message becomes visible (for retry backoff)
            request_id: Optional request ID for log correlation
            doctor_id: Doctor ID for multi-doctor isolation (required for multi-doctor support)

        Returns:
            Message ID
        """
        message = {
            "job_type": "transcription",
            "patient_id": patient_id,
            "visit_id": visit_id,
            "audio_file_id": audio_file_id,
            "doctor_id": doctor_id,
            "language": language,
            "created_at": datetime.utcnow().isoformat(),
            "retry_count": retry_count,
        }
        if request_id:
            message["request_id"] = request_id
        if doctor_id:
            message["doctor_id"] = doctor_id

        try:
            # For new jobs: visibility_timeout=0 (immediate visibility).
            # For retry jobs: visibility_timeout=delay_seconds (initial invisibility for backoff).
            visibility_timeout = delay_seconds if delay_seconds > 0 else 0

            # Enqueue message (non-blocking)
            response = await run_blocking(
                self.queue_client.send_message,
                json.dumps(message),
                visibility_timeout=visibility_timeout,
            )

            logger.info(
                "✅ Transcription job enqueued: "
                f"queue={self.settings.queue_name}, visit={visit_id}, "
                f"audio_file={audio_file_id}, message_id={response.id}, "
                f"retry_count={retry_count}, delay={delay_seconds}s"
            )

            return response.id
        except Exception as e:
            logger.error(
                "❌ Failed to enqueue transcription job: "
                f"queue={self.settings.queue_name}, visit={visit_id}, audio_file={audio_file_id}, "
                f"retry_count={retry_count}, delay={delay_seconds}s, error={e}"
            )
            raise

    async def dequeue_transcription_job(self, max_messages: int = 1) -> Optional[Dict[str, Any]]:
        """
        Dequeue transcription job(s) (non-blocking).

        Args:
            max_messages: Maximum number of messages to dequeue (default: 1 for backward compatibility)

        Returns:
            Dict with 'data', 'message_id', 'pop_receipt' or None (single message)
            OR List[Dict] if max_messages > 1 (returns all available messages up to max_messages)
        """
        try:
            messages = await run_blocking(
                lambda: list(
                    self.queue_client.receive_messages(
                        messages_per_page=max_messages,
                        visibility_timeout=self.settings.visibility_timeout,
                    )
                )
            )

            if not messages:
                # Log empty queue periodically (every ~30s) to avoid log spam
                current_time = time.time()
                if current_time - self._last_empty_poll_log >= self._empty_poll_log_interval:
                    logger.debug(f"Queue '{self.settings.queue_name}' is empty (no messages available)")
                    self._last_empty_poll_log = current_time
                return None

            # Process all messages and collect valid ones
            valid_messages = []
            for message in messages:
                message_id = message.id
                pop_receipt = message.pop_receipt

                try:
                    message_data = json.loads(message.content)
                    visit_id = message_data.get("visit_id", "unknown")
                    audio_file_id = message_data.get("audio_file_id", "unknown")
                    retry_count = message_data.get("retry_count", 0)

                    # Check if this is a poison message (too many retries)
                    if retry_count >= self.settings.max_dequeue_count:
                        reason = (
                            f"POISON_MESSAGE: retry_count={retry_count} >= "
                            f"max_dequeue_count={self.settings.max_dequeue_count}"
                        )
                        logger.warning(
                            f"⚠️  Poison message detected: message_id={message_id}, "
                            f"visit={visit_id}, retry_count={retry_count}"
                        )
                        job_dict = {
                            "data": message_data,
                            "message_id": message_id,
                            "pop_receipt": pop_receipt,
                            "poison": True,
                            "poison_reason": reason,
                            "retry_count": retry_count,
                            "raw_content": message.content,
                        }
                        # Return immediately for single-message dequeue
                        if max_messages == 1:
                            return job_dict
                        valid_messages.push(job_dict)
                        continue

                    # Log message details with insertion time if available
                    insertion_time = getattr(message, "insertion_time", None)
                    insertion_str = f", insertion_time={insertion_time.isoformat()}" if insertion_time else ""

                    logger.info(
                        f"📥 Dequeued transcription job: visit={visit_id}, "
                        f"audio_file={audio_file_id}, message_id={message_id}, "
                        f"retry={retry_count}{insertion_str}"
                    )

                    job_dict = {
                        "data": message_data,
                        "message_id": message_id,
                        "pop_receipt": pop_receipt,
                        "poison": False,
                        "retry_count": retry_count,
                        "raw_content": message.content,
                    }

                    # If max_messages == 1, return first valid message (backward compatible)
                    if max_messages == 1:
                        return job_dict

                    # Otherwise, collect for batch return
                    valid_messages.append(job_dict)

                except json.JSONDecodeError as e:
                    # Invalid JSON - log warning with truncated content and handle as poison in dev
                    content_preview = message.content[:200] if len(message.content) > 200 else message.content
                    logger.warning(
                        f"⚠️  Failed to parse queue message JSON: message_id={message_id}, "
                        f"error={e}, content_preview={content_preview}"
                    )

                    # In development, move to poison queue; in production, delete
                    import os

                    is_dev = (
                        os.getenv("APP_ENV", "production") == "development"
                        or os.getenv("DEBUG", "false").lower() == "true"
                    )

                    if is_dev:
                        reason = f"INVALID_JSON: {str(e)[:100]}"
                        await self._move_to_poison_queue(message_id, pop_receipt, message.content, reason)
                    else:
                        # Delete invalid message
                        await run_blocking(self.queue_client.delete_message, message_id, pop_receipt)
                    continue

            # Return batch if max_messages > 1, otherwise None (no valid messages found)
            if max_messages > 1:
                return valid_messages if valid_messages else None
            return None
        except Exception as e:
            logger.error(f"❌ Failed to dequeue transcription job: {e}")
            return None

    async def delete_message(self, message_id: str, pop_receipt: str) -> bool:
        """
        Delete a processed message from the queue (non-blocking).

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            await run_blocking(self.queue_client.delete_message, message_id, pop_receipt)
            logger.debug(f"✅ Deleted message: {message_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete message {message_id}: {e}", exc_info=True)
            # Raise exception so caller can handle failure
            raise

    async def update_message_visibility(self, message_id: str, pop_receipt: str, visibility_timeout: int) -> str:
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
                visibility_timeout=visibility_timeout,
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
