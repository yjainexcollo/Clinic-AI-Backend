"""
Background worker for processing transcription jobs from Azure Queue Storage.
This runs as a separate process/service.
"""
import asyncio
import logging
import sys
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
current_dir = Path(__file__).parent.parent.parent.parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

from clinicai.adapters.queue.azure_queue_service import get_azure_queue_service
from clinicai.adapters.db.mongo.repositories.patient_repository import MongoPatientRepository
from clinicai.adapters.db.mongo.repositories.visit_repository import MongoVisitRepository
from clinicai.adapters.db.mongo.repositories.audio_repository import AudioRepository
from clinicai.api.deps import get_transcription_service
from clinicai.application.use_cases.transcribe_audio import TranscribeAudioUseCase
from clinicai.application.dto.patient_dto import AudioTranscriptionRequest
from clinicai.core.config import get_settings
from clinicai.domain.value_objects.patient_id import PatientId
from clinicai.domain.value_objects.visit_id import VisitId

logger = logging.getLogger(__name__)


class TranscriptionWorker:
    """Worker that processes transcription jobs from Azure Queue."""
    
    def __init__(self):
        self.queue_service = get_azure_queue_service()
        self.patient_repo = MongoPatientRepository()
        self.visit_repo = MongoVisitRepository()
        self.audio_repo = AudioRepository()
        self.transcription_service = get_transcription_service()  # Uses same service selection as API
        self.settings = get_settings()
        
        # Worker configuration
        self.poll_interval = self.settings.azure_queue.poll_interval
        self.max_processing_time = 1800  # 30 minutes max processing time
        
    async def initialize(self):
        """Initialize worker (database connections, etc.)."""
        # Ensure queue exists
        # Note: Queue existence is ensured at startup, not per worker run
        logger.info("✅ Transcription worker initialized")
    
    async def get_audio_data(self, audio_file_id: str) -> Optional[bytes]:
        """Get audio file data from blob storage with logging."""
        download_start_time = time.time()
        try:
            audio_file = await self.audio_repo.get_audio_file_by_id(audio_file_id)
            if not audio_file:
                logger.error(f"Audio file {audio_file_id} not found")
                return None
            
            # Get blob reference
            from clinicai.adapters.db.mongo.repositories.blob_file_repository import BlobFileRepository
            blob_repo = BlobFileRepository()
            blob_ref = await blob_repo.get_blob_reference_by_id(audio_file.blob_reference_id)
            
            if not blob_ref:
                logger.error(f"Blob reference {audio_file.blob_reference_id} not found")
                return None
            
            # Download from blob storage (with timeout and retry handled in blob_service)
            from clinicai.adapters.storage.azure_blob_service import get_azure_blob_service
            blob_service = get_azure_blob_service()
            audio_data = await blob_service.download_file(blob_ref.blob_path)
            
            download_duration = time.time() - download_start_time
            logger.info(
                f"✅ Downloaded audio data: {audio_file_id}, "
                f"size={len(audio_data)} bytes ({len(audio_data) / (1024*1024):.2f}MB), "
                f"duration={download_duration:.2f}s"
            )
            
            return audio_data
        except Exception as e:
            download_duration = time.time() - download_start_time
            logger.error(
                f"Failed to get audio data: {e} (duration: {download_duration:.2f}s)",
                exc_info=True
            )
            return None
    
    async def process_job(self, job_data: dict, message_id: str, pop_receipt: str):
        """Process a single transcription job with improved logging and error handling."""
        job_start_time = time.time()
        patient_id = job_data["patient_id"]
        visit_id = job_data["visit_id"]
        audio_file_id = job_data["audio_file_id"]
        language = job_data.get("language", "en")
        retry_count = job_data.get("retry_count", 0)
        
        logger.info(
            f"Processing transcription job: patient={patient_id}, visit={visit_id}, "
            f"audio_file={audio_file_id}, language={language}, retry={retry_count}"
        )
        print("🔵 === Worker: Processing transcription job ===")
        print(f"🔵 patient_id={patient_id}, visit_id={visit_id}, audio_file_id={audio_file_id}, language={language}, retry={retry_count}")
        
        temp_file_path = None
        visibility_task = None
        
        try:
            # Get audio file data from blob storage
            download_start = time.time()
            audio_data = await self.get_audio_data(audio_file_id)
            download_duration = time.time() - download_start
            
            if not audio_data:
                raise ValueError(f"Failed to retrieve audio data for {audio_file_id}")
            
            logger.info(
                f"Downloaded audio data: size={len(audio_data)} bytes, "
                f"duration={download_duration:.2f}s"
            )
            print(f"🔵 Worker: downloaded audio data size = {len(audio_data)} bytes")
            
            # Get audio file metadata to determine extension
            audio_file = await self.audio_repo.get_audio_file_by_id(audio_file_id)
            if not audio_file:
                raise ValueError(f"Audio file {audio_file_id} not found")
            
            # Create temp file for transcription
            ext = audio_file.filename.split('.')[-1] if '.' in audio_file.filename else 'mp3'
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            logger.info(f"Created temp file: {temp_file_path}")
            print(f"🔵 Worker: temp file created at {temp_file_path}")
            
            # Extend message visibility periodically during processing
            async def extend_visibility():
                nonlocal pop_receipt
                while True:
                    await asyncio.sleep(300)  # Every 5 minutes
                    try:
                        new_pop_receipt = await self.queue_service.update_message_visibility(
                            message_id,
                            pop_receipt,
                            visibility_timeout=self.settings.azure_queue.visibility_timeout
                        )
                        logger.debug(f"Extended message visibility: {message_id}")
                        pop_receipt = new_pop_receipt
                    except Exception as e:
                        logger.warning(f"Failed to extend visibility: {e}")
            
            # Start visibility extension task
            visibility_task = asyncio.create_task(extend_visibility())
            
            try:
                # Create transcription request
                request = AudioTranscriptionRequest(
                    patient_id=patient_id,
                    visit_id=visit_id,
                    audio_file_path=temp_file_path,
                    language=language,
                )
                
                # Execute transcription use case
                use_case = TranscribeAudioUseCase(
                    self.patient_repo,
                    self.visit_repo,
                    self.transcription_service
                )
                
                # Process transcription with timeout (this can take 10+ minutes)
                transcription_start = time.time()
                logger.info(f"Starting transcription processing for {patient_id}/{visit_id}")
                print(f"🔵 Worker: starting transcription for {patient_id}/{visit_id}")
                
                # Add timeout for transcription (30 minutes max)
                try:
                    result = await asyncio.wait_for(
                        use_case.execute(request),
                        timeout=1800.0  # 30 minutes
                    )
                except asyncio.TimeoutError:
                    transcription_duration = time.time() - transcription_start
                    total_duration = time.time() - job_start_time
                    error_msg = (
                        f"Transcription processing timed out after {transcription_duration:.2f}s "
                        f"(total job duration: {total_duration:.2f}s)"
                    )
                    logger.error(f"❌ {error_msg}")
                    raise TimeoutError(error_msg)
                
                transcription_duration = time.time() - transcription_start
                total_duration = time.time() - job_start_time
                
                logger.info(
                    f"✅ Transcription completed: {result}, "
                    f"transcription_duration={transcription_duration:.2f}s, "
                    f"total_job_duration={total_duration:.2f}s"
                )
                print(f"✅ Worker: transcription completed. duration={result.audio_duration}, words={result.word_count}")
                
                # Check if transcription actually failed
                if result.transcription_status == "failed" or not result.transcript:
                    error_msg = result.message or "Transcription returned empty result"
                    logger.error(f"❌ Transcription failed: {error_msg}")
                    logger.error(f"❌ Transcription status: {result.transcription_status}, transcript length: {len(result.transcript) if result.transcript else 0}")
                    print(f"❌ Worker: transcription failed - {error_msg}")
                    raise ValueError(f"Transcription failed: {error_msg}")
                
                # Update audio file with duration if we have the result
                if result.audio_duration:
                    await self.audio_repo.update_audio_metadata(
                        audio_file_id,
                        duration_seconds=result.audio_duration
                    )
                    logger.info(f"Updated audio file duration: {result.audio_duration} seconds")
                    print(f"🔵 Worker: updated audio duration to {result.audio_duration} seconds")
                
                # Delete message from queue (job completed successfully)
                await self.queue_service.delete_message(message_id, pop_receipt)
                logger.info(f"✅ Job completed and removed from queue: {message_id}")
                print(f"✅ Worker: job removed from queue: {message_id}")
                
            finally:
                # Cancel visibility extension task
                if visibility_task:
                    visibility_task.cancel()
                    try:
                        await visibility_task
                    except asyncio.CancelledError:
                        pass
                    
        except Exception as e:
            total_duration = time.time() - job_start_time
            import traceback
            error_details = traceback.format_exc()
            logger.error(
                f"❌ Transcription job failed after {total_duration:.2f}s: {e}",
                exc_info=True
            )
            logger.error(f"Full error traceback:\n{error_details}")
            print(f"❌ Worker: transcription job failed: {str(e)}")
            print(f"❌ Full error traceback:\n{error_details}")
            
            # Cancel visibility task if running
            if visibility_task:
                visibility_task.cancel()
            
            # Check if this is a permanent error that shouldn't be retried
            from clinicai.domain.errors import VisitNotFoundError
            is_permanent_error = isinstance(e, VisitNotFoundError)
            
            if is_permanent_error:
                # Permanent error - delete message immediately (no retries)
                logger.warning(f"Permanent error detected ({type(e).__name__}), not retrying: {str(e)}")
                await self.queue_service.delete_message(message_id, pop_receipt)
                logger.error(f"❌ Job failed with permanent error, removed from queue")
            # Handle retries for transient errors
            elif retry_count < self.settings.azure_queue.max_retry_attempts:
                # Re-enqueue with incremented retry count
                job_data["retry_count"] = retry_count + 1
                await self.queue_service.enqueue_transcription_job(
                    job_data["patient_id"],
                    job_data["visit_id"],
                    job_data["audio_file_id"],
                    job_data["language"]
                )
                logger.info(f"Re-queued job for retry {retry_count + 1}/{self.settings.azure_queue.max_retry_attempts}")
            else:
                # Max retries exceeded - delete message and mark as failed
                await self.queue_service.delete_message(message_id, pop_receipt)
                
                # Mark visit transcription as failed
                try:
                    visit = await self.visit_repo.find_by_patient_and_visit_id(
                        patient_id, VisitId(visit_id)
                    )
                    if visit:
                        visit.fail_transcription(f"Job failed after {retry_count} retries: {str(e)}")
                        await self.visit_repo.save(visit)
                        logger.info(f"Marked transcription as failed for {patient_id}/{visit_id}")
                except Exception as db_error:
                    logger.error(f"Failed to mark transcription as failed: {db_error}")
                
                logger.error(f"❌ Job failed permanently after {retry_count} retries")
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temp file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}")
    
    async def run(self):
        """Main worker loop."""
        logger.info("🚀 Starting transcription worker...")
        await self.initialize()
        
        while True:
            try:
                # Poll queue for messages (non-blocking)
                job = await self.queue_service.dequeue_transcription_job()
                
                if job:
                    await self.process_job(
                        job["data"],
                        job["message_id"],
                        job["pop_receipt"]
                    )
                else:
                    # No messages, wait before next poll
                    await asyncio.sleep(self.poll_interval)
                    
            except KeyboardInterrupt:
                logger.info("🛑 Worker stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Worker error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)


async def main():
    """Entry point for worker process."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    worker = TranscriptionWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

