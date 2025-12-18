"""
Background worker for processing transcription jobs from Azure Queue Storage.
This runs as a separate process/service.
"""
import asyncio
import json
import logging
import sys
import os
import tempfile
import time
from datetime import datetime, timedelta
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
        timings = {
            "dequeue_wait": 0.0,  # Not tracked here (would need dequeue timestamp)
            "blob_sas_generation": 0.0,
            "job_create": 0.0,
            "poll": 0.0,
            "results_fetch": 0.0,
            "postprocess": 0.0,
            "db_save": 0.0,
        }
        
        patient_id = job_data["patient_id"]
        visit_id = job_data["visit_id"]
        audio_file_id = job_data["audio_file_id"]
        language = job_data.get("language", "en")
        retry_count = job_data.get("retry_count", 0)
        request_id = job_data.get("request_id")
        
        # Generate unique worker ID for this worker instance
        import socket
        worker_id = f"{socket.gethostname()}:{os.getpid()}"
        
        # Get stale seconds from settings
        stale_seconds = self.settings.azure_queue.processing_stale_seconds
        
        # IDEMPOTENCY GUARD: Check visit status BEFORE attempting claim
        try:
            visit = await self.visit_repo.find_by_patient_and_visit_id(
                patient_id, VisitId(visit_id)
            )
            if not visit:
                logger.warning(f"Visit {visit_id} not found, deleting message {message_id}")
                try:
                    await self.queue_service.delete_message(message_id, pop_receipt)
                except Exception:
                    pass  # Best effort cleanup
                return

            # Check if already completed - skip and delete message
            if visit.transcription_session and visit.transcription_session.transcription_status == "completed":
                logger.info(f"Transcription already completed for visit {visit_id}, skipping duplicate job {message_id}")
                try:
                    await self.queue_service.delete_message(message_id, pop_receipt)
                except Exception:
                    pass  # Best effort cleanup
                return
            
            # Check if already failed - skip and delete message (don't retry failed jobs)
            if visit.transcription_session and visit.transcription_session.transcription_status == "failed":
                logger.info(f"Transcription already marked as failed for visit {visit_id}, skipping job {message_id}")
                try:
                    await self.queue_service.delete_message(message_id, pop_receipt)
                except Exception:
                    pass  # Best effort cleanup
                return
            
            # Attempt atomic claim of the job
            claimed = await self.visit_repo.try_mark_processing(
                patient_id=patient_id,
                visit_id=VisitId(visit_id),
                worker_id=worker_id,
                stale_seconds=stale_seconds
            )
            
            if not claimed:
                # Job was not claimed (likely being processed by another worker or already completed/failed)
                # Use short backoff (30s) so we can re-check quickly for stale takeover
                queue_name = getattr(self.queue_service, 'queue_name', None)
                if queue_name is None:
                    queue_name = self.queue_service.settings.queue_name
                claim_backoff_seconds = min(30, self.settings.azure_queue.visibility_timeout)
                logger.info(
                    f"Job not claimed → visibility backoff={claim_backoff_seconds}s: visit={visit_id}, "
                    f"message_id={message_id}, retry={retry_count}, queue_name={queue_name} "
                    f"(likely being processed elsewhere)"
                )
                # Extend visibility with short backoff - let the message become visible again soon for stale takeover
                try:
                    await self.queue_service.update_message_visibility(
                        message_id,
                        pop_receipt,
                        visibility_timeout=claim_backoff_seconds
                    )
                except Exception as visibility_error:
                    logger.warning(f"Failed to extend visibility for unclaimed message: {visibility_error}")
                return
            
            # Job was successfully claimed - log and proceed with processing
            logger.info(
                f"✅ Job claimed: visit={visit_id}, message_id={message_id}, "
                f"claimed_by={worker_id}, retry={retry_count}"
            )
            
            # Reload visit to get the updated state from the claim
            visit = await self.visit_repo.find_by_patient_and_visit_id(
                patient_id, VisitId(visit_id)
            )
            if not visit:
                logger.error(f"Visit {visit_id} not found after claim, this should not happen")
                return
            
        except Exception as idempotency_check_error:
            logger.error(f"Error during idempotency check/claim: {idempotency_check_error}", exc_info=True)
            # On error, extend visibility with short backoff and skip to avoid processing corrupted state
            try:
                claim_backoff_seconds = min(30, self.settings.azure_queue.visibility_timeout)
                await self.queue_service.update_message_visibility(
                    message_id,
                    pop_receipt,
                    visibility_timeout=claim_backoff_seconds
                )
            except Exception:
                pass
            return
        
        # Get dequeued_at from visit (set by atomic claim)
        dequeued_at = visit.transcription_session.dequeued_at if visit.transcription_session else None
        logger.info(
            f"Processing transcription job: visit={visit_id}, "
            f"audio_file={audio_file_id}, language={language}, retry={retry_count}, "
            f"message_id={message_id}, request_id={request_id or 'none'}, "
            f"dequeued_at={dequeued_at.isoformat() if dequeued_at else 'N/A'}, "
            f"worker_id={worker_id}"
        )
        
        temp_file_path = None
        visibility_task = None
        latest_pop_receipt = pop_receipt  # Track latest pop_receipt for deletion
        
        try:
            # Get audio file metadata and blob reference (for SAS URL)
            audio_file = await self.audio_repo.get_audio_file_by_id(audio_file_id)
            if not audio_file:
                raise ValueError(f"Audio file {audio_file_id} not found")
            
            from clinicai.adapters.db.mongo.repositories.blob_file_repository import BlobFileRepository
            blob_repo = BlobFileRepository()
            blob_ref = await blob_repo.get_blob_reference_by_id(audio_file.blob_reference_id)
            if not blob_ref:
                raise ValueError(f"Blob reference {audio_file.blob_reference_id} not found")
            
            # Generate SAS URL for existing audio blob (avoids re-upload for Azure Speech)
            sas_start = time.time()
            from clinicai.adapters.storage.azure_blob_service import get_azure_blob_service
            blob_service = get_azure_blob_service()
            sas_url = blob_service.generate_signed_url(
                blob_path=blob_ref.blob_path,
                expires_in_hours=24,
            )
            timings["blob_sas_generation"] = time.time() - sas_start
            logger.debug(f"Generated SAS URL for transcription blob in {timings['blob_sas_generation']:.2f}s")
            
            # OPTIMIZATION: Skip blob download - use SAS URL directly
            # The transcription service can use SAS URL without local file
            # Only create temp file path as placeholder (transcription_service will handle SAS URL)
            ext = audio_file.filename.split('.')[-1] if '.' in audio_file.filename else 'mp3'
            temp_file_path = None  # Not needed when using SAS URL directly
            
            # Extend message visibility periodically during processing
            async def extend_visibility():
                nonlocal latest_pop_receipt
                while True:
                    await asyncio.sleep(300)  # Every 5 minutes
                    try:
                        new_pop_receipt = await self.queue_service.update_message_visibility(
                            message_id,
                            latest_pop_receipt,
                            visibility_timeout=self.settings.azure_queue.visibility_timeout
                        )
                        logger.debug(f"Extended message visibility: {message_id}")
                        latest_pop_receipt = new_pop_receipt
                    except Exception as e:
                        logger.warning(f"Failed to extend visibility: {e}")
            
            # Start visibility extension task
            visibility_task = asyncio.create_task(extend_visibility())
            heartbeat_task = None  # Initialize for cleanup
            
            try:
                # Create transcription request (use SAS URL, no local file needed)
                request = AudioTranscriptionRequest(
                    patient_id=patient_id,
                    visit_id=visit_id,
                    audio_file_path=None,  # Not needed when sas_url provided
                    language=language,
                    sas_url=sas_url,
                )
                
                # Execute transcription use case
                use_case = TranscribeAudioUseCase(
                    self.patient_repo,
                    self.visit_repo,
                    self.transcription_service
                )
                
                # Process transcription with timeout (this can take 10+ minutes)
                job_create_start = time.time()
                logger.debug(f"Starting transcription processing for visit {visit_id}")
                
                # Add heartbeat logging task to show progress (INFO level for visibility)
                async def heartbeat_logger():
                    """Log progress every 60 seconds at INFO level to show worker is still processing."""
                    heartbeat_interval = 60  # 60 seconds
                    while True:
                        await asyncio.sleep(heartbeat_interval)
                        elapsed = time.time() - job_create_start
                        
                        # Read transcription_id from database
                        transcription_id_from_db = "N/A"
                        try:
                            current_visit = await self.visit_repo.find_by_patient_and_visit_id(
                                patient_id, VisitId(visit_id)
                            )
                            if current_visit and current_visit.transcription_session:
                                transcription_id_from_db = current_visit.transcription_session.transcription_id or "N/A"
                        except Exception as db_error:
                            logger.debug(f"Failed to read transcription_id from DB for heartbeat: {db_error}")
                        
                        logger.info(
                            f"💓 Transcription heartbeat: visit={visit_id}, "
                            f"transcription_id={transcription_id_from_db}, "
                            f"elapsed={elapsed:.1f}s, still processing..."
                        )
                
                heartbeat_task = asyncio.create_task(heartbeat_logger())
                
                # Add timeout for transcription (30 minutes max)
                try:
                    result = await asyncio.wait_for(
                        use_case.execute(request),
                        timeout=1800.0  # 30 minutes
                    )
                except asyncio.TimeoutError:
                    transcription_duration = time.time() - job_create_start
                    total_duration = time.time() - job_start_time
                    error_msg = f"Transcription processing timed out after {transcription_duration:.2f}s"
                    logger.error(f"❌ {error_msg}")
                    raise TimeoutError(error_msg)
                finally:
                    # Cancel heartbeat task when done (success or timeout)
                    if heartbeat_task:
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass
                
                # Track timings: job_create includes all transcription work (speech + LLM + PII removal)
                timings["job_create"] = time.time() - job_create_start
                timings["postprocess"] = timings["job_create"]  # For backward compatibility
                
                # STRICT VALIDATION: Never log "completed" for empty/failed transcripts
                if result.transcription_status != "completed":
                    error_msg = result.message or "Transcription status not completed"
                    logger.error(f"❌ Transcription failed: status={result.transcription_status}, message={error_msg}")
                    raise ValueError(f"Transcription failed: {error_msg}")
                
                if not result.transcript or result.transcript.strip() == "":
                    error_msg = "Transcription returned empty transcript"
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)
                
                if result.word_count is None or result.word_count == 0:
                    error_msg = "Transcription returned zero word count"
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)
                
                # Update audio file with duration if we have the result
                if result.audio_duration:
                    await self.audio_repo.update_audio_metadata(
                        audio_file_id,
                        duration_seconds=result.audio_duration
                    )
                    logger.debug(f"Updated audio file duration: {result.audio_duration} seconds")
                
                # P0-1: DB save is done inside use_case.execute() - verify it succeeded
                # The use case saves the visit with transcript, so we can now safely delete the queue message
                # Retry DB save if needed (though it should already be saved by use case)
                db_save_start = time.time()
                db_save_attempts = 0
                max_db_save_attempts = 3
                db_save_success = False
                
                while db_save_attempts < max_db_save_attempts and not db_save_success:
                    try:
                        # Verify visit was saved by reloading and checking transcript exists
                        saved_visit = await self.visit_repo.find_by_patient_and_visit_id(
                            patient_id, VisitId(visit_id)
                        )
                        if saved_visit and saved_visit.transcription_session and saved_visit.transcription_session.transcript:
                            db_save_success = True
                            logger.info(f"✅ DB save verified: transcript exists for visit {visit_id}")
                        else:
                            # Transcript not found - try to save again
                            logger.warning(f"⚠️  Transcript not found in DB after use case, attempting save (attempt {db_save_attempts + 1}/{max_db_save_attempts})")
                            # Reload visit and save again
                            visit = await self.visit_repo.find_by_patient_and_visit_id(
                                patient_id, VisitId(visit_id)
                            )
                            if visit and visit.transcription_session:
                                await self.visit_repo.save(visit)
                                db_save_attempts += 1
                                if db_save_attempts < max_db_save_attempts:
                                    await asyncio.sleep(2 ** db_save_attempts)  # Exponential backoff
                            else:
                                raise ValueError("Visit or transcription session not found")
                    except Exception as db_save_error:
                        db_save_attempts += 1
                        logger.error(f"❌ DB save attempt {db_save_attempts}/{max_db_save_attempts} failed: {db_save_error}")
                        if db_save_attempts < max_db_save_attempts:
                            await asyncio.sleep(2 ** db_save_attempts)  # Exponential backoff
                        else:
                            # Max attempts exceeded - do NOT delete queue message
                            logger.error(
                                f"❌ ACK_SKIPPED_DB_SAVE_FAILED: visit={visit_id}, message_id={message_id}, "
                                f"db_save_attempts={db_save_attempts}. Queue message will remain for retry."
                            )
                            raise ValueError(f"DB save failed after {max_db_save_attempts} attempts: {db_save_error}")
                
                timings["db_save"] = time.time() - db_save_start
                
                # P0-1: Only delete queue message AFTER successful DB save
                if db_save_success:
                    try:
                        await self.queue_service.delete_message(message_id, latest_pop_receipt)
                        logger.info(f"✅ ACK_AFTER_DB_SAVE_OK: visit={visit_id}, message_id={message_id}")
                    except Exception as delete_error:
                        logger.error(f"❌ Failed to delete queue message after DB save: {delete_error}", exc_info=True)
                        # Message will become visible again after visibility timeout - acceptable
                
                # Structured success log
                total_duration = time.time() - job_start_time
                log_data = {
                    "event": "transcription_job_completed",
                    "message_id": message_id,
                    "visit_id": visit_id,
                    "audio_file_id": audio_file_id,
                    "retry_count": retry_count,
                    "request_id": request_id,
                    "status": "success",
                    "timings": timings,
                    "total_time_seconds": total_duration,
                    "word_count": result.word_count,
                    "audio_duration": result.audio_duration,
                }
                logger.info(json.dumps(log_data))
                
            finally:
                # Cancel visibility extension task and heartbeat task
                if visibility_task:
                    visibility_task.cancel()
                    try:
                        await visibility_task
                    except asyncio.CancelledError:
                        pass
                if heartbeat_task:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                    
        except Exception as e:
            total_duration = time.time() - job_start_time
            
            # Extract clean error information (no PHI, no __name__ bug)
            error_type = type(e).__name__  # Use type(e).__name__ not e.__name__
            error_message = str(e)
            error_code = getattr(e, 'error_code', 'UNKNOWN_ERROR') if hasattr(e, 'error_code') else 'UNKNOWN_ERROR'
            
            # Avoid double-prefixing error messages
            if error_message.startswith("Transcription failed:"):
                clean_error_message = error_message
            else:
                clean_error_message = f"{error_type}: {error_message}"
            
            logger.error(
                f"❌ Transcription job failed: visit={visit_id}, message_id={message_id}, "
                f"retry={retry_count}, duration={total_duration:.2f}s, "
                f"error_type={error_type}, error_code={error_code}",
                exc_info=True
            )
            
            # Cancel visibility task and heartbeat task if running
            if visibility_task:
                visibility_task.cancel()
            if heartbeat_task:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Check if this is a permanent error that shouldn't be retried
            from clinicai.domain.errors import VisitNotFoundError
            is_permanent_error = isinstance(e, VisitNotFoundError)
            
            if is_permanent_error:
                # Permanent error - mark as failed, but only delete if DB save succeeds
                logger.warning(f"Permanent error detected ({error_type}), not retrying: {clean_error_message}")
                db_save_success = False
                try:
                    visit = await self.visit_repo.find_by_patient_and_visit_id(
                        patient_id, VisitId(visit_id)
                    )
                    if visit:
                        error_info = f"{error_code}: {clean_error_message}"
                        visit.fail_transcription(error_info)
                        await self.visit_repo.save(visit)
                        db_save_success = True
                except Exception as db_error:
                    logger.error(f"Failed to mark transcription as failed: {db_error}", exc_info=True)
                
                # P0-1: Only delete message if DB save succeeded
                if db_save_success:
                    try:
                        await self.queue_service.delete_message(message_id, latest_pop_receipt)
                        logger.info(f"✅ ACK_AFTER_DB_SAVE_OK (permanent error): visit={visit_id}, message_id={message_id}")
                    except Exception as delete_error:
                        logger.error(f"Failed to delete message after permanent error: {delete_error}", exc_info=True)
                else:
                    logger.error(
                        f"❌ ACK_SKIPPED_DB_SAVE_FAILED (permanent error): visit={visit_id}, message_id={message_id}. "
                        f"Queue message will remain for manual review."
                    )
                
                # Structured failure log
                log_data = {
                    "event": "transcription_job_failed",
                    "message_id": message_id,
                    "visit_id": visit_id,
                    "audio_file_id": audio_file_id,
                    "retry_count": retry_count,
                    "request_id": request_id,
                    "status": "failed",
                    "error_type": error_type,
                    "error_code": error_code,
                    "error_message": clean_error_message,
                    "timings": timings,
                    "total_time_seconds": total_duration,
                    "is_permanent": True,
                }
                logger.info(json.dumps(log_data))
                return
            
            # Handle retries for transient errors
            if retry_count < self.settings.azure_queue.max_retry_attempts:
                # Calculate exponential backoff delay
                delay_seconds = min(60 * (2 ** retry_count), 300)  # Max 5 minutes
                new_retry_count = retry_count + 1
                
                # Re-enqueue with incremented retry count and delay
                try:
                    await self.queue_service.enqueue_transcription_job(
                        job_data["patient_id"],
                        job_data["visit_id"],
                        job_data["audio_file_id"],
                        job_data["language"],
                        retry_count=new_retry_count,
                        delay_seconds=delay_seconds,
                        request_id=request_id
                    )
                    logger.info(f"Re-queued job for retry {new_retry_count}/{self.settings.azure_queue.max_retry_attempts} with {delay_seconds}s delay")
                except Exception as requeue_error:
                    logger.error(f"Failed to re-enqueue job: {requeue_error}", exc_info=True)
                    # If re-enqueue fails, we cannot proceed - original message remains for retry
                    return
                
                # P0-1 CRITICAL: Do NOT delete original message in retry path
                # The original message must remain until DB save is verified by the new retry attempt.
                # If we delete the original now and the new message fails DB save, we lose the job.
                # The original becomes visible again after visibility_timeout (no TTL set on messages),
                # providing a safety net if the new retry fails DB save.
                # If the new retry succeeds, the original will be handled by idempotency checks (already completed).
                logger.info(
                    f"✅ ACK_SKIPPED_RETRY_PATH_ORIGINAL_RETAINED: visit={visit_id}, message_id={message_id}, "
                    f"retry_count={retry_count}→{new_retry_count}. Original message retained for safety. "
                    f"New retry message enqueued. Original becomes visible again after visibility timeout if new retry fails."
                )
                
                # Structured retry log
                log_data = {
                    "event": "transcription_job_retry",
                    "message_id": message_id,
                    "visit_id": visit_id,
                    "audio_file_id": audio_file_id,
                    "retry_count": retry_count,
                    "new_retry_count": new_retry_count,
                    "request_id": request_id,
                    "status": "retrying",
                    "error_type": error_type,
                    "error_code": error_code,
                    "delay_seconds": delay_seconds,
                    "timings": timings,
                    "total_time_seconds": total_duration,
                }
                logger.info(json.dumps(log_data))
            else:
                # Max retries exceeded - mark as failed, but only delete message if DB save succeeds
                # Mark visit transcription as failed with clean error message
                db_save_success = False
                try:
                    visit = await self.visit_repo.find_by_patient_and_visit_id(
                        patient_id, VisitId(visit_id)
                    )
                    if visit:
                        # Store structured error info (no PHI)
                        error_info = f"{error_code}: {clean_error_message}"
                        visit.fail_transcription(error_info)
                        await self.visit_repo.save(visit)
                        db_save_success = True
                        logger.info(f"Marked transcription as failed for visit {visit_id}")
                except Exception as db_error:
                    logger.error(f"Failed to mark transcription as failed: {db_error}", exc_info=True)
                
                # P0-1: Only delete message if DB save succeeded
                if db_save_success:
                    try:
                        await self.queue_service.delete_message(message_id, latest_pop_receipt)
                        logger.info(f"✅ ACK_AFTER_DB_SAVE_OK (failed job): visit={visit_id}, message_id={message_id}")
                    except Exception as delete_error:
                        logger.error(f"Failed to delete message after max retries: {delete_error}", exc_info=True)
                else:
                    logger.error(
                        f"❌ ACK_SKIPPED_DB_SAVE_FAILED (max retries): visit={visit_id}, message_id={message_id}. "
                        f"Queue message will remain for manual review."
                    )
                
                # Structured permanent failure log
                log_data = {
                    "event": "transcription_job_failed",
                    "message_id": message_id,
                    "visit_id": visit_id,
                    "audio_file_id": audio_file_id,
                    "retry_count": retry_count,
                    "request_id": request_id,
                    "status": "failed",
                    "error_type": error_type,
                    "error_code": error_code,
                    "error_message": clean_error_message,
                    "timings": timings,
                    "total_time_seconds": total_duration,
                    "is_permanent": False,
                    "max_retries_exceeded": True,
                }
                logger.info(json.dumps(log_data))
        finally:
            # Clean up temp file (if created)
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temp file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}")
    
    async def _handle_poison_job(self, job: dict) -> None:
        """
        Handle a poison message (max dequeue count exceeded) in a DB-safe way.

        - Marks the corresponding visit's transcription as failed.
        - Only moves the message to the poison queue after DB save succeeds.
        - If DB save fails, the message remains in the main queue for manual / later retry.
        """
        message_id: str = job.get("message_id")
        pop_receipt: str = job.get("pop_receipt")
        job_data: dict = job.get("data") or {}
        retry_count: int = int(job.get("retry_count", 0))
        poison_reason: str = job.get("poison_reason", "POISON_MESSAGE")
        raw_content: str = job.get("raw_content", "")

        patient_id = job_data.get("patient_id")
        visit_id = job_data.get("visit_id")

        if not patient_id or not visit_id:
            logger.error(
                "⚠️  POISON_MESSAGE_WITHOUT_IDS: message_id=%s, reason=%s, retry_count=%s",
                message_id,
                poison_reason,
                retry_count,
            )
            # We cannot update DB, but we can still move to poison queue to avoid infinite retries
            try:
                await self.queue_service.move_to_poison_message(
                    message_id=message_id,
                    pop_receipt=pop_receipt,
                    content=raw_content,
                    reason=f"{poison_reason} (missing patient_id/visit_id)",
                )
                logger.warning(
                    "POISON_MARKED_NO_DB_UPDATE_MOVED_TO_POISON: message_id=%s, reason=%s",
                    message_id,
                    poison_reason,
                )
            except Exception as e:  # noqa: PERF203
                logger.error(
                    "POISON_MOVE_TO_POISON_FAILED_NO_IDS: message_id=%s, error=%s",
                    message_id,
                    e,
                    exc_info=True,
                )
            return

        error_info = (
            f"{poison_reason}; retry_count={retry_count}; "
            f"max_dequeue={self.settings.azure_queue.max_dequeue_count}; message_id={message_id}"
        )

        db_save_success = False
        try:
            visit = await self.visit_repo.find_by_patient_and_visit_id(
                patient_id, VisitId(visit_id)
            )
            if not visit:
                logger.warning(
                    "POISON_VISIT_NOT_FOUND: patient_id=%s visit_id=%s message_id=%s",
                    patient_id,
                    visit_id,
                    message_id,
                )
            else:
                visit.fail_transcription(error_message=error_info)
                await self.visit_repo.save(visit)
                db_save_success = True
                logger.info(
                    "POISON_MARKED_FAILED_DB_OK: patient_id=%s visit_id=%s message_id=%s retry_count=%s",
                    patient_id,
                    visit_id,
                    message_id,
                    retry_count,
                )
        except Exception as e:  # noqa: PERF203
            logger.error(
                "POISON_DB_SAVE_FAILED: patient_id=%s visit_id=%s message_id=%s error=%s",
                patient_id,
                visit_id,
                message_id,
                e,
                exc_info=True,
            )

        if db_save_success:
            try:
                await self.queue_service.move_to_poison_message(
                    message_id=message_id,
                    pop_receipt=pop_receipt,
                    content=raw_content,
                    reason=error_info,
                )
                logger.info(
                    "POISON_MARKED_FAILED_AND_MOVED_TO_POISON: patient_id=%s visit_id=%s message_id=%s",
                    patient_id,
                    visit_id,
                    message_id,
                )
            except Exception as e:  # noqa: PERF203
                logger.error(
                    "POISON_MOVE_TO_POISON_FAILED_AFTER_DB_OK: patient_id=%s visit_id=%s message_id=%s error=%s",
                    patient_id,
                    visit_id,
                    message_id,
                    e,
                    exc_info=True,
                )
        else:
            # Do not delete/move the message; leave it for later inspection or retry.
            logger.error(
                "POISON_DB_SAVE_FAILED_ACK_SKIPPED: patient_id=%s visit_id=%s message_id=%s retry_count=%s",
                patient_id,
                visit_id,
                message_id,
                retry_count,
            )

    async def run(self):
        """Main worker loop with bounded concurrency and batch dequeue."""
        # Worker startup guard: check if already running
        worker_process_id = os.getpid()
        logger.info(f"🚀 Starting transcription worker (PID: {worker_process_id})...")
        
        # Check if ENABLE_TRANSCRIPTION_WORKER is set (only warn if both are running)
        if os.getenv("ENABLE_TRANSCRIPTION_WORKER", "false").lower() == "true":
            logger.warning(
                "⚠️  ENABLE_TRANSCRIPTION_WORKER=true detected. "
                "If this worker is running as a separate process, you may have duplicate workers. "
                "Recommended: run worker either in-process (ENABLE_TRANSCRIPTION_WORKER=true) "
                "OR as separate process (worker_startup.py), not both."
            )
        
        await self.initialize()
        
        # Read concurrency from environment (default to 5 for dev, 2 for production)
        default_concurrency = 5  # Increased default for better throughput
        try:
            max_concurrent_jobs = int(os.getenv("TRANSCRIPTION_WORKER_CONCURRENCY", str(default_concurrency)))
        except ValueError:
            max_concurrent_jobs = default_concurrency
        if max_concurrent_jobs < 1:
            max_concurrent_jobs = 1
        
        logger.info(f"Transcription worker concurrency set to {max_concurrent_jobs}")
        semaphore = asyncio.Semaphore(max_concurrent_jobs)
        
        # P0-2: Track active tasks for true concurrency
        active_tasks: set[asyncio.Task] = set()
        
        # Track queue polling for observability
        poll_count = 0
        last_queue_status_log = time.time()
        queue_status_log_interval = 300  # Log queue status every 5 minutes

        async def handle_job(job: dict):
            """Handle a single job with semaphore and task tracking."""
            task = asyncio.current_task()
            if task:
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

            async with semaphore:
                # Poison messages are handled separately to ensure DB is updated before moving to poison queue
                if job.get("poison"):
                    await self._handle_poison_job(job)
                else:
                    await self.process_job(
                        job["data"],
                        job["message_id"],
                        job["pop_receipt"],
                    )
        
        # Graceful shutdown handler
        shutdown_event = asyncio.Event()
        
        def signal_handler():
            logger.info("🛑 Shutdown signal received, stopping worker gracefully...")
            shutdown_event.set()
        
        import signal
        if hasattr(signal, 'SIGTERM'):
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
            loop.add_signal_handler(signal.SIGINT, signal_handler)
        
        while not shutdown_event.is_set():
            try:
                # P0-2: Calculate free slots and batch size
                active_jobs = len(active_tasks)
                free_slots = max_concurrent_jobs - active_jobs
                batch_size = min(free_slots, max_concurrent_jobs) if free_slots > 0 else 0
                
                # Poll queue for messages (batch dequeue if slots available)
                if batch_size > 0:
                    jobs = await self.queue_service.dequeue_transcription_job(max_messages=batch_size)
                    poll_count += 1
                    
                    if jobs:
                        # Handle single job (backward compatible) or batch
                        job_list = jobs if isinstance(jobs, list) else [jobs]
                        
                        for job in job_list:
                            if job:
                                # Process job in background with concurrency limit
                                task = asyncio.create_task(handle_job(job))
                                active_tasks.add(task)
                                task.add_done_callback(active_tasks.discard)
                    else:
                        # No messages, wait before next poll
                        await asyncio.sleep(self.poll_interval)
                else:
                    # No free slots, wait a bit before checking again
                    await asyncio.sleep(1)
                
                # Periodic queue status logging (every 5 minutes)
                current_time = time.time()
                if current_time - last_queue_status_log >= queue_status_log_interval:
                    # Defensive: use property if available, otherwise fallback to settings
                    queue_name = getattr(self.queue_service, 'queue_name', None)
                    if queue_name is None:
                        queue_name = self.queue_service.settings.queue_name
                    logger.info(
                        f"📊 Worker status: poll_count={poll_count}, "
                        f"active_jobs={active_jobs}, free_slots={free_slots}, "
                        f"batch_size={batch_size}, queue_name={queue_name}"
                    )
                    last_queue_status_log = current_time
                    
            except KeyboardInterrupt:
                logger.info("🛑 Worker stopped by user")
                shutdown_event.set()
                break
            except Exception as e:
                logger.error(f"❌ Worker error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
        
        # Graceful shutdown: wait for active tasks with timeout
        if active_tasks:
            logger.info(f"⏳ Waiting for {len(active_tasks)} active job(s) to complete (max 60s)...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*active_tasks, return_exceptions=True),
                    timeout=60.0
                )
                logger.info("✅ All active jobs completed")
            except asyncio.TimeoutError:
                logger.warning(f"⚠️  Timeout waiting for {len(active_tasks)} job(s) to complete. Cancelling...")
                for task in active_tasks:
                    if not task.done():
                        task.cancel()
                # Wait a bit more for cancellations
                await asyncio.sleep(2)
                unfinished = [t for t in active_tasks if not t.done()]
                if unfinished:
                    logger.error(f"❌ {len(unfinished)} job(s) did not complete gracefully")


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

