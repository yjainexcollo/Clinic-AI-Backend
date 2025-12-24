"""Note-related API endpoints for Step-03 functionality."""
import logging
import time

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import Response as FastAPIResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import tempfile
import os
import asyncio
from datetime import datetime

from ...application.dto.patient_dto import (
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    SoapGenerationRequest,
    SoapGenerationResponse,
    SoapNoteDTO,
    TranscriptionSessionDTO
)
from ..schemas.medical import SOAPNoteRequest
from pydantic import BaseModel
from ...application.use_cases.transcribe_audio import TranscribeAudioUseCase
from ...application.use_cases.generate_soap_note import GenerateSoapNoteUseCase
from ...domain.errors import (
    PatientNotFoundError,
    VisitNotFoundError,
)
from ..deps import PatientRepositoryDep, VisitRepositoryDep, TranscriptionServiceDep, AudioRepositoryDep, SoapServiceDep
from ...core.utils.crypto import decode_patient_id
from ..schemas import ErrorResponse
from ...core.config import get_settings
from ..schemas.common import ApiResponse, ErrorResponse
from ..utils.responses import ok, fail

router = APIRouter(prefix="/notes")
logger = logging.getLogger("clinicai")

try:
    from ...adapters.queue.azure_queue_service import get_azure_queue_service
    QUEUE_SERVICE_AVAILABLE = True
except ImportError:
    QUEUE_SERVICE_AVAILABLE = False
    logger.warning("Azure Queue Storage not available. Install azure-storage-queue package.")


# Vitals data models
class VitalsData(BaseModel):
    # Make types align with JSON schema
    systolic: int
    diastolic: int
    bpArm: str = ""  # optional: "Left" | "Right"
    bpPosition: str = ""  # optional: "Sitting" | "Standing" | "Lying"
    heartRate: int
    rhythm: str = ""  # optional
    respiratoryRate: int
    temperature: float
    tempUnit: str = "C"  # "C" | "F"
    tempMethod: str = ""  # optional
    oxygenSaturation: int
    height: Optional[float] = None  # optional cm
    heightUnit: str = "cm"
    weight: float
    weightUnit: str = "kg"
    painScore: Optional[int] = None
    notes: str = ""


class VitalsRequest(BaseModel):
    patient_id: str
    visit_id: str
    vitals: VitalsData


class VitalsResponse(BaseModel):
    success: bool
    message: str
    vitals_id: str


@router.options("/transcribe", include_in_schema=False)
async def transcribe_audio_options():
    """Handle preflight OPTIONS request for transcribe endpoint."""
    return {"message": "OK"}


@router.get("/test-cors", include_in_schema=False)
async def test_cors():
    """Test endpoint to verify CORS is working."""
    return {"message": "CORS is working", "timestamp": "2024-01-01T00:00:00Z"}


@router.post(
    "/transcribe",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Vitals and Transcript Generation"],
    responses={
        202: {"description": "Queued for transcription"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "Invalid audio file or visit status"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def transcribe_audio(
    request: Request,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
    transcription_service: TranscriptionServiceDep,
    audio_repo: AudioRepositoryDep,
    background: BackgroundTasks,
    patient_id: str = Form(...),
    visit_id: str = Form(...),
    language: str = Form("en"),
    audio_file: UploadFile = File(...),
):
    """Queue audio transcription and return immediately (202)."""
    # Set IDs in request state for HIPAA audit middleware
    request.state.audit_patient_id = patient_id
    request.state.audit_visit_id = visit_id
    # Log request (no PHI)
    logger.info(f"Transcribe audio request received: visit_id={visit_id}, language={language}, filename={audio_file.filename if audio_file else 'None'}")

    if not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "NO_AUDIO_FILE", "message": "No audio file provided", "details": {}},
        )

    # Validate file type (allow common audio types and mpeg containers)
    print("🔵 Step 2: Validating file type...")
    content_type = audio_file.content_type or ""
    is_audio_like = content_type.startswith("audio/")
    is_mpeg_container = content_type in ("video/mpeg", "video/mpg", "application/mpeg")
    is_generic_stream = content_type in ("application/octet-stream",)
    print(f"🔵 Step 2 complete: content_type={content_type}, is_audio_like={is_audio_like}, is_mpeg_container={is_mpeg_container}, is_generic_stream={is_generic_stream}")
    if not (is_audio_like or is_mpeg_container or is_generic_stream):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_FILE_TYPE",
                "message": "File must be an audio file",
                "details": {"content_type": audio_file.content_type},
            },
        )

    # Decode opaque patient id from client
    print("🔵 Step 3: Decoding patient ID...")
    # Check if this looks like an internal patient ID (format: name_mobile)
    if '_' in patient_id:
        parts = patient_id.split('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            # This looks like an internal patient ID, skip decryption
            internal_patient_id = patient_id
            logger.debug(f"Using internal patient ID for transcription: {internal_patient_id}")
        else:
            # Try to decrypt as opaque token
            try:
                internal_patient_id = decode_patient_id(patient_id)
            except Exception as e:
                logger.warning(f"Failed to decode patient_id '{patient_id}': {e}")
                internal_patient_id = patient_id
    else:
        # Try to decrypt as opaque token
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception as e:
            logger.warning(f"Failed to decode patient_id '{patient_id}': {e}")
            internal_patient_id = patient_id

    # Stream to temp file WITHOUT accumulating in memory
    print("🔵 Step 4: Starting file streaming...")
    upload_start_time = time.time()
    temp_file_path = None
    file_size = 0
    audio_file_record = None  # Initialize outside try block for background task access
    try:
        # Extract doctor_id
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "MISSING_DOCTOR_ID",
                    "message": "X-Doctor-ID header is required",
                    "details": {},
                },
            )
        logger.info("Starting streaming upload to temp file (no memory accumulation)...")
        print("🔵 Step 4a: Creating temp file...")
        ext = (audio_file.filename or 'audio').split('.')[-1]
        # Normalize common MPEG container cases
        if ext.lower() in {"mpeg", "mpg"}:
            ext = "mpeg"
        suffix = f".{ext}"
        
        # Stream directly to temp file - NO memory accumulation
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            print(f"🔵 Step 4b: Temp file created at {temp_file_path}, starting to read chunks...")
            chunk_size = 1024 * 1024  # 1MB chunks for optimal performance
            chunk_count = 0
            while True:
                print(f"🔵 Step 4c: Reading chunk {chunk_count}...")
                chunk = await audio_file.read(chunk_size)
                if not chunk:
                    print(f"🔵 Step 4d: Finished reading all chunks (total: {chunk_count} chunks, {file_size} bytes)")
                    break
                temp_file.write(chunk)
                file_size += len(chunk)  # Track size only, don't accumulate data
                chunk_count += 1
                if chunk_count % 10 == 0:  # Log every 10 chunks to avoid spam
                    print(f"🔵 Step 4c: Read {chunk_count} chunks, {file_size} bytes so far...")
        
        upload_duration = time.time() - upload_start_time
        logger.info(
            f"Temp file created: {temp_file_path}, size={file_size} bytes "
            f"({file_size / (1024*1024):.2f}MB), upload_duration={upload_duration:.2f}s"
        )
        print(f"🔵 Step 4e: Temp file created successfully in {upload_duration:.2f}s")

        # Audio normalization: convert video/mpeg or non-audio .mpeg to WAV 16kHz mono PCM
        normalized_file_path = temp_file_path
        conversion_time = 0.0
        normalized_audio = False
        original_content_type = content_type
        normalized_format = None
        final_content_type = content_type
        audio_duration_seconds = None
        
        normalize_enabled = os.getenv("NORMALIZE_AUDIO", "true").lower() == "true"
        needs_normalization = (
            normalize_enabled and
            (content_type.startswith("video/") or (not is_audio_like and content_type in ("video/mpeg", "video/mpg", "application/mpeg")))
        )
        
        if needs_normalization:
            try:
                from ...core.audio_utils import normalize_audio_to_wav, get_audio_duration
                
                logger.info(f"Audio normalization required: content_type={content_type}, is_audio_like={is_audio_like}")
                
                # Create normalized output file
                normalized_fd, normalized_file_path = tempfile.mkstemp(suffix=".wav", prefix="normalized_")
                os.close(normalized_fd)
                
                # Convert to WAV 16kHz mono PCM
                normalized_file_path, conversion_time = normalize_audio_to_wav(
                    input_path=temp_file_path,
                    output_path=normalized_file_path,
                    timeout_seconds=300,
                )
                
                # Get audio duration using ffprobe
                try:
                    audio_duration_seconds = get_audio_duration(normalized_file_path, timeout_seconds=30)
                except Exception as duration_error:
                    logger.warning(f"Could not get audio duration: {duration_error}, will get from Azure Speech result")
                    # Continue without duration - Azure Speech will provide it
                
                normalized_audio = True
                normalized_format = "wav_16khz_mono_pcm"
                final_content_type = "audio/wav"
                
                logger.info(
                    f"Audio normalized: {temp_file_path} -> {normalized_file_path}, "
                    f"conversion_time={conversion_time:.2f}s, duration={audio_duration_seconds}s"
                )
                
            except FileNotFoundError as e:
                # ffmpeg not found - return clear error
                logger.error(f"ffmpeg not found for audio normalization: {e}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "AUDIO_NORMALIZATION_FAILED",
                        "message": "Audio normalization requires ffmpeg. Please install ffmpeg or set FFMPEG_PATH env var.",
                        "details": {"original_error": str(e)},
                    },
                )
            except (TimeoutError, RuntimeError) as e:
                # Conversion failed - return clear error
                logger.error(f"Audio normalization failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "AUDIO_NORMALIZATION_FAILED",
                        "message": f"Failed to normalize audio file: {str(e)}",
                        "details": {"original_error": str(e)},
                    },
                )
        else:
            # Try to get duration for non-normalized files too (if ffprobe available)
            try:
                from ...core.audio_utils import get_audio_duration
                audio_duration_seconds = get_audio_duration(temp_file_path, timeout_seconds=30)
            except Exception:
                # Duration extraction is optional - Azure Speech will provide it
                pass

        # Mark visit as queued if possible (best-effort via use case during processing)
        try:
            # Check if patient and visit exist and are in correct status
            from clinicai.domain.value_objects.patient_id import PatientId
            patient = await patient_repo.find_by_id(PatientId(internal_patient_id), doctor_id)
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "PATIENT_NOT_FOUND", "message": "Patient not found", "details": {}},
                )
            
            from ...domain.value_objects.visit_id import VisitId
            print(f"🔵 Step 5b: Looking up visit {visit_id}...")
            visit_id_obj = VisitId(visit_id)
            visit = await visit_repo.find_by_patient_and_visit_id(
                internal_patient_id, visit_id_obj, doctor_id
            )
            if not visit:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "VISIT_NOT_FOUND", "message": f"Visit {visit_id} not found", "details": {}},
                )
            
            logger.info(f"Patient and visit {visit_id} found. Visit status: {visit.status}")
            
            # Check if visit is ready for transcription before starting
            if not visit.can_proceed_to_transcription():
                if visit.is_scheduled_workflow():
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "error": "VISIT_NOT_READY_FOR_TRANSCRIPTION",
                            "message": f"Scheduled visit not ready for transcription. Current status: {visit.status}. Please fill vitals first before uploading transcript.",
                            "details": {"current_status": visit.status, "required_status": "vitals or vitals_completed"}
                        }
                    )
                elif visit.is_walk_in_workflow():
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "error": "VISIT_NOT_READY_FOR_TRANSCRIPTION",
                            "message": f"Walk-in visit not ready for transcription. Current status: {visit.status}. Please complete vitals first.",
                            "details": {"current_status": visit.status, "required_status": "vitals_completed"}
                        }
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "error": "VISIT_NOT_READY_FOR_TRANSCRIPTION",
                            "message": f"Visit not ready for transcription. Current status: {visit.status}.",
                            "details": {"current_status": visit.status}
                        }
                    )
            
            # Save audio file to database first (before queuing)
            # Upload directly from temp file (streaming, no memory loading)
            print("🔵 Step 6: Starting blob upload...")
            blob_upload_start = time.time()
            logger.info(f"Uploading audio file from temp file: {temp_file_path}, size={file_size} bytes")
            print(f"🔵 Step 6a: Calling create_audio_file_from_path with file_path={temp_file_path}, size={file_size} bytes")
            
            repo_call_start = time.time()
            # Use normalized file if normalization was performed
            upload_file_path = normalized_file_path if normalized_audio else temp_file_path
            upload_filename = (audio_file.filename or "unknown_audio")
            if normalized_audio:
                # Change extension to .wav if normalized
                if "." in upload_filename:
                    upload_filename = upload_filename.rsplit(".", 1)[0] + ".wav"
                else:
                    upload_filename = upload_filename + ".wav"
            
            audio_file_record = await audio_repo.create_audio_file_from_path(
                file_path=upload_file_path,
                filename=upload_filename,
                content_type=final_content_type,
                patient_id=internal_patient_id,
                visit_id=visit_id,
                audio_type="visit",
            )
            repo_call_duration = time.time() - repo_call_start
            blob_upload_duration = time.time() - blob_upload_start
            logger.info(
                f"Audio file saved to database: {audio_file_record.audio_id}, "
                f"repo_call_duration={repo_call_duration:.2f}s, blob_upload_duration={blob_upload_duration:.2f}s"
            )
            print(f"✅ Audio file saved to database: {audio_file_record.audio_id} (repo_call={repo_call_duration:.3f}s, total={blob_upload_duration:.3f}s)")

            # ---------------------------
            # Phase 1: mark enqueue pending
            # ---------------------------
            requested_at = datetime.utcnow()
            visit.mark_transcription_enqueue_pending(audio_file_path=None, requested_at=requested_at)

            # Store normalization metadata and duration
            if visit.transcription_session:
                visit.transcription_session.normalized_audio = normalized_audio
                visit.transcription_session.original_content_type = original_content_type
                visit.transcription_session.normalized_format = normalized_format
                visit.transcription_session.file_content_type = final_content_type
                if audio_duration_seconds is not None:
                    visit.transcription_session.audio_duration_seconds = audio_duration_seconds

            await visit_repo.save(visit)
            logger.info(
                f"[Transcribe] Enqueue pending for visit {visit_id}, "
                f"requested_at={requested_at.isoformat()}, "
                f"normalized_audio={normalized_audio}, file_content_type={final_content_type}, "
                f"audio_duration_seconds={audio_duration_seconds}"
            )

            # Validate audio reference BEFORE enqueue
            if not audio_file_record or not getattr(audio_file_record, "audio_id", None):
                error_msg = "missing_audio_reference"
                visit.mark_transcription_enqueue_failed(error_msg)
                await visit_repo.save(visit)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "QUEUE_ERROR",
                        "message": "Failed to queue transcription: missing audio reference",
                        "details": {"reason": error_msg},
                    },
                )

            # ---------------------------
            # Phase 2: enqueue with retry/backoff
            # ---------------------------
            if not QUEUE_SERVICE_AVAILABLE:
                visit.mark_transcription_enqueue_failed("QUEUE_SERVICE_UNAVAILABLE")
                await visit_repo.save(visit)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "QUEUE_SERVICE_UNAVAILABLE",
                        "message": "Azure Queue Storage service is not available. Please install azure-storage-queue package.",
                        "details": {},
                    },
                )

            queue_service = get_azure_queue_service()
            # Note: Queue existence is ensured at startup, not per request

            request_id = getattr(request.state, "request_id", None)

            backoff_schedule = [0.5, 2.0, 5.0]
            last_error: Optional[Exception] = None
            message_id: Optional[str] = None

            for attempt, delay in enumerate(backoff_schedule, start=1):
                try:
                    message_id = await queue_service.enqueue_transcription_job(
                        patient_id=internal_patient_id,
                        visit_id=visit_id,
                        audio_file_id=audio_file_record.audio_id,
                        language=language,
                        retry_count=attempt - 1,
                        delay_seconds=0,
                        doctor_id=doctor_id,
                        request_id=request_id,
                    )
                    break
                except Exception as e:  # noqa: PERF203
                    last_error = e
                    logger.error(
                        f"[Transcribe] enqueue_transcription_job attempt {attempt} failed "
                        f"for visit={visit_id}, audio_file_id={audio_file_record.audio_id}: {e}",
                        exc_info=True,
                    )
                    if attempt < len(backoff_schedule):
                        await asyncio.sleep(delay)

            if not message_id:
                # All attempts failed
                err_str = str(last_error) if last_error else "unknown_error"
                visit.mark_transcription_enqueue_failed(err_str)
                await visit_repo.save(visit)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "QUEUE_ERROR",
                        "message": f"Failed to queue transcription: {err_str}",
                        "details": {},
                    },
                )

            # Mark enqueued only AFTER queue send success
            enqueued_at = datetime.utcnow()
            visit.mark_transcription_enqueued(message_id=message_id, enqueued_at=enqueued_at)
            await visit_repo.save(visit)

            total_duration = time.time() - upload_start_time
            logger.info(
                f"[Transcribe] Transcription job enqueued: message_id={message_id}, "
                f"visit={visit_id}, total_request_duration={total_duration:.2f}s, file_size={file_size} bytes"
            )
            print(f"✅ Transcription job enqueued: message_id={message_id}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to queue transcription: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "QUEUE_ERROR", "message": f"Failed to queue transcription: {str(e)}", "details": {}},
            )
        finally:
            # Clean up temp files (original and normalized if created)
            cleanup_paths = []
            if temp_file_path and os.path.exists(temp_file_path):
                cleanup_paths.append(temp_file_path)
            if normalized_audio and normalized_file_path and os.path.exists(normalized_file_path) and normalized_file_path != temp_file_path:
                cleanup_paths.append(normalized_file_path)
            
            for cleanup_path in cleanup_paths:
                try:
                    os.unlink(cleanup_path)
                    logger.debug(f"Cleaned up temp file: {cleanup_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file {cleanup_path}: {cleanup_error}")

        # Return 202 Accepted for async operation
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "queued",
                "patient_id": internal_patient_id,
                "visit_id": visit_id,
                "message_id": message_id,
                "message": "Transcription queued successfully. Poll /notes/transcribe/status/{patient_id}/{visit_id} for status."
            }
        )

    except HTTPException:
        # Reraise HTTPException directly
        raise
    except Exception as e:
        logger.error("Unhandled error queuing transcription", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e), "details": {}},
        )


@router.get(
    "/transcribe/status/{patient_id}/{visit_id}",
    status_code=status.HTTP_200_OK,
    tags=["Vitals and Transcript Generation"],
    responses={
        200: {"description": "Transcription status retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_transcription_status(
    request: Request,
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """Get transcription status for polling."""
    try:
        # Get doctor_id from request state
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        # Decode patient ID
        if '_' in patient_id:
            parts = patient_id.split('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                internal_patient_id = patient_id
            else:
                try:
                    internal_patient_id = decode_patient_id(patient_id)
                except Exception:
                    internal_patient_id = patient_id
        else:
            try:
                internal_patient_id = decode_patient_id(patient_id)
            except Exception:
                internal_patient_id = patient_id
        
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj, doctor_id
        )
        
        if not visit:
            return fail(request, error="VISIT_NOT_FOUND", message="Visit not found")
        
        if not visit.transcription_session:
            return ok(request, data={
                "status": "pending",
                "message": "Transcription not started"
            })
        
        transcription_session = visit.transcription_session
        transcription_status = transcription_session.transcription_status
        
        # Check for stale processing state (> 30 minutes with no recent activity)
        from datetime import timedelta
        now = datetime.utcnow()
        is_stale = False
        if transcription_status == "processing" and transcription_session.started_at:
            age_minutes = (now - transcription_session.started_at).total_seconds() / 60
            last_poll_age_minutes = (now - transcription_session.last_poll_at).total_seconds() / 60 if transcription_session.last_poll_at else age_minutes
            # Consider stale if processing > 30 min and last poll > 20 min ago (or never polled)
            if age_minutes > 30 and (last_poll_age_minutes > 20 or not transcription_session.last_poll_at):
                is_stale = True
                transcription_status = "stale_processing"
        
        status_info = {
            "status": transcription_status,  # pending, processing, completed, failed, stale_processing
            "transcription_id": transcription_session.transcription_id,
            "started_at": transcription_session.started_at.isoformat() if transcription_session.started_at else None,
            "last_poll_status": transcription_session.last_poll_status,
            "last_poll_at": transcription_session.last_poll_at.isoformat() if transcription_session.last_poll_at else None,
            "error_message": transcription_session.error_message,
            "enqueued_at": transcription_session.enqueued_at.isoformat() if transcription_session.enqueued_at else None,
            "dequeued_at": transcription_session.dequeued_at.isoformat() if transcription_session.dequeued_at else None,
        }
        
        if transcription_status == "completed":
            status_info["transcript_available"] = True
            status_info["word_count"] = transcription_session.word_count
            status_info["duration"] = transcription_session.audio_duration_seconds
            status_info["audio_duration_seconds"] = transcription_session.audio_duration_seconds
            status_info["file_content_type"] = transcription_session.file_content_type
            status_info["normalized_audio"] = transcription_session.normalized_audio
            status_info["completed_at"] = transcription_session.completed_at.isoformat() if transcription_session.completed_at else None
            status_info["message"] = "Transcription completed successfully"
        elif transcription_status in ["processing", "stale_processing"]:
            status_info["audio_duration_seconds"] = transcription_session.audio_duration_seconds
            status_info["file_content_type"] = transcription_session.file_content_type
            status_info["normalized_audio"] = transcription_session.normalized_audio
        elif transcription_status == "stale_processing":
            status_info["message"] = f"Transcription appears stuck (processing for {age_minutes:.1f} minutes). May need manual intervention."
            status_info["next_action"] = "retry_or_reset"
        elif transcription_status == "processing":
            age_seconds = (now - transcription_session.started_at).total_seconds() if transcription_session.started_at else 0
            status_info["message"] = f"Transcription in progress (running for {age_seconds:.0f} seconds)"
            status_info["progress_seconds"] = age_seconds
        elif transcription_status == "failed":
            status_info["error"] = transcription_session.error_message
            status_info["message"] = f"Transcription failed: {transcription_session.error_message}"
        else:
            status_info["message"] = f"Transcription status: {transcription_status}"
        
        return ok(request, data=status_info)
    except Exception as e:
        logger.error(f"Error getting transcription status: {e}", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=str(e))


@router.post(
    "/soap/generate",
    response_model=SoapGenerationResponse,
    status_code=status.HTTP_200_OK,
    tags=["SOAP Note Generation"],
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "No transcript available or invalid visit status"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def generate_soap_note(
    http_request: Request,
    request: SOAPNoteRequest,  # Use Pydantic schema instead of dataclass
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
    soap_service: SoapServiceDep,
):
    """
    Generate SOAP note from transcript.
    
    This endpoint:
    1. Validates patient and visit exist
    2. Checks transcript is available
    3. Generates SOAP note using AI
    4. Updates visit status to prescription_analysis
    5. Returns structured SOAP note
    """
    try:
        # Set IDs in request state for HIPAA audit middleware
        http_request.state.audit_patient_id = request.patient_id
        http_request.state.audit_visit_id = request.visit_id

        # Extract doctor_id
        doctor_id = getattr(http_request.state, "doctor_id", None)
        if not doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "MISSING_DOCTOR_ID",
                    "message": "X-Doctor-ID header is required",
                    "details": {},
                },
            )
        
        # Decode opaque patient_id if provided by client
        from ...core.utils.crypto import decode_patient_id
        try:
            # Some environments may URL-encode the token inadvertently
            import urllib.parse as _up
            decoded_input = _up.unquote(request.patient_id)
        except Exception:
            decoded_input = request.patient_id
        try:
            internal_patient_id = decode_patient_id(decoded_input)
        except Exception:
            internal_patient_id = decoded_input
        
        # Create DTO for use case (dataclass)
        decoded_request = SoapGenerationRequest(
            patient_id=internal_patient_id,
            visit_id=request.visit_id,
            transcript=request.transcript,
            template=request.template.dict() if getattr(request, "template", None) else None,
        )

        # Execute use case
        use_case = GenerateSoapNoteUseCase(patient_repo, visit_repo, soap_service)
        result = await use_case.execute(decoded_request, doctor_id=doctor_id)
        
        # Convert DTO to response format (encode patient_id for client)
        from ...core.utils.crypto import encode_patient_id
        return {
            "patient_id": encode_patient_id(result.patient_id),
            "visit_id": result.visit_id,
            "soap_note": result.soap_note,
            "generated_at": result.generated_at,
            "message": result.message
        }
        
    except ValueError as e:
        error_message = str(e)
        logger.warning(f"SOAP generation validation failed: {error_message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_REQUEST",
                "message": error_message,
                "details": {"reason": "Visit not ready for SOAP generation"},
            },
        )
    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "PATIENT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "VISIT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error("Unhandled error in generate_soap_note", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


# ------------------------
# Vitals endpoints
# ------------------------

class VitalsPayload(BaseModel):
    patient_id: str = Field(...)
    visit_id: str = Field(...)
    vitals: Dict[str, Any] = Field(...)


@router.post(
    "/vitals",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    include_in_schema=False
)
async def store_vitals(
    http_request: Request,
    payload: VitalsPayload,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """Store objective vitals for a visit."""
    try:
        # Set IDs in request state for HIPAA audit middleware
        http_request.state.audit_patient_id = payload.patient_id
        http_request.state.audit_visit_id = payload.visit_id
        
        from ...domain.value_objects.patient_id import PatientId
        # decode opaque id if needed
        try:
            internal_patient_id = decode_patient_id(payload.patient_id)
        except Exception:
            internal_patient_id = payload.patient_id
        doctor_id = getattr(http_request.state, "doctor_id", None)
        if not doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "MISSING_DOCTOR_ID",
                    "message": "X-Doctor-ID header is required",
                    "details": {},
                },
            )

        patient = await patient_repo.find_by_id(PatientId(internal_patient_id), doctor_id)
        if not patient:
            raise PatientNotFoundError(payload.patient_id)
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(payload.visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj, doctor_id
        )
        if not visit:
            raise VisitNotFoundError(payload.visit_id)
        visit.store_vitals(payload.vitals)
        
        # Update visit status appropriately after storing vitals
        visit.complete_vitals()  # Handles both walk-in and scheduled workflows
        
        # Additional status updates for scheduled visits with existing transcripts
        if visit.is_scheduled_workflow() and visit.is_transcription_complete():
            # If transcript exists, update to soap_generation
            if visit.status not in ["soap_generation", "prescription_analysis", "completed"]:
                visit.status = "soap_generation"
        
        await visit_repo.save(visit)
        return {"success": True, "message": "Vitals stored", "vitals_id": f"{payload.visit_id}:vitals"}
    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "PATIENT_NOT_FOUND", "message": e.message, "details": e.details},
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "VISIT_NOT_FOUND", "message": e.message, "details": e.details},
        )
    except Exception as e:
        logger.error("Unhandled error in store_vitals", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


@router.get(
    "/{patient_id}/visits/{visit_id}/vitals",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    include_in_schema=False
)
async def get_vitals(
    request: Request,
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
) -> Dict[str, Any]:
    try:
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "MISSING_DOCTOR_ID",
                    "message": "X-Doctor-ID header is required",
                    "details": {},
                },
            )
        from ...domain.value_objects.patient_id import PatientId
        # decode opaque id
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id), doctor_id)
        if not patient:
            raise PatientNotFoundError(patient_id)
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj, doctor_id
        )
        if not visit:
            raise VisitNotFoundError(visit_id)
        if not visit.vitals:
            raise HTTPException(status_code=404, detail={"error": "VITALS_NOT_FOUND", "message": "No vitals found"})
        return visit.vitals
    except HTTPException:
        # Re-raise HTTPException directly (like 404 for vitals not found)
        raise
    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "PATIENT_NOT_FOUND", "message": e.message, "details": e.details},
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "VISIT_NOT_FOUND", "message": e.message, "details": e.details},
        )
    except Exception as e:
        logger.error("Unhandled error in get_vitals", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


@router.get(
    "/{patient_id}/visits/{visit_id}/dialogue",
    response_model=ApiResponse[TranscriptionSessionDTO],
    status_code=status.HTTP_200_OK,
    tags=["Vitals and Transcript Generation"],
    summary="Get transcript with doctor/patient dialogue",
    responses={
        200: {"description": "Transcript dialogue returned"},
        404: {"model": ErrorResponse, "description": "Transcript not found"},
        422: {"model": ErrorResponse, "description": "Invalid input"},
    },
)
async def get_transcription_dialogue(request: Request, patient_id: str, visit_id: str, patient_repo: PatientRepositoryDep, visit_repo: VisitRepositoryDep):
    """Get transcript + doctor/patient dialogue for a visit."""
    try:
        from ...domain.value_objects.patient_id import PatientId
        import urllib.parse
        
        # Find patient (decode opaque id from client)
        # URL-decode first to restore any encoded '=' characters in Fernet tokens
        decoded_path_param = urllib.parse.unquote(patient_id)
        
        # Check if this looks like an internal patient ID (format: name_mobile)
        # If it contains underscore and the part after underscore is all digits, treat as internal ID
        if '_' in decoded_path_param:
            parts = decoded_path_param.split('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                # This looks like an internal patient ID, skip decryption
                internal_patient_id = decoded_path_param
                logger.debug(f"Using internal patient ID: {internal_patient_id}")
            else:
                # Try to decrypt as opaque token
                try:
                    internal_patient_id = decode_patient_id(decoded_path_param)
                except Exception as e:
                    logger.warning(f"Failed to decode patient_id '{decoded_path_param}': {e}")
                    internal_patient_id = decoded_path_param
        else:
            # Try to decrypt as opaque token
            try:
                internal_patient_id = decode_patient_id(decoded_path_param)
            except Exception as e:
                logger.warning(f"Failed to decode patient_id '{decoded_path_param}': {e}")
                internal_patient_id = decoded_path_param
        try:
            patient_id_obj = PatientId(internal_patient_id)
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "INVALID_PATIENT_ID",
                    "message": str(ve),
                    "details": {
                        "hint": "Provide a valid opaque patient_id token or an internal id like {name}_{phone}",
                    },
                },
            )
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "MISSING_DOCTOR_ID",
                    "message": "X-Doctor-ID header is required",
                    "details": {},
                },
            )
        patient = await patient_repo.find_by_id(patient_id_obj, doctor_id)
        if not patient:
            raise PatientNotFoundError(patient_id)

        # Find visit
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj, doctor_id
        )
        if not visit:
            raise VisitNotFoundError(visit_id)

        # If transcript not ready, or transcription is in-progress, advertise processing with Retry-After
        # Status values: "pending" (not started), "processing" (in progress), "completed" (done), "failed" (error)
        transcription_status = getattr(visit.transcription_session, "transcription_status", "").lower() if visit.transcription_session else "pending"
        if (
            not visit.transcription_session
            or not getattr(visit.transcription_session, "transcript", None)
            or transcription_status in {"pending", "processing"}
        ):
            headers = {"Retry-After": "60"}  # 1 minute polling interval
            return FastAPIResponse(content=b"", status_code=status.HTTP_202_ACCEPTED, headers=headers)

        # Get the transcription session
        session = visit.transcription_session

        # Return transcript data including any stored structured dialogue
        return ok(request, data=TranscriptionSessionDTO(
            audio_file_path=session.audio_file_path,
            transcript=session.transcript,
            transcription_status=session.transcription_status,
            started_at=session.started_at.isoformat() if session.started_at else None,
            completed_at=session.completed_at.isoformat() if session.completed_at else None,
            error_message=session.error_message,
            audio_duration_seconds=session.audio_duration_seconds,
            word_count=session.word_count,
            structured_dialogue=getattr(session, "structured_dialogue", None),  # Return stored structured dialogue from database
        ), message="Success")

    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "PATIENT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "VISIT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error("Unhandled error in get_transcript", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


@router.get(
    "/{patient_id}/visits/{visit_id}/soap",
    # Use Dict[str, Any] so we can control key insertion order at runtime
    response_model=ApiResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    tags=["SOAP Note Generation"],
    responses={
        404: {"model": ErrorResponse, "description": "Patient, visit, or SOAP note not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_soap_note(request: Request, patient_id: str, visit_id: str, patient_repo: PatientRepositoryDep, visit_repo: VisitRepositoryDep):
    """Get SOAP note for a visit."""
    try:
        from ...domain.value_objects.patient_id import PatientId
        from ...core.utils.crypto import decode_patient_id
        import urllib.parse

        # Support opaque patient_id tokens from clients
        decoded_param = urllib.parse.unquote(patient_id)
        try:
            internal_patient_id = decode_patient_id(decoded_param)
        except Exception:
            internal_patient_id = decoded_param

        # Get doctor_id from request state for multi-doctor isolation
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "MISSING_DOCTOR_ID",
                    "message": "X-Doctor-ID header is required",
                    "details": {},
                },
            )

        # Find patient (scoped by doctor_id)
        patient_id_obj = PatientId(internal_patient_id)
        patient = await patient_repo.find_by_id(patient_id_obj, doctor_id)
        if not patient:
            raise PatientNotFoundError(internal_patient_id)

        # Find visit (scoped by doctor_id)
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj, doctor_id
        )
        if not visit:
            raise VisitNotFoundError(visit_id)

        # Check if SOAP note exists
        if not visit.soap_note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "SOAP_NOTE_NOT_FOUND",
                    "message": f"No SOAP note found for visit {visit_id}",
                    "details": {"visit_id": visit_id},
                },
            )

        # Return SOAP note data
        soap = visit.soap_note

        # Load doctor preferences to determine soap_order
        try:
            from ...adapters.db.mongo.models.patient_m import DoctorPreferencesMongo
            prefs = await DoctorPreferencesMongo.find_one(
                DoctorPreferencesMongo.doctor_id == doctor_id
            )
            default_order = ["subjective", "objective", "assessment", "plan"]
            raw_order = getattr(prefs, "soap_order", None) if prefs else None
            if raw_order and isinstance(raw_order, list) and raw_order:
                soap_order = [s for s in raw_order if s in default_order]
                if len(soap_order) != len(default_order):
                    soap_order = default_order
            else:
                soap_order = default_order
        except Exception:
            # Fail-open: if preferences cannot be loaded, use default order
            soap_order = ["subjective", "objective", "assessment", "plan"]

        # Build payload in the desired key order.
        # JSON key order will follow insertion order of this dict.
        section_map: Dict[str, Any] = {
            "subjective": soap.subjective or "",
            "objective": soap.objective or {},
            "assessment": soap.assessment or "",
            "plan": soap.plan or "",
        }

        payload: Dict[str, Any] = {}
        for key in soap_order:
            # Only insert known sections to avoid unexpected keys
            if key in section_map:
                payload[key] = section_map[key]

        # Append non-ordered metadata fields
        payload["highlights"] = soap.highlights or []
        payload["red_flags"] = soap.red_flags or []
        payload["generated_at"] = (
            soap.generated_at.isoformat() if soap.generated_at else ""
        )
        payload["model_info"] = soap.model_info or {}
        payload["confidence_score"] = soap.confidence_score
        payload["soap_order"] = soap_order

        return ok(request, data=payload, message="Success")

    except HTTPException:
        # Re-raise HTTPException directly (like 404 for SOAP note not found)
        raise
    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "PATIENT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "VISIT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error("Unhandled error in get_soap_note", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


@router.post(
    "/{patient_id}/visits/{visit_id}/dialogue/structure",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    responses={
        404: {"model": ErrorResponse, "description": "Patient, visit, or transcript not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def structure_dialogue(request: Request, patient_id: str, visit_id: str, patient_repo: PatientRepositoryDep, visit_repo: VisitRepositoryDep) -> Dict[str, Any]:
    """Clean PII and structure transcript into alternating Doctor/Patient JSON using LLM."""
    try:
        # Resolve patient and transcript
        from ...domain.value_objects.patient_id import PatientId
        from urllib.parse import unquote

        decoded = unquote(patient_id)
        try:
            internal_patient_id = decode_patient_id(decoded)
        except Exception:
            internal_patient_id = decoded

        try:
            pid = PatientId(internal_patient_id)
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "INVALID_PATIENT_ID",
                    "message": str(ve),
                    "details": {},
                },
            )

        patient = await patient_repo.find_by_id(pid)
        if not patient:
            raise PatientNotFoundError(internal_patient_id)

        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj
        )
        if not visit:
            raise VisitNotFoundError(visit_id)
        if not visit.transcription_session or not visit.transcription_session.transcript:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "TRANSCRIPT_NOT_FOUND",
                    "message": f"No transcript found for visit {visit_id}",
                    "details": {"visit_id": visit_id},
                },
            )

        raw_transcript = visit.transcription_session.transcript

        # Use the same logic as adhoc transcribe (no caching to ensure fresh processing)
        from ...application.utils.structure_dialogue import structure_dialogue_from_text

        settings = get_settings()
        model = settings.azure_openai.deployment_name

        logger.info(f"Calling structure_dialogue_from_text with deployment: {model}")
        dialogue = await structure_dialogue_from_text(
            raw_transcript, 
            model=model,
            azure_endpoint=settings.azure_openai.endpoint,
            azure_api_key=settings.azure_openai.api_key
        )
        logger.info(f"Structure dialogue result: {type(dialogue)}, length: {len(dialogue) if isinstance(dialogue, list) else 'N/A'}")

        # Normalize dialogue to a list (empty list if None)
        normalized_dialogue: List[Dict[str, str]] = dialogue if isinstance(dialogue, list) else []

        # Apply PII removal to the structured dialogue (even though LLM should have done it)
        if normalized_dialogue:
            from ...application.use_cases.transcribe_audio import TranscribeAudioUseCase
            # Create a temporary instance just to use the PII removal methods
            # We don't need the full use case, just the PII removal logic
            temp_transcription_service = None  # Not needed for PII removal
            temp_use_case = TranscribeAudioUseCase(None, None, temp_transcription_service)  # type: ignore
            
            # Apply PII removal (standard + aggressive)
            normalized_dialogue = temp_use_case._remove_pii_from_dialogue(normalized_dialogue)
            normalized_dialogue = temp_use_case._aggressive_pii_removal_from_dialogue(normalized_dialogue)
            
            logger.info(f"Applied PII removal to structured dialogue")

        # Save the structured dialogue back to the database for future use
        if normalized_dialogue and visit.transcription_session:
            try:
                visit.transcription_session.structured_dialogue = normalized_dialogue
                await patient_repo.save(patient)
                logger.info(f"Saved structured dialogue to database with {len(normalized_dialogue)} turns")
            except Exception as e:
                logger.warning(f"Failed to save structured dialogue to database: {e}")

        logger.info(f"Returning dialogue with {len(normalized_dialogue)} turns")
        return ok(request, data={"dialogue": normalized_dialogue}, message="Success")


    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "PATIENT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "VISIT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error("Unhandled error in get_vitals", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )