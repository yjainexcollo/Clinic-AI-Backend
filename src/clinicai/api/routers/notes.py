"""Note-related API endpoints for Step-03 functionality."""
import logging

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import Response as FastAPIResponse
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
from ...adapters.db.mongo.models.patient_m import AdhocTranscriptMongo
try:
    from ...adapters.queue.azure_queue_service import get_azure_queue_service
    QUEUE_SERVICE_AVAILABLE = True
except ImportError:
    QUEUE_SERVICE_AVAILABLE = False
    logger.warning("Azure Queue Storage not available. Install azure-storage-queue package.")
from ..schemas.common import ApiResponse, ErrorResponse
from ..utils.responses import ok, fail

router = APIRouter(prefix="/notes")
logger = logging.getLogger("clinicai")


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
    
    logger.info(f"Transcribe audio request received for patient_id: {patient_id}, visit_id: {visit_id}, language: {language}")

    if not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "NO_AUDIO_FILE", "message": "No audio file provided", "details": {}},
        )

    # Validate file type (allow common audio types and mpeg containers)
    content_type = audio_file.content_type or ""
    is_audio_like = content_type.startswith("audio/")
    is_mpeg_container = content_type in ("video/mpeg", "video/mpg", "application/mpeg")
    is_generic_stream = content_type in ("application/octet-stream",)
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

    # Stream to temp file to avoid loading entire file in memory
    temp_file_path = None
    audio_data = b""
    audio_file_record = None  # Initialize outside try block for background task access
    try:
        ext = (audio_file.filename or 'audio').split('.')[-1]
        # Normalize common MPEG container cases
        if ext.lower() in {"mpeg", "mpg"}:
            ext = "mpeg"
        suffix = f".{ext}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            while True:
                chunk = await audio_file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
                audio_data += chunk

        # Mark visit as queued if possible (best-effort via use case during processing)
        try:
            # Check if patient and visit exist and are in correct status
            from clinicai.domain.value_objects.patient_id import PatientId
            patient = await patient_repo.find_by_id(PatientId(internal_patient_id))
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "PATIENT_NOT_FOUND", "message": f"Patient {internal_patient_id} not found", "details": {}},
                )
            
            from ...domain.value_objects.visit_id import VisitId
            visit_id_obj = VisitId(visit_id)
            visit = await visit_repo.find_by_patient_and_visit_id(
                internal_patient_id, visit_id_obj
            )
            if not visit:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "VISIT_NOT_FOUND", "message": f"Visit {visit_id} not found", "details": {}},
                )
            
            logger.info(f"Patient {internal_patient_id} and visit {visit_id} found. Visit status: {visit.status}")
            
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
            logger.info(f"Attempting to save audio file to database. Audio data size: {len(audio_data)} bytes")
            audio_file_record = await audio_repo.create_audio_file(
                audio_data=audio_data,
                filename=audio_file.filename or "unknown_audio",
                content_type=audio_file.content_type or "audio/mpeg",
                patient_id=internal_patient_id,
                visit_id=visit_id,
                audio_type="visit",
            )
            logger.info(f"Audio file saved to database: {audio_file_record.audio_id}")

            # Start transcription session
            visit.start_transcription(None)  # No file path needed
            await visit_repo.save(visit)
            logger.info(f"Started transcription session for patient {internal_patient_id}, visit {visit_id}")
            
            # Enqueue transcription job to Azure Queue
            if not QUEUE_SERVICE_AVAILABLE:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "QUEUE_SERVICE_UNAVAILABLE",
                        "message": "Azure Queue Storage service is not available. Please install azure-storage-queue package.",
                        "details": {}
                    }
                )
            
            queue_service = get_azure_queue_service()
            await queue_service.ensure_queue_exists()  # Ensure queue exists
            
            message_id = queue_service.enqueue_transcription_job(
                patient_id=internal_patient_id,
                visit_id=visit_id,
                audio_file_id=audio_file_record.audio_id,
                language=language
            )
            
            logger.info(f"Transcription job enqueued: message_id={message_id}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to queue transcription: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "QUEUE_ERROR", "message": f"Failed to queue transcription: {str(e)}", "details": {}},
            )
        finally:
            # Clean up temp file
            try:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temp file: {cleanup_error}")

        return {
            "status": "queued",
            "patient_id": internal_patient_id,
            "visit_id": visit_id,
            "message_id": message_id,
            "message": "Transcription queued successfully. Poll /notes/transcribe/status/{patient_id}/{visit_id} for status."
        }

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
            internal_patient_id, visit_id_obj
        )
        
        if not visit:
            return fail(request, error="VISIT_NOT_FOUND", message="Visit not found")
        
        if not visit.transcription_session:
            return ok(request, data={
                "status": "pending",
                "message": "Transcription not started"
            })
        
        transcription_status = visit.transcription_session.transcription_status
        status_info = {
            "status": transcription_status,  # pending, processing, completed, failed
            "progress": "unknown"
        }
        
        if transcription_status == "completed":
            status_info["transcript_available"] = True
            status_info["word_count"] = visit.transcription_session.word_count
            status_info["duration"] = visit.transcription_session.audio_duration_seconds
            status_info["message"] = "Transcription completed successfully"
        elif transcription_status == "processing":
            status_info["message"] = "Transcription in progress"
        elif transcription_status == "failed":
            status_info["error"] = visit.transcription_session.error_message
            status_info["message"] = f"Transcription failed: {visit.transcription_session.error_message}"
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
        )

        # Execute use case
        use_case = GenerateSoapNoteUseCase(patient_repo, visit_repo, soap_service)
        result = await use_case.execute(decoded_request)
        
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
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id))
        if not patient:
            raise PatientNotFoundError(payload.patient_id)
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(payload.visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj
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
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
) -> Dict[str, Any]:
    try:
        from ...domain.value_objects.patient_id import PatientId
        # decode opaque id
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id))
        if not patient:
            raise PatientNotFoundError(patient_id)
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj
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
    "/{patient_id}/visits/{visit_id}/transcript",
    response_model=ApiResponse[TranscriptionSessionDTO],
    status_code=status.HTTP_200_OK,
    tags=["Vitals and Transcript Generation"],
    responses={
        200: {"description": "Transcript returned"},
        404: {"model": ErrorResponse, "description": "Transcript not found"},
        422: {"model": ErrorResponse, "description": "Invalid input"},
    },
)
async def get_transcript(request: Request, patient_id: str, visit_id: str, patient_repo: PatientRepositoryDep, visit_repo: VisitRepositoryDep):
    """Get transcript for a visit."""
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
        patient = await patient_repo.find_by_id(patient_id_obj)
        if not patient:
            raise PatientNotFoundError(patient_id)

        # Find visit
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj
        )
        if not visit:
            raise VisitNotFoundError(visit_id)

        # If transcript not ready, or transcription is in-progress, advertise processing with Retry-After
        if (
            not visit.transcription_session
            or not getattr(visit.transcription_session, "transcript", None)
            or getattr(visit.transcription_session, "transcription_status", "").lower() in {"queued", "processing", "in_progress"}
        ):
            headers = {"Retry-After": "5"}
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
    response_model=ApiResponse[SoapNoteDTO],
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

        # Find patient
        patient_id_obj = PatientId(internal_patient_id)
        patient = await patient_repo.find_by_id(patient_id_obj)
        if not patient:
            raise PatientNotFoundError(internal_patient_id)

        # Find visit
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj
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
        soap_data = SoapNoteDTO(
            subjective=soap.subjective or "",
            objective=soap.objective or {},
            assessment=soap.assessment or "",
            plan=soap.plan or "",
            highlights=soap.highlights or [],
            red_flags=soap.red_flags or [],
            generated_at=soap.generated_at.isoformat() if soap.generated_at else "",
            model_info=soap.model_info or {},
            confidence_score=soap.confidence_score
        )
        return ok(request, data=soap_data, message="Success")

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