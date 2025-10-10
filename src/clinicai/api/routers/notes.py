"""Note-related API endpoints for Step-03 functionality."""
import logging

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import tempfile
import os
import asyncio
from datetime import datetime

from openai import OpenAI  # type: ignore

from clinicai.application.dto.patient_dto import (
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    SoapGenerationRequest,
    SoapGenerationResponse,
    SoapNoteDTO,
    TranscriptionSessionDTO
)
from pydantic import BaseModel
from clinicai.application.use_cases.transcribe_audio import TranscribeAudioUseCase
from ...application.utils.structure_dialogue import structure_dialogue_from_text
from clinicai.application.use_cases.generate_soap_note import GenerateSoapNoteUseCase
from clinicai.domain.errors import (
    PatientNotFoundError,
    VisitNotFoundError,
)
from ..deps import PatientRepositoryDep, TranscriptionServiceDep, AudioRepositoryDep, SoapServiceDep
from ...core.utils.crypto import decode_patient_id
from ..schemas.patient import ErrorResponse
from ...core.config import get_settings
from ...adapters.db.mongo.models.patient_m import AdhocTranscriptMongo

router = APIRouter(prefix="/notes", tags=["notes"])
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


@router.options("/transcribe")
async def transcribe_audio_options():
    """Handle preflight OPTIONS request for transcribe endpoint."""
    return {"message": "OK"}


@router.get("/test-cors")
async def test_cors():
    """Test endpoint to verify CORS is working."""
    return {"message": "CORS is working", "timestamp": "2024-01-01T00:00:00Z"}


@router.post(
    "/transcribe",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Queued for transcription"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "Invalid audio file or visit status"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def transcribe_audio(
    patient_repo: PatientRepositoryDep,
    transcription_service: TranscriptionServiceDep,
    audio_repo: AudioRepositoryDep,
    background: BackgroundTasks,
    patient_id: str = Form(...),
    visit_id: str = Form(...),
    language: str = Form("en"),
    audio_file: UploadFile = File(...),
):
    """Queue audio transcription and return immediately (202)."""
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
    try:
        internal_patient_id = decode_patient_id(patient_id)
    except Exception:
        if '_' in patient_id and patient_id.count('_') >= 1:
            internal_patient_id = patient_id
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "INVALID_PATIENT_ID", "message": "Invalid patient ID format", "details": {}},
            )

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
            patient = await patient_repo.find_by_id(internal_patient_id)
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "PATIENT_NOT_FOUND", "message": f"Patient {internal_patient_id} not found", "details": {}},
                )
            
            visit = patient.get_visit_by_id(visit_id)
            if not visit:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "VISIT_NOT_FOUND", "message": f"Visit {visit_id} not found", "details": {}},
                )
            
            logger.info(f"Patient {internal_patient_id} and visit {visit_id} found. Visit status: {visit.status}")
            
            # Audio file will be saved in background task where database context is properly initialized

            # Start transcription session (no longer using file path)
            visit.start_transcription(None)  # No file path needed
            await patient_repo.save(patient)
            logger.info(f"Started transcription session for patient {internal_patient_id}, visit {visit_id}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Failed to start transcription session: {e}. Continuing with background processing...")

        async def _run_background():
            try:
                logger.info(f"Starting background transcription for patient {internal_patient_id}, visit {visit_id}")
                
                # Save audio file to database in background task where database context is properly initialized
                audio_file_record = None
                try:
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
                except Exception as e:
                    logger.error(f"Failed to save audio file to database: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    # Continue without failing the transcription
                
                # Get audio data from database if we have an audio file record
                if audio_file_record:
                    logger.info(f"Using audio data from database: {audio_file_record.audio_id}")
                    # Create a temporary file with the audio data for transcription
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file_record.filename.split('.')[-1]}") as temp_audio_file:
                        temp_audio_file.write(audio_data)
                        temp_audio_path = temp_audio_file.name
                    
                    req = AudioTranscriptionRequest(
                        patient_id=internal_patient_id,
                        visit_id=visit_id,
                        audio_file_path=temp_audio_path,
                        language=language,
                    )
                else:
                    # Fallback to temp file if no database record
                    logger.info(f"Using temp file path: {temp_file_path}")
                    req = AudioTranscriptionRequest(
                        patient_id=internal_patient_id,
                        visit_id=visit_id,
                        audio_file_path=temp_file_path,
                        language=language,
                    )
                
                use_case = TranscribeAudioUseCase(patient_repo, transcription_service)
                
                logger.info("Executing transcription use case...")
                result = await use_case.execute(req)
                logger.info(f"Transcription completed successfully for patient {internal_patient_id}: {result}")
                
                # Update audio file with duration if we have the result
                if audio_file_record and result.audio_duration:
                    await audio_repo.update_audio_metadata(
                        audio_file_record.audio_id,
                        duration_seconds=result.audio_duration
                    )
                    logger.info(f"Updated audio file duration: {result.audio_duration} seconds")
                
            except Exception as e:
                logger.error(f"Background transcription failed for patient {internal_patient_id}, visit {visit_id}: {e}", exc_info=True)
                # Mark transcription as failed in the database
                try:
                    patient = await patient_repo.find_by_id(internal_patient_id)
                    if patient:
                        visit = patient.get_visit_by_id(visit_id)
                        if visit and visit.transcription_session:
                            visit.fail_transcription(str(e))
                            await patient_repo.save(patient)
                            logger.info(f"Marked transcription as failed for patient {internal_patient_id}")
                except Exception as db_error:
                    logger.error(f"Failed to mark transcription as failed in database: {db_error}")
            finally:
                try:
                    # Clean up temp files
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        logger.info(f"Cleaned up temp file: {temp_file_path}")
                    # Clean up any additional temp files created in background
                    if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                        logger.info(f"Cleaned up background temp file: {temp_audio_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up temp files: {cleanup_error}")

        # Schedule background processing
        asyncio.create_task(_run_background())

        return {"status": "queued", "patient_id": internal_patient_id, "visit_id": visit_id}

    except HTTPException:
        # Reraise HTTPException directly
        raise
    except Exception as e:
        logger.error("Unhandled error queuing transcription", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e), "details": {}},
        )


@router.post(
    "/soap/generate",
    response_model=SoapGenerationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "No transcript available or invalid visit status"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def generate_soap_note(
    request: SoapGenerationRequest,
    patient_repo: PatientRepositoryDep,
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
        decoded_request = SoapGenerationRequest(
            patient_id=internal_patient_id,
            visit_id=request.visit_id,
            transcript=request.transcript,
        )

        # Execute use case
        use_case = GenerateSoapNoteUseCase(patient_repo, soap_service)
        result = await use_case.execute(decoded_request)
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_REQUEST",
                "message": str(e),
                "details": {},
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
)
async def store_vitals(
    payload: VitalsPayload,
    patient_repo: PatientRepositoryDep,
):
    """Store objective vitals for a visit."""
    try:
        from ...domain.value_objects.patient_id import PatientId
        # decode opaque id if needed
        try:
            internal_patient_id = decode_patient_id(payload.patient_id)
        except Exception:
            internal_patient_id = payload.patient_id
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id))
        if not patient:
            raise PatientNotFoundError(payload.patient_id)
        visit = patient.get_visit_by_id(payload.visit_id)
        if not visit:
            raise VisitNotFoundError(payload.visit_id)
        visit.store_vitals(payload.vitals)
        await patient_repo.save(patient)
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
)
async def get_vitals(
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
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
        visit = patient.get_visit_by_id(visit_id)
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
    response_model=TranscriptionSessionDTO,
    status_code=status.HTTP_200_OK,
    responses={
        202: {"description": "Transcript processing; client should retry after delay"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_transcript(
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
):
    """Get transcript for a visit."""
    try:
        from ...domain.value_objects.patient_id import PatientId
        import urllib.parse
        
        # Find patient (decode opaque id from client)
        # URL-decode first to restore any encoded '=' characters in Fernet tokens
        decoded_path_param = urllib.parse.unquote(patient_id)
        # Attempt to decrypt opaque token. If decryption fails, only accept the raw value
        # if it already conforms to our internal PatientId format; otherwise return 422.
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
        visit = patient.get_visit_by_id(visit_id)
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
        return TranscriptionSessionDTO(
            audio_file_path=session.audio_file_path,
            transcript=session.transcript,
            transcription_status=session.transcription_status,
            started_at=session.started_at.isoformat() if session.started_at else None,
            completed_at=session.completed_at.isoformat() if session.completed_at else None,
            error_message=session.error_message,
            audio_duration_seconds=session.audio_duration_seconds,
            word_count=session.word_count,
            structured_dialogue=getattr(session, "structured_dialogue", None),  # Return stored structured dialogue from database
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
    response_model=SoapNoteDTO,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Patient, visit, or SOAP note not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_soap_note(
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
):
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
        visit = patient.get_visit_by_id(visit_id)
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
        return SoapNoteDTO(
            subjective=soap.subjective,
            objective=soap.objective,
            assessment=soap.assessment,
            plan=soap.plan,
            highlights=soap.highlights,
            red_flags=soap.red_flags,
            generated_at=soap.generated_at.isoformat(),
            model_info=soap.model_info,
            confidence_score=soap.confidence_score
        )

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
    responses={
        404: {"model": ErrorResponse, "description": "Patient, visit, or transcript not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def structure_dialogue(
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
) -> Dict[str, Any]:
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

        visit = patient.get_visit_by_id(visit_id)
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
        model = settings.openai.model
        api_key = settings.openai.api_key

        logger.info(f"Calling structure_dialogue_from_text with model: {model}")
        dialogue = await structure_dialogue_from_text(raw_transcript, model=model, api_key=api_key)
        logger.info(f"Structure dialogue result: {type(dialogue)}, length: {len(dialogue) if isinstance(dialogue, list) else 'N/A'}")

        # Normalize dialogue to a list (empty list if None)
        normalized_dialogue: List[Dict[str, str]] = dialogue if isinstance(dialogue, list) else []

        # Save the structured dialogue back to the database for future use
        if normalized_dialogue and visit.transcription_session:
            try:
                visit.transcription_session.structured_dialogue = normalized_dialogue
                await patient_repo.save(patient)
                logger.info(f"Saved structured dialogue to database with {len(normalized_dialogue)} turns")
            except Exception as e:
                logger.warning(f"Failed to save structured dialogue to database: {e}")

        logger.info(f"Returning dialogue with {len(normalized_dialogue)} turns")
        return {"dialogue": normalized_dialogue}


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