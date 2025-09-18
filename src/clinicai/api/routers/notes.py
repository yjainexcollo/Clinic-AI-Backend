"""Note-related API endpoints for Step-03 functionality."""
import logging

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from typing import Optional
import tempfile
import os

from clinicai.application.dto.patient_dto import (
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    SoapGenerationRequest,
    SoapGenerationResponse,
    SoapNoteDTO,
    TranscriptionSessionDTO
)
from clinicai.application.use_cases.transcribe_audio import TranscribeAudioUseCase
from clinicai.application.use_cases.generate_soap_note import GenerateSoapNoteUseCase
from clinicai.domain.errors import (
    PatientNotFoundError,
    VisitNotFoundError,
)
from ..deps import PatientRepositoryDep, TranscriptionServiceDep, SoapServiceDep
from ...core.utils.crypto import decode_patient_id
from ..schemas.patient import ErrorResponse

router = APIRouter(prefix="/notes", tags=["notes"])
logger = logging.getLogger("clinicai")


@router.post(
    "/transcribe",
    response_model=AudioTranscriptionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "Invalid audio file or visit status"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def transcribe_audio(
    patient_repo: PatientRepositoryDep,
    transcription_service: TranscriptionServiceDep,
    patient_id: str = Form(...),
    visit_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    Transcribe audio file for a visit.
    
    This endpoint:
    1. Validates the audio file format and size
    2. Saves the file temporarily
    3. Transcribes using AI service
    4. Updates visit status to soap_generation
    5. Cleans up temporary file
    """
    temp_file_path = None
    
    try:
        # Validate file type (allow common audio types and mpeg containers)
        content_type = audio_file.content_type or ""
        is_audio_like = content_type.startswith("audio/")
        is_mpeg_container = content_type in ("video/mpeg",)
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
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Decode opaque patient id from client
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id  # fallback if already internal

        # Create request
        request = AudioTranscriptionRequest(
            patient_id=internal_patient_id,
            visit_id=visit_id,
            audio_file_path=temp_file_path
        )
        
        # Execute use case
        use_case = TranscribeAudioUseCase(patient_repo, transcription_service)
        result = await use_case.execute(request)
        
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
        logger.error("Unhandled error in transcribe_audio", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass  # Ignore cleanup errors


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
        # Execute use case
        use_case = GenerateSoapNoteUseCase(patient_repo, soap_service)
        result = await use_case.execute(request)
        
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


@router.get(
    "/{patient_id}/visits/{visit_id}/transcript",
    response_model=TranscriptionSessionDTO,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Patient, visit, or transcript not found"},
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
        from ...core.utils.crypto import decode_patient_id
        
        # Find patient (decode opaque id from client)
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        patient_id_obj = PatientId(internal_patient_id)
        patient = await patient_repo.find_by_id(patient_id_obj)
        if not patient:
            raise PatientNotFoundError(patient_id)

        # Find visit
        visit = patient.get_visit_by_id(visit_id)
        if not visit:
            raise VisitNotFoundError(visit_id)

        # Check if transcript exists
        if not visit.transcription_session or not visit.transcription_session.transcript:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "TRANSCRIPT_NOT_FOUND",
                    "message": f"No transcript found for visit {visit_id}",
                    "details": {"visit_id": visit_id},
                },
            )

        # Return transcript data
        session = visit.transcription_session
        return TranscriptionSessionDTO(
            audio_file_path=session.audio_file_path,
            transcript=session.transcript,
            transcription_status=session.transcription_status,
            started_at=session.started_at.isoformat() if session.started_at else None,
            completed_at=session.completed_at.isoformat() if session.completed_at else None,
            error_message=session.error_message,
            audio_duration_seconds=session.audio_duration_seconds,
            word_count=session.word_count
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
        
        # Find patient
        patient_id_obj = PatientId(patient_id)
        patient = await patient_repo.find_by_id(patient_id_obj)
        if not patient:
            raise PatientNotFoundError(patient_id)

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
