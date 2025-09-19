"""Note-related API endpoints for Step-03 functionality."""
import logging

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from typing import Optional, Dict, Any
import tempfile
import os
import asyncio

from openai import OpenAI  # type: ignore

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
from ...core.config import get_settings

router = APIRouter(prefix="/notes", tags=["notes"])
logger = logging.getLogger("clinicai")


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
    background: BackgroundTasks,
    patient_id: str = Form(...),
    visit_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """Queue audio transcription and return immediately (202)."""
    logger.info(f"Transcribe audio request received for patient_id: {patient_id}, visit_id: {visit_id}")

    if not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "NO_AUDIO_FILE", "message": "No audio file provided", "details": {}},
        )

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
    try:
        suffix = f".{(audio_file.filename or 'audio').split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            while True:
                chunk = await audio_file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)

        # Mark visit as queued if possible (best-effort via use case during processing)

        async def _run_background():
            try:
                req = AudioTranscriptionRequest(
                    patient_id=internal_patient_id,
                    visit_id=visit_id,
                    audio_file_path=temp_file_path,
                )
                use_case = TranscribeAudioUseCase(patient_repo, transcription_service)
                await use_case.execute(req)
                logger.info(f"Transcription completed for patient {internal_patient_id}")
            except Exception as e:
                logger.error(f"Background transcription failed: {e}", exc_info=True)
            finally:
                try:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                except Exception:
                    pass

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
            from fastapi import HTTPException, status
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

        # Build prompt
        system_prompt = (
            "You are an AI assistant processing raw transcripts generated from audio recordings.\n"
            "Your tasks are:\n"
            "1) Understand and clean the transcript: remove any names, phone numbers, or personal identifiers;\n"
            "   correct obvious transcription errors (spelling, spacing); keep only conversational content.\n"
            "2) Format the dialogue clearly into structured JSON with alternating keys \"Doctor\" and \"Patient\".\n"
            "   Preserve the natural flow and output valid JSON only, no extra commentary."
        )
        user_prompt = (
            "Example:\n"
            "{\n  \"Doctor\": \"How are you feeling today?\",\n  \"Patient\": \"I have been coughing for three days.\",\n"
            "  \"Doctor\": \"Do you have a fever?\",\n  \"Patient\": \"Yes, since yesterday.\"\n}\n\n"
            "Transcript to process (clean PII and structure):\n" + (raw_transcript or "")
        )

        settings = get_settings()
        client = OpenAI(api_key=settings.openai.api_key)

        def _call_openai() -> str:
            resp = client.chat.completions.create(
                model=settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=min(2000, settings.openai.max_tokens),
                temperature=0.1,
            )
            return (resp.choices[0].message.content or "").strip()

        content = await asyncio.to_thread(_call_openai)

        # Try to parse JSON; if not valid JSON object, return as text under 'dialogue'
        import json
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return {"dialogue": data}
            # If it's a list of pairs, convert to dict-like sequence
            if isinstance(data, list):
                merged: Dict[str, Any] = {}
                for item in data:
                    if isinstance(item, dict):
                        merged.update(item)
                return {"dialogue": merged or {"text": content}}
        except Exception:
            pass

        return {"dialogue": {"text": content}}

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
        logger.error("Unhandled error in structure_dialogue", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )
