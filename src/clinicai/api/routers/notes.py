"""Note-related API endpoints for Step-03 functionality."""
import logging

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
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
from ...application.utils.structure_dialogue import structure_dialogue_from_text
from clinicai.application.use_cases.generate_soap_note import GenerateSoapNoteUseCase
from clinicai.domain.errors import (
    PatientNotFoundError,
    VisitNotFoundError,
)
from ..deps import PatientRepositoryDep, TranscriptionServiceDep, SoapServiceDep
from ...core.utils.crypto import decode_patient_id
# removed adhoc structured dialogue generation for ad-hoc transcribe
from ..schemas.patient import ErrorResponse
from ...core.config import get_settings
from ...adapters.db.mongo.models.patient_m import AdhocTranscriptMongo

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

        # Mark visit as queued if possible (best-effort via use case during processing)
        try:
            # Check if patient and visit exist and are in correct status
            patient = await patient_repo.get_by_id(internal_patient_id)
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
            
            # Start transcription session
            visit.start_transcription(temp_file_path)
            await patient_repo.save(patient)
            logger.info(f"Started transcription session for patient {internal_patient_id}, visit {visit_id}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Failed to start transcription session: {e}. Continuing with background processing...")

        async def _run_background():
            try:
                logger.info(f"Starting background transcription for patient {internal_patient_id}, visit {visit_id}")
                logger.info(f"Audio file path: {temp_file_path}")
                logger.info(f"File exists: {os.path.exists(temp_file_path) if temp_file_path else False}")
                
                req = AudioTranscriptionRequest(
                    patient_id=internal_patient_id,
                    visit_id=visit_id,
                    audio_file_path=temp_file_path,
                )
                use_case = TranscribeAudioUseCase(patient_repo, transcription_service)
                
                logger.info("Executing transcription use case...")
                result = await use_case.execute(req)
                logger.info(f"Transcription completed successfully for patient {internal_patient_id}: {result}")
                
            except Exception as e:
                logger.error(f"Background transcription failed for patient {internal_patient_id}, visit {visit_id}: {e}", exc_info=True)
                # Mark transcription as failed in the database
                try:
                    patient = await patient_repo.get_by_id(internal_patient_id)
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
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        logger.info(f"Cleaned up temp file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up temp file {temp_file_path}: {cleanup_error}")

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


# ------------------------
# (adhoc transcription moved to /transcription router)
# ------------------------


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
        # Decode opaque patient id from client before passing to use case
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

        # If no structured dialogue cached yet, auto-structure and persist (server-side) for fast future loads
        session = visit.transcription_session
        try:
            if session and not getattr(session, "structured_dialogue", None) and session.transcript:
                raw = session.transcript
                # Chunk long transcripts
                import re as _re, json as _json
                sents = [_s.strip() for _s in _re.split(r"(?<=[.!?])\s+", raw) if _s.strip()]
                chunks: list[str] = []
                buf = ""
                for s in sents:
                    if len(buf) + len(s) + 1 > 6000 and buf:
                        chunks.append(buf)
                        buf = s
                    else:
                        buf = f"{buf} {s}".strip()
                if buf:
                    chunks.append(buf)

                settings = get_settings()
                client = OpenAI(api_key=settings.openai.api_key)
                sys_prompt = (
                    "You are an AI assistant that structures a medical visit transcript.\n"
                    "Return ONLY valid JSON. Do not include text outside JSON.\n"
                    "Rules:\n- Remove PII; fix typos; output as an array of ordered turns with keys Doctor/Patient."
                )
                def _call(text: str) -> list[dict]:
                    user_prompt = "Transcript:\n" + text + "\n\nReturn JSON ONLY, no markdown, no comments."
                    try:
                        resp = client.chat.completions.create(
                            model=settings.openai.model,
                            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                            max_tokens=min(1200, settings.openai.max_tokens or 1200),
                            temperature=0.1,
                        )
                        content = (resp.choices[0].message.content or "").strip()
                        parsed = _json.loads(content)
                        return parsed if isinstance(parsed, list) else []
                    except Exception:
                        # Heuristic split
                        turns: list[dict] = []
                        next_role = "Doctor"
                        for s in [_s.strip() for _s in _re.split(r"(?<=[.!?])\s+", text) if _s.strip()]:
                            lower = s.lower()
                            if lower.startswith("doctor:"):
                                turns.append({"Doctor": s.split(":",1)[1].strip()})
                                next_role = "Patient"
                            elif lower.startswith("patient:"):
                                turns.append({"Patient": s.split(":",1)[1].strip()})
                                next_role = "Doctor"
                            else:
                                turns.append({next_role: s})
                                next_role = "Patient" if next_role == "Doctor" else "Doctor"
                        return turns

                parts: list[dict] = []
                for ch in (chunks or [raw]):
                    parts.extend(_call(ch))
                if parts:
                    session.structured_dialogue = parts
                    await patient_repo.save(patient)
        except Exception:
            pass

        # Return transcript data including structured dialogue if cached
        return TranscriptionSessionDTO(
            audio_file_path=session.audio_file_path,
            transcript=session.transcript,
            transcription_status=session.transcription_status,
            started_at=session.started_at.isoformat() if session.started_at else None,
            completed_at=session.completed_at.isoformat() if session.completed_at else None,
            error_message=session.error_message,
            audio_duration_seconds=session.audio_duration_seconds,
            word_count=session.word_count,
            structured_dialogue=getattr(session, "structured_dialogue", None),
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
        
        # Find patient (decode opaque id)
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
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

        # Fast path: if structured dialogue was already cached during a prior request, return it
        if getattr(visit.transcription_session, "structured_dialogue", None):
            return {"dialogue": visit.transcription_session.structured_dialogue}

        # Use improved chunking strategy for long transcripts
        safe_transcript = raw_transcript or ""
        
        # Improved system prompt matching the transcribe_audio.py implementation
        system_prompt = (
            "You are a medical transcript processing AI. Your task is to create a structured dialogue between a Doctor and Patient.\n\n"
            "CRITICAL RULES:\n"
            "1. Remove all personal identifiers (names, addresses, phone numbers, specific dates)\n"
            "2. Fix obvious transcription errors while preserving medical meaning\n"
            "3. Output ONLY a JSON array where each element is an object with one key: either \"Doctor\" or \"Patient\"\n"
            "4. Use these speaker identification rules in priority order:\n"
            "   DOCTOR indicators:\n"
            "   - Questions about symptoms, history, medications\n"
            "   - Medical instructions, prescriptions, recommendations\n"
            "   - Examination procedures, test orders\n"
            "   - Professional language, medical terminology\n"
            "   - Phrases: \"Let me examine\", \"I'll prescribe\", \"We'll schedule\", \"Any allergies?\"\n"
            "   PATIENT indicators:\n"
            "   - Personal symptoms, feelings, experiences\n"
            "   - Answers to doctor's questions\n"
            "   - Personal history, family history\n"
            "   - Questions about treatment, concerns\n"
            "   - Phrases: \"I feel\", \"I have\", \"My pain\", \"It started\", \"I took\"\n"
            "5. Maintain conversation flow - if uncertain, consider context from previous turns\n"
            "6. DO NOT use markdown, explanations, or anything outside the JSON array\n"
            "OUTPUT FORMAT (JSON array only):\n"
            "[{\"Doctor\": \"...\"}, {\"Patient\": \"...\"}, {\"Doctor\": \"...\"}]"
        )

        settings = get_settings()
        client = OpenAI(api_key=settings.openai.api_key)

        # Use improved chunking strategy for long transcripts
        max_chars_per_chunk = 8000 if settings.openai.model.startswith('gpt-4') else 6000
        overlap_chars = 500
        
        # Split into sentences for better chunking
        import re as _re
        sentences = [_s.strip() for _s in _re.split(r"(?<=[.!?])\s+", safe_transcript) if _s.strip()]
        
        if len(safe_transcript) <= max_chars_per_chunk:
            # Single chunk processing
            user_prompt = f"Process this medical consultation transcript into structured dialogue:\n\n{safe_transcript}\n\nReturn JSON array only."
            
            def _call_openai() -> str:
                try:
                    max_tokens = 4000 if settings.openai.model.startswith('gpt-4') else 2000
                    logger.info(f"Structure dialogue: Calling OpenAI API with model: {settings.openai.model}, max_tokens: {max_tokens}")
                    resp = client.chat.completions.create(
                        model=settings.openai.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                    content = (resp.choices[0].message.content or "").strip()
                    logger.info(f"Structure dialogue: OpenAI API succeeded. Response length: {len(content)} characters")
                    return content
                except Exception as e:
                    logger.error(f"Structure dialogue: OpenAI API failed: {str(e)}", exc_info=True)
                    logger.error(f"Structure dialogue: Raw transcript length: {len(raw_transcript)} characters")
                    raise
            
            content = await asyncio.to_thread(_call_openai)
        else:
            # Multi-chunk processing with overlap
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > max_chars_per_chunk and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from previous
                    overlap_start = max(0, len(current_chunk) - overlap_chars)
                    current_chunk = current_chunk[overlap_start:] + " " + sentence
                else:
                    current_chunk += (" " + sentence) if current_chunk else sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            logger.info(f"Processing transcript in {len(chunks)} chunks with {overlap_chars} char overlap")
            
            def _call_openai_chunk(text: str) -> str:
                try:
                    max_tokens = 4000 if settings.openai.model.startswith('gpt-4') else 2000
                    user_prompt = f"Process this part of a medical consultation transcript:\n\n{text}\n\nReturn JSON array only."
                    resp = client.chat.completions.create(
                        model=settings.openai.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                    content = (resp.choices[0].message.content or "").strip()
                    return content
                except Exception as e:
                    logger.error(f"Chunk processing failed: {str(e)}", exc_info=True)
                    return ""
            
            # Process each chunk
            chunk_results = []
            for i, chunk in enumerate(chunks):
                chunk_result = await asyncio.to_thread(_call_openai_chunk, chunk)
                
                if chunk_result:
                    try:
                        import json as _json
                        parsed = _json.loads(chunk_result)
                        if isinstance(parsed, list):
                            chunk_results.extend(parsed)
                        else:
                            logger.warning(f"Chunk {i+1} returned invalid format")
                            chunk_results.append({"Doctor": f"[Chunk {i+1} processing failed]"})
                    except:
                        logger.warning(f"Chunk {i+1} JSON parsing failed")
                        chunk_results.append({"Doctor": f"[Chunk {i+1} processing failed]"})
                else:
                    logger.warning(f"Chunk {i+1} processing failed")
                    chunk_results.append({"Doctor": f"[Chunk {i+1} processing failed]"})
            
            # Merge and clean up overlapping content
            merged = [chunk_results[0]] if chunk_results else []
            for i in range(1, len(chunk_results)):
                current_chunk = chunk_results[i]
                if (merged and current_chunk and 
                    len(merged[-1]) == 1 and len(current_chunk[0]) == 1 and
                    list(merged[-1].keys())[0] == list(current_chunk[0].keys())[0] and
                    list(merged[-1].values())[0] == list(current_chunk[0].values())[0]):
                    merged.extend(current_chunk[1:])
                else:
                    merged.extend(current_chunk)
            
            # Convert back to JSON string
            import json as _json
            content = _json.dumps(merged)
        
        try:
            # Cache parsed structured dialogue to DB for fast retrieval
            import json as _json
            try:
                parsed = _json.loads(content)
                if isinstance(parsed, list):
                    if not visit.transcription_session:
                        raise RuntimeError("No transcription session to cache dialogue")
                    visit.transcription_session.structured_dialogue = parsed
                    await patient_repo.save(patient)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to structure dialogue: {str(e)}")
            # Heuristic fallback: split into sentences and alternate speakers
            import re
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", safe_transcript) if s.strip()]
            turns: list[dict[str, str]] = []
            next_role = "Doctor"
            for s in sentences:
                # Detect explicit speaker labels if present
                lower = s.lower()
                if lower.startswith("doctor:"):
                    turns.append({"Doctor": s.split(":", 1)[1].strip()})
                    next_role = "Patient"
                elif lower.startswith("patient:"):
                    turns.append({"Patient": s.split(":", 1)[1].strip()})
                    next_role = "Doctor"
                else:
                    turns.append({next_role: s})
                    next_role = "Patient" if next_role == "Doctor" else "Doctor"
            try:
                if not visit.transcription_session:
                    raise RuntimeError("No transcription session to cache dialogue")
                visit.transcription_session.structured_dialogue = turns
                await patient_repo.save(patient)
            except Exception:
                pass
            return {"dialogue": turns if turns else {"text": safe_transcript}}

        # Try to parse JSON; if not valid JSON object, return as text under 'dialogue'
        import json
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # If dict, coerce to ordered list preserving keys order where possible
                ordered: list[dict[str, str]] = []
                for k, v in data.items():
                    if k in ("Doctor", "Patient") and isinstance(v, str):
                        ordered.append({k: v})
                if ordered:
                    return {"dialogue": ordered}
                logger.warning("Structured content doesn't contain Doctor/Patient keys")
                return {"dialogue": {"text": content}}
            # If it's a list of pairs, convert to dict-like sequence
            if isinstance(data, list):
                merged: Dict[str, Any] = {}
                for item in data:
                    if isinstance(item, dict):
                        merged.update(item)
                if any(key in merged for key in ["Doctor", "Patient"]):
                    # Prefer returning the original list to preserve order
                    return {"dialogue": data}
                return {"dialogue": {"text": content}}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse structured content as JSON: {e}")
            return {"dialogue": {"text": content}}
        except Exception as e:
            logger.warning(f"Error processing structured content: {e}")
            return {"dialogue": {"text": content}}

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
