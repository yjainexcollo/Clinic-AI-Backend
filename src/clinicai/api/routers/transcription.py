from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import tempfile
import logging
import json
from datetime import datetime

from ..deps import TranscriptionServiceDep, AudioRepositoryDep, ActionPlanServiceDep
from ...adapters.db.mongo.models.patient_m import AdhocTranscriptMongo
from ...adapters.db.mongo.repositories.audio_repository import AudioRepository
from beanie import PydanticObjectId
from ...core.config import get_settings
from ...core.utils.file_utils import save_audio_file
from ...application.utils.structure_dialogue import structure_dialogue_from_text
from ..schemas.common import ApiResponse, ErrorResponse
from ..utils.responses import ok, fail


router = APIRouter(prefix="/transcription", tags=["transcription"])
logger = logging.getLogger("clinicai")


# LLM processing removed - Azure Speech Service provides structured dialogue directly with speaker diarization

class AdhocTranscriptionResponse(BaseModel):
    transcript: str
    structured_dialogue: Optional[List[Dict[str, str]]] = None
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    word_count: Optional[int] = None
    model: Optional[str] = None
    filename: Optional[str] = None
    adhoc_id: Optional[str] = None


@router.post(
    "",
    response_model=ApiResponse[AdhocTranscriptionResponse],
    status_code=status.HTTP_200_OK,
)
async def transcribe_audio_adhoc(
    request: Request,
    transcription_service: TranscriptionServiceDep,
    audio_repo: AudioRepositoryDep,
    audio_file: Optional[UploadFile] = File(None),
    language: Optional[str] = Form("en"),
):
    """Transcribe audio file (adhoc - not tied to a visit)."""
    print("🔵 === Adhoc transcription request received ===")
    print(f"🔵 Content-Type: {request.headers.get('content-type', 'not set')}")
    print(f"🔵 Filename: {audio_file.filename if audio_file else 'None'}")
    logger.info(f"=== Adhoc transcription request received ===")
    logger.info(f"  Content-Type: {request.headers.get('content-type', 'not set')}")
    logger.info(f"  Filename: {audio_file.filename if audio_file else 'None'}")
    
    # Handle case where file might not be in the expected field name
    if audio_file is None or not audio_file.filename:
        try:
            form = await request.form()
            # Try common field names
            for field_name in ["audio_file", "file", "audio", "file_upload", "upload"]:
                if field_name in form:
                    file_item = form[field_name]
                    if isinstance(file_item, UploadFile):
                        audio_file = file_item
                        logger.info(f"Found audio file in field: {field_name}")
                        break
        except Exception as e:
            logger.warning(f"Failed to parse form data: {e}")
    
    if audio_file is None or not audio_file.filename:
        logger.error("No audio file filename provided")
        return fail(request, error="NO_AUDIO_FILE", message="No audio file provided")

    content_type = audio_file.content_type or ""
    is_audio_like = content_type.startswith("audio/")
    is_supported_video = content_type in ("video/mpeg", "video/webm", "video/mp4")
    is_generic_stream = content_type in ("application/octet-stream",)
    
    logger.info(f"Content type validation: audio_like={is_audio_like}, supported_video={is_supported_video}, generic_stream={is_generic_stream}")
    
    if not (is_audio_like or is_supported_video or is_generic_stream):
        logger.error(f"Invalid file type: {content_type}")
        return fail(request, error="INVALID_FILE_TYPE", message="File must be an audio file")

    temp_file_path = None
    audio_data = b""
    try:
        # First, read all the audio data
        print("🔵 Reading audio file data...")
        audio_data = await audio_file.read()
        print(f"🔵 Read {len(audio_data)} bytes of audio data")
        
        # Reset file pointer for transcription service
        print("🔵 Resetting file pointer...")
        await audio_file.seek(0)
        
        # Create temp file for transcription
        # Extract file extension properly (handle filenames with or without extensions)
        filename = audio_file.filename or 'audio'
        if '.' in filename:
            file_ext = filename.split('.')[-1].lower()
            # Normalize common extensions
            # MPEG/MPG audio files are typically MP3 format, so convert to .mp3 for Azure Speech Service
            if file_ext in ['mpeg', 'mpg']:
                file_ext = 'mp3'  # Convert to .mp3 since Azure Speech Service accepts .mp3 but not .mpeg
            suffix = f".{file_ext}"
        else:
            # No extension - try to infer from content type
            content_type = audio_file.content_type or ""
            if 'mp3' in content_type or 'mpeg' in content_type:
                suffix = ".mp3"
            elif 'wav' in content_type:
                suffix = ".wav"
            elif 'm4a' in content_type or 'mp4' in content_type:
                suffix = ".m4a"
            else:
                suffix = ".mp3"  # Default fallback
        print(f"🔵 Creating temp file with suffix: {suffix} (from filename: {filename})")
        logger.info(f"Creating temp file with suffix: {suffix} (from filename: {filename})")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_data)
            print(f"🔵 Temp file created: {temp_file_path}")
            logger.info(f"Temp file created: {temp_file_path}")

        # Validate and transcribe
        try:
            print(f"🔵 Validating audio file: {temp_file_path}")
            logger.info(f"Validating audio file: {temp_file_path}")
            meta = await transcription_service.validate_audio_file(temp_file_path)
            print(f"🔵 Validation result: {meta}")
            logger.info(f"Validation result: {meta}")
            if not meta.get("is_valid"):
                print(f"❌ Audio validation failed: {meta}")
                logger.error(f"Audio validation failed: {meta}")
                return fail(request, error="INVALID_AUDIO", message=meta.get("error") or "Invalid audio")
        except HTTPException as e:
            raise
        except Exception as e:
            print(f"⚠️ Audio validation error: {e}")
            logger.warning(f"Audio validation error: {e}")

        print(f"🔵 Starting transcription for file: {temp_file_path}, language: {language}")
        logger.info(f"Starting transcription for file: {temp_file_path}, language: {language}")
        result = await transcription_service.transcribe_audio(temp_file_path, language=language)
        
        raw_transcript = result.get("transcript") or ""
        print(f"🔵 === Transcription completed. Transcript length: {len(raw_transcript)} characters ===")
        logger.info(f"=== Transcription completed. Transcript length: {len(raw_transcript)} characters ===")
        
        if not raw_transcript or raw_transcript.strip() == "":
            print("❌ === ERROR: Empty transcript returned - stopping before database save ===")
            logger.error("=== ERROR: Empty transcript returned - stopping before database save ===")
            return fail(request, error="EMPTY_TRANSCRIPT", message="Transcription returned empty transcript")

        # Azure Speech Service provides structured dialogue with speaker diarization
        pre_structured_dialogue = result.get("structured_dialogue")
        speaker_info = result.get("speaker_labels", {})
        print("🔵 Extracting structured dialogue from result...")
        print(f"🔵 pre_structured_dialogue type: {type(pre_structured_dialogue)}, is_list: {isinstance(pre_structured_dialogue, list)}")
        print(f"🔵 speaker_info: {speaker_info}")
        
        if not pre_structured_dialogue or not isinstance(pre_structured_dialogue, list):
            print(f"❌ === ERROR: No structured dialogue - stopping before database save ===")
            print(f"   pre_structured_dialogue type: {type(pre_structured_dialogue)}, value: {pre_structured_dialogue}")
            logger.error(f"=== ERROR: No structured dialogue - stopping before database save ===")
            logger.error(f"  pre_structured_dialogue type: {type(pre_structured_dialogue)}, value: {pre_structured_dialogue}")
            return fail(
                request, 
                error="NO_STRUCTURED_DIALOGUE", 
                message="Azure Speech Service did not provide structured dialogue. Ensure speaker diarization is enabled."
            )
        
        # Map speakers from Azure Speech Service to Doctor/Patient
        print(f"🔵 Using pre-structured dialogue from Azure Speech Service ({len(pre_structured_dialogue)} turns)")
        logger.info(f"Using pre-structured dialogue from Azure Speech Service ({len(pre_structured_dialogue)} turns)")
        from ...application.utils.speaker_mapping import map_speakers_to_doctor_patient
        print("🔵 Mapping speakers to Doctor/Patient...")
        structured_dialogue = map_speakers_to_doctor_patient(
            pre_structured_dialogue,
            speaker_info=speaker_info,
            language=language
        )
        print(f"🔵 Mapped speakers to Doctor/Patient: {len(structured_dialogue)} turns")
        logger.info(f"Mapped speakers to Doctor/Patient: {len(structured_dialogue)} turns")
        
        if structured_dialogue:
            print(f"✅ Successfully mapped structured dialogue with {len(structured_dialogue)} turns")
            logger.info(f"✅ Successfully mapped structured dialogue with {len(structured_dialogue)} turns")
        else:
            print("❌ Failed to map structured dialogue from Azure Speech Service")
            logger.error("❌ Failed to map structured dialogue from Azure Speech Service")
            return fail(request, error="DIALOGUE_MAPPING_FAILED", message="Failed to map speakers to Doctor/Patient")

        # Save audio file to database
        print("🔵 === Starting database save operations ===")
        print(f"🔵 Audio data size: {len(audio_data)} bytes")
        print(f"🔵 Filename: {audio_file.filename}")
        print(f"🔵 Content type: {audio_file.content_type}")
        logger.info(f"=== Starting database save operations ===")
        logger.info(f"Audio data size: {len(audio_data)} bytes")
        logger.info(f"Filename: {audio_file.filename}")
        logger.info(f"Content type: {audio_file.content_type}")
        audio_file_record = None
        try:
            print("🔵 Calling audio_repo.create_audio_file()...")
            logger.info(f"Calling audio_repo.create_audio_file()...")
            audio_file_record = await audio_repo.create_audio_file(
                audio_data=audio_data,
                filename=audio_file.filename or "unknown_audio",
                content_type=audio_file.content_type or "audio/mpeg",
                audio_type="adhoc",
                duration_seconds=result.get("duration"),
            )
            print(f"✅ Audio file saved to database: {audio_file_record.audio_id}")
            print(f"   Audio ID: {audio_file_record.audio_id}")
            print(f"   Filename: {audio_file_record.filename}")
            print(f"   File size: {audio_file_record.file_size} bytes")
            logger.info(f"✅ Audio file saved to database: {audio_file_record.audio_id}")
            logger.info(f"   Audio ID: {audio_file_record.audio_id}")
            logger.info(f"   Filename: {audio_file_record.filename}")
            logger.info(f"   File size: {audio_file_record.file_size} bytes")
        except Exception as e:
            print(f"❌ Failed to save audio file to database: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Full traceback: {error_traceback}")
            logger.error(f"❌ Failed to save audio file to database: {e}")
            logger.error(f"Full traceback: {error_traceback}")
            # Don't fail the request, but log prominently so we can debug

        # Persist ad-hoc transcript with structured dialogue
        print("🔵 === Attempting to save adhoc transcript to database ===")
        print(f"🔵 Transcript length: {len(raw_transcript)} characters")
        print(f"🔵 Structured dialogue turns: {len(structured_dialogue)}")
        logger.info(f"=== Attempting to save adhoc transcript to database ===")
        logger.info(f"Transcript length: {len(raw_transcript)} characters")
        logger.info(f"Structured dialogue turns: {len(structured_dialogue)}")
        adhoc_id = None
        try:
            print("🔵 Creating AdhocTranscriptMongo document...")
            logger.info(f"Creating AdhocTranscriptMongo document...")
            doc = AdhocTranscriptMongo(
                transcript=raw_transcript,
                structured_dialogue=structured_dialogue,  # Save structured dialogue automatically
                language=result.get("language"),
                confidence=result.get("confidence"),
                duration=result.get("duration"),
                word_count=result.get("word_count"),
                model=result.get("model"),
                filename=audio_file.filename or None,
                audio_file_path=None,  # No longer using file paths
            )
            print("🔵 Calling doc.insert()...")
            logger.info(f"Calling doc.insert()...")
            await doc.insert()
            adhoc_id = str(doc.id)
            result["adhoc_id"] = adhoc_id
            print(f"✅ Adhoc transcript saved to database: {adhoc_id}")
            print(f"   MongoDB ID: {doc.id}")
            print(f"   Filename: {doc.filename}")
            logger.info(f"✅ Adhoc transcript saved to database: {adhoc_id}")
            logger.info(f"   MongoDB ID: {doc.id}")
            logger.info(f"   Filename: {doc.filename}")
            
            # Link audio file to adhoc transcript if we have both
            if audio_file_record:
                print(f"🔵 Linking audio file {audio_file_record.audio_id} to adhoc transcript {adhoc_id}...")
                logger.info(f"Linking audio file {audio_file_record.audio_id} to adhoc transcript {adhoc_id}...")
                link_success = await audio_repo.link_audio_to_adhoc(audio_file_record.audio_id, adhoc_id)
                if link_success:
                    print(f"✅ Linked audio file {audio_file_record.audio_id} to adhoc transcript {adhoc_id}")
                    logger.info(f"✅ Linked audio file {audio_file_record.audio_id} to adhoc transcript {adhoc_id}")
                else:
                    print(f"⚠️ Failed to link audio file {audio_file_record.audio_id} to adhoc transcript {adhoc_id}")
                    logger.warning(f"⚠️ Failed to link audio file {audio_file_record.audio_id} to adhoc transcript {adhoc_id}")
            else:
                print("⚠️ No audio_file_record to link to adhoc transcript")
                logger.warning(f"⚠️ No audio_file_record to link to adhoc transcript")
                
        except Exception as e:
            print(f"❌ Failed to persist adhoc transcript: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Full traceback: {error_traceback}")
            logger.error(f"❌ Failed to persist adhoc transcript: {e}")
            logger.error(f"Full traceback: {error_traceback}")
            # Don't fail the request, but log prominently so we can debug

        # Return response with both raw transcript and structured dialogue
        print(f"🔵 === Preparing response. Audio saved: {audio_file_record is not None}, Adhoc saved: {adhoc_id is not None} ===")
        logger.info(f"=== Preparing response. Audio saved: {audio_file_record is not None}, Adhoc saved: {adhoc_id is not None} ===")
        response_data = {
            **result,
            "filename": audio_file.filename or None,
            "structured_dialogue": structured_dialogue,  # Include structured dialogue in response
        }
        print("🔵 === Returning successful response ===")
        logger.info(f"=== Returning successful response ===")
        return ok(request, data=AdhocTranscriptionResponse(**response_data))
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"=== UNEXPECTED ERROR in transcribe_audio_adhoc ===")
        logger.error(f"Error: {e}")
        logger.error(f"Full traceback: {error_traceback}")
        raise
    finally:
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                print(f"🔵 Cleaning up temp file: {temp_file_path}")
                os.unlink(temp_file_path)
        except Exception:
            pass


class StructureTextRequest(BaseModel):
    transcript: str
    model: Optional[str] = None
    adhoc_id: Optional[str] = None


@router.post(
    "/structure",
    response_model=ApiResponse[Dict[str, List[Dict[str, str]]]],
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def structure_transcript_text(request: Request, payload: StructureTextRequest):
    logger.info(f"Structure endpoint called with adhoc_id: {payload.adhoc_id}, transcript length: {len(payload.transcript) if payload.transcript else 0}")
    
    if not payload.transcript or not payload.transcript.strip():
        return fail(request, error="EMPTY_TRANSCRIPT", message="Transcript is empty")

    settings = get_settings()
    model = payload.model or settings.azure_openai.deployment_name

    logger.info(f"Calling structure_dialogue_from_text with deployment: {model}")
    dialogue = await structure_dialogue_from_text(
        payload.transcript, 
        model=model,
        azure_endpoint=settings.azure_openai.endpoint,
        azure_api_key=settings.azure_openai.api_key
    )
    logger.info(f"Structure dialogue result: {type(dialogue)}, length: {len(dialogue) if isinstance(dialogue, list) else 'N/A'}")

    # Normalize dialogue to a list (empty list if None)
    normalized_dialogue: List[Dict[str, str]] = dialogue if isinstance(dialogue, list) else []

    # Optionally persist structured dialogue back to adhoc record
    if payload.adhoc_id:
        try:
            logger.info(f"Attempting to save structured dialogue to adhoc_id: {payload.adhoc_id}")
            oid = PydanticObjectId(payload.adhoc_id)
            doc = await AdhocTranscriptMongo.get(oid)  # type: ignore
            if doc is not None:
                logger.info(f"Found adhoc document, updating structured_dialogue with {len(normalized_dialogue)} turns")
                doc.structured_dialogue = normalized_dialogue
                await doc.save()
                logger.info("Successfully saved structured dialogue to database")
            else:
                logger.warning(f"Adhoc document not found for id: {payload.adhoc_id}")
        except Exception as e:
            logger.error(f"Failed to persist structured dialogue: {e}")
            # Swallow persistence errors to keep endpoint responsive
            pass

    logger.info(f"Returning dialogue with {len(normalized_dialogue)} turns")
    return ok(request, data={"dialogue": normalized_dialogue})


class ActionPlanRequest(BaseModel):
    adhoc_id: str


class ActionPlanResponse(BaseModel):
    adhoc_id: str
    status: str
    message: str


@router.post(
    "/adhoc/action-plan",
    response_model=ApiResponse[ActionPlanResponse],
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Adhoc transcript not found"},
        422: {"model": ErrorResponse, "description": "No transcript available"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def generate_action_plan(
    http_request: Request,
    payload: ActionPlanRequest,
    action_plan_service: ActionPlanServiceDep,
):
    """
    Generate Action and Plan for an adhoc transcript.
    This endpoint queues the action plan generation and returns immediately.
    """
    logger.info(f"Action plan generation request for adhoc_id: {payload.adhoc_id}")
    
    # Validate adhoc_id format (MongoDB ObjectId is 24 hex characters)
    if not payload.adhoc_id or len(payload.adhoc_id) != 24 or not all(c in '0123456789abcdefABCDEF' for c in payload.adhoc_id):
        logger.warning(f"Invalid adhoc_id format: {payload.adhoc_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INVALID_ADHOC_ID", "message": f"Invalid adhoc_id format: {payload.adhoc_id}"}
        )
    
    try:
        # Find the adhoc transcript
        try:
            oid = PydanticObjectId(payload.adhoc_id)
        except Exception as e:
            logger.error(f"Invalid ObjectId format: {payload.adhoc_id}, error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "INVALID_ADHOC_ID", "message": f"Invalid adhoc_id format: {payload.adhoc_id}"}
            )
        
        adhoc_doc = await AdhocTranscriptMongo.get(oid)  # Use .get() for better error handling
        
        if not adhoc_doc:
            return fail(http_request, error="ADHOC_NOT_FOUND", message=f"Adhoc transcript {payload.adhoc_id} not found")
        
        # Check if transcript exists
        if not adhoc_doc.transcript:
            return fail(http_request, error="NO_TRANSCRIPT", message="No transcript available for action plan generation")
        
        # Check if action plan is already being processed or completed
        if adhoc_doc.action_plan_status in ["processing", "completed"]:
            return ok(http_request, data=ActionPlanResponse(
                adhoc_id=payload.adhoc_id,
                status=adhoc_doc.action_plan_status,
                message="Action plan already processed or in progress"
            ))
        
        # Start action plan generation
        adhoc_doc.action_plan_status = "processing"
        adhoc_doc.action_plan_started_at = datetime.utcnow()
        await adhoc_doc.save()
        
        # Queue background processing
        import asyncio
        asyncio.create_task(_generate_action_plan_background(
            payload.adhoc_id, 
            adhoc_doc.transcript, 
            adhoc_doc.structured_dialogue,
            adhoc_doc.language or "en",
            action_plan_service
        ))
        
        logger.info(f"Action plan generation queued for adhoc_id: {payload.adhoc_id}")
        
        return ok(http_request, data=ActionPlanResponse(
            adhoc_id=payload.adhoc_id,
            status="processing",
            message="Action plan generation started"
        ))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting action plan generation: {e}")
        return fail(http_request, error="INTERNAL_ERROR", message=str(e))


async def _generate_action_plan_background(
    adhoc_id: str,
    transcript: str,
    structured_dialogue: list[dict],
    language: str,
    action_plan_service: ActionPlanServiceDep
):
    """Background task to generate action plan."""
    try:
        logger.info(f"Starting background action plan generation for adhoc_id: {adhoc_id}")
        
        # Generate action plan
        action_plan_data = await action_plan_service.generate_action_plan(
            transcript=transcript,
            structured_dialogue=structured_dialogue,
            language=language
        )
        
        # Update the adhoc transcript with the generated action plan
        try:
            oid = PydanticObjectId(adhoc_id)
            adhoc_doc = await AdhocTranscriptMongo.get(oid)  # Use .get() for better error handling
        except Exception as e:
            logger.error(f"Invalid ObjectId format or not found: {adhoc_id}, error: {e}")
            adhoc_doc = None
        
        if adhoc_doc:
            adhoc_doc.action_plan = action_plan_data
            adhoc_doc.action_plan_status = "completed"
            adhoc_doc.action_plan_completed_at = datetime.utcnow()
            await adhoc_doc.save()
            
            logger.info(f"Action plan generation completed for adhoc_id: {adhoc_id}")
        else:
            logger.error(f"Adhoc document not found during action plan completion: {adhoc_id}")
            
    except Exception as e:
        logger.error(f"Error in background action plan generation: {e}")
        
        # Update status to failed
        try:
            oid = PydanticObjectId(adhoc_id)
            adhoc_doc = await AdhocTranscriptMongo.get(oid)  # Use .get() for better error handling
            if adhoc_doc:
                adhoc_doc.action_plan_status = "failed"
                adhoc_doc.action_plan_error_message = str(e)
                adhoc_doc.action_plan_completed_at = datetime.utcnow()
                await adhoc_doc.save()
        except Exception as update_error:
            logger.error(f"Failed to update action plan status to failed: {update_error}")


@router.get(
    "/adhoc/{adhoc_id}/action-plan/status",
    response_model=ApiResponse[Dict],
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Adhoc transcript not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_action_plan_status(request: Request, adhoc_id: str):
    """
    Get the status of action plan generation for an adhoc transcript.
    """
    logger.info(f"Getting action plan status for adhoc_id: {adhoc_id}")
    try:
        # Validate adhoc_id format (MongoDB ObjectId is 24 hex characters)
        if not adhoc_id or len(adhoc_id) != 24 or not all(c in '0123456789abcdefABCDEF' for c in adhoc_id):
            logger.warning(f"Invalid adhoc_id format: {adhoc_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "INVALID_ADHOC_ID", "message": f"Invalid adhoc_id format: {adhoc_id}"}
            )
        
        try:
            oid = PydanticObjectId(adhoc_id)
        except Exception as e:
            logger.error(f"Invalid ObjectId format: {adhoc_id}, error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "INVALID_ADHOC_ID", "message": f"Invalid adhoc_id format: {adhoc_id}"}
            )
        
        adhoc_doc = await AdhocTranscriptMongo.get(oid)  # Use .get() for better error handling
        
        if not adhoc_doc:
            return fail(request, error="ADHOC_NOT_FOUND", message=f"Adhoc transcript {adhoc_id} not found")
        
        return ok(request, data={
            "adhoc_id": adhoc_id,
            "status": adhoc_doc.action_plan_status,
            "started_at": adhoc_doc.action_plan_started_at.isoformat() if adhoc_doc.action_plan_started_at else None,
            "completed_at": adhoc_doc.action_plan_completed_at.isoformat() if adhoc_doc.action_plan_completed_at else None,
            "error_message": adhoc_doc.action_plan_error_message,
            "has_action_plan": adhoc_doc.action_plan is not None
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting action plan status: {e}")
        return fail(request, error="INTERNAL_ERROR", message=str(e))


@router.get(
    "/adhoc/{adhoc_id}/action-plan",
    response_model=ApiResponse[Dict],
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Adhoc transcript or action plan not found"},
        422: {"model": ErrorResponse, "description": "Action plan not ready"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_action_plan(request: Request, adhoc_id: str):
    """
    Get the generated action plan for an adhoc transcript.
    """
    logger.info(f"Getting action plan for adhoc_id: {adhoc_id}")
    try:
        # Validate adhoc_id format (MongoDB ObjectId is 24 hex characters)
        if not adhoc_id or len(adhoc_id) != 24 or not all(c in '0123456789abcdefABCDEF' for c in adhoc_id):
            logger.warning(f"Invalid adhoc_id format: {adhoc_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "INVALID_ADHOC_ID", "message": f"Invalid adhoc_id format: {adhoc_id}"}
            )
        
        try:
            oid = PydanticObjectId(adhoc_id)
        except Exception as e:
            logger.error(f"Invalid ObjectId format: {adhoc_id}, error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "INVALID_ADHOC_ID", "message": f"Invalid adhoc_id format: {adhoc_id}"}
            )
        
        adhoc_doc = await AdhocTranscriptMongo.get(oid)  # Use .get() for better error handling
        
        if not adhoc_doc:
            return fail(request, error="ADHOC_NOT_FOUND", message=f"Adhoc transcript {adhoc_id} not found")
        
        if adhoc_doc.action_plan_status != "completed":
            return fail(request, error="NOT_READY", message=f"Action plan status: {adhoc_doc.action_plan_status}")
        
        if not adhoc_doc.action_plan:
            return fail(request, error="NO_ACTION_PLAN", message="No action plan available")
        
        return ok(request, data={
            "adhoc_id": adhoc_id,
            "action_plan": adhoc_doc.action_plan,
            "generated_at": adhoc_doc.action_plan_completed_at.isoformat() if adhoc_doc.action_plan_completed_at else None
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting action plan: {e}")
        return fail(request, error="INTERNAL_ERROR", message=str(e))


