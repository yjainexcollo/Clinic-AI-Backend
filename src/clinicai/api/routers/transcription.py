from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import tempfile
import logging
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


class AdhocTranscriptionResponse(BaseModel):
    transcript: str
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
    audio_file: UploadFile = File(...),
    language: str = Form("en"),
):
    logger.info(f"Adhoc transcription request: filename={audio_file.filename}, content_type={audio_file.content_type}, language={language}")
    
    if not audio_file.filename:
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
        audio_data = await audio_file.read()
        
        # Reset file pointer for transcription service
        await audio_file.seek(0)
        
        # Create temp file for transcription
        suffix = f".{(audio_file.filename or 'audio').split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_data)

        # Validate and transcribe
        try:
            logger.info(f"Validating audio file: {temp_file_path}")
            meta = await transcription_service.validate_audio_file(temp_file_path)
            logger.info(f"Validation result: {meta}")
            if not meta.get("is_valid"):
                logger.error(f"Audio validation failed: {meta}")
                return fail(request, error="INVALID_AUDIO", message=meta.get("error") or "Invalid audio")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Audio validation error: {e}")

        result = await transcription_service.transcribe_audio(temp_file_path, language=language)

        # Save audio file to database
        audio_file_record = None
        try:
            logger.info(f"Attempting to save audio file to database. Audio data size: {len(audio_data)} bytes")
            audio_file_record = await audio_repo.create_audio_file(
                audio_data=audio_data,
                filename=audio_file.filename or "unknown_audio",
                content_type=audio_file.content_type or "audio/mpeg",
                audio_type="adhoc",
                duration_seconds=result.get("duration"),
            )
            logger.info(f"Audio file saved to database: {audio_file_record.audio_id}")
        except Exception as e:
            logger.error(f"Failed to save audio file to database: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Continue without failing the transcription

        # Persist ad-hoc transcript
        try:
            doc = AdhocTranscriptMongo(
                transcript=result.get("transcript") or "",
                structured_dialogue=None,
                language=result.get("language"),
                confidence=result.get("confidence"),
                duration=result.get("duration"),
                word_count=result.get("word_count"),
                model=result.get("model"),
                filename=audio_file.filename or None,
                audio_file_path=None,  # No longer using file paths
            )
            await doc.insert()
            result["adhoc_id"] = str(doc.id)
            
            # Link audio file to adhoc transcript if we have both
            if audio_file_record:
                await audio_repo.link_audio_to_adhoc(audio_file_record.audio_id, str(doc.id))
                logger.info(f"Linked audio file {audio_file_record.audio_id} to adhoc transcript {doc.id}")
                
        except Exception as e:
            logger.error(f"Failed to persist adhoc transcript: {e}")
            pass

        return ok(request, data=AdhocTranscriptionResponse(**{**result, "filename": audio_file.filename or None}))
    finally:
        try:
            if temp_file_path and os.path.exists(temp_file_path):
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
)
async def structure_transcript_text(request: Request, payload: StructureTextRequest):
    logger.info(f"Structure endpoint called with adhoc_id: {payload.adhoc_id}, transcript length: {len(payload.transcript) if payload.transcript else 0}")
    
    if not payload.transcript or not payload.transcript.strip():
        return fail(request, error="EMPTY_TRANSCRIPT", message="Transcript is empty")

    settings = get_settings()
    model = payload.model or settings.openai.model
    api_key = settings.openai.api_key

    logger.info(f"Calling structure_dialogue_from_text with model: {model}")
    dialogue = await structure_dialogue_from_text(payload.transcript, model=model, api_key=api_key)
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
)
async def generate_action_plan(
    request: Request,
    action_plan_service: ActionPlanServiceDep,
):
    """
    Generate Action and Plan for an adhoc transcript.
    This endpoint queues the action plan generation and returns immediately.
    """
    logger.info(f"Action plan generation request for adhoc_id: {request.adhoc_id}")
    
    try:
        # Find the adhoc transcript
        adhoc_doc = await AdhocTranscriptMongo.find_one(
            AdhocTranscriptMongo.id == PydanticObjectId(request.adhoc_id)
        )
        
        if not adhoc_doc:
            return fail(request, error="ADHOC_NOT_FOUND", message=f"Adhoc transcript {request.adhoc_id} not found")
        
        # Check if transcript exists
        if not adhoc_doc.transcript:
            return fail(request, error="NO_TRANSCRIPT", message="No transcript available for action plan generation")
        
        # Check if action plan is already being processed or completed
        if adhoc_doc.action_plan_status in ["processing", "completed"]:
            return ok(request, data=ActionPlanResponse(
                adhoc_id=request.adhoc_id,
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
            request.adhoc_id, 
            adhoc_doc.transcript, 
            adhoc_doc.structured_dialogue,
            adhoc_doc.language or "en",
            action_plan_service
        ))
        
        logger.info(f"Action plan generation queued for adhoc_id: {request.adhoc_id}")
        
        return ok(request, data=ActionPlanResponse(
            adhoc_id=request.adhoc_id,
            status="processing",
            message="Action plan generation started"
        ))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting action plan generation: {e}")
        return fail(request, error="INTERNAL_ERROR", message=str(e))


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
        adhoc_doc = await AdhocTranscriptMongo.find_one(
            AdhocTranscriptMongo.id == PydanticObjectId(adhoc_id)
        )
        
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
            adhoc_doc = await AdhocTranscriptMongo.find_one(
                AdhocTranscriptMongo.id == PydanticObjectId(adhoc_id)
            )
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
)
async def get_action_plan_status(request: Request, adhoc_id: str):
    """
    Get the status of action plan generation for an adhoc transcript.
    """
    try:
        adhoc_doc = await AdhocTranscriptMongo.find_one(
            AdhocTranscriptMongo.id == PydanticObjectId(adhoc_id)
        )
        
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
)
async def get_action_plan(request: Request, adhoc_id: str):
    """
    Get the generated action plan for an adhoc transcript.
    """
    try:
        adhoc_doc = await AdhocTranscriptMongo.find_one(
            AdhocTranscriptMongo.id == PydanticObjectId(adhoc_id)
        )
        
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


