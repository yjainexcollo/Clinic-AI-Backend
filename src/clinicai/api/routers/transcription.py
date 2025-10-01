from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import tempfile
import logging

from ..deps import TranscriptionServiceDep
from ...adapters.db.mongo.models.patient_m import AdhocTranscriptMongo
from beanie import PydanticObjectId
from ...core.config import get_settings
from ...application.utils.structure_dialogue import structure_dialogue_from_text


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
    response_model=AdhocTranscriptionResponse,
    status_code=status.HTTP_200_OK,
)
async def transcribe_audio_adhoc(
    transcription_service: TranscriptionServiceDep,
    audio_file: UploadFile = File(...),
    language: str = Form("en"),
):
    logger.info(f"Adhoc transcription request: filename={audio_file.filename}, content_type={audio_file.content_type}, language={language}")
    
    if not audio_file.filename:
        logger.error("No audio file filename provided")
        raise HTTPException(
            status_code=400,
            detail={"error": "NO_AUDIO_FILE", "message": "No audio file provided", "details": {}},
        )

    content_type = audio_file.content_type or ""
    is_audio_like = content_type.startswith("audio/")
    is_supported_video = content_type in ("video/mpeg", "video/webm", "video/mp4")
    is_generic_stream = content_type in ("application/octet-stream",)
    
    logger.info(f"Content type validation: audio_like={is_audio_like}, supported_video={is_supported_video}, generic_stream={is_generic_stream}")
    
    if not (is_audio_like or is_supported_video or is_generic_stream):
        logger.error(f"Invalid file type: {content_type}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_FILE_TYPE",
                "message": "File must be an audio file",
                "details": {"content_type": audio_file.content_type},
            },
        )

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

        # Validate and transcribe
        try:
            logger.info(f"Validating audio file: {temp_file_path}")
            meta = await transcription_service.validate_audio_file(temp_file_path)
            logger.info(f"Validation result: {meta}")
            if not meta.get("is_valid"):
                logger.error(f"Audio validation failed: {meta}")
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "INVALID_AUDIO",
                        "message": meta.get("error") or "Invalid audio",
                        "details": meta,
                    },
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Audio validation error: {e}")

        result = await transcription_service.transcribe_audio(temp_file_path, language=language)

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
            )
            await doc.insert()
            result["adhoc_id"] = str(doc.id)
        except Exception:
            pass

        return AdhocTranscriptionResponse(**{**result, "filename": audio_file.filename or None})
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
    status_code=status.HTTP_200_OK,
)
async def structure_transcript_text(payload: StructureTextRequest) -> Dict[str, List[Dict[str, str]]]:
    if not payload.transcript or not payload.transcript.strip():
        raise HTTPException(
            status_code=422,
            detail={"error": "EMPTY_TRANSCRIPT", "message": "Transcript is empty", "details": {}},
        )

    settings = get_settings()
    model = payload.model or settings.openai.model
    api_key = settings.openai.api_key

    dialogue = await structure_dialogue_from_text(payload.transcript, model=model, api_key=api_key)

    # Normalize dialogue to a list (empty list if None)
    normalized_dialogue: List[Dict[str, str]] = dialogue if isinstance(dialogue, list) else []

    # Optionally persist structured dialogue back to adhoc record
    if payload.adhoc_id:
        try:
            oid = PydanticObjectId(payload.adhoc_id)
            doc = await AdhocTranscriptMongo.get(oid)  # type: ignore
            if doc is not None:
                doc.structured_dialogue = normalized_dialogue
                await doc.save()
        except Exception:
            # Swallow persistence errors to keep endpoint responsive
            pass

    return {"dialogue": normalized_dialogue}


