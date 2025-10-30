"""
Audio management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Response, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import io

from ..deps import AudioRepositoryDep
from ...adapters.db.mongo.repositories.audio_repository import AudioRepository
from ...adapters.db.mongo.models.patient_m import AudioFileMongo
from ..schemas.common import ApiResponse, ErrorResponse
from ..utils.responses import ok, fail

router = APIRouter(prefix="/audio", tags=["audio"])
logger = logging.getLogger("clinicai")


class AudioFileResponse(BaseModel):
    """Response model for audio file metadata."""
    audio_id: str
    filename: str
    content_type: str
    file_size: int
    duration_seconds: Optional[float]
    patient_id: Optional[str]
    visit_id: Optional[str]
    adhoc_id: Optional[str]
    audio_type: str
    created_at: str
    updated_at: str

    @classmethod
    def from_audio_file(cls, audio_file: AudioFileMongo) -> "AudioFileResponse":
        """Create response from AudioFileMongo model."""
        return cls(
            audio_id=audio_file.audio_id,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
            file_size=audio_file.file_size,
            duration_seconds=audio_file.duration_seconds,
            patient_id=audio_file.patient_id,
            visit_id=audio_file.visit_id,
            adhoc_id=audio_file.adhoc_id,
            audio_type=audio_file.audio_type,
            created_at=audio_file.created_at.isoformat(),
            updated_at=audio_file.updated_at.isoformat(),
        )


class AudioListResponse(BaseModel):
    """Response model for audio file list."""
    files: List[AudioFileResponse]
    total_count: int
    limit: int
    offset: int


class AudioDialogueResponse(BaseModel):
    """Response model for audio dialogue data."""
    audio_id: str
    filename: str
    duration_seconds: Optional[float]
    patient_id: Optional[str]
    visit_id: Optional[str]
    adhoc_id: Optional[str]
    audio_type: str
    created_at: str
    structured_dialogue: Optional[List[Dict[str, str]]]


class AudioDialogueListResponse(BaseModel):
    """Response model for audio dialogue list."""
    dialogues: List[AudioDialogueResponse]
    total_count: int
    limit: int
    offset: int


@router.get(
    "/",
    response_model=ApiResponse[AudioListResponse],
    status_code=status.HTTP_200_OK,
)
async def list_audio_files(
    request: Request,
    audio_repo: AudioRepositoryDep,
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    visit_id: Optional[str] = Query(None, description="Filter by visit ID"),
    audio_type: Optional[str] = Query(None, description="Filter by audio type (adhoc, visit)"),
    limit: int = Query(50, ge=1, le=100, description="Number of files to return"),
    offset: int = Query(0, ge=0, description="Number of files to skip"),
):
    """List all audio files with optional filtering."""
    try:
        logger.info(f"Listing audio files: patient_id={patient_id}, visit_id={visit_id}, audio_type={audio_type}, limit={limit}, offset={offset}")
        
        # Get audio files
        audio_files = await audio_repo.list_audio_files(
            patient_id=patient_id,
            visit_id=visit_id,
            audio_type=audio_type,
            limit=limit,
            offset=offset,
        )
        
        # Get total count
        total_count = await audio_repo.get_audio_count(
            patient_id=patient_id,
            audio_type=audio_type,
        )
        
        # Convert to response models
        file_responses = [AudioFileResponse.from_audio_file(f) for f in audio_files]
        
        return ok(request, data=AudioListResponse(
            files=file_responses,
            total_count=total_count,
            limit=limit,
            offset=offset,
        ), message="Audio files listed")
        
    except Exception as e:
        logger.error(f"Failed to list audio files: {e}")
        return fail(request, error="LIST_AUDIO_FAILED", message="Failed to list audio files", details=str(e))


@router.get(
    "/dialogue",
    response_model=ApiResponse[AudioDialogueListResponse],
    status_code=status.HTTP_200_OK,
)
async def list_audio_dialogues(
    request: Request,
    audio_repo: AudioRepositoryDep,
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    visit_id: Optional[str] = Query(None, description="Filter by visit ID"),
    audio_type: Optional[str] = Query(None, description="Filter by audio type (adhoc, visit)"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date (ISO format)"),
    limit: int = Query(50, ge=1, le=100, description="Number of dialogues to return"),
    offset: int = Query(0, ge=0, description="Number of dialogues to skip"),
):
    """List structured dialogues for audio files instead of full metadata."""
    try:
        logger.info(f"Listing audio dialogues: patient_id={patient_id}, visit_id={visit_id}, audio_type={audio_type}, start_date={start_date}, end_date={end_date}, limit={limit}, offset={offset}")
        
        # Get dialogue data
        dialogue_data = await audio_repo.get_audio_dialogue_list(
            patient_id=patient_id,
            visit_id=visit_id,
            audio_type=audio_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        
        # Get total count
        total_count = await audio_repo.get_audio_count(
            patient_id=patient_id,
            audio_type=audio_type,
        )
        
        # Convert to response models
        dialogue_responses = [
            AudioDialogueResponse(**data) for data in dialogue_data
        ]
        
        return ok(request, data=AudioDialogueListResponse(
            dialogues=dialogue_responses,
            total_count=total_count,
            limit=limit,
            offset=offset,
        ), message="Audio dialogues listed")
        
    except Exception as e:
        logger.error(f"Failed to list audio dialogues: {e}")
        return fail(request, error="LIST_DIALOGUE_FAILED", message="Failed to list audio dialogues", details=str(e))


@router.get(
    "/{audio_id}",
    response_model=ApiResponse[AudioFileResponse],
    status_code=status.HTTP_200_OK,
)
async def get_audio_metadata(
    audio_id: str,
    audio_repo: AudioRepositoryDep,
    request: Request,
):
    """Get audio file metadata by ID."""
    try:
        logger.info(f"Getting audio metadata: {audio_id}")
        
        audio_file = await audio_repo.get_audio_file_by_id(audio_id)
        if not audio_file:
            raise HTTPException(
                status_code=404,
                detail={"error": "AUDIO_NOT_FOUND", "message": "Audio file not found", "details": {"audio_id": audio_id}},
            )
        
        return ok(request, data=AudioFileResponse.from_audio_file(audio_file), message="Audio metadata retrieved")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get audio metadata {audio_id}: {e}")
        return fail(request, error="GET_AUDIO_FAILED", message="Failed to get audio file", details=str(e))


@router.get(
    "/{audio_id}/download",
    status_code=status.HTTP_200_OK,
)
async def download_audio_file(
    audio_id: str,
    audio_repo: AudioRepositoryDep,
    request: Request,
):
    """Download audio file by ID."""
    try:
        logger.info(f"Downloading audio file: {audio_id}")
        
        # Get audio file metadata
        audio_file = await audio_repo.get_audio_file_by_id(audio_id)
        if not audio_file:
            raise HTTPException(
                status_code=404,
                detail={"error": "AUDIO_NOT_FOUND", "message": "Audio file not found", "details": {"audio_id": audio_id}},
            )
        
        # Get audio data
        audio_data = await audio_repo.get_audio_data(audio_id)
        if not audio_data:
            raise HTTPException(
                status_code=404,
                detail={"error": "AUDIO_DATA_NOT_FOUND", "message": "Audio data not found", "details": {"audio_id": audio_id}},
            )
        
        # Create streaming response
        audio_stream = io.BytesIO(audio_data)
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=audio_file.content_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{audio_file.filename}\"",
                "Content-Length": str(audio_file.file_size),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download audio file {audio_id}: {e}")
        return fail(request, error="DOWNLOAD_AUDIO_FAILED", message="Failed to download audio file", details=str(e))


@router.get(
    "/{audio_id}/stream",
    status_code=status.HTTP_200_OK,
)
async def stream_audio_file(
    audio_id: str,
    audio_repo: AudioRepositoryDep,
    request: Request,
):
    """Stream audio file for playback."""
    try:
        logger.info(f"Streaming audio file: {audio_id}")
        
        # Get audio file metadata
        audio_file = await audio_repo.get_audio_file_by_id(audio_id)
        if not audio_file:
            raise HTTPException(
                status_code=404,
                detail={"error": "AUDIO_NOT_FOUND", "message": "Audio file not found", "details": {"audio_id": audio_id}},
            )
        
        # Get audio data
        audio_data = await audio_repo.get_audio_data(audio_id)
        if not audio_data:
            raise HTTPException(
                status_code=404,
                detail={"error": "AUDIO_DATA_NOT_FOUND", "message": "Audio data not found", "details": {"audio_id": audio_id}},
            )
        
        # Create streaming response for audio playback
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=audio_file.content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(audio_file.file_size),
                "Cache-Control": "public, max-age=3600",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stream audio file {audio_id}: {e}")
        return fail(request, error="STREAM_AUDIO_FAILED", message="Failed to stream audio file", details=str(e))


@router.delete(
    "/{audio_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_audio_file(
    audio_id: str,
    audio_repo: AudioRepositoryDep,
    request: Request,
):
    """Delete audio file by ID."""
    try:
        logger.info(f"Deleting audio file: {audio_id}")
        
        success = await audio_repo.delete_audio_file(audio_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={"error": "AUDIO_NOT_FOUND", "message": "Audio file not found", "details": {"audio_id": audio_id}},
            )
        
        return Response(status_code=204)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete audio file {audio_id}: {e}")
        return fail(request, error="DELETE_AUDIO_FAILED", message="Failed to delete audio file", details=str(e))


@router.get(
    "/stats/summary",
    status_code=status.HTTP_200_OK,
)
async def get_audio_stats(
    audio_repo: AudioRepositoryDep,
    request: Request,
):
    """Get audio storage statistics."""
    try:
        logger.info("Getting audio storage statistics")
        
        # Get counts by type
        adhoc_count = await audio_repo.get_audio_count(audio_type="adhoc")
        visit_count = await audio_repo.get_audio_count(audio_type="visit")
        total_count = await audio_repo.get_audio_count()
        
        return ok(request, data={
            "total_files": total_count,
            "adhoc_files": adhoc_count,
            "visit_files": visit_count,
            "other_files": total_count - adhoc_count - visit_count,
        }, message="Audio stats retrieved")
        
    except Exception as e:
        logger.error(f"Failed to get audio stats: {e}")
        return fail(request, error="GET_STATS_FAILED", message="Failed to get audio statistics", details=str(e))
