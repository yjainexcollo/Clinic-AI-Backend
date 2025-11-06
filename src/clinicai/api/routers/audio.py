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

router = APIRouter(prefix="/audio")
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


class EnhancedAudioFileResponse(BaseModel):
    """Enhanced response model for audio file with URLs."""
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
    download_url: str
    stream_url: str

    @classmethod
    def from_audio_file(cls, audio_file: AudioFileMongo, base_url: str = "", encoded_patient_id: Optional[str] = None) -> "EnhancedAudioFileResponse":
        """Create enhanced response from AudioFileMongo model."""
        return cls(
            audio_id=audio_file.audio_id,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
            file_size=audio_file.file_size,
            duration_seconds=audio_file.duration_seconds,
            patient_id=encoded_patient_id if encoded_patient_id else audio_file.patient_id,
            visit_id=audio_file.visit_id,
            adhoc_id=audio_file.adhoc_id,
            audio_type=audio_file.audio_type,
            created_at=audio_file.created_at.isoformat(),
            updated_at=audio_file.updated_at.isoformat(),
            download_url=f"{base_url}/audio/{audio_file.audio_id}/download",
            stream_url=f"{base_url}/audio/{audio_file.audio_id}/stream",
        )


class EnhancedAudioListResponse(BaseModel):
    """Enhanced response model for audio file list with pagination."""
    audio_files: List[EnhancedAudioFileResponse]
    pagination: Dict[str, Any]


@router.get(
    "/",
    response_model=ApiResponse[AudioListResponse],
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
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
    include_in_schema=False,
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
        
        logger.info(f"Retrieved {len(dialogue_data) if dialogue_data else 0} dialogue records from repository")
        
        # Get total count
        total_count = await audio_repo.get_audio_count(
            patient_id=patient_id,
            audio_type=audio_type,
        )
        
        logger.info(f"Total audio count: {total_count}")
        
        # Convert to response models with validation
        dialogue_responses = []
        for data in dialogue_data:
            # Ensure structured_dialogue is always a list (never None)
            if "structured_dialogue" not in data or data["structured_dialogue"] is None:
                data["structured_dialogue"] = []
            elif not isinstance(data["structured_dialogue"], list):
                logger.warning(f"structured_dialogue is not a list for {data.get('audio_id')}, converting to empty list")
                data["structured_dialogue"] = []
            
            try:
                dialogue_responses.append(AudioDialogueResponse(**data))
            except Exception as e:
                logger.error(f"Failed to create AudioDialogueResponse from data: {e}, data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                continue
        
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
    "/files",
    response_model=ApiResponse[EnhancedAudioListResponse],
    status_code=status.HTTP_200_OK,
    tags=["Audio Management"],
    summary="Get all original audio files with sorting",
    responses={
        200: {"description": "Audio files retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_all_audio_files(
    request: Request,
    audio_repo: AudioRepositoryDep,
    limit: int = Query(100, ge=1, le=200, description="Number of files to return"),
    offset: int = Query(0, ge=0, description="Number of files to skip"),
    sort_by: str = Query("created_at", description="Sort field: created_at, filename, file_size, duration_seconds"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
):
    """
    Get all original audio files from the database with sorting and pagination.
    
    This endpoint:
    1. Retrieves ALL audio files in the database with metadata
    2. Supports sorting by created_at, filename, file_size, or duration_seconds
    3. Returns paginated results with download and stream URLs
    
    Query Parameters:
    - limit: Number of results per page (default: 100, max: 200)
    - offset: Number of results to skip (default: 0)
    - sort_by: Sort field - "created_at", "filename", "file_size", or "duration_seconds" (default: "created_at")
    - sort_order: Sort direction - "asc" or "desc" (default: "desc")
    """
    try:
        logger.info(
            f"Listing all audio files: sort_by={sort_by}, sort_order={sort_order}, "
            f"limit={limit}, offset={offset}"
        )
        
        # Validate sort parameters
        valid_sort_fields = ["created_at", "filename", "file_size", "duration_seconds", "updated_at"]
        if sort_by not in valid_sort_fields:
            sort_by = "created_at"
        if sort_order not in ["asc", "desc"]:
            sort_order = "desc"
        
        # Get all audio files with enhanced sorting (no filters)
        audio_files = await audio_repo.list_audio_files(
            patient_id=None,
            visit_id=None,
            audio_type=None,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        
        # Get total count for pagination (all files, no filters)
        total_count = await audio_repo.get_audio_count()
        
        # Get base URL for generating download/stream URLs
        base_url = str(request.base_url).rstrip("/")
        
        # Encode patient IDs if present
        from ...core.utils.crypto import encode_patient_id
        
        # Convert to enhanced response models
        enhanced_files = []
        for audio_file in audio_files:
            # Encode patient_id if present
            encoded_patient_id = None
            if audio_file.patient_id:
                try:
                    encoded_patient_id = encode_patient_id(audio_file.patient_id)
                except Exception:
                    encoded_patient_id = audio_file.patient_id  # Fallback to original
            
            # Create enhanced response with encoded patient_id
            enhanced_file = EnhancedAudioFileResponse.from_audio_file(
                audio_file, 
                base_url, 
                encoded_patient_id=encoded_patient_id
            )
            enhanced_files.append(enhanced_file)
        
        # Calculate pagination info
        has_more = len(enhanced_files) == limit and (offset + limit) < total_count
        
        return ok(
            request,
            data=EnhancedAudioListResponse(
                audio_files=enhanced_files,
                pagination={
                    "limit": limit,
                    "offset": offset,
                    "count": len(enhanced_files),
                    "total": total_count,
                    "has_more": has_more
                }
            ),
            message="Audio files retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to list all audio files: {e}", exc_info=True)
        return fail(request, error="LIST_AUDIO_FAILED", message="Failed to list audio files", details=str(e))


@router.get(
    "/{audio_id}",
    response_model=ApiResponse[AudioFileResponse],
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
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
    include_in_schema=False,
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
    include_in_schema=False,
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
    include_in_schema=False,
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
    include_in_schema=False,
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
