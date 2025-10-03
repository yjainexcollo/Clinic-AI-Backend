"""
Audio repository for MongoDB operations.
"""

from typing import List, Optional
from datetime import datetime
import uuid
import logging

from beanie import PydanticObjectId
from ..models.patient_m import AudioFileMongo

logger = logging.getLogger("clinicai")


class AudioRepository:
    """Repository for audio file operations."""

    async def create_audio_file(
        self,
        audio_data: bytes,
        filename: str,
        content_type: str,
        patient_id: Optional[str] = None,
        visit_id: Optional[str] = None,
        adhoc_id: Optional[str] = None,
        audio_type: str = "adhoc",
        duration_seconds: Optional[float] = None,
    ) -> AudioFileMongo:
        """Create a new audio file record in the database."""
        try:
            audio_id = str(uuid.uuid4())
            
            audio_file = AudioFileMongo(
                audio_id=audio_id,
                filename=filename,
                content_type=content_type,
                audio_data=audio_data,
                file_size=len(audio_data),
                duration_seconds=duration_seconds,
                patient_id=patient_id,
                visit_id=visit_id,
                adhoc_id=adhoc_id,
                audio_type=audio_type,
            )
            
            await audio_file.insert()
            logger.info(f"Created audio file: {audio_id} ({filename})")
            return audio_file
            
        except Exception as e:
            logger.error(f"Failed to create audio file: {e}")
            raise

    async def get_audio_file_by_id(self, audio_id: str) -> Optional[AudioFileMongo]:
        """Get audio file by audio_id."""
        try:
            return await AudioFileMongo.find_one(AudioFileMongo.audio_id == audio_id)
        except Exception as e:
            logger.error(f"Failed to get audio file by ID {audio_id}: {e}")
            return None

    async def get_audio_file_by_mongo_id(self, mongo_id: str) -> Optional[AudioFileMongo]:
        """Get audio file by MongoDB ObjectId."""
        try:
            oid = PydanticObjectId(mongo_id)
            return await AudioFileMongo.get(oid)
        except Exception as e:
            logger.error(f"Failed to get audio file by MongoDB ID {mongo_id}: {e}")
            return None

    async def list_audio_files(
        self,
        patient_id: Optional[str] = None,
        visit_id: Optional[str] = None,
        adhoc_id: Optional[str] = None,
        audio_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AudioFileMongo]:
        """List audio files with optional filtering."""
        try:
            query = {}
            
            if patient_id:
                query["patient_id"] = patient_id
            if visit_id:
                query["visit_id"] = visit_id
            if adhoc_id:
                query["adhoc_id"] = adhoc_id
            if audio_type:
                query["audio_type"] = audio_type
            
            # Get files without audio_data to reduce memory usage
            files = await AudioFileMongo.find(
                query,
                skip=offset,
                limit=limit,
                sort=[("created_at", -1)]  # Most recent first
            ).to_list()
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list audio files: {e}")
            return []

    async def get_audio_data(self, audio_id: str) -> Optional[bytes]:
        """Get audio file binary data by audio_id."""
        try:
            # Get the full audio file document
            audio_file = await AudioFileMongo.find_one(
                AudioFileMongo.audio_id == audio_id
            )
            return audio_file.audio_data if audio_file else None
        except Exception as e:
            logger.error(f"Failed to get audio data for ID {audio_id}: {e}")
            return None

    async def update_audio_metadata(
        self,
        audio_id: str,
        duration_seconds: Optional[float] = None,
        **kwargs
    ) -> Optional[AudioFileMongo]:
        """Update audio file metadata."""
        try:
            audio_file = await self.get_audio_file_by_id(audio_id)
            if not audio_file:
                return None
            
            if duration_seconds is not None:
                audio_file.duration_seconds = duration_seconds
            
            # Update any other provided fields
            for key, value in kwargs.items():
                if hasattr(audio_file, key):
                    setattr(audio_file, key, value)
            
            audio_file.updated_at = datetime.utcnow()
            await audio_file.save()
            
            logger.info(f"Updated audio file metadata: {audio_id}")
            return audio_file
            
        except Exception as e:
            logger.error(f"Failed to update audio file metadata {audio_id}: {e}")
            return None

    async def delete_audio_file(self, audio_id: str) -> bool:
        """Delete audio file by audio_id."""
        try:
            audio_file = await self.get_audio_file_by_id(audio_id)
            if not audio_file:
                return False
            
            await audio_file.delete()
            logger.info(f"Deleted audio file: {audio_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete audio file {audio_id}: {e}")
            return False

    async def get_audio_count(
        self,
        patient_id: Optional[str] = None,
        adhoc_id: Optional[str] = None,
        audio_type: Optional[str] = None,
    ) -> int:
        """Get count of audio files with optional filtering."""
        try:
            query = {}
            
            if patient_id:
                query["patient_id"] = patient_id
            if adhoc_id:
                query["adhoc_id"] = adhoc_id
            if audio_type:
                query["audio_type"] = audio_type
            
            count = await AudioFileMongo.find(query).count()
            return count
            
        except Exception as e:
            logger.error(f"Failed to get audio count: {e}")
            return 0

    async def link_audio_to_adhoc(self, audio_id: str, adhoc_id: str) -> bool:
        """Link audio file to adhoc transcript."""
        try:
            audio_file = await self.get_audio_file_by_id(audio_id)
            if not audio_file:
                return False
            
            audio_file.adhoc_id = adhoc_id
            audio_file.audio_type = "adhoc"
            audio_file.updated_at = datetime.utcnow()
            await audio_file.save()
            
            logger.info(f"Linked audio file {audio_id} to adhoc transcript {adhoc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to link audio to adhoc: {e}")
            return False

    async def link_audio_to_visit(self, audio_id: str, patient_id: str, visit_id: str) -> bool:
        """Link audio file to visit."""
        try:
            audio_file = await self.get_audio_file_by_id(audio_id)
            if not audio_file:
                return False
            
            audio_file.patient_id = patient_id
            audio_file.visit_id = visit_id
            audio_file.audio_type = "visit"
            audio_file.updated_at = datetime.utcnow()
            await audio_file.save()
            
            logger.info(f"Linked audio file {audio_id} to visit {visit_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to link audio to visit: {e}")
            return False
