"""
Audio repository for MongoDB operations with Azure Blob Storage integration.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from beanie import PydanticObjectId

from ....storage.azure_blob_service import get_azure_blob_service
from ..models.patient_m import AudioFileMongo
from .blob_file_repository import BlobFileRepository

logger = logging.getLogger("clinicai")


class AudioRepository:
    """Repository for audio file operations with Azure Blob Storage."""

    def __init__(self):
        self.blob_service = get_azure_blob_service()
        self.blob_repo = BlobFileRepository()

    async def create_audio_file_from_path(
        self,
        file_path: str,
        filename: str,
        content_type: str,
        patient_id: Optional[str] = None,
        visit_id: Optional[str] = None,
        adhoc_id: Optional[str] = None,
        audio_type: str = "adhoc",
        duration_seconds: Optional[float] = None,
    ) -> AudioFileMongo:
        """Create a new audio file record with blob storage from file path (streaming, no memory)."""
        import os

        try:
            print(f"🔵 [AudioRepo] Step 1: Creating audio file record for {filename}")
            audio_id = str(uuid.uuid4())
            file_size = os.path.getsize(file_path)
            print(f"🔵 [AudioRepo] Step 2: File size = {file_size} bytes")

            # Upload to blob storage by streaming from file (no memory loading)
            print(f"🔵 [AudioRepo] Step 3: Starting blob upload...")
            blob_info = await self.blob_service.upload_file_from_path(
                file_path=file_path,
                filename=filename,
                content_type=content_type,
                file_type="audio",
                patient_id=patient_id,
                visit_id=visit_id,
                adhoc_id=adhoc_id,
                audio_type=audio_type,
                duration_seconds=duration_seconds,
            )
            print(f"🔵 [AudioRepo] Step 4: Blob upload completed, creating blob reference...")

            # Create blob reference
            blob_reference = await self.blob_repo.create_blob_reference(
                blob_path=blob_info["blob_path"],
                container_name=blob_info["container_name"],
                original_filename=filename,
                content_type=content_type,
                file_size=file_size,
                blob_url=blob_info["blob_url"],
                file_type="audio",
                category=audio_type,
                patient_id=patient_id,
                visit_id=visit_id,
                adhoc_id=adhoc_id,
                metadata={
                    "duration_seconds": duration_seconds,
                    "audio_type": audio_type,
                },
            )
            print(f"🔵 [AudioRepo] Step 5: Blob reference created, creating audio file record...")

            # Create audio file record
            audio_file = AudioFileMongo(
                audio_id=audio_id,
                filename=filename,
                content_type=content_type,
                file_size=file_size,
                duration_seconds=duration_seconds,
                blob_reference_id=blob_reference.file_id,
                patient_id=patient_id,
                visit_id=visit_id,
                adhoc_id=adhoc_id,
                audio_type=audio_type,
            )

            print(f"🔵 [AudioRepo] Step 6: Inserting audio file record into database...")
            await audio_file.insert()
            logger.info(f"Created audio file with blob storage: {audio_id} ({filename})")
            print(f"🔵 [AudioRepo] Step 7: Audio file record created successfully!")
            return audio_file

        except Exception as e:
            logger.error(f"Failed to create audio file: {e}")
            raise

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
        """Create a new audio file record with blob storage."""
        try:
            audio_id = str(uuid.uuid4())

            # Upload to blob storage
            blob_info = await self.blob_service.upload_file(
                file_data=audio_data,
                filename=filename,
                content_type=content_type,
                file_type="audio",
                patient_id=patient_id,
                visit_id=visit_id,
                adhoc_id=adhoc_id,
                audio_type=audio_type,
                duration_seconds=duration_seconds,
            )

            # Create blob reference
            blob_reference = await self.blob_repo.create_blob_reference(
                blob_path=blob_info["blob_path"],
                container_name=blob_info["container_name"],
                original_filename=filename,
                content_type=content_type,
                file_size=len(audio_data),
                blob_url=blob_info["blob_url"],
                file_type="audio",
                category=audio_type,
                patient_id=patient_id,
                visit_id=visit_id,
                adhoc_id=adhoc_id,
                metadata={
                    "duration_seconds": duration_seconds,
                    "audio_type": audio_type,
                },
            )

            # Create audio file record
            audio_file = AudioFileMongo(
                audio_id=audio_id,
                filename=filename,
                content_type=content_type,
                file_size=len(audio_data),
                duration_seconds=duration_seconds,
                blob_reference_id=blob_reference.file_id,
                patient_id=patient_id,
                visit_id=visit_id,
                adhoc_id=adhoc_id,
                audio_type=audio_type,
            )

            await audio_file.insert()
            logger.info(f"Created audio file with blob storage: {audio_id} ({filename})")
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
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> List[AudioFileMongo]:
        """List audio files with optional filtering and sorting."""
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

            # Determine sort field and direction
            sort_direction = -1 if sort_order == "desc" else 1
            valid_sort_fields = [
                "created_at",
                "filename",
                "file_size",
                "duration_seconds",
                "updated_at",
            ]

            if sort_by not in valid_sort_fields:
                sort_by = "created_at"

            # Get files without audio_data to reduce memory usage
            files = await AudioFileMongo.find(
                query, skip=offset, limit=limit, sort=[(sort_by, sort_direction)]
            ).to_list()

            return files

        except Exception as e:
            logger.error(f"Failed to list audio files: {e}")
            return []

    async def get_audio_data(self, audio_id: str) -> Optional[bytes]:
        """Get audio file binary data by audio_id from blob storage."""
        try:
            # Get the audio file document
            audio_file = await AudioFileMongo.find_one(AudioFileMongo.audio_id == audio_id)
            if not audio_file:
                return None

            # Get blob reference
            blob_reference = await self.blob_repo.get_blob_reference_by_id(audio_file.blob_reference_id)
            if not blob_reference:
                logger.error(f"Blob reference not found for audio file: {audio_id}")
                return None

            # Download from blob storage
            audio_data = await self.blob_service.download_file(blob_reference.blob_path)
            return audio_data

        except Exception as e:
            logger.error(f"Failed to get audio data for ID {audio_id}: {e}")
            return None

    async def update_audio_metadata(
        self, audio_id: str, duration_seconds: Optional[float] = None, **kwargs
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

    async def get_audio_stats(self) -> dict:
        """Get audio statistics using individual count queries."""
        try:
            # Use individual count queries for reliability
            adhoc_count = await self.get_audio_count(audio_type="adhoc")
            visit_count = await self.get_audio_count(audio_type="visit")
            total_count = await self.get_audio_count()

            return {
                "total_files": total_count,
                "adhoc_files": adhoc_count,
                "visit_files": visit_count,
                "other_files": total_count - adhoc_count - visit_count,
            }

        except Exception as e:
            logger.error(f"Failed to get audio stats: {e}")
            return {
                "total_files": 0,
                "adhoc_files": 0,
                "visit_files": 0,
                "other_files": 0,
            }

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

    async def get_audio_dialogue_list(
        self,
        patient_id: Optional[str] = None,
        visit_id: Optional[str] = None,
        audio_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get structured dialogue for audio files instead of full metadata."""
        try:
            from ..models.patient_m import VisitMongo

            dialogue_list = []

            # Get audio files (without audio_data to save memory)
            query = {}
            if patient_id:
                query["patient_id"] = patient_id
            if visit_id:
                query["visit_id"] = visit_id
            if audio_type:
                query["audio_type"] = audio_type

            # Add timestamp filtering
            if start_date or end_date:
                date_query = {}
                if start_date:
                    date_query["$gte"] = start_date
                if end_date:
                    date_query["$lte"] = end_date
                query["created_at"] = date_query

            # Query audio files (blob_reference_id is now optional to support legacy records)
            audio_files = await AudioFileMongo.find(
                query, skip=offset, limit=limit, sort=[("created_at", -1)]
            ).to_list()

            logger.info(f"Found {len(audio_files)} audio files matching query: {query}")

            for audio_file in audio_files:
                dialogue_data = {
                    "audio_id": audio_file.audio_id,
                    "filename": audio_file.filename,
                    "duration_seconds": audio_file.duration_seconds,
                    "patient_id": audio_file.patient_id,
                    "visit_id": audio_file.visit_id,
                    "adhoc_id": audio_file.adhoc_id,
                    "audio_type": audio_file.audio_type,
                    "created_at": audio_file.created_at.isoformat(),
                    "structured_dialogue": [],  # Initialize as empty list instead of None
                }

                # Get structured dialogue based on audio type
                # NOTE: AdhocTranscriptMongo has been removed from the codebase.
                # For audio_type == "adhoc", structured_dialogue will remain an empty list.
                if audio_file.audio_type == "visit" and audio_file.patient_id and audio_file.visit_id:
                    try:
                        visit = await VisitMongo.find_one(
                            VisitMongo.patient_id == audio_file.patient_id,
                            VisitMongo.visit_id == audio_file.visit_id,
                        )
                        if visit and visit.transcription_session and visit.transcription_session.structured_dialogue:
                            # Ensure structured_dialogue is a list
                            if isinstance(visit.transcription_session.structured_dialogue, list):
                                dialogue_data["structured_dialogue"] = visit.transcription_session.structured_dialogue
                            else:
                                logger.warning(f"structured_dialogue is not a list for visit {audio_file.audio_id}")
                                dialogue_data["structured_dialogue"] = []
                    except Exception as e:
                        logger.warning(f"Failed to get visit dialogue for {audio_file.audio_id}: {e}")
                        dialogue_data["structured_dialogue"] = []

                dialogue_list.append(dialogue_data)

            return dialogue_list

        except Exception as e:
            logger.error(f"Failed to get audio dialogue list: {e}")
            return []
