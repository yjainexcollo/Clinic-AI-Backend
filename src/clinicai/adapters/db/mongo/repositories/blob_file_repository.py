"""
Blob file repository for managing blob file references in MongoDB.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from beanie import PydanticObjectId

from ..models.blob_file_reference import BlobFileReference

logger = logging.getLogger("clinicai")


class BlobFileRepository:
    """Repository for blob file reference operations."""

    async def create_blob_reference(
        self,
        blob_path: str,
        container_name: str,
        original_filename: str,
        content_type: str,
        file_size: int,
        blob_url: str,
        file_type: str,
        category: str = "general",
        patient_id: Optional[str] = None,
        visit_id: Optional[str] = None,
        adhoc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> BlobFileReference:
        """Create a new blob file reference record."""
        try:
            file_id = str(uuid.uuid4())

            blob_reference = BlobFileReference(
                file_id=file_id,
                blob_path=blob_path,
                container_name=container_name,
                original_filename=original_filename,
                content_type=content_type,
                file_size=file_size,
                blob_url=blob_url,
                file_type=file_type,
                category=category,
                patient_id=patient_id,
                visit_id=visit_id,
                adhoc_id=adhoc_id,
                metadata=metadata or {},
                expires_at=expires_at,
            )

            await blob_reference.insert()
            logger.info(f"Created blob file reference: {file_id} ({original_filename})")
            return blob_reference

        except Exception as e:
            logger.error(f"Failed to create blob file reference: {e}")
            raise

    async def get_blob_reference_by_id(self, file_id: str) -> Optional[BlobFileReference]:
        """Get blob file reference by file_id."""
        try:
            return await BlobFileReference.find_one(BlobFileReference.file_id == file_id)
        except Exception as e:
            logger.error(f"Failed to get blob file reference by ID {file_id}: {e}")
            return None

    async def get_blob_reference_by_mongo_id(self, mongo_id: str) -> Optional[BlobFileReference]:
        """Get blob file reference by MongoDB ObjectId."""
        try:
            return await BlobFileReference.get(PydanticObjectId(mongo_id))
        except Exception as e:
            logger.error(f"Failed to get blob file reference by MongoDB ID {mongo_id}: {e}")
            return None

    async def get_blob_references_by_patient(
        self, patient_id: str, limit: int = 50, offset: int = 0
    ) -> List[BlobFileReference]:
        """Get blob file references by patient ID."""
        try:
            return (
                await BlobFileReference.find(
                    BlobFileReference.patient_id == patient_id,
                    BlobFileReference.is_active == True,
                    BlobFileReference.is_deleted == False,
                )
                .skip(offset)
                .limit(limit)
                .to_list()
            )
        except Exception as e:
            logger.error(f"Failed to get blob file references by patient {patient_id}: {e}")
            return []

    async def get_blob_references_by_visit(
        self, visit_id: str, limit: int = 50, offset: int = 0
    ) -> List[BlobFileReference]:
        """Get blob file references by visit ID."""
        try:
            return (
                await BlobFileReference.find(
                    BlobFileReference.visit_id == visit_id,
                    BlobFileReference.is_active == True,
                    BlobFileReference.is_deleted == False,
                )
                .skip(offset)
                .limit(limit)
                .to_list()
            )
        except Exception as e:
            logger.error(f"Failed to get blob file references by visit {visit_id}: {e}")
            return []

    async def get_blob_references_by_type(
        self, file_type: str, limit: int = 50, offset: int = 0
    ) -> List[BlobFileReference]:
        """Get blob file references by file type."""
        try:
            return (
                await BlobFileReference.find(
                    BlobFileReference.file_type == file_type,
                    BlobFileReference.is_active == True,
                    BlobFileReference.is_deleted == False,
                )
                .skip(offset)
                .limit(limit)
                .to_list()
            )
        except Exception as e:
            logger.error(f"Failed to get blob file references by type {file_type}: {e}")
            return []

    async def update_blob_reference(self, file_id: str, update_data: Dict[str, Any]) -> Optional[BlobFileReference]:
        """Update blob file reference."""
        try:
            blob_reference = await self.get_blob_reference_by_id(file_id)
            if not blob_reference:
                return None

            # Update fields
            for key, value in update_data.items():
                if hasattr(blob_reference, key):
                    setattr(blob_reference, key, value)

            blob_reference.updated_at = datetime.utcnow()
            await blob_reference.save()

            logger.info(f"Updated blob file reference: {file_id}")
            return blob_reference

        except Exception as e:
            logger.error(f"Failed to update blob file reference {file_id}: {e}")
            return None

    async def delete_blob_reference(self, file_id: str) -> bool:
        """Mark blob file reference as deleted."""
        try:
            blob_reference = await self.get_blob_reference_by_id(file_id)
            if not blob_reference:
                return False

            blob_reference.is_deleted = True
            blob_reference.is_active = False
            blob_reference.updated_at = datetime.utcnow()
            await blob_reference.save()

            logger.info(f"Marked blob file reference as deleted: {file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete blob file reference {file_id}: {e}")
            return False

    async def hard_delete_blob_reference(self, file_id: str) -> bool:
        """Permanently delete blob file reference."""
        try:
            blob_reference = await self.get_blob_reference_by_id(file_id)
            if not blob_reference:
                return False

            await blob_reference.delete()
            logger.info(f"Permanently deleted blob file reference: {file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to permanently delete blob file reference {file_id}: {e}")
            return False

    async def cleanup_expired_files(self) -> int:
        """Clean up expired blob file references."""
        try:
            current_time = datetime.utcnow()
            expired_files = await BlobFileReference.find(
                BlobFileReference.expires_at < current_time,
                BlobFileReference.is_deleted == False,
            ).to_list()

            count = 0
            for file_ref in expired_files:
                file_ref.is_deleted = True
                file_ref.is_active = False
                file_ref.updated_at = current_time
                await file_ref.save()
                count += 1

            logger.info(f"Cleaned up {count} expired blob file references")
            return count

        except Exception as e:
            logger.error(f"Failed to cleanup expired files: {e}")
            return 0

    async def get_file_statistics(self) -> Dict[str, Any]:
        """Get file statistics."""
        try:
            total_files = await BlobFileReference.count()
            active_files = await BlobFileReference.find(
                BlobFileReference.is_active == True,
                BlobFileReference.is_deleted == False,
            ).count()

            # Get file type breakdown
            audio_files = await BlobFileReference.find(
                BlobFileReference.file_type == "audio",
                BlobFileReference.is_active == True,
                BlobFileReference.is_deleted == False,
            ).count()

            image_files = await BlobFileReference.find(
                BlobFileReference.file_type == "image",
                BlobFileReference.is_active == True,
                BlobFileReference.is_deleted == False,
            ).count()

            return {
                "total_files": total_files,
                "active_files": active_files,
                "deleted_files": total_files - active_files,
                "audio_files": audio_files,
                "image_files": image_files,
                "other_files": active_files - audio_files - image_files,
            }

        except Exception as e:
            logger.error(f"Failed to get file statistics: {e}")
            return {}
