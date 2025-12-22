"""
Visit repository interface for managing visit data.
"""

from typing import List, Optional, Dict, Any

from clinicai.domain.entities.visit import Visit
from clinicai.domain.value_objects.visit_id import VisitId
from clinicai.domain.enums.workflow import VisitWorkflowType


class VisitRepository:
    """Repository interface for managing visits."""

    async def save(self, visit: Visit) -> Visit:
        """Save a visit to the repository."""
        raise NotImplementedError

    async def find_by_id(self, visit_id: VisitId, doctor_id: str) -> Optional[Visit]:
        """Find a visit by ID."""
        raise NotImplementedError

    async def find_by_patient_id(self, patient_id: str, doctor_id: str) -> List[Visit]:
        """Find all visits for a specific patient."""
        raise NotImplementedError

    async def find_by_patient_and_visit_id(self, patient_id: str, visit_id: VisitId, doctor_id: str) -> Optional[Visit]:
        """Find a specific visit for a patient."""
        raise NotImplementedError

    async def find_latest_by_patient_id(self, patient_id: str, doctor_id: str) -> Optional[Visit]:
        """Find the latest visit for a specific patient."""
        raise NotImplementedError

    async def exists_by_id(self, visit_id: VisitId, doctor_id: str) -> bool:
        """Check if a visit exists by ID."""
        raise NotImplementedError

    async def delete(self, visit_id: VisitId) -> bool:
        """Delete a visit by ID."""
        raise NotImplementedError

    async def find_all(self, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find all visits with pagination."""
        raise NotImplementedError

    async def find_by_status(self, status: str, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find visits by status with pagination."""
        raise NotImplementedError

    async def count_by_patient_id(self, patient_id: str, doctor_id: str) -> int:
        """Count total visits for a patient."""
        raise NotImplementedError

    async def find_by_workflow_type(self, workflow_type: VisitWorkflowType, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find visits by workflow type with pagination."""
        raise NotImplementedError

    async def find_walk_in_visits(self, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find walk-in visits with pagination."""
        raise NotImplementedError

    async def find_scheduled_visits(self, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find scheduled visits with pagination."""
        raise NotImplementedError

    async def find_patients_with_visits(
        self,
        doctor_id: str,
        workflow_type: Optional[VisitWorkflowType] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        Find patients with aggregated visit information.
        
        Returns list of dicts with patient info and visit statistics.
        Each dict contains:
        - patient_id: str
        - name: str
        - mobile: str
        - age: int
        - gender: Optional[str]
        - latest_visit: dict with visit_id, workflow_type, status, created_at
        - total_visits: int
        - scheduled_visits_count: int
        - walk_in_visits_count: int
        """
        raise NotImplementedError
    
    async def try_mark_processing(
        self,
        patient_id: str,
        visit_id: VisitId,
        worker_id: str,
        stale_seconds: int,
        doctor_id: str,
    ) -> bool:
        """
        Atomically claim a transcription job by marking it as processing.
        
        Returns True only if the job was successfully claimed (modified_count == 1).
        This ensures only one worker processes each job.
        
        Conditions for claiming:
        - status == "queued"
        - OR status == "processing" but started_at is None
        - OR status == "processing" and started_at <= now - stale_seconds (stale)
        
        Updates on successful claim:
        - status = "processing"
        - started_at = now (UTC)
        - dequeued_at = now (UTC)
        - worker_id = worker_id
        - error_message = None
        """
        raise NotImplementedError
    
    async def update_transcription_session_fields(
        self,
        patient_id: str,
        visit_id: VisitId,
        doctor_id: str,
        fields: Dict[str, Any]
    ) -> bool:
        """
        Atomically update specific fields in the transcription_session of a visit.
        
        Args:
            patient_id: Patient ID
            visit_id: Visit ID
            fields: Dictionary of field names and values to update.
                   Field names should be top-level transcription_session field names
                   (e.g., "transcription_id", "last_poll_status", "last_poll_at")
        
        Returns:
            True if the visit was found and updated (modified_count == 1), False otherwise
        """
        raise NotImplementedError