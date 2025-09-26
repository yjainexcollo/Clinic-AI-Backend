from datetime import datetime, timezone
from typing import Optional

from ..dto.patient_dto import (
    PostVisitSummaryRequest,
    PostVisitSummaryResponse,
)
from ..ports.repositories.patient_repo import PatientRepository
from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId


class GeneratePostVisitSummaryUseCase:
    """Generate a post-visit recap from stored visit data.

    Minimal implementation: validates patient and visit exist and returns a
    templated recap. Can be extended to use SOAP/transcript as needed.
    """

    def __init__(self, patient_repository: PatientRepository) -> None:
        self._patient_repository = patient_repository

    async def execute(self, request: PostVisitSummaryRequest) -> PostVisitSummaryResponse:
        patient = await self._patient_repository.find_by_id(PatientId(request.patient_id))
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        visit = patient.get_visit_by_id(request.visit_id)
        if not visit:
            raise VisitNotFoundError(request.visit_id)

        # Compose a simple post-visit recap. In the future, include SOAP/transcript highlights
        now_iso = datetime.now(timezone.utc).isoformat()
        summary = (
            "Post-Consultation Recap\n\n"
            f"Patient: {getattr(patient, 'name', 'N/A')}\n"
            f"Visit ID: {visit.visit_id.value}\n"
            f"Date: {now_iso}\n\n"
            "Key points:\n"
            "- Review pre-visit summary and intake answers.\n"
            "- Outline assessment and plan discussed.\n"
            "- Next steps: follow-up, labs, or prescriptions as advised.\n"
        )

        return PostVisitSummaryResponse(
            patient_id=patient.patient_id.value,
            visit_id=visit.visit_id.value,
            summary=summary,
            generated_at=now_iso,
        )


