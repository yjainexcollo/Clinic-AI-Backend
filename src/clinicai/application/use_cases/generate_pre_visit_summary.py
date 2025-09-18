"""Generate Pre-Visit Summary use case for Step-02 functionality."""

from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId
from ..dto.patient_dto import PreVisitSummaryRequest, PreVisitSummaryResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.services.question_service import QuestionService


class GeneratePreVisitSummaryUseCase:
    """Use case for generating pre-visit clinical summaries."""

    def __init__(
        self, 
        patient_repository: PatientRepository, 
        question_service: QuestionService
    ):
        self._patient_repository = patient_repository
        self._question_service = question_service

    async def execute(self, request: PreVisitSummaryRequest) -> PreVisitSummaryResponse:
        """Execute the pre-visit summary generation use case."""
        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit
        visit = patient.get_visit_by_id(request.visit_id)
        if not visit:
            raise VisitNotFoundError(request.visit_id)

        # Check if intake is completed
        if not visit.is_intake_complete():
            raise ValueError("Cannot generate summary for incomplete intake")

        # Prepare patient data
        patient_data = {
            "patient_id": patient.patient_id.value,
            "name": patient.name,
            "age": patient.age,
            "mobile": patient.mobile,
            "symptom": visit.symptom
        }

        # Get intake answers
        intake_answers = visit.get_intake_summary()

        # Generate summary using AI service
        summary_result = await self._question_service.generate_pre_visit_summary(
            patient_data, intake_answers
        )

        # Store minimal summary in visit for EHR
        visit.store_pre_visit_summary(summary_result["summary"])

        # Save the updated visit to repository
        await self._patient_repository.save(patient)

        return PreVisitSummaryResponse(
            patient_id=patient.patient_id.value,
            visit_id=visit.visit_id.value,
            summary=summary_result["summary"],
            generated_at=visit.updated_at.isoformat(),
        )
