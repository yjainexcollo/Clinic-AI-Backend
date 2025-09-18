"""Generate SOAP note use case for Step-03 functionality."""

from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId
from ..dto.patient_dto import SoapGenerationRequest, SoapGenerationResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.services.soap_service import SoapService


class GenerateSoapNoteUseCase:
    """Use case for generating SOAP notes."""

    def __init__(
        self, 
        patient_repository: PatientRepository, 
        soap_service: SoapService
    ):
        self._patient_repository = patient_repository
        self._soap_service = soap_service

    async def execute(self, request: SoapGenerationRequest) -> SoapGenerationResponse:
        """Execute the SOAP note generation use case."""
        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit
        visit = patient.get_visit_by_id(request.visit_id)
        if not visit:
            raise VisitNotFoundError(request.visit_id)

        # Check if visit is ready for SOAP generation
        if not visit.can_generate_soap():
            raise ValueError(f"Visit not ready for SOAP generation. Current status: {visit.status}")

        # Get transcript (use provided or stored)
        transcript = request.transcript or visit.get_transcript()
        if not transcript:
            raise ValueError("No transcript available for SOAP generation")

        # Prepare patient context
        patient_context = {
            "patient_id": patient.patient_id.value,
            "name": patient.name,
            "age": patient.age,
            "mobile": patient.mobile,
            "symptom": visit.symptom
        }

        # Get intake data and pre-visit summary
        intake_data = visit.get_intake_summary() if visit.is_intake_complete() else None
        pre_visit_summary = visit.get_pre_visit_summary()

        try:
            # Generate SOAP note
            soap_result = await self._soap_service.generate_soap_note(
                transcript=transcript,
                patient_context=patient_context,
                intake_data=intake_data,
                pre_visit_summary=pre_visit_summary
            )

            # Validate SOAP structure
            is_valid = await self._soap_service.validate_soap_structure(soap_result)
            if not is_valid:
                raise ValueError("Generated SOAP note failed validation")

            # Store SOAP note
            visit.store_soap_note(soap_result)

            # Save updated visit
            await self._patient_repository.save(patient)

            return SoapGenerationResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                soap_note=soap_result,
                generated_at=visit.soap_note.generated_at.isoformat(),
                message="SOAP note generated successfully"
            )

        except Exception as e:
            raise ValueError(f"SOAP generation failed: {str(e)}")
