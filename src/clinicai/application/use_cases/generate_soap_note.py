"""Generate SOAP note use case for Step-03 functionality."""

from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId
from ...domain.value_objects.visit_id import VisitId
from ..dto.patient_dto import SoapGenerationRequest, SoapGenerationResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.repositories.visit_repo import VisitRepository
from ..ports.services.soap_service import SoapService


class GenerateSoapNoteUseCase:
    """Use case for generating SOAP notes."""

    def __init__(
        self,
        patient_repository: PatientRepository,
        visit_repository: VisitRepository, 
        soap_service: SoapService
    
    ):
        self._patient_repository = patient_repository
        self._visit_repository = visit_repository
        self._soap_service = soap_service

    async def execute(self, request: SoapGenerationRequest) -> SoapGenerationResponse:
        """Execute the SOAP note generation use case."""
        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit
        visit_id = VisitId(request.visit_id)
        visit = await self._visit_repository.find_by_patient_and_visit_id(
            request.patient_id, visit_id
        )
        if not visit:
            raise VisitNotFoundError(request.visit_id)

        # Check if visit is ready for SOAP generation
        has_transcript = visit.is_transcription_complete()
        can_generate = visit.can_generate_soap()
        
        # Get more detailed info for debugging
        transcript_text = visit.get_transcript()
        has_transcript_text = bool(transcript_text)
        transcription_status = visit.transcription_session.transcription_status if visit.transcription_session else "None"
        
        # Log detailed information for debugging
        import logging
        logger = logging.getLogger("clinicai")
        logger.info(f"[GenerateSOAP] Visit check - workflow_type: {visit.workflow_type.value}, status: {visit.status}")
        logger.info(f"[GenerateSOAP] Transcription - status: {transcription_status}, has_transcript_check: {has_transcript}, has_transcript_text: {has_transcript_text}, transcript_length: {len(transcript_text) if transcript_text else 0}")
        logger.info(f"[GenerateSOAP] Vitals - exists: {bool(visit.vitals)}, can_generate: {can_generate}")
        
        if not can_generate:
            # More detailed error message
            error_details = []
            if not has_transcript:
                error_details.append(f"Transcript not complete (transcription_status: {transcription_status})")
            if not visit.vitals:
                error_details.append("Vitals not stored")
            if visit.workflow_type.value == "scheduled" and has_transcript and visit.vitals:
                error_details.append(f"Status issue: current_status='{visit.status}', expected one of ['soap_generation', 'transcription_completed', 'transcription']")
            
            error_msg = (
                f"Visit not ready for SOAP generation. "
                f"Workflow: {visit.workflow_type.value}, "
                f"Status: {visit.status}, "
                f"Transcription status: {transcription_status}, "
                f"Has transcript (check): {has_transcript}, "
                f"Has transcript (text): {has_transcript_text}, "
                f"Has vitals: {bool(visit.vitals)}. "
                f"Issues: {'; '.join(error_details) if error_details else 'Unknown'}"
            )
            logger.warning(f"[GenerateSOAP] Validation failed: {error_msg}")
            raise ValueError(error_msg)

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
        vitals = visit.get_vitals()

        try:
            # Get patient language for SOAP generation
            patient_language = getattr(patient, 'language', 'en') or 'en'
            # Normalize language code (handle both 'sp' and 'es' for backward compatibility)
            if patient_language in ['es', 'sp']:
                patient_language = 'sp'
            
            # Generate SOAP note
            soap_result = await self._soap_service.generate_soap_note(
                transcript=transcript,
                patient_context=patient_context,
                intake_data=intake_data,
                pre_visit_summary=pre_visit_summary,
                vitals=vitals,
                language=patient_language,
                doctor_id="D123",  # Temporary hardcoded doctor_id for preferences
            )

            # Validate SOAP structure
            is_valid = await self._soap_service.validate_soap_structure(soap_result)
            if not is_valid:
                raise ValueError("Generated SOAP note failed validation")

            # Store SOAP note
            visit.store_soap_note(soap_result)

            # Save updated visit
            await self._visit_repository.save(visit)

            return SoapGenerationResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                soap_note=soap_result,
                generated_at=visit.soap_note.generated_at.isoformat(),
                message="SOAP note generated successfully"
            )

        except Exception as e:
            raise ValueError(f"SOAP generation failed: {str(e)}")
