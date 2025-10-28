"""Generate Post-Visit Summary use case for Step-04 functionality."""

from datetime import datetime
from typing import Dict, Any, Optional

from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId
from ..dto.patient_dto import PostVisitSummaryRequest
from ...api.schemas import PostVisitSummaryResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.services.soap_service import SoapService


class GeneratePostVisitSummaryUseCase:
    """Use case for generating post-visit patient summaries."""

    def __init__(
        self, 
        patient_repository: PatientRepository, 
        soap_service: SoapService
    ):
        self._patient_repository = patient_repository
        self._soap_service = soap_service

    async def execute(self, request: PostVisitSummaryRequest) -> PostVisitSummaryResponse:
        """Generate a comprehensive post-visit summary for patient sharing."""
        
        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit
        visit = patient.get_visit_by_id(request.visit_id)
        if not visit:
            raise VisitNotFoundError(request.visit_id)

        # Check if visit has SOAP note generated
        if not visit.is_soap_generated():
            raise ValueError("Cannot generate post-visit summary. SOAP note must be generated first.")
        
        # Check if post-visit summary already exists
        if visit.has_post_visit_summary():
            raise ValueError("Post-visit summary already exists for this visit. Use GET endpoint to retrieve it.")

        # Get SOAP note data
        soap_note = visit.get_soap_note()
        if not soap_note:
            raise ValueError("SOAP note not found for this visit")

        # Prepare patient data for summary generation
        patient_data = {
            "patient_id": patient.patient_id.value,
            "name": patient.name,
            "age": patient.age,
            "mobile": patient.mobile,
            "symptom": visit.symptom,
            "visit_date": visit.created_at.isoformat(),
            "visit_id": visit.visit_id.value
        }

        # Prepare SOAP note data - handle both string and dict formats
        def safe_get_soap_field(field_value):
            """Safely extract SOAP field value, handling both string and dict formats."""
            if isinstance(field_value, str):
                return field_value
            elif isinstance(field_value, dict):
                return str(field_value)
            else:
                return str(field_value) if field_value else ""
        
        soap_data = {
            "subjective": safe_get_soap_field(soap_note.subjective),
            "objective": safe_get_soap_field(soap_note.objective),
            "assessment": safe_get_soap_field(soap_note.assessment),
            "plan": safe_get_soap_field(soap_note.plan),
            "highlights": soap_note.highlights or [],
            "red_flags": soap_note.red_flags or []
        }

        # Generate post-visit summary using AI service
        try:
            summary_result = await self._soap_service.generate_post_visit_summary(
                patient_data, soap_data, language=patient.language
            )
        except Exception as e:
            raise ValueError(f"Failed to generate post-visit summary: {str(e)}")

        # Parse the AI response and structure it according to recommended format
        parsed_summary = self._parse_summary_result(summary_result, visit.symptom)

        # Get chief complaint - use visit symptom or fallback to first question answer
        chief_complaint = visit.symptom
        if not chief_complaint and visit.intake_session and visit.intake_session.questions_asked:
            chief_complaint = visit.intake_session.questions_asked[0].answer
        
        response = PostVisitSummaryResponse(
            chief_complaint=chief_complaint or "General consultation",
            key_findings=parsed_summary.get("key_findings", []),
            diagnosis=parsed_summary.get("diagnosis", "Please consult with your doctor for diagnosis"),
            medications=parsed_summary.get("medications", []),
            other_recommendations=parsed_summary.get("other_recommendations", []),
            tests_ordered=parsed_summary.get("tests_ordered", []),
            next_appointment=parsed_summary.get("next_appointment"),
            red_flag_symptoms=parsed_summary.get("red_flag_symptoms", []),
            patient_instructions=parsed_summary.get("patient_instructions", []),
            reassurance_note=parsed_summary.get("reassurance_note", "Please contact us if symptoms worsen or if you have questions."),
            generated_at=datetime.utcnow().isoformat()
        )

        # Persist to visit for future retrieval
        try:
            visit.store_post_visit_summary(response.dict())
            await self._patient_repository.save(patient)
        except Exception as e:
            # Non-fatal: log and continue returning response
            print(f"ERROR: Failed to persist post-visit summary: {e}")
            print(f"ERROR: Exception type: {type(e)}")
            import traceback
            print(f"ERROR: Traceback: {traceback.format_exc()}")

        return response

    def _parse_summary_result(self, summary_result: Dict[str, Any], chief_complaint: str) -> Dict[str, Any]:
        """Parse and structure the AI-generated summary result according to recommended format."""
        
        # Default structure following the recommended format
        parsed = {
            "key_findings": [],
            "diagnosis": "",
            "medications": [],
            "other_recommendations": [],
            "tests_ordered": [],
            "next_appointment": None,
            "red_flag_symptoms": [],
            "patient_instructions": [],
            "reassurance_note": "Please contact us if symptoms worsen or if you have questions."
        }

        # If the AI service returns a structured response, use it
        if isinstance(summary_result, dict):
            # Map AI response to our structured format
            parsed.update({
                "key_findings": summary_result.get("key_findings", []),
                "diagnosis": summary_result.get("diagnosis", ""),
                "medications": summary_result.get("medications", []),
                "other_recommendations": summary_result.get("other_recommendations", []),
                "tests_ordered": summary_result.get("tests_ordered", []),
                "next_appointment": summary_result.get("next_appointment"),
                "red_flag_symptoms": summary_result.get("red_flag_symptoms", []),
                "patient_instructions": summary_result.get("patient_instructions", []),
                "reassurance_note": summary_result.get("reassurance_note", parsed["reassurance_note"])
            })
        else:
            # If it's a string, create basic structure
            parsed.update({
                "diagnosis": f"Based on your symptoms of {chief_complaint}, please follow the treatment plan as discussed.",
                "patient_instructions": [
                    "Take medications as prescribed",
                    "Rest and avoid strenuous activities",
                    "Monitor symptoms and report any changes"
                ],
                "red_flag_symptoms": [
                    "Severe pain that doesn't improve",
                    "High fever (>38.5°C)",
                    "Difficulty breathing",
                    "Signs of allergic reaction"
                ]
            })

        return parsed