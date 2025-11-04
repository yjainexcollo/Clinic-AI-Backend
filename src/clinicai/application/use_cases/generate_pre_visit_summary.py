"""Generate Pre-Visit Summary use case for Step-02 functionality."""

import logging
from datetime import datetime
from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId
from ...domain.value_objects.visit_id import VisitId
from ..dto.patient_dto import PreVisitSummaryRequest
from ...api.schemas import PreVisitSummaryResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.repositories.visit_repo import VisitRepository
from ..ports.services.question_service import QuestionService

logger = logging.getLogger(__name__)


class GeneratePreVisitSummaryUseCase:
    """Use case for generating pre-visit clinical summaries."""

    def __init__(
        self, 
        patient_repository: PatientRepository, 
        visit_repository: VisitRepository,
        question_service: QuestionService
    ):
        self._patient_repository = patient_repository
        self._visit_repository = visit_repository
        self._question_service = question_service

    async def execute(self, request: PreVisitSummaryRequest) -> PreVisitSummaryResponse:
        """Execute the pre-visit summary generation use case."""
        # Store original request patient_id for image querying
        original_request_patient_id = request.patient_id
        
        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit using VisitRepository
        visit_id = VisitId(request.visit_id)
        visit = await self._visit_repository.find_by_patient_and_visit_id(
            request.patient_id, visit_id
        )
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
        
        # Check for medication images first (before AI generation) so we can pass info to AI
        medication_images_info = None
        medication_images_list = None
        try:
            from clinicai.adapters.db.mongo.models.patient_m import MedicationImageMongo
            from clinicai.core.utils.crypto import encode_patient_id
            
            # Try both internal ID and encoded ID (in case images were stored with encoded ID)
            patient_internal_id = str(patient.patient_id.value)
            patient_encoded_id = encode_patient_id(patient_internal_id)
            
            # Query with OR condition to find images stored with either format
            from beanie.operators import Or
            logger.info(f"[GeneratePreVisitSummary] Querying medication images for visit {visit.visit_id.value}")
            logger.info(f"[GeneratePreVisitSummary] Patient internal_id: {patient_internal_id[:50]}..., encoded_id: {patient_encoded_id[:50]}...")
            
            # Also check original request patient_id (might be encoded and different from our encoding)
            docs = await MedicationImageMongo.find(
                Or(
                    MedicationImageMongo.patient_id == patient_internal_id,
                    MedicationImageMongo.patient_id == patient_encoded_id,
                    MedicationImageMongo.patient_id == original_request_patient_id  # Check original request ID
                ),
                MedicationImageMongo.visit_id == visit.visit_id.value,
            ).to_list()
            
            # If no docs found, try querying all images for this visit to see what patient_ids exist
            if not docs:
                all_visit_images = await MedicationImageMongo.find(
                    MedicationImageMongo.visit_id == visit.visit_id.value,
                ).to_list()
                if all_visit_images:
                    logger.warning(f"[GeneratePreVisitSummary] No images found with patient_id match, but found {len(all_visit_images)} images for visit {visit.visit_id.value}")
                    logger.warning(f"[GeneratePreVisitSummary] Images have patient_ids: {[str(img.patient_id)[:50] for img in all_visit_images]}")
            
            if docs:
                logger.info(f"[GeneratePreVisitSummary] Found {len(docs)} medication images for visit {visit.visit_id.value}")
                medication_images_list = [
                    {
                        "id": str(getattr(d, "id", "")),
                        "filename": getattr(d, "filename", "unknown"),
                        "content_type": getattr(d, "content_type", ""),
                    }
                    for d in docs
                ]
                medication_images_info = f"Patient uploaded {len(docs)} medication image(s): {', '.join([d.filename for d in docs])}"
                logger.info(f"[GeneratePreVisitSummary] Medication images list: {medication_images_list}")
            else:
                logger.warning(f"[GeneratePreVisitSummary] No medication images found for visit {visit.visit_id.value} (checked both internal_id={patient_internal_id[:50]}... and encoded_id={patient_encoded_id[:50]}...)")
        except Exception as e:
            logger.warning(f"Failed to query medication images before summary generation: {e}", exc_info=True)

        # Generate summary using AI service
        summary_result = await self._question_service.generate_pre_visit_summary(
            patient_data, intake_answers, language=patient.language, medication_images_info=medication_images_info
        )

        # Attach medication images to summary result (already queried above)
        if medication_images_list:
            summary_result["medication_images"] = medication_images_list

        # Store minimal summary in visit for EHR
        visit.store_pre_visit_summary(
            summary_result["summary"], 
            red_flags=summary_result.get("red_flags", [])
        )

        # Optionally reflect step completion in workflow
        try:
            visit.status = "pre_visit_summary_generated"
        except Exception:
            pass

        # Save the updated visit to repository (critical)
        await self._visit_repository.save(visit)

        # Get the generated_at timestamp from the stored summary (not visit.updated_at)
        stored_summary = visit.get_pre_visit_summary()
        generated_at = stored_summary.get("generated_at") if stored_summary else datetime.utcnow().isoformat()

        return PreVisitSummaryResponse(
            patient_id=patient.patient_id.value,
            visit_id=visit.visit_id.value,
            summary=summary_result["summary"],
            generated_at=generated_at,
            medication_images=summary_result.get("medication_images") if isinstance(summary_result, dict) else None,
            red_flags=summary_result.get("red_flags") if isinstance(summary_result, dict) else None,
        )

def clean_summary_for_patient(response_dict):
    # Remove common internal fields not meant for patients
    forbidden = ["clinic_name", "doctor_name", "patient_name", "visit_date", "storage_path"]
    return {k: v for k, v in response_dict.items() if k not in forbidden}