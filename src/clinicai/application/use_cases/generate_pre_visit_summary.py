"""Generate Pre-Visit Summary use case for Step-02 functionality."""

import logging
from datetime import datetime

from ...adapters.db.mongo.repositories.llm_interaction_repository import (
    append_phase_call,
)
from ...api.schemas import PreVisitSummaryResponse
from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId
from ...domain.value_objects.visit_id import VisitId
from ..dto.patient_dto import PreVisitSummaryRequest
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
        question_service: QuestionService,
    ):
        self._patient_repository = patient_repository
        self._visit_repository = visit_repository
        self._question_service = question_service

    async def execute(self, request: PreVisitSummaryRequest, doctor_id: str) -> PreVisitSummaryResponse:
        """Execute the pre-visit summary generation use case."""
        try:
            # Store original request patient_id for image querying
            original_request_patient_id = request.patient_id

            # Find patient - handle both PatientId objects and strings
            try:
                patient_id = PatientId(request.patient_id)
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Invalid patient_id format: {request.patient_id[:50] if request.patient_id else None}, "
                    f"error: {e}"
                )
                raise PatientNotFoundError(request.patient_id) from e

            patient = await self._patient_repository.find_by_id(patient_id, doctor_id)
            if not patient:
                logger.warning(f"Patient not found: {request.patient_id[:50]}")
                raise PatientNotFoundError(request.patient_id)

            # Find visit using VisitRepository
            try:
                visit_id = VisitId(request.visit_id)
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid visit_id format: {request.visit_id}, error: {e}")
                raise VisitNotFoundError(request.visit_id) from e

            visit = await self._visit_repository.find_by_patient_and_visit_id(request.patient_id, visit_id, doctor_id)
            if not visit:
                logger.warning(f"Visit not found: visit_id={request.visit_id}, patient_id={request.patient_id[:50]}")
                raise VisitNotFoundError(request.visit_id)

            # Always allow pre-visit summary generation, regardless of intake completion.
            # If intake is incomplete, log for observability only.
            if not visit.is_intake_complete():
                question_count = visit.intake_session.current_question_count if visit.intake_session else 0
                max_questions = visit.intake_session.max_questions if visit.intake_session else 0
                logger.warning(
                    "Generating pre-visit summary for INCOMPLETE intake: " "visit_id=%s, question_count=%s/%s",
                    request.visit_id,
                    question_count,
                    max_questions,
                )

            # Build intake summary structure (may contain 0 questions)
            intake_answers = visit.get_intake_summary()
            total_questions = intake_answers.get("total_questions", 0)

            # Branch 1: No answers at all -> fixed summary, no LLM call
            if total_questions == 0:
                logger.info(
                    "Pre-visit summary requested for visit %s with zero intake answers; "
                    "returning fixed 'patient did not fill the form' summary.",
                    request.visit_id,
                )
                summary_result = {
                    "summary": "The patient did not fill the intake form. No intake answers are available.",
                    "red_flags": [],
                }
                medication_images_list = None

            # Branch 2: At least one answer -> AI-powered summary (two-step evidence-based pipeline)
            else:
                # Prepare patient data
                patient_data = {
                    "patient_id": patient.patient_id.value,
                    "name": patient.name,
                    "age": patient.age,
                    "mobile": patient.mobile,
                    "symptom": visit.symptom,
                    "visit_id": visit.visit_id.value,
                }

                # Check for medication images first (before AI generation) so we can pass info to AI
                medication_images_info = None
                medication_images_list = None
                try:
                    from beanie.operators import Or

                    from clinicai.adapters.db.mongo.models.patient_m import (
                        MedicationImageMongo,
                    )
                    from clinicai.core.utils.crypto import encode_patient_id

                    # Try both internal ID and encoded ID (in case images were stored with encoded ID)
                    patient_internal_id = str(patient.patient_id.value)
                    patient_encoded_id = encode_patient_id(patient_internal_id)

                    docs = await MedicationImageMongo.find(
                        Or(
                            MedicationImageMongo.patient_id == patient_internal_id,
                            MedicationImageMongo.patient_id == patient_encoded_id,
                            MedicationImageMongo.patient_id == original_request_patient_id,
                        ),
                        MedicationImageMongo.visit_id == visit.visit_id.value,
                    ).to_list()

                    if docs:
                        medication_images_list = [
                            {
                                "id": str(getattr(d, "id", "")),
                                "filename": getattr(d, "filename", "unknown"),
                                "content_type": getattr(d, "content_type", ""),
                            }
                            for d in docs
                        ]
                        medication_images_info = (
                            "Patient uploaded "
                            f"{len(docs)} medication image(s): "
                            f"{', '.join([d.filename for d in docs])}"
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to query medication images before summary generation: %s",
                        e,
                        exc_info=True,
                    )

                # Generate summary using AI service (internally evidence-based)
                try:
                    logger.info(
                        "Generating pre-visit summary (evidence-based) for visit %s",
                        request.visit_id,
                    )
                    summary_result = await self._question_service.generate_pre_visit_summary(
                        patient_data,
                        intake_answers,
                        language=patient.language,
                        medication_images_info=medication_images_info,
                        doctor_id=doctor_id,
                    )
                except Exception as ai_error:
                    logger.error(
                        "AI service failed to generate pre-visit summary: %s: %s",
                        type(ai_error).__name__,
                        ai_error,
                        exc_info=True,
                        extra={
                            "patient_id": request.patient_id[:50] if request.patient_id else None,
                            "visit_id": request.visit_id,
                            "error_type": type(ai_error).__name__,
                        },
                    )
                    raise ValueError(f"Failed to generate summary: {str(ai_error)}") from ai_error

                # Attach medication images to summary result (already queried above)
                if medication_images_list and isinstance(summary_result, dict):
                    summary_result["medication_images"] = medication_images_list

            # Store minimal summary in visit for EHR
            try:
                visit.store_pre_visit_summary(
                    summary_result["summary"],
                    red_flags=summary_result.get("red_flags", []),
                )
            except Exception as store_error:
                logger.error(f"Failed to store pre-visit summary: {store_error}", exc_info=True)
                raise ValueError(f"Failed to store summary: {str(store_error)}") from store_error

            # Optionally reflect step completion in workflow
            try:
                visit.status = "pre_visit_summary_generated"
            except Exception as status_error:
                logger.warning(f"Failed to update visit status: {status_error}")

            # Save the updated visit to repository (critical)
            try:
                await self._visit_repository.save(visit)
            except Exception as save_error:
                logger.error(
                    f"Failed to save visit after summary generation: {save_error}",
                    exc_info=True,
                )
                raise ValueError(f"Failed to save visit: {str(save_error)}") from save_error

            # Get the generated_at timestamp from the stored summary (not visit.updated_at)
            stored_summary = visit.get_pre_visit_summary()
            generated_at = stored_summary.get("generated_at") if stored_summary else datetime.utcnow().isoformat()

            # Structured per-visit LLM interaction log (no system prompt)
            try:
                # Construct the user prompt data (excluding patient name and phone for privacy)
                # Remove name and mobile from patient_data before logging
                patient_data_for_log = {
                    "patient_id": patient_data.get("patient_id"),
                    "age": patient_data.get("age"),
                    "symptom": patient_data.get("symptom"),
                    "visit_id": patient_data.get("visit_id"),
                }
                user_prompt_dict = {
                    "patient_data": patient_data_for_log,
                    "intake_answers": intake_answers,
                    "medication_images_info": medication_images_info,
                }
                await append_phase_call(
                    visit_id=visit.visit_id.value,
                    patient_id=patient.patient_id.value,
                    phase="pre_visit_summary",
                    agent_name="previsit_summary_generator",
                    user_prompt=user_prompt_dict,  # Will be converted to string in append_phase_call
                    response_text=summary_result["summary"],
                    metadata={
                        "prompt_version": "previsit_v1",
                        "language": patient.language,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to append structured pre-visit log: {e}")

            logger.info(f"Successfully generated pre-visit summary for visit {request.visit_id}")
            return PreVisitSummaryResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                summary=summary_result["summary"],
                generated_at=generated_at,
                medication_images=(
                    summary_result.get("medication_images") if isinstance(summary_result, dict) else None
                ),
                red_flags=(summary_result.get("red_flags") if isinstance(summary_result, dict) else None),
            )

        except (PatientNotFoundError, VisitNotFoundError, ValueError) as e:
            # Re-raise domain errors as-is (they will be handled by the router)
            raise
        except Exception as e:
            # Wrap unexpected errors with context
            logger.error(
                f"Unexpected error in GeneratePreVisitSummaryUseCase.execute: {type(e).__name__}: {e}",
                exc_info=True,
                extra={
                    "patient_id": (request.patient_id[:50] if request.patient_id else None),
                    "visit_id": request.visit_id,
                    "error_type": type(e).__name__,
                },
            )
            raise


def clean_summary_for_patient(response_dict):
    # Remove common internal fields not meant for patients
    forbidden = [
        "clinic_name",
        "doctor_name",
        "patient_name",
        "visit_date",
        "storage_path",
    ]
    return {k: v for k, v in response_dict.items() if k not in forbidden}
