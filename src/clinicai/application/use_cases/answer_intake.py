"""Answer Intake use case for Step-01 functionality.
Formatting-only changes; behavior preserved.
"""
import logging
from ...domain.entities.visit import Visit
from ...domain.errors import (
    IntakeAlreadyCompletedError,
    PatientNotFoundError,
    VisitNotFoundError,
    DuplicateQuestionError,
)
from ...domain.value_objects.patient_id import PatientId
from ...domain.value_objects.visit_id import VisitId
from ...core.config import get_settings
from ...core.constants import TRAVEL_KEYWORDS
from ..dto.patient_dto import (
AnswerIntakeRequest,
AnswerIntakeResponse,
EditAnswerRequest,
EditAnswerResponse,
PreVisitSummaryRequest,
)
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.repositories.visit_repo import VisitRepository
from ..ports.services.question_service import QuestionService
logger = logging.getLogger("clinicai")
class AnswerIntakeUseCase:
    """Use case for answering intake questions."""

    def __init__(
        self, patient_repository: PatientRepository, visit_repository: VisitRepository, question_service: QuestionService
    ):
        self._patient_repository = patient_repository
        self._visit_repository = visit_repository
        self._question_service = question_service

    async def execute(self, request: AnswerIntakeRequest) -> AnswerIntakeResponse:
        """Execute the answer intake use case."""
        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit using VisitRepository
        visit_id = VisitId(request.visit_id)
        visit = await self._visit_repository.find_by_patient_and_visit_id(request.patient_id, visit_id)
        if not visit:
            raise VisitNotFoundError(request.visit_id)

        # Build prior context from latest completed visit (excluding current)
        from typing import Optional, List

        prior_summary: Optional[str] = None
        prior_qas: Optional[List[str]] = None
        try:
            latest = await self._visit_repository.find_latest_by_patient_id(request.patient_id)
            if latest and latest.visit_id.value != visit.visit_id.value:
                if latest.pre_visit_summary and latest.pre_visit_summary.get("summary"):
                    prior_summary = latest.pre_visit_summary.get("summary")
                if latest.intake_session and latest.intake_session.questions_asked:
                    prior_qas = [
                        f"Q: {qa.question} | A: {qa.answer}"
                        for qa in latest.intake_session.questions_asked[:8]
                    ]
        except Exception:
            prior_summary = None
            prior_qas = None

        # Check if intake is already completed
        if visit.is_intake_complete():
            raise IntakeAlreadyCompletedError(request.visit_id)

        # Determine the question being answered.
        # Prefer a previously served pending question to avoid mismatch between UI and storage.
        current_question = visit.intake_session.pending_question
        if not current_question:
            if visit.intake_session.current_question_count == 0:
                current_question = await self._question_service.generate_first_question(
                    disease=visit.symptom or "general consultation",
                    language=patient.language,
                )
            else:
                previous_answers = [qa.answer for qa in visit.intake_session.questions_asked]
                asked_questions = [qa.question for qa in visit.intake_session.questions_asked]
                # Robust uniqueness loop: never surface a duplicate question
                max_attempts = 6
                attempt = 0
                current_question = None
                while attempt < max_attempts and not current_question:
                    attempt += 1
                    try:
                        candidate = await self._question_service.generate_next_question(
                            disease=visit.symptom,
                            previous_answers=previous_answers,
                            asked_questions=asked_questions,
                            current_count=visit.intake_session.current_question_count,
                            max_count=visit.intake_session.max_questions,
                            language=patient.language,
                            recently_travelled=visit.recently_travelled,  # Use visit.recently_travelled (moved from Patient)
                            travel_questions_count=visit.intake_session.travel_questions_count,
                            prior_summary=prior_summary,
                            prior_qas=prior_qas,
                            patient_gender=patient.gender,
                            patient_age=patient.age,
                        )
                        if candidate and candidate.strip() and candidate not in asked_questions:
                            current_question = candidate
                            break
                    except DuplicateQuestionError:
                        # Service signaled a duplicate; try again
                        pass
                    except Exception:
                        # On any transient failure, try again once or twice
                        pass
                # Fallback: if still none, pick a safe generic non-duplicate question
                # NOTE: Excluded allergies and PMH from fallback pool - these should only be asked conditionally
                # based on medical context (chronic/allergy-related/high-risk conditions)
                if not current_question:
                    generic_pool = [
                        "Are you currently taking any medications?",
                        "Have you experienced fever, cough, or shortness of breath recently?",
                        "Can you describe when the symptoms started?",
                        "How would you rate the severity of your symptoms on a scale of 1 to 10?",
                    ]
                    current_question = next((q for q in generic_pool if q not in asked_questions), generic_pool[0])

        # Add the question and answer
        visit.add_question_answer(
            current_question,
            request.answer,
        )

        # Track travel-related questions
        if any(kw in current_question.lower() for kw in TRAVEL_KEYWORDS):
            visit.intake_session.travel_questions_count += 1
            logger.info(
                f"Travel question detected: '{current_question[:50]}...', "
                f"travel_questions_count now: {visit.intake_session.travel_questions_count}"
            )
            # Save visit immediately to persist travel_questions_count
            await self._visit_repository.save(visit)

        # If this is the first answer, set the visit.symptom from patient's response
        if visit.symptom == "" and visit.intake_session.current_question_count == 1:
            visit.symptom = request.answer.strip()

        self._apply_diagnostic_consent_limit(visit)

        # Check if we should stop asking questions
        should_stop = await self._question_service.should_stop_asking(
            disease=visit.symptom,
            previous_answers=[qa.answer for qa in visit.intake_session.questions_asked],
            current_count=visit.intake_session.current_question_count,
            max_count=visit.intake_session.max_questions,
        )

        next_question = None
        is_complete = False

        logger.info(
            "Completion check: should_stop=%s, can_ask_more=%s, current_count=%s",
            should_stop,
            visit.can_ask_more_questions(),
            visit.intake_session.current_question_count,
        )

        if should_stop or not visit.can_ask_more_questions():
            # Complete the intake
            visit.complete_intake()
            is_complete = True
            message = "Intake completed successfully. Ready for next step."
            logger.info(
                "Intake completed for visit %s with %s questions",
                request.visit_id,
                visit.intake_session.current_question_count,
            )
        else:
            # Generate next question for the NEXT round and cache it as pending
            previous_answers = [qa.answer for qa in visit.intake_session.questions_asked]
            asked_questions = [qa.question for qa in visit.intake_session.questions_asked]
            # Robust uniqueness loop: avoid duplicates for the NEXT question
            max_attempts = 6
            attempt = 0
            next_question = None
            while attempt < max_attempts and not next_question:
                attempt += 1
                try:
                    candidate = await self._question_service.generate_next_question(
                        disease=visit.symptom,
                        previous_answers=previous_answers,
                        asked_questions=asked_questions,
                        current_count=visit.intake_session.current_question_count,
                        max_count=visit.intake_session.max_questions,
                        language=patient.language,
                        recently_travelled=visit.recently_travelled,  # Use visit.recently_travelled (moved from Patient)
                        travel_questions_count=visit.intake_session.travel_questions_count,
                        prior_summary=prior_summary,
                        prior_qas=prior_qas,
                        patient_gender=patient.gender,
                        patient_age=patient.age,
                    )
                    if candidate and candidate.strip() and candidate not in asked_questions:
                        next_question = candidate
                        break
                except DuplicateQuestionError:
                    continue
                except Exception:
                    continue
            if not next_question:
                # NOTE: Excluded allergies and PMH from fallback pool - these should only be asked conditionally
                # based on medical context (chronic/allergy-related/high-risk conditions)
                generic_pool = [
                    "Are you currently taking any medications?",
                    "Have you experienced fever, cough, or shortness of breath recently?",
                    "Can you describe when the symptoms started?",
                    "How would you rate the severity of your symptoms on a scale of 1 to 10?",
                ]
                next_question = next((q for q in generic_pool if q not in asked_questions), generic_pool[0])
            visit.set_pending_question(next_question)
            message = (
                f"Question {visit.intake_session.current_question_count + 1} "
                f"of {visit.intake_session.max_questions}"
            )

        # Compute completion percent (LLM or deterministic fallback)
        completion_percent = await self._question_service.assess_completion_percent(
            disease=visit.symptom,
            previous_answers=[qa.answer for qa in visit.intake_session.questions_asked],
            asked_questions=[qa.question for qa in visit.intake_session.questions_asked],
            current_count=visit.intake_session.current_question_count,
            max_count=visit.intake_session.max_questions,
            prior_summary=prior_summary,
            prior_qas=prior_qas,
        )

        # Force 100% on completion
        if is_complete:
            completion_percent = 100

        # Save the updated visit
        await self._visit_repository.save(visit)

        allows_image_upload = False
        if next_question:
            allows_image_upload = await self._question_service.is_medication_question(next_question)

        return AnswerIntakeResponse(
            next_question=next_question,
            is_complete=is_complete,
            question_count=visit.intake_session.current_question_count,
            max_questions=visit.intake_session.max_questions,
            completion_percent=completion_percent,
            message=message,
            allows_image_upload=allows_image_upload,
        )

    async def edit(self, request: EditAnswerRequest) -> EditAnswerResponse:
        """Edit an existing answer by question number (1-based)."""
        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit using VisitRepository
        visit_id = VisitId(request.visit_id)
        visit = await self._visit_repository.find_by_patient_and_visit_id(request.patient_id, visit_id)
        if not visit:
            raise VisitNotFoundError(request.visit_id)

        # Validate question number
        idx = request.question_number - 1
        if idx < 0 or idx >= len(visit.intake_session.questions_asked):
            raise ValueError("Invalid question_number")

        # Apply edit (update answer)
        qa = visit.intake_session.questions_asked[idx]
        qa.answer = request.new_answer.strip()

        # Truncate all questions AFTER the edited one to allow dynamic regeneration
        visit.truncate_questions_after(request.question_number)
        self._apply_diagnostic_consent_limit(visit)

        # Recompute and set pending next question from updated context
        previous_answers = [x.answer for x in visit.intake_session.questions_asked]
        asked_questions = [x.question for x in visit.intake_session.questions_asked]
        current_count = visit.intake_session.current_question_count
        max_count = visit.intake_session.max_questions

        next_question = await self._question_service.generate_next_question(
            disease=visit.symptom or "general consultation",
            previous_answers=previous_answers,
            asked_questions=asked_questions,
            current_count=current_count,
            max_count=max_count,
            language=patient.language,
            recently_travelled=visit.recently_travelled,  # Use visit.recently_travelled (moved from Patient)
            travel_questions_count=visit.intake_session.travel_questions_count,
            patient_gender=patient.gender,
            patient_age=patient.age,
        )
        visit.set_pending_question(next_question)

        # Recalculate completion percent after truncation and edit
        completion_percent = await self._question_service.assess_completion_percent(
            disease=visit.symptom or "general consultation",
            previous_answers=previous_answers,
            asked_questions=asked_questions,
            current_count=visit.intake_session.current_question_count,
            max_count=max_count,
        )

        allows_image_upload = False
        if next_question:
            allows_image_upload = await self._question_service.is_medication_question(next_question)

        # Persist changes
        await self._visit_repository.save(visit)

        return EditAnswerResponse(
            success=True,
            message="Answer updated and subsequent questions regenerated",
            next_question=next_question,
            question_count=visit.intake_session.current_question_count,
            max_questions=max_count,
            completion_percent=completion_percent,
            allows_image_upload=allows_image_upload,
        )

    @staticmethod
    def _apply_diagnostic_consent_limit(visit: Visit) -> None:
        """Set intake max question limit based on diagnostic consent responses."""
        if not visit or not visit.intake_session:
            return

        settings = get_settings()
        base_max = settings.intake.max_questions  # Default: 12

        consent_prompts = {
            "would you like to answer some detailed diagnostic questions related to your symptoms?",
            "¿le gustaría responder algunas preguntas diagnósticas detalladas relacionadas con sus síntomas?",
        }
        positive_responses = {
            "yes", "y", "yeah", "yep", "sure", "ok", "okay", "of course", "absolutely",
            "sí", "si", "claro", "por supuesto", "por supuesto que sí", "de acuerdo",
        }
        negative_responses = {
            "no", "n", "nope", "nah", "not really", "don't", "dont",
            "no gracias", "no quiero", "no, gracias", "no, no quiero",
        }

        # Base limit is max_questions from settings unless a positive consent appears
        new_limit = base_max
        for qa in visit.intake_session.questions_asked:
            question_text = (qa.question or "").strip().lower()
            if question_text in consent_prompts:
                answer_text = (qa.answer or "").strip().lower()
                if any(resp in answer_text for resp in positive_responses):
                    # With consent, allow up to max_questions (already 12)
                    new_limit = base_max
                    break
                if any(resp in answer_text for resp in negative_responses):
                    # Without consent, use a lower limit (10, but capped by settings)
                    new_limit = min(10, base_max)
                    break

        # Ensure max never exceeds settings limit
        visit.intake_session.max_questions = max(1, min(base_max, new_limit))