"""Answer Intake use case for Step-01 functionality.
Formatting-only changes; behavior preserved.
"""
import logging
from typing import Optional, List
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
                    visit_id=visit.visit_id.value,
                    patient_id=patient.patient_id.value,
                    question_number=1,
                )
            else:
                previous_answers = [qa.answer for qa in visit.intake_session.questions_asked]
                asked_questions = [qa.question for qa in visit.intake_session.questions_asked]

                # Initialize asked_categories list for tracking (code-truth)
                # First, try to use existing asked_categories from session
                asked_categories: List[str] = list(visit.intake_session.asked_categories) if visit.intake_session.asked_categories else []
                # If asked_categories is empty or shorter than expected, reconstruct from existing questions
                if len(asked_categories) < len(visit.intake_session.questions_asked) - 1:  # -1 because Q1 has no category
                    asked_categories = []
                    # Reconstruct from existing questions based on strict sequence
                    for i, qa in enumerate(visit.intake_session.questions_asked):
                        q_num = qa.question_number  # 1-based
                        if q_num == 1:
                            continue  # Chief complaint - no category
                        elif q_num == 2:
                            asked_categories.append("duration")
                        elif q_num == 3:
                            asked_categories.append("associated_symptoms")
                        elif q_num == 4:
                            asked_categories.append("current_medications")
                        elif q_num == 5:
                            asked_categories.append("past_medical_history")
                        elif q_num == 6:
                            asked_categories.append("triggers")
                        elif q_num == 7:
                            q_lower = qa.question.lower()
                            if any(kw in q_lower for kw in ["travel", "trip", "abroad", "viaje"]):
                                asked_categories.append("travel_history")
                            else:
                                asked_categories.append("lifestyle_functional_impact")
                        elif q_num == 8:
                            q_lower = qa.question.lower()
                            if any(kw in q_lower for kw in ["family", "relative", "mother", "father"]):
                                asked_categories.append("family_history")
                            elif any(kw in q_lower for kw in ["allerg", "reaction"]):
                                asked_categories.append("allergies")
                            elif any(kw in q_lower for kw in ["pain", "severity", "0 to 10", "scale"]):
                                asked_categories.append("pain_assessment")
                            else:
                                asked_categories.append("temporal")
                        elif q_num == 9:
                            q_lower = qa.question.lower()
                            if "detailed diagnostic" in q_lower or "preguntas diagnósticas detalladas" in q_lower:
                                continue  # Consent question - no category
                            elif any(kw in q_lower for kw in ["menstrual", "period", "cycle", "lmp"]):
                                asked_categories.append("menstrual_cycle")
                            else:
                                asked_categories.append("past_evaluation")
                        elif q_num >= 10:
                            deep_num = q_num - 9
                            if deep_num == 1:
                                asked_categories.append("chronic_monitoring")
                            elif deep_num == 2:
                                asked_categories.append("screening")
                            elif deep_num == 3:
                                asked_categories.append("screening")



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
                            asked_categories=asked_categories,
                            language=patient.language,
                            recently_travelled=visit.recently_travelled,  # Use visit.recently_travelled (moved from Patient)
                            travel_questions_count=visit.intake_session.travel_questions_count,
                            prior_summary=prior_summary,
                            prior_qas=prior_qas,
                            patient_gender=patient.gender,
                            patient_age=patient.age,
                            visit_id=visit.visit_id.value,
                            patient_id=patient.patient_id.value,
                            question_number=visit.intake_session.current_question_count + 1,
                        )
                        if candidate and candidate.strip() and candidate not in asked_questions:
                            current_question = candidate

                            visit.intake_session.asked_categories = list(asked_categories)
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
                    # Avoid asking generic medication question if medications were already clearly asked
                    meds_keywords = [
                        "medication", "medications", "medicine", "medicines",
                        "drug", "drugs", "tablet", "tablets", "capsule", "capsules",
                        "insulin", "supplement", "supplements",
                    ]
                    meds_already_asked = any(
                        any(kw in (q or "").lower() for kw in meds_keywords)
                        for q in asked_questions
                    )

                    generic_pool = []
                    if not meds_already_asked:
                        generic_pool.append("Are you currently taking any medications?")
                    generic_pool.extend(
                        [
                            "Have you experienced fever, cough, or shortness of breath recently?",
                            "Can you describe when the symptoms started?",
                            "How would you rate the severity of your symptoms on a scale of 1 to 10?",
                        ]
                    )
                    current_question = next(
                        (q for q in generic_pool if q not in asked_questions),
                        generic_pool[0],
                    )

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

            # Initialize asked_categories list for tracking (code-truth)
            # First, try to use existing asked_categories from session
            asked_categories: List[str] = list(visit.intake_session.asked_categories) if visit.intake_session.asked_categories else []
            # If asked_categories is empty or shorter than expected, reconstruct from existing questions
            if len(asked_categories) < len(visit.intake_session.questions_asked) - 1:  # -1 because Q1 has no category
                asked_categories = []
                # Try to infer categories from existing questions based on strict sequence
                # This ensures tracking is accurate even if categories weren't stored before
                for i, qa in enumerate(visit.intake_session.questions_asked):
                    q_num = qa.question_number  # 1-based
                    # Map question number to expected category based on strict sequence
                    if q_num == 1:
                        # Chief complaint - no category
                        continue
                    elif q_num == 2:
                        asked_categories.append("duration")
                    elif q_num == 3:
                        asked_categories.append("associated_symptoms")
                    elif q_num == 4:
                        asked_categories.append("current_medications")
                    elif q_num == 5:
                        asked_categories.append("past_medical_history")
                    elif q_num == 6:
                        asked_categories.append("triggers")
                    elif q_num == 7:
                        # Could be travel_history or lifestyle_functional_impact
                        # Infer from question content
                        q_lower = qa.question.lower()
                        if any(kw in q_lower for kw in ["travel", "trip", "abroad", "viaje"]):
                            asked_categories.append("travel_history")
                        else:
                            asked_categories.append("lifestyle_functional_impact")
                    elif q_num == 8:
                        # Could be family_history, allergies, pain_assessment, or temporal
                        # Infer from question content
                        q_lower = qa.question.lower()
                        if any(kw in q_lower for kw in ["family", "relative", "mother", "father"]):
                            asked_categories.append("family_history")
                        elif any(kw in q_lower for kw in ["allerg", "reaction"]):
                            asked_categories.append("allergies")
                        elif any(kw in q_lower for kw in ["pain", "severity", "0 to 10", "scale"]):
                            asked_categories.append("pain_assessment")
                        else:
                            asked_categories.append("temporal")
                    elif q_num == 9:
                        # Could be consent question (no category), menstrual_cycle, or past_evaluation
                        q_lower = qa.question.lower()
                        if "detailed diagnostic" in q_lower or "preguntas diagnósticas detalladas" in q_lower:
                            # Consent question - no category
                            continue
                        elif any(kw in q_lower for kw in ["menstrual", "period", "cycle", "lmp"]):
                            asked_categories.append("menstrual_cycle")
                        else:
                            asked_categories.append("past_evaluation")
                    elif q_num >= 10:
                        # Deep diagnostic questions (chronic only)
                        deep_num = q_num - 9
                        if deep_num == 1:
                            asked_categories.append("chronic_monitoring")
                        elif deep_num == 2:
                            asked_categories.append("screening")  # Lab tests
                        elif deep_num == 3:
                            asked_categories.append("screening")  # Screening/complications



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
                        asked_categories=asked_categories,
                        language=patient.language,
                        recently_travelled=visit.recently_travelled,  # Use visit.recently_travelled (moved from Patient)
                        travel_questions_count=visit.intake_session.travel_questions_count,
                        prior_summary=prior_summary,
                        prior_qas=prior_qas,
                        patient_gender=patient.gender,
                        patient_age=patient.age,
                        visit_id=visit.visit_id.value,
                        patient_id=patient.patient_id.value,
                        question_number=visit.intake_session.current_question_count + 1,
                    )
                    if candidate and candidate.strip() and candidate not in asked_questions:
                        next_question = candidate
                        visit.intake_session.asked_categories = list(asked_categories)
                        break
                except DuplicateQuestionError:
                    # Service signaled a duplicate; try again with a new candidate
                    continue
                except Exception:
                    # Any unexpected failure from the multi-agent pipeline – break out
                    # and let completion logic decide whether to stop instead of
                    # falling back to generic questions.
                    break

            if not next_question:
                # No safe next question from the multi-agent pipeline after several
                # attempts. Instead of asking a generic fallback question (which can
                # re-open already covered topics), treat this as an early completion.
                visit.complete_intake()
                is_complete = True
                message = "Intake completed; no further questions were needed."
            else:
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
            visit_id=visit.visit_id.value,
            patient_id=patient.patient_id.value,
            question_number=visit.intake_session.current_question_count + 1,
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

        # Base limit starts at 10 (for non-chronic or negative consent)
        new_limit = 10
        for qa in visit.intake_session.questions_asked:
            question_text = (qa.question or "").strip().lower()
            if question_text in consent_prompts:
                answer_text = (qa.answer or "").strip().lower()
                if any(resp in answer_text for resp in positive_responses):
                    # With positive consent, allow up to 13 questions (for deep diagnostic questions)
                    new_limit = 13
                    break
                if any(resp in answer_text for resp in negative_responses):
                    # Without consent, use limit of 10 questions
                    new_limit = 10
                    break

        # Ensure max never exceeds settings limit
        visit.intake_session.max_questions = min(new_limit, base_max)