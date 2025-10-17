"""Answer Intake use case for Step-01 functionality.
Formatting-only changes; behavior preserved.
"""
from ...domain.errors import (
    IntakeAlreadyCompletedError,
    PatientNotFoundError,
    VisitNotFoundError,
    DuplicateQuestionError,
)
from ...domain.value_objects.patient_id import PatientId
from ..dto.patient_dto import (
    AnswerIntakeRequest,
    AnswerIntakeResponse,
    EditAnswerRequest,
    EditAnswerResponse,
    OCRQualityInfo,
    PreVisitSummaryRequest,
)
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.services.question_service import QuestionService


class AnswerIntakeUseCase:
    """Use case for answering intake questions."""

    def __init__(
        self, patient_repository: PatientRepository, question_service: QuestionService
    ):
        self._patient_repository = patient_repository
        self._question_service = question_service

    async def execute(self, request: AnswerIntakeRequest) -> AnswerIntakeResponse:
        """Execute the answer intake use case."""
        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit
        visit = patient.get_visit_by_id(request.visit_id)
        if not visit:
            raise VisitNotFoundError(request.visit_id)
        

        # Build prior context from latest completed visit (excluding current)
        from typing import Optional, List
        prior_summary: Optional[str] = None
        prior_qas: Optional[List[str]] = None
        try:
            latest = patient.get_latest_visit()
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
                    language=patient.language
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
                            recently_travelled=patient.recently_travelled,
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
                if not current_question:
                    generic_pool = [
                        "Do you have any known allergies?",
                        "Are you currently taking any medications?",
                        "Have you experienced fever, cough, or shortness of breath recently?",
                        "Do you have any past medical conditions we should know about?",
                    ]
                    current_question = next((q for q in generic_pool if q not in asked_questions), generic_pool[0])

        # Add the question and answer
        visit.add_question_answer(
            current_question,
            request.answer,
        )
        # If this is the first answer, set the visit.symptom from patient's response
        if visit.symptom == "" and visit.intake_session.current_question_count == 1:
            visit.symptom = request.answer.strip()

        # Check if we should stop asking questions
        should_stop = await self._question_service.should_stop_asking(
            disease=visit.symptom,
            previous_answers=[qa.answer for qa in visit.intake_session.questions_asked],
            current_count=visit.intake_session.current_question_count,
            max_count=visit.intake_session.max_questions,
        )

        next_question = None
        is_complete = False

        # Enforce minimum of 5 questions before completion unless service decides to stop after >=5
        min_questions_required = 5
        reached_minimum = visit.intake_session.current_question_count >= min_questions_required
        
        # Log completion decision for debugging
        import logging
        logger = logging.getLogger("clinicai")
        logger.info(f"Completion check: should_stop={should_stop}, reached_minimum={reached_minimum}, "
                   f"can_ask_more={visit.can_ask_more_questions()}, current_count={visit.intake_session.current_question_count}")

        if (should_stop and reached_minimum) or not visit.can_ask_more_questions():
            # Complete the intake
            visit.complete_intake()
            is_complete = True
            message = "Intake completed successfully. Ready for next step."
            logger.info(f"Intake completed for visit {request.visit_id} with {visit.intake_session.current_question_count} questions")
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
                        recently_travelled=patient.recently_travelled,
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
                generic_pool = [
                    "Do you have any known allergies?",
                    "Are you currently taking any medications?",
                    "Have you experienced fever, cough, or shortness of breath recently?",
                    "Do you have any past medical conditions we should know about?",
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

        # Save the updated patient
        await self._patient_repository.save(patient)

        # Note: Pre-visit summary generation is now manual via /patients/summary/previsit endpoint
        # This allows for better control over when summaries are generated

        # Raise domain events
        # Note: In a real implementation, you'd have an event bus

        # Check if next question allows image upload
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

        # Find visit
        visit = patient.get_visit_by_id(request.visit_id)
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
            recently_travelled=patient.recently_travelled,
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
        await self._patient_repository.save(patient)

        return EditAnswerResponse(
            success=True,
            message="Answer updated and subsequent questions regenerated",
            next_question=next_question,
            question_count=visit.intake_session.current_question_count,
            max_questions=max_count,
            completion_percent=completion_percent,
            allows_image_upload=allows_image_upload,
        )