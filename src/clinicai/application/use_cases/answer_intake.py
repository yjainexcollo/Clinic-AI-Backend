"""Answer Intake use case for Step-01 functionality.
Formatting-only changes; behavior preserved.
"""
from ...domain.errors import (
    IntakeAlreadyCompletedError,
    PatientNotFoundError,
    VisitNotFoundError,
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

        # Check if intake is already completed
        if visit.is_intake_complete():
            raise IntakeAlreadyCompletedError(request.visit_id)

        # Determine the question being answered.
        # Prefer a previously served pending question to avoid mismatch between UI and storage.
        current_question = visit.intake_session.pending_question
        if not current_question:
            if visit.intake_session.current_question_count == 0:
                current_question = await self._question_service.generate_first_question(
                    disease=visit.symptom or "general consultation"
                )
            else:
                previous_answers = [qa.answer for qa in visit.intake_session.questions_asked]
                asked_questions = [qa.question for qa in visit.intake_session.questions_asked]
                current_question = await self._question_service.generate_next_question(
                    disease=visit.symptom,
                    previous_answers=previous_answers,
                    asked_questions=asked_questions,
                    current_count=visit.intake_session.current_question_count,
                    max_count=visit.intake_session.max_questions,
                )

        # Add the question and answer
        # Extract OCR texts from embedded markers in answer if present
        ocr_texts: list[str] = []
        ocr_quality_info = None
        
        if "[OCR]:" in request.answer:
            try:
                marker = request.answer.split("[OCR]:", 1)[1].strip()
                # split on '|' separator when we added multiple texts
                parts = [p.strip() for p in marker.split("|") if p.strip()]
                if parts:
                    ocr_texts = parts
            except Exception:
                ocr_texts = []
        
        # Process OCR quality if we have image attachments
        if request.attachment_image_paths and len(request.attachment_image_paths) > 0:
            # Use the first image for quality assessment
            first_image_path = request.attachment_image_paths[0]
            try:
                from clinicai.core.utils.image_ocr import extract_text_with_quality
                ocr_result = extract_text_with_quality(first_image_path)
                
                ocr_quality_info = OCRQualityInfo(
                    quality=ocr_result.quality,
                    confidence=ocr_result.confidence,
                    extracted_text=ocr_result.text,
                    extracted_medications=ocr_result.extracted_medications,
                    suggestions=ocr_result.suggestions,
                    word_count=ocr_result.word_count,
                    has_medication_keywords=ocr_result.has_medication_keywords
                )
            except Exception:
                # If OCR processing fails, create a failed quality info
                ocr_quality_info = OCRQualityInfo(
                    quality="failed",
                    confidence=0.0,
                    extracted_text="",
                    extracted_medications=[],
                    suggestions=["OCR processing failed. Please try uploading a clearer image."],
                    word_count=0,
                    has_medication_keywords=False
                )
        
        visit.add_question_answer(
            current_question,
            request.answer,
            attachment_image_paths=request.attachment_image_paths,
            ocr_texts=ocr_texts or None,
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
            next_question = await self._question_service.generate_next_question(
                disease=visit.symptom,
                previous_answers=previous_answers,
                asked_questions=asked_questions,
                current_count=visit.intake_session.current_question_count,
                max_count=visit.intake_session.max_questions,
            )
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
        )

        # Force 100% on completion
        if is_complete:
            completion_percent = 100

        # Save the updated patient
        await self._patient_repository.save(patient)

        # Generate pre-visit summary if intake was completed
        if is_complete:
            try:
                from .generate_pre_visit_summary import GeneratePreVisitSummaryUseCase
                summary_use_case = GeneratePreVisitSummaryUseCase(self._patient_repository, self._question_service)
                summary_request = PreVisitSummaryRequest(
                    patient_id=request.patient_id,
                    visit_id=request.visit_id,
                )
                await summary_use_case.execute(summary_request)
                logger.info(f"Pre-visit summary generated successfully for visit {request.visit_id}")
            except Exception as e:
                # Log error but don't fail the intake completion
                logger.warning(f"Failed to generate pre-visit summary: {e}")

        # Raise domain events
        # Note: In a real implementation, you'd have an event bus

        # Check if next question allows image upload
        allows_image_upload = False
        if next_question:
            allows_image_upload = self._question_service.is_medication_question(next_question)

        return AnswerIntakeResponse(
            next_question=next_question,
            is_complete=is_complete,
            question_count=visit.intake_session.current_question_count,
            max_questions=visit.intake_session.max_questions,
            completion_percent=completion_percent,
            message=message,
            allows_image_upload=allows_image_upload,
            ocr_quality=ocr_quality_info,
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

        # Apply edit (update answer and any attachments/ocr markers)
        qa = visit.intake_session.questions_asked[idx]
        qa.answer = request.new_answer.strip()
        # Parse markers for attachments and OCR
        try:
            if "[IMAGES]:" in qa.answer:
                # extract comma-separated paths after [IMAGES]:
                trailing = qa.answer.split("[IMAGES]:", 1)[1]
                first_line = trailing.split("\n", 1)[0]
                paths = [p.strip() for p in first_line.split(",") if p.strip()]
                if paths:
                    qa.attachment_image_paths = paths
            if "[OCR]:" in qa.answer:
                trailing = qa.answer.split("[OCR]:", 1)[1]
                first_line = trailing.split("\n", 1)[0]
                texts = [p.strip() for p in first_line.split("|") if p.strip()]
                if texts:
                    qa.ocr_texts = texts
        except Exception:
            # ignore parsing errors; keep best-effort
            pass

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
            allows_image_upload = self._question_service.is_medication_question(next_question)

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