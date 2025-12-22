"""Register Patient use case for Step-01 functionality.

Formatting-only changes; behavior preserved.
"""

# Unused imports removed

from ...domain.entities.patient import Patient
from ...domain.entities.visit import Visit

# Domain events currently not dispatched here; keeping behavior unchanged.
from ...domain.value_objects.patient_id import PatientId
from ...domain.value_objects.visit_id import VisitId
from ...core.config import get_settings
from ..dto.patient_dto import RegisterPatientRequest, RegisterPatientResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.repositories.visit_repo import VisitRepository
from ..ports.services.question_service import QuestionService


class RegisterPatientUseCase:
    """Use case for registering a new patient and starting intake."""

    def __init__(
        self, patient_repository: PatientRepository, visit_repository: VisitRepository, question_service: QuestionService
    ):
        self._patient_repository = patient_repository
        self._visit_repository = visit_repository
        self._question_service = question_service

    async def execute(self, request: RegisterPatientRequest, doctor_id: str) -> RegisterPatientResponse:
        """Execute the register patient use case."""
        # Enforce consent gating
        if not getattr(request, "consent", False):
            raise ValueError("Consent is required to proceed with registration")
        # Normalize gender string
        request.gender = request.gender.strip() if request.gender else ""

        # Check if patient already exists (exact match)
        existing_patient = await self._patient_repository.find_by_name_and_mobile(
            f"{request.first_name} {request.last_name}", request.mobile, doctor_id
        )
        if existing_patient:
            # Start a new visit for the existing patient instead of raising duplicate
            visit_id = VisitId.generate()
            visit = Visit(
                visit_id=visit_id,
                patient_id=existing_patient.patient_id.value,
                doctor_id=doctor_id,
                symptom="",
                recently_travelled=request.recently_travelled,  # Store travel history on visit (visit-specific)
            )
            settings = get_settings()
            visit.intake_session.max_questions = settings.intake.max_questions
            # Update patient's language preference for this visit
            existing_patient.language = request.language
            first_question = await self._question_service.generate_first_question(
                disease=visit.symptom or "general consultation",
                language=request.language
            )
            visit.set_pending_question(first_question)
            
            # Save patient and visit separately
            await self._patient_repository.save(existing_patient)
            await self._visit_repository.save(visit)
            
            return RegisterPatientResponse(
                patient_id=existing_patient.patient_id.value,
                visit_id=visit_id.value,
                first_question=first_question,
                message="Existing patient found. New visit started.",
            )

        # Generate patient ID using FIRST NAME only (lowercased inside VO) and mobile
        patient_id = PatientId.generate(request.first_name, request.mobile)

        # Check for family members (mobile-only match) for analytics
        family_members = await self._patient_repository.find_by_mobile(request.mobile, doctor_id)  # noqa: F841
        # Note: We don't prevent registration here, just log for analytics
        # The frontend should handle family member detection via resolve endpoint

        # Create patient entity
        patient = Patient(
            patient_id=patient_id,
            doctor_id=doctor_id,
            name=f"{request.first_name} {request.last_name}",
            mobile=request.mobile,
            age=request.age,
            gender=request.gender,
            # recently_travelled removed from Patient - now stored on Visit
            language=request.language,
        )

        # Generate visit ID
        visit_id = VisitId.generate()

        # Create visit entity with recently_travelled (travel is visit-specific, not lifetime patient attribute)
        visit = Visit(
            visit_id=visit_id,
            patient_id=patient_id.value,
            doctor_id=doctor_id,
            symptom="",
            recently_travelled=request.recently_travelled,
        )
        settings = get_settings()
        visit.intake_session.max_questions = settings.intake.max_questions

        # Generate first question via QuestionService for consistency
        first_question = await self._question_service.generate_first_question(
            disease=visit.symptom or "general consultation",
            language=patient.language
        )

        # Set pending question on visit
        visit.set_pending_question(first_question)

        # Save patient and visit separately
        await self._patient_repository.save(patient)
        await self._visit_repository.save(visit)

        # Raise domain events
        # Note: In a real implementation, you'd have an event bus
        # For now, we'll just log or handle events directly

        return RegisterPatientResponse(
            patient_id=patient_id.value,
            visit_id=visit_id.value,
            first_question=first_question,
            message="Patient registered successfully. Intake session started.",
        )