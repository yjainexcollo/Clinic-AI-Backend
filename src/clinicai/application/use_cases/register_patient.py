"""Register Patient use case for Step-01 functionality.

Formatting-only changes; behavior preserved.
"""

# Unused imports removed

from ...domain.entities.patient import Patient
from ...domain.entities.visit import Visit

# Domain events currently not dispatched here; keeping behavior unchanged.
from ...domain.value_objects.patient_id import PatientId
from ...domain.value_objects.visit_id import VisitId
from ..dto.patient_dto import RegisterPatientRequest, RegisterPatientResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.services.question_service import QuestionService


class RegisterPatientUseCase:
    """Use case for registering a new patient and starting intake."""

    def __init__(
        self, patient_repository: PatientRepository, question_service: QuestionService
    ):
        self._patient_repository = patient_repository
        self._question_service = question_service

    async def execute(self, request: RegisterPatientRequest) -> RegisterPatientResponse:
        """Execute the register patient use case."""
        # Enforce consent gating
        if not getattr(request, "consent", False):
            raise ValueError("Consent is required to proceed with registration")
        # Normalize gender string
        request.gender = request.gender.strip() if request.gender else ""

        # Check if patient already exists (exact match)
        existing_patient = await self._patient_repository.find_by_name_and_mobile(
            request.name, request.mobile
        )
        if existing_patient:
            # Start a new visit for the existing patient instead of raising duplicate
            visit_id = VisitId.generate()
            visit = Visit(
                visit_id=visit_id, patient_id=existing_patient.patient_id.value, symptom=""
            )
            # Update patient's language preference for this visit
            existing_patient.language = request.language
            first_question = await self._question_service.generate_first_question(
                disease=visit.symptom or "general consultation",
                language=request.language
            )
            existing_patient.add_visit(visit)
            visit.set_pending_question(first_question)
            await self._patient_repository.save(existing_patient)
            return RegisterPatientResponse(
                patient_id=existing_patient.patient_id.value,
                visit_id=visit_id.value,
                first_question=first_question,
                message="Existing patient found. New visit started.",
            )

        # Generate patient ID using FIRST NAME only (lowercased inside VO) and mobile
        first_name_only = (request.name or "").strip().split(" ")[0]
        patient_id = PatientId.generate(first_name_only, request.mobile)

        # Check for family members (mobile-only match) for analytics
        family_members = await self._patient_repository.find_by_mobile(request.mobile)  # noqa: F841
        # Note: We don't prevent registration here, just log for analytics
        # The frontend should handle family member detection via resolve endpoint

        # Create patient entity
        patient = Patient(
            patient_id=patient_id,
            name=request.name,
            mobile=request.mobile,
            age=request.age,
            gender=request.gender,
            recently_travelled=request.recently_travelled,
            language=request.language,
        )

        # Generate visit ID
        visit_id = VisitId.generate()

        # Create visit entity
        visit = Visit(
            visit_id=visit_id, patient_id=patient_id.value, symptom=""
        )

        # Generate first question via QuestionService for consistency
        first_question = await self._question_service.generate_first_question(
            disease=visit.symptom or "general consultation",
            language=patient.language
        )

        # Add visit to patient and cache pending question to ensure UI/DB match
        patient.add_visit(visit)
        visit.set_pending_question(first_question)

        # Save patient (which includes the visit)
        await self._patient_repository.save(patient)

        # Raise domain events
        # Note: In a real implementation, you'd have an event bus
        # For now, we'll just log or handle events directly

        return RegisterPatientResponse(
            patient_id=patient_id.value,
            visit_id=visit_id.value,
            first_question=first_question,
            message="Patient registered successfully. Intake session started.",
        )