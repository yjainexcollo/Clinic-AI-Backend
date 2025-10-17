"""Create Walk-in Visit use case for walk-in patients without intake."""

from clinicai.domain.entities.patient import Patient
from clinicai.domain.entities.visit import Visit
from clinicai.domain.value_objects.patient_id import PatientId
from clinicai.domain.value_objects.visit_id import VisitId
from clinicai.domain.enums.workflow import VisitWorkflowType
from clinicai.application.ports.repositories.patient_repo import PatientRepository
from clinicai.application.ports.repositories.visit_repo import VisitRepository
from clinicai.domain.errors import DuplicatePatientError, PatientNotFoundError


class CreateWalkInVisitRequest:
    """Request for creating a walk-in visit."""
    def __init__(self, name: str, mobile: str, age: int = None, gender: str = None):
        self.name = name
        self.mobile = mobile
        self.age = age
        self.gender = gender


class CreateWalkInVisitResponse:
    """Response for creating a walk-in visit."""
    def __init__(self, patient_id: str, visit_id: str, workflow_type: str, status: str, message: str):
        self.patient_id = patient_id
        self.visit_id = visit_id
        self.workflow_type = workflow_type
        self.status = status
        self.message = message


class CreateWalkInVisitUseCase:
    """Use case for creating walk-in visits."""

    def __init__(self, patient_repository: PatientRepository, visit_repository: VisitRepository):
        self._patient_repository = patient_repository
        self._visit_repository = visit_repository

    async def execute(self, request: CreateWalkInVisitRequest) -> CreateWalkInVisitResponse:
        """Execute the create walk-in visit use case."""
        
        # Check if patient already exists
        existing_patient = await self._patient_repository.find_by_name_and_mobile(
            request.name, request.mobile
        )
        
        if existing_patient:
            # Use existing patient for walk-in visit
            patient = existing_patient
        else:
            # Create new patient
            patient_id = PatientId.generate(request.name.split(" ")[0], request.mobile)
            
            patient = Patient(
                patient_id=patient_id,
                name=request.name,
                mobile=request.mobile,
                age=request.age or 0,
                gender=request.gender,
                recently_travelled=False,
                language="en",
            )
            
            # Save patient
            await self._patient_repository.save(patient)

        # Generate visit ID
        visit_id = VisitId.generate()

        # Create walk-in visit
        visit = Visit(
            visit_id=visit_id,
            patient_id=patient.patient_id.value,
            symptom="",  # No symptom for walk-in
            workflow_type=VisitWorkflowType.WALK_IN,
            status="walk_in_patient"
        )

        # Save visit
        await self._visit_repository.save(visit)

        return CreateWalkInVisitResponse(
            patient_id=patient.patient_id.value,
            visit_id=visit_id.value,
            workflow_type=VisitWorkflowType.WALK_IN.value,
            status="walk_in_patient",
            message="Walk-in visit created successfully. Patient can proceed to transcription."
        )
