"""Unified patient registration use case (scheduled + walk-in)."""

from dataclasses import dataclass
from typing import Optional

from ...application.dto.patient_dto import RegisterPatientRequest, RegisterPatientResponse
from ...application.ports.repositories.patient_repo import PatientRepository
from ...application.ports.repositories.visit_repo import VisitRepository
from ...application.ports.services.question_service import QuestionService
from .create_walk_in_visit import (
    CreateWalkInVisitRequest,
    CreateWalkInVisitResponse,
    CreateWalkInVisitUseCase,
)
from .register_patient import RegisterPatientUseCase


@dataclass
class UnifiedRegisterPatientResult:
    """Result of unified patient registration."""

    patient_id: str
    visit_id: str
    workflow_type: str
    status: str
    first_question: Optional[str]
    message: str


class UnifiedRegisterPatientUseCase:
    """Unified use case for registering patients (scheduled or walk-in)."""

    def __init__(
        self,
        patient_repository: PatientRepository,
        visit_repository: VisitRepository,
        question_service: Optional[QuestionService] = None,
    ):
        self._patient_repository = patient_repository
        self._visit_repository = visit_repository
        self._question_service = question_service

    async def execute(
        self,
        workflow_type: str,
        mobile: str,
        age: Optional[int],
        gender: Optional[str],
        language: str,
        *,
        first_name: str,
        last_name: str,
        recently_travelled: bool = False,
        consent: bool = False,
        country: str = "US",
        doctor_id: str = "",
    ) -> UnifiedRegisterPatientResult:
        """Execute unified registration; delegates to scheduled or walk-in use case."""
        wf = (workflow_type or "").strip().lower()
        if wf not in ("scheduled", "walk_in"):
            raise ValueError("workflow_type must be 'scheduled' or 'walk_in'")

        if wf == "scheduled":
            return await self._register_scheduled(
                first_name=first_name,
                last_name=last_name,
                mobile=mobile,
                age=age or 0,
                gender=gender or "",
                recently_travelled=recently_travelled,
                consent=consent,
                country=country,
                language=language,
                doctor_id=doctor_id,
            )
        return await self._register_walk_in(
            first_name=first_name,
            last_name=last_name,
            mobile=mobile,
            age=age,
            gender=gender,
            language=language,
            doctor_id=doctor_id,
        )

    async def _register_scheduled(
        self,
        *,
        first_name: str,
        last_name: str,
        mobile: str,
        age: int,
        gender: str,
        recently_travelled: bool,
        consent: bool,
        country: str,
        language: str,
        doctor_id: str,
    ) -> UnifiedRegisterPatientResult:
        if not self._question_service:
            raise ValueError("Question service is required for scheduled workflow")
        dto = RegisterPatientRequest(
            first_name=first_name,
            last_name=last_name,
            mobile=mobile,
            age=age,
            gender=gender,
            recently_travelled=recently_travelled,
            consent=consent,
            country=country,
            language=language,
        )
        use_case = RegisterPatientUseCase(
            self._patient_repository,
            self._visit_repository,
            self._question_service,
        )
        result: RegisterPatientResponse = await use_case.execute(dto, doctor_id=doctor_id)
        return UnifiedRegisterPatientResult(
            patient_id=result.patient_id,
            visit_id=result.visit_id,
            workflow_type="scheduled",
            status="intake",
            first_question=result.first_question,
            message=result.message,
        )

    async def _register_walk_in(
        self,
        *,
        first_name: str,
        last_name: str,
        mobile: str,
        age: Optional[int],
        gender: Optional[str],
        language: str,
        doctor_id: str,
    ) -> UnifiedRegisterPatientResult:
        dto = CreateWalkInVisitRequest(
            first_name=first_name,
            last_name=last_name,
            mobile=mobile,
            age=age,
            gender=gender,
            language=language,
        )
        use_case = CreateWalkInVisitUseCase(
            self._patient_repository,
            self._visit_repository,
        )
        result: CreateWalkInVisitResponse = await use_case.execute(dto, doctor_id=doctor_id)
        return UnifiedRegisterPatientResult(
            patient_id=result.patient_id,
            visit_id=result.visit_id,
            workflow_type=result.workflow_type,
            status=result.status,
            first_question=None,
            message=result.message,
        )
