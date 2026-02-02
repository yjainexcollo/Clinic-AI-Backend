"""
Status checking routes for workflow flow visualization and visit status tracking.
"""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status

from ...core.utils.crypto import decode_patient_id
from ...domain.enums.workflow import VisitWorkflowType
from ...domain.value_objects.patient_id import PatientId
from ...domain.value_objects.visit_id import VisitId
from ..deps import PatientRepositoryDep, VisitRepositoryDep
from ..schemas.common import ApiResponse, ErrorResponse, fail, ok

router = APIRouter(prefix="/status")
logger = logging.getLogger("clinicai")


@router.get(
    "/workflow-flow",
    response_model=ApiResponse[dict],
    status_code=status.HTTP_200_OK,
    tags=["Check Status"],
    summary="Get workflow flow steps for scheduled or walk-in visits",
    description=(
        "Returns the sequence of steps for a given workflow type. "
        "Scheduled flow: Patient Registration → Intake → Pre-Visit Summary → Vitals → Transcript → SOAP → Post-Visit. "
        "Walk-in flow: Patient Registration → Vitals → Transcript → SOAP → Post-Visit."
    ),
    responses={
        200: {"description": "Workflow flow retrieved successfully"},
        400: {"model": ErrorResponse, "description": "Invalid workflow type"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_workflow_flow(
    request: Request,
    workflow_type: Literal["scheduled", "walk_in"] = "scheduled",
):
    """
    Get the workflow flow steps for a given workflow type.

    Returns the sequence of steps that a visit goes through based on the workflow type.
    """
    try:
        if workflow_type == "scheduled":
            flow_steps = [
                {
                    "step": 1,
                    "name": "Patient Registration",
                    "description": "Register a new patient and create a visit",
                    "endpoint": "POST /patients/",
                },
                {
                    "step": 2,
                    "name": "Intake",
                    "description": "Answer intake questions",
                    "endpoint": "POST /patients/consultations/answer",
                },
                {
                    "step": 3,
                    "name": "Pre-Visit Summary",
                    "description": "Generate pre-visit summary from intake data",
                    "endpoint": "POST /patients/{patient_id}/visits/{visit_id}/pre-visit-summary",
                },
                {
                    "step": 4,
                    "name": "Vitals",
                    "description": "Record patient vitals (requires pre-visit summary)",
                    "endpoint": "POST /patients/{patient_id}/visits/{visit_id}/vitals",
                },
                {
                    "step": 5,
                    "name": "Transcript",
                    "description": "Upload and transcribe audio recording",
                    "endpoint": "POST /notes/transcribe",
                },
                {
                    "step": 6,
                    "name": "SOAP",
                    "description": "Generate SOAP note from transcript and vitals",
                    "endpoint": "POST /notes/soap",
                },
                {
                    "step": 7,
                    "name": "Post-Visit Summary",
                    "description": "Generate post-visit summary for patient sharing",
                    "endpoint": "POST /patients/{patient_id}/visits/{visit_id}/post-visit-summary",
                },
            ]
        elif workflow_type == "walk_in":
            flow_steps = [
                {
                    "step": 1,
                    "name": "Patient Registration",
                    "description": "Register a new patient and create a walk-in visit",
                    "endpoint": "POST /workflow/walk-in/create-visit",
                },
                {
                    "step": 2,
                    "name": "Vitals",
                    "description": "Record patient vitals",
                    "endpoint": "POST /patients/{patient_id}/visits/{visit_id}/vitals",
                },
                {
                    "step": 3,
                    "name": "Transcript",
                    "description": "Upload and transcribe audio recording",
                    "endpoint": "POST /notes/transcribe",
                },
                {
                    "step": 4,
                    "name": "SOAP",
                    "description": "Generate SOAP note from transcript and vitals",
                    "endpoint": "POST /notes/soap",
                },
                {
                    "step": 5,
                    "name": "Post-Visit Summary",
                    "description": "Generate post-visit summary for patient sharing",
                    "endpoint": "POST /patients/{patient_id}/visits/{visit_id}/post-visit-summary",
                },
            ]
        else:
            return fail(
                request,
                error="INVALID_WORKFLOW_TYPE",
                message=f"Invalid workflow type: {workflow_type}. Must be 'scheduled' or 'walk_in'",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        return ok(
            request,
            data={
                "workflow_type": workflow_type,
                "flow_steps": flow_steps,
                "total_steps": len(flow_steps),
            },
            message=f"Workflow flow for {workflow_type} retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Error getting workflow flow: {e}", exc_info=True)
        return fail(
            request,
            error="INTERNAL_ERROR",
            message=f"An unexpected error occurred: {str(e)}",
        )


@router.get(
    "/{patient_id}/visits/{visit_id}",
    response_model=ApiResponse[dict],
    status_code=status.HTTP_200_OK,
    tags=["Check Status"],
    summary="Get visit status information (previous, current, next)",
    description=(
        "Returns the previous status, current status, and next status for a specific visit. "
        "This helps track the visit's progress through the workflow."
    ),
    responses={
        200: {"description": "Visit status retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_visit_status(
    request: Request,
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """
    Get the status information for a specific visit.

    Returns:
    - previous_status: The previous status before the current one
    - current_status: The current status of the visit
    - next_status: The predicted next status in the workflow
    """
    try:
        # Get doctor_id from request state
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Decode patient ID if needed
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id

        # Get patient to verify it exists
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id), doctor_id)
        if not patient:
            return fail(
                request,
                error="PATIENT_NOT_FOUND",
                message=f"Patient {patient_id} not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        # Get visit
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(internal_patient_id, visit_id_obj, doctor_id)
        if not visit:
            return fail(
                request,
                error="VISIT_NOT_FOUND",
                message=f"Visit {visit_id} not found for patient {patient_id}",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        # Return status information
        return ok(
            request,
            data={
                "visit_id": visit.visit_id.value,
                "patient_id": patient_id,  # Return encoded patient_id
                "workflow_type": visit.workflow_type.value,
                "previous_status": visit.previous_status,
                "current_status": visit.status,
                "next_status": visit.next_status,
            },
            message="Visit status retrieved successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visit status: {e}", exc_info=True)
        return fail(
            request,
            error="INTERNAL_ERROR",
            message=f"An unexpected error occurred: {str(e)}",
        )
