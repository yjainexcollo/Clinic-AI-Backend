"""Workflow-related API endpoints for conditional workflow support."""

import logging

from fastapi import APIRouter, HTTPException, Request, status

from ..deps import VisitRepositoryDep
from ..schemas.common import ErrorResponse, fail, ok

router = APIRouter(prefix="/workflow")
logger = logging.getLogger("clinicai")


@router.get(
    "/visit/{visit_id}/available-steps",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    responses={
        404: {"model": ErrorResponse, "description": "Visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_available_workflow_steps(
    request: Request,
    visit_id: str,
    visit_repo: VisitRepositoryDep,
):
    """
    Get available workflow steps for a visit based on its workflow type and status.

    This endpoint:
    1. Finds the visit
    2. Determines available steps based on workflow type
    3. Returns list of available steps
    """
    try:
        # Convert string visit_id to VisitId object
        from ...domain.value_objects.visit_id import VisitId

        visit_id_obj = VisitId(visit_id)

        # Extract doctor_id
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Find visit by ID
        visit = await visit_repo.find_by_id(visit_id_obj, doctor_id)
        if not visit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "VISIT_NOT_FOUND",
                    "message": f"Visit {visit_id} not found",
                },
            )

        # Get available steps
        available_steps = visit.get_available_steps()

        return ok(
            request,
            data={
                "visit_id": visit_id,
                "workflow_type": visit.workflow_type.value,
                "current_status": visit.status,
                "available_steps": available_steps,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting available workflow steps", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message="An unexpected error occurred")
