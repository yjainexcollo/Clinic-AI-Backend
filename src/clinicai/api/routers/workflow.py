"""Workflow-related API endpoints for conditional workflow support."""

from fastapi import APIRouter, HTTPException, status, Depends, Request
import logging
from typing import List
from pydantic import BaseModel, Field

from ...application.use_cases.create_walk_in_visit import (
    CreateWalkInVisitUseCase,
    CreateWalkInVisitRequest,
    CreateWalkInVisitResponse
)
from ...domain.enums.workflow import VisitWorkflowType
from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ..deps import PatientRepositoryDep, VisitRepositoryDep
from ..schemas.common import ApiResponse, ErrorResponse
from ..utils.responses import ok, fail

router = APIRouter(prefix="/workflow", tags=["workflow"])
logger = logging.getLogger("clinicai")


class CreateWalkInVisitRequestSchema(BaseModel):
    """Request schema for creating walk-in visit."""
    name: str = Field(..., description="Patient name")
    mobile: str = Field(..., description="Patient mobile number")
    age: int = Field(None, description="Patient age")
    gender: str = Field(None, description="Patient gender")


class CreateWalkInVisitResponseSchema(BaseModel):
    """Response schema for creating walk-in visit."""
    patient_id: str = Field(..., description="Patient ID")
    visit_id: str = Field(..., description="Visit ID")
    workflow_type: str = Field(..., description="Workflow type")
    status: str = Field(..., description="Visit status")
    message: str = Field(..., description="Response message")


@router.post(
    "/walk-in/create-visit",
    response_model=ApiResponse[CreateWalkInVisitResponseSchema],
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_walk_in_visit(
    http_request: Request,
    request: CreateWalkInVisitRequestSchema,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """
    Create a walk-in visit for patients without intake.
    
    This endpoint:
    1. Creates or finds existing patient
    2. Creates walk-in visit with workflow_type = "walk_in"
    3. Sets status = "walk_in_patient"
    4. Returns patient_id, visit_id, and next steps
    """
    try:
        # Convert schema to use case request
        use_case_request = CreateWalkInVisitRequest(
            name=request.name,
            mobile=request.mobile,
            age=request.age,
            gender=request.gender
        )
        
        # Execute use case
        use_case = CreateWalkInVisitUseCase(patient_repo, visit_repo)
        response = await use_case.execute(use_case_request)
        
        # Set IDs in request state for HIPAA audit middleware
        http_request.state.audit_patient_id = response.patient_id
        http_request.state.audit_visit_id = response.visit_id
        
        return ok(http_request, data=CreateWalkInVisitResponseSchema(
            patient_id=response.patient_id,
            visit_id=response.visit_id,
            workflow_type=response.workflow_type,
            status=response.status,
            message=response.message
        ), message="Walk-in visit created successfully")
        
    except Exception as e:
        logger.error("Error creating walk-in visit", exc_info=True)
        return fail(http_request, error="INTERNAL_ERROR", message="An unexpected error occurred")


@router.get(
    "/visit/{visit_id}/available-steps",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_available_workflow_steps(
    visit_id: str,
    patient_repo: PatientRepositoryDep,
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
        
        # Find visit by ID
        visit = await visit_repo.find_by_id(visit_id_obj)
        if not visit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "VISIT_NOT_FOUND", "message": f"Visit {visit_id} not found"}
            )
        
        # Get available steps
        available_steps = visit.get_available_steps()
        
        return ok(None, data={
            "visit_id": visit_id,
            "workflow_type": visit.workflow_type.value,
            "current_status": visit.status,
            "available_steps": available_steps
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting available workflow steps", exc_info=True)
        return fail(None, error="INTERNAL_ERROR", message="An unexpected error occurred")


@router.get(
    "/visits/walk-in",
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_walk_in_visits(
    visit_repo: VisitRepositoryDep,
    limit: int = 100,
    offset: int = 0,
):
    """
    List walk-in visits with pagination.
    
    This endpoint:
    1. Finds all walk-in visits
    2. Returns paginated list
    """
    try:
        visits = await visit_repo.find_walk_in_visits(limit, offset)
        
        return ok(None, data={
            "visits": [
                {
                    "visit_id": visit.visit_id.value,
                    "patient_id": visit.patient_id,
                    "workflow_type": visit.workflow_type.value,
                    "status": visit.status,
                    "created_at": visit.created_at.isoformat(),
                    "updated_at": visit.updated_at.isoformat()
                }
                for visit in visits
            ],
            "limit": limit,
            "offset": offset,
            "count": len(visits)
        })
        
    except Exception as e:
        logger.error("Error listing walk-in visits", exc_info=True)
        return fail(None, error="INTERNAL_ERROR", message="An unexpected error occurred")
