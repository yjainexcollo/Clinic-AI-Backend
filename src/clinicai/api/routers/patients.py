"""Patient-related API endpoints.

Formatting-only changes; behavior preserved.
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, Request, Form, File
import logging
import traceback
from typing import Union, Optional, List

from ...application.dto.patient_dto import (
    AnswerIntakeRequest,
    PreVisitSummaryRequest,
    PostVisitSummaryRequest,
    RegisterPatientRequest,
)
from ...application.dto.patient_dto import EditAnswerRequest
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
from ...application.use_cases.answer_intake import AnswerIntakeUseCase
from ...application.use_cases.generate_pre_visit_summary import GeneratePreVisitSummaryUseCase
from ...application.use_cases.generate_post_visit_summary import GeneratePostVisitSummaryUseCase
from ...core.utils.crypto import decode_patient_id
from ...application.use_cases.register_patient import RegisterPatientUseCase
from ...domain.errors import (
    DuplicatePatientError,
    DuplicateQuestionError,
    IntakeAlreadyCompletedError,
    InvalidDiseaseError,
    InvalidPatientDataError,
    PatientNotFoundError,
    QuestionLimitExceededError,
    VisitNotFoundError,
    PatientAlreadyExistsError,
)
from ...domain.enums.workflow import VisitWorkflowType

from ..deps import PatientRepositoryDep, VisitRepositoryDep, QuestionServiceDep, SoapServiceDep
from ...core.utils.crypto import encode_patient_id, decode_patient_id
from ..schemas import (
    AnswerIntakeResponse,
    ErrorResponse,
    AnswerIntakeRequest as AnswerIntakeRequestSchema,
    PatientSummarySchema,
    PreVisitSummaryResponse,
    PostVisitSummaryResponse,
    RegisterPatientResponse,
    EditAnswerRequest as EditAnswerRequestSchema,
    EditAnswerResponse as EditAnswerResponseSchema,
    RegisterPatientRequest as RegisterPatientRequestSchema,
    IntakeSummarySchema,
    PatientWithVisitsSchema,
    PatientListResponse,
    LatestVisitInfo,
    VisitListItemSchema,
    VisitDetailSchema,
    VisitListResponse,
    TranscriptionSessionSchema,
    SoapNoteSchema
)
from fastapi import UploadFile, File, Form
from fastapi.responses import Response
from pathlib import Path
from datetime import datetime
import os
from beanie import PydanticObjectId
from ..schemas.common import ApiResponse, ErrorResponse
from ..utils.responses import ok, fail

router = APIRouter(prefix="/patients")
logger = logging.getLogger("clinicai")


@router.post(
    "/",
    response_model=ApiResponse[RegisterPatientResponse],
    status_code=status.HTTP_201_CREATED,
    tags=["Patient Registration"],
    summary="Register a new patient and start intake session",
    responses={
        201: {"description": "Patient registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        409: {"model": ErrorResponse, "description": "Duplicate patient"},
        422: {"model": ErrorResponse, "description": "Invalid symptom"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def register_patient(
    http_request: Request,
    request: RegisterPatientRequestSchema,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
    question_service: QuestionServiceDep,
):
    """
    Register a new patient and start intake session.

    This endpoint:
    1. Validates patient data
    2. Generates patient_id and visit_id
    3. Creates patient and visit entities
    4. Generates first question based on primary symptom
    5. Returns patient_id, visit_id, and first question
    """
    try:
        # Extract doctor_id from middleware
        doctor_id = getattr(http_request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                http_request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Convert Pydantic model to DTO
        dto_request = RegisterPatientRequest(
            first_name=request.first_name,
            last_name=request.last_name,
            mobile=request.mobile,
            age=request.age,
            gender=request.gender,
            recently_travelled=request.recently_travelled,
            consent=request.consent,
            country=request.country,
            language=request.language,
        )

        # Execute use case
        use_case = RegisterPatientUseCase(patient_repo, visit_repo, question_service)
        result = await use_case.execute(dto_request, doctor_id=doctor_id)

        # Set IDs in request state for HIPAA audit middleware
        http_request.state.audit_patient_id = encode_patient_id(result.patient_id)
        http_request.state.audit_visit_id = result.visit_id

        # Only return opaque patient_id/visit_id, do NOT include internal details
        response = RegisterPatientResponse(
            patient_id=encode_patient_id(result.patient_id),
            visit_id=result.visit_id,
            first_question=result.first_question,
            message="Patient registered successfully. Intake session started."
        )
        return ok(http_request, data=response, message="Created")

    except DuplicatePatientError as e:
        return fail(
            http_request, 
            error="DUPLICATE_PATIENT", 
            message=e.message, 
            status_code=status.HTTP_409_CONFLICT
        )
    except InvalidDiseaseError as e:
        return fail(
            http_request, 
            error="INVALID_DISEASE", 
            message=e.message, 
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except InvalidPatientDataError as e:
        # InvalidPatientDataError indicates validation failure (mobile number, name, age, etc.)
        return fail(
            http_request,
            error="INVALID_PATIENT_DATA",
            message=e.message or "Invalid patient data provided",
            details=e.details or {},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except Exception as e:
        # Log full exception details for debugging
        error_type = type(e).__name__
        error_message = str(e)
        logger.exception(
            f"Unhandled error in patient registration: {error_type}: {error_message}",
            exc_info=True,
            extra={
                "error_type": error_type,
                "error_message": error_message,
                "request_data": {
                    "first_name": getattr(request, 'first_name', None),
                    "mobile": getattr(request, 'mobile', None),
                }
            }
        )
        # Return error response with more detail in development mode
        import os
        is_dev = os.getenv("APP_ENV", "production") == "development" or os.getenv("DEBUG", "false").lower() == "true"
        error_msg = f"{error_type}: {error_message}" if is_dev else "Internal error occurred."
        return fail(
            http_request, 
            error="INTERNAL_ERROR", 
            message=error_msg,
            details={"error_type": error_type} if is_dev else {},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get(
    "/",
    response_model=ApiResponse[PatientListResponse],
    status_code=status.HTTP_200_OK,
    tags=["Patient Management"],
    summary="Get all patients with visit information",
    responses={
        200: {"description": "List of patients retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_patients(
    request: Request,
    visit_repo: VisitRepositoryDep,
    workflow_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "created_at",
    sort_order: str = "desc",
):
    """
    Get all patients with aggregated visit information, sorted by workflow type.
    
    This endpoint:
    1. Retrieves patients with their visit statistics
    2. Filters by workflow type if specified (scheduled or walk_in)
    3. Sorts by latest visit date or patient name
    4. Returns paginated results with visit counts
    
    Query Parameters:
    - workflow_type: Optional filter by "scheduled" or "walk_in"
    - limit: Number of results per page (default: 100)
    - offset: Number of results to skip (default: 0)
    - sort_by: Sort field - "created_at" or "name" (default: "created_at")
    - sort_order: Sort direction - "asc" or "desc" (default: "desc")
    """
    try:
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Parse workflow_type if provided
        workflow_type_enum = None
        if workflow_type:
            workflow_type_lower = workflow_type.lower()
            if workflow_type_lower == "scheduled":
                workflow_type_enum = VisitWorkflowType.SCHEDULED
            elif workflow_type_lower == "walk_in" or workflow_type_lower == "walk-in":
                workflow_type_enum = VisitWorkflowType.WALK_IN
            else:
                return fail(
                    request,
                    error="INVALID_WORKFLOW_TYPE",
                    message=f"Invalid workflow_type: {workflow_type}. Must be 'scheduled' or 'walk_in'",
                    status_code=status.HTTP_400_BAD_REQUEST
                )
        
        # Validate sort parameters
        if sort_by not in ["created_at", "name"]:
            sort_by = "created_at"
        if sort_order not in ["asc", "desc"]:
            sort_order = "desc"
        
        # Get patients with visits for this doctor
        patients_data = await visit_repo.find_patients_with_visits(
            doctor_id=doctor_id,
            workflow_type=workflow_type_enum,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Encode patient IDs and format response
        formatted_patients = []
        for patient_data in patients_data:
            # Encode patient_id for client
            encoded_patient_id = encode_patient_id(patient_data["patient_id"])
            
            # Format latest visit if present
            latest_visit_info = None
            if patient_data.get("latest_visit"):
                latest_visit = patient_data["latest_visit"]
                latest_visit_info = LatestVisitInfo(
                    visit_id=latest_visit.get("visit_id", ""),
                    workflow_type=latest_visit.get("workflow_type", ""),
                    status=latest_visit.get("status", ""),
                    created_at=latest_visit.get("created_at", datetime.utcnow())
                )
            
            formatted_patients.append(PatientWithVisitsSchema(
                patient_id=encoded_patient_id,
                name=patient_data.get("name", "Unknown"),
                mobile=patient_data.get("mobile", ""),
                age=patient_data.get("age", 0),
                gender=patient_data.get("gender"),
                latest_visit=latest_visit_info,
                total_visits=patient_data.get("total_visits", 0),
                scheduled_visits_count=patient_data.get("scheduled_visits_count", 0),
                walk_in_visits_count=patient_data.get("walk_in_visits_count", 0)
            ))
        
        # Calculate total count (approximate - for pagination)
        # Note: For accurate total count, we'd need a separate aggregation
        # For now, we'll use the returned count as an indicator
        total_count = len(formatted_patients)
        has_more = len(formatted_patients) == limit  # If we got full limit, there might be more
        
        return ok(
            request,
            data=PatientListResponse(
                patients=formatted_patients,
                pagination={
                    "limit": limit,
                    "offset": offset,
                    "count": total_count,
                    "has_more": has_more
                }
            ),
            message="Patients retrieved successfully"
        )
        
    except Exception as e:
        logger.error("Error listing patients", exc_info=True)
        return fail(
            request, 
            error="INTERNAL_ERROR", 
            message="An unexpected error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post(
    "/consultations/answer",
    response_model=AnswerIntakeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Intake + Pre-Visit Summary"],
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        409: {"model": ErrorResponse, "description": "Intake already completed"},
        422: {
            "model": ErrorResponse,
            "description": "Question limit exceeded or duplicate question",
        },
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def answer_intake_question(
    request: Request,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
    question_service: QuestionServiceDep,
    # Optional multipart fields (to enable Swagger form UI)
    form_patient_id: Optional[str] = Form(None),
    form_visit_id: Optional[str] = Form(None),
    form_answer: Optional[str] = Form(None),
):
    """
    Answer an intake question and get the next question.

    This endpoint:
    1. Validates the answer
    2. Finds the patient and visi
    3. Adds the answer to the intake session
    4. Generates next question or completes intake
    5. Returns next question or completion status
    """
    try:
        content_type = request.headers.get("content-type", "")
        logger.info(f"[AnswerIntake] Incoming content-type: {content_type}")
        if content_type.startswith("application/json"):
            body = await request.json()
            raw_pid = (body.get("patient_id", "").strip())
            visit_id_from_body = (body.get("visit_id", "").strip())
            
            # Set IDs in request state for HIPAA audit middleware
            request.state.audit_patient_id = raw_pid
            request.state.audit_visit_id = visit_id_from_body
            
            try:
                internal_pid = decode_patient_id(raw_pid)
            except Exception as e:
                logger.warning("[AnswerIntake][JSON] Failed to decode patient_id '%s': %s; using raw value", raw_pid, e)
                internal_pid = raw_pid
            # If internal_pid looks like an opaque token (has non-alnum/underscore), don't construct PatientId later
            # The use case accepts raw string; repository lookup will handle both forms.
            dto_request = AnswerIntakeRequest(
                patient_id=internal_pid,
                visit_id=visit_id_from_body,
                answer=(body.get("answer", "").strip()),
            )
        elif content_type.startswith("multipart/form-data") or content_type.startswith("application/x-www-form-urlencoded"):
            # Prefer explicitly bound form fields first; fallback to reading the form
            if form_patient_id is None or form_visit_id is None or form_answer is None:
                form = await request.form()
                form_patient_id = form_patient_id or (form.get("patient_id") or "").strip()
                form_visit_id = form_visit_id or (form.get("visit_id") or "").strip()
                form_answer = form_answer or (form.get("answer") or "").strip()
            
            # Set IDs in request state for HIPAA audit middleware
            request.state.audit_patient_id = form_patient_id
            request.state.audit_visit_id = form_visit_id
            
            logger.info(
                "[AnswerIntake][FORM] Received form fields: patient_id=%s visit_id=%s answer_len=%s",
                (form_patient_id or "").strip(),
                (form_visit_id or "").strip(),
                len((form_answer or "").strip()),
            )

            if not (form_patient_id and form_visit_id and form_answer):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "VALIDATION_ERROR",
                        "message": "Missing required form fields: patient_id, visit_id, answer",
                        "details": {},
                    },
                )

            # Decode patient_id for use case
            try:
                internal_pid = decode_patient_id(form_patient_id)
            except Exception as e:
                logger.warning("[AnswerIntake][FORM] Failed to decode patient_id '%s': %s; using raw value", form_patient_id, e)
                internal_pid = form_patient_id
            dto_request = AnswerIntakeRequest(
                patient_id=internal_pid,
                visit_id=form_visit_id,
                answer=form_answer,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={
                    "error": "UNSUPPORTED_MEDIA_TYPE",
                    "message": "Content-Type must be application/json, multipart/form-data, or application/x-www-form-urlencoded",
                    "details": {"content_type": content_type},
                },
            )

        # Execute use case
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "MISSING_DOCTOR_ID",
                    "message": "X-Doctor-ID header is required",
                    "details": {},
                },
            )
        use_case = AnswerIntakeUseCase(patient_repo, visit_repo, question_service)
        result = await use_case.execute(dto_request, doctor_id=doctor_id)

        return AnswerIntakeResponse(
            next_question=result.next_question,
            is_complete=result.is_complete,
            question_count=result.question_count,
            max_questions=result.max_questions,
            completion_percent=result.completion_percent,
            message=result.message,
            allows_image_upload=result.allows_image_upload,
        )
    except HTTPException:
        # Re-raise HTTPException so FastAPI can handle it properly
        raise
    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "PATIENT_NOT_FOUND",
                "message": e.message,
                "details": e.details or {},
            },
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "VISIT_NOT_FOUND",
                "message": e.message,
                "details": e.details or {},
            },
        )
    except IntakeAlreadyCompletedError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "INTAKE_ALREADY_COMPLETED",
                "message": e.message,
                "details": e.details or {},
            },
        )

    except DuplicateQuestionError as e:
        # Legacy DuplicateQuestionError does not expose structured error codes;
        # return a safe, explicit payload instead of 500.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "DUPLICATE_QUESTION",
                "message": str(e),
                "details": {},
            },
        )

    
    except (QuestionLimitExceededError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details or {},
            },
        )
    except Exception as e:
        logger.error("Unhandled error in answer_intake_question", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {},
            },
        )


@router.patch(
    "/consultations/answer",
    response_model=EditAnswerResponseSchema,
    status_code=status.HTTP_200_OK,
    tags=["Intake + Pre-Visit Summary"],
)
async def edit_intake_answer(
    http_request: Request,
    request: EditAnswerRequestSchema,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
    question_service: QuestionServiceDep,
):
    """Edit an existing answer by question number."""
    try:
        # Set IDs in request state for HIPAA audit middleware
        http_request.state.audit_patient_id = request.patient_id
        http_request.state.audit_visit_id = request.visit_id
        
        doctor_id = getattr(http_request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                http_request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        use_case = AnswerIntakeUseCase(patient_repo, visit_repo, question_service)
        dto_request = EditAnswerRequest(
            patient_id=decode_patient_id(request.patient_id),
            visit_id=request.visit_id,
            question_number=request.question_number,
            new_answer=request.new_answer,
        )
        result = await use_case.edit(dto_request, doctor_id=doctor_id)
        return EditAnswerResponseSchema(
            success=result.success,
            message=result.message,
            next_question=result.next_question,
            question_count=result.question_count,
            max_questions=result.max_questions,
            completion_percent=result.completion_percent,
            allows_image_upload=result.allows_image_upload,
        )
    except PatientNotFoundError as e:
        return fail(
            http_request, 
            error="PATIENT_NOT_FOUND", 
            message=e.message,
            status_code=status.HTTP_404_NOT_FOUND
        )
    except VisitNotFoundError as e:
        return fail(
            http_request, 
            error="VISIT_NOT_FOUND", 
            message=e.message,
            status_code=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error("Unhandled error in edit_intake_answer", exc_info=True)
        return fail(
            http_request, 
            error="INTERNAL_ERROR", 
            message="An unexpected error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# Endpoint for medication image uploads (supports both single and multiple images)
@router.post("/webhook/images", tags=["Intake + Pre-Visit Summary"])
async def upload_medication_images(
    request: Request,
    patient_id: str = Form(...),
    visit_id: str = Form(...),
    images: Optional[List[UploadFile]] = File(None),
):
    try:
        # Set IDs in request state for HIPAA audit middleware
        request.state.audit_patient_id = patient_id
        request.state.audit_visit_id = visit_id
        
        logger.info("[WebhookImages] Incoming upload for patient_id=%s visit_id=%s", patient_id, visit_id)
        # Safely resolve patient ID (handles both encrypted and plain text)
        from ...core.utils.patient_id_resolver import resolve_patient_id
        internal_patient_id = resolve_patient_id(patient_id, "patient endpoint")
        
        valid_types = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/heic", "image/heif"}
        uploaded_images = []
        errors = []
        
        # Prefer FastAPI-bound files; fallback to manual parse
        file_list: List[UploadFile] = [f for f in (images or []) if getattr(f, "filename", None)]
        if not file_list:
            try:
                form = await request.form()
                candidates: List[UploadFile] = []
                for key in ["images", "medication_images", "medication_image", "image", "file", "photo"]:
                    try:
                        items = form.getlist(key)
                        for item in items:
                            if isinstance(item, UploadFile) and getattr(item, "filename", None):
                                candidates.append(item)
                    except Exception:
                        pass
                file_list = candidates
            except Exception:
                file_list = []

        logger.info("[WebhookImages] Parsed %d file(s) from form", len(file_list))
        for i, image in enumerate(file_list):
            try:
                logger.info("[WebhookImages] File[%d]: name=%s type=%s", i, getattr(image, "filename", None), getattr(image, "content_type", None))
            except Exception:
                pass

        # Store each image using blob storage
        from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
        from ...adapters.storage.azure_blob_service import get_azure_blob_service
        from ...adapters.db.mongo.repositories.blob_file_repository import BlobFileRepository
        
        blob_service = get_azure_blob_service()
        blob_repo = BlobFileRepository()
        
        for i, image in enumerate(file_list):
            try:
                # Validate content type
                raw_ct = (image.content_type or "").lower()
                content_type = raw_ct.split(";")[0].strip().strip(",")
                if content_type not in valid_types:
                    errors.append(f"Image {i+1}: Invalid file type {content_type}")
                    continue
                
                # Read image data
                content = await image.read()
                
                # Upload to blob storage
                blob_info = await blob_service.upload_file(
                    file_data=content,
                    filename=image.filename or f"image_{i+1}",
                    content_type=content_type,
                    file_type="image",
                    patient_id=str(internal_patient_id),
                    visit_id=str(visit_id)
                )
                
                # Create blob reference
                blob_reference = await blob_repo.create_blob_reference(
                    blob_path=blob_info["blob_path"],
                    container_name=blob_info["container_name"],
                    original_filename=image.filename or f"image_{i+1}",
                    content_type=content_type,
                    file_size=len(content),
                    blob_url=blob_info["blob_url"],
                    file_type="image",
                    category="medication",
                    patient_id=str(internal_patient_id),
                    visit_id=str(visit_id)
                )
                
                # Store DB record with blob reference
                doc = MedicationImageMongo(
                    patient_id=str(internal_patient_id),
                    visit_id=str(visit_id),
                    content_type=content_type,
                    filename=image.filename or f"image_{i+1}",
                    file_size=len(content),
                    blob_reference_id=blob_reference.file_id,
                )
                inserted = await doc.insert()
                
                uploaded_images.append({
                    "id": str(getattr(inserted, "id", "")),
                    "filename": image.filename or f"image_{i+1}",
                    "content_type": content_type,
                    "blob_url": blob_info["blob_url"],
                    "file_size": len(content)
                })
                logger.info("[WebhookImages] Stored image[%d] id=%s filename=%s", i, str(getattr(inserted, "id", "")), image.filename or f"image_{i+1}")
                
            except Exception as e:
                errors.append(f"Image {i+1}: {str(e)}")
                logger.error(f"Error uploading image {i+1}: {e}")
        
        return ok(request, data={
            "status": "success" if not errors else "partial_success",
            "patient_id": internal_patient_id,
            "visit_id": visit_id,
            "uploaded_images": uploaded_images,
            "errors": errors,
            "total_uploaded": len(uploaded_images),
            "total_errors": len(errors)
        })
    except Exception as e:
        logger.error("Unhandled error in upload_multiple_medication_images", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message="Failed to upload images")


# Get intake medication image content (with security validation)
@router.get("/{patient_id}/visits/{visit_id}/intake-images/{image_id}/content", include_in_schema=False)
async def get_intake_medication_image_content(
    request: Request,
    patient_id: str, 
    visit_id: str, 
    image_id: str
):
    """Get medication image content uploaded during intake with proper access control."""
    from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
    try:
        logger.info(f"[GetMedicationImage] Request: patient_id={patient_id[:50]}..., visit_id={visit_id}, image_id={image_id}")
        
        # Normalize patient ID
        try:
            internal_patient_id = decode_patient_id(patient_id)
            logger.info(f"[GetMedicationImage] Decoded patient_id: {internal_patient_id[:50]}...")
        except Exception as e:
            logger.warning(f"[GetMedicationImage] Could not decode patient_id, using as-is: {e}")
            internal_patient_id = patient_id
        
        # Find the image with security validation
        try:
            # Try to parse as ObjectId
            try:
                obj_id = PydanticObjectId(image_id)
            except Exception as parse_error:
                logger.error(f"[GetMedicationImage] Invalid ObjectId format: {image_id}, error: {parse_error}")
                raise HTTPException(
                    status_code=400,
                    detail={"error": "INVALID_IMAGE_ID", "message": f"Invalid image ID format: {image_id}"}
                )
            
            doc = await MedicationImageMongo.get(obj_id)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[GetMedicationImage] Failed to get image document with ID {image_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=404, 
                detail={"error": "NOT_FOUND", "message": f"Image not found: {image_id}"}
            )
            
        if not doc:
            logger.warning(f"[GetMedicationImage] Image document not found for ID: {image_id}")
            raise HTTPException(
                status_code=404, 
                detail={"error": "NOT_FOUND", "message": "Image not found"}
            )
        
        logger.info(f"[GetMedicationImage] Found image document: patient_id={doc.patient_id[:50] if len(str(doc.patient_id)) > 50 else doc.patient_id}..., visit_id={doc.visit_id}, blob_reference_id={doc.blob_reference_id}")
        
        # Security check: verify the image belongs to the specified patient and visit
        # Check both internal and encoded patient_id (images might be stored with either format)
        from ...core.utils.crypto import encode_patient_id
        patient_encoded_id = encode_patient_id(str(internal_patient_id))
        patient_id_matches = (
            doc.patient_id == str(internal_patient_id) or 
            doc.patient_id == patient_encoded_id or
            doc.patient_id == patient_id  # Also check original request ID
        )
        
        logger.info(f"[GetMedicationImage] Patient ID match check: doc.patient_id={doc.patient_id[:50] if len(str(doc.patient_id)) > 50 else doc.patient_id}, internal={str(internal_patient_id)[:50]}..., encoded={patient_encoded_id[:50]}..., request={patient_id[:50]}..., matches={patient_id_matches}")
        
        if not patient_id_matches or doc.visit_id != str(visit_id):
            logger.warning(f"[GetMedicationImage] Access denied: patient_id_match={patient_id_matches}, visit_match={doc.visit_id == str(visit_id)}")
            raise HTTPException(
                status_code=403, 
                detail={"error": "FORBIDDEN", "message": "Access denied to this image"}
            )
        
        # Fetch image from blob storage
        from ...adapters.db.mongo.repositories.blob_file_repository import BlobFileRepository
        from ...adapters.storage.azure_blob_service import get_azure_blob_service
        
        try:
            logger.info(f"[GetMedicationImage] Fetching blob reference: {doc.blob_reference_id}")
            blob_repo = BlobFileRepository()
            blob_service = get_azure_blob_service()
            
            # Get blob reference
            blob_ref = await blob_repo.get_blob_reference_by_id(doc.blob_reference_id)
            if not blob_ref:
                logger.error(f"[GetMedicationImage] Blob reference not found: {doc.blob_reference_id}")
                raise HTTPException(
                    status_code=404,
                    detail={"error": "NOT_FOUND", "message": "Blob reference not found"}
                )
            
            logger.info(f"[GetMedicationImage] Blob reference found: blob_path={blob_ref.blob_path}")
            
            # Download from blob storage
            file_data = await blob_service.download_file(
                blob_path=blob_ref.blob_path
            )
            
            logger.info(f"[GetMedicationImage] Successfully downloaded {len(file_data)} bytes from blob storage")
            
            return Response(
                content=file_data,
                media_type=doc.content_type or "application/octet-stream"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[GetMedicationImage] Error fetching image from blob storage: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"error": "INTERNAL_ERROR", "message": "Failed to retrieve image from storage"}
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error reading intake medication image", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=str(e))


# List uploaded images for a visit
@router.get("/{patient_id}/visits/{visit_id}/images", tags=["Intake + Pre-Visit Summary"])
async def list_medication_images(request: Request, patient_id: str, visit_id: str):
    from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
    from ...adapters.db.mongo.repositories.blob_file_repository import BlobFileRepository
    from ...adapters.storage.azure_blob_service import get_azure_blob_service
    try:
        original_patient_id = patient_id
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        candidate_patient_ids = {str(internal_patient_id), str(original_patient_id)}
        try:
            candidate_patient_ids.add(encode_patient_id(str(internal_patient_id)))
        except Exception:
            pass
        docs = await MedicationImageMongo.find(
            {
                "patient_id": {"$in": list(candidate_patient_ids)},
                "visit_id": str(visit_id),
            }
        ).to_list()
        
        # Fetch blob references for blob_url
        blob_repo = BlobFileRepository()
        blob_service = get_azure_blob_service()
        images_list = []
        for d in docs:
            blob_url = None
            signed_url = None
            try:
                if hasattr(d, "blob_reference_id") and d.blob_reference_id:
                    blob_ref = await blob_repo.get_blob_reference_by_id(d.blob_reference_id)
                    if blob_ref:
                        blob_url = blob_ref.blob_url
                        if getattr(blob_ref, "blob_path", None):
                            try:
                                signed_url = blob_service.generate_signed_url(
                                    blob_path=blob_ref.blob_path,
                                    expires_in_hours=1,
                                )
                            except Exception as sas_error:
                                logger.warning(
                                    "Failed to generate SAS URL for blob %s: %s",
                                    blob_ref.blob_path,
                                    sas_error,
                                )
            except Exception as e:
                logger.warning(f"Failed to fetch blob URL for image {getattr(d, 'id', 'unknown')}: {e}")
            
            images_list.append({
                "id": str(getattr(d, "id", "")),
                "filename": getattr(d, "filename", "unknown"),
                "content_type": getattr(d, "content_type", ""),
                "file_size": getattr(d, "file_size", 0),
                "blob_url": blob_url,
                "signed_url": signed_url,
                "uploaded_at": getattr(d, "uploaded_at", None),
            })
        
        return ok(request, data={
            "patient_id": internal_patient_id,
            "visit_id": visit_id,
            "images": images_list,
        })
    except Exception as e:
        logger.error("Error listing medication images", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=str(e))


# Delete one uploaded image by id
@router.delete("/images/{image_id}", tags=["Intake + Pre-Visit Summary"])
async def delete_medication_image(request: Request, image_id: str):
    from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
    try:
        doc = await MedicationImageMongo.get(PydanticObjectId(image_id))
        if not doc:
            raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": "Image not found"})
        await doc.delete()
        return {"status": "deleted", "id": image_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting medication image", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=str(e))


@router.post(
    "/{patient_id}/visits/{visit_id}/intake/reset",
    response_model=ApiResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def reset_intake_session(
    request: Request,
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """Reset intake session - clear all questions and start fresh."""
    try:
        # Decode patient ID if needed
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        
        # Get patient and visit
        from ...domain.value_objects.patient_id import PatientId
        from ...domain.value_objects.visit_id import VisitId
        
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id))
        if not patient:
            return fail(request, error="PATIENT_NOT_FOUND", message=f"Patient {patient_id} not found", status_message=status.HTTP_404_NOT_FOUND)
        
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(internal_patient_id, visit_id_obj)
        if not visit:
            return fail(request, error="VISIT_NOT_FOUND", message=f"Visit {visit_id} not found", status_message=status.HTTP_404_NOT_FOUND)
        
        # Reset intake session by truncating all questions (pass -1 to truncate_after to clear all)
        if visit.intake_session:
            visit.intake_session.truncate_after(-1)
            visit.symptom = ""  # Reset symptom too
            visit.status = "intake"  # Reset visit status
            visit.updated_at = datetime.utcnow()
            await visit_repo.save(visit)
            logger.info(f"Reset intake session for patient {internal_patient_id}, visit {visit_id}")
        
        return ok(request, data={"message": "Intake session reset successfully", "patient_id": internal_patient_id, "visit_id": visit_id})
        
    except Exception as e:
        logger.error("Unhandled error in reset_intake_session", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=str(e))


@router.get(
    "/{patient_id}/visits/{visit_id}/intake/status",
    response_model=ApiResponse[IntakeSummarySchema],
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    responses={
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_intake_status(
    request: Request,
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """Get the current status and history of an intake session for a given patient and visit."""
    from ...domain.value_objects.patient_id import PatientId
    from ...domain.value_objects.visit_id import VisitId
    from ..schemas.common import QuestionAnswer
    
    try:
        # Decode patient ID if needed
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id))
        if not patient:
            return fail(request, error="PATIENT_NOT_FOUND", message=f"Patient {patient_id} not found", status_message=status.HTTP_404_NOT_FOUND)

        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(internal_patient_id, visit_id_obj)
        if not visit:
            return fail(request, error="VISIT_NOT_FOUND", message=f"Visit {visit_id} not found for patient {patient_id}", status_message=status.HTTP_404_NOT_FOUND)

        if not visit.intake_session:
            return fail(request, error="INTAKE_SESSION_NOT_FOUND", message="No intake session found for this visit", status_message=status.HTTP_404_NOT_FOUND)

        intake_session = visit.intake_session
        questions_asked_schema = [
            QuestionAnswer(
                question_id=qa.question_id.value,
                question=qa.question,
                answer=qa.answer,
                timestamp=qa.timestamp,
                question_number=qa.question_number
            ) for qa in intake_session.questions_asked
        ]

        response_data = IntakeSummarySchema(
            visit_id=visit_id,
            status=visit.status,
            questions_asked=questions_asked_schema,
            total_questions=intake_session.current_question_count,
            max_questions=intake_session.max_questions,
            intake_status=intake_session.status,
            started_at=intake_session.started_at,
            completed_at=intake_session.completed_at,
            pending_question=intake_session.pending_question
        )
        return ok(request, data=response_data)

    except Exception as e:
        logger.error("Error getting intake status", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message="An unexpected error occurred while retrieving intake status")


@router.post(
    "/summary/previsit",
    response_model=PreVisitSummaryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Intake + Pre-Visit Summary"],
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "Intake not completed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def generate_pre_visit_summary(
    http_request: Request,
    request: PreVisitSummaryRequest,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
    question_service: QuestionServiceDep,
):
    """
    Generate pre-visit clinical summary from completed intake data.

    This endpoint:
    1. Validates patient and visit exist
    2. Checks intake is completed
    3. Generates AI-powered clinical summary
    4. Returns structured summary for doctor review
    """
    try:
        # Set IDs in request state for HIPAA audit middleware
        http_request.state.audit_patient_id = request.patient_id
        http_request.state.audit_visit_id = request.visit_id
        
        # Extract doctor_id
        doctor_id = getattr(http_request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                http_request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Convert Pydantic model to DTO - handle patient_id decoding gracefully
        try:
            decoded_patient_id = decode_patient_id(request.patient_id)
        except (ValueError, Exception) as decode_error:
            # If decoding fails, try using the patient_id as-is (might already be decoded)
            logger.warning(
                f"Failed to decode patient_id {request.patient_id[:20] if request.patient_id else 'None'}..., "
                f"using as-is: {decode_error}"
            )
            decoded_patient_id = request.patient_id
        
        dto_request = PreVisitSummaryRequest(
            patient_id=decoded_patient_id,
            visit_id=request.visit_id,
        )

        # Execute use case
        use_case = GeneratePreVisitSummaryUseCase(patient_repo, visit_repo, question_service)
        result = await use_case.execute(dto_request, doctor_id=doctor_id)

        return PreVisitSummaryResponse(
            patient_id=encode_patient_id(result.patient_id),
            visit_id=result.visit_id,
            summary=result.summary,
            generated_at=result.generated_at,
        )

    except ValueError as e:
        logger.error(f"ValueError in generate_pre_visit_summary: {e}", exc_info=True)
        return fail(
            http_request, 
            error="INTAKE_NOT_COMPLETED", 
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except PatientNotFoundError as e:
        logger.error(f"PatientNotFoundError in generate_pre_visit_summary: patient_id={request.patient_id[:20] if request.patient_id else None}, error={e.message}")
        return fail(
            http_request, 
            error="PATIENT_NOT_FOUND", 
            message=e.message,
            status_code=status.HTTP_404_NOT_FOUND
        )
    except VisitNotFoundError as e:
        logger.error(f"VisitNotFoundError in generate_pre_visit_summary: visit_id={request.visit_id}, error={e.message}")
        return fail(
            http_request, 
            error="VISIT_NOT_FOUND", 
            message=e.message,
            status_code=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        # Log the full error with traceback and context for debugging
        error_type = type(e).__name__
        error_message = str(e)
        logger.error(
            f"Unhandled error in generate_pre_visit_summary: {error_type}: {error_message}",
            exc_info=True,
            extra={
                "patient_id": request.patient_id[:20] if request.patient_id else None,
                "visit_id": request.visit_id,
                "error_type": error_type,
                "error_message": error_message,
            }
        )
        # Return more helpful error message in development, generic in production
        import os
        is_dev = os.getenv("APP_ENV", "production") == "development" or os.getenv("DEBUG", "false").lower() == "true"
        error_msg = f"{error_type}: {error_message}" if is_dev else "An unexpected error occurred"
        return fail(
            http_request, 
            error="INTERNAL_ERROR", 
            message=error_msg,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get(
    "/{patient_id}/visits/{visit_id}/summary",
    response_model=PreVisitSummaryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Intake + Pre-Visit Summary"],
    responses={
        404: {"model": ErrorResponse, "description": "Patient, visit, or summary not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_pre_visit_summary(
    request: Request,
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """
    Retrieve stored pre-visit summary from EHR.

    This endpoint:
    1. Validates patient and visit exist
    2. Retrieves stored pre-visit summary from EHR
    3. Returns the clinical summary for doctor review
    """
    try:
        from ...domain.value_objects.patient_id import PatientId
        
        # Get doctor_id for tenant isolation
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        # Find patient (decode opaque id from client)
        internal_patient_id = decode_patient_id(patient_id)
        patient_id_obj = PatientId(internal_patient_id)
        patient = await patient_repo.find_by_id(patient_id_obj, doctor_id)
        if not patient:
            raise PatientNotFoundError(patient_id)

        # Find visit using VisitRepository
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj, doctor_id
        )
        if not visit:
            raise VisitNotFoundError(visit_id)

        # Check if summary exists, if not try to generate it
        if not visit.has_pre_visit_summary():
            logger.warning(f"No pre-visit summary found for visit {visit_id}, attempting to generate...")
            
            # Check if intake is completed
            if not visit.is_intake_complete():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "INTAKE_NOT_COMPLETED",
                        "message": f"Intake not completed for visit {visit_id}",
                        "details": {"visit_id": visit_id},
                    },
                )
            
            # Try to generate summary on-demand
            try:
                from ...application.use_cases.generate_pre_visit_summary import GeneratePreVisitSummaryUseCase
                from ...application.dto.patient_dto import PreVisitSummaryRequest
                from ...adapters.external.question_service_openai import OpenAIQuestionService
                
                # Create question service directly instead of using container
                question_service = OpenAIQuestionService()
                
                summary_use_case = GeneratePreVisitSummaryUseCase(patient_repo, visit_repo, question_service)
                # Pass the original patient_id (encoded) to use case so it can query images correctly
                summary_request = PreVisitSummaryRequest(
                    patient_id=patient_id,  # Use original encoded patient_id from request
                    visit_id=visit_id,
                )
                
                result = await summary_use_case.execute(summary_request)
                logger.info(f"Successfully generated pre-visit summary for visit {visit_id}")
                
                # Attach images explicitly in case use case didn't find them
                if not result.medication_images:
                    logger.warning(f"[GetPreVisitSummary] Use case returned no images, querying directly...")
                    from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
                    from ...adapters.db.mongo.repositories.blob_file_repository import BlobFileRepository
                    from ...adapters.storage.azure_blob_service import get_azure_blob_service
                    from beanie.operators import Or
                    from ...core.utils.crypto import encode_patient_id
                    
                    patient_internal_id = str(patient.patient_id.value)
                    patient_encoded_id = encode_patient_id(patient_internal_id)
                    
                    docs = await MedicationImageMongo.find(
                        Or(
                            MedicationImageMongo.patient_id == patient_internal_id,
                            MedicationImageMongo.patient_id == patient_encoded_id,
                            MedicationImageMongo.patient_id == patient_id  # Original request ID
                        ),
                        MedicationImageMongo.visit_id == visit_id,
                    ).to_list()
                    
                    if docs:
                        blob_repo = BlobFileRepository()
                        blob_service = get_azure_blob_service()
                        enriched_images = []
                        for d in docs:
                            signed_url = None
                            try:
                                if getattr(d, "blob_reference_id", None):
                                    blob_ref = await blob_repo.get_blob_reference_by_id(d.blob_reference_id)
                                    if blob_ref and getattr(blob_ref, "blob_path", None):
                                        signed_url = blob_service.generate_signed_url(
                                            blob_path=blob_ref.blob_path,
                                            expires_in_hours=1,
                                        )
                            except Exception as sas_error:
                                logger.warning(
                                    "Failed to generate SAS URL for medication image %s: %s",
                                    getattr(d, "id", "unknown"),
                                    sas_error,
                                )
                            enriched_images.append(
                                {
                                    "id": str(getattr(d, "id", "")),
                                    "filename": getattr(d, "filename", "unknown"),
                                    "content_type": getattr(d, "content_type", ""),
                                    "signed_url": signed_url,
                                }
                            )
                        result.medication_images = enriched_images
                        logger.info(f"[GetPreVisitSummary] Found {len(result.medication_images)} images in direct query")
                
                return PreVisitSummaryResponse(
                    patient_id=encode_patient_id(result.patient_id),
                    visit_id=result.visit_id,
                    summary=result.summary,
                    generated_at=result.generated_at,
                    medication_images=result.medication_images,
                    red_flags=result.red_flags,
                )
                
            except Exception as e:
                logger.error(f"Failed to generate pre-visit summary for visit {visit_id}: {e}")
                return fail(request, error="SUMMARY_GENERATION_FAILED", message=f"Failed to generate pre-visit summary for visit {visit_id}")

        # Get stored summary
        summary_data = visit.get_pre_visit_summary()

        # Attach any uploaded medication images
        from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
        from ...adapters.db.mongo.repositories.blob_file_repository import BlobFileRepository
        from ...adapters.storage.azure_blob_service import get_azure_blob_service
        from beanie.operators import Or
        from ...core.utils.crypto import encode_patient_id
        
        # Try both internal ID and encoded ID (in case images were stored with encoded ID)
        patient_internal_id = str(patient.patient_id.value)
        patient_encoded_id = encode_patient_id(patient_internal_id)
        
        # Also try with the original patient_id from request (might be encoded)
        # This handles cases where images were stored with the exact encoded ID from the request
        docs = await MedicationImageMongo.find(
            Or(
                MedicationImageMongo.patient_id == patient_internal_id,
                MedicationImageMongo.patient_id == patient_encoded_id,
                MedicationImageMongo.patient_id == patient_id  # Also check the original request ID
            ),
            MedicationImageMongo.visit_id == visit.visit_id.value,
        ).to_list()
        
        logger.info(f"[GetPreVisitSummary] Querying images for visit {visit.visit_id.value} with patient_id variants: internal={patient_internal_id[:20]}..., encoded={patient_encoded_id[:20]}..., request={patient_id[:20]}...")
        logger.info(f"[GetPreVisitSummary] Found {len(docs) if docs else 0} medication images")
        
        images_meta = None
        if docs:
            blob_repo = BlobFileRepository()
            blob_service = get_azure_blob_service()
            images_meta = []
            for d in docs:
                signed_url = None
                try:
                    if getattr(d, "blob_reference_id", None):
                        blob_ref = await blob_repo.get_blob_reference_by_id(d.blob_reference_id)
                        if blob_ref and getattr(blob_ref, "blob_path", None):
                            signed_url = blob_service.generate_signed_url(
                                blob_path=blob_ref.blob_path,
                                expires_in_hours=1,
                            )
                except Exception as sas_error:
                    logger.warning(
                        "Failed to generate SAS URL for medication image %s: %s",
                        getattr(d, "id", "unknown"),
                        sas_error,
                    )
                images_meta.append(
                    {
                        "id": str(getattr(d, "id", "")),
                        "filename": getattr(d, "filename", "unknown"),
                        "content_type": getattr(d, "content_type", ""),
                        "signed_url": signed_url,
                    }
                )
        
        if images_meta:
            logger.info(f"[GetPreVisitSummary] Returning {len(images_meta)} medication images: {[img['filename'] for img in images_meta]}")

        return PreVisitSummaryResponse(
            patient_id=encode_patient_id(patient.patient_id.value),
            visit_id=visit.visit_id.value,
            summary=summary_data["summary"],
            generated_at=summary_data["generated_at"],
            medication_images=images_meta,
            red_flags=summary_data.get("red_flags", []),
        )

    except PatientNotFoundError as e:
        return fail(request, error="PATIENT_NOT_FOUND", message=e.message)
    except VisitNotFoundError as e:
        return fail(request, error="VISIT_NOT_FOUND", message=e.message)
    except Exception as e:
        logger.error("Unhandled error in get_pre_visit_summary", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message="An unexpected error occurred")


@router.post(
    "/summary/postvisit",
    response_model=ApiResponse[PostVisitSummaryResponse],
    status_code=status.HTTP_200_OK,
    tags=["Post-Visit Summary"],
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "Visit not completed or SOAP note not available"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def generate_post_visit_summary(
    http_request: Request,
    request: PostVisitSummaryRequest,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
    soap_service: SoapServiceDep,
):
    """
    Generate post-visit summary for patient sharing.
    
    This endpoint:
    1. Validates patient and visit exist
    2. Checks if visit is completed with SOAP note
    3. Generates patient-friendly post-visit summary
    4. Returns structured summary for WhatsApp sharing
    """
    try:
        # Get doctor_id from request state
        doctor_id = getattr(http_request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                http_request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        # Set IDs in request state for HIPAA audit middleware
        http_request.state.audit_patient_id = request.patient_id
        http_request.state.audit_visit_id = request.visit_id
        
        # Decode the opaque patient_id to get the internal patient_id
        try:
            internal_patient_id = decode_patient_id(request.patient_id)
            logger.info(f"Decoded patient_id: {internal_patient_id}")
        except Exception as decode_error:
            logger.error(f"Failed to decode patient_id: {request.patient_id}, error: {decode_error}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "INVALID_PATIENT_ID",
                    "message": "Invalid patient ID format",
                    "details": {"patient_id": request.patient_id}
                }
            )
        
        logger.info(f"Generating post-visit summary for internal_patient_id: {internal_patient_id}, visit_id: {request.visit_id}, doctor_id: {doctor_id}")
        
        # Create request with decoded patient_id
        decoded_request = PostVisitSummaryRequest(
            patient_id=internal_patient_id,
            visit_id=request.visit_id
        )
        
        # Create use case instance (patient_repo, visit_repo, soap_service)
        use_case = GeneratePostVisitSummaryUseCase(patient_repo, visit_repo, soap_service)
        
        # Execute use case - pass doctor_id as second argument
        result = await use_case.execute(decoded_request, doctor_id)
        
        return ok(http_request, data=result)
        
    except PatientNotFoundError as e:
        return fail(http_request, error="PATIENT_NOT_FOUND", message=f"Patient {request.patient_id} not found")
    except VisitNotFoundError as e:
        return fail(http_request, error="VISIT_NOT_FOUND", message=f"Visit {request.visit_id} not found")
    except ValueError as e:
        return fail(http_request, error="INVALID_VISIT_STATE", message=str(e))
    except Exception as e:
        logger.error("Unhandled error in generate_post_visit_summary", exc_info=True)
        return fail(http_request, error="INTERNAL_ERROR", message="An unexpected error occurred")


@router.get(
    "/{patient_id}/visits/{visit_id}/summary/postvisit",
    response_model=ApiResponse[PostVisitSummaryResponse],
    status_code=status.HTTP_200_OK,
    tags=["Post-Visit Summary"]
)
async def get_post_visit_summary(
    request: Request,
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """Retrieve stored post-visit summary from visit (if available)."""
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
        
        from ...domain.value_objects.patient_id import PatientId
        
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception as e:
            internal_patient_id = patient_id
            
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id), doctor_id)
        if not patient:
            raise PatientNotFoundError(patient_id)
        
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj, doctor_id
        )
        if not visit:
            raise VisitNotFoundError(visit_id)
        
        data = visit.get_post_visit_summary()
            
        if not data:
            raise HTTPException(status_code=404, detail={"error": "POST_VISIT_SUMMARY_NOT_FOUND", "message": "No post-visit summary stored"})
        return ok(request, data=PostVisitSummaryResponse(**data))
    except PatientNotFoundError as e:
        return fail(request, error="PATIENT_NOT_FOUND", message=e.message)
    except VisitNotFoundError as e:
        return fail(request, error="VISIT_NOT_FOUND", message=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled error in get_post_visit_summary", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=str(e))


# ------------------------
# Vitals endpoints
# ------------------------

class VitalsPayload(BaseModel):
    """Payload for storing vitals data."""
    bloodPressure: Optional[str] = Field(None, description="Blood pressure")
    heartRate: Optional[str] = Field(None, description="Heart rate")
    temperature: Optional[str] = Field(None, description="Temperature")
    respiratoryRate: Optional[str] = Field(None, description="Respiratory rate")
    oxygenSaturation: Optional[str] = Field(None, description="Oxygen saturation")
    weight: Optional[str] = Field(None, description="Weight")
    height: Optional[str] = Field(None, description="Height")
    bmi: Optional[str] = Field(None, description="BMI")
    notes: Optional[str] = Field(None, description="Additional notes")


@router.post(
    "/{patient_id}/visits/{visit_id}/vitals",
    status_code=status.HTTP_200_OK,
    tags=["Vitals and Transcript Generation"],
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def store_vitals(
    request: Request,
    patient_id: str,
    visit_id: str,
    vitals: VitalsPayload,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """Store vitals data for a visit."""
    try:
        # Get doctor_id from request state
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "MISSING_DOCTOR_ID",
                    "message": "X-Doctor-ID header is required",
                    "details": {},
                },
            )
        
        from ...domain.value_objects.patient_id import PatientId
        from ...domain.value_objects.visit_id import VisitId
        
        # Decode patient ID if needed
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        
        # Get patient and visit - with doctor_id for data isolation
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id), doctor_id)
        if not patient:
            raise HTTPException(
                status_code=404,
                detail={"error": "PATIENT_NOT_FOUND", "message": f"Patient {patient_id} not found", "details": {}}
            )
        
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj, doctor_id
        )
        if not visit:
            raise HTTPException(
                status_code=404,
                detail={"error": "VISIT_NOT_FOUND", "message": f"Visit {visit_id} not found", "details": {}}
            )
        
        # Convert vitals to dict format
        vitals_dict = {
            "blood_pressure": vitals.bloodPressure,
            "heart_rate": vitals.heartRate,
            "temperature": vitals.temperature,
            "respiratory_rate": vitals.respiratoryRate,
            "oxygen_saturation": vitals.oxygenSaturation,
            "weight": vitals.weight,
            "height": vitals.height,
            "bmi": vitals.bmi,
            "notes": vitals.notes,
            "recorded_at": datetime.utcnow().isoformat()
        }
        
        # Store vitals in visit
        visit.store_vitals(vitals_dict)
        
        # Update visit status based on workflow type
        visit.complete_vitals()  # Handles both walk-in and scheduled workflows
        
        # Additional status updates for scheduled visits with existing transcripts
        if visit.is_scheduled_workflow() and visit.is_transcription_complete():
            # If transcript exists, update to soap_generation
            if visit.status not in ["soap_generation", "prescription_analysis", "completed"]:
                visit.status = "soap_generation"
        
        # Persist visit (not patient)
        await visit_repo.save(visit)
        
        return ok(request, data={"success": True, "message": "Vitals stored successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled error in store_vitals", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=str(e))


@router.get(
    "/{patient_id}/visits/{visit_id}/vitals",
    status_code=status.HTTP_200_OK,
    tags=["Vitals and Transcript Generation"],
    responses={
        404: {"model": ErrorResponse, "description": "Patient, visit, or vitals not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_vitals(
    request: Request,
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """Get vitals data for a visit."""
    try:
        # Get doctor_id from request state
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "MISSING_DOCTOR_ID",
                    "message": "X-Doctor-ID header is required",
                    "details": {},
                },
            )
        
        from ...domain.value_objects.patient_id import PatientId
        
        # Decode patient ID if needed
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        
        # Get patient and visit - with doctor_id for data isolation
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id), doctor_id)
        if not patient:
            raise HTTPException(
                status_code=404,
                detail={"error": "PATIENT_NOT_FOUND", "message": f"Patient {patient_id} not found", "details": {}}
            )
        
        from ...domain.value_objects.visit_id import VisitId
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(
            internal_patient_id, visit_id_obj, doctor_id
        )
        if not visit:
            raise HTTPException(
                status_code=404,
                detail={"error": "VISIT_NOT_FOUND", "message": f"Visit {visit_id} not found", "details": {}}
            )
        
        if not visit.vitals:
            raise HTTPException(
                status_code=404,
                detail={"error": "VITALS_NOT_FOUND", "message": "No vitals found for this visit", "details": {}}
            )
        
        return ok(request, data=visit.vitals)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled error in get_vitals", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=str(e))


@router.get(
    "/{patient_id}/visits",
    response_model=ApiResponse[VisitListResponse],
    status_code=status.HTTP_200_OK,
    tags=["Patient Management"],
    summary="Get all visits for a patient",
    responses={
        200: {"description": "List of visits retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Patient not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_patient_visits(
    request: Request,
    patient_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """
    Get all visits for a specific patient.
    
    Returns a list of visits with summary information including:
    - Visit ID, status, workflow type
    - Timestamps
    - Flags indicating what data is available (transcript, SOAP, vitals, etc.)
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
        
        from ...domain.value_objects.patient_id import PatientId
        import urllib.parse
        
        # URL decode the patient_id in case it's URL encoded
        decoded_path_param = urllib.parse.unquote(patient_id)
        
        # Check if this looks like an internal patient ID (format: name_mobile)
        # If it contains underscore and the part after underscore is all digits, treat as internal ID
        if '_' in decoded_path_param:
            parts = decoded_path_param.split('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                # This looks like an internal patient ID, skip decryption
                internal_patient_id = decoded_path_param
                logger.debug(f"Using internal patient ID: {internal_patient_id}")
            else:
                # Try to decrypt as opaque token
                try:
                    internal_patient_id = decode_patient_id(decoded_path_param)
                except Exception as e:
                    logger.warning(f"Failed to decode patient_id '{decoded_path_param}': {e}")
                    internal_patient_id = decoded_path_param
        else:
            # Try to decrypt as opaque token
            try:
                internal_patient_id = decode_patient_id(decoded_path_param)
            except Exception as e:
                logger.warning(f"Failed to decode patient_id '{decoded_path_param}': {e}")
                internal_patient_id = decoded_path_param
        
        # Create PatientId value object with proper error handling
        try:
            patient_id_obj = PatientId(internal_patient_id)
        except ValueError as ve:
            logger.error(f"Invalid patient ID format: {ve}")
            return fail(
                request,
                error="INVALID_PATIENT_ID",
                message=f"Invalid patient ID format: {str(ve)}",
                status_message=status.HTTP_422_UNPROCESSABLE_ENTITY
            )
        
        # Verify patient exists - with doctor_id for data isolation
        patient = await patient_repo.find_by_id(patient_id_obj, doctor_id)
        if not patient:
            return fail(
                request,
                error="PATIENT_NOT_FOUND",
                message=f"Patient {patient_id} not found",
                status_message=status.HTTP_404_NOT_FOUND
            )
        
        # Get all visits for this patient - with doctor_id for data isolation
        visits = await visit_repo.find_by_patient_id(internal_patient_id, doctor_id)
        
        # Convert to schema
        visit_items = []
        for visit in visits:
            # Ensure all boolean values are explicitly converted to bool
            # Python's 'and' operator in is_transcription_complete() can return None
            # when transcription_session is None, so we explicitly handle None cases
            try:
                transcript_result = visit.is_transcription_complete()
                has_transcript = transcript_result if isinstance(transcript_result, bool) else False
            except (AttributeError, TypeError):
                has_transcript = False
            
            try:
                soap_result = visit.is_soap_generated()
                has_soap = soap_result if isinstance(soap_result, bool) else False
            except (AttributeError, TypeError):
                has_soap = False
            
            try:
                has_vitals = bool(visit.vitals is not None)
            except (AttributeError, TypeError):
                has_vitals = False
            
            try:
                pre_visit_result = visit.has_pre_visit_summary()
                has_pre_visit_summary = pre_visit_result if isinstance(pre_visit_result, bool) else False
            except (AttributeError, TypeError):
                has_pre_visit_summary = False
            
            try:
                post_visit_result = visit.has_post_visit_summary()
                has_post_visit_summary = post_visit_result if isinstance(post_visit_result, bool) else False
            except (AttributeError, TypeError):
                has_post_visit_summary = False
            
            visit_items.append(VisitListItemSchema(
                visit_id=visit.visit_id.value,
                symptom=visit.symptom or "",
                workflow_type=visit.workflow_type.value,
                status=visit.status,
                created_at=visit.created_at,
                updated_at=visit.updated_at,
                has_transcript=has_transcript,
                has_soap=has_soap,
                has_vitals=has_vitals,
                has_pre_visit_summary=has_pre_visit_summary,
                has_post_visit_summary=has_post_visit_summary,
            ))
        
        return ok(
            request,
            data=VisitListResponse(
                visits=visit_items,
                total=len(visit_items)
            ),
            message="Visits retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error listing patient visits: {e}", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=f"An unexpected error occurred: {str(e)}")


@router.get(
    "/{patient_id}/visits/{visit_id}",
    response_model=ApiResponse[VisitDetailSchema],
    status_code=status.HTTP_200_OK,
    tags=["Patient Management"],
    summary="Get full details of a specific visit",
    responses={
        200: {"description": "Visit details retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_visit_detail(
    request: Request,
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
    visit_repo: VisitRepositoryDep,
):
    """
    Get full details of a specific visit.
    
    Returns complete visit information including:
    - Visit metadata (ID, status, timestamps, workflow type)
    - Intake session (if scheduled workflow)
    - Pre-visit summary
    - Transcription data (transcript, structured dialogue, audio file info)
    - SOAP note
    - Vitals
    - Post-visit summary
    - Associated audio files
    """
    try:
        from ...domain.value_objects.patient_id import PatientId
        from ...domain.value_objects.visit_id import VisitId
        from ...adapters.db.mongo.repositories.audio_repository import AudioRepository
        import urllib.parse
        
        # URL decode the patient_id in case it's URL encoded
        decoded_path_param = urllib.parse.unquote(patient_id)
        
        # Check if this looks like an internal patient ID (format: name_mobile)
        # If it contains underscore and the part after underscore is all digits, treat as internal ID
        if '_' in decoded_path_param:
            parts = decoded_path_param.split('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                # This looks like an internal patient ID, skip decryption
                internal_patient_id = decoded_path_param
                logger.debug(f"Using internal patient ID: {internal_patient_id}")
            else:
                # Try to decrypt as opaque token
                try:
                    internal_patient_id = decode_patient_id(decoded_path_param)
                except Exception as e:
                    logger.warning(f"Failed to decode patient_id '{decoded_path_param}': {e}")
                    internal_patient_id = decoded_path_param
        else:
            # Try to decrypt as opaque token
            try:
                internal_patient_id = decode_patient_id(decoded_path_param)
            except Exception as e:
                logger.warning(f"Failed to decode patient_id '{decoded_path_param}': {e}")
                internal_patient_id = decoded_path_param
        
        # Create PatientId value object with proper error handling
        try:
            patient_id_obj = PatientId(internal_patient_id)
        except ValueError as ve:
            logger.error(f"Invalid patient ID format: {ve}")
            return fail(
                request,
                error="INVALID_PATIENT_ID",
                message=f"Invalid patient ID format: {str(ve)}",
                status_message=status.HTTP_422_UNPROCESSABLE_ENTITY
            )
        
        # Get doctor_id from request state
        doctor_id = getattr(request.state, "doctor_id", None)
        if not doctor_id:
            return fail(
                request,
                error="MISSING_DOCTOR_ID",
                message="X-Doctor-ID header is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Verify patient exists
        patient = await patient_repo.find_by_id(patient_id_obj, doctor_id)
        if not patient:
            return fail(
                request,
                error="PATIENT_NOT_FOUND",
                message=f"Patient {patient_id} not found",
                status_message=status.HTTP_404_NOT_FOUND
            )
        
        # Get visit
        visit_id_obj = VisitId(visit_id)
        visit = await visit_repo.find_by_patient_and_visit_id(internal_patient_id, visit_id_obj, doctor_id)
        if not visit:
            return fail(
                request,
                error="VISIT_NOT_FOUND",
                message=f"Visit {visit_id} not found for patient {patient_id}",
                status_message=status.HTTP_404_NOT_FOUND
            )
        
        # Format intake session
        intake_session_data = None
        if visit.intake_session:
            intake_session_data = {
                "symptom": visit.symptom,
                "questions_asked": [
                    {
                        "question_id": qa.question_id.value,
                        "question": qa.question,
                        "answer": qa.answer,
                        "timestamp": qa.timestamp.isoformat(),
                        "question_number": qa.question_number,
                    }
                    for qa in visit.intake_session.questions_asked
                ],
                "total_questions": visit.intake_session.current_question_count,
                "max_questions": visit.intake_session.max_questions,
                "status": visit.intake_session.status,
                "started_at": visit.intake_session.started_at.isoformat(),
                "completed_at": visit.intake_session.completed_at.isoformat() if visit.intake_session.completed_at else None,
                "pending_question": visit.intake_session.pending_question,
            }
        
        # Format transcription session
        transcription_session_data = None
        if visit.transcription_session:
            transcription_session_data = TranscriptionSessionSchema(
                audio_file_path=visit.transcription_session.audio_file_path,
                transcript=visit.transcription_session.transcript,
                transcription_status=visit.transcription_session.transcription_status,
                started_at=visit.transcription_session.started_at.isoformat() if visit.transcription_session.started_at else None,
                completed_at=visit.transcription_session.completed_at.isoformat() if visit.transcription_session.completed_at else None,
                error_message=visit.transcription_session.error_message,
                audio_duration_seconds=visit.transcription_session.audio_duration_seconds,
                word_count=visit.transcription_session.word_count,
                structured_dialogue=visit.transcription_session.structured_dialogue,
            )
        
        # Format SOAP note
        soap_note_data = None
        if visit.soap_note:
            soap_note_data = SoapNoteSchema(
                subjective=visit.soap_note.subjective,
                objective=visit.soap_note.objective,
                assessment=visit.soap_note.assessment,
                plan=visit.soap_note.plan,
                highlights=visit.soap_note.highlights,
                red_flags=visit.soap_note.red_flags,
                generated_at=visit.soap_note.generated_at.isoformat(),
                model_info=visit.soap_note.model_info,
                confidence_score=visit.soap_note.confidence_score,
            )
        
        # Note: Audio files are stored internally for transcription but not exposed in API
        
        # Build response
        visit_detail = VisitDetailSchema(
            visit_id=visit.visit_id.value,
            patient_id=encode_patient_id(internal_patient_id),
            symptom=visit.symptom or "",
            workflow_type=visit.workflow_type.value,
            status=visit.status,
            created_at=visit.created_at,
            updated_at=visit.updated_at,
            intake_session=intake_session_data,
            pre_visit_summary=visit.pre_visit_summary,
            transcription_session=transcription_session_data,
            soap_note=soap_note_data,
            vitals=visit.vitals,
            post_visit_summary=visit.post_visit_summary,
        )
        
        return ok(
            request,
            data=visit_detail,
            message="Visit details retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting visit detail: {e}", exc_info=True)
        return fail(request, error="INTERNAL_ERROR", message=f"An unexpected error occurred: {str(e)}")

 