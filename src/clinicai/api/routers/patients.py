"""Patient-related API endpoints.

Formatting-only changes; behavior preserved.
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, Request, Form, File
import logging
import traceback
from typing import Union, Optional, List

from clinicai.application.dto.patient_dto import (
    AnswerIntakeRequest,
    PreVisitSummaryRequest,
    PostVisitSummaryRequest,
    RegisterPatientRequest,
)
from clinicai.application.dto.patient_dto import EditAnswerRequest
from clinicai.application.use_cases.answer_intake import AnswerIntakeUseCase
from clinicai.application.use_cases.generate_pre_visit_summary import GeneratePreVisitSummaryUseCase
from clinicai.application.use_cases.generate_post_visit_summary import GeneratePostVisitSummaryUseCase
from clinicai.core.utils.crypto import decode_patient_id
from clinicai.application.use_cases.register_patient import RegisterPatientUseCase
from clinicai.domain.errors import (
    DuplicatePatientError,
    DuplicateQuestionError,
    IntakeAlreadyCompletedError,
    InvalidDiseaseError,
    PatientNotFoundError,
    QuestionLimitExceededError,
    VisitNotFoundError,
)

from ..deps import PatientRepositoryDep, QuestionServiceDep, SoapServiceDep
from ...core.utils.crypto import encode_patient_id, decode_patient_id
from ..schemas.patient import AnswerIntakeResponse, ErrorResponse
from ..schemas.patient import AnswerIntakeRequest as AnswerIntakeRequestSchema
from ..schemas.patient import (
    PatientSummarySchema,
    PreVisitSummaryResponse,
    PostVisitSummaryResponse,
    RegisterPatientResponse,
    EditAnswerRequest as EditAnswerRequestSchema,
    EditAnswerResponse as EditAnswerResponseSchema,
)
from ..schemas.patient import RegisterPatientRequest as RegisterPatientRequestSchema
from fastapi import UploadFile, File, Form
from fastapi.responses import Response
from pathlib import Path
from datetime import datetime
import os
from beanie import PydanticObjectId

router = APIRouter(prefix="/patients", tags=["patients"])
logger = logging.getLogger("clinicai")


@router.post(
    "/",
    response_model=RegisterPatientResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        409: {"model": ErrorResponse, "description": "Duplicate patient"},
        422: {"model": ErrorResponse, "description": "Invalid symptom"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def register_patient(
    request: RegisterPatientRequestSchema,
    patient_repo: PatientRepositoryDep,
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
        # Convert Pydantic model to DTO
        full_name = f"{request.first_name.strip()} {request.last_name.strip()}".strip()
        dto_request = RegisterPatientRequest(
            name=full_name,
            mobile=request.mobile,
            age=request.age,
            gender=request.gender,
            recently_travelled=request.recently_travelled,
            consent=request.consent,
            language=request.language,
        )

        # Execute use case
        use_case = RegisterPatientUseCase(patient_repo, question_service)
        result = await use_case.execute(dto_request)

        # Return opaque patient_id to callers
        return RegisterPatientResponse(
            patient_id=encode_patient_id(result.patient_id),
            visit_id=result.visit_id,
            first_question=result.first_question,
            message=result.message,
        )

    except DuplicatePatientError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "DUPLICATE_PATIENT",
                "message": e.message,
                "details": e.details,
            },
        )
    except InvalidDiseaseError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_DISEASE",
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error("Unhandled error in register_patient", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


@router.post(
    "/consultations/answer",
    response_model=AnswerIntakeResponse,
    status_code=status.HTTP_200_OK,
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
    question_service: QuestionServiceDep,
    # Optional multipart fields (to enable Swagger file upload UI)
    form_patient_id: Optional[str] = Form(None),
    form_visit_id: Optional[str] = Form(None),
    form_answer: Optional[str] = Form(None),
    medication_images: Optional[List[UploadFile]] = File(None),
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
            try:
                internal_pid = decode_patient_id(raw_pid)
            except Exception as e:
                logger.warning("[AnswerIntake][JSON] Failed to decode patient_id '%s': %s; using raw value", raw_pid, e)
                internal_pid = raw_pid
            # If internal_pid looks like an opaque token (has non-alnum/underscore), don't construct PatientId later
            # The use case accepts raw string; repository lookup will handle both forms.
            dto_request = AnswerIntakeRequest(
                patient_id=internal_pid,
                visit_id=(body.get("visit_id", "").strip()),
                answer=(body.get("answer", "").strip()),
            )
        elif content_type.startswith("multipart/form-data"):
            # Prefer explicitly bound form fields first; fallback to reading the form
            if form_patient_id is None or form_visit_id is None or form_answer is None:
                form = await request.form()
                form_patient_id = form_patient_id or (form.get("patient_id") or "").strip()
                form_visit_id = form_visit_id or (form.get("visit_id") or "").strip()
                form_answer = form_answer or (form.get("answer") or "").strip()
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

            files: List[UploadFile] = medication_images or []
            if not files:
                # Fallback to reading via form in case Swagger binds differently
                try:
                    form = await request.form()
                    # Try multiple common field names
                    candidates: List[UploadFile] = []
                    for key in ["medication_images", "medication_image", "image", "file", "photo"]:
                        try:
                            candidates.extend([f for f in form.getlist(key) if isinstance(f, UploadFile)])
                        except Exception:
                            pass
                    files = [f for f in candidates if getattr(f, "filename", None)]
                except Exception:
                    files = []
            
            # Store images directly in database if any
            if files:
                logger.info("[AnswerIntake] Received %d file(s) for upload", len(files))
                try:
                    for i, f in enumerate(files):
                        logger.info("[AnswerIntake] File[%d]: name=%s type=%s", i, getattr(f, "filename", None), getattr(f, "content_type", None))
                except Exception:
                    pass
                try:
                    internal_pid = decode_patient_id(form_patient_id)
                except Exception as e:
                    logger.warning("[AnswerIntake][FORM] Failed to decode patient_id '%s': %s; using raw value", form_patient_id, e)
                    internal_pid = form_patient_id
                
                # Store each image in database
                from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
                for file in files:
                    if isinstance(file, UploadFile) and file.filename:
                        raw_ct = (file.content_type or "").lower()
                        content_type = raw_ct.split(";")[0].strip().strip(",")
                        valid_types = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/heic", "image/heif"}
                        if content_type in valid_types:
                            content = await file.read()
                            doc = MedicationImageMongo(
                                patient_id=str(internal_pid),
                                visit_id=str(form_visit_id),
                                image_data=content,
                                content_type=content_type,
                                filename=file.filename,
                            )
                            await doc.insert()
                            logger.info(f"[AnswerIntake] Stored image {file.filename} in database")
                        else:
                            logger.warning(f"[AnswerIntake] Skipping invalid file type: {content_type}")
            
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
                    "message": "Content-Type must be application/json or multipart/form-data",
                    "details": {"content_type": content_type},
                },
            )

        # Execute use case
        use_case = AnswerIntakeUseCase(patient_repo, question_service)
        result = await use_case.execute(dto_request)

        return AnswerIntakeResponse(
            next_question=result.next_question,
            is_complete=result.is_complete,
            question_count=result.question_count,
            max_questions=result.max_questions,
            completion_percent=result.completion_percent,
            message=result.message,
            allows_image_upload=result.allows_image_upload,
            ocr_quality=result.ocr_quality,
        )
    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "PATIENT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "VISIT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except IntakeAlreadyCompletedError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "INTAKE_ALREADY_COMPLETED",
                "message": e.message,
                "details": e.details,
            },
        )
    except (QuestionLimitExceededError, DuplicateQuestionError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": e.error_code, "message": e.message, "details": e.details},
        )
    except Exception as e:
        logger.error("Unhandled error in answer_intake_question", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


@router.patch(
    "/consultations/answer",
    response_model=EditAnswerResponseSchema,
    status_code=status.HTTP_200_OK,
)
async def edit_intake_answer(
    request: EditAnswerRequestSchema,
    patient_repo: PatientRepositoryDep,
    question_service: QuestionServiceDep,
):
    """Edit an existing answer by question number."""
    try:
        use_case = AnswerIntakeUseCase(patient_repo, question_service)
        dto_request = EditAnswerRequest(
            patient_id=decode_patient_id(request.patient_id),
            visit_id=request.visit_id,
            question_number=request.question_number,
            new_answer=request.new_answer,
        )
        result = await use_case.edit(dto_request)
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "PATIENT_NOT_FOUND", "message": e.message, "details": e.details},
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "VISIT_NOT_FOUND", "message": e.message, "details": e.details},
        )
    except Exception as e:
        logger.error("Unhandled error in edit_intake_answer", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


# Lightweight webhook to receive medication images during intake (per requirements)
@router.post("/webhook/image")
async def upload_medication_image(
    image: UploadFile = File(...),
    patient_id: str = Form(...),
    visit_id: str = Form(...),
):
    try:
        # Normalize/resolve patient id (opaque token from client → internal id)
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        # Validate content type
        raw_ct = (image.content_type or "").lower()
        content_type = raw_ct.split(";")[0].strip().strip(",")
        valid_types = {"image/jpeg", "image/jpg", "image/png"}
        if content_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "INVALID_FILE_TYPE",
                    "message": f"Only jpg, jpeg, png allowed (got {content_type or 'unknown'})",
                    "details": {},
                },
            )

        # Read image data
        content = await image.read()
        
        # Store DB record with image data
        from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
        doc = MedicationImageMongo(
            patient_id=str(internal_patient_id),
            visit_id=str(visit_id),
            image_data=content,
            content_type=content_type,
            filename=image.filename or "unknown",
        )
        inserted = await doc.insert()

        return {
            "status": "success",
            "patient_id": internal_patient_id,
            "visit_id": visit_id,
            "id": str(getattr(inserted, "id", "")),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled error in upload_medication_image", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to upload image",
                "details": {"exception": str(e), "type": e.__class__.__name__},
            },
        )


# Endpoint for multiple image uploads
@router.post("/webhook/images")
async def upload_multiple_medication_images(
    request: Request,
    patient_id: str = Form(...),
    visit_id: str = Form(...),
    images: Optional[List[UploadFile]] = File(None),
):
    try:
        logger.info("[WebhookImages] Incoming upload for patient_id=%s visit_id=%s", patient_id, visit_id)
        # Normalize/resolve patient id (opaque token from client → internal id)
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        
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

        # Store each image in database
        from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
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
                
                # Store DB record with image data
                doc = MedicationImageMongo(
                    patient_id=str(internal_patient_id),
                    visit_id=str(visit_id),
                    image_data=content,
                    content_type=content_type,
                    filename=image.filename or f"image_{i+1}",
                )
                inserted = await doc.insert()
                
                uploaded_images.append({
                    "id": str(getattr(inserted, "id", "")),
                    "filename": image.filename or f"image_{i+1}",
                    "content_type": content_type
                })
                logger.info("[WebhookImages] Stored image[%d] id=%s filename=%s", i, str(getattr(inserted, "id", "")), image.filename or f"image_{i+1}")
                
            except Exception as e:
                errors.append(f"Image {i+1}: {str(e)}")
                logger.error(f"Error uploading image {i+1}: {e}")
        
        return {
            "status": "success" if not errors else "partial_success",
            "patient_id": internal_patient_id,
            "visit_id": visit_id,
            "uploaded_images": uploaded_images,
            "errors": errors,
            "total_uploaded": len(uploaded_images),
            "total_errors": len(errors)
        }
    except Exception as e:
        logger.error("Unhandled error in upload_multiple_medication_images", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to upload images",
                "details": {"exception": str(e), "type": e.__class__.__name__},
            },
        )


# Stream image bytes by id
@router.get("/images/{image_id}/content")
async def get_medication_image_content(image_id: str):
    from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
    try:
        doc = await MedicationImageMongo.get(PydanticObjectId(image_id))
        if not doc:
            raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": "Image not found"})
        return Response(content=doc.image_data, media_type=getattr(doc, "content_type", "application/octet-stream"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error reading medication image", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(e)})


# List uploaded images for a visit
@router.get("/{patient_id}/visits/{visit_id}/images")
async def list_medication_images(patient_id: str, visit_id: str):
    from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
    try:
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception:
            internal_patient_id = patient_id
        docs = await MedicationImageMongo.find(
            MedicationImageMongo.patient_id == str(internal_patient_id),
            MedicationImageMongo.visit_id == str(visit_id),
        ).to_list()
        return {
            "patient_id": internal_patient_id,
            "visit_id": visit_id,
            "images": [
                {
                    "id": str(getattr(d, "id", "")),
                    "filename": getattr(d, "filename", "unknown"),
                    "content_type": getattr(d, "content_type", ""),
                    "uploaded_at": getattr(d, "uploaded_at", None),
                }
                for d in docs
            ],
        }
    except Exception as e:
        logger.error("Error listing medication images", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(e)})


# Delete one uploaded image by id
@router.delete("/images/{image_id}")
async def delete_medication_image(image_id: str):
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
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(e)})


@router.post(
    "/summary/previsit",
    response_model=PreVisitSummaryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "Intake not completed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def generate_pre_visit_summary(
    request: PreVisitSummaryRequest,
    patient_repo: PatientRepositoryDep,
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
        # Convert Pydantic model to DTO
        dto_request = PreVisitSummaryRequest(
            patient_id=decode_patient_id(request.patient_id),
            visit_id=request.visit_id,
        )

        # Execute use case
        use_case = GeneratePreVisitSummaryUseCase(patient_repo, question_service)
        result = await use_case.execute(dto_request)

        return PreVisitSummaryResponse(
            patient_id=result.patient_id,
            visit_id=result.visit_id,
            summary=result.summary,
            generated_at=result.generated_at,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INTAKE_NOT_COMPLETED",
                "message": str(e),
                "details": {},
            },
        )
    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "PATIENT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "VISIT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error("Unhandled error in generate_pre_visit_summary", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


@router.get(
    "/{patient_id}/visits/{visit_id}/summary",
    response_model=PreVisitSummaryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Patient, visit, or summary not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_pre_visit_summary(
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
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
        
        # Find patient (decode opaque id from client)
        internal_patient_id = decode_patient_id(patient_id)
        patient_id_obj = PatientId(internal_patient_id)
        patient = await patient_repo.find_by_id(patient_id_obj)
        if not patient:
            raise PatientNotFoundError(patient_id)

        # Find visit
        visit = patient.get_visit_by_id(visit_id)
        if not visit:
            raise VisitNotFoundError(visit_id)

        # Check if summary exists
        if not visit.has_pre_visit_summary():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "SUMMARY_NOT_FOUND",
                    "message": f"No pre-visit summary found for visit {visit_id}",
                    "details": {"visit_id": visit_id},
                },
            )

        # Get stored summary
        summary_data = visit.get_pre_visit_summary()

        # Attach any uploaded medication images
        from ...adapters.db.mongo.models.patient_m import MedicationImageMongo
        docs = await MedicationImageMongo.find(
            MedicationImageMongo.patient_id == patient.patient_id.value,
            MedicationImageMongo.visit_id == visit.visit_id.value,
        ).to_list()
        images_meta = [
            {
                "id": str(getattr(d, "id", "")),
                "filename": getattr(d, "filename", "unknown"),
                "content_type": getattr(d, "content_type", ""),
            }
            for d in docs
        ] if docs else None

        return PreVisitSummaryResponse(
            patient_id=encode_patient_id(patient.patient_id.value),
            visit_id=visit.visit_id.value,
            summary=summary_data["summary"],
            generated_at=summary_data["generated_at"],
            medication_images=images_meta,
        )

    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "PATIENT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "VISIT_NOT_FOUND",
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error("Unhandled error in get_pre_visit_summary", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


@router.post(
    "/summary/postvisit",
    response_model=PostVisitSummaryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "Visit not completed or SOAP note not available"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def generate_post_visit_summary(
    request: PostVisitSummaryRequest,
    patient_repo: PatientRepositoryDep,
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
        
        logger.info(f"Generating post-visit summary for internal_patient_id: {internal_patient_id}, visit_id: {request.visit_id}")
        
        # Create request with decoded patient_id
        decoded_request = PostVisitSummaryRequest(
            patient_id=internal_patient_id,
            visit_id=request.visit_id
        )
        
        # Create use case instance
        use_case = GeneratePostVisitSummaryUseCase(patient_repo, soap_service)
        
        # Execute use case
        result = await use_case.execute(decoded_request)
        
        return result
        
    except PatientNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "PATIENT_NOT_FOUND",
                "message": f"Patient {request.patient_id} not found",
                "details": {"patient_id": request.patient_id},
            },
        )
    except VisitNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "VISIT_NOT_FOUND",
                "message": f"Visit {request.visit_id} not found",
                "details": {"visit_id": request.visit_id},
            },
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_VISIT_STATE",
                "message": str(e),
                "details": {"patient_id": request.patient_id, "visit_id": request.visit_id},
            },
        )
    except Exception as e:
        logger.error("Unhandled error in generate_post_visit_summary", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(e) or repr(e), "type": e.__class__.__name__},
            },
        )


@router.get(
    "/{patient_id}/visits/{visit_id}/summary/postvisit",
    response_model=PostVisitSummaryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_post_visit_summary(
    patient_id: str,
    visit_id: str,
    patient_repo: PatientRepositoryDep,
):
    """Retrieve stored post-visit summary from visit (if available)."""
    try:
        from ...domain.value_objects.patient_id import PatientId
        
        try:
            internal_patient_id = decode_patient_id(patient_id)
        except Exception as e:
            internal_patient_id = patient_id
            
        patient = await patient_repo.find_by_id(PatientId(internal_patient_id))
        if not patient:
            raise PatientNotFoundError(patient_id)
        
        visit = patient.get_visit_by_id(visit_id)
        if not visit:
            raise VisitNotFoundError(visit_id)
        
        data = visit.get_post_visit_summary()
            
        if not data:
            raise HTTPException(status_code=404, detail={"error": "POST_VISIT_SUMMARY_NOT_FOUND", "message": "No post-visit summary stored"})
        return PostVisitSummaryResponse(**data)
    except PatientNotFoundError as e:
        raise HTTPException(status_code=404, detail={"error": "PATIENT_NOT_FOUND", "message": e.message, "details": e.details})
    except VisitNotFoundError as e:
        raise HTTPException(status_code=404, detail={"error": "VISIT_NOT_FOUND", "message": e.message, "details": e.details})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled error in get_post_visit_summary", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(e)})

 