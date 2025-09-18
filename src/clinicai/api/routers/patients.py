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
    RegisterPatientRequest,
)
from clinicai.application.dto.patient_dto import EditAnswerRequest
from clinicai.application.use_cases.answer_intake import AnswerIntakeUseCase
from clinicai.application.use_cases.generate_pre_visit_summary import GeneratePreVisitSummaryUseCase
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

from ..deps import PatientRepositoryDep, QuestionServiceDep
from ...core.utils.crypto import encode_patient_id, decode_patient_id
from ..schemas.patient import AnswerIntakeResponse, ErrorResponse
from ..schemas.patient import AnswerIntakeRequest as AnswerIntakeRequestSchema
from ..schemas.patient import (
    PatientSummarySchema,
    PreVisitSummaryResponse,
    RegisterPatientResponse,
    EditAnswerRequest as EditAnswerRequestSchema,
    EditAnswerResponse as EditAnswerResponseSchema,
)
from ..schemas.patient import RegisterPatientRequest as RegisterPatientRequestSchema

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
            dto_request = AnswerIntakeRequest(
                patient_id=decode_patient_id((body.get("patient_id", "").strip())),
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

            if not (form_patient_id and form_visit_id and form_answer):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "VALIDATION_ERROR",
                        "message": "Missing required form fields: patient_id, visit_id, answer",
                        "details": {},
                    },
                )

            image_paths: List[str] = []
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
            if files:
                import os
                from uuid import uuid4
                from clinicai.core.utils.file_utils import create_directory
                from clinicai.core.utils.image_ocr import extract_text_from_image

                uploads_dir = os.getenv("UPLOADS_DIR", "/tmp/clinicai_uploads")
                create_directory(uploads_dir)

                ocr_texts: list[str] = []
                logger.info(f"[AnswerIntake] Received {len(files)} file(s) for upload")
                for file in files:
                    if isinstance(file, UploadFile) and file.filename:
                        filename = f"med_{uuid4().hex}_{file.filename}"
                        dest_path = os.path.join(uploads_dir, filename)
                        with open(dest_path, "wb") as f:
                            f.write(await file.read())
                        image_paths.append(dest_path)
                        # OCR extraction (best-effort)
                        text = extract_text_from_image(dest_path)
                        if text:
                            ocr_texts.append(text)

            # Pass images and OCR text through the answer payload via markers for downstream usage
            if image_paths:
                form_answer = f"{form_answer}\n[IMAGES]: {', '.join(image_paths)}"
                if 'ocr_texts' in locals() and ocr_texts:
                    form_answer = f"{form_answer}\n[OCR]: {' | '.join(ocr_texts)}"
            dto_request = AnswerIntakeRequest(
                patient_id=decode_patient_id(form_patient_id),
                visit_id=form_visit_id,
                answer=form_answer,
                attachment_image_paths=image_paths if image_paths else None,
            )
            logger.info(f"[AnswerIntake] Attachment paths to persist: {image_paths}")
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

        return PreVisitSummaryResponse(
            patient_id=encode_patient_id(patient.patient_id.value),
            visit_id=visit.visit_id.value,
            summary=summary_data["summary"],
            generated_at=summary_data["generated_at"],
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


 
