"""
Prescription-related API endpoints for image upload and analysis.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status

from clinicai.application.ports.services.prescription_service import PrescriptionService
from clinicai.domain.errors import PatientNotFoundError, VisitNotFoundError
from clinicai.domain.value_objects.patient_id import PatientId

from ..deps import PatientRepositoryDep
from ..schemas.prescription import PrescriptionResponse, ErrorResponse, Medicine

router = APIRouter(prefix="/prescriptions", tags=["Prescriptions"])
logger = logging.getLogger("clinicai")

# Dependency to get prescription service
def get_prescription_service() -> PrescriptionService:
    """Get prescription service instance."""
    # Import here to avoid initialization errors at module level
    from clinicai.adapters.external.prescription_service_mistral import MistralPrescriptionService
    return MistralPrescriptionService()


@router.post(
    "/upload",
    response_model=PrescriptionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Patient or visit not found"},
        422: {"model": ErrorResponse, "description": "Invalid file format"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def upload_prescriptions(
    patient_id: str = Form(..., description="Patient ID"),
    visit_id: str = Form(..., description="Visit ID"),
    files: List[UploadFile] = File(..., description="Prescription images to analyze"),
    patient_repo: PatientRepositoryDep = None,
):
    """
    Upload and analyze prescription images using Mistral AI.
    
    This endpoint:
    1. Validates patient and visit exist
    2. Accepts multiple prescription images
    3. Uses Mistral AI for OCR and text extraction
    4. Parses medicines, tests, and instructions into structured data
    5. Returns structured JSON response with extracted information
    
    **Supported image formats:** JPG, JPEG, PNG, GIF, BMP, WebP, TIFF
    **Maximum file size:** 50MB per file
    **Maximum files:** 10 images per request
    """
    try:
        logger.info(f"Processing prescription upload for patient {patient_id}, visit {visit_id}")
        
        # Validate inputs
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "VALIDATION_ERROR",
                    "message": "At least one prescription image must be uploaded",
                    "details": {},
                },
            )
        
        logger.info(f"Received {len(files)} files for processing")
        
        if len(files) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "VALIDATION_ERROR",
                    "message": "Maximum 10 images allowed per request",
                    "details": {"uploaded_count": len(files), "max_allowed": 10},
                },
            )
        
        # Validate patient exists
        try:
            patient_id_obj = PatientId(patient_id)
            patient = await patient_repo.find_by_id(patient_id_obj)
            if not patient:
                raise PatientNotFoundError(patient_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "PATIENT_NOT_FOUND",
                    "message": f"Patient {patient_id} not found",
                    "details": {"patient_id": patient_id},
                },
            )
        
        # Validate visit exists
        visit = patient.get_visit_by_id(visit_id)
        if not visit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "VISIT_NOT_FOUND",
                    "message": f"Visit {visit_id} not found for patient {patient_id}",
                    "details": {"patient_id": patient_id, "visit_id": visit_id},
                },
            )
        
        # Validate file types and sizes
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        max_size_mb = 50
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "INVALID_FILE",
                        "message": "File must have a filename",
                        "details": {},
                    },
                )
            
            # Check file extension
            file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
            if f'.{file_ext}' not in valid_extensions:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "INVALID_FILE_FORMAT",
                        "message": f"Unsupported file format: {file_ext}",
                        "details": {
                            "filename": file.filename,
                            "supported_formats": list(valid_extensions)
                        },
                    },
                )
            
            # Check file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to beginning
            
            if file_size > max_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "FILE_TOO_LARGE",
                        "message": f"File {file.filename} is too large",
                        "details": {
                            "filename": file.filename,
                            "size_mb": round(file_size / (1024 * 1024), 2),
                            "max_size_mb": max_size_mb
                        },
                    },
                )
        
        logger.info("File validation passed, proceeding with prescription processing")
        
        # Process prescriptions using Mistral AI
        prescription_service = get_prescription_service()
        result = await prescription_service.process_prescriptions(
            patient_id=patient_id,
            visit_id=visit_id,
            files=files
        )
        
        # Convert medicines to Medicine objects
        medicines = []
        for med_data in result.get("medicines", []):
            medicines.append(Medicine(
                name=med_data.get("name", ""),
                dose=med_data.get("dose"),
                frequency=med_data.get("frequency"),
                duration=med_data.get("duration")
            ))
        
        logger.info(f"Prescription processing completed with status: {result.get('processing_status')}")
        
        # Return structured response
        return PrescriptionResponse(
            patient_id=result["patient_id"],
            visit_id=result["visit_id"],
            medicines=medicines,
            tests=result.get("tests", []),
            instructions=result.get("instructions", []),
            raw_text=result.get("raw_text", ""),
            processing_status=result.get("processing_status", "unknown"),
            message=result.get("message", "Processing completed"),
            debug_info=result.get("debug_info")
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("Unhandled error in upload_prescriptions", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred while processing prescriptions",
                "details": {"exception": str(e), "type": e.__class__.__name__},
            },
        )