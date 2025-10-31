"""
FastAPI application factory and main app configuration.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import logging
import os

from .api.routers import health, patients, notes, workflow
from .api.routers import doctor as doctor_router
from .api.routers import transcription as transcription_router
from .api.routers import audio as audio_router
from .api.routers import intake as intake_router
from .core.config import get_settings
from .domain.errors import DomainError
from .core.hipaa_audit import get_audit_logger
from .middleware.hipaa_middleware import HIPAAAuditMiddleware
from .middleware.performance_middleware import PerformanceMiddleware
from clinicai.middleware.request_id_middleware import RequestIDMiddleware
from clinicai.api.errors import APIError, ValidationError, NotFoundError
from clinicai.api.schemas.common import ErrorResponse
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
# Azure Monitor removed - was causing issues
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"🚀 Starting Clinic-AI Intake Assistant v{settings.app_version}")
    print(f"📊 Environment: {settings.app_env}")
    print(f"🔧 Debug mode: {settings.debug}")
    
    # Azure Monitor removed - was causing issues
    # try:
    #     azure_monitor = get_azure_monitor()
    #     print("✅ Azure Monitor initialized")
    # except Exception as e:
    #     print(f"⚠️  Azure Monitor initialization failed: {e}")
    #     logging.error(f"Azure Monitor failed to initialize: {e}")
    
    
    # Initialize database connection (MongoDB + Beanie)
    try:
        from beanie import init_beanie  # type: ignore
        from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
        import certifi  # type: ignore

        # Import models for registration
        from .adapters.db.mongo.models.patient_m import (
            PatientMongo,
            VisitMongo,
            MedicationImageMongo,
            AdhocTranscriptMongo,
            DoctorPreferencesMongo,
            AudioFileMongo,
        )
        from .adapters.db.mongo.models.blob_file_reference import BlobFileReference

        # Use configured URI
        mongo_uri = settings.database.uri
        db_name = settings.database.db_name

        # Enable TLS only for Atlas SRV URIs
        if mongo_uri.startswith("mongodb+srv://"):
            ca_path = certifi.where()
            client = AsyncIOMotorClient(
                mongo_uri,
                serverSelectionTimeoutMS=15000,
                tls=True,
                tlsCAFile=ca_path,
                tlsAllowInvalidCertificates=False,
            )
        else:
            # Local/standard connection (no TLS)
            client = AsyncIOMotorClient(
                mongo_uri,
                serverSelectionTimeoutMS=15000,
            )

        db = client[db_name]
        await init_beanie(
            database=db,
            document_models=[PatientMongo, VisitMongo, MedicationImageMongo, AdhocTranscriptMongo, DoctorPreferencesMongo, AudioFileMongo, BlobFileReference],
        )
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

    # Initialize Azure Blob Storage
    try:
        from .adapters.storage.azure_blob_service import get_azure_blob_service
        blob_service = get_azure_blob_service()
        await blob_service.ensure_container_exists()
        print("✅ Azure Blob Storage initialized")
    except Exception as e:
        print(f"⚠️  Azure Blob Storage initialization failed: {e}")
        logging.error(f"Azure Blob Storage failed to initialize: {e}")

    # Initialize HIPAA audit logger
    try:
        audit_logger = get_audit_logger()
        await audit_logger.initialize(
            mongo_uri=mongo_uri,
            db_name=db_name
        )
        print("✅ HIPAA Audit Logger initialized")
    except Exception as e:
        print(f"⚠️  HIPAA Audit Logger initialization failed: {e}")
        # Don't fail startup, but log the error
        logging.error(f"HIPAA Audit Logger failed to initialize: {e}")

    # Whisper warm-up disabled to reduce startup memory footprint

    yield

    # Shutdown
    print("🛑 Shutting down Clinic-AI Intake Assistant")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Clinic-AI Intake Assistant",
        description="AI-powered clinical intake system for small and mid-sized clinics",
        version=settings.app_version,
        docs_url="/docs",   # Restore Swagger UI
        redoc_url="/redoc", # Restore ReDoc UI (optional)
        openapi_url="/openapi.json", # Restore OpenAPI JSON
        lifespan=lifespan,
    )

    # CORS middleware
    # Allow all origins (no credentials) to resolve preflight failures
    allow_methods = settings.cors.allowed_methods or ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    allow_methods = list({m.upper() for m in allow_methods} | {"PATCH", "OPTIONS"})
    allow_headers = settings.cors.allowed_headers or ["*"]
    if allow_headers != ["*"] and "content-type" not in {h.lower() for h in allow_headers}:
        allow_headers = [*allow_headers, "content-type"]
    
    # Add specific headers for file uploads
    if allow_headers != ["*"]:
        upload_headers = ["content-type", "content-disposition", "authorization", "x-requested-with"]
        for header in upload_headers:
            if header not in {h.lower() for h in allow_headers}:
                allow_headers.append(header)

    # Debug CORS configuration
    print(f"🔧 CORS Configuration:")
    print(f"   - Origins: * (all)")
    print(f"   - Methods: {allow_methods}")
    print(f"   - Headers: {allow_headers}")
    print(f"   - Credentials: False")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        max_age=600,
        expose_headers=["*"],  # Expose all headers to client
    )
    
    # Add HIPAA audit middleware
    app.add_middleware(HIPAAAuditMiddleware)
    
    # Add performance tracking middleware
    app.add_middleware(PerformanceMiddleware)
    
    # Add request logging middleware for debugging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger = logging.getLogger("clinicai")
        logger.info(f"🌐 {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
        logger.info(f"   Headers: {dict(request.headers)}")
        
        response = await call_next(request)
        
        logger.info(f"   Response: {response.status_code}")
        return response
    
    # Add explicit CORS headers for all responses
    @app.middleware("http")
    async def add_cors_headers(request: Request, call_next):
        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            response = JSONResponse(content={"message": "OK"})
        else:
            response = await call_next(request)
        
        # Add CORS headers to all responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "600"
        
        return response

    # Register X-Request-ID middleware after CORS etc.
    app.add_middleware(RequestIDMiddleware)

    # Include routers in logical order: Health → Patients → Intake → Workflow → Notes → Transcription → Audio → Doctor
    app.include_router(health.router)
    app.include_router(patients.router)
    app.include_router(intake_router.router)
    app.include_router(workflow.router)
    app.include_router(notes.router)
    app.include_router(transcription_router.router)
    app.include_router(audio_router.router)
    app.include_router(doctor_router.router)

    # Global exception handler for domain errors
    @app.exception_handler(DomainError)
    async def domain_error_handler(request, exc: DomainError):
        return JSONResponse(
            status_code=400,
            content={
                "error": exc.error_code or "DOMAIN_ERROR",
                "message": exc.message,
                "details": exc.details,
            },
        )

    # Global exception handler for validation errors
    @app.exception_handler(ValueError)
    async def validation_error_handler(request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={"error": "VALIDATION_ERROR", "message": str(exc), "details": {}},
        )

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        req_id = getattr(request.state, "request_id", None)
        logging.error(f"APIError: {exc.code} ({exc.http_status}) {exc.message} | request_id={req_id}")
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(
                error=exc.code,
                message=exc.message,
                request_id=req_id or "",
                details=exc.details or {}
            ).dict(),
        )

    @app.exception_handler(RequestValidationError)
    async def pydantic_error_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", None)
        logging.error(f"ValidationError: {exc.errors()} | request_id={req_id}")
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="INVALID_INPUT",
                message="Input validation failed.",
                request_id=req_id or "",
                details={"errors": exc.errors()}
            ).dict()
        )

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        req_id = getattr(request.state, "request_id", None)
        logging.error(f"NotFoundError: {exc.message} | request_id={req_id}")
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="NOT_FOUND",
                message=exc.message,
                request_id=req_id or "",
                details=exc.details or {}
            ).dict()
        )

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", None)
        logging.error(f"Unhandled error: {type(exc)} | request_id={req_id}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="INTERNAL_ERROR",
                message="An unexpected error has occurred. Please try again later.",
                request_id=req_id or "",
            ).dict(),
        )

    return app


# Create the app instance
app = create_app()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    return {
        "service": "Clinic-AI Intake Assistant",
        "version": settings.app_version,
        "environment": settings.app_env,
        "status": "running",
        "docs": "/docs",
        "swagger_yaml": "/swagger.yaml",
        "endpoints": {
            "health": "/health",
            "register_patient": "POST /patients/",
            "answer_intake": "POST /patients/consultations/answer",
            "pre_visit_summary": "POST /patients/summary/previsit",
            "get_summary": "GET /patients/{patient_id}/visits/{visit_id}/summary",
            # Image upload endpoints
            "upload_images": "POST /patients/webhook/images",
            "get_intake_image_content": "GET /patients/{patient_id}/visits/{visit_id}/intake-images/{image_id}/content",
            "list_images": "GET /patients/{patient_id}/visits/{visit_id}/images",
            "delete_image": "DELETE /patients/images/{image_id}",
            # Step-03 endpoints
            "transcribe_audio": "POST /notes/transcribe",
            "generate_soap": "POST /notes/soap/generate",
            "get_transcript": "GET /notes/{patient_id}/visits/{visit_id}/transcript",
            "get_soap": "GET /notes/{patient_id}/visits/{visit_id}/soap",
            # Vitals endpoints
            "store_vitals": "POST /notes/vitals",
            "get_vitals": "GET /notes/{patient_id}/visits/{visit_id}/vitals",
            # Doctor preferences
            "get_doctor_preferences": "GET /doctor/preferences",
            "save_doctor_preferences": "POST /doctor/preferences",
            # Intake session (preferences-aware)
            "start_intake": "POST /intake/start",
            "next_question": "POST /intake/next-question",
            # Audio management
            "list_audio_files": "GET /audio/",
            "get_audio_metadata": "GET /audio/{audio_id}",
            "download_audio": "GET /audio/{audio_id}/download",
            "stream_audio": "GET /audio/{audio_id}/stream",
            "delete_audio": "DELETE /audio/{audio_id}",
            "audio_stats": "GET /audio/stats/summary",
        },
    }


@app.get("/swagger.yaml")
async def get_swagger_yaml():
    """Serve the custom Swagger YAML file."""
    swagger_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "swagger.yaml")
    if os.path.exists(swagger_path):
        return FileResponse(
            path=swagger_path,
            media_type="application/x-yaml",
            filename="swagger.yaml"
        )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Swagger file not found"}
        )