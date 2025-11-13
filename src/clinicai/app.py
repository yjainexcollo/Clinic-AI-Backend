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
from .middleware.auth_middleware import AuthenticationMiddleware
from .middleware.hipaa_middleware import HIPAAAuditMiddleware
from .middleware.performance_middleware import PerformanceMiddleware
from clinicai.middleware.request_id_middleware import RequestIDMiddleware
from clinicai.api.errors import APIError, ValidationError, NotFoundError
from clinicai.api.schemas.common import ErrorResponse
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"🚀 Starting Clinic-AI Intake Assistant v{settings.app_version}")
    print(f"📊 Environment: {settings.app_env}")
    print(f"🔧 Debug mode: {settings.debug}")
    
    # Note: Azure Application Insights is initialized in create_app() before middleware
    # to ensure proper instrumentation order and request capture
    
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

    # Initialize Azure Key Vault (if configured)
    try:
        from .core.key_vault import get_key_vault_service
        key_vault = get_key_vault_service()
        if key_vault and key_vault.is_available:
            print("✅ Azure Key Vault initialized")
            logging.info("Azure Key Vault initialized successfully")
        else:
            print("⚠️  Azure Key Vault not available (using environment variables)")
            logging.debug("Azure Key Vault not configured or not accessible")
    except Exception as e:
        print(f"⚠️  Azure Key Vault initialization failed: {e}")
        logging.debug(f"Azure Key Vault initialization skipped: {e}")

    # Initialize Azure Blob Storage
    try:
        from .adapters.storage.azure_blob_service import get_azure_blob_service
        blob_service = get_azure_blob_service()
        await blob_service.ensure_container_exists()
        print("✅ Azure Blob Storage initialized")
    except Exception as e:
        print(f"⚠️  Azure Blob Storage initialization failed: {e}")
        logging.error(f"Azure Blob Storage failed to initialize: {e}")

    # Initialize Azure Queue Storage and start worker (if enabled)
    worker_task = None
    try:
        try:
            from .adapters.queue.azure_queue_service import get_azure_queue_service
            queue_service = get_azure_queue_service()
            await queue_service.ensure_queue_exists()
            print("✅ Azure Queue Storage initialized")
            
            # Start transcription worker if enabled (set ENABLE_TRANSCRIPTION_WORKER=true)
            if os.getenv("ENABLE_TRANSCRIPTION_WORKER", "false").lower() == "true":
                from .workers.transcription_worker import TranscriptionWorker
                worker = TranscriptionWorker()
                worker_task = asyncio.create_task(worker.run())
                print("✅ Transcription worker started")
                logging.info("Transcription worker started as background task")
            else:
                print("ℹ️  Transcription worker disabled (set ENABLE_TRANSCRIPTION_WORKER=true to enable)")
        except ImportError:
            print("⚠️  Azure Queue Storage not available (azure-storage-queue package not installed)")
            logging.warning("Azure Queue Storage not available - install azure-storage-queue package")
    except Exception as e:
        print(f"⚠️  Azure Queue Storage initialization failed: {e}")
        logging.error(f"Azure Queue Storage failed to initialize: {e}")

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

    # Validate Azure OpenAI configuration
    try:
        azure_openai_configured = (
            settings.azure_openai.endpoint and 
            settings.azure_openai.api_key
        )
        
        if azure_openai_configured:
            # Validate endpoint format
            if not settings.azure_openai.endpoint.startswith("https://") or ".openai.azure.com" not in settings.azure_openai.endpoint:
                raise ValueError(
                    f"Invalid Azure OpenAI endpoint format: {settings.azure_openai.endpoint}. "
                    "Must be: https://xxx.openai.azure.com/"
                )
            
            # Validate deployment names are configured
            if not settings.azure_openai.deployment_name:
                raise ValueError(
                    "Azure OpenAI chat deployment name is required. "
                    "Please set AZURE_OPENAI_DEPLOYMENT_NAME."
                )
            
            # Azure OpenAI Whisper deployment not required - using Azure Speech Service for transcription
            
            # Validate API key is not empty
            if not settings.azure_openai.api_key or len(settings.azure_openai.api_key.strip()) == 0:
                raise ValueError(
                    "Azure OpenAI API key is required. "
                    "Please set AZURE_OPENAI_API_KEY."
                )
            
            # Validate deployment actually exists by making a test call
            print(f"🔍 Validating Azure OpenAI deployments...")
            print(f"   Endpoint: {settings.azure_openai.endpoint}")
            print(f"   API Version: {settings.azure_openai.api_version}")
            print(f"   Deployment: {settings.azure_openai.deployment_name}")
            from .core.azure_openai_client import validate_azure_openai_deployment
            
            # Validate chat deployment (will try multiple API versions automatically)
            is_valid, error_msg = await validate_azure_openai_deployment(
                endpoint=settings.azure_openai.endpoint,
                api_key=settings.azure_openai.api_key,
                api_version=settings.azure_openai.api_version,
                deployment_name=settings.azure_openai.deployment_name,
                timeout=10.0,
                try_alternative_versions=True
            )
            
            if not is_valid:
                print(f"❌ Deployment validation failed:")
                print(f"   {error_msg}")
                raise ValueError(
                    f"Azure OpenAI chat deployment validation failed: {error_msg}"
                )
            
            # If validation succeeded but with a different API version, show a warning
            if error_msg and "API version" in error_msg:
                print(f"⚠️  {error_msg}")
                logging.warning(error_msg)
            
            print(f"✅ Azure OpenAI configuration validated")
            print(f"   Endpoint: {settings.azure_openai.endpoint}")
            print(f"   API Version: {settings.azure_openai.api_version}")
            print(f"   Chat Deployment: {settings.azure_openai.deployment_name} ✓")
            print(f"   Transcription Service: Azure Speech Service (batch with speaker diarization)")
            logging.info(
                f"Azure OpenAI validated - endpoint={settings.azure_openai.endpoint}, "
                f"chat_deployment={settings.azure_openai.deployment_name}, "
                f"transcription_service=azure_speech"
            )
        else:
            # Azure OpenAI is required - fail startup if not configured
            raise ValueError(
                "Azure OpenAI is required but not configured. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY. "
                "Fallback to standard OpenAI is disabled for data security."
            )
    except ValueError as e:
        print(f"❌ Azure OpenAI validation failed: {e}")
        logging.error(f"Azure OpenAI validation failed: {e}")
        raise  # Fail startup if Azure OpenAI is misconfigured
    except Exception as e:
        print(f"⚠️  Azure OpenAI validation error: {e}")
        logging.warning(f"Azure OpenAI validation error: {e}")
        # Don't fail startup for unexpected errors, but log them

    # Azure Speech Service transcription - no warm-up needed

    yield

    # Shutdown
    print("🛑 Shutting down Clinic-AI Intake Assistant")
    
    # Stop transcription worker if running
    if worker_task:
        print("🛑 Stopping transcription worker...")
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        print("✅ Transcription worker stopped")


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

    # Initialize Azure Application Insights BEFORE middleware
    # This must happen early to capture all requests and ensure proper instrumentation order
    try:
        app_insights_connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if app_insights_connection_string:
            from azure.monitor.opentelemetry import configure_azure_monitor
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            
            # Configure Azure Monitor with OpenTelemetry
            configure_azure_monitor(
                connection_string=app_insights_connection_string
            )
            # Instrument FastAPI app for automatic telemetry
            # This must be done BEFORE adding middleware to ensure all requests are captured
            FastAPIInstrumentor.instrument_app(app)
            print("✅ Azure Application Insights initialized (before middleware)")
            logging.info("Azure Application Insights initialized successfully")
        else:
            print("⚠️  APPLICATIONINSIGHTS_CONNECTION_STRING not set, Application Insights disabled")
            logging.debug("Application Insights not configured (connection string missing)")
    except ImportError as e:
        print(f"⚠️  Application Insights packages not installed: {e}")
        logging.warning(f"Application Insights not available: {e}")
    except Exception as e:
        print(f"⚠️  Azure Application Insights initialization failed: {e}")
        logging.error(f"Azure Application Insights failed to initialize: {e}", exc_info=True)

    # Customize OpenAPI schema to control tag order
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        # Define tag order: Health, Patient Registration, Patient Management, Intake + Previsit, Vitals and Transcript, Soap, Postvisit, Audio Management
        openapi_schema["tags"] = [
            {"name": "health", "description": "Health and readiness checks"},
            {"name": "Patient Registration", "description": "Patient registration for scheduled and walk-in visits"},
            {"name": "Patient Management", "description": "Patient listing and management operations"},
            {"name": "Intake + Pre-Visit Summary", "description": "Intake form, medication image, and previsit summary"},
            {"name": "Vitals and Transcript Generation", "description": "Vitals and transcript routes"},
            {"name": "SOAP Note Generation", "description": "SOAP note generation and retrieval"},
            {"name": "Post-Visit Summary", "description": "Post-visit summary generation and retrieval"},
            {"name": "Audio Management", "description": "Audio file listing and management operations"},
        ]
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi

    # CORS middleware
    # Allow all origins (no credentials) to resolve preflight failures
    allow_methods = settings.cors.allowed_methods or ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    allow_methods = list({m.upper() for m in allow_methods} | {"PATCH", "OPTIONS"})
    allow_headers = settings.cors.allowed_headers or ["*"]
    if allow_headers != ["*"] and "content-type" not in {h.lower() for h in allow_headers}:
        allow_headers = [*allow_headers, "content-type"]
    
    # Add specific headers for file uploads
    if allow_headers != ["*"]:
        upload_headers = [
            "content-type",
            "content-disposition",
            "authorization",
            "x-requested-with",
            "x-api-key",
        ]
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
    
    # Add authentication middleware FIRST (before HIPAA audit)
    # This ensures all PHI endpoints require authentication
    app.add_middleware(AuthenticationMiddleware)
    
    # Add HIPAA audit middleware (runs after authentication)
    # This logs all PHI access with authenticated user IDs
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
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, X-API-Key, Accept, Origin"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "600"
        
        return response

    # Register X-Request-ID middleware after CORS etc.
    app.add_middleware(RequestIDMiddleware)

    # Include routers in logical order: Health → Patients → Intake → Workflow → Notes → Transcription → Audio → Doctor
    app.include_router(health.router)
    app.include_router(patients.router)
    app.include_router(intake_router.router, include_in_schema=False)
    app.include_router(workflow.router)
    app.include_router(notes.router)
    app.include_router(transcription_router.router, include_in_schema=False)
    app.include_router(audio_router.router)
    app.include_router(doctor_router.router, include_in_schema=False)

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
        error_details = exc.errors()
        # Log full validation error details for debugging
        print(f"❌ ValidationError on {request.method} {request.url.path}")
        print(f"   Error details: {error_details}")
        print(f"   Request headers: {dict(request.headers)}")
        print(f"   Content-type: {request.headers.get('content-type', 'not set')}")
        logging.error(f"ValidationError on {request.method} {request.url.path}: {error_details} | request_id={req_id}")
        logging.error(f"Request headers: {dict(request.headers)}")
        logging.error(f"Request content-type: {request.headers.get('content-type', 'not set')}")
        
        # Create user-friendly error message
        error_messages = []
        for error in error_details:
            loc = " -> ".join(str(x) for x in error.get("loc", []))
            msg = error.get("msg", "Validation error")
            error_messages.append(f"{loc}: {msg}")
        
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="INVALID_INPUT",
                message=f"Input validation failed: {'; '.join(error_messages)}",
                request_id=req_id or "",
                details={"errors": error_details, "path": request.url.path}
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
@app.get("/", tags=["health"])
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


@app.get("/swagger.yaml", include_in_schema=False)
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