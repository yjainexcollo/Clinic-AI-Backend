"""
FastAPI application factory and main app configuration.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from .api.routers import health, patients, notes, prescriptions
from .api.routers import doctor as doctor_router
from .api.routers import transcription as transcription_router
from .core.config import get_settings
from .domain.errors import DomainError
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"🚀 Starting Clinic-AI Intake Assistant v{settings.app_version}")
    print(f"📊 Environment: {settings.app_env}")
    print(f"🔧 Debug mode: {settings.debug}")
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
        )

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
            document_models=[PatientMongo, VisitMongo, MedicationImageMongo, AdhocTranscriptMongo, DoctorPreferencesMongo],
        )
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

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
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
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

    # Include routers
    app.include_router(health.router)
    app.include_router(patients.router)
    app.include_router(notes.router)
    app.include_router(prescriptions.router)
    app.include_router(transcription_router.router)
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
        "docs": "/docs" if settings.debug else "disabled",
        "endpoints": {
            "health": "/health",
            "register_patient": "POST /patients/",
            "answer_intake": "POST /patients/consultations/answer",
            "pre_visit_summary": "POST /patients/summary/previsit",
            "get_summary": "GET /patients/{patient_id}/visits/{visit_id}/summary",
            # Image upload endpoints
            "upload_images": "POST /patients/webhook/images",
            "upload_single_image": "POST /patients/webhook/image",
            "get_image_content": "GET /patients/images/{image_id}/content",
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
            # Prescription endpoints
            "upload_prescriptions": "POST /prescriptions/upload",
            # Doctor preferences
            "get_doctor_preferences": "GET /doctor/preferences",
            "save_doctor_preferences": "POST /doctor/preferences",
            # Intake session (preferences-aware)
            "start_intake": "POST /intake/start",
            "next_question": "POST /intake/next-question",
        },
    }