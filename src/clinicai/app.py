"""
FastAPI application factory and main app configuration.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routers import health, patients, notes, prescriptions
from .core.config import get_settings
from .domain.errors import DomainError


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
        )
        # stable_* models removed

        # Use configured URI and enable TLS with CA bundle for Atlas
        mongo_uri = settings.database.uri
        db_name = settings.database.db_name
        ca_path = certifi.where()
        print(f"🔐 Using certifi CA bundle: {ca_path}")
        client = AsyncIOMotorClient(
            mongo_uri,
            serverSelectionTimeoutMS=15000,
            tls=True,
            tlsCAFile=ca_path,
            tlsAllowInvalidCertificates=False,
        )
        db = client[db_name]
        await init_beanie(
            database=db,
            document_models=[PatientMongo, VisitMongo],
        )
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

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
    # Always include common local dev origins; merge with configured origins
    common_local_origins = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    configured_origins = settings.cors.allowed_origins or []
    allow_origins = list({*common_local_origins, *configured_origins}) or ["*"]

    allow_methods = settings.cors.allowed_methods or ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    # Ensure PATCH and OPTIONS are always allowed for browser preflights
    allow_methods = list({m.upper() for m in allow_methods} | {"PATCH", "OPTIONS"})
    allow_headers = settings.cors.allowed_headers or ["*"]
    # Ensure common headers present
    if allow_headers != ["*"] and "content-type" not in {h.lower() for h in allow_headers}:
        allow_headers = [*allow_headers, "content-type"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=r"https?://.*",
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(patients.router)
    app.include_router(notes.router)
    app.include_router(prescriptions.router)

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
            # Step-03 endpoints
            "transcribe_audio": "POST /notes/transcribe",
            "generate_soap": "POST /notes/soap/generate",
            "get_transcript": "GET /notes/{patient_id}/visits/{visit_id}/transcript",
            "get_soap": "GET /notes/{patient_id}/visits/{visit_id}/soap",
            # Prescription endpoints
            "upload_prescriptions": "POST /prescriptions/upload",
        },
    }