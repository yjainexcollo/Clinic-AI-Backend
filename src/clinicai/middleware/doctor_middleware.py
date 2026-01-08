"""Doctor middleware to extract and validate X-Doctor-ID per request."""

import logging
import os
import re

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from clinicai.adapters.db.mongo.models.doctor_m import DoctorMongo

logger = logging.getLogger(__name__)


class DoctorMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce and attach doctor_id from X-Doctor-ID header."""

    # Align public paths with AuthenticationMiddleware
    PUBLIC_PATHS = {
        "/",
        "/favicon.ico",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/health/live",
        "/health/ready",
    }
    PUBLIC_PATH_PREFIXES = {
        "/docs",
        "/redoc",
    }

    def is_public_endpoint(self, path: str) -> bool:
        normalized_path = path.rstrip("/") or "/"
        if normalized_path in self.PUBLIC_PATHS:
            return True
        for prefix in self.PUBLIC_PATH_PREFIXES:
            if path.startswith(prefix + "/"):
                return True
        return False

    def _validate_doctor_id(self, doctor_id: str) -> bool:
        if not doctor_id or len(doctor_id) < 1 or len(doctor_id) > 100:
            return False
        return bool(re.match(r"^[A-Za-z0-9_-]+$", doctor_id))

    async def dispatch(self, request: Request, call_next):
        # Skip public endpoints
        if self.is_public_endpoint(request.url.path):
            return await call_next(request)

        doctor_id = request.headers.get("X-Doctor-ID") or request.headers.get("x-doctor-id")

        # In strict mode we require the header. In non-strict mode we fall back to a default doctor ID.
        require_header = os.getenv("REQUIRE_DOCTOR_ID", "false").lower() == "true"

        if not doctor_id:
            if require_header:
                logger.warning(
                    "❌ Missing X-Doctor-ID header for %s %s",
                    request.method,
                    request.url.path,
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "MISSING_DOCTOR_ID",
                        "message": "X-Doctor-ID header is required",
                        "details": {"path": request.url.path, "method": request.method},
                    },
                )
            # Fallback: use default doctor ID for development / non-strict environments
            doctor_id = os.getenv("DEFAULT_DOCTOR_ID", "D123")
            logger.warning(
                "⚠️ Missing X-Doctor-ID header for %s %s, falling back to DEFAULT_DOCTOR_ID=%s",
                request.method,
                request.url.path,
                doctor_id,
            )

        if not self._validate_doctor_id(doctor_id):
            logger.warning("❌ Invalid doctor_id format: %s", doctor_id[:80])
            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_DOCTOR_ID",
                    "message": "doctor_id must be 1-100 chars, alphanumeric, hyphen or underscore",
                    "details": {"doctor_id": doctor_id[:80]},
                },
            )

        # Auto-create doctor if missing
        try:
            doctor = await DoctorMongo.find_one(DoctorMongo.doctor_id == doctor_id)
            if not doctor:
                doctor = DoctorMongo(doctor_id=doctor_id, name=f"Doctor {doctor_id}", status="active")
                await doctor.save()
                logger.info("🆕 Created doctor %s", doctor_id)

            request.state.doctor_id = doctor_id
            request.state.doctor = doctor
        except Exception as exc:
            logger.error("Doctor middleware error: %s", exc, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "DOCTOR_MIDDLEWARE_ERROR",
                    "message": "Failed to process doctor_id",
                },
            )

        return await call_next(request)
