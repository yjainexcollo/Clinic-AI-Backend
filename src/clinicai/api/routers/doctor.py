from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Request, status

from ...adapters.db.mongo.models.patient_m import DoctorPreferencesMongo
from ...core.config import get_settings
from ..schemas.common import ApiResponse
from ..schemas.doctor_preferences import (
    DoctorPreferencesResponse,
    UpsertDoctorPreferencesRequest,
    PreVisitSectionConfig,
)
from ..utils.responses import fail, ok

router = APIRouter(prefix="/doctor")


DEFAULT_DOCTOR_ID = "D123"
DEFAULT_GLOBAL_CATEGORIES = [
    "duration", "triggers", "pain", "temporal", "travel",
    "allergies", "medications", "hpi", "family", "lifestyle",
    "gyn", "functional", "other"
]


async def _get_or_default(doctor_id: str) -> Optional[DoctorPreferencesMongo]:
    return await DoctorPreferencesMongo.find_one(DoctorPreferencesMongo.doctor_id == doctor_id)


def _defaults_for(doctor_id: str) -> DoctorPreferencesResponse:
    settings = get_settings()
    return DoctorPreferencesResponse(
        doctor_id=doctor_id,
        global_categories=DEFAULT_GLOBAL_CATEGORIES.copy(),
        selected_categories=[],
        max_questions=settings.intake.max_questions,
        soap_order=[],
        pre_visit_config=[],
        pre_visit_ai_config=None,
        soap_ai_config=None,
    )


def _merge_intake_prefs(existing: DoctorPreferencesMongo, payload: UpsertDoctorPreferencesRequest, settings) -> None:
    # Normalize incoming legacy intake fields
    incoming_categories = []
    if payload.categories is not None:
        incoming_categories = [c.strip().lower() for c in payload.categories if c and isinstance(c, str)]
        incoming_categories = list(dict.fromkeys(incoming_categories))

    if payload.global_categories is not None:
        incoming_global = [c.strip().lower() for c in payload.global_categories if c and isinstance(c, str)]
        incoming_global = list(dict.fromkeys(incoming_global))
    else:
        incoming_global = None

    if payload.max_questions is not None:
        try:
            max_q = max(1, min(settings.intake.max_questions, int(payload.max_questions)))
        except Exception:
            max_q = settings.intake.max_questions
    else:
        max_q = existing.max_questions if existing and existing.max_questions else settings.intake.max_questions

    if not existing.global_categories:
        existing.global_categories = DEFAULT_GLOBAL_CATEGORIES.copy()

    if incoming_global is not None:
        merged_global = list(dict.fromkeys(incoming_global))
    else:
        merged_global = list(dict.fromkeys((existing.global_categories or []) + incoming_categories))

    existing.global_categories = merged_global
    existing.selected_categories = [c for c in incoming_categories if c in set(merged_global)] if incoming_categories else (existing.selected_categories or [])
    existing.max_questions = max_q


def _apply_new_prefs(existing: DoctorPreferencesMongo, payload: UpsertDoctorPreferencesRequest) -> None:
    # New fields are optional; only set if provided
    if payload.soap_order is not None:
        existing.soap_order = payload.soap_order
    if payload.pre_visit_config is not None:
        # Store as plain dicts for flexibility / backward compatibility
        existing.pre_visit_config = [cfg.dict() for cfg in payload.pre_visit_config]
    if payload.pre_visit_ai_config is not None:
        existing.pre_visit_ai_config = payload.pre_visit_ai_config.dict()
    if payload.soap_ai_config is not None:
        existing.soap_ai_config = payload.soap_ai_config.dict()


def _build_response(doc: DoctorPreferencesMongo, settings) -> DoctorPreferencesResponse:
    return DoctorPreferencesResponse(
        doctor_id=doc.doctor_id,
        global_categories=sorted(list(dict.fromkeys([c.lower() for c in (doc.global_categories or [])]))),
        selected_categories=sorted(list(dict.fromkeys([c.lower() for c in (doc.selected_categories or [])]))),
        max_questions=int(doc.max_questions or settings.intake.max_questions),
        soap_order=doc.soap_order or [],
        pre_visit_config=[PreVisitSectionConfig(**cfg) if isinstance(cfg, dict) else cfg for cfg in (doc.pre_visit_config or [])],
        pre_visit_ai_config=doc.pre_visit_ai_config,
        soap_ai_config=doc.soap_ai_config,
    )


@router.get(
    "/preferences",
    response_model=ApiResponse[DoctorPreferencesResponse],
    include_in_schema=True,
    tags=["Doctor Preferences"],
    summary="Get doctor preferences for AI summaries and SOAP notes",
    description=(
        "Returns doctor-specific preferences that customize:\n"
        "- Pre-visit summaries\n"
        "- SOAP note generation\n\n"
        "Includes both:\n"
        "- Frontend configuration (soap_order, pre_visit_config)\n"
        "- AI configuration (style, detail level, formatting)\n\n"
        "Backward compatible with legacy intake preferences."
    ),
)
async def get_doctor_preferences(
    request: Request,
    doctor_id: str = DEFAULT_DOCTOR_ID,
):
    """Return latest preferences for the given doctor; independent of intake flow."""
    try:
        doc = await _get_or_default(doctor_id)
        if not doc:
            return ok(request, data=_defaults_for(doctor_id), message="Doctor preferences loaded")
        return ok(request, data=_build_response(doc, get_settings()), message="Doctor preferences loaded")
    except Exception:
        return fail(request, error="INTERNAL_ERROR", message="Failed to load preferences")


@router.post(
    "/preferences",
    response_model=ApiResponse[DoctorPreferencesResponse],
    status_code=status.HTTP_200_OK,
    include_in_schema=True,
    tags=["Doctor Preferences"],
    summary="Create or update doctor preferences",
    description=(
        "Creates or updates doctor-specific preferences that customize:\n"
        "- Pre-visit summaries\n"
        "- SOAP note generation\n\n"
        "Accepts both:\n"
        "- Frontend configuration (soap_order, pre_visit_config)\n"
        "- AI configuration (style, detail level, formatting)\n\n"
        "Backward compatible with legacy intake preference fields."
    ),
)
async def set_doctor_preferences(request: Request, payload: UpsertDoctorPreferencesRequest):
    """Merge and persist doctor preferences; independent of intake flow."""
    try:
        doctor_id = payload.doctor_id or DEFAULT_DOCTOR_ID
        existing = await _get_or_default(doctor_id)
        settings = get_settings()

        if not existing:
            existing = DoctorPreferencesMongo(
                doctor_id=doctor_id,
                global_categories=DEFAULT_GLOBAL_CATEGORIES.copy(),
                selected_categories=[],
                max_questions=settings.intake.max_questions,
                updated_at=datetime.utcnow(),
            )

        # Merge legacy intake prefs (if provided)
        _merge_intake_prefs(existing, payload, settings)

        # Apply new doctor preference fields (soap/pre-visit)
        _apply_new_prefs(existing, payload)

        existing.updated_at = datetime.utcnow()
        await existing.save()

        return ok(request, data=_build_response(existing, settings), message="Preferences updated")
    except Exception:
        return fail(request, error="INTERNAL_ERROR", message="Failed to update preferences")

