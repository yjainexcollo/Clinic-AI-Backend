from fastapi import APIRouter, status, Request
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

from ...adapters.db.mongo.models.patient_m import DoctorPreferencesMongo
from ...core.config import get_settings
from ..schemas.common import ApiResponse, ErrorResponse
from ..utils.responses import ok, fail

router = APIRouter(prefix="/doctor")


DEFAULT_DOCTOR_ID = "D123"
DEFAULT_GLOBAL_CATEGORIES = [
    "duration","triggers","pain","temporal","travel",
    "allergies","medications","hpi","family","lifestyle",
    "gyn","functional","other"
]


class DoctorPreferencesResponse(BaseModel):
    doctor_id: str
    global_categories: List[str]
    selected_categories: List[str]
    max_questions: int


class UpsertPreferencesRequest(BaseModel):
    categories: List[str] = Field(default_factory=list)
    max_questions: int = Field(ge=1, le=12)
    # Optional authoritative list of global categories (for rename/remove operations)
    global_categories: Optional[List[str]] = None


async def _get_or_default(doctor_id: str) -> Optional[DoctorPreferencesMongo]:
    doc = await DoctorPreferencesMongo.find_one(DoctorPreferencesMongo.doctor_id == doctor_id)
    return doc


def _defaults_for(doctor_id: str) -> DoctorPreferencesResponse:
    from clinicai.core.config import get_settings
    settings = get_settings()
    return DoctorPreferencesResponse(
        doctor_id=doctor_id,
        global_categories=DEFAULT_GLOBAL_CATEGORIES.copy(),
        selected_categories=[],
        max_questions=settings.intake.max_questions,
    )


@router.get("/preferences", response_model=ApiResponse[DoctorPreferencesResponse], include_in_schema=False)
async def get_doctor_preferences(request: Request):
    """Return latest preferences for the default doctor; independent of intake flow."""
    try:
        doctor_id = DEFAULT_DOCTOR_ID
        doc = await _get_or_default(doctor_id)
        if not doc:
            return ok(request, data=_defaults_for(doctor_id), message="Doctor preferences loaded")
        return ok(request, data=DoctorPreferencesResponse(
            doctor_id=doc.doctor_id,
            global_categories=sorted(list(dict.fromkeys([c.lower() for c in (doc.global_categories or [])]))),
            selected_categories=sorted(list(dict.fromkeys([c.lower() for c in (doc.selected_categories or [])]))),
            max_questions=int(doc.max_questions or get_settings().intake.max_questions),
        ), message="Doctor preferences loaded")
    except Exception as e:
        return fail(request, error="INTERNAL_ERROR", message="Failed to load preferences")


@router.post("/preferences", response_model=ApiResponse[DoctorPreferencesResponse], status_code=status.HTTP_200_OK, include_in_schema=False)
async def set_doctor_preferences(request: Request, payload: UpsertPreferencesRequest):
    """Merge and persist doctor preferences; independent of intake flow."""
    try:
        doctor_id = DEFAULT_DOCTOR_ID
        existing = await _get_or_default(doctor_id)

        # Normalize inputs
        incoming_categories = [c.strip().lower() for c in payload.categories if c and isinstance(c, str)]
        incoming_categories = list(dict.fromkeys(incoming_categories))  # dedupe order-preserving
        settings = get_settings()
        max_q = max(1, min(settings.intake.max_questions, int(payload.max_questions)))
        incoming_global = None
        if payload.global_categories is not None:
            incoming_global = [c.strip().lower() for c in payload.global_categories if c and isinstance(c, str)]
            incoming_global = list(dict.fromkeys(incoming_global))

        if not existing:
            # Create new document from defaults and apply incoming selections
            if incoming_global is not None and len(incoming_global) > 0:
                global_categories = list(dict.fromkeys(DEFAULT_GLOBAL_CATEGORIES + incoming_global))
            else:
                global_categories = list(dict.fromkeys(DEFAULT_GLOBAL_CATEGORIES + incoming_categories))
            doc = DoctorPreferencesMongo(
                doctor_id=doctor_id,
                global_categories=global_categories,
                selected_categories=incoming_categories,
                max_questions=max_q,
                updated_at=datetime.utcnow(),
            )
            await doc.insert()
            return ok(request, data=DoctorPreferencesResponse(
                doctor_id=doctor_id,
                global_categories=sorted(global_categories),
                selected_categories=sorted(incoming_categories),
                max_questions=max_q,
            ), message="Preferences updated")

        # Compute next global categories
        if incoming_global is not None:
            # Authoritative replacement, but ensure defaults are preserved initially only; here honor client list
            merged_global = list(dict.fromkeys(incoming_global))
        else:
            # Merge in any new selections to existing global list
            merged_global = list(dict.fromkeys([(existing.global_categories or []) + incoming_categories][0]))
        existing.global_categories = merged_global
        # Only keep selections that exist in global to support removals/renames handled client-side
        existing.selected_categories = [c for c in incoming_categories if c in set(merged_global)]
        existing.max_questions = max_q
        existing.updated_at = datetime.utcnow()
        await existing.save()

        return ok(request, data=DoctorPreferencesResponse(
            doctor_id=doctor_id,
            global_categories=sorted(merged_global),
            selected_categories=sorted([c for c in incoming_categories if c in set(merged_global)]),
            max_questions=max_q,
        ), message="Preferences updated")
    except Exception as e:
        return fail(request, error="INTERNAL_ERROR", message="Failed to update preferences")

