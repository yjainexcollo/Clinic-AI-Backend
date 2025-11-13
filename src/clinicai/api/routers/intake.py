"""
Independent intake session endpoints that leverage doctor's preferences.

This module does NOT modify the existing intake flow endpoints. It provides a
lightweight session keyed by patient_id, storing the doctor's selected
categories and max_questions for the duration of a single intake.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel, Field

from ...adapters.db.mongo.models.patient_m import DoctorPreferencesMongo
from ..deps import QuestionServiceDep
from ..schemas.common import ApiResponse, ErrorResponse
from ..utils.responses import ok, fail


router = APIRouter(prefix="/intake", tags=["intake"])


# Simple in-memory session store
_SESSION_TTL_MINUTES = 120
_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _now() -> datetime:
    return datetime.utcnow()


def _cleanup_sessions() -> None:
    cutoff = _now() - timedelta(minutes=_SESSION_TTL_MINUTES)
    stale_keys = [k for k, v in _SESSIONS.items() if (v.get("updated_at") or v.get("created_at") or cutoff) < cutoff]
    for k in stale_keys:
        _SESSIONS.pop(k, None)


class IntakeStartRequest(BaseModel):
    patient_id: str = Field(...)
    doctor_id: str = Field(...)


class IntakeStartResponse(BaseModel):
    patient_id: str
    doctor_id: str
    categories: List[str]
    max_questions: int
    asked_count: int
    created_at: datetime


class NextQuestionRequest(BaseModel):
    patient_id: str
    last_answer: Optional[str] = None


class NextQuestionResponse(BaseModel):
    question: str
    asked_count: int
    max_questions: int


async def _load_preferences(doctor_id: str) -> Dict[str, Any]:
    doc = await DoctorPreferencesMongo.find_one(DoctorPreferencesMongo.doctor_id == doctor_id)
    if not doc:
        # Fallback defaults (match doctor router defaults)
        default_globals = [
            "duration","triggers","pain","temporal","travel",
            "allergies","medications","hpi","family","lifestyle",
            "gyn","functional","other",
        ]
        return {
            "doctor_id": doctor_id,
            "categories": [],
            "max_questions": 10,
            "global_categories": default_globals,
        }
    return {
        "doctor_id": doctor_id,
        "categories": list(dict.fromkeys([c.strip().lower() for c in (doc.selected_categories or [])])),
        "max_questions": max(1, min(14, int(doc.max_questions or 10))),
        "global_categories": list(dict.fromkeys([c.strip().lower() for c in (doc.global_categories or [])])),
    }


@router.post("/start", response_model=ApiResponse[IntakeStartResponse], status_code=status.HTTP_200_OK)
async def start_intake(request: Request, req: IntakeStartRequest):
    """Initialize an intake session for a patient using the doctor's preferences."""
    _cleanup_sessions()
    try:
        prefs = await _load_preferences(req.doctor_id)

        session = {
            "patient_id": req.patient_id,
            "doctor_id": prefs["doctor_id"],
            "categories": prefs["categories"],
            "max_questions": max(1, min(14, int(prefs["max_questions"]))),
            "asked_count": 0,
            "asked_questions": [],  # type: List[str]
            "previous_answers": [],  # type: List[str]
            "created_at": _now(),
            "updated_at": _now(),
        }
        _SESSIONS[req.patient_id] = session

        result = IntakeStartResponse(
            patient_id=req.patient_id,
            doctor_id=prefs["doctor_id"],
            categories=prefs["categories"],
            max_questions=prefs["max_questions"],
            asked_count=0,
            created_at=session["created_at"],
        )
        return ok(request, data=result, message="Intake started")
    except Exception as e:
        return fail(request, error="INTERNAL_ERROR", message="Failed to start intake")


@router.post("/next-question", response_model=ApiResponse[NextQuestionResponse])
async def next_question(request: Request, req: NextQuestionRequest, question_service: QuestionServiceDep):
    """Generate the next intake question using session-configured limits."""
    session = _SESSIONS.get(req.patient_id)
    if not session:
        raise HTTPException(status_code=404, detail={"error": "SESSION_NOT_FOUND", "message": "Start intake first."})

    asked_count = int(session.get("asked_count", 0))
    max_questions = max(1, min(14, int(session.get("max_questions", 10))))

    # Stop if limit reached
    if asked_count >= max_questions:
        return NextQuestionResponse(question="COMPLETE", asked_count=asked_count, max_questions=max_questions)

    # Update previous answers
    if req.last_answer:
        prev = list(session.get("previous_answers") or [])
        prev.append(req.last_answer)
        session["previous_answers"] = prev

    # Build inputs for the question service
    disease = ""  # Not provided here; prompt update will be handled separately
    previous_answers: List[str] = list(session.get("previous_answers") or [])
    asked_questions: List[str] = list(session.get("asked_questions") or [])
    current_count = asked_count
    max_count = max_questions

    # Generate next question
    question_text = await question_service.generate_next_question(
        disease=disease,
        previous_answers=previous_answers,
        asked_questions=asked_questions,
        current_count=current_count,
        max_count=max_count,
        recently_travelled=False,
        prior_summary=None,
        prior_qas=None,
        patient_gender=None,
        patient_age=None,
    )

    # Update session state
    asked_questions.append(question_text)
    asked_count += 1
    session["asked_questions"] = asked_questions
    session["asked_count"] = asked_count
    session["updated_at"] = _now()
    _SESSIONS[req.patient_id] = session

    # Enforce stopping after reaching limit
    if asked_count >= max_questions:
        # Optionally return closing question already generated; client can decide to stop
        pass

    result = NextQuestionResponse(
        question=question_text if asked_count <= max_questions else "COMPLETE",
        asked_count=asked_count,
        max_questions=max_questions,
    )
    return ok(request, data=result, message="Next question")

