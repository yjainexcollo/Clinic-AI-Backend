"""
Helper functions for structured per-visit LLM interaction logging.
Collection: llm_interaction
"""

from datetime import datetime
from typing import Any, Dict, Optional
import logging

from clinicai.adapters.db.mongo.models.patient_m import (
    AgentLog,
    IntakeQuestionLog,
    LLMCallLog,
    LLMInteractionVisit,
)
from clinicai.core.config import get_settings

logger = logging.getLogger("clinicai")


async def _get_or_create(visit_id: str, patient_id: str) -> LLMInteractionVisit:
    doc = await LLMInteractionVisit.find_one(LLMInteractionVisit.visit_id == visit_id)
    if doc:
        return doc
    doc = LLMInteractionVisit(visit_id=visit_id, patient_id=patient_id)
    await doc.insert()
    return doc


async def append_intake_agent_log(
    *,
    visit_id: str,
    patient_id: str,
    question_number: int,
    question_text: Optional[str],
    agent_name: str,
    user_prompt: Any,
    response_text: Any,
    metadata: Optional[Dict[str, Any]] = None,
    question_id: Optional[str] = None,
    asked_at: Optional[datetime] = None,
) -> None:
    """Append an agent interaction under a specific intake question."""
    # Check if LLM interaction logging is enabled
    settings = get_settings()
    if not settings.llm_interaction.enabled:
        return
    
    # Skip logging if visit_id or patient_id are empty
    if not visit_id or not patient_id:
        if settings.llm_interaction.enable_debug_logging:
            logger.debug(f"[{agent_name}] Skipping log: empty visit_id or patient_id")
        return
    
    # Debug logging if enabled
    if settings.llm_interaction.enable_debug_logging:
        logger.debug(
            f"[{agent_name}] Logging: visit_id={visit_id}, patient_id={patient_id}, "
            f"question_number={question_number}"
        )
    
    doc = await _get_or_create(visit_id, patient_id)

    # find or create the question entry
    q = next((q for q in doc.intake if q.question_number == question_number), None)
    if not q:
        q = IntakeQuestionLog(
            question_number=question_number,
            question_id=question_id,
            question_text=question_text,
            asked_at=asked_at or datetime.utcnow(),
        )
        doc.intake.append(q)
    else:
        # Update question_text if provided and current one is None/empty
        if question_text and (not q.question_text or q.question_text is None):
            q.question_text = question_text
        # Update question_id if provided and current one is None/empty
        if question_id and (not q.question_id or q.question_id is None):
            q.question_id = question_id

    # append agent log
    q.agents.append(
        AgentLog(
            agent_name=agent_name,
            user_prompt=user_prompt,
            response_text=response_text,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
        )
    )

    doc.updated_at = datetime.utcnow()
    await doc.save()


async def append_phase_call(
    *,
    visit_id: str,
    patient_id: str,
    phase: str,
    agent_name: str,
    user_prompt: Any,
    response_text: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Set an LLM call for phases other than intake (only one per visit per phase).
    phase: one of ["pre_visit_summary", "soap", "post_visit_summary"]
    
    Note: user_prompt will be converted to string if it's a dict.
    """
    # Check if LLM interaction logging is enabled
    settings = get_settings()
    if not settings.llm_interaction.enabled:
        return
    
    # Skip logging if visit_id or patient_id are empty
    if not visit_id or not patient_id:
        if settings.llm_interaction.enable_debug_logging:
            logger.debug(f"[{agent_name}] Skipping log: empty visit_id or patient_id")
        return
    
    # Debug logging if enabled
    if settings.llm_interaction.enable_debug_logging:
        logger.debug(
            f"[{agent_name}] Logging phase call: visit_id={visit_id}, patient_id={patient_id}, phase={phase}"
        )
    
    doc = await _get_or_create(visit_id, patient_id)

    # Convert user_prompt to string if it's a dict
    if isinstance(user_prompt, dict):
        import json
        user_prompt_str = json.dumps(user_prompt, indent=2, ensure_ascii=False)
    else:
        user_prompt_str = str(user_prompt) if user_prompt is not None else ""

    call = LLMCallLog(
        agent_name=agent_name,
        user_prompt=user_prompt_str,
        response_text=response_text,
        metadata=metadata or {},
        created_at=datetime.utcnow(),
    )

    if phase == "pre_visit_summary":
        doc.pre_visit_summary = call
    elif phase == "soap":
        doc.soap = call
    elif phase == "post_visit_summary":
        doc.post_visit_summary = call
    else:
        raise ValueError(f"Unsupported phase '{phase}' for LLM interaction logging.")

    doc.updated_at = datetime.utcnow()
    await doc.save()

