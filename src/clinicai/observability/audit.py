"""Audit logging utilities.

MVP: log structured audit events via standard logger. Later, persist immutably.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional


logger = logging.getLogger("clinicai.audit")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def audit_log_event(
    *,
    event: str,
    patient_id: Optional[str] = None,
    visit_id: Optional[str] = None,
    user_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    record = {
        "ts": _now_iso(),
        "event": event,
        "patient_id": patient_id,
        "visit_id": visit_id,
        "user_id": user_id,
        "payload": payload or {},
    }
    try:
        logger.info("AUDIT %s", json.dumps(record, ensure_ascii=False))
    except Exception:
        # Fallback
        logger.info("AUDIT %s", record)


