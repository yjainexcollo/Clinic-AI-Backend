import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from clinicai.adapters.db.mongo.models.patient_m import VisitMongo
from clinicai.adapters.queue.azure_queue_service import get_azure_queue_service
from clinicai.core.config import get_settings

logger = logging.getLogger("clinicai")


async def _sweep_once(threshold_seconds: int) -> None:
    """
    Perform a single sweep to find and re-enqueue stuck queued transcription jobs.
    """
    now = datetime.utcnow()
    threshold = now - timedelta(seconds=threshold_seconds)

    # Find visits with queued transcription that appear stuck
    cursor = VisitMongo.find(
        {
            "transcription_session.transcription_status": "queued",
            "transcription_session.dequeued_at": None,
            "transcription_session.worker_id": None,
            "transcription_session.transcription_id": None,
            "transcription_session.enqueued_at": {"$lte": threshold},
        }
    )

    queue_service = get_azure_queue_service()
    settings = get_settings()

    async for visit in cursor:
        visit_id = visit.visit_id
        patient_id = visit.patient_id
        # Extract doctor_id from visit (required for multi-doctor support)
        doctor_id = getattr(visit, "doctor_id", None) or "D123"  # Fallback for backward compatibility
        ts = visit.transcription_session
        if not ts:
            continue

        logger.info(
            "[StuckSweeper] Found stuck queued visit=%s patient=%s doctor_id=%s enqueued_at=%s queue_message_id=%s",
            visit_id,
            patient_id,
            doctor_id,
            getattr(ts, "enqueued_at", None),
            getattr(ts, "queue_message_id", None),
        )

        # If we already have a queue_message_id and enqueue_state queued, leave it;
        # infra/worker should pick it up once available.
        if getattr(ts, "enqueue_state", None) == "queued" and getattr(ts, "queue_message_id", None):
            continue

        # We need an audio reference to re-enqueue. Use AudioFileMongo by visit_id.
        from clinicai.adapters.db.mongo.models.audio_m import (  # lazy import to avoid cycles
            AudioFileMongo,
        )

        audio = await AudioFileMongo.find_one(AudioFileMongo.visit_id == visit_id, AudioFileMongo.audio_type == "visit")
        if not audio or not getattr(audio, "audio_id", None):
            logger.warning(
                "[StuckSweeper] Cannot re-enqueue visit=%s patient=%s: missing audio reference",
                visit_id,
                patient_id,
            )
            continue

        # Build minimal enqueue payload
        request_id: Optional[str] = None
        backoff_schedule = [0.5, 2.0, 5.0]
        last_error: Optional[Exception] = None
        message_id: Optional[str] = None

        for attempt, delay in enumerate(backoff_schedule, start=1):
            try:
                message_id = await queue_service.enqueue_transcription_job(
                    patient_id=patient_id,
                    visit_id=visit_id,
                    audio_file_id=audio.audio_id,
                    doctor_id=getattr(visit, "doctor_id", "D123"),
                    language="en",  # language is not stored per visit audio today; default to en
                    retry_count=attempt - 1,
                    delay_seconds=0,
                    request_id=request_id,
                    doctor_id=doctor_id,
                )
                break
            except Exception as e:  # noqa: PERF203
                last_error = e
                logger.error(
                    "[StuckSweeper] enqueue_transcription_job attempt %d failed for visit=%s, audio_file_id=%s: %s",
                    attempt,
                    visit_id,
                    audio.audio_id,
                    e,
                    exc_info=True,
                )
                if attempt < len(backoff_schedule):
                    await asyncio.sleep(delay)

        if not message_id:
            logger.error(
                "[StuckSweeper] STUCK_QUEUED_REENQUEUE_FAILED visit=%s patient=%s error=%s",
                visit_id,
                patient_id,
                last_error,
            )
            # Mark enqueue_state failed for observability
            ts.enqueue_state = "failed"
            ts.enqueue_failed_at = datetime.utcnow()
            ts.enqueue_last_error = f"STUCK_SWEEPER_REENQUEUE_FAILED: {last_error}"
            await visit.save()
            continue

        # Success: mark as queued in Mongo; keep transcription_status as 'queued'
        ts.enqueue_state = "queued"
        ts.queue_message_id = message_id
        ts.enqueued_at = datetime.utcnow()
        await visit.save()

        logger.info(
            "[StuckSweeper] STUCK_QUEUED_REENQUEUE success visit=%s patient=%s message_id=%s",
            visit_id,
            patient_id,
            message_id,
        )


async def run_stuck_sweeper_forever() -> None:
    """
    Run the stuck queued sweeper in a loop, controlled by environment settings.
    """
    settings = get_settings()
    if not settings.transcription_stuck_sweeper_enabled:
        logger.info("[StuckSweeper] Disabled via TRANSCRIPTION_STUCK_SWEEPER_ENABLED")
        return

    interval = max(30, settings.transcription_stuck_sweeper_interval_seconds)
    threshold = max(60, settings.transcription_stuck_threshold_seconds)

    logger.info(
        "[StuckSweeper] Starting (interval=%ss, threshold=%ss, queue=%s)",
        interval,
        threshold,
        settings.azure_queue.queue_name,
    )

    while True:
        try:
            await _sweep_once(threshold_seconds=threshold)
        except Exception as e:  # noqa: PERF203
            logger.error("[StuckSweeper] Sweep iteration failed: %s", e, exc_info=True)
        await asyncio.sleep(interval)
