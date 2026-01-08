import asyncio
import logging
import os
from typing import Optional

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from clinicai.adapters.db.mongo.models.audio_m import AudioFileMongo
from clinicai.adapters.db.mongo.models.patient_m import (  # ensure models are imported for Beanie
    VisitMongo,
)
from clinicai.core.config import get_settings
from clinicai.workers.transcription_stuck_sweeper import run_stuck_sweeper_forever

logger = logging.getLogger("clinicai")


async def _init_db(settings) -> Optional[AsyncIOMotorClient]:
    """Initialize MongoDB connection for the sweeper."""
    client = AsyncIOMotorClient(settings.database.uri)
    db = client[settings.database.db_name]
    await init_beanie(
        database=db,
        document_models=[VisitMongo, AudioFileMongo],
    )
    return client


async def main() -> None:
    """
    Entry point for the transcription stuck sweeper.

    This process is intended to be run separately from the main worker:
        PYTHONPATH=./src python3 sweeper_startup.py
    """
    settings = get_settings()
    if not settings.transcription_stuck_sweeper_enabled:
        logger.info(
            "Transcription stuck sweeper is disabled. "
            "Set TRANSCRIPTION_STUCK_SWEEPER_ENABLED=true to enable."
        )
        return

    # Startup banner
    logger.info("🚀 Starting transcription stuck sweeper…")
    logger.info(
        "Sweeper config: queue=%s, interval=%ss, threshold=%ss",
        settings.azure_queue.queue_name,
        settings.transcription_stuck_sweeper_interval_seconds,
        settings.transcription_stuck_threshold_seconds,
    )

    client = await _init_db(settings)

    # Graceful shutdown via signals
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("🛑 Shutdown signal received for sweeper, stopping gracefully…")
        stop_event.set()

    import signal

    loop = asyncio.get_event_loop()
    if hasattr(signal, "SIGTERM"):
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGINT"):
        loop.add_signal_handler(signal.SIGINT, _signal_handler)

    try:
        # Run the sweeper loop until stop_event is set
        sweeper_task = asyncio.create_task(run_stuck_sweeper_forever())
        await stop_event.wait()
        sweeper_task.cancel()
        try:
            await sweeper_task
        except asyncio.CancelledError:
            logger.info("Sweeper task cancelled.")
    finally:
        if client:
            client.close()
            logger.info("Sweeper MongoDB client closed.")


if __name__ == "__main__":
    # Allow running as: PYTHONPATH=./src python3 sweeper_startup.py
    asyncio.run(main())
