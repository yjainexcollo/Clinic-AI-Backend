"""
Standalone startup script for transcription worker.
Run this as a separate process/service for production deployments.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Entry point for standalone worker process."""
    try:
        # Initialize database connection (required for worker)
        from beanie import init_beanie
        from motor.motor_asyncio import AsyncIOMotorClient
        import certifi
        
        from clinicai.core.config import get_settings
        from clinicai.adapters.db.mongo.models.patient_m import (
            PatientMongo,
            VisitMongo,
            MedicationImageMongo,
            AdhocTranscriptMongo,
            DoctorPreferencesMongo,
            AudioFileMongo,
        )
        from clinicai.adapters.db.mongo.models.blob_file_reference import BlobFileReference
        
        settings = get_settings()
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
            client = AsyncIOMotorClient(
                mongo_uri,
                serverSelectionTimeoutMS=15000,
            )
        
        db = client[db_name]
        await init_beanie(
            database=db,
            document_models=[
                PatientMongo,
                VisitMongo,
                MedicationImageMongo,
                AdhocTranscriptMongo,
                DoctorPreferencesMongo,
                AudioFileMongo,
                BlobFileReference
            ],
        )
        logger.info("✅ Database connection established")
        
        # Start worker
        from clinicai.workers.transcription_worker import TranscriptionWorker
        worker = TranscriptionWorker()
        await worker.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Worker stopped by user")
    except Exception as e:
        logger.error(f"❌ Worker startup failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

