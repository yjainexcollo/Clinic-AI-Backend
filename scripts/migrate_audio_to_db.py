#!/usr/bin/env python3
"""
Migration script to move existing file-based audio to database storage.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clinicai.core.config import get_settings
from clinicai.adapters.db.mongo.repositories.audio_repository import AudioRepository
from clinicai.adapters.db.mongo.models.patient_m import AdhocTranscriptMongo
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import certifi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_audio_files():
    """Migrate existing audio files from file system to database."""
    settings = get_settings()
    
    # Initialize database connection
    try:
        from clinicai.adapters.db.mongo.models.patient_m import (
            PatientMongo,
            VisitMongo,
            MedicationImageMongo,
            AdhocTranscriptMongo,
            DoctorPreferencesMongo,
            AudioFileMongo,
        )

        mongo_uri = settings.database.uri
        db_name = settings.database.db_name

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
            document_models=[PatientMongo, VisitMongo, MedicationImageMongo, AdhocTranscriptMongo, DoctorPreferencesMongo, AudioFileMongo],
        )
        logger.info("✅ Database connection established")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return

    audio_repo = AudioRepository()
    audio_storage_path = settings.file_storage.audio_storage_path
    
    if not os.path.exists(audio_storage_path):
        logger.info(f"📁 Audio storage path does not exist: {audio_storage_path}")
        return

    # Find all audio files in the storage directory
    audio_files = []
    for file_path in Path(audio_storage_path).glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mpeg', '.mpg']:
            audio_files.append(file_path)
    
    logger.info(f"📁 Found {len(audio_files)} audio files to migrate")
    
    if not audio_files:
        logger.info("✅ No audio files to migrate")
        return

    migrated_count = 0
    failed_count = 0

    for file_path in audio_files:
        try:
            logger.info(f"🔄 Migrating: {file_path.name}")
            
            # Read the audio file
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # Determine content type based on file extension
            content_type_map = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.m4a': 'audio/mp4',
                '.flac': 'audio/flac',
                '.ogg': 'audio/ogg',
                '.mpeg': 'audio/mpeg',
                '.mpg': 'audio/mpeg',
            }
            content_type = content_type_map.get(file_path.suffix.lower(), 'audio/mpeg')
            
            # Try to find associated adhoc transcript
            adhoc_id = None
            try:
                # Look for adhoc transcripts with this file path
                adhoc_transcripts = await AdhocTranscriptMongo.find(
                    AdhocTranscriptMongo.audio_file_path == str(file_path)
                ).to_list()
                
                if adhoc_transcripts:
                    adhoc_id = str(adhoc_transcripts[0].id)
                    logger.info(f"📝 Found associated adhoc transcript: {adhoc_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not find associated adhoc transcript: {e}")
            
            # Create audio file record in database
            audio_file_record = await audio_repo.create_audio_file(
                audio_data=audio_data,
                filename=file_path.name,
                content_type=content_type,
                adhoc_id=adhoc_id,
                audio_type="adhoc" if adhoc_id else "migrated",
            )
            
            logger.info(f"✅ Migrated: {file_path.name} -> {audio_file_record.audio_id}")
            migrated_count += 1
            
            # Update adhoc transcript to link to the new audio file
            if adhoc_id:
                try:
                    adhoc_transcript = await AdhocTranscriptMongo.get(adhoc_id)
                    if adhoc_transcript:
                        adhoc_transcript.audio_file_path = None  # Remove file path reference
                        await adhoc_transcript.save()
                        logger.info(f"📝 Updated adhoc transcript {adhoc_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not update adhoc transcript: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate {file_path.name}: {e}")
            failed_count += 1

    logger.info(f"🎉 Migration completed!")
    logger.info(f"   ✅ Migrated: {migrated_count} files")
    logger.info(f"   ❌ Failed: {failed_count} files")
    
    if migrated_count > 0:
        logger.info("💡 You can now safely delete the audio files from the file system if desired.")
        logger.info(f"   Files are located at: {audio_storage_path}")


async def main():
    """Main function."""
    logger.info("🚀 Starting audio file migration to database...")
    await migrate_audio_files()
    logger.info("🏁 Migration script completed")


if __name__ == "__main__":
    asyncio.run(main())
