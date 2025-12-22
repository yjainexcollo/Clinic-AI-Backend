"""Migration script to backfill doctor_id for existing patients and visits.

Usage (from backend root, with venv active):

    python -m clinicai.scripts.migrate_to_multi_doctor

This will:
1. Ensure a default doctor with doctor_id="D123" exists.
2. Set doctor_id="D123" for any PatientMongo/VisitMongo that do not have doctor_id.
3. Ensure default DoctorPreferencesMongo for "D123".
"""

import asyncio
from datetime import datetime

from beanie import init_beanie  # type: ignore
from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore

from clinicai.core.config import get_settings
from clinicai.adapters.db.mongo.models.patient_m import (
    PatientMongo,
    VisitMongo,
    DoctorPreferencesMongo,
)
from clinicai.adapters.db.mongo.models.doctor_m import DoctorMongo
from clinicai.adapters.db.mongo.models.blob_file_reference import BlobFileReference
from clinicai.adapters.db.mongo.models.prompt_version_m import PromptVersionMongo


async def migrate() -> None:
    settings = get_settings()
    client = AsyncIOMotorClient(settings.database.uri)
    db = client[settings.database.db_name]

    await init_beanie(
        database=db,
        document_models=[
            PatientMongo,
            VisitMongo,
            DoctorMongo,
            DoctorPreferencesMongo,
            BlobFileReference,
            PromptVersionMongo,
        ],
    )

    default_doctor_id = "D123"

    # 1) Ensure default doctor exists
    doctor = await DoctorMongo.find_one(DoctorMongo.doctor_id == default_doctor_id)
    if not doctor:
        doctor = DoctorMongo(
            doctor_id=default_doctor_id,
            name="Default Doctor",
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        await doctor.save()
        print(f"Created default doctor with doctor_id={default_doctor_id}")
    else:
        print(f"Default doctor already exists with doctor_id={default_doctor_id}")

    # 2) Backfill doctor_id on patients
    patients_without_doctor = await PatientMongo.find(
        {"$or": [{"doctor_id": {"$exists": False}}, {"doctor_id": ""}]}
    ).to_list()
    print(f"Found {len(patients_without_doctor)} patients without doctor_id")
    for p in patients_without_doctor:
        p.doctor_id = default_doctor_id
        p.updated_at = datetime.utcnow()
        await p.save()
    print(f"Updated {len(patients_without_doctor)} patients with doctor_id={default_doctor_id}")

    # 3) Backfill doctor_id on visits
    visits_without_doctor = await VisitMongo.find(
        {"$or": [{"doctor_id": {"$exists": False}}, {"doctor_id": ""}]}
    ).to_list()
    print(f"Found {len(visits_without_doctor)} visits without doctor_id")
    for v in visits_without_doctor:
        v.doctor_id = default_doctor_id
        v.updated_at = datetime.utcnow()
        await v.save()
    print(f"Updated {len(visits_without_doctor)} visits with doctor_id={default_doctor_id}")

    # 4) Ensure default doctor preferences exist
    prefs = await DoctorPreferencesMongo.find_one(
        DoctorPreferencesMongo.doctor_id == default_doctor_id
    )
    if not prefs:
        prefs = DoctorPreferencesMongo(
            doctor_id=default_doctor_id,
            global_categories=[],
            selected_categories=[],
            max_questions=settings.intake.max_questions,
            updated_at=datetime.utcnow(),
        )
        await prefs.save()
        print(f"Created default DoctorPreferences for doctor_id={default_doctor_id}")
    else:
        print(f"DoctorPreferences already exist for doctor_id={default_doctor_id}")

    print("Migration to multi-doctor complete.")


if __name__ == "__main__":
    asyncio.run(migrate())


