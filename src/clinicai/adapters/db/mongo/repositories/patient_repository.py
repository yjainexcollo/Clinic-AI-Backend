"""
MongoDB implementation of PatientRepository.
"""

from datetime import datetime
from typing import List, Optional

from clinicai.application.ports.repositories.patient_repo import PatientRepository
from clinicai.domain.entities.patient import Patient
from clinicai.domain.value_objects.patient_id import PatientId

from ..models.patient_m import PatientMongo


class MongoPatientRepository(PatientRepository):
    """MongoDB implementation of PatientRepository."""

    async def save(self, patient: Patient) -> Patient:
        """Save a patient to MongoDB."""
        # Convert domain entity to MongoDB model
        patient_mongo = await self._domain_to_mongo(patient)

        # Save to database
        await patient_mongo.save()

        # Remove revision_id from the document after saving using raw MongoDB operations
        if patient_mongo.id:
            from motor.motor_asyncio import AsyncIOMotorClient

            from clinicai.core.config import get_settings

            settings = get_settings()
            client = AsyncIOMotorClient(settings.database.uri)
            db = client[settings.database.db_name]
            collection = db["patients"]

            await collection.update_one({"_id": patient_mongo.id}, {"$unset": {"revision_id": ""}})

        # Return the domain entity
        return await self._mongo_to_domain(patient_mongo)

    async def find_by_id(self, patient_id: PatientId, doctor_id: str) -> Optional[Patient]:
        """Find a patient by ID."""
        patient_mongo = await PatientMongo.find_one({"patient_id": patient_id.value, "doctor_id": doctor_id})

        if not patient_mongo:
            return None

        return await self._mongo_to_domain(patient_mongo)

    async def find_by_name_and_mobile(self, name: str, mobile: str, doctor_id: str) -> Optional[Patient]:
        """Find a patient by name and mobile number using optimized indexes."""
        patient_mongo = await PatientMongo.find_one({"name": name, "mobile": mobile, "doctor_id": doctor_id})

        if not patient_mongo:
            return None

        return await self._mongo_to_domain(patient_mongo)

    async def exists_by_id(self, patient_id: PatientId, doctor_id: str) -> bool:
        """Check if a patient exists by ID."""
        count = await PatientMongo.find({"patient_id": patient_id.value, "doctor_id": doctor_id}).count()

        return count > 0

    async def find_all(self, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Patient]:
        """Find all patients with pagination."""
        patients_mongo = await PatientMongo.find({"doctor_id": doctor_id}).skip(offset).limit(limit).to_list()

        result = []
        for patient_mongo in patients_mongo:
            result.append(await self._mongo_to_domain(patient_mongo))

        return result

    async def find_by_mobile(self, mobile: str, doctor_id: str) -> List[Patient]:
        """Find all patients with the same mobile number (family members)."""
        patients_mongo = await PatientMongo.find({"mobile": mobile, "doctor_id": doctor_id}).to_list()

        result = []
        for patient_mongo in patients_mongo:
            result.append(await self._mongo_to_domain(patient_mongo))

        return result

    async def delete(self, patient_id: PatientId, doctor_id: str) -> bool:
        """Delete a patient by ID."""
        result = await PatientMongo.find_one({"patient_id": patient_id.value, "doctor_id": doctor_id}).delete()

        return result is not None

    async def cleanup_revision_ids(self) -> int:
        """Remove revision_id from all patient documents."""
        result = await PatientMongo.update_many({}, {"$unset": {"revision_id": ""}})
        return result.modified_count

    async def _domain_to_mongo(self, patient: Patient) -> PatientMongo:
        """Convert domain entity to MongoDB model."""
        # Check if patient already exists
        existing_patient = await PatientMongo.find_one(
            {"patient_id": patient.patient_id.value, "doctor_id": patient.doctor_id}
        )

        if existing_patient:
            # Update existing patient
            existing_patient.name = patient.name
            existing_patient.mobile = patient.mobile
            existing_patient.age = patient.age
            existing_patient.gender = patient.gender
            # recently_travelled removed - now stored on Visit
            existing_patient.language = patient.language
            existing_patient.updated_at = datetime.utcnow()
            return existing_patient
        else:
            # Create new patient
            return PatientMongo(
                patient_id=patient.patient_id.value,
                name=patient.name,
                mobile=patient.mobile,
                age=patient.age,
                gender=patient.gender,
                # recently_travelled removed - now stored on Visit
                language=patient.language,
                doctor_id=patient.doctor_id,
                created_at=patient.created_at,
                updated_at=patient.updated_at,
            )

    async def _mongo_to_domain(self, patient_mongo: PatientMongo) -> Patient:
        """Convert MongoDB model to domain entity."""
        return Patient(
            patient_id=PatientId(patient_mongo.patient_id),
            doctor_id=getattr(patient_mongo, "doctor_id", ""),
            name=patient_mongo.name,
            mobile=patient_mongo.mobile,
            age=patient_mongo.age,
            gender=getattr(patient_mongo, "gender", None),
            # recently_travelled removed - now stored on Visit
            language=getattr(patient_mongo, "language", "en"),
            created_at=patient_mongo.created_at,
            updated_at=patient_mongo.updated_at,
        )
