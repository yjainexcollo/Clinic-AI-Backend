"""
MongoDB implementation of PatientRepository.
"""

from datetime import datetime
from typing import List, Optional

from clinicai.application.ports.repositories.patient_repo import PatientRepository
from clinicai.application.ports.repositories.visit_repo import VisitRepository
from clinicai.domain.entities.patient import Patient
from clinicai.domain.entities.visit import Visit
from clinicai.domain.value_objects.patient_id import PatientId
from clinicai.domain.value_objects.visit_id import VisitId

from ..models.patient_m import PatientMongo


class MongoPatientRepository(PatientRepository):
    """MongoDB implementation of PatientRepository."""

    def __init__(self, visit_repository: VisitRepository):
        """Initialize with visit repository dependency."""
        self.visit_repository = visit_repository

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
            
            await collection.update_one(
                {"_id": patient_mongo.id},
                {"$unset": {"revision_id": ""}}
            )

        # Save visits separately using visit repository
        for visit in patient.visits:
            await self.visit_repository.save(visit)

        # Return the domain entity with visits
        return await self._mongo_to_domain(patient_mongo, patient.visits)

    async def find_by_id(self, patient_id: PatientId) -> Optional[Patient]:
        """Find a patient by ID."""
        patient_mongo = await PatientMongo.find_one(
            PatientMongo.patient_id == patient_id.value
        )

        if not patient_mongo:
            return None

        # Get visits from visit repository
        visits = await self.visit_repository.find_by_patient_id(patient_id.value)
        
        return await self._mongo_to_domain(patient_mongo, visits)

    async def find_by_name_and_mobile(
        self, name: str, mobile: str
    ) -> Optional[Patient]:
        """Find a patient by name and mobile number."""
        patient_mongo = await PatientMongo.find_one(
            PatientMongo.name == name, PatientMongo.mobile == mobile
        )

        if not patient_mongo:
            return None

        # Get visits from visit repository
        visits = await self.visit_repository.find_by_patient_id(patient_mongo.patient_id)
        
        return await self._mongo_to_domain(patient_mongo, visits)

    async def exists_by_id(self, patient_id: PatientId) -> bool:
        """Check if a patient exists by ID."""
        count = await PatientMongo.find(
            PatientMongo.patient_id == patient_id.value
        ).count()

        return count > 0

    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Patient]:
        """Find all patients with pagination."""
        patients_mongo = await PatientMongo.find().skip(offset).limit(limit).to_list()

        result = []
        for patient_mongo in patients_mongo:
            # Get visits from visit repository
            visits = await self.visit_repository.find_by_patient_id(patient_mongo.patient_id)
            result.append(await self._mongo_to_domain(patient_mongo, visits))
        
        return result

    async def find_by_mobile(self, mobile: str) -> List[Patient]:
        """Find all patients with the same mobile number (family members)."""
        patients_mongo = await PatientMongo.find(
            PatientMongo.mobile == mobile
        ).to_list()

        result = []
        for patient_mongo in patients_mongo:
            # Get visits from visit repository
            visits = await self.visit_repository.find_by_patient_id(patient_mongo.patient_id)
            result.append(await self._mongo_to_domain(patient_mongo, visits))
        
        return result

    async def delete(self, patient_id: PatientId) -> bool:
        """Delete a patient by ID."""
        result = await PatientMongo.find_one(
            PatientMongo.patient_id == patient_id.value
        ).delete()

        return result is not None

    async def cleanup_revision_ids(self) -> int:
        """Remove revision_id from all patient documents."""
        result = await PatientMongo.update_many(
            {},
            {
                "$unset": {
                    "revision_id": ""
                }
            }
        )
        return result.modified_count

    async def _domain_to_mongo(self, patient: Patient) -> PatientMongo:
        """Convert domain entity to MongoDB model."""
        # Check if patient already exists
        existing_patient = await PatientMongo.find_one(
            PatientMongo.patient_id == patient.patient_id.value
        )

        if existing_patient:
            # Update existing patient
            existing_patient.name = patient.name
            existing_patient.mobile = patient.mobile
            existing_patient.age = patient.age
            existing_patient.gender = patient.gender
            existing_patient.recently_travelled = patient.recently_travelled
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
                recently_travelled=patient.recently_travelled,
                language=patient.language,
                created_at=patient.created_at,
                updated_at=patient.updated_at,
            )

    async def _mongo_to_domain(self, patient_mongo: PatientMongo, visits: List[Visit] = None) -> Patient:
        """Convert MongoDB model to domain entity."""
        return Patient(
            patient_id=PatientId(patient_mongo.patient_id),
            name=patient_mongo.name,
            mobile=patient_mongo.mobile,
            age=patient_mongo.age,
            gender=getattr(patient_mongo, "gender", None),
            recently_travelled=getattr(patient_mongo, "recently_travelled", False),
            language=getattr(patient_mongo, "language", "en"),
            visits=visits or [],
            created_at=patient_mongo.created_at,
            updated_at=patient_mongo.updated_at,
        )
