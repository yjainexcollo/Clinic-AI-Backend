"""
MongoDB implementation of PatientRepository.
"""

from datetime import datetime
from typing import List, Optional

from clinicai.application.ports.repositories.patient_repo import PatientRepository
from clinicai.domain.entities.patient import Patient
from clinicai.domain.entities.visit import IntakeSession, QuestionAnswer, Visit, TranscriptionSession, SoapNote
from clinicai.domain.value_objects.patient_id import PatientId
from clinicai.domain.value_objects.question_id import QuestionId
from clinicai.domain.value_objects.visit_id import VisitId

from ..models.patient_m import (
    IntakeSessionMongo,
    PatientMongo,
    QuestionAnswerMongo,
    VisitMongo,
    TranscriptionSessionMongo,
    SoapNoteMongo,
)


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
            
            await collection.update_one(
                {"_id": patient_mongo.id},
                {"$unset": {"revision_id": "", "visits.$[].revision_id": ""}}
            )

        # Return the domain entity
        return await self._mongo_to_domain(patient_mongo)

    async def find_by_id(self, patient_id: PatientId) -> Optional[Patient]:
        """Find a patient by ID."""
        patient_mongo = await PatientMongo.find_one(
            PatientMongo.patient_id == patient_id.value
        )

        if not patient_mongo:
            return None

        return await self._mongo_to_domain(patient_mongo)

    async def find_by_name_and_mobile(
        self, name: str, mobile: str
    ) -> Optional[Patient]:
        """Find a patient by name and mobile number."""
        patient_mongo = await PatientMongo.find_one(
            PatientMongo.name == name, PatientMongo.mobile == mobile
        )

        if not patient_mongo:
            return None

        return await self._mongo_to_domain(patient_mongo)

    async def exists_by_id(self, patient_id: PatientId) -> bool:
        """Check if a patient exists by ID."""
        count = await PatientMongo.find(
            PatientMongo.patient_id == patient_id.value
        ).count()

        return count > 0

    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Patient]:
        """Find all patients with pagination."""
        patients_mongo = await PatientMongo.find().skip(offset).limit(limit).to_list()

        return [
            await self._mongo_to_domain(patient_mongo)
            for patient_mongo in patients_mongo
        ]

    async def find_by_mobile(self, mobile: str) -> List[Patient]:
        """Find all patients with the same mobile number (family members)."""
        patients_mongo = await PatientMongo.find(
            PatientMongo.mobile == mobile
        ).to_list()

        return [
            await self._mongo_to_domain(patient_mongo)
            for patient_mongo in patients_mongo
        ]

    async def delete(self, patient_id: PatientId) -> bool:
        """Delete a patient by ID."""
        result = await PatientMongo.find_one(
            PatientMongo.patient_id == patient_id.value
        ).delete()

        return result is not None

    async def cleanup_revision_ids(self) -> int:
        """Remove revision_id from all patient documents and nested visits."""
        result = await PatientMongo.update_many(
            {},
            {
                "$unset": {
                    "revision_id": "",
                    "visits.$[].revision_id": ""
                }
            }
        )
        return result.modified_count

    async def _domain_to_mongo(self, patient: Patient) -> PatientMongo:
        """Convert domain entity to MongoDB model."""
        # Convert visits
        visits_mongo = []
        for visit in patient.visits:
            # Convert intake session
            intake_session_mongo = None
            if visit.intake_session:
                # Convert question answers
                questions_asked_mongo = []
                for qa in visit.intake_session.questions_asked:
                    qa_mongo = QuestionAnswerMongo(
                        question_id=qa.question_id.value,
                        question=qa.question,
                        answer=qa.answer,
                        timestamp=qa.timestamp,
                        question_number=qa.question_number,
                    )
                    questions_asked_mongo.append(qa_mongo)

                intake_session_mongo = IntakeSessionMongo(
                    questions_asked=questions_asked_mongo,
                    current_question_count=visit.intake_session.current_question_count,
                    max_questions=visit.intake_session.max_questions,
                    status=visit.intake_session.status,
                    started_at=visit.intake_session.started_at,
                    completed_at=visit.intake_session.completed_at,
                    pending_question=visit.intake_session.pending_question,
                )

            # Convert transcription session
            transcription_session_mongo = None
            if visit.transcription_session:
                transcription_session_mongo = TranscriptionSessionMongo(
                    audio_file_path=visit.transcription_session.audio_file_path,
                    transcript=visit.transcription_session.transcript,
                    transcription_status=visit.transcription_session.transcription_status,
                    started_at=visit.transcription_session.started_at,
                    completed_at=visit.transcription_session.completed_at,
                    error_message=visit.transcription_session.error_message,
                    audio_duration_seconds=visit.transcription_session.audio_duration_seconds,
                    word_count=visit.transcription_session.word_count,
                    structured_dialogue=getattr(visit.transcription_session, "structured_dialogue", None),
                )

            # Convert SOAP note
            soap_note_mongo = None
            if visit.soap_note:
                soap_note_mongo = SoapNoteMongo(
                    subjective=visit.soap_note.subjective,
                    objective=visit.soap_note.objective,
                    assessment=visit.soap_note.assessment,
                    plan=visit.soap_note.plan,
                    highlights=visit.soap_note.highlights,
                    red_flags=visit.soap_note.red_flags,
                    generated_at=visit.soap_note.generated_at,
                    model_info=visit.soap_note.model_info,
                    confidence_score=visit.soap_note.confidence_score,
                )

            visit_mongo = VisitMongo(
                visit_id=visit.visit_id.value,
                patient_id=visit.patient_id,
                status=visit.status,
                created_at=visit.created_at,
                updated_at=visit.updated_at,
                intake_session=intake_session_mongo,
                pre_visit_summary=visit.pre_visit_summary,
                transcription_session=transcription_session_mongo,
                soap_note=soap_note_mongo,
                vitals=visit.vitals,
                post_visit_summary=visit.post_visit_summary,
            )
            visits_mongo.append(visit_mongo)

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
            
            # Merge visits instead of replacing them
            # Create a map of existing visits by visit_id for efficient lookup
            existing_visits_map = {visit.visit_id: visit for visit in existing_patient.visits}
            
            # Add or update visits from the domain entity
            for visit_mongo in visits_mongo:
                if visit_mongo.visit_id in existing_visits_map:
                    # Update existing visit
                    existing_visit = existing_visits_map[visit_mongo.visit_id]
                    existing_visit.status = visit_mongo.status
                    existing_visit.updated_at = visit_mongo.updated_at
                    existing_visit.intake_session = visit_mongo.intake_session
                    existing_visit.pre_visit_summary = visit_mongo.pre_visit_summary
                    existing_visit.transcription_session = visit_mongo.transcription_session
                    existing_visit.soap_note = visit_mongo.soap_note
                    existing_visit.vitals = visit_mongo.vitals
                    existing_visit.post_visit_summary = visit_mongo.post_visit_summary
                else:
                    # Add new visit
                    existing_patient.visits.append(visit_mongo)
            
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
                visits=visits_mongo,
                created_at=patient.created_at,
                updated_at=patient.updated_at,
            )

    async def _mongo_to_domain(self, patient_mongo: PatientMongo) -> Patient:
        """Convert MongoDB model to domain entity."""
        # Convert visits
        visits = []
        for visit_mongo in patient_mongo.visits:
            # Convert intake session
            intake_session = None
            if visit_mongo.intake_session:
                # Convert question answers
                questions_asked = []
                for qa_mongo in visit_mongo.intake_session.questions_asked:
                    qa = QuestionAnswer(
                        question_id=QuestionId(qa_mongo.question_id),
                        question=qa_mongo.question,
                        answer=qa_mongo.answer,
                        timestamp=qa_mongo.timestamp,
                        question_number=qa_mongo.question_number,
                    )
                    questions_asked.append(qa)

                intake_session = IntakeSession(
                    symptom="",  # No symptom field in database, use empty string
                    questions_asked=questions_asked,
                    current_question_count=visit_mongo.intake_session.current_question_count,
                    max_questions=visit_mongo.intake_session.max_questions,
                    status=visit_mongo.intake_session.status,
                    started_at=visit_mongo.intake_session.started_at,
                    completed_at=visit_mongo.intake_session.completed_at,
                )
                intake_session.pending_question = getattr(visit_mongo.intake_session, "pending_question", None)

            # Convert transcription session
            transcription_session = None
            if visit_mongo.transcription_session:
                transcription_session = TranscriptionSession(
                    audio_file_path=visit_mongo.transcription_session.audio_file_path,
                    transcript=visit_mongo.transcription_session.transcript,
                    transcription_status=visit_mongo.transcription_session.transcription_status,
                    started_at=visit_mongo.transcription_session.started_at,
                    completed_at=visit_mongo.transcription_session.completed_at,
                    error_message=visit_mongo.transcription_session.error_message,
                    audio_duration_seconds=visit_mongo.transcription_session.audio_duration_seconds,
                    word_count=visit_mongo.transcription_session.word_count,
                    structured_dialogue=getattr(visit_mongo.transcription_session, "structured_dialogue", None),
                )

            # Convert SOAP note
            soap_note = None
            if visit_mongo.soap_note:
                # Handle objective field - it might be a string from old data
                objective = visit_mongo.soap_note.objective
                if isinstance(objective, str):
                    try:
                        # Try to parse as JSON if it looks like a dict string
                        if objective.strip().startswith('{') and objective.strip().endswith('}'):
                            import json
                            objective = json.loads(objective)
                        else:
                            # If it's not JSON, create a basic structure
                            objective = {
                                "vital_signs": {},
                                "physical_exam": {"general_appearance": objective or "Not discussed"}
                            }
                    except:
                        # If parsing fails, create a basic structure
                        objective = {
                            "vital_signs": {},
                            "physical_exam": {"general_appearance": objective or "Not discussed"}
                        }
                elif not isinstance(objective, dict):
                    # If it's neither string nor dict, create a basic structure
                    objective = {
                        "vital_signs": {},
                        "physical_exam": {"general_appearance": "Not discussed"}
                    }
                
                soap_note = SoapNote(
                    subjective=visit_mongo.soap_note.subjective,
                    objective=objective,  # Now guaranteed to be a dict
                    assessment=visit_mongo.soap_note.assessment,
                    plan=visit_mongo.soap_note.plan,
                    highlights=visit_mongo.soap_note.highlights,
                    red_flags=visit_mongo.soap_note.red_flags,
                    generated_at=visit_mongo.soap_note.generated_at,
                    model_info=visit_mongo.soap_note.model_info,
                    confidence_score=visit_mongo.soap_note.confidence_score,
                )

            visit = Visit(
                visit_id=VisitId(visit_mongo.visit_id),
                patient_id=visit_mongo.patient_id,
                symptom=intake_session.symptom if intake_session else "",  # Use intake session symptom or empty string
                status=visit_mongo.status,
                created_at=visit_mongo.created_at,
                updated_at=visit_mongo.updated_at,
                intake_session=intake_session,
                pre_visit_summary=visit_mongo.pre_visit_summary,
                transcription_session=transcription_session,
                soap_note=soap_note,
                post_visit_summary=getattr(visit_mongo, "post_visit_summary", None),
            )
            # Attach vitals if present
            visit.vitals = getattr(visit_mongo, "vitals", None)
            visits.append(visit)

        return Patient(
            patient_id=PatientId(patient_mongo.patient_id),
            name=patient_mongo.name,
            mobile=patient_mongo.mobile,
            age=patient_mongo.age,
            gender=getattr(patient_mongo, "gender", None),
            recently_travelled=getattr(patient_mongo, "recently_travelled", False),
            language=getattr(patient_mongo, "language", "en"),
            visits=visits,
            created_at=patient_mongo.created_at,
            updated_at=patient_mongo.updated_at,
        )
