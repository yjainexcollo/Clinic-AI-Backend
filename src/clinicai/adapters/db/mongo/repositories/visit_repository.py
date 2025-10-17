"""
MongoDB implementation of VisitRepository.
"""

from datetime import datetime
from typing import List, Optional

from clinicai.application.ports.repositories.visit_repo import VisitRepository
from clinicai.domain.entities.visit import IntakeSession, QuestionAnswer, Visit, TranscriptionSession, SoapNote
from clinicai.domain.value_objects.question_id import QuestionId
from clinicai.domain.value_objects.visit_id import VisitId
from clinicai.domain.enums.workflow import VisitWorkflowType

from ..models.patient_m import (
    IntakeSessionMongo,
    VisitMongo,
    QuestionAnswerMongo,
    TranscriptionSessionMongo,
    SoapNoteMongo,
)


class MongoVisitRepository(VisitRepository):
    """MongoDB implementation of VisitRepository."""

    async def save(self, visit: Visit) -> Visit:
        """Save a visit to MongoDB."""
        import logging
        logger = logging.getLogger("clinicai")
        
        logger.info(f"Saving visit {visit.visit_id.value} to database")
        logger.info(f"Visit status: {visit.status}")
        if visit.transcription_session:
            logger.info(f"Transcription session status: {visit.transcription_session.transcription_status}")
            logger.info(f"Transcript length: {len(visit.transcription_session.transcript) if visit.transcription_session.transcript else 0}")
            logger.info(f"Structured dialogue turns: {len(visit.transcription_session.structured_dialogue) if visit.transcription_session.structured_dialogue else 0}")
        
        # Convert domain entity to MongoDB model
        visit_mongo = await self._domain_to_mongo(visit)

        # Save to database
        await visit_mongo.save()
        logger.info(f"Visit {visit.visit_id.value} saved to database with ID: {visit_mongo.id}")

        # Remove revision_id from the document after saving using raw MongoDB operations
        if visit_mongo.id:
            from motor.motor_asyncio import AsyncIOMotorClient
            from clinicai.core.config import get_settings
            
            settings = get_settings()
            client = AsyncIOMotorClient(settings.database.uri)
            db = client[settings.database.db_name]
            collection = db["visits"]
            
            await collection.update_one(
                {"_id": visit_mongo.id},
                {"$unset": {"revision_id": ""}}
            )
            logger.info(f"Removed revision_id from visit {visit.visit_id.value}")

        # Return the domain entity
        return await self._mongo_to_domain(visit_mongo)

    async def find_by_id(self, visit_id: VisitId) -> Optional[Visit]:
        """Find a visit by ID."""
        visit_mongo = await VisitMongo.find_one(
            VisitMongo.visit_id == visit_id.value
        )

        if not visit_mongo:
            return None

        return await self._mongo_to_domain(visit_mongo)

    async def find_by_patient_id(self, patient_id: str) -> List[Visit]:
        """Find all visits for a specific patient."""
        visits_mongo = await VisitMongo.find(
            VisitMongo.patient_id == patient_id
        ).sort([("created_at", -1)]).to_list()

        return [
            await self._mongo_to_domain(visit_mongo)
            for visit_mongo in visits_mongo
        ]

    async def find_by_patient_and_visit_id(self, patient_id: str, visit_id: VisitId) -> Optional[Visit]:
        """Find a specific visit for a patient."""
        visit_mongo = await VisitMongo.find_one(
            VisitMongo.patient_id == patient_id,
            VisitMongo.visit_id == visit_id.value
        )

        if not visit_mongo:
            return None

        return await self._mongo_to_domain(visit_mongo)

    async def find_latest_by_patient_id(self, patient_id: str) -> Optional[Visit]:
        """Find the latest visit for a specific patient."""
        visit_mongo = await VisitMongo.find(
            VisitMongo.patient_id == patient_id
        ).sort([("created_at", -1)]).first()

        if not visit_mongo:
            return None

        return await self._mongo_to_domain(visit_mongo)

    async def exists_by_id(self, visit_id: VisitId) -> bool:
        """Check if a visit exists by ID."""
        count = await VisitMongo.find(
            VisitMongo.visit_id == visit_id.value
        ).count()

        return count > 0

    async def delete(self, visit_id: VisitId) -> bool:
        """Delete a visit by ID."""
        result = await VisitMongo.find_one(
            VisitMongo.visit_id == visit_id.value
        ).delete()

        return result is not None

    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find all visits with pagination."""
        visits_mongo = await VisitMongo.find().skip(offset).limit(limit).sort([("created_at", -1)]).to_list()

        return [
            await self._mongo_to_domain(visit_mongo)
            for visit_mongo in visits_mongo
        ]

    async def find_by_status(self, status: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find visits by status with pagination."""
        visits_mongo = await VisitMongo.find(
            VisitMongo.status == status
        ).skip(offset).limit(limit).sort([("created_at", -1)]).to_list()

        return [
            await self._mongo_to_domain(visit_mongo)
            for visit_mongo in visits_mongo
        ]

    async def count_by_patient_id(self, patient_id: str) -> int:
        """Count total visits for a patient."""
        return await VisitMongo.find(
            VisitMongo.patient_id == patient_id
        ).count()

    async def find_by_workflow_type(self, workflow_type: VisitWorkflowType, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find visits by workflow type with pagination."""
        visits_mongo = await VisitMongo.find(
            VisitMongo.workflow_type == workflow_type.value
        ).skip(offset).limit(limit).sort([("created_at", -1)]).to_list()

        return [
            await self._mongo_to_domain(visit_mongo)
            for visit_mongo in visits_mongo
        ]

    async def find_walk_in_visits(self, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find walk-in visits with pagination."""
        return await self.find_by_workflow_type(VisitWorkflowType.WALK_IN, limit, offset)

    async def find_scheduled_visits(self, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find scheduled visits with pagination."""
        return await self.find_by_workflow_type(VisitWorkflowType.SCHEDULED, limit, offset)

    async def _domain_to_mongo(self, visit: Visit) -> VisitMongo:
        """Convert domain entity to MongoDB model."""
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

        # Check if visit already exists
        existing_visit = await VisitMongo.find_one(
            VisitMongo.visit_id == visit.visit_id.value
        )

        if existing_visit:
            # Update existing visit
            existing_visit.patient_id = visit.patient_id
            existing_visit.status = visit.status
            existing_visit.updated_at = datetime.utcnow()
            existing_visit.intake_session = intake_session_mongo
            existing_visit.pre_visit_summary = visit.pre_visit_summary
            existing_visit.transcription_session = transcription_session_mongo
            existing_visit.soap_note = soap_note_mongo
            existing_visit.vitals = visit.vitals
            existing_visit.post_visit_summary = visit.post_visit_summary
            return existing_visit
        else:
            # Create new visit
            return VisitMongo(
                visit_id=visit.visit_id.value,
                patient_id=visit.patient_id,
                workflow_type=visit.workflow_type.value,
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

    async def _mongo_to_domain(self, visit_mongo: VisitMongo) -> Visit:
        """Convert MongoDB model to domain entity."""
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
            workflow_type=VisitWorkflowType(visit_mongo.workflow_type),
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
        
        return visit
