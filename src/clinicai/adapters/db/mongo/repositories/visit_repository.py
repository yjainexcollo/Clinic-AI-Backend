"""
MongoDB implementation of VisitRepository.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from clinicai.application.ports.repositories.visit_repo import VisitRepository
from clinicai.domain.entities.visit import (
    IntakeSession,
    QuestionAnswer,
    SoapNote,
    TranscriptionSession,
    Visit,
)
from clinicai.domain.enums.workflow import VisitWorkflowType
from clinicai.domain.value_objects.question_id import QuestionId
from clinicai.domain.value_objects.visit_id import VisitId

from ..models.patient_m import (
    IntakeSessionMongo,
    PatientMongo,
    QuestionAnswerMongo,
    SoapNoteMongo,
    TranscriptionSessionMongo,
    VisitMongo,
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
            logger.info(
                f"Transcript length: {len(visit.transcription_session.transcript) if visit.transcription_session.transcript else 0}"
            )
            logger.info(
                f"Structured dialogue turns: {len(visit.transcription_session.structured_dialogue) if visit.transcription_session.structured_dialogue else 0}"
            )

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

            await collection.update_one({"_id": visit_mongo.id}, {"$unset": {"revision_id": ""}})
            logger.info(f"Removed revision_id from visit {visit.visit_id.value}")

        # Return the domain entity
        return await self._mongo_to_domain(visit_mongo)

    async def find_by_id(self, visit_id: VisitId, doctor_id: str) -> Optional[Visit]:
        """Find a visit by ID."""
        visit_mongo = await VisitMongo.find_one(
            VisitMongo.visit_id == visit_id.value,
            VisitMongo.doctor_id == doctor_id,
        )

        if not visit_mongo:
            return None

        return await self._mongo_to_domain(visit_mongo)

    async def find_by_patient_id(self, patient_id: str, doctor_id: str) -> List[Visit]:
        """Find all visits for a specific patient."""
        visits_mongo = (
            await VisitMongo.find(
                VisitMongo.patient_id == patient_id,
                VisitMongo.doctor_id == doctor_id,
            )
            .sort([("created_at", -1)])
            .to_list()
        )

        return [await self._mongo_to_domain(visit_mongo) for visit_mongo in visits_mongo]

    async def find_by_patient_and_visit_id(self, patient_id: str, visit_id: VisitId, doctor_id: str) -> Optional[Visit]:
        """Find a specific visit for a patient."""
        visit_mongo = await VisitMongo.find_one(
            VisitMongo.patient_id == patient_id,
            VisitMongo.visit_id == visit_id.value,
            VisitMongo.doctor_id == doctor_id,
        )

        if not visit_mongo:
            return None

        return await self._mongo_to_domain(visit_mongo)

    async def find_latest_by_patient_id(self, patient_id: str, doctor_id: str) -> Optional[Visit]:
        """Find the latest visit for a specific patient."""
        visit_mongo = (
            await VisitMongo.find(
                VisitMongo.patient_id == patient_id,
                VisitMongo.doctor_id == doctor_id,
            )
            .sort([("created_at", -1)])
            .first()
        )

        if not visit_mongo:
            return None

        return await self._mongo_to_domain(visit_mongo)

    async def exists_by_id(self, visit_id: VisitId, doctor_id: str) -> bool:
        """Check if a visit exists by ID."""
        count = await VisitMongo.find(
            VisitMongo.visit_id == visit_id.value,
            VisitMongo.doctor_id == doctor_id,
        ).count()

        return count > 0

    async def delete(self, visit_id: VisitId, doctor_id: str) -> bool:
        """Delete a visit by ID."""
        result = await VisitMongo.find_one(
            VisitMongo.visit_id == visit_id.value,
            VisitMongo.doctor_id == doctor_id,
        ).delete()

        return result is not None

    async def find_all(self, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find all visits with pagination."""
        visits_mongo = (
            await VisitMongo.find(VisitMongo.doctor_id == doctor_id)
            .skip(offset)
            .limit(limit)
            .sort([("created_at", -1)])
            .to_list()
        )

        return [await self._mongo_to_domain(visit_mongo) for visit_mongo in visits_mongo]

    async def find_by_status(self, status: str, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find visits by status with pagination."""
        visits_mongo = (
            await VisitMongo.find(
                VisitMongo.status == status,
                VisitMongo.doctor_id == doctor_id,
            )
            .skip(offset)
            .limit(limit)
            .sort([("created_at", -1)])
            .to_list()
        )

        return [await self._mongo_to_domain(visit_mongo) for visit_mongo in visits_mongo]

    async def count_by_patient_id(self, patient_id: str, doctor_id: str) -> int:
        """Count total visits for a patient."""
        return await VisitMongo.find(
            VisitMongo.patient_id == patient_id,
            VisitMongo.doctor_id == doctor_id,
        ).count()

    async def find_by_workflow_type(
        self,
        workflow_type: VisitWorkflowType,
        doctor_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Visit]:
        """Find visits by workflow type with pagination."""
        visits_mongo = (
            await VisitMongo.find(
                VisitMongo.workflow_type == workflow_type.value,
                VisitMongo.doctor_id == doctor_id,
            )
            .skip(offset)
            .limit(limit)
            .sort([("created_at", -1)])
            .to_list()
        )

        return [await self._mongo_to_domain(visit_mongo) for visit_mongo in visits_mongo]

    async def find_walk_in_visits(self, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find walk-in visits with pagination."""
        return await self.find_by_workflow_type(VisitWorkflowType.WALK_IN, doctor_id, limit, offset)

    async def find_scheduled_visits(self, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Visit]:
        """Find scheduled visits with pagination."""
        return await self.find_by_workflow_type(VisitWorkflowType.SCHEDULED, doctor_id, limit, offset)

    async def find_patients_with_visits(
        self,
        doctor_id: str,
        workflow_type: Optional[VisitWorkflowType] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """
        Find patients with aggregated visit information using MongoDB aggregation.
        """
        from motor.motor_asyncio import AsyncIOMotorClient

        from clinicai.core.config import get_settings

        settings = get_settings()
        client = AsyncIOMotorClient(settings.database.uri)
        db = client[settings.database.db_name]
        visits_collection = db["visits"]

        # Build match stage for workflow_type filter
        match_stage = {"doctor_id": doctor_id}
        if workflow_type:
            match_stage["workflow_type"] = workflow_type.value

        # Aggregation pipeline
        pipeline = [
            # Match visits by workflow_type if specified
            {"$match": match_stage},
            # Sort visits by created_at to get latest first
            {"$sort": {"created_at": -1}},
            # Group by patient_id to aggregate visit data
            {
                "$group": {
                    "_id": "$patient_id",
                    "latest_visit": {"$first": "$$ROOT"},
                    "total_visits": {"$sum": 1},
                    "scheduled_visits_count": {"$sum": {"$cond": [{"$eq": ["$workflow_type", "scheduled"]}, 1, 0]}},
                    "walk_in_visits_count": {"$sum": {"$cond": [{"$eq": ["$workflow_type", "walk_in"]}, 1, 0]}},
                }
            },
            # Lookup patient information
            {
                "$lookup": {
                    "from": "patients",
                    "localField": "_id",
                    "foreignField": "patient_id",
                    "as": "patient",
                }
            },
            # Unwind patient array (should be single patient)
            {"$unwind": {"path": "$patient", "preserveNullAndEmptyArrays": True}},
            # Project final structure
            {
                "$project": {
                    "patient_id": "$_id",
                    "name": "$patient.name",
                    "mobile": "$patient.mobile",
                    "age": "$patient.age",
                    "gender": "$patient.gender",
                    "latest_visit": {
                        "visit_id": "$latest_visit.visit_id",
                        "workflow_type": "$latest_visit.workflow_type",
                        "status": "$latest_visit.status",
                        "created_at": "$latest_visit.created_at",
                    },
                    "total_visits": 1,
                    "scheduled_visits_count": 1,
                    "walk_in_visits_count": 1,
                }
            },
        ]

        # Add sorting
        sort_direction = -1 if sort_order == "desc" else 1
        if sort_by == "name":
            pipeline.append({"$sort": {"name": sort_direction}})
        elif sort_by == "created_at":
            pipeline.append({"$sort": {"latest_visit.created_at": sort_direction}})
        else:
            pipeline.append({"$sort": {"latest_visit.created_at": sort_direction}})

        # Add pagination
        pipeline.extend([{"$skip": offset}, {"$limit": limit}])

        # Execute aggregation
        cursor = visits_collection.aggregate(pipeline)
        results = await cursor.to_list(length=limit)

        # Format results
        formatted_results = []
        for result in results:
            latest_visit = result.get("latest_visit")
            # MongoDB aggregation returns datetime objects directly, but ensure it's a dict
            if latest_visit and not isinstance(latest_visit, dict):
                latest_visit = None

            formatted_results.append(
                {
                    "patient_id": result.get("patient_id", ""),
                    "name": result.get("name", "Unknown"),
                    "mobile": result.get("mobile", ""),
                    "age": result.get("age", 0),
                    "gender": result.get("gender"),
                    "latest_visit": latest_visit,
                    "total_visits": result.get("total_visits", 0),
                    "scheduled_visits_count": result.get("scheduled_visits_count", 0),
                    "walk_in_visits_count": result.get("walk_in_visits_count", 0),
                }
            )

        return formatted_results

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
                travel_questions_count=getattr(visit.intake_session, "travel_questions_count", 0),
                asked_categories=getattr(visit.intake_session, "asked_categories", []),
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
                worker_id=getattr(visit.transcription_session, "worker_id", None),
                audio_duration_seconds=visit.transcription_session.audio_duration_seconds,
                word_count=visit.transcription_session.word_count,
                structured_dialogue=getattr(visit.transcription_session, "structured_dialogue", None),
                transcription_id=getattr(visit.transcription_session, "transcription_id", None),
                last_poll_status=getattr(visit.transcription_session, "last_poll_status", None),
                last_poll_at=getattr(visit.transcription_session, "last_poll_at", None),
                enqueued_at=getattr(visit.transcription_session, "enqueued_at", None),
                dequeued_at=getattr(visit.transcription_session, "dequeued_at", None),
                azure_job_created_at=getattr(visit.transcription_session, "azure_job_created_at", None),
                first_poll_at=getattr(visit.transcription_session, "first_poll_at", None),
                results_downloaded_at=getattr(visit.transcription_session, "results_downloaded_at", None),
                db_saved_at=getattr(visit.transcription_session, "db_saved_at", None),
                normalized_audio=getattr(visit.transcription_session, "normalized_audio", None),
                original_content_type=getattr(visit.transcription_session, "original_content_type", None),
                normalized_format=getattr(visit.transcription_session, "normalized_format", None),
                file_content_type=getattr(visit.transcription_session, "file_content_type", None),
                enqueue_state=getattr(visit.transcription_session, "enqueue_state", None),
                enqueue_attempts=getattr(visit.transcription_session, "enqueue_attempts", None),
                enqueue_last_error=getattr(visit.transcription_session, "enqueue_last_error", None),
                enqueue_requested_at=getattr(visit.transcription_session, "enqueue_requested_at", None),
                enqueue_failed_at=getattr(visit.transcription_session, "enqueue_failed_at", None),
                queue_message_id=getattr(visit.transcription_session, "queue_message_id", None),
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
            VisitMongo.visit_id == visit.visit_id.value,
            VisitMongo.doctor_id == visit.doctor_id,
        )

        if existing_visit:
            # Update existing visit
            existing_visit.patient_id = visit.patient_id
            existing_visit.doctor_id = visit.doctor_id
            existing_visit.status = visit.status
            existing_visit.updated_at = datetime.utcnow()
            existing_visit.recently_travelled = getattr(visit, "recently_travelled", False)
            existing_visit.symptom = visit.symptom
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
                doctor_id=visit.doctor_id,
                symptom=visit.symptom,
                workflow_type=visit.workflow_type.value,
                status=visit.status,
                created_at=visit.created_at,
                updated_at=visit.updated_at,
                recently_travelled=getattr(visit, "recently_travelled", False),
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
                symptom=getattr(visit_mongo, "symptom", "") or "",
                questions_asked=questions_asked,
                current_question_count=visit_mongo.intake_session.current_question_count,
                max_questions=visit_mongo.intake_session.max_questions,
                status=visit_mongo.intake_session.status,
                started_at=visit_mongo.intake_session.started_at,
                completed_at=visit_mongo.intake_session.completed_at,
                travel_questions_count=getattr(visit_mongo.intake_session, "travel_questions_count", 0),
                asked_categories=getattr(visit_mongo.intake_session, "asked_categories", []),
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
                worker_id=getattr(visit_mongo.transcription_session, "worker_id", None),
                audio_duration_seconds=visit_mongo.transcription_session.audio_duration_seconds,
                word_count=visit_mongo.transcription_session.word_count,
                structured_dialogue=getattr(visit_mongo.transcription_session, "structured_dialogue", None),
                transcription_id=getattr(visit_mongo.transcription_session, "transcription_id", None),
                last_poll_status=getattr(visit_mongo.transcription_session, "last_poll_status", None),
                last_poll_at=getattr(visit_mongo.transcription_session, "last_poll_at", None),
                enqueued_at=getattr(visit_mongo.transcription_session, "enqueued_at", None),
                dequeued_at=getattr(visit_mongo.transcription_session, "dequeued_at", None),
                azure_job_created_at=getattr(visit_mongo.transcription_session, "azure_job_created_at", None),
                first_poll_at=getattr(visit_mongo.transcription_session, "first_poll_at", None),
                results_downloaded_at=getattr(visit_mongo.transcription_session, "results_downloaded_at", None),
                db_saved_at=getattr(visit_mongo.transcription_session, "db_saved_at", None),
                normalized_audio=getattr(visit_mongo.transcription_session, "normalized_audio", None),
                original_content_type=getattr(visit_mongo.transcription_session, "original_content_type", None),
                normalized_format=getattr(visit_mongo.transcription_session, "normalized_format", None),
                file_content_type=getattr(visit_mongo.transcription_session, "file_content_type", None),
                enqueue_state=getattr(visit_mongo.transcription_session, "enqueue_state", None),
                enqueue_attempts=getattr(visit_mongo.transcription_session, "enqueue_attempts", None),
                enqueue_last_error=getattr(visit_mongo.transcription_session, "enqueue_last_error", None),
                enqueue_requested_at=getattr(visit_mongo.transcription_session, "enqueue_requested_at", None),
                enqueue_failed_at=getattr(visit_mongo.transcription_session, "enqueue_failed_at", None),
                queue_message_id=getattr(visit_mongo.transcription_session, "queue_message_id", None),
            )

        # Convert SOAP note
        soap_note = None
        if visit_mongo.soap_note:
            # Handle objective field - it might be a string from old data
            objective = visit_mongo.soap_note.objective
            if isinstance(objective, str):
                try:
                    # Try to parse as JSON if it looks like a dict string
                    if objective.strip().startswith("{") and objective.strip().endswith("}"):
                        import json

                        objective = json.loads(objective)
                    else:
                        # If it's not JSON, create a basic structure
                        objective = {
                            "vital_signs": {},
                            "physical_exam": {"general_appearance": objective or "Not discussed"},
                        }
                except:
                    # If parsing fails, create a basic structure
                    objective = {
                        "vital_signs": {},
                        "physical_exam": {"general_appearance": objective or "Not discussed"},
                    }
            elif not isinstance(objective, dict):
                # If it's neither string nor dict, create a basic structure
                objective = {
                    "vital_signs": {},
                    "physical_exam": {"general_appearance": "Not discussed"},
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

        visit_symptom = getattr(visit_mongo, "symptom", "") or ""

        visit = Visit(
            visit_id=VisitId(visit_mongo.visit_id),
            patient_id=visit_mongo.patient_id,
            doctor_id=getattr(visit_mongo, "doctor_id", ""),
            symptom=visit_symptom,
            workflow_type=VisitWorkflowType(visit_mongo.workflow_type),
            status=visit_mongo.status,
            created_at=visit_mongo.created_at,
            updated_at=visit_mongo.updated_at,
            recently_travelled=getattr(visit_mongo, "recently_travelled", False),
            intake_session=intake_session,
            pre_visit_summary=visit_mongo.pre_visit_summary,
            transcription_session=transcription_session,
            soap_note=soap_note,
            post_visit_summary=getattr(visit_mongo, "post_visit_summary", None),
        )
        # Attach vitals if present
        visit.vitals = getattr(visit_mongo, "vitals", None)
        if intake_session:
            intake_session.symptom = visit_symptom

        return visit

    async def try_mark_processing(
        self,
        patient_id: str,
        visit_id: VisitId,
        worker_id: str,
        stale_seconds: int,
        doctor_id: str,
    ) -> bool:
        """
        Atomically claim a transcription job by marking it as processing.

        Returns True only if the job was successfully claimed (modified_count == 1).
        This ensures only one worker processes each job.

        Conditions for claiming:
        - status == "queued"
        - OR status == "processing" but started_at is None
        - OR status == "processing" and started_at <= now - stale_seconds (stale)

        Updates on successful claim:
        - status = "processing"
        - started_at = now (UTC)
        - dequeued_at = now (UTC)
        - worker_id = worker_id
        - error_message = None
        """
        from datetime import datetime, timedelta

        from motor.motor_asyncio import AsyncIOMotorClient

        from clinicai.core.config import get_settings

        settings = get_settings()
        client = AsyncIOMotorClient(settings.database.uri)
        db = client[settings.database.db_name]
        collection = db["visits"]

        now = datetime.utcnow()
        stale_threshold = now - timedelta(seconds=stale_seconds)

        # Build query conditions for claiming
        # Condition 1: status == "queued"
        # Condition 2: status == "processing" and started_at is None
        # Condition 3: status == "processing" and started_at <= stale_threshold
        claim_conditions = {
            "patient_id": patient_id,
            "visit_id": visit_id.value,
            "doctor_id": doctor_id,
            "$or": [
                # Status is queued
                {"transcription_session.transcription_status": "queued"},
                # Status is processing but started_at is None (inconsistent state)
                {
                    "transcription_session.transcription_status": "processing",
                    "$or": [
                        {"transcription_session.started_at": None},
                        {"transcription_session.started_at": {"$exists": False}},
                    ],
                },
                # Status is processing but stale
                {
                    "transcription_session.transcription_status": "processing",
                    "transcription_session.started_at": {"$lte": stale_threshold},
                },
            ],
        }

        # Update operation - atomically claim the job
        update_operation = {
            "$set": {
                "transcription_session.transcription_status": "processing",
                "transcription_session.started_at": now,
                "transcription_session.dequeued_at": now,
                "transcription_session.worker_id": worker_id,
                "transcription_session.error_message": None,
                "updated_at": now,
            }
        }

        result = await collection.update_one(claim_conditions, update_operation)

        return result.modified_count == 1

    async def update_transcription_session_fields(
        self, patient_id: str, visit_id: VisitId, doctor_id: str, fields: Dict[str, Any]
    ) -> bool:
        """
        Atomically update specific fields in the transcription_session of a visit.

        Args:
            patient_id: Patient ID
            visit_id: Visit ID
            fields: Dictionary of field names and values to update.
                   Field names should be top-level transcription_session field names
                   (e.g., "transcription_id", "last_poll_status", "last_poll_at")

        Returns:
            True if the visit was found and updated (modified_count == 1), False otherwise
        """
        from datetime import datetime

        from motor.motor_asyncio import AsyncIOMotorClient

        from clinicai.core.config import get_settings

        settings = get_settings()
        client = AsyncIOMotorClient(settings.database.uri)
        db = client[settings.database.db_name]
        collection = db["visits"]

        # Build update operation using dot notation for nested transcription_session fields
        update_operation = {"$set": {"updated_at": datetime.utcnow()}}

        # Add transcription_session fields with dot notation
        for field_name, field_value in fields.items():
            update_operation["$set"][f"transcription_session.{field_name}"] = field_value

        # Update the visit document
        result = await collection.update_one(
            {
                "patient_id": patient_id,
                "visit_id": visit_id.value,
                "doctor_id": doctor_id,
            },
            update_operation,
        )

        return result.modified_count == 1
