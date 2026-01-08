"""Transcribe audio use case for Step-03 functionality."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ...core.ai_client import AzureAIClient
from ...core.ai_factory import get_ai_client
from ...core.config import get_settings
from ...domain.enums.workflow import VisitWorkflowType
from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId
from ...domain.value_objects.visit_id import VisitId
from ..dto.patient_dto import AudioTranscriptionRequest, AudioTranscriptionResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.repositories.visit_repo import VisitRepository
from ..ports.services.transcription_service import TranscriptionService

# Module-level logger to avoid UnboundLocalError from function-level assignments
LOGGER = logging.getLogger("clinicai")


class TranscribeAudioUseCase:
    """Use case for transcribing audio files."""

    def __init__(
        self,
        patient_repository: PatientRepository,
        visit_repository: VisitRepository,
        transcription_service: TranscriptionService,
    ):
        self._patient_repository = patient_repository
        self._visit_repository = visit_repository
        self._transcription_service = transcription_service

        # Medical terms that should NOT be removed as PII (medications, medical conditions, etc.)
        # Use frozenset for faster O(1) lookups instead of O(n) set iteration
        self._medical_term_whitelist = frozenset(
            {
                # Common medications
                "metformin",
                "jardiance",
                "giordians",
                "lisinopril",
                "amlodipine",
                "lidocaine",
                "aspirin",
                "ibuprofen",
                "acetaminophen",
                "tylenol",
                "advil",
                "motrin",
                "naproxen",
                "atorvastatin",
                "simvastatin",
                "pravastatin",
                "rosuvastatin",
                "hydrochlorothiazide",
                "furosemide",
                "spironolactone",
                "omeprazole",
                "esomeprazole",
                "pantoprazole",
                "amlodipine",
                "losartan",
                "valsartan",
                "carvedilol",
                "gabapentin",
                "pregabalin",
                "duloxetine",
                "insulin",
                "glipizide",
                "glyburide",
                "pioglitazone",
                # Medical conditions/anatomy (should not be removed)
                "diabetes",
                "hypertension",
                "osteoporosis",
                "arthritis",
                "pneumonia",
                "bronchitis",
                "asthma",
                "copd",
                "kidney",
                "liver",
                "heart",
                "lung",
                "shoulder",
                "neck",
                # Common medical terms
                "a1c",
                "hemoglobin",
                "glucose",
                "blood pressure",
                "physical therapy",
                "pt",
                "mri",
                "ct",
                "xray",
            }
        )

        # PII detection patterns (raw strings)
        pii_patterns_raw = {
            "phone": [
                r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US phone formats
                r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}",  # Generic phone XXX-XXX-XXXX
                r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}",  # (XXX) XXX-XXXX
            ],
            "email": [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            ],
            "ssn": [
                r"\b\d{3}-?\d{2}-?\d{4}\b",  # Social Security Number
            ],
            "date": [
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # Dates MM/DD/YYYY
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            ],
            "age": [
                r"\bage\s+(\d{1,3})\b",  # Age X
                r"\b(\d{1,3})\s+years?\s+old\b",
                r"\b(\d{1,3})\s+y\.o\.?\b",
            ],
            "zipcode": [
                r"\b\d{5}(?:-\d{4})?\b",  # ZIP codes
            ],
            "mrn": [
                r"\bMRN[:\s]*\d+\b",  # Medical Record Number
                r"\bPatient\s+ID[:\s]*\d+\b",
            ],
            "name": [
                # Doctor titles with names (most specific - highest priority)
                # Match: Dr. Prasad, Dr. John Smith, Doctor Kumar, Dr Prasad
                r"\b(?:Dr|Doctor|Dr\.|MD|MD\.)\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?(?=\s|\.|,|:|$)",  # Dr. Prasad, Dr. John Smith, Dr Prasad
                # Title prefixes with names
                r"\b(Mr|Mrs|Ms|Miss|Mr\.|Mrs\.|Ms\.|Miss\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?=\s|\.|,|$)",  # Mr. Smith, Mrs. Johnson
                # Names after greetings/addresses (exclude titles like Dr, Doctor)
                # Match: Hello John, Hi Mary Smith, Hey Prasad
                r"\b(?:Hello|Hi|Hey|Dear),?\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?)(?=\s|\.|,|:|$)",  # Hello John, Hi Mary Smith, Hey Prasad
                # Full names in context
                r"\b(?:I'?m|name is|called|named|this is)\s+([A-Z][a-z]{2,}\s+[A-Z][a-z]+)(?=\s|\.|,|$)",  # I'm John Smith, name is Mary Johnson
                # Patient/Doctor name mentions with context
                r"\b(?:patient|doctor)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)(?=\s|\.|,|$)",  # patient John Smith, doctor Mary Johnson
                # Standalone capitalized names after "Thank you" (excluding titles)
                r"\b(?:Thank you|Thanks),?\s+([A-Z][a-z]{2,})(?=\s|\.|,|:|$)",  # Thank you, John (but not "Thank you, Dr")
                # Standalone capitalized first names in medical context (common doctor names)
                # Match: "Prasad said", "Kumar will", "Smith mentioned" but NOT medication names
                r"\b([A-Z][a-z]{3,})(?:\s+(?:said|will|mentioned|told|asked|examined|checked|found|sees|sees|prescribed|ordered))",  # Prasad said, Kumar will
                # Capitalized words at start of sentences that might be names (context-dependent)
                # Match: "Prasad, how are you?" but not "Diabetes is..."
                r"(?:^|\.\s+)([A-Z][a-z]{3,})(?:,\s*(?:how|what|when|where|why|can|do|are|is|have))",  # Prasad, how are you?
                # Names mentioned as "Mr/Mrs [Name]" pattern
                r"\b(?:Mr|Mrs|Ms|Miss)\s+([A-Z][a-z]{2,})\b",  # Mr Prasad, Mrs Smith
            ],
        }

        # Pre-compile all regex patterns for 2-3x faster PII removal
        # Store as dict of lists of compiled patterns
        self._compiled_pii_patterns = {}
        for pii_type, patterns in pii_patterns_raw.items():
            self._compiled_pii_patterns[pii_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

        # Store raw patterns for backward compatibility (if needed)
        self._pii_patterns = pii_patterns_raw

        # Pre-compile aggressive PII removal patterns
        self._aggressive_name_patterns = [
            re.compile(
                r"\b([A-Z][a-z]{3,})(?:,?\s+(?:how|what|when|where|why|can|do|are|is|have|will|said|told|asked))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:I\'m|I am|name is|called|named)\s+([A-Z][a-z]{3,})\b",
                re.IGNORECASE,
            ),
            re.compile(r"\bDr\.\s+([A-Z][a-z]{3,})\b", re.IGNORECASE),
            re.compile(
                r"\b([A-Z][a-z]{3,})\s+(?:prescribed|ordered|recommended|suggested)",
                re.IGNORECASE,
            ),
        ]

    async def execute(self, request: AudioTranscriptionRequest, doctor_id: str) -> AudioTranscriptionResponse:
        """Execute the audio transcription use case."""
        LOGGER.info(f"TranscribeAudioUseCase.execute called for patient {request.patient_id}, visit {request.visit_id}")

        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id, doctor_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit
        visit_id = VisitId(request.visit_id)
        visit = await self._visit_repository.find_by_patient_and_visit_id(request.patient_id, visit_id, doctor_id)
        if not visit:
            raise VisitNotFoundError(request.visit_id)

        # Check if transcription is already completed - return existing data if so
        if visit.transcription_session and visit.transcription_session.transcription_status == "completed":
            LOGGER.info(f"Transcription already completed for visit {request.visit_id}, returning existing data")
            return AudioTranscriptionResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                transcript=visit.transcription_session.transcript or "",
                word_count=visit.transcription_session.word_count or 0,
                audio_duration=visit.transcription_session.audio_duration_seconds,
                transcription_status=visit.transcription_session.transcription_status,
                message="Transcription already completed",
            )

        # Check if visit is ready for transcription based on workflow type
        if not visit.can_proceed_to_transcription():
            if visit.is_scheduled_workflow():
                raise ValueError(
                    f"Scheduled visit not ready for transcription. Current status: {visit.status}. Complete intake first."
                )
            elif visit.is_walk_in_workflow():
                raise ValueError(f"Walk-in visit not ready for transcription. Current status: {visit.status}.")
            else:
                raise ValueError(f"Visit not ready for transcription. Current status: {visit.status}.")

        try:
            # Validate audio file only if local file path provided (skip if SAS URL only)
            if request.audio_file_path and not getattr(request, "sas_url", None):
                validation_result = await self._transcription_service.validate_audio_file(request.audio_file_path)

                if not validation_result.get("is_valid", False):
                    raise ValueError(f"Invalid audio file: {validation_result.get('error', 'Unknown error')}")
            elif getattr(request, "sas_url", None):
                # SAS URL provided - skip local file validation
                LOGGER.debug(f"Skipping local file validation (SAS URL provided for visit {request.visit_id})")

            # Get language from request or fallback to patient language
            transcription_language = request.language or getattr(patient, "language", "en") or "en"
            # Normalize language code (handle both 'sp' and 'es' for backward compatibility)
            if transcription_language in ["es", "sp"]:
                transcription_language = "sp"

            # Start transcription process (only if not already started)
            # The API endpoint may have already called start_transcription(None) to mark it as processing
            # In that case, just update the audio_file_path instead of creating a new session
            if visit.transcription_session and visit.transcription_session.transcription_status == "processing":
                # Session already exists and is processing, just update the audio_file_path
                LOGGER.info(
                    f"Transcription session already exists for visit {request.visit_id}, updating audio_file_path"
                )
                visit.transcription_session.audio_file_path = (
                    request.audio_file_path or ""
                )  # Can be None when using SAS URL
                visit.updated_at = datetime.utcnow()
            else:
                # No session or session is not processing, create new one
                visit.start_transcription(request.audio_file_path or "")  # Can be None when using SAS URL

            await self._visit_repository.save(visit)

            # Transcribe audio using Azure Speech Service
            audio_source = getattr(request, "sas_url", None) or request.audio_file_path or "unknown"
            LOGGER.info(
                f"Starting transcription for visit {request.visit_id}, source={('SAS URL' if getattr(request, 'sas_url', None) else 'local file')}, language: {transcription_language}"
            )

            from ...core.utils.timing import TimingContext

            # Create callback to persist transcription metadata during job creation and polling
            async def status_update_callback(update_fields: Dict[str, Any]) -> None:
                """Callback to persist transcription session fields to database."""
                try:
                    # Filter and map fields that should be persisted
                    fields_to_update = {}
                    if "transcription_id" in update_fields:
                        fields_to_update["transcription_id"] = update_fields["transcription_id"]
                    if "azure_job_created_at" in update_fields:
                        fields_to_update["azure_job_created_at"] = update_fields["azure_job_created_at"]
                    if "first_poll_at" in update_fields:
                        fields_to_update["first_poll_at"] = update_fields["first_poll_at"]
                    if "results_downloaded_at" in update_fields:
                        fields_to_update["results_downloaded_at"] = update_fields["results_downloaded_at"]
                    if "last_poll_status" in update_fields:
                        fields_to_update["last_poll_status"] = update_fields["last_poll_status"]
                    if "last_poll_at" in update_fields:
                        fields_to_update["last_poll_at"] = update_fields["last_poll_at"]
                    if "transcription_status" in update_fields:
                        fields_to_update["transcription_status"] = update_fields["transcription_status"]
                    if "error_message" in update_fields:
                        fields_to_update["error_message"] = update_fields["error_message"]

                    if fields_to_update:
                        await self._visit_repository.update_transcription_session_fields(
                            request.patient_id,
                            visit_id,
                            doctor_id,
                            fields_to_update,
                        )
                        LOGGER.debug(
                            f"Persisted transcription session fields: {list(fields_to_update.keys())} for visit {request.visit_id}"
                        )
                except Exception as e:
                    LOGGER.warning(f"Failed to persist transcription session fields: {e}")

            # P1-4: Timestamps are now set exactly by Azure Speech service via callbacks
            # No more approximate calculations needed
            speech_start = time.time()
            with TimingContext("AzureSpeech_Transcription", LOGGER) as timing_ctx:
                timing_ctx.set_input_size(len(audio_source) if audio_source else 0)
            transcription_result = await self._transcription_service.transcribe_audio(
                request.audio_file_path or "",  # Can be None when using SAS URL only
                language=transcription_language,
                medical_context=True,
                sas_url=getattr(request, "sas_url", None),
                status_update_callback=status_update_callback,
                enable_diarization=getattr(request, "enable_diarization", None),  # P1-5: Pass diarization toggle
            )
            speech_end = time.time()
            # Calculate duration after context exits (duration is set in __exit__)
            speech_latency = (speech_end - speech_start) if speech_start else 0.0

            # Check for transcription errors
            if transcription_result.get("error"):
                error_info = transcription_result["error"]
                error_code = error_info.get("code", "UNKNOWN_ERROR")
                error_message = error_info.get("message", "Transcription failed")
                error_details = error_info.get("details", {})

                LOGGER.error(
                    f"Transcription failed: {error_code} - {error_message}",
                    extra={
                        "error_code": error_code,
                        "error_details": error_details,
                        "visit_id": visit.visit_id,
                        "patient_id": visit.patient_id,
                    },
                )

                # Mark visit transcription as failed with error info
                visit.mark_transcription_failed(error_message=f"{error_code}: {error_message}")
                await self._visit_repository.save(visit)

                # Raise with detailed error information
                raise ValueError(f"Transcription failed: {error_message} (code: {error_code})")

            # P1-4: transcription_id and timestamps are already persisted via callbacks
            # No need to update them here - they're set exactly when events occur
            transcription_id = transcription_result.get("transcription_id")
            if transcription_id:
                LOGGER.info(
                    f"Transcription completed with transcription_id={transcription_id} for visit {request.visit_id}"
                )

            raw_transcript = transcription_result.get("transcript", "") or ""
            LOGGER.info(
                f"Transcription completed. Transcript length: {len(raw_transcript)} characters, word_count: {transcription_result.get('word_count', 0)}"
            )
            # REMOVED: Raw transcript preview logging (PHI safety)

            if not raw_transcript or raw_transcript.strip() == "":
                raise ValueError("Transcription returned empty transcript")

            # Azure Speech Service provides structured dialogue with speaker diarization
            pre_structured_dialogue = transcription_result.get("structured_dialogue")
            speaker_info = transcription_result.get("speaker_labels", {})

            # P1-5: Handle case where diarization is disabled (may have empty or single-speaker dialogue)
            if not pre_structured_dialogue or not isinstance(pre_structured_dialogue, list):
                LOGGER.warning(
                    f"Azure Speech Service did not provide structured dialogue for visit {request.visit_id}. "
                    f"This may occur if diarization is disabled. Creating single-speaker dialogue from transcript."
                )
                # Create a single-speaker dialogue from the raw transcript
                pre_structured_dialogue = [{"Speaker 1": raw_transcript}] if raw_transcript else []
                speaker_info = {"speakers": [{"label": "Speaker 1"}]}

            # Map speakers from Azure Speech Service (Speaker 1, Speaker 2) to Doctor/Patient
            LOGGER.info(
                f"Using pre-structured dialogue from Azure Speech Service ({len(pre_structured_dialogue)} turns)"
            )
            from ...application.utils.speaker_mapping import (
                map_speakers_to_doctor_patient,
            )

            with TimingContext("Speaker_Mapping", LOGGER) as timing_ctx:
                timing_ctx.set_input_size(len(pre_structured_dialogue))
                # Normalize Azure dialogue into {"Speaker 1": "text"} format expected by mapper
                normalized_dialogue: List[Dict[str, str]] = []
                for turn in pre_structured_dialogue:
                    if isinstance(turn, dict):
                        speaker_label = turn.get("speaker")
                        text = turn.get("text")
                        if speaker_label and text:
                            normalized_dialogue.append({speaker_label: text})
                        else:
                            normalized_dialogue.append(turn)
                    else:
                        normalized_dialogue.append(turn)

                structured_dialogue = map_speakers_to_doctor_patient(
                    normalized_dialogue,
                    speaker_info=speaker_info,
                    language=transcription_language,
                )
                timing_ctx.set_output_size(len(structured_dialogue))
                timing_ctx.add_metadata(turns=len(structured_dialogue))

            LOGGER.info(f"Mapped speakers to Doctor/Patient: {len(structured_dialogue)} turns")

            LOGGER.info(f"Raw transcript length: {len(raw_transcript)} characters")
            LOGGER.info(f"Structured dialogue turns: {len(structured_dialogue) if structured_dialogue else 0}")

            # Apply PII removal to raw transcript and structured dialogue
            with TimingContext("PII_Removal", LOGGER) as timing_ctx:
                timing_ctx.set_input_size(len(raw_transcript))

                # Parallel PII removal for raw transcript and structured dialogue (CPU-bound, use threads)
                if structured_dialogue:
                    raw_transcript_cleaned, structured_dialogue_cleaned = await asyncio.gather(
                        asyncio.to_thread(self._remove_pii_from_text, raw_transcript),
                        asyncio.to_thread(self._remove_pii_from_dialogue, structured_dialogue),
                    )
                    raw_transcript = raw_transcript_cleaned
                    structured_dialogue = structured_dialogue_cleaned
                else:
                    raw_transcript = self._remove_pii_from_text(raw_transcript)

                timing_ctx.set_output_size(len(raw_transcript))
                timing_ctx.add_metadata(
                    transcript_chars=len(raw_transcript),
                    dialogue_turns=(len(structured_dialogue) if structured_dialogue else 0),
                )

            # Validate PII removal (conditional aggressive pass only if needed)
            with TimingContext("PII_Validation", LOGGER) as timing_ctx:
                pii_validation = self._validate_pii_removal(raw_transcript, structured_dialogue)
                timing_ctx.add_metadata(
                    pii_detected=pii_validation["pii_detected"],
                    pii_count=pii_validation["pii_count"],
                )

            if pii_validation["pii_detected"]:
                LOGGER.warning(
                    f"⚠️ PII validation: {pii_validation['pii_count']} PII items still detected after removal"
                )
                for pii_type, value, location in pii_validation["pii_items"][:10]:  # Show first 10
                    LOGGER.warning(f"  - {pii_type}: {value} (in {location})")
                # Conditional aggressive pass: only if PII still detected
                if structured_dialogue and pii_validation["pii_count"] > 0:
                    LOGGER.info("Attempting conditional aggressive PII removal pass...")
                    structured_dialogue = self._aggressive_pii_removal_from_dialogue(structured_dialogue)
                    # Re-validate
                    pii_validation_retry = self._validate_pii_removal(raw_transcript, structured_dialogue)
                    if pii_validation_retry["pii_detected"]:
                        LOGGER.warning(f"⚠️ PII still detected after retry: {pii_validation_retry['pii_count']} items")
                    else:
                        LOGGER.info("✓ Additional PII removal pass successful")
            else:
                LOGGER.info("✓ PII validation: No PII detected in final output")

            # Measure LLM/PII/dialogue post-processing latency
            # (All operations from after Speech call to just before persistence)
            # NOTE: speech_latency was measured around the Azure Speech call above.
            llm_start = speech_end
            llm_end = time.time()
            llm_latency = llm_end - llm_start if llm_start else 0.0
            # Ensure speech_latency is not None before adding
            speech_latency_safe = speech_latency if speech_latency is not None else 0.0
            total_transcript_time = speech_latency_safe + llm_latency

            LOGGER.info(
                f"⏱️ Transcription pipeline: speech={speech_latency_safe:.2f}s, llm={llm_latency:.2f}s, total={total_transcript_time:.2f}s",
                extra={
                    "event": "transcription_pipeline_timings",
                    "patient_id": request.patient_id,
                    "visit_id": request.visit_id,
                    "speech_latency_sec": speech_latency_safe,
                    "llm_latency_sec": llm_latency,
                    "total_transcript_time_sec": total_transcript_time,
                    "transcript_chars": len(raw_transcript),
                    "structured_dialogue_turns": (len(structured_dialogue) if structured_dialogue else 0),
                },
            )

            # Validate completeness
            if structured_dialogue:
                completeness = self._validate_completeness(structured_dialogue, raw_transcript)
                LOGGER.info(f"Completeness check:")
                LOGGER.info(f"  - Dialogue turns: {completeness['dialogue_turns']}")
                LOGGER.info(f"  - Transcript sentences: {completeness['transcript_sentences']}")
                LOGGER.info(f"  - Completeness ratio: {completeness['completeness_ratio']:.2%}")
                LOGGER.info(f"  - Character ratio: {completeness['char_ratio']:.2%}")

                if not completeness["is_complete"]:
                    LOGGER.warning(
                        f"⚠️  Dialogue completeness is below threshold (0.70): {completeness['completeness_ratio']:.2%}"
                    )
                else:
                    LOGGER.info(f"✓ Dialogue completeness meets threshold: {completeness['completeness_ratio']:.2%}")

            # Complete transcription with both raw transcript and structured dialogue
            visit.complete_transcription_with_data(
                transcript=raw_transcript,  # Store raw transcript
                audio_duration=transcription_result.get("duration"),
                structured_dialogue=structured_dialogue,  # Store structured dialogue separately
            )

            LOGGER.info(f"About to save patient {request.patient_id} with visit {request.visit_id}")
            LOGGER.info(f"Visit status: {visit.status}")
            LOGGER.info(
                f"Transcription session status: {visit.transcription_session.transcription_status if visit.transcription_session else 'None'}"
            )
            LOGGER.info(f"Transcript length: {len(raw_transcript) if raw_transcript else 0}")
            LOGGER.info(f"Structured dialogue turns: {len(structured_dialogue) if structured_dialogue else 0}")

            # Save updated visit with db_saved_at timestamp
            db_saved_at = datetime.utcnow()
            if visit.transcription_session:
                visit.transcription_session.db_saved_at = db_saved_at

            with TimingContext("Database_Save", LOGGER) as timing_ctx:
                timing_ctx.set_input_size(len(raw_transcript))
            await self._visit_repository.save(visit)

            # Calculate latency timings
            enqueued_at = visit.transcription_session.enqueued_at if visit.transcription_session else None
            dequeued_at = visit.transcription_session.dequeued_at if visit.transcription_session else None
            azure_job_created_at = (
                visit.transcription_session.azure_job_created_at if visit.transcription_session else None
            )
            first_poll_at = visit.transcription_session.first_poll_at if visit.transcription_session else None
            results_downloaded_at = (
                visit.transcription_session.results_downloaded_at if visit.transcription_session else None
            )
            started_at = visit.transcription_session.started_at if visit.transcription_session else None

            # Calculate stage durations - ensure all are numeric (default to 0.0 if missing)
            queue_wait = (dequeued_at - enqueued_at).total_seconds() if enqueued_at and dequeued_at else 0.0
            azure_run = (
                (results_downloaded_at - azure_job_created_at).total_seconds()
                if azure_job_created_at and results_downloaded_at
                else 0.0
            )
            download_parse = (
                (db_saved_at - results_downloaded_at).total_seconds() if results_downloaded_at and db_saved_at else 0.0
            )
            total_elapsed = (db_saved_at - started_at).total_seconds() if started_at and db_saved_at else 0.0

            # Clear INFO-level success log with all metrics (always numeric)
            LOGGER.info(
                f"✅ TRANSCRIPTION COMPLETED: visit_id={visit.visit_id.value}, "
                f"transcription_id={transcription_id or 'N/A'}, "
                f"duration_seconds={transcription_result.get('duration') or 0.0}, "
                f"word_count={transcription_result.get('word_count', 0)}, "
                f"transcript_chars={len(raw_transcript)}, "
                f"structured_dialogue_turns={len(structured_dialogue) if structured_dialogue else 0}, "
                f"total_elapsed_seconds={total_elapsed:.2f}, "
                f"timeline=[queue_wait={queue_wait:.2f}s, azure_run={azure_run:.2f}s, download_parse={download_parse:.2f}s]"
            )

            return AudioTranscriptionResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                transcript=raw_transcript,  # Return raw transcript
                word_count=transcription_result.get("word_count", 0),
                audio_duration=transcription_result.get("duration"),
                transcription_status=visit.transcription_session.transcription_status,
                message="Audio transcribed successfully",
            )

        except Exception as e:
            # Extract clean error information (no __name__ bug)
            error_type = type(e).__name__
            error_message = str(e)

            # Avoid double-prefixing error messages
            if error_message.startswith("Transcription failed:"):
                clean_error_message = error_message
            else:
                clean_error_message = f"{error_type}: {error_message}"

            # Log the full error for debugging (no PHI in message)
            LOGGER.error(
                f"Transcription failed for visit {visit.visit_id.value}: {clean_error_message}",
                exc_info=True,
            )

            # Calculate elapsed time for failure log - ensure numeric
            started_at = visit.transcription_session.started_at if visit.transcription_session else None
            failure_time = datetime.utcnow()
            total_elapsed = (failure_time - started_at).total_seconds() if started_at else 0.0
            transcription_id = visit.transcription_session.transcription_id if visit.transcription_session else None

            # Mark transcription as failed with clean error message
            visit.fail_transcription(clean_error_message)
            await self._visit_repository.save(visit)

            # Clear ERROR-level failure log with all identifiers (always numeric)
            LOGGER.error(
                f"❌ TRANSCRIPTION FAILED: visit_id={visit.visit_id.value}, "
                f"transcription_id={transcription_id or 'N/A'}, "
                f"error_type={error_type}, "
                f"error_message={clean_error_message[:200]}, "  # Truncate long messages
                f"total_elapsed_seconds={total_elapsed:.2f}"
            )

            return AudioTranscriptionResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                transcript="",
                word_count=0,
                audio_duration=None,
                transcription_status="failed",
                message=clean_error_message,
            )

    def _remove_pii_from_text(self, text: str) -> str:
        """Remove PII from raw text transcript using pre-compiled patterns (2-3x faster)."""
        if not text:
            return text

        cleaned_text = text
        pii_removed = []

        # Remove phone numbers (using pre-compiled patterns)
        for compiled_pattern in self._compiled_pii_patterns["phone"]:
            cleaned_text = compiled_pattern.sub("[PHONE]", cleaned_text)

        # Remove email addresses
        for compiled_pattern in self._compiled_pii_patterns["email"]:
            cleaned_text = compiled_pattern.sub("[EMAIL]", cleaned_text)

        # Remove SSN
        for compiled_pattern in self._compiled_pii_patterns["ssn"]:
            cleaned_text = compiled_pattern.sub("[SSN]", cleaned_text)

        # Remove specific dates (keep relative dates like "yesterday")
        relative_date_words = frozenset(
            [
                "yesterday",
                "today",
                "tomorrow",
                "last week",
                "next week",
                "last month",
                "next month",
            ]
        )
        for compiled_pattern in self._compiled_pii_patterns["date"]:
            # Find all matches first
            for match in compiled_pattern.finditer(cleaned_text):
                match_text = match.group(0)
                # Check if it's a relative date by checking context
                match_start = match.start()
                context_before = cleaned_text[max(0, match_start - 30) : match_start].lower()
                if not any(rel_word in context_before for rel_word in relative_date_words):
                    cleaned_text = cleaned_text[:match_start] + "[DATE]" + cleaned_text[match.end() :]
                    pii_removed.append(("date", match_text))
                    break  # Only replace first occurrence per pattern

        # Remove ages
        for compiled_pattern in self._compiled_pii_patterns["age"]:
            for match in compiled_pattern.finditer(cleaned_text):
                age_value = match.group(1) if match.groups() else match.group(0)
                cleaned_text = cleaned_text[: match.start()] + "[AGE]" + cleaned_text[match.end() :]
                pii_removed.append(("age", age_value))
                break  # Only replace first occurrence per pattern

        # Remove ZIP codes
        for compiled_pattern in self._compiled_pii_patterns["zipcode"]:
            cleaned_text = compiled_pattern.sub("[ZIPCODE]", cleaned_text)

        # Remove MRN/Patient IDs
        for compiled_pattern in self._compiled_pii_patterns["mrn"]:
            cleaned_text = compiled_pattern.sub("[MRN]", cleaned_text)

        # Remove names using comprehensive patterns (process in reverse order to maintain positions)
        # Process from most specific to least specific to avoid double-matching
        name_matches = []
        for compiled_pattern in self._compiled_pii_patterns["name"]:
            for match in compiled_pattern.finditer(cleaned_text):
                matched_text = match.group(0)
                # Extract the actual name part (remove greeting/prefix)
                name_part = matched_text
                # If pattern has capture groups, extract the name part
                if match.groups():
                    # Use the first capture group if available
                    name_part = match.group(1) if match.group(1) else matched_text
                else:
                    # For patterns without capture groups, extract name after prefix
                    if "Dr" in matched_text or "Doctor" in matched_text or "MD" in matched_text:
                        name_part = re.sub(
                            r"^(?:Dr|Doctor|Dr\.|MD|MD\.)\s+",
                            "",
                            matched_text,
                            flags=re.IGNORECASE,
                        )
                    elif any(prefix in matched_text for prefix in ["Mr", "Mrs", "Ms", "Miss"]):
                        name_part = re.sub(
                            r"^(?:Mr|Mrs|Ms|Miss|Mr\.|Mrs\.|Ms\.|Miss\.)\s+",
                            "",
                            matched_text,
                            flags=re.IGNORECASE,
                        )
                    elif any(
                        greeting in matched_text
                        for greeting in [
                            "Hello",
                            "Hi",
                            "Hey",
                            "Dear",
                            "Thank you",
                            "Thanks",
                        ]
                    ):
                        name_part = re.sub(
                            r"^(?:Hello|Hi|Hey|Dear|Thank you|Thanks),?\s+",
                            "",
                            matched_text,
                            flags=re.IGNORECASE,
                        )

                # Check if this is a medical term (medication, condition, etc.) - should NOT be removed
                # Use frozenset for O(1) lookup instead of O(n) iteration
                name_lower = name_part.lower().strip()

                # Fast medical term check using frozenset (O(1) lookup)
                is_medical_term = name_lower in self._medical_term_whitelist

                # Also check if any medical term is contained in the name (for compound terms)
                if not is_medical_term:
                    is_medical_term = any(
                        med_term in name_lower or name_lower in med_term for med_term in self._medical_term_whitelist
                    )

                # Additional check: if the matched word is part of a medical phrase, don't remove it
                # Check context around the match for medical keywords
                if not is_medical_term:
                    context_start = max(0, match.start() - 30)
                    context_end = min(len(cleaned_text), match.end() + 30)
                    context = cleaned_text[context_start:context_end].lower()

                    # Common medical phrases that might contain capitalized words
                    medical_phrases = [
                        "medication",
                        "prescribed",
                        "taking",
                        "dosage",
                        "milligrams",
                        "mg",
                        "ml",
                        "diagnosed with",
                        "suffering from",
                        "condition",
                        "treatment",
                        "therapy",
                        "blood pressure",
                        "heart rate",
                        "blood test",
                        "lab results",
                    ]

                    # If context suggests medical content, be more conservative
                    if any(phrase in context for phrase in medical_phrases):
                        # Double-check if the matched word could be a medication name
                        # Common medication name patterns (usually lowercase in speech, but might be capitalized)
                        common_med_patterns = [
                            r"\b" + re.escape(name_lower) + r"\s+(?:mg|milligrams|ml|milliliters|tablets|capsules)\b"
                        ]
                        if not any(re.search(pattern, context, re.IGNORECASE) for pattern in common_med_patterns):
                            # If it's not clearly a medication, it might be a name - proceed with removal
                            pass
                        else:
                            is_medical_term = True

                # Only add to matches if it's NOT a medical term
                if not is_medical_term:
                    name_matches.append((match.start(), match.end(), matched_text))

        # Sort by start position and remove overlaps
        name_matches.sort(key=lambda x: x[0])
        non_overlapping = []
        for start, end, text in name_matches:
            if not non_overlapping or start >= non_overlapping[-1][1]:
                non_overlapping.append((start, end, text))

        # Replace in reverse order to maintain positions
        for start, end, matched_text in reversed(non_overlapping):
            cleaned_text = cleaned_text[:start] + "[NAME]" + cleaned_text[end:]
            pii_removed.append(("name", matched_text))

        if pii_removed:
            LOGGER.info(f"PII removed from text: {len(pii_removed)} items")
            for pii_type, value in pii_removed[:5]:  # Log first 5
                LOGGER.debug(f"  - {pii_type}: {value}")

        return cleaned_text

    def _remove_pii_from_dialogue(self, dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove PII from structured dialogue."""
        if not dialogue:
            return dialogue

        cleaned_dialogue = []
        for turn in dialogue:
            if not isinstance(turn, dict) or len(turn) != 1:
                cleaned_dialogue.append(turn)
                continue

            speaker = list(turn.keys())[0]
            text = list(turn.values())[0]

            # Remove PII from the dialogue text
            cleaned_text = self._remove_pii_from_text(text)
            cleaned_dialogue.append({speaker: cleaned_text})

        return cleaned_dialogue

    def _aggressive_pii_removal_from_dialogue(self, dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply additional aggressive PII removal pass to structured dialogue using pre-compiled patterns."""
        if not dialogue:
            return dialogue

        cleaned_dialogue = []
        for turn in dialogue:
            if not isinstance(turn, dict) or len(turn) != 1:
                cleaned_dialogue.append(turn)
                continue

            speaker = list(turn.keys())[0]
            text = list(turn.values())[0]

            # Additional aggressive patterns for common name variations (using pre-compiled patterns)
            cleaned_text = text

            # Use pre-compiled aggressive patterns (faster than compiling on each call)
            for compiled_pattern in self._aggressive_name_patterns:
                for match in compiled_pattern.finditer(cleaned_text):
                    name_candidate = match.group(1) if match.groups() else match.group(0)
                    name_lower = name_candidate.lower().strip()

                    # Fast medical term check using frozenset (O(1) lookup)
                    is_medical = name_lower in self._medical_term_whitelist

                    # Also check if any medical term is contained in the name
                    if not is_medical:
                        is_medical = any(
                            med_term in name_lower or name_lower in med_term
                            for med_term in self._medical_term_whitelist
                        )

                    if not is_medical:
                        # Replace with [NAME]
                        cleaned_text = cleaned_text[: match.start()] + "[NAME]" + cleaned_text[match.end() :]
                        LOGGER.debug(f"Aggressive PII removal: removed '{name_candidate}' from dialogue")
                        break  # Break to avoid multiple replacements of same match

            cleaned_dialogue.append({speaker: cleaned_text})

        return cleaned_dialogue

    def _validate_pii_removal(self, text: str, dialogue: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Validate that PII has been removed from output."""
        pii_found = []

        # Check text
        for pii_type, patterns in self._pii_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    pii_found.extend([(pii_type, m, "text") for m in matches])

        # Check dialogue if provided
        if dialogue:
            for turn in dialogue:
                if isinstance(turn, dict):
                    for speaker, text in turn.items():
                        for pii_type, patterns in self._pii_patterns.items():
                            for pattern in patterns:
                                matches = re.findall(pattern, text, re.IGNORECASE)
                                if matches:
                                    pii_found.extend([(pii_type, m, f"dialogue-{speaker}") for m in matches])

        return {
            "pii_detected": len(pii_found) > 0,
            "pii_count": len(pii_found),
            "pii_items": pii_found[:10],  # First 10 items
        }

    async def _retry_chunk_processing(
        self,
        client: AzureAIClient,
        system_prompt: str,
        user_prompt: str,
        settings,
        logger,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 5.0,
    ) -> Tuple[str, bool]:
        """Retry chunk processing with exponential backoff."""
        last_error = None
        for attempt in range(max_retries):
            try:
                result = await self._process_single_chunk(client, system_prompt, user_prompt, settings, logger)
                if result and result.strip():
                    logger.info(f"✓ Chunk processing succeeded on attempt {attempt + 1}/{max_retries}")
                    return result, True
                else:
                    logger.warning(f"Chunk processing returned empty result (attempt {attempt + 1}/{max_retries})")
                    last_error = "Empty result from LLM"
            except Exception as e:
                last_error = str(e)
                error_str = str(e).lower()

                # Check for specific error types
                is_rate_limit = any(
                    keyword in error_str for keyword in ["rate limit", "429", "too many requests", "quota"]
                )
                is_timeout = any(keyword in error_str for keyword in ["timeout", "timed out", "504"])
                is_connection = any(keyword in error_str for keyword in ["connection", "network", "dns"])
                is_service_error = any(
                    keyword in error_str
                    for keyword in [
                        "service unavailable",
                        "503",
                        "502",
                        "500",
                        "internal server",
                    ]
                )

                # Only retry on transient errors
                is_transient = is_rate_limit or is_timeout or is_connection or is_service_error

                if is_transient:
                    error_type = (
                        "rate limit"
                        if is_rate_limit
                        else ("timeout" if is_timeout else ("connection" if is_connection else "service error"))
                    )
                    logger.warning(f"Transient {error_type} error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.info(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Non-transient error, log and return
                    logger.error(f"Non-transient error on attempt {attempt + 1}/{max_retries}: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Don't retry on non-transient errors
                    return "", False

            # If we get here, it's an empty result - retry if not last attempt
            if attempt < max_retries - 1:
                delay = min(base_delay * (2**attempt), max_delay)
                logger.info(f"Retrying in {delay} seconds due to empty result...")
                await asyncio.sleep(delay)

        logger.error(f"Failed to process chunk after {max_retries} attempts. Last error: {last_error}")
        return "", False

    def _validate_completeness(
        self, dialogue: List[Dict[str, str]], transcript: str, threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Validate that dialogue captures sufficient content from transcript."""
        if not dialogue or not transcript:
            return {
                "completeness_ratio": 0.0,
                "is_complete": False,
                "dialogue_turns": len(dialogue) if dialogue else 0,
                "transcript_sentences": 0,
            }

        # Count sentences in transcript
        transcript_sentences = len([s for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()])

        # Count dialogue turns
        dialogue_turns = len(dialogue)

        # Calculate completeness ratio
        # Note: One dialogue turn can contain multiple sentences, so we compare turns to sentences
        # A reasonable ratio would be: dialogue_turns / transcript_sentences
        # But we need to account for the fact that dialogue turns are often multiple sentences
        # So we use a more lenient threshold
        completeness_ratio = dialogue_turns / transcript_sentences if transcript_sentences > 0 else 0.0

        # Alternative: Estimate based on character count
        dialogue_text = " ".join([list(turn.values())[0] for turn in dialogue if isinstance(turn, dict)])
        dialogue_chars = len(dialogue_text)
        transcript_chars = len(transcript)
        char_ratio = dialogue_chars / transcript_chars if transcript_chars > 0 else 0.0

        # Use the higher ratio (more lenient)
        final_ratio = max(completeness_ratio, char_ratio * 0.8)  # Scale char ratio slightly

        is_complete = final_ratio >= threshold

        return {
            "completeness_ratio": final_ratio,
            "is_complete": is_complete,
            "dialogue_turns": dialogue_turns,
            "transcript_sentences": transcript_sentences,
            "dialogue_chars": dialogue_chars,
            "transcript_chars": transcript_chars,
            "char_ratio": char_ratio,
        }

    def _recover_partial_json(self, partial_json: str, logger) -> Optional[List[Dict[str, str]]]:
        """Try to recover dialogue from partial or malformed JSON."""
        if not partial_json:
            return None

        recovered = []

        # Strategy 1: Try to extract valid JSON objects using regex
        json_object_pattern = r'\{"(Doctor|Patient)":\s*"[^"]*"\}'
        matches = re.findall(json_object_pattern, partial_json)

        if matches:
            # Try to reconstruct
            dialogue_pattern = r'\{"(Doctor|Patient)":\s*"([^"]*)"\}'
            for match in re.finditer(dialogue_pattern, partial_json):
                speaker = match.group(1)
                text = match.group(2)
                recovered.append({speaker: text})

            if recovered:
                logger.info(f"Recovered {len(recovered)} dialogue turns from partial JSON using regex")
                return recovered

        # Strategy 2: Try to fix common JSON issues
        cleaned = partial_json.strip()

        # Remove incomplete objects at the end
        while cleaned.endswith(","):
            cleaned = cleaned[:-1]

        # Try to close the array
        if cleaned.startswith("[") and not cleaned.endswith("]"):
            # Find last complete object
            last_complete = cleaned.rfind("},")
            if last_complete != -1:
                cleaned = cleaned[: last_complete + 1] + "]"
            else:
                cleaned = cleaned + "]"

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                logger.info(f"Recovered {len(parsed)} dialogue turns from fixed JSON")
                return parsed
        except json.JSONDecodeError:
            pass

        return None

    async def _process_transcript_with_chunking(
        self,
        client: AzureAIClient,
        raw_transcript: str,
        settings,
        logger,
        language: str = "en",
    ) -> str:
        """Process transcript with robust chunking strategy for long content."""

        # Highly refined system prompt for superior speaker attribution (language-aware)
        if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
            system_prompt = (
                "Eres un analista experto de diálogos médicos. Convierte transcripciones crudas de consultas médicas en diálogos estructurados Doctor-Paciente.\n\n"
                "🎯 TAREA PRINCIPAL:\n"
                'Devuelve un arreglo JSON donde cada elemento es {"Doctor": "..."} o {"Paciente": "..."} - UNA clave por turno.\n\n'
                "📋 REGLAS DE IDENTIFICACIÓN DE HABLANTE (Aplicar en orden):\n\n"
                "1. ANÁLISIS BASADO EN CONTEXTO (MÁS IMPORTANTE):\n"
                "   • SIEMPRE analiza el turno PREVIO para determinar el hablante\n"
                "   • Si el turno anterior fue Doctor haciendo pregunta → la siguiente respuesta es Paciente\n"
                "   • Si el turno anterior fue Paciente respondiendo → la siguiente declaración es Doctor\n"
                "   • El examen físico sigue patrón: Doctor da instrucción → Paciente responde → Doctor observa\n\n"
                "2. SEÑALES DEL DOCTOR (99% precisión cuando están presentes):\n"
                '   • PREGUNTAS (interrogativas): "¿Cuándo...?", "¿Cuánto tiempo...?", "¿Puedes...?", "¿Qué...?", "¿Alguna...?"\n'
                '   • INSTRUCCIONES (imperativas): "Déjame...", "Voy a...", "Vamos a...", "Puede mover...", "Levante...", "Resista..."\n'
                '   • EVALUACIONES CLÍNICAS: "Veo...", "No veo...", "Parece...", "Es una buena señal", "Sospecho..."\n'
                "   • TERMINOLOGÍA MÉDICA: nombres de fármacos, términos anatómicos, diagnósticos, procedimientos\n"
                '   • DECLARACIONES DE AUTORIDAD: "Recomiendo", "Debe", "Es importante", "Necesitamos"\n'
                '   • PLAN/PRESCRIPCIÓN: "Voy a ordenar", "Voy a prescribir", "Voy a referir", "Vamos a programar"\n'
                '   • COMANDOS DE EXAMEN: "Mueva su...", "Levante...", "Resista...", "¿Puede sentir...?", "¿Siente algún dolor?"\n'
                '   • SALUDOS/APERTURAS: "Hola soy el Dr.", "Mucho gusto", "¿En qué puedo ayudarle?"\n\n'
                "3. SEÑALES DEL PACIENTE (99% precisión cuando están presentes):\n"
                '   • EXPERIENCIAS EN PRIMERA PERSONA: "Tengo", "Siento", "He estado", "Tomé", "Fui", "Estoy aquí por"\n'
                '   • RESPUESTAS DIRECTAS: "Sí", "No", "Alrededor de...", "Fue...", "No..."\n'
                '   • DESCRIPCIONES DE SÍNTOMAS: "Me duele", "Es doloroso", "Comenzó...", "Empeora cuando..."\n'
                '   • HISTORIA PERSONAL: "Usualmente...", "Trato de...", "No he...", "Mi última..."\n'
                '   • RESPUESTAS A INSTRUCCIONES: "Bien", "Sí doctor", "No duele", "Está bien", "De acuerdo" (DESPUÉS del comando del doctor)\n'
                '   • CONFIRMACIÓN: "Sí, está bien", "Entiendo", "Comprendo", "Suena bien"\n'
                '   • PREGUNTAS AL DOCTOR: "¿Qué significa eso?", "¿Es grave?", "¿Cuánto tiempo...?", "¿Necesito...?"\n\n'
                "4. CASOS ESPECIALES:\n"
                "   • Durante exámenes físicos: Doctor da comando → Paciente responde brevemente → Doctor hace observación\n"
                '   • Ejemplo: Doctor: "¿Puede mover su hombro?" → Paciente: "Bien" → Doctor: "Excelente. ¿Siente algún dolor?" → Paciente: "No duele"\n'
                '   • Cuando doctor pregunta "¿Siente algún dolor?" → la respuesta es SIEMPRE Paciente\n'
                '   • Cuando doctor dice "Déjame..." o "Voy a..." → SIEMPRE Doctor\n'
                '   • Cuando paciente dice "Estoy aquí por..." o "Necesito..." → SIEMPRE Paciente\n'
                '   • Resúmenes al final: "Solo para recapitular" = Doctor, "No tengo preguntas" = Paciente\n\n'
                "5. ÁRBOL DE DECISIÓN PARA CASOS AMBIGUOS:\n"
                "   • Si contiene signo de interrogación (?) → probablemente Doctor preguntando\n"
                '   • Si empieza con "Yo" + verbo + experiencia personal → Paciente\n'
                "   • Si contiene términos médicos (diagnóstico, nombres de fármacos) → probablemente Doctor explicando\n"
                '   • Si respuesta corta (">Bien", ">Excelente", ">Sí") DESPUÉS de instrucción del doctor → Paciente\n'
                '   • Si describe lo que el doctor hará (">Voy a...", ">Vamos a...") → Doctor\n'
                "   • Si no está seguro, verifica CONTEXTO: ¿qué se dijo antes?\n\n"
                "📝 INSTRUCCIONES DE PROCESAMIENTO:\n"
                '• ELIMINA TODOS LOS NOMBRES: Nombres de doctores ("Dr. Prasad", "Dr. García"), nombres de pacientes ("Juan", "María López"), TODOS los nombres propios → Reemplazar con [NAME]\n'
                "• ELIMINA: direcciones, teléfonos, fechas específicas, edades\n"
                "• ⚠️ CRÍTICO: NO ELIMINES NOMBRES DE MEDICAMENTOS - Estos son términos médicos, NO PII:\n"
                '  - Ejemplos: "metformina", "jardiance", "lisinopril", "amlodipino", "lidocaína" → MANTENER COMO ESTÁ\n'
                '  - "Sí, metformina y jardiance" → MANTENER "metformina" y "jardiance" (NO cambiar a [NAME])\n'
                "  - Los nombres de medicamentos son información médica esencial y deben preservarse\n"
                "• NO ELIMINES: Condiciones médicas, síntomas, partes del cuerpo, dosificaciones, mediciones médicas\n"
                "• CORRIGE: errores obvios de transcripción preservando el significado\n"
                "• LIMPIA: muletillas (eh, em, este), falsos comienzos, repeticiones\n"
                "• COMBINA: oraciones relacionadas del mismo hablante en UN solo turno\n"
                "• PRESERVE: terminología médica, contexto clínico, flujo de conversación\n\n"
                "🔄 PATRÓN DE CONVERSACIÓN:\n"
                "Doctor saluda → Paciente indica razón → Doctor hace preguntas → Paciente responde → Doctor examina → Paciente responde → Doctor resume → Paciente confirma\n\n"
                "⚠️ REQUISITOS CRÍTICOS DE SALIDA:\n"
                '• Devuelve SOLO arreglo JSON válido: [{"Doctor": "..."}, {"Paciente": "..."}]\n'
                "• SIN markdown, SIN bloques de código, SIN explicaciones, SIN comentarios\n"
                "• SIN envolver en ```json``` - empieza directamente con [\n"
                "• Cada turno = UNA idea o respuesta completa\n"
                "• Procesa transcripción COMPLETA - incluye TODOS los turnos de diálogo\n"
                "• NO trunques ni te detengas temprano\n"
                "• Escapa comillas correctamente en JSON\n"
                "• Termina con ]\n\n"
                "📤 EJEMPLO DE SALIDA:\n"
                '[{"Doctor": "Hola, soy el [NAME]. ¿En qué puedo ayudarle?"}, {"Paciente": "Estoy aquí para mi examen físico y recargas de medicamentos."}, {"Doctor": "¿Cuándo fue diagnosticado con diabetes?"}, {"Paciente": "Hace unos cinco años."}, {"Doctor": "¿Puede mover su hombro arriba y abajo?"}, {"Paciente": "Sí."}, {"Doctor": "¿Siente algún dolor?"}, {"Paciente": "No duele."}]'
            )
        else:
            system_prompt = (
                "You are an expert medical dialogue analyzer. Convert raw medical consultation transcripts into structured Doctor-Patient dialogue.\n\n"
                "🎯 CORE TASK:\n"
                'Output a JSON array where each element is {"Doctor": "..."} or {"Patient": "..."} - ONE key per turn.\n\n'
                "📋 SPEAKER IDENTIFICATION RULES (Apply in order):\n\n"
                "1. CONTEXT-BASED ANALYSIS (MOST IMPORTANT):\n"
                "   • ALWAYS analyze the PREVIOUS turn to determine speaker\n"
                "   • If previous turn was Doctor asking a question → next response is Patient\n"
                "   • If previous turn was Patient answering → next statement is Doctor\n"
                "   • Physical exam follows pattern: Doctor gives instruction → Patient responds → Doctor observes\n\n"
                "2. DOCTOR SIGNALS (99% accuracy when present):\n"
                '   • QUESTIONS (interrogative): "When...?", "How long...?", "Can you...?", "What...?", "Any...?"\n'
                '   • INSTRUCTIONS (imperative): "Let me...", "I\'ll...", "We\'ll...", "Can you move...", "Raise your...", "Resist against..."\n'
                '   • CLINICAL ASSESSMENTS: "I see...", "I don\'t see...", "It appears...", "That\'s a good sign", "I suspect..."\n'
                "   • MEDICAL TERMINOLOGY: drug names, anatomical terms, diagnoses, procedures\n"
                '   • AUTHORITY STATEMENTS: "I recommend", "You should", "It\'s important", "We need to"\n'
                '   • PLAN/PRESCRIPTION: "I\'ll order", "I\'ll prescribe", "I\'ll refer", "We\'ll schedule"\n'
                '   • EXAM COMMANDS: "Move your...", "Raise...", "Resist...", "Can you feel...", "Do you feel any pain?"\n'
                '   • GREETINGS/OPENINGS: "Hi I\'m Dr.", "Nice to meet you", "How can I help?"\n\n'
                "3. PATIENT SIGNALS (99% accuracy when present):\n"
                '   • FIRST-PERSON EXPERIENCES: "I have", "I feel", "I\'ve been", "I took", "I went", "I\'m here for"\n'
                '   • DIRECT ANSWERS: "Yes", "No", "About...", "It was...", "I don\'t..."\n'
                '   • SYMPTOM DESCRIPTIONS: "It hurts", "It\'s painful", "It started...", "It gets worse when..."\n'
                '   • PERSONAL HISTORY: "I usually...", "I try to...", "I haven\'t...", "My last..."\n'
                '   • RESPONSES TO INSTRUCTIONS: "Okay", "Yes doctor", "No pain", "That\'s fine", "Alright" (AFTER doctor\'s command)\n'
                '   • CONFIRMATION: "Yes, that\'s okay", "I understand", "Got it", "Sounds good"\n'
                '   • QUESTIONS TO DOCTOR: "What does that mean?", "Is it serious?", "How long...?", "Do I need...?"\n\n'
                "4. SPECIAL CASES:\n"
                "   • During physical exams: Doctor gives command → Patient responds briefly → Doctor makes observation\n"
                '   • Example: Doctor: "Can you move your shoulder?" → Patient: "Okay" → Doctor: "Great. Do you feel any pain?" → Patient: "No pain"\n'
                '   • When doctor asks "Do you feel any pain?" → response is ALWAYS Patient\n'
                '   • When doctor says "Let me..." or "I\'m going to..." → ALWAYS Doctor\n'
                '   • When patient says "I\'m here for..." or "I need..." → ALWAYS Patient\n'
                '   • Recaps/summaries at end: "Just to recap" = Doctor, "No questions" = Patient\n\n'
                "5. DECISION TREE FOR AMBIGUOUS CASES:\n"
                "   • If contains question mark (?) → likely Doctor asking\n"
                '   • If starts with "I" + verb + personal experience → Patient\n'
                "   • If contains medical terms (diagnosis, drug names) → likely Doctor explaining\n"
                '   • If short response (">Okay", ">Great", ">Yes") AFTER doctor\'s instruction → Patient\n'
                '   • If describes what doctor will do (">I\'ll...", ">We\'ll...") → Doctor\n'
                "   • If unsure, check CONTEXT: what was said before?\n\n"
                "📝 PROCESSING INSTRUCTIONS:\n"
                '• REMOVE ALL NAMES: Doctor names ("Dr. Prasad", "Dr. Smith"), Patient names ("John", "Mary Johnson"), ALL proper names → Replace with [NAME]\n'
                "• REMOVE: addresses, phone numbers, specific dates, ages\n"
                "• ⚠️ CRITICAL: DO NOT REMOVE MEDICATION NAMES - These are medical terms, NOT PII:\n"
                '  - Examples: "metformin", "jardiance", "lisinopril", "amlodipine", "lidocaine" → KEEP AS IS\n'
                '  - "Yes, metformin and jardiance" → KEEP "metformin" and "jardiance" (do NOT change to [NAME])\n'
                "  - Medication names are essential medical information and must be preserved\n"
                "• DO NOT REMOVE: Medical conditions, symptoms, body parts, dosages, medical measurements\n"
                "• FIX: obvious transcription errors while preserving meaning\n"
                "• CLEAN: filler words (um, uh, like), false starts, repetitions\n"
                "• COMBINE: related sentences from same speaker into ONE turn\n"
                "• PRESERVE: medical terminology, clinical context, conversation flow\n\n"
                "🔄 CONVERSATION PATTERN:\n"
                "Doctor greets → Patient states reason → Doctor asks questions → Patient answers → Doctor examines → Patient responds → Doctor summarizes → Patient confirms\n\n"
                "⚠️ CRITICAL OUTPUT REQUIREMENTS:\n"
                '• Output ONLY valid JSON array: [{"Doctor": "..."}, {"Patient": "..."}]\n'
                "• NO markdown, NO code blocks, NO explanations, NO comments\n"
                "• NO ```json``` wrapper - start directly with [\n"
                "• Each turn = ONE complete thought or response\n"
                "• Process COMPLETE transcript - include ALL dialogue turns\n"
                "• DO NOT truncate or stop early\n"
                "• Escape quotes properly in JSON\n"
                "• End with ]\n\n"
                "📤 EXAMPLE OUTPUT:\n"
                '[{"Doctor": "Hi, I\'m [NAME]. How can I help you today?"}, {"Patient": "I\'m here for my physical exam and medication refills."}, {"Doctor": "When were you diagnosed with diabetes?"}, {"Patient": "About five years ago."}, {"Doctor": "Can you move your shoulder up and down?"}, {"Patient": "Yes."}, {"Doctor": "Do you feel any pain?"}, {"Patient": "No pain."}]'
            )

        # Calculate optimal chunk size based on deployment context
        # Optimized: Larger chunks = fewer LLM calls = faster + cheaper
        # More specific model detection for chunk sizing
        deployment_name = settings.azure_openai.deployment_name.lower()

        if "gpt-4o-mini" in deployment_name:
            # GPT-4o-mini: 128k context, but optimized for efficiency - use medium chunks
            max_chars_per_chunk = 4000
        elif deployment_name.startswith("gpt-4"):
            # Full GPT-4 models (gpt-4, gpt-4o, gpt-4-turbo): 128k context - use large chunks
            max_chars_per_chunk = 5000
        else:
            # Other models (gpt-3.5, etc.): smaller context - use smaller chunks
            max_chars_per_chunk = 3000

        overlap_chars = 200  # Reduced overlap (200 chars is sufficient for context, saves processing)

        # Split into sentences for better chunking
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_transcript) if s.strip()]

        if len(raw_transcript) <= max_chars_per_chunk:
            # Single chunk processing
            logger.info("Processing as single chunk (no chunking needed)")
            if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
                user_prompt = (
                    f"TRANSCRIPCIÓN DE CONSULTA MÉDICA:\n"
                    f"{raw_transcript}\n\n"
                    f"TAREA: Convierte esta transcripción cruda en un diálogo estructurado Doctor-Paciente.\n"
                    f"Sigue el flujo de la conversación y usa las reglas de identificación para asignar correctamente los hablantes.\n\n"
                    f"SALIDA: Devuelve SOLO un arreglo JSON que empiece con [ y termine con ]. No uses markdown, bloques de código ni otro formato."
                )
            else:
                user_prompt = (
                    f"MEDICAL CONSULTATION TRANSCRIPT:\n"
                    f"{raw_transcript}\n\n"
                    f"TASK: Convert this raw transcript into a structured Doctor-Patient dialogue.\n"
                    f"Follow the conversation flow and use the identification rules to assign speakers correctly.\n\n"
                    f"OUTPUT: Return ONLY a JSON array starting with [ and ending with ]. Do not use markdown, code blocks, or any other formatting."
                )
            return await self._process_single_chunk(client, system_prompt, user_prompt, settings, logger)

        # Multi-chunk processing with overlap
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_chars_per_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous
                overlap_start = max(0, len(current_chunk) - overlap_chars)
                current_chunk = current_chunk[overlap_start:] + " " + sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"Processing transcript in {len(chunks)} chunks with {overlap_chars} char overlap")
        logger.info(f"Total chunks to process: {len(chunks)}")

        # Log chunk details for debugging
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1}/{len(chunks)}: {len(chunk)} chars, preview: {chunk[:100]}...")

        # Process chunks with parallel execution (bounded concurrency)
        # Strategy: Process chunks in parallel with semaphore limit (3 concurrent LLM calls)
        # Each chunk gets context from previous chunk (which should be completed)
        # This provides 3-5x speedup for multi-chunk transcripts without increasing API costs
        from ...core.utils.timing import TimingContext

        with TimingContext("LLM_ChunkProcessing", logger) as timing_ctx:
            timing_ctx.add_metadata(chunk_count=len(chunks), max_chars_per_chunk=max_chars_per_chunk)

            # Bounded concurrency: max 3 concurrent LLM calls (cost-neutral, just faster)
            semaphore = asyncio.Semaphore(3)
            chunk_results = [None] * len(chunks)  # Pre-allocate list to maintain order
            previous_chunk_turns = []  # Store last 3 turns for context

            async def process_chunk_with_context(chunk_idx: int, chunk_text: str, prev_turns: List[Dict[str, str]]):
                """Process a single chunk with context and retry logic."""
                async with semaphore:  # Limit concurrent LLM calls
                    chunk_start_time = time.time()

                    # Build context from previous chunk if available
                    context_text = ""
                    if prev_turns and chunk_idx > 0:
                        context_turns = prev_turns[-3:]  # Last 3 turns
                        if (language or "en").lower() in [
                            "sp",
                            "es",
                            "es-es",
                            "es-mx",
                            "spanish",
                        ]:
                            context_text = (
                                "CONTEXTO DE CONVERSACIÓN PREVIA:\n"
                                + "\n".join(
                                    [f"{list(turn.keys())[0]}: {list(turn.values())[0]}" for turn in context_turns]
                                )
                                + "\n\n"
                            )
                        else:
                            context_text = (
                                "PREVIOUS CONVERSATION CONTEXT:\n"
                                + "\n".join(
                                    [f"{list(turn.keys())[0]}: {list(turn.values())[0]}" for turn in context_turns]
                                )
                                + "\n\n"
                            )

                    if (language or "en").lower() in [
                        "sp",
                        "es",
                        "es-es",
                        "es-mx",
                        "spanish",
                    ]:
                        chunk_prompt = (
                            f"{context_text}"
                            f"FRAGMENTO DE TRANSCRIPCIÓN DE CONSULTA MÉDICA {chunk_idx+1}:\n"
                            f"{chunk_text}\n\n"
                            f"TAREA: Convierte este fragmento en diálogo estructurado Doctor-Paciente.\n"
                            f"Nota: Es parte de una conversación más larga. Usa el contexto previo y las pistas de contexto para mantener continuidad.\n\n"
                            f"SALIDA: Devuelve SOLO un arreglo JSON que empiece con [ y termine con ]. No uses markdown ni bloques de código."
                        )
                    else:
                        chunk_prompt = (
                            f"{context_text}"
                            f"MEDICAL CONSULTATION TRANSCRIPT CHUNK {chunk_idx+1}:\n"
                            f"{chunk_text}\n\n"
                            f"TASK: Convert this transcript chunk into structured Doctor-Patient dialogue.\n"
                            f"Note: This is part of a larger conversation. Use the previous context and context clues to maintain continuity.\n\n"
                            f"OUTPUT: Return ONLY a JSON array starting with [ and ending with ]. Do not use markdown, code blocks, or any other formatting."
                        )

                    # Use retry logic for chunk processing
                    chunk_result, success = await self._retry_chunk_processing(
                        client, system_prompt, chunk_prompt, settings, logger
                    )

                    chunk_processing_time = time.time() - chunk_start_time
                    logger.info(f"Chunk {chunk_idx+1}/{len(chunks)} processing time: {chunk_processing_time:.2f}s")

                    if not success or not chunk_result:
                        logger.warning(
                            f"Chunk {chunk_idx+1} processing failed after retries, attempting fallback extraction"
                        )
                        fallback_dialogue = self._extract_dialogue_fallback(chunk_text, logger, language)
                        if fallback_dialogue:
                            return fallback_dialogue, (
                                fallback_dialogue[-3:] if len(fallback_dialogue) >= 3 else fallback_dialogue
                            )
                        return [{"Doctor": f"[Chunk {chunk_idx+1} processing failed - unable to extract dialogue]"}], []

                    # Parse result
                    parsed = None
                    if chunk_result and chunk_result != chunk_text:
                        try:
                            cleaned_result = chunk_result.strip()
                            if not cleaned_result.endswith("]"):
                                recovered = self._recover_partial_json(cleaned_result, logger)
                                if recovered:
                                    parsed = recovered
                                else:
                                    if cleaned_result.startswith("["):
                                        last_complete_idx = cleaned_result.rfind("},")
                                        if last_complete_idx != -1:
                                            cleaned_result = cleaned_result[: last_complete_idx + 1] + "]"
                                        else:
                                            cleaned_result = cleaned_result + "]"

                            if not parsed:
                                try:
                                    parsed = json.loads(cleaned_result)
                                except json.JSONDecodeError:
                                    parsed = self._recover_partial_json(chunk_result, logger)

                            if parsed and isinstance(parsed, list):
                                return parsed, (parsed[-3:] if len(parsed) >= 3 else parsed)
                        except Exception as e:
                            logger.warning(f"Chunk {chunk_idx+1} parsing error: {e}")

                    return [{"Doctor": f"[Chunk {chunk_idx+1} processing failed - invalid format]"}], []

            # Process chunks sequentially but with optimized chunk sizes
            # Note: Parallel processing is limited by context dependency (each chunk needs previous chunk's context)
            # Main optimization: Larger chunks = fewer LLM calls = faster + cheaper
            start_time = time.time()

            for i, chunk in enumerate(chunks):
                chunk_start_time = time.time()
                progress_percent = ((i + 1) / len(chunks)) * 100

                logger.info(f"=" * 60)
                logger.info(f"📦 Processing chunk {i+1}/{len(chunks)} ({progress_percent:.1f}% complete)")
                if i > 0:
                    elapsed = time.time() - start_time
                    avg_time_per_chunk = elapsed / i
                    remaining_chunks = len(chunks) - (i + 1)
                    estimated_remaining = avg_time_per_chunk * remaining_chunks
                    logger.info(f"⏱️  Elapsed: {elapsed:.1f}s | Estimated remaining: {estimated_remaining:.1f}s")

                # Process chunk with context
                result, prev_turns = await process_chunk_with_context(i, chunk, previous_chunk_turns)
                chunk_results[i] = result
                if prev_turns:
                    previous_chunk_turns = prev_turns

                chunk_processing_time = time.time() - chunk_start_time
                logger.info(f"Chunk {i+1}/{len(chunks)} completed in {chunk_processing_time:.2f}s")

            # Filter out None results (shouldn't happen, but safety check)
            chunk_results = [r for r in chunk_results if r is not None]

        total_processing_time = time.time() - start_time
        timing_ctx.duration = total_processing_time
        timing_ctx.add_metadata(avg_time_per_chunk=total_processing_time / len(chunks) if chunks else 0)
        logger.info(f"=" * 60)
        logger.info(f"📊 All {len(chunks)} chunks processed in {total_processing_time:.2f}s (parallel execution)")

        # Merge and clean up overlapping content
        merged_dialogue = self._merge_chunk_results(chunk_results, logger)

        # Log final result for debugging
        logger.info(f"Final merged dialogue: {len(merged_dialogue)} turns")
        if merged_dialogue:
            logger.info(f"First turn: {merged_dialogue[0]}")
            logger.info(f"Last turn: {merged_dialogue[-1]}")

        # Convert back to JSON string
        result_json = json.dumps(merged_dialogue)
        return result_json

    async def _process_single_chunk(
        self,
        client: AzureAIClient,
        system_prompt: str,
        user_prompt: str,
        settings,
        logger,
    ) -> str:
        """Process a single chunk of transcript."""

        async def _call_openai() -> str:
            try:
                # Use appropriate max_tokens for chunk processing - increased to prevent truncation
                # More specific model detection
                deployment_name = settings.azure_openai.deployment_name.lower()
                if "gpt-4o-mini" in deployment_name:
                    max_tokens = 3000  # Conservative for mini model
                elif deployment_name.startswith("gpt-4"):
                    max_tokens = 4000  # Full GPT-4 models
                else:
                    max_tokens = 3000  # Other models

                logger.info(f"=== STARTING LLM CALL ===")
                logger.info(f"Deployment: {settings.azure_openai.deployment_name}")
                logger.info(f"Max tokens: {max_tokens}")
                logger.info(f"Azure OpenAI configured: {bool(settings.azure_openai.api_key)}")
                logger.info(f"System prompt length: {len(system_prompt)} characters")
                logger.info(f"User prompt length: {len(user_prompt)} characters")
                logger.info(f"User prompt preview: {user_prompt[:300]}...")

                # Use unified client.chat() method
                resp = await client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=settings.azure_openai.deployment_name,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.1,
                )

                content = (resp.choices[0].message.content or "").strip()

                # Clean up markdown formatting if present
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()

                logger.info(f"=== LLM CALL SUCCESS ===")
                logger.info(f"Response length: {len(content)} characters")
                logger.info(f"Full response: {content}")
                logger.info(f"=== END LLM CALL ===")
                return content

            except Exception as e:

                logger.error(f"=== LLM CALL FAILED ===")
                logger.error(f"Error: {str(e)}")
                logger.error(f"Deployment: {settings.azure_openai.deployment_name}")
                logger.error(f"Azure OpenAI configured: {bool(settings.azure_openai.api_key)}")
                logger.error(f"Endpoint: {settings.azure_openai.endpoint}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.error(f"=== END ERROR ===")
                return ""

        return await _call_openai()

    def _extract_dialogue_fallback(
        self, raw_chunk: str, logger, language: str = "en"
    ) -> Optional[List[Dict[str, str]]]:
        """Fallback method to extract dialogue from raw chunk when LLM processing fails."""
        if not raw_chunk or not raw_chunk.strip():
            return None

        try:
            # Simple heuristic-based extraction
            # Split by sentences and alternate between Doctor and Patient based on patterns
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_chunk) if s.strip()]

            if not sentences:
                return None

            dialogue = []
            patient_label = (
                "Paciente" if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"] else "Patient"
            )
            next_speaker = "Doctor"

            for sentence in sentences:
                sentence_lower = sentence.lower()

                # Doctor signals
                is_doctor = (
                    sentence.endswith("?")  # Questions
                    or sentence.startswith(
                        (
                            "Let me",
                            "I'll",
                            "We'll",
                            "I'm going to",
                            "Déjame",
                            "Voy a",
                            "Vamos a",
                        )
                    )
                    or any(
                        word in sentence_lower
                        for word in [
                            "recommend",
                            "prescribe",
                            "order",
                            "examine",
                            "check",
                            "recomiendo",
                            "prescribir",
                            "ordenar",
                            "examinar",
                            "revisar",
                        ]
                    )
                    or any(
                        pattern in sentence_lower
                        for pattern in [
                            "can you",
                            "do you",
                            "have you",
                            "¿puedes",
                            "¿tienes",
                        ]
                    )
                )

                # Patient signals
                is_patient = (
                    sentence.startswith(("I", "I've", "I'm", "I have", "Yo", "Tengo", "He estado"))
                    or any(word in sentence_lower for word in ["yes", "no", "okay", "sí", "no", "bien"])
                    or any(pattern in sentence_lower for pattern in ["it hurts", "i feel", "me duele", "siento"])
                )

                # Determine speaker
                if is_doctor and not is_patient:
                    speaker = "Doctor"
                    next_speaker = patient_label
                elif is_patient and not is_doctor:
                    speaker = patient_label
                    next_speaker = "Doctor"
                else:
                    # Use context-based assignment
                    speaker = next_speaker
                    next_speaker = patient_label if next_speaker == "Doctor" else "Doctor"

                dialogue.append({speaker: sentence})

            if dialogue:
                logger.info(
                    f"Fallback extraction created {len(dialogue)} dialogue turns from {len(sentences)} sentences"
                )
                return dialogue

        except Exception as e:
            logger.error(f"Fallback extraction error: {e}")

        return None

    def _merge_chunk_results(self, chunk_results: list, logger) -> list:
        """Merge chunk results and remove overlapping content with enhanced detection."""

        if not chunk_results:
            return []

        merged = []

        for i, chunk in enumerate(chunk_results):

            if not chunk or not isinstance(chunk, list) or len(chunk) == 0:
                logger.warning(f"Chunk {i} is empty or invalid, skipping")
                continue

            # If merged is empty, just add the chunk
            if not merged:
                merged.extend(chunk)
                logger.info(f"Added first chunk with {len(chunk)} turns")
                continue

            # Enhanced overlap detection: exact match + semantic similarity
            last_merged = merged[-1] if merged else None
            first_chunk = chunk[0] if chunk else None

            # Strategy 1: Exact match (highest confidence)
            if (
                isinstance(last_merged, dict)
                and isinstance(first_chunk, dict)
                and len(last_merged) == 1
                and len(first_chunk) == 1
                and list(last_merged.keys())[0] == list(first_chunk.keys())[0]
                and list(last_merged.values())[0] == list(first_chunk.values())[0]
            ):

                # Skip first item in current chunk to avoid duplication
                merged.extend(chunk[1:])
                logger.info(
                    f"Chunk {i}: Exact match detected - skipped duplicate first turn, added {len(chunk)-1} turns"
                )
                continue

            # Strategy 2: Check for partial overlap by comparing last 2-3 turns
            overlap_found = False
            max_overlap_size = min(4, len(chunk), len(merged))

            # Try exact match first (most reliable)
            for overlap_size in range(1, max_overlap_size + 1):
                if len(merged) >= overlap_size:
                    merged_tail = merged[-overlap_size:]
                    chunk_head = chunk[:overlap_size]

                    # Exact match
                    if merged_tail == chunk_head:
                        merged.extend(chunk[overlap_size:])
                        logger.info(
                            f"Chunk {i}: Found {overlap_size}-turn exact overlap, added {len(chunk)-overlap_size} turns"
                        )
                        overlap_found = True
                        break

                    # Semantic similarity check (fuzzy match)
                    # Compare similarity of text content (not exact dict match)
                    similarity_score = self._calculate_similarity(merged_tail, chunk_head)
                    if similarity_score > 0.85:  # 85% similarity threshold
                        merged.extend(chunk[overlap_size:])
                        logger.info(
                            f"Chunk {i}: Found {overlap_size}-turn semantic overlap (similarity: {similarity_score:.2f}), added {len(chunk)-overlap_size} turns"
                        )
                        overlap_found = True
                        break

            if not overlap_found:
                merged.extend(chunk)
                logger.info(f"Chunk {i}: No overlap found, added {len(chunk)} turns")

        logger.info(f"Merged {len(chunk_results)} chunks into {len(merged)} dialogue turns")
        return merged

    def _calculate_similarity(self, turns1: List[Dict[str, str]], turns2: List[Dict[str, str]]) -> float:
        """Calculate semantic similarity between two turn sequences."""
        if len(turns1) != len(turns2):
            return 0.0

        if not turns1:
            return 1.0

        matches = 0
        total = len(turns1)

        for turn1, turn2 in zip(turns1, turns2):
            if not (isinstance(turn1, dict) and isinstance(turn2, dict)):
                continue

            speaker1 = list(turn1.keys())[0] if turn1 else None
            speaker2 = list(turn2.keys())[0] if turn2 else None

            text1 = list(turn1.values())[0] if turn1 else ""
            text2 = list(turn2.values())[0] if turn2 else ""

            # Speaker must match
            if speaker1 != speaker2:
                continue

            # Calculate text similarity (simple word overlap)
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 and not words2:
                matches += 1
            elif words1 and words2:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                similarity = intersection / union if union > 0 else 0.0

                # Consider it a match if similarity > 0.8
                if similarity > 0.8:
                    matches += 1

        return matches / total if total > 0 else 0.0

    async def _process_transcript_simple(
        self,
        client: AzureAIClient,
        raw_transcript: str,
        settings,
        logger,
        language: str = "en",
    ) -> str:
        """Process transcript with simplified approach for very short transcripts (<5000 chars)."""

        # Simplified system prompt for faster processing
        if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
            system_prompt = (
                "Convierte esta transcripción médica en diálogo Doctor-Paciente. "
                "Elimina información personal. Devuelve SOLO un JSON array con turnos de diálogo. "
                'Formato: [{"Doctor": "texto"}, {"Paciente": "texto"}]'
            )
        else:
            system_prompt = (
                "Convert this medical transcript into Doctor-Patient dialogue. "
                "Remove personal information. Return ONLY a JSON array with dialogue turns. "
                'Format: [{"Doctor": "text"}, {"Patient": "text"}]'
            )

        # For transcripts above a configurable threshold, use robust chunking strategy instead
        from clinicai.core.config import get_settings

        settings_obj = get_settings()
        try:
            simple_max_chars = int(settings_obj.transcription_simple_max_chars)
        except (TypeError, ValueError):
            simple_max_chars = 20000

        if len(raw_transcript) > simple_max_chars:
            logger.info(
                f"Transcript length {len(raw_transcript)} exceeds simple_max_chars={simple_max_chars}, "
                "redirecting to chunking strategy"
            )
            return await self._process_transcript_with_chunking(client, raw_transcript, settings, logger, language)

        if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
            user_prompt = f"TRANSCRIPCIÓN: {raw_transcript}\n\nConvierte a diálogo Doctor-Paciente en formato JSON."
        else:
            user_prompt = f"TRANSCRIPT: {raw_transcript}\n\nConvert to Doctor-Patient dialogue in JSON format."

        try:
            logger.info("Starting simplified LLM processing...")
            result = await self._process_single_chunk(client, system_prompt, user_prompt, settings, logger)
            logger.info(f"Simplified processing completed: {len(result) if result else 0} characters")
            return result
        except Exception as e:
            logger.error(f"Simplified processing failed: {e}")
            return ""
