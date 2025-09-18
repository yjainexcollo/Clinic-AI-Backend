"""Transcribe audio use case for Step-03 functionality."""

from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId
from ..dto.patient_dto import AudioTranscriptionRequest, AudioTranscriptionResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.services.transcription_service import TranscriptionService
from ...core.config import get_settings
from typing import Dict, Any
import asyncio
import logging

from openai import OpenAI  # type: ignore


class TranscribeAudioUseCase:
    """Use case for transcribing audio files."""

    def __init__(
        self, 
        patient_repository: PatientRepository, 
        transcription_service: TranscriptionService
    ):
        self._patient_repository = patient_repository
        self._transcription_service = transcription_service

    async def execute(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        """Execute the audio transcription use case."""
        # Find patient
        patient_id = PatientId(request.patient_id)
        patient = await self._patient_repository.find_by_id(patient_id)
        if not patient:
            raise PatientNotFoundError(request.patient_id)

        # Find visit
        visit = patient.get_visit_by_id(request.visit_id)
        if not visit:
            raise VisitNotFoundError(request.visit_id)

        # Check if visit is ready for transcription
        if not visit.can_start_transcription():
            raise ValueError(f"Visit not ready for transcription. Current status: {visit.status}")

        try:
            # Validate audio file
            validation_result = await self._transcription_service.validate_audio_file(
                request.audio_file_path
            )
            
            if not validation_result.get("is_valid", False):
                raise ValueError(f"Invalid audio file: {validation_result.get('error', 'Unknown error')}")

            # Start transcription process
            visit.start_transcription(request.audio_file_path)
            await self._patient_repository.save(patient)

            # Transcribe audio
            logger = logging.getLogger("clinicai")
            logger.info(f"Starting Whisper transcription for file: {request.audio_file_path}")
            
            transcription_result = await self._transcription_service.transcribe_audio(
                request.audio_file_path,
                medical_context=True
            )

            raw_transcript = transcription_result.get("transcript", "") or ""
            logger.info(f"Whisper transcription completed. Transcript length: {len(raw_transcript)} characters")

            # Post-process with LLM to clean PII and structure Doctor/Patient dialogue
            logger.info("Starting LLM processing for transcript cleaning and structuring")
            settings = get_settings()
            client = OpenAI(api_key=settings.openai.api_key)

            system_prompt = (
                "You are an AI assistant processing clinical consultation transcripts.\n"
                "Goal: produce highly accurate speaker-attributed dialogue.\n"
                "STRICT RULES:\n"
                "- Remove personal identifiers (names, phone numbers, addresses).\n"
                "- Correct obvious transcription errors (spelling, spacing, casing) without changing meaning.\n"
                "- Attribute each utterance to the correct speaker: Doctor vs Patient.\n"
                "- Use medical context cues to determine speaker.\n"
                "- Output valid JSON ONLY, with alternating keys 'Doctor' and 'Patient' where possible.\n"
                "- If the same speaker talks twice in a row, still output two consecutive keys (do not invent turns).\n"
                "- Do not include any commentary or Markdown. JSON only."
            )
            user_prompt = (
                "Use these diarization heuristics in order of priority:\n"
                "1) The Doctor typically greets, asks questions, gives instructions, summarizes, or explains plans.\n"
                "2) The Patient typically reports symptoms, answers, describes history, denies symptoms, or asks for help.\n"
                "3) Question-like sentences (who/what/when/where/why/how, or ending with '?') are usually Doctor unless clearly the Patient asking.\n"
                "4) Phrases like 'I prescribe', 'Let's order', 'We will do', 'Follow up', 'Take this', 'I'll examine' => Doctor.\n"
                "5) Phrases like 'I feel', 'I have', 'It started', 'My pain', 'Since yesterday', 'I took' => Patient.\n"
                "6) If ambiguous, prefer continuity with previous speaker unless a question/answer pattern indicates a switch.\n"
                "7) Keep the original order of utterances.\n\n"
                "Few-shot examples (not part of output):\n"
                "RAW: 'Hi, I'm Dr. Smith. How can I help you today?' → {\"Doctor\": \"Hello, how can I help you today?\"}\n"
                "RAW: 'I've had a cough for three days.' → {\"Patient\": \"I have had a cough for three days.\"}\n"
                "RAW: 'Do you have fever or shortness of breath?' → {\"Doctor\": \"Do you have a fever or shortness of breath?\"}\n"
                "RAW: 'Yes, fever since yesterday.' → {\"Patient\": \"Yes, fever since yesterday.\"}\n\n"
                "OUTPUT FORMAT (single JSON object; keys repeat as turns):\n"
                "{\n  \"Doctor\": \"...\",\n  \"Patient\": \"...\",\n  \"Doctor\": \"...\",\n  \"Patient\": \"...\"\n}\n\n"
                "Transcript to process (clean PII and structure; JSON only):\n" + raw_transcript
            )

            def _call_openai() -> str:
                try:
                    resp = client.chat.completions.create(
                        model=settings.openai.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=min(8000, settings.openai.max_tokens),
                        temperature=0.0,
                        top_p=0.9,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    logger = logging.getLogger("clinicai")
                    logger.error(f"OpenAI LLM processing failed: {str(e)}")
                    # Return the raw transcript if LLM processing fails
                    return raw_transcript

            structured_content = await asyncio.to_thread(_call_openai)
            logger.info(f"LLM processing completed. Structured content length: {len(structured_content)} characters")

            # Prefer storing the structured JSON text in place of raw transcript
            cleaned_transcript_text = structured_content or raw_transcript

            # Complete transcription
            visit.complete_transcription(
                transcript=cleaned_transcript_text,
                audio_duration=transcription_result.get("duration")
            )

            # Save updated visit
            await self._patient_repository.save(patient)
            logger.info(f"Transcription completed successfully for patient {patient.patient_id.value}, visit {visit.visit_id.value}")

            return AudioTranscriptionResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                transcript=cleaned_transcript_text,
                word_count=transcription_result.get("word_count", 0),
                audio_duration=transcription_result.get("duration"),
                transcription_status=visit.transcription_session.transcription_status,
                message="Audio transcribed successfully"
            )

        except Exception as e:
            # Log the full error for debugging
            logger = logging.getLogger("clinicai")
            logger.error(f"Transcription failed for patient {patient.patient_id.value}, visit {visit.visit_id.value}: {str(e)}", exc_info=True)
            
            # Mark transcription as failed
            visit.fail_transcription(str(e))
            await self._patient_repository.save(patient)
            
            return AudioTranscriptionResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                transcript="",
                word_count=0,
                audio_duration=None,
                transcription_status="failed",
                message=f"Transcription failed: {str(e)}"
            )
