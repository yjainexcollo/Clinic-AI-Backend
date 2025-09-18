"""Transcribe audio use case for Step-03 functionality."""

from ...domain.errors import PatientNotFoundError, VisitNotFoundError
from ...domain.value_objects.patient_id import PatientId
from ..dto.patient_dto import AudioTranscriptionRequest, AudioTranscriptionResponse
from ..ports.repositories.patient_repo import PatientRepository
from ..ports.services.transcription_service import TranscriptionService


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
            transcription_result = await self._transcription_service.transcribe_audio(
                request.audio_file_path,
                medical_context=True
            )

            # Complete transcription
            visit.complete_transcription(
                transcript=transcription_result["transcript"],
                audio_duration=transcription_result.get("duration")
            )

            # Save updated visit
            await self._patient_repository.save(patient)

            return AudioTranscriptionResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                transcript=transcription_result["transcript"],
                word_count=transcription_result.get("word_count", 0),
                audio_duration=transcription_result.get("duration"),
                transcription_status=visit.transcription_session.transcription_status,
                message="Audio transcribed successfully"
            )

        except Exception as e:
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
