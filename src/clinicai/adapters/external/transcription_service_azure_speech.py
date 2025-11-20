"""
Azure Speech Service batch transcription with speaker diarization.
Uses REST API for batch transcription (more reliable than SDK).
"""

import os
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

import aiohttp

from clinicai.application.ports.services.transcription_service import TranscriptionService
from clinicai.core.config import get_settings

logger = logging.getLogger(__name__)


class AzureSpeechTranscriptionService(TranscriptionService):
    """
    Azure Speech Service transcription with speaker diarization.
    Uses batch transcription REST API for better accuracy and cost efficiency.
    """
    
    def __init__(self) -> None:
        self._settings = get_settings()
        
        # Validate Azure Speech Service configuration
        if not self._settings.azure_speech.subscription_key:
            raise ValueError(
                "Azure Speech Service subscription key is required. "
                "Please set AZURE_SPEECH_SUBSCRIPTION_KEY environment variable."
            )
        
        if not self._settings.azure_speech.region and not self._settings.azure_speech.endpoint:
            raise ValueError(
                "Azure Speech Service region is required unless AZURE_SPEECH_ENDPOINT is provided. "
                "Please set AZURE_SPEECH_REGION environment variable (e.g., 'eastus', 'westus2')."
            )
        
        # Build endpoint (use explicit override if provided)
        self._endpoint = (
            self._settings.azure_speech.endpoint
            or f"https://{self._settings.azure_speech.region}.api.cognitive.microsoft.com"
        )
        self._subscription_key = self._settings.azure_speech.subscription_key
        
        logger.info(
            "✅ Azure Speech Service initialized (endpoint: %s, mode: %s)",
            self._endpoint,
            self._settings.azure_speech.transcription_mode,
        )
    
    async def transcribe_audio(
        self,
        audio_file_path: str,
        language: str = "en",
        medical_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Azure Speech Service batch transcription with speaker diarization.
        
        Returns:
            Dict containing:
            - transcript: Full transcript text
            - structured_dialogue: List of dialogue turns with speaker labels (Speaker 1, Speaker 2)
            - speaker_labels: Raw speaker labels from Azure
            - confidence: Average confidence score
            - duration: Audio duration in seconds
            - word_count: Number of words
            - language: Language code
            - model: Service name
        """
        if self._settings.azure_speech.transcription_mode == "batch":
            return await self._transcribe_batch(audio_file_path, language, medical_context)
        else:
            raise ValueError("Real-time transcription not supported. Use batch mode.")
    
    async def _transcribe_batch(
        self,
        audio_file_path: str,
        language: str,
        medical_context: bool,
    ) -> Dict[str, Any]:
        """Batch transcription with speaker diarization using REST API."""
        try:
            # Map language codes
            language_map = {
                "en": "en-US",
                "sp": "es-ES",
                "es": "es-ES",
            }
            speech_language = language_map.get(language, "en-US")
            
            logger.info(f"Starting Azure Speech Service batch transcription: {audio_file_path}, language: {speech_language}")
            
            # Read audio file
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Upload audio to Azure Blob Storage and get SAS URL
            blob_url = await self._upload_audio_to_blob(audio_file_path, audio_data)
            
            # Create transcription job
            transcription_id = await self._create_transcription_job(blob_url, speech_language)
            
            logger.info(f"Batch transcription job created: {transcription_id}")
            
            # Poll for completion
            transcription_status = await self._poll_transcription_status(transcription_id)
            
            if transcription_status["status"] != "Succeeded":
                error_msg = f"Transcription failed with status: {transcription_status['status']}"
                if transcription_status.get("error"):
                    error_msg += f", error: {transcription_status['error']}"
                raise ValueError(error_msg)
            
            # Get transcription results
            results = await self._get_transcription_results(transcription_id)
            logger.info(f"Retrieved {len(results)} result files from transcription job")
            
            if not results:
                logger.error("No transcription results returned from Azure Speech Service")
                raise ValueError("No transcription results returned from Azure Speech Service")
            
            # Log first result structure for debugging
            print("🔵 === Azure Speech Service Results Debug ===")
            print(f"🔵 Number of result files: {len(results)}")
            if results:
                print(f"🔵 First result keys: {list(results[0].keys())}")
                logger.info(f"First result keys: {list(results[0].keys())}")
                if "recognizedPhrases" in results[0]:
                    print(f"🔵 Found {len(results[0]['recognizedPhrases'])} recognized phrases in first result")
                    logger.info(f"Found {len(results[0]['recognizedPhrases'])} recognized phrases in first result")
                    if results[0]['recognizedPhrases']:
                        print(f"🔵 First phrase keys: {list(results[0]['recognizedPhrases'][0].keys())}")
                        print(f"🔵 First phrase sample: {str(results[0]['recognizedPhrases'][0])[:200]}")
                if "combinedRecognizedPhrases" in results[0]:
                    print(f"🔵 Found {len(results[0]['combinedRecognizedPhrases'])} combined recognized phrases")
                    logger.info(f"Found {len(results[0]['combinedRecognizedPhrases'])} combined recognized phrases")
            
            # Process results to extract transcript and speaker information
            transcript_text, structured_dialogue, speaker_info = self._process_transcription_results(results)
            print("🔵 === Processing Complete ===")
            print(f"🔵 Transcript length: {len(transcript_text)} characters")
            print(f"🔵 Structured dialogue turns: {len(structured_dialogue)}")
            print(f"🔵 Speaker info: {speaker_info}")
            logger.info(f"Processed transcript: {len(transcript_text)} characters, {len(structured_dialogue)} dialogue turns")
            
            # Clean up transcription job
            try:
                await self._delete_transcription_job(transcription_id)
                logger.info(f"Cleaned up transcription job: {transcription_id}")
            except Exception as e:
                logger.warning(f"Failed to delete transcription job {transcription_id}: {e}")
            
            return {
                "transcript": transcript_text,
                "structured_dialogue": structured_dialogue,  # Pre-structured with speaker labels
                "speaker_labels": speaker_info,
                "confidence": self._calculate_average_confidence(results),
                "duration": self._extract_duration(results),
                "word_count": len(transcript_text.split()) if transcript_text else 0,
                "language": language,
                "model": "azure-speech-batch",
            }
            
        except Exception as e:
            logger.error(f"Azure Speech Service batch transcription failed: {e}", exc_info=True)
            raise ValueError(f"Transcription failed: {str(e)}")
    
    async def _upload_audio_to_blob(self, audio_file_path: str, audio_data: bytes) -> str:
        """Upload audio file to Azure Blob Storage and return SAS URL."""
        try:
            from clinicai.adapters.storage.azure_blob_service import get_azure_blob_service
            
            blob_service = get_azure_blob_service()
            
            # Determine content type from file extension
            content_type_map = {
                ".mp3": "audio/mpeg",
                ".mpeg": "audio/mpeg",
                ".mpg": "audio/mpeg",
                ".wav": "audio/wav",
                ".m4a": "audio/mp4",
                ".mp4": "audio/mp4",
                ".flac": "audio/flac",
                ".ogg": "audio/ogg",
                ".opus": "audio/opus",
                ".amr": "audio/amr",
                ".webm": "audio/webm",
                ".aac": "audio/aac",
                ".wma": "audio/x-ms-wma",
            }
            file_ext = Path(audio_file_path).suffix.lower()
            content_type = content_type_map.get(file_ext, "audio/mpeg")
            
            # Upload to blob
            upload_result = await blob_service.upload_file(
                file_data=audio_data,
                filename=Path(audio_file_path).name,
                content_type=content_type,
                file_type="audio"
            )
            
            # Generate SAS URL (signed URL) for Azure Speech Service
            # Azure Speech Service needs a publicly accessible URL
            blob_path = upload_result["blob_path"]
            sas_url = blob_service.generate_signed_url(
                blob_path=blob_path,
                expires_in_hours=24,  # 24 hours should be enough for transcription
            )
            
            logger.info(f"Uploaded audio to blob storage for transcription: {blob_path}")
            return sas_url
            
        except Exception as e:
            logger.error(f"Failed to upload audio for transcription: {e}", exc_info=True)
            raise
    
    async def _create_transcription_job(self, blob_url: str, language: str) -> str:
        """Create Azure Speech transcription job with retry logic."""
        transcription_name = f"clinicai-transcription-{uuid.uuid4()}"
        
        payload = {
            "contentUrls": [blob_url],
            "locale": language,
            "displayName": transcription_name,
            "properties": {
                "diarizationEnabled": self._settings.azure_speech.enable_speaker_diarization,
                "wordLevelTimestampsEnabled": True,
                "punctuationMode": "DictatedAndAutomatic",
                "profanityFilterMode": "Masked"
            }
        }
        
        headers = {
            "Ocp-Apim-Subscription-Key": self._subscription_key,
            "Content-Type": "application/json"
        }
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    url = f"{self._endpoint}/speechtotext/v3.1/transcriptions"
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status not in (200, 201, 202):
                            error_text = await response.text()
                            logger.error(
                                f"Failed to create transcription job: {response.status} {error_text} "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                await asyncio.sleep(delay)
                                continue
                            raise ValueError(f"Failed to create transcription job: {response.status} {error_text}")
                        
                        location = response.headers.get("Location")
                        if not location:
                            raise ValueError("Azure Speech Service did not return transcription location URL")
                        
                        transcription_id = location.rstrip("/").split("/")[-1]
                        logger.info(f"✅ Created transcription job: {transcription_id} (attempt {attempt + 1})")
                        return transcription_id
                        
            except asyncio.TimeoutError:
                logger.error(f"Timeout creating transcription job (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise TimeoutError("Failed to create transcription job: timeout after 3 attempts")
                
            except Exception as e:
                logger.error(f"Error creating transcription job (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
    
    async def _poll_transcription_status(self, transcription_id: str) -> Dict[str, Any]:
        """Poll transcription job status until completion with increased timeout."""
        status_url = f"{self._endpoint}/speechtotext/v3.1/transcriptions/{transcription_id}"
        headers = {"Ocp-Apim-Subscription-Key": self._subscription_key}
        poll_interval = max(3, self._settings.azure_speech.batch_polling_interval)  # Minimum 3 seconds
        timeout_seconds = self._settings.azure_speech.batch_max_wait_time
        start_time = time.time()
        poll_count = 0
        
        logger.info(f"Starting status polling for transcription: {transcription_id}, poll_interval={poll_interval}s")
        
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout per request (increased from 30s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                poll_count += 1
                try:
                    async with session.get(status_url, headers=headers) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Failed to get transcription status: {response.status} {error_text}")
                            raise ValueError(f"Failed to get transcription status: {response.status} {error_text}")
                        
                        status_data = await response.json()
                        status = status_data.get("status")
                        
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Transcription status poll #{poll_count}: status={status}, "
                            f"elapsed={elapsed:.1f}s/{timeout_seconds}s"
                        )
                        
                        if status in ("Succeeded", "Failed"):
                            total_duration = time.time() - start_time
                            logger.info(
                                f"✅ Transcription job completed: {transcription_id}, "
                                f"status={status}, total_duration={total_duration:.2f}s, "
                                f"polls={poll_count}"
                            )
                            return status_data
                        
                        if time.time() - start_time > timeout_seconds:
                            raise TimeoutError(
                                f"Transcription job timed out after {timeout_seconds} seconds. "
                                f"Last status: {status}, polls: {poll_count}"
                            )
                        
                        await asyncio.sleep(poll_interval)
                        
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.warning(
                        f"Timeout during status poll #{poll_count} for {transcription_id} "
                        f"(elapsed: {elapsed:.1f}s)"
                    )
                    # Continue polling unless overall timeout exceeded
                    if time.time() - start_time > timeout_seconds:
                        raise TimeoutError(
                            f"Transcription job timed out after {timeout_seconds} seconds. "
                            f"Last poll timed out."
                        )
                    await asyncio.sleep(poll_interval)
                    continue
    
    async def _get_transcription_results(self, transcription_id: str) -> List[Dict[str, Any]]:
        """Retrieve transcription results."""
        files_url = f"{self._endpoint}/speechtotext/v3.1/transcriptions/{transcription_id}/files"
        headers = {"Ocp-Apim-Subscription-Key": self._subscription_key}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(files_url, headers=headers, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to get transcription files: {response.status} {error_text}")
                    raise ValueError(f"Failed to get transcription files: {response.status} {error_text}")
                
                files_data = await response.json()
                result_files = files_data.get("values", [])
                
                transcripts = []
                for file_info in result_files:
                    if file_info.get("kind") != "Transcription":
                        continue
                    
                    content_url = file_info.get("links", {}).get("contentUrl")
                    if not content_url:
                        continue
                    
                    async with session.get(content_url, timeout=60) as content_response:
                        if content_response.status != 200:
                            logger.warning(f"Failed to download transcription result: {content_response.status}")
                            continue
                        
                        transcript_json = await content_response.json()
                        transcripts.append(transcript_json)
                
                return transcripts
    
    def _process_transcription_results(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Process Azure Speech transcription results into structured dialogue."""
        transcript_text = []
        structured_dialogue = []
        speaker_info = {"speakers": []}
        seen_speakers = set()
        
        for result in results:
            recognized_phrases = result.get("recognizedPhrases", [])
            
            for phrase in recognized_phrases:
                nbest = phrase.get("nBest", [])
                if not nbest:
                    continue
                
                best_result = nbest[0]
                text = best_result.get("display") or best_result.get("lexical") or ""
                if not text.strip():
                    continue
                
                transcript_text.append(text)
                
                # Prefer diarization speaker ID; fallback to channel if missing
                speaker_id = phrase.get("speaker")
                if speaker_id is None:
                    channel = phrase.get("channel", 0)
                    speaker_id = channel + 1
                
                speaker_label = f"Speaker {speaker_id}"
                seen_speakers.add(speaker_label)
                
                structured_dialogue.append(
                    {
                        "speaker": speaker_label,
                        "text": text,
                        "offset": phrase.get("offset"),
                        "duration": phrase.get("duration"),
                    }
                )
        
        if seen_speakers:
            speaker_info["speakers"] = [{"label": label} for label in sorted(seen_speakers)]
        else:
            speaker_info["speakers"] = [{"label": "Speaker 1"}]
        
        return " ".join(transcript_text).strip(), structured_dialogue, speaker_info
    
    def _calculate_average_confidence(self, results: List[Dict[str, Any]]) -> float:
        confidences = []
        for result in results:
            recognized_phrases = result.get("recognizedPhrases", [])
            for phrase in recognized_phrases:
                nbest = phrase.get("nBest", [])
                if nbest:
                    confidences.append(nbest[0].get("confidence", 0.0))
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
    
    def _extract_duration(self, results: List[Dict[str, Any]]) -> float:
        durations = []
        for result in results:
            recognized_phrases = result.get("recognizedPhrases", [])
            for phrase in recognized_phrases:
                duration_text = phrase.get("duration")
                if duration_text:
                    try:
                        if duration_text.startswith("PT") and duration_text.endswith("S"):
                            duration_seconds = float(duration_text[2:-1])
                            durations.append(duration_seconds)
                    except ValueError:
                        continue
        return max(durations) if durations else 0.0
    
    async def _delete_transcription_job(self, transcription_id: str) -> None:
        """Delete transcription job to clean up resources."""
        delete_url = f"{self._endpoint}/speechtotext/v3.1/transcriptions/{transcription_id}"
        headers = {"Ocp-Apim-Subscription-Key": self._subscription_key}
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(delete_url, headers=headers, timeout=30) as response:
                if response.status not in (200, 202, 204):
                    logger.warning(f"Failed to delete transcription job: {response.status}")
    
    async def validate_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Validate audio file format and quality for Azure Speech Service.
        
        Azure Speech Service supports the following formats:
        - WAV (PCM, A-law, mu-law)
        - MP3 (MPEG-1/2 Audio Layer 3)
        - M4A (MPEG-4 Audio)
        - FLAC (Free Lossless Audio Codec)
        - OGG (Ogg Vorbis)
        - OPUS (Opus codec)
        - AMR (Adaptive Multi-Rate)
        - WebM (WebM audio)
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dict containing validation results and metadata:
            - is_valid: bool
            - error: Optional error message
            - file_size: File size in bytes
            - duration: Audio duration in seconds (0 if unknown)
            - format: File format/extension
        """
        try:
            p = Path(audio_file_path)
            if not p.exists() or not p.is_file():
                return {
                    "is_valid": False,
                    "error": "Audio file not found",
                    "file_size": 0,
                    "duration": 0,
                    "format": None,
                }
            
            file_size = p.stat().st_size
            if file_size <= 0:
                return {
                    "is_valid": False,
                    "error": "Empty file",
                    "file_size": 0,
                    "duration": 0,
                    "format": None,
                }
            
            # Azure Speech Service supports files up to 1GB, but we'll use a reasonable limit
            # Check against configured max file size if available
            max_size_mb = (
                self._settings.file_storage.max_file_size_mb 
                if hasattr(self._settings, 'file_storage') and hasattr(self._settings.file_storage, 'max_file_size_mb')
                else 100
            )
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                return {
                    "is_valid": False,
                    "error": f"Audio file too large ({file_size / (1024*1024):.1f}MB, max {max_size_mb}MB)",
                    "file_size": file_size,
                    "duration": 0,
                    "format": p.suffix.lower().lstrip("."),
                }
            
            # Validate file format - Azure Speech Service supported formats
            # Including MPEG variants and other common formats
            supported_formats = {
                ".wav",      # WAV (PCM, A-law, mu-law)
                ".mp3",      # MP3 (MPEG-1/2 Audio Layer 3)
                ".mpeg",     # MPEG audio
                ".mpg",      # MPEG audio (alternative extension)
                ".m4a",      # M4A (MPEG-4 Audio)
                ".mp4",      # MP4 (can contain audio)
                ".flac",     # FLAC (Free Lossless Audio Codec)
                ".ogg",      # OGG (Ogg Vorbis)
                ".opus",     # OPUS (Opus codec)
                ".amr",      # AMR (Adaptive Multi-Rate)
                ".webm",     # WebM audio
                ".aac",      # AAC (Advanced Audio Coding)
                ".wma",      # WMA (Windows Media Audio) - if supported
            }
            file_ext = p.suffix.lower()
            
            if file_ext not in supported_formats:
                return {
                    "is_valid": False,
                    "error": f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(sorted(supported_formats))}",
                    "file_size": file_size,
                    "duration": 0,
                    "format": file_ext.lstrip("."),
                }
            
            return {
                "is_valid": True,
                "error": None,
                "file_size": file_size,
                "duration": 0,  # Duration would require audio processing library to determine
                "format": file_ext.lstrip("."),
            }
        except Exception as e:
            logger.error(f"Error validating audio file {audio_file_path}: {e}", exc_info=True)
            return {
                "is_valid": False,
                "error": f"Validation error: {e}",
                "file_size": 0,
                "duration": 0,
                "format": None,
            }