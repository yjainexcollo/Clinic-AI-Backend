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
        
        if not self._settings.azure_speech.region:
            raise ValueError(
                "Azure Speech Service region is required. "
                "Please set AZURE_SPEECH_REGION environment variable (e.g., 'eastus', 'westus2')."
            )
        
        # Build endpoint
        self._endpoint = f"https://{self._settings.azure_speech.region}.api.cognitive.microsoft.com"
        self._subscription_key = self._settings.azure_speech.subscription_key
        
        logger.info(f"✅ Azure Speech Service initialized (region: {self._settings.azure_speech.region}, mode: {self._settings.azure_speech.transcription_mode})")
    
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
                ".wav": "audio/wav",
                ".m4a": "audio/mp4",
                ".flac": "audio/flac",
                ".ogg": "audio/ogg",
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
                permissions="r"
            )
            
            logger.info(f"Uploaded audio to blob and generated SAS URL: {blob_path}")
            return sas_url
            
        except Exception as e:
            logger.error(f"Failed to upload audio to blob: {e}", exc_info=True)
            raise ValueError(f"Failed to upload audio file: {str(e)}")
    
    async def _create_transcription_job(self, blob_url: str, locale: str) -> str:
        """Create a batch transcription job."""
        url = f"{self._endpoint}/speechtotext/v3.1/transcriptions"
        
        # Verify diarization settings
        if self._settings.azure_speech.enable_speaker_diarization:
            logger.info(f"✅ Speaker diarization enabled: max_speakers={self._settings.azure_speech.max_speakers}")
            print(f"🔵 Diarization config: enabled=True, max_speakers={self._settings.azure_speech.max_speakers}")
        else:
            logger.warning("⚠️ Speaker diarization is DISABLED in settings!")
            print("⚠️ WARNING: Speaker diarization is disabled!")
        
        # Configure transcription properties
        properties = {
            "diarizationEnabled": self._settings.azure_speech.enable_speaker_diarization,
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "Dictated",
            "profanityFilterMode": "Masked",
        }
        
        if self._settings.azure_speech.enable_speaker_diarization:
            properties["diarization"] = {
                "speakers": {
                    "minCount": 1,
                    "maxCount": self._settings.azure_speech.max_speakers
                }
            }
        
        payload = {
            "contentUrls": [blob_url],
            "locale": locale,
            "displayName": f"transcription_{int(time.time())}",
            "description": "Medical consultation transcription",
            "properties": properties
        }
        
        headers = {
            "Ocp-Apim-Subscription-Key": self._subscription_key,
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    error_json = {}
                    try:
                        error_json = await response.json()
                    except:
                        pass
                    
                    # Check for subscription tier error
                    error_code = error_json.get("error", {}).get("code", "") if error_json.get("error") else ""
                    if "InvalidSubscription" in error_text or error_code == "InvalidSubscription":
                        raise ValueError(
                            "Azure Speech Service batch transcription requires a 'Standard' tier subscription. "
                            "Your current subscription appears to be 'Free' tier. "
                            "Please upgrade your Azure Speech Service subscription to 'Standard' tier in the Azure Portal. "
                            f"Original error: {error_text}"
                        )
                    
                    raise ValueError(f"Failed to create transcription job: {response.status} - {error_text}")
                
                result = await response.json()
                transcription_id = result.get("self", "").split("/")[-1]
                return transcription_id
    
    async def _poll_transcription_status(self, transcription_id: str) -> Dict[str, Any]:
        """Poll transcription status until complete."""
        url = f"{self._endpoint}/speechtotext/v3.1/transcriptions/{transcription_id}"
        headers = {
            "Ocp-Apim-Subscription-Key": self._subscription_key
        }
        
        max_wait_time = self._settings.azure_speech.batch_max_wait_time
        poll_interval = self._settings.azure_speech.batch_polling_interval
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Failed to get transcription status: {response.status} - {error_text}")
                    
                    result = await response.json()
                    status = result.get("status", "Unknown")
                    
                    logger.info(f"Transcription status: {status}")
                    
                    if status in ["Succeeded", "Failed", "Cancelled"]:
                        return {
                            "status": status,
                            "error": result.get("properties", {}).get("error") if status == "Failed" else None
                        }
                    
                    elapsed = time.time() - start_time
                    if elapsed >= max_wait_time:
                        raise TimeoutError(f"Transcription timed out after {max_wait_time} seconds")
                    
                    await asyncio.sleep(poll_interval)
    
    async def _get_transcription_results(self, transcription_id: str) -> List[Dict[str, Any]]:
        """Download and parse transcription results."""
        url = f"{self._endpoint}/speechtotext/v3.1/transcriptions/{transcription_id}/files"
        headers = {
            "Ocp-Apim-Subscription-Key": self._subscription_key
        }
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            # Get list of result files
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Failed to get transcription files: {response.status} - {error_text}")
                
                files_response = await response.json()
                result_files = files_response.get("values", [])
                
                # Download each result file
                for result_file in result_files:
                    if result_file.get("kind") == "Transcription":
                        result_url = result_file.get("links", {}).get("contentUrl")
                        if result_url:
                            async with session.get(result_url) as file_response:
                                if file_response.status == 200:
                                    result_data = await file_response.json()
                                    results.append(result_data)
                                else:
                                    logger.warning(f"Failed to download result from {result_url}: {file_response.status}")
        
        return results
    
    def _process_transcription_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Process Azure Speech Service results to extract:
        1. Full transcript text
        2. Structured dialogue with speaker labels
        3. Speaker information
        """
        if not results:
            logger.warning("No results to process")
            return "", [], {}
        
        # Combine all result files
        all_segments = []
        for i, result in enumerate(results):
            print(f"🔵 Processing result {i+1}/{len(results)}")
            logger.info(f"Processing result {i+1}/{len(results)}")
            if "recognizedPhrases" in result:
                phrases = result["recognizedPhrases"]
                print(f"🔵 Found {len(phrases)} recognized phrases in result {i+1}")
                logger.info(f"Found {len(phrases)} recognized phrases in result {i+1}")
                all_segments.extend(phrases)
            else:
                print(f"⚠️ Result {i+1} does not contain 'recognizedPhrases' key. Keys: {list(result.keys())}")
                logger.warning(f"Result {i+1} does not contain 'recognizedPhrases' key. Keys: {list(result.keys())}")
        
        print(f"🔵 Total segments collected: {len(all_segments)}")
        logger.info(f"Total segments collected: {len(all_segments)}")
        
        # Sort by offset (time)
        all_segments.sort(key=lambda x: x.get("offsetInTicks", 0))
        
        # Prefer recognizedPhrases for speaker diarization (has individual speaker segments)
        # Only use combinedRecognizedPhrases as fallback if recognizedPhrases is empty
        if not all_segments and results and "combinedRecognizedPhrases" in results[0]:
            combined = results[0]["combinedRecognizedPhrases"]
            logger.info(f"Found combinedRecognizedPhrases with {len(combined)} items (using as fallback)")
            if combined:
                # Use combinedRecognizedPhrases as fallback (but we lose speaker separation)
                all_segments = combined
                logger.warning("Using combinedRecognizedPhrases - speaker diarization may not work properly")
        else:
            logger.info(f"Using recognizedPhrases with {len(all_segments)} segments for proper speaker diarization")
        
        # Extract full transcript
        transcript_parts = []
        structured_dialogue = []
        speaker_info = {}
        
        logger.info(f"Processing {len(all_segments)} segments")
        
        # Diagnostic: Check speaker distribution
        speaker_ids_found = set()
        segments_with_speaker = 0
        segments_without_speaker = 0
        
        for i, segment in enumerate(all_segments):
            # Try different possible field names for text
            # Azure Speech Service can return text in different places:
            # 1. Direct "text" field (for combinedRecognizedPhrases)
            # 2. Direct "display" field (alternative format)
            # 3. Inside nBest[0].text (for recognizedPhrases with speaker diarization)
            text = ""
            
            # First try direct fields
            if "text" in segment:
                text = segment["text"]
            elif "display" in segment:
                text = segment["display"]
            # Then try nBest array (most common for speaker diarization)
            elif "nBest" in segment and isinstance(segment["nBest"], list) and len(segment["nBest"]) > 0:
                nbest_item = segment["nBest"][0]
                if isinstance(nbest_item, dict):
                    text = nbest_item.get("text", "") or nbest_item.get("display", "") or nbest_item.get("lexical", "")
            
            # Clean up text
            if isinstance(text, str):
                text = text.strip()
            else:
                text = ""
            
            if not text:
                if i < 3:
                    print(f"⚠️ Segment {i+1} has no text. Keys: {list(segment.keys())}")
                    if "nBest" in segment:
                        print(f"   nBest type: {type(segment['nBest'])}, length: {len(segment['nBest']) if isinstance(segment['nBest'], list) else 'N/A'}")
                        if isinstance(segment['nBest'], list) and len(segment['nBest']) > 0:
                            print(f"   nBest[0] keys: {list(segment['nBest'][0].keys()) if isinstance(segment['nBest'][0], dict) else 'Not a dict'}")
                            print(f"   nBest[0] data: {str(segment['nBest'][0])[:300]}")
                logger.warning(f"Segment {i+1} has no text. Keys: {list(segment.keys())}")
                continue
            else:
                if i < 3:
                    print(f"✅ Segment {i+1} has text: {text[:100]}...")
            
            # Get speaker ID if diarization is enabled
            # Speaker ID can be in different places depending on the result structure
            speaker_id = segment.get("speaker", None) or segment.get("speakerId", None)
            
            # If using nBest format, get speaker from first best
            if speaker_id is None and "nBest" in segment and len(segment["nBest"]) > 0:
                speaker_id = segment["nBest"][0].get("speaker", None)
            
            # Diagnostic tracking
            if speaker_id is not None:
                speaker_ids_found.add(speaker_id)
                segments_with_speaker += 1
            else:
                segments_without_speaker += 1
                if i < 5:  # Log first few segments without speaker
                    logger.warning(f"Segment {i+1} has no speaker ID. Segment keys: {list(segment.keys())}")
                    if "nBest" in segment and isinstance(segment.get("nBest"), list) and len(segment.get("nBest", [])) > 0:
                        logger.warning(f"  nBest[0] keys: {list(segment['nBest'][0].keys()) if isinstance(segment['nBest'][0], dict) else 'N/A'}")
            
            # Build transcript
            transcript_parts.append(text)
            
            # Build structured dialogue
            if speaker_id is not None:
                # Map speaker ID to label (will be mapped to Doctor/Patient later)
                speaker_label = f"Speaker {speaker_id}"
                
                # Track speaker info
                if speaker_id not in speaker_info:
                    speaker_info[speaker_id] = {
                        "speaker_id": speaker_id,
                        "total_segments": 0,
                        "total_words": 0
                    }
                speaker_info[speaker_id]["total_segments"] += 1
                speaker_info[speaker_id]["total_words"] += len(text.split())
                
                # Add to structured dialogue (will be mapped to Doctor/Patient)
                structured_dialogue.append({
                    speaker_label: text
                })
            else:
                # No speaker info, add as unknown
                structured_dialogue.append({
                    "Unknown": text
                })
        
        full_transcript = " ".join(transcript_parts)
        
        # Log speaker diagnostics
        logger.info(f"🔵 Speaker diarization diagnostics:")
        logger.info(f"   - Unique speakers detected: {len(speaker_ids_found)} ({sorted(speaker_ids_found)})")
        logger.info(f"   - Segments with speaker ID: {segments_with_speaker}")
        logger.info(f"   - Segments without speaker ID: {segments_without_speaker}")
        print(f"🔵 Speaker diarization: {len(speaker_ids_found)} speakers found ({sorted(speaker_ids_found)})")
        if len(speaker_ids_found) == 1:
            logger.warning("⚠️ Only one speaker detected. This may indicate:")
            logger.warning("   1. Audio actually has only one speaker")
            logger.warning("   2. Speaker diarization not working properly")
            logger.warning("   3. Audio quality issues preventing speaker separation")
            print(f"⚠️ Warning: Only one speaker detected in audio")
        elif len(speaker_ids_found) == 0:
            logger.warning("⚠️ No speakers detected - diarization may not be enabled or working")
            print(f"⚠️ Warning: No speakers detected in audio")
        
        return full_transcript, structured_dialogue, speaker_info
    
    def _calculate_average_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average confidence from results."""
        confidences = []
        for result in results:
            if "recognizedPhrases" in result:
                for phrase in result["recognizedPhrases"]:
                    confidence = phrase.get("confidence", 0.0)
                    if confidence > 0:
                        confidences.append(confidence)
        
        if confidences:
            return sum(confidences) / len(confidences)
        return 0.0
    
    def _extract_duration(self, results: List[Dict[str, Any]]) -> Optional[float]:
        """Extract audio duration from results."""
        if not results:
            return None
        
        # Get duration from first result
        duration_ticks = results[0].get("durationInTicks", 0)
        if duration_ticks > 0:
            # Convert ticks to seconds (1 tick = 100 nanoseconds)
            return duration_ticks / 10_000_000
        
        return None
    
    async def _delete_transcription_job(self, transcription_id: str) -> None:
        """Delete transcription job."""
        url = f"{self._endpoint}/speechtotext/v3.1/transcriptions/{transcription_id}"
        headers = {
            "Ocp-Apim-Subscription-Key": self._subscription_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers) as response:
                if response.status not in [200, 204]:
                    error_text = await response.text()
                    logger.warning(f"Failed to delete transcription job: {response.status} - {error_text}")
    
    async def validate_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """Validate audio file format and quality."""
        try:
            p = Path(audio_file_path)
            if not p.exists() or not p.is_file():
                return {"is_valid": False, "error": "Audio file not found", "file_size": 0, "duration": 0}
            
            file_size = p.stat().st_size
            if file_size <= 0:
                return {"is_valid": False, "error": "Empty file", "file_size": 0, "duration": 0}
            
            # Azure Speech Service supports up to 1GB for batch transcription
            max_size = 1024 * 1024 * 1024  # 1GB
            if file_size > max_size:
                return {
                    "is_valid": False,
                    "error": f"File too large ({file_size / (1024*1024):.1f}MB, max 1GB)",
                    "file_size": file_size,
                    "duration": 0
                }
            
            # Check file extension
            # Note: .mpeg/.mpg files should be normalized to .mp3 before reaching here
            valid_extensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"]
            file_ext = p.suffix.lower()
            # Also accept .mpeg/.mpg as they're typically MP3 format
            if file_ext in [".mpeg", ".mpg"]:
                file_ext = ".mp3"
            if file_ext not in valid_extensions:
                return {
                    "is_valid": False,
                    "error": f"Unsupported file format: {p.suffix}. Supported: {', '.join(valid_extensions)}",
                    "file_size": file_size,
                    "duration": 0
                }
            
            return {
                "is_valid": True,
                "file_size": file_size,
                "duration": 0,  # Will be determined during transcription
                "format": p.suffix.lower()
            }
            
        except Exception as e:
            logger.error(f"Error validating audio file: {e}", exc_info=True)
            return {"is_valid": False, "error": str(e), "file_size": 0, "duration": 0}
