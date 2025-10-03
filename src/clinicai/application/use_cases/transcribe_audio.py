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
import json
import re

from openai import OpenAI  # type: ignore

# Module-level logger to avoid UnboundLocalError from function-level assignments
LOGGER = logging.getLogger("clinicai")


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
        LOGGER.info(f"TranscribeAudioUseCase.execute called for patient {request.patient_id}, visit {request.visit_id}")
        
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
            LOGGER.info(f"Starting Whisper transcription for file: {request.audio_file_path}")
            
            transcription_result = await self._transcription_service.transcribe_audio(
                request.audio_file_path,
                language=request.language,
                medical_context=True
            )

            raw_transcript = transcription_result.get("transcript", "") or ""
            LOGGER.info(f"Whisper transcription completed. Transcript length: {len(raw_transcript)} characters")
            LOGGER.info(f"Raw transcript preview: {raw_transcript[:300]}...")
            
            if not raw_transcript or raw_transcript.strip() == "":
                raise ValueError("Whisper transcription returned empty transcript")

            # Post-process with LLM to clean PII and structure Doctor/Patient dialogue
            LOGGER.info("Starting LLM processing for transcript cleaning and structuring")
            settings = get_settings()
            client = OpenAI(api_key=settings.openai.api_key)

            # Process transcript with improved chunking strategy
            LOGGER.info(f"Starting transcript processing for {len(raw_transcript)} characters")
            LOGGER.info(f"Raw transcript preview: {raw_transcript[:200]}...")
            
            # For very long transcripts, use simpler processing to avoid timeouts
            if len(raw_transcript) > 5000:
                LOGGER.info("Long transcript detected, using simplified processing to avoid timeout")
                structured_content = await self._process_transcript_simple(
                    client, raw_transcript, settings, LOGGER, request.language or "en"
                )
            else:
                structured_content = await self._process_transcript_with_chunking(
                    client, raw_transcript, settings, LOGGER, request.language or "en"
                )

            # Process structured dialogue separately from raw transcript
            structured_dialogue = None
            LOGGER.info(f"LLM processing result length: {len(structured_content) if structured_content else 0}")
            
            if structured_content and structured_content != raw_transcript:
                try:
                    parsed = json.loads(structured_content)
                    LOGGER.info(f"Parsed JSON structure: {type(parsed)} with {len(parsed) if isinstance(parsed, list) else 'N/A'} items")
                    
                    if isinstance(parsed, list) and all(
                        isinstance(item, dict) and 
                        len(item) == 1 and 
                        list(item.keys())[0] in ["Doctor", "Patient"]
                        for item in parsed
                    ):
                        structured_dialogue = parsed
                        LOGGER.info(f"Successfully validated structured dialogue with {len(parsed)} turns")
                    else:
                        LOGGER.warning(f"Structured content is not valid dialogue format. Type: {type(parsed)}, Content: {structured_content[:200]}...")
                except json.JSONDecodeError as e:
                    LOGGER.warning(f"Structured content is not valid JSON: {e}. Content: {structured_content[:200]}...")
                except Exception as e:
                    LOGGER.warning(f"Error validating structured content: {e}. Content: {structured_content[:200] if structured_content else 'None'}...")
            
            # Create intelligent fallback structured dialogue if LLM processing failed
            if not structured_dialogue and raw_transcript and raw_transcript.strip():
                try:
                    import re
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_transcript) if s.strip()]
                    fallback_dialogue = []
                    current_speaker = "Doctor"  # Medical consultations typically start with doctor
                    
                    for sentence in sentences:
                        # Simple heuristic for better speaker attribution
                        sentence_lower = sentence.lower()
                        
                        # Doctor indicators
                        if any(indicator in sentence_lower for indicator in [
                            'what brings you', 'how can i help', 'tell me about', 'how long have you',
                            'can you describe', 'do you have', 'are you taking', 'any allergies',
                            'let me examine', 'i recommend', 'you should', 'we need to',
                            'on a scale', 'when did this', 'where does it', 'how often'
                        ]):
                            current_speaker = "Doctor"
                        
                        # Patient indicators
                        elif any(indicator in sentence_lower for indicator in [
                            'i have', 'i feel', 'i think', 'i took', 'i went', 'my pain',
                            'it started', 'it hurts', 'i don\'t', 'i can\'t', 'i\'m worried',
                            'yes', 'no', 'maybe', 'i think so', 'i\'m not sure'
                        ]):
                            current_speaker = "Patient"
                        
                        fallback_dialogue.append({current_speaker: sentence})
                        # Alternate speaker for next turn if no clear indicator
                        if not any(indicator in sentence_lower for indicator in [
                            'what brings you', 'how can i help', 'tell me about', 'how long have you',
                            'can you describe', 'do you have', 'are you taking', 'any allergies',
                            'let me examine', 'i recommend', 'you should', 'we need to',
                            'on a scale', 'when did this', 'where does it', 'how often',
                            'i have', 'i feel', 'i think', 'i took', 'i went', 'my pain',
                            'it started', 'it hurts', 'i don\'t', 'i can\'t', 'i\'m worried',
                            'yes', 'no', 'maybe', 'i think so', 'i\'m not sure'
                        ]):
                            current_speaker = "Patient" if current_speaker == "Doctor" else "Doctor"
                    
                    structured_dialogue = fallback_dialogue
                    LOGGER.info(f"Created intelligent fallback structured dialogue with {len(fallback_dialogue)} turns")
                except Exception as e:
                    LOGGER.warning(f"Failed to create fallback structured dialogue: {e}")

            LOGGER.info(f"Raw transcript length: {len(raw_transcript)} characters")
            LOGGER.info(f"Structured dialogue turns: {len(structured_dialogue) if structured_dialogue else 0}")

            # Complete transcription with both raw transcript and structured dialogue
            visit.complete_transcription(
                transcript=raw_transcript,  # Store raw transcript
                audio_duration=transcription_result.get("duration"),
                structured_dialogue=structured_dialogue  # Store structured dialogue separately
            )

            # Save updated visit
            await self._patient_repository.save(patient)
            LOGGER.info(f"Transcription completed successfully for patient {patient.patient_id.value}, visit {visit.visit_id.value}")

            return AudioTranscriptionResponse(
                patient_id=patient.patient_id.value,
                visit_id=visit.visit_id.value,
                transcript=raw_transcript,  # Return raw transcript
                word_count=transcription_result.get("word_count", 0),
                audio_duration=transcription_result.get("duration"),
                transcription_status=visit.transcription_session.transcription_status,
                message="Audio transcribed successfully"
            )

        except Exception as e:
            # Log the full error for debugging
            LOGGER.error(f"Transcription failed for patient {patient.patient_id.value}, visit {visit.visit_id.value}: {str(e)}", exc_info=True)
            
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

    async def _process_transcript_with_chunking(
        self, 
        client: OpenAI, 
        raw_transcript: str, 
        settings, 
        logger,
        language: str = "en"
    ) -> str:
        """Process transcript with robust chunking strategy for long content."""
        
        # Highly refined system prompt for superior speaker attribution (language-aware)
        if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
            system_prompt = (
                "Eres un analista experto de conversaciones médicas. Tu tarea es convertir una transcripción cruda de una consulta médica en un diálogo perfectamente estructurado entre un Doctor y un Paciente.\n\n"
                "🎯 OBJETIVO PRINCIPAL:\n"
                "Transforma la transcripción en un arreglo JSON donde cada elemento es un turno de diálogo con exactamente una clave: \"Doctor\" o \"Paciente\"\n\n"
                "📋 REGLAS DE PROCESAMIENTO:\n"
                "1. ELIMINA identificadores personales (nombres, direcciones, teléfonos, fechas específicas, edades)\n"
                "2. CORRIGE errores obvios de transcripción manteniendo el significado médico\n"
                "3. LIMPIA muletillas (eh, em, este) y falsos comienzos\n"
                "4. MANTIÉN el flujo natural de la conversación y el contexto clínico\n\n"
                "👨‍⚕️ IDENTIFICACIÓN DEL DOCTOR (Alta prioridad):\n"
                "• Preguntas sobre: síntomas, antecedentes, medicamentos, alergias, historia familiar\n"
                "• Instrucciones médicas: recetas, tratamientos, seguimientos, derivaciones\n"
                "• Lenguaje clínico: exploración, órdenes de pruebas, diagnósticos\n"
                "• Frases profesionales: \"Voy a examinar\", \"Voy a prescribir\", \"Programaremos\", \"¿Alguna alergia?\", \"¿Desde cuándo...?\", \"¿Puede describir...?\", \"En una escala del 1 al 10\"\n"
                "• Terminología médica: términos anatómicos, patologías, fármacos\n"
                "• Frases de autoridad: \"Recomiendo\", \"Debe\", \"Es importante que\"\n\n"
                "🤒 IDENTIFICACIÓN DEL PACIENTE (Alta prioridad):\n"
                "• Experiencias personales: síntomas, sensaciones, descripciones del dolor\n"
                "• Respuestas a preguntas: \"Sí\", \"No\", \"Creo\", \"Tal vez\", \"No estoy seguro\"\n"
                "• Historia personal: \"Tengo\", \"Tuve\", \"Tomé\", \"Fui\", \"Siento\"\n"
                "• Dudas y preocupaciones: \"¿Qué significa?\", \"¿Es grave?\", \"¿Cuánto tardará?\"\n"
                "• Respuestas emocionales: \"Me preocupa\", \"Me da miedo\", \"Espero\", \"Me alivia\"\n"
                "• Contexto personal: \"En el trabajo\", \"Anoche\", \"Al despertar\"\n\n"
                "🔄 REGLAS DE FLUJO DE CONVERSACIÓN:\n"
                "• Las consultas suelen empezar con el Doctor saludando y preguntando por el problema\n"
                "• El Paciente responde con su motivo de consulta\n"
                "• El Doctor hace preguntas de seguimiento\n"
                "• El Paciente aporta respuestas y detalles adicionales\n"
                "• El Doctor puede preguntar por antecedentes, medicación, etc.\n"
                "• El Paciente comparte información relevante\n"
                "• El Doctor ofrece evaluación, recomendaciones o plan terapéutico\n"
                "• El Paciente puede pedir aclaraciones\n\n"
                "⚠️ REQUISITOS CRÍTICOS:\n"
                "• Devuelve SOLO un arreglo JSON: sin explicaciones, sin markdown, sin comentarios, sin bloques de código\n"
                "• NO envuelvas el JSON en ```json``` ni en otro formato\n"
                "• Cada turno debe ser una idea completa\n"
                "• Combina oraciones relacionadas del mismo hablante en un solo turno\n"
                "• Si dudas del hablante, usa el contexto y el flujo típico de consulta\n"
                "• Asegura formato JSON correcto con comillas escapadas\n"
                "• Empieza directamente con [ y termina con ]\n\n"
                "📤 FORMATO DE SALIDA:\n"
                "[{\"Doctor\": \"Hola, ¿qué le trae hoy?\"}, {\"Paciente\": \"Tengo dolor en el pecho desde hace tres días.\"}, {\"Doctor\": \"¿Puede describirme el dolor?\"}]"
            )
        else:
            system_prompt = (
                "You are an expert medical conversation analyzer. Your task is to convert a raw medical consultation transcript into a perfectly structured dialogue between a Doctor and Patient.\n\n"
                "🎯 PRIMARY OBJECTIVE:\n"
                "Transform the transcript into a JSON array where each element is a dialogue turn with exactly one key: \"Doctor\" or \"Patient\"\n\n"
                "📋 PROCESSING RULES:\n"
                "1. REMOVE all personal identifiers (names, addresses, phone numbers, specific dates, ages)\n"
                "2. FIX obvious transcription errors while preserving medical meaning\n"
                "3. CLEAN up filler words (um, uh, like, you know) and false starts\n"
                "4. MAINTAIN natural conversation flow and medical context\n\n"
                "👨‍⚕️ DOCTOR IDENTIFICATION (High Priority):\n"
                "• Questions about: symptoms, medical history, medications, allergies, family history\n"
                "• Medical instructions: prescriptions, treatments, follow-ups, referrals\n"
                "• Clinical language: examination procedures, test orders, diagnoses\n"
                "• Professional phrases: \"Let me examine\", \"I'll prescribe\", \"We'll schedule\", \"Any allergies?\", \"How long have you had\", \"Can you describe\", \"On a scale of 1-10\"\n"
                "• Medical terminology: anatomical terms, medical conditions, drug names\n"
                "• Authority statements: \"I recommend\", \"You should\", \"It's important that\"\n\n"
                "🤒 PATIENT IDENTIFICATION (High Priority):\n"
                "• Personal experiences: symptoms, feelings, pain descriptions\n"
                "• Answers to questions: \"Yes\", \"No\", \"I think\", \"Maybe\", \"I'm not sure\"\n"
                "• Personal history: \"I have\", \"I had\", \"I took\", \"I went to\", \"I feel\"\n"
                "• Concerns and questions: \"What does this mean?\", \"Is it serious?\", \"How long will it take?\"\n"
                "• Emotional responses: \"I'm worried\", \"I'm scared\", \"I hope\", \"I'm relieved\"\n"
                "• Personal context: \"At work\", \"Last night\", \"When I woke up\", \"My husband said\"\n\n"
                "🔄 CONVERSATION FLOW RULES:\n"
                "• Medical consultations typically start with the Doctor greeting and asking about the problem\n"
                "• Patient responds with their main complaint\n"
                "• Doctor asks follow-up questions\n"
                "• Patient provides answers and additional details\n"
                "• Doctor may ask about medical history, medications, etc.\n"
                "• Patient shares relevant information\n"
                "• Doctor provides assessment, recommendations, or treatment plan\n"
                "• Patient may ask clarifying questions\n\n"
                "⚠️ CRITICAL REQUIREMENTS:\n"
                "• Output ONLY a JSON array - no explanations, no markdown, no comments, no code blocks\n"
                "• DO NOT wrap the JSON in ```json``` or any other formatting\n"
                "• Each dialogue turn must be a complete thought or response\n"
                "• Combine related sentences from the same speaker into one turn\n"
                "• If uncertain about speaker, consider the conversation context and typical medical consultation flow\n"
                "• Ensure proper JSON formatting with proper escaping of quotes\n"
                "• Start your response directly with [ and end with ]\n\n"
                "📤 OUTPUT FORMAT:\n"
                "[{\"Doctor\": \"Hello, what brings you in today?\"}, {\"Patient\": \"I've been having chest pain for three days.\"}, {\"Doctor\": \"Can you describe the pain for me?\"}]"
            )
        
        # Calculate optimal chunk size based on model context
        # Further reduce chunk size to ensure reliable processing
        max_chars_per_chunk = 2000 if settings.openai.model.startswith('gpt-4') else 1500
        overlap_chars = 200  # Overlap between chunks to preserve context
        
        # Split into sentences for better chunking
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_transcript) if s.strip()]
        
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
        
        # Log chunk details for debugging
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1}: {len(chunk)} chars, preview: {chunk[:100]}...")
        
        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
                chunk_prompt = (
                    f"FRAGMENTO DE TRANSCRIPCIÓN DE CONSULTA MÉDICA {i+1}:\n"
                    f"{chunk}\n\n"
                    f"TAREA: Convierte este fragmento en diálogo estructurado Doctor-Paciente.\n"
                    f"Nota: Es parte de una conversación más larga. Usa pistas de contexto y patrones típicos de consulta.\n\n"
                    f"SALIDA: Devuelve SOLO un arreglo JSON que empiece con [ y termine con ]. No uses markdown ni bloques de código."
                )
            else:
                chunk_prompt = (
                    f"MEDICAL CONSULTATION TRANSCRIPT CHUNK {i+1}:\n"
                    f"{chunk}\n\n"
                    f"TASK: Convert this transcript chunk into structured Doctor-Patient dialogue.\n"
                    f"Note: This is part of a larger conversation. Use context clues and medical consultation patterns.\n\n"
                    f"OUTPUT: Return ONLY a JSON array starting with [ and ending with ]. Do not use markdown, code blocks, or any other formatting."
                )
            
            logger.info(f"Processing chunk {i+1}...")
            chunk_result = await self._process_single_chunk(
                client, system_prompt, chunk_prompt, settings, logger
            )
            
            
            logger.info(f"Chunk {i+1} result: {chunk_result[:200] if chunk_result else 'None'}...")
            logger.info(f"Chunk {i+1} input length: {len(chunk)}")
            logger.info(f"Chunk {i+1} result length: {len(chunk_result) if chunk_result else 0}")
            logger.info(f"Chunk {i+1} result != input: {chunk_result != chunk if chunk_result else False}")
            
            
            if chunk_result and chunk_result != chunk:
                try:
                    parsed = json.loads(chunk_result)
                    
                    if isinstance(parsed, list):
                        chunk_results.append(parsed)  # Append the entire list, don't extend
                        logger.info(f"Chunk {i+1} processed successfully: {len(parsed)} dialogue turns")
                    else:
                        logger.warning(f"Chunk {i+1} returned invalid format: {type(parsed)}, using fallback")
                        chunk_results.append([{"Doctor": f"[Chunk {i+1} processing failed - invalid format]"}])
                except json.JSONDecodeError as e:
                    logger.warning(f"Chunk {i+1} JSON parsing failed: {e}, using fallback")
                    logger.warning(f"Chunk {i+1} content: {chunk_result[:200]}...")
                    chunk_results.append([{"Doctor": f"[Chunk {i+1} processing failed - JSON error]"}])
                except Exception as e:
                    logger.warning(f"Chunk {i+1} processing error: {e}, using fallback")
                    chunk_results.append([{"Doctor": f"[Chunk {i+1} processing failed - {str(e)}]"}])
            else:
                logger.warning(f"Chunk {i+1} processing failed - no result or same as input, using fallback")
                logger.warning(f"Chunk {i+1} input length: {len(chunk)}, result: {chunk_result[:100] if chunk_result else 'None'}...")
                chunk_results.append([{"Doctor": f"[Chunk {i+1} processing failed - no result]"}])
        
        # Merge and clean up overlapping content
        merged_dialogue = self._merge_chunk_results(chunk_results, logger)
        
        # Convert back to JSON string
        result_json = json.dumps(merged_dialogue)
        return result_json
    
    async def _process_single_chunk(
        self, 
        client: OpenAI, 
        system_prompt: str, 
        user_prompt: str, 
        settings, 
        logger
    ) -> str:
        """Process a single chunk of transcript."""
        
        def _call_openai() -> str:
            try:
                # Use appropriate max_tokens for chunk processing
                max_tokens = 2000 if settings.openai.model.startswith('gpt-4') else 1500
                
                
                logger.info(f"=== STARTING LLM CALL ===")
                logger.info(f"Model: {settings.openai.model}")
                logger.info(f"Max tokens: {max_tokens}")
                logger.info(f"API Key present: {bool(settings.openai.api_key)}")
                logger.info(f"System prompt length: {len(system_prompt)} characters")
                logger.info(f"User prompt length: {len(user_prompt)} characters")
                logger.info(f"User prompt preview: {user_prompt[:300]}...")
                
                resp = client.chat.completions.create(
                    model=settings.openai.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
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
                logger.error(f"Model: {settings.openai.model}")
                logger.error(f"API Key present: {bool(settings.openai.api_key)}")
                logger.error(f"API Key preview: {settings.openai.api_key[:10]}..." if settings.openai.api_key else "No API key")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.error(f"=== END ERROR ===")
                return ""
        
        return await asyncio.to_thread(_call_openai)
    
    def _merge_chunk_results(self, chunk_results: list, logger) -> list:
        """Merge chunk results and remove overlapping content."""
        
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
                continue
            
            # Check for overlap with the last item in merged
            last_merged = merged[-1]
            first_chunk = chunk[0]
            
            
            # Simple deduplication: if last turn in merged is same as first turn in current chunk,
            # skip the first turn of current chunk
            if (isinstance(last_merged, dict) and isinstance(first_chunk, dict) and
                len(last_merged) == 1 and len(first_chunk) == 1 and
                list(last_merged.keys())[0] == list(first_chunk.keys())[0] and
                list(last_merged.values())[0] == list(first_chunk.values())[0]):
                
                # Skip first item in current chunk
                merged.extend(chunk[1:])
            else:
                merged.extend(chunk)
            
        
        logger.info(f"Merged {len(chunk_results)} chunks into {len(merged)} dialogue turns")
        return merged
    
    async def _process_transcript_simple(
        self, 
        client: OpenAI, 
        raw_transcript: str, 
        settings, 
        logger,
        language: str = "en"
    ) -> str:
        """Process transcript with simplified approach for long transcripts to avoid timeouts."""
        
        # Simplified system prompt for faster processing
        if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
            system_prompt = (
                "Convierte esta transcripción médica en diálogo Doctor-Paciente. "
                "Elimina información personal. Devuelve SOLO un JSON array con turnos de diálogo. "
                "Formato: [{\"Doctor\": \"texto\"}, {\"Paciente\": \"texto\"}]"
            )
        else:
            system_prompt = (
                "Convert this medical transcript into Doctor-Patient dialogue. "
                "Remove personal information. Return ONLY a JSON array with dialogue turns. "
                "Format: [{\"Doctor\": \"text\"}, {\"Patient\": \"text\"}]"
            )
        
        # Truncate very long transcripts to avoid API limits
        max_length = 8000  # Reduced from chunking approach
        if len(raw_transcript) > max_length:
            logger.info(f"Truncating transcript from {len(raw_transcript)} to {max_length} characters for faster processing")
            raw_transcript = raw_transcript[:max_length] + "..."
        
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