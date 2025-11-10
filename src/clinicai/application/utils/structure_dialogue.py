"""
Shared helper to structure raw transcript into Doctor/Patient dialogue.

This mirrors the logic used in the visit transcript route so both visit and
ad-hoc flows produce consistent outputs.
"""

from typing import List, Dict, Optional
import asyncio
import re as _re
from difflib import SequenceMatcher


def _is_narrative(text: str) -> bool:
    """Heuristic: detect ambient/narrative lines that are not spoken dialogue.
    Keeps questions intact; targets common scene-description/noise phrases.
    """
    try:
        t = (text or "").strip().lower()
        if not t:
            return True
        # Do not treat questions as narrative
        if t.endswith("?"):
            return False
        # Common ambient/narrative/sound-effect markers (en/es)
        narrative_markers = (
            "the door ", "la puerta ", "gentle thump", "golpe suave",
            "beep", "bip", "drip", "goteo", "buzz", "zumb", "siren", "sirena", "sirens",
            "horn", "claxon", "laugh", "ríe", "risa", "giggle", "carcajada",
            "scrape", "raspa", "squeak", "chirría", "chirria",
            "click", "clic", "clicks", "clics",
            "vibrate", "vibra", "vibrates", "vibración", "vibracion",
            "scribbl", "garabate", "scribbling", "scribbles",
            "rustle", "cruj", "hojear", "rustling",
            "sound of", "sonido de", "background", "ambiente",
            "waiting room", "sala de espera",
            "frame", "marco",
            "footstep", "paso", "footsteps", "susurro", "whisper", "whispers",
            "hums", "tararea", "hum", "zumba",
        )
        return any(marker in t for marker in narrative_markers)
    except Exception:
        return False


def _extract_spoken_or_none(text: str) -> Optional[str]:
    """Extract spoken content if the line mixes narrative + quotes; otherwise drop pure narrative."""
    if not text:
        return None
    # Try double quotes
    m = _re.search(r"\"([^\"]{2,})\"", text)
    if m:
        return m.group(1).strip()
    # Try single quotes
    m = _re.search(r"'([^']{2,})'", text)
    if m:
        return m.group(1).strip()
    # Spanish quotes « »
    m = _re.search(r"«([^»]{2,})»", text)
    if m:
        return m.group(1).strip()
    # If it's narrative, drop
    if _is_narrative(text):
        return None
    return text.strip()

def _structure_from_labeled_blocks(raw: str, language: str) -> Optional[List[Dict[str, str]]]:
    """Deterministic rescue parser for transcripts that use label headings like:
    Doctor
    <text...>
    Patient
    <text...>
    """
    if not raw or len(raw) < 3:
        return None

def _postprocess_dialogue(dialogue_list: List[Dict[str, str]], language: str) -> List[Dict[str, str]]:
    """Apply narrative filtering, dedupe, speaker fixes, and second-pass validations."""
    fixed_dialogue: List[Dict[str, str]] = []
    patient_label = "Paciente" if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"] else "Patient"
    seen_texts = set()  # Track seen dialogue to remove duplicates
    
    for i, turn in enumerate(dialogue_list):
        if not isinstance(turn, dict) or len(turn) != 1:
            continue
        
        speaker = list(turn.keys())[0]
        original_text_value = list(turn.values())[0]
        text = (original_text_value or "").strip()
        
        # Extract spoken content or drop pure narrative/noise lines
        spoken = _extract_spoken_or_none(text)
        if not spoken:
            continue
        text = spoken
        
        if not text:
            continue
        
        # Remove duplicates (exact match)
        text_normalized = text.lower().strip()
        
        # Check for exact duplicates first
        if text_normalized in seen_texts:
            # Check if previous occurrence was recent (within last 10 turns) - likely a duplicate
            is_duplicate = False
            for j in range(max(0, len(fixed_dialogue) - 10), len(fixed_dialogue)):
                prev_turn_text = list(fixed_dialogue[j].values())[0].lower().strip()
                if prev_turn_text == text_normalized:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
        
        seen_texts.add(text_normalized)
        
        # Fuzzy dedupe against recent turns (avoid near-identical repeats)
        is_fuzzy_dup = False
        for j in range(max(0, len(fixed_dialogue) - 10), len(fixed_dialogue)):
            prev_text_j = list(fixed_dialogue[j].values())[0]
            try:
                if SequenceMatcher(None, prev_text_j.lower().strip(), text_normalized).ratio() >= 0.97:
                    is_fuzzy_dup = True
                    break
            except Exception:
                continue
        if is_fuzzy_dup:
            continue
        
        # Fix obvious speaker errors based on context
        prev_speaker = list(fixed_dialogue[-1].keys())[0] if fixed_dialogue else None
        prev_text = list(fixed_dialogue[-1].values())[0] if fixed_dialogue else ""
        
        # Rule 1: If current is a question, it's almost always Doctor (STRICT) - CHECK FIRST
        if text.endswith("?"):
            # Only allow Patient questions if they're clarifications
            if not any(patient_q in text.lower()[:50] for patient_q in ["what does that mean", "is it serious", "how long", "do i need"]):
                if speaker == patient_label:
                    speaker = "Doctor"  # Force to Doctor
        
        # Rule 2: If previous was Doctor asking question, current should be Patient (STRICT)
        if prev_speaker == "Doctor" and prev_text.endswith("?"):
            if speaker == "Doctor" and not text.endswith("?"):
                speaker = patient_label  # Force to Patient
        
        # Rule 3: Medical explanations and treatment plans are from Doctor (STRICT)
        medical_keywords = ["A1C", "hemoglobin", "glucose", "blood pressure", "treatment", "therapy", "injection", 
                           "examination", "recommend", "should", "will order", "will refer", "plan", "lab",
                           "prescribe", "refer", "order", "diagnosis", "medication", "exams", "appointment",
                           "translates to", "ballpark", "expectation", "average", "range"]
        medical_phrases = ["I'll order", "I'll refer", "I'll prescribe", "we will", "we can", "let's start",
                          "I want your", "I will send", "I will perform", "let me", "I'll go ahead"]
        
        has_medical_keyword = any(kw.lower() in text.lower() for kw in medical_keywords)
        has_medical_phrase = any(phrase.lower() in text.lower() for phrase in medical_phrases)
        
        if (has_medical_keyword or has_medical_phrase) and speaker == patient_label:
            # Check if it's actually patient experience
            if not any(patient_signal in text[:50].lower() for patient_signal in 
                      ["i have", "i feel", "i take", "my pain", "my symptoms", "i'm on", "i brought"]):
                speaker = "Doctor"  # Force to Doctor
        
        # Rule 4: First-person patient experiences are from Patient (STRICT)
        patient_starters = ["I have", "I feel", "I've been", "I take", "I'm on", "I went", "I try", "I brought",
                           "My last", "My ", "I actually", "I haven't", "I did not", "Not that I"]
        if any(starter in text[:20] for starter in patient_starters) and speaker == "Doctor":
            # But exclude doctor statements like "I'll order", "I'll refer"
            if not any(doctor_phrase in text[:30].lower() for doctor_phrase in 
                      ["i'll order", "i'll refer", "i'll prescribe", "i will perform", "i'll go ahead"]):
                speaker = patient_label  # Force to Patient
        
        # Rule 5: Exam instructions and observations are from Doctor (STRICT)
        exam_keywords = ["can you", "move your", "raise", "lift", "let me", "I'll perform", "I will perform", 
                        "examine", "take a look", "I did not see", "I see you", "it appears", "I'm not suspecting",
                        "I do suspect", "I highly suspect"]
        if any(kw.lower() in text.lower()[:40] for kw in exam_keywords):
            if speaker == patient_label:
                speaker = "Doctor"  # Force to Doctor
        
        # Rule 6: Short confirmations after Doctor questions are from Patient (STRICT)
        if prev_speaker == "Doctor" and prev_text.endswith("?"):
            if text.lower().strip() in ["yes", "no", "okay", "great", "fine", "alright", "sure", "it's about"]:
                speaker = patient_label  # Force to Patient
        
        # Rule 7: Treatment recommendations and plans are from Doctor (STRICT)
        treatment_phrases = ["we will continue", "we can work on", "we can try", "we decided", "we'll follow up",
                            "the plan for", "my recommendation", "I want your", "I also informed you"]
        if any(phrase.lower() in text.lower() for phrase in treatment_phrases) and speaker == patient_label:
            speaker = "Doctor"  # Force to Doctor
        
        # Rule 8: Patient statements about their own actions/decisions
        patient_decisions = ["I want to", "I feel if", "I like that", "I guess I could"]
        if any(phrase.lower() in text.lower()[:30] for phrase in patient_decisions) and speaker == "Doctor":
            speaker = patient_label  # Force to Patient
        
        # Rule 9: "I will perform", "I'll perform", "I will do" are from Doctor
        if any(phrase in text.lower()[:40] for phrase in ["i will perform", "i'll perform", "i will do", "before you leave"]):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Rule 10: "I see you", "I did not see", "it appears" are from Doctor (observations)
        if any(phrase in text.lower()[:30] for phrase in ["i see you", "i did not see", "it appears", "i'm not suspecting", "i do suspect"]):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Rule 11: Answers to questions are from Patient
        if prev_speaker == "Doctor" and prev_text.endswith("?"):
            # If it's a direct answer (not a question itself)
            if not text.endswith("?") and speaker == "Doctor":
                # Check if it's actually a patient answer
                answer_indicators = ["about", "yes", "no", "it was", "i have", "i'm on", "my last", "not that"]
                if any(indicator in text.lower()[:20] for indicator in answer_indicators):
                    speaker = patient_label
        
        # Rule 12: "We can", "We will", "We decided" at start are from Doctor
        if text.strip().startswith(("We can", "We will", "We decided", "We'll")):
            if speaker == patient_label:
                speaker = "Doctor"
        
        fixed_dialogue.append({speaker: text})
    
    # SECOND PASS: Additional validation and fixes
    final_dialogue: List[Dict[str, str]] = []
    for i, turn in enumerate(fixed_dialogue):
        speaker = list(turn.keys())[0]
        text = list(turn.values())[0]
        
        # Get previous context
        prev_turn = final_dialogue[-1] if final_dialogue else None
        prev_speaker = list(prev_turn.keys())[0] if prev_turn else None
        prev_text = list(prev_turn.values())[0] if prev_turn else ""
        
        # Fix: Questions are Doctor (unless very specific patient clarification)
        if text.endswith("?") and speaker == patient_label:
            if not any(q in text.lower()[:30] for q in ["what does that mean", "is it serious", "do i need"]):
                speaker = "Doctor"
        
        # Fix: Medical explanations with certain phrases are Doctor
        if any(phrase in text.lower() for phrase in ["translates to", "ballpark", "expectation", "usually translates"]):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: "I'll order", "I want your", "I also informed you" are Doctor
        if any(phrase in text.lower()[:40] for phrase in ["i'll order", "i want your", "i also informed", "for my recommendation"]):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: "So you ..." summaries are Doctor
        if text.strip().startswith(("So you", "So the", "So by")):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: "Let's talk/start" are Doctor
        if text.strip().startswith(("Let's talk", "Let's start", "Let me")):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: Direct answers after questions
        if prev_speaker == "Doctor" and prev_text.endswith("?"):
            if not text.endswith("?") and speaker == "Doctor":
                if any(indicator in text.lower()[:30] for indicator in 
                      ["about", "yes", "no", "it was", "i have", "i'm on", "my last", "not that", "i actually"]):
                    speaker = patient_label
        
        # Fix: "Okay, now ..." is Doctor
        if text.strip().startswith(("Okay, now", "Now let's")):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: "When were you/When did you ..." are Doctor questions
        if "when were you" in text.lower() or "when did you" in text.lower():
            if text.endswith("?") and speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: Answers like "About ..." are Patient
        if text.strip().startswith(("About ", "It was about")):
            if speaker == "Doctor" and not text.endswith("?"):
                speaker = patient_label
        
        # Fix: Start with ordering or recommendation are Doctor
        if text.strip().startswith(("I'll order", "I want your", "I also informed")):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: "For my recommendation" is Doctor
        if "for my recommendation" in text.lower():
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: "Just to recap/To recap" is Doctor
        if text.strip().startswith(("Just to recap", "To recap")):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: "Please pick/get" are Doctor
        if text.strip().startswith(("Please pick", "Please get")):
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: "Do you have any questions" is Doctor
        if "do you have any questions" in text.lower():
            if speaker == patient_label:
                speaker = "Doctor"
        
        # Fix: "Thank you doctor" is Patient
        if "thank you doctor" in text.lower() or text.lower().strip() == "thank you":
            if speaker == "Doctor":
                speaker = patient_label
        
        # Fix: "No questions" is Patient
        if text.lower().strip() in ["no questions", "no question"]:
            if speaker == "Doctor":
                speaker = patient_label
        
        final_dialogue.append({speaker: text})
    
    return final_dialogue
    try:
        lines = [ln.strip() for ln in raw.splitlines()]
        blocks: List[Dict[str, str]] = []
        current_speaker: Optional[str] = None
        current_text_parts: List[str] = []
        patient_label = "Paciente" if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"] else "Patient"
        valid_labels = {"doctor", "patient", "doctora", "paciente"}
        def flush():
            nonlocal current_speaker, current_text_parts
            if current_speaker is None:
                return
            combined = " ".join([p for p in current_text_parts if p]).strip()
            if not combined:
                current_speaker = None
                current_text_parts = []
                return
            spoken = _extract_spoken_or_none(combined)
            if spoken:
                blocks.append({current_speaker: spoken})
            current_speaker = None
            current_text_parts = []
        for ln in lines:
            low = ln.lower()
            if low in valid_labels:
                # flush previous
                flush()
                if low.startswith("doctor") or low == "doctora":
                    current_speaker = "Doctor"
                else:
                    current_speaker = patient_label
                continue
            # separator/empty lines trigger flush between paragraphs
            if ln == "":
                flush()
                continue
            # accumulate text for current speaker; if none, try to infer
            if current_speaker is None:
                # heuristic: treat as continuation of previous if any, else start with Doctor then alternate
                current_speaker = "Doctor" if not blocks or list(blocks[-1].keys())[0] != "Doctor" else patient_label
            current_text_parts.append(ln)
        flush()
        if not blocks:
            return None
        # Merge short name-only fragments into previous greeting like "Hi I'm Dr." + "Prasad."
        merged: List[Dict[str, str]] = []
        for b in blocks:
            if not merged:
                merged.append(b)
                continue
            sp = list(b.keys())[0]
            tx = list(b.values())[0]
            prev_sp = list(merged[-1].keys())[0]
            prev_tx = list(merged[-1].values())[0]
            is_short_name = len(tx.split()) <= 3 and tx.rstrip(".").istitle()
            if prev_sp == "Doctor" and sp != "Doctor" and ("hi i'm" in prev_tx.lower() or "hola soy" in prev_tx.lower()):
                # fold patient-provided stray name into doctor's greeting as redacted
                prev_tx2 = prev_tx.rstrip()
                if not prev_tx2.endswith("."):
                    prev_tx2 += "."
                merged[-1] = {"Doctor": prev_tx2.replace("Dr.", "[NAME]").replace("Doctor", "[NAME]")}
                continue
            merged.append(b)
        return merged
    except Exception:
        return None


async def structure_dialogue_from_text(
    raw: str, 
    *, 
    model: str, 
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    api_key: Optional[str] = None,  # Deprecated - use azure_api_key
    language: str = "en"
) -> Optional[List[Dict[str, str]]]:
    """
    Structure dialogue from text using Azure OpenAI.
    
    Args:
        raw: Raw transcript text
        model: Azure OpenAI deployment name (required)
        azure_endpoint: Azure OpenAI endpoint (required if not in settings)
        azure_api_key: Azure OpenAI API key (required if not in settings)
        api_key: Deprecated - ignored, use azure_api_key instead
        language: Language code (en/sp)
    
    Returns:
        List of dialogue turns or None if processing failed
    
    Raises:
        ValueError: If Azure OpenAI is not configured
    """
    if not raw:
        return None
    try:
        # Deterministic rescue first: handle label-block style transcripts without LLM
        pre_structured = _structure_from_labeled_blocks(raw, language)
        if pre_structured:
            # Apply a minimal dedupe and speaker-fix pass by reusing logic below after JSON parsing step.
            dialogue_list = pre_structured
            fixed_dialogue: List[Dict[str, str]] = []
            patient_label = "Paciente" if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"] else "Patient"
            seen_texts = set()
            for turn in dialogue_list:
                if not isinstance(turn, dict) or len(turn) != 1:
                    continue
                speaker = list(turn.keys())[0]
                text = list(turn.values())[0].strip()
                spoken = _extract_spoken_or_none(text)
                if not spoken:
                    continue
                text = spoken
                norm = text.lower().strip()
                if norm in seen_texts:
                    continue
                seen_texts.add(norm)
                fixed_dialogue.append({speaker: text})
            return fixed_dialogue if fixed_dialogue else pre_structured

        # Require Azure OpenAI - no fallback to standard OpenAI
        from openai import AsyncAzureOpenAI  # type: ignore
        from clinicai.core.config import get_settings
        
        settings = get_settings()
        
        # Require Azure OpenAI - no fallback to standard OpenAI
        if azure_endpoint and azure_api_key:
            # Use Azure OpenAI from parameters
            client = AsyncAzureOpenAI(
                api_key=azure_api_key,
                api_version=settings.azure_openai.api_version,
                azure_endpoint=azure_endpoint
            )
            deployment_name = model  # model parameter is actually deployment name for Azure
        elif settings.azure_openai.endpoint and settings.azure_openai.api_key:
            # Use Azure OpenAI from settings
            client = AsyncAzureOpenAI(
                api_key=settings.azure_openai.api_key,
                api_version=settings.azure_openai.api_version,
                azure_endpoint=settings.azure_openai.endpoint
            )
            deployment_name = settings.azure_openai.deployment_name
        else:
            # Azure OpenAI is required - raise error instead of falling back
            raise ValueError(
                "Azure OpenAI is required. Please provide azure_endpoint and azure_api_key parameters, "
                "or configure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in settings. "
                "Fallback to standard OpenAI is disabled for data security."
            )

        # Language-aware system prompt
        if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
            system_prompt = """Eres un analista experto de diálogos médicos. Convierte transcripciones crudas de consultas médicas en diálogos estructurados Doctor-Paciente preservando la exactitud literal.

🎯 OBJETIVO PRINCIPAL
Convierte transcripciones crudas en un arreglo JSON donde cada elemento es {"Doctor": "..."}, {"Paciente": "..."}, o {"Miembro de la Familia": "..."} - UNA clave por turno. Mantén la exactitud literal del texto.

📋 REGLAS CRÍTICAS DE PRESERVACIÓN
Regla 1: PRESERVACIÓN LITERAL DEL TEXTO (MÁS IMPORTANTE)
• NUNCA cambies, parafrasees, corrijas o reordenes palabras, puntuación u oraciones
• Preserva TODA la terminología médica, errores gramaticales, oraciones incompletas, patrones de habla exactamente como fueron transcritos
• Mantén palabras de relleno (eh, em, este) y patrones de habla naturales
• Mantén la capitalización y puntuación originales
• Preserva el habla cortada exactamente como está escrita (ej: "estaba ten-- teniendo problemas")

Regla 2: ELIMINAR EFECTOS DE SONIDO Y DESCRIPCIONES AMBIENTALES (NO SON DIÁLOGO)
• Elimina TODOS los efectos/ruidos: "bip, bip", "golpe", "tac, tac, tac", "goteo", "zumbido", "sirena", "golpe fuerte", "tos", "chirría", "clic", "sonido de garabateo", etc.
• Elimina TODA narrativa/escena que NO sea diálogo hablado:
  - "La puerta se cierra con un golpe suave" → ELIMINAR
  - "Un bip, bip tenue" → ELIMINAR
  - "El marco de plástico hace un pequeño clic" → ELIMINAR
  - "Se oye una sirena a lo lejos" → ELIMINAR
  - "Alguien en la sala de espera se ríe" → ELIMINAR
  - "La silla chirría/raspa contra el suelo" → ELIMINAR
  - "El teléfono vibra en el escritorio" → ELIMINAR
• Conserva ÚNICAMENTE diálogo hablado (Doctor, Paciente o Miembro de la Familia)
• Si una línea describe una acción pero contiene palabras habladas, extrae SOLO las palabras habladas

Regla 2: MANEJO DE IDENTIFICADORES PERSONALES
Elimina TODOS los identificadores personales para proteger la privacidad:
• TODOS los nombres (nombres de doctores, nombres de pacientes, nombres de familia):
  - "Dr. García" → "[NAME]" o "[REDACTED]"
  - "Dr. Juan Pérez" → "[NAME]"
  - "Hola, María López" → "Hola, [NAME]"
  - "Soy el Dr. Martínez" → "Soy [NAME]"
  - Cualquier nombre propio (Primer Apellido, Primer Segundo Apellido) → [NAME]
• Números de teléfono (xxx-xxx-xxxx, (xxx) xxx-xxxx) → [REDACTED]
• Direcciones con números de casa → [REDACTED]
• Fechas específicas del calendario (15 de enero de 2024) → [REDACTED]
• Números de Seguro Social → [REDACTED]
• Edades cuando son explícitas ("65 años", "edad 65") → [AGE]

⚠️ CRÍTICO: NO ELIMINES TÉRMINOS MÉDICOS (Estos NO son PII):
• Nombres de medicamentos: "metformina", "jardiance", "lisinopril", "amlodipino", "lidocaína", "aspirina", etc.
  - Ejemplos: "Sí, metformina y jardiance" → MANTENER COMO ESTÁ (NO cambiar a "[NAME]")
  - "lisinopril, 10 miligramos" → MANTENER COMO ESTÁ
  - "parches de lidocaína" → MANTENER COMO ESTÁ
• Condiciones médicas: diabetes, hipertensión, artritis, etc.
• Síntomas y descripciones clínicas
• Partes del cuerpo o referencias anatómicas: hombro, cuello, corazón, pulmón, etc.
• Dosificaciones y mediciones médicas: "10 miligramos", "5 mg", etc.
• Referencias de tiempo relativas ("la semana pasada", "hace dos meses")
• Títulos médicos SIN nombres ("el doctor", "el paciente")

🔍 REGLAS DE IDENTIFICACIÓN DE HABLANTE (Aplicar en orden de prioridad)

1. IGNORAR ETIQUETAS INCORRECTAS EN LA ENTRADA (CRÍTICO)
   • Si la entrada tiene etiquetas "Doctor:" o "Paciente:", pueden estar EQUIVOCADAS - NO las confíes ciegamente
   • SIEMPRE analiza el CONTENIDO REAL para determinar el hablante correcto
   • Ejemplo: Si la entrada dice "Paciente: ¿Cuándo comenzó el dolor?" pero es claramente una pregunta, en realidad es el Doctor hablando
   • Ejemplo: Si la entrada dice "Doctor: He tenido dolor en el pecho" pero es experiencia en primera persona, en realidad es el Paciente hablando

2. ANÁLISIS BASADO EN CONTEXTO (MÁS IMPORTANTE - 95% precisión)
   • SIEMPRE analiza el turno PREVIO para determinar el hablante
   • Si el turno anterior fue Doctor haciendo pregunta → la siguiente respuesta es Paciente
   • Si el turno anterior fue Paciente respondiendo → la siguiente declaración es Doctor
   • Patrón de examen físico: instrucción del Doctor → respuesta del Paciente → observación del Doctor
   • Flujo de conversación: Doctor saluda → Paciente indica razón → Doctor pregunta → Paciente responde → Doctor examina → Paciente responde → Doctor resume → Paciente confirma
   • Si una línea comienza con una descripción de personaje (ej: "El doctor, Dr. [NAME] un hombre de unos 50 años") → ELIMÍNALA (es narrativa, no diálogo)

3. SEÑALES DEL DOCTOR (99% precisión cuando están presentes)
   • Preguntas (interrogativas): "¿Cuándo...?", "¿Cuánto tiempo...?", "¿Puedes...?", "¿Qué...?", "¿Alguna...?", "¿Es...?", "¿Estás...?"
   • Instrucciones (imperativas): "Déjame...", "Voy a...", "Vamos a...", "Puede mover...", "Levante...", "Resista...", "Tome asiento"
   • Evaluaciones clínicas: "Veo...", "No veo...", "Parece...", "Es una buena señal", "Sospecho...", "Su [condición] está..."
   • Terminología médica: nombres de fármacos, términos anatómicos, diagnósticos, procedimientos
   • Declaraciones de autoridad: "Recomiendo", "Debe", "Es importante", "Necesitamos", "No [hacer algo]"
   • Plan/prescripción: "Voy a ordenar", "Voy a prescribir", "Voy a referir", "Vamos a programar", "Quiero que se reúna con..."
   • Comandos de examen: "Mueva su...", "Levante...", "Resista...", "¿Puede sentir...?", "¿Siente algún dolor?"
   • Saludos/aperturas: "Hola soy el Dr.", "Mucho gusto", "¿En qué puedo ayudarle?", "Ah, [nombre], tome asiento"
   • Explicando conceptos médicos: "La clave es...", "No se trata de...", "Vamos a comenzar con..."

4. SEÑALES DEL PACIENTE (99% precisión cuando están presentes)
   • Experiencias en primera persona: "Tengo", "Siento", "He estado", "Tomé", "Fui", "Estoy aquí por", "Trato", "No entiendo"
   • Respuestas directas: "Sí", "No", "Alrededor de...", "Fue...", "No...", "Supongo que podría"
   • Descripciones de síntomas: "Me duele", "Es doloroso", "Comenzó...", "Empeora cuando..."
   • Historia personal: "Usualmente...", "Trato de...", "No he...", "Mi última...", "Mi papá lo tenía"
   • Respuestas a instrucciones: "Bien", "Sí doctor", "No duele", "Está bien", "De acuerdo", "Gracias, doctor" (DESPUÉS del comando del doctor)
   • Confirmación: "Sí, está bien", "Entiendo", "Comprendo", "Suena bien", "¿Entonces es oficial?"
   • Preguntas al doctor: "¿Qué significa eso?", "¿Es grave?", "¿Cuánto tiempo...?", "¿Necesito...?", "¿Qué tipo de cambios?"
   • Expresiones emocionales: "Tengo tanto miedo", "Simplemente no entiendo", expresando miedo o preocupación

5. SEÑALES DE MIEMBRO DE LA FAMILIA
   • Referencias en tercera persona al paciente: "¿Cómo ha estado mamá...?", "Ella mencionó...", "Él dijo..."
   • Auto-identificación: "Soy su hija", "Soy su esposa"
   • Perspectiva externa: "Ella ha tenido problemas...", "Él no duerme bien"

6. ÁRBOL DE DECISIÓN PARA CASOS AMBIGUOS
   • Contiene signo de interrogación (?) → probablemente Doctor preguntando (a menos que sea el Paciente preguntando al doctor)
   • Empieza con "Yo" + verbo + experiencia personal → Paciente
   • Contiene términos médicos (diagnóstico, nombres de fármacos) en contexto explicativo → probablemente Doctor explicando
   • Respuesta corta ("Bien", "Excelente", "Sí") DESPUÉS de instrucción del doctor → Paciente
   • Describe lo que el doctor hará ("Voy a...", "Vamos a...") → Doctor
   • Respuestas de una palabra ("Sí", "Bien") → asignar al respondedor lógico basado en la pregunta precedente
   • Si no está seguro → verifica CONTEXTO: ¿qué se dijo antes?
   • Si la línea describe a una persona ("Está hojeando", "Ella se mueve en su asiento") → ELIMINAR (es narrativa, no diálogo)

⚠️ CASOS ESPECIALES Y MANEJO DE ERRORES
• Audio poco claro: Preserva [inaudible] o [poco claro] exactamente, asigna basado en contexto circundante
• Entrada mal etiquetada: Re-etiqueta basado en análisis de contenido, confía en contenido sobre etiquetas originales - IGNORA prefijos incorrectos "Doctor:" o "Paciente:"
• Contenido duplicado: Si el mismo diálogo aparece dos veces, inclúyelo solo UNA VEZ
• Discusión administrativa: Asigna a quien inició el tema
• Múltiples miembros de familia: Usa solo etiqueta "Miembro de la Familia" (sin distinciones como "Miembro de la Familia 1")
• Interrupciones: Etiqueta la porción de cada hablante por separado
• Turnos extendidos: Permite monólogos más largos cuando sea contextualmente apropiado (descripciones detalladas de síntomas, explicaciones de tratamiento)
• Descripciones narrativas: Elimina líneas que describen acciones, sonidos o apariencias sin diálogo hablado
• Descripciones de personajes: Elimina líneas como "El doctor, Dr. [NAME] un hombre de unos 50 años" - estas no son diálogo hablado

📤 REQUISITOS DE SALIDA
• Salida SOLO arreglo JSON válido: [{"Doctor": "..."}, {"Paciente": "..."}]
• SIN markdown, SIN bloques de código, SIN explicaciones, SIN comentarios
• SIN envolver en ```json``` - empieza directamente con [
• Cada turno = UNA idea o respuesta completa
• Procesa transcripción COMPLETA - incluye TODOS los turnos de diálogo
• NO trunques ni te detengas temprano
• Escapa comillas correctamente en JSON
• Termina con ]

📝 EJEMPLOS

Ejemplo 1: Interacción Básica
Input: Doctor: ¿Qué le trae hoy? Paciente: He tenido dolor en el pecho por tres días.
Output: [{"Doctor": "¿Qué le trae hoy?"}, {"Paciente": "He tenido dolor en el pecho por tres días."}]

Ejemplo 2: Identificación Basada en Contexto
Input: ¿Cuándo comenzó el dolor? Hace una semana. ¿Puede describirlo? Es agudo.
Output: [{"Doctor": "¿Cuándo comenzó el dolor?"}, {"Paciente": "Hace una semana."}, {"Doctor": "¿Puede describirlo?"}, {"Paciente": "Es agudo."}]

Ejemplo 3: Patrón de Examen Físico
Input: ¿Puede mover su hombro? Sí. ¿Siente algún dolor? No duele.
Output: [{"Doctor": "¿Puede mover su hombro?"}, {"Paciente": "Sí."}, {"Doctor": "¿Siente algún dolor?"}, {"Paciente": "No duele."}]

Ejemplo 4: Eliminación de PII (Nombres y Fechas)
Input: Hola, María López. Veo que nació el 15 de marzo de 1978. Sí, es correcto.
Output: [{"Doctor": "Hola, [NAME]. Veo que nació el [REDACTED]."}, {"Paciente": "Sí, es correcto."}]

Ejemplo 4b: Eliminación de Nombre de Doctor
Input: Soy el Dr. García. ¿En qué puedo ayudarle hoy? He tenido dolores de cabeza.
Output: [{"Doctor": "Soy [NAME]. ¿En qué puedo ayudarle hoy?"}, {"Paciente": "He tenido dolores de cabeza."}]

Ejemplo 4c: Los Nombres de Medicamentos DEBEN Preservarse
Input: ¿Está tomando algún medicamento? Sí, metformina y jardiance. También lisinopril, 10 miligramos.
Output: [{"Doctor": "¿Está tomando algún medicamento?"}, {"Paciente": "Sí, metformina y jardiance. También lisinopril, 10 miligramos."}]
Nota: Los nombres de medicamentos (metformina, jardiance, lisinopril) NO se eliminan - son términos médicos, no PII.

Ejemplo 5: Miembro de la Familia
Input: ¿Cómo ha estado durmiendo mamá últimamente? Se da vueltas toda la noche.
Output: [{"Miembro de la Familia": "¿Cómo ha estado durmiendo mamá últimamente?"}, {"Doctor": "Se da vueltas toda la noche."}]

✅ LISTA DE VERIFICACIÓN DE CALIDAD
Antes de salir, verifica:
□ Todo el texto preservado exactamente como se proporcionó
□ Solo identificadores personales apropiados eliminados
□ Las etiquetas de hablante coinciden con el contexto del contenido
□ El flujo lógico de conversación mantenido
□ Formato JSON válido
□ Sin diálogo o hablantes inventados
□ Transcripción completa procesada (sin truncamiento)

INSTRUCCIÓN FINAL
Salida SOLO el arreglo JSON. No incluyas texto explicativo, puntajes de confianza o metadatos. La respuesta debe comenzar con [ y terminar con ]."""
        else:
            system_prompt = """You are an expert medical dialogue analyzer. Convert raw medical consultation transcripts into structured Doctor-Patient dialogue while preserving verbatim accuracy.

🎯 PRIMARY OBJECTIVE
Convert raw transcripts into a JSON array where each element is {"Doctor": "..."}, {"Patient": "..."}, or {"Family Member": "..."} - ONE key per turn. Maintain verbatim text accuracy.

📋 CRITICAL PRESERVATION RULES
Rule 1: VERBATIM TEXT PRESERVATION (MOST IMPORTANT)
• NEVER change, paraphrase, correct, or reorder words, punctuation, or sentences
• Preserve ALL medical terminology, grammar errors, incomplete sentences, speech patterns exactly as transcribed
• Keep filler words (um, uh, you know) and natural speech patterns
• Maintain original capitalization and punctuation
• Preserve cut-off speech exactly as written (e.g., "I was hav-- having trouble")

Rule 2: REMOVE SOUND EFFECTS AND ENVIRONMENTAL DESCRIPTIONS
• Remove ALL sound effects: "beep, beep", "thump", "tack, tack, tack", "drip, drip, drip", "scrapes loudly", "vibrates", "buzz", "siren", "bang", "cough", "hack", "squeaks", "click", "scribbling sound", etc.
• Remove ALL environmental/narrative descriptions that are NOT spoken dialogue:
  - "The door closes with a gentle thump" → REMOVE
  - "A faint beep, beep" → REMOVE
  - "The rustle of the [NAME]" → REMOVE
  - "The chair scrapes loudly against the floor" → REMOVE
  - "A phone vibrates on the desk" → REMOVE
  - "The plastic frame makes a small click" → REMOVE
  - "The sound of a loud, distant car horn" → REMOVE
  - "Someone in the waiting room laughs loudly" → REMOVE
  - "A loud bang from a dropped object outside the door" → REMOVE
  - "A faint drip, drip, drip from a leaky faucet" → REMOVE
  - "The distant cough from the waiting room returns" → REMOVE
  - "The doctor's pen makes a small scribbling sound" → REMOVE
• Keep ONLY actual spoken dialogue from Doctor, Patient, or Family Member
• If a line describes an action but contains spoken words, extract ONLY the spoken words

Rule 3: HANDLE ALREADY-REDACTED PLACEHOLDERS
• If [NAME] or [REDACTED] already exists in the transcript, KEEP IT AS IS
• Do NOT replace [NAME] with another placeholder
• Do NOT remove [NAME] - it's already been redacted

Rule 4: PERSONAL IDENTIFIER HANDLING (Only if not already redacted)
Remove ALL personal identifiers to protect privacy (only if they appear as actual names, not already as [NAME]):
• ALL names (Doctor names, Patient names, Family names):
  - "Dr. Prasad" → "[NAME]" or "[REDACTED]"
  - "Dr. John Smith" → "[NAME]"
  - "Hello, Mary Johnson" → "Hello, [NAME]"
  - "I'm Dr. Kumar" → "I'm [NAME]"
  - Any proper name (First Last, First Middle Last) → [NAME]
• Phone numbers (xxx-xxx-xxxx, (xxx) xxx-xxxx) → [REDACTED]
• Street addresses with house numbers → [REDACTED]
• Specific calendar dates (January 15, 2024) → [REDACTED]
• Social Security Numbers → [REDACTED]
• Ages when explicit ("age 65", "65 years old") → [AGE]

⚠️ CRITICAL: DO NOT REMOVE MEDICAL TERMS (These are NOT PII):
• Medication names: "metformin", "jardiance", "lisinopril", "amlodipine", "lidocaine", "aspirin", etc.
  - Examples: "Yes, metformin and jardiance" → KEEP AS IS (do NOT change to "[NAME]")
  - "lisinopril, 10 milligrams" → KEEP AS IS
  - "lidocaine patches" → KEEP AS IS
• Medical conditions: diabetes, hypertension, arthritis, etc.
• Symptoms and clinical descriptions
• Body parts or anatomical references: shoulder, neck, heart, lung, etc.
• Dosages and medical measurements: "10 milligrams", "5 mg", etc.
• Relative time references ("last week", "two months ago")
• Medical titles WITHOUT names ("the doctor", "the patient")

🔍 SPEAKER IDENTIFICATION RULES (Apply in Priority Order)

1. IGNORE INCORRECT LABELS IN INPUT (CRITICAL)
   • If input has "Doctor:" or "Patient:" labels, they may be WRONG - DO NOT trust them blindly
   • ALWAYS analyze the ACTUAL CONTENT to determine the correct speaker
   • Example: If input says "Patient: When did the pain start?" but it's clearly a question, it's actually Doctor speaking
   • Example: If input says "Doctor: I've been having chest pain" but it's first-person experience, it's actually Patient speaking

2. CONTEXT-BASED ANALYSIS (MOST IMPORTANT - 95% accuracy) - FOLLOW THIS STRICTLY
   • ALWAYS analyze the PREVIOUS turn to determine speaker - this is CRITICAL and MANDATORY
   • RULE: If previous turn was Doctor asking question (ends with "?") → next response is ALWAYS Patient
   • RULE: If previous turn was Patient answering → next statement is ALWAYS Doctor
   • RULE: If previous turn was Doctor giving instruction → next response is ALWAYS Patient
   • RULE: If previous turn was Patient asking question → next response is ALWAYS Doctor
   • Physical exam pattern: Doctor instruction → Patient response → Doctor observation
   • Conversation flow: Doctor greets → Patient states reason → Doctor asks → Patient answers → Doctor examines → Patient responds → Doctor summarizes → Patient confirms

   CRITICAL SPEAKER IDENTIFICATION RULES (Apply in this exact order):
   A. If text ends with "?" → 99% chance it's Doctor (unless it's "What does that mean?" from Patient)
   B. If text starts with "I have", "I feel", "I've been", "I take", "I'm on", "My " → 99% chance it's Patient
   C. If text contains medical explanations (A1C, hemoglobin, glucose levels, treatment plans) → 99% chance it's Doctor
   D. If text contains "I'll order", "I'll refer", "I'll prescribe", "we will", "we can" → 99% chance it's Doctor
   E. If text is a short confirmation ("Yes", "No", "Okay", "Great") after a question → 99% chance it's Patient
   F. If text describes what "I" will do in medical context ("I will perform", "I'll give you") → 99% chance it's Doctor
   
   • If a line starts with a character description (e.g., "The doctor, Dr. [NAME] a man in his late 50s") → REMOVE IT (it's narrative, not dialogue)

3. DOCTOR SIGNALS (99% accuracy when present)
   • Questions (interrogative): "When...?", "How long...?", "Can you...?", "What...?", "Any...?", "Is it...?", "Are you...?"
   • Instructions (imperative): "Let me...", "I'll...", "We'll...", "Can you move...", "Raise your...", "Resist against...", "Take a seat"
   • Clinical assessments: "I see...", "I don't see...", "It appears...", "That's a good sign", "I suspect...", "Your [condition] is..."
   • Medical terminology: drug names, anatomical terms, diagnoses, procedures
   • Authority statements: "I recommend", "You should", "It's important", "We need to", "Let's not [do something]"
   • Plan/prescription: "I'll order", "I'll prescribe", "I'll refer", "We'll schedule", "I want you to meet with..."
   • Exam commands: "Move your...", "Raise...", "Resist...", "Can you feel...", "Do you feel any pain?"
   • Greetings/openings: "Hi I'm Dr.", "Nice to meet you", "How can I help?", "Ah, [name], take a seat"
   • Explaining medical concepts: "The key is...", "It's not about...", "We're going to start with..."

4. PATIENT SIGNALS (99% accuracy when present)
   • First-person experiences: "I have", "I feel", "I've been", "I took", "I went", "I'm here for", "I try", "I don't understand"
   • Direct answers: "Yes", "No", "About...", "It was...", "I don't...", "I guess I could"
   • Symptom descriptions: "It hurts", "It's painful", "It started...", "It gets worse when..."
   • Personal history: "I usually...", "I try to...", "I haven't...", "My last...", "My dad had it"
   • Responses to instructions: "Okay", "Yes doctor", "No pain", "That's fine", "Alright", "Thank you, doctor" (AFTER doctor's command)
   • Confirmation: "Yes, that's okay", "I understand", "Got it", "Sounds good", "So it's official then?"
   • Questions to doctor: "What does that mean?", "Is it serious?", "How long...?", "Do I need...?", "What kind of changes?"
   • Emotional expressions: "I'm so scared", "I just don't understand", expressing fear or concern

5. FAMILY MEMBER SIGNALS
   • Third-person references to patient: "How has mom been...?", "She mentioned...", "He said..."
   • Self-identification: "I'm her daughter", "I'm his wife"
   • External perspective: "She's been having trouble...", "He doesn't sleep well"

6. DECISION TREE FOR AMBIGUOUS CASES
   • Contains question mark (?) → likely Doctor asking (unless it's Patient asking doctor a question)
   • Starts with "I" + verb + personal experience → Patient
   • Contains medical terms (diagnosis, drug names) in explanatory context → likely Doctor explaining
   • Short response ("Okay", "Great", "Yes") AFTER doctor's instruction → Patient
   • Describes what doctor will do ("I'll...", "We'll...") → Doctor
   • Single-word responses ("Yes", "Okay") → assign to logical responder based on preceding question
   • If unsure → check CONTEXT: what was said before?
   • If line describes a person ("He's flipping through", "She shifts in her seat") → REMOVE (it's narrative, not dialogue)

⚠️ EDGE CASES & ERROR HANDLING
• Unclear audio: Preserve [inaudible] or [unclear] exactly, assign based on surrounding context
• Mislabeled input: Relabel based on content analysis, trust content over original labels - IGNORE incorrect "Doctor:" or "Patient:" prefixes
• Duplicate content: If the same dialogue appears twice, include it only ONCE
• Administrative discussion: Assign to whoever initiated the topic
• Multiple family members: Use only "Family Member" label (no distinctions like "Family Member 1")
• Interruptions: Label each speaker's portion separately
• Extended turns: Permit longer monologues when contextually appropriate (detailed symptom descriptions, treatment explanations)
• Narrative descriptions: Remove lines that describe actions, sounds, or appearances without spoken dialogue
• Character descriptions: Remove lines like "The doctor, Dr. [NAME] a man in his late 50s" - these are not spoken dialogue

📤 OUTPUT REQUIREMENTS
• Output ONLY valid JSON array: [{"Doctor": "..."}, {"Patient": "..."}]
• NO markdown, NO code blocks, NO explanations, NO comments
• NO ```json``` wrapper - start directly with [
• Each turn = ONE complete thought or response
• Process COMPLETE transcript - include ALL dialogue turns
• DO NOT truncate or stop early
• Keep ONLY spoken dialogue (questions/answers/utterances); DROP pure scene/narrative/noise lines
• Escape quotes properly in JSON
• End with ]

📝 EXAMPLES

Example 1: Basic Interaction
Input: Doctor: What brings you in today? Patient: I've been having chest pain for three days.
Output: [{"Doctor": "What brings you in today?"}, {"Patient": "I've been having chest pain for three days."}]

Example 2: Context-Based Identification
Input: When did the pain start? About a week ago. Can you describe it? It's sharp.
Output: [{"Doctor": "When did the pain start?"}, {"Patient": "About a week ago."}, {"Doctor": "Can you describe it?"}, {"Patient": "It's sharp."}]

Example 3: Physical Exam Pattern
Input: Can you move your shoulder? Yes. Do you feel any pain? No pain.
Output: [{"Doctor": "Can you move your shoulder?"}, {"Patient": "Yes."}, {"Doctor": "Do you feel any pain?"}, {"Patient": "No pain."}]

Example 4: PII Removal (Names & Dates)
Input: Hello, Mary Johnson. I see you were born on March 15, 1978. Yes, that's correct.
Output: [{"Doctor": "Hello, [NAME]. I see you were born on [REDACTED]."}, {"Patient": "Yes, that's correct."}]

Example 4b: Doctor Name Removal
Input: I'm Dr. Prasad. How can I help you today? I've been having headaches.
Output: [{"Doctor": "I'm [NAME]. How can I help you today?"}, {"Patient": "I've been having headaches."}]

Example 4c: Medication Names MUST Be Preserved
Input: Are you on any medications? Yes, metformin and jardiance. Also lisinopril, 10 milligrams.
Output: [{"Doctor": "Are you on any medications?"}, {"Patient": "Yes, metformin and jardiance. Also lisinopril, 10 milligrams."}]
Note: Medication names (metformin, jardiance, lisinopril) are NOT removed - they are medical terms, not PII.

Example 5: Remove Sound Effects and Narrative
Input: Doctor: The door closes with a gentle thump. Ah, Sarah, take a seat. Patient: The chair scrapes loudly. Thank you, doctor.
Output: [{"Doctor": "Ah, [NAME], take a seat."}, {"Patient": "Thank you, doctor."}]
Note: Sound effects ("thump", "scrapes loudly") and narrative ("The door closes") are removed.

Example 6: Ignore Incorrect Labels and Use Context
Input: Patient: When did the pain start? Doctor: I've been having chest pain for three days.
Output: [{"Doctor": "When did the pain start?"}, {"Patient": "I've been having chest pain for three days."}]
Note: The input labels were wrong - the question is from Doctor, the first-person experience is from Patient.

Example 7: Context-Based Question-Answer Pattern
Input: Doctor: Hi I'm Dr. Patient: Prasad. Patient: It's nice to meet you. Patient: It looks like you're new to the clinic. Doctor: I reviewed your past medical notes, so how can I help? Patient: When were you diagnosed with diabetes? Doctor: About five years ago.
Output: [{"Doctor": "Hi I'm [NAME]."}, {"Doctor": "It's nice to meet you."}, {"Doctor": "It looks like you're new to the clinic."}, {"Doctor": "I reviewed your past medical notes, so how can I help?"}, {"Patient": "When were you diagnosed with diabetes?"}, {"Doctor": "About five years ago."}]
Note: "Hi I'm Dr." + "Prasad" should be combined as one Doctor turn. "It's nice to meet you" and "It looks like you're new" are Doctor statements. "When were you diagnosed" is a Doctor question (not Patient). "About five years ago" is Patient answering.

Example 8: Correct Question-Answer Attribution
Input: Doctor: Are you on any medications? Patient: Yes, I'm on metformin. Doctor: When did you have your last A1C checked? Patient: It was about nine months ago.
Output: [{"Doctor": "Are you on any medications?"}, {"Patient": "Yes, I'm on metformin."}, {"Doctor": "When did you have your last A1C checked?"}, {"Patient": "It was about nine months ago."}]
Note: Questions come from Doctor, answers come from Patient. ALWAYS alternate after a question.

Example 9: Handle Already-Redacted Names
Input: Doctor: Hello, [NAME]. How can I help? Patient: I'm here to see Dr. [NAME].
Output: [{"Doctor": "Hello, [NAME]. How can I help?"}, {"Patient": "I'm here to see [NAME]."}]
Note: [NAME] placeholders are kept as-is, but "Dr. [NAME]" becomes just "[NAME]" to avoid redundancy.

Example 10: Family Member
Input: How has mom been sleeping lately? She tosses and turns all night.
Output: [{"Family Member": "How has mom been sleeping lately?"}, {"Doctor": "She tosses and turns all night."}]

Example 11: Medical Explanations are from Doctor
Input: Patient: As complex sugars, they get broken down and turn into simple sugars which increases your blood glucose level. Patient: I know it gets hard at times, but start by reducing small amounts of carbs.
Output: [{"Doctor": "As complex sugars, they get broken down and turn into simple sugars which increases your blood glucose level."}, {"Doctor": "I know it gets hard at times, but start by reducing small amounts of carbs."}]
Note: Medical explanations and treatment recommendations are from Doctor, not Patient.

Example 12: Physical Exam Instructions
Input: Doctor: Can you move your shoulder up and down? Patient: Okay, great. Doctor: Can you move your arm front to back? Patient: Great. Patient: I did not see any restrictions.
Output: [{"Doctor": "Can you move your shoulder up and down?"}, {"Patient": "Okay, great."}, {"Doctor": "Can you move your arm front to back?"}, {"Patient": "Great."}, {"Doctor": "I did not see any restrictions."}]
Note: Exam instructions and observations are from Doctor. Patient responses are short confirmations.

✅ QUALITY CHECKLIST
Before outputting, verify:
□ All text preserved exactly as provided
□ Only appropriate personal identifiers removed
□ Speaker labels match content context
□ Logical conversation flow maintained
□ Valid JSON format
□ No invented dialogue or speakers
□ Complete transcript processed (no truncation)

FINAL INSTRUCTION
Output ONLY the JSON array. Do not include explanatory text, confidence scores, or metadata. The response must begin with [ and end with ]."""

        import json as _json
        sentences = [_s.strip() for _s in _re.split(r"(?<=[.!?])\s+", raw) if _s.strip()]
        # Only treat as GPT-4 if it's explicitly gpt-4 (not gpt-4o-mini or other variants)
        is_gpt4 = str(deployment_name).startswith("gpt-4") and "mini" not in str(deployment_name).lower() and "o" not in str(deployment_name).lower()
        # Optimize for gpt-4o-mini: smaller chunks, lower token limits
        max_chars_per_chunk = 8000 if is_gpt4 else 3500  # Further reduced for gpt-4o-mini
        overlap_chars = 500

        if len(raw) <= max_chars_per_chunk:
            if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
                user_prompt = (
                    "TRANSCRIPCIÓN DE CONSULTA MÉDICA:\n"
                    f"{raw}\n\n"
                    "TAREA: Convierte esta transcripción en diálogo estructurado Doctor-Paciente.\n"
                    "• Preserva TODO el texto literalmente - no modifiques, parafrasees o corrijas\n"
                    "• Usa análisis basado en contexto: analiza el turno previo para determinar el hablante\n"
                    "• Elimina SOLO identificadores personales independientes (nombres, números de teléfono, direcciones, fechas específicas, SSN)\n"
                    "• Devuelve un objeto JSON con clave 'dialogue' conteniendo el arreglo, o devuelve el arreglo directamente\n\n"
                    "SALIDA: Arreglo JSON válido que empiece con [ y termine con ]"
                )
            else:
                user_prompt = (
                    "MEDICAL CONSULTATION TRANSCRIPT:\n"
                    f"{raw}\n\n"
                    "TASK: Convert this transcript into structured Doctor-Patient dialogue.\n"
                    "• Preserve ALL text verbatim - do not modify, paraphrase, or correct\n"
                    "• Use context-based analysis: analyze previous turn to determine speaker\n"
                    "• Remove ONLY standalone personal identifiers (names, phone numbers, addresses, specific dates, SSN)\n"
                    "• Return a JSON object with key 'dialogue' containing the array, or return the array directly\n\n"
                    "OUTPUT: Valid JSON array starting with [ and ending with ]"
                )

            async def _call_openai() -> str:
                try:
                    resp = await client.chat.completions.create(
                        model=deployment_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4000 if is_gpt4 else 3000,  # Increased for gpt-4o-mini to handle longer responses
                        temperature=0.0,
                        response_format={"type": "json_object"},  # enforce strict JSON when supported
                    )
                except Exception:
                    # Fallback without response_format if unsupported
                    resp = await client.chat.completions.create(
                        model=deployment_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4000 if is_gpt4 else 3000,  # Increased for gpt-4o-mini
                        temperature=0.0,
                    )
                return (resp.choices[0].message.content or "").strip()

            content = await _call_openai()
        else:
            chunks: List[str] = []
            current_chunk = ""
            for s in sentences:
                if len(current_chunk) + len(s) + 1 > max_chars_per_chunk and current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_start = max(0, len(current_chunk) - overlap_chars)
                    current_chunk = current_chunk[overlap_start:] + " " + s
                else:
                    current_chunk += (" " + s) if current_chunk else s
            if current_chunk:
                chunks.append(current_chunk.strip())

            async def _call_openai_chunk(text: str, prev_turns: Optional[List[Dict[str, str]]] = None) -> str:
                # Prepare brief context from previous chunk's last few turns
                context_block = ""
                if prev_turns:
                    last = prev_turns[-4:]
                    ctx_lines = []
                    for t in last:
                        sp = list(t.keys())[0]
                        tx = list(t.values())[0]
                        ctx_lines.append(f"{sp}: {tx}")
                    context_block = ("PREVIOUS DIALOGUE CONTEXT (maintain continuity and correct speakers):\n" + "\n".join(ctx_lines) + "\n\n")
                if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
                    user_prompt = (
                        f"{context_block}"
                        "FRAGMENTO DE TRANSCRIPCIÓN (Parte de conversación más larga):\n"
                        f"{text}\n\n"
                        "TAREA: Convierte este fragmento en diálogo estructurado Doctor-Paciente.\n"
                        "• Preserva TODO el texto literalmente - no modifiques, parafrasees o corrijas\n"
                        "• Usa análisis basado en contexto: analiza el turno previo para determinar el hablante\n"
                        "• Esto es parte de una conversación más larga - mantén continuidad\n"
                        "• Elimina SOLO identificadores personales independientes (nombres, números de teléfono, direcciones, fechas específicas, SSN)\n"
                        "• Devuelve un objeto JSON con clave 'dialogue' conteniendo el arreglo, o devuelve el arreglo directamente\n\n"
                        "SALIDA: Arreglo JSON válido que empiece con [ y termine con ]"
                    )
                else:
                    user_prompt = (
                        f"{context_block}"
                        "TRANSCRIPT CHUNK (Part of larger conversation):\n"
                        f"{text}\n\n"
                        "TASK: Convert this chunk into structured Doctor-Patient dialogue.\n"
                        "• Preserve ALL text verbatim - do not modify, paraphrase, or correct\n"
                        "• Use context-based analysis: analyze previous turn to determine speaker\n"
                        "• This is part of a larger conversation - maintain continuity\n"
                        "• Remove ONLY standalone personal identifiers (names, phone numbers, addresses, specific dates, SSN)\n"
                        "• Return a JSON object with key 'dialogue' containing the array, or return the array directly\n\n"
                        "OUTPUT: Valid JSON array starting with [ and ending with ]"
                    )
                try:
                    resp = await client.chat.completions.create(
                        model=deployment_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4000 if is_gpt4 else 3000,  # Increased for gpt-4o-mini
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    resp = await client.chat.completions.create(
                        model=deployment_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4000 if is_gpt4 else 3000,  # Increased for gpt-4o-mini
                        temperature=0.0,
                    )
                return (resp.choices[0].message.content or "").strip()

            def _extract_json_array(text: str) -> Optional[List[Dict[str, str]]]:
                try:
                    # Prefer JSON object with 'dialogue'
                    parsed = _json.loads(text)
                    if isinstance(parsed, dict) and isinstance(parsed.get("dialogue"), list):
                        return parsed["dialogue"]  # type: ignore
                    if isinstance(parsed, list):
                        return parsed  # type: ignore
                except Exception:
                    pass
                # Try to extract the first top-level JSON array substring
                try:
                    m = _re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
                    if m:
                        arr = _json.loads(m.group(0))
                        if isinstance(arr, list):
                            return arr  # type: ignore
                    # Try to extract object with dialogue key
                    m2 = _re.search(r"\{[\s\S]*?\"dialogue\"\s*:\s*\[[\s\S]*?\][\s\S]*?\}", text)
                    if m2:
                        obj = _json.loads(m2.group(0))
                        if isinstance(obj, dict) and isinstance(obj.get("dialogue"), list):
                            return obj["dialogue"]  # type: ignore
                except Exception:
                    pass
                return None

            parts: List[Dict[str, str]] = []
            prev_turns_for_context: List[Dict[str, str]] = []
            for ch in chunks:
                chunk_result = await _call_openai_chunk(ch, prev_turns_for_context)
                parsed = _extract_json_array(chunk_result)
                if isinstance(parsed, list):
                    parts.extend(parsed)
                    # update context with last few turns of this parsed chunk
                    for it in parsed[-4:]:
                        prev_turns_for_context.append(it)
                        # Keep context window modest
                        if len(prev_turns_for_context) > 8:
                            prev_turns_for_context = prev_turns_for_context[-8:]

            # Merge trivial consecutive duplicates
            merged: List[Dict[str, str]] = []
            for item in parts:
                if not merged:
                    merged.append(item)
                    continue
                try:
                    if (
                        len(item) == 1
                        and len(merged[-1]) == 1
                        and list(item.keys())[0] == list(merged[-1].keys())[0]
                        and list(item.values())[0] == list(merged[-1].values())[0]
                    ):
                        continue
                except Exception:
                    pass
                merged.append(item)
            import json as _json2
            if not merged:
                # Heuristic fallback if model returned nothing useful
                turns: List[Dict[str, str]] = []
                patient_label = "Paciente" if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"] else "Patient"
                next_role = "Doctor"
                for s in sentences:
                    low = s.lower()
                    if low.startswith("doctor:") or low.startswith("doctora:"):
                        turns.append({"Doctor": s.split(":", 1)[1].strip()})
                        next_role = patient_label
                    elif low.startswith("patient:") or low.startswith("paciente:"):
                        turns.append({patient_label: s.split(":", 1)[1].strip()})
                        next_role = "Doctor"
                    else:
                        turns.append({next_role: s})
                        next_role = patient_label if next_role == "Doctor" else "Doctor"
                return turns
            content = _json2.dumps(merged)

        import json
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and isinstance(parsed.get("dialogue"), list):
                dialogue_list = parsed["dialogue"]
            elif isinstance(parsed, list):
                dialogue_list = parsed
            else:
                raise ValueError("Invalid response format")
            
            # Post-process to fix common errors and remove duplicates
            fixed_dialogue: List[Dict[str, str]] = []
            patient_label = "Paciente" if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"] else "Patient"
            seen_texts = set()  # Track seen dialogue to remove duplicates
            
            for i, turn in enumerate(dialogue_list):
                if not isinstance(turn, dict) or len(turn) != 1:
                    continue
                
                speaker = list(turn.keys())[0]
                original_text_value = list(turn.values())[0]
                text = (original_text_value or "").strip()
                
                # Extract spoken content or drop pure narrative/noise lines
                spoken = _extract_spoken_or_none(text)
                if not spoken:
                    continue
                text = spoken
                
                if not text:
                    continue
                
                # Remove duplicates (exact match)
                text_normalized = text.lower().strip()
                
                # Check for exact duplicates first
                if text_normalized in seen_texts:
                    # Check if previous occurrence was recent (within last 10 turns) - likely a duplicate
                    is_duplicate = False
                    for j in range(max(0, len(fixed_dialogue) - 10), len(fixed_dialogue)):
                        prev_turn_text = list(fixed_dialogue[j].values())[0].lower().strip()
                        if prev_turn_text == text_normalized:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        continue
                
                seen_texts.add(text_normalized)
                
                # Fuzzy dedupe against recent turns (avoid near-identical repeats)
                is_fuzzy_dup = False
                for j in range(max(0, len(fixed_dialogue) - 10), len(fixed_dialogue)):
                    prev_text_j = list(fixed_dialogue[j].values())[0]
                    try:
                        if SequenceMatcher(None, prev_text_j.lower().strip(), text_normalized).ratio() >= 0.97:
                            is_fuzzy_dup = True
                            break
                    except Exception:
                        continue
                if is_fuzzy_dup:
                    continue
                
                # Fix obvious speaker errors based on context
                prev_speaker = list(fixed_dialogue[-1].keys())[0] if fixed_dialogue else None
                prev_text = list(fixed_dialogue[-1].values())[0] if fixed_dialogue else ""
                
                # AGGRESSIVE POST-PROCESSING TO FIX SPEAKER ERRORS
                # Apply rules in priority order for best results
                
                # Rule 1: If current is a question, it's almost always Doctor (STRICT) - CHECK FIRST
                if text.endswith("?"):
                    # Only allow Patient questions if they're clarifications
                    if not any(patient_q in text.lower()[:50] for patient_q in ["what does that mean", "is it serious", "how long", "do i need"]):
                        if speaker == patient_label:
                            speaker = "Doctor"  # Force to Doctor
                
                # Rule 2: If previous was Doctor asking question, current should be Patient (STRICT)
                if prev_speaker == "Doctor" and prev_text.endswith("?"):
                    if speaker == "Doctor" and not text.endswith("?"):
                        speaker = patient_label  # Force to Patient
                
                # Rule 3: Medical explanations and treatment plans are from Doctor (STRICT)
                medical_keywords = ["A1C", "hemoglobin", "glucose", "blood pressure", "treatment", "therapy", "injection", 
                                   "examination", "recommend", "should", "will order", "will refer", "plan", "lab",
                                   "prescribe", "refer", "order", "diagnosis", "medication", "exams", "appointment",
                                   "translates to", "ballpark", "expectation", "average", "range"]
                medical_phrases = ["I'll order", "I'll refer", "I'll prescribe", "we will", "we can", "let's start",
                                  "I want your", "I will send", "I will perform", "let me", "I'll go ahead"]
                
                has_medical_keyword = any(kw.lower() in text.lower() for kw in medical_keywords)
                has_medical_phrase = any(phrase.lower() in text.lower() for phrase in medical_phrases)
                
                if (has_medical_keyword or has_medical_phrase) and speaker == patient_label:
                    # Check if it's actually patient experience
                    if not any(patient_signal in text[:50].lower() for patient_signal in 
                              ["i have", "i feel", "i take", "my pain", "my symptoms", "i'm on", "i brought"]):
                        speaker = "Doctor"  # Force to Doctor
                
                # Rule 4: First-person patient experiences are from Patient (STRICT)
                patient_starters = ["I have", "I feel", "I've been", "I take", "I'm on", "I went", "I try", "I brought",
                                   "My last", "My ", "I actually", "I haven't", "I did not", "Not that I"]
                if any(starter in text[:20] for starter in patient_starters) and speaker == "Doctor":
                    # But exclude doctor statements like "I'll order", "I'll refer"
                    if not any(doctor_phrase in text[:30].lower() for doctor_phrase in 
                              ["i'll order", "i'll refer", "i'll prescribe", "i will perform", "i'll go ahead"]):
                        speaker = patient_label  # Force to Patient
                
                # Rule 5: Exam instructions and observations are from Doctor (STRICT)
                exam_keywords = ["can you", "move your", "raise", "lift", "let me", "I'll perform", "I will perform", 
                                "examine", "take a look", "I did not see", "I see you", "it appears", "I'm not suspecting",
                                "I do suspect", "I highly suspect"]
                if any(kw.lower() in text.lower()[:40] for kw in exam_keywords):
                    if speaker == patient_label:
                        speaker = "Doctor"  # Force to Doctor
                
                # Rule 6: Short confirmations after Doctor questions are from Patient (STRICT)
                if prev_speaker == "Doctor" and prev_text.endswith("?"):
                    if text.lower().strip() in ["yes", "no", "okay", "great", "fine", "alright", "sure", "it's about"]:
                        speaker = patient_label  # Force to Patient
                
                # Rule 7: Treatment recommendations and plans are from Doctor (STRICT)
                treatment_phrases = ["we will continue", "we can work on", "we can try", "we decided", "we'll follow up",
                                    "the plan for", "my recommendation", "I want your", "I also informed you"]
                if any(phrase.lower() in text.lower() for phrase in treatment_phrases) and speaker == patient_label:
                    speaker = "Doctor"  # Force to Doctor
                
                # Rule 8: Patient statements about their own actions/decisions
                patient_decisions = ["I want to", "I feel if", "I like that", "I guess I could"]
                if any(phrase.lower() in text.lower()[:30] for phrase in patient_decisions) and speaker == "Doctor":
                    speaker = patient_label  # Force to Patient
                
                # Rule 9: "I will perform", "I'll perform", "I will do" are from Doctor
                if any(phrase in text.lower()[:40] for phrase in ["i will perform", "i'll perform", "i will do", "before you leave"]):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # Rule 10: "I see you", "I did not see", "it appears" are from Doctor (observations)
                if any(phrase in text.lower()[:30] for phrase in ["i see you", "i did not see", "it appears", "i'm not suspecting", "i do suspect"]):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # Rule 11: Answers to questions are from Patient
                if prev_speaker == "Doctor" and prev_text.endswith("?"):
                    # If it's a direct answer (not a question itself)
                    if not text.endswith("?") and speaker == "Doctor":
                        # Check if it's actually a patient answer
                        answer_indicators = ["about", "yes", "no", "it was", "i have", "i'm on", "my last", "not that"]
                        if any(indicator in text.lower()[:20] for indicator in answer_indicators):
                            speaker = patient_label
                
                # Rule 12: "We can", "We will", "We decided" at start are from Doctor
                if text.strip().startswith(("We can", "We will", "We decided", "We'll")):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                fixed_dialogue.append({speaker: text})
            
            # SECOND PASS: Additional validation and fixes
            final_dialogue: List[Dict[str, str]] = []
            for i, turn in enumerate(fixed_dialogue):
                speaker = list(turn.keys())[0]
                text = list(turn.values())[0]
                
                # Get previous and next context
                prev_turn = final_dialogue[-1] if final_dialogue else None
                prev_speaker = list(prev_turn.keys())[0] if prev_turn else None
                prev_text = list(prev_turn.values())[0] if prev_turn else ""
                
                # Additional fixes based on patterns from user's examples
                
                # Fix: Questions are Doctor (unless very specific patient clarification)
                if text.endswith("?") and speaker == patient_label:
                    if not any(q in text.lower()[:30] for q in ["what does that mean", "is it serious", "do i need"]):
                        speaker = "Doctor"
                
                # Fix: Medical explanations with "translates to", "ballpark", "expectation"
                if any(phrase in text.lower() for phrase in ["translates to", "ballpark", "expectation", "usually translates"]):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # Fix: "I'll order", "I want your", "I also informed you" are Doctor
                if any(phrase in text.lower()[:40] for phrase in ["i'll order", "i want your", "i also informed", "for my recommendation"]):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # Fix: "So you came in", "So you will be" are Doctor (summarizing)
                if text.strip().startswith(("So you", "So the", "So by")):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # Fix: "Let's talk about", "Let's start" are Doctor
                if text.strip().startswith(("Let's talk", "Let's start", "Let me")):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # Fix: Direct answers after questions
                if prev_speaker == "Doctor" and prev_text.endswith("?"):
                    if not text.endswith("?") and speaker == "Doctor":
                        # It's likely a Patient answer
                        if any(indicator in text.lower()[:30] for indicator in 
                              ["about", "yes", "no", "it was", "i have", "i'm on", "my last", "not that", "i actually"]):
                            speaker = patient_label
                
                # Fix: "Okay, now let's" is Doctor
                if text.strip().startswith(("Okay, now", "Now let's")):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # Fix: Specific patterns from user examples
                # "When were you diagnosed" is a Doctor question
                if "when were you" in text.lower() or "when did you" in text.lower():
                    if text.endswith("?") and speaker == patient_label:
                        speaker = "Doctor"
                
                # "About five years ago", "About nine months ago" are Patient answers
                if text.strip().startswith(("About ", "It was about")):
                    if speaker == "Doctor" and not text.endswith("?"):
                        speaker = patient_label
                
                # "I'll order", "I want your" at start are Doctor
                if text.strip().startswith(("I'll order", "I want your", "I also informed")):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # "For my recommendation" is Doctor
                if "for my recommendation" in text.lower():
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # "Just to recap" is Doctor
                if text.strip().startswith(("Just to recap", "To recap")):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # "Please pick up", "Please get" are Doctor
                if text.strip().startswith(("Please pick", "Please get")):
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # "Do you have any questions" is Doctor
                if "do you have any questions" in text.lower():
                    if speaker == patient_label:
                        speaker = "Doctor"
                
                # "Thank you doctor" is Patient
                if "thank you doctor" in text.lower() or text.lower().strip() == "thank you":
                    if speaker == "Doctor":
                        speaker = patient_label
                
                # "No questions" is Patient
                if text.lower().strip() in ["no questions", "no question"]:
                    if speaker == "Doctor":
                        speaker = patient_label
                
                final_dialogue.append({speaker: text})
            
            return final_dialogue
            
        except Exception:
            # Heuristic fallback: alternate speakers
            turns: List[Dict[str, str]] = []
            patient_label = "Paciente" if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"] else "Patient"
            next_role = "Doctor"
            for s in sentences:
                low = s.lower()
                if low.startswith("doctor:") or low.startswith("doctora:"):
                    turns.append({"Doctor": s.split(":", 1)[1].strip()})
                    next_role = patient_label
                elif low.startswith("patient:") or low.startswith("paciente:"):
                    turns.append({patient_label: s.split(":", 1)[1].strip()})
                    next_role = "Doctor"
                else:
                    turns.append({next_role: s})
                    next_role = patient_label if next_role == "Doctor" else "Doctor"
            return turns
    except Exception:
        return None


