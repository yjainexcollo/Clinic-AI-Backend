"""
Shared helper to structure raw transcript into Doctor/Patient dialogue.

This mirrors the logic used in the visit transcript route so both visit and
ad-hoc flows produce consistent outputs.
"""

from typing import List, Dict, Optional
import asyncio
import re as _re


async def structure_dialogue_from_text(raw: str, *, model: str, api_key: str, language: str = "en") -> Optional[List[Dict[str, str]]]:
    if not raw:
        return None
    try:
        # Local import to avoid import cost when unused
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)

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

1. ANÁLISIS BASADO EN CONTEXTO (MÁS IMPORTANTE - 95% precisión)
   • SIEMPRE analiza el turno PREVIO para determinar el hablante
   • Si el turno anterior fue Doctor haciendo pregunta → la siguiente respuesta es Paciente
   • Si el turno anterior fue Paciente respondiendo → la siguiente declaración es Doctor
   • Patrón de examen físico: instrucción del Doctor → respuesta del Paciente → observación del Doctor
   • Flujo de conversación: Doctor saluda → Paciente indica razón → Doctor pregunta → Paciente responde → Doctor examina → Paciente responde → Doctor resume → Paciente confirma

2. SEÑALES DEL DOCTOR (99% precisión cuando están presentes)
   • Preguntas (interrogativas): "¿Cuándo...?", "¿Cuánto tiempo...?", "¿Puedes...?", "¿Qué...?", "¿Alguna...?"
   • Instrucciones (imperativas): "Déjame...", "Voy a...", "Vamos a...", "Puede mover...", "Levante...", "Resista..."
   • Evaluaciones clínicas: "Veo...", "No veo...", "Parece...", "Es una buena señal", "Sospecho..."
   • Terminología médica: nombres de fármacos, términos anatómicos, diagnósticos, procedimientos
   • Declaraciones de autoridad: "Recomiendo", "Debe", "Es importante", "Necesitamos"
   • Plan/prescripción: "Voy a ordenar", "Voy a prescribir", "Voy a referir", "Vamos a programar"
   • Comandos de examen: "Mueva su...", "Levante...", "Resista...", "¿Puede sentir...?", "¿Siente algún dolor?"
   • Saludos/aperturas: "Hola soy el Dr.", "Mucho gusto", "¿En qué puedo ayudarle?"

3. SEÑALES DEL PACIENTE (99% precisión cuando están presentes)
   • Experiencias en primera persona: "Tengo", "Siento", "He estado", "Tomé", "Fui", "Estoy aquí por"
   • Respuestas directas: "Sí", "No", "Alrededor de...", "Fue...", "No..."
   • Descripciones de síntomas: "Me duele", "Es doloroso", "Comenzó...", "Empeora cuando..."
   • Historia personal: "Usualmente...", "Trato de...", "No he...", "Mi última..."
   • Respuestas a instrucciones: "Bien", "Sí doctor", "No duele", "Está bien", "De acuerdo" (DESPUÉS del comando del doctor)
   • Confirmación: "Sí, está bien", "Entiendo", "Comprendo", "Suena bien"
   • Preguntas al doctor: "¿Qué significa eso?", "¿Es grave?", "¿Cuánto tiempo...?", "¿Necesito...?"

4. SEÑALES DE MIEMBRO DE LA FAMILIA
   • Referencias en tercera persona al paciente: "¿Cómo ha estado mamá...?", "Ella mencionó...", "Él dijo..."
   • Auto-identificación: "Soy su hija", "Soy su esposa"
   • Perspectiva externa: "Ella ha tenido problemas...", "Él no duerme bien"

5. ÁRBOL DE DECISIÓN PARA CASOS AMBIGUOS
   • Contiene signo de interrogación (?) → probablemente Doctor preguntando
   • Empieza con "Yo" + verbo + experiencia personal → Paciente
   • Contiene términos médicos (diagnóstico, nombres de fármacos) → probablemente Doctor explicando
   • Respuesta corta ("Bien", "Excelente", "Sí") DESPUÉS de instrucción del doctor → Paciente
   • Describe lo que el doctor hará ("Voy a...", "Vamos a...") → Doctor
   • Respuestas de una palabra ("Sí", "Bien") → asignar al respondedor lógico basado en la pregunta precedente
   • Si no está seguro → verifica CONTEXTO: ¿qué se dijo antes?

⚠️ CASOS ESPECIALES Y MANEJO DE ERRORES
• Audio poco claro: Preserva [inaudible] o [poco claro] exactamente, asigna basado en contexto circundante
• Entrada mal etiquetada: Re-etiqueta basado en análisis de contenido, confía en contenido sobre etiquetas originales
• Discusión administrativa: Asigna a quien inició el tema
• Múltiples miembros de familia: Usa solo etiqueta "Miembro de la Familia" (sin distinciones como "Miembro de la Familia 1")
• Interrupciones: Etiqueta la porción de cada hablante por separado
• Turnos extendidos: Permite monólogos más largos cuando sea contextualmente apropiado (descripciones detalladas de síntomas, explicaciones de tratamiento)

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

Rule 2: PERSONAL IDENTIFIER HANDLING
Remove ALL personal identifiers to protect privacy:
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

1. CONTEXT-BASED ANALYSIS (MOST IMPORTANT - 95% accuracy)
   • ALWAYS analyze the PREVIOUS turn to determine speaker
   • If previous turn was Doctor asking question → next response is Patient
   • If previous turn was Patient answering → next statement is Doctor
   • Physical exam pattern: Doctor instruction → Patient response → Doctor observation
   • Conversation flow: Doctor greets → Patient states reason → Doctor asks → Patient answers → Doctor examines → Patient responds → Doctor summarizes → Patient confirms

2. DOCTOR SIGNALS (99% accuracy when present)
   • Questions (interrogative): "When...?", "How long...?", "Can you...?", "What...?", "Any...?"
   • Instructions (imperative): "Let me...", "I'll...", "We'll...", "Can you move...", "Raise your...", "Resist against..."
   • Clinical assessments: "I see...", "I don't see...", "It appears...", "That's a good sign", "I suspect..."
   • Medical terminology: drug names, anatomical terms, diagnoses, procedures
   • Authority statements: "I recommend", "You should", "It's important", "We need to"
   • Plan/prescription: "I'll order", "I'll prescribe", "I'll refer", "We'll schedule"
   • Exam commands: "Move your...", "Raise...", "Resist...", "Can you feel...", "Do you feel any pain?"
   • Greetings/openings: "Hi I'm Dr.", "Nice to meet you", "How can I help?"

3. PATIENT SIGNALS (99% accuracy when present)
   • First-person experiences: "I have", "I feel", "I've been", "I took", "I went", "I'm here for"
   • Direct answers: "Yes", "No", "About...", "It was...", "I don't..."
   • Symptom descriptions: "It hurts", "It's painful", "It started...", "It gets worse when..."
   • Personal history: "I usually...", "I try to...", "I haven't...", "My last..."
   • Responses to instructions: "Okay", "Yes doctor", "No pain", "That's fine", "Alright" (AFTER doctor's command)
   • Confirmation: "Yes, that's okay", "I understand", "Got it", "Sounds good"
   • Questions to doctor: "What does that mean?", "Is it serious?", "How long...?", "Do I need...?"

4. FAMILY MEMBER SIGNALS
   • Third-person references to patient: "How has mom been...?", "She mentioned...", "He said..."
   • Self-identification: "I'm her daughter", "I'm his wife"
   • External perspective: "She's been having trouble...", "He doesn't sleep well"

5. DECISION TREE FOR AMBIGUOUS CASES
   • Contains question mark (?) → likely Doctor asking
   • Starts with "I" + verb + personal experience → Patient
   • Contains medical terms (diagnosis, drug names) → likely Doctor explaining
   • Short response ("Okay", "Great", "Yes") AFTER doctor's instruction → Patient
   • Describes what doctor will do ("I'll...", "We'll...") → Doctor
   • Single-word responses ("Yes", "Okay") → assign to logical responder based on preceding question
   • If unsure → check CONTEXT: what was said before?

⚠️ EDGE CASES & ERROR HANDLING
• Unclear audio: Preserve [inaudible] or [unclear] exactly, assign based on surrounding context
• Mislabeled input: Relabel based on content analysis, trust content over original labels
• Administrative discussion: Assign to whoever initiated the topic
• Multiple family members: Use only "Family Member" label (no distinctions like "Family Member 1")
• Interruptions: Label each speaker's portion separately
• Extended turns: Permit longer monologues when contextually appropriate (detailed symptom descriptions, treatment explanations)

📤 OUTPUT REQUIREMENTS
• Output ONLY valid JSON array: [{"Doctor": "..."}, {"Patient": "..."}]
• NO markdown, NO code blocks, NO explanations, NO comments
• NO ```json``` wrapper - start directly with [
• Each turn = ONE complete thought or response
• Process COMPLETE transcript - include ALL dialogue turns
• DO NOT truncate or stop early
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

Example 5: Family Member
Input: How has mom been sleeping lately? She tosses and turns all night.
Output: [{"Family Member": "How has mom been sleeping lately?"}, {"Doctor": "She tosses and turns all night."}]

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
        is_gpt4 = str(model).startswith("gpt-4")
        max_chars_per_chunk = 8000 if is_gpt4 else 6000
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

            def _call_openai() -> str:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4000 if is_gpt4 else 2000,
                        temperature=0.0,
                        response_format={"type": "json_object"},  # enforce strict JSON when supported
                    )
                except Exception:
                    # Fallback without response_format if unsupported
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4000 if is_gpt4 else 2000,
                        temperature=0.0,
                    )
                return (resp.choices[0].message.content or "").strip()

            content = await asyncio.to_thread(_call_openai)
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

            def _call_openai_chunk(text: str) -> str:
                if (language or "en").lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
                    user_prompt = (
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
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4000 if is_gpt4 else 2000,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4000 if is_gpt4 else 2000,
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
            for ch in chunks:
                chunk_result = await asyncio.to_thread(_call_openai_chunk, ch)
                parsed = _extract_json_array(chunk_result)
                if isinstance(parsed, list):
                    parts.extend(parsed)

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
                return parsed["dialogue"]
            if isinstance(parsed, list):
                return parsed
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


