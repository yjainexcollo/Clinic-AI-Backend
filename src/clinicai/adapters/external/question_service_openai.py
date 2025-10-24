import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from clinicai.application.ports.services.question_service import QuestionService
from clinicai.core.config import get_settings
from clinicai.core.helicone_client import create_helicone_client


# ----------------------
# Pure LLM Approach - No Templates Needed
# ----------------------

# ----------------------
# OpenAI QuestionService
# ----------------------
class OpenAIQuestionService(QuestionService):
    def __init__(self) -> None:
        import os
        from pathlib import Path
        from dotenv import load_dotenv

        try:
            # dotenv is optional but helps in local development
            from dotenv import load_dotenv  # type: ignore
        except Exception:
            load_dotenv = None

        # Load project settings (includes model name, tokens, temperature, etc.)
        self._settings = get_settings()
        self._debug_prompts = getattr(self._settings, "debug_prompts", False)

        api_key = self._settings.openai.api_key or os.getenv("OPENAI_API_KEY", "")
        self._mode = (os.getenv("QUESTION_MODE", "autonomous") or "autonomous").strip().lower()

        if not api_key and load_dotenv is not None:
            cwd = Path(os.getcwd()).resolve()
            for parent in [cwd, *cwd.parents]:
                candidate = parent / ".env"
                if candidate.exists():
                    load_dotenv(dotenv_path=str(candidate), override=False)
                    api_key = os.getenv("OPENAI_API_KEY", "")
                    if api_key:
                        break
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        # Use Helicone client for AI observability
        self._client = create_helicone_client()

    async def _chat_completion(
        self, messages: List[Dict[str, str]], max_tokens: int = 64, temperature: float = 0.3,
        patient_id: str = None, prompt_name: str = None
    ) -> str:
        logger = logging.getLogger("clinicai")
        try:
            if self._debug_prompts:
                logger.debug("[QuestionService] Sending messages to OpenAI:\n%s", messages)

            # Use Helicone client with tracking
            resp, metrics = await self._client.chat_completion(
                model=self._settings.openai.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                patient_id=patient_id,
                prompt_name=prompt_name or "question_service",
                custom_properties={
                    "service": "question_generation",
                    "message_count": len(messages)
                }
            )
            output = resp.choices[0].message.content.strip()

            if self._debug_prompts:
                logger.debug("[QuestionService] Received response: %s", output)
                logger.debug(f"[QuestionService] Metrics: {metrics}")

            return output
        except Exception:
            logger.error("[QuestionService] OpenAI call failed", exc_info=True)
            return ""

    # ----------------------
    # AI-powered classifier
    # ----------------------
    async def _classify_question(self, question: str) -> str:
        """Classify a question into a canonical category using AI for duplicate detection.

        This method uses the LLM to intelligently classify questions into categories
        to help prevent semantic duplicates while allowing contextually different
        questions in the same category.
        """
        if not question:
            return "other"

        # Use AI to classify the question
        classification_prompt = f"""You are a medical question classifier. Classify the following question into ONE of these categories:

CATEGORIES:
- duration: Questions about how long symptoms have been present
- pain: Questions about pain assessment, severity, location, triggers
- medications: Questions about current medications, prescriptions, drug history
- family: Questions about family medical history, hereditary conditions
- travel: Questions about recent travel, exposure history
- allergies: Questions about allergies, allergic reactions
- hpi: Questions about past medical history, previous conditions
- associated: Questions about associated symptoms, related symptoms
- chronic_monitoring: Questions about chronic disease monitoring, tests, screenings
- womens_health: Questions about women's health, menstrual, pregnancy, gynecological
- functional: Questions about daily activities, energy, appetite, weight, functional status
- lifestyle: Questions about lifestyle factors, smoking, drinking, exercise
- triggers: Questions about what makes symptoms worse or better
- temporal: Questions about past episodes, similar experiences, timing
- other: Any other category not listed above

QUESTION TO CLASSIFY: "{question}"

Return ONLY the category name (e.g., "duration", "pain", "medications", etc.). No explanations."""

        try:
            messages = [{"role": "user", "content": classification_prompt}]
            result = await self._chat_completion(messages, max_tokens=20, temperature=0.1)
            category = result.strip().lower()

            # Validate the category
            valid_categories = ["duration", "pain", "medications", "family", "travel", "allergies", 
                              "hpi", "associated", "chronic_monitoring", "womens_health", 
                              "functional", "lifestyle", "triggers", "temporal", "other"]

            if category in valid_categories:
                return category
            else:
                return "other"

        except Exception as e:
            logger = logging.getLogger("clinicai")
            logger.error(f"AI classification failed: {e}")
        return "other"

    # ----------------------
    # Question generation
    # ----------------------
    async def generate_first_question(self, disease: str, language: str = "en") -> str:
        if language == "sp":
            question = "¿Por qué ha venido hoy? ¿Cuál es la principal preocupación con la que necesita ayuda?"
            return question
        question = "Why have you come in today? What is the main concern you want help with?"
        return question

    async def generate_next_question(
        self,
        disease: str,
        previous_answers: List[str],
        asked_questions: List[str],
        current_count: int,
        max_count: int,
        asked_categories: Optional[List[str]] = None,
        recently_travelled: bool = False,
        prior_summary: Optional[Any] = None,
        prior_qas: Optional[List[str]] = None,
        patient_gender: Optional[str] = None,
        patient_age: Optional[int] = None,
        language: str = "en",
    ) -> str:
        # Force closing question at the end
        if current_count + 1 >= max_count:
            if language == "sp":
                return "¿Hemos pasado por alto algo importante sobre su salud o hay otras preocupaciones que desee que el médico sepa?"
            return "Have we missed anything important about your health, or any other concerns you want the doctor to know?"

        # Build prior context block
        prior_block = ""
        if prior_summary:
            ps = str(prior_summary)
            prior_block += f"Prior summary: {ps[:400]}\n"
        if prior_qas:
            prior_block += "Prior QAs: " + "; ".join(prior_qas[:6]) + "\n"

        # Build sophisticated prompt with detailed medical knowledge
        if language == "sp":
            system_prompt = f"""
SISTEMA

Eres un Asistente Inteligente de Admisión Clínica con conocimientos médicos avanzados. Conduce una entrevista médica completa pero enfocada, haciendo una sola pregunta clara y comprensible para el paciente a la vez, siempre terminando con un signo de interrogación.

1. Parámetros del Médico

El médico selecciona categorías y límites máximos de preguntas.
Debes mantenerte estrictamente dentro de estos límites.

Cuando el número de preguntas sea limitado, prioriza siempre la información médicamente esencial primero.

2. Contexto a Usar

Motivo(s) principal(es) de consulta: {disease or "N/A"}
Género del paciente: {patient_gender or "No especificado"}
Edad del paciente: {patient_age or "No especificada"}

ANÁLISIS OBLIGATORIO DE CONDICIONES:

PASO 1: Analiza el motivo de consulta "{disease}" y clasifica CADA condición presente:

CONDICIONES CRÓNICAS (requieren historia familiar y monitoreo):
- diabetes, hipertensión, cardiopatías, asma, trastornos tiroideos, cáncer, enfermedades renales, hepáticas, EPOC, artritis, dolor crónico, enfermedades autoinmunes, condiciones de salud mental

CONDICIONES AGUDAS (NO requieren historia familiar ni monitoreo):
- fiebre, resfriado, gripe, tos, dolor de garganta, infecciones agudas, lesiones, dolor agudo, náusea, vómito, diarrea, infecciones respiratorias agudas

CONDICIONES DE DOLOR (NO requieren historia familiar ni monitoreo):
- cefaleas, dolor corporal, dolor torácico, dolor lumbar, dolor articular, dolor muscular, dolor abdominal, dolor pélvico, dolor menstrual

CONDICIONES DE SALUD DE LA MUJER (requieren historia familiar si es mujer y apropiado para la edad):
- problemas menstruales, embarazo, menopausia, SOP, endometriosis, condiciones ginecológicas, problemas mamarios, dolor pélvico/abdominal en mujeres
- CONSIDERACIONES DE EDAD: Solo preguntar sobre temas menstruales para mujeres de 12-60 años

CONDICIONES ALÉRGICAS (requieren preguntas sobre alergias):
- reacciones alérgicas, problemas cutáneos, alergias alimentarias, ambientales, asma inducida por alergia

CONDICIONES RELACIONADAS CON VIAJES (requieren preguntas sobre viajes):
- condiciones asociadas a viajes recientes, exposición a nuevos ambientes, enfermedades infecciosas

PASO 2: Si hay MÚLTIPLES condiciones, aplica reglas para CADA una:
- Si hay condición CRÓNICA → preguntar historia familiar y monitoreo
- Si hay condición AGUDA → NO preguntar historia familiar ni monitoreo
- Si hay condición de DOLOR → NO preguntar historia familiar ni monitoreo
- Si hay condición de SALUD DE LA MUJER Y el paciente es MUJER Y edad 12-60 → preguntar historia familiar y preguntas menstruales
- Si hay condición de SALUD DE LA MUJER Y el paciente es MUJER Y edad <12 o >60 → preguntar historia familiar pero NO preguntas menstruales
- Si hay condición de SALUD DE LA MUJER Y el paciente es HOMBRE → NO preguntar historia familiar
- Si hay condición ALÉRGICA → preguntar sobre alergias y exposiciones
- Si hay condición RELACIONADA CON VIAJES → preguntar sobre viajes recientes

PASO 3: Para dolor abdominal/pélvico en MUJERES, considerar si puede ser relacionado con salud de la mujer:
- Si edad 12-60: Preguntar sobre ciclo menstrual, embarazo, historia ginecológica
- Si edad <12 o >60: Enfocarse en causas abdominales generales, evitar preguntas menstruales

PASO 4: Determina qué preguntas hacer basado en TODAS las condiciones identificadas.

Respuestas recientes: {', '.join(previous_answers[-3:])}
Preguntas ya realizadas: {asked_questions}

VALIDACIÓN OBLIGATORIA: Antes de generar CUALQUIER pregunta, debes:

Leer la lista "Preguntas ya realizadas"

Comprobar si tu pregunta propuesta es similar a alguna de esa lista

Si es similar, elige OTRA categoría y genera una pregunta DIFERENTE

NUNCA generes una pregunta igual o parecida a las ya realizadas

CRÍTICO: No repitas ninguna pregunta anterior. Cada pregunta debe ser única.

Categorías disponibles: {asked_categories or "TODAS"}
Progreso: {current_count}/{max_count}
¿Viaje reciente?: {recently_travelled}
Contexto previo: {prior_block or "ninguno"}

3. Reglas por Categoría

Síntomas asociados → Máx. 3

Monitoreo de enfermedades crónicas → Máx. 3 (ver detalles abajo)

Todas las demás categorías → Máx. 1

4. Filtrado Inteligente - REGLAS CRÍTICAS

**HISTORIA FAMILIAR**: SOLO para condiciones crónicas/genéticas (diabetes, hipertensión, cardiopatías, cáncer, asma, tiroides). NUNCA para condiciones agudas, dolor simple, o infecciones.

**MONITOREO CRÓNICO**: SOLO para enfermedades crónicas (diabetes, HTA, asma, tiroides, cardiopatías, cáncer). NUNCA para condiciones agudas, dolor simple, o infecciones.

**Antecedentes médicos**: solo si son relevantes

**Historia de viajes**: solo si recently_travelled=True y la condición lo amerita

**Alergias**: solo si hay síntomas alérgicos (erupciones, urticaria, sibilancias, rinitis)

**Dolor**: preguntar solo si forma parte del motivo de consulta

**REGLA CRÍTICA**: Si la condición es AGUDA (fiebre, tos, resfriado, infección, dolor agudo) → NO preguntar historia familiar ni monitoreo crónico.

5. Monitoreo de Enfermedades Crónicas (máx. 3)

Cubrir cada área una sola vez, escoger las 3 más relevantes:

Mediciones en casa → presión arterial, glucosa, flujo espiratorio, tiroides

Análisis recientes → HbA1c, función renal/hepática, colesterol, ECG

Cribados y complicaciones (combinados) → exámenes de ojos, pies, dentales + heridas, dolor torácico, disnea

Adherencia a medicamentos y efectos secundarios

CRÍTICO: Nunca separar cribados y complicaciones. Combinar siempre en una sola pregunta.

6. Flujo de Prioridad

Duración de síntomas → siempre primero

Remedios caseros + medicamentos actuales → siempre segundo (en una sola pregunta)

Preguntas específicas según clasificación de la enfermedad

Síntomas asociados (si relevante, máx. 3)

Monitoreo crónico (si aplica, máx. 3)

Otras categorías (máx. 1 cada una)

Pregunta de cierre → solo al final (después de ≥7 preguntas o cuando ya no haya más útiles)

7. Priorización Inteligente (cuando el médico selecciona categorías)

Condiciones crónicas

Alta prioridad: duración, medicamentos, mediciones, análisis, cribados/complicaciones, historia familiar

Media: síntomas asociados, estilo de vida, impacto funcional

Baja: viajes, alergias (si no son relevantes)

Chequeos específicos

Diabetes → ojos, pies, dientes, riñón

Hipertensión → corazón, riñón, ojos

Cardiopatía → pruebas cardíacas, colesterol, esfuerzo

Asma → función pulmonar, alergias, radiografía, flujo espiratorio

Tiroides → TSH, función tiroidea, cuello, frecuencia cardíaca

Cáncer → cribados, imágenes, marcadores

Salud de la mujer → Papanicolau, mamografía, examen pélvico, densidad ósea, hormonas

CRÍTICO

Historia familiar: esencial solo para crónicas (dentro de las primeras 5).

En agudas/dolor: preguntar en cambio por episodios previos.

Cribados + complicaciones siempre en UNA sola pregunta.

Nunca duplicar síntomas en preguntas separadas (ej. apetito, peso, energía → combinarlos).

8. Reglas Específicas

**CONDICIONES AGUDAS** (fiebre, tos, resfriado, infección)

Alta: duración, síntomas asociados, medicamentos, viajes (si aplica), episodios previos

Media: dolor, estilo de vida

**PROHIBIDO**: historia familiar, monitoreo crónico

**CONDICIONES DE DOLOR** (cabeza, tórax, cuerpo, espalda, articulaciones, abdomen, pelvis)

Alta: duración, intensidad (0–10), desencadenantes, impacto funcional, medicamentos, episodios previos

Media: síntomas asociados, estilo de vida

**PROHIBIDO**: historia familiar, monitoreo crónico (a menos que sea dolor crónico)

**SALUD DE LA MUJER** (menstruación, embarazo, menopausia, SOP, dolor pélvico/abdominal)

Alta: duración, medicamentos, síntomas hormonales, cribados recientes, historia familiar

Media: síntomas asociados, estilo de vida, impacto funcional

Baja: viajes, alergias (si no aplican)

**CONDICIONES CRÓNICAS** (diabetes, hipertensión, asma, tiroides, cardiopatías, cáncer)

Alta: duración, medicamentos, monitoreo crónico, historia familiar

Media: síntomas asociados, estilo de vida, impacto funcional

Baja: viajes, alergias (si no aplican)

9. Condiciones Múltiples

Ejemplos:

Mujer + dolor → duración, medicamentos, hormonas, cribados, historia familiar → luego dolor, síntomas, menstruación

Dolor + crónica → duración (ambas), medicamentos (ambas), monitoreo crónico, historia familiar (crónica) → luego dolor, síntomas

Crónica + aguda → duración (ambas), medicamentos (ambas), monitoreo crónico, historia familiar (crónica) → luego síntomas, viajes

Aguda + dolor → duración (ambas), dolor, síntomas, episodios previos → luego medicamentos, viajes

CRÍTICO: Siempre cubrir las áreas de alta prioridad para cada condición antes de pasar a media.

10. Combinación de Preguntas

Cuando un área aplica a más de una condición, pregunta combinando:

Duración → "¿Desde hace cuánto tiempo presenta [condición 1] y [condición 2]?"

Medicamentos → "¿Está tomando algún medicamento para [condición 1] o [condición 2] (incluyendo remedios caseros y de venta libre)?"

Síntomas asociados → "¿Ha notado otros síntomas junto con [condición 1] y [condición 2]?"

Viajes → "¿Ha viajado recientemente que pudiera relacionarse con [condición 1] o [condición 2]?"

CRÍTICO: Solo UNA pregunta sobre medicamentos que incluya todo (recetados, caseros, OTC).

11. Conteo Limitado (ej. 6 preguntas)

Solo alta prioridad

Síntomas asociados → máx. 1–2

Monitoreo crónico → máx. 1–2

Omitir bajas prioridades

12. Evitar Redundancia

OBLIGATORIO: Revisar "Preguntas ya realizadas" antes de generar una nueva.

Nunca repetir preguntas ya hechas (ni con distinta redacción).

Validación paso a paso:

Si ya se preguntó episodios previos → no volver a preguntar

Si ya se preguntó sobre medicamentos → no repetir

Si ya se preguntó duración → no repetir

Escoge otra categoría distinta.

Cada área de monitoreo → solo una vez.

CRÍTICO: Nunca generar la misma pregunta dos veces.

13. Inteligencia Contextual

Agudas → duración, síntomas asociados, medicamentos, viajes

Crónicas → monitoreo, adherencia, complicaciones, historia familiar

Dolor → dolor, desencadenantes, funcionalidad

Alergias → exposiciones, antecedentes familiares, desencadenantes ambientales

Relacionadas con viajes → viajes, contactos, brotes

14. Criterios de Detención

Parar si current_count ≥ max_count

Parar si ≥5 preguntas con cobertura suficiente

Siempre terminar con:

"¿Hay algo más sobre su salud que le gustaría comentar?"

15. Regla de Salida

Devuelve solo una pregunta clara y amigable para el paciente terminada en "?".

No muestres razonamiento, categorías ni explicaciones.

16. VALIDACIÓN FINAL CRÍTICA: Antes de devolver tu pregunta, verifica:

PASO 1: Analiza el motivo de consulta "{disease}" y clasifica CADA condición:
- ¿Hay condición CRÓNICA (diabetes, asma, hipertensión, etc.)? → SÍ preguntar historia familiar y monitoreo
- ¿Hay condición AGUDA (fiebre, tos, resfriado, infección)? → NO preguntar historia familiar ni monitoreo
- ¿Hay condición de DOLOR (dolor de cabeza, dolor abdominal, etc.)? → NO preguntar historia familiar ni monitoreo
- ¿Hay condición de SALUD DE LA MUJER Y el paciente es MUJER? → SÍ preguntar historia familiar
- ¿Hay condición de SALUD DE LA MUJER Y el paciente es HOMBRE? → NO preguntar historia familiar
- ¿Hay condición ALÉRGICA (reacciones alérgicas, asma alérgica, etc.)? → SÍ preguntar sobre alergias
- ¿Hay condición RELACIONADA CON VIAJES (infecciones, exposición)? → SÍ preguntar sobre viajes

PASO 2: Verifica que la pregunta no haya sido hecha antes.

Enfoque Específico por Edad

<12 años: NO preguntas menstruales - enfocarse en causas generales de síntomas

10–18: Irregularidades menstruales, pubertad, SOP, dolor abdominal con períodos

19–40: Problemas menstruales, embarazo, fertilidad, endometriosis

41–60: Perimenopausia, menopausia, cambios hormonales, salud ósea

>60 años: NO preguntas menstruales - enfocarse en causas generales, problemas relacionados con menopausia

Todas las edades: Historia familiar de cáncer de mama/ovario, trastornos hormonales

**REGLA ABSOLUTA**: Si el motivo es fiebre, tos, resfriado, infección, dolor agudo → NUNCA preguntar sobre historia familiar o monitoreo crónico.
"""
        else:
            system_prompt = f"""

SYSTEM PROMPT
You are an intelligent Clinical Intake Assistant with advanced medical knowledge. Conduct a comprehensive yet focused medical interview by asking one clear, patient-friendly question at a time, always ending with a question mark.

1. Doctor's Parameters

The doctor selects categories and maximum question limits. Stay strictly within these limits.

When questions are limited, always prioritize medically essential information first.

2. Context to Use

Chief complaint(s): {disease or "N/A"}
Patient gender: {patient_gender or "Not specified"}
Patient age: {patient_age or "Not specified"}

MANDATORY CONDITION ANALYSIS:

STEP 1: Analyze the chief complaint "{disease}" and classify EACH condition present:

CHRONIC CONDITIONS (require family history and monitoring):
- diabetes, hypertension, heart disease, asthma, thyroid disorders, cancer, kidney disease, liver disease, COPD, arthritis, chronic pain conditions, autoimmune diseases, mental health conditions

ACUTE CONDITIONS (DO NOT require family history or monitoring):
- fever, cold, flu, cough, sore throat, acute infections, injuries, acute pain, nausea, vomiting, diarrhea, acute respiratory infections

PAIN CONDITIONS (DO NOT require family history or monitoring):
- headaches, body pain, chest pain, back pain, joint pain, muscle pain, abdominal pain, pelvic pain, menstrual pain

WOMEN'S HEALTH CONDITIONS (require family history if female and age-appropriate):
- menstrual issues, pregnancy-related, menopause, PCOS, endometriosis, gynecological conditions, breast issues, pelvic/abdominal pain in women
- AGE CONSIDERATIONS: Only ask menstrual-related questions for females aged 12-60 years

ALLERGY-RELATED CONDITIONS (require allergy questions):
- allergic reactions, skin conditions, food allergies, environmental allergies, asthma (when allergy-triggered)

TRAVEL-RELATED CONDITIONS (require travel questions):
- conditions that may be affected by recent travel, exposure to new environments, infectious diseases

STEP 2: If there are MULTIPLE conditions, apply rules for EACH one:
- If there's a CHRONIC condition → ask family history and monitoring
- If there's an ACUTE condition → DO NOT ask family history or monitoring
- If there's a PAIN condition → DO NOT ask family history or monitoring
- If there's a WOMEN'S HEALTH condition AND patient is FEMALE AND age 12-60 → ask family history and menstrual questions
- If there's a WOMEN'S HEALTH condition AND patient is FEMALE AND age <12 or >60 → ask family history but NOT menstrual questions
- If there's a WOMEN'S HEALTH condition AND patient is MALE → DO NOT ask family history
- If there's an ALLERGY-RELATED condition → ask about allergies and exposures
- If there's a TRAVEL-RELATED condition → ask about recent travel

STEP 3: For abdominal/pelvic pain in FEMALES, consider if it could be women's health related:
- If age 12-60: Ask about menstrual cycle, pregnancy, gynecological history
- If age <12 or >60: Focus on general abdominal causes, avoid menstrual questions

STEP 4: Determine what questions to ask based on ALL identified conditions.

Recent answers: {', '.join(previous_answers[-3:])}

Already asked: {asked_questions}

**MANDATORY VALIDATION**: Before generating ANY question, you MUST:
1. Read the "Already asked" list above
2. Check if your proposed question is similar to any question in that list
3. If it is similar, choose a DIFFERENT category and generate a DIFFERENT question
4. NEVER generate a question that is the same or similar to any question in the "Already asked" list

**CRITICAL**: Do NOT repeat any of the above questions. Each question must be unique and different from all previously asked questions.

Available categories: {asked_categories or "ALL"}

Progress: {current_count}/{max_count}

Recently travelled: {recently_travelled}

Prior context: {prior_block or "none"}

3. Category Rules

Associated Symptoms → Max 3

Chronic Disease Monitoring → Max 3 (see specifics below)

All other categories → Max 1

4. Smart Filtering - CRITICAL RULES

**FAMILY HISTORY**: ONLY for chronic/genetic conditions (diabetes, hypertension, heart disease, cancer, asthma, thyroid). NEVER for acute conditions, simple pain, or infections.

**CHRONIC MONITORING**: ONLY for chronic diseases (diabetes, HTN, asthma, thyroid, heart disease, cancer). NEVER for acute conditions, simple pain, or infections.

Past Medical History → Only if relevant to the illness

Travel History → Only if recently_travelled=True and condition fits (GI, infectious, fever, cough)

Allergies → Only if allergy-related symptoms are present (rashes, wheeze, hives, respiratory issues)

Pain Assessment → Only if pain is part of the complaint

**CRITICAL RULE**: If the condition is ACUTE (fever, cough, cold, infection, acute pain) → DO NOT ask family history or chronic monitoring.

5. Chronic Disease Monitoring Coverage (max 3 total)

Cover each area once only, choose the most relevant three:

Home readings → BP, glucose, peak flow, thyroid tests

Recent labs → HbA1c, kidney/liver, cholesterol, ECG

Screenings & complications (combined) → eye, foot, dental exams + wounds, chest pain, breathlessness

Medication adherence & side effects

CRITICAL: Do not split screenings and complications into separate questions. Always combine.

6. Flow Priority

Duration of symptoms → Always first

Self-care/Home remedies + Current medications → Always second (combine into one question)

Disease-specific questions (based on classification)

Associated Symptoms → If relevant (≤3)

Chronic Monitoring → If chronic disease (≤3)

Other categories → ≤1 each

Closing question → Only at the end (after ≥7 questions OR when no further useful questions remain)

7. Intelligent Prioritization (when doctor-selected categories)

You must prioritize questions within selected categories by medical importance. Act as a general physician would.

Chronic Conditions

High priority: Duration, medications, home readings, recent labs, complications/screenings, family history

Medium: Associated symptoms, lifestyle factors, functional impact

Low: Travel history, allergies (unless relevant)

Specific Check-Ups by Condition

Diabetes → Eye exams, foot exams, dental, kidney tests

Hypertension → Heart checks, kidney function, eye exams

Heart Disease → Cardiac tests, cholesterol, stress tests

Asthma → Lung function, allergy tests, chest X-ray, peak flow

Thyroid → TSH, thyroid function, neck exam, heart rate

Cancer → Screening tests, imaging, blood markers

Women's Health → Pap smear, mammogram, pelvic exam, bone density, hormone levels

CRITICAL:

Family history → essential only for chronic (ask within first 5).

Acute/pain → ask instead about past similar episodes.

Combine screenings + complications into ONE question.

Never duplicate symptom questions (e.g., appetite + weight + energy separately).

8. Condition-Specific Rules

**ACUTE CONDITIONS** (fever, cough, cold, infection)

High priority: Duration, associated symptoms, medications, travel history (if applicable), past similar episodes

Medium: Pain assessment, lifestyle

**FORBIDDEN**: Family history, chronic monitoring

**PAIN CONDITIONS** (headache, chest pain, body pain)

High priority: Duration, pain assessment, triggers, functional impact, medications, past similar episodes

Medium: Associated symptoms, lifestyle

**FORBIDDEN**: Family history, chronic monitoring (unless it's chronic pain)

**WOMEN'S HEALTH CONDITIONS** (menstrual, pregnancy, menopause, PCOS, pelvic/stomach pain)

High priority: Duration, medications, hormonal symptoms, recent screenings, family history

Medium: Associated symptoms, lifestyle, functional impact

Low: Travel history, allergies (unless relevant)

**CHRONIC CONDITIONS** (diabetes, hypertension, asthma, thyroid, heart disease, cancer)

High priority: Duration, medications, chronic monitoring, family history

Medium: Associated symptoms, lifestyle, functional impact

Low: Travel history, allergies (unless relevant)

Age-Specific Focus

<12 years: NO menstrual questions - focus on general causes of symptoms

10–18: Menstrual irregularities, puberty, PCOS, stomach pain with periods

19–40: Menstrual issues, pregnancy, fertility, endometriosis

41–60: Perimenopause, menopause, hormonal changes, bone health

>60 years: NO menstrual questions - focus on general causes, menopause-related issues

All ages: Family history of breast/ovarian cancer, hormonal disorders

9. Multiple Conditions Prioritization

When multiple conditions exist (e.g., body pain + diabetes, or chronic + acute):

Women's Health + Pain → Duration, medications, hormonal symptoms, screenings, family history → then pain assessment, associated symptoms, menstrual history

Pain + Chronic → Duration (both), medications (both), chronic monitoring, family history (chronic) → then pain assessment, associated symptoms

Chronic + Acute → Duration (both), medications (both), **chronic monitoring**, **family history (chronic)** → then associated symptoms, travel history

**CRITICAL**: For chronic conditions like asthma, diabetes, hypertension, ALWAYS ask:
- Family history (within first 5 questions)
- Chronic monitoring (tests, screenings, home readings)


Acute + Pain → Duration (both), pain assessment, associated symptoms, past similar episodes → then medications, travel history

CRITICAL: Always cover all high-priority categories first for each condition before moving to medium.

10. Combining Questions Across Conditions

When a category applies to multiple conditions, ask one combined question:

Duration → "How long have you been experiencing [condition 1] and [condition 2]?" (ALWAYS mention ALL conditions)

Medications → "Are you taking any medications for [condition 1] or [condition 2]?" (INCLUDE prescribed medications, home remedies, over-the-counter medications)

Associated Symptoms → "Have you noticed any other symptoms along with [condition 1] and [condition 2]?" (ALWAYS mention ALL conditions)

Travel History → "What countries or regions did you visit, and were you exposed to any unusual environments or outbreaks that could be related to [condition 1] or [condition 2]?"

**CRITICAL**: For medications, ask ONE question that covers ALL types of medications (prescribed, home remedies, over-the-counter). Do NOT ask separate medication questions.

11. Limited Question Count (e.g., 6 questions)

Focus only on high-priority categories

Associated Symptoms → max 1–2

Chronic Monitoring → max 1–2

Skip low-priority categories entirely

12. Avoid Redundancy

**MANDATORY**: Check the "Already asked" list above before generating any question

**NEVER** repeat any question from the "Already asked" list - even if worded slightly differently

**STEP-BY-STEP VALIDATION**:
1. Look at the "Already asked" list
2. If you see "past episodes" questions → Do NOT generate another "past episodes" question
3. If you see "medications" questions → Do NOT generate another "medications" question 
4. If you see "duration" questions → Do NOT generate another "duration" question
5. Choose a DIFFERENT category that hasn't been covered yet

Do not generate similar questions with different wording

If a category is covered, move to the next relevant one

Each monitoring area asked once only

CRITICAL: Never generate the same question twice.

13. Contextual Intelligence

Acute → duration, associated symptoms, medications, travel

Chronic → monitoring, adherence, complications, family history

Pain → pain assessment, triggers, functional impact

Allergy → exposures, allergy history, family history of allergies

Travel-related → travel, exposures, outbreak/contact risks

14. Stopping Criteria

Stop if current_count ≥ max_count

Stop if ≥5 questions asked with sufficient coverage

Always end with Closing:

"Is there anything else about your health you'd like to discuss?"

15. Output Rule

Return only one clear, patient-friendly question ending with "?".

Do not show reasoning, category names, or explanations.

**FINAL VALIDATION**: Before returning your question, verify:

STEP 1: Analyze the chief complaint "{disease}" and classify EACH condition:
- Is there a CHRONIC condition (diabetes, asthma, hypertension, etc.)? → YES, ask family history and monitoring
- Is there an ACUTE condition (fever, cough, cold, infection)? → DO NOT ask family history or monitoring
- Is there a PAIN condition (headache, abdominal pain, etc.)? → DO NOT ask family history or monitoring
- Is there a WOMEN'S HEALTH condition AND patient is FEMALE AND age 12-60? → YES, ask family history and menstrual questions
- Is there a WOMEN'S HEALTH condition AND patient is FEMALE AND age <12 or >60? → YES, ask family history but NOT menstrual questions
- Is there a WOMEN'S HEALTH condition AND patient is MALE? → DO NOT ask family history
- Is there an ALLERGY-RELATED condition (allergic reactions, allergic asthma, etc.)? → YES, ask about allergies
- Is there a TRAVEL-RELATED condition (infections, exposure)? → YES, ask about travel

STEP 2: Verify the question hasn't been asked before.

**ABSOLUTE RULE**: If the chief complaint is fever, cough, cold, infection, acute pain → NEVER ask about family history or chronic monitoring.

""" 

        try:
            text = await self._chat_completion(
                messages=[
                    {"role": "system", "content": "You are a clinical intake assistant."},
                    {"role": "user", "content": system_prompt},
                ],
                max_tokens=min(256, self._settings.openai.max_tokens),
                temperature=0.3,
            )
            text = text.replace("\n", " ").strip()
            if not text.endswith("?"):
                text = text.rstrip(".") + "?"

            return text
        except Exception:
            raise

    async def should_stop_asking(
        self,
        disease: str,
        previous_answers: List[str],
        current_count: int,
        max_count: int,
    ) -> bool:
        if current_count >= max_count:
            return True
        return False

    async def assess_completion_percent(
        self,
        disease: str,
        previous_answers: List[str],
        asked_questions: List[str],
        current_count: int,
        max_count: int,
        prior_summary: Optional[Any] = None,
        prior_qas: Optional[List[str]] = None,
    ) -> int:
        try:
            progress_ratio = 0.0
            if max_count > 0:
                progress_ratio = min(max(current_count / max_count, 0.0), 1.0)
            return int(progress_ratio * 100)
        except Exception:
            if max_count <= 0:
                return 0
            return max(0, min(int(round((current_count / max_count) * 100.0)), 100))

    # ----------------------
    # Medication check
    # ----------------------
    async def is_medication_question(self, question: str) -> bool:
        """Return True if the question is about current medications, home remedies, or self-care.

        This includes the combined self-care/home remedies/medications questions that have image upload capability.
        """
        question_lower = (question or "").lower()

        # Check for combined self-care/home remedies/medications questions
        if ("home remedies" in question_lower and "medications" in question_lower) or \
           ("self-care" in question_lower and "medications" in question_lower) or \
           ("remedies" in question_lower and "supplements" in question_lower) or \
           ("autocuidado" in question_lower and "medicamentos" in question_lower) or \
           ("remedios caseros" in question_lower and "medicamentos" in question_lower):
            return True

        # Check for individual medication-related terms
        medication_terms = [
            "medication", "medications", "medicines", "medicine", "drug", "drugs",
            "prescription", "prescribed", "tablet", "tablets", "capsule", "capsules",
            "syrup", "dose", "dosage", "frequency", "supplement", "supplements",
            "insulin", "otc", "over-the-counter", "medicamento", "medicamentos",
            "medicina", "medicinas", "medicamento recetado", "suplemento", "suplementos"
        ]

        return any(term in question_lower for term in medication_terms)

    # ----------------------
    # Pre-visit summary
    # ----------------------
    async def generate_pre_visit_summary(
        self,
        patient_data: Dict[str, Any],
        intake_answers: Dict[str, Any],
        language: str = "en",
    ) -> Dict[str, Any]:
        """Generate pre-visit clinical summary from intake data with red flag detection."""
        
        if language == "sp":
            prompt = (
                "Rol y Tarea\n"
                "Eres un Asistente de Admisión Clínica.\n"
                "Tu tarea es generar un Resumen Pre-Consulta conciso y clínicamente útil (~180-200 palabras) basado estrictamente en las respuestas de admisión proporcionadas.\n\n"
                "Reglas Críticas\n"
                "- No inventes, adivines o expandas más allá de la entrada proporcionada.\n"
                "- La salida debe ser texto plano con encabezados de sección, una sección por línea (sin líneas en blanco adicionales).\n"
                "- Usa solo los encabezados exactos listados a continuación. No agregues, renombres o reordenes encabezados.\n"
                "- Sin viñetas, numeración o formato markdown.\n"
                "- Escribe en un tono de entrega clínica: corto, factual, sin duplicados y neutral.\n"
                "- Incluye una sección SOLO si contiene contenido real de las respuestas del paciente.\n"
                "- No uses marcadores de posición como \"N/A\", \"No proporcionado\", \"no reportado\", o \"niega\".\n"
                "- No incluyas secciones para temas que no fueron preguntados o discutidos.\n"
                "- Usa frases orientadas al paciente: \"El paciente reporta...\", \"Niega...\", \"En medicamentos:...\".\n"
                "- No incluyas observaciones clínicas, diagnósticos, planes, signos vitales o hallazgos del examen (la pre-consulta es solo lo reportado por el paciente).\n"
                "- Normaliza pronunciaciones médicas obvias a términos correctos sin agregar nueva información.\n\n"
                "Encabezados (usa MAYÚSCULAS EXACTAS; incluye solo si tienes datos reales de las respuestas del paciente)\n"
                "Motivo de Consulta:\n"
                "HPI:\n"
                "Historia:\n"
                "Revisión de Sistemas:\n"
                "Medicación Actual:\n\n"
                "Pautas de Contenido por Sección\n"
                "- Motivo de Consulta: Una línea en las propias palabras del paciente si está disponible.\n"
                "- HPI: UN párrafo legible tejiendo OLDCARTS en prosa:\n"
                "  Inicio, Localización, Duración, Caracterización/calidad, Factores agravantes, Factores aliviadores, Radiación,\n"
                "  Patrón temporal, Severidad (1-10), Síntomas asociados, Negativos relevantes.\n"
                "  Manténlo natural y coherente (ej., \"El paciente reporta...\"). Si algunos elementos OLDCARTS son desconocidos, simplemente omítelos.\n"
                "- Historia: Una línea combinando cualquier elemento reportado por el paciente usando punto y coma en este orden si está presente:\n"
                "  Médica: ...; Quirúrgica: ...; Familiar: ...; Estilo de vida: ...\n"
                "  (Incluye SOLO las partes que fueron realmente preguntadas y respondidas por el paciente. Si un tema no fue discutido, no lo incluyas en absoluto).\n"
                "- Revisión de Sistemas: Una línea narrativa resumiendo positivos/negativos basados en sistemas mencionados explícitamente por el paciente. Mantén como prosa, no como lista. Solo incluye si los sistemas fueron realmente revisados.\n"
                "- Medicación Actual: Una línea narrativa con medicamentos/suplementos realmente declarados por el paciente (nombre/dosis/frecuencia si se proporciona). Incluye declaraciones de alergia solo si el paciente las reportó explícitamente. Solo incluye si los medicamentos fueron realmente discutidos.\n\n"
                "Ejemplo de Formato\n"
                "(Estructura y tono solamente—el contenido será diferente; cada sección en una sola línea.)\n"
                "Motivo de Consulta: El paciente reporta dolor de cabeza severo por 3 días.\n"
                "HPI: El paciente describe una semana de dolores de cabeza persistentes que comienzan en la mañana y empeoran durante el día, llegando hasta 8/10 en los últimos 3 días. El dolor es sobre ambas sienes y se siente diferente de migrañas previas; la fatiga es prominente y se niega náusea. Los episodios se agravan por estrés y más tarde en el día, con alivio mínimo de analgésicos de venta libre y algo de alivio usando compresas frías.\n"
                "Historia: Médica: hipertensión; Quirúrgica: colecistectomía hace cinco años; Estilo de vida: no fumador, alcohol ocasional, trabajo de alto estrés.\n"
                "Medicación Actual: En medicamentos: lisinopril 10 mg diario e ibuprofeno según necesidad; alergias incluidas solo si el paciente las declaró explícitamente.\n\n"
                f"Respuestas de Admisión:\n{self._format_intake_answers(intake_answers)}"
            )
        else:
            prompt = (
                "Role & Task\n"
                "You are a Clinical Intake Assistant.\n"
                "Your task is to generate a concise, clinically useful Pre-Visit Summary (~180–200 words) based strictly on the provided intake responses.\n\n"
                "Critical Rules\n"
                "- Do not invent, guess, or expand beyond the provided input.\n"
                "- Output must be plain text with section headings, one section per line (no extra blank lines).\n"
                "- Use only the exact headings listed below. Do not add, rename, or reorder headings.\n"
                "- No bullets, numbering, or markdown formatting.\n"
                "- Write in a clinical handover tone: short, factual, deduplicated, and neutral.\n"
                "- Include a section ONLY if it contains actual content from the patient's responses.\n"
                "- Do not use placeholders like \"N/A\", \"Not provided\", \"not reported\", or \"denies\".\n"
                "- Do not include sections for topics that were not asked about or discussed.\n"
                "- Use patient-facing phrasing: \"Patient reports …\", \"Denies …\", \"On meds: …\".\n"
                "- Do not include clinician observations, diagnoses, plans, vitals, or exam findings (previsit is patient-reported only).\n"
                '- Normalize obvious medical mispronunciations to correct terms (e.g., "diabetes mellitus" -> "diabetes mellitus") without adding new information.\n\n'
                "Headings (use EXACT casing; include only if you have actual data from patient responses)\n"
                "Chief Complaint:\n"
                "HPI:\n"
                "History:\n"
                "Review of Systems:\n"
                "Current Medication:\n\n"
                "Content Guidelines per Section\n"
                "- Chief Complaint: One line in the patient's own words if available.\n"
                "- HPI: ONE readable paragraph weaving OLDCARTS into prose:\n"
                "  Onset, Location, Duration, Characterization/quality, Aggravating factors, Relieving factors, Radiation,\n"
                "  Temporal pattern, Severity (1–10), Associated symptoms, Relevant negatives.\n"
                "  Keep it natural and coherent (e.g., \"The patient reports …\"). If some OLDCARTS elements are unknown, simply omit them (do not write placeholders).\n"
                "- History: One line combining any patient-reported items using semicolons in this order if present:\n"
                "  Medical: …; Surgical: …; Family: …; Lifestyle: …\n"
                "  (Include ONLY parts that were actually asked about and answered by the patient. If a topic was not discussed, do not include it at all.)\n"
                "- Review of Systems: One narrative line summarizing system-based positives/negatives explicitly mentioned by the patient (e.g., General, Neuro, Eyes, Resp, GI). Keep as prose, not a list. Only include if systems were actually reviewed.\n"
                "- Current Medication: One narrative line with meds/supplements actually stated by the patient (name/dose/frequency if provided). Include allergy statements only if the patient explicitly reported them. Only include if medications were actually discussed.\n\n"
                "Example Format\n"
                "(Structure and tone only—content will differ; each section on a single line.)\n"
                "Chief Complaint: Patient reports severe headache for 3 days.\n"
                "HPI: The patient describes a week of persistent headaches that begin in the morning and worsen through the day, reaching up to 8/10 over the last 3 days. Pain is over both temples and feels different from prior migraines; fatigue is prominent and nausea is denied. Episodes are aggravated by stress and later in the day, with minimal relief from over-the-counter analgesics and some relief using cold compresses. No radiation is reported, evenings are typically worse, and there have been no recent changes in medications or lifestyle.\n"
                "History: Medical: hypertension; Surgical: cholecystectomy five years ago; Lifestyle: non-smoker, occasional alcohol, high-stress job.\n"
                "Current Medication: On meds: lisinopril 10 mg daily and ibuprofen as needed; allergies included only if the patient explicitly stated them.\n\n"
                f"Intake Responses:\n{self._format_intake_answers(intake_answers)}"
            )

        try:
            # Detect abusive language red flags
            try:
                red_flags = await self._detect_red_flags(intake_answers, language)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Red flag detection failed, continuing without flags: {e}")
                red_flags = []
            
            response = await self._chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical assistant generating pre-visit summaries. Focus on accuracy, completeness, "
                            "and clinical relevance. Do not make diagnoses."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=min(2000, self._settings.openai.max_tokens),
                temperature=0.3,
            )
            cleaned = self._clean_summary_markdown(response)
            
            # Red flags are handled separately in the frontend, not included in summary text
            
            return {
                "summary": cleaned,
                "structured_data": {
                    "chief_complaint": "See summary",
                    "key_findings": ["See summary"],
                },
                "red_flags": red_flags,
            }
        except Exception:
            return await self._generate_fallback_summary(patient_data, intake_answers)

    # ----------------------
    # Red Flag Detection
    # ----------------------
    async def _detect_red_flags(self, intake_answers: Dict[str, Any], language: str = "en") -> List[Dict[str, str]]:
        """Hybrid abusive language detection: hardcoded rules + LLM analysis."""
        red_flags = []
        
        if not isinstance(intake_answers, dict) or "questions_asked" not in intake_answers:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Invalid intake_answers format for red flag detection")
            return red_flags
            
        questions_asked = intake_answers.get("questions_asked", [])
        if not questions_asked:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("No questions found in intake_answers")
            return red_flags
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Starting hybrid abusive language detection for {len(questions_asked)} questions")
        
        # Step 1: Fast hardcoded detection for obvious cases
        obvious_flags = self._detect_obvious_abusive_language(questions_asked, language)
        red_flags.extend(obvious_flags)
        logger.info(f"Obvious abusive language flags detected: {len(obvious_flags)}")
        
        # Step 2: LLM analysis for subtle/contextual abusive language
        complex_flags = await self._detect_subtle_abusive_language_with_llm(questions_asked, language)
        red_flags.extend(complex_flags)
        logger.info(f"Subtle abusive language flags detected: {len(complex_flags)}")
        
        logger.info(f"Total abusive language red flags detected: {len(red_flags)}")
        return red_flags
    
    def _detect_obvious_abusive_language(self, questions_asked: List[Dict[str, Any]], language: str = "en") -> List[Dict[str, str]]:
        """Fast hardcoded detection for obvious abusive language."""
        red_flags = []
        
        for qa in questions_asked:
            answer = qa.get("answer", "").strip()
            question = qa.get("question", "").strip()
            
            if not answer or answer.lower() in ["", "n/a", "not provided", "unknown", "don't know", "no se", "no proporcionado"]:
                continue
            
            # Check for obvious abusive language
            if self._contains_abusive_language(answer, language):
                red_flags.append({
                    "type": "abusive_language",
                    "question": question,
                    "answer": answer,
                    "message": self._get_abusive_language_message(language),
                    "detection_method": "hardcoded"
                })
        
        return red_flags
    
    async def _detect_subtle_abusive_language_with_llm(self, questions_asked: List[Dict[str, Any]], language: str = "en") -> List[Dict[str, str]]:
        """Use LLM to detect subtle, contextual, or creative abusive language."""
        try:
            # Filter out obvious cases already detected
            subtle_cases = []
            for qa in questions_asked:
                answer = qa.get("answer", "").strip()
                question = qa.get("question", "").strip()
                
                if (answer and answer.lower() not in ["", "n/a", "not provided", "unknown", "don't know", "no se", "no proporcionado"] 
                    and not self._contains_abusive_language(answer, language)):
                    subtle_cases.append(qa)
            
            if not subtle_cases:
                return []
            
            # Use LLM to analyze subtle cases
            return await self._analyze_abusive_language_with_llm(subtle_cases, language)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"LLM abusive language analysis failed: {e}")
            return []
    
    async def _analyze_abusive_language_with_llm(self, questions_asked: List[Dict[str, Any]], language: str = "en") -> List[Dict[str, str]]:
        """Use LLM to analyze question-answer pairs for subtle abusive language."""
        
        if language == "sp":
            prompt = f"""
Analiza estas respuestas de admisión del paciente para detectar LENGUAJE ABUSIVO O INAPROPIADO:

IMPORTANTE: Busca lenguaje abusivo, ofensivo, o inapropiado, incluyendo:
- Profanidad directa o disfrazada
- Insultos, desprecios, o lenguaje degradante
- Lenguaje sexual inapropiado
- Comentarios racistas, sexistas, o discriminatorios
- Amenazas o lenguaje agresivo
- Sarcasmo ofensivo o pasivo-agresivo
- Lenguaje que sea inapropiado en un contexto médico

NO marques como abusivo:
- Lenguaje médico técnico
- Descripciones de síntomas
- Respuestas apropiadas a preguntas médicas
- Expresiones de dolor o frustración legítima

Para cada respuesta, determina si contiene lenguaje abusivo o inapropiado.

Responde SOLO en formato JSON con este esquema:
{{
    "abusive_language": [
        {{
            "question": "pregunta completa",
            "answer": "respuesta completa",
            "reason": "explicación específica de por qué es lenguaje abusivo"
        }}
    ]
}}

Si no hay lenguaje abusivo, devuelve: {{"abusive_language": []}}

Respuestas a analizar:
{self._format_qa_pairs(questions_asked)}
"""
        else:
            prompt = f"""
Analyze these patient intake responses for ABUSIVE OR INAPPROPRIATE LANGUAGE:

IMPORTANT: Look for abusive, offensive, or inappropriate language, including:
- Direct or disguised profanity
- Insults, slurs, or degrading language
- Inappropriate sexual language
- Racist, sexist, or discriminatory comments
- Threats or aggressive language
- Offensive sarcasm or passive-aggressive language
- Language inappropriate in a medical context

DO NOT flag as abusive:
- Medical terminology
- Symptom descriptions
- Appropriate responses to medical questions
- Legitimate expressions of pain or frustration

For each response, determine if it contains abusive or inappropriate language.

Respond ONLY in JSON format with this schema:
{{
    "abusive_language": [
        {{
            "question": "full question",
            "answer": "full answer",
            "reason": "specific explanation of why this is abusive language"
        }}
    ]
}}

If no abusive language found, return: {{"abusive_language": []}}

Responses to analyze:
{self._format_qa_pairs(questions_asked)}
"""

        try:
            response = await self._chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a clinical assistant analyzing patient responses for abusive language. Be precise and only flag truly inappropriate or abusive language."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent analysis
            )
            
            # Parse LLM response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                abusive_cases = result.get("abusive_language", [])
                
                # Convert to our format
                formatted_flags = []
                for case in abusive_cases:
                    formatted_flags.append({
                        "type": "abusive_language",
                        "question": case.get("question", ""),
                        "answer": case.get("answer", ""),
                        "message": self._get_llm_abusive_language_message(case.get("reason", ""), language),
                        "detection_method": "llm"
                    })
                
                return formatted_flags
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to parse LLM abusive language response: {e}")
        
        return []
    
    def _format_qa_pairs(self, questions_asked: List[Dict[str, Any]]) -> str:
        """Format question-answer pairs for LLM analysis."""
        formatted = []
        for i, qa in enumerate(questions_asked, 1):
            question = qa.get("question", "N/A")
            answer = qa.get("answer", "N/A")
            formatted.append(f"{i}. Q: {question}\n   A: {answer}")
        return "\n\n".join(formatted)
    
    def _get_llm_abusive_language_message(self, reason: str, language: str = "en") -> str:
        """Get message for LLM-detected abusive language."""
        if language == "sp":
            return f"⚠️ BANDERA ROJA: Lenguaje abusivo detectado. Razón: {reason}"
        else:
            return f"⚠️ RED FLAG: Abusive language detected. Reason: {reason}"
    
    def _contains_abusive_language(self, text: str, language: str = "en") -> bool:
        """Check if text contains abusive or inappropriate language."""
        text_lower = text.lower()
        
        # Common abusive words/phrases in English
        english_abusive = [
            "fuck", "shit", "damn", "hell", "bitch", "asshole", "bastard", "crap",
            "stupid", "idiot", "moron", "retard", "gay", "fag", "nigger", "whore",
            "slut", "cunt", "piss", "pissed", "fucking", "bullshit", "goddamn"
        ]
        
        # Common abusive words/phrases in Spanish
        spanish_abusive = [
            "puta", "puto", "mierda", "joder", "coño", "cabrón", "hijo de puta",
            "estúpido", "idiota", "imbécil", "retrasado", "maricón", "joto",
            "pinche", "chingado", "verga", "pendejo", "culero", "mamón"
        ]
        
        abusive_words = spanish_abusive if language == "sp" else english_abusive
        
        return any(word in text_lower for word in abusive_words)
    
    
    def _get_abusive_language_message(self, language: str = "en") -> str:
        """Get message for abusive language red flag."""
        if language == "sp":
            return "⚠️ BANDERA ROJA: El paciente utilizó lenguaje inapropiado o abusivo en sus respuestas."
        else:
            return "⚠️ RED FLAG: Patient used inappropriate or abusive language in their responses."
    
    

    # ----------------------
    # Helpers
    # ----------------------
    def _format_intake_answers(self, intake_answers: Dict[str, Any]) -> str:
        """Format intake answers for prompt."""
        if isinstance(intake_answers, dict) and "questions_asked" in intake_answers:
            formatted: List[str] = []
            for qa in intake_answers.get("questions_asked", []):
                q = qa.get("question", "N/A")
                a = qa.get("answer", "N/A")
                formatted.append(f"Q: {q}")
                formatted.append(f"A: {a}")
                formatted.append("")
            return "\n".join(formatted)
        return "\n".join([f"{k}: {v}" for k, v in intake_answers.items()])

    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format (JSON if present)."""
        try:
            import json
            import re as _re

            text = (response or "").strip()

            # 1) Prefer fenced ```json ... ```
            fence_json = _re.search(r"```json\s*([\s\S]*?)\s*```", text, _re.IGNORECASE)
            if fence_json:
                candidate = fence_json.group(1).strip()
                return json.loads(candidate)

            # 2) Any fenced block without language
            fence_any = _re.search(r"```\s*([\s\S]*?)\s*```", text)
            if fence_any:
                candidate = fence_any.group(1).strip()
                try:
                    return json.loads(candidate)
                except Exception:
                    pass

            # 3) Raw JSON between first '{' and last '}'
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                candidate = text[first : last + 1]
                return json.loads(candidate)

            # Fallback to basic structure
            return {
                "summary": text,
                "structured_data": {
                    "chief_complaint": "See summary",
                    "key_findings": ["See summary"],
                    "recommendations": ["See summary"],
                },
            }
        except Exception:
            return {
                "summary": response,
                "structured_data": {
                    "chief_complaint": "Unable to parse",
                    "key_findings": ["See summary"],
                },
            }

    async def _generate_fallback_summary(self, patient_data: Dict[str, Any], intake_answers: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic fallback summary with red flag detection."""
        # Still detect red flags even in fallback mode
        red_flags = await self._detect_red_flags(intake_answers, "en")  # Default to English for fallback
        
        summary = f"Pre-visit summary for {patient_data.get('name', 'Patient')}"
        
        # Red flags are handled separately in the frontend, not included in summary text
        
        return {
            "summary": summary,
            "structured_data": {
                "chief_complaint": patient_data.get("symptom") or patient_data.get("complaint") or "N/A",
                "key_findings": ["See intake responses"],
            },
            "red_flags": red_flags,
        }

    async def _normalize_summary_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure result contains 'summary' and 'structured_data' keys with sane defaults."""
        if not isinstance(result, dict):
            return await self._generate_fallback_summary({}, {})

        summary = result.get("summary") or result.get("markdown") or result.get("content") or ""
        structured = result.get("structured_data") or result.get("structuredData") or result.get("data") or {}

        if not isinstance(summary, str):
            summary = str(summary)
        if not isinstance(structured, dict):
            structured = {"raw": structured}

        if "chief_complaint" not in structured:
            structured["chief_complaint"] = "See summary"
        if "key_findings" not in structured:
            structured["key_findings"] = ["See summary"]

        # Attempt extraction from markdown if missing
        def _extract_from_markdown(md: str) -> Dict[str, Any]:
            data: Dict[str, Any] = {}
            current_section = None
            key_findings: List[str] = []
            chief_bullets: List[str] = []
            for raw in (md or "").splitlines():
                line = raw.strip()
                if line.startswith("## "):
                    title = line[3:].strip().lower()
                    if "key clinical points" in title:
                        current_section = "key_points"
                    elif "chief complaint" in title:
                        current_section = "chief"
                    else:
                        current_section = None
                    continue
                if line.startswith("- "):
                    text2 = line[2:].strip()
                    if current_section == "key_points" and text2:
                        key_findings.append(text2)
                    if current_section == "chief" and text2:
                        chief_bullets.append(text2)
            if key_findings:
                data["key_findings"] = key_findings
            if chief_bullets:
                data["chief_complaint"] = ", ".join(chief_bullets)
            return data

        if structured.get("key_findings") == ["See summary"] or not structured.get("key_findings"):
            extracted = _extract_from_markdown(summary)
            if extracted.get("key_findings"):
                structured["key_findings"] = extracted["key_findings"]
            if extracted.get("chief_complaint") and structured.get("chief_complaint") in (None, "See summary", "N/A"):
                structured["chief_complaint"] = extracted["chief_complaint"]

        return {
            "summary": summary, 
            "structured_data": structured,
            "red_flags": result.get("red_flags", [])
        }

    def _clean_summary_markdown(self, summary_md: str) -> str:
        """Remove placeholder lines like [Insert ...] and drop now-empty sections."""
        if not isinstance(summary_md, str) or not summary_md.strip():
            return summary_md
        lines = summary_md.splitlines()
        cleaned: List[str] = []
        current_section_start = -1
        section_has_bullets = False

        def flush_section() -> None:
            nonlocal current_section_start, section_has_bullets
            if current_section_start == -1:
                return
            if not section_has_bullets:
                if cleaned:
                    cleaned.pop()
            current_section_start = -1
            section_has_bullets = False

        for raw in lines:
            line = raw.rstrip()
            if line.startswith("## ") or line.startswith("# "):
                flush_section()
                current_section_start = len(cleaned)
                section_has_bullets = False
                cleaned.append(line)
                continue
            low = line.lower()
            if "[insert" in low:
                continue
            if line.strip().startswith("- "):
                section_has_bullets = True
                cleaned.append(line)
            else:
                cleaned.append(line)
        flush_section()
        return "\n".join(cleaned)
