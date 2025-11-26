"""
Multi-Agent Question Generation System
Optimized for medical accuracy with intelligent reasoning

Architecture:
- Agent 1: Medical Context Analyzer - Analyzes chief complaint and determines medical priorities
- Agent 2: Answer Extractor - Reviews previous Q&A to avoid redundancy
- Agent 3: Question Generator - Generates medically-relevant interview questions
- Agent 4: Safety Validator - Enforces hard safety constraints

This replaces the previous 837-line prompt with dynamic medical reasoning.
"""

import asyncio
import logging
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from clinicai.application.ports.services.question_service import QuestionService
from clinicai.core.config import get_settings
from clinicai.core.ai_factory import get_ai_client


logger = logging.getLogger("clinicai")


# ============================================================================
# AGENT 1: Medical Context Analyzer
# ============================================================================

@dataclass
class MedicalContext:
    """Structured medical context from condition analysis"""
    chief_complaint: str
    condition_properties: Dict[str, bool]  # is_chronic, is_hereditary, has_complications, etc.
    priority_topics: List[str]  # Ordered list of what to ask about
    avoid_topics: List[str]  # Topics that should NOT be asked
    medical_reasoning: str  # Why these priorities
    patient_age: Optional[int]
    patient_gender: Optional[str]


class MedicalContextAnalyzer:
    """Agent 1: Analyzes medical context using LLM reasoning"""
    
    def __init__(self, client, settings):
        self._client = client
        self._settings = settings
    
    async def analyze_condition(
        self,
        chief_complaint: str,
        patient_age: Optional[int],
        patient_gender: Optional[str],
        recently_travelled: bool = False,
        language: str = "en"
    ) -> MedicalContext:
        """
        Analyze the medical condition to understand what information is needed.
        Uses LLM's medical knowledge instead of hardcoded rules.
        """
        
        lang = self._normalize_language(language)
        
        if lang == "sp":
            system_prompt = """Eres un médico experto analizando una queja principal del paciente.

Tu trabajo es determinar QUÉ información médica es esencial recopilar para este caso específico.

Analiza la condición médicamente y devuelve un análisis estructurado en formato JSON."""

            user_prompt = f"""
Analiza esta queja principal del paciente: "{chief_complaint}"
Edad del paciente: {patient_age or "Desconocida"}
Género del paciente: {patient_gender or "No especificado"}
Ha viajado recientemente: {"Sí" if recently_travelled else "No"}

Realiza un análisis médico completo y devuelve un JSON con esta estructura EXACTA:

REGLAS MÉDICAS PARA PRIORITY_TOPICS (CUMPLE EXACTAMENTE):

PRIORIDAD DE FLUJO - aplica a todas las entrevistas (omite solo lo que no sea relevante):
   1. Duración - siempre la primera pregunta.
   2. Medicamentos + cuidados caseros combinados - segunda pregunta; incluye fármacos prescritos y todo lo que el paciente use en casa (OTC, hierbas, remedios).
   3. Temas de alta prioridad según la enfermedad - aborda inmediatamente después (ver secciones). Si el médico selecciona categorías, cúbrelas primero pero respetando este orden global y eligiendo la pregunta más relevante médicamente.
   4. Síntomas asociados - solo si aportan valor, máximo 3 preguntas únicas (no reformules).
   5. Monitoreo crónico - si is_chronic=true, formula exactamente DOS preguntas de monitoreo (lecturas en casa, adherencia, tendencias).
   6. Detección/complicaciones - si is_chronic=true, formula exactamente UNA pregunta de detección que cite exámenes/laboratorios reales (combina chequeos relacionados en una sola pregunta).
   7. Otras categorías - cualquier otro tema (impacto funcional, estilo de vida, alergias, desencadenantes, temporalidad, etc.) queda limitado a UNA pregunta por categoría.
   8. Pregunta de cierre - "algo más" solo después de >=10 preguntas totales o cuando ya no existan temas de alto valor pendientes.

EVITA DUPLICADOS:
   - No separes apetito, peso y energía en múltiples preguntas; si uno se cubre, omite los demás.
   - No reformules preguntas ya realizadas; usa el contexto semántico para evitar redundancias.

PRIORIDADES SEGÚN TIPO DE CONDICIÓN (generaliza con criterio clínico):

**Enfermedades crónicas (diabetes, hipertensión, asma, tiroides, cardiopatías, cáncer, EPOC, etc.)**
   - Alta prioridad: duración, combinación medicamentos/cuidados caseros, monitoreo crónico (2 preguntas), detección/complicaciones (1 pregunta), historia familiar (dentro de las primeras 5), laboratorios/imágenes recientes.
   - Ejemplos de monitoreo:
      - Diabetes - registros de glucosa, CGM, HbA1c, hipoglucemias.
      - Hipertensión - registros de PA domiciliaria, calibración de tensiómetro.
      - Asma - registros de peak flow, uso de inhalador de rescate, diario de síntomas.
      - Tiroides - TSH/T4 recientes, frecuencia cardíaca, síntomas de hipo/hipertiroidismo.
      - Cardiopatías - BP/HR domiciliarios, saturación/oxígeno, control de peso.
   - Ejemplos de detección/complicaciones (mención en la pregunta):
      - Diabetes - fondo de ojo, examen de pies, revisión dental, microalbuminuria/eGFR.
      - Hipertensión - función renal, examen retinal, pruebas cardíacas (ECG/eco/estrés).
      - Asma - espirometría, panel de alergias, radiografía/TC de tórax, peak flow basal.
      - Tiroides - examen de cuello/ultrasonido, densitometría ósea, evaluación cardíaca.
      - Cardiopatías - panel lipídico, pruebas de esfuerzo, Holter/ECG, ecocardiograma.
      - Cáncer - imágenes (CT/MRI/PET), marcadores tumorales, biopsias, estadificación.
   - Prioridad media: síntomas asociados, estilo de vida (dieta, ejercicio, tabaco), impacto funcional.
   - Baja prioridad: viajes, alergias (salvo relación evidente).
   - Prohibido salvo que el médico lo solicite explícitamente: preguntas genéricas de historial médico/pasos similares en el pasado ("¿Ha tenido episodios parecidos antes?", "¿Qué enfermedades ha tenido?", etc.). Mantén el enfoque en el control actual y el monitoreo.
   - Combina detección y complicaciones en una sola pregunta rica.

**Condiciones agudas/infecciosas (fiebre, tos, diarrea, infecciones, etc.)**
   - Alta prioridad: duración, síntomas asociados, medicamentos/cuidados caseros, viajes (solo si hay riesgo), episodios previos similares.
   - Media: evaluación de dolor (si corresponde), impacto en estilo de vida/aislamiento.
   - Prohibido: historia familiar, monitoreo/detección crónica.

**Condiciones dolorosas (cefalea, dolor torácico, musculoesquelético)**
   - Alta prioridad: duración, caracterización del dolor, desencadenantes/alivio, impacto funcional, medicamentos/cuidados caseros, episodios previos similares.
   - Media: síntomas asociados, factores de estilo de vida (sueño, estrés).
   - Prohibido: historia familiar, monitoreo crónico (salvo dolor crónico documentado).

**Salud de la mujer (menstruación, fertilidad, embarazo, menopausia, SOP, dolor pélvico)**
   - Alta prioridad: duración, combinación medicamentos/cuidados caseros, síntomas hormonales/ginecológicos, tamizajes recientes (Pap, mamografía, densitometría, análisis hormonales), historia familiar de cáncer de mama/ovario o trastornos endocrinos (dentro de las primeras 5 si es crónico).
   - Media: síntomas asociados, estilo de vida, impacto funcional.
   - Baja: viajes, alergias (salvo relación directa).

ENFOQUE POR EDAD:
   - <12 años: nunca hacer preguntas menstruales; prioriza causas generales y seguridad.
   - 10-18: abordar regularidad menstrual, pubertad, SOP, cólicos/pain con la regla.
   - 19-40: fertilidad, endometriosis, embarazo/obstetricia, alteraciones menstruales.
   - 41-60: perimenopausia, menopausia, salud ósea, cambios hormonales.
   - >60: sin preguntas menstruales; enfócate en integración con enfermedades crónicas y tamizajes.
   - Todas las edades: si hay enfermedad crónica, pregunta historia familiar una vez y en las primeras 5 preguntas.

REGLAS GENERALES:
   - Respeta estrictamente los límites (síntomas asociados <=3, monitoreo crónico =2, detección =1, resto <=1).
   - Preguntas de viaje solo si la casilla está marcada y el cuadro clínico lo amerita.
   - Preguntas menstruales/pregnancy solo con género, edad y síntomas pertinentes.
   - No formules la pregunta final antes de tiempo.
   - Las categorías elegidas por el médico tienen prioridad, sin romper el flujo global; elige la pregunta más relevante médicamente dentro de cada categoría seleccionada.

Los ejemplos son ilustrativos: aplica tu mejor criterio clínico para enfermedades raras o complejas usando estas mismas prioridades.

{{
    "condition_properties": {{
        "is_chronic": <true/false - ¿Es una condición crónica que dura >3 meses?>,
        "is_hereditary": <true/false - ¿Tiene componente genético/familiar?>,
        "has_complications": <true/false - ¿Requiere monitoreo de complicaciones?>,
        "is_acute_emergency": <true/false - ¿Es aguda y de inicio reciente <2 semanas?>,
        "is_pain_related": <true/false - ¿El dolor es síntoma principal?>,
        "is_womens_health": <true/false - ¿Relacionado con salud de la mujer (menstruación, embarazo, ginecología)?>,
        "is_allergy_related": <true/false - ¿Involucra alergias o reacciones alérgicas?>,
        "requires_lifestyle_assessment": <true/false - ¿Factores de estilo de vida son relevantes?>
    }},
    "priority_topics": [
        <lista ordenada de temas médicos a explorar según REGLAS arriba>
        Opciones disponibles (EN ESTE ORDEN EXACTO DE SECUENCIA - solo incluir si es relevante para los síntomas):
        1. "duración" - SIEMPRE PRIMERO si no se ha preguntado
        2. "síntomas_asociados", "caracterización_síntomas" - Después de duración (máximo 2-3 preguntas)
        3. "medicamentos_actuales" - Después de síntomas asociados (incluye medicamentos Y remedios caseros, máximo 1 pregunta)
        4. "evaluación_dolor", "caracterización_dolor" - Solo si el dolor es parte de los síntomas
        5. "historial_viajes" - Solo si recently_travelled=True Y síntomas son relevantes para viajes
        6. "monitoreo_crónico" - Solo para enfermedades crónicas (OBLIGATORIO: hacer exactamente 2 preguntas)
        7. "detección" - Solo para enfermedades crónicas (OBLIGATORIO: hacer exactamente 1 pregunta sobre exámenes específicos)
        8. "alergias" - Solo si síntomas relacionados con alergias
        9. "historial_médico_previo", "hpi" - Solo si es relevante para síntomas actuales O el médico lo seleccionó explícitamente (omitir en seguimientos crónicos) (máximo 1 pregunta)
        10. "ciclo_menstrual" - SOLO para condiciones de salud de la mujer (NO para hipertensión/diabetes/dolor de cabeza)
        11. "impacto_funcional", "impacto_diario" - Después de preguntas centrales
        12. "factores_estilo_vida" - Después de funcional (máximo 1 pregunta)
        13. "desencadenantes", "factores_agravantes" - Después de estilo de vida
        14. "temporal", "progresión", "frecuencia" - Después de desencadenantes
        15. "otros", "exploratorio" - Último, después de todas las categorías anteriores
        También disponible (agregar cuando sea relevante):
        "historia_familiar" - OBLIGATORIO para enfermedades crónicas (máximo 1 pregunta)
        
        IMPORTANTE:
        - Sigue esta secuencia ESTRICTAMENTE - NO te saltes categorías anteriores si aún son relevantes
        - Solo incluye temas que cumplen las REGLAS MÉDICAS arriba Y son relevantes para los síntomas
        - Revisa brechas de información para ver qué categorías aún se necesitan
    ],
    "avoid_topics": [
        <lista de temas que NO son relevantes para esta condición>
    ],
    "medical_reasoning": "<1-2 oraciones explicando el razonamiento médico>"
}}

REGLAS CRÍTICAS DE SEGURIDAD (no violar):
1. Si género = masculino → NUNCA incluir "ciclo_menstrual" en priority_topics
2. Si edad < 12 o > 60 → NUNCA incluir preguntas menstruales
3. Si "Ha viajado recientemente" = No → NUNCA incluir "historial_viajes" en priority_topics
4. Priorizar información médicamente relevante según el tipo de condición
5. Solo incluir temas que son REALMENTE necesarios para evaluar esta condición específica

Devuelve SOLO el JSON, sin texto adicional."""

        else:  # English
            system_prompt = """You are an expert physician analyzing a patient's chief complaint.

Your job is to determine WHAT medical information is essential to collect for this specific case.

Analyze the condition medically and return a structured analysis in JSON format."""

            user_prompt = f"""
Analyze this patient's chief complaint: "{chief_complaint}"
Patient age: {patient_age or "Unknown"}
Patient gender: {patient_gender or "Not specified"}
Recently traveled: {"Yes" if recently_travelled else "No"}

Perform a comprehensive medical analysis and return a JSON with this EXACT structure:

MEDICAL RULES FOR PRIORITY_TOPICS (FOLLOW EXACTLY):

FLOW PRIORITY - apply to every intake (skip only when medically irrelevant):
   1. Duration - must be the very first question.
   2. Combined medications + home/self-care - second question, covering prescriptions AND over-the-counter/herbal/home remedies in one question.
   3. Disease-specific high-priority items - ask next (see sections below). If the doctor supplied selected categories, address those first while keeping this global order. Within a selected category, choose the question a general physician would prioritise.
   4. Associated symptoms - only if relevant, maximum 3 unique questions (never rephrase the same symptom).
   5. Chronic monitoring - if is_chronic=true, ask exactly TWO monitoring questions (home logs, device readings, treatment adherence, trend tracking).
   6. Chronic screening/complications - if is_chronic=true, ask exactly ONE screening question referencing real exams/labs (combine related checks in a single question).
   7. Other categories - everything else (functional impact, lifestyle, allergies, triggers, temporal progression, etc.) is limited to ONE question each.
   8. Closing question - "anything else" only after >=10 total questions or when no high-value topics remain.

DUPLICATE SAFEGUARDS:
   - Never split appetite/weight/energy across separate questions; once one is covered, do not ask the others.
   - Do not rephrase previously covered information; rely on semantic understanding to avoid duplicates.

CONDITION-SPECIFIC PRIORITIES (examples generalise to similar illnesses; adapt using clinical reasoning):

**Chronic diseases (diabetes, hypertension, asthma, thyroid, heart disease, cancer, COPD, etc.)**
   - High priority (ask early): duration, combined meds/home care, chronic monitoring (2 questions), screening/complication exam (1 question), family history (within first 5), recent labs/imaging.
   - Monitoring examples:
      - Diabetes - home glucose logs, CGM trends, HbA1c, hypoglycaemia episodes.
      - Hypertension - home BP logs, cuff calibration, sudden spikes.
      - Asthma - peak-flow records, rescue inhaler frequency, symptom diary.
      - Thyroid - recent TSH/FT4 results, heart rate checks.
      - Heart disease - BP/HR logs, home oxygen use if applicable.
   - Screening/complication examples (reference real tests in the screening question):
      - Diabetes - eye exam, foot exam, dental check, kidney labs (microalbumin, eGFR).
      - Hypertension - kidney function labs, retinal exam, cardiac imaging/stress tests.
      - Asthma - pulmonary function test, allergy panels, chest imaging, baseline peak-flow.
      - Thyroid - neck exam/ultrasound, bone density, cardiac rhythm evaluation.
      - Heart disease - lipid panel, stress test, echocardiogram, Holter/ECG.
      - Cancer - imaging (CT/MRI/PET), tumour markers, biopsies, staging labs.
   - Medium priority: associated symptoms, lifestyle factors (diet, exercise, smoking), functional impact.
   - Low priority: travel history, allergies (unless clearly linked).
   - Forbidden unless clinician explicitly requests: generic past medical history questions ("Have you experienced this before?", "Any similar episodes in the past?", "What past conditions have you had?"). Stay focused on current control and monitoring.
   - Combine screening + complication context in ONE rich question.

**Acute infections (fever, cough, gastroenteritis, URI, etc.)**
   - High priority: duration, associated symptoms, combined meds/home care, travel history (only if risk factors), past similar episodes.
   - Medium: pain assessment (when applicable), lifestyle/contagion considerations.
   - Forbidden: family history, chronic monitoring/screening.

**Pain conditions (headache, chest pain, musculoskeletal pain)**
   - High priority: duration, pain character/severity, triggers/relieving factors, functional impact, combined meds/home care, past similar episodes.
   - Medium: associated symptoms, lifestyle factors (sleep, stress).
   - Forbidden: family history, chronic monitoring (unless chronic pain management is the focus).

**Women's health (menstrual, fertility, pregnancy, menopause, PCOS, pelvic pain)**
   - High priority: duration, combined meds/home care, hormonal/gynecologic symptoms, recent screenings (Pap smear, pelvic exam, mammogram, bone density, hormone panels), family history of breast/ovarian/endocrine cancers (within first 5 if chronic context).
   - Medium: associated symptoms, lifestyle, functional impact.
   - Low: travel, allergies (unless related).

AGE-FOCUSED GUIDANCE:
   - <12 years: never ask menstrual questions; emphasise general causes and safety.
   - 10-18: include puberty, cycle regularity, PCOS, cramps when relevant.
   - 19-40: fertility, endometriosis, pregnancy history, menstrual irregularities.
   - 41-60: perimenopause, menopause, hormone changes, bone health.
   - >60: no menstrual questions; focus on chronic disease interplay and relevant screenings.
   - All ages: when chronic disease present, ask family history once early (within first 5).

GENERAL RULES:
   - Respect per-category limits strictly (associated symptoms <=3, chronic monitoring =2, screening =1, every other category <=1).
   - Travel questions only when checkbox is true AND symptoms justify it.
   - Menstrual/pregnancy questions require appropriate gender, age, and symptom relevance.
   - Never ask the final "anything else" question prematurely.
   - Doctor-selected categories override the generic order, but do NOT violate the flow; choose the most medically relevant question within each selected category.

Examples are illustrative—always apply best clinical reasoning for any condition, including rare or complex presentations.

IMPORTANT: These examples are illustrative—always apply your best clinical reasoning for any condition.

{{
    "condition_properties": {{
        "is_chronic": <true/false - Is this a chronic condition lasting >3 months?>,
        "is_hereditary": <true/false - Does it have genetic/familial component?>,
        "has_complications": <true/false - Does it require complication monitoring?>,
        "is_acute_emergency": <true/false - Is it acute with recent onset <2 weeks?>,
        "is_pain_related": <true/false - Is pain the primary symptom?>,
        "is_womens_health": <true/false - Related to women's health (menstruation, pregnancy, gynecology)?>,
        "is_allergy_related": <true/false - Involves allergies or allergic reactions?>,
        "requires_lifestyle_assessment": <true/false - Are lifestyle factors relevant?>
    }},
    "priority_topics": [
        <ordered list of medical topics to explore according to RULES above>
        Available options (IN THIS EXACT SEQUENCE ORDER - only include if relevant to symptoms):
        1. "duration" - ALWAYS FIRST if not already asked
        2. "associated_symptoms", "symptom_characterization" - After duration (2-3 questions max)
        3. "current_medications" - After associated symptoms (includes medications AND home remedies, 1 question max)
        4. "pain_assessment", "pain_characterization" - Only if pain is part of symptoms
        5. "travel_history" - Only if recently_travelled=True AND symptoms are travel-relevant
        6. "chronic_monitoring" - Only for chronic diseases (MANDATORY: ask exactly 2 questions)
        7. "screening" - Only for chronic diseases (MANDATORY: ask exactly 1 question about specific exams/checkups)
        8. "allergies" - Only if allergy-related symptoms
        9. "past_medical_history", "hpi" - Only if relevant to current symptoms OR doctor explicitly selected it (skip for chronic follow-up)
        10. "menstrual_cycle" - ONLY for women's health conditions (NOT for hypertension/diabetes/headache)
        11. "functional_impact", "daily_impact" - After core questions
        12. "lifestyle_factors" - After functional (1 question max)
        13. "triggers", "aggravating_factors" - After lifestyle
        14. "temporal", "progression", "frequency" - After triggers
        15. "other", "exploratory" - Last, after all above categories
        Also available (add when relevant):
        "family_history" - MANDATORY for chronic diseases (1 question max)
        
        IMPORTANT: 
        - Follow this sequence STRICTLY - do NOT skip ahead to later categories if earlier ones are still relevant
        - Only include topics that meet the MEDICAL RULES above AND are relevant to the symptoms
        - Check information_gaps to see which categories are still needed
    ],
    "avoid_topics": [
        <list of topics that are NOT relevant for this condition>
    ],
    "medical_reasoning": "<1-2 sentences explaining the medical reasoning>"
}}

CRITICAL SAFETY RULES (do not violate):
1. If gender = male → NEVER include "menstrual_cycle" in priority_topics
2. If age < 12 or > 60 → NEVER include menstrual questions
3. If "Recently traveled" = No → NEVER include "travel_history" in priority_topics
4. Prioritize medically relevant information based on condition type
5. Only include topics that are TRULY necessary to evaluate this specific condition

Return ONLY the JSON, no additional text."""

        try:
            response = await self._client.chat(
                model=self._settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.2,  # Low temperature for consistency
            )
            
            # Extract text from response object
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            analysis = json.loads(json_match.group())
            
            # Validate and enforce hard constraints
            analysis = self._enforce_safety_constraints(
                analysis, 
                patient_age, 
                patient_gender,
                recently_travelled
            )
            
            logger.info(
                f"Medical context analysis: {analysis['condition_properties']}, "
                f"Priority topics: {analysis['priority_topics']}"
            )
            
            return MedicalContext(
                chief_complaint=chief_complaint,
                condition_properties=analysis["condition_properties"],
                priority_topics=analysis["priority_topics"],
                avoid_topics=analysis.get("avoid_topics", []),
                medical_reasoning=analysis.get("medical_reasoning", ""),
                patient_age=patient_age,
                patient_gender=patient_gender
            )
            
        except Exception as e:
            logger.error(f"Medical context analysis failed: {e}")
            # Fallback to conservative approach
            return self._create_fallback_context(chief_complaint, patient_age, patient_gender, recently_travelled)
    
    def _enforce_safety_constraints(
        self,
        analysis: Dict[str, Any],
        patient_age: Optional[int],
        patient_gender: Optional[str],
        recently_travelled: bool = False
    ) -> Dict[str, Any]:
        """Enforce hard safety constraints that cannot be violated"""
        
        priority_topics = analysis.get("priority_topics", [])
        avoid_topics = analysis.get("avoid_topics", [])
        
        # HARD CONSTRAINT 1: Male patients - no menstrual questions
        if patient_gender and patient_gender.lower() in ["male", "m", "masculino", "hombre"]:
            if "menstrual_cycle" in priority_topics:
                priority_topics.remove("menstrual_cycle")
            if "ciclo_menstrual" in priority_topics:
                priority_topics.remove("ciclo_menstrual")
            if "menstrual_cycle" not in avoid_topics:
                avoid_topics.append("menstrual_cycle")
            if "ciclo_menstrual" not in avoid_topics:
                avoid_topics.append("ciclo_menstrual")
            logger.warning("Removed menstrual questions for male patient (safety constraint)")
        
        # HARD CONSTRAINT 2: Age-based menstrual filtering
        if patient_age is not None:
            if patient_age < 12 or patient_age > 60:
                if "menstrual_cycle" in priority_topics:
                    priority_topics.remove("menstrual_cycle")
                if "ciclo_menstrual" in priority_topics:
                    priority_topics.remove("ciclo_menstrual")
                if "menstrual_cycle" not in avoid_topics:
                    avoid_topics.append("menstrual_cycle")
                if "ciclo_menstrual" not in avoid_topics:
                    avoid_topics.append("ciclo_menstrual")
                logger.warning(f"Removed menstrual questions for age {patient_age} (safety constraint)")
        
        # HARD CONSTRAINT 3: Travel history only if patient traveled
        if not recently_travelled:
            if "travel_history" in priority_topics:
                priority_topics.remove("travel_history")
            if "historial_viajes" in priority_topics:
                priority_topics.remove("historial_viajes")
            if "travel_history" not in avoid_topics:
                avoid_topics.append("travel_history")
            if "historial_viajes" not in avoid_topics:
                avoid_topics.append("historial_viajes")
            logger.info("Removed travel_history - patient has not traveled recently (safety constraint)")
        
        analysis["priority_topics"] = priority_topics
        analysis["avoid_topics"] = list(set(avoid_topics))  # Remove duplicates
        
        return analysis
    
    def _create_fallback_context(
        self,
        chief_complaint: str,
        patient_age: Optional[int],
        patient_gender: Optional[str],
        recently_travelled: bool = False
    ) -> MedicalContext:
        """Create safe fallback context if analysis fails"""
        
        priority_topics = [
            "duration",
            "current_medications",
            "symptom_characterization",
            "associated_symptoms"
        ]
        
        avoid_topics = []
        
        # Apply safety constraints
        is_male = patient_gender and patient_gender.lower() in ["male", "m", "masculino", "hombre"]
        age_inappropriate = patient_age and (patient_age < 12 or patient_age > 60)
        
        if is_male or age_inappropriate:
            avoid_topics.extend(["menstrual_cycle", "ciclo_menstrual"])
        
        # Apply travel constraint
        if not recently_travelled:
            avoid_topics.extend(["travel_history", "historial_viajes"])
        
        return MedicalContext(
            chief_complaint=chief_complaint,
            condition_properties={
                "is_chronic": False,
                "is_hereditary": False,
                "has_complications": False,
                "is_acute_emergency": True,
                "is_pain_related": False,
                "is_womens_health": False,
                "is_allergy_related": False,
                "requires_lifestyle_assessment": False
            },
            priority_topics=priority_topics,
            avoid_topics=avoid_topics,
            medical_reasoning="Using conservative fallback approach",
            patient_age=patient_age,
            patient_gender=patient_gender
        )
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code"""
        if not language:
            return "en"
        normalized = language.lower().strip()
        if normalized in ['es', 'sp']:
            return 'sp'
        return normalized if normalized in ['en', 'sp'] else 'en'


# ============================================================================
# AGENT 2: Answer Extractor
# ============================================================================

@dataclass
class ExtractedInformation:
    """Information already collected from patient"""
    topics_covered: List[str]  # What has been asked
    information_gaps: List[str]  # What's still needed
    extracted_facts: Dict[str, str]  # Key facts from answers
    already_mentioned_duration: bool
    already_mentioned_medications: bool
    redundant_categories: List[str]
    topic_counts: Optional[Dict[str, int]] = None  # Count of how many times each topic was asked


class AnswerExtractor:
    """Agent 2: Extracts information from previous Q&A to avoid redundancy"""
    
    def __init__(self, client, settings):
        self._client = client
        self._settings = settings
    
    async def extract_covered_information(
        self,
        asked_questions: List[str],
        previous_answers: List[str],
        medical_context: MedicalContext,
        language: str = "en"
    ) -> ExtractedInformation:
        """
        Analyze ALL Q&A pairs to understand what's been covered.
        Prevents asking redundant questions by tracking all previously asked topics.
        """
        
        # Get ALL Q&A pairs for comprehensive duplicate checking
        all_qa = []
        for i in range(len(asked_questions)):
            if i < len(previous_answers):
                all_qa.append({
                    "question": asked_questions[i],
                    "answer": previous_answers[i]
                })
        
        # Get last 3 Q&A pairs for detailed context analysis
        recent_qa = []
        for i in range(max(0, len(asked_questions) - 3), len(asked_questions)):
            if i < len(previous_answers):
                recent_qa.append({
                    "question": asked_questions[i],
                    "answer": previous_answers[i]
                })
        
        if not all_qa:
            # No previous Q&A, nothing covered yet
            return ExtractedInformation(
                topics_covered=[],
                information_gaps=medical_context.priority_topics,
                extracted_facts={},
                already_mentioned_duration=False,
                already_mentioned_medications=False,
                redundant_categories=[],
                topic_counts=None
            )
        
        lang = self._normalize_language(language)
        
        if lang == "sp":
            system_prompt = """Eres un asistente médico analizando respuestas de admisión del paciente.

Tu trabajo es extraer QUÉ información ya se ha recopilado para evitar preguntas redundantes."""

            user_prompt = f"""
TODAS LAS PREGUNTAS Y RESPUESTAS (para verificación de duplicados):
{self._format_qa_pairs(all_qa)}

{"ANÁLISIS DETALLADO (últimas 3 respuestas):" if len(recent_qa) < len(all_qa) else ""}
{self._format_qa_pairs(recent_qa) if len(recent_qa) < len(all_qa) else ""}

Contexto médico:
- Queja principal: {medical_context.chief_complaint}
- Temas prioritarios a cubrir: {medical_context.priority_topics}

TAREA IMPORTANTE: Analiza TODAS las {len(all_qa)} preguntas arriba para determinar qué información YA se ha recopilado.

Devuelve un JSON con esta estructura EXACTA:

{{
    "topics_covered": [
        <lista de temas que YA fueron preguntados/respondidos EN CUALQUIER PREGUNTA>
        Opciones (en orden de secuencia): "duración", "síntomas_asociados", "caracterización_síntomas",
                 "medicamentos_actuales", "evaluación_dolor", "caracterización_dolor",
                 "historial_viajes", "monitoreo_crónico", "detección", "alergias",
                 "historial_médico_previo", "hpi", "ciclo_menstrual", "impacto_funcional",
                 "impacto_diario", "factores_estilo_vida", "desencadenantes", "factores_agravantes",
                 "temporal", "progresión", "frecuencia", "otros", "exploratorio",
                 "historia_familiar"
    ],
    "information_gaps": [
        <lista de temas prioritarios que AÚN NO se han preguntado>
    ],
    "extracted_facts": {{
        "duration": "<si se mencionó duración EN CUALQUIER RESPUESTA, extrae aquí (ej. '3 semanas'), sino null>",
        "medications": "<si se mencionaron medicamentos EN CUALQUIER RESPUESTA, lista aquí, sino null>",
        "pain_severity": "<si se mencionó severidad EN CUALQUIER RESPUESTA (1-10), sino null>",
        "associated_symptoms": "<si se mencionaron síntomas EN CUALQUIER RESPUESTA, lista aquí, sino null>"
    }},
    "already_mentioned_duration": <true/false - ¿El paciente ya mencionó duración EN CUALQUIER RESPUESTA?>,
    "already_mentioned_medications": <true/false - ¿Ya se preguntó sobre medicamentos EN CUALQUIER PREGUNTA?>,
    "redundant_categories": [
        <categorías que NO deben preguntarse de nuevo porque ya fueron cubiertas EN CUALQUIER PREGUNTA>
    ]
}}

REGLAS CRÍTICAS PARA MARCAR REDUNDANT_CATEGORIES (revisa TODAS las categorías en orden de secuencia):
1. Revisa TODAS las {len(all_qa)} preguntas arriba, no solo las últimas 3
2. Si "duración" fue preguntada EXPLÍCITAMENTE en CUALQUIER pregunta → already_mentioned_duration = true Y añade "duración" a redundant_categories
3. Si "síntomas asociados" o "otros síntomas" fueron preguntados EXPLÍCITAMENTE → añade "síntomas_asociados" a redundant_categories SOLO SI YA FUERON PREGUNTADOS (máximo 2-3 preguntas)
4. Si "caracterización de síntomas" o "describe los síntomas" fue preguntada EXPLÍCITAMENTE → añade "caracterización_síntomas" a redundant_categories SOLO SI YA FUE PREGUNTADA
5. Si "medicamentos" o "remedios caseros" fueron preguntados EXPLÍCITAMENTE en CUALQUIER pregunta → already_mentioned_medications = true Y añade "medicamentos_actuales" a redundant_categories (máximo 1 pregunta)
6. Si "dolor" o "evaluación del dolor" fue preguntado EXPLÍCITAMENTE → añade "evaluación_dolor" o "caracterización_dolor" a redundant_categories SOLO SI YA FUE PREGUNTADO
7. Si "viajes" o "historial de viajes" fue preguntado EXPLÍCITAMENTE → añade "historial_viajes" a redundant_categories SOLO SI YA FUE PREGUNTADO
8. Si "monitoreo crónico" o monitoreo de lecturas/registros fue preguntado EXPLÍCITAMENTE → añade "monitoreo_crónico" a redundant_categories SOLO SI YA FUE PREGUNTADO (máximo 2 preguntas en total)
9. Si "detección" o "chequeos" o exámenes específicos (ojos/pies/dental) fueron preguntados EXPLÍCITAMENTE → añade "detección" a redundant_categories SOLO SI YA FUE PREGUNTADO (máximo 1 pregunta en total)
10. Si "alergias" fue preguntado EXPLÍCITAMENTE → añade "alergias" a redundant_categories SOLO SI YA FUE PREGUNTADO
11. Si "historial médico previo" o "episodios anteriores" o "HPI" fue preguntado EXPLÍCITAMENTE → añade "historial_médico_previo" o "hpi" a redundant_categories SOLO SI YA FUE PREGUNTADO (máximo 1 pregunta)
12. Si "ciclo menstrual" o "período" fue preguntado EXPLÍCITAMENTE → añade "ciclo_menstrual" a redundant_categories SOLO SI YA FUE PREGUNTADO
13. Si "impacto funcional" o "actividades diarias" o "vida diaria" fue preguntado EXPLÍCITAMENTE → añade "impacto_funcional" o "impacto_diario" a redundant_categories SOLO SI YA FUE PREGUNTADO
14. Si "factores de estilo de vida" o "cambios en el estilo de vida" fue preguntado EXPLÍCITAMENTE → añade "factores_estilo_vida" a redundant_categories SOLO SI YA FUE PREGUNTADO (máximo 1 pregunta)
15. Si "desencadenantes" o "qué lo empeora/mejora" o "factores agravantes" fue preguntado EXPLÍCITAMENTE → añade "desencadenantes" o "factores_agravantes" a redundant_categories SOLO SI YA FUE PREGUNTADO
16. Si "temporal" o "progresión" o "frecuencia" o "con qué frecuencia" o "con el tiempo" fue preguntado EXPLÍCITAMENTE → añade "temporal" o "progresión" o "frecuencia" a redundant_categories SOLO SI YA FUE PREGUNTADO
17. Si "historia familiar" fue preguntada EXPLÍCITAMENTE → añade "historia_familiar" a redundant_categories SOLO SI YA FUE PREGUNTADA (máximo 1 pregunta, obligatorio para enfermedades crónicas)

IMPORTANTE:
- SÉ CONSERVADOR: solo marca un tema como redundante si la pregunta REALMENTE lo preguntó directamente
- NO marques temas que solo se mencionaron de paso o implícitamente
- information_gaps debe contener temas que AÚN NO se han preguntado específicamente
- Solo añade a redundant_categories cuando estés 100% seguro de que ese tema específico fue cubierto

MANTÉN information_gaps con al menos 2-3 temas hasta que el paciente haya respondido al menos 5-6 preguntas.

Devuelve SOLO el JSON, sin texto adicional."""

        else:  # English
            system_prompt = """You are a medical assistant analyzing patient intake responses.

Your job is to extract WHAT information has already been collected to avoid redundant questions."""

            user_prompt = f"""
ALL QUESTIONS AND ANSWERS (for duplicate checking):
{self._format_qa_pairs(all_qa)}

{"DETAILED ANALYSIS (last 3 responses):" if len(recent_qa) < len(all_qa) else ""}
{self._format_qa_pairs(recent_qa) if len(recent_qa) < len(all_qa) else ""}

Medical context:
- Chief complaint: {medical_context.chief_complaint}
- Priority topics to cover: {medical_context.priority_topics}

IMPORTANT TASK: Analyze ALL {len(all_qa)} questions above to determine what information HAS been collected.

Return a JSON with this EXACT structure:

{{
    "topics_covered": [
        <list of topics that were ALREADY asked/answered IN ANY QUESTION>
        Options (in sequence order): "duration", "associated_symptoms", 
                 "current_medications", "pain_assessment", "pain_characterization",
                 "travel_history", "chronic_monitoring", "screening", "allergies",
                 "past_medical_history", "hpi", "menstrual_cycle", "functional_impact",
                 "daily_impact", "lifestyle_factors", "triggers", "aggravating_factors",
                 "temporal", "progression", "frequency", "other", "exploratory",
                 "family_history"
    ],
    "information_gaps": [
        <list of priority topics that have NOT been asked yet>
    ],
    "extracted_facts": {{
        "duration": "<if duration mentioned IN ANY ANSWER, extract here (e.g. '3 weeks'), else null>",
        "medications": "<if medications mentioned IN ANY ANSWER, list here, else null>",
        "pain_severity": "<if pain severity mentioned IN ANY ANSWER (1-10), else null>",
        "associated_symptoms": "<if associated symptoms mentioned IN ANY ANSWER, list here, else null>"
    }},
    "already_mentioned_duration": <true/false - Did patient already mention duration IN ANY ANSWER?>,
    "already_mentioned_medications": <true/false - Were medications already asked IN ANY QUESTION?>,
    "redundant_categories": [
        <categories that should NOT be asked again because already covered IN ANY QUESTION>
    ]
}}

CRITICAL RULES FOR MARKING REDUNDANT_CATEGORIES (check ALL categories in sequence order):
1. Review ALL {len(all_qa)} questions above, not just the last 3
2. If "duration" was asked EXPLICITLY in ANY previous question → already_mentioned_duration = true AND add "duration" to redundant_categories
3. If "associated symptoms" or "other symptoms" were asked EXPLICITLY → add "associated_symptoms" to redundant_categories ONLY IF ALREADY ASKED (max 2-3 questions)
4. If "symptom characterization" or "describe symptoms" was asked EXPLICITLY → add "symptom_characterization" to redundant_categories ONLY IF ALREADY ASKED
5. If "medications" or "home remedies" were asked EXPLICITLY in ANY previous question → already_mentioned_medications = true AND add "current_medications" to redundant_categories (max 1 question)
6. If "pain" or "pain assessment" was asked EXPLICITLY → add "pain_assessment" or "pain_characterization" to redundant_categories ONLY IF ALREADY ASKED
7. If "travel" or "travel history" was asked EXPLICITLY → add "travel_history" to redundant_categories ONLY IF ALREADY ASKED
8. If "chronic monitoring" or monitoring of readings/logs was asked EXPLICITLY → add "chronic_monitoring" to redundant_categories ONLY IF ALREADY ASKED (max 2 questions total)
9. If "screening" or "checkups" or specific exams (eye/foot/dental) were asked EXPLICITLY → add "screening" to redundant_categories ONLY IF ALREADY ASKED (max 1 question total)
10. If "allergies" was asked EXPLICITLY → add "allergies" to redundant_categories ONLY IF ALREADY ASKED
11. If "past medical history" or "previous episodes" or "HPI" was asked EXPLICITLY → add "past_medical_history" or "hpi" to redundant_categories ONLY IF ALREADY ASKED (max 1 question)
12. If "menstrual cycle" or "period" was asked EXPLICITLY → add "menstrual_cycle" to redundant_categories ONLY IF ALREADY ASKED
13. If "functional impact" or "daily activities" or "daily life" was asked EXPLICITLY → add "functional_impact" or "daily_impact" to redundant_categories ONLY IF ALREADY ASKED
14. If "lifestyle factors" or "lifestyle changes" was asked EXPLICITLY → add "lifestyle_factors" to redundant_categories ONLY IF ALREADY ASKED (max 1 question)
15. If "triggers" or "what makes it worse/better" or "aggravating factors" was asked EXPLICITLY → add "triggers" or "aggravating_factors" to redundant_categories ONLY IF ALREADY ASKED
16. If "temporal" or "progression" or "frequency" or "how often" or "over time" was asked EXPLICITLY → add "temporal" or "progression" or "frequency" to redundant_categories ONLY IF ALREADY ASKED
17. If "family history" was asked EXPLICITLY → add "family_history" to redundant_categories ONLY IF ALREADY ASKED (max 1 question, mandatory for chronic diseases)

IMPORTANT:
- Be CONSERVATIVE: only mark a topic as redundant if the question ACTUALLY asked about it directly
- DO NOT mark topics that were only mentioned in passing or implicitly
- information_gaps should contain topics that have NOT been specifically asked yet
- Only add to redundant_categories when you are 100% certain that specific topic was covered

KEEP information_gaps with at least 2-3 topics until the patient has answered at least 5-6 questions.

Return ONLY the JSON, no additional text."""

        try:
            response = await self._client.chat(
                model=self._settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.1,  # Very low for consistency
            )
            
            # Extract text from response object
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response with robust error handling
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON found in Agent 2 response: {response_text[:200]}")
                raise ValueError("No JSON found in response")
            
            try:
                extraction = json.loads(json_match.group())
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error in Agent 2: {je}. Response: {json_match.group()[:200]}")
                raise
            
            # Validate JSON structure
            required_keys = ["topics_covered", "information_gaps", "redundant_categories"]
            missing_keys = [k for k in required_keys if k not in extraction]
            if missing_keys:
                logger.warning(f"Agent 2 JSON missing keys: {missing_keys}. Using defaults.")
            
            # Extract values with defaults
            topics_covered = extraction.get("topics_covered", [])
            information_gaps = extraction.get("information_gaps", medical_context.priority_topics)
            redundant_categories = extraction.get("redundant_categories", [])
            
            # Ensure lists are actually lists
            if not isinstance(topics_covered, list):
                topics_covered = []
            if not isinstance(information_gaps, list):
                information_gaps = medical_context.priority_topics
            if not isinstance(redundant_categories, list):
                redundant_categories = []
            
            logger.info(
                f"Agent 2 Results: Covered={topics_covered}, "
                f"Gaps={information_gaps}, Redundant={redundant_categories} "
                f"(analyzed {len(all_qa)} total Q&A pairs)"
            )
            
            # Calculate topic_counts from topics_covered if not provided
            topic_counts = extraction.get("topic_counts")
            if topic_counts is None:
                # Count occurrences of each topic in topics_covered
                from collections import Counter
                topic_counts = dict(Counter(topics_covered))
            
            return ExtractedInformation(
                topics_covered=topics_covered,
                information_gaps=information_gaps,
                extracted_facts=extraction.get("extracted_facts", {}),
                already_mentioned_duration=extraction.get("already_mentioned_duration", False),
                already_mentioned_medications=extraction.get("already_mentioned_medications", False),
                redundant_categories=redundant_categories,
                topic_counts=topic_counts
            )
            
        except Exception as e:
            logger.error(f"Answer extraction failed: {e}")
            # Fallback: assume nothing covered
            return ExtractedInformation(
                topics_covered=[],
                information_gaps=medical_context.priority_topics,
                extracted_facts={},
                already_mentioned_duration=False,
                already_mentioned_medications=False,
                redundant_categories=[],
                topic_counts=None
            )
    
    def _format_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> str:
        """Format Q&A pairs for analysis"""
        formatted = []
        for i, qa in enumerate(qa_pairs, 1):
            formatted.append(f"{i}. Q: {qa['question']}\n   A: {qa['answer']}")
        return "\n\n".join(formatted)
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code"""
        if not language:
            return "en"
        normalized = language.lower().strip()
        if normalized in ['es', 'sp']:
            return 'sp'
        return normalized if normalized in ['en', 'sp'] else 'en'


# ============================================================================
# AGENT 3: Question Generator
# ============================================================================

class QuestionGenerator:
    """Agent 3: Generates the actual medical question"""
    
    def __init__(self, client, settings):
        self._client = client
        self._settings = settings
    
    async def generate_question(
        self,
        medical_context: MedicalContext,
        extracted_info: ExtractedInformation,
        current_count: int,
        max_count: int,
        language: str = "en",
        avoid_similar_to: Optional[str] = None,
        asked_questions: Optional[List[str]] = None,
        previous_answers: Optional[List[str]] = None,
        is_deep_diagnostic: bool = False
    ) -> str:
        """
        Generate next medical question based on context and what's been covered.
        This is a FOCUSED prompt that leverages medical reasoning.
        
        Args:
            avoid_similar_to: A question that was just rejected as duplicate - generate something DIFFERENT
            asked_questions: All questions asked so far (for full context)
            previous_answers: All answers received so far (for full context)
        """
        
        lang = self._normalize_language(language)
        
        # Format all Q&A pairs for context
        qa_history = ""
        if asked_questions and previous_answers:
            all_qa = []
            for i in range(len(asked_questions)):
                if i < len(previous_answers):
                    all_qa.append({
                        "question": asked_questions[i],
                        "answer": previous_answers[i]
                    })
            if all_qa:
                qa_history = self._format_qa_pairs(all_qa)
        
        # Determine if this is the closing question
        if current_count + 1 >= max_count:
            if lang == "sp":
                return "¿Hemos pasado por alto algo importante sobre su salud o hay otras preocupaciones que desee que el médico sepa?"
            return "Have we missed anything important about your health, or any other concerns you want the doctor to know?"
        
        if lang == "sp":
            system_prompt = """Eres un médico experimentado realizando una entrevista médica.

Tu trabajo es generar UNA pregunta médicamente relevante, clara y amigable para el paciente.

La pregunta debe ser apropiada para un contexto de entrevista médica."""

            deep_diagnostic_block_sp = ""
            if is_deep_diagnostic:
                deep_diagnostic_block_sp = (
                    "🔍 MODO DIAGNÓSTICO PROFUNDO: El paciente aceptó responder preguntas diagnósticas detalladas. "
                    "Esta es la pregunta diagnóstica profunda #" + str(current_count - 9) + " de 3.\n\n"
                    "CRÍTICO: Estas preguntas deben ser PROFUNDAMENTE DIAGNÓSTICAS - NO preguntas básicas como 'cuánto tiempo', "
                    "'cuándo comenzó', 'duración', 'medicamentos', etc. Esas ya fueron preguntadas.\n\n"
                    "Genera preguntas diagnósticas ALTAMENTE ESPECÍFICAS que exploren:\n"
                    "- Patrones específicos de síntomas, desencadenantes o asociaciones\n"
                    "- Caracterización detallada de síntomas (calidad, radiación, momento)\n"
                    "- Impacto en sistemas corporales o funciones específicas\n"
                    "- Historia detallada relacionada con la queja principal\n"
                    "- Pistas diagnósticas específicas o señales de alerta\n\n"
                    "Ejemplos de BUENAS preguntas diagnósticas profundas:\n"
                    "- '¿Puede describir el patrón exacto de su [síntoma] - viene en oleadas, es constante, o tiene desencadenantes específicos?'\n"
                    "- 'Cuando experimenta [síntoma], ¿se irradia a alguna otra parte de su cuerpo, y si es así, dónde exactamente?'\n"
                    "- '¿Ha notado alguna hora específica del día o circunstancias cuando [síntoma] empeora o mejora?'\n"
                    "- '¿Hay alguna actividad, posición o movimiento específico que desencadene o empeore su [síntoma]?'\n\n"
                    "Ejemplos de MALAS preguntas (NO hagas estas - son demasiado básicas):\n"
                    "- '¿Desde hace cuánto tiempo tiene esto?' (ya preguntado)\n"
                    "- '¿Qué medicamentos está tomando?' (ya preguntado)\n"
                    "- '¿Puede describir sus síntomas?' (demasiado genérico)\n\n"
                )

            duplicate_warning_sp = ""
            if avoid_similar_to:
                duplicate_warning_sp = (
                    f"⚠️ ADVERTENCIA CRÍTICA: La pregunta '{avoid_similar_to}' fue RECHAZADA como duplicado. "
                    "Debes generar una pregunta COMPLETAMENTE DIFERENTE sobre un TEMA DIFERENTE. NO reformules la misma pregunta.\n"
                )

            user_prompt = f"""
Contexto médico:
- Queja principal: {medical_context.chief_complaint}
- Edad del paciente: {medical_context.patient_age or "Desconocida"}
- Género del paciente: {medical_context.patient_gender or "No especificado"}
- Propiedades de la condición: {json.dumps(medical_context.condition_properties, indent=2)}
- Razonamiento médico: {medical_context.medical_reasoning}

Temas prioritarios a cubrir: {medical_context.priority_topics}
Temas a EVITAR: {medical_context.avoid_topics}

{"=" * 60}
TODAS LAS PREGUNTAS Y RESPUESTAS ANTERIORES (CONTEXTO COMPLETO):
{qa_history if qa_history else "No hay preguntas previas - esta es la primera pregunta."}
{"=" * 60}

IMPORTANTE: Revisa TODAS las preguntas y respuestas anteriores para asegurarte de NO repetir preguntas similares o duplicadas. Usa este contexto para entender qué información ya se ha recopilado y qué temas ya se han cubierto.

Información ya recopilada:
- Temas cubiertos: {extracted_info.topics_covered}
- Brechas de información: {extracted_info.information_gaps if extracted_info.information_gaps else "No hay brechas listadas - genera una pregunta exploratoria relevante según el contexto médico"}
- Hechos extraídos: {json.dumps(extracted_info.extracted_facts, indent=2)}
- ¿Ya se mencionó duración?: {extracted_info.already_mentioned_duration}
- ¿Ya se preguntó sobre medicamentos?: {extracted_info.already_mentioned_medications}
- Categorías redundantes (NO repetir): {extracted_info.redundant_categories}

Progreso: Pregunta {current_count + 1} de {max_count}

{deep_diagnostic_block_sp}

"REGLAS DE FLUJO Y CATEGORÍAS (OBLIGATORIAS):\n"
"- Sigue la secuencia global: 1) duración (ya debería estar cubierta salvo que sea la primera pregunta), 2) UNA pregunta combinada de medicamentos + cuidados/remedios caseros, 3) temas de alta prioridad según la enfermedad, 4) síntomas asociados (<=3 únicos), 5) monitoreo crónico (exactamente 2 si es crónica), 6) detección/complicaciones (exactamente 1 si es crónica), 7) categorías restantes (<=1 pregunta cada una), 8) pregunta de cierre solo al final (>=7 preguntas totales o sin temas valiosos pendientes).\n"
"- Si el médico seleccionó categorías, cúbrelas primero dentro de esta secuencia eligiendo la pregunta más relevante clínicamente.\n"
"- No preguntes historia familiar en cuadros agudos/infecciosos ni en dolor aislado. En condiciones crónicas, omite preguntas de historial médico / episodios similares ("¿Ha pasado antes?", etc.) a menos que el médico lo haya solicitado explícitamente.\n"
"- Escenarios agudos/infecciosos: enfócate en duración, síntomas asociados, medicamentos/cuidados caseros, viajes relevantes, episodios previos similares. NO hagas preguntas de monitoreo/detección crónica.\n"
"- Escenarios de dolor: prioriza duración, caracterización/severidad, desencadenantes/alivio, impacto funcional, medicamentos/cuidados caseros, episodios previos. Evita historia familiar salvo dolor crónico.\n"
"- Salud de la mujer: incluye síntomas hormonales/ginecológicos, tamizajes recientes (Pap, examen pélvico, mamografía, densitometría, laboratorios hormonales) e historia familiar pertinente. Viajes/alergias solo con justificación.\n"
"- Enfermedades crónicas (diabetes, hipertensión, asma, tiroides, cardiopatías, cáncer, EPOC, etc.): formula DOS preguntas de monitoreo y UNA de detección/complicaciones mencionando pruebas reales (fondo de ojo/examen de pies/consulta dental + laboratorios renales para diabetes; laboratorios renales, examen retinal y pruebas cardíacas para hipertensión; espirometría, pruebas de alergia, imágenes torácicas y peak-flow basal para asma; TSH/FT4, examen de cuello/ultrasonido, densitometría ósea y ritmo cardíaco para tiroides; panel lipídico, prueba de esfuerzo, Holter/ECG, ecocardiograma para cardiopatías; CT/MRI/PET, marcadores tumorales, biopsias y estudios de estadificación para oncología). Combina screening y complicaciones en una sola pregunta rica.\n"
"- Guía por edad: <12 y >60 -> sin preguntas menstruales. Ajusta para pubertad, fertilidad o menopausia según corresponda.\n"
"- No dividas apetito, peso y energía en preguntas separadas; una vez cubierto un aspecto, los demás quedan redundantes.\n"
"- Preguntas de viaje solo cuando la casilla esté marcada Y el cuadro clínico lo amerite.\n\n"

{duplicate_warning_sp}

INSTRUCCIÓN ESPECIAL: Si las brechas de información están vacías, genera una pregunta exploratoria médicamente relevante basada en la queja principal y las propiedades de la condición. NO hagas una pregunta genérica de "¿algo más?" a menos que sea la última pregunta.

PREVENCIÓN DE DUPLICADOS:

CRÍTICO: Revisa "Temas cubiertos" y "Categorías redundantes" - estos temas YA fueron preguntados y NO deben repetirse.
La detección programática de duplicados rechazará cualquier pregunta duplicada, así que asegúrate de que tu pregunta sea sobre un NUEVO tema de "Brechas de información".

VALIDACIÓN OBLIGATORIA antes de generar:
Paso 1: Identifica qué INFORMACIÓN CENTRAL busca tu pregunta
Paso 2: Busca en "Temas cubiertos" y "Categorías redundantes"
Paso 3: Si esa información YA fue preguntada (aunque con palabras diferentes) → OMITE este tema
Paso 4: Escoge un tema de "Brechas de información" que NO haya sido cubierto

REGLAS CRÍTICAS:
1. Genera UNA pregunta sobre el siguiente tema más importante que AÚN NO se ha cubierto
2. Prioriza brechas de información sobre temas prioritarios
3. NUNCA preguntes sobre temas en "Temas a EVITAR"
4. NUNCA repitas categorías en "Categorías redundantes" (incluye duración y medicamentos si ya fueron preguntados)
5. Haz la pregunta clara, específica y amigable para el paciente
6. La pregunta debe ser tipo entrevista, NO tipo cuestionario
7. Termina siempre con "?"
8. NO incluyas razonamiento, categorías o explicaciones - SOLO la pregunta

VALIDACIÓN FINAL OBLIGATORIA:
Antes de responder, VERIFICA:
✓ ¿Esta pregunta busca información que YA está en "Temas cubiertos"? → Si SÍ, OMITE este tema
✓ ¿Esta pregunta es semánticamente similar a alguna pregunta anterior? → Si SÍ, OMITE este tema
✓ ¿Esta pregunta está en "Categorías redundantes"? → Si SÍ, OMITE este tema
✓ Escoge SOLO de "Brechas de información"

FORMATO DE SALIDA:
Responde ÚNICAMENTE con el texto de la pregunta, sin comillas, sin numeración, sin explicaciones.
Ejemplo de respuesta correcta: ¿Qué medicamentos está tomando actualmente?

Genera LA SIGUIENTE pregunta más importante ahora:"""

        else:  # English
            system_prompt = """You are an experienced physician conducting a medical interview.

Your job is to generate ONE medically relevant, clear, and patient-friendly question.

The question should be appropriate for a medical interview context."""

            deep_diagnostic_block_en = ""
            if is_deep_diagnostic:
                deep_diagnostic_block_en = (
                    "🔍 DEEP DIAGNOSTIC MODE: The patient agreed to answer detailed diagnostic questions. "
                    "This is deep diagnostic question #" + str(current_count - 9) + " of 3.\n\n"
                    "CRITICAL: These questions must be DEEPLY DIAGNOSTIC - NOT basic questions like 'how long', "
                    "'when did it start', 'duration', 'medications', etc. Those have already been asked.\n\n"
                    "Generate HIGHLY SPECIFIC diagnostic questions that explore:\n"
                    "- Specific symptom patterns, triggers, or associations\n"
                    "- Detailed characterization of symptoms (quality, radiation, timing)\n"
                    "- Impact on specific body systems or functions\n"
                    "- Detailed history related to the chief complaint\n"
                    "- Specific diagnostic clues or red flags\n\n"
                    "Examples of GOOD deep diagnostic questions:\n"
                    "- 'Can you describe the exact pattern of your [symptom] - does it come in waves, is it constant, or does it have specific triggers?'\n"
                    "- 'When you experience [symptom], does it radiate to any other part of your body, and if so, where exactly?'\n"
                    "- 'Have you noticed any specific time of day or circumstances when [symptom] is worse or better?'\n"
                    "- 'Are there any specific activities, positions, or movements that trigger or worsen your [symptom]?'\n\n"
                    "Examples of BAD questions (DO NOT ask these - they're too basic):\n"
                    "- 'How long have you had this?' (already asked)\n"
                    "- 'What medications are you taking?' (already asked)\n"
                    "- 'Can you describe your symptoms?' (too generic)\n\n"
                )

            duplicate_warning_en = ""
            if avoid_similar_to:
                duplicate_warning_en = (
                    f"⚠️ CRITICAL WARNING: The question '{avoid_similar_to}' was REJECTED as duplicate. "
                    "You MUST generate a COMPLETELY DIFFERENT question about a DIFFERENT TOPIC. DO NOT rephrase the same question.\n"
                )

            user_prompt = f"""
Medical context:
- Chief complaint: {medical_context.chief_complaint}
- Patient age: {medical_context.patient_age or "Unknown"}
- Patient gender: {medical_context.patient_gender or "Not specified"}
- Condition properties: {json.dumps(medical_context.condition_properties, indent=2)}
- Medical reasoning: {medical_context.medical_reasoning}

Priority topics to cover: {medical_context.priority_topics}
Topics to AVOID: {medical_context.avoid_topics}

{"=" * 60}
ALL PREVIOUS QUESTIONS AND ANSWERS (FULL CONTEXT):
{qa_history if qa_history else "No previous questions - this is the first question."}
{"=" * 60}

IMPORTANT: Review ALL previous questions and answers above to ensure you do NOT repeat similar or duplicate questions. Use this context to understand what information has already been collected and what topics have already been covered.

Information already collected:
- Topics covered: {extracted_info.topics_covered}
- Information gaps: {extracted_info.information_gaps if extracted_info.information_gaps else "No gaps listed - generate an exploratory follow-up question based on medical context"}
- Extracted facts: {json.dumps(extracted_info.extracted_facts, indent=2)}
- Duration already mentioned?: {extracted_info.already_mentioned_duration}
- Medications already asked?: {extracted_info.already_mentioned_medications}
- Redundant categories (DO NOT repeat): {extracted_info.redundant_categories}

Progress: Question {current_count + 1} of {max_count}

{deep_diagnostic_block_en}

"FLOW & CATEGORY RULES (MANDATORY):\n"
"- Follow the global sequence: 1) duration (already asked unless this is Q1), 2) one combined medications + home/self-care question, 3) disease-specific high-priority items, 4) associated symptoms (<=3 unique), 5) chronic monitoring (exactly 2 when chronic), 6) chronic screening/complication (exactly 1 when chronic), 7) remaining categories (<=1 question each), 8) closing question only at the end (>=7 total questions or no high-value topics left).\n"
"- If the doctor selected categories, address them first within that sequence while choosing the most clinically relevant question.\n"
"- Do not ask family history for purely acute infections or isolated pain complaints. For chronic conditions, SKIP any past medical history / "have you had this before" questions unless the doctor explicitly selected that category.\n"
"- Acute/infectious scenarios: focus on duration, associated symptoms, combined meds/home care, relevant travel exposure, prior similar episodes. NO chronic monitoring/screening questions.\n"
"- Pain scenarios: emphasise duration, pain character/severity, triggers/relief, functional impact, combined meds/home care, previous similar episodes. Avoid family history unless managing chronic pain.\n"
"- Women's health: include hormonal/gynecologic symptoms, recent screenings (Pap, pelvic exam, mammogram, bone density, hormone labs) and pertinent family history. Travel/allergy questions only when justified.\n"
"- Chronic diseases (diabetes, hypertension, asthma, thyroid, heart disease, cancer, COPD, etc.): ensure TWO monitoring questions and ONE screening/complication question referencing real tests (eye/foot/dental + kidney labs for diabetes; kidney labs, retinal exam and cardiac testing for hypertension; pulmonary function test, allergy testing, chest imaging and peak-flow baseline for asthma; TSH/FT4, neck exam/ultrasound, bone density and heart rhythm checks for thyroid; lipid panel, stress test, Holter/ECG, echocardiogram for heart disease; CT/MRI/PET, tumour markers, biopsies and staging labs for oncology). Combine screening and complications into a single rich question.\n"
"- Apply age guidance: <12 and >60 -> no menstrual questions. Adjust for puberty, fertility or menopause as appropriate.\n"
"- Do not split appetite, weight and energy into separate questions; once covered, treat the others as redundant.\n"
"- Travel questions only when the travel checkbox is true AND symptoms justify it.\n\n"

{duplicate_warning_en}

SPECIAL INSTRUCTION: If information_gaps is empty, generate a medically relevant exploratory question based on the chief complaint and condition properties. Do NOT ask a generic "anything else?" question unless this is the last question.

DUPLICATE PREVENTION:

CRITICAL: Check "Topics covered" and "Redundant categories" - these topics have ALREADY been asked and must NOT be repeated.
Programmatic duplicate detection will reject any duplicate questions, so ensure your question is about a NEW topic from "Information gaps".

MANDATORY VALIDATION before generating:
Step 1: Identify what CORE INFORMATION your question seeks
Step 2: Check "Topics covered" and "Redundant categories"
Step 3: If that information was ALREADY asked (even with different words) → SKIP this topic
Step 4: Choose a topic from "Information gaps" that has NOT been covered

CRITICAL RULES:
1. Generate ONE question about the next most important topic that has NOT been covered
2. Prioritize information gaps from priority topics
3. NEVER ask about topics in "Topics to AVOID"
4. NEVER repeat categories in "Redundant categories" (includes duration and medications if already asked)
5. Make the question clear, specific, and patient-friendly
6. Question should be interview-style, NOT questionnaire-style
7. Always end with "?"
8. DO NOT include reasoning, categories, or explanations - ONLY the question

MANDATORY FINAL VALIDATION:
Before responding, VERIFY:
✓ Does this question seek information ALREADY in "Topics covered"? → If YES, SKIP this topic
✓ Is this question semantically similar to any previous question? → If YES, SKIP this topic
✓ Is this question in "Redundant categories"? → If YES, SKIP this topic
✓ Choose ONLY from "Information gaps"

OUTPUT FORMAT:
Respond with ONLY the question text, without quotes, without numbering, without explanations.
Example of correct response: What medications are you currently taking?

Generate THE NEXT most important question now:"""

        try:
            response = await self._client.chat(
                model=self._settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,  # Increased to avoid truncation
                temperature=0.2,  # Lower for stricter adherence to rules and less creativity
            )
            
            # Extract text from response object
            response_text = response.choices[0].message.content.strip()
            
            # Clean up the question with robust formatting
            question = response_text.strip()
            
            # Remove wrapping quotes (single or double) that LLM might add
            if (question.startswith('"') and question.endswith('"')) or \
               (question.startswith("'") and question.endswith("'")):
                question = question[1:-1].strip()
            
            # Remove newlines and extra spaces
            question = question.replace("\n", " ").strip()
            question = ' '.join(question.split())  # Normalize whitespace
            
            # Remove any trailing punctuation before adding final "?"
            question = question.rstrip('?.!,;:')
            
            # Add single question mark
            if not question.endswith("?"):
                question = question + "?"
            
            # Final validation: remove double question marks if any
            while "??" in question:
                question = question.replace("??", "?")
            
            logger.info(f"Generated question (cleaned): {question}")
            
            return question
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            # Fallback question
            if lang == "sp":
                return "¿Puede contarme más sobre cómo se siente?"
            return "Can you tell me more about how you're feeling?"
    
    def _format_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> str:
        """Format Q&A pairs for context"""
        formatted = []
        for i, qa in enumerate(qa_pairs, 1):
            formatted.append(f"{i}. Q: {qa['question']}\n   A: {qa['answer']}")
        return "\n\n".join(formatted)
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code"""
        if not language:
            return "en"
        normalized = language.lower().strip()
        if normalized in ['es', 'sp']:
            return 'sp'
        return normalized if normalized in ['en', 'sp'] else 'en'


# ============================================================================
# AGENT 4: Safety Validator
# ============================================================================

@dataclass
class ValidationResult:
    """Result of safety validation"""
    is_valid: bool
    issues: List[str]
    corrected_question: Optional[str]


class SafetyValidator:
    """Agent 4: Validates questions against hard safety constraints"""
    
    def __init__(self, client, settings):
        self._client = client
        self._settings = settings
    
    async def validate_question(
        self,
        question: str,
        medical_context: MedicalContext,
        asked_questions: List[str],
        language: str = "en"
    ) -> ValidationResult:
        """
        Validate question against hard safety constraints.
        This is the final safety check before returning to patient.
        """
        
        issues = []
        
        # VALIDATION 1: Format check
        if not question or len(question) < 10:
            issues.append("Question too short")
        
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"
        
        # VALIDATION 2: Gender appropriateness
        if medical_context.patient_gender:
            gender_lower = medical_context.patient_gender.lower()
            question_lower = question.lower()
            
            # Check for menstrual questions to male patients
            if gender_lower in ["male", "m", "masculino", "hombre"]:
                menstrual_keywords = [
                    "menstrual", "period", "menstruation", "menstruo", "periodo",
                    "menstruación", "regla", "ciclo menstrual", "menstrual cycle"
                ]
                if any(keyword in question_lower for keyword in menstrual_keywords):
                    issues.append("Menstrual question asked to male patient (CRITICAL VIOLATION)")
        
        # VALIDATION 3: Age appropriateness
        if medical_context.patient_age is not None:
            age = medical_context.patient_age
            question_lower = question.lower()
            
            # Check for menstrual questions outside 12-60 age range
            if age < 12 or age > 60:
                menstrual_keywords = [
                    "menstrual", "period", "menstruation", "menstruo", "periodo",
                    "menstruación", "regla", "ciclo menstrual", "menstrual cycle"
                ]
                if any(keyword in question_lower for keyword in menstrual_keywords):
                    issues.append(f"Menstrual question asked to age {age} (CRITICAL VIOLATION)")
        
        # VALIDATION 4: Enhanced duplication check with semantic awareness
        question_normalized = question.lower().strip().rstrip("?.")
        
        # Define semantic equivalence patterns
        semantic_patterns = {
            "symptom_description": ["describe", "symptoms", "feeling", "experiencing", "concern"],
            "duration": ["how long", "when", "started", "began", "duration"],
            "medications": ["medication", "medicine", "drug", "taking", "treatment"],
            "other_symptoms": ["other symptoms", "additional", "associated", "along with"],
            "progression": ["changed", "progressed", "worse", "better", "over time", "past week"],
            "daily_impact": ["daily", "activities", "life", "affected", "impact"],
        }
        
        def get_question_intent(q: str) -> set:
            """Extract semantic intent from question"""
            q_lower = q.lower()
            intents = set()
            for intent, keywords in semantic_patterns.items():
                if sum(1 for kw in keywords if kw in q_lower) >= 2:
                    intents.add(intent)
            return intents
        
        current_intent = get_question_intent(question_normalized)
        
        for prev_q in asked_questions:
            prev_normalized = prev_q.lower().strip().rstrip("?.")
            
            # Check 1: Exact or very similar text match
            if question_normalized == prev_normalized:
                issues.append(f"Question is exact duplicate of: '{prev_q}' (CRITICAL VIOLATION)")
                continue
            
            # Check 2: High word overlap similarity
            similarity_score = self._calculate_similarity(question_normalized, prev_normalized)
            if similarity_score > 0.85:
                issues.append(f"Question is very similar to: '{prev_q}' (similarity: {similarity_score:.2f})")
                continue
            
            # Check 3: Semantic intent overlap
            prev_intent = get_question_intent(prev_normalized)
            if current_intent and prev_intent and current_intent == prev_intent:
                issues.append(f"Question has same semantic intent as: '{prev_q}' (intent: {current_intent})")
                continue
            
            # Check 4: High word overlap with lower threshold but same key concepts
            if similarity_score > 0.65:
                # Check if they share the same core medical concept
                core_medical_terms = ["symptom", "medication", "duration", "pain", "condition", "treatment"]
                q_terms = [term for term in core_medical_terms if term in question_normalized]
                prev_terms = [term for term in core_medical_terms if term in prev_normalized]
                if q_terms and prev_terms and set(q_terms) == set(prev_terms):
                    issues.append(f"Question semantically similar to: '{prev_q}' (similarity: {similarity_score:.2f})")
        
        # VALIDATION 5: Output format
        if "\n" in question:
            question = question.replace("\n", " ")
            issues.append("Question contained newlines (auto-corrected)")
        
        # If critical violations, generate fallback or flag for regeneration
        critical_violations = [i for i in issues if "CRITICAL VIOLATION" in i]
        if critical_violations:
            logger.error(f"Critical safety violation detected: {critical_violations}")
            # For duplicate questions, we need to force Agent 3 to generate a different question
            # Return None to signal that question generation should be retried
            if any("duplicate" in v.lower() for v in critical_violations):
                logger.error(f"Duplicate question detected and blocked: '{question}'")
                # Generate a diversified safe exploratory question based on asked questions
                lang = self._normalize_language(language)
                question = self._generate_diverse_fallback(asked_questions, lang)
            else:
                # For other critical violations (gender/age)
                lang = self._normalize_language(language)
                if lang == "sp":
                    question = "¿Hay algo más sobre su condición que le gustaría compartir?"
                else:
                    question = "Is there anything else about your condition you'd like to share?"
        
        is_valid = len(critical_violations) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            corrected_question=question
        )
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Simple similarity calculation based on word overlap"""
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_diverse_fallback(self, asked_questions: List[str], language: str) -> str:
        """Generate a diverse fallback question that won't duplicate previous ones"""
        fallback_options_en = [
            "Can you share more details that might help us better understand your condition?",
            "Is there anything about your symptoms that we haven't covered yet?",
            "What other information do you think would be helpful for us to know?",
            "Are there any recent changes in your health that you'd like to mention?",
            "Is there anything else about your condition that concerns you?"
        ]
        
        fallback_options_sp = [
            "¿Puede compartir más detalles que puedan ayudarnos a entender mejor su condición?",
            "¿Hay algo sobre sus síntomas que aún no hayamos cubierto?",
            "¿Qué otra información cree que sería útil que supiéramos?",
            "¿Ha habido cambios recientes en su salud que le gustaría mencionar?",
            "¿Hay algo más sobre su condición que le preocupe?"
        ]
        
        options = fallback_options_sp if language == "sp" else fallback_options_en
        
        # Find a fallback that hasn't been asked yet
        asked_lower = [q.lower() for q in asked_questions]
        for fallback in options:
            if fallback.lower() not in asked_lower:
                return fallback
        
        # If all have been asked (unlikely), return the first one
        return options[0]
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code"""
        if not language:
            return "en"
        normalized = language.lower().strip()
        if normalized in ['es', 'sp']:
            return 'sp'
        return normalized if normalized in ['en', 'sp'] else 'en'


# ============================================================================
# MAIN SERVICE: Multi-Agent Orchestrator
# ============================================================================

class OpenAIQuestionService(QuestionService):
    """
    Main Question Service using Multi-Agent Architecture
    Optimized for medical accuracy with intelligent reasoning
    
    This replaces the previous 837-line prompt system with dynamic agents.
    """
    
    def __init__(self) -> None:
        import os
        from pathlib import Path
        
        try:
            from dotenv import load_dotenv
        except Exception:
            load_dotenv = None
        
        # Load settings
        self._settings = get_settings()
        self._debug_prompts = getattr(self._settings, "debug_prompts", False)
        
        # Require Azure OpenAI - no fallback to standard OpenAI
        azure_openai_configured = (
            self._settings.azure_openai.endpoint and 
            self._settings.azure_openai.api_key
        )
        
        if not azure_openai_configured:
            raise ValueError(
                "Azure OpenAI is required. Please configure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY. "
                "Fallback to standard OpenAI is disabled for data security."
            )
        
        # Verify deployment names are configured
        if not self._settings.azure_openai.deployment_name:
            raise ValueError(
                "Azure OpenAI deployment name is required. Please set AZURE_OPENAI_DEPLOYMENT_NAME."
            )
        
        if not self._settings.azure_openai.whisper_deployment_name:
            raise ValueError(
                "Azure OpenAI Whisper deployment name is required. Please set AZURE_OPENAI_WHISPER_DEPLOYMENT_NAME."
            )
        
        # Initialize centralized Azure AI client (no fallback)
        self._client = get_ai_client()
        
        # Initialize agents
        self._context_analyzer = MedicalContextAnalyzer(self._client, self._settings)
        self._answer_extractor = AnswerExtractor(self._client, self._settings)
        self._question_generator = QuestionGenerator(self._client, self._settings)
        self._safety_validator = SafetyValidator(self._client, self._settings)
        
        logger.info("Multi-Agent Question Service initialized")
    
    async def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 64,
        temperature: float = 0.3,
        patient_id: str = None,
        prompt_name: str = None,
    ) -> str:
        """Helper method for direct chat completion (used by other methods)"""
        try:
            if self._debug_prompts:
                logger.debug("[QuestionService] Sending messages to OpenAI:\n%s", messages)

            resp = await self._client.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                model=self._settings.openai.model,
            )
            output = resp.choices[0].message.content.strip()

            if self._debug_prompts:
                logger.debug("[QuestionService] Received response: %s", output)

            return output
        except Exception:
            logger.error("[QuestionService] OpenAI call failed", exc_info=True)
            return ""
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code to handle both 'sp' and 'es' for backward compatibility."""
        if not language:
            return "en"
        normalized = language.lower().strip()
        if normalized in ['es', 'sp']:
            return 'sp'
        return normalized if normalized in ['en', 'sp'] else 'en'

    def _get_topic_count(
        self,
        extracted_info: ExtractedInformation,
        topic_keys: List[str],
    ) -> int:
        """Count how many times any of the given topic keys have been asked."""
        if not extracted_info:
            return 0
        
        total = 0
        if extracted_info.topic_counts:
            for key in topic_keys:
                total += extracted_info.topic_counts.get(key, 0)
        else:
            counter = Counter(extracted_info.topics_covered or [])
            for key in topic_keys:
                total += counter.get(key, 0)
        return total

    def _enforce_topic_requirement(
        self,
        extracted_info: ExtractedInformation,
        topic_en: str,
        topic_sp: str,
        required_count: int,
        language: str,
        insert_after: Optional[List[str]] = None,
        log_context: Optional[str] = None,
    ) -> int:
        """
        Ensure a topic appears the required number of times by adjusting information_gaps/redundant_categories.
        """
        lang = self._normalize_language(language)
        topic_name = topic_sp if lang == "sp" else topic_en
        topic_keys = [topic_en, topic_sp]
        
        count = self._get_topic_count(extracted_info, topic_keys)
        
        info_gaps = list(extracted_info.information_gaps or [])
        redundant = list(extracted_info.redundant_categories or [])
        
        if count < required_count:
            if topic_name not in info_gaps:
                insert_pos = 0
                if insert_after:
                    insert_pos = len(info_gaps)
                    for i, name in enumerate(info_gaps):
                        if name in insert_after:
                            insert_pos = i + 1
                insert_pos = max(0, min(insert_pos, len(info_gaps)))
                info_gaps.insert(insert_pos, topic_name)
                if log_context:
                    logger.warning(
                        f"{log_context} requirement not met (count={count} < {required_count}). "
                        f"Added '{topic_name}' to information_gaps at position {insert_pos}"
                    )
            if topic_name in redundant:
                redundant.remove(topic_name)
        else:
            if topic_name in info_gaps:
                info_gaps.remove(topic_name)
            if topic_name not in redundant:
                redundant.append(topic_name)
        
        extracted_info.information_gaps = info_gaps
        extracted_info.redundant_categories = redundant
        return count
    
    async def generate_first_question(self, disease: str, language: str = "en") -> str:
        """Generate the first intake question"""
        lang = self._normalize_language(language)
        if lang == "sp":
            return "¿Por qué ha venido hoy? ¿Cuál es la principal preocupación con la que necesita ayuda?"
        return "Why have you come in today? What is the main concern you want help with?"
    
    async def generate_next_question(
        self,
        disease: str,
        previous_answers: List[str],
        asked_questions: List[str],
        current_count: int,
        max_count: int = 10,  # Default max
        asked_categories: Optional[List[str]] = None,  # Ignored (deprecated)
        recently_travelled: bool = False,
        prior_summary: Optional[Any] = None,
        prior_qas: Optional[List[str]] = None,
        patient_gender: Optional[str] = None,
        patient_age: Optional[int] = None,
        language: str = "en",
    ) -> str:
        """
        Generate next question using multi-agent pipeline.
        
        This replaces the massive 837-line prompt with intelligent reasoning.
        """
        
        try:
            max_count = max(1, min(14, max_count))
            # AGENT 1: Analyze medical context
            logger.info(f"Agent 1: Analyzing medical context for '{disease}'")
            medical_context = await self._context_analyzer.analyze_condition(
                chief_complaint=disease,
                patient_age=patient_age,
                patient_gender=patient_gender,
                recently_travelled=recently_travelled,
                language=language
            )
            
            # AGENT 2: Extract what's been covered
            logger.info(f"Agent 2: Extracting covered information from {len(asked_questions)} previous questions")
            extracted_info = await self._answer_extractor.extract_covered_information(
                asked_questions=asked_questions,
                previous_answers=previous_answers,
                medical_context=medical_context,
                language=language
            )
            
            # Log Agent 2 output for debugging
            logger.info(f"Agent 2 Results - Topics covered: {extracted_info.topics_covered}")
            logger.info(f"Agent 2 Results - Information gaps: {extracted_info.information_gaps}")
            logger.info(f"Agent 2 Results - Redundant categories: {extracted_info.redundant_categories}")
            
            condition_props = medical_context.condition_properties or {}
            is_chronic = condition_props.get("is_chronic", False)
            is_pain_related = condition_props.get("is_pain_related", False)

            if is_chronic:
                monitoring_count = self._enforce_topic_requirement(
                    extracted_info=extracted_info,
                    topic_en="chronic_monitoring",
                    topic_sp="monitoreo_crónico",
                    required_count=2,
                    language=language,
                    insert_after=["current_medications", "medicamentos_actuales"],
                    log_context="Chronic monitoring",
                )
                screening_count = self._enforce_topic_requirement(
                    extracted_info=extracted_info,
                    topic_en="screening",
                    topic_sp="detección",
                    required_count=1,
                    language=language,
                    insert_after=[
                        "chronic_monitoring", "monitoreo_crónico",
                        "screening", "detección",
                        "current_medications", "medicamentos_actuales"
                    ],
                    log_context="Screening",
                )
                logger.info(
                    f"Chronic disease topic counts -> monitoring: {monitoring_count}, screening: {screening_count}"
                )

            if not is_pain_related:
                pain_topics = ["pain_assessment", "pain_characterization", "evaluación_dolor", "caracterización_dolor"]
                info_gaps = list(extracted_info.information_gaps or [])
                redundant = list(extracted_info.redundant_categories or [])
                for pain_topic in pain_topics:
                    if pain_topic in info_gaps:
                        info_gaps.remove(pain_topic)
                        logger.info(f"Removed {pain_topic} from information_gaps - condition not pain-related")
                    if pain_topic not in redundant:
                        redundant.append(pain_topic)
                extracted_info.information_gaps = info_gaps
                extracted_info.redundant_categories = redundant

            # CRITICAL: Check if patient answered "yes" to diagnostic consent question
            # If yes, we need to generate deep diagnostic questions (3 questions)
            diagnostic_consent_patterns = [
                "would you like to answer some detailed diagnostic questions",
                "detailed diagnostic questions related to your symptoms",
                "le gustaría responder algunas preguntas diagnósticas detalladas",
                "preguntas diagnósticas detalladas relacionadas con sus síntomas"
            ]
            
            # Check if last question was diagnostic consent
            is_after_consent = False
            if asked_questions and previous_answers and len(asked_questions) == len(previous_answers):
                last_question = asked_questions[-1].lower()
                last_answer = previous_answers[-1].lower().strip()
                is_consent_question = any(pattern in last_question for pattern in diagnostic_consent_patterns)
                
                if is_consent_question:
                    # Check if answer is positive
                    positive_responses = ["yes", "y", "yeah", "yep", "sure", "ok", "okay", "of course", "absolutely",
                                        "sí", "si", "claro", "por supuesto", "por supuesto que sí", "de acuerdo"]
                    is_positive = any(pos in last_answer for pos in positive_responses) and len(last_answer) < 50
                    
                    if is_positive:
                        is_after_consent = True
                        logger.info(f"Patient answered YES to diagnostic consent - will generate deep diagnostic questions")
                        # Update max_count to 14 to allow for 3 deep diagnostic questions + final question
                        if max_count < 14:
                            max_count = 14
                            logger.info(f"Updated max_count to 14 for deep diagnostic questions")
                    else:
                        negative_responses = [
                            "no", "n", "nope", "nah", "not really", "don't", "dont",
                            "no gracias", "no quiero", "no, gracias", "no, no quiero"
                        ]
                        is_negative = any(neg in last_answer for neg in negative_responses) and len(last_answer) < 50
                        if is_negative:
                            logger.info("Patient declined diagnostic consent - returning final question immediately")
                            lang = self._normalize_language(language)
                            if lang == "sp":
                                return "¿Hay algo más que le gustaría compartir sobre su condición?"
                            return "Is there anything else you'd like to share about your condition?"
            
            # Gather condition properties (for logging/prompts only, no hardcoded enforcement)
            
            # CRITICAL: Check if this is Q10 (current_count == 9) - ask diagnostic consent question
            if current_count == 9:
                logger.info("Q10 reached - generating diagnostic consent question")
                lang = self._normalize_language(language)
                if lang == "sp":
                    return "¿Le gustaría responder algunas preguntas diagnósticas detalladas relacionadas con sus síntomas?"
                return "Would you like to answer some detailed diagnostic questions related to your symptoms?"
            
            # Validation: Check if we've reached max questions before checking information gaps
            # This prevents premature fallback to generic questions
            if current_count >= max_count - 1:
                logger.info("Reached maximum question count - generating closing question")
                lang = self._normalize_language(language)
                if lang == "sp":
                    return "¿Hay algo más que le gustaría compartir sobre su condición?"
                return "Is there anything else you'd like to share about your condition?"
            
            # If no information gaps but we haven't reached max, let Agent 3 handle it
            # Agent 3 should generate a relevant follow-up question
            if not extracted_info.information_gaps or len(extracted_info.information_gaps) == 0:
                logger.warning("No information gaps in priority list - Agent 3 will generate exploratory question")
                # Don't return here - let Agent 3 decide what to ask
            
            # AGENT 3: Generate question with retry mechanism for duplicates
            logger.info(f"Agent 3: Generating question {current_count + 1}/{max_count}")
            
            max_retries = 2  # Try up to 3 times total
            question = None
            validation = None
            rejected_question = None  # Track the question that was rejected
            
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    logger.warning(f"Retry attempt {attempt} due to duplicate detection")
                
                # Count deep diagnostic questions if we're after consent
                deeper_count = 0
                if is_after_consent and asked_questions:
                    # Count questions asked after the consent question
                    consent_index = -1
                    for i, q in enumerate(asked_questions):
                        if any(pattern in q.lower() for pattern in diagnostic_consent_patterns):
                            consent_index = i
                            break
                    if consent_index >= 0:
                        # Count questions after consent (excluding the consent question itself)
                        deeper_count = len(asked_questions) - consent_index - 1
                
                # If we've asked 3 deep diagnostic questions, ask final question
                if is_after_consent and deeper_count >= 3:
                    logger.info(f"3 deep diagnostic questions completed ({deeper_count} asked) - generating final question")
                    lang = self._normalize_language(language)
                    if lang == "sp":
                        return "¿Hay algo más que le gustaría compartir sobre su condición?"
                    return "Is there anything else you'd like to share about your condition?"
                
                question = await self._question_generator.generate_question(
                    medical_context=medical_context,
                    extracted_info=extracted_info,
                    current_count=current_count,
                    max_count=max_count,
                    language=language,
                    avoid_similar_to=rejected_question,  # Tell Agent 3 to avoid this specific question
                    asked_questions=asked_questions,  # Full Q&A history for context
                    previous_answers=previous_answers,  # Full Q&A history for context
                    is_deep_diagnostic=is_after_consent and deeper_count < 3  # Generate deep diagnostic if after consent and less than 3 asked
                )
                
                # AGENT 4: Validate safety
                logger.info(f"Agent 4: Validating question safety (attempt {attempt + 1})")
                validation = await self._safety_validator.validate_question(
                    question=question,
                    medical_context=medical_context,
                    asked_questions=asked_questions,
                    language=language
                )
                
                # Check if it's a duplicate
                has_duplicate = any("duplicate" in issue.lower() for issue in validation.issues)
                
                if not has_duplicate or attempt == max_retries:
                    # Either valid or we've exhausted retries
                    break
                
                # Add the failed question to asked_questions temporarily to avoid regenerating it
                asked_questions = asked_questions + [question]
                rejected_question = question  # Remember this question for the next attempt
                logger.warning(f"Question was duplicate, retrying with updated context")
            
            if not validation.is_valid:
                logger.warning(f"Question validation issues: {validation.issues}")
            
            final_question = validation.corrected_question or question
            
            logger.info(
                f"✅ Multi-agent pipeline completed successfully: '{final_question}' "
                f"(count={current_count+1}/{max_count})"
            )
            
            return final_question
            
        except Exception as e:
            logger.error(f"Multi-agent question generation failed: {e}", exc_info=True)
            # Fallback
            lang = self._normalize_language(language)
            if lang == "sp":
                return "¿Puede contarme más sobre cómo se siente?"
            return "Can you tell me more about how you're feeling?"
    
    async def should_stop_asking(
        self,
        disease: str,
        previous_answers: List[str],
        current_count: int,
        max_count: int = 10,
    ) -> bool:
        """Determine if we should stop asking questions"""
        if current_count >= max_count:
            return True
        return False
    
    async def assess_completion_percent(
        self,
        disease: str,
        previous_answers: List[str],
        asked_questions: List[str],
        current_count: int,
        max_count: int = 10,
        prior_summary: Optional[Any] = None,
        prior_qas: Optional[List[str]] = None,
    ) -> int:
        """Calculate completion percentage"""
        try:
            progress_ratio = 0.0
            if max_count > 0:
                progress_ratio = min(max(current_count / max_count, 0.0), 1.0)
            return int(progress_ratio * 100)
        except Exception:
            return 0
    
    async def is_medication_question(self, question: str) -> bool:
        """Check if question is about medications (for image upload)"""
        question_lower = (question or "").lower()
        
        # Check for combined self-care/home remedies/medications questions
        if ("home remedies" in question_lower and "medications" in question_lower) or \
           ("self-care" in question_lower and "medications" in question_lower) or \
           ("remedies" in question_lower and "supplements" in question_lower) or \
           ("autocuidado" in question_lower and "medicamentos" in question_lower) or \
           ("remedios caseros" in question_lower and "medicamentos" in question_lower):
            return True
        
        medication_terms = [
            "medication", "medications", "medicines", "medicine", "drug", "drugs",
            "prescription", "prescribed", "tablet", "tablets", "capsule", "capsules",
            "syrup", "dose", "dosage", "frequency", "supplement", "supplements",
            "insulin", "otc", "over-the-counter", "medicamento", "medicamentos",
            "medicina", "medicinas", "medicamento recetado", "suplemento", "suplementos"
        ]
        
        return any(term in question_lower for term in medication_terms)
    
    # ============================================================================
    # PRESERVED METHODS: Pre-visit summary and red flag detection
    # ============================================================================
    
    async def generate_pre_visit_summary(
        self,
        patient_data: Dict[str, Any],
        intake_answers: Dict[str, Any],
        language: str = "en",
        medication_images_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate pre-visit clinical summary from intake data with red flag detection."""
        
        # Normalize language code
        lang = self._normalize_language(language)
        
        if lang == "sp":
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
                "- Medicación Actual: Una línea narrativa con medicamentos/suplementos realmente declarados por el paciente (nombre/dosis/frecuencia si se proporciona). Incluye declaraciones de alergia solo si el paciente las reportó explícitamente. Si el paciente subió imágenes de medicamentos (incluso si no mencionó explícitamente los nombres), menciona esto: \"El paciente proporcionó imágenes de medicamentos: [nombre(s) de archivo(s)]\". Incluye esta sección si los medicamentos fueron discutidos O si se subieron imágenes de medicamentos.\n\n"
                "Ejemplo de Formato\n"
                "(Estructura y tono solamente—el contenido será diferente; cada sección en una sola línea.)\n"
                "Motivo de Consulta: El paciente reporta dolor de cabeza severo por 3 días.\n"
                "HPI: El paciente describe una semana de dolores de cabeza persistentes que comienzan en la mañana y empeoran durante el día, llegando hasta 8/10 en los últimos 3 días. El dolor es sobre ambas sienes y se siente diferente de migrañas previas; la fatiga es prominente y se niega náusea. Los episodios se agravan por estrés y más tarde en el día, con alivio mínimo de analgésicos de venta libre y algo de alivio usando compresas frías.\n"
                "Historia: Médica: hipertensión; Quirúrgica: colecistectomía hace cinco años; Estilo de vida: no fumador, alcohol ocasional, trabajo de alto estrés.\n"
                "Medicación Actual: En medicamentos: lisinopril 10 mg diario e ibuprofeno según necesidad; alergias incluidas solo si el paciente las declaró explícitamente.\n\n"
                f"{f'Imágenes de Medicamentos: {medication_images_info}' if medication_images_info else ''}\n\n"
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
                "- Current Medication: One narrative line with meds/supplements actually stated by the patient (name/dose/frequency if provided). Include allergy statements only if the patient explicitly reported them. If the patient uploaded medication images (even if they didn't explicitly name the medications), mention this: \"Patient provided medication images: [filename(s)]\". Include this section if medications were discussed OR if medication images were uploaded.\n\n"
                "Example Format\n"
                "(Structure and tone only—content will differ; each section on a single line.)\n"
                "Chief Complaint: Patient reports severe headache for 3 days.\n"
                "HPI: The patient describes a week of persistent headaches that begin in the morning and worsen through the day, reaching up to 8/10 over the last 3 days. Pain is over both temples and feels different from prior migraines; fatigue is prominent and nausea is denied. Episodes are aggravated by stress and later in the day, with minimal relief from over-the-counter analgesics and some relief using cold compresses. No radiation is reported, evenings are typically worse, and there have been no recent changes in medications or lifestyle.\n"
                "History: Medical: hypertension; Surgical: cholecystectomy five years ago; Lifestyle: non-smoker, occasional alcohol, high-stress job.\n"
                "Current Medication: On meds: lisinopril 10 mg daily and ibuprofen as needed; allergies included only if the patient explicitly stated them.\n\n"
                f"{f'Medication Images: {medication_images_info}' if medication_images_info else ''}\n\n"
                f"Intake Responses:\n{self._format_intake_answers(intake_answers)}"
            )

        try:
            # Detect abusive language red flags
            try:
                red_flags = await self._detect_red_flags(intake_answers, lang)
            except Exception as e:
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
                max_tokens=min(2000, self._settings.openai.max_tokens),  # Use max_tokens from settings
                temperature=0.3,
            )
            cleaned = self._clean_summary_markdown(response)
            
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
        lang = self._normalize_language(language)
        
        red_flags = []
        
        if not isinstance(intake_answers, dict) or "questions_asked" not in intake_answers:
            logger.warning("Invalid intake_answers format for red flag detection")
            return red_flags
            
        questions_asked = intake_answers.get("questions_asked", [])
        if not questions_asked:
            logger.warning("No questions found in intake_answers")
            return red_flags
        
        logger.info(f"Starting hybrid abusive language detection for {len(questions_asked)} questions")
        
        # Step 1: Fast hardcoded detection for obvious cases
        obvious_flags = self._detect_obvious_abusive_language(questions_asked, lang)
        red_flags.extend(obvious_flags)
        logger.info(f"Obvious abusive language flags detected: {len(obvious_flags)}")
        
        # Step 2: LLM analysis for subtle/contextual abusive language
        complex_flags = await self._detect_subtle_abusive_language_with_llm(questions_asked, lang)
        red_flags.extend(complex_flags)
        logger.info(f"Subtle abusive language flags detected: {len(complex_flags)}")
        
        logger.info(f"Total abusive language red flags detected: {len(red_flags)}")
        return red_flags
    
    def _detect_obvious_abusive_language(self, questions_asked: List[Dict[str, Any]], language: str = "en") -> List[Dict[str, str]]:
        """Fast hardcoded detection for obvious abusive language."""
        lang = self._normalize_language(language)
        red_flags = []
        
        for qa in questions_asked:
            answer = qa.get("answer", "").strip()
            question = qa.get("question", "").strip()
            
            if not answer or answer.lower() in ["", "n/a", "not provided", "unknown", "don't know", "no se", "no proporcionado"]:
                continue
            
            # Check for obvious abusive language
            if self._contains_abusive_language(answer, lang):
                red_flags.append({
                    "type": "abusive_language",
                    "question": question,
                    "answer": answer,
                    "message": self._get_abusive_language_message(lang),
                    "detection_method": "hardcoded"
                })
        
        return red_flags
    
    async def _detect_subtle_abusive_language_with_llm(self, questions_asked: List[Dict[str, Any]], language: str = "en") -> List[Dict[str, str]]:
        """Use LLM to detect subtle, contextual, or creative abusive language."""
        try:
            lang = self._normalize_language(language)
            
            # Filter out obvious cases already detected
            subtle_cases = []
            for qa in questions_asked:
                answer = qa.get("answer", "").strip()
                question = qa.get("question", "").strip()
                
                if (answer and answer.lower() not in ["", "n/a", "not provided", "unknown", "don't know", "no se", "no proporcionado"] 
                    and not self._contains_abusive_language(answer, lang)):
                    subtle_cases.append(qa)
            
            if not subtle_cases:
                return []
            
            # Use LLM to analyze subtle cases
            return await self._analyze_abusive_language_with_llm(subtle_cases, lang)
            
        except Exception as e:
            logger.warning(f"LLM abusive language analysis failed: {e}")
            return []
    
    async def _analyze_abusive_language_with_llm(self, questions_asked: List[Dict[str, Any]], language: str = "en") -> List[Dict[str, str]]:
        """Use LLM to analyze question-answer pairs for subtle abusive language."""
        lang = self._normalize_language(language)
        
        if lang == "sp":
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
                        "message": self._get_llm_abusive_language_message(case.get("reason", ""), lang),
                        "detection_method": "llm"
                    })
                
                return formatted_flags
            
        except Exception as e:
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
        lang = self._normalize_language(language)
        if lang == "sp":
            return f"⚠️ BANDERA ROJA: Lenguaje abusivo detectado. Razón: {reason}"
        else:
            return f"⚠️ RED FLAG: Abusive language detected. Reason: {reason}"
    
    def _contains_abusive_language(self, text: str, language: str = "en") -> bool:
        """Check if text contains abusive or inappropriate language."""
        lang = self._normalize_language(language)
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
        
        abusive_words = spanish_abusive if lang == "sp" else english_abusive
        
        return any(word in text_lower for word in abusive_words)
    
    def _get_abusive_language_message(self, language: str = "en") -> str:
        """Get message for abusive language red flag."""
        lang = self._normalize_language(language)
        if lang == "sp":
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
            text = (response or "").strip()

            # 1) Prefer fenced ```json ... ```
            fence_json = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
            if fence_json:
                candidate = fence_json.group(1).strip()
                return json.loads(candidate)

            # 2) Any fenced block without language
            fence_any = re.search(r"```\s*([\s\S]*?)\s*```", text)
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
