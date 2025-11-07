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
from dataclasses import dataclass
from enum import Enum

from clinicai.application.ports.services.question_service import QuestionService
from clinicai.core.config import get_settings
from clinicai.core.helicone_client import create_helicone_client


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

REGLAS MÉDICAS PARA PRIORITY_TOPICS:

ATENCIÓN: Todos los ejemplos siguientes (diabetes, asma, hipertensión, epilepsia, EPOC, etc.) son SOLO ILUSTRATIVOS. DEBES generalizar esta lógica a cualquier enfermedad crónica, aguda, rara o no listada según el mejor criterio médico. Al identificar cronicidad, riesgo de complicaciones, herencia genética, etc., aplícalo aunque la enfermedad no esté en los ejemplos.

1. MONITOREO CRÓNICO (monitoreo_crónico):
   INCLUIR SI: is_chronic = true Y has_complications = true
   Ejemplos ilustrativos: diabetes (glucosa/HbA1c), hipertensión (presión arterial), asma (flujo espiratorio/peak flow), epilepsia (crisis), EPOC, enfermedades autoinmunes…
   IMPORTANTE: Asma SIEMPRE requiere monitoreo (peak flow, frecuencia de inhalador, exacerbaciones)
   Si recibes otra condición crónica (no listada), pregunta acerca del monitoreo relevante igualmente.

2. HISTORIA FAMILIAR (historia_familiar):
   INCLUIR SI: is_hereditary = true
   Ejemplos: diabetes, hipertensión, asma, cáncer, enfermedades cardíacas, trastornos tiroideos (SOLO ILUSTRATIVOS)
   Si la enfermedad es genética/hereditaria aunque no esté en la lista, pregunta la historia familiar apropiada.

3. ALERGIAS (alergias):
   INCLUIR SI: is_allergy_related = true o hay síntomas de alergia
   Uso idéntico para cualquier condición relevante.

4. SIEMPRE INCLUIR (todas las condiciones):
   - duración (primero)
   - medicamentos_actuales (segundo)
   - caracterización_síntomas o síntomas_asociados

PARA CUALQUIER NUEVA, RARA O COMPLEJA condición usa juicio clínico: si reconoces cronicidad/herencia/complicaciones sigue esta lógica aunque no esté en la lista.

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
        Opciones disponibles:
        "duración", "medicamentos_actuales", "caracterización_síntomas", "síntomas_asociados",
        "historia_familiar", "monitoreo_crónico", "ciclo_menstrual", "historial_viajes",
        "alergias", "factores_estilo_vida"
        
        IMPORTANTE: Solo incluye temas que cumplen las REGLAS MÉDICAS arriba
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

MEDICAL RULES FOR PRIORITY_TOPICS:

NOTE: All examples below (e.g., diabetes, asthma, hypertension, epilepsy, COPD, etc.) are for ILLUSTRATION ONLY. You MUST generalize this logic to ANY chronic, acute, rare, or unlisted disease according to best medical practice. Always apply this reasoning to any disease that fits the pattern (chronic, hereditary, risk of complication…) even if not explicitly named.

1. CHRONIC MONITORING (chronic_monitoring):
   INCLUDE IF: is_chronic = true AND has_complications = true
   Examples (ONLY illustration): diabetes (glucose logs/HbA1c), hypertension (BP readings), asthma (peak flow/inhaler frequency), epilepsy (seizure record), COPD, autoimmune disorders...
   IMPORTANT: Asthma ALWAYS requires monitoring (peak flow, inhaler usage, exacerbations)
   If you encounter ANY other chronic disease, always ask about pertinent monitoring, even if it's not listed.

2. FAMILY HISTORY (family_history):
   INCLUDE IF: is_hereditary = true
   Examples: diabetes, hypertension, asthma, cancer, heart disease (ONLY ILLUSTRATIVE)
   If the disease is hereditary/genetic but not listed, apply best judgment and ask relevant family history.

3. ALLERGIES (allergies):
   INCLUDE IF: is_allergy_related = true OR allergic symptoms present
   Use for ANY relevant case, not just those listed.

4. ALWAYS INCLUDE (all conditions):
   - duration (always first)
   - current_medications (always second)
   - symptom_characterization OR associated_symptoms

FOR ANY NEW, RARE OR COMPLEX CONDITION, use clinical reasoning: if chronic, ask about monitoring/adherence/complication; if hereditary, ask about family history, etc.

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
        Available options:
        "duration", "current_medications", "symptom_characterization", "associated_symptoms",
        "family_history", "chronic_monitoring", "menstrual_cycle", "travel_history",
        "allergies", "lifestyle_factors"
        
        IMPORTANT: Only include topics that meet the MEDICAL RULES above
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
            response, metrics = await self._client.chat_completion(
                model=self._settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.2,  # Low temperature for consistency
                prompt_name="medical_context_analyzer",
                custom_properties={
                    "agent": "context_analyzer",
                    "chief_complaint": chief_complaint
                }
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
                redundant_categories=[]
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
        Opciones: "duración", "medicamentos_actuales", "caracterización_síntomas", 
                 "síntomas_asociados", "historia_familiar", "monitoreo_crónico",
                 "ciclo_menstrual", "historial_viajes", "alergias", "factores_estilo_vida"
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

REGLAS CRÍTICAS PARA MARCAR REDUNDANT_CATEGORIES:
1. Revisa TODAS las {len(all_qa)} preguntas arriba, no solo las últimas 3
2. Si "duración" fue preguntada EXPLÍCITAMENTE en CUALQUIER pregunta → already_mentioned_duration = true Y añade "duración" a redundant_categories
3. Si "medicamentos" fueron preguntados EXPLÍCITAMENTE en CUALQUIER pregunta → already_mentioned_medications = true Y añade "medicamentos_actuales" a redundant_categories
4. Si "síntomas asociados" o "otros síntomas" fueron preguntados EXPLÍCITAMENTE → añade "síntomas_asociados" a redundant_categories SOLO SI YA FUERON PREGUNTADOS
5. Si "caracterización de síntomas" fue preguntada EXPLÍCITAMENTE → añade "caracterización_síntomas" SOLO SI YA FUE PREGUNTADA
6. Si "actividades diarias" fue preguntado EXPLÍCITAMENTE → añade "factores_estilo_vida" SOLO SI YA FUE PREGUNTADO
7. Si "progresión" fue preguntada EXPLÍCITAMENTE → marca como redundante SOLO SI YA FUE PREGUNTADA

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
        Options: "duration", "current_medications", "symptom_characterization", 
                 "associated_symptoms", "family_history", "chronic_monitoring",
                 "menstrual_cycle", "travel_history", "allergies", "lifestyle_factors"
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

CRITICAL RULES FOR MARKING REDUNDANT_CATEGORIES:
1. Review ALL {len(all_qa)} questions above, not just the last 3
2. If "duration" was asked EXPLICITLY in ANY previous question → already_mentioned_duration = true AND add "duration" to redundant_categories
3. If "medications" were asked EXPLICITLY in ANY previous question → already_mentioned_medications = true AND add "current_medications" to redundant_categories
4. If "associated symptoms" were asked EXPLICITLY → add "associated_symptoms" to redundant_categories ONLY IF ALREADY ASKED
5. If "symptom characterization" was asked EXPLICITLY → add "symptom_characterization" ONLY IF ALREADY ASKED
6. If "daily activities" was asked EXPLICITLY → add "lifestyle_factors" ONLY IF ALREADY ASKED
7. If "progression" was asked EXPLICITLY → mark as redundant ONLY IF ALREADY ASKED

IMPORTANT:
- Be CONSERVATIVE: only mark a topic as redundant if the question ACTUALLY asked about it directly
- DO NOT mark topics that were only mentioned in passing or implicitly
- information_gaps should contain topics that have NOT been specifically asked yet
- Only add to redundant_categories when you are 100% certain that specific topic was covered

KEEP information_gaps with at least 2-3 topics until the patient has answered at least 5-6 questions.

Return ONLY the JSON, no additional text."""

        try:
            response, metrics = await self._client.chat_completion(
                model=self._settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.1,  # Very low for consistency
                prompt_name="answer_extractor",
                custom_properties={
                    "agent": "answer_extractor",
                    "qa_count": len(all_qa),
                    "total_questions": len(asked_questions)
                }
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
            
            return ExtractedInformation(
                topics_covered=topics_covered,
                information_gaps=information_gaps,
                extracted_facts=extraction.get("extracted_facts", {}),
                already_mentioned_duration=extraction.get("already_mentioned_duration", False),
                already_mentioned_medications=extraction.get("already_mentioned_medications", False),
                redundant_categories=redundant_categories
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
                redundant_categories=[]
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
        previous_answers: Optional[List[str]] = None
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

{"⚠️ ADVERTENCIA CRÍTICA: La pregunta '" + avoid_similar_to + "' fue RECHAZADA como duplicado. Debes generar una pregunta COMPLETAMENTE DIFERENTE sobre un TEMA DIFERENTE. NO reformules la misma pregunta." if avoid_similar_to else ""}

INSTRUCCIÓN ESPECIAL: Si las brechas de información están vacías, genera una pregunta exploratoria médicamente relevante basada en la queja principal y las propiedades de la condición. NO hagas una pregunta genérica de "¿algo más?" a menos que sea la última pregunta.

DETECCIÓN UNIVERSAL DE DUPLICADOS SEMÁNTICOS:

PRINCIPIO CLAVE: Dos preguntas son DUPLICADAS si buscan el MISMO TIPO de información del paciente, sin importar cómo estén formuladas.

EQUIVALENCIAS SEMÁNTICAS (estas preguntas son LA MISMA pregunta):
1. DESCRIPCIÓN DE SÍNTOMAS:
   - "¿Cuál es su principal preocupación?" (Q1 queja principal)
   = "Describa sus síntomas"
   = "¿Qué síntomas está experimentando?"
   = "¿Puede decirme qué está sintiendo?"
   ↑ Si el paciente YA describió su queja principal, NO vuelvas a preguntar por síntomas

2. SÍNTOMAS ADICIONALES:
   - "¿Ha experimentado otros síntomas?"
   = "Síntomas asociados"
   = "Síntomas específicos adicionales"
   = "Otros síntomas relacionados"
   ↑ Si YA se preguntó sobre "otros síntomas", NO repitas en ninguna forma

3. DURACIÓN:
   - "¿Desde hace cuánto tiempo?"
   = "¿Cuándo comenzó?"
   = "Duración de los síntomas"
   ↑ Si YA se preguntó duración, NO repitas

4. MEDICAMENTOS:
   - "¿Qué medicamentos está tomando?"
   = "Tratamientos que ha probado"
   = "Medicamentos o suplementos"
   ↑ Si YA se preguntó medicamentos, NO repitas

VALIDACIÓN OBLIGATORIA antes de generar:
Paso 1: Identifica qué INFORMACIÓN CENTRAL busca tu pregunta
Paso 2: Busca en "Temas cubiertos" y "Categorías redundantes"
Paso 3: Si esa información YA fue preguntada (aunque con palabras diferentes) → OMITE este tema
Paso 4: Escoge un tema de "Brechas de información" que NO haya sido cubierto

REGLAS CRÍTICAS:
1. Genera UNA pregunta sobre el siguiente tema más importante que AÚN NO se ha cubierto
2. Prioriza brechas de información sobre temas prioritarios
3. NUNCA preguntes sobre temas en "Temas a EVITAR"
4. NUNCA repitas categorías en "Categorías redundantes"
5. Si ya se mencionó duración, NO preguntes sobre duración de nuevo
6. Si ya se preguntó sobre medicamentos, NO repitas esa pregunta
7. Haz la pregunta clara, específica y amigable para el paciente
8. La pregunta debe ser tipo entrevista, NO tipo cuestionario
9. Termina siempre con "?"
10. NO incluyas razonamiento, categorías o explicaciones - SOLO la pregunta

EJEMPLOS DE BUENAS PREGUNTAS (estilo entrevista):
- "¿Qué tratamientos o medicamentos ha probado para este problema?"
- "¿Hay algo que haga que el dolor empeore o mejore?"
- "¿Ha notado otros síntomas junto con esto?"
- "¿Alguien en su familia ha tenido problemas similares?"

EJEMPLOS DE MALAS PREGUNTAS (evitar):
- "Descríbame su historial médico completo" (demasiado amplio)
- "¿Tiene diabetes?" (sí/no cerrado, no entrevista)
- "Califique su dolor del 1 al 10" (tipo cuestionario)

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

{"⚠️ CRITICAL WARNING: The question '" + avoid_similar_to + "' was REJECTED as duplicate. You MUST generate a COMPLETELY DIFFERENT question about a DIFFERENT TOPIC. DO NOT rephrase the same question." if avoid_similar_to else ""}

SPECIAL INSTRUCTION: If information_gaps is empty, generate a medically relevant exploratory question based on the chief complaint and condition properties. Do NOT ask a generic "anything else?" question unless this is the last question.

UNIVERSAL SEMANTIC DUPLICATE DETECTION:

KEY PRINCIPLE: Two questions are DUPLICATES if they seek the SAME TYPE of patient information, regardless of how they're worded.

SEMANTIC EQUIVALENCES (these questions are THE SAME question):
1. SYMPTOM DESCRIPTION:
   - "What is your main concern?" (Q1 chief complaint)
   = "Describe your symptoms"
   = "What symptoms are you experiencing?"
   = "Can you tell me what you're feeling?"
   ↑ If patient ALREADY described their chief complaint, DO NOT ask about symptoms again

2. ADDITIONAL SYMPTOMS:
   - "Have you experienced any other symptoms?"
   = "Associated symptoms"
   = "Specific additional symptoms"
   = "Other related symptoms"
   ↑ If "other symptoms" was ALREADY asked, DO NOT repeat in any form

3. DURATION:
   - "How long have you had this?"
   = "When did it start?"
   = "Duration of symptoms"
   ↑ If duration was ALREADY asked, DO NOT repeat

4. MEDICATIONS:
   - "What medications are you taking?"
   = "Treatments you've tried"
   = "Medications or supplements"
   ↑ If medications were ALREADY asked, DO NOT repeat

MANDATORY VALIDATION before generating:
Step 1: Identify what CORE INFORMATION your question seeks
Step 2: Check "Topics covered" and "Redundant categories"
Step 3: If that information was ALREADY asked (even with different words) → SKIP this topic
Step 4: Choose a topic from "Information gaps" that has NOT been covered

CRITICAL RULES:
1. Generate ONE question about the next most important topic that has NOT been covered
2. Prioritize information gaps from priority topics
3. NEVER ask about topics in "Topics to AVOID"
4. NEVER repeat categories in "Redundant categories"
5. If duration already mentioned, DO NOT ask about duration again
6. If medications already asked, DO NOT repeat that question
7. Make the question clear, specific, and patient-friendly
8. Question should be interview-style, NOT questionnaire-style
9. Always end with "?"
10. DO NOT include reasoning, categories, or explanations - ONLY the question

EXAMPLES OF GOOD QUESTIONS (interview style):
- "What treatments or medications have you tried for this problem?"
- "Is there anything that makes the pain worse or better?"
- "Have you noticed any other symptoms along with this?"
- "Has anyone in your family had similar issues?"

EXAMPLES OF BAD QUESTIONS (avoid):
- "Describe your complete medical history" (too broad)
- "Do you have diabetes?" (yes/no closed, not interview)
- "Rate your pain 1-10" (questionnaire style)

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
            response, metrics = await self._client.chat_completion(
                model=self._settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,  # Increased to avoid truncation
                temperature=0.2,  # Lower for stricter adherence to rules and less creativity
                prompt_name="question_generator",
                custom_properties={
                    "agent": "question_generator",
                    "question_number": current_count + 1
                }
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
        
        # Initialize Azure OpenAI client (no fallback)
        self._client = create_helicone_client()
        
        # Initialize agents
        self._context_analyzer = MedicalContextAnalyzer(self._client, self._settings)
        self._answer_extractor = AnswerExtractor(self._client, self._settings)
        self._question_generator = QuestionGenerator(self._client, self._settings)
        self._safety_validator = SafetyValidator(self._client, self._settings)
        
        logger.info("Multi-Agent Question Service initialized")
    
    async def _chat_completion(
        self, messages: List[Dict[str, str]], max_tokens: int = 64, temperature: float = 0.3,
        patient_id: str = None, prompt_name: str = None
    ) -> str:
        """Helper method for direct chat completion (used by other methods)"""
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
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code to handle both 'sp' and 'es' for backward compatibility."""
        if not language:
            return "en"
        normalized = language.lower().strip()
        if normalized in ['es', 'sp']:
            return 'sp'
        return normalized if normalized in ['en', 'sp'] else 'en'
    
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
                
                question = await self._question_generator.generate_question(
                    medical_context=medical_context,
                    extracted_info=extracted_info,
                    current_count=current_count,
                    max_count=max_count,
                    language=language,
                    avoid_similar_to=rejected_question,  # Tell Agent 3 to avoid this specific question
                    asked_questions=asked_questions,  # Full Q&A history for context
                    previous_answers=previous_answers  # Full Q&A history for context
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
                max_tokens=min(2000, self._settings.openai.max_tokens),
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
