import asyncio
import logging
import json
import re
from typing import Any, Dict, List, Optional
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from clinicai.application.ports.services.question_service import QuestionService
from clinicai.core.config import get_settings
from clinicai.core.ai_factory import get_ai_client
from clinicai.core.constants import (
    ALLOWED_TOPICS,
    TRAVEL_KEYWORDS,
    MENSTRUAL_KEYWORDS,
    HIGH_RISK_COMPLAINT_KEYWORDS,
    SIMILARITY_STOPWORDS,
)

logger = logging.getLogger("clinicai")


# ============================================================================
# AGENT 1: Medical Context Analyzer
# ============================================================================

class ConditionPropertiesModel(BaseModel):
    """Pydantic schema for condition_properties returned by Agent-01."""

    is_chronic: Optional[bool] = None
    is_hereditary: Optional[bool] = None
    has_complications: Optional[bool] = None
    is_acute_emergency: Optional[bool] = None
    is_pain_related: Optional[bool] = None
    is_womens_health: Optional[bool] = None
    is_allergy_related: Optional[bool] = None
    requires_lifestyle_assessment: Optional[bool] = None
    is_travel_related: Optional[bool] = None
    severity_level: Optional[str] = Field(default=None, pattern="^(mild|moderate|severe)$")
    acuity_level: Optional[str] = Field(default=None, pattern="^(acute|subacute|chronic)$")
    is_new_problem: Optional[bool] = None
    is_followup: Optional[bool] = None


class MedicalContextLLMResponse(BaseModel):
    """Pydantic schema for Agent-01 LLM JSON response."""

    condition_properties: ConditionPropertiesModel
    triage_level: Optional[str] = "routine"
    red_flags: Optional[Dict[str, Any]] = None
    normalized_complaint: Optional[str] = None
    priority_topics: List[str] = Field(default_factory=list)
    avoid_topics: List[str] = Field(default_factory=list)
    medical_reasoning: str = ""
    prompt_version: Optional[str] = None
    core_symptom_phrase: Optional[str] = None
    topic_plan: List[str] = Field(default_factory=list)


@dataclass
class MedicalContext:
    """Structured medical context from condition analysis."""

    # Core fields
    chief_complaint: str
    condition_properties: Dict[str, Any]  # validated via ConditionPropertiesModel
    priority_topics: List[str]            # Ordered list of what to ask about
    avoid_topics: List[str]               # Topics that should NOT be asked
    medical_reasoning: str                # Why these priorities
    patient_age: Optional[int]
    patient_gender: Optional[str]

    # Phase 1 richer context (all optional, LLM-populated when available)
    severity_level: Optional[str] = None          # "mild" | "moderate" | "severe"
    acuity_level: Optional[str] = None            # "acute" | "subacute" | "chronic"
    triage_level: Optional[str] = None            # "routine" | "urgent" | "emergency"
    normalized_complaint: Optional[str] = None    # Short normalized label, e.g. "type 2 diabetes follow-up"
    red_flags: Optional[Dict[str, Any]] = None    # Flexible structure for red-flag metadata
    core_symptom_phrase: Optional[str] = None     # e.g. "blurry vision"
    topic_plan: List[str] = field(default_factory=list)  # ordered plan from Agent-01


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
            system_prompt = """You are AGENT-01 "MEDICAL CONTEXT ANALYZER" / Clinical Strategist for a clinical intake assistant.

Your ONLY job is to analyze the chief complaint and basic patient info and return a JSON "plan" object.
You DO NOT write questions for the patient. You decide:
- what kind of problem this is (chronic/acute, red flags, etc.),
- which topics are important,
- in what order they should be asked,
- and which topics must NOT be asked in this case.

TOPIC LABELS (DEBES USAR SOLO ESTAS)
- Allowed topic labels (sensible a mayúsculas/minúsculas, SIEMPRE en inglés):
  "duration", "associated_symptoms", "symptom_characterization",
  "current_medications", "pain_assessment", "pain_characterization",
  "location_radiation", "travel_history", "chronic_monitoring", "screening",
  "allergies", "past_medical_history", "hpi", "menstrual_cycle",
  "functional_impact", "daily_impact", "lifestyle_factors",
  "triggers", "aggravating_factors", "temporal", "progression",
  "frequency", "past_evaluation", "other", "exploratory", "family_history"
- DEBES usar SOLO estas etiquetas de temas en priority_topics, avoid_topics y topic_plan.
- NUNCA inventes nuevas etiquetas. NUNCA traduzcas las etiquetas. Deben permanecer en inglés.

ESQUEMA JSON (DEVUELVE SOLO ESTO)
Debes devolver UN SOLO objeto JSON con esta estructura:
{
  "condition_properties": {
    "is_chronic": <true/false/null>,
    "is_hereditary": <true/false/null>,
    "has_complications": <true/false/null>,
    "is_acute_emergency": <true/false/null>,
    "is_pain_related": <true/false/null>,
    "is_womens_health": <true/false/null>,
    "is_allergy_related": <true/false/null>,
    "requires_lifestyle_assessment": <true/false/null>,
    "is_travel_related": <true/false/null>,
    "severity_level": "<mild|moderate|severe|null>",
    "acuity_level": "<acute|subacute|chronic|null>",
    "is_new_problem": <true/false/null>,
    "is_followup": <true/false/null>
  },
  "triage_level": "<routine|urgent|emergency>",
  "red_flags": {
    "possible_emergency": <true/false>,
    "needs_urgent_attention": <true/false>,
    "example_flags": ["<frases cortas o vacío>"]
  },
  "normalized_complaint": "<etiqueta breve y normalizada para el problema principal>",
  "core_symptom_phrase": "<frase corta para el síntoma principal o null>",
  "priority_topics": ["<etiquetas de la lista permitida>", "..."],
  "avoid_topics": ["<etiquetas de la lista permitida>", "..."],
  "topic_plan": ["<secuencia ideal de temas, subconjunto de priority_topics>", "..."],
  "medical_reasoning": "<explicación breve de por qué elegiste estas propiedades y temas>",
  "prompt_version": "med_ctx_v3"
}

REGLAS:
- Sé conservador: si no estás seguro sobre un booleano, usa null en lugar de adivinar.
- triage_level:
  - "emergency": dolor de pecho con signos de alarma, dificultad respiratoria severa, síntomas tipo ACV, anafilaxia, etc.
  - "urgent": alto riesgo pero estable (por ejemplo, malestar torácico que empeora, fiebre alta con signos de alarma).
  - "routine": la mayoría de las consultas estables.
- Para niños (<12), evita temas menstruales y enfócate en infección, hidratación, preocupación de los padres, etc.
- Para pacientes masculinos o edad <12 o >60, NUNCA incluyas "menstrual_cycle" en priority_topics.
- Para problemas claramente crónicos no infecciosos (por ejemplo, seguimiento de diabetes sin infección), normalmente evita "travel_history".

Devuelve SOLO JSON. Sin comentarios ni explicaciones fuera del JSON.

IMPORTANTE: Las etiquetas de temas deben permanecer EXACTAMENTE en inglés como se listan arriba."""
            user_prompt = f"""
Analiza la queja principal del paciente: "{chief_complaint}"
Edad del paciente: {patient_age or "Desconocida"}
Género del paciente: {patient_gender or "No especificado"}
Viajó recientemente: {"Sí" if recently_travelled else "No"}

Realiza un análisis médico completo según las instrucciones del sistema y devuelve SOLO el objeto JSON descrito."""
        else:  # English
            system_prompt = """You are AGENT-01 "MEDICAL CONTEXT ANALYZER" / Clinical Strategist for a clinical intake assistant.

Your ONLY job is to analyze the chief complaint and basic patient info and return a JSON "plan" object.
You DO NOT write questions for the patient. You decide:
- what kind of problem this is (chronic/acute, red flags, etc.),
- which topics are important,
- in what order they should be asked,
- and which topics must NOT be asked in this case.

TOPIC LABELS (MUST USE THESE ONLY)
- Allowed topic labels (case-sensitive, always English):
  "duration", "associated_symptoms", "symptom_characterization",
  "current_medications", "pain_assessment", "pain_characterization",
  "location_radiation", "travel_history", "chronic_monitoring", "screening",
  "allergies", "past_medical_history", "hpi", "menstrual_cycle",
  "functional_impact", "daily_impact", "lifestyle_factors",
  "triggers", "aggravating_factors", "temporal", "progression",
  "frequency", "past_evaluation", "other", "exploratory", "family_history"
- You MUST only use these topic labels in priority_topics, avoid_topics, and topic_plan.
- NEVER invent new topic labels. NEVER translate labels. They must stay in English.

JSON SCHEMA (RETURN THIS ONLY)
You MUST return ONE JSON object with this structure:
{
  "condition_properties": {
    "is_chronic": <true/false/null>,
    "is_hereditary": <true/false/null>,
    "has_complications": <true/false/null>,
    "is_acute_emergency": <true/false/null>,
    "is_pain_related": <true/false/null>,
    "is_womens_health": <true/false/null>,
    "is_allergy_related": <true/false/null>,
    "requires_lifestyle_assessment": <true/false/null>,
    "is_travel_related": <true/false/null>,
    "severity_level": "<mild|moderate|severe|null>",
    "acuity_level": "<acute|subacute|chronic|null>",
    "is_new_problem": <true/false/null>,
    "is_followup": <true/false/null>
  },
  "triage_level": "<routine|urgent|emergency>",
  "red_flags": {
    "possible_emergency": <true/false>,
    "needs_urgent_attention": <true/false>,
    "example_flags": [ "<short phrases or empty>" ]
  },
  "normalized_complaint": "<short normalized label for the main problem>",
  "core_symptom_phrase": "<short phrase for main symptom or null>",
  "priority_topics": ["<topic label from the allowed list>", "..."],
  "avoid_topics": ["<topic label from the allowed list>", "..."],
  "topic_plan": ["<ideal questioning sequence, subset of priority_topics>", "..."],
  "medical_reasoning": "<short explanation of why you chose these properties and topics>",
  "prompt_version": "med_ctx_v3"
}

RULES:
- Be conservative: if unsure about a boolean, use null instead of guessing.
- triage_level:
  - "emergency": chest pain with red-flag features, severe breathing difficulty, stroke-like symptoms, anaphylaxis, etc.
  - "urgent": high-risk but stable (e.g. worsening chest discomfort, high fever with red flags).
  - "routine": most stable consultations.
- For children (<12), avoid menstrual topics and focus on infection, hydration, parental concern, etc.
- For males or for age <12 or >60, NEVER include "menstrual_cycle" in priority_topics.
- For clearly chronic non-infectious problems (e.g. diabetes follow-up without infection), usually avoid "travel_history".

IMPORTANT: Topic labels MUST remain EXACTLY in English as listed above.
Return ONLY JSON, no commentary or explanation outside the JSON."""

            user_prompt = f"""
Analyze this patient's chief complaint: "{chief_complaint}"
Patient age: {patient_age or "Unknown"}
Patient gender: {patient_gender or "Not specified"}
Recently traveled: {"Yes" if recently_travelled else "No"}

Perform a comprehensive medical analysis following the system instructions and return ONLY the JSON object described."""

        try:
            response = await self._client.chat(
                model=self._settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.2,
            )

            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.error("Agent 1: No JSON found in response: %s", response_text[:200])
                return self._create_fallback_context(
                    chief_complaint, patient_age, patient_gender, recently_travelled
                )

            # Parse and validate JSON via Pydantic schema
            try:
                raw = json.loads(json_match.group())
                parsed = MedicalContextLLMResponse.model_validate(raw)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error("Agent 1: JSON/validation error: %s", e)
                return self._create_fallback_context(
                    chief_complaint, patient_age, patient_gender, recently_travelled
                )

            # ---- Phase 1: normalize new rich fields (triage, severity, acuity, red_flags, normalized_complaint) ----
            condition_properties = parsed.condition_properties.model_dump()

            # Severity and acuity inside condition_properties
            severity = condition_properties.get("severity_level")
            if severity not in {"mild", "moderate", "severe"}:
                severity = None

            acuity = condition_properties.get("acuity_level")
            if acuity not in {"acute", "subacute", "chronic"}:
                acuity = None

            condition_properties["severity_level"] = severity
            condition_properties["acuity_level"] = acuity

            # Triage level: clamp to allowed values
            triage_level = parsed.triage_level or "routine"
            allowed_triage = {"routine", "urgent", "emergency"}
            if triage_level not in allowed_triage:
                triage_level = "routine"

            # Topics and reasoning (clamped to allow-list)
            priority_topics_raw = parsed.priority_topics or []
            avoid_topics_raw = parsed.avoid_topics or []
            topic_plan_raw = parsed.topic_plan or []
            medical_reasoning = parsed.medical_reasoning or ""

            allowed_topics_set = set(ALLOWED_TOPICS)
            unknown_topics = (
                set(priority_topics_raw)
                .union(avoid_topics_raw)
                .union(topic_plan_raw)
                - allowed_topics_set
            )
            if unknown_topics:
                logger.warning(
                    "Agent 1: Dropping unknown topic labels from LLM: %s",
                    sorted(unknown_topics),
                )

            priority_topics = [t for t in priority_topics_raw if t in allowed_topics_set]
            avoid_topics = [t for t in avoid_topics_raw if t in allowed_topics_set]
            # topic_plan must be subset of priority_topics and allowed topics
            priority_set = set(priority_topics)
            topic_plan = [t for t in topic_plan_raw if t in priority_set and t in allowed_topics_set]

            # Other optional rich fields
            red_flags = parsed.red_flags or None
            normalized_complaint = parsed.normalized_complaint or None
            core_symptom_phrase = parsed.core_symptom_phrase or None

            # Build analysis dict for safety constraints and logging
            analysis: Dict[str, Any] = {
                "chief_complaint": chief_complaint,
                "condition_properties": condition_properties,
                "priority_topics": priority_topics,
                "avoid_topics": avoid_topics,
                "medical_reasoning": medical_reasoning,
            }

            # Apply hard safety constraints (travel, family history, PMH/allergies, menstrual)
            analysis = self._enforce_safety_constraints(
                analysis,
                patient_age,
                patient_gender,
                recently_travelled
            )

            # Enhanced logging: log full condition properties and topics after safety constraints
            logger.info(
                f"Medical context analysis completed - "
                f"chief_complaint: '{chief_complaint}', "
                f"is_travel_related: {analysis['condition_properties'].get('is_travel_related')}, "
                f"recently_travelled: {recently_travelled}, "
                f"condition_properties: {json.dumps(analysis['condition_properties'], indent=2)}, "
                f"priority_topics: {analysis['priority_topics']}, "
                f"avoid_topics: {analysis.get('avoid_topics', [])}"
            )

            # Additional compact log for triage summary
            logger.info(
                "Medical context triage summary - triage_level=%s, severity=%s, acuity=%s, normalized_complaint='%s'",
                triage_level,
                severity,
                acuity,
                normalized_complaint or "",
            )

            return MedicalContext(
                chief_complaint=chief_complaint,
                condition_properties=analysis["condition_properties"],
                priority_topics=analysis["priority_topics"],
                avoid_topics=analysis.get("avoid_topics", []),
                medical_reasoning=medical_reasoning,
                patient_age=patient_age,
                patient_gender=patient_gender,
                severity_level=severity,
                acuity_level=acuity,
                triage_level=triage_level,
                normalized_complaint=normalized_complaint,
                red_flags=red_flags,
                core_symptom_phrase=core_symptom_phrase,
                topic_plan=topic_plan,
            )

        except Exception as e:
            logger.error(f"Medical context analysis failed: {e}")
            return self._create_fallback_context(chief_complaint, patient_age, patient_gender, recently_travelled)

    def _enforce_safety_constraints(
        self,
        analysis: Dict[str, Any],
        patient_age: Optional[int],
        patient_gender: Optional[str],
        recently_travelled: bool = False
    ) -> Dict[str, Any]:
        """
        Enforce hard safety constraints that cannot be violated.

        This method enforces conditional topic rules:
        - Travel history: only if recently_travelled AND is_travel_related
        - Family history: only if condition is chronic or hereditary
        - PMH/allergies: only if chronic, allergy-related, or high-risk
        """
        priority_topics = analysis.get("priority_topics", [])
        avoid_topics = analysis.get("avoid_topics", [])
        condition_props = analysis.get("condition_properties", {})
        chief_complaint = analysis.get("chief_complaint", "")

        # NEW: Detect travel-illness keywords vs chronic non-infectious conditions
        chief_complaint_lower = (chief_complaint or "").lower()
        travel_illness_keywords = [
            "fever", "diarrhea", "loose motion", "vomiting", "stomach pain",
            "infection", "malaria", "dengue", "chikungunya", "jaundice", "rash"
        ]
        chronic_non_infectious_keywords = [
            "diabetes", "hypertension", "high blood pressure", "asthma", "copd",
            "chronic", "heart failure", "ckd", "chronic kidney", "thyroid"
        ]

        looks_like_travel_illness = any(kw in chief_complaint_lower for kw in travel_illness_keywords)
        is_chronic_non_infectious = any(kw in chief_complaint_lower for kw in chronic_non_infectious_keywords)

        # Override is_travel_related for chronic non-infectious conditions
        is_travel_related = condition_props.get("is_travel_related", False)
        if is_chronic_non_infectious and not looks_like_travel_illness:
            condition_props["is_travel_related"] = False
            logger.info(
                f"Overrode is_travel_related=False for chronic non-infectious condition: "
                f"chief_complaint='{chief_complaint_lower}', "
                f"chronic_keywords_matched={[kw for kw in chronic_non_infectious_keywords if kw in chief_complaint_lower]}"
            )

        # HARD CONSTRAINT: Travel history only if patient travelled AND symptoms are travel-related
        is_travel_related = condition_props.get("is_travel_related", False)
        if not recently_travelled or not is_travel_related:
            for key in ["travel_history", "historial_viajes"]:
                if key in priority_topics:
                    priority_topics.remove(key)
                if key not in avoid_topics:
                    avoid_topics.append(key)
            if not recently_travelled:
                logger.info(
                    f"Safety constraint: Removed travel_history from priority_topics "
                    f"(reason: patient has not traveled recently, recently_travelled={recently_travelled}, "
                    f"chief_complaint='{chief_complaint_lower}')"
                )
            elif not is_travel_related:
                logger.info(
                    f"Safety constraint: Removed travel_history from priority_topics "
                    f"(reason: symptoms are not travel-related, is_travel_related={is_travel_related}, "
                    f"chronic_non_infectious={is_chronic_non_infectious}, chief_complaint='{chief_complaint_lower}')"
                )

        # HARD CONSTRAINT: Family history only for hereditary/chronic conditions
        is_hereditary = condition_props.get("is_hereditary", False)
        is_chronic = condition_props.get("is_chronic", False)
        removed_family = False
        if not (is_hereditary or is_chronic):
            for key in ["family_history"]:
                if key in priority_topics:
                    priority_topics.remove(key)
                    removed_family = True
                if key not in avoid_topics:
                    avoid_topics.append(key)
            logger.info(
                f"Safety constraint: Removed family_history from priority_topics "
                f"(reason: condition is not hereditary/chronic, is_hereditary={is_hereditary}, is_chronic={is_chronic})"
            )

        # HARD CONSTRAINT: PMH/allergies only when medically relevant
        is_allergy_related = condition_props.get("is_allergy_related", False)
        is_chronic = condition_props.get("is_chronic", False)
        # High-risk conditions that require PMH/allergy screening
        chief_complaint_lower = (analysis.get("chief_complaint", "") or "").lower()
        high_risk_keywords = ["chest pain", "shortness of breath", "anaphylaxis", "allergic reaction", "medication", "prescription"]
        is_high_risk = any(kw in chief_complaint_lower for kw in high_risk_keywords)
        
        should_ask_pmh_allergy = is_chronic or is_allergy_related or is_high_risk
        removed_pmh_allergy = False
        
        if not should_ask_pmh_allergy:
            for key in ["past_medical_history", "allergies"]:
                if key in priority_topics:
                    priority_topics.remove(key)
                    removed_pmh_allergy = True
                if key not in avoid_topics:
                    avoid_topics.append(key)
            logger.info(
                f"Safety constraint: Removed PMH/allergies from priority_topics "
                f"(reason: not medically relevant, is_chronic={is_chronic}, "
                f"is_allergy_related={is_allergy_related}, is_high_risk={is_high_risk})"
            )

        # HARD CONSTRAINT: Menstrual questions filtered by age & gender
        if patient_gender:
            gender_lower = patient_gender.lower()
            question_menstrual_topics = ["menstrual_cycle", "ciclo_menstrual"]
            if gender_lower in ["male", "m", "masculino", "hombre"]:
                for key in question_menstrual_topics:
                    if key in priority_topics:
                        priority_topics.remove(key)
                    if key not in avoid_topics:
                        avoid_topics.append(key)
                logger.warning("Removed menstrual questions for male patient (safety constraint)")

        if patient_age is not None and (patient_age < 12 or patient_age > 60):
            for key in ["menstrual_cycle", "ciclo_menstrual"]:
                if key in priority_topics:
                    priority_topics.remove(key)
                if key not in avoid_topics:
                    avoid_topics.append(key)
            logger.warning(f"Removed menstrual questions for age {patient_age} (safety constraint)")

        analysis["priority_topics"] = priority_topics
        analysis["avoid_topics"] = list(set(avoid_topics))
        return analysis

    def _create_fallback_context(
        self,
        chief_complaint: str,
        patient_age: Optional[int],
        patient_gender: Optional[str],
        recently_travelled: bool = False
    ) -> MedicalContext:
        """Safe fallback context if analysis fails completely."""

        priority_topics = [
            "duration",
            "current_medications",
            "symptom_characterization",
            "associated_symptoms",
        ]

        # If the main complaint is pain, ensure we ask about location & radiation early
        chief_lower = (chief_complaint or "").lower()
        if "pain" in chief_lower and "location_radiation" not in priority_topics:
            insert_idx = 2 if len(priority_topics) >= 2 else len(priority_topics)
            priority_topics.insert(insert_idx, "location_radiation")

        avoid_topics: List[str] = []

        is_male = patient_gender and patient_gender.lower() in ["male", "m", "masculino", "hombre"]
        age_inappropriate = patient_age and (patient_age < 12 or patient_age > 60)

        if is_male or age_inappropriate:
            avoid_topics.extend(["menstrual_cycle", "ciclo_menstrual"])

        if not recently_travelled:
            avoid_topics.extend(["travel_history", "historial_viajes"])

        topic_plan = list(priority_topics)

        return MedicalContext(
            chief_complaint=chief_complaint,
            condition_properties={
                "is_chronic": False,
                "is_hereditary": False,
                "has_complications": False,
                "is_acute_emergency": True,
                "is_pain_related": "pain" in chief_lower,
                "is_womens_health": False,
                "is_allergy_related": False,
                "requires_lifestyle_assessment": False,
                "is_travel_related": False,  # Default to False in fallback
                "severity_level": None,
                "acuity_level": None,
            },
            priority_topics=priority_topics,
            avoid_topics=avoid_topics,
            medical_reasoning="Using conservative fallback approach",
            patient_age=patient_age,
            patient_gender=patient_gender,
            severity_level=None,
            acuity_level=None,
            triage_level="routine",
            normalized_complaint=None,
            red_flags=None,
            core_symptom_phrase=None,
            topic_plan=topic_plan,
        )

    def _normalize_language(self, language: str) -> str:
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
    topics_covered: List[str]
    information_gaps: List[str]
    extracted_facts: Dict[str, str]
    already_mentioned_duration: bool
    already_mentioned_medications: bool
    redundant_categories: List[str]
    topic_counts: Optional[Dict[str, int]] = None


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

        all_qa = []
        for i in range(len(asked_questions)):
            if i < len(previous_answers):
                all_qa.append({
                    "question": asked_questions[i],
                    "answer": previous_answers[i]
                })

        recent_qa = []
        for i in range(max(0, len(asked_questions) - 3), len(asked_questions)):
            if i < len(previous_answers):
                recent_qa.append({
                    "question": asked_questions[i],
                    "answer": previous_answers[i]
                })

        if not all_qa:
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
            system_prompt = """Eres AGENT-02 "COVERAGE & FACT EXTRACTOR" para un asistente de admisión clínica.

Tu ÚNICO trabajo es revisar todas las preguntas y respuestas previas, junto con el plan de Agent-01, y decidir:
- qué temas ya están cubiertos,
- qué temas son claramente redundantes,
- qué temas del topic_plan todavía necesitan preguntarse.

No creas nuevas preguntas para el paciente.
Solo resumes la cobertura actual.

Etiquetas de temas permitidas (mismas que Agent-01, NO inventes etiquetas nuevas):
"duration", "associated_symptoms", "symptom_characterization",
"current_medications", "pain_assessment", "pain_characterization",
"location_radiation", "travel_history", "chronic_monitoring", "screening",
"allergies", "past_medical_history", "hpi", "menstrual_cycle",
"functional_impact", "daily_impact", "lifestyle_factors",
"triggers", "aggravating_factors", "temporal", "progression",
"frequency", "past_evaluation", "other", "exploratory", "family_history"

ESQUEMA JSON:
{
  "topics_covered": ["<etiqueta de tema>", "..."],
  "information_gaps": ["<etiqueta de tema>", "..."],
  "extracted_facts": {
    "duration": "<texto o null>",
    "medications": "<texto o null>",
    "pain_severity": "<texto o null>",
    "associated_symptoms": "<texto o null>"
  },
  "already_mentioned_duration": <true/false>,
  "already_mentioned_medications": <true/false>,
  "redundant_categories": ["<etiqueta de tema>", "..."]
}

REGLAS:
- topics_covered: temas donde hubo AL MENOS una pregunta clara y directa con respuesta.
- information_gaps:
  - DEBEN ser un subconjunto de medical_context.priority_topics.
  - DEBEN excluir todo lo que esté en medical_context.avoid_topics.
  - Deben seguir aproximadamente el orden de medical_context.topic_plan.
- duration, current_medications, travel_history y daily_impact son "UNA SOLA VEZ":
  - Una vez haya habido una pregunta clara y enfocada sobre el tema, márcalo como cubierto y agrégalo a redundant_categories.
- Sé conservador con redundant_categories:
  - Solo marca un tema como redundante si hubo una pregunta DIRECTA y enfocada con una respuesta clara.
  - En caso de duda, NO lo incluyas en redundant_categories.

Devuelve SOLO JSON, sin texto adicional."""
            user_prompt = f"""
TODAS LAS PREGUNTAS Y RESPUESTAS (para verificación de duplicados):
{self._format_qa_pairs(all_qa)}

{"ANÁLISIS DETALLADO (últimas 3 respuestas):" if len(recent_qa) < len(all_qa) else ""}
{self._format_qa_pairs(recent_qa) if len(recent_qa) < len(all_qa) else ""}

Contexto médico:
- Queja principal: {medical_context.chief_complaint}
- Temas prioritarios a cubrir: {medical_context.priority_topics}

TAREA IMPORTANTE: Analiza TODAS las {len(all_qa)} preguntas anteriores para determinar qué información YA se ha recopilado.

Devuelve un JSON con esta estructura EXACTA:

{{
    "topics_covered": [
        "duration",
        "associated_symptoms"
    ],
    "information_gaps": [
        "current_medications"
    ],
    "extracted_facts": {{
        "duration": "<si se mencionó duración EN CUALQUIER RESPUESTA, extrae aquí (p. ej. '3 semanas'), si no null>",
        "medications": "<si se mencionaron medicamentos EN CUALQUIER RESPUESTA, lista aquí, si no null>",
        "pain_severity": "<si se mencionó severidad del dolor EN CUALQUIER RESPUESTA (1-10), si no null>",
        "associated_symptoms": "<si se mencionaron síntomas asociados EN CUALQUIER RESPUESTA, lista aquí, si no null>"
    }},
    "already_mentioned_duration": <true/false>,
    "already_mentioned_medications": <true/false>,
    "redundant_categories": [
        "duration"
    ]
}}

Reglas importantes:
- Solo uses etiquetas de temas EXACTAS de la lista permitida. NUNCA inventes etiquetas nuevas.
- "topics_covered": temas que YA fueron preguntados y respondidos de forma clara en CUALQUIER pregunta.
- "information_gaps": temas prioritarios que AÚN NO se han cubierto adecuadamente.
- "redundant_categories": temas que NO deben preguntarse de nuevo porque ya se cubrieron claramente.
- Sé conservador al marcar categorías como redundantes: solo marca un tema como redundante si hubo al menos una pregunta directa y específica sobre ese tema con una respuesta clara. En caso de duda, NO lo incluyas.

Etiquetas de temas permitidas (usa SOLO estas cadenas EXACTAS, nunca inventes etiquetas nuevas):
"duration", "associated_symptoms", "symptom_characterization", "current_medications", "pain_assessment",
"pain_characterization", "location_radiation", "travel_history", "chronic_monitoring", "screening",
"allergies", "past_medical_history", "hpi", "menstrual_cycle", "functional_impact", "daily_impact",
"lifestyle_factors", "triggers", "aggravating_factors", "temporal", "progression", "frequency",
"past_evaluation", "other", "exploratory", "family_history".

Devuelve SOLO el JSON, sin texto adicional."""
        else:
            system_prompt = """You are AGENT-02 "COVERAGE & FACT EXTRACTOR" for a clinical intake assistant.

Your ONLY job is to review all previous questions and answers plus the plan from Agent-01 and decide:
- which topics are already covered,
- which topics are clearly redundant,
- which topics from the topic_plan still need to be asked.

You DO NOT create new questions for the patient.
You ONLY summarize what has been covered so far.

TOPIC LABELS (same allow-list as Agent-01, do NOT invent new labels):
"duration", "associated_symptoms", "symptom_characterization",
"current_medications", "pain_assessment", "pain_characterization",
"location_radiation", "travel_history", "chronic_monitoring", "screening",
"allergies", "past_medical_history", "hpi", "menstrual_cycle",
"functional_impact", "daily_impact", "lifestyle_factors",
"triggers", "aggravating_factors", "temporal", "progression",
"frequency", "past_evaluation", "other", "exploratory", "family_history"

JSON SCHEMA:
{
  "topics_covered": ["<topic label>", "..."],
  "information_gaps": ["<topic label>", "..."],
  "extracted_facts": {
    "duration": "<text or null>",
    "medications": "<text or null>",
    "pain_severity": "<text or null>",
    "associated_symptoms": "<text or null>"
  },
  "already_mentioned_duration": <true/false>,
  "already_mentioned_medications": <true/false>,
  "redundant_categories": ["<topic label>", "..."]
}

RULES:
- topics_covered: topics where at least ONE CLEAR, direct question was asked and answered.
- information_gaps MUST:
  - be a subset of medical_context.priority_topics,
  - exclude everything in medical_context.avoid_topics,
  - be ordered roughly following medical_context.topic_plan.
- Once there has been a clear, focused Q&A about a topic, treat these as ONE-AND-DONE and add to redundant_categories:
  - "duration", "current_medications", "travel_history", "daily_impact".
- Be conservative with redundant_categories:
  - Only mark a topic as redundant if there was a DIRECT, focused question with a clear answer.
  - If in doubt, leave the topic OUT of redundant_categories.

Return ONLY JSON, no extra text."""

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
        "duration",
        "associated_symptoms"
    ],
    "information_gaps": [
        "current_medications"
    ],
    "extracted_facts": {{
        "duration": "<if duration mentioned IN ANY ANSWER, extract here (e.g. '3 weeks'), else null>",
        "medications": "<if medications mentioned IN ANY ANSWER, list here, else null>",
        "pain_severity": "<if pain severity mentioned IN ANY ANSWER (1-10), else null>",
        "associated_symptoms": "<if associated symptoms mentioned IN ANY ANSWER, list here, else null>"
    }},
    "already_mentioned_duration": <true/false>,
    "already_mentioned_medications": <true/false>,
    "redundant_categories": [
        "duration"
    ]
}}

Important rules:
- You MUST only use these exact topic labels. Never invent new labels.
- "topics_covered": topics that were ALREADY asked and clearly answered in ANY question.
- "information_gaps": priority topics that have NOT yet been adequately covered.
- "redundant_categories": topics that should NOT be asked again because they are clearly covered.
- Be conservative when marking categories as redundant: only mark a topic as redundant if there has been at least one direct, focused question about that topic with a clear, specific answer. If in doubt, do NOT include it.

Allowed topic labels (use ONLY these exact strings, never invent new ones):
"duration", "associated_symptoms", "symptom_characterization", "current_medications", "pain_assessment",
"pain_characterization", "location_radiation", "travel_history", "chronic_monitoring", "screening",
"allergies", "past_medical_history", "hpi", "menstrual_cycle", "functional_impact", "daily_impact",
"lifestyle_factors", "triggers", "aggravating_factors", "temporal", "progression", "frequency",
"past_evaluation", "other", "exploratory", "family_history".

Return ONLY the JSON, no additional text."""

        try:
            response = await self._client.chat(
                model=self._settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.1,
            )

            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON found in Agent 2 response: {response_text[:200]}")
                raise ValueError("No JSON found in response")

            extraction = json.loads(json_match.group())

            # ---- Normalize topic lists and clamp to allowed set ----
            def _to_str_list(value: Any) -> List[str]:
                if not isinstance(value, list):
                    return []
                result: List[str] = []
                for v in value:
                    if isinstance(v, (str, int, float)):
                        s = str(v).strip()
                        if s:
                            result.append(s)
                return result

            raw_topics_covered = extraction.get("topics_covered", []) or []
            raw_information_gaps = extraction.get("information_gaps", medical_context.priority_topics) or []
            raw_redundant_categories = extraction.get("redundant_categories", []) or []

            topics_covered = _to_str_list(raw_topics_covered)
            information_gaps = _to_str_list(raw_information_gaps)
            redundant_categories = _to_str_list(raw_redundant_categories)

            allowed = set(ALLOWED_TOPICS)
            unknown_topics = (
                set(topics_covered)
                .union(information_gaps)
                .union(redundant_categories)
                - allowed
            )
            if unknown_topics:
                logger.warning(
                    f"Agent 2: Dropping unknown topic labels from LLM: {sorted(unknown_topics)}"
                )

            topics_covered = [t for t in topics_covered if t in allowed]
            information_gaps = [t for t in information_gaps if t in allowed]
            redundant_categories = [t for t in redundant_categories if t in allowed]

            # ---- Ensure travel topic is treated as covered once a travel question was asked ----
            if any(
                any(kw in (q or "").lower() for kw in TRAVEL_KEYWORDS)
                for q in asked_questions or []
            ):
                if "travel_history" in information_gaps:
                    information_gaps.remove("travel_history")
                if "travel_history" not in redundant_categories:
                    redundant_categories.append("travel_history")

            # ---- One-and-done hard rule for daily_impact (similar to travel_history) ----
            try:
                any_daily_impact_q = any(
                    "daily activities" in (q or "").lower()
                    or "overall quality of life" in (q or "").lower()
                    or "daily life" in (q or "").lower()
                    for q in (asked_questions or [])
                )
            except Exception:
                any_daily_impact_q = False

            if any_daily_impact_q:
                if "daily_impact" in information_gaps:
                    information_gaps.remove("daily_impact")
                if "daily_impact" not in redundant_categories:
                    redundant_categories.append("daily_impact")

            # ---- Tie gaps to priority_topics AND avoid_topics from Agent-01 ----
            priority = set(medical_context.priority_topics or [])
            avoid = set(medical_context.avoid_topics or [])

            information_gaps = [
                t for t in information_gaps
                if t in priority and t not in avoid
            ]

            if not information_gaps and priority:
                logger.info(
                    "Agent 2: information_gaps empty after filtering; "
                    "falling back to priority_topics minus topics_covered/avoid."
                )
                information_gaps = [
                    t for t in medical_context.priority_topics
                    if t not in topics_covered and t not in avoid
                ]

            # ---- Extracted facts normalization ----
            extracted_facts = extraction.get("extracted_facts", {}) or {}
            if not isinstance(extracted_facts, dict):
                extracted_facts = {}

            for key in ["duration", "medications", "pain_severity", "associated_symptoms"]:
                value = extracted_facts.get(key)
                if value is not None and not isinstance(value, str):
                    extracted_facts[key] = str(value)

            already_mentioned_duration = bool(extraction.get("already_mentioned_duration", False))
            already_mentioned_medications = bool(extraction.get("already_mentioned_medications", False))

            # ---- Make redundant_categories more conservative ----
            SENSITIVE_TOPICS = {"associated_symptoms", "current_medications"}
            redundant_categories = [
                t for t in redundant_categories
                if t not in SENSITIVE_TOPICS
            ]

            # ---- Topic counts: use provided or derive from topics_covered ----
            topic_counts = extraction.get("topic_counts")
            if not isinstance(topic_counts, dict):
                topic_counts = dict(Counter(topics_covered))

            logger.info(
                f"Agent 2 Results: Covered={topics_covered}, "
                f"Gaps={information_gaps}, Redundant={redundant_categories}"
            )

            return ExtractedInformation(
                topics_covered=topics_covered,
                information_gaps=information_gaps,
                extracted_facts=extracted_facts,
                already_mentioned_duration=already_mentioned_duration,
                already_mentioned_medications=already_mentioned_medications,
                redundant_categories=redundant_categories,
                topic_counts=topic_counts,
            )

        except Exception as e:
            logger.error(f"Answer extraction failed: {e}")
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
        formatted = []
        for i, qa in enumerate(qa_pairs, 1):
            formatted.append(f"{i}. Q: {qa['question']}\n   A: {qa['answer']}")
        return "\n\n".join(formatted)

    def _normalize_language(self, language: str) -> str:
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
        is_deep_diagnostic: bool = False,
        deep_diagnostic_question_num: Optional[int] = None,
    ) -> str:
        """
        Generate next medical question based on context and what's been covered.

        KEY:
        - In deep diagnostic mode, we IGNORE information_gaps for topic selection and force
          the question to be about a specific domain.
          • CHRONIC: monitoring / labs / screening
          • NON-CHRONIC: triggers / functional_impact / past_evaluation
        """

        # 1) Determine if condition is chronic
        #    - Prefer Agent-01 decision; use keyword-based detection only as fallback.
        condition_props = {}
        try:
            if medical_context and medical_context.condition_properties:
                condition_props = dict(medical_context.condition_properties)
        except Exception:
            condition_props = medical_context.condition_properties or {}

        is_chronic_flag = condition_props.get("is_chronic", None)  # None = unknown
        is_chronic_condition = bool(is_chronic_flag) if is_chronic_flag is not None else False

        # Build combined text from chief complaint + ALL answers for chronic keyword detection (fallback only)
        combined_text_parts: List[str] = []
        try:
            if medical_context and medical_context.chief_complaint:
                combined_text_parts.append(medical_context.chief_complaint.lower())
        except Exception:
            pass
        try:
            if previous_answers:
                combined_text_parts.append(" ".join(a.lower() for a in previous_answers if a))
        except Exception:
            pass

        combined_text = " ".join(combined_text_parts)

        chronic_keywords = [
            "diabetes",
            "type 1 diabetes",
            "type 2 diabetes",
            "hypertension",
            "high blood pressure",
            "asthma",
            "copd",
            "chronic obstructive",
            "chronic kidney",
            "ckd",
            "heart failure",
            "coronary artery",
            "epilepsy",
            "thyroid",
        ]

        # Only infer chronic if Agent-01 did not specify (fallback)
        if is_chronic_flag is None and any(kw in combined_text for kw in chronic_keywords):
            condition_props["is_chronic"] = True
            if medical_context:
                medical_context.condition_properties = condition_props
            is_chronic_condition = True
            logger.info(
                "Agent 3: is_chronic inferred from keywords in Q&A (fallback), chief_complaint='%s'",
                medical_context.chief_complaint if medical_context else "",
            )

        lang = self._normalize_language(language)

        # Format Q&A history (for context in the prompt)
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

        # Closing question for standard flow
        if current_count + 1 >= max_count and not is_deep_diagnostic:
            if lang == "sp":
                return "¿Hay algo más que le gustaría compartir sobre su condición?"
            return "Is there anything else you'd like to share about your condition?"

        # ------------------------------------------------------------------ #
        # DEEP DIAGNOSTIC MODE
        # ------------------------------------------------------------------ #
        if is_deep_diagnostic:
            if lang == "sp":
                system_prompt = """Eres un médico experimentado realizando una entrevista médica.

Tu trabajo es generar UNA pregunta diagnóstica detallada, clara y amigable relacionada
con el dominio específico que se te indica."""
                
                q_num = deep_diagnostic_question_num or 1
                q_num = max(1, min(3, q_num))

                if is_chronic_condition:
                    # CHRONIC: fixed domains (1 = monitoring, 2 = labs, 3 = screening)
                    if q_num == 1:
                        domain_title = "MONITOREO EN CASA / LECTURAS"
                        domain_desc = (
                            "Cómo el paciente ha estado monitoreando esta condición crónica en casa "
                            "(por ejemplo, azúcar en sangre, presión arterial, flujo pico, registros de peso, lecturas de dispositivos)."
                        )
                    elif q_num == 2:
                        domain_title = "RESULTADOS RECIENTES DE PRUEBAS DE LABORATORIO"
                        domain_desc = (
                            "Pruebas de laboratorio recientes relevantes para esta enfermedad crónica (por ejemplo, HbA1c, función renal, "
                            "colesterol, pruebas de tiroides) y lo que el paciente recuerda sobre los resultados."
                        )
                    else:
                        domain_title = "EXÁMENES DE DETECCIÓN / VERIFICACIÓN DE COMPLICACIONES"
                        domain_desc = (
                            "Exámenes formales y pruebas de detección de complicaciones (por ejemplo, exámenes de ojos/pies para diabetes, "
                            "pruebas cardíacas, imágenes renales, pruebas de función pulmonar) realizadas debido a esta condición."
                        )

                    user_prompt = f"""
Contexto médico:
- Motivo de consulta: {medical_context.chief_complaint}
- Edad del paciente: {medical_context.patient_age or "Desconocida"}
- Género del paciente: {medical_context.patient_gender or "No especificado"}
- Propiedades de la condición: {json.dumps(medical_context.condition_properties, indent=2)}
- Esta es una condición CRÓNICA (por ejemplo, diabetes, hipertensión, asma, etc.)

Modo Diagnóstico Profundo – Pregunta {q_num} de 3 (enfermedad CRÓNICA)

DOMINIO DIAGNÓSTICO SOLO PARA ESTA PREGUNTA:
- Dominio: {domain_title}
- Descripción: {domain_desc}

Reglas para esta pregunta de diagnóstico profundo:
- IGNORA information_gaps y redundant_categories para la selección de temas; son solo para contexto.
- DEBES hacer EXACTAMENTE UNA pregunta sobre ESTE dominio solamente. NO combines con otros dominios.
- CRÍTICO: Mantén la pregunta CORTA (15-25 palabras). Enfócate en UN aspecto específico de este dominio.
- Incluso si este dominio se tocó antes (por ejemplo, se mencionó una lectura única de presión arterial),
  DEBES hacer una pregunta de diagnóstico profundo enfocada que agregue MÁS detalle que antes.
  NO omitas este dominio.
- NO preguntes sobre severidad de síntomas, detalles genéricos de dolor de cabeza, peso/apetito, alergias, sueño,
  o estilo de vida general en esta pregunta.
- NO preguntes sobre otros dominios (por ejemplo, si esta es la pregunta de monitoreo, NO preguntes sobre pruebas de laboratorio o exámenes de detección).
- Haz la pregunta específica, concreta, concisa y fácil de responder para el paciente en sus propias palabras.
- Fraseala como una pregunta única, directa, CORTA que termine con "?".

Historial de Q&A (solo contexto – NO elijas tema de esto, solo evita duplicación obvia):
{qa_history if qa_history else "No hay preguntas previas."}

REQUISITOS DE SALIDA:
- Devuelve SOLO el texto de la pregunta, sin comillas, sin numeración, sin explicaciones.
- Ejemplo de estilo correcto (para dominio de monitoreo): 
  "¿Con qué frecuencia revisas tu presión arterial en casa y cómo han sido tus lecturas recientes generalmente?"

Genera la siguiente pregunta de diagnóstico profundo ahora, estrictamente sobre este dominio."""
                else:
                    # NON-CHRONIC: deep diagnostic sequence = triggers, functional impact, past evaluation
                    if q_num == 1:
                        domain_title = "DESENCADENANTES, ALIVIADORES Y CONTEXTO"
                        domain_desc = (
                            "Qué tiende a provocar o empeorar el síntoma principal, qué lo mejora, "
                            "y en qué situaciones o momentos del día suele aparecer."
                        )
                    elif q_num == 2:
                        domain_title = "IMPACTO FUNCIONAL"
                        domain_desc = (
                            "Cómo este problema está afectando la capacidad del paciente para realizar actividades diarias, "
                            "trabajo, ejercicio, sueño o autocuidado."
                        )
                    else:
                        domain_title = "EVALUACIÓN PASADA / TRABAJO DIAGNÓSTICO"
                        domain_desc = (
                            "Evaluaciones médicas previas específicamente para este problema – por ejemplo visitas a clínica o urgencias, "
                            "ECG, rayos X, escaneos, análisis de sangre, o consultas con especialistas – y lo que se le dijo al paciente."
                        )

                    user_prompt = f"""
Contexto médico:
- Motivo de consulta: {medical_context.chief_complaint}
- Edad del paciente: {medical_context.patient_age or "Desconocida"}
- Género del paciente: {medical_context.patient_gender or "No especificado"}
- Propiedades de la condición: {json.dumps(medical_context.condition_properties, indent=2)}

Modo Diagnóstico Profundo – Pregunta {q_num} de 3 (condición NO CRÓNICA)

DOMINIO DIAGNÓSTICO SOLO PARA ESTA PREGUNTA:
- Dominio: {domain_title}
- Descripción: {domain_desc}

Reglas para esta pregunta de diagnóstico profundo:
- IGNORA information_gaps y redundant_categories para la selección de temas; son solo para contexto.
- DEBES hacer EXACTAMENTE UNA pregunta que profundice la comprensión del MOTIVO DE CONSULTA dentro de este dominio.
- CRÍTICO: Mantén la pregunta CORTA (15-25 palabras). Enfócate en UN aspecto específico de este dominio.
- Incluso si esta área se mencionó antes, DEBES hacer una pregunta de diagnóstico profundo enfocada para este dominio.
  NO omitas este dominio.
- NO preguntes sobre temas separados y no relacionados como alergias, estilo de vida general, o nuevos sistemas de órganos.
- NO combines múltiples aspectos o temas en una pregunta.
- Haz la pregunta específica, concreta, concisa y fácil de responder para el paciente.
- Frasea la pregunta como una pregunta única, directa, CORTA que termine con "?".

Historial de Q&A (solo contexto – NO elijas tema de esto, solo evita duplicación obvia):
{qa_history if qa_history else "No hay preguntas previas."}

REQUISITOS DE SALIDA:
- Devuelve SOLO el texto de la pregunta, sin comillas, sin numeración, sin explicaciones.
- Ejemplo de estilo correcto (para dominio de desencadenantes):
  "¿Has notado algo que tiende a desencadenar o empeorar este problema, o algo que confiablemente lo hace sentir mejor?"

Genera la siguiente pregunta de diagnóstico profundo ahora, estrictamente sobre este dominio."""
            else:
                system_prompt = """You are an experienced physician conducting a focused diagnostic interview.

Your job is to generate ONE detailed, clinically relevant, patient-friendly question
about the EXACT diagnostic domain specified below."""

                q_num = deep_diagnostic_question_num or 1
                q_num = max(1, min(3, q_num))

                if is_chronic_condition:
                    # CHRONIC: fixed domains (1 = monitoring, 2 = labs, 3 = screening)
                    if q_num == 1:
                        domain_title = "HOME MONITORING / READINGS"
                        domain_desc = (
                            "How the patient has been monitoring this chronic condition at home "
                            "(e.g., blood sugar, blood pressure, peak flow, weight logs, device readings)."
                        )
                    elif q_num == 2:
                        domain_title = "RECENT LAB TEST RESULTS"
                        domain_desc = (
                            "Recent lab tests relevant to this chronic disease (e.g., HbA1c, kidney function, "
                            "cholesterol, thyroid labs) and what the patient remembers about the results."
                        )
                    else:
                        domain_title = "SCREENING / COMPLICATION CHECKS"
                        domain_desc = (
                            "Formal exams and screening tests for complications (e.g., diabetic eye/foot exams, "
                            "cardiac tests, kidney imaging, lung function tests) done because of this condition."
                        )

                    user_prompt = f"""
Medical context:
- Chief complaint: {medical_context.chief_complaint}
- Patient age: {medical_context.patient_age or "Unknown"}
- Patient gender: {medical_context.patient_gender or "Not specified"}
- Condition properties: {json.dumps(medical_context.condition_properties, indent=2)}
- This is a CHRONIC condition (for example, diabetes, hypertension, asthma, etc.)

Deep Diagnostic Mode – Question {q_num} of 3 (CHRONIC disease)

DIAGNOSTIC DOMAIN FOR THIS QUESTION ONLY:
- Domain: {domain_title}
- Description: {domain_desc}

Rules for this deep diagnostic question:
- IGNORE information_gaps and redundant_categories for topic selection; they are for context only.
- You MUST ask EXACTLY ONE question about THIS domain only. Do NOT combine with other domains.
- CRITICAL: Keep the question SHORT (15-25 words). Focus on ONE specific aspect of this domain.
- Even if this domain was touched earlier (for example, a single BP reading was mentioned),
  you MUST still ask a focused deep diagnostic question that adds MORE detail than before.
  Do NOT skip this domain.
- Do NOT ask about symptom severity, generic headache details, weight/appetite, allergies, sleep,
  or general lifestyle in this question.
- Do NOT ask about other domains (for example, if this is the monitoring question, do NOT ask about lab tests or screening exams).
- Make the question specific, concrete, concise, and easy for the patient to answer in their own words.
- Phrase it as a single, direct, SHORT question ending with "?".

Q&A history (context only – do NOT choose topic from this, just avoid obvious duplication):
{qa_history if qa_history else "No previous questions."}

OUTPUT REQUIREMENTS:
- Return ONLY the question text, no quotes, no numbering, no explanations.
- Example of correct style (for monitoring domain): 
  "How often do you check your blood pressure at home, and what have your recent readings usually been like?"

Generate the next deep diagnostic question now, strictly about this domain."""
                else:
                    # NON-CHRONIC: deep diagnostic sequence = triggers, functional impact, past evaluation
                    if q_num == 1:
                        domain_title = "TRIGGERS, RELIEVERS AND CONTEXT"
                        domain_desc = (
                            "What tends to bring on or worsen the main symptom, what makes it better, "
                            "and in what situations or times of day it usually appears."
                        )
                    elif q_num == 2:
                        domain_title = "FUNCTIONAL IMPACT"
                        domain_desc = (
                            "How this problem is affecting the patient's ability to carry out daily activities, "
                            "work, exercise, sleep, or self-care."
                        )
                    else:
                        domain_title = "PAST EVALUATION / WORKUP"
                        domain_desc = (
                            "Previous medical evaluations specifically for this problem – for example clinic or ER visits, "
                            "ECG, X-ray, scans, blood tests, or specialist consultations – and what the patient was told."
                        )

                    user_prompt = f"""
Medical context:
- Chief complaint: {medical_context.chief_complaint}
- Patient age: {medical_context.patient_age or "Unknown"}
- Patient gender: {medical_context.patient_gender or "Not specified"}
- Condition properties: {json.dumps(medical_context.condition_properties, indent=2)}

Deep Diagnostic Mode – Question {q_num} of 3 (NON-CHRONIC condition)

DIAGNOSTIC DOMAIN FOR THIS QUESTION ONLY:
- Domain: {domain_title}
- Description: {domain_desc}

Rules for this deep diagnostic question:
- IGNORE information_gaps and redundant_categories for topic selection; they are for context only.
- You MUST ask EXACTLY ONE question that deepens understanding of the CHIEF COMPLAINT within this domain.
- CRITICAL: Keep the question SHORT (15-25 words). Focus on ONE specific aspect of this domain.
- Even if this area was mentioned earlier, you MUST still ask one focused deep diagnostic question for this domain.
  Do NOT skip this domain.
- Do NOT ask about separate, unrelated topics such as allergies, general lifestyle, or new organ systems.
- Do NOT combine multiple aspects or topics in one question.
- Make the question specific, concrete, concise, and easy for the patient to answer.
- Phrase the question as a single, direct, SHORT question ending with "?".

Q&A history (context only – do NOT choose topic from this, just avoid obvious duplication):
{qa_history if qa_history else "No previous questions."}

OUTPUT REQUIREMENTS:
- Return ONLY the question text, no quotes, no numbering, no explanations.
- Example of correct style (for triggers domain):
  "Have you noticed anything that tends to trigger or worsen this problem, or anything that reliably makes it feel better?"

Generate the next deep diagnostic question now, strictly about this domain."""

            try:
                response = await self._client.chat(
                    model=self._settings.openai.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=100,  # Reduced to enforce shorter, single-topic questions
                    temperature=0.2,
                )
                question = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Deep diagnostic question generation failed: {e}")
                # Safe fallback
                if lang == "sp":
                    return "¿Puede contarme un poco más de detalle sobre cómo se ha presentado este problema?"
                return "Can you tell me a bit more detail about how this problem has been occurring for you?"

            question = self._postprocess_question_text(question)
            logger.info(f"Generated deep diagnostic question (cleaned): {question}")
            return question
        # ------------------------------------------------------------------ #
        # NORMAL (NON-DEEP) QUESTION GENERATION
        # ------------------------------------------------------------------ #

        # Determine chosen_topic using topic_plan + information_gaps
        info_gaps = list(extracted_info.information_gaps or [])
        redundant = set(extracted_info.redundant_categories or [])
        avoid = set(medical_context.avoid_topics or [])
        topic_plan = list(getattr(medical_context, "topic_plan", []) or [])

        base_candidates = [t for t in info_gaps if t not in avoid and t not in redundant]
        ordered_candidates = [t for t in topic_plan if t in base_candidates]
        chosen_topic = ordered_candidates[0] if ordered_candidates else (base_candidates[0] if base_candidates else None)

        if lang == "sp":
            system_prompt = """Eres un médico experimentado realizando una entrevista médica.

Tu trabajo es generar UNA pregunta médicamente relevante, clara y amigable para el paciente."""

            avoid_similar_block = ""
            if avoid_similar_to:
                avoid_similar_block = f"""

PREGUNTA A EVITAR:
- NO hagas ninguna pregunta semánticamente similar a esta pregunta previa:
  "{avoid_similar_to}"
- No repitas la misma intención ni el mismo enunciado. Elige un ángulo o tema nuevo.
"""

            topic_line = (
                f"Tema que DEBES abordar en esta pregunta: {chosen_topic}"
                if chosen_topic
                else "Tema que DEBES abordar en esta pregunta: general_cierre"
            )

            user_prompt = f"""
Contexto médico:
- Motivo de consulta: {medical_context.chief_complaint}
- Edad del paciente: {medical_context.patient_age or "Desconocida"}
- Género del paciente: {medical_context.patient_gender or "No especificado"}
- Propiedades de la condición: {json.dumps(medical_context.condition_properties, indent=2)}
- Razonamiento médico: {medical_context.medical_reasoning}
{topic_line}

Temas prioritarios a cubrir: {medical_context.priority_topics}
Temas a EVITAR: {medical_context.avoid_topics}

{"=" * 60}
TODAS LAS PREGUNTAS Y RESPUESTAS ANTERIORES (CONTEXTO COMPLETO):
{qa_history if qa_history else "No hay preguntas previas - esta es la primera pregunta."}
{"=" * 60}

Información ya recolectada:
- Temas cubiertos: {extracted_info.topics_covered}
- Brechas de información (temas que aún necesitan ser cubiertos): {extracted_info.information_gaps if extracted_info.information_gaps else "[]"}
- Hechos extraídos: {json.dumps(extracted_info.extracted_facts, indent=2)}
- ¿Ya se mencionó la duración?: {extracted_info.already_mentioned_duration}
- ¿Ya se preguntó sobre medicamentos?: {extracted_info.already_mentioned_medications}
- Categorías redundantes (NO deben repetirse): {extracted_info.redundant_categories}

Progreso: Pregunta {current_count + 1} de {max_count}
{avoid_similar_block}

REGLAS DE FLUJO Y CATEGORÍA:
- Elige la siguiente pregunta de information_gaps, dando prioridad a temas médicamente importantes.
- NUNCA elijas un tema que aparezca en redundant_categories o avoid_topics.
- CRÍTICO: Pregunta sobre SOLO UNA categoría/tema por pregunta. NO combines múltiples categorías.
- La duración debe preguntarse una vez temprano en la entrevista (SOLO UNA pregunta).
- Los medicamentos combinados + cuidado en casa/autocuidado deben preguntarse una vez temprano como una sola pregunta (esta es la ÚNICA excepción a la regla de una categoría).
- Los síntomas asociados deben preguntarse solo una vez, pero para enfermedades crónicas es de alta prioridad temprano.
- Para condiciones crónicas, las preguntas de chronic_monitoring y screening están reservadas SOLO para modo diagnóstico profundo (después de consentimiento explícito).
- Para condiciones no crónicas, las preguntas de past_evaluation (pruebas previas o visitas a clínica/urgencias para este problema) están reservadas
  SOLO para modo diagnóstico profundo; NO preguntes sobre evaluación pasada en el flujo normal.
- Los temas relacionados con dolor (pain_assessment, pain_characterization, location_radiation) son relevantes solo cuando el dolor es parte de la presentación.
- Cuando el dolor es un síntoma clave, incluye exactamente una pregunta corta de estilo 'location_radiation' aclarando dónde está el dolor y
  si se desplaza a algún lugar (por ejemplo al brazo, mandíbula, espalda u otras áreas).
- La pregunta debe ser CORTA, estilo entrevista (lenguaje natural), no una lista de verificación.
- Mantén las preguntas concisas: máximo 15-25 palabras. NO crees preguntas largas de múltiples partes.

REGLAS DE TEMA CONDICIONAL (CRÍTICO):
- Historial de viajes: SOLO pregunta si los síntomas son plausiblemente relacionados con viajes (infecciones, enfermedades endémicas, tiempo de viaje reciente).
  NO hagas preguntas genéricas de viajes solo porque el paciente viajó.
- Historial familiar: SOLO pregunta si la condición es crónica o hereditaria (is_hereditary=true o is_chronic=true).
  Pregunta sobre el historial familiar de la condición ESPECÍFICA, no historial familiar genérico.
- Historial médico pasado / Alergias: SOLO pregunta si:
  a) La condición es crónica (interacciones de medicamentos, cuidado continuo), O
  b) La condición está relacionada con alergias, O
  c) Queja de alto riesgo (dolor de pecho, dificultad para respirar, anafilaxis, relacionado con medicamentos)
  NO hagas preguntas genéricas de PMH/alergias solo para llenar el conteo de preguntas.

PREVENCIÓN DE DUPLICADOS:
- NO repitas lo que ya se ha preguntado claramente en preguntas anteriores.
- Usa topics_covered y redundant_categories para evitar duplicados.

FORMATO DE SALIDA:
- Responde con SOLO el texto de la pregunta.
- Sin comillas, sin numeración, sin explicaciones.
- Termina con un solo carácter "?".
- Mantén la pregunta CORTA (máximo 15-25 palabras).
- Enfócate en SOLO UNA categoría/tema.

REGLAS DE SELECCIÓN DE TEMA:
- Primero, elige internamente EXACTAMENTE UN tema de information_gaps en el que enfocarte.
- Ese tema DEBE ser una de estas etiquetas (no inventes etiquetas nuevas):
  "duration", "associated_symptoms", "symptom_characterization", "current_medications",
  "pain_assessment", "pain_characterization", "location_radiation", "travel_history",
  "chronic_monitoring", "screening", "allergies", "past_medical_history", "hpi",
  "menstrual_cycle", "functional_impact", "daily_impact", "lifestyle_factors",
  "triggers", "aggravating_factors", "temporal", "progression", "frequency",
  "past_evaluation", "other", "exploratory", "family_history".
- NUNCA selecciones un tema que aparezca en redundant_categories o avoid_topics, y NO vuelvas a preguntar sobre ese concepto.
- Además, si un tema ya aparece en topics_covered, debes evitar volver a preguntarlo,
  incluso si todavía aparece en information_gaps (asume que eso es ruido).
- Si "travel_history" está en los temas a EVITAR, no hagas ninguna pregunta relacionada con viajes.
- Si "family_history" está en los temas a EVITAR, no preguntes sobre historial familiar.
- Si "allergies" está en los temas a EVITAR, no preguntes sobre alergias.
- Haz EXACTAMENTE UNA pregunta que se enfoque SOLO en el tema elegido. No mezcles varios temas en una sola pregunta.

Genera la siguiente pregunta MÁS importante ahora eligiendo UN tema de information_gaps que
no esté en redundant_categories o avoid_topics."""
        else:
            system_prompt = """You are an experienced physician conducting a medical interview.

Your job is to generate ONE medically relevant, clear, and patient-friendly question."""

            avoid_similar_block = ""
            if avoid_similar_to:
                avoid_similar_block = f"""

AVOID SIMILAR QUESTION:
- Do NOT ask anything semantically similar to this previous question:
  "{avoid_similar_to}"
- Do not repeat the same intent or wording. Choose a new angle or topic.
"""

            topic_line = (
                f"Topic you MUST focus on for this question: {chosen_topic}"
                if chosen_topic
                else "Topic you MUST focus on for this question: general_closing"
            )

            user_prompt = f"""
Medical context:
- Chief complaint: {medical_context.chief_complaint}
- Patient age: {medical_context.patient_age or "Unknown"}
- Patient gender: {medical_context.patient_gender or "Not specified"}
- Condition properties: {json.dumps(medical_context.condition_properties, indent=2)}
- Medical reasoning: {medical_context.medical_reasoning}
{topic_line}

Priority topics to cover: {medical_context.priority_topics}
Topics to AVOID: {medical_context.avoid_topics}

{"=" * 60}
ALL PREVIOUS QUESTIONS AND ANSWERS (FULL CONTEXT):
{qa_history if qa_history else "No previous questions - this is the first question."}
{"=" * 60}

Information already collected:
- Topics covered: {extracted_info.topics_covered}
- Information gaps (topics that still need to be covered): {extracted_info.information_gaps if extracted_info.information_gaps else "[]"}
- Extracted facts: {json.dumps(extracted_info.extracted_facts, indent=2)}
- Duration already mentioned?: {extracted_info.already_mentioned_duration}
- Medications already asked?: {extracted_info.already_mentioned_medications}
- Redundant categories (must NOT be repeated): {extracted_info.redundant_categories}

Progress: Question {current_count + 1} of {max_count}
{avoid_similar_block}

FLOW & CATEGORY RULES:
- Choose the next question from information_gaps, giving priority to medically important topics.
- NEVER pick a topic that appears in redundant_categories or avoid_topics.
- CRITICAL: Ask about ONLY ONE category/topic per question. Do NOT combine multiple categories.
- Duration should be asked once early in the interview (ONE question only).
- Combined medications + home/self-care should be asked once early as a single question (this is the ONLY exception to one-category rule).
- Associated symptoms should be asked only once, but for chronic diseases it is high-priority early.
- For chronic conditions, chronic_monitoring and screening questions are reserved for deep diagnostic mode ONLY (after explicit consent).
- For non-chronic conditions, past_evaluation questions (previous tests or ER/clinic visits for this problem) are reserved
  for deep diagnostic mode ONLY; do NOT ask about past evaluation in the normal flow.
- Pain-related topics (pain_assessment, pain_characterization, location_radiation) are relevant only when pain is part of the presentation.
- When pain is a key symptom, include exactly one short 'location_radiation' style question clarifying where the pain is and
  whether it travels anywhere (for example to arm, jaw, back, or other areas).
- Question must be SHORT, interview-style (natural language), not a checklist.
- Keep questions concise: 15-25 words maximum. Do NOT create long multi-part questions.

CONDITIONAL TOPIC RULES (CRITICAL):
- Travel history: ONLY ask if symptoms are plausibly travel-related (infections, endemic diseases, recent travel timing).
  Do NOT ask generic travel questions just because patient traveled.
- Family history: ONLY ask if condition is chronic or hereditary (is_hereditary=true or is_chronic=true).
  Ask about family history of the SPECIFIC condition, not generic family history.
- Past medical history / Allergies: ONLY ask if:
  a) Condition is chronic (medication interactions, ongoing care), OR
  b) Condition is allergy-related, OR
  c) High-risk complaint (chest pain, SOB, anaphylaxis, medication-related)
  Do NOT ask generic PMH/allergy questions just to fill question count.

DUPLICATE PREVENTION:
- Do NOT repeat what has already been clearly asked in previous questions.
- Use topics_covered and redundant_categories to avoid duplicates.

OUTPUT FORMAT:
- Respond with ONLY the question text.
- No quotes, no numbering, no explanations.
- End with a single "?" character.
- Keep the question SHORT (15-25 words maximum).
- Focus on ONE category/topic only.

TOPIC SELECTION RULES:
- First, internally choose EXACTLY ONE topic from information_gaps to focus on.
- That topic MUST be one of these labels only (do not invent new labels):
  "duration", "associated_symptoms", "symptom_characterization", "current_medications",
  "pain_assessment", "pain_characterization", "location_radiation", "travel_history",
  "chronic_monitoring", "screening", "allergies", "past_medical_history", "hpi",
  "menstrual_cycle", "functional_impact", "daily_impact", "lifestyle_factors",
  "triggers", "aggravating_factors", "temporal", "progression", "frequency",
  "past_evaluation", "other", "exploratory", "family_history".
- NEVER select a topic that appears in redundant_categories or avoid_topics, and do NOT ask about that concept again.
- Additionally, if a topic appears in topics_covered, strongly avoid asking about it again,
  even if it still appears in information_gaps (assume that is noise).
- If "travel_history" is in Topics to AVOID, do not ask any travel-related question at all.
- If "family_history" is in Topics to AVOID, do not ask about family history.
- If "allergies" is in Topics to AVOID, do not ask about allergies.
- Ask exactly ONE question that focuses ONLY on the chosen topic. Do not mix multiple topics in one question.

Generate the NEXT most important question now by choosing ONE topic from information_gaps that
is not in redundant_categories or avoid_topics."""

        try:
            response = await self._client.chat(
                model=self._settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,  # Reduced to enforce shorter, single-topic questions
                temperature=0.2,
            )
            response_text = response.choices[0].message.content.strip()
            question = self._postprocess_question_text(response_text)

            # Optional semantic similarity guard for avoid_similar_to (normal mode only)
            if avoid_similar_to and self._too_similar(question, avoid_similar_to):
                logger.warning(
                    "Agent 3: Generated question is too similar to avoid_similar_to; "
                    "falling back to a generic clarification question."
                )
                if lang == "sp":
                    question = "¿Puede contarme un poco más sobre cómo se ha presentado este problema para usted?"
                else:
                    question = "Can you tell me a bit more detail about how this problem has been occurring for you?"

            logger.info(f"Generated question (cleaned): {question}")
            return question

        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            if lang == "sp":
                return "¿Puede contarme más sobre cómo se siente?"
            return "Can you tell me more about how you're feeling?"

    def _too_similar(self, a: Optional[str], b: Optional[str]) -> bool:
        """Heuristic check for semantic similarity between two questions."""
        if not a or not b:
            return False

        a_norm = a.lower().strip(" ?!.;,")
        b_norm = b.lower().strip(" ?!.;,")
        if not a_norm or not b_norm:
            return False

        a_tokens = set(a_norm.split())
        b_tokens = set(b_norm.split())
        if not a_tokens or not b_tokens:
            return False

        overlap = len(a_tokens & b_tokens)
        min_len = min(len(a_tokens), len(b_tokens))
        return overlap >= max(3, int(0.6 * min_len))

    def _postprocess_question_text(self, question: str) -> str:
        """Normalize and clean up question text."""
        if not question:
            return question

        q = question.strip()

        # Strip surrounding quotes
        if (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'")):
            q = q[1:-1].strip()

        # Remove leading "Q:" or numbering (e.g., "1. ")
        q = re.sub(r"^(q:|\d+\.)\s*", "", q, flags=re.IGNORECASE)

        # Normalize whitespace
        q = q.replace("\n", " ")
        q = " ".join(q.split())

        # Remove trailing punctuation (we will add a single "?")
        q = q.rstrip("?.!,;:")

        # Soft word limit (keep questions reasonably short)
        words = q.split()
        if len(words) > 30:
            q = " ".join(words[:30]).rstrip("?.!,;:")

        if not q.endswith("?"):
            q = q + "?"

        while "??" in q:
            q = q.replace("??", "?")

        return q

    def _format_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> str:
        formatted = []
        for i, qa in enumerate(qa_pairs, 1):
            formatted.append(f"{i}. Q: {qa['question']}\n   A: {qa['answer']}")
        return "\n\n".join(formatted)

    def _normalize_language(self, language: str) -> str:
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
        language: str = "en",
        recently_travelled: bool = False,
    ) -> ValidationResult:

        issues: List[str] = []

        condition_props = getattr(medical_context, "condition_properties", {}) or {}
        is_chronic = bool(condition_props.get("is_chronic", False))
        is_hereditary = bool(condition_props.get("is_hereditary", False))
        is_allergy_related = bool(condition_props.get("is_allergy_related", False))
        is_travel_related = bool(condition_props.get("is_travel_related", False))
        chief_complaint_lower = (getattr(medical_context, "chief_complaint", "") or "").lower()

        if not question or len(question.strip()) < 10:
            issues.append("Question too short")

        if not question.endswith("?"):
            question = question.rstrip(".") + "?"

        # Gender / age safety checks (menstrual questions)
        if medical_context.patient_gender:
            gender_lower = medical_context.patient_gender.lower()
            question_lower = question.lower()
            if gender_lower in ["male", "m", "masculino", "hombre"]:
                if any(k in question_lower for k in MENSTRUAL_KEYWORDS):
                    issues.append("Menstrual question asked to male patient (CRITICAL VIOLATION)")

        if medical_context.patient_age is not None:
            age = medical_context.patient_age
            question_lower = question.lower()
            if age < 12 or age > 60:
                if any(k in question_lower for k in MENSTRUAL_KEYWORDS):
                    issues.append(f"Menstrual question asked to age {age} (CRITICAL VIOLATION)")

        # Travel safety: only ask travel-related questions if condition is travel-related AND patient recently traveled
        question_lower = question.lower()
        if (not recently_travelled or not is_travel_related) and any(
            kw in question_lower for kw in TRAVEL_KEYWORDS
        ):
            issues.append(
                "Travel-related question asked for non-travel condition or patient not recently traveled (CRITICAL VIOLATION)"
            )

        # Family history safety: warn if asked for non-chronic/non-hereditary conditions
        family_history_keywords = [
            "family history",
            "family medical history",
            "historial familiar",
            "antecedentes familiares",
        ]
        if (not is_chronic and not is_hereditary) and any(
            kw in question_lower for kw in family_history_keywords
        ):
            issues.append(
                "Family history asked for non-chronic/non-hereditary condition (safety warning)"
            )

        # PMH / allergies safety: warn if not clinically indicated
        is_high_risk_complaint = any(
            kw in chief_complaint_lower for kw in HIGH_RISK_COMPLAINT_KEYWORDS
        )
        pmh_allergy_keywords = [
            "past medical history",
            "medical history",
            "conditions you have had",
            "allergies",
            "drug allergies",
            "alergias",
            "antecedentes médicos",
            "antecedentes de salud",
        ]
        should_ask_pmh_or_allergies = is_chronic or is_allergy_related or is_high_risk_complaint
        if not should_ask_pmh_or_allergies and any(
            kw in question_lower for kw in pmh_allergy_keywords
        ):
            issues.append(
                "Past medical history / allergy question asked when not clinically indicated (safety warning)"
            )

        # Duplicate / semantic similarity checks
        question_normalized = question.lower().strip().rstrip("?.")
        semantic_patterns = {
            "symptom_description": [
                "describe", "symptoms", "feeling", "experiencing", "concern",
                "síntomas", "sentir",
            ],
            "duration": [
                "how long", "when", "started", "began", "duration",
                "tiempo", "cuánto tiempo", "desde cuándo",
            ],
            "medications": [
                "medication", "medicine", "drug", "taking", "treatment",
                "medicamentos", "medicina", "tratamiento",
            ],
            "other_symptoms": [
                "other symptoms", "additional", "associated", "along with",
                "otros síntomas", "además de",
            ],
            "progression": [
                "changed", "progressed", "worse", "better", "over time", "past week",
                "empeorado", "mejorado",
            ],
            "daily_impact": [
                "daily", "activities", "life", "affected", "impact",
                "día a día", "actividades", "vida diaria",
            ],
        }

        def get_question_intent(q: str) -> set:
            q_lower = q.lower()
            intents = set()
            for intent, keywords in semantic_patterns.items():
                if sum(1 for kw in keywords if kw in q_lower) >= 2:
                    intents.add(intent)
            return intents

        current_intent = get_question_intent(question_normalized)

        for prev_q in asked_questions:
            prev_normalized = prev_q.lower().strip().rstrip("?.")
            if question_normalized == prev_normalized:
                issues.append(f"Question is exact duplicate of: '{prev_q}' (CRITICAL VIOLATION)")
                continue

            similarity_score = self._calculate_similarity(question_normalized, prev_normalized)

            prev_intent = get_question_intent(prev_normalized)

            # High similarity tier – near-exact duplicate
            if similarity_score >= 0.9:
                issues.append(
                    f"Question is near-exact duplicate of: '{prev_q}' "
                    f"(similarity: {similarity_score:.2f}, CRITICAL VIOLATION)"
                )
                continue

            # Very similar AND same intent → treat as critical duplicate
            if similarity_score >= 0.8 and current_intent and prev_intent and current_intent == prev_intent:
                issues.append(
                    f"Question is very similar with same intent as: '{prev_q}' "
                    f"(similarity: {similarity_score:.2f}, intent: {current_intent}, CRITICAL VIOLATION)"
                )
                continue

            # Medium similarity tier – strong warning (different intent)
            if 0.8 <= similarity_score < 0.9:
                issues.append(
                    f"Question is very similar to: '{prev_q}' (similarity: {similarity_score:.2f})"
                )
                continue

            if current_intent and prev_intent and current_intent == prev_intent:
                issues.append(f"Question has same semantic intent as: '{prev_q}' (intent: {current_intent})")
                continue

            if similarity_score > 0.65:
                core_medical_terms = [
                    "symptom", "síntoma", "medication", "medicamento",
                    "duration", "duración", "pain", "dolor",
                    "condition", "condición", "treatment", "tratamiento",
                ]
                q_terms = [t for t in core_medical_terms if t in question_normalized]
                prev_terms = [t for t in core_medical_terms if t in prev_normalized]
                if q_terms and prev_terms and set(q_terms) == set(prev_terms):
                    issues.append(f"Question semantically similar to: '{prev_q}' (similarity: {similarity_score:.2f})")

        if "\n" in question:
            question = question.replace("\n", " ")
            issues.append("Question contained newlines (auto-corrected)")

        # Question complexity / length checks (non-critical for now)
        word_count = len(question.split())
        if word_count > 30:
            issues.append("Question too long / possibly multi-part")
        if question.count(" and ") >= 2 or question.count(" o ") >= 2:
            issues.append("Question appears multi-part (more than one 'and'/'o')")

        # Optional: core_symptom_phrase overuse warning
        core_symptom = getattr(medical_context, "core_symptom_phrase", None)
        if core_symptom:
            core_lower = core_symptom.lower().strip()
            if core_lower:
                prev_core_uses = sum(
                    1 for q in asked_questions if core_lower in (q or "").lower()
                )
                if prev_core_uses >= 2 and core_lower in question.lower():
                    issues.append(
                        f"Core symptom phrase '{core_symptom}' over-used across questions"
                    )

        critical_violations = [i for i in issues if "CRITICAL VIOLATION" in i]
        if critical_violations:
            logger.error(f"Critical safety violation detected: {critical_violations}")
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
        if not str1 or not str2:
            return 0.0
        words1 = {w for w in str1.split() if w and w not in SIMILARITY_STOPWORDS}
        words2 = {w for w in str2.split() if w and w not in SIMILARITY_STOPWORDS}
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

    def _normalize_language(self, language: str) -> str:
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
    """

    def __init__(self) -> None:
        try:
            from dotenv import load_dotenv  # noqa: F401
        except Exception:
            load_dotenv = None

        self._settings = get_settings()
        self._debug_prompts = getattr(self._settings, "debug_prompts", False)

        azure_openai_configured = (
            self._settings.azure_openai.endpoint and
            self._settings.azure_openai.api_key
        )
        if not azure_openai_configured:
            raise ValueError(
                "Azure OpenAI is required. Please configure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
            )
        if not self._settings.azure_openai.deployment_name:
            raise ValueError(
                "Azure OpenAI deployment name is required. Please set AZURE_OPENAI_DEPLOYMENT_NAME."
            )

        self._client = get_ai_client()
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
        try:
            if self._debug_prompts:
                logger.debug("[QuestionService] Sending messages:\n%s", messages)

            resp = await self._client.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                model=self._settings.openai.model,
            )
            output = resp.choices[0].message.content.strip()

            if self._debug_prompts:
                logger.debug("[QuestionService] Response: %s", output)

            return output
        except Exception:
            logger.error("[QuestionService] OpenAI call failed", exc_info=True)
            return ""

    def _normalize_language(self, language: str) -> str:
        if not language:
            return "en"
        normalized = language.lower().strip()
        if normalized in ['es', 'sp']:
            return 'sp'
        return normalized if normalized in ['en', 'sp'] else 'en'

    async def generate_first_question(self, disease: str, language: str = "en") -> str:
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
        max_count: int = 10,
        asked_categories: Optional[List[str]] = None,
        recently_travelled: bool = False,
        travel_questions_count: int = 0,
        prior_summary: Optional[Any] = None,
        prior_qas: Optional[List[str]] = None,
        patient_gender: Optional[str] = None,
        patient_age: Optional[int] = None,
        language: str = "en",
    ) -> str:

        try:
            # Clamp max_count to settings limit
            settings = self._settings
            max_count = max(1, min(settings.intake.max_questions, max_count))
            lang = self._normalize_language(language)

            # Q8: explicit consent question for deep diagnostic
            # current_count is zero-based, so current_count == 7 means we are about to ask Question 8
            if current_count == 7:
                logger.info("Q8 reached - generating diagnostic consent question")
                if lang == "sp":
                    return "¿Le gustaría responder algunas preguntas diagnósticas detalladas relacionadas con sus síntomas?"
                return "Would you like to answer some detailed diagnostic questions related to your symptoms?"

            # ------------------------------------------------------------------
            # Agent 1: medical context (condition-level reasoning)
            # ------------------------------------------------------------------
            logger.info(f"Agent 1: Analyzing medical context for '{disease}'")
            medical_context = await self._context_analyzer.analyze_condition(
                chief_complaint=disease,
                patient_age=patient_age,
                patient_gender=patient_gender,
                recently_travelled=recently_travelled,
                language=language,
            )

            # ------------------------------------------------------------------
            # Agent 2: what has already been covered so far
            # ------------------------------------------------------------------
            logger.info(f"Agent 2: Extracting coverage from {len(asked_questions)} questions")
            extracted_info = await self._answer_extractor.extract_covered_information(
                asked_questions=asked_questions,
                previous_answers=previous_answers,
                medical_context=medical_context,
                language=language,
            )

            # Basic flags from context
            condition_props = medical_context.condition_properties or {}
            is_chronic_flag = condition_props.get("is_chronic", None)  # None = unknown
            is_chronic = bool(is_chronic_flag) if is_chronic_flag is not None else False
            is_pain_related = bool(condition_props.get("is_pain_related", False))

            # ------------------------------------------------------------------
            # 1) UPGRADE CHRONIC FLAG BASED ON DURATION / DISEASE NAME (FALLBACK ONLY)
            #    - Only when Agent-01 did not specify is_chronic
            # ------------------------------------------------------------------
            if is_chronic_flag is None:
                try:
                    duration_text = (extracted_info.extracted_facts or {}).get("duration") or ""
                except Exception:
                    duration_text = ""
                duration_lower = duration_text.lower()

                months_chronic = False
                months_match = re.search(r"(\d+)\s*(month|months|mo)", duration_lower)
                if months_match:
                    try:
                        months_value = int(months_match.group(1))
                        if months_value >= 3:
                            months_chronic = True
                    except Exception:
                        months_chronic = False

                years_match = re.search(r"(\d+)\s*(year|years|yr|yrs)", duration_lower)
                has_years = bool(years_match)

                if any(
                    phrase in duration_lower
                    for phrase in ["many months", "long time", "since childhood", "since i was a child"]
                ):
                    months_chronic = True

                if has_years or months_chronic:
                    is_chronic = True
                    condition_props["is_chronic"] = True
                    medical_context.condition_properties = condition_props

                # Also upgrade chronic flag from disease name if it clearly indicates a chronic disease
                disease_lower = (disease or "").lower()
                chronic_name_keywords = [
                    "diabetes",
                    "hypertension",
                    "high blood pressure",
                    "asthma",
                    "copd",
                    "chronic",
                    "heart failure",
                    "ckd",
                    "chronic kidney",
                    "thyroid",
                ]
                if any(kw in disease_lower for kw in chronic_name_keywords):
                    is_chronic = True
                    condition_props["is_chronic"] = True
                    medical_context.condition_properties = condition_props

            # ------------------------------------------------------------------
            # 2) REMOVE DEEP-DIAGNOSTIC-ONLY TOPICS FROM NORMAL FLOW (BEFORE CONSENT)
            #    - monitoring / labs / screening topics
            #    - past_evaluation (for non-chronic deep diagnostic domain)
            # ------------------------------------------------------------------
            if current_count < 7:
                info_gaps = list(extracted_info.information_gaps or [])
                filtered_gaps: List[str] = []
                for gap in info_gaps:
                    gl = gap.lower()
                    if any(
                        k in gl
                        for k in [
                            "monitor",
                            "lab",
                            "screen",
                            "test",
                            "exam",
                            "complication",
                            "hba1c",
                            "cholesterol",
                            "kidney",
                            "glucose",
                            "blood sugar",
                            "eye exam",
                            "foot exam",
                        ]
                    ) or gl == "past_evaluation":
                        logger.debug(f"Stripping deep diagnostic topic before consent: {gap}")
                        continue
                    filtered_gaps.append(gap)
                extracted_info.information_gaps = filtered_gaps

            # ------------------------------------------------------------------
            # 3) ENFORCE DURATION & MEDICATION QUESTIONS EARLY
            #    + pain-location/radiation early if pain-related
            # ------------------------------------------------------------------
            try:
                info_gaps = list(extracted_info.information_gaps or [])
                redundant = list(extracted_info.redundant_categories or [])

                # Duration once and early
                if not extracted_info.already_mentioned_duration:
                    info_gaps = [g for g in info_gaps if g != "duration"]
                    info_gaps.insert(0, "duration")

                # Medications early, just after duration if present
                if not extracted_info.already_mentioned_medications:
                    info_gaps = [g for g in info_gaps if g != "current_medications"]
                    insert_idx = 1 if "duration" in info_gaps else 0
                    info_gaps.insert(insert_idx, "current_medications")

                # Location & radiation – only when pain-related
                if is_pain_related:
                    info_gaps = [g for g in info_gaps if g != "location_radiation"]
                    insert_idx = 0
                    if "duration" in info_gaps:
                        insert_idx = max(insert_idx, info_gaps.index("duration") + 1)
                    if "current_medications" in info_gaps:
                        insert_idx = max(insert_idx, info_gaps.index("current_medications") + 1)
                    info_gaps.insert(insert_idx, "location_radiation")

                extracted_info.information_gaps = info_gaps
                extracted_info.redundant_categories = redundant
            except Exception as e:
                logger.warning(f"Failed to enforce duration/medication/location_radiation ordering: {e}")

            # ------------------------------------------------------------------
            # 4) REMOVE PAIN TOPICS IF CONDITION IS NOT PAIN-RELATED
            # ------------------------------------------------------------------
            if not is_pain_related:
                pain_topics = ["pain_assessment", "pain_characterization", "location_radiation"]
                info_gaps = list(extracted_info.information_gaps or [])
                redundant = list(extracted_info.redundant_categories or [])
                for p in pain_topics:
                    if p in info_gaps:
                        info_gaps.remove(p)
                    if p not in redundant:
                        redundant.append(p)
                extracted_info.information_gaps = info_gaps
                extracted_info.redundant_categories = redundant

            # ------------------------------------------------------------------
            # 5) TRAVEL HISTORY: Only ask if symptoms are travel-related + cap after first question
            #    (enforced in medical context analysis, but double-check here)
            # ------------------------------------------------------------------
            condition_props = medical_context.condition_properties or {}
            is_travel_related = bool(condition_props.get("is_travel_related", False))

            # Use persisted travel_questions_count instead of recalculating
            if not recently_travelled or not is_travel_related or travel_questions_count >= 1:
                info_gaps = list(extracted_info.information_gaps or [])
                redundant = list(extracted_info.redundant_categories or [])
                filtered_travel = False
                for key in ["travel_history", "historial_viajes"]:
                    if key in info_gaps:
                        info_gaps.remove(key)
                        filtered_travel = True
                    if key not in redundant:
                        redundant.append(key)
                extracted_info.information_gaps = info_gaps
                extracted_info.redundant_categories = redundant
                # Also make sure it is in avoid_topics for the generator prompt
                if "travel_history" not in medical_context.avoid_topics:
                    medical_context.avoid_topics.append("travel_history")
                
                # Log the reason for filtering
                if filtered_travel:
                    if not recently_travelled:
                        logger.info(
                            f"Filtered travel_history from information_gaps: "
                            f"recently_travelled=False, chief_complaint='{disease}'"
                        )
                    elif not is_travel_related:
                        logger.info(
                            f"Filtered travel_history from information_gaps: "
                            f"is_travel_related=False, chief_complaint='{disease}'"
                        )
                    elif travel_questions_count >= 1:
                        logger.info(
                            f"Filtered travel_history from information_gaps: "
                            f"travel_questions_count={travel_questions_count} >= 1 (travel cap enforced)"
                        )

            # ------------------------------------------------------------------
            # 5b) EXPLICIT FILTERING: Remove any topics in avoid_topics from information_gaps
            #    This is a defensive check to ensure conditional rules are never violated
            # ------------------------------------------------------------------
            info_gaps_before = list(extracted_info.information_gaps or [])
            avoid_topics = list(medical_context.avoid_topics or [])
            filtered_out = []
            
            for topic in avoid_topics:
                if topic in info_gaps_before:
                    info_gaps_before.remove(topic)
                    filtered_out.append(topic)
            
            if filtered_out:
                logger.info(
                    f"Information gaps BEFORE filtering avoid_topics: {extracted_info.information_gaps}. "
                    f"Filtered out topics: {filtered_out}. "
                    f"Information gaps AFTER filtering: {info_gaps_before}"
                )
            
            extracted_info.information_gaps = info_gaps_before

            # ------------------------------------------------------------------
            # 6) EVALUATE DIAGNOSTIC CONSENT STATE (ROBUST, NOT ONLY LAST Q)
            # ------------------------------------------------------------------
            diagnostic_consent_patterns = [
                "would you like to answer some detailed diagnostic questions",
                "detailed diagnostic questions related to your symptoms",
                "le gustaría responder algunas preguntas diagnósticas detalladas",
                "preguntas diagnósticas detalladas relacionadas con sus síntomas",
            ]

            consent_index: Optional[int] = None
            has_positive_consent = False
            has_negative_consent = False

            if asked_questions and previous_answers:
                # Use the *last* matching consent question if multiple exist
                for idx, q in enumerate(asked_questions):
                    q_low = q.lower()
                    if any(p in q_low for p in diagnostic_consent_patterns):
                        consent_index = idx

                if consent_index is not None and consent_index < len(previous_answers):
                    consent_answer = previous_answers[consent_index].lower().strip()
                    positive_kw = [
                        "yes",
                        "y",
                        "yeah",
                        "yep",
                        "sure",
                        "ok",
                        "okay",
                        "of course",
                        "absolutely",
                        "sí",
                        "si",
                        "claro",
                        "por supuesto",
                    ]
                    negative_kw = [
                        "no",
                        "n",
                        "nope",
                        "nah",
                        "not really",
                        "no gracias",
                        "no quiero",
                    ]
                    if any(k in consent_answer for k in positive_kw):
                        has_positive_consent = True
                    elif any(k in consent_answer for k in negative_kw):
                        has_negative_consent = True

            # If patient clearly declined detailed questions → go straight to closing question
            if has_negative_consent and not has_positive_consent:
                logger.info("Patient declined detailed diagnostic questions, returning closing question.")
                if lang == "sp":
                    return "¿Hay algo más que le gustaría compartir sobre su condición?"
                return "Is there anything else you'd like to share about your condition?"

            # If patient accepted → allow up to 3 deep diagnostic questions
            deep_questions_asked = 0
            is_deep_diagnostic_mode = False
            deep_diagnostic_question_num: Optional[int] = None

            if has_positive_consent and consent_index is not None:
                # Ensure there is room in max_count for 3 deep questions
                # Use settings max_questions (default 12) as the limit
                settings = self._settings
                if max_count < settings.intake.max_questions:
                    max_count = settings.intake.max_questions

                # How many questions have already been asked AFTER consent?
                deep_questions_asked = max(0, len(asked_questions) - (consent_index + 1))

                if deep_questions_asked >= 3:
                    # Already asked all 3 (monitoring, labs, screening OR triggers, impact, past evaluation)
                    logger.info("Completed 3 deep diagnostic questions - closing.")
                    if lang == "sp":
                        return "¿Hay algo más que le gustaría compartir sobre su condición?"
                    return "Is there anything else you'd like to share about your condition?"
                else:
                    # We are inside the 3-question deep diagnostic window
                    is_deep_diagnostic_mode = True
                    deep_diagnostic_question_num = deep_questions_asked + 1

            # ------------------------------------------------------------------
            # 7) NORMAL MAX-COUNT CLOSE (ONLY WHEN NOT IN DEEP-DIAGNOSTIC MODE)
            #     - Adjust effective max_count using triage_level (emergency/urgent/routine)
            # ------------------------------------------------------------------

            # Adjust max_count based on triage_level from medical context
            triage_level = medical_context.triage_level or "routine"
            if triage_level == "emergency":
                max_count = min(max_count, 6)
            elif triage_level == "urgent":
                max_count = min(max_count, 8)

            if not is_deep_diagnostic_mode and current_count >= max_count - 1:
                logger.info("Reached maximum question count - closing.")
                if lang == "sp":
                    return "¿Hay algo más que le gustaría compartir sobre su condición?"
                return "Is there anything else you'd like to share about your condition?"

            # ------------------------------------------------------------------
            # 8) Agent 3: generate question (deep-diagnostic or normal)
            # ------------------------------------------------------------------
            logger.info(
                f"Agent 3: Generating question {current_count + 1}/{max_count} "
                f"(deep_diagnostic={is_deep_diagnostic_mode}, deep_num={deep_diagnostic_question_num})"
            )
            last_question = asked_questions[-1] if asked_questions else None
            question = await self._question_generator.generate_question(
                medical_context=medical_context,
                extracted_info=extracted_info,
                current_count=current_count,
                max_count=max_count,
                language=language,
                avoid_similar_to=last_question,
                asked_questions=asked_questions,
                previous_answers=previous_answers,
                is_deep_diagnostic=is_deep_diagnostic_mode,
                deep_diagnostic_question_num=deep_diagnostic_question_num,
            )

            # ------------------------------------------------------------------
            # 9) Agent 4: safety validation
            # ------------------------------------------------------------------
            validation = await self._safety_validator.validate_question(
                question=question,
                medical_context=medical_context,
                asked_questions=asked_questions,
                language=language,
                recently_travelled=recently_travelled,
            )

            final_question = validation.corrected_question or question
            # Enhanced logging: log final question with context
            logger.info(
                f"Multi-agent pipeline completed: Question {current_count+1}/{max_count} = '{final_question}'. "
                f"Condition: {disease}, Issues: {validation.issues}, "
                f"Remaining gaps: {extracted_info.information_gaps}"
            )
            return final_question

        except Exception as e:
            logger.error(f"Multi-agent question generation failed: {e}", exc_info=True)
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
        return current_count >= max_count

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
        try:
            if max_count <= 0:
                return 0
            return int(min(max(current_count / max_count, 0.0), 1.0) * 100)
        except Exception:
            return 0

    async def is_medication_question(self, question: str) -> bool:
        """Check if question is about medications (for image upload)."""
        q = (question or "").lower()
        if ("home remedies" in q and "medications" in q) or \
           ("self-care" in q and "medications" in q) or \
           ("remedios caseros" in q and "medicamentos" in q):
            return True

        medication_terms = [
            "medication", "medications", "medicines", "medicine", "drug", "drugs",
            "prescription", "prescribed", "tablet", "tablets", "capsule", "capsules",
            "syrup", "dose", "dosage", "frequency", "supplement", "supplements",
            "insulin", "otc", "over-the-counter", "medicamento", "medicamentos",
            "medicina", "medicinas", "suplemento", "suplementos"
        ]
        return any(term in q for term in medication_terms)

    # ========================================================================
    # PRE-VISIT SUMMARY & RED-FLAG METHODS ARE INTENTIONALLY EXCLUDED HERE
    # ========================================================================


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
                "Tu tarea es generar un Resumen Pre-Consulta conciso y clínicamente útil (~180-200 palabras) "
                "basado estrictamente en las respuestas de admisión proporcionadas.\n\n"
                "Reglas Críticas\n"
                "- No inventes, adivines o expandas más allá de la entrada proporcionada.\n"
                "- La salida debe ser texto plano con encabezados de sección, una sección por línea "
                "(sin líneas en blanco adicionales).\n"
                "- Usa solo los encabezados exactos listados a continuación. No agregues, renombres o "
                "reordenes encabezados.\n"
                "- Sin viñetas, numeración o formato markdown.\n"
                "- Escribe en un tono de entrega clínica: corto, factual, sin duplicados y neutral.\n"
                "- Incluye una sección SOLO si contiene contenido real de las respuestas del paciente.\n"
                "- No uses marcadores de posición como \"N/A\", \"No proporcionado\", \"no reportado\", o \"niega\".\n"
                "- No incluyas secciones para temas que no fueron preguntados o discutidos.\n"
                "- Usa frases orientadas al paciente: \"El paciente reporta...\", \"Niega...\", \"En medicamentos:...\".\n"
                "- No incluyas observaciones clínicas, diagnósticos, planes, signos vitales o hallazgos del examen "
                "(la pre-consulta es solo lo reportado por el paciente).\n"
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
                "  Inicio, Localización, Duración, Caracterización/calidad, Factores agravantes, Factores aliviadores, "
                "Radiación,\n"
                "  Patrón temporal, Severidad (1-10), Síntomas asociados, Negativos relevantes.\n"
                "  Manténlo natural y coherente (ej., \"El paciente reporta...\"). Si algunos elementos OLDCARTS son "
                "desconocidos, simplemente omítelos.\n"
                "- Historia: Una línea combinando cualquier elemento reportado por el paciente usando punto y coma en este "
                "orden si está presente:\n"
                "  Médica: ...; Quirúrgica: ...; Familiar: ...; Estilo de vida: ...\n"
                "  (Incluye SOLO las partes que fueron realmente preguntadas y respondidas por el paciente. Si un tema no fue "
                "discutido, no lo incluyas en absoluto).\n"
                "- Revisión de Sistemas: Una línea narrativa resumiendo positivos/negativos basados en sistemas mencionados "
                "explícitamente por el paciente. Mantén como prosa, no como lista. Solo incluye si los sistemas fueron "
                "realmente revisados.\n"
                "- Medicación Actual: Una línea narrativa con medicamentos/suplementos realmente declarados por el paciente "
                "(nombre/dosis/frecuencia si se proporciona). Incluye declaraciones de alergia solo si el paciente las "
                "reportó explícitamente. Si el paciente subió imágenes de medicamentos (incluso si no mencionó "
                "explícitamente los nombres), menciona esto: \"El paciente proporcionó imágenes de medicamentos: "
                "[nombre(s) de archivo(s)]\". Incluye esta sección si los medicamentos fueron discutidos O si se subieron "
                "imágenes de medicamentos.\n\n"
                "Ejemplo de Formato\n"
                "(Estructura y tono solamente—el contenido será diferente; cada sección en una sola línea.)\n"
                "Motivo de Consulta: El paciente reporta dolor de cabeza severo por 3 días.\n"
                "HPI: El paciente describe una semana de dolores de cabeza persistentes que comienzan en la mañana y empeoran "
                "durante el día, llegando hasta 8/10 en los últimos 3 días. El dolor es sobre ambas sienes y se siente "
                "diferente de migrañas previas; la fatiga es prominente y se niega náusea. Los episodios se agravan por "
                "estrés y más tarde en el día, con alivio mínimo de analgésicos de venta libre y algo de alivio usando "
                "compresas frías.\n"
                "Historia: Médica: hipertensión; Quirúrgica: colecistectomía hace cinco años; Estilo de vida: no fumador, "
                "alcohol ocasional, trabajo de alto estrés.\n"
                "Medicación Actual: En medicamentos: lisinopril 10 mg diario e ibuprofeno según necesidad; alergias incluidas "
                "solo si el paciente las declaró explícitamente.\n\n"
                f"{f'Imágenes de Medicamentos: {medication_images_info}' if medication_images_info else ''}\n\n"
                f"Respuestas de Admisión:\n{self._format_intake_answers(intake_answers)}"
            )
        else:
            prompt = (
                "Role & Task\n"
                "You are a Clinical Intake Assistant.\n"
                "Your task is to generate a concise, clinically useful Pre-Visit Summary (~180–200 words) based strictly on "
                "the provided intake responses.\n\n"
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
                "- Do not include clinician observations, diagnoses, plans, vitals, or exam findings "
                "(previsit is patient-reported only).\n"
                "- Normalize obvious medical mispronunciations to correct terms (e.g., \"diabities\" -> \"diabetes\") "
                "without adding new information.\n\n"
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
                "  Keep it natural and coherent (e.g., \"The patient reports …\"). If some OLDCARTS elements are unknown, "
                "simply omit them (do not write placeholders).\n"
                "- History: One line combining any patient-reported items using semicolons in this order if present:\n"
                "  Medical: …; Surgical: …; Family: …; Lifestyle: …\n"
                "  (Include ONLY parts that were actually asked about and answered by the patient. If a topic was not "
                "discussed, do not include it at all.)\n"
                "- Review of Systems: One narrative line summarizing system-based positives/negatives explicitly mentioned by "
                "the patient (e.g., General, Neuro, Eyes, Resp, GI). Keep as prose, not a list. Only include if systems were "
                "actually reviewed.\n"
                "- Current Medication: One narrative line with meds/supplements actually stated by the patient "
                "(name/dose/frequency if provided). Include allergy statements only if the patient explicitly reported them. "
                "If the patient uploaded medication images (even if they didn't explicitly name the medications), mention "
                "this: \"Patient provided medication images: [filename(s)]\". Include this section if medications were "
                "discussed OR if medication images were uploaded.\n\n"
                "Example Format\n"
                "(Structure and tone only—content will differ; each section on a single line.)\n"
                "Chief Complaint: Patient reports severe headache for 3 days.\n"
                "HPI: The patient describes a week of persistent headaches that begin in the morning and worsen through the "
                "day, reaching up to 8/10 over the last 3 days. Pain is over both temples and feels different from prior "
                "migraines; fatigue is prominent and nausea is denied. Episodes are aggravated by stress and later in the "
                "day, with minimal relief from over-the-counter analgesics and some relief using cold compresses. No "
                "radiation is reported, evenings are typically worse, and there have been no recent changes in medications "
                "or lifestyle.\n"
                "History: Medical: hypertension; Surgical: cholecystectomy five years ago; Lifestyle: non-smoker, occasional "
                "alcohol, high-stress job.\n"
                "Current Medication: On meds: lisinopril 10 mg daily and ibuprofen as needed; allergies included only if the "
                "patient explicitly stated them.\n\n"
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

    async def _detect_red_flags(
        self,
        intake_answers: Dict[str, Any],
        language: str = "en",
    ) -> List[Dict[str, str]]:
        """Hybrid abusive language detection: hardcoded rules + LLM analysis."""
        lang = self._normalize_language(language)

        red_flags: List[Dict[str, str]] = []

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

    def _detect_obvious_abusive_language(
        self,
        questions_asked: List[Dict[str, Any]],
        language: str = "en",
    ) -> List[Dict[str, str]]:
        """Fast hardcoded detection for obvious abusive language."""
        lang = self._normalize_language(language)
        red_flags: List[Dict[str, str]] = []

        for qa in questions_asked:
            answer = qa.get("answer", "").strip()
            question = qa.get("question", "").strip()

            if not answer or answer.lower() in [
                "",
                "n/a",
                "not provided",
                "unknown",
                "don't know",
                "dont know",
                "no se",
                "no proporcionado",
            ]:
                continue

            # Check for obvious abusive language
            if self._contains_abusive_language(answer, lang):
                red_flags.append(
                    {
                        "type": "abusive_language",
                        "question": question,
                        "answer": answer,
                        "message": self._get_abusive_language_message(lang),
                        "detection_method": "hardcoded",
                    }
                )

        return red_flags

    async def _detect_subtle_abusive_language_with_llm(
        self,
        questions_asked: List[Dict[str, Any]],
        language: str = "en",
    ) -> List[Dict[str, str]]:
        """Use LLM to detect subtle, contextual, or creative abusive language."""
        try:
            lang = self._normalize_language(language)

            # Filter out obvious cases already detected
            subtle_cases: List[Dict[str, Any]] = []
            for qa in questions_asked:
                answer = qa.get("answer", "").strip()
                question = qa.get("question", "").strip()

                if (
                    answer
                    and answer.lower()
                    not in [
                        "",
                        "n/a",
                        "not provided",
                        "unknown",
                        "don't know",
                        "dont know",
                        "no se",
                        "no proporcionado",
                    ]
                    and not self._contains_abusive_language(answer, lang)
                ):
                    subtle_cases.append(qa)

            if not subtle_cases:
                return []

            # Use LLM to analyze subtle cases
            return await self._analyze_abusive_language_with_llm(subtle_cases, lang)

        except Exception as e:
            logger.warning(f"LLM abusive language analysis failed: {e}")
            return []

    async def _analyze_abusive_language_with_llm(
        self,
        questions_asked: List[Dict[str, Any]],
        language: str = "en",
    ) -> List[Dict[str, str]]:
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
                        "content": (
                            "You are a clinical assistant analyzing patient responses for abusive language. "
                            "Be precise and only flag truly inappropriate or abusive language."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.1,
            )

            # Parse LLM response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                abusive_cases = result.get("abusive_language", [])

                formatted_flags: List[Dict[str, str]] = []
                for case in abusive_cases:
                    formatted_flags.append(
                        {
                            "type": "abusive_language",
                            "question": case.get("question", "") or "",
                            "answer": case.get("answer", "") or "",
                            "message": self._get_llm_abusive_language_message(
                                case.get("reason", "") or "",
                                lang,
                            ),
                            "detection_method": "llm",
                        }
                    )

                return formatted_flags

        except Exception as e:
            logger.warning(f"Failed to parse LLM abusive language response: {e}")

        return []

    def _format_qa_pairs(self, questions_asked: List[Dict[str, Any]]) -> str:
        """Format question-answer pairs for LLM analysis."""
        formatted: List[str] = []
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

        english_abusive = [
            "fuck",
            "shit",
            "damn",
            "hell",
            "bitch",
            "asshole",
            "bastard",
            "crap",
            "stupid",
            "idiot",
            "moron",
            "retard",
            "gay",
            "fag",
            "nigger",
            "whore",
            "slut",
            "cunt",
            "piss",
            "pissed",
            "fucking",
            "bullshit",
            "goddamn",
        ]

        spanish_abusive = [
            "puta",
            "puto",
            "mierda",
            "joder",
            "coño",
            "cabrón",
            "hijo de puta",
            "estúpido",
            "idiota",
            "imbécil",
            "retrasado",
            "maricón",
            "joto",
            "pinche",
            "chingado",
            "verga",
            "pendejo",
            "culero",
            "mamón",
        ]

        abusive_words = spanish_abusive if lang == "sp" else english_abusive

        return any(word in text_lower for word in abusive_words)

    def _get_abusive_language_message(self, language: str = "en") -> str:
        """Get message for abusive language red flag."""
        lang = self._normalize_language(language)
        if lang == "sp":
            return (
                "⚠️ BANDERA ROJA: El paciente utilizó lenguaje inapropiado o abusivo en sus respuestas."
            )
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

    async def _generate_fallback_summary(
        self,
        patient_data: Dict[str, Any],
        intake_answers: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate basic fallback summary with red flag detection."""
        red_flags = await self._detect_red_flags(intake_answers, "en")

        summary = f"Pre-visit summary for {patient_data.get('name', 'Patient')}"

        return {
            "summary": summary,
            "structured_data": {
                "chief_complaint": patient_data.get("symptom")
                or patient_data.get("complaint")
                or "N/A",
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

        def _extract_from_markdown(md: str) -> Dict[str, Any]:
            data: Dict[str, Any] = {}
            current_section: Optional[str] = None
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
            if extracted.get("chief_complaint") and structured.get("chief_complaint") in (
                None,
                "See summary",
                "N/A",
            ):
                structured["chief_complaint"] = extracted["chief_complaint"]

        return {
            "summary": summary,
            "structured_data": structured,
            "red_flags": result.get("red_flags", []),
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