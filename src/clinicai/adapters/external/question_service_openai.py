import asyncio
import logging
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from pydantic import BaseModel, Field, ValidationError
from pydantic import ConfigDict # Pydantic v2
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
from clinicai.adapters.db.mongo.repositories.llm_interaction_repository import (
    append_intake_agent_log,
    append_phase_call,
)
from clinicai.adapters.db.mongo.models.patient_m import DoctorPreferencesMongo
from clinicai.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS
from clinicai.adapters.external.llm_gateway import call_llm_with_telemetry
logger = logging.getLogger("clinicai")
# =============================================================================
# SHARED UTILITIES
# =============================================================================
def _normalize_language(language: str) -> str:
    if not language:
        return "en"
    normalized = language.lower().strip()
    if normalized in ["es", "sp"]:
        return "sp"
    return normalized if normalized in ["en", "sp"] else "en"


def _format_qa_pairs(qa_pairs: List[Dict[str, str]]) -> str:
    formatted = []
    for i, qa in enumerate(qa_pairs, 1):
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        formatted.append(f"{i}. Q: {q}\n A: {a}")
    return "\n\n".join(formatted)


def _safe_str_list(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    out: List[str] = []
    for x in v:
        if isinstance(x, (str, int, float)):
            s = str(x).strip()
            if s:
                out.append(s)
    return out


def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    Robust-ish JSON extraction:
    - Prefer direct json.loads(text)
    - Else regex pull the first {...} block
    """
    if not text:
        return None
    t = text.strip()
    try:
        if t.startswith("{") and t.endswith("}"):
            return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def _clamp_topics(topics: List[str]) -> List[str]:
    allowed = set(ALLOWED_TOPICS)
    return [t for t in topics if t in allowed]


def _ensure_nonempty_topic_plan(priority_topics: List[str], topic_plan: List[str]) -> List[str]:
    """
    Guarantee topic_plan exists and is valid:
    - If empty → use priority_topics
    - Ensure topic_plan is subset of priority_topics and allowed topics
    """
    pr = _clamp_topics(priority_topics)
    tp = _clamp_topics(topic_plan)
    pr_set = set(pr)
    tp = [t for t in tp if t in pr_set]
    if not tp:
        tp = pr[:]  # fallback to priority list (still LLM-driven list)
    return tp


def _recompute_gaps_from_plan(
    medical_context: "MedicalContext",
    topics_covered: List[str],
) -> List[str]:
    """
    Authoritative gaps (code-truth):
    gaps = topic_plan - covered - avoid
    """
    plan = list(medical_context.topic_plan or medical_context.priority_topics or [])
    avoid = set(medical_context.avoid_topics or [])
    covered = set(topics_covered or [])
    gaps = [t for t in plan if t not in avoid and t not in covered]
    return gaps
# =============================================================================
# ✅ NEW: CODE-TRUTH TOPIC TRACKING + CHRONIC TAIL
# =============================================================================
def _topic_counts_from_asked_categories(asked_categories: Optional[List[str]]) -> Dict[str, int]:
    """
    Code-truth topic counts from asked_categories.
    asked_categories MUST be appended with chosen_topic at question-send time.
    """
    if not asked_categories:
        return {}
    allowed = set(ALLOWED_TOPICS)
    out: Dict[str, int] = {}
    for t in asked_categories:
        if t in allowed:
            out[t] = out.get(t, 0) + 1
    return out


_POSITIVE = {"yes", "y", "yeah", "yep", "sure", "of course", "i do", "i have", "sometimes", "often", "frequently"}
_NEGATIVE = {"no", "n", "nope", "never", "not really", "i don't", "i do not", "haven't", "have not"}


def _is_positive_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(k in t for k in _NEGATIVE):
        return False
    if any(k in t for k in _POSITIVE):
        return True
    return "yes" in t


def _is_chronic_case(medical_context: "MedicalContext") -> bool:
    props = medical_context.condition_properties or {}
    return bool(props.get("is_chronic")) or (props.get("acuity_level") == "chronic")


def _duration_implies_chronic(answers: List[str]) -> bool:
    """
    Heuristic: if any prior answer mentions duration >= 3 months, treat as chronic.
    Looks for patterns like "3 months", "three months", "over 3 months".
    """
    if not answers:
        return False
    month_patterns = [
        r"(\d+)\s*(?:month|months|mo)\b",
        r"(three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months",
    ]
    number_words = {
        "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12,
    }
    for ans in answers:
        a = (ans or "").lower()
        for pat in month_patterns:
            m = re.search(pat, a)
            if m:
                val = m.group(1)
                if val.isdigit():
                    if int(val) >= 3:
                        return True
                elif val in number_words and number_words[val] >= 3:
                    return True
        # catch generic phrasing
        if "over 3 months" in a or "more than 3 months" in a or "more than three months" in a:
            return True
    return False
# =============================================================================
# AGENT 1: Medical Context Analyzer
# =============================================================================
class ConditionPropertiesModel(BaseModel):
    """
    IMPORTANT FIX:
    - extra="ignore" so random keys like {"fever": true} DO NOT break validation.
    """
    model_config = ConfigDict(extra="ignore")
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
    model_config = ConfigDict(extra="ignore")
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
    chief_complaint: str
    condition_properties: Dict[str, Any]
    priority_topics: List[str]
    avoid_topics: List[str]
    medical_reasoning: str
    patient_age: Optional[int]
    patient_gender: Optional[str]
    severity_level: Optional[str] = None
    acuity_level: Optional[str] = None
    triage_level: Optional[str] = None
    normalized_complaint: Optional[str] = None
    red_flags: Optional[Dict[str, Any]] = None
    core_symptom_phrase: Optional[str] = None
    topic_plan: List[str] = field(default_factory=list)
    recently_travelled: bool = False


class MedicalContextAnalyzer:
    def __init__(self, client, settings):
        self._client = client
        self._settings = settings

    def _normalize_language(self, language: str) -> str:
        return _normalize_language(language)

    async def analyze_condition(
        self,
        chief_complaint: str,
        patient_age: Optional[int],
        patient_gender: Optional[str],
        recently_travelled: bool = False,
        language: str = "en",
        visit_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        question_number: Optional[int] = None,
    ) -> MedicalContext:
        lang = self._normalize_language(language)
        system_prompt = """You are AGENT-01 "MEDICAL CONTEXT ANALYZER" - Clinical Strategist.
Return ONLY one JSON object with this schema:
{
"condition_properties": {
"is_chronic": <true|false|null>,
"is_hereditary": <true|false|null>,
"has_complications": <true|false|null>,
"is_acute_emergency": <true|false|null>,
"is_pain_related": <true|false|null>,
"is_womens_health": <true|false|null>,
"is_allergy_related": <true|false|null>,
"requires_lifestyle_assessment": <true|false|null>,
"is_travel_related": <true|false|null>,
"severity_level": "<mild|moderate|severe|null>",
"acuity_level": "<acute|subacute|chronic|null>",
"is_new_problem": <true|false|null>,
"is_followup": <true|false|null>
},
"triage_level": "<routine|urgent|emergency>",
"red_flags": {
"possible_emergency": <true|false>,
"needs_urgent_attention": <true|false>,
"identified_flags": ["<string>", "..."]
},
"normalized_complaint": "<short label or null>",
"core_symptom_phrase": "<short phrase or null>",
"priority_topics": ["<topic>", "..."],
"avoid_topics": ["<topic>", "..."],
"topic_plan": ["<topic>", "..."],
"medical_reasoning": "<2-4 sentences>",
"prompt_version": "med_ctx_v4"
}
ALLOWED TOPICS (must use exactly):
duration, associated_symptoms, current_medications, past_medical_history,
triggers, travel_history, lifestyle_functional_impact, family_history,
allergies, pain_assessment, temporal, menstrual_cycle, past_evaluation,
chronic_monitoring,lab_tests, screening
Rules:
- topic_plan must be a subset of priority_topics
- if travel checkbox is NO, set is_travel_related=false and avoid_topics includes travel_history
- avoid menstrual_cycle for males or age<12 or age>60
Return ONLY JSON.
"""
        user_prompt = f"""
PATIENT:
Chief Complaint: "{chief_complaint}"
Age: {patient_age or "Unknown"}
Gender: {patient_gender or "Not specified"}
Recent Travel Checkbox: {"Yes" if recently_travelled else "No"}
Return the JSON plan now.
"""
        logger.info("Agent1: calling LLM for context plan chief_complaint='%s'", chief_complaint)
        resp = await call_llm_with_telemetry(
            ai_client=self._client,
            scenario=PromptScenario.RED_FLAG,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self._settings.openai.model,
            max_tokens=1200,
            temperature=0.1,
        )
        response_text = (resp.choices[0].message.content or "").strip()
        log_msg = (
            "[Agent1-MedicalContextAnalyzer] Raw LLM response received\n"
            f"{'=' * 80}\n{response_text}\n{'=' * 80}"
        )
        print(log_msg, flush=True)
        logger.info(log_msg)
        # Log to structured llm_interaction collection (only user_prompt, no system_prompt)
        if visit_id and patient_id and question_number is not None:
            try:
                await append_intake_agent_log(
                    visit_id=visit_id,
                    patient_id=patient_id,
                    question_number=question_number,
                    question_text=None,  # Not known yet at Agent 1 stage
                    agent_name="agent1_medical_context",
                    user_prompt=user_prompt,  # Only user_prompt, NO system_prompt
                    response_text=response_text,
                    metadata={
                        "chief_complaint": chief_complaint,
                        "prompt_version": PROMPT_VERSIONS.get(PromptScenario.INTAKE, "UNKNOWN"),
                    },
                )
            except Exception as e:
                logger.warning("Agent1: failed to persist interaction: %s", e)
        raw = _extract_first_json_object(response_text)
        if not raw:
            raise ValueError("Agent1 returned no valid JSON object")
        parsed = MedicalContextLLMResponse.model_validate(raw)
        condition_props = parsed.condition_properties.model_dump()
        triage_level = parsed.triage_level or "routine"
        if triage_level not in {"routine", "urgent", "emergency"}:
            triage_level = "routine"
        priority_topics = _clamp_topics(parsed.priority_topics or [])
        avoid_topics = _clamp_topics(parsed.avoid_topics or [])
        if not recently_travelled:
            if "travel_history" in priority_topics:
                priority_topics = [t for t in priority_topics if t != "travel_history"]
            if "travel_history" not in avoid_topics:
                avoid_topics.append("travel_history")
            condition_props["is_travel_related"] = False
        if patient_gender:
            g = patient_gender.lower().strip()
            is_male = g in ["male", "m", "masculino", "hombre"]
            if is_male:
                priority_topics = [t for t in priority_topics if t != "menstrual_cycle"]
                if "menstrual_cycle" not in avoid_topics:
                    avoid_topics.append("menstrual_cycle")
        if patient_age is not None and (patient_age < 12 or patient_age > 60):
            priority_topics = [t for t in priority_topics if t != "menstrual_cycle"]
            if "menstrual_cycle" not in avoid_topics:
                avoid_topics.append("menstrual_cycle")
        topic_plan = _ensure_nonempty_topic_plan(priority_topics, parsed.topic_plan or [])
        avoid_set = set(avoid_topics)
        topic_plan = [t for t in topic_plan if t not in avoid_set]
        if not topic_plan:
            topic_plan = [t for t in priority_topics if t not in avoid_set]
        if not priority_topics:
            raise ValueError("Agent1 returned empty priority_topics; fix prompt or model behavior")
        logger.info(
            "Agent1 parsed: priority_topics=%s avoid_topics=%s topic_plan=%s triage=%s",
            priority_topics, avoid_topics, topic_plan, triage_level
        )
        return MedicalContext(
            chief_complaint=chief_complaint,
            condition_properties=condition_props,
            priority_topics=priority_topics,
            avoid_topics=avoid_topics,
            medical_reasoning=parsed.medical_reasoning or "",
            patient_age=patient_age,
            patient_gender=patient_gender,
            severity_level=condition_props.get("severity_level"),
            acuity_level=condition_props.get("acuity_level"),
            triage_level=triage_level,
            normalized_complaint=parsed.normalized_complaint,
            red_flags=parsed.red_flags,
            core_symptom_phrase=parsed.core_symptom_phrase,
            topic_plan=topic_plan,
            recently_travelled=recently_travelled,
        )
# =============================================================================
# AGENT 2: Answer Extractor
# =============================================================================
@dataclass
class ExtractedInformation:
    topics_covered: List[str]
    information_gaps: List[str]
    extracted_facts: Dict[str, str]
    already_mentioned_duration: bool
    already_mentioned_medications: bool
    redundant_categories: List[str]
    topic_counts: Optional[Dict[str, int]] = None


class AnswerExtractor:
    def __init__(self, client, settings):
        self._client = client
        self._settings = settings

    def _normalize_language(self, language: str) -> str:
        return _normalize_language(language)

    def _format_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> str:
        return _format_qa_pairs(qa_pairs)

    async def extract_covered_information(
        self,
        asked_questions: List[str],
        previous_answers: List[str],
        medical_context: MedicalContext,
        language: str = "en",
        visit_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        question_number: Optional[int] = None,
    ) -> ExtractedInformation:
        all_qa: List[Dict[str, str]] = []
        for i in range(len(asked_questions or [])):
            if i < len(previous_answers or []):
                all_qa.append({"question": asked_questions[i], "answer": previous_answers[i]})
        # HARD RULE: if nothing asked yet => nothing covered
        if not all_qa:
            gaps = _recompute_gaps_from_plan(medical_context=medical_context, topics_covered=[])
            return ExtractedInformation(
                topics_covered=[],
                information_gaps=gaps,
                extracted_facts={},
                already_mentioned_duration=False,
                already_mentioned_medications=False,
                redundant_categories=[],
                topic_counts=None,
            )
        system_prompt = """You are AGENT-02 "COVERAGE & FACT EXTRACTOR".
Return ONLY JSON:
{
"topics_covered": ["<topic>", "..."],
"information_gaps": ["<topic>", "..."],
"extracted_facts": {
"duration": "<text or null>",
"medications": "<text or null>",
"pain_severity": "<text or null>",
"associated_symptoms": "<text or null>"
},
"already_mentioned_duration": <true|false>,
"already_mentioned_medications": <true|false>,
"redundant_categories": ["<topic>", "..."],
"topic_counts": { "<topic>": <int>, "...": <int> }
}
CRITICAL RULES:
- topics_covered and information_gaps MUST be DISJOINT (no overlap).
- topics_covered may include a topic ONLY if a question about that topic exists in history.
- information_gaps MUST equal:
(medical_context.topic_plan minus topics_covered minus medical_context.avoid_topics)
- If the conversation has 0 Q/A pairs, topics_covered MUST be [] and information_gaps MUST be topic_plan (minus avoid_topics).
Other rules:
- Allowed topics must match the system allowed list exactly.
- If a topic was asked clearly once, it counts as covered (even if answer is "no/none").
Return ONLY JSON.
"""
        user_prompt = f"""
CONVERSATION HISTORY:
{self._format_qa_pairs(all_qa)}
MEDICAL CONTEXT:
Chief complaint: {medical_context.chief_complaint}
Priority topics: {medical_context.priority_topics}
Avoid topics: {medical_context.avoid_topics}
Recommended plan order: {medical_context.topic_plan}
Return the JSON now.
"""
        resp = await call_llm_with_telemetry(
            ai_client=self._client,
            scenario=PromptScenario.INTAKE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self._settings.openai.model,
            max_tokens=700,
            temperature=0.1,
        )
        response_text = (resp.choices[0].message.content or "").strip()
        log_msg = (
            "[Agent2-AnswerExtractor] Raw LLM response received\n"
            f"{'=' * 80}\n{response_text}\n{'=' * 80}"
        )
        print(log_msg, flush=True)
        logger.info(log_msg)
        raw = _extract_first_json_object(response_text)
        if not raw:
            raise ValueError("Agent2 returned no valid JSON")
        topics_covered = _clamp_topics(_safe_str_list(raw.get("topics_covered")))
        information_gaps = _clamp_topics(_safe_str_list(raw.get("information_gaps")))
        redundant_categories = _clamp_topics(_safe_str_list(raw.get("redundant_categories")))
        extracted_facts = raw.get("extracted_facts") or {}
        if not isinstance(extracted_facts, dict):
            extracted_facts = {}
        for k in ["duration", "medications", "pain_severity", "associated_symptoms"]:
            v = extracted_facts.get(k)
            if v is None:
                continue
            if not isinstance(v, str):
                extracted_facts[k] = str(v)
        already_duration = bool(raw.get("already_mentioned_duration", False))
        already_meds = bool(raw.get("already_mentioned_medications", False))
        priority = set(medical_context.priority_topics or [])
        avoid = set(medical_context.avoid_topics or [])
        topics_covered = [t for t in topics_covered if t in priority and t not in avoid]
        redundant_categories = [t for t in redundant_categories if t in set(ALLOWED_TOPICS)]
        covered_set = set(topics_covered)
        information_gaps = [t for t in information_gaps if t in priority and t not in avoid and t not in covered_set]
        recomputed_gaps = _recompute_gaps_from_plan(medical_context=medical_context, topics_covered=topics_covered)
        if not information_gaps or (set(information_gaps) == set(topics_covered)):
            information_gaps = recomputed_gaps
        topic_counts = raw.get("topic_counts")
        if not isinstance(topic_counts, dict):
            topic_counts = dict(Counter(topics_covered))
        # Log to structured llm_interaction collection (only user_prompt, no system_prompt)
        if visit_id and patient_id and question_number is not None:
            try:
                await append_intake_agent_log(
                    visit_id=visit_id,
                    patient_id=patient_id,
                    question_number=question_number,
                    question_text=None,  # Not known yet at Agent 2 stage
                    agent_name="agent2_extractor",
                    user_prompt=user_prompt,  # Only user_prompt, NO system_prompt
                    response_text=response_text,
                    metadata={
                        "topics_covered": topics_covered,
                        "information_gaps": information_gaps,
                        "redundant_categories": redundant_categories,
                        "extracted_facts": extracted_facts,
                        "already_mentioned_duration": already_duration,
                        "already_mentioned_medications": already_meds,
                        "topic_counts": topic_counts,
                        "prompt_version": PROMPT_VERSIONS.get(PromptScenario.INTAKE, "UNKNOWN"),
                    },
                )
            except Exception as e:
                logger.warning("Agent2: failed to persist interaction: %s", e)
        return ExtractedInformation(
            topics_covered=topics_covered,
            information_gaps=information_gaps,
            extracted_facts=extracted_facts,
            already_mentioned_duration=already_duration,
            already_mentioned_medications=already_meds,
            redundant_categories=redundant_categories,
            topic_counts=topic_counts,
        )
# =============================================================================
# AGENT 3: Question Generator (topic-forced) + ✅ TOPIC ENFORCEMENT
# =============================================================================
class QuestionGenerator:
    def __init__(self, client, settings):
        self._client = client
        self._settings = settings

    def _normalize_language(self, language: str) -> str:
        return _normalize_language(language)

    def _format_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> str:
        return _format_qa_pairs(qa_pairs)

    def _postprocess_question_text(self, question: str) -> str:
        if not question:
            return question
        q = question.strip()
        if (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'")):
            q = q[1:-1].strip()
        q = re.sub(r"^(q:|\d+\.)\s*", "", q, flags=re.IGNORECASE)
        q = q.replace("\n", " ")
        q = " ".join(q.split())
        q = q.rstrip("?.!,;:")
        if not q.endswith("?"):
            q += "?"
        return q

    # ----------------------------
    # ✅ Topic rulebook + validator
    # ----------------------------
    _TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "duration": ["how long", "when did", "since when", "start", "began", "duration"],
        "associated_symptoms": [
            "symptom", "fever", "fatigue", "nausea", "vomit", "cough", "breath",
            "dizzy", "headache", "urination", "thirst", "blur", "weight loss",
            "tingling", "numb", "swelling"
        ],
        "current_medications": ["medication", "medicine", "tablet", "pill", "insulin", "metformin", "dose", "mg", "units", "taking"],
        "past_medical_history": ["history", "diagnosed", "past", "previous", "surgery", "hospital", "condition"],
        "triggers": ["trigger", "worse", "better", "after", "before", "food", "exercise", "stress", "sleep"],
        "travel_history": ["travel", "trip", "flight", "abroad", "out of town"],
        "lifestyle_functional_impact": ["daily life", "work", "sleep", "appetite", "activity", "exercise", "diet", "routine", "impact", "function"],
        "family_history": ["family", "mother", "father", "siblings", "runs in family", "relative"],
        "allergies": ["allergy", "allergic", "rash", "hives", "reaction"],
        "pain_assessment": ["pain", "severity", "0 to 10", "scale", "where", "location", "radiate"],
        "temporal": ["pattern", "frequency", "how often", "timing", "episodes", "comes and goes", "progress"],
        "menstrual_cycle": ["period", "menstrual", "cycle", "pregnant", "lmp", "bleeding"],
        "past_evaluation": ["tests", "scan", "x-ray", "lab", "doctor", "evaluation", "report"],
        "chronic_monitoring": ["monitor", "check", "readings", "hba1c", "fasting", "logs", "follow up"],
        "lab_tests": ["lab", "blood test", "results", "hba1c", "cholesterol", "thyroid", "creatinine", "test results"],
        "screening": ["screening", "eye", "kidney", "feet", "foot", "retina", "complications", "imaging", "stress test"],

    }
    _TOPIC_FALLBACK_Q: Dict[str, str] = {
        "duration": "How long have you had this problem?",
        "associated_symptoms": "What other symptoms have you noticed along with this?",
        "current_medications": "Are you currently taking any medicines or insulin? If yes, which ones and what doses?",
        "past_medical_history": "Do you have any past medical conditions or surgeries I should know about?",
        "triggers": "Have you noticed anything that makes it better or worse (food, stress, activity, time of day)?",
        "travel_history": "Have you travelled anywhere recently (within the last few weeks)?",
        "lifestyle_functional_impact": "How is this affecting your daily routine—sleep, work, diet, or activity?",
        "family_history": "Does anyone in your family have similar health conditions (like diabetes, BP, thyroid)?",
        "allergies": "Do you have any allergies to medicines, foods, or anything else?",
        "pain_assessment": "If you have pain, where is it and how severe is it on a scale of 0 to 10?",
        "temporal": "How often does this happen, and is it getting better, worse, or staying the same?",
        "menstrual_cycle": "When was your last menstrual period, and are your cycles regular?",
        "past_evaluation": "Have you had any tests or doctor visits for this? What were the results?",
        "chronic_monitoring": "Do you regularly monitor your condition (e.g., sugar readings)? If yes, what are typical values?",
        "lab_tests": "Have you had any recent lab tests for this condition (like blood tests), and what do you remember about the results?",
        "screening": "Have you had any screening tests for complications (like eye, heart, or kidney exams) recently?",

    }

    def _question_matches_topic(self, chosen_topic: str, question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False
        kws = self._TOPIC_KEYWORDS.get(chosen_topic, [])
        if not kws:
            return True  # if no rulebook, don't block
        return any(kw in q for kw in kws)

    def _get_fallback_question(self, chosen_topic: str) -> str:
        return self._TOPIC_FALLBACK_Q.get(chosen_topic, "Could you tell me more about that?")

    async def _llm_generate_once(self, system_prompt: str, user_prompt: str) -> str:
        resp = await call_llm_with_telemetry(
            ai_client=self._client,
            scenario=PromptScenario.INTAKE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self._settings.openai.model,
            max_tokens=250,
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()

    async def generate_question_for_topic(
        self,
        medical_context: MedicalContext,
        extracted_info: ExtractedInformation,
        chosen_topic: str,
        asked_questions: Optional[List[str]] = None,
        previous_answers: Optional[List[str]] = None,
        language: str = "en",
        max_words: int = 25,
        deep_diagnostic_question_num: Optional[int] = None,
        visit_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        question_number: Optional[int] = None,
    ) -> str:
        lang = self._normalize_language(language)
        qa_history = ""
        if asked_questions and previous_answers:
            all_qa = []
            for i in range(len(asked_questions)):
                if i < len(previous_answers):
                    all_qa.append({"question": asked_questions[i], "answer": previous_answers[i]})
            qa_history = self._format_qa_pairs(all_qa) if all_qa else ""
        # ✅ Clear, focused prompt without refusal option
        system_prompt = f"""You are AGENT-03 "INTAKE QUESTION GENERATOR" for clinical intake interviews.
Your task: Generate ONE clear, concise medical question about the topic: {chosen_topic}
Requirements:
- Output ONLY the question text (no quotes, no numbering, no explanations)
- Question must end with "?"
- Keep it under {max_words} words
- Focus ONLY on {chosen_topic} - do not combine with other topics
- Use natural, conversational language appropriate for a medical interview
- Do not repeat questions from the conversation history
- Make the question specific and easy for the patient to answer
Topic-specific guidance:
- duration: Ask how long they've had the problem AND what might have caused it
- current_medications: Ask about medications, home remedies, AND dosages/frequency
- past_medical_history: If chronic condition, focus on related conditions only. If non-chronic, ask about conditions related to chief complaint AND associated symptoms
- pain_assessment: Ask about severity (0-10), characterization, AND location/radiation
- triggers: Ask about what brings it on AND what makes it worse
- lifestyle_functional_impact: Ask about daily activities, work, AND routine changes
- temporal: Ask about frequency AND progression (better/worse/same)
- chronic_monitoring: Ask about BOTH home monitoring (self-checks) AND professional/clinical monitoring (clinic/doctor checks), including frequency AND typical readings/values (e.g., "How often do you or your doctors check your [condition-specific metric], and what are your typical readings?")
 - lab_tests: Ask specifically about LAB TEST RESULTS (what lab tests, what results) for this condition.
 - screening: Ask specifically about FORMAL SCREENING EXAMS/COMPLICATION CHECKS (what screening exams, when last done) — DO NOT mix lab tests into screening.
IMPORTANT: For deep diagnostic questions (chronic_monitoring, lab_tests, screening), follow the SPECIAL INSTRUCTION provided in the user prompt below.
Generate the question now.
"""
        deep_diag_note = ""
        if deep_diagnostic_question_num is not None:
            if deep_diagnostic_question_num == 1 and chosen_topic == "chronic_monitoring":
                deep_diag_note = "\n\nDEEP DIAGNOSTIC QUESTION #1 - HOME & CLINICAL MONITORING:\n" \
                                 "Ask about how the patient monitors this chronic condition BOTH at home and in clinical settings.\n" \
                                 "Examples: home blood sugar readings, home BP checks, and clinic-based device or doctor checks.\n" \
                                 "Format: Ask about frequency of monitoring AND typical values/readings (home and/or clinic).\n" \
                                 "Example: 'How often do you or your doctors check your readings for this condition, and what are your usual values?'"
            elif deep_diagnostic_question_num == 2 and chosen_topic == "lab_tests":
                deep_diag_note = "\n\nDEEP DIAGNOSTIC QUESTION #2 - LAB TEST RESULTS ONLY:\n" \
                    "Ask about RECENT LABORATORY TEST RESULTS relevant to this chronic condition.\n" \
                    "Focus on: HbA1c, fasting glucose, kidney function tests, cholesterol panels, thyroid labs, or other condition-specific LAB tests.\n" \
                    "Format: Ask what recent lab tests they've had AND what they remember about the lab results.\n" \
                    "Example: 'Have you had any recent lab tests for this condition, and what do you remember about the results?'"
            elif deep_diagnostic_question_num == 3 and chosen_topic == "screening":
                deep_diag_note = "\n\nDEEP DIAGNOSTIC QUESTION #3 - SCREENING/COMPLICATION CHECKS (NO LABS):\n" \
                                 "Ask about FORMAL SCREENING EXAMS and COMPLICATION CHECKS (not routine labs) done because of this chronic condition.\n" \
                                 "Focus on: eye/foot exams, cardiac imaging or stress tests, kidney imaging, lung function tests, or other screening exams.\n" \
                    "Format: Ask if they've had screening exams AND when they were last done.\n" \
                    "Example: 'Have you had any screening tests for complications related to this condition (like eye, heart, or kidney exams), and when were they last done?'"
        user_prompt = f"""
CHOSEN TOPIC (MUST FOLLOW): {chosen_topic}{deep_diag_note}
Patient:
- Chief complaint: {medical_context.chief_complaint}
- Age: {medical_context.patient_age or "Unknown"}
- Gender: {medical_context.patient_gender or "Not specified"}
- Recent travel checkbox: {"Yes" if medical_context.recently_travelled else "No"}
Agent-1 context:
Priority topics: {medical_context.priority_topics}
Avoid topics: {medical_context.avoid_topics}
Topic plan: {medical_context.topic_plan}
Condition properties: {json.dumps(medical_context.condition_properties, indent=2)}
Agent-2 coverage:
Covered: {extracted_info.topics_covered}
Redundant: {extracted_info.redundant_categories}
Gaps: {extracted_info.information_gaps}
Facts: {json.dumps(extracted_info.extracted_facts, indent=2)}
Conversation history:
{qa_history or "No previous questions."}
Generate ONE question now, strictly about {chosen_topic}.
"""
        # 1) attempt
        response_text = await self._llm_generate_once(system_prompt, user_prompt)
        q1 = self._postprocess_question_text(response_text)
        log_msg = (
            "[Agent3-QuestionGenerator] Raw LLM response received\n"
            f"{'=' * 80}\n{response_text}\n{'=' * 80}"
        )
        print(log_msg, flush=True)
        logger.info(log_msg)
        # Log to structured llm_interaction collection (only user_prompt, no system_prompt)
        if visit_id and patient_id and question_number is not None:
            try:
                await append_intake_agent_log(
                    visit_id=visit_id,
                    patient_id=patient_id,
                    question_number=question_number,
                    question_text=q1,  # Question text from attempt #1
                    agent_name="agent3_question_generator",
                    user_prompt=user_prompt,  # Only user_prompt, NO system_prompt
                    response_text=response_text,
                    metadata={
                        "chosen_topic": chosen_topic,
                        "attempt": 1,
                        "prompt_version": PROMPT_VERSIONS.get(PromptScenario.INTAKE, "UNKNOWN"),
                    },
                )
            except Exception as e:
                logger.warning("Agent3: failed to persist interaction (attempt 1): %s", e)
        # If question is empty or doesn't match topic => retry once
        if (not q1) or (not self._question_matches_topic(chosen_topic, q1)):
            correction = f"""
Your previous question was not clearly focused on the topic: {chosen_topic}
Please generate a question that is SPECIFICALLY about {chosen_topic}.
Make it clear, concise, and directly relevant to this topic.
Return ONE question ONLY.
"""
            response_text_2 = await self._llm_generate_once(system_prompt, user_prompt + "\n\n" + correction)
            q2 = self._postprocess_question_text(response_text_2)
            log_msg2 = (
                "[Agent3-QuestionGenerator] Raw LLM response received (retry)\n"
                f"{'=' * 80}\n{response_text_2}\n{'=' * 80}"
            )
            print(log_msg2, flush=True)
            logger.info(log_msg2)
            # Log retry attempt #2 to structured llm_interaction collection
            if visit_id and patient_id and question_number is not None:
                try:
                    await append_intake_agent_log(
                        visit_id=visit_id,
                        patient_id=patient_id,
                        question_number=question_number,
                        question_text=q2 if q2 and self._question_matches_topic(chosen_topic, q2) else None,
                        agent_name="agent3_question_generator",
                        user_prompt=user_prompt + "\n\n" + correction,  # Only user_prompt, NO system_prompt
                        response_text=response_text_2,
                        metadata={
                            "chosen_topic": chosen_topic,
                            "attempt": 2,
                            "prompt_version": PROMPT_VERSIONS.get(PromptScenario.INTAKE, "UNKNOWN"),
                        },
                    )
                except Exception as e:
                    logger.warning("Agent3: failed to persist interaction (attempt 2): %s", e)
            if q2 and self._question_matches_topic(chosen_topic, q2):
                return q2
            # deterministic fallback
            return self._postprocess_question_text(self._get_fallback_question(chosen_topic))
        return q1
# =============================================================================
# OPTIONAL: Safety Validator
# =============================================================================
@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str]
    corrected_question: Optional[str]


class SafetyValidator:
    def __init__(self, client, settings):
        self._client = client
        self._settings = settings

    def _normalize_language(self, language: str) -> str:
        return _normalize_language(language)

    async def validate_question(
        self,
        question: str,
        medical_context: MedicalContext,
        asked_questions: List[str],
        language: str = "en",
        recently_travelled: bool = False,
    ) -> ValidationResult:
        issues: List[str] = []
        q = (question or "").strip()
        ql = q.lower()
        prev_norm = {(x or "").strip().lower().rstrip("?.") for x in (asked_questions or [])}
        if q.lower().rstrip("?.") in prev_norm:
            issues.append("CRITICAL VIOLATION: exact duplicate question")
        if medical_context.patient_gender:
            g = medical_context.patient_gender.lower().strip()
            if g in ["male", "m", "masculino", "hombre"]:
                if any(k in ql for k in MENSTRUAL_KEYWORDS):
                    issues.append("CRITICAL VIOLATION: menstrual asked to male")
        if medical_context.patient_age is not None and (medical_context.patient_age < 12 or medical_context.patient_age > 60):
            if any(k in ql for k in MENSTRUAL_KEYWORDS):
                issues.append("CRITICAL VIOLATION: menstrual asked outside age window")
        is_travel_related = bool((medical_context.condition_properties or {}).get("is_travel_related", False))
        if (not recently_travelled or not is_travel_related) and any(k in ql for k in TRAVEL_KEYWORDS):
            issues.append("CRITICAL VIOLATION: travel asked when not allowed")
        corrected = q
        if not corrected.endswith("?"):
            corrected = corrected.rstrip(".") + "?"
        is_valid = not any("CRITICAL VIOLATION" in x for x in issues)
        return ValidationResult(is_valid=is_valid, issues=issues, corrected_question=(corrected if is_valid else None))
# =============================================================================
# OPTION A: LLM decides next topic each turn (with guardrails)
# =============================================================================
class PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    next_topic: str
    stop_intake: bool = False
    reason: Optional[str] = None


class TopicPlannerAgent:
    def __init__(self, client, settings):
        self._client = client
        self._settings = settings

    async def decide_next_topic(
        self,
        medical_context: MedicalContext,
        extracted: ExtractedInformation,
        asked_questions: List[str],
        current_count: int,
        max_count: int,
        language: str = "en",
    ) -> PlannerDecision:
        system_prompt = """You are the TOPIC PLANNER for an intake interview.
Return ONLY JSON:
{
"next_topic": "<one of allowed topics>",
"stop_intake": <true|false>,
"reason": "<short>"
}
Allowed topics:
duration, associated_symptoms, current_medications, past_medical_history,
triggers, travel_history, lifestyle_functional_impact, family_history,
allergies, pain_assessment, temporal, menstrual_cycle, past_evaluation,
chronic_monitoring, lab_tests, screening
Rules:
- next_topic must be a SINGLE topic
- stop_intake true only if no valid topics remain
- Always prioritize medical_context.topic_plan order, then extracted.information_gaps
- Never select avoid_topics or already covered topics
Return ONLY JSON.
"""
        user_prompt = f"""
Agent1:
topic_plan={medical_context.topic_plan}
priority_topics={medical_context.priority_topics}
avoid_topics={medical_context.avoid_topics}
condition_properties={json.dumps(medical_context.condition_properties, indent=2)}
recent_travel_checkbox={"Yes" if medical_context.recently_travelled else "No"}
Agent2:
topics_covered={extracted.topics_covered}
information_gaps={extracted.information_gaps}
Progress:
current_count={current_count}
max_count={max_count}
Decide the NEXT topic now.
"""
        resp = await call_llm_with_telemetry(
            ai_client=self._client,
            scenario=PromptScenario.INTAKE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self._settings.openai.model,
            max_tokens=250,
            temperature=0.1,
        )
        txt = (resp.choices[0].message.content or "").strip()
        raw = _extract_first_json_object(txt)
        if not raw:
            raise ValueError("Planner returned no JSON")
        decision = PlannerDecision.model_validate(raw)
        return decision
class OpenAIQuestionService(QuestionService):
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = get_ai_client()
        self._context_analyzer = MedicalContextAnalyzer(self._client, self._settings)
        self._answer_extractor = AnswerExtractor(self._client, self._settings)
        self._planner = TopicPlannerAgent(self._client, self._settings)
        self._question_generator = QuestionGenerator(self._client, self._settings)
        self._safety_validator = SafetyValidator(self._client, self._settings)
        logger.info("Multi-Agent Question Service initialized")

    def _normalize_language(self, language: str) -> str:
        return _normalize_language(language)

    def _closing(self, language: str) -> str:
        lang = self._normalize_language(language)
        return "¿Hay algo más que le gustaría compartir sobre su condición?" if lang == "sp" else \
            "Is there anything else you'd like to share about your condition?"

    async def generate_first_question(
        self,
        disease: str,
        language: str = "en",
        visit_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        question_number: Optional[int] = None,
    ) -> str:
        lang = self._normalize_language(language)
        return "¿Por qué ha venido hoy? ¿Cuál es la principal preocupación con la que necesita ayuda?" if lang == "sp" else \
            "Why have you come in today? What is the main concern you want help with?"

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
        visit_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        question_number: Optional[int] = None,
    ) -> str:
        chief = (disease or "").strip() or "general consultation"
        medical_context = await self._context_analyzer.analyze_condition(
            chief_complaint=chief,
            patient_age=patient_age,
            patient_gender=patient_gender,
            recently_travelled=recently_travelled,
            language=language,
            visit_id=visit_id,
            patient_id=patient_id,
            question_number=question_number,
        )
        extracted = await self._answer_extractor.extract_covered_information(
            asked_questions=asked_questions or [],
            previous_answers=previous_answers or [],
            medical_context=medical_context,
            language=language,
            visit_id=visit_id,
            patient_id=patient_id,
            question_number=question_number,
        )
        # =============================================================================
        # ✅ COVERAGE/REDUNDANCY IS 100% CODE-TRUTH (asked_categories-driven)
        # =============================================================================
        topic_counts = _topic_counts_from_asked_categories(asked_categories)
        topics_covered_truth = list(topic_counts.keys())
        redundant_truth = [t for t, c in topic_counts.items() if c > 1]
        extracted.topics_covered = topics_covered_truth
        extracted.redundant_categories = redundant_truth
        extracted.topic_counts = topic_counts
        extracted.information_gaps = _recompute_gaps_from_plan(
            medical_context=medical_context,
            topics_covered=extracted.topics_covered,
        )
        # =============================================================================
        # =============================================================================
        # ✅ STRICT SEQUENTIAL TOPIC SELECTION (Steps 2-13)
        # =============================================================================
        condition_props = medical_context.condition_properties or {}
        is_chronic = bool(condition_props.get("is_chronic", False))
        is_hereditary = bool(condition_props.get("is_hereditary", False))
        is_allergy_related = bool(condition_props.get("is_allergy_related", False))
        is_pain_related = bool(condition_props.get("is_pain_related", False))
        is_women_health = bool(condition_props.get("is_womens_health", False))
        is_travel_related = bool(condition_props.get("is_travel_related", False))
        # 🔒 Rule: any chief complaint > 3 months → treat as chronic (overrides LLM)
        # We infer from prior answers (duration question is step 2).
        if not is_chronic and _duration_implies_chronic(previous_answers):
            is_chronic = True
            condition_props["is_chronic"] = True  # keep downstream logic aligned
        # =============================================================================
        # ✅ MAX QUESTION LIMITS (based on chronicity and consent)
        # =============================================================================
        # Check consent status early to determine max_count
        consent_index = None
        has_positive_consent = False
        has_negative_consent = False
        if asked_questions and previous_answers:
            for idx, q in enumerate(asked_questions):
                q_low = q.lower()
                if "detailed diagnostic questions" in q_low or "preguntas diagnósticas detalladas" in q_low:
                    consent_index = idx
            if consent_index is not None and consent_index < len(previous_answers):
                consent_answer = previous_answers[consent_index].lower().strip()
                consent_answer = re.sub(r'[?!.;,\s]+', ' ', consent_answer)
                consent_answer = ' '.join(consent_answer.split())
                positive_kw = ["yes", "y", "yeah", "yep", "sure", "ok", "okay", "of course", "absolutely",
                               "sí", "si", "claro", "por supuesto", "vale", "de acuerdo"]
                negative_kw = ["no", "n", "nope", "nah", "not really", "no thanks", "no thank you",
                               "no gracias", "no quiero", "no deseo", "prefiero no"]
                if any(k in consent_answer for k in positive_kw):
                    has_positive_consent = True
                elif any(k in consent_answer for k in negative_kw):
                    has_negative_consent = True
                else:
                    has_negative_consent = True  # Ambiguous = negative
        # Set max_count based on chronicity and consent
        if not is_chronic:
            # Non-chronic: max 10 questions (step 10 is closing)
            max_count = min(max_count, 10)
        else:
            # Chronic conditions
            if has_positive_consent:
                # Chronic + positive consent: max 13 questions (step 13 is closing)
                max_count = min(max_count, 13)
            else:
                # Chronic + negative consent OR no consent yet: max 10 questions (step 10 is closing)
                max_count = min(max_count, 10)
        # =============================================================================
        step_number = current_count + 1  # 1-based (step 1 = chief complaint, handled separately)
        next_topic: Optional[str] = None
        # Step 2: duration (include cause)
        if step_number == 2:
            next_topic = "duration"
        # Step 3: associated_symptoms
        elif step_number == 3:
            next_topic = "associated_symptoms"
        # Step 4: current_medications
        elif step_number == 4:
            next_topic = "current_medications"
        # Step 5: past_medical_history
        elif step_number == 5:
            next_topic = "past_medical_history"
        # Step 6: triggers
        elif step_number == 6:
            next_topic = "triggers"
        # Step 7: travel_history (if travel checkbox) OR lifestyle_functional_impact
        elif step_number == 7:
            if recently_travelled and is_travel_related:
                next_topic = "travel_history"
            else:
                next_topic = "lifestyle_functional_impact"
        # Step 8: family_history / allergies / pain_assessment / temporal
        elif step_number == 8:
            if is_chronic or is_hereditary:
                next_topic = "family_history"
            elif is_allergy_related:
                next_topic = "allergies"
            elif is_pain_related:
                next_topic = "pain_assessment"
            else:
                next_topic = "temporal"
        # Step 9: deep_diagnostic_consent (if chronic/hereditary) OR menstrual_cycle OR past_evaluation
        elif step_number == 9:
            if is_chronic or is_hereditary:
                # Return consent question directly (not a topic)
                lang = self._normalize_language(language)
                if lang == "sp":
                    return "¿Le gustaría responder algunas preguntas diagnósticas detalladas relacionadas con sus síntomas?"
                return "Would you like to answer some detailed diagnostic questions related to your symptoms?"
            elif is_women_health:
                next_topic = "menstrual_cycle"
            else:
                next_topic = "past_evaluation"
        # Step 10-13: Deep diagnostic flow (chronic only, if consent positive)
        elif step_number >= 10 and is_chronic:
            # Consent status already checked above
            # If negative consent OR no consent yet, return closing question at step 10
            if has_negative_consent or not has_positive_consent:
                if step_number == 10:
                    logger.info("Chronic condition with negative/no consent - returning closing question at step 10")
                    return self._closing(language)
                else:
                    # Should not reach here, but safety check
                    return self._closing(language)
            # If positive consent, continue with deep diagnostic questions
            if has_positive_consent:
                deep_question_num = step_number - 9  # 1, 2, 3, or 4
                if deep_question_num == 1:
                    # Step 10: Home monitoring
                    next_topic = "chronic_monitoring"
                elif deep_question_num == 2:
                    # Step 11: Lab test results (use screening topic but will be phrased as labs)
                    next_topic = "lab_tests"
                elif deep_question_num == 3:
                    # Step 12: Screening/complications
                    next_topic = "screening"
                elif deep_question_num == 4:
                    # Step 13: Closing question (mandatory for chronic + positive consent)
                    logger.info("Step 13 reached (chronic + positive consent) - returning mandatory closing question")
                    return self._closing(language)
                else:
                    # Beyond step 13, return closing
                    return self._closing(language)
        # Step 10: Non-chronic OR chronic with negative consent gets closing question
        elif step_number == 10:
            if not is_chronic:
                logger.info("Step 10 reached (non-chronic) - returning closing question")
            elif is_chronic and (has_negative_consent or not has_positive_consent):
                logger.info("Step 10 reached (chronic + negative/no consent) - returning closing question")
            return self._closing(language)
        # Safety check: beyond max_count (should not reach here normally due to step-based logic)
        elif current_count >= max_count:
            logger.info("Max count reached (current_count=%d, max_count=%d) - returning closing question", current_count, max_count)
            return self._closing(language)
        # Safety check: beyond step 13 (should not reach here normally)
        elif step_number > 13:
            logger.warning("Step number %d exceeds maximum - returning closing question", step_number)
            return self._closing(language)
        # If no topic determined, fallback to closing
        if not next_topic:
            logger.warning("Strict sequence: no topic for step %d -> closing", step_number)
            return self._closing(language)
        # Validate topic is allowed and not in avoid list
        allowed = set(ALLOWED_TOPICS)
        avoid = set(medical_context.avoid_topics or [])
        if next_topic not in allowed:
            logger.warning("Strict sequence: topic '%s' not in allowed list -> closing", next_topic)
            return self._closing(language)
        if next_topic in avoid:
            logger.warning("Strict sequence: topic '%s' in avoid list -> closing", next_topic)
            return self._closing(language)
        # =============================================================================
        # =============================================================================
        # ✅ CRITICAL: append chosen topic to asked_categories (code-truth tracking)
        # =============================================================================
        if asked_categories is not None:
            asked_categories.append(next_topic)
        # =============================================================================
        # Determine deep diagvnostic question number if applicable
        deep_diag_num = None
        if step_number >= 10 and is_chronic:
            # Check if we have positive consent
            consent_index = None
            has_positive_consent = False
            if asked_questions and previous_answers:
                for idx, q in enumerate(asked_questions):
                    q_low = q.lower()
                    if "detailed diagnostic questions" in q_low or "preguntas diagnósticas detalladas" in q_low:
                        consent_index = idx
            if consent_index is not None and consent_index < len(previous_answers):
                consent_answer = previous_answers[consent_index].lower().strip()
                consent_answer = re.sub(r'[?!.;,\s]+', ' ', consent_answer)
                consent_answer = ' '.join(consent_answer.split())
                positive_kw = ["yes", "y", "yeah", "yep", "sure", "ok", "okay", "of course", "absolutely",
                               "sí", "si", "claro", "por supuesto", "vale", "de acuerdo"]
                if any(k in consent_answer for k in positive_kw):
                    has_positive_consent = True
            if has_positive_consent:
                deep_diag_num = step_number - 9  # 1, 2, or 3
        logger.info("Strict sequence: step %d, chosen_topic='%s', deep_diag_num=%s", step_number, next_topic, deep_diag_num)
        q = await self._question_generator.generate_question_for_topic(
            medical_context=medical_context,
            extracted_info=extracted,
            chosen_topic=next_topic,
            asked_questions=asked_questions or [],
            previous_answers=previous_answers or [],
            language=language,
            deep_diagnostic_question_num=deep_diag_num,
            visit_id=visit_id,
            patient_id=patient_id,
            question_number=question_number,
        )
        validation = await self._safety_validator.validate_question(
            question=q,
            medical_context=medical_context,
            asked_questions=asked_questions or [],
            language=language,
            recently_travelled=recently_travelled,
        )
        if not validation.is_valid:
            logger.error("OptionA: safety invalid: %s", validation.issues)
            return self._closing(language)
        return validation.corrected_question or q

    async def should_stop_asking(self, disease: str, previous_answers: List[str], current_count: int, max_count: int = 10) -> bool:
        return current_count >= max_count

    async def assess_completion_percent(
        self,
        disease: str,
        previous_answers: List[str],
        asked_questions: List[str],
        current_count: int,
        max_count: int = 10,
        prior_summary: Optional[Any] = None,
        prior_qas: Optional[List[str]] = None
    ) -> int:
        if max_count <= 0:
            return 0
        return int(min(max(current_count / max_count, 0.0), 1.0) * 100)

    async def is_medication_question(self, question: str) -> bool:
        q = (question or "").lower()
        return "medication" in q or "medicine" in q or "medicamentos" in q


    # ========================================================================
    # PRE-VISIT SUMMARY & RED-FLAG METHODS ARE INTENTIONALLY EXCLUDED HERE
    # ========================================================================

    async def _get_doctor_preferences(self, doctor_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch doctor preferences with 1s timeout; fail-open on error."""
        if not doctor_id:
            return None
        try:
            prefs = await asyncio.wait_for(
                DoctorPreferencesMongo.find_one(DoctorPreferencesMongo.doctor_id == doctor_id),
                timeout=1.0
            )
            return prefs.dict() if prefs else None
        except Exception as e:
            logger.warning(f"[DoctorPrefs] Failed to load preferences for doctor_id={doctor_id}: {e}")
            return None


    async def generate_pre_visit_summary(
        self,
        patient_data: Dict[str, Any],
        intake_answers: Dict[str, Any],
        language: str = "en",
        medication_images_info: Optional[str] = None,
        doctor_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate pre-visit clinical summary from intake data with red flag detection."""

        # Normalize language code
        lang = self._normalize_language(language)

        # Load doctor preferences with fail-open defaults
        prefs = await self._get_doctor_preferences(doctor_id)
        pv_config = (prefs or {}).get("pre_visit_ai_config") or {}

        # AI style preferences (always optional, safe defaults)
        style_pref = (pv_config.get("style") or "standard").strip().lower()
        focus_areas = pv_config.get("focus_areas") or []
        include_red_flags = pv_config.get("include_red_flags")
        if include_red_flags is None:
            include_red_flags = True
        focus_text = ", ".join(focus_areas) if focus_areas else "all relevant areas"
        prefs_snippet = (
            f"\nDoctor Preferences:\n"
            f"- Summary style: {style_pref}\n"
            f"- Focus areas: {focus_text}\n"
            f"- Include red flags: {'yes' if include_red_flags else 'no'}\n"
        )

        # Section configuration from doctor preferences (pre_visit_config)
        raw_sections = (prefs or {}).get("pre_visit_config") or []
        # Default behavior:
        # - If NO config present at all -> all sections enabled (fail-open, legacy behavior).
        # - If ANY config present       -> sections are opt-in and must be explicitly enabled.
        if raw_sections:
            default_section_state = {
                "chief_complaint": False,
                "hpi": False,
                "history": False,
                "review_of_systems": False,
                "current_medication": False,
            }
        else:
            default_section_state = {
                "chief_complaint": True,
                "hpi": True,
                "history": True,
                "review_of_systems": True,
                "current_medication": True,
            }
        enabled_sections = default_section_state.copy()
        try:
            for sec in raw_sections:
                key = sec.get("section_key") if isinstance(sec, dict) else getattr(sec, "section_key", None)
                if key in enabled_sections:
                    enabled_sections[key] = bool(sec.get("enabled", True) if isinstance(sec, dict) else getattr(sec, "enabled", True))
        except Exception as e:
            # Fail-open: if malformed, keep defaults and log at debug level
            logger.debug(f"[DoctorPrefs] Failed to parse pre_visit_config for doctor_id={doctor_id}: {e}")

        enable_cc = enabled_sections.get("chief_complaint", True)
        enable_hpi = enabled_sections.get("hpi", True)
        enable_history = enabled_sections.get("history", True)
        enable_ros = enabled_sections.get("review_of_systems", True)
        enable_meds = enabled_sections.get("current_medication", True)

        if lang == "sp":
            # Build dynamic Spanish headings based on enabled sections
            headings_lines_es: list[str] = []
            if enable_cc:
                headings_lines_es.append("Motivo de Consulta:")
            if enable_hpi:
                headings_lines_es.append("HPI:")
            if enable_history:
                headings_lines_es.append("Historia:")
            if enable_ros:
                headings_lines_es.append("Revisión de Sistemas:")
            if enable_meds:
                headings_lines_es.append("Medicación Actual:")
            headings_text_es = "\n".join(headings_lines_es) + ("\n\n" if headings_lines_es else "\n\n")

            # Dynamic example block based on enabled sections
            example_lines_es: list[str] = []
            if enable_cc:
                example_lines_es.append("Motivo de Consulta: El paciente reporta dolor de cabeza severo por 3 días.")
            if enable_hpi:
                example_lines_es.append(
                    "HPI: El paciente describe una semana de dolores de cabeza persistentes que comienzan en la mañana "
                    "y empeoran durante el día, llegando hasta 8/10 en los últimos 3 días."
                )
            if enable_history:
                example_lines_es.append(
                    "Historia: Médica: hipertensión; Quirúrgica: colecistectomía hace cinco años; Estilo de vida: no fumador."
                )
            if enable_meds:
                example_lines_es.append(
                    "Medicación Actual: En medicamentos: lisinopril 10 mg diario e ibuprofeno según necesidad."
                )
            example_block_es = "\n".join(example_lines_es) + ("\n\n" if example_lines_es else "\n\n")

            # Dynamic guidelines text based on enabled sections
            guidelines_es_lines: list[str] = []
            if enable_cc:
                guidelines_es_lines.append(
                    "- Motivo de Consulta: Una línea en las propias palabras del paciente si está disponible."
                )
            if enable_hpi:
                guidelines_es_lines.append(
                    "- HPI: UN párrafo legible tejiendo OLDCARTS en prosa."
                )
            if enable_history:
                guidelines_es_lines.append(
                    "- Historia: Una línea combinando elementos médicos, quirúrgicos, familiares y de estilo de vida "
                    "(solo si 'Historia' está en los encabezados)."
                )
            if enable_ros:
                guidelines_es_lines.append(
                    "- Revisión de Sistemas: Una línea narrativa resumiendo positivos/negativos por sistemas "
                    "(solo si 'Revisión de Sistemas' está en los encabezados)."
                )
            if enable_meds:
                guidelines_es_lines.append(
                    "- Medicación Actual: Una línea narrativa con medicamentos/suplementos realmente declarados por el "
                    "paciente o mención de imágenes de medicamentos (solo si 'Medicación Actual' está en los encabezados)."
                )
            guidelines_text_es = "\n".join(guidelines_es_lines) + ("\n\n" if guidelines_es_lines else "\n\n")

            prompt = (
                "Rol y Tarea\n"
                "Eres un Asistente de Admisión Clínica.\n"
                "Tu tarea es generar un Resumen Pre-Consulta conciso y clínicamente útil (~180-200 palabras) "
                "basado estrictamente en las respuestas de admisión proporcionadas.\n\n"
                f"{prefs_snippet}"
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
                "- No incluyas secciones que NO estén presentes en la lista de encabezados proporcionada.\n"
                "- Usa frases orientadas al paciente: \"El paciente reporta...\", \"Niega...\", \"En medicamentos:...\".\n"
                "- No incluyas observaciones clínicas, diagnósticos, planes, signos vitales o hallazgos del examen "
                "(la pre-consulta es solo lo reportado por el paciente).\n"
                "- Normaliza pronunciaciones médicas obvias a términos correctos sin agregar nueva información.\n\n"
                "Encabezados (usa MAYÚSCULAS EXACTAS; incluye solo si tienes datos reales de las respuestas del paciente)\n"
                f"{headings_text_es}"
                "Pautas de Contenido por Sección (aplican solo a los encabezados listados arriba)\n"
                f"{guidelines_text_es}"
                "Ejemplo de Formato\n"
                "(Estructura y tono solamente—el contenido será diferente; cada sección en una sola línea.)\n"
                f"{example_block_es}"
                f"{f'Imágenes de Medicamentos: {medication_images_info}' if medication_images_info else ''}\n\n"
                f"Respuestas de Admisión:\n{self._format_intake_answers(intake_answers)}"
            )
        else:
            # Build dynamic English headings based on enabled sections
            headings_lines = []
            if enable_cc:
                headings_lines.append("Chief Complaint:")
            if enable_hpi:
                headings_lines.append("HPI:")
            if enable_history:
                headings_lines.append("History:")
            if enable_ros:
                headings_lines.append("Review of Systems:")
            if enable_meds:
                headings_lines.append("Current Medication:")
            headings_text = "\n".join(headings_lines) + ("\n\n" if headings_lines else "\n\n")

            # Dynamic example block based on enabled sections
            example_lines: list[str] = []
            if enable_cc:
                example_lines.append("Chief Complaint: Patient reports severe headache for 3 days.")
            if enable_hpi:
                example_lines.append(
                    "HPI: The patient describes a week of persistent headaches that begin in the morning and worsen through "
                    "the day, reaching up to 8/10 over the last 3 days."
                )
            if enable_history:
                example_lines.append(
                    "History: Medical: hypertension; Surgical: cholecystectomy five years ago; Lifestyle: non-smoker."
                )
            if enable_meds:
                example_lines.append(
                    "Current Medication: On meds: lisinopril 10 mg daily and ibuprofen as needed; allergies included only if "
                    "the patient explicitly stated them."
                )
            example_block = "\n".join(example_lines) + ("\n\n" if example_lines else "\n\n")

            # Dynamic guidelines text based on enabled sections
            guidelines_lines: list[str] = []
            if enable_cc:
                guidelines_lines.append(
                    "- Chief Complaint: One line in the patient's own words if available."
                )
            if enable_hpi:
                guidelines_lines.append(
                    "- HPI: ONE readable paragraph weaving OLDCARTS into prose (only if HPI is listed)."
                )
            if enable_history:
                guidelines_lines.append(
                    "- History: One line combining medical/surgical/family/lifestyle history (only if History is listed)."
                )
            if enable_ros:
                guidelines_lines.append(
                    "- Review of Systems: One narrative line summarizing system-based positives/negatives "
                    "(only if Review of Systems is listed)."
                )
            if enable_meds:
                guidelines_lines.append(
                    "- Current Medication: One narrative line with meds/supplements actually stated by the patient or "
                    "mention of medication images (only if Current Medication is listed)."
                )
            guidelines_text = "\n".join(guidelines_lines) + ("\n\n" if guidelines_lines else "\n\n")

            # Extra hard rule driven by doctor preferences:
            # if the Current Medication section is disabled, the LLM must not generate it at all.
            disabled_section_rules: list[str] = []
            if not enable_meds:
                disabled_section_rules.append(
                    "- Do NOT include any 'Current Medication' section or mention medications anywhere in the summary "
                    "when doctor preferences have this section disabled.\n"
                )
            disabled_section_rules_text = "".join(disabled_section_rules)

            prompt = (
                "Role & Task\n"
                "You are a Clinical Intake Assistant.\n"
                "Your task is to generate a concise, clinically useful Pre-Visit Summary (~180–200 words) based strictly on "
                "the provided intake responses.\n\n"
                f"{prefs_snippet}"
                "Critical Rules\n"
                "- Do not invent, guess, or expand beyond the provided input.\n"
                "- Output must be plain text with section headings, one section per line (no extra blank lines).\n"
                "- Use only the exact headings listed below. Do not add, rename, or reorder headings.\n"
                "- No bullets, numbering, or markdown formatting.\n"
                "- Write in a clinical handover tone: short, factual, deduplicated, and neutral.\n"
                "- Include a section ONLY if it contains actual content from the patient's responses.\n"
                "- Do not use placeholders like \"N/A\", \"Not provided\", \"not reported\", or \"denies\".\n"
                "- Do not include sections for topics that were not asked about or discussed.\n"
                "- Do NOT include sections that are not present in the headings list below (for example, omit 'History' if it is not listed).\n"
                "- Use patient-facing phrasing: \"Patient reports …\", \"Denies …\", \"On meds: …\".\n"
                "- Do not include clinician observations, diagnoses, plans, vitals, or exam findings "
                "(previsit is patient-reported only).\n"
                "- Normalize obvious medical mispronunciations to correct terms (e.g., \"diabities\" -> \"diabetes\") "
                 "without adding new information.\n"
                 f"{disabled_section_rules_text}\n"
                "Headings (use EXACT casing; include only if you have actual data from patient responses)\n"
                f"{headings_text}"
                "Content Guidelines per Section (apply only to the headings listed above)\n"
                f"{guidelines_text}"
                "Example Format\n"
                "(Structure and tone only—content will differ; each section on a single line.)\n"
                f"{example_block}"
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

            resp = await call_llm_with_telemetry(
                ai_client=self._client,
                scenario=PromptScenario.PREVISIT_SUMMARY,
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
                model=self._settings.openai.model,
                max_tokens=min(2000, self._settings.openai.max_tokens),
                temperature=0.1,
            )
            response_text = (resp.choices[0].message.content or "").strip()
            cleaned = self._clean_summary_markdown(response_text)

            # Post-process to hard-enforce disabled sections (History, Current Medication, etc.)
            cleaned = self._strip_disabled_sections(
                cleaned,
                lang=lang,
                enable_cc=enable_cc,
                enable_hpi=enable_hpi,
                enable_history=enable_history,
                enable_ros=enable_ros,
                enable_meds=enable_meds,
            )

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
            resp = await call_llm_with_telemetry(
                ai_client=self._client,
                scenario=PromptScenario.RED_FLAG,
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
                model=self._settings.openai.model,
                max_tokens=1000,
                temperature=0.1,
            )

            # Parse LLM response
            # TODO (2025-12): Consider using json.loads() directly with more robust error handling
            # instead of regex extraction, to handle edge cases like nested JSON or malformed responses
            response_text = (resp.choices[0].message.content or "").strip()
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
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

    def _strip_disabled_sections(
        self,
        summary: str,
        lang: str,
        enable_cc: bool,
        enable_hpi: bool,
        enable_history: bool,
        enable_ros: bool,
        enable_meds: bool,
    ) -> str:
        """
        Final safety net: remove any section headings that are disabled in doctor preferences.
        This protects against the LLM occasionally emitting a disallowed section.
        """
        if not summary:
            return summary

        is_spanish = self._normalize_language(lang) == "sp"
        lines = summary.splitlines()
        out_lines: List[str] = []

        for line in lines:
            stripped = line.lstrip()
            # English headings
            if not is_spanish:
                if not enable_cc and stripped.startswith("Chief Complaint:"):
                    continue
                if not enable_hpi and stripped.startswith("HPI:"):
                    continue
                if not enable_history and stripped.startswith("History:"):
                    continue
                if not enable_ros and stripped.startswith("Review of Systems:"):
                    continue
                if not enable_meds and stripped.startswith("Current Medication:"):
                    continue
            else:
                # Spanish headings
                if not enable_cc and stripped.startswith("Motivo de Consulta:"):
                    continue
                if not enable_hpi and stripped.startswith("HPI:"):
                    continue
                if not enable_history and stripped.startswith("Historia:"):
                    continue
                if not enable_ros and stripped.startswith("Revisión de Sistemas:"):
                    continue
                if not enable_meds and stripped.startswith("Medicación Actual:"):
                    continue

            out_lines.append(line)

        return "\n".join(out_lines)