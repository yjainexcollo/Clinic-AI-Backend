"""
OpenAI implementation of QuestionService for AI-powered question generation.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from clinicai.application.ports.services.question_service import QuestionService
from clinicai.core.config import get_settings


class OpenAIQuestionService(QuestionService):
    """OpenAI-based implementation of the QuestionService for intake question generation."""

    def __init__(self) -> None:
        import os
        from pathlib import Path

        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:
            load_dotenv = None

        self._settings = get_settings()
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

        self._client = OpenAI(api_key=api_key)

    async def _chat_completion(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        def _run() -> str:
            logger = logging.getLogger("clinicai")
            logger.info(
                "[QuestionService] Calling OpenAI chat.completions",
                extra={
                    "model": self._settings.openai.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            try:
                resp = self._client.chat.completions.create(
                    model=self._settings.openai.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = resp.choices[0].message.content.strip()
                logger.info("[QuestionService] OpenAI call succeeded")
                return text
            except Exception:
                logger.error("[QuestionService] OpenAI call failed", exc_info=True)
                raise

        return await asyncio.to_thread(_run)

    # ----------------------
    # Shared classifier
    # ----------------------
    def _classify_question(self, text: str) -> str:
        t = (text or "").lower().strip()

        if any(k in t for k in ["how long", "since when", "duration"]):
            return "duration"
        if any(k in t for k in ["trigger", "worse", "aggrav", "what makes", "factors", "relieving"]):
            return "triggers"
        if any(
            k in t
            for k in [
                "temporal",
                "time of day",
                "morning",
                "evening",
                "night",
                "seasonal",
                "cyclical",
                "timing",
                "what time",
            ]
        ):
            return "temporal"
        if "pain" in t or any(
            k in t
            for k in ["scale", "intensity", "sharp", "dull", "burning", "throbbing", "radiat", "spreads", "0-10"]
        ):
            return "pain"
        if any(k in t for k in ["travel", "traveled", "trip", "abroad", "sick contact", "visited"]):
            return "travel"
        if any(k in t for k in ["allerg", "hives", "rash", "swelling", "wheeze", "sneeze", "runny nose"]):
            return "allergies"
        if any(k in t for k in ["medicat", "treatment", "drug", "dose", "frequency", "remedy", "otc", "supplement"]):
            return "medications"
        if any(
            k in t
            for k in [
                "history of present illness",
                "hpi",
                "fever",
                "cough",
                "infection",
                "diabetes",
                "hypertension",
                "chest pain",
                "stomach pain",
                "abdominal pain",
            ]
        ):
            return "hpi"
        # Expanded family detection
        if any(
            k in t
            for k in [
                "family",
                "family history",
                "anyone in your family",
                "relatives",
                "parents",
                "siblings",
                "grandparents",
                "hereditary",
                "genetic",
                "runs in your family",
                "chronic conditions in family",
            ]
        ):
            return "family"
        if any(k in t for k in ["smoke", "alcohol", "diet", "exercise", "occupation", "work", "exposure", "routine"]):
            return "lifestyle"
        if any(k in t for k in ["pregnan", "gyneco", "obstet", "menstru", "period", "lmp", "vaginal"]):
            return "gyn"
        if any(k in t for k in ["functional", "mobility", "walk", "daily activity", "impair", "adl"]):
            return "functional"

        return "other"

    # ----------------------
    # Question generation
    # ----------------------
    async def generate_first_question(self, disease: str) -> str:
        return "Why have you come in today? What is the main concern you want help with?"

    async def generate_next_question(
        self,
        disease: str,
        previous_answers: List[str],
        asked_questions: List[str],
        current_count: int,
        max_count: int,
        recently_travelled: bool = False,
        prior_summary: Optional[Any] = None,
        prior_qas: Optional[List[str]] = None,
    ) -> str:
        mandatory_closing = "Have we missed anything important about your condition or any concerns you want us to address?"
        if current_count >= max_count - 1:
            already_asked = any(
                mandatory_closing.lower() in (q or "").lower() for q in (asked_questions or [])
            )
            if not already_asked:
                return mandatory_closing

        covered_categories = sorted({self._classify_question(q) for q in (asked_questions or []) if q})
        covered_categories_str = ", ".join(c for c in covered_categories if c and c != "other") or "none"

        # Travel pre-check (symptoms + travel flag)
        symptom_text = (f"{disease or ''} " + " ".join(previous_answers or [])).lower()
        travel_keywords = [
            "fever",
            "cough",
            "cold",
            "sore throat",
            "diarrhea",
            "vomit",
            "stomach",
            "abdomen",
            "rash",
            "jaundice",
            "infection",
        ]
        allow_travel = bool(recently_travelled) and any(k in symptom_text for k in travel_keywords)
        travel_already_covered = "travel" in {self._classify_question(q) for q in (asked_questions or []) if q}

        if allow_travel and not travel_already_covered:
            gi_related = any(k in symptom_text for k in ["diarrhea", "vomit", "stomach", "abdomen"])
            if gi_related:
                return (
                    "Have you travelled in the last 1–3 months? If yes, where did you go, did you eat street food or "
                    "raw/undercooked foods, drink untreated water, or did others with you have similar stomach symptoms?"
                )
            return (
                "Have you travelled domestically or internationally in the last 1–3 months? If yes, where did you go, "
                "was it a known infectious area, and did you have any sick contacts?"
            )

        # Build prior context block
        prior_block = ""
        if prior_summary:
            ps = str(prior_summary)
            prior_block += f"Prior summary: {ps[:400]}\n"
        if prior_qas:
            prior_block += "Prior QAs: " + "; ".join(prior_qas[:6]) + "\n"

        prompt = f"""
SYSTEM PROMPT:
You are a Clinical Intake Assistant.
Ask ONE and only ONE medically relevant question at a time.
Do not repeat or rephrase any question that is already in "Already asked" or "Covered categories."
CONTEXT:
- Chief complaint(s): {disease or "N/A"}
- Last 3 answers: {', '.join(previous_answers[-3:])}
- Already asked: {asked_questions}
- Covered categories: {covered_categories_str}
- Progress: {current_count}/{max_count}
- Recently travelled: {recently_travelled}
- Prior context: {prior_block or 'none'}
RULES:
1. Each category can only be asked ONCE.
2. Always advance to the NEXT relevant category in the sequence.
3. If a patient response is vague, mark category complete and move forward.
4. Never invent new categories.
5. Do NOT re-ask stable facts documented in Prior context unless the patient indicates a change.
CATEGORY RELEVANCE:
- Duration: always ask (once).
- Triggers: ask only if symptoms vary with food, activity, stress, or environment.
- Pain: ask only if complaint/answers mention pain terms.
- Temporal (timing/pattern): ask if timing patterns would clarify acute symptoms.
- Travel: ask if recently_travelled is true AND infection-like symptoms are present.
- Allergies: ask only if allergy terms are present.
- Medications: always ask (once).
- HPI (acute): ask if acute terms like fever, cough, diarrhea, infection are present.
- Family history: ask if chronic or hereditary disease is present (diabetes, hypertension, cancer, etc.).
- Lifestyle: ask if chronic metabolic/cardiovascular disease is present (diabetes, hypertension, obesity, thyroid).
- Gynecologic/obstetric: ask only if female age 10–60 with gyn/obstetric complaints.
- Functional status: ask if mobility, weakness, or neurologic limitation is implied.
NO-REPEAT RULES
- If a category or synonym was asked, mark it covered.
- If the answer was vague (“no, none, nothing, not sure”), mark it complete and move on.
- Treat synonyms as the same (e.g., "anyone in your family" = Family history).
SCENARIO FLOWS
1. Acute only → Duration → HPI → Travel (if eligible) → Medications → Allergies (if relevant) → Temporal.
2. Chronic only → Duration → Medications → Family history → Lifestyle.
3. Mixed (chronic + acute) → Duration → HPI → Travel → Medications → Family history → Lifestyle → Allergies → Temporal.
4. Pain-led → Duration → Pain → Triggers (if relevant) → Medications → HPI (if other acute terms).
SPECIAL RULE
If Travel is eligible, it must be asked within two turns after HPI (or immediately if HPI already asked).
STOPPING
- Stop if current_count ≥ max_count.
- Stop early if ≥6 questions and information is sufficient.
- Closing question: "Have we missed anything important about your condition or any concerns you want us to address?"
OUTPUT
Return only one patient-friendly question ending with "?".
No lists, no category names, no explanations.
"""

        try:
            text = await self._chat_completion(
                messages=[
                    {"role": "system", "content": "You are a clinical intake assistant. Follow instructions strictly."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=min(256, self._settings.openai.max_tokens),
                temperature=0.2,
            )
            text = text.replace("\n", " ").strip()
            if not text.endswith("?"):
                text = text.rstrip(".") + "?"

            # Normalize to canonical category for deduplication
            def _normalize(q: str) -> str:
                cat = self._classify_question(q)
                if cat and cat != "other":
                    return f"category::{cat}"
                t2 = (q or "").lower().strip()
                t2 = re.sub(r"[\?\.!,:;()\[\]\-]", " ", t2)
                t2 = re.sub(r"\s+", " ", t2)
                return t2

            asked_set = {_normalize(q) for q in (asked_questions or []) if q}
            generated_norm = _normalize(text)
            generated_category = self._classify_question(text)
            covered_categories_set = {self._classify_question(q) for q in (asked_questions or []) if q}

            if generated_norm in asked_set or (
                generated_category in covered_categories_set and generated_category != "other"
            ):
                return self._deterministic_next_question(
                    disease=disease,
                    asked_questions=asked_questions,
                    previous_answers=previous_answers,
                    recently_travelled=recently_travelled,
                )

            return text
        except Exception:
            raise

    def _deterministic_next_question(
        self,
        disease: str,
        asked_questions: List[str],
        previous_answers: List[str],
        recently_travelled: bool,
    ) -> str:
        covered = {self._classify_question(q) for q in (asked_questions or []) if q}
        sequence = [
            "duration",
            "triggers",
            "pain",
            "temporal",
            "travel",
            "allergies",
            "medications",
            "hpi",
            "family",
            "lifestyle",
            "gyn",
            "functional",
        ]

        def relevant(key: str) -> bool:
            if key == "travel":
                combined_text = f"{disease or ''} " + " ".join(previous_answers or [])
                infection_terms = [
                    "fever",
                    "cough",
                    "cold",
                    "stomach",
                    "abdominal",
                    "pain",
                    "infection",
                    "vomit",
                    "diarrhea",
                    "rash",
                    "jaundice",
                ]
                return bool(recently_travelled) and any(term in combined_text.lower() for term in infection_terms)
            return True

        for key in sequence:
            if key in covered or not relevant(key):
                continue
            if key == "duration":
                return "How long have you been experiencing these symptoms?"
            if key == "triggers":
                return "What makes your symptoms better or worse, such as activity, food, or stress?"
            if key == "pain":
                return "If you have any pain, where is it located and how severe is it on a 0-10 scale?"
            if key == "temporal":
                return "Do your symptoms vary by time of day or season (morning/evening or cyclical)?"
            if key == "travel":
                combined_text = f"{disease or ''} " + " ".join(previous_answers or [])
                gi_related = any(k in combined_text.lower() for k in ["diarrhea", "vomit", "stomach", "abdomen"])
                if gi_related:
                    return (
                        "Have you travelled in the last 1–3 months? If yes, where did you go, did you eat street food or "
                        "raw/undercooked foods, drink untreated water, or did others with you have similar stomach symptoms?"
                    )
                return (
                    "Have you travelled domestically or internationally in the last 1–3 months? If yes, where did you go, "
                    "was it a known infectious area, and did you have any sick contacts?"
                )
            if key == "allergies":
                return "Do you have any allergies such as rash, hives, swelling, wheeze, sneezing, or runny nose?"
            if key == "medications":
                return "What medications or remedies are you currently taking, including OTC or supplements?"
            if key == "hpi":
                return "Have you had any recent acute symptoms like fever, cough, chest pain, stomach pain, or infection?"
            if key == "family":
                return "Does anyone in your family have diabetes, hypertension, heart disease, cancer, or similar conditions?"
            if key == "lifestyle":
                return "Are there any lifestyle factors—diet, exercise, occupation, or exposures—relevant to your symptoms?"
            if key == "gyn":
                return "If applicable, when was your last menstrual period and are your cycles regular?"
            if key == "functional":
                return "Do your symptoms affect your daily activities or mobility?"

        return "Have we missed anything important about your condition or any concerns you want us to address?"

    async def should_stop_asking(
        self,
        disease: str,
        previous_answers: List[str],
        current_count: int,
        max_count: int,
        recently_travelled: bool = False,
    ) -> bool:
        if current_count >= max_count:
            return True
        if current_count < 6:
            return False
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
            covered = {self._classify_question(q) for q in (asked_questions or [])}
            key_set = {
                "duration",
                "triggers",
                "temporal",
                "pain",
                "travel",
                "allergies",
                "medications",
                "hpi",
                "family",
                "lifestyle",
                "gyn",
                "functional",
            }
            covered_keys = len(covered & key_set)
            coverage_ratio = covered_keys / len(key_set)
            progress_ratio = 0.0
            if max_count > 0:
                progress_ratio = min(max(current_count / max_count, 0.0), 1.0)
            score = (0.7 * coverage_ratio + 0.3 * progress_ratio) * 100.0
            return max(0, min(int(round(score)), 100))
        except Exception:
            if max_count <= 0:
                return 0
            return max(0, min(int(round((current_count / max_count) * 100.0)), 100))

    def is_medication_question(self, question: str) -> bool:
        return self._classify_question(question) == "medications"

    # ----------------------
    # Pre-visit summary
    # ----------------------
    async def generate_pre_visit_summary(
        self,
        patient_data: Dict[str, Any],
        intake_answers: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate pre-visit clinical summary from intake data (concise bullets)."""
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
            "- Include a section only if it contains content; omit sections with no data.\n"
            "- Do not use placeholders like \"N/A\" or \"Not provided\".\n"
            "- Use patient-facing phrasing: \"Patient reports …\", \"Denies …\", \"On meds: …\".\n"
            "- Do not include clinician observations, diagnoses, plans, vitals, or exam findings (previsit is patient-reported only).\n"
            "- Normalize obvious medical mispronunciations to correct terms (e.g., \"die-a-bee-tees mellitis\" → \"diabetes mellitus\") without adding new information.\n\n"
            "Headings (use EXACT casing; include only if you have data)\n"
            "Chief Complaint:\n"
            "HPI:\n"
            "History:\n"
            "Current Medication:\n\n"
            "Content Guidelines per Section\n"
            "- Chief Complaint: One line in the patient’s own words if available.\n"
            "- HPI: ONE readable paragraph weaving OLDCARTS into prose:\n"
            "  Onset, Location, Duration, Characterization/quality, Aggravating factors, Relieving factors, Radiation,\n"
            "  Temporal pattern, Severity (1–10), Associated symptoms, Relevant negatives.\n"
            "  Keep it natural and coherent (e.g., \"The patient reports …\"). If some OLDCARTS elements are unknown, simply omit them (do not write placeholders).\n"
            "- History: One line combining any patient-reported items using semicolons in this order if present:\n"
            "  Medical: …; Surgical: …; Family: …; Lifestyle: …\n"
            "  (Include only parts provided by the patient; omit absent parts entirely.)\n"
            "- Review of Systems: One narrative line summarizing system-based positives/negatives explicitly mentioned by the patient (e.g., General, Neuro, Eyes, Resp, GI). Keep as prose, not a list.\n"
            "- Current Medication: One narrative line with meds/supplements actually stated by the patient (name/dose/frequency if provided). Include allergy statements only if the patient explicitly reported them.\n\n"
            "Example Format\n"
            "(Structure and tone only—content will differ; each section on a single line.)\n"
            "Chief Complaint: Patient reports severe headache for 3 days.\n"
            "HPI: The patient describes a week of persistent headaches that begin in the morning and worsen through the day, reaching up to 8/10 over the last 3 days. Pain is over both temples and feels different from prior migraines; fatigue is prominent and nausea is denied. Episodes are aggravated by stress and later in the day, with minimal relief from over-the-counter analgesics and some relief using cold compresses. No radiation is reported, evenings are typically worse, and there have been no recent changes in medications or lifestyle.\n"
            "History: Medical: hypertension; Surgical: cholecystectomy five years ago; Family: not reported; Lifestyle: non-smoker, occasional alcohol, high-stress job.\n"
            "Current Medication: On meds: lisinopril 10 mg daily and ibuprofen as needed; allergies included only if the patient explicitly stated them.\n\n"
            f"Intake Responses:\n{self._format_intake_answers(intake_answers)}"
        )

        try:
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
            }
        except Exception:
            return self._generate_fallback_summary(patient_data, intake_answers)

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

    def _generate_fallback_summary(self, patient_data: Dict[str, Any], intake_answers: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic fallback summary."""
        return {
            "summary": f"Pre-visit summary for {patient_data.get('name', 'Patient')}",
            "structured_data": {
                "chief_complaint": patient_data.get("symptom") or patient_data.get("complaint") or "N/A",
                "key_findings": ["See intake responses"],
            },
        }

    def _normalize_summary_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure result contains 'summary' and 'structured_data' keys with sane defaults."""
        if not isinstance(result, dict):
            return self._generate_fallback_summary({}, {})

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

        return {"summary": summary, "structured_data": structured}

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
