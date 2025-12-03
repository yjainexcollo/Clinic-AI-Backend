import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from clinicai.adapters.external.question_service_openai import OpenAIQuestionService
from clinicai.core.ai_factory import get_ai_client
from clinicai.core.config import get_settings


# -----------------------------------------------------------------------------
# Scenario & evaluation models
# -----------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    language: str  # "en" or "sp"
    patient_age: int
    patient_gender: str  # "male" | "female" | "other"
    chief_complaint: str
    recently_travelled: bool = False
    travel_questions_count: int = 0
    initial_answer: Optional[str] = None  # Answer to the very first question
    # Optional scripted answers for subsequent questions (Q2, Q3, ...)
    # Indexing: index 0 corresponds to Q1 (if initial_answer is not provided),
    # index 1 to Q2, etc.
    scripted_answers: List[str] = field(default_factory=list)
    # How many questions to allow in this scenario (so we can see deep-diagnostic mode)
    max_questions: int = 8


@dataclass
class QuestionEvaluation:
    rating: float
    coverage_ok: bool
    safety_ok: bool
    redundancy_ok: bool
    summary: str


# -----------------------------------------------------------------------------
# Scenarios (25 total)
# -----------------------------------------------------------------------------


SCENARIOS: List[Scenario] = [
    # 1–3 Stable chronic conditions
    Scenario(
        name="Chronic - Type 2 diabetes follow-up with blurred vision",
        language="en",
        patient_age=58,
        patient_gender="female",
        chief_complaint="Type 2 diabetes follow-up, noticing episodes of blurred vision recently",
        # Example scripted conversation for more realistic testing
        scripted_answers=[
            "I'm here for a follow-up on my type 2 diabetes and because I've been having some episodes of blurred vision.",
            "I first noticed the blurred vision about two weeks ago.",
            "It happens a few times a day, not every single day but most days.",
            "It usually lasts a few minutes and then clears up on its own.",
            "It feels like things are slightly out of focus, especially when I'm reading or looking at my phone.",
            "Sometimes I also get a mild headache when the blurred vision happens, but no other major symptoms.",
            "I'm taking metformin twice a day and a blood pressure pill every morning.",
            "Yes, I'm okay with answering some more detailed diagnostic questions if it helps.",
        ],
        # Allow room for deep-diagnostic questions after consent
        max_questions=12,
    ),
    Scenario(
        name="Chronic - Hypertension follow-up with mild headache",
        language="en",
        patient_age=62,
        patient_gender="male",
        chief_complaint="Hypertension follow-up, mild intermittent headaches",
    ),
    Scenario(
        name="Chronic - Asthma with occasional wheeze",
        language="en",
        patient_age=40,
        patient_gender="female",
        chief_complaint="Known asthma, recently more frequent episodes of wheezing",
    ),

    # 4–7 Acute high-risk complaints (non-trauma)
    Scenario(
        name="Acute high-risk - Sudden chest pain in 55M",
        language="en",
        patient_age=55,
        patient_gender="male",
        chief_complaint="Sudden chest pain that started 30 minutes ago while walking",
    ),
    Scenario(
        name="Acute high-risk - Unilateral weakness and slurred speech",
        language="en",
        patient_age=68,
        patient_gender="female",
        chief_complaint="New left arm weakness and slurred speech since this morning",
    ),
    Scenario(
        name="Acute high-risk - Dyspnea on exertion in older adult",
        language="en",
        patient_age=72,
        patient_gender="male",
        chief_complaint="Shortness of breath on exertion over the last week",
    ),
    Scenario(
        name="Acute high-risk - High fever with neck stiffness",
        language="en",
        patient_age=27,
        patient_gender="female",
        chief_complaint="High fever and neck stiffness since yesterday",
    ),

    # 8–11 Acute moderate/low-risk complaints
    Scenario(
        name="Acute low-risk - Sore throat and fever in young adult",
        language="en",
        patient_age=23,
        patient_gender="female",
        chief_complaint="Sore throat and low-grade fever for 2 days",
        scripted_answers=[
            "My main concern is a sore throat and low-grade fever I've had for the last two days.",
            "The sore throat and fever both started about two days ago.",
            "It's mostly a constant sore throat that gets a bit worse when I swallow.",
            "I've taken some ibuprofen and used throat lozenges, which help a little.",
            "I don't have any serious medical conditions and I'm not on any regular prescription medications.",
            "I don't have any allergies that I know of.",
            "I haven't noticed anything like trouble breathing, chest pain, or a stiff neck.",
            "Yes, I'm okay with answering more detailed questions if needed.",
        ],
        max_questions=12,
    ),
    Scenario(
        name="Acute moderate - Low back pain after lifting",
        language="en",
        patient_age=35,
        patient_gender="male",
        chief_complaint="Acute low back pain after lifting a heavy box yesterday",
    ),
    Scenario(
        name="Acute moderate - Viral gastroenteritis symptoms",
        language="en",
        patient_age=30,
        patient_gender="female",
        chief_complaint="Vomiting and diarrhea with abdominal cramps since last night",
    ),
    Scenario(
        name="Acute moderate - Simple UTI symptoms",
        language="en",
        patient_age=29,
        patient_gender="female",
        chief_complaint="Burning with urination and urinary frequency for 3 days",
    ),

    # 12–14 Travel-related scenarios
    Scenario(
        name="Travel-related - Dengue-endemic fever and body aches",
        language="en",
        patient_age=34,
        patient_gender="male",
        chief_complaint="Fever and severe body aches after returning from a dengue-endemic area",
        recently_travelled=True,
    ),
    Scenario(
        name="Travel-related - Traveler's diarrhea",
        language="en",
        patient_age=41,
        patient_gender="female",
        chief_complaint="Diarrhea and abdominal cramps after recent international travel",
        recently_travelled=True,
    ),
    Scenario(
        name="Travel but non-travel complaint - Ankle sprain",
        language="en",
        patient_age=38,
        patient_gender="male",
        chief_complaint="Sprained ankle from missing a step yesterday; just returned from travel but no infection symptoms",
        recently_travelled=True,
    ),

    # 15–19 Menstrual / reproductive (including edge cases)
    Scenario(
        name="Menstrual - Irregular periods in 25F",
        language="en",
        patient_age=25,
        patient_gender="female",
        chief_complaint="Irregular menstrual periods over the last 6 months",
    ),
    Scenario(
        name="Menstrual - Heavy menstrual bleeding in 35F",
        language="en",
        patient_age=35,
        patient_gender="female",
        chief_complaint="Heavy menstrual bleeding with clots each month",
    ),
    Scenario(
        name="Reproductive - Early pregnancy symptoms",
        language="en",
        patient_age=29,
        patient_gender="female",
        chief_complaint="Missed period and morning nausea, possible early pregnancy",
    ),
    Scenario(
        name="Reproductive - Pelvic pain in reproductive-age female",
        language="en",
        patient_age=32,
        patient_gender="female",
        chief_complaint="Pelvic pain and discomfort during intercourse",
    ),
    Scenario(
        name="Menstrual edge - Male with abdominal pain (no menstrual questions)",
        language="en",
        patient_age=30,
        patient_gender="male",
        chief_complaint="Intermittent lower abdominal pain",
    ),
    Scenario(
        name="Menstrual edge - Girl <12 with abdominal pain",
        language="en",
        patient_age=10,
        patient_gender="female",
        chief_complaint="Stomach pain and low appetite",
    ),
    Scenario(
        name="Menstrual edge - Woman >60 with abdominal pain",
        language="en",
        patient_age=68,
        patient_gender="female",
        chief_complaint="Lower abdominal discomfort and bloating",
    ),

    # 20–22 Pediatric
    Scenario(
        name="Pediatric - 8-year-old with cough and fever",
        language="en",
        patient_age=8,
        patient_gender="male",
        chief_complaint="Cough and fever for 3 days",
    ),
    Scenario(
        name="Pediatric - 3-year-old with vomiting and poor intake",
        language="en",
        patient_age=3,
        patient_gender="female",
        chief_complaint="Vomiting and not eating well for 1 day",
    ),
    Scenario(
        name="Pediatric - Adolescent with sports-related knee pain",
        language="en",
        patient_age=15,
        patient_gender="male",
        chief_complaint="Right knee pain after soccer game injury",
    ),

    # 23–24 Geriatric / multi-morbid
    Scenario(
        name="Geriatric - CKD and fatigue",
        language="en",
        patient_age=72,
        patient_gender="female",
        chief_complaint="Chronic kidney disease follow-up with increasing fatigue",
    ),
    Scenario(
        name="Geriatric - CHF and leg swelling",
        language="en",
        patient_age=68,
        patient_gender="male",
        chief_complaint="Chronic heart failure with worsening leg swelling and shortness of breath when lying flat",
    ),

    # 25–26 Non-medical / mental health / lifestyle (include 1 ES scenario)
    Scenario(
        name="Mental health - Anxiety and panic symptoms",
        language="en",
        patient_age=30,
        patient_gender="female",
        chief_complaint="Episodes of anxiety with palpitations and feeling like I can't breathe, but normal tests so far",
    ),
    Scenario(
        name="Lifestyle / sleep - Insomnia in working adult (Spanish)",
        language="sp",
        patient_age=37,
        patient_gender="male",
        chief_complaint="Dificultad para dormir y despertarse varias veces en la noche por estrés laboral",
    ),
]

# Ensure exactly 25 (trim to 25 if we accidentally added 26)
SCENARIOS = SCENARIOS[:25]


# -----------------------------------------------------------------------------
# Helpers to run the intake engine
# -----------------------------------------------------------------------------


CLOSING_EN = "Is there anything else you'd like to share about your condition?"
CLOSING_SP = "¿Hay algo más que le gustaría compartir sobre su condición?"


async def run_intake_for_scenario(
    service: OpenAIQuestionService,
    scenario: Scenario,
) -> List[tuple[str, str]]:
    """Run the full intake Q&A loop for a scenario and return (question, answer) pairs."""
    language = scenario.language
    disease = scenario.chief_complaint

    asked_questions: List[str] = []
    previous_answers: List[str] = []

    # First question
    first_q = await service.generate_first_question(disease=disease, language=language)
    asked_questions.append(first_q)

    # Simple plausible first answer (scripted if provided)
    if scenario.initial_answer is not None:
        first_a = scenario.initial_answer
    elif scenario.scripted_answers:
        # Use scripted answer[0] if present
        first_a = scenario.scripted_answers[0]
    else:
        if language == "sp":
            first_a = f"Vengo porque {scenario.chief_complaint.lower()}."
        else:
            first_a = f"My main concern is {scenario.chief_complaint.lower()}."
    previous_answers.append(first_a)

    max_questions = max(1, scenario.max_questions or 8)

    # Subsequent questions
    for _ in range(max_questions - 1):
        current_count = len(asked_questions)

        # Stop early if we already reached a closing question
        last_q = asked_questions[-1]
        if last_q.strip() in (CLOSING_EN, CLOSING_SP):
            break

        next_q = await service.generate_next_question(
            disease=disease,
            previous_answers=previous_answers,
            asked_questions=asked_questions,
            current_count=current_count,
            max_count=max_questions,
            asked_categories=None,
            recently_travelled=scenario.recently_travelled,
            travel_questions_count=scenario.travel_questions_count,
            prior_summary=None,
            prior_qas=None,
            patient_gender=scenario.patient_gender,
            patient_age=scenario.patient_age,
            language=language,
        )

        asked_questions.append(next_q)

        # Use scripted answer if provided; otherwise a generic placeholder
        # previous_answers already has one entry (Q1), so current index is len(previous_answers)
        answer_idx = len(previous_answers)
        if scenario.scripted_answers and answer_idx < len(scenario.scripted_answers):
            answer_text = scenario.scripted_answers[answer_idx]
        else:
            if language == "sp":
                answer_text = "El paciente proporciona una respuesta breve relacionada con la pregunta."
            else:
                answer_text = "The patient provides a brief answer related to the question."
        previous_answers.append(answer_text)

        # Stop if we hit closing question
        if next_q.strip() in (CLOSING_EN, CLOSING_SP):
            break

    # Return paired questions and answers for richer inspection
    return list(zip(asked_questions, previous_answers))


# -----------------------------------------------------------------------------
# LLM-based question quality evaluator
# -----------------------------------------------------------------------------


async def evaluate_questions_with_llm(
    scenario: Scenario,
    questions: List[str],
) -> QuestionEvaluation:
    """Use an LLM to rate question quality for a given scenario."""
    settings = get_settings()
    client = get_ai_client()

    # Build scenario description
    scenario_desc = {
        "name": scenario.name,
        "language": scenario.language,
        "patient_age": scenario.patient_age,
        "patient_gender": scenario.patient_gender,
        "chief_complaint": scenario.chief_complaint,
    }

    system_prompt = (
        "You are a senior clinician and clinical question design reviewer.\n"
        "You will be given:\n"
        "- A brief patient scenario (age, gender, chief complaint).\n"
        "- A list of intake questions generated by a system.\n\n"
        "Your tasks:\n"
        "1. Judge whether the questions are medically appropriate, safe, and relevant.\n"
        "2. Check if they:\n"
        "   - Cover key aspects (onset/duration, severity, associated symptoms, red flags,\n"
        "     past medical history/medications when needed, travel/menstrual only when relevant).\n"
        "   - Avoid redundant or near-duplicate questions.\n"
        "   - Respect obvious safety rules (no menstrual questions for male/<12/>60; no irrelevant heavy travel questions;\n"
        "     no invasive diagnostic interrogation without context; etc.).\n"
        "3. Assign an overall score from 0 to 10 (0 = completely inappropriate; 10 = excellent clinic-grade intake).\n"
        "4. Briefly explain the reasoning.\n\n"
        "IMPORTANT:\n"
        "- Do NOT change or invent new questions.\n"
        "- Only judge the quality of the questions provided.\n"
        "- Be conservative about safety: if in doubt, mark safety_ok=false.\n"
        "- Return STRICT JSON only, with keys: rating, coverage_ok, safety_ok, redundancy_ok, summary.\n"
    )

    user_content = {
        "scenario": scenario_desc,
        "questions": questions,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]

    try:
        resp = await client.chat(
            model=settings.openai.model,
            messages=messages,
            max_tokens=256,
            temperature=0.1,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Try to extract JSON
        json_match = None
        try:
            # Look for JSON object in response
            json_match = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to find first {...} block
            import re

            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                json_match = json.loads(m.group())

        if not isinstance(json_match, dict):
            raise ValueError("Evaluator did not return a JSON object")

        rating = float(json_match.get("rating", 0.0))
        coverage_ok = bool(json_match.get("coverage_ok", False))
        safety_ok = bool(json_match.get("safety_ok", False))
        redundancy_ok = bool(json_match.get("redundancy_ok", False))
        summary = str(json_match.get("summary", "") or "")

        return QuestionEvaluation(
            rating=rating,
            coverage_ok=coverage_ok,
            safety_ok=safety_ok,
            redundancy_ok=redundancy_ok,
            summary=summary,
        )
    except Exception as e:
        # On any failure, log and return a conservative low score
        print(f"[Evaluator] Failed for scenario '{scenario.name}': {e}")
        return QuestionEvaluation(
            rating=0.0,
            coverage_ok=False,
            safety_ok=False,
            redundancy_ok=False,
            summary="Evaluation failed or returned invalid JSON.",
        )


# -----------------------------------------------------------------------------
# Pytest entrypoint
# -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
async def test_question_quality_across_25_scenarios() -> None:
    """Integration test: run multi-agent intake and LLM evaluator across 25 scenarios."""
    service = OpenAIQuestionService()

    results: List[tuple[Scenario, List[tuple[str, str]], QuestionEvaluation]] = []

    for scenario in SCENARIOS:
        # Progress log so long-running runs show where we are
        print(
            f"\n=== Running scenario: {scenario.name} "
            f"(lang={scenario.language}, age={scenario.patient_age}, gender={scenario.patient_gender}) ==="
        )
        qa_pairs = await run_intake_for_scenario(service, scenario)
        questions_only = [q for q, _ in qa_pairs]
        eval_result = await evaluate_questions_with_llm(scenario, questions_only)
        results.append((scenario, qa_pairs, eval_result))

        # Show generated questions and answers for this scenario
        print("Generated Q&A:")
        for idx, (q, a) in enumerate(qa_pairs, start=1):
            print(f"  Q{idx}: {q}")
            print(f"  A{idx}: {a}")

        # Show evaluation summary
        print(
            f"Evaluation -> rating={eval_result.rating:.2f}, "
            f"coverage_ok={eval_result.coverage_ok}, "
            f"safety_ok={eval_result.safety_ok}, "
            f"redundancy_ok={eval_result.redundancy_ok}"
        )
        print(f"Summary: {eval_result.summary}\n")

    # Print a compact table
    header = (
        "Scenario".ljust(55)
        + " | Lang | Rating | Coverage | Safety | Redundancy"
    )
    print("\n" + header)
    print("-" * len(header))
    for scenario, _qa_pairs, ev in results:
        row = (
            scenario.name[:55].ljust(55)
            + f" | {scenario.language:4}"
            + f" | {ev.rating:6.2f}"
            + f" | {str(ev.coverage_ok):8}"
            + f" | {str(ev.safety_ok):6}"
            + f" | {str(ev.redundancy_ok):10}"
        )
        print(row)

    # Soft assertion: most non-edge cases should be at least mildly acceptable
    non_edge_results = [
        ev.rating
        for scenario, _qa_pairs, ev in results
        if "edge" not in scenario.name.lower()
    ]
    if non_edge_results:
        avg_rating = sum(non_edge_results) / len(non_edge_results)
        # Do not fail the suite aggressively yet, but ensure we don't regress to very low quality overall
        assert avg_rating >= 5.0, (
            f"Average rating across non-edge scenarios is low ({avg_rating:.2f}); "
            "question quality may have regressed."
        )


