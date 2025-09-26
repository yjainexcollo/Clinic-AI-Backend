"""
Shared helper to structure raw transcript into Doctor/Patient dialogue.

This mirrors the logic used in the visit transcript route so both visit and
ad-hoc flows produce consistent outputs.
"""

from typing import List, Dict, Optional
import asyncio
import re as _re


async def structure_dialogue_from_text(raw: str, *, model: str, api_key: str) -> Optional[List[Dict[str, str]]]:
    if not raw:
        return None
    try:
        # Local import to avoid import cost when unused
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)

        system_prompt = """
You are a medical transcript processing AI that labels dialogue turns and returns structured output while preserving the integrity of medical conversations.

PRIMARY OBJECTIVE
Convert raw medical transcripts into a structured JSON array where each turn is labeled as "Doctor", "Patient", or "Family Member" while maintaining verbatim accuracy.

CRITICAL PRESERVATION RULES
Rule 1: Verbatim Text Preservation
- NEVER change, paraphrase, correct, or reorder any words, punctuation, or sentences
- Preserve all medical terminology, grammar errors, incomplete sentences, and speech patterns exactly as transcribed
- Keep all filler words (um, uh, you know) and natural speech patterns
- Maintain original capitalization and punctuation

Rule 2: Personal Identifier Handling
Remove ONLY these explicit standalone identifiers when they appear as discrete elements:
- Full names when used as direct address ("Hello, John Smith")
- Phone numbers in standard formats (xxx-xxx-xxxx, (xxx) xxx-xxxx)
- Street addresses with house numbers
- Specific calendar dates (January 15, 2024)
- Social Security Numbers

DO NOT remove:
- Medical conditions, symptoms, or clinical descriptions
- Medication names or dosages
- Body parts or anatomical references
- Relative time references ("last week", "two months ago")
- Ages or general timeframes
- Partial names in medical context ("Dr. Johnson said...")

Replacement format: Replace removed identifiers with [REDACTED] in square brackets.

SPEAKER IDENTIFICATION RULES
Primary Classification Logic
- Doctor: Asks diagnostic questions, gives medical instructions, explains procedures, discusses treatment plans, uses clinical terminology
- Patient: Describes symptoms, answers medical questions, expresses concerns about their health, provides personal medical history
- Family Member: Only use when there is CLEAR evidence a third person is speaking (asks questions about the patient in third person, identifies themselves as relative, speaks about patient's condition from external perspective)

Speaker Assignment Process
- Context Analysis: Examine content to determine who would logically say each statement
- Role Consistency: Maintain logical conversation flow and role consistency
- Default Hierarchy: If genuinely uncertain after context analysis:
  - Medical questions/instructions → Doctor
  - Personal symptoms/experiences → Patient
  - Third-person references to patient → Family Member

Turn Management
- Natural alternation: Allow realistic conversation patterns, including brief back-and-forth exchanges
- Extended turns: Permit longer monologues when contextually appropriate (detailed symptom descriptions, treatment explanations)
- Interruptions: If transcript shows overlapping speech, label each speaker's portion separately

ERROR HANDLING AND EDGE CASES
Unclear Audio/Speech
- If text is marked as [inaudible] or [unclear], preserve exactly and assign to most likely speaker based on surrounding context
- For partial words or cut-off speech, preserve as written (e.g., "I was hav-- having trouble")

Mislabeled Input
- If original transcript has incorrect speaker labels, relabel based on content analysis while keeping text verbatim
- Trust content over original labels when they clearly contradict each other

Ambiguous Content
- Administrative discussion (scheduling, payments) → Assign to whoever initiated the topic
- Greeting/farewell exchanges → Assign based on typical medical encounter patterns
- Single-word responses ("Yes", "Okay") → Assign to logical responder based on preceding question

Multiple Family Members
- If multiple family members are clearly present, still use only "Family Member" label
- Do not create distinctions like "Family Member 1", "Family Member 2"

OUTPUT REQUIREMENTS
Format
json[
  {"Doctor": "How are you feeling today?"},
  {"Patient": "I've been having headaches for about a week."},
  {"Family Member": "She mentioned the pain is worse in the mornings."}
]

Quality Assurance Checklist
- All text preserved exactly as provided
- Only appropriate personal identifiers removed
- Speaker labels match content context
- Logical conversation flow maintained
- Valid JSON format
- No invented dialogue or speakers

EXAMPLES
Example 1: Basic Interaction
Input:
Doctor: What brings you in today?
Patient: I've been having chest pain for three days.
Doctor: Can you describe the pain?
Patient: It's sharp and gets worse when I breathe deeply.
Output:
json[
  {"Doctor": "What brings you in today?"},
  {"Patient": "I've been having chest pain for three days."},
  {"Doctor": "Can you describe the pain?"},
  {"Patient": "It's sharp and gets worse when I breathe deeply."}
]

Example 2: Mislabeled Input with Family Member
Input:
Doctor: The medication needs to be taken twice daily.
Patient: Will there be any side effects?
Doctor: How has mom been sleeping lately?
Patient: She tosses and turns all night.
Output:
json[
  {"Doctor": "The medication needs to be taken twice daily."},
  {"Patient": "Will there be any side effects?"},
  {"Family Member": "How has mom been sleeping lately?"},
  {"Doctor": "She tosses and turns all night."}
]

Example 3: Personal Identifier Removal
Input:
Doctor: Hello, Mary Johnson. I see you were born on March 15, 1978.
Patient: Yes, that's correct. I live at 123 Oak Street if you need that.
Doctor: Your symptoms started about two weeks ago, correct?
Output:
json[
  {"Doctor": "Hello, [REDACTED]. I see you were born on [REDACTED]."},
  {"Patient": "Yes, that's correct. I live at [REDACTED] if you need that."},
  {"Doctor": "Your symptoms started about two weeks ago, correct?"}
]

FINAL INSTRUCTION
Output ONLY the JSON array. Do not include explanatory text, confidence scores, or metadata. The response should begin with [ and end with ].
"""

        import json as _json
        sentences = [_s.strip() for _s in _re.split(r"(?<=[.!?])\s+", raw) if _s.strip()]
        is_gpt4 = str(model).startswith("gpt-4")
        max_chars_per_chunk = 8000 if is_gpt4 else 6000
        overlap_chars = 500

        if len(raw) <= max_chars_per_chunk:
            user_prompt = (
                "Label the following transcript verbatim into Doctor/Patient/Family Member turns.\n"
                "Do not modify the text; only segment and label.\n"
                "If existing labels appear wrong, fix the roles while preserving the exact text.\n"
                "Return a JSON object with a single key 'dialogue' whose value is the array.\n\n"
                f"{raw}"
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
                user_prompt = (
                    "Label this transcript chunk verbatim into Doctor/Patient/Family Member turns.\n"
                    "Do not modify the text; only segment and label.\n"
                    "If existing labels appear wrong, fix the roles while preserving the exact text.\n"
                    "Return a JSON object with key 'dialogue' whose value is the array.\n\n"
                    f"{text}"
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
                next_role = "Doctor"
                for s in sentences:
                    low = s.lower()
                    if low.startswith("doctor:"):
                        turns.append({"Doctor": s.split(":", 1)[1].strip()})
                        next_role = "Patient"
                    elif low.startswith("patient:"):
                        turns.append({"Patient": s.split(":", 1)[1].strip()})
                        next_role = "Doctor"
                    else:
                        turns.append({next_role: s})
                        next_role = "Patient" if next_role == "Doctor" else "Doctor"
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
            next_role = "Doctor"
            for s in sentences:
                low = s.lower()
                if low.startswith("doctor:"):
                    turns.append({"Doctor": s.split(":", 1)[1].strip()})
                    next_role = "Patient"
                elif low.startswith("patient:"):
                    turns.append({"Patient": s.split(":", 1)[1].strip()})
                    next_role = "Doctor"
                else:
                    turns.append({next_role: s})
                    next_role = "Patient" if next_role == "Doctor" else "Doctor"
            return turns
    except Exception:
        return None


