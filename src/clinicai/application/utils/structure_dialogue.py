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


