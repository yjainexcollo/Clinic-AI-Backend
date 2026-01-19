# Clinic-AI: LLM Usage and Prompts Documentation

**Document Type:** Final Closure Documentation  
**Project:** Clinic-AI  
**Date:** December 2024  
**Audience:** Engineering Leads, Product/RM Stakeholders, Future Developers

---

## Table of Contents

1. [Project Overview (LLM Perspective)](#1-project-overview-llm-perspective)
2. [Inventory of LLM Prompts](#2-inventory-of-llm-prompts)
3. [Detailed Prompt Breakdown](#3-detailed-prompt-breakdown)
4. [Multilingual Strategy](#4-multilingual-strategy)
5. [Safety, Compliance & Guardrails](#5-safety-compliance--guardrails)
6. [Prompt Evaluation & Quality Control](#6-prompt-evaluation--quality-control)
7. [Known Limitations & Future Improvements](#7-known-limitations--future-improvements)

---

## 1. Project Overview (LLM Perspective)

### What Clinic-AI Does

Clinic-AI is a clinical intake and documentation system that uses Large Language Models (LLMs) to automate patient intake interviews, generate clinical summaries, and produce structured medical documentation (SOAP notes). The system operates as a multi-agent architecture where specialized LLM agents collaborate to conduct intelligent medical interviews, extract structured information, and generate clinician-ready documentation.

### Why LLMs Are Used

LLMs are essential to Clinic-AI because they enable:

1. **Natural Language Understanding**: Patients can describe symptoms and medical history in their own words, and the system extracts structured clinical information.

2. **Adaptive Question Generation**: The system dynamically generates contextually appropriate follow-up questions based on patient responses and medical reasoning, rather than using static questionnaires.

3. **Clinical Summarization**: Patient-reported information is condensed into concise, clinically structured summaries (pre-visit summaries, SOAP notes) that follow medical documentation standards.

4. **Multilingual Support**: The system conducts interviews and generates documentation in both English and Spanish, with proper medical terminology handling for each language.

5. **Contextual Safety**: LLMs help identify red flags, abusive language, and potential emergency situations that require immediate clinical attention.

### High-Level Workflow

The LLM-powered workflow consists of four main phases:

1. **Intake Interview (Multi-Agent System)**: Three specialized agents collaborate to conduct the interview:
   - **Agent-01 (Medical Context Analyzer)**: Analyzes chief complaint and determines clinical strategy
   - **Agent-02 (Coverage & Fact Extractor)**: Tracks what information has been covered and identifies gaps
   - **Agent-03 (Question Generator)**: Generates the next question based on context and gaps

2. **Pre-Visit Summary Generation**: After intake completion, an LLM generates a structured clinical summary for the doctor, incorporating doctor preferences and filtering disabled sections.

3. **SOAP Note Generation**: During or after the visit, an LLM processes the consultation transcript and context to generate structured SOAP (Subjective, Objective, Assessment, Plan) notes.

4. **Post-Visit Summary**: An LLM generates patient-friendly summaries for sharing via WhatsApp or other channels.

---

## 2. Inventory of LLM Prompts

### Prompt Inventory Table

| Prompt Name | Purpose | File/Module | Input Variables | Output Format | Language Handling |
|-------------|---------|-------------|-----------------|---------------|-------------------|
| **Agent-01: Medical Context Analyzer** | Analyzes chief complaint, determines condition properties, triage level, and question plan | `question_service_openai.py` (MedicalContextAnalyzer) | chief_complaint, patient_age, patient_gender, recently_travelled, language | JSON (MedicalContext schema) | English (unified prompt with dynamic language instructions) |
| **Agent-02: Coverage & Fact Extractor** | Extracts covered topics, identifies information gaps, extracts facts from conversation history | `question_service_openai.py` (AnswerExtractor) | asked_questions, previous_answers, medical_context, language | JSON (ExtractedInformation schema) | English (unified prompt with dynamic language instructions) |
| **Agent-03: Question Generator** | Generates next question for a specific topic | `question_service_openai.py` (QuestionGenerator) | medical_context, extracted_info, chosen_topic, asked_questions, previous_answers, language, max_words, deep_diagnostic_question_num | Plain text question string | Dynamic (output language: English or Spanish) |
| **Pre-Visit Summary Generator** | Generates clinical summary from intake responses | `question_service_openai.py` (generate_pre_visit_summary) | patient_data, intake_answers, language, medication_images_info, doctor_id | Plain text with section headings | Dynamic (output language: English or Spanish) |
| **SOAP Note Generator** | Generates SOAP note from consultation transcript | `soap_service_openai.py` (generate_soap_note) | transcript, patient_context, intake_data, pre_visit_summary, vitals, language, doctor_id, template | JSON (SOAP schema) | Dynamic (output language: English or Spanish) |
| **Post-Visit Summary Generator** | Generates patient-friendly summary | `soap_service_openai.py` (generate_post_visit_summary) | patient_data, soap_data, language | JSON (structured summary schema) | Dynamic (output language: English or Spanish) |
| **Dialogue Structure Analyzer** | Converts raw transcript into structured Doctor-Patient dialogue | `structure_dialogue.py` (structure_dialogue_from_text) | raw_transcript, model, language | JSON array of dialogue turns | Dynamic (output language: English or Spanish) |
| **Abusive Language Detector** | Detects inappropriate or abusive language in patient responses | `question_service_openai.py` (_analyze_abusive_language_with_llm) | questions_asked (QA pairs), language | JSON (list of abusive cases) | Dynamic (output language: English or Spanish) |
| **Question Quality Evaluator** | Evaluates question quality in test scenarios | `test_intake_question_quality_scenarios.py` (evaluate_questions_with_llm) | scenario, questions | JSON (evaluation scores) | English (test-only) |

### Prompt Scenario Classification

All prompts are tracked via the `PromptScenario` enum:

- `INTAKE`: Covers Agent-01, Agent-02, and Agent-03 prompts
- `PREVISIT_SUMMARY`: Pre-visit summary generation
- `RED_FLAG`: Agent-01 (red flag detection) and abusive language detection
- `SOAP`: SOAP note generation
- `POSTVISIT_SUMMARY`: Post-visit summary generation

---

## 3. Detailed Prompt Breakdown

### 3.1 Agent-01: Medical Context Analyzer

**Location:** `question_service_openai.py`, `MedicalContextAnalyzer.analyze_condition()`

**Intent:**
This prompt analyzes the patient's chief complaint and generates a strategic plan for the intake interview. It determines condition properties (chronic, acute, pain-related, etc.), triage level, red flags, and creates a prioritized list of topics to explore.

**Business/Clinical Reasoning:**
- Ensures the intake interview is clinically appropriate and focused on relevant information
- Identifies emergency situations that require immediate attention
- Prevents asking irrelevant questions (e.g., menstrual cycle questions for males)
- Establishes the foundation for subsequent question generation

**Inputs:**
- Chief complaint (text)
- Patient age (integer or null)
- Patient gender (string or null)
- Recent travel checkbox status (boolean)
- Language preference (en/es)

**Expected Output:**
JSON object with schema:
```json
{
  "condition_properties": {
    "is_chronic": true|false|null,
    "is_hereditary": true|false|null,
    "has_complications": true|false|null,
    "is_acute_emergency": true|false|null,
    "is_pain_related": true|false|null,
    "is_womens_health": true|false|null,
    "is_allergy_related": true|false|null,
    "requires_lifestyle_assessment": true|false|null,
    "is_travel_related": true|false|null,
    "severity_level": "mild|moderate|severe|null",
    "acuity_level": "acute|subacute|chronic|null",
    "is_new_problem": true|false|null,
    "is_followup": true|false|null
  },
  "triage_level": "routine|urgent|emergency",
  "red_flags": {
    "possible_emergency": true|false,
    "needs_urgent_attention": true|false,
    "identified_flags": ["string", ...]
  },
  "normalized_complaint": "short label or null",
  "core_symptom_phrase": "short phrase or null",
  "priority_topics": ["topic", ...],
  "avoid_topics": ["topic", ...],
  "topic_plan": ["topic", ...],
  "medical_reasoning": "2-4 sentences",
  "prompt_version": "med_ctx_v4"
}
```

**Constraints:**
- Must use only allowed topics (16 predefined medical topics)
- `topic_plan` must be a subset of `priority_topics`
- Must respect gender/age constraints for sensitive topics (menstrual_cycle)
- Must set `is_travel_related=false` if travel checkbox is "No"
- Must include women's health topics for female patients aged 12-60 with abdominal/pelvic symptoms

**Error Handling:**
- JSON parsing failures raise `ValueError`
- Invalid topic names are filtered via code (`_clamp_topics`)
- Missing required fields trigger validation errors
- Logged to structured `llm_interaction` collection for audit

**LLM Configuration:**
- Model: `settings.openai.model` (default: gpt-4o-mini)
- Temperature: 0.1 (low for consistency)
- Max tokens: 1200
- Scenario: `PromptScenario.RED_FLAG`

---

### 3.2 Agent-02: Coverage & Fact Extractor

**Location:** `question_service_openai.py`, `AnswerExtractor.extract_covered_information()`

**Intent:**
This prompt analyzes the conversation history to determine which topics have been covered, which remain as gaps, and extracts key facts (duration, medications, pain severity, associated symptoms).

**Business/Clinical Reasoning:**
- Prevents redundant questions by tracking what has already been asked
- Ensures comprehensive coverage of all relevant topics
- Maintains a code-truth tracking system (asked_categories) for authoritative coverage status

**Inputs:**
- Asked questions (list of strings)
- Previous answers (list of strings)
- Medical context (from Agent-01)
- Language preference

**Expected Output:**
JSON object with schema:
```json
{
  "topics_covered": ["topic", ...],
  "information_gaps": ["topic", ...],
  "extracted_facts": {
    "duration": "text or null",
    "medications": "text or null",
    "pain_severity": "text or null",
    "associated_symptoms": "text or null"
  },
  "already_mentioned_duration": true|false,
  "already_mentioned_medications": true|false,
  "redundant_categories": ["topic", ...],
  "topic_counts": {"topic": int, ...}
}
```

**Critical Rules:**
- `topics_covered` and `information_gaps` must be DISJOINT (no overlap)
- `topics_covered` may include a topic ONLY if a question about that topic exists in history
- `information_gaps` MUST equal: (topic_plan - topics_covered - avoid_topics)
- If conversation has 0 Q/A pairs, topics_covered = [] and information_gaps = topic_plan (minus avoid_topics)

**Error Handling:**
- Code recomputes gaps if LLM output is invalid: `_recompute_gaps_from_plan()`
- Topics are clamped to allowed list via `_clamp_topics()`
- Logged to structured `llm_interaction` collection

**LLM Configuration:**
- Model: `settings.openai.model`
- Temperature: 0.1
- Max tokens: 700
- Scenario: `PromptScenario.INTAKE`

---

### 3.3 Agent-03: Question Generator

**Location:** `question_service_openai.py`, `QuestionGenerator.generate_question_for_topic()`

**Intent:**
Generates a single, concise medical question about a specific topic. The question must be contextually appropriate, avoid repetition, and follow topic-specific guidelines.

**Business/Clinical Reasoning:**
- Produces natural, conversational questions that patients can easily answer
- Ensures questions are medically appropriate and focused
- Respects topic-specific requirements (e.g., duration questions must ask about cause)

**Inputs:**
- Medical context (from Agent-01)
- Extracted information (from Agent-02)
- Chosen topic (required topic string)
- Asked questions and previous answers (conversation history)
- Language preference
- Max words (default: 25)
- Deep diagnostic question number (optional: 1, 2, or 3 for chronic conditions)

**Expected Output:**
Plain text question string (must end with "?")

**Topic-Specific Guidance (from prompt):**
- **duration**: Must ask how long AND what might have caused it (cause is mandatory)
- **current_medications**: Must ask about medications, home remedies, AND dosages/frequency
- **past_medical_history**: If chronic, focus on related conditions; if non-chronic, ask about conditions related to chief complaint AND associated symptoms
- **pain_assessment**: Must ask about severity (0-10), characterization, AND location/radiation
- **triggers**: Must ask about what brings it on, what makes it worse, AND what relieves it
- **travel_history**: Must ask about destination, time, activities, exposures, symptoms during/after travel, preventive measures
- **chronic_monitoring**: Must ask about BOTH home monitoring AND professional monitoring, including frequency and typical readings
- **lab_tests**: Must ask specifically about LAB TEST RESULTS (what tests, what results)
- **screening**: Must ask specifically about FORMAL SCREENING EXAMS/COMPLICATION CHECKS (not lab tests)

**Deep Diagnostic Questions:**
For chronic/hereditary conditions with positive consent, three deep diagnostic questions are asked:
1. **chronic_monitoring**: Home and clinical monitoring (frequency and values)
2. **lab_tests**: Recent lab test results (HbA1c, kidney function, etc.)
3. **screening**: Screening exams for complications (eye exams, stress tests, etc.)

**Validation & Fallback:**
- Generated question is validated against topic keywords (`_question_matches_topic()`)
- If validation fails, one retry is attempted with correction instruction
- If retry fails, deterministic fallback question is used (`_get_fallback_question()`)

**Error Handling:**
- Empty or invalid questions trigger retry
- Topic mismatch triggers retry with correction
- Final fallback uses hardcoded question templates

**LLM Configuration:**
- Model: `settings.openai.model`
- Temperature: 0.1
- Max tokens: 250
- Scenario: `PromptScenario.INTAKE`

---

### 3.4 Pre-Visit Summary Generator

**Location:** `question_service_openai.py`, `OpenAIQuestionService.generate_pre_visit_summary()`

**Intent:**
Generates a concise clinical summary (~180-200 words) from patient intake responses, structured according to doctor preferences and enabled sections.

**Business/Clinical Reasoning:**
- Provides clinicians with a quick, structured overview before the visit
- Respects doctor preferences for sections (Chief Complaint, HPI, History, Review of Systems, Current Medication)
- Filters out disabled sections to prevent data leakage
- Maintains clinical handover tone: short, factual, deduplicated, neutral

**Inputs:**
- Patient data (name, age, etc.)
- Intake answers (Q/A pairs, asked_categories)
- Language preference
- Medication images info (optional)
- Doctor ID (for preferences)

**Expected Output:**
Plain text summary with section headings:
- Chief Complaint:
- HPI:
- History:
- Review of Systems:
- Current Medication:

**Section Definitions:**
- **Chief Complaint**: Primary reason for visit in patient's words (from Q1)
- **HPI**: ONE paragraph weaving OLDCARTS (Onset, Location, Duration, Characterization, Aggravating/Relieving factors, Radiation, Temporal pattern, Severity)
- **History**: Past medical, surgical, family, lifestyle, travel history
- **Review of Systems**: System-based positives/negatives
- **Current Medication**: Current medications, supplements, dosages, allergies

**Critical Rules:**
- Must NOT invent, guess, or expand beyond provided input
- Must NOT use placeholders like "N/A", "Not provided", "denies"
- Must NOT include sections that were not asked about
- Must respect exclusion rules for disabled sections (complete exclusion, no data leakage)
- Each section must contain ONLY information belonging to that section

**Doctor Preferences:**
- Section enable/disable flags
- Selected fields per section (e.g., HPI can focus on specific OLDCARTS elements)
- Style preferences (standard, concise, detailed)
- Focus areas
- Red flag inclusion toggle

**Error Handling:**
- Falls back to `_generate_fallback_summary()` if LLM fails
- Post-processing strips disabled sections via `_strip_disabled_sections()`
- Red flag detection runs separately (hybrid: hardcoded rules + LLM)

**LLM Configuration:**
- Model: `settings.openai.model`
- Temperature: 0.1
- Max tokens: 2000 (capped)
- Scenario: `PromptScenario.PREVISIT_SUMMARY`

---

### 3.5 SOAP Note Generator

**Location:** `soap_service_openai.py`, `OpenAISoapService.generate_soap_note()`

**Intent:**
Generates structured SOAP (Subjective, Objective, Assessment, Plan) notes from consultation transcripts, incorporating patient context, intake data, vitals, and pre-visit summary.

**Business/Clinical Reasoning:**
- Automates clinical documentation to reduce clinician documentation burden
- Ensures structured, comprehensive notes following medical standards
- Respects doctor preferences for detail level, formatting, section order, and language

**Inputs:**
- Consultation transcript (raw text)
- Patient context (name, age, chief complaint)
- Intake data (Q/A pairs)
- Pre-visit summary
- Vitals (structured dict)
- Language preference
- Doctor ID (for preferences)
- Template (optional per-visit custom template)

**Expected Output:**
JSON object with schema:
```json
{
  "subjective": "Patient's reported symptoms and history",
  "objective": {
    "vital_signs": {
      "blood_pressure": "120/80 mmHg",
      "heart_rate": "74 bpm",
      ...
    },
    "physical_exam": {
      "general_appearance": "...",
      "HEENT": "...",
      "cardiac": "...",
      ...
    }
  },
  "assessment": "Clinical impressions and reasoning",
  "plan": "Treatment plan and next steps",
  "highlights": ["Key point 1", ...],
  "red_flags": ["Flag 1", ...],
  "model_info": {...},
  "confidence_score": 0.95
}
```

**Critical Rules:**
- Must NOT make diagnoses unless explicitly stated by physician
- Must be objective and factual
- Must mark unclear information as "Unclear" or "Not discussed"
- Must include BOTH vitals from provided data AND transcript-derived exam findings
- Must respect doctor preferences:
  - Detail level (standard, concise, detailed)
  - Formatting (bullet_points, paragraphs)
  - Section order (configurable SOAP order)
  - Language override

**Error Handling:**
- JSON parsing failures trigger extraction from code fences (```json ... ```)
- Final fallback returns minimal structure with "Not discussed" placeholders
- Normalization via `_normalize_soap()` ensures valid structure

**LLM Configuration:**
- Model: `settings.soap.model` (default: gpt-4o-mini)
- Temperature: `settings.soap.temperature` (default: 0.3)
- Max tokens: `settings.soap.max_tokens` (default: 4000)
- Scenario: `PromptScenario.SOAP`

---

### 3.6 Post-Visit Summary Generator

**Location:** `soap_service_openai.py`, `OpenAISoapService.generate_post_visit_summary()`

**Intent:**
Generates patient-friendly summary for sharing via WhatsApp or other channels, translating clinical SOAP note into accessible language.

**Business/Clinical Reasoning:**
- Improves patient engagement and understanding
- Provides actionable instructions and reassurance
- Ensures patients remember key findings, medications, and next steps

**Inputs:**
- Patient data (name, age, visit date, chief complaint)
- SOAP note data (all sections)
- Language preference

**Expected Output:**
JSON object with schema:
```json
{
  "key_findings": ["Finding 1", ...],
  "diagnosis": "Diagnosis in simple language",
  "medications": [
    {
      "name": "Medication name",
      "dosage": "Dosage",
      "frequency": "Frequency",
      "duration": "Duration",
      "purpose": "Purpose"
    }
  ],
  "other_recommendations": ["Recommendation 1", ...],
  "tests_ordered": [
    {
      "test_name": "Test name",
      "purpose": "Why needed",
      "instructions": "When/where to get it"
    }
  ],
  "next_appointment": "Next appointment details",
  "red_flag_symptoms": ["Warning sign 1", ...],
  "patient_instructions": ["Instruction 1", ...],
  "reassurance_note": "Encouraging message"
}
```

**Critical Rules:**
- Must use simple, clear language
- Must be patient-friendly and easy to understand
- Must be comprehensive but concise
- Must be actionable with specific instructions
- Must be reassuring and supportive

**Error Handling:**
- Falls back to minimal structure with placeholder messages if parsing fails
- Normalization via `_normalize_post_visit_summary()` ensures required fields exist

**LLM Configuration:**
- Model: `settings.soap.model`
- Temperature: `settings.soap.temperature`
- Max tokens: `settings.soap.max_tokens`
- Scenario: `PromptScenario.POSTVISIT_SUMMARY`

---

### 3.7 Dialogue Structure Analyzer

**Location:** `structure_dialogue.py`, `structure_dialogue_from_text()`

**Intent:**
Converts raw medical consultation transcripts into structured Doctor-Patient dialogue format, preserving verbatim accuracy while removing sound effects and redacting personal identifiers.

**Business/Clinical Reasoning:**
- Enables downstream processing of consultation transcripts
- Protects patient privacy by redacting PII
- Maintains verbatim accuracy for clinical accuracy

**Inputs:**
- Raw transcript text (potentially long, chunked if >8000 chars)
- Deployment name (model)
- Language preference

**Expected Output:**
JSON array of dialogue turns:
```json
[
  {"Doctor": "What brings you in today?"},
  {"Patient": "I've been having chest pain."},
  ...
]
```

**Critical Rules:**
- **VERBATIM TEXT PRESERVATION**: NEVER change, paraphrase, or correct words
- Remove ALL sound effects and environmental descriptions
- Remove personal identifiers (names → [NAME], phone numbers → [REDACTED], etc.)
- Preserve medical terminology (medication names, conditions, symptoms)
- Use context-based speaker identification (not blind trust of input labels)

**Speaker Identification Priority:**
1. Context-based analysis (95% accuracy)
2. Doctor signals (questions, instructions, clinical assessments)
3. Patient signals (first-person experiences, symptom descriptions)
4. Family Member signals (third-person references)

**Error Handling:**
- Chunking for long transcripts (>8000 chars for GPT-4, >6000 for GPT-4o-mini)
- Heuristic fallback if LLM fails (alternating speakers based on patterns)
- JSON extraction from code fences if present

**LLM Configuration:**
- Model: Azure OpenAI deployment (from settings)
- Temperature: 0.0 (deterministic)
- Max tokens: 4000 (GPT-4) or 2000 (other models)
- Response format: JSON object (when supported)

---

### 3.8 Abusive Language Detector

**Location:** `question_service_openai.py`, `_analyze_abusive_language_with_llm()`

**Intent:**
Detects subtle, contextual, or creative abusive language in patient responses that may not be caught by hardcoded keyword lists.

**Business/Clinical Reasoning:**
- Protects clinicians from inappropriate patient interactions
- Flags potentially problematic patients for review
- Maintains professional clinical environment

**Inputs:**
- Question-answer pairs (filtered to exclude obvious cases already detected)
- Language preference

**Expected Output:**
JSON object with schema:
```json
{
  "abusive_language": [
    {
      "question": "full question",
      "answer": "full answer",
      "reason": "explanation (in output language)"
    }
  ]
}
```

**What to Flag:**
- Direct or disguised profanity
- Insults, slurs, degrading language
- Inappropriate sexual language
- Racist, sexist, discriminatory comments
- Threats or aggressive language
- Offensive sarcasm or passive-aggressive language

**What NOT to Flag:**
- Medical terminology
- Symptom descriptions
- Appropriate responses to medical questions
- Legitimate expressions of pain or frustration

**Hybrid Detection:**
- **Step 1**: Fast hardcoded detection for obvious cases (`_detect_obvious_abusive_language()`)
- **Step 2**: LLM analysis for subtle/contextual cases (this prompt)

**Error Handling:**
- Returns empty list on parsing failures
- Logs warnings for failures

**LLM Configuration:**
- Model: `settings.openai.model`
- Temperature: 0.1
- Max tokens: 1000
- Scenario: `PromptScenario.RED_FLAG`

---

## 4. Multilingual Strategy

### Language Normalization

The system normalizes language codes consistently:
- **Frontend uses**: `'en'` or `'sp'`
- **Backend normalizes to**: `'en'` or `'es'` (for LLM prompts)
- **Mapping**: `'sp'`, `'es'`, `'spanish'`, `'español'`, `'es-es'`, `'es-mx'` → `'es'`
- **Default**: Unknown/empty → `'en'`

### Unified Prompt Strategy

**All prompts use a unified English system prompt with dynamic language instructions.** This means:
- System prompts are written in English
- Language instructions are injected dynamically: `f"Write all natural-language text values in {output_language}"`
- Output language name is derived: `"Spanish"` if `lang == "es"`, else `"English"`

### JSON Key Preservation Rule

**CRITICAL RULE**: Do NOT translate JSON keys, enums, codes, field names, or IDs. Only natural-language text values are translated.

This ensures:
- JSON parsing remains consistent across languages
- Database schemas are language-agnostic
- API contracts are stable

### Language-Specific Handling

**Agent-03 (Question Generator)** is the only prompt that produces output in the target language:
- System prompt instructs: `"Write the question in {output_language}"`
- Fallback questions are pre-translated (separate English and Spanish dictionaries)
- Topic keyword matching supports both languages (`_TOPIC_KEYWORDS_EN`, `_TOPIC_KEYWORDS_ES`)

**All other prompts** generate content that is then displayed/used in the target language but may internally use English for structured fields.

### Vitals Translation

SOAP notes include manual translation for vitals text:
- English vitals labels are translated to Spanish via `_translate_vitals_to_spanish()`
- Translation uses a hardcoded mapping dictionary
- This is a post-processing step, not LLM-based

### Known Limitations

1. **Dialogue Structure Analyzer**: Speaker labels ("Doctor", "Patient") remain in English even for Spanish transcripts. Consider using "Doctora", "Paciente" for Spanish outputs.

2. **Consistency**: Some prompts may occasionally produce mixed-language outputs if language instructions are not strictly followed. Post-processing validation is recommended.

3. **Medical Terminology**: Medical terms are preserved in their original language. Consider adding terminology translation dictionaries if full localization is required.

---

## 5. Safety, Compliance & Guardrails

### Clinical Constraints

**No Diagnosis Rule:**
- SOAP note prompt explicitly states: "Do NOT make diagnoses or treatment recommendations unless explicitly stated by the physician"
- Pre-visit summary prompt states: "Do not invent, guess, or expand beyond the provided input"
- System extracts and structures information; clinical judgment remains with physicians

**Factual Extraction Only:**
- All prompts emphasize extracting only what was actually said or reported
- No hallucination or expansion beyond provided context
- Placeholder handling: "Unclear" or "Not discussed" for missing information

### Red Flag Detection

**Two-Level Detection:**
1. **Agent-01 (Medical Context Analyzer)**: Identifies emergency situations, urgent attention needs, and clinical red flags
2. **Abusive Language Detector**: Identifies inappropriate patient language (hardcoded + LLM)

**Emergency Handling:**
- Triage levels: `routine`, `urgent`, `emergency`
- Red flags include: `possible_emergency`, `needs_urgent_attention`, `identified_flags`
- These are surfaced in pre-visit summaries and SOAP notes

### Topic Safety Rules

**Gender/Age Constraints:**
- Menstrual cycle questions are blocked for:
  - Males
  - Age < 12 or age > 60
  - Code enforces: `priority_topics` filtering, `avoid_topics` addition

**Travel Questions:**
- Only asked if:
  - Travel checkbox is "Yes" AND
  - `is_travel_related=true` from Agent-01
- Code enforces: `avoid_topics` includes `travel_history` if checkbox is "No"

**Women's Health Logic:**
- For female patients aged 12-60 with stomach/abdominal/pelvic pain symptoms:
  - `is_womens_health=true` is set
  - `menstrual_cycle` is added to `priority_topics`
- Code enforces this logic explicitly (lines 401-422 in question_service_openai.py)

### PHI Handling

**Dialogue Structure Analyzer** redacts:
- Names (doctor, patient, family) → `[NAME]`
- Phone numbers → `[REDACTED]`
- Street addresses → `[REDACTED]`
- Specific calendar dates → `[REDACTED]`
- Social Security Numbers → `[REDACTED]`
- Ages (explicit) → `[AGE]`

**Medical Terminology Preservation:**
- Medication names (metformin, lisinopril, etc.) are NOT redacted (they are not PII)
- Medical conditions, symptoms, anatomical references are preserved
- This ensures clinical accuracy while protecting privacy

### LLM Training/Memory Prevention

**Azure OpenAI Configuration:**
- System uses Azure OpenAI (not standard OpenAI) for data security
- Azure OpenAI is configured with data residency controls
- No explicit opt-out flags are set in prompts (rely on Azure configuration)

**Recommendation:**
- Verify Azure OpenAI resource configuration for data handling policies
- Consider adding explicit opt-out instructions if required by compliance

### Data Leakage Prevention

**Pre-Visit Summary:**
- Doctor preferences can disable sections (Chief Complaint, HPI, History, Review of Systems, Current Medication)
- Disabled sections are:
  1. Filtered from intake answers before prompt (`_filter_intake_answers_by_prefs()`)
  2. Explicitly excluded in prompt instructions
  3. Post-processed and stripped via `_strip_disabled_sections()`

This triple-layer protection ensures no data leakage into disabled sections.

---

## 6. Prompt Evaluation & Quality Control

### Prompt Version Tracking

**Automatic Version Management:**
- Prompt templates are extracted automatically via `prompt_extractors.py`
- Templates are normalized (dynamic variables replaced with placeholders)
- Versions are stored in MongoDB (`PromptVersionMongo` collection)
- Versions follow format: `{SCENARIO}_V_{MAJOR}.{MINOR}` (e.g., `INTAKE_V_1.0`)

**Version Detection:**
- On startup, `PromptVersionManager` compares current template hash with stored version
- If template changed, version is incremented automatically
- Global registry `PROMPT_VERSIONS` is updated with current versions

**Telemetry Integration:**
- All LLM calls include `prompt_version` in telemetry (`call_llm_with_telemetry()`)
- Application Insights traces include version information
- This enables A/B testing and rollback if issues occur

### Test Scenarios

**File:** `test_intake_question_quality_scenarios.py`

**Evaluation Method:**
- 25 test scenarios covering diverse patient profiles (age, gender, complaints, languages)
- Each scenario runs full multi-agent intake pipeline
- LLM evaluator (`evaluate_questions_with_llm()`) rates question quality:
  - Overall rating (0-10)
  - Coverage OK (boolean)
  - Safety OK (boolean)
  - Redundancy OK (boolean)
  - Summary (text explanation)

**Evaluator Prompt:**
- System prompt: "You are a senior clinician and clinical question design reviewer"
- Evaluates: medical appropriateness, safety, relevance, coverage, redundancy
- Returns JSON with scores and reasoning

**Quality Criteria:**
- Questions must be medically appropriate and safe
- Must cover key aspects (onset, duration, severity, associated symptoms, red flags, PMH/meds when needed)
- Must avoid redundant or near-duplicate questions
- Must respect safety rules (no menstrual for males, no irrelevant travel, etc.)

**Known Test Scenarios:**
- Acute conditions (headache, chest pain, fever)
- Chronic conditions (diabetes, hypertension)
- Women's health (abdominal pain, menstrual issues)
- Travel-related (recent travel with symptoms)
- Pediatric and geriatric cases
- Spanish language scenarios

### Golden Datasets

**Not Currently Implemented:**
- No explicit golden dataset for prompt evaluation
- Test scenarios serve as ad-hoc validation
- Recommendation: Create golden dataset with expected outputs for regression testing

### Evaluation Criteria

**JSON Validity:**
- All prompts requiring JSON output use regex extraction or structured parsing
- Fallback handling for code fences (```json ... ```)
- Minimal structure fallbacks if parsing fails

**Coverage:**
- Agent-02 explicitly computes coverage gaps
- Code recomputes gaps if LLM output is invalid (`_recompute_gaps_from_plan()`)
- Topic counts tracked via `asked_categories` (code-truth)

**Correctness:**
- Safety validator (`SafetyValidator`) checks for critical violations:
  - Exact duplicate questions
  - Menstrual questions for invalid demographics
  - Travel questions when not allowed
- Falls back to closing question if validation fails

**Similarity:**
- Question similarity checked via topic keyword matching (`_question_matches_topic()`)
- Retry mechanism if topic mismatch detected

### What "Good Output" Means

**For Agent-01:**
- Valid JSON with all required fields
- Topics are from allowed list
- Triage level is appropriate
- Red flags are identified if present

**For Agent-02:**
- Topics_covered and information_gaps are disjoint
- Gaps correctly computed: (topic_plan - covered - avoid)
- Facts extracted accurately

**For Agent-03:**
- Question is on-topic (validated via keywords)
- Question is concise (<25 words typically)
- Question ends with "?"
- Question is not duplicate

**For Pre-Visit Summary:**
- All enabled sections present (if data exists)
- No disabled sections present
- Concise (~180-200 words)
- Clinical handover tone

**For SOAP Notes:**
- Valid JSON structure
- All sections present (may be "Not discussed")
- No invented diagnoses
- Objective includes both vitals and exam findings

---

## 7. Known Limitations & Future Improvements

### Prompt Complexity Issues

**1. Agent-01 Prompt Length:**
- Current prompt includes extensive rules and topic definitions
- Consider splitting into structured prompt sections or using few-shot examples
- **Impact**: Potential token cost and clarity issues

**2. Agent-03 Topic-Specific Guidance:**
- Extensive topic-specific rules embedded in system prompt
- Consider externalizing to a topic configuration file
- **Impact**: Harder to maintain and extend

**3. Pre-Visit Summary Prompt:**
- Very long prompt with dynamic section definitions
- Consider template-based approach with separate prompts per section
- **Impact**: Token cost, maintenance complexity

### Token Cost Considerations

**Current Estimates (approximate):**
- Agent-01: ~800 input tokens, ~600 output tokens
- Agent-02: ~500 input tokens, ~400 output tokens
- Agent-03: ~600 input tokens, ~50 output tokens
- Pre-Visit Summary: ~2000 input tokens, ~400 output tokens
- SOAP Note: ~3000 input tokens, ~1500 output tokens
- Dialogue Structure: Variable (chunked for long transcripts)

**Optimization Opportunities:**
1. **Truncate conversation history**: Only include last N Q/A pairs for Agent-02 and Agent-03
2. **Summarize context**: Pre-summarize long conversation history before passing to LLM
3. **Use smaller models**: Consider GPT-4o-mini for less critical tasks (already used)
4. **Cache Agent-01 output**: Medical context doesn't change during interview (already cached in code)

### Areas for Simplification

**1. Prompt Abstraction:**
- Create a prompt template system with variables
- Reduce duplication between English/Spanish instructions
- **Benefit**: Easier maintenance, version control

**2. Topic Configuration:**
- Externalize topic definitions, keywords, fallback questions to config files
- Enable non-developer updates
- **Benefit**: Faster iteration, less code changes

**3. Error Handling Consolidation:**
- Standardize JSON extraction, fallback handling across all prompts
- Create utility functions for common patterns
- **Benefit**: Consistent behavior, easier debugging

### Future Prompt Consolidation

**Potential Consolidations:**
1. **Agent-01 + Agent-02**: Could potentially merge into single analysis step (trade-off: less modularity)
2. **Pre-Visit + SOAP**: Could share common summarization logic (trade-off: different output formats)

**Not Recommended:**
- Keeping agents separate maintains clear responsibilities and easier debugging
- Different output formats (text vs JSON) require different prompt structures

### Suggestions for Future Improvements

**1. Prompt A/B Testing Framework:**
- Implement systematic A/B testing for prompt variations
- Track metrics: question quality, coverage, clinician satisfaction
- Use prompt version tracking for rollback

**2. Few-Shot Examples:**
- Add few-shot examples to prompts for better consistency
- Examples should cover edge cases (chronic conditions, complex histories)

**3. Prompt Validation:**
- Pre-validate prompts for JSON schema compliance
- Test prompts against golden datasets before deployment
- Automated regression testing for prompt changes

**4. Documentation:**
- Maintain prompt change log
- Document prompt design decisions and trade-offs
- Include prompt examples in documentation

**5. Multilingual Expansion:**
- Consider additional languages beyond English/Spanish
- Create language-specific prompt templates if needed
- Test multilingual scenarios more thoroughly

**6. Clinical Review:**
- Regular clinical review of prompts for medical accuracy
- Incorporate clinician feedback into prompt improvements
- Periodic audit of red flag detection effectiveness

**7. Token Optimization:**
- Implement conversation summarization for long interviews
- Use embedding-based similarity to reduce redundant context
- Consider fine-tuning smaller models for specific tasks

---

## Appendix: File References

### Core Prompt Files
- `Clinic-AI-Backend/src/clinicai/adapters/external/question_service_openai.py` - Agent-01, Agent-02, Agent-03, Pre-Visit Summary, Abusive Language Detection
- `Clinic-AI-Backend/src/clinicai/adapters/external/soap_service_openai.py` - SOAP Note, Post-Visit Summary
- `Clinic-AI-Backend/src/clinicai/application/utils/structure_dialogue.py` - Dialogue Structure Analyzer

### Prompt Management
- `Clinic-AI-Backend/src/clinicai/adapters/external/prompt_registry.py` - PromptScenario enum, PROMPT_VERSIONS registry
- `Clinic-AI-Backend/src/clinicai/adapters/external/prompt_extractors.py` - Automatic prompt template extraction
- `Clinic-AI-Backend/src/clinicai/adapters/external/prompt_version_manager.py` - Version tracking and management
- `Clinic-AI-Backend/src/clinicai/adapters/external/llm_gateway.py` - Centralized LLM call gateway with telemetry

### Configuration
- `Clinic-AI-Backend/src/clinicai/core/config.py` - LLM model, temperature, max_tokens settings
- `Clinic-AI-Backend/src/clinicai/core/ai_factory.py` - Azure OpenAI client factory

### Testing
- `Clinic-AI-Backend/tests/test_intake_question_quality_scenarios.py` - LLM-based question quality evaluation

---

## Document Revision History

- **v1.0** (December 2024): Initial comprehensive documentation for project closure

---

**End of Document**



