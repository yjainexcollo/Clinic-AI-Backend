# LLM Interaction Logging (Per Visit)

## Goal
Capture all LLM prompts/responses per visit in a **structured, system-prompt-free** document for auditing and replay. The collection is `llm_interaction`.

## Collection: `llm_interaction`
One document per visit:
```json
{
  "visit_id": "VISIT_UUID",
  "patient_id": "PATIENT_UUID",
  "intake": {
    "intake_prompt_log": [
      {
        "question_number": 1,
        "question_id": "Q1",
        "question_text": "What symptoms are you experiencing today?",
        "asked_at": "...",
        "agents": [
          {
            "agent_name": "agent1_medical_context",
            "user_prompt": "...",
            "response_text": "...",
            "metadata": { "prompt_version": "med_ctx_v3", "chief_complaint": "..." },
            "created_at": "..."
          },
          {
            "agent_name": "agent2_extractor",
            "user_prompt": "...",
            "response_text": "...",
            "metadata": { "topics_covered": [], "information_gaps": [] },
            "created_at": "..."
          },
          {
            "agent_name": "agent3_question_generator",
            "user_prompt": "...",
            "response_text": "next question text",
            "metadata": { "chosen_topic": "...", "prompt_version": "qgen_v2" },
            "created_at": "..."
          }
        ]
      }
    ]
  },
  "pre_visit_summary": { "llm_calls": [ /* user_prompt, response_text, metadata */ ] },
  "soap": { "llm_calls": [ /* user_prompt, response_text, metadata */ ] },
  "post_visit_summary": { "llm_calls": [ /* user_prompt, response_text, metadata */ ] },
  "created_at": "...",
  "updated_at": "..."
}
```

### Notes
- **System prompts are not stored** in this structured log.
- Legacy `llm_interactions` collection remains unchanged for backward compatibility.

## Code Points
- **Models**: `LLMInteractionVisit`, `IntakeQuestionLog`, `AgentLog`, `LLMCallLog` in `adapters/db/mongo/models/patient_m.py`.
- **Repo helpers**: `append_intake_agent_log`, `append_phase_call` in `adapters/db/mongo/repositories/llm_interaction_repository.py`.

## Intake Flow (Agents 1–3)
- Wired in `question_service_openai.py` inside `generate_next_question`.
- Each agent appends to `intake_prompt_log` with `question_number`, `question_text` (when known), `user_prompt`, `response_text`, `metadata`.
- **Important**: All three agents (Agent 1: Medical Context Analyzer, Agent 2: Coverage & Fact Extractor, Agent 3: Question Generator) store their **actual user prompt strings** (not structured dictionaries) in the `user_prompt` field.
- For each question, all 3 agents' interactions are stored in the `agents` array, allowing full traceability of the question generation process.

## Pre-Visit Summary
- Logged in `question_service_openai.generate_pre_visit_summary`.
- `user_prompt` includes `patient_data`, `intake_answers`, `medication_images_info`.
- Phase key: `pre_visit_summary`.

## SOAP
- Logged in `generate_soap_note` use case (`application/use_cases/generate_soap_note.py`) after SOAP is generated.
- `user_prompt` includes transcript, patient_context, intake_data, pre_visit_summary, vitals, language.
- Phase key: `soap`.

## Post-Visit Summary
- Logged in `generate_post_visit_summary` use case after generation.
- `user_prompt` includes `patient_data`, `soap_data`, `language`.
- Phase key: `post_visit_summary`.

## Storage Rules
- Only `user_prompt`, `response_text`, `metadata` are stored per agent/call.
- **User prompts are stored as actual strings** - the exact prompt text sent to the LLM (not structured summaries or dictionaries).
- Timestamps: `created_at` per entry, `updated_at` per document.
- Indexes on `visit_id`, `patient_id`, `updated_at`.

## Model Structure

The following models are defined in `adapters/db/mongo/models/patient_m.py`:

- **`AgentLog`**: Individual agent interaction log
  - `agent_name`: Agent identifier (e.g., "agent1_medical_context", "agent2_extractor", "agent3_question_generator")
  - `user_prompt`: Actual user prompt string sent to LLM
  - `response_text`: LLM response text
  - `metadata`: Additional metadata (prompt_version, extracted facts, etc.)
  - `created_at`: Timestamp

- **`IntakeQuestionLog`**: All agents for one intake question
  - `question_number`: Sequential question number
  - `question_id`: Optional question ID
  - `question_text`: The actual question text
  - `asked_at`: When question was asked
  - `agents`: List of `AgentLog` entries (typically 3: Agent 1, Agent 2, Agent 3)

- **`LLMCallLog`**: LLM call log for other phases (previsit, SOAP, postvisit)
  - Same structure as `AgentLog` but used for non-intake phases

- **`LLMInteractionVisit`**: Main document (Beanie Document)
  - `visit_id`: Unique visit identifier
  - `patient_id`: Patient identifier
  - `intake`: `IntakePromptLogSection` containing list of `IntakeQuestionLog`
  - `pre_visit_summary`: `PhaseLogSection` containing list of `LLMCallLog`
  - `soap`: `PhaseLogSection` containing list of `LLMCallLog`
  - `post_visit_summary`: `PhaseLogSection` containing list of `LLMCallLog`
  - `created_at`, `updated_at`: Timestamps

## Implementation Details

### Intake Flow
- **Agent 1 (Medical Context Analyzer)**: Called first, analyzes chief complaint and patient demographics
  - Stores actual user prompt with patient data (chief complaint, age, gender, travel status)
  - Response: Medical context JSON with condition properties, triage level, priority topics
  
- **Agent 2 (Coverage & Fact Extractor)**: Called after Agent 1, analyzes conversation history
  - Stores actual user prompt with Q&A history and medical context
  - Response: Coverage analysis JSON with topics_covered, information_gaps, extracted_facts
  
- **Agent 3 (Question Generator)**: Called after Agent 2, generates the actual question
  - Stores actual user prompt with medical context, coverage analysis, and conversation history
  - Response: Generated question text

All three agents log their interactions to the same `question_number` entry in the `agents` array.

## Changelog

### 2025-01 (Latest Update)
- **Fixed**: All three intake agents now store actual user prompt strings instead of structured dictionaries
- **Added**: Missing model classes (`AgentLog`, `IntakeQuestionLog`, `LLMCallLog`, `LLMInteractionVisit`, etc.) to `patient_m.py`
- **Improved**: Consistency across all workflow phases (intake, previsit, SOAP, postvisit) - all store actual user prompts
- **Result**: Complete traceability of all LLM interactions with actual prompts for testing and debugging

### Previous Implementation
- Previously, Agent 1 and Agent 2 stored structured context dictionaries instead of actual user prompts
- This made it difficult to replay or debug LLM interactions
- Agent 3 already stored actual prompts, creating inconsistency

### Current Implementation  
- All agents (1, 2, 3) now store the exact user prompt strings sent to the LLM
- This enables full replay capability and easier debugging
- Test cases can now capture complete prompt/response pairs for regression testing


