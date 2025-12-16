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
- Timestamps: `created_at` per entry, `updated_at` per document.
- Indexes on `visit_id`, `patient_id`, `updated_at`.


