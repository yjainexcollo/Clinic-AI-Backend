## Intake Form Setup Notes (Himanshu Code Setup)

### What was fixed
- Added `asked_categories` to the intake domain entity and persisted it to Mongo so topic tracking survives across requests.
- Ensured Mongo repository reads/writes `asked_categories` on visits.
- Replaced missing `_chat_completion` calls with the unified `call_llm_with_telemetry` for pre-visit summary and abusive-language analysis.

### Key files
- `domain/entities/visit.py` — `IntakeSession` now tracks `asked_categories`.
- `adapters/db/mongo/repositories/visit_repository.py` — persists and restores `asked_categories`.
- `adapters/external/question_service_openai.py` — uses telemetry gateway instead of the missing `_chat_completion`.

### Impact
- Intake topic sequencing is now durable across API calls and persisted in the DB.
- LLM calls for summaries and abuse checks use the standard telemetry path, avoiding runtime errors.

### Follow-up checks
- Run the intake flow end-to-end to confirm `asked_categories` is stored and reloaded.
- Verify pre-visit summary and abusive-language checks execute without `_chat_completion` errors.

