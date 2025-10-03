# Clinic-AI 🏥🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AI-powered clinic management system with speech-to-text, large language model integration, and Electronic Health Record (EHR) capabilities. Built with Clean Architecture principles for scalability and maintainability.

## ✨ Features

- **🤖 AI-Powered Services**

  - Speech-to-Text transcription for medical consultations
  - Large Language Model integration for medical documentation
  - OCR capabilities for document processing
  - Intelligent SOAP note generation
  - Prescription image analysis with Mistral AI

- **🏥 Healthcare Management**

  - Patient registration and management
  - Consultation scheduling and tracking
  - Medical record management
  - EHR integration capabilities

- **🎵 Audio Management System**

  - Database-stored audio files (no file system dependencies)
  - Comprehensive audio player with full controls
  - Audio file listing and management interface
  - Support for both adhoc and visit recordings
  - Audio streaming and download capabilities
  - Migration tools for existing file-based audio

- **🏗️ Modern Architecture**

  - Clean Architecture with domain-driven design
  - FastAPI for high-performance API
  - MongoDB with Beanie ODM
  - Redis caching layer
  - Comprehensive testing strategy

- **🔒 Security & Compliance**
  - JWT-based authentication
  - Role-based access control
  - HIPAA-compliant data handling
  - Secure external service integration

## 🧱 Architecture and Tech Stack (Backend)

This backend follows Clean Architecture and domain-driven design. Responsibilities are separated into presentation (APIs), application (use cases), domain (entities, value objects), and infrastructure (adapters for DB, AI, storage). Below is a concrete mapping of each tool/technology, why it is used, and where it lives in the codebase.

### Core Runtime
- **Python 3.11**: Modern typing, performance improvements, and ecosystem compatibility for async workloads.
- **Uvicorn** (`uvicorn[standard]`): ASGI server to run FastAPI in production/dev.
  - Used via `make dev` and `src/clinicai/app.py` app factory.

### Web Framework & API Layer
- **FastAPI**: High-performance async web framework with automatic OpenAPI.
  - Entry/app wiring: `src/clinicai/app.py`
  - Routers (presentation layer):
    - `api/routers/patients.py` – intake lifecycle, pre-visit summary
    - `api/routers/notes.py` – transcription, SOAP generation, retrieval
    - `api/routers/prescriptions.py` – prescription upload & analysis
    - `api/routers/audio.py` – audio file management, streaming, and playback
    - `api/routers/health.py` – health and readiness
  - Schemas (request/response models): `api/schemas/*`
  - Dependencies: `api/deps.py`

### Validation & Settings
- **Pydantic v2** (`pydantic`, `pydantic-settings`): Data validation and typed settings.
  - Settings/config: `core/config.py` (loads env like `MONGO_DB_NAME`, models, limits)
  - DTOs: `application/dto/*` and `api/schemas/*`

### Database & Persistence
- **MongoDB** with **Motor** (async driver) and **Beanie** (ODM): Flexible document storage for patients, visits, transcripts, SOAP, etc.
  - Models: `adapters/db/mongo/models/*` (e.g., `patient_m.py`)
  - Repositories: `adapters/db/mongo/repositories/*` (e.g., `patient_repository.py`)
  - Unit of Work: `application/uow.py`
  - Notes:
    - Connection configured in `core/config.py` and app bootstrap `bootstrap.py`/`app.py`.
    - `dnspython` and `certifi` support SRV and TLS trust stores.

### Caching & State
- **Redis**: Optional caching/session layer for low-latency operations.
  - Accessed via adapters/utilities when enabled (see `core/config.py`).

### AI & NLP Services
- **OpenAI** (`openai`):
  - Intake Q&A, next-question generation, and pre-visit summaries.
  - Files: `adapters/external/question_service_openai.py`, `adapters/external/soap_service_openai.py`
  - Config: `OPENAI_API_KEY`, `OPENAI_MODEL`, `SOAP_MODEL`, token/temperature limits in `.env` via `core/config.py`.

- **Mistral AI** (`mistralai`):
  - Prescription understanding from images; structured extraction of medicines/tests/instructions.
  - File: `adapters/external/prescription_service_mistral.py`
  - Config: `MISTRAL_API_KEY` via `core/config.py`.

### Speech-to-Text & Audio Processing (Step‑03)
- **OpenAI Whisper** (`openai-whisper`, `torch`, `torchaudio`): Local/hosted transcription of consultations.
  - File: `adapters/external/transcription_service_whisper.py`
  - API route: `api/routers/notes.py` → `POST /notes/transcribe`
  - System dependency: `ffmpeg` (see Quick Start).

### Security & Auth
- **python-jose[cryptography]**: JWT creation/verification; future-proofing for auth.
- **passlib[bcrypt]**: Password hashing if/when user auth is enabled.
- **python-multipart**: Multipart parsing for audio/image uploads.
  - Usage: `api/routers/notes.py` and `api/routers/prescriptions.py` handle multipart forms.

### HTTP & Async IO
- **httpx**, **aiohttp**: Async HTTP clients for calling external AI or webhooks when needed.
  - Used within external adapters and future integrations.

### Observability
- **structlog**: Structured logging with context-rich events.
  - Config: `observability/logging_conf.py`
  - Usage across layers via logger injection.
- Custom modules:
  - `observability/tracing.py` – tracing hooks (extensible for OpenTelemetry)
  - `observability/metrics.py` – counters/histograms (extensible for Prometheus)
  - `observability/audit.py` – audit logging for PHI-sensitive events

### Error Handling & Exceptions
- Centralized exceptions in `core/exceptions.py` and `domain/errors.py`.
- API-level error handlers in router layer; consistent 4xx/5xx mapping.

### Utilities & Core
- `core/utils/*`: datetime, strings, crypto helpers, OCR scaffolding, patient matching.
- `core/container.py`: DI/service registry for ports/adapters.
- `bootstrap.py`: Wiring of infrastructure and app startup.

### Testing & Quality
- **pytest**, **pytest-asyncio**, **pytest-cov**: Unit/integration/e2e tests with coverage.
- **black**, **isort**, **flake8**, **mypy**: Formatting, import order, linting, and static typing.
  - Commands: `make test`, `make lint`, `make format`.

### Packaging & Dependencies
- `requirements.txt`: Runtime and dev dependencies (pinned by ranges).
- `pyproject.toml`: Tooling config, project metadata.

### Containerization & Local Dev
- **Docker** and **docker-compose**: Local orchestration of API, MongoDB, Redis.
  - Files: `docker/Dockerfile`, `docker-compose.yml`, `docker/mongo-init/`
  - Make targets: `make docker-build`, `make docker-run`.

### API Schema & Docs
- OpenAPI/Swagger auto-generated by FastAPI at `/docs` and `/openapi.json`.
- Hand-authored API reference: `swagger.yaml` (source of truth for external consumers).

### Data Privacy & Compliance
- No training on PHI; prompts/responses are not persisted beyond required storage.
- Full audit logging for significant actions; transport secured via TLS.
- Encryption utilities available in `core/utils/crypto.py` and `crypto_utils.py`.

### Service-by-Service Mapping
- Patients & Intake (Step‑01/02):
  - Routes: `api/routers/patients.py`
  - External services: `question_service_openai.py` (next question, completion), `soap_service_openai.py` (pre-visit summary when applicable)
  - Persistence: `adapters/db/mongo/...`

- Notes: Transcription & SOAP (Step‑03):
  - Routes: `api/routers/notes.py`
  - External services: `transcription_service_whisper.py` (STT), `soap_service_openai.py` (SOAP)
  - Persistence: transcripts/SOAP stored via Mongo repositories

- Prescriptions:
  - Routes: `api/routers/prescriptions.py`
  - External services: `prescription_service_mistral.py`
  - Storage of results and raw text via Mongo repositories

- Health & Readiness:
  - Route: `api/routers/health.py`
  - Checks DB, cache, and overall service liveness

- Audio Management:
  - Routes: `api/routers/audio.py`
  - Database storage: `AudioFileMongo` model in `adapters/db/mongo/models/patient_m.py`
  - Repository: `adapters/db/mongo/repositories/audio_repository.py`
  - Frontend: Audio management page with full-featured player
  - Migration: `scripts/migrate_audio_to_db.py` for existing file-based audio

### Configuration Snapshot (env)
- `MONGO_URI`, `MONGO_DB_NAME` – database connection and DB name
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_MAX_TOKENS`, `OPENAI_TEMPERATURE`
- `SOAP_MODEL`, `SOAP_MAX_TOKENS`, `SOAP_TEMPERATURE`
- `WHISPER_MODEL`, `WHISPER_LANGUAGE`
- `MISTRAL_API_KEY`
- `AUDIO_MAX_SIZE_MB`, `AUDIO_ALLOWED_FORMATS`, `FILE_STORAGE_TYPE`
- `FILE_SAVE_AUDIO_FILES` (deprecated - now defaults to False, use database storage)

Where to look: `core/config.py` centralizes reading and validation of these environment variables.

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- MongoDB 6.0+
- Redis 6.0+
- Docker (optional)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/clinicai.git
   cd clinicai
   ```

2. **Set up virtual environment**

   macOS (zsh):

   ```bash
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   py -3.11 -m venv .venv311
   .\.venv311\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Install system dependency for transcription (ffmpeg)**

   macOS (Homebrew):

   ```bash
   brew install ffmpeg
   ```

   Ubuntu/Debian:

   ```bash
   sudo apt update && sudo apt install -y ffmpeg
   ```

6. **Run the application**
   ```bash
   make dev
   ```

The API will be available at `http://localhost:8000`

### Environment configuration (.env)

Key variables you should set (see `.env.example` for the full list):

```bash
# Application
APP_ENV=development
DEBUG=true
PORT=8000

# Database
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=clinicai

# OpenAI (Intake Q&A – Step‑01)
OPENAI_API_KEY=sk-REPLACE_ME
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=800
OPENAI_TEMPERATURE=0.6

# SOAP generation (Step‑03)
SOAP_MODEL=gpt-4
SOAP_MAX_TOKENS=2000
SOAP_TEMPERATURE=0.3

# Whisper (Transcription – Step‑03)
WHISPER_MODEL=base
WHISPER_LANGUAGE=en

# Mistral AI (Prescription Analysis)
MISTRAL_API_KEY=your_mistral_api_key_here

# Audio/File
AUDIO_MAX_SIZE_MB=50
AUDIO_ALLOWED_FORMATS=["mp3","wav","m4a","flac","ogg","mpeg","mpg"]
FILE_STORAGE_TYPE=local
```

Model choices:
- Intake questions (Step‑01): defaults to `gpt-4o-mini` (fast/cost‑effective). Change via `OPENAI_MODEL`.
- SOAP generation (Step‑03): defaults to `gpt-4` for higher quality. Change via `SOAP_MODEL`.

### Key API endpoints

Step‑01 & Step‑02 (Patients):

- `POST /patients/` – Register patient and start intake. Returns `patient_id`, `visit_id`, and first question.
- `POST /patients/consultations/answer` – Submit an answer, get next question or completion.
- `POST /patients/summary/previsit` – Generate pre‑visit summary once intake is completed.
- `GET /patients/{patient_id}/visits/{visit_id}/summary` – Retrieve stored pre‑visit summary.

Step‑03 (Notes):

- `POST /notes/transcribe` – Upload audio (multipart/form‑data) to transcribe a visit.
- `POST /notes/soap/generate` – Generate SOAP note using stored transcript + intake + pre‑visit summary.
- `GET /notes/{patient_id}/visits/{visit_id}/transcript` – Get stored transcript for a visit.
- `GET /notes/{patient_id}/visits/{visit_id}/soap` – Get stored SOAP note for a visit.

Prescriptions:

- `POST /prescriptions/upload` – Upload prescription images for AI analysis and structured data extraction.

### Example requests

Register a patient (Step‑01):

```bash
curl -X POST http://localhost:8000/patients/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "mobile": "9999999999",
    "age": 35,
    "disease": "Fever"
  }'
```

Answer an intake question:

```bash
curl -X POST http://localhost:8000/patients/consultations/answer \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "<PID>",
    "visit_id": "<VID>",
    "answer": "It started 3 days ago"
  }'
```

Transcribe audio (Step‑03):

```bash
curl -X POST http://localhost:8000/notes/transcribe \
  -F patient_id=<PID> \
  -F visit_id=<VID> \
  -F audio_file=@sample.m4a
```

Generate SOAP note (Step‑03):

```bash
curl -X POST http://localhost:8000/notes/soap/generate \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "<PID>",
    "visit_id": "<VID>"
  }'
```

Upload prescription images:

```bash
curl -X POST http://localhost:8000/prescriptions/upload \
  -F patient_id=<PID> \
  -F visit_id=<VID> \
  -F files=@prescription1.jpg \
  -F files=@prescription2.png
```

**Response example:**
```json
{
  "patient_id": "ABC123_001",
  "visit_id": "CONSULT-20241201-001",
  "medicines": [
    {
      "name": "Amoxicillin",
      "dose": "500mg",
      "frequency": "Three times daily",
      "duration": "7 days"
    },
    {
      "name": "Ibuprofen",
      "dose": "400mg",
      "frequency": "As needed for pain",
      "duration": "3 days"
    }
  ],
  "tests": [
    "Complete Blood Count (CBC)",
    "C-Reactive Protein"
  ],
  "instructions": [
    "Take with food",
    "Complete full course of antibiotics",
    "Return if symptoms worsen"
  ],
  "raw_text": "Dr. Smith - Amoxicillin 500mg TID x 7 days...",
  "processing_status": "success",
  "message": "Successfully processed 2 prescription images"
}
```

## 🩺 Troubleshooting

- ffmpeg not found during transcription
  - Install ffmpeg (see Quick Start) and restart the app.

- 500 INTERNAL_ERROR on intake/notes routes
  - Check server logs – the app now logs exception types and traces in development.
  - Verify `.env` is loaded and `OPENAI_API_KEY` is set in the running shell/process.

- 400 invalid_request_error about `response_format`
  - Fixed: SOAP generator no longer passes unsupported `response_format` to some models (e.g., 4o/mini).

- SOAP generation failed validation
  - Fixed: output normalization ensures required fields are present; validation thresholds relaxed for concise outputs.

- Port 8000 already in use
  - Stop previous instance: `pkill -f uvicorn` (macOS/Linux) or use a different port.

- Prescription upload fails with 401 Unauthorized
  - Check that `MISTRAL_API_KEY` is set in your `.env` file.
  - Verify the API key is valid and has sufficient credits.

- Prescription analysis returns empty results
  - Ensure uploaded images are clear and readable.
  - Check that images are in supported formats (JPG, PNG, GIF, BMP, WebP, TIFF).
  - Try with higher resolution images for better OCR accuracy.

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Architecture Guide](docs/architecture.md) - System design and architecture
- [Development Guide](docs/development.md) - Contributing guidelines

## 🏗️ Project Structure

```
clinicai/
├── src/clinicai/           # Source code
│   ├── api/               # FastAPI presentation layer
│   ├── application/       # Use cases and application services
│   ├── domain/           # Business logic and entities
│   ├── adapters/         # Infrastructure implementations
│   ├── core/             # Configuration and utilities
│   └── observability/    # Logging, tracing, metrics
├── tests/                # Test suite
├── docs/                 # Documentation
├── scripts/              # Utility scripts
└── docker/               # Containerization
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test types
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run with coverage
pytest --cov=src/clinicai tests/
```

## 🔧 Development

### Code Quality

```bash
# Format code
make format

# Check code quality
make lint

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🐳 Docker

```bash
# Build image
make docker-build

# Run container
make docker-run
```

## 📋 Available Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make test          # Run tests
make lint          # Check code quality
make format        # Format code
make clean         # Clean generated files
make dev           # Start development server
make docker-build  # Build Docker image
make docker-run    # Run Docker container
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure all checks pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 Email: support@clinicai.com
- 💬 Discord: [Clinic-AI Community](https://discord.gg/clinicai)
- 📖 Documentation: [docs.clinicai.com](https://docs.clinicai.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/clinicai/issues)

## 🙏 Acknowledgments

- FastAPI community for the excellent web framework
- MongoDB team for the robust database
- OpenAI and Mistral for AI capabilities
- Healthcare professionals for domain expertise

---

**Made with ❤️ for better healthcare**