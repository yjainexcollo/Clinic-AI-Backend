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