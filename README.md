# Clinic-AI Backend

A comprehensive medical consultation platform that provides AI-powered patient intake, audio transcription, SOAP note generation, and prescription analysis. Built with FastAPI and following Clean Architecture principles.

## 🚀 Features

### Core Functionality
- **Patient Management**: Register patients and manage their medical records with opaque ID encoding
- **Intelligent Intake**: AI-powered adaptive questioning system with doctor-configurable preferences
- **Audio Transcription**: Convert consultation audio to text using OpenAI Whisper (local or API)
- **SOAP Notes**: Generate structured medical notes from transcripts using OpenAI
- **Prescription Analysis**: Extract structured data from prescription images using Azure OpenAI
- **Multi-language Support**: English and Spanish language support throughout the platform
- **Doctor Preferences**: Configurable intake session parameters and question categories

### Advanced Features
- **Structured Dialogue**: Clean PII and structure transcripts into Doctor/Patient conversations
- **Action Plans**: Generate comprehensive treatment plans from consultation transcripts
- **Vitals Management**: Store and retrieve objective patient vitals data
- **Image Management**: Upload, store, and analyze medication images with OCR
- **Audio Management**: Comprehensive audio file storage and streaming capabilities
- **Background Processing**: Asynchronous task processing for transcription and analysis

## 🏗️ Architecture

This backend follows **Clean Architecture** principles with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   FastAPI       │  │   API Schemas   │  │   Routers    │ │
│  │   Application   │  │   (Pydantic)    │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Use Cases     │  │   DTOs          │  │   Ports      │ │
│  │                 │  │                 │  │   (Interfaces)│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                     Domain Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Entities      │  │  Value Objects  │  │   Events     │ │
│  │   (Patient,     │  │   (IDs, etc.)   │  │              │ │
│  │    Visit)       │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Adapters      │  │   External      │  │   Database   │ │
│  │   (MongoDB,     │  │   Services      │  │   (Beanie    │ │
│  │    OpenAI,      │  │   (OpenAI,      │  │    ODM)      │ │
│  │    Azure)       │  │    Azure)       │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

- **Presentation Layer**: FastAPI endpoints, request/response schemas, and routing
- **Application Layer**: Use cases, business logic orchestration, and data transfer objects
- **Domain Layer**: Core business entities, value objects, and domain events
- **Infrastructure Layer**: External service integrations, database persistence, and adapters

## 🛠️ Tech Stack

### Core Framework
- **FastAPI**: High-performance async web framework
- **Pydantic v2**: Data validation and settings management
- **Python 3.11+**: Modern Python with async/await support

### Database & Storage
- **MongoDB**: Document database for flexible medical data storage
- **Beanie ODM**: Async MongoDB object-document mapper
- **Motor**: Async MongoDB driver

### AI & External Services
- **OpenAI**: Question generation, SOAP notes, post-visit summaries, and transcription
- **Azure OpenAI**: Prescription image analysis and OCR
- **Whisper**: Audio transcription (local or OpenAI API)

### Development & Deployment
- **Docker**: Containerization
- **Docker Compose**: Local development environment
- **Render**: Cloud deployment platform
- **Make**: Build automation

## 📁 Project Structure

```
backend/
├── src/clinicai/
│   ├── api/                    # FastAPI presentation layer
│   │   ├── routers/           # API endpoint definitions
│   │   │   ├── health.py      # Health check endpoints
│   │   │   ├── patients.py    # Patient management
│   │   │   ├── notes.py       # Medical notes & transcription
│   │   │   ├── prescriptions.py # Prescription analysis
│   │   │   ├── audio.py       # Audio file management
│   │   │   ├── doctor.py      # Doctor preferences
│   │   │   ├── transcription.py # Ad-hoc transcription
│   │   │   └── intake.py      # Independent intake sessions
│   │   ├── schemas/           # Pydantic request/response models
│   │   └── deps.py            # FastAPI dependency injection
│   ├── application/           # Application layer
│   │   ├── use_cases/         # Business logic orchestration
│   │   ├── dto/               # Data transfer objects
│   │   ├── ports/             # Interface definitions
│   │   └── uow.py             # Unit of work pattern
│   ├── domain/                # Domain layer
│   │   ├── entities/          # Core business entities
│   │   ├── value_objects/     # Domain value objects
│   │   └── events/            # Domain events
│   ├── adapters/              # Infrastructure layer
│   │   ├── db/                # Database adapters
│   │   ├── external/          # External service adapters
│   │   └── services/          # Service implementations
│   ├── core/                  # Core utilities
│   │   ├── config.py          # Configuration management
│   │   ├── container.py       # Dependency injection
│   │   └── utils/             # Utility functions
│   ├── observability/         # Monitoring & logging
│   └── app.py                 # FastAPI application factory
├── docker/                    # Docker configuration
├── scripts/                   # Utility scripts
├── storage/                   # Local file storage
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── docker-compose.yml        # Local development setup
├── Makefile                  # Build automation
└── swagger.yaml              # API documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- MongoDB (local or cloud)
- OpenAI API key

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Clinic-AI/backend
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   ```

3. **Configure environment variables**
   ```bash
   # Required
   OPENAI_API_KEY=your_openai_api_key
   MONGODB_URL=mongodb://localhost:27017/clinicai
   
   # Optional
   TRANSCRIPTION_SERVICE=azure_speech  # Azure Speech Service with speaker diarization
   ```

### Local Development

1. **Start services with Docker Compose**
   ```bash
   make dev
   # or
   docker-compose up -d
   ```

2. **Install Python dependencies**
   ```bash
   make install
   # or
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   make run
   # or
   python -m uvicorn src.clinicai.app:create_app --factory --reload
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Production Deployment

1. **Build Docker image**
   ```bash
   make build
   ```

2. **Deploy to Render**
   ```bash
   # Configure render.yaml with your settings
   make deploy
   ```

## 📚 API Documentation

### Core Endpoint

#### Health & Monitoring
- `GET /health` - Basic health check
- `GET /health/ready` - Readiness check (database, external services)
- `GET /health/live` - Liveness check

#### Patient Management
- `POST /patients/` - Register new patient
- `POST /patients/consultations/answer` - Answer intake questions
- `PATCH /patients/consultations/answer` - Edit previous answers
- `GET /patients/{patient_id}/visits/{visit_id}/summary` - Get pre-visit summary
- `GET /patients/{patient_id}/visits/{visit_id}/summary/postvisit` - Get post-visit summary

#### Medical Notes & Transcription
- `POST /notes/transcribe` - Queue audio transcription
- `POST /notes/soap/generate` - Generate SOAP note
- `POST /notes/vitals` - Store patient vitals
- `GET /patients/{patient_id}/visits/{visit_id}/transcript` - Get transcript
- `GET /patients/{patient_id}/visits/{visit_id}/soap` - Get SOAP note

#### Prescription Analysis
- `POST /prescriptions/upload` - Upload and analyze prescription images

#### Audio Management
- `GET /audio/` - List audio files
- `GET /audio/{audio_id}/stream` - Stream audio for playback
- `DELETE /audio/{audio_id}` - Delete audio file

#### Doctor Preferences
- `GET /doctor/preferences` - Get doctor preferences
- `POST /doctor/preferences` - Save doctor preferences

#### Ad-hoc Services
- `POST /transcription` - Ad-hoc audio transcription
- `POST /adhoc/action-plan` - Generate action plan
- `GET /adhoc/{adhoc_id}/action-plan` - Get action plan

### Request/Response Examples

#### Register Patient
```bash
curl -X POST "http://localhost:8000/patients/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "age": 35,
    "gender": "male",
    "chief_complaint": "Chest pain and shortness of breath",
    "language": "en"
  }'
```

#### Answer Intake Question
```bash
curl -X POST "http://localhost:8000/patients/consultations/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "pat_abc123def456",
    "visit_id": "vis_xyz789uvw012",
    "question_id": "q_001",
    "answer": "The pain is sharp and located in the center of my chest."
  }'
```

#### Transcribe Audio
```bash
curl -X POST "http://localhost:8000/notes/transcribe" \
  -F "patient_id=pat_abc123def456" \
  -F "visit_id=vis_xyz789uvw012" \
  -F "audio_file=@consultation.mp3" \
  -F "language=en"
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for AI services | - | Yes |
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017/clinicai` | Yes |
| `TRANSCRIPTION_SERVICE` | Transcription service type (`openai` or `local`) | `openai` | No |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:3000"]` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |

### Service Configuration

#### OpenAI Settings
```python
OPENAI_MODEL = "gpt-4"  # Model for text generation
OPENAI_TEMPERATURE = 0.3  # Response creativity
OPENAI_MAX_TOKENS = 2000  # Maximum response length
```

#### Whisper Settings
```python
WHISPER_MODEL = "base"  # Whisper model size
WHISPER_LANGUAGE = "en"  # Default language
WHISPER_MEDICAL_CONTEXT = True  # Medical context optimization
```

## 🧪 Testing

### Run Tests
```bash
make test
# or
pytest tests/
```

### Test Coverage
```bash
make test-coverage
# or
pytest --cov=src tests/
```

### Integration Tests
```bash
make test-integration
# or
pytest tests/integration/
```

## 📊 Monitoring & Observability

### Health Checks
- **Basic Health**: `/health` - Service availability
- **Readiness**: `/health/ready` - Database and external service connectivity
- **Liveness**: `/health/live` - Process health

### Logging
- Structured JSON logging
- Request/response logging
- Error tracking and alerting
- Performance metrics

### Metrics
- Request duration and throughput
- Error rates by endpoint
- External service response times
- Database query performance

## 🔒 Security

### Data Protection
- **Opaque Patient IDs**: External-facing IDs are encoded/encrypted
- **PII Handling**: Automatic PII detection and cleaning in transcripts
- **Secure Storage**: Encrypted data at rest in MongoDB
- **CORS Configuration**: Configurable cross-origin resource sharing

### API Security
- **Input Validation**: Comprehensive Pydantic schema validation
- **Rate Limiting**: Configurable rate limits per endpoint
- **Error Handling**: Secure error responses without sensitive data exposure

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t clinic-ai-backend .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e MONGODB_URL=your_mongodb_url \
  clinic-ai-backend
```

### Render Deployment
The project includes `render.yaml` for easy deployment to Render:

```yaml
services:
  - type: web
    name: clinic-ai-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python -m uvicorn src.clinicai.app:create_app --factory --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: MONGODB_URL
        sync: false
```

### Environment-Specific Configuration
- **Development**: Local MongoDB, debug logging
- **Staging**: Cloud MongoDB, structured logging
- **Production**: High-availability MongoDB, monitoring integration

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the test suite
5. Submit a pull request

### Code Standards
- **Type Hints**: All functions must have type annotations
- **Docstrings**: Comprehensive documentation for all public APIs
- **Testing**: Minimum 80% test coverage
- **Linting**: Black formatting, isort imports, flake8 linting

### Pre-commit Hooks
```bash
make install-hooks
# or
pre-commit install
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

#### MongoDB Connection Issues
```bash
# Check MongoDB status
docker-compose ps mongodb

# View MongoDB logs
docker-compose logs mongodb

# Restart MongoDB
docker-compose restart mongodb
```

#### OpenAI API Issues
```bash
# Verify API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Check rate limits
# Monitor usage in OpenAI dashboard
```

#### Transcription Service Issues
```bash
# Check service type
echo $TRANSCRIPTION_SERVICE

# Test local Whisper
python -c "import whisper; print('Whisper available')"

# Test OpenAI Whisper
curl -X POST "https://api.openai.com/v1/audio/transcriptions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "file=@test.mp3" \
  -F "model=whisper-1"
```

### Performance Optimization

#### Database Optimization
- Index frequently queried fields
- Use connection pooling
- Monitor query performance

#### AI Service Optimization
- Cache frequently used responses
- Implement request batching
- Monitor token usage and costs

#### Memory Management
- Use streaming for large file uploads
- Implement proper cleanup for temporary files
- Monitor memory usage in production

## 📞 Support

For support and questions:
- **Documentation**: Check this README and API docs
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Email**: support@clinic-ai.com

## 🔄 Changelog

### Version 1.0.0
- Initial release with core functionality
- Patient management and intake system
- Audio transcription and SOAP note generation
- Prescription analysis with Azure OpenAI
- Multi-language support (English/Spanish)
- Comprehensive API documentation

---

**Built with ❤️ for better healthcare through AI**# Azure deployment fix

# Trigger deployment with Service Principal
# Force fresh deployment
