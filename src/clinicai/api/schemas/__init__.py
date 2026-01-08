"""
API schemas package.
"""

# Common schemas
from .common import (
    BaseResponse,
    BlobFileInfo,
    ContactInfo,
    ErrorResponse,
    PersonalInfo,
    QuestionAnswer,
)

# Doctor preferences
from .doctor_preferences import (
    DoctorPreferencesResponse,
    PreVisitAIConfig,
    PreVisitSectionConfig,
    SoapAIConfig,
    UpsertDoctorPreferencesRequest,
)

# Intake schemas
from .intake import (
    AnswerIntakeRequest,
    AnswerIntakeResponse,
    EditAnswerRequest,
    EditAnswerResponse,
    IntakeSummarySchema,
)

# Medical schemas
from .medical import (
    ActionPlanRequest,
    ActionPlanResponse,
    MedicalRecommendation,
    MedicationInfo,
    PhysicalExam,
    SOAPNoteRequest,
    SOAPNoteResponse,
    TestOrderInfo,
    VitalsData,
)

# Patient registration schemas
from .patient_registration import (
    LatestVisitInfo,
    PatientListResponse,
    PatientSummarySchema,
    PatientWithVisitsSchema,
    RegisterPatientRequest,
    RegisterPatientResponse,
)

# Summary schemas
from .summaries import (
    PostVisitSummaryRequest,
    PostVisitSummaryResponse,
    PreVisitSummaryRequest,
    PreVisitSummaryResponse,
)

# Visit schemas
from .visits import (
    SoapNoteSchema,
    TranscriptionSessionSchema,
    VisitDetailSchema,
    VisitListItemSchema,
    VisitListResponse,
)

# Health schemas
# from .health import (
#     HealthResponse,
#     ReadyResponse,
#     LiveResponse
# )

__all__ = [
    # Common
    "BaseResponse",
    "ErrorResponse",
    "ContactInfo",
    "PersonalInfo",
    "QuestionAnswer",
    "BlobFileInfo",
    # Patient registration
    "RegisterPatientRequest",
    "RegisterPatientResponse",
    "PatientSummarySchema",
    "PatientWithVisitsSchema",
    "PatientListResponse",
    "LatestVisitInfo",
    # Intake
    "AnswerIntakeRequest",
    "AnswerIntakeResponse",
    "EditAnswerRequest",
    "EditAnswerResponse",
    "IntakeSummarySchema",
    # Summaries
    "PreVisitSummaryRequest",
    "PreVisitSummaryResponse",
    "PostVisitSummaryRequest",
    "PostVisitSummaryResponse",
    # Medical
    "VitalsData",
    "PhysicalExam",
    "SOAPNoteRequest",
    "SOAPNoteResponse",
    "MedicationInfo",
    "TestOrderInfo",
    "MedicalRecommendation",
    "ActionPlanRequest",
    "ActionPlanResponse",
    # Visits
    "VisitListItemSchema",
    "VisitDetailSchema",
    "VisitListResponse",
    "TranscriptionSessionSchema",
    "SoapNoteSchema",
    # Doctor preferences
    "PreVisitSectionConfig",
    "PreVisitAIConfig",
    "SoapAIConfig",
    "DoctorPreferencesResponse",
    "UpsertDoctorPreferencesRequest",
    # Health
    # "HealthResponse",
    # "ReadyResponse",
    # "LiveResponse"
]
