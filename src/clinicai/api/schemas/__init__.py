"""
API schemas package.
"""

# Common schemas
from .common import (
    BaseResponse,
    ErrorResponse,
    ContactInfo,
    PersonalInfo,
    QuestionAnswer,
    BlobFileInfo
)

# Patient registration schemas
from .patient_registration import (
    RegisterPatientRequest,
    RegisterPatientResponse,
    PatientSummarySchema,
    PatientWithVisitsSchema,
    PatientListResponse,
    LatestVisitInfo
)

# Intake schemas
from .intake import (
    AnswerIntakeRequest,
    AnswerIntakeResponse,
    EditAnswerRequest,
    EditAnswerResponse,
    IntakeSummarySchema
)

# Summary schemas
from .summaries import (
    PreVisitSummaryRequest,
    PreVisitSummaryResponse,
    PostVisitSummaryRequest,
    PostVisitSummaryResponse
)

# Medical schemas
from .medical import (
    VitalsData,
    PhysicalExam,
    SOAPNoteRequest,
    SOAPNoteResponse,
    MedicationInfo,
    TestOrderInfo,
    MedicalRecommendation,
    ActionPlanRequest,
    ActionPlanResponse
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
    
    # Health
    # "HealthResponse",
    # "ReadyResponse",
    # "LiveResponse"
]
