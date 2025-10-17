"""
Workflow type and status enums for conditional workflow support.
"""

from enum import Enum


class VisitWorkflowType(str, Enum):
    """Types of visit workflows."""
    SCHEDULED = "scheduled"  # With intake (Steps 1-5)
    WALK_IN = "walk_in"      # Without intake (Steps 3-5)


class VisitStatus(str, Enum):
    """Visit status values for different workflow types."""
    
    # Scheduled workflow statuses (existing)
    INTAKE = "intake"
    TRANSCRIPTION = "transcription"
    SOAP_GENERATION = "soap_generation"
    PRESCRIPTION_ANALYSIS = "prescription_analysis"
    COMPLETED = "completed"
    
    # Walk-in workflow statuses (sequential)
    WALK_IN_PATIENT = "walk_in_patient"           # Initial state
    TRANSCRIPTION_PENDING = "transcription_pending"  # Ready for audio upload
    TRANSCRIPTION_COMPLETED = "transcription_completed"  # Audio transcribed
    VITALS_PENDING = "vitals_pending"             # Ready for vitals input
    VITALS_COMPLETED = "vitals_completed"         # Vitals entered
    SOAP_PENDING = "soap_pending"                 # Ready for SOAP generation
    SOAP_COMPLETED = "soap_completed"             # SOAP generated
    POST_VISIT_PENDING = "post_visit_pending"     # Ready for post-visit summary
    POST_VISIT_COMPLETED = "post_visit_completed" # Post-visit summary generated
    
    # Common statuses
    CANCELLED = "cancelled"
