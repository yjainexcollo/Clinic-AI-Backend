"""
Workflow type and status enums for conditional workflow support.
"""

from enum import Enum


class VisitWorkflowType(str, Enum):
    """Types of visit workflows."""

    SCHEDULED = "scheduled"  # With intake (Steps 1-5)
    WALK_IN = "walk_in"  # Without intake (Steps 3-5)


class VisitStatus(str, Enum):
    """Visit status values for different workflow stages.

    We intentionally expose only high-level flow names (no *_pending /
    *_in_progress / *_completed suffixes). The eight primary stages are:

    - patient_registered
    - intake
    - pre_visit_summary
    - vitals
    - transcription
    - soap_generation
    - post_visit_summary
    - completed
    """

    # Core visit lifecycle (shared between scheduled and walk-in)
    PATIENT_REGISTERED = "patient_registered"  # After patient registration
    INTAKE = "intake"  # Intake form in progress/completed
    PRE_VISIT_SUMMARY = "pre_visit_summary"  # Pre-visit summary generated
    VITALS = "vitals"  # Vitals form (entered / editable)
    TRANSCRIPTION = "transcription"  # Transcript captured / in progress
    SOAP_GENERATION = "soap_generation"  # SOAP / prescription analysis step
    POST_VISIT_SUMMARY = "post_visit_summary"  # Post-visit summary generated
    COMPLETED = "completed"  # Visit fully completed

    # Optional / legacy statuses
    WALK_IN_PATIENT = "walk_in_patient"  # Legacy initial state for walk-in workflow
    CANCELLED = "cancelled"
