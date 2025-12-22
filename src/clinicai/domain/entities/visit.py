"""Visit domain entity representing a single consultation visit.

Includes the intake session for Step-01 functionality. Formatting-only changes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..errors import (
    DuplicateQuestionError,
    IntakeAlreadyCompletedError,
    QuestionLimitExceededError,
)
from ..value_objects.question_id import QuestionId
from ..value_objects.visit_id import VisitId
from ..enums.workflow import VisitWorkflowType


@dataclass
class QuestionAnswer:
    """Question and answer pair."""

    question_id: QuestionId
    question: str
    answer: str
    timestamp: datetime
    question_number: int


@dataclass
class IntakeSession:
    """Intake session data for Step-01."""

    symptom: str = ""  # Make symptom optional with default empty string
    questions_asked: List[QuestionAnswer] = field(default_factory=list)
    current_question_count: int = 0
    max_questions: int = 12  # Default from settings (can be overridden)
    status: str = "in_progress"  # in_progress, completed, cancelled
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    pending_question: Optional[str] = None
    travel_questions_count: int = 0  # Track how many travel-related questions asked
    asked_categories: List[str] = field(default_factory=list)  # Code-truth topic tracking

    def __post_init__(self) -> None:
        """Normalize primary symptom string (no fixed whitelist)."""
        if self.symptom:
            self.symptom = self.symptom.strip()

    def add_question_answer(
        self,
        question: str,
        answer: str,
        attachment_image_paths: Optional[List[str]] = None,
        ocr_texts: Optional[List[str]] = None,
    ) -> None:
        # Check limit
        if self.current_question_count >= self.max_questions:
            raise QuestionLimitExceededError(
                self.current_question_count, self.max_questions
            )

        # Check for duplicate questions - exact text match only
        # Note: Semantic duplicate checking is handled at the application layer
        for qa in self.questions_asked:
            # Exact text match (case-insensitive)
            if qa.question.lower().strip() == question.lower().strip():
                raise DuplicateQuestionError(question)

        question_id = QuestionId.generate()
        question_answer = QuestionAnswer(
            question_id=question_id,
            question=question,
            answer=answer,
            timestamp=datetime.utcnow(),
            question_number=self.current_question_count + 1,
        )

        self.questions_asked.append(question_answer)
        self.current_question_count += 1
        # Clear pending once answered
        self.pending_question = None

    def truncate_after(self, index_inclusive: int) -> None:
        """Remove all questions after the given 0-based index, keep up to index, and reset state.

        If index_inclusive is the last kept item index, questions with indices > index_inclusive are dropped.
        If index_inclusive is -1, all questions are dropped.
        """
        if index_inclusive < -1 or index_inclusive >= len(self.questions_asked):
            # Nothing to do if index is out of bounds for truncation semantics
            if index_inclusive == len(self.questions_asked) - 1:
                return
        keep_upto = index_inclusive + 1
        self.questions_asked = self.questions_asked[:keep_upto]
        # Re-number remaining questions to be sequential starting at 1
        for idx, qa in enumerate(self.questions_asked, start=1):
            qa.question_number = idx
        self.current_question_count = len(self.questions_asked)
        # Reset status to in_progress and clear completion markers/pending
        self.status = "in_progress"
        self.completed_at = None
        self.pending_question = None

    def can_ask_more_questions(self) -> bool:
        """Check if more questions can be asked."""
        return (
            self.current_question_count < self.max_questions
            and self.status == "in_progress"
        )

    def is_complete(self) -> bool:
        """Check if intake is complete."""
        return self.status == "completed"

    def complete_intake(self) -> None:
        """Mark intake as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()

    def set_pending_question(self, question: Optional[str]) -> None:
        """Set the pending question to be asked next to the patient (not yet answered)."""
        self.pending_question = (question or None)

    def get_question_context(self) -> str:
        """Get context for AI to generate next question."""
        if not self.questions_asked:
            return f"Primary symptom: {self.symptom}. Generate the first symptom-focused question."

        # Get last 3 answers for context
        recent_answers = [qa.answer for qa in self.questions_asked[-3:]]
        asked_questions = [qa.question for qa in self.questions_asked]

        context = f"""
        Primary Symptom: {self.symptom}
        Recent answers: {'; '.join(recent_answers)}
        Already asked questions: {asked_questions}
        Current question count: {self.current_question_count}/{self.max_questions}

        Generate the next symptom-focused question. Do not repeat any already asked questions.
        """
        return context.strip()


@dataclass
class TranscriptionSession:
    """Transcription session data for Step-03."""
    
    audio_file_path: Optional[str] = None
    transcript: Optional[str] = None
    transcription_status: str = "pending"  # pending, queued, processing, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    worker_id: Optional[str] = None  # Worker that claimed/processed this job (format: "hostname:pid")
    audio_duration_seconds: Optional[float] = None
    word_count: Optional[int] = None
    # Cached structured dialogue turns (ordered Doctor/Patient), to avoid re-structuring on the fly
    structured_dialogue: Optional[List[Dict[str, Any]]] = None
    transcription_id: Optional[str] = None  # Azure Speech Service transcription job ID for tracking
    last_poll_status: Optional[str] = None  # Last polled status from Azure Speech Service (Succeeded, Running, Failed, etc.)
    last_poll_at: Optional[datetime] = None  # Timestamp of last status poll from Azure Speech Service
    # Observability timestamps for latency analysis
    enqueued_at: Optional[datetime] = None  # When job was enqueued to Azure Queue
    dequeued_at: Optional[datetime] = None  # When worker dequeued the job
    azure_job_created_at: Optional[datetime] = None  # When Azure Speech job was created
    first_poll_at: Optional[datetime] = None  # When first status poll was made
    results_downloaded_at: Optional[datetime] = None  # When transcription results were downloaded
    db_saved_at: Optional[datetime] = None  # When transcript was saved to database
    # Audio normalization metadata
    normalized_audio: Optional[bool] = None  # Whether audio was normalized/converted
    original_content_type: Optional[str] = None  # Original content type before normalization
    normalized_format: Optional[str] = None  # Format after normalization (e.g., wav_16khz_mono_pcm)
    file_content_type: Optional[str] = None  # Final content type used for transcription
    # Enqueue tracking (two-phase enqueue state machine)
    enqueue_state: Optional[str] = None  # "pending" | "queued" | "failed"
    enqueue_attempts: Optional[int] = None
    enqueue_last_error: Optional[str] = None
    enqueue_requested_at: Optional[datetime] = None
    enqueue_failed_at: Optional[datetime] = None
    queue_message_id: Optional[str] = None


@dataclass
class SoapNote:
    """SOAP note data for Step-03."""
    
    subjective: str
    objective: Dict[str, Any]
    assessment: str
    plan: str
    highlights: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    model_info: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None


@dataclass
class Visit:
    """Visit domain entity."""

    visit_id: VisitId
    patient_id: str  # Reference to patient
    doctor_id: str   # Owning doctor
    symptom: str
    workflow_type: VisitWorkflowType = VisitWorkflowType.SCHEDULED
    status: str = (
        "intake"  # intake, transcription, soap_generation, prescription_analysis, completed, walk_in_patient
    )
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Visit-specific travel history (moved from Patient - travel is visit-specific, not lifetime patient attribute)
    recently_travelled: bool = False

    # Step 1: Pre-Visit Intake
    intake_session: Optional[IntakeSession] = None
    
    # Step 2: Pre-Visit Summary (EHR Storage)
    pre_visit_summary: Optional[Dict[str, Any]] = None
    
    # Step 3: Audio Transcription & SOAP Generation
    transcription_session: Optional[TranscriptionSession] = None
    soap_note: Optional[SoapNote] = None
    # Optional per-visit SOAP template used to guide generation (not a global default).
    soap_template: Optional[Dict[str, Any]] = None
    # Objective Vitals (optional)
    vitals: Optional[Dict[str, Any]] = None
    # Step 4: Post-Visit Summary for patient sharing (stored JSON)
    post_visit_summary: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize intake session."""
        if self.intake_session is None:
            self.intake_session = IntakeSession(symptom=self.symptom)

    def add_question_answer(
        self,
        question: str,
        answer: str,
    ) -> None:
        """Add a question and answer to the intake session."""
        self.intake_session.add_question_answer(
            question,
            answer,
        )
        self.updated_at = datetime.utcnow()

    def set_pending_question(self, question: Optional[str]) -> None:
        """Set the next pending question on the intake session."""
        self.intake_session.set_pending_question(question)
        self.updated_at = datetime.utcnow()

    def complete_intake(self) -> None:
        """Complete the intake process."""
        self.intake_session.complete_intake()
        self.status = "transcription"  # Ready for next step
        self.updated_at = datetime.utcnow()

    def can_ask_more_questions(self) -> bool:
        """Check if more questions can be asked."""
        return self.intake_session.can_ask_more_questions()

    def is_intake_complete(self) -> bool:
        """Check if intake is complete."""
        return self.intake_session.is_complete()

    def get_question_context(self) -> str:
        """Get context for AI to generate next question."""
        return self.intake_session.get_question_context()

    def update_answer(self, question_number: int, new_answer: str) -> None:
        """Update an existing answer by question number."""
        if not self.intake_session or question_number < 1 or question_number > len(self.intake_session.questions_asked):
            raise ValueError("Invalid question number")
        self.intake_session.questions_asked[question_number - 1].answer = new_answer
        self.updated_at = datetime.utcnow()

    def truncate_questions_after(self, question_number: int) -> None:
        """Drop all questions after the given 1-based question number (keeping that one)."""
        index_inclusive = question_number - 1
        self.intake_session.truncate_after(index_inclusive)
        # If the first answer was changed, align primary symptom to the first answer
        if self.intake_session.current_question_count >= 1:
            self.symptom = self.intake_session.questions_asked[0].answer.strip()
        else:
            # No answers left, fall back to empty symptom
            self.symptom = ""
        self.updated_at = datetime.utcnow()

    def get_intake_summary(self) -> Dict[str, Any]:
        """Get summary of intake session."""
        return {
            "visit_id": self.visit_id.value,
            "symptom": self.symptom,
            "status": self.status,
            "questions_asked": [
                {
                    "question_id": qa.question_id.value,
                    "question": qa.question,
                    "answer": qa.answer,
                    "timestamp": qa.timestamp.isoformat(),
                    "question_number": qa.question_number,
                }
                for qa in self.intake_session.questions_asked
            ],
            "total_questions": self.intake_session.current_question_count,
            "max_questions": self.intake_session.max_questions,
            "intake_status": self.intake_session.status,
            "started_at": self.intake_session.started_at.isoformat(),
            "completed_at": (
                self.intake_session.completed_at.isoformat()
                if self.intake_session.completed_at
                else None
            ),
        }

    def store_pre_visit_summary(self, summary: str, red_flags: Optional[List[Dict[str, str]]] = None) -> None:
        """Store pre-visit summary in EHR (minimal schema)."""
        self.pre_visit_summary = {
            "summary": summary,
            "generated_at": datetime.utcnow().isoformat(),
            "red_flags": red_flags or [],
        }
        self.updated_at = datetime.utcnow()

    def get_pre_visit_summary(self) -> Optional[Dict[str, Any]]:
        """Get stored pre-visit summary from EHR."""
        return self.pre_visit_summary

    def has_pre_visit_summary(self) -> bool:
        """Check if pre-visit summary exists."""
        return self.pre_visit_summary is not None

    # Step-04: Post-Visit Summary Methods
    def store_post_visit_summary(self, summary: Dict[str, Any]) -> None:
        """Persist post-visit summary JSON for patient sharing."""
        self.post_visit_summary = {
            **(summary or {}),
            "stored_at": datetime.utcnow().isoformat(),
        }
        self.updated_at = datetime.utcnow()
    
    def get_post_visit_summary(self) -> Optional[Dict[str, Any]]:
        return self.post_visit_summary
    
    def has_post_visit_summary(self) -> bool:
        return self.post_visit_summary is not None

    # Workflow validation methods
    def is_scheduled_workflow(self) -> bool:
        """Check if this is a scheduled workflow (with intake)."""
        return self.workflow_type == VisitWorkflowType.SCHEDULED

    def is_walk_in_workflow(self) -> bool:
        """Check if this is a walk-in workflow (without intake)."""
        return self.workflow_type == VisitWorkflowType.WALK_IN

    def can_proceed_to_transcription(self) -> bool:
        """Check if visit can proceed to transcription based on workflow type."""
        if self.is_scheduled_workflow():
            # For scheduled: after vitals are filled (status can be pre_visit_summary_generated with vitals stored) or if already in transcription
            # Allow pre_visit_summary_generated if vitals are stored (vitals can be filled at this stage)
            if self.status == "pre_visit_summary_generated" and self.vitals:
                return True
            return self.status in ["vitals", "vitals_completed", "transcription"]
        elif self.is_walk_in_workflow():
            # For walk-in: after vitals are completed
            return self.status in ["vitals_completed", "transcription_pending", "transcription"]
        return False

    def can_proceed_to_vitals(self) -> bool:
        """Check if visit can proceed to vitals input."""
        if self.is_scheduled_workflow():
            # For scheduled: after pre-visit summary is generated
            return self.status in ["pre_visit_summary_generated", "vitals", "vitals_pending"]
        elif self.is_walk_in_workflow():
            # For walk-in: right after registration (walk_in_patient status)
            return self.status in ["walk_in_patient", "vitals_pending", "vitals"]
        return False

    def can_proceed_to_soap(self) -> bool:
        """Check if visit can proceed to SOAP generation."""
        if self.is_walk_in_workflow():
            return self.status in ["vitals_completed", "soap_pending", "soap_generation"]
        return False

    def can_proceed_to_post_visit(self) -> bool:
        """Check if visit can proceed to post-visit summary."""
        if self.is_walk_in_workflow():
            return self.status in ["soap_completed", "post_visit_pending", "post_visit_summary"]
        return False

    def get_available_steps(self) -> List[str]:
        """Get list of available workflow steps based on type and status."""
        if self.is_scheduled_workflow():
            return self._get_scheduled_workflow_steps()
        else:
            return self._get_walk_in_workflow_steps()

    def _get_scheduled_workflow_steps(self) -> List[str]:
        """Get available steps for scheduled workflow."""
        steps = []
        if self.status == "intake":
            steps.extend(["intake", "pre_visit_summary"])
        elif self.status == "pre_visit_summary_generated":
            # After pre-visit summary, vitals form comes next
            steps.extend(["vitals", "transcription", "soap_generation", "post_visit_summary"])
        elif self.status in ["vitals", "vitals_pending", "transcription"]:
            # After vitals (or during transcription), all subsequent steps are available
            steps.extend(["transcription", "soap_generation", "post_visit_summary"])
        else:
            # For other statuses, include common steps
            if self.status in ["soap_generation", "prescription_analysis"]:
                steps.extend(["soap_generation", "post_visit_summary"])
            elif self.status not in ["completed"]:
                steps.extend(["post_visit_summary"])
        return steps

    def _get_walk_in_workflow_steps(self) -> List[str]:
        """Get available steps for walk-in workflow (sequential)."""
        steps = []
        
        # Sequential workflow based on current status
        # Walk-in flow: registration -> vitals -> transcription -> soap -> post-visit
        if self.status == "walk_in_patient":
            steps.append("vitals")  # Vitals form comes first after registration
        elif self.status in ["vitals_pending", "vitals"]:
            steps.append("vitals")
        elif self.status == "vitals_completed":
            steps.append("transcription")  # Transcription comes after vitals
        elif self.status in ["transcription_pending", "transcription"]:
            steps.append("transcription")
        elif self.status == "transcription_completed":
            steps.append("soap_generation")
        elif self.status in ["soap_pending", "soap_generation"]:
            steps.append("soap_generation")
        elif self.status == "soap_completed":
            steps.append("post_visit_summary")
        elif self.status in ["post_visit_pending", "post_visit_summary"]:
            steps.append("post_visit_summary")
        
        return steps

    # Step-03: Audio Transcription & SOAP Generation Methods

    def queue_transcription(self, audio_file_path: Optional[str] = None, enqueued_at: Optional[datetime] = None) -> None:
        """
        Queue a transcription job (sets status to 'queued', not 'processing').
        Used when enqueuing to Azure Queue - actual processing is claimed by worker atomically.
        """
        if not self.can_proceed_to_transcription():
            raise ValueError(f"Cannot queue transcription. Current status: {self.status}")
        
        now = datetime.utcnow()
        # Only update if not already completed (idempotency)
        if self.transcription_session and self.transcription_session.transcription_status == "completed":
            return  # Already completed, don't overwrite
        
        # Preserve existing transcription_session fields if it exists (e.g., transcription_id, last_poll_status)
        if self.transcription_session:
            # Update existing session instead of replacing it
            if audio_file_path is not None:
                self.transcription_session.audio_file_path = audio_file_path
            self.transcription_session.transcription_status = "queued"
            self.transcription_session.started_at = None  # Will be set when worker claims it
            self.transcription_session.enqueued_at = enqueued_at or now
            self.transcription_session.dequeued_at = None  # Will be set when worker claims it
            self.transcription_session.worker_id = None  # Will be set when worker claims it
            self.transcription_session.error_message = None  # Clear any previous errors
        else:
            # Create new session only if none exists
            self.transcription_session = TranscriptionSession(
                audio_file_path=audio_file_path,
                transcription_status="queued",  # Set to queued, not processing
                started_at=None,  # Will be set when worker claims it
                enqueued_at=enqueued_at or now,
                dequeued_at=None,  # Will be set when worker claims it
                worker_id=None,  # Will be set when worker claims it
                error_message=None  # Clear any previous errors
            )
        if self.is_walk_in_workflow():
            self.status = "transcription"
        else:
            self.status = "transcription"
        self.updated_at = datetime.utcnow()

    # ------------------------------------------------------------------
    # Two-phase enqueue helpers (new, preferred API)
    # ------------------------------------------------------------------

    def _ensure_transcription_session(self) -> None:
        """Ensure transcription_session exists (backward-safe)."""
        if not self.transcription_session:
            self.transcription_session = TranscriptionSession()

    def mark_transcription_enqueue_pending(
        self,
        audio_file_path: Optional[str] = None,
        requested_at: Optional[datetime] = None,
    ) -> None:
        """
        Phase 1: mark that enqueue has been requested, before calling the queue.

        - Does NOT set transcription_status="queued".
        - Increments enqueue_attempts.
        - Resets worker/dequeued/transcription_id/started_at for safety.
        """
        now = requested_at or datetime.utcnow()
        self._ensure_transcription_session()
        ts = self.transcription_session

        if audio_file_path is not None:
            ts.audio_file_path = audio_file_path

        # Initialize attempts if None
        ts.enqueue_attempts = (ts.enqueue_attempts or 0) + 1
        ts.enqueue_state = "pending"
        ts.enqueue_requested_at = now
        ts.enqueue_last_error = None
        ts.enqueue_failed_at = None

        # Reset claim/processing-specific fields (idempotent safety)
        ts.dequeued_at = None
        ts.worker_id = None
        ts.transcription_id = None
        ts.error_message = None
        ts.started_at = None
        # Do NOT touch transcription_status here; it will be set to "queued" only on success

        self.updated_at = datetime.utcnow()

    def mark_transcription_enqueued(
        self,
        message_id: str,
        enqueued_at: Optional[datetime] = None,
    ) -> None:
        """
        Phase 2 success: queue message was enqueued successfully.

        - Sets enqueue_state="queued"
        - Sets queue_message_id
        - Sets transcription_status="queued"
        - Updates enqueued_at timestamp
        """
        now = enqueued_at or datetime.utcnow()
        self._ensure_transcription_session()
        ts = self.transcription_session

        ts.enqueue_state = "queued"
        ts.queue_message_id = message_id
        ts.enqueued_at = now
        ts.transcription_status = "queued"
        ts.error_message = None

        self.updated_at = datetime.utcnow()

    def mark_transcription_enqueue_failed(self, error: str) -> None:
        """
        Phase 2 failure: enqueue failed after retries.

        - Sets enqueue_state="failed"
        - Records enqueue_failed_at + enqueue_last_error
        - Does NOT set transcription_status="queued"
        """
        self._ensure_transcription_session()
        ts = self.transcription_session

        ts.enqueue_state = "failed"
        ts.enqueue_failed_at = datetime.utcnow()
        ts.enqueue_last_error = error

        # Preserve previous transcription_status; explicitly avoid setting it to "queued"
        self.updated_at = datetime.utcnow()

    def start_transcription(self, audio_file_path: str, enqueued_at: Optional[datetime] = None) -> None:
        """Start the transcription process."""
        if not self.can_proceed_to_transcription():
            raise ValueError(f"Cannot start transcription. Current status: {self.status}")
        
        now = datetime.utcnow()
        # Preserve existing transcription_session fields if it exists (e.g., transcription_id, last_poll_status)
        if self.transcription_session:
            # Update existing session instead of replacing it
            self.transcription_session.audio_file_path = audio_file_path
            self.transcription_session.transcription_status = "processing"
            if not self.transcription_session.started_at:
                self.transcription_session.started_at = now
            if enqueued_at:
                self.transcription_session.enqueued_at = enqueued_at
            elif not self.transcription_session.enqueued_at:
                self.transcription_session.enqueued_at = now
        else:
            # Create new session only if none exists
            self.transcription_session = TranscriptionSession(
                audio_file_path=audio_file_path,
                transcription_status="processing",
                started_at=now,
                enqueued_at=enqueued_at or now  # Use provided enqueued_at or fallback to now
            )
        if self.is_walk_in_workflow():
            self.status = "transcription"
        else:
            self.status = "transcription"
        self.updated_at = datetime.utcnow()

    def complete_transcription(self) -> None:
        """Complete the transcription process."""
        if self.transcription_session:
            self.transcription_session.transcription_status = "completed"
            self.transcription_session.completed_at = datetime.utcnow()
        
        if self.is_walk_in_workflow():
            self.status = "transcription_completed"
        else:
            self.status = "transcription"
        self.updated_at = datetime.utcnow()

    def start_vitals(self) -> None:
        """Start the vitals input process."""
        if not self.can_proceed_to_vitals():
            raise ValueError(f"Cannot start vitals. Current status: {self.status}")
        
        if self.is_walk_in_workflow():
            self.status = "vitals_pending"
        self.updated_at = datetime.utcnow()

    def complete_vitals(self) -> None:
        """Complete the vitals input process."""
        if self.is_walk_in_workflow():
            self.status = "vitals_completed"
        elif self.is_scheduled_workflow():
            # For scheduled visits: update status to allow transcription
            if self.status == "pre_visit_summary_generated":
                self.status = "vitals_completed"
            # If already in transcription or later stages, don't change status
        self.updated_at = datetime.utcnow()

    def start_soap_generation(self) -> None:
        """Start the SOAP generation process."""
        if not self.can_proceed_to_soap():
            raise ValueError(f"Cannot start SOAP generation. Current status: {self.status}")
        
        if self.is_walk_in_workflow():
            self.status = "soap_pending"
        else:
            self.status = "soap_generation"
        self.updated_at = datetime.utcnow()

    def complete_soap_generation(self) -> None:
        """Complete the SOAP generation process."""
        if self.is_walk_in_workflow():
            self.status = "soap_completed"
        else:
            self.status = "soap_generation"
        self.updated_at = datetime.utcnow()

    def start_post_visit_summary(self) -> None:
        """Start the post-visit summary generation process."""
        if not self.can_proceed_to_post_visit():
            raise ValueError(f"Cannot start post-visit summary. Current status: {self.status}")
        
        if self.is_walk_in_workflow():
            self.status = "post_visit_pending"
        else:
            self.status = "post_visit_summary"
        self.updated_at = datetime.utcnow()

    def complete_post_visit_summary(self) -> None:
        """Complete the post-visit summary generation process."""
        if self.is_walk_in_workflow():
            self.status = "post_visit_completed"
        else:
            self.status = "completed"
        self.updated_at = datetime.utcnow()

    def complete_transcription_with_data(self, transcript: str, audio_duration: Optional[float] = None, structured_dialogue: Optional[List[Dict[str, Any]]] = None) -> None:
        """Complete the transcription process with data."""
        if not self.transcription_session:
            raise ValueError("No active transcription session")
        
        self.transcription_session.transcript = transcript
        self.transcription_session.transcription_status = "completed"
        self.transcription_session.completed_at = datetime.utcnow()
        self.transcription_session.audio_duration_seconds = audio_duration
        self.transcription_session.word_count = len(transcript.split()) if transcript else 0
        
        # Store structured dialogue if provided
        if structured_dialogue:
            self.transcription_session.structured_dialogue = structured_dialogue
        
        # Update status based on workflow type
        if self.is_walk_in_workflow():
            self.status = "transcription_completed"  # Next step is vitals
        else:
            self.status = "soap_generation"  # Scheduled workflow goes directly to SOAP
        self.updated_at = datetime.utcnow()

    def fail_transcription(self, error_message: str) -> None:
        """Mark transcription as failed."""
        if not self.transcription_session:
            # Create a failed transcription session if none exists
            self.transcription_session = TranscriptionSession(
                audio_file_path="",
                transcription_status="failed",
                error_message=error_message,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
        else:
            self.transcription_session.transcription_status = "failed"
            self.transcription_session.error_message = error_message
            self.transcription_session.completed_at = datetime.utcnow()
        
        self.updated_at = datetime.utcnow()

    def store_soap_note(self, soap_data: Dict[str, Any]) -> None:
        """Store generated SOAP note."""
        # Allow storing SOAP note from various valid statuses
        valid_statuses_for_soap = ["soap_generation", "soap_pending"]
        
        # For walk-in workflows, also allow vitals_completed (next step after vitals)
        if self.is_walk_in_workflow():
            valid_statuses_for_soap.extend(["vitals_completed", "transcription_completed"])
        # For scheduled workflows, also allow transcription and transcription_completed
        elif self.is_scheduled_workflow():
            valid_statuses_for_soap.extend(["transcription", "transcription_completed"])
        
        if self.status not in valid_statuses_for_soap:
            raise ValueError(f"Cannot store SOAP note. Current status: {self.status}, valid statuses: {valid_statuses_for_soap}")
        
        self.soap_note = SoapNote(
            subjective=soap_data.get("subjective", ""),
            objective=soap_data.get("objective", {}),
            assessment=soap_data.get("assessment", ""),
            plan=soap_data.get("plan", ""),
            highlights=soap_data.get("highlights", []),
            red_flags=soap_data.get("red_flags", []),
            model_info=soap_data.get("model_info", {}),
            confidence_score=soap_data.get("confidence_score")
        )
        
        # Update status appropriately based on workflow type
        if self.is_walk_in_workflow():
            self.status = "soap_completed"
        else:
            self.status = "prescription_analysis"
        self.updated_at = datetime.utcnow()

    def store_vitals(self, vitals: Dict[str, Any]) -> None:
        """Store objective vitals for the visit."""
        self.vitals = vitals or {}
        self.updated_at = datetime.utcnow()

    def get_vitals(self) -> Optional[Dict[str, Any]]:
        """Get stored objective vitals, if any."""
        return self.vitals

    # -------------------------------------------------------------------------
    # SOAP Template helpers (per-visit, optional)
    # -------------------------------------------------------------------------

    def store_soap_template(self, template: Optional[Dict[str, Any]]) -> None:
        """Store a SOAP template for this visit (for auditing or reuse within visit).

        This does not make the template a global default; it is scoped to this visit.
        Passing None will clear the template.
        """
        self.soap_template = template or None
        self.updated_at = datetime.utcnow()

    def get_soap_template(self) -> Optional[Dict[str, Any]]:
        """Return the stored SOAP template for this visit, if any."""
        return self.soap_template

    def get_transcript(self) -> Optional[str]:
        """Get the transcript if available."""
        if self.transcription_session and self.transcription_session.transcript:
            return self.transcription_session.transcript
        return None

    def get_soap_note(self) -> Optional[SoapNote]:
        """Get the SOAP note if available."""
        return self.soap_note

    def is_transcription_complete(self) -> bool:
        """Check if transcription is complete."""
        return (self.transcription_session and 
                self.transcription_session.transcription_status == "completed")

    def is_soap_generated(self) -> bool:
        """Check if SOAP note is generated."""
        return self.soap_note is not None

    def can_start_transcription(self) -> bool:
        """Check if transcription can be started."""
        return self.status == "transcription"

    def can_generate_soap(self) -> bool:
        """Check if SOAP can be generated."""
        has_transcript = self.is_transcription_complete()
        if not has_transcript:
            return False
        
        if self.workflow_type == VisitWorkflowType.WALK_IN:
            # For walk-in: can generate SOAP after transcription and vitals are completed
            return self.status in ["transcription_completed", "vitals", "vitals_completed", "soap_generation", "soap_pending"]
        else:
            # For scheduled: can generate SOAP if transcript exists AND vitals exist
            # This is more permissive - allows generation regardless of exact status
            # Status-based checks (fallback for backward compatibility):
            if self.status == "soap_generation":
                return True
            if self.status == "transcription_completed":
                return True
            # Most permissive: if transcript exists and vitals exist, allow SOAP generation
            # This handles cases where status wasn't updated correctly
            if has_transcript and self.vitals:
                return True
            # Fallback: status is transcription and vitals exist
            if self.status == "transcription" and self.vitals:
                return True
            return False

    def store_vitals(self, vitals_data: Dict[str, Any]) -> None:
        """Store vitals data for the visit."""
        self.vitals = vitals_data
        self.updated_at = datetime.utcnow()