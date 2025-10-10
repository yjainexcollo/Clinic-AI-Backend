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
    max_questions: int = 10
    status: str = "in_progress"  # in_progress, completed, cancelled
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    pending_question: Optional[str] = None

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
    transcription_status: str = "pending"  # pending, processing, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    audio_duration_seconds: Optional[float] = None
    word_count: Optional[int] = None
    # Cached structured dialogue turns (ordered Doctor/Patient), to avoid re-structuring on the fly
    structured_dialogue: Optional[List[Dict[str, Any]]] = None


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
    symptom: str
    status: str = (
        "intake"  # intake, transcription, soap_generation, prescription_analysis, completed
    )
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Step 1: Pre-Visit Intake
    intake_session: Optional[IntakeSession] = None
    
    # Step 2: Pre-Visit Summary (EHR Storage)
    pre_visit_summary: Optional[Dict[str, Any]] = None
    
    # Step 3: Audio Transcription & SOAP Generation
    transcription_session: Optional[TranscriptionSession] = None
    soap_note: Optional[SoapNote] = None
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

    # Step-03: Audio Transcription & SOAP Generation Methods

    def start_transcription(self, audio_file_path: str) -> None:
        """Start the transcription process."""
        if self.status != "transcription":
            raise ValueError(f"Cannot start transcription. Current status: {self.status}")
        
        self.transcription_session = TranscriptionSession(
            audio_file_path=audio_file_path,
            transcription_status="processing",
            started_at=datetime.utcnow()
        )
        self.status = "transcription"
        self.updated_at = datetime.utcnow()

    def complete_transcription(self, transcript: str, audio_duration: Optional[float] = None, structured_dialogue: Optional[List[Dict[str, Any]]] = None) -> None:
        """Complete the transcription process."""
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
        
        # Move to SOAP generation status
        self.status = "soap_generation"
        self.updated_at = datetime.utcnow()

    def fail_transcription(self, error_message: str) -> None:
        """Mark transcription as failed."""
        if not self.transcription_session:
            raise ValueError("No active transcription session")
        
        self.transcription_session.transcription_status = "failed"
        self.transcription_session.error_message = error_message
        self.transcription_session.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def store_soap_note(self, soap_data: Dict[str, Any]) -> None:
        """Store generated SOAP note."""
        if self.status != "soap_generation":
            raise ValueError(f"Cannot store SOAP note. Current status: {self.status}")
        
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
        
        # Move to prescription analysis status
        self.status = "prescription_analysis"
        self.updated_at = datetime.utcnow()

    def store_vitals(self, vitals: Dict[str, Any]) -> None:
        """Store objective vitals for the visit."""
        self.vitals = vitals or {}
        self.updated_at = datetime.utcnow()

    def get_vitals(self) -> Optional[Dict[str, Any]]:
        """Get stored objective vitals, if any."""
        return self.vitals

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
        return self.status == "soap_generation" and self.is_transcription_complete()