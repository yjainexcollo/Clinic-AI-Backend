"""
Question service interface for AI-powered question generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class QuestionService(ABC):
    """Abstract service for generating adaptive questions."""

    @abstractmethod
    async def generate_first_question(self, disease: str) -> str:
        """Generate the first question based on primary symptom (backward param name)."""
        pass

    @abstractmethod
    async def generate_next_question(
        self,
        disease: str,
        previous_answers: List[str],
        asked_questions: List[str],
        current_count: int,
        max_count: int,
    ) -> str:
        """Generate the next question based on context."""
        pass

    @abstractmethod
    async def should_stop_asking(
        self,
        disease: str,
        previous_answers: List[str],
        current_count: int,
        max_count: int,
    ) -> bool:
        """Determine if sufficient information has been collected."""
        pass

    @abstractmethod
    async def generate_pre_visit_summary(
        self, 
        patient_data: Dict[str, Any], 
        intake_answers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate pre-visit clinical summary from intake data."""
        pass

    @abstractmethod
    async def assess_completion_percent(
        self,
        disease: str,
        previous_answers: List[str],
        asked_questions: List[str],
        current_count: int,
        max_count: int,
    ) -> int:
        """Return completion percent (0-100) based on information coverage."""
        pass

    @abstractmethod
    async def assess_completion_percent(
        self,
        disease: str,
        previous_answers: List[str],
        asked_questions: List[str],
        current_count: int,
        max_count: int,
    ) -> int:
        """Return completion percent (0-100) based on information coverage."""
        pass

    @abstractmethod
    def is_medication_question(self, question: str) -> bool:
        """Check if a question is about medications and allows image upload."""
        pass