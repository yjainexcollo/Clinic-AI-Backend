"""
Action Plan Service interface for generating Action and Plan from medical transcripts.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ActionPlanService(ABC):
    """Abstract interface for Action Plan generation service."""

    @abstractmethod
    async def generate_action_plan(
        self, 
        transcript: str, 
        structured_dialogue: list[dict] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate Action and Plan from medical transcript.
        
        Args:
            transcript: Raw transcript text
            structured_dialogue: Optional structured dialogue turns
            language: Language of the transcript
            
        Returns:
            Dictionary containing:
            - action: List of recommended actions
            - plan: Treatment plan details
            - confidence: Confidence score (0-1)
            - reasoning: Explanation of the recommendations
        """
        pass
