"""
SOAP generation service interface for medical documentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class SoapService(ABC):
    """Abstract service for SOAP note generation."""

    @abstractmethod
    async def generate_soap_note(
        self,
        transcript: str,
        patient_context: Optional[Dict[str, Any]] = None,
        intake_data: Optional[Dict[str, Any]] = None,
        pre_visit_summary: Optional[Dict[str, Any]] = None,
        vitals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate SOAP note from transcript and context.
        
        Args:
            transcript: Conversation transcript
            patient_context: Patient demographics and basic info
            intake_data: Pre-visit intake responses
            pre_visit_summary: Pre-visit summary from Step-02
            vitals: Objective vitals captured via vitals form
            
        Returns:
            Dict containing structured SOAP note data
        """
        pass

    @abstractmethod
    async def validate_soap_structure(self, soap_data: Dict[str, Any]) -> bool:
        """
        Validate SOAP note structure and completeness.
        
        Args:
            soap_data: SOAP note data to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    async def generate_post_visit_summary(
        self,
        patient_data: Dict[str, Any],
        soap_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate post-visit summary for patient sharing.
        
        Args:
            patient_data: Patient demographics and basic info
            soap_data: SOAP note data from completed visit
            
        Returns:
            Dict containing structured post-visit summary data
        """
        pass
