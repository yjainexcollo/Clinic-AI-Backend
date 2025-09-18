"""
Prescription service interface for AI-powered prescription analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from fastapi import UploadFile


class PrescriptionService(ABC):
    """Abstract service for analyzing prescription images and extracting structured data."""

    @abstractmethod
    async def process_prescriptions(
        self, 
        patient_id: str, 
        visit_id: str, 
        files: List[UploadFile]
    ) -> Dict[str, Any]:
        """
        Process uploaded prescription images and extract structured data.
        
        Args:
            patient_id: Patient identifier
            visit_id: Visit identifier
            files: List of uploaded prescription image files
            
        Returns:
            Dictionary containing extracted medicines, tests, instructions, and raw text
        """
        pass

    @abstractmethod
    async def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from a single prescription image using OCR/vision AI.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text content
        """
        pass

    @abstractmethod
    async def parse_prescription_data(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse raw prescription text into structured data.
        
        Args:
            raw_text: Raw text extracted from prescription images
            
        Returns:
            Structured data with medicines, tests, and instructions
        """
        pass