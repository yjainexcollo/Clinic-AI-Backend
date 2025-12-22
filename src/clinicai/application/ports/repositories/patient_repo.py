"""
Patient repository interface for data access abstraction.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ....domain.entities.patient import Patient
from ....domain.value_objects.patient_id import PatientId


class PatientRepository(ABC):
    """Abstract repository for patient data access."""

    @abstractmethod
    async def save(self, patient: Patient) -> Patient:
        """Save a patient to the repository."""
        pass

    @abstractmethod
    async def find_by_id(self, patient_id: PatientId, doctor_id: str) -> Optional[Patient]:
        """Find a patient by ID."""
        pass

    @abstractmethod
    async def find_by_name_and_mobile(
        self, name: str, mobile: str, doctor_id: str
    ) -> Optional[Patient]:
        """Find a patient by name and mobile number."""
        pass

    @abstractmethod
    async def exists_by_id(self, patient_id: PatientId, doctor_id: str) -> bool:
        """Check if a patient exists by ID."""
        pass

    @abstractmethod
    async def find_all(self, doctor_id: str, limit: int = 100, offset: int = 0) -> List[Patient]:
        """Find all patients with pagination."""
        pass

    @abstractmethod
    async def find_by_mobile(self, mobile: str, doctor_id: str) -> List[Patient]:
        """Find all patients with the same mobile number (family members)."""
        pass

    @abstractmethod
    async def delete(self, patient_id: PatientId, doctor_id: str) -> bool:
        """Delete a patient by ID."""
        pass
