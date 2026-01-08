"""
Storage adapters for Clinic-AI.

This module contains storage-related adapters for file management,
including Azure Blob Storage integration.
"""

from .azure_blob_service import AzureBlobStorageService, get_azure_blob_service

__all__ = [
    "get_azure_blob_service",
    "AzureBlobStorageService",
]
