"""Queue adapters for background job processing."""

from .azure_queue_service import AzureQueueService, get_azure_queue_service

__all__ = [
    "AzureQueueService",
    "get_azure_queue_service",
]
