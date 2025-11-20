"""
Debug endpoints for environment/storage diagnostics.

These are protected by the existing authentication middleware (X-API-Key).
"""
from fastapi import APIRouter
from azure.storage.blob import BlobServiceClient
from azure.storage.queue import QueueServiceClient
import os


router = APIRouter(prefix="/debug", tags=["Debug"])


def _mask(value: str) -> str:
    if not value:
        return "****"
    if len(value) <= 8:
        return value[:2] + "***"
    return f"{value[:4]}***{value[-4:]}"


@router.get("/env-storage")
async def debug_env_storage():
    slot = os.getenv("WEBSITE_SLOT_NAME", "Production")
    version = os.getenv("APP_VERSION", "unknown")
    blob_cs = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
    queue_cs = os.getenv("AZURE_QUEUE_CONNECTION_STRING", "")

    blob_is_kv = blob_cs.startswith("@Microsoft.KeyVault")
    queue_is_kv = queue_cs.startswith("@Microsoft.KeyVault")

    blob_account = "unknown"
    queue_account = "unknown"

    if not blob_is_kv and blob_cs:
        try:
            blob_account = BlobServiceClient.from_connection_string(blob_cs).account_name
        except Exception as exc:
            blob_account = f"ERROR: {exc}"
    else:
        blob_account = "KeyVault-reference"

    if not queue_is_kv and queue_cs:
        try:
            queue_account = QueueServiceClient.from_connection_string(queue_cs).account_name
        except Exception as exc:
            queue_account = f"ERROR: {exc}"
    else:
        queue_account = "KeyVault-reference"

    return {
        "slot": slot,
        "app_version": version,
        "blob_account": blob_account,
        "queue_account": queue_account,
        "blob_connection_masked": _mask(blob_cs),
        "queue_connection_masked": _mask(queue_cs),
        "blob_from_key_vault": blob_is_kv,
        "queue_from_key_vault": queue_is_kv,
    }


@router.get("/ping-blob-queue")
async def debug_ping_blob_queue():
    blob_cs = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
    queue_cs = os.getenv("AZURE_QUEUE_CONNECTION_STRING", "")

    blob_status = "not-configured"
    queue_status = "not-configured"

    if blob_cs:
        try:
            blob_client = BlobServiceClient.from_connection_string(blob_cs)
            blob_client.get_service_properties()
            blob_status = "ok"
        except Exception as exc:
            blob_status = f"ERROR: {exc}"

    if queue_cs:
        try:
            queue_client = QueueServiceClient.from_connection_string(queue_cs)
            queue_client.get_service_properties()
            queue_status = "ok"
        except Exception as exc:
            queue_status = f"ERROR: {exc}"

    return {"blob": blob_status, "queue": queue_status}

