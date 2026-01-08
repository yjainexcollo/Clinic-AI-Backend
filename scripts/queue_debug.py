#!/usr/bin/env python3
"""
Debug script to print Azure Queue information.
Uses Settings (NOT direct env vars) to ensure consistency with app/worker.
"""

import os
import sys
from pathlib import Path

# Bootstrap: Add src directory to Python path for src-layout convenience
# This allows the script to be run directly without PYTHONPATH=./src
# Uses resolve() to get absolute paths, works from any working directory
_script_dir = Path(__file__).resolve().parent
_backend_dir = _script_dir.parent
_src_dir = _backend_dir / "src"
_src_dir_str = str(_src_dir)
if _src_dir_str not in sys.path:
    sys.path.insert(0, _src_dir_str)

import asyncio

from clinicai.adapters.queue.azure_queue_service import get_azure_queue_service
from clinicai.core.config import get_settings


def mask_connection_string(conn_str: str) -> str:
    """Mask sensitive parts of connection string for display."""
    if not conn_str:
        return "not set"
    try:
        # Show format: AccountName=XXX;AccountKey=***masked***;...
        parts = []
        for part in conn_str.split(";"):
            if part.startswith("AccountKey="):
                parts.append("AccountKey=***masked***")
            elif part.startswith("SharedAccessKey="):
                parts.append("SharedAccessKey=***masked***")
            else:
                parts.append(part)
        return ";".join(parts)
    except Exception:
        return "***error parsing***"


def extract_storage_account(conn_str: str) -> str:
    """Extract storage account name from connection string."""
    if not conn_str:
        return "unknown"
    try:
        for part in conn_str.split(";"):
            if part.startswith("AccountName="):
                return part.split("=", 1)[1]
    except Exception:
        pass
    return "unknown"


async def main():
    """Print queue information."""
    print("=" * 60)
    print("Azure Queue Storage Debug Information")
    print("=" * 60)
    print()

    try:
        # Get settings (handles all fallbacks)
        settings = get_settings()
        queue_settings = settings.azure_queue

        print("📋 Configuration (from Settings):")
        print(f"  Queue Name: {queue_settings.queue_name}")
        print(f"  Visibility Timeout: {queue_settings.visibility_timeout}s")
        print(f"  Poll Interval: {queue_settings.poll_interval}s")
        print(f"  Max Retry Attempts: {queue_settings.max_retry_attempts}")
        print()

        print("🔗 Connection String (masked):")
        conn_str = queue_settings.connection_string
        if not conn_str:
            # Try blob fallback
            blob_settings = settings.azure_blob
            conn_str = blob_settings.connection_string
            print(f"  Queue connection string: not set (using blob fallback)")
        else:
            print(f"  Queue connection string: {mask_connection_string(conn_str)}")

        storage_account = extract_storage_account(conn_str)
        print(f"  Storage Account: {storage_account}")
        print()

        # Check environment variables for debugging
        print("🔍 Environment Variables Check:")
        env_vars = {
            "AZURE_QUEUE_CONNECTION_STRING": os.getenv("AZURE_QUEUE_CONNECTION_STRING"),
            "AZURE_STORAGE_CONNECTION_STRING": os.getenv(
                "AZURE_STORAGE_CONNECTION_STRING"
            ),
            "AZURE_BLOB_CONNECTION_STRING": os.getenv("AZURE_BLOB_CONNECTION_STRING"),
            "AZURE_QUEUE_QUEUE_NAME": os.getenv("AZURE_QUEUE_QUEUE_NAME"),
            "AZURE_QUEUE_NAME": os.getenv("AZURE_QUEUE_NAME"),
        }
        for var_name, var_value in env_vars.items():
            if var_value:
                if "CONNECTION_STRING" in var_name:
                    print(f"  ✅ {var_name}: {mask_connection_string(var_value)}")
                else:
                    print(f"  ✅ {var_name}: {var_value}")
            else:
                print(f"  ⚪ {var_name}: not set")
        print()

        # Test queue connection
        print("🔌 Testing Queue Connection:")
        try:
            queue_service = get_azure_queue_service()
            # Access queue_client property to trigger initialization
            queue_client = queue_service.queue_client

            # Get queue length using the service method
            message_count = await queue_service.get_queue_length()

            print(f"  ✅ Connected successfully")
            print(f"  Queue Name: {queue_settings.queue_name}")
            print(f"  Approximate Message Count: {message_count}")

        except Exception as e:
            print(f"  ❌ Connection failed: {e}")
            print(f"     Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()

        print()
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
