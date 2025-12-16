#!/usr/bin/env python3
"""
Print Azure Queue statistics and current configuration.
Shows approximate message count, queue properties, and settings.
"""

import sys
import os
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

from clinicai.core.config import get_settings
from clinicai.adapters.queue.azure_queue_service import get_azure_queue_service
import asyncio


def mask_connection_string(conn_str: str) -> str:
    """Mask sensitive parts of connection string for display."""
    if not conn_str:
        return "not set"
    try:
        parts = []
        for part in conn_str.split(';'):
            if part.startswith('AccountKey='):
                parts.append('AccountKey=***masked***')
            elif part.startswith('SharedAccessKey='):
                parts.append('SharedAccessKey=***masked***')
            else:
                parts.append(part)
        return ';'.join(parts)
    except Exception:
        return "***error parsing***"


async def main():
    """Print queue statistics."""
    print("=" * 70)
    print("Azure Queue Statistics")
    print("=" * 70)
    print()
    
    try:
        settings = get_settings()
        queue_settings = settings.azure_queue
        
        print("📋 Configuration (from Settings):")
        print(f"  Queue Name: {queue_settings.queue_name}")
        print(f"  Visibility Timeout: {queue_settings.visibility_timeout}s")
        print(f"  Poll Interval: {queue_settings.poll_interval}s")
        print(f"  Max Retry Attempts: {queue_settings.max_retry_attempts}")
        print(f"  Max Dequeue Count: {queue_settings.max_dequeue_count}")
        print()
        
        print("🔗 Connection String (masked):")
        conn_str = queue_settings.connection_string
        if not conn_str:
            blob_settings = settings.azure_blob
            conn_str = blob_settings.connection_string
            print(f"  Queue connection string: not set (using blob fallback)")
        else:
            print(f"  Queue connection string: {mask_connection_string(conn_str)}")
        print()
        
        # Get queue statistics
        print("📊 Queue Statistics:")
        try:
            queue_service = get_azure_queue_service()
            message_count = await queue_service.get_queue_length()
            
            print(f"  ✅ Connected successfully")
            print(f"  Approximate Message Count: {message_count}")
            
            # Get queue properties for additional info
            try:
                queue_client = queue_service.queue_client
                properties = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: queue_client.get_queue_properties()
                )
                # Safely access last_modified - may not exist in all SDK versions
                last_modified = getattr(properties, "last_modified", None)
                if last_modified:
                    print(f"  Last Modified: {last_modified.isoformat()}")
                else:
                    print(f"  Last Modified: (not available in this SDK object)")
            except Exception as e:
                print(f"  ⚠️  Could not fetch queue properties: {e}")
            
        except Exception as e:
            print(f"  ❌ Connection failed: {e}")
            print(f"     Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
        
        print()
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

