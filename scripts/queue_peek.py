#!/usr/bin/env python3
"""
Peek at the first 5 messages in the Azure Queue without removing them.
Useful for debugging queue contents and payload structure.
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

import json
from datetime import datetime

from clinicai.core.config import get_settings
from clinicai.adapters.queue.azure_queue_service import get_azure_queue_service
import asyncio


def truncate_content(content: str, max_chars: int = 500) -> str:
    """Truncate content safely for display."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + f"... (truncated, total length: {len(content)} chars)"


async def main():
    """Peek at queue messages."""
    print("=" * 70)
    print("Azure Queue Message Peek")
    print("=" * 70)
    print()
    
    try:
        settings = get_settings()
        queue_settings = settings.azure_queue
        
        print(f"📋 Queue: {queue_settings.queue_name}")
        print()
        
        queue_service = get_azure_queue_service()
        queue_client = queue_service.queue_client
        
        # Peek messages (doesn't remove them from queue)
        messages = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(queue_client.peek_messages(max_messages=5))
        )
        
        if not messages:
            print("✅ Queue is empty (no messages to peek)")
            print()
            print("=" * 70)
            return
        
        print(f"📨 Found {len(messages)} message(s):")
        print()
        
        for idx, message in enumerate(messages, 1):
            print(f"--- Message #{idx} ---")
            print(f"Message ID: {message.id}")
            
            # Try to parse content as JSON
            try:
                content_dict = json.loads(message.content)
                print(f"Content (JSON):")
                print(json.dumps(content_dict, indent=2))
            except json.JSONDecodeError:
                print(f"Content (raw, first 500 chars):")
                print(truncate_content(message.content, 500))
            
            print()
        
        print("=" * 70)
        print("Note: These messages are still in the queue (peek only)")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

