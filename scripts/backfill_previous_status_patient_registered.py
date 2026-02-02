#!/usr/bin/env python3
"""
One-time backfill script to set previous_status = "patient_registered"
for existing visits where previous_status is null or missing.

Characteristics:
- Uses MongoDB updateMany on the visits collection.
- Idempotent and safe to run multiple times.

Usage:
    python scripts/backfill_previous_status_patient_registered.py --dry-run
    python scripts/backfill_previous_status_patient_registered.py --execute
"""

import argparse
import asyncio
import sys
from typing import Any, Dict

from motor.motor_asyncio import AsyncIOMotorClient

# Ensure we can import clinicai settings when run as a standalone script
sys.path.insert(0, "src")

from clinicai.core.config import get_settings  # noqa: E402


async def backfill_previous_status(dry_run: bool = True) -> Dict[str, Any]:
    """
    Backfill previous_status for existing visit records.

    Filter:
      - previous_status does not exist OR
      - previous_status is null

    Update:
      - set previous_status = "patient_registered"
    """
    settings = get_settings()
    client = AsyncIOMotorClient(settings.database.uri)
    db = client[settings.database.db_name]
    visits = db["visits"]

    # Match documents where previous_status is missing or null and not already set
    query = {
        "$or": [
            {"previous_status": {"$exists": False}},
            {"previous_status": None},
        ]
    }

    # Count documents that would be affected
    to_update = await visits.count_documents(query)

    result: Dict[str, Any] = {
        "matched_count": to_update,
        "modified_count": 0,
        "dry_run": dry_run,
    }

    if dry_run or to_update == 0:
        client.close()
        return result

    # Perform idempotent updateMany
    update_result = await visits.update_many(
        query,
        {"$set": {"previous_status": "patient_registered"}},
    )

    result["modified_count"] = update_result.modified_count
    client.close()
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill visits.previous_status to 'patient_registered' where null/missing."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many documents would be updated, without modifying data.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the backfill and update documents in-place.",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        parser.print_help()
        return

    dry_run = args.dry_run and not args.execute
    result = await backfill_previous_status(dry_run=dry_run)

    mode = "DRY RUN" if dry_run else "EXECUTE"
    print(f"[{mode}] previous_status backfill to 'patient_registered'")
    print(f"  Matched documents:  {result['matched_count']}")
    print(f"  Modified documents: {result['modified_count']}")


if __name__ == "__main__":
    asyncio.run(main())

