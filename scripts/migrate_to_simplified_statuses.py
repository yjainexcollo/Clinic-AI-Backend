"""
Migration script to update visit statuses from detailed codes to simplified flow names.

This script updates all visit records in MongoDB to use simplified status names:
- Removes _pending, _in_progress, _completed, _processing suffixes
- Maps old statuses to new simplified ones:
  - intake_pending, intake_in_progress -> intake
  - pre_visit_summary_generated -> pre_visit_summary
  - vitals_pending, vitals_in_progress, vitals_completed -> vitals
  - transcription_pending, transcription_processing, transcription_completed -> transcription
  - soap_pending, soap_completed, prescription_analysis -> soap_generation
  - post_visit_pending, post_visit_completed -> post_visit_summary

Usage:
    python scripts/migrate_to_simplified_statuses.py [--dry-run]
"""

import asyncio
import sys
from typing import Any, Dict

from motor.motor_asyncio import AsyncIOMotorClient

# Add src to path
sys.path.insert(0, "src")

from clinicai.core.config import get_settings


# Mapping from old detailed statuses to new simplified ones
STATUS_MAPPING: Dict[str, str] = {
    # Scheduled intake / pre-visit
    "intake_pending": "intake",
    "intake_in_progress": "intake",
    "pre_visit_summary_generated": "pre_visit_summary",
    # Vitals
    "vitals_pending": "vitals",
    "vitals_in_progress": "vitals",
    "vitals_completed": "vitals",
    # Transcription
    "transcription_pending": "transcription",
    "transcription_processing": "transcription",
    "transcription_completed": "transcription",
    # SOAP / prescription analysis
    "soap_pending": "soap_generation",
    "soap_completed": "soap_generation",
    "prescription_analysis": "soap_generation",
    # Post-visit summary
    "post_visit_pending": "post_visit_summary",
    "post_visit_completed": "post_visit_summary",
    # Walk-in specific (already simplified, but keep for completeness)
    "walk_in_patient": "walk_in_patient",  # Keep as-is
}

# Statuses that don't need mapping (already simplified)
SIMPLIFIED_STATUSES = {
    "patient_registered",
    "intake",
    "pre_visit_summary",
    "vitals",
    "transcription",
    "soap_generation",
    "post_visit_summary",
    "completed",
    "cancelled",
    "walk_in_patient",
}


def map_status(status: str) -> str:
    """Map old detailed status to new simplified status."""
    if status in SIMPLIFIED_STATUSES:
        return status  # Already simplified
    return STATUS_MAPPING.get(status, status)  # Return mapped or original if not found


async def migrate_visits(dry_run: bool = True) -> Dict[str, Any]:
    """
    Migrate visit statuses from detailed codes to simplified flow names.

    Args:
        dry_run: If True, only report what would be changed without making changes

    Returns:
        Dictionary with migration statistics
    """
    settings = get_settings()
    client = AsyncIOMotorClient(settings.database.uri)
    db = client[settings.database.db_name]
    visits = db["visits"]

    # Find all visits that need migration
    query = {
        "$or": [
            {"status": {"$in": list(STATUS_MAPPING.keys())}},
            {"previous_status": {"$in": list(STATUS_MAPPING.keys())}},
            {"next_status": {"$in": list(STATUS_MAPPING.keys())}},
        ]
    }

    total_visits = await visits.count_documents({})
    visits_to_migrate = await visits.count_documents(query)

    print(f"Total visits in database: {total_visits}")
    print(f"Visits that need migration: {visits_to_migrate}")

    if visits_to_migrate == 0:
        print("No visits need migration. All statuses are already simplified.")
        return {
            "total_visits": total_visits,
            "visits_migrated": 0,
            "dry_run": dry_run,
        }

    stats = {
        "status_updates": 0,
        "previous_status_updates": 0,
        "next_status_updates": 0,
    }

    # Process visits in batches
    batch_size = 100
    cursor = visits.find(query)
    batch = []

    async for visit_doc in cursor:
        update_fields = {}
        visit_id = visit_doc.get("visit_id", "unknown")

        # Map status
        old_status = visit_doc.get("status")
        if old_status and old_status in STATUS_MAPPING:
            new_status = map_status(old_status)
            if new_status != old_status:
                update_fields["status"] = new_status
                stats["status_updates"] += 1
                if not dry_run:
                    print(f"  Visit {visit_id}: status '{old_status}' -> '{new_status}'")

        # Map previous_status
        old_previous_status = visit_doc.get("previous_status")
        if old_previous_status and old_previous_status in STATUS_MAPPING:
            new_previous_status = map_status(old_previous_status)
            if new_previous_status != old_previous_status:
                update_fields["previous_status"] = new_previous_status
                stats["previous_status_updates"] += 1
                if not dry_run:
                    print(f"  Visit {visit_id}: previous_status '{old_previous_status}' -> '{new_previous_status}'")

        # Map next_status
        old_next_status = visit_doc.get("next_status")
        if old_next_status and old_next_status in STATUS_MAPPING:
            new_next_status = map_status(old_next_status)
            if new_next_status != old_next_status:
                update_fields["next_status"] = new_next_status
                stats["next_status_updates"] += 1
                if not dry_run:
                    print(f"  Visit {visit_id}: next_status '{old_next_status}' -> '{new_next_status}'")

        if update_fields:
            batch.append((visit_doc["_id"], update_fields))

        # Process batch when it reaches batch_size
        if len(batch) >= batch_size:
            if not dry_run:
                bulk_ops = [
                    visits.update_one({"_id": doc_id}, {"$set": updates})
                    for doc_id, updates in batch
                ]
                # Note: Motor doesn't support bulk_write directly, so we do individual updates
                # In production, you might want to use pymongo's bulk_write for better performance
                for doc_id, updates in batch:
                    await visits.update_one({"_id": doc_id}, {"$set": updates})
            batch = []

    # Process remaining batch
    if batch:
        if not dry_run:
            for doc_id, updates in batch:
                await visits.update_one({"_id": doc_id}, {"$set": updates})

    client.close()

    result = {
        "total_visits": total_visits,
        "visits_to_migrate": visits_to_migrate,
        "status_updates": stats["status_updates"],
        "previous_status_updates": stats["previous_status_updates"],
        "next_status_updates": stats["next_status_updates"],
        "dry_run": dry_run,
    }

    print("\nMigration Summary:")
    print(f"  Total visits: {result['total_visits']}")
    print(f"  Visits to migrate: {result['visits_to_migrate']}")
    print(f"  Status field updates: {result['status_updates']}")
    print(f"  Previous status field updates: {result['previous_status_updates']}")
    print(f"  Next status field updates: {result['next_status_updates']}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")

    return result


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate visit statuses to simplified flow names")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (default: True, set --no-dry-run to actually migrate)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Actually perform the migration (default: dry-run mode)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Visit Status Migration Script")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE MIGRATION'}")
    print()

    try:
        result = await migrate_visits(dry_run=args.dry_run)
        if args.dry_run:
            print("\n" + "=" * 60)
            print("This was a DRY RUN. No changes were made.")
            print("Run with --no-dry-run to actually perform the migration.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Migration completed successfully!")
            print("=" * 60)
    except Exception as e:
        print(f"\nError during migration: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
