#!/usr/bin/env python3
"""
Migration script to add workflow_type field to existing visits.

This script:
1. Updates all existing visits to have workflow_type = "scheduled"
2. Ensures backward compatibility
3. Creates necessary indexes

Usage:
    python scripts/migrate_workflow_types.py --dry-run
    python scripts/migrate_workflow_types.py --execute
"""

import argparse
import asyncio
import sys
from typing import Any, Dict

# Add the src directory to the Python path
sys.path.insert(0, "src")

from motor.motor_asyncio import AsyncIOMotorClient

from clinicai.core.config import get_settings


class WorkflowTypeMigration:
    """Handles migration of workflow types for existing visits."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncIOMotorClient(self.settings.database.uri)
        self.db = self.client[self.settings.database.db_name]
        self.visits_collection = self.db["visits"]

    async def analyze_current_state(self) -> Dict[str, Any]:
        """Analyze the current state of visits collection."""
        print("🔍 Analyzing current visits collection...")

        # Count total visits
        total_visits = await self.visits_collection.count_documents({})

        # Count visits with workflow_type field
        visits_with_workflow_type = await self.visits_collection.count_documents(
            {"workflow_type": {"$exists": True}}
        )

        # Count visits without workflow_type field
        visits_without_workflow_type = total_visits - visits_with_workflow_type

        # Get sample visits
        sample_visits = await self.visits_collection.find({}).limit(3).to_list(3)

        analysis = {
            "total_visits": total_visits,
            "visits_with_workflow_type": visits_with_workflow_type,
            "visits_without_workflow_type": visits_without_workflow_type,
            "sample_visits": sample_visits,
        }

        print(f"📊 Analysis Results:")
        print(f"   Total visits: {total_visits}")
        print(f"   Visits with workflow_type: {visits_with_workflow_type}")
        print(f"   Visits without workflow_type: {visits_without_workflow_type}")

        return analysis

    async def dry_run(self) -> None:
        """Perform a dry run of the migration."""
        print("🧪 Performing dry run...")

        analysis = await self.analyze_current_state()

        if analysis["visits_without_workflow_type"] == 0:
            print(
                "✅ No visits need migration. All visits already have workflow_type field."
            )
        else:
            print(f"📋 Dry Run Summary:")
            print(f"   Visits to update: {analysis['visits_without_workflow_type']}")
            print(f"   Will set workflow_type = 'scheduled' for all existing visits")
            print(f"   Will create workflow_type index")
            print("🚀 Ready to migrate. Use --execute to perform the actual migration.")

    async def execute_migration(self) -> None:
        """Execute the migration."""
        print("🚀 Starting workflow type migration...")

        # Update all visits without workflow_type to have workflow_type = "scheduled"
        result = await self.visits_collection.update_many(
            {"workflow_type": {"$exists": False}},
            {"$set": {"workflow_type": "scheduled"}},
        )

        print(f"✅ Migration completed!")
        print(f"   Updated visits: {result.modified_count}")

        # Create workflow_type index
        try:
            await self.visits_collection.create_index("workflow_type")
            print("   ✅ Created workflow_type index")
        except Exception as e:
            print(f"   ⚠️  Index creation failed (may already exist): {e}")

        # Create compound index for workflow_type and status
        try:
            await self.visits_collection.create_index(
                [("workflow_type", 1), ("status", 1)]
            )
            print("   ✅ Created compound index (workflow_type, status)")
        except Exception as e:
            print(f"   ⚠️  Compound index creation failed (may already exist): {e}")

    async def verify_migration(self) -> None:
        """Verify the migration was successful."""
        print("🔍 Verifying migration...")

        # Check that all visits now have workflow_type
        total_visits = await self.visits_collection.count_documents({})
        visits_with_workflow_type = await self.visits_collection.count_documents(
            {"workflow_type": {"$exists": True}}
        )

        if total_visits == visits_with_workflow_type:
            print("✅ Migration verification successful!")
            print(f"   All {total_visits} visits have workflow_type field")
        else:
            print("❌ Migration verification failed!")
            print(f"   Total visits: {total_visits}")
            print(f"   Visits with workflow_type: {visits_with_workflow_type}")

        # Check workflow_type distribution
        scheduled_count = await self.visits_collection.count_documents(
            {"workflow_type": "scheduled"}
        )
        walk_in_count = await self.visits_collection.count_documents(
            {"workflow_type": "walk_in"}
        )

        print(f"📊 Workflow Type Distribution:")
        print(f"   Scheduled visits: {scheduled_count}")
        print(f"   Walk-in visits: {walk_in_count}")

    async def close(self):
        """Close database connections."""
        self.client.close()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Migrate workflow types for existing visits"
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run")
    parser.add_argument("--execute", action="store_true", help="Execute the migration")
    parser.add_argument("--verify", action="store_true", help="Verify the migration")

    args = parser.parse_args()

    if not any([args.dry_run, args.execute, args.verify]):
        parser.print_help()
        return

    migration = WorkflowTypeMigration()

    try:
        if args.dry_run:
            await migration.dry_run()
        elif args.execute:
            await migration.execute_migration()
            await migration.verify_migration()
        elif args.verify:
            await migration.verify_migration()
    finally:
        await migration.close()


if __name__ == "__main__":
    asyncio.run(main())
