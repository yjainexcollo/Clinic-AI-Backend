#!/usr/bin/env python3
"""
Migration script to move visits from embedded documents in patients collection
to a separate visits collection.

This script:
1. Reads all patients with embedded visits
2. Extracts visits and saves them to the visits collection
3. Removes the visits field from patients collection
4. Provides rollback functionality

Usage:
    python scripts/migrate_visits_to_separate_collection.py --dry-run
    python scripts/migrate_visits_to_separate_collection.py --execute
    python scripts/migrate_visits_to_separate_collection.py --rollback
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Dict, List

# Add the src directory to the Python path
sys.path.insert(0, "src")

from motor.motor_asyncio import AsyncIOMotorClient

from clinicai.adapters.db.mongo.models.patient_m import PatientMongo, VisitMongo
from clinicai.adapters.db.mongo.repositories.visit_repository import (
    MongoVisitRepository,
)
from clinicai.core.config import get_settings
from clinicai.domain.value_objects.visit_id import VisitId


class VisitMigration:
    """Handles migration of visits from embedded to separate collection."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncIOMotorClient(self.settings.database.uri)
        self.db = self.client[self.settings.database.db_name]
        self.patients_collection = self.db["patients"]
        self.visits_collection = self.db["visits"]
        self.visit_repository = MongoVisitRepository()
        self.migrated_count = 0
        self.error_count = 0
        self.errors = []

    async def analyze_current_state(self) -> Dict[str, Any]:
        """Analyze the current state of the database."""
        print("🔍 Analyzing current database state...")

        # Count patients with embedded visits
        patients_with_visits = await self.patients_collection.count_documents(
            {"visits": {"$exists": True, "$ne": []}}
        )

        # Count total patients
        total_patients = await self.patients_collection.count_documents({})

        # Count existing visits in separate collection
        existing_visits = await self.visits_collection.count_documents({})

        # Get sample of patients with visits
        sample_patients = (
            await self.patients_collection.find(
                {"visits": {"$exists": True, "$ne": []}}
            )
            .limit(3)
            .to_list(3)
        )

        # Check current indexes
        current_indexes = await self.patients_collection.list_indexes().to_list(None)

        analysis = {
            "total_patients": total_patients,
            "patients_with_visits": patients_with_visits,
            "existing_visits_in_separate_collection": existing_visits,
            "sample_patients": sample_patients,
            "current_indexes": current_indexes,
        }

        print(f"📊 Analysis Results:")
        print(f"   Total patients: {total_patients}")
        print(f"   Patients with embedded visits: {patients_with_visits}")
        print(f"   Existing visits in separate collection: {existing_visits}")
        print(f"   Current indexes: {len(current_indexes)}")

        return analysis

    async def dry_run(self) -> None:
        """Perform a dry run of the migration."""
        print("🧪 Performing dry run...")

        analysis = await self.analyze_current_state()

        if analysis["existing_visits_in_separate_collection"] > 0:
            print("⚠️  WARNING: Visits already exist in separate collection!")
            print("   This might indicate a previous migration or data inconsistency.")

        # Get all patients with visits
        patients_cursor = self.patients_collection.find(
            {"visits": {"$exists": True, "$ne": []}}
        )

        total_visits_to_migrate = 0
        async for patient in patients_cursor:
            visits = patient.get("visits", [])
            total_visits_to_migrate += len(visits)

            print(
                f"   Patient {patient.get('patient_id', 'unknown')}: {len(visits)} visits"
            )
            for visit in visits:
                print(
                    f"     - Visit {visit.get('visit_id', 'unknown')}: {visit.get('status', 'unknown')}"
                )

        print(f"\n📋 Dry Run Summary:")
        print(f"   Total visits to migrate: {total_visits_to_migrate}")
        print(f"   Patients to update: {analysis['patients_with_visits']}")

        if total_visits_to_migrate == 0:
            print("✅ No visits to migrate. Database is already in the correct state.")
        else:
            print("🚀 Ready to migrate. Use --execute to perform the actual migration.")

    async def execute_migration(self) -> None:
        """Execute the migration."""
        print("🚀 Starting migration...")

        # Check if visits already exist in separate collection
        existing_visits = await self.visits_collection.count_documents({})
        if existing_visits > 0:
            print(
                f"⚠️  WARNING: {existing_visits} visits already exist in separate collection!"
            )
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                print("❌ Migration cancelled.")
                return

        # Get all patients with visits
        patients_cursor = self.patients_collection.find(
            {"visits": {"$exists": True, "$ne": []}}
        )

        async for patient in patients_cursor:
            try:
                await self._migrate_patient_visits(patient)
            except Exception as e:
                self.error_count += 1
                error_msg = f"Error migrating patient {patient.get('patient_id', 'unknown')}: {str(e)}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")

        print(f"\n✅ Migration completed!")
        print(f"   Migrated visits: {self.migrated_count}")
        print(f"   Errors: {self.error_count}")

        if self.errors:
            print("\n❌ Errors encountered:")
            for error in self.errors:
                print(f"   - {error}")

        # Suggest index optimization
        print(f"\n💡 Next Steps:")
        print(
            f"   1. Run index optimization: python scripts/optimize_patient_indexes.py --analyze"
        )
        print(
            f"   2. Create optimized indexes: python scripts/optimize_patient_indexes.py --create-indexes"
        )
        print(
            f"   3. Test performance: python scripts/optimize_patient_indexes.py --performance-test"
        )

    async def _migrate_patient_visits(self, patient: Dict[str, Any]) -> None:
        """Migrate visits for a single patient."""
        patient_id = patient.get("patient_id")
        visits = patient.get("visits", [])

        print(f"🔄 Migrating {len(visits)} visits for patient {patient_id}...")

        for visit_data in visits:
            try:
                # Create VisitMongo document
                visit_doc = VisitMongo(
                    visit_id=visit_data.get("visit_id"),
                    patient_id=patient_id,
                    status=visit_data.get("status", "intake"),
                    created_at=visit_data.get("created_at", datetime.utcnow()),
                    updated_at=visit_data.get("updated_at", datetime.utcnow()),
                    intake_session=visit_data.get("intake_session"),
                    pre_visit_summary=visit_data.get("pre_visit_summary"),
                    transcription_session=visit_data.get("transcription_session"),
                    soap_note=visit_data.get("soap_note"),
                    vitals=visit_data.get("vitals"),
                    post_visit_summary=visit_data.get("post_visit_summary"),
                )

                # Save to visits collection
                await visit_doc.save()
                self.migrated_count += 1

                print(f"   ✅ Migrated visit {visit_data.get('visit_id')}")

            except Exception as e:
                raise Exception(
                    f"Failed to migrate visit {visit_data.get('visit_id')}: {str(e)}"
                )

        # Remove visits field from patient document
        await self.patients_collection.update_one(
            {"_id": patient["_id"]}, {"$unset": {"visits": ""}}
        )

        print(f"   🗑️  Removed visits field from patient {patient_id}")

    async def rollback(self) -> None:
        """Rollback the migration by moving visits back to embedded structure."""
        print("🔄 Starting rollback...")

        # Get all visits
        visits_cursor = self.visits_collection.find({})

        # Group visits by patient_id
        visits_by_patient = {}
        async for visit in visits_cursor:
            patient_id = visit.get("patient_id")
            if patient_id not in visits_by_patient:
                visits_by_patient[patient_id] = []

            # Convert back to embedded format
            visit_data = {
                "visit_id": visit.get("visit_id"),
                "status": visit.get("status"),
                "created_at": visit.get("created_at"),
                "updated_at": visit.get("updated_at"),
                "intake_session": visit.get("intake_session"),
                "pre_visit_summary": visit.get("pre_visit_summary"),
                "transcription_session": visit.get("transcription_session"),
                "soap_note": visit.get("soap_note"),
                "vitals": visit.get("vitals"),
                "post_visit_summary": visit.get("post_visit_summary"),
            }
            visits_by_patient[patient_id].append(visit_data)

        # Update patients with embedded visits
        for patient_id, visits in visits_by_patient.items():
            await self.patients_collection.update_one(
                {"patient_id": patient_id}, {"$set": {"visits": visits}}
            )
            print(f"   ✅ Restored {len(visits)} visits to patient {patient_id}")

        # Drop the visits collection
        await self.visits_collection.drop()
        print("   🗑️  Dropped visits collection")

        print(
            f"✅ Rollback completed! Restored {len(visits_by_patient)} patients with embedded visits."
        )

    async def close(self):
        """Close database connections."""
        self.client.close()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Migrate visits to separate collection"
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run")
    parser.add_argument("--execute", action="store_true", help="Execute the migration")
    parser.add_argument(
        "--rollback", action="store_true", help="Rollback the migration"
    )

    args = parser.parse_args()

    if not any([args.dry_run, args.execute, args.rollback]):
        parser.print_help()
        return

    migration = VisitMigration()

    try:
        if args.dry_run:
            await migration.dry_run()
        elif args.execute:
            await migration.execute_migration()
        elif args.rollback:
            await migration.rollback()
    finally:
        await migration.close()


if __name__ == "__main__":
    asyncio.run(main())
