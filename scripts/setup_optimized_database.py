#!/usr/bin/env python3
"""
Comprehensive database setup script for Clinic-AI.

This script performs:
1. Visit migration from embedded to separate collection
2. Patient index optimization with compound indexes
3. Performance testing and validation
4. Database health checks

Usage:
    python scripts/setup_optimized_database.py --full-setup
    python scripts/setup_optimized_database.py --migrate-only
    python scripts/setup_optimized_database.py --optimize-only
    python scripts/setup_optimized_database.py --health-check
"""

import argparse
import asyncio
import sys
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, 'src')

from motor.motor_asyncio import AsyncIOMotorClient
from clinicai.core.config import get_settings


class DatabaseOptimizer:
    """Comprehensive database optimization and setup."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncIOMotorClient(self.settings.database.uri)
        self.db = self.client[self.settings.database.db_name]

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive database health check."""
        print("🏥 Performing database health check...")
        
        # Check collections
        collections = await self.db.list_collection_names()
        
        # Check patients collection
        patients_count = await self.db.patients.count_documents({})
        patients_with_visits = await self.db.patients.count_documents({
            "visits": {"$exists": True, "$ne": []}
        })
        
        # Check visits collection
        visits_count = await self.db.visits.count_documents({})
        
        # Check indexes
        patients_indexes = await self.db.patients.list_indexes().to_list(None)
        visits_indexes = await self.db.visits.list_indexes().to_list(None)
        
        # Check for compound indexes
        has_mobile_name_index = any(
            idx.get("name") == "mobile_name_compound" 
            for idx in patients_indexes
        )
        
        health_status = {
            "collections": collections,
            "patients": {
                "total": patients_count,
                "with_embedded_visits": patients_with_visits,
                "indexes": len(patients_indexes),
                "has_optimized_indexes": has_mobile_name_index
            },
            "visits": {
                "total": visits_count,
                "indexes": len(visits_indexes)
            },
            "migration_needed": patients_with_visits > 0,
            "optimization_needed": not has_mobile_name_index
        }
        
        print(f"📊 Health Check Results:")
        print(f"   Collections: {collections}")
        print(f"   Patients: {patients_count} total, {patients_with_visits} with embedded visits")
        print(f"   Visits: {visits_count} in separate collection")
        print(f"   Patient indexes: {len(patients_indexes)}")
        print(f"   Visit indexes: {len(visits_indexes)}")
        print(f"   Migration needed: {'Yes' if health_status['migration_needed'] else 'No'}")
        print(f"   Optimization needed: {'Yes' if health_status['optimization_needed'] else 'No'}")
        
        return health_status

    async def run_migration(self) -> bool:
        """Run visit migration from embedded to separate collection."""
        print("🔄 Running visit migration...")
        
        try:
            # Import and run migration
            from migrate_visits_to_separate_collection import VisitMigration
            
            migration = VisitMigration()
            await migration.execute_migration()
            await migration.close()
            
            print("✅ Migration completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False

    async def run_index_optimization(self) -> bool:
        """Run patient index optimization."""
        print("⚡ Running index optimization...")
        
        try:
            # Import and run optimization
            from optimize_patient_indexes import PatientIndexOptimizer
            
            optimizer = PatientIndexOptimizer()
            await optimizer.create_optimized_indexes()
            await optimizer.close()
            
            print("✅ Index optimization completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Index optimization failed: {e}")
            return False

    async def run_performance_test(self) -> bool:
        """Run performance tests."""
        print("🧪 Running performance tests...")
        
        try:
            from optimize_patient_indexes import PatientIndexOptimizer
            
            optimizer = PatientIndexOptimizer()
            await optimizer.performance_test()
            await optimizer.explain_plans()
            await optimizer.close()
            
            print("✅ Performance tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Performance tests failed: {e}")
            return False

    async def full_setup(self) -> bool:
        """Run complete database setup and optimization."""
        print("🚀 Starting full database setup and optimization...")
        
        # Health check
        health = await self.health_check()
        
        success = True
        
        # Run migration if needed
        if health["migration_needed"]:
            print("\n📦 Step 1: Migrating visits to separate collection...")
            if not await self.run_migration():
                success = False
        else:
            print("\n✅ Step 1: No migration needed - visits already separated")
        
        # Run index optimization if needed
        if health["optimization_needed"]:
            print("\n⚡ Step 2: Optimizing patient indexes...")
            if not await self.run_index_optimization():
                success = False
        else:
            print("\n✅ Step 2: No optimization needed - indexes already optimized")
        
        # Run performance tests
        print("\n🧪 Step 3: Running performance tests...")
        if not await self.run_performance_test():
            success = False
        
        # Final health check
        print("\n🏥 Step 4: Final health check...")
        final_health = await self.health_check()
        
        if success:
            print("\n🎉 Full database setup completed successfully!")
            print("   ✅ Visits migrated to separate collection")
            print("   ✅ Patient indexes optimized")
            print("   ✅ Performance tests passed")
        else:
            print("\n⚠️  Database setup completed with some issues.")
            print("   Please review the logs above for details.")
        
        return success

    async def close(self):
        """Close database connections."""
        self.client.close()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive database setup and optimization")
    parser.add_argument("--full-setup", action="store_true", help="Run complete setup")
    parser.add_argument("--migrate-only", action="store_true", help="Run migration only")
    parser.add_argument("--optimize-only", action="store_true", help="Run optimization only")
    parser.add_argument("--health-check", action="store_true", help="Run health check only")
    parser.add_argument("--performance-test", action="store_true", help="Run performance tests only")
    
    args = parser.parse_args()
    
    if not any([args.full_setup, args.migrate_only, args.optimize_only, 
                args.health_check, args.performance_test]):
        parser.print_help()
        return
    
    optimizer = DatabaseOptimizer()
    
    try:
        if args.full_setup:
            success = await optimizer.full_setup()
            sys.exit(0 if success else 1)
        elif args.migrate_only:
            success = await optimizer.run_migration()
            sys.exit(0 if success else 1)
        elif args.optimize_only:
            success = await optimizer.run_index_optimization()
            sys.exit(0 if success else 1)
        elif args.health_check:
            await optimizer.health_check()
        elif args.performance_test:
            success = await optimizer.run_performance_test()
            sys.exit(0 if success else 1)
                
    finally:
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())
