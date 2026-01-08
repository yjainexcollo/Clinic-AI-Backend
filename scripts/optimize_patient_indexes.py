#!/usr/bin/env python3
"""
Index optimization script for patient collection.

This script:
1. Analyzes current index usage and performance
2. Creates optimized compound indexes
3. Removes redundant single-field indexes
4. Provides performance comparison

Usage:
    python scripts/optimize_patient_indexes.py --analyze
    python scripts/optimize_patient_indexes.py --create-indexes
    python scripts/optimize_patient_indexes.py --drop-old-indexes
    python scripts/optimize_patient_indexes.py --performance-test
"""

import argparse
import asyncio
import sys
import time
from typing import Any, Dict, List

# Add the src directory to the Python path
sys.path.insert(0, "src")

from motor.motor_asyncio import AsyncIOMotorClient

from clinicai.core.config import get_settings


class PatientIndexOptimizer:
    """Handles patient collection index optimization."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncIOMotorClient(self.settings.database.uri)
        self.db = self.client[self.settings.database.db_name]
        self.patients_collection = self.db["patients"]

    async def analyze_current_indexes(self) -> Dict[str, Any]:
        """Analyze current indexes on the patients collection."""
        print("🔍 Analyzing current indexes...")

        # Get current indexes
        indexes = await self.patients_collection.list_indexes().to_list(None)

        # Get collection stats
        stats = await self.db.command("collStats", "patients")

        # Get sample queries for analysis
        sample_queries = [
            {"mobile": {"$exists": True}},
            {"name": {"$exists": True}},
            {"mobile": {"$exists": True}, "name": {"$exists": True}},
            {"patient_id": {"$exists": True}},
        ]

        analysis = {
            "current_indexes": indexes,
            "collection_stats": {
                "count": stats.get("count", 0),
                "size": stats.get("size", 0),
                "avgObjSize": stats.get("avgObjSize", 0),
                "storageSize": stats.get("storageSize", 0),
                "indexes": stats.get("nindexes", 0),
                "totalIndexSize": stats.get("totalIndexSize", 0),
            },
            "sample_queries": sample_queries,
        }

        print(f"📊 Current Index Analysis:")
        print(f"   Collection: patients")
        print(f"   Document count: {analysis['collection_stats']['count']:,}")
        print(f"   Total indexes: {analysis['collection_stats']['indexes']}")
        print(
            f"   Total index size: {analysis['collection_stats']['totalIndexSize']:,} bytes"
        )

        print(f"\n📋 Current Indexes:")
        for idx in indexes:
            idx_name = idx.get("name", "unknown")
            idx_key = idx.get("key", {})
            idx_size = idx.get("size", 0)
            print(f"   - {idx_name}: {idx_key} ({idx_size:,} bytes)")

        return analysis

    async def create_optimized_indexes(self) -> None:
        """Create the optimized compound indexes."""
        print("🚀 Creating optimized indexes...")

        # Define optimized indexes
        optimized_indexes = [
            {
                "name": "mobile_name_compound",
                "key": [("mobile", 1), ("name", 1)],
                "background": True,
            },
            {
                "name": "mobile_created_compound",
                "key": [("mobile", 1), ("created_at", -1)],
                "background": True,
            },
        ]

        created_indexes = []
        for idx_spec in optimized_indexes:
            try:
                print(f"   Creating index: {idx_spec['name']} on {idx_spec['key']}")
                result = await self.patients_collection.create_index(
                    idx_spec["key"],
                    name=idx_spec["name"],
                    background=idx_spec["background"],
                )
                created_indexes.append(result)
                print(f"   ✅ Created: {result}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"   ⚠️  Index already exists: {idx_spec['name']}")
                else:
                    print(f"   ❌ Error creating {idx_spec['name']}: {e}")

        print(f"\n✅ Index creation completed!")
        print(f"   Created indexes: {len(created_indexes)}")

    async def drop_redundant_indexes(self) -> None:
        """Drop redundant single-field indexes that are covered by compound indexes."""
        print("🗑️  Dropping redundant indexes...")

        # Indexes to potentially drop (after compound indexes are created)
        redundant_indexes = [
            "name_1",  # Covered by mobile_name_compound
            "mobile_1",  # Covered by mobile_name_compound and mobile_created_compound
        ]

        dropped_indexes = []
        for idx_name in redundant_indexes:
            try:
                print(f"   Dropping index: {idx_name}")
                await self.patients_collection.drop_index(idx_name)
                dropped_indexes.append(idx_name)
                print(f"   ✅ Dropped: {idx_name}")
            except Exception as e:
                if "not found" in str(e).lower():
                    print(f"   ⚠️  Index not found: {idx_name}")
                else:
                    print(f"   ❌ Error dropping {idx_name}: {e}")

        print(f"\n✅ Index cleanup completed!")
        print(f"   Dropped indexes: {len(dropped_indexes)}")

    async def performance_test(self) -> None:
        """Test query performance with different index configurations."""
        print("⚡ Running performance tests...")

        # Test queries
        test_queries = [
            {
                "name": "find_by_mobile",
                "query": {"mobile": "+1234567890"},
                "description": "Find patients by mobile number",
            },
            {
                "name": "find_by_name_and_mobile",
                "query": {"mobile": "+1234567890", "name": "John Doe"},
                "description": "Find patient by mobile and name (most common)",
            },
            {
                "name": "find_by_mobile_and_date",
                "query": {
                    "mobile": "+1234567890",
                    "created_at": {"$gte": "2024-01-01"},
                },
                "description": "Find patients by mobile and date range",
            },
            {
                "name": "find_by_patient_id",
                "query": {"patient_id": "patient_123"},
                "description": "Find patient by ID",
            },
        ]

        # Get some sample data for realistic testing
        sample_patients = await self.patients_collection.find({}).limit(10).to_list(10)

        if not sample_patients:
            print("   ⚠️  No sample data found. Creating test queries with dummy data.")
            # Use dummy data for testing
            test_queries = [
                {
                    "name": "find_by_mobile",
                    "query": {"mobile": {"$regex": "^\\+"}},
                    "description": "Find patients by mobile pattern",
                },
                {
                    "name": "find_by_name_and_mobile",
                    "query": {"mobile": {"$exists": True}, "name": {"$exists": True}},
                    "description": "Find patients with both mobile and name",
                },
            ]
        else:
            # Use real data for testing
            sample_patient = sample_patients[0]
            test_queries[0]["query"] = {
                "mobile": sample_patient.get("mobile", "+1234567890")
            }
            test_queries[1]["query"] = {
                "mobile": sample_patient.get("mobile", "+1234567890"),
                "name": sample_patient.get("name", "John Doe"),
            }

        results = []
        for test in test_queries:
            print(f"\n   🧪 Testing: {test['description']}")

            # Run query multiple times for average
            times = []
            for i in range(5):
                start_time = time.time()
                cursor = self.patients_collection.find(test["query"])
                count = await cursor.count()
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            result = {
                "query_name": test["name"],
                "description": test["description"],
                "avg_time_ms": round(avg_time * 1000, 2),
                "min_time_ms": round(min_time * 1000, 2),
                "max_time_ms": round(max_time * 1000, 2),
                "result_count": count,
            }
            results.append(result)

            print(f"      Results: {count} documents")
            print(f"      Avg time: {result['avg_time_ms']}ms")
            print(f"      Min time: {result['min_time_ms']}ms")
            print(f"      Max time: {result['max_time_ms']}ms")

        print(f"\n📊 Performance Test Summary:")
        for result in results:
            print(
                f"   {result['query_name']}: {result['avg_time_ms']}ms avg ({result['result_count']} results)"
            )

        return results

    async def explain_query_plans(self) -> None:
        """Explain query execution plans to verify index usage."""
        print("🔍 Analyzing query execution plans...")

        # Sample queries to explain
        explain_queries = [
            {"mobile": "+1234567890", "name": "John Doe"},
            {"mobile": "+1234567890"},
            {"patient_id": "patient_123"},
        ]

        for i, query in enumerate(explain_queries):
            print(f"\n   📋 Query {i+1}: {query}")
            try:
                explain_result = await self.patients_collection.find(query).explain()

                # Extract key information
                execution_stats = explain_result.get("executionStats", {})
                winning_plan = explain_result.get("queryPlanner", {}).get(
                    "winningPlan", {}
                )

                print(f"      Index used: {winning_plan.get('indexName', 'COLLSCAN')}")
                print(
                    f"      Execution time: {execution_stats.get('executionTimeMillis', 'N/A')}ms"
                )
                print(
                    f"      Documents examined: {execution_stats.get('totalDocsExamined', 'N/A')}"
                )
                print(
                    f"      Documents returned: {execution_stats.get('totalDocsReturned', 'N/A')}"
                )

                if winning_plan.get("indexName") == "COLLSCAN":
                    print(
                        f"      ⚠️  Warning: Collection scan detected - index may not be optimal"
                    )
                else:
                    print(f"      ✅ Index scan detected - good performance")

            except Exception as e:
                print(f"      ❌ Error explaining query: {e}")

    async def close(self):
        """Close database connections."""
        self.client.close()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimize patient collection indexes")
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze current indexes"
    )
    parser.add_argument(
        "--create-indexes", action="store_true", help="Create optimized indexes"
    )
    parser.add_argument(
        "--drop-old-indexes", action="store_true", help="Drop redundant indexes"
    )
    parser.add_argument(
        "--performance-test", action="store_true", help="Run performance tests"
    )
    parser.add_argument(
        "--explain-plans", action="store_true", help="Explain query execution plans"
    )
    parser.add_argument(
        "--full-optimization", action="store_true", help="Run complete optimization"
    )

    args = parser.parse_args()

    if not any(
        [
            args.analyze,
            args.create_indexes,
            args.drop_old_indexes,
            args.performance_test,
            args.explain_plans,
            args.full_optimization,
        ]
    ):
        parser.print_help()
        return

    optimizer = PatientIndexOptimizer()

    try:
        if args.full_optimization:
            print("🚀 Running full index optimization...")
            await optimizer.analyze_current_indexes()
            await optimizer.create_optimized_indexes()
            await optimizer.performance_test()
            await optimizer.explain_plans()
            print(
                "\n⚠️  Note: Run --drop-old-indexes separately after verifying performance"
            )
        else:
            if args.analyze:
                await optimizer.analyze_current_indexes()
            if args.create_indexes:
                await optimizer.create_optimized_indexes()
            if args.drop_old_indexes:
                await optimizer.drop_redundant_indexes()
            if args.performance_test:
                await optimizer.performance_test()
            if args.explain_plans:
                await optimizer.explain_plans()

    finally:
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())
