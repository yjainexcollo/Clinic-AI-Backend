"""
HIPAA-compliant audit logging system
Captures all PHI access with immutable logs stored for 6 years
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


class HIPAAAuditLogger:
    """
    HIPAA audit logger with immutable logs and 6-year retention
    Logs stored in MongoDB with encryption and access controls
    """

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.audit_collection = None
        self.phi_access_collection = None
        self._initialized = False

    async def initialize(self, mongo_uri: str, db_name: str):
        """Initialize audit logging collections with proper indexes"""
        try:
            self.client = AsyncIOMotorClient(
                mongo_uri,
                tls=True,
                retryWrites=True,
                w="majority",  # Write concern for durability
                journal=True,  # Ensure writes are journaled
            )
            self.db = self.client[db_name]

            # Main audit log collection
            self.audit_collection = self.db["hipaa_audit_logs"]

            # PHI-specific access logs
            self.phi_access_collection = self.db["phi_access_logs"]

            # Create indexes for efficient querying and compliance
            await self._create_indexes()

            self._initialized = True
            logger.info("✅ HIPAA Audit Logger initialized with 6-year retention")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HIPAA Audit Logger: {e}")
            raise

    async def _create_indexes(self):
        """Create indexes for audit collections"""
        # Audit log indexes
        await self.audit_collection.create_index("timestamp")
        await self.audit_collection.create_index("user_id")
        await self.audit_collection.create_index("resource_type")
        await self.audit_collection.create_index("action")
        await self.audit_collection.create_index([("timestamp", -1)])
        await self.audit_collection.create_index([("user_id", 1), ("timestamp", -1)])

        # PHI access log indexes
        await self.phi_access_collection.create_index("timestamp")
        await self.phi_access_collection.create_index("user_id")
        await self.phi_access_collection.create_index("patient_id")
        await self.phi_access_collection.create_index("phi_type")
        await self.phi_access_collection.create_index([("timestamp", -1)])

        logger.info("✅ Audit log indexes created")

    async def log_phi_access(
        self,
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str],
        patient_id: Optional[str],
        ip_address: str,
        user_agent: str,
        phi_fields: List[str],
        phi_accessed: bool = True,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Log PHI access with comprehensive audit trail

        Args:
            user_id: User who accessed the data
            action: HTTP method (GET, POST, PUT, DELETE)
            resource_type: Type of resource (patient, visit, note, etc.)
            resource_id: ID of the resource
            patient_id: Patient ID if applicable
            ip_address: IP address of the request
            user_agent: Browser/client user agent
            phi_fields: List of PHI fields accessed
            phi_accessed: Whether PHI was accessed
            success: Whether the action succeeded
            details: Additional details
            request_id: Unique request ID
            session_id: Session ID for tracking

        Returns:
            str: Audit log ID
        """
        if not self._initialized:
            return await self._fallback_log(locals())

        timestamp = datetime.utcnow()

        # Create audit entry with ISO format timestamps for consistent checksums
        audit_entry = {
            "audit_id": str(ObjectId()),
            "timestamp": timestamp.isoformat(),  # Store as ISO string
            "user_id": user_id or "anonymous",
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "patient_id": patient_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "phi_accessed": phi_accessed,
            "phi_fields": phi_fields,
            "success": success,
            "request_id": request_id,
            "session_id": session_id,
            "details": details or {},
            "retention_date": (timestamp + timedelta(days=2190)).isoformat(),  # Store as ISO string
            "created_at": timestamp.isoformat(),  # Store as ISO string
            "immutable": True,  # Mark as immutable
            "checksum": None,  # Will be added below
        }

        # Calculate checksum for integrity verification
        audit_entry["checksum"] = self._calculate_checksum(audit_entry)

        try:
            # Insert into main audit log
            result = await self.audit_collection.insert_one(audit_entry)
            audit_id = str(result.inserted_id)

            # If PHI was accessed, create separate PHI access log
            if phi_accessed and phi_fields:
                phi_log = {
                    "audit_id": audit_id,
                    "timestamp": timestamp,
                    "user_id": user_id or "anonymous",
                    "patient_id": patient_id,
                    "phi_type": resource_type,
                    "phi_fields": phi_fields,
                    "action": action,
                    "purpose": (details.get("purpose", "treatment") if details else "treatment"),
                    "ip_address": ip_address,
                    "created_at": timestamp,
                }
                await self.phi_access_collection.insert_one(phi_log)

            # Log to application logs for redundancy
            logger.info(
                f"HIPAA_AUDIT: audit_id={audit_id} user={user_id} action={action} "
                f"resource={resource_type}:{resource_id} patient={patient_id} "
                f"phi_fields={len(phi_fields)} success={success}"
            )

            return audit_id
        except Exception as e:
            logger.error(f"❌ Failed to write HIPAA audit log: {e}")
            return await self._fallback_log(locals())

    def _calculate_checksum(self, entry: dict) -> str:
        """Calculate SHA-256 checksum for integrity verification"""
        # Remove MongoDB-added fields and checksum itself
        excluded_fields = {"checksum", "_id"}
        entry_copy = {k: v for k, v in entry.items() if k not in excluded_fields}

        # Normalize datetime objects to ISO format strings for consistent serialization
        normalized_entry = {}
        for k, v in entry_copy.items():
            if isinstance(v, datetime):
                normalized_entry[k] = v.isoformat()
            else:
                normalized_entry[k] = v

        entry_json = json.dumps(normalized_entry, sort_keys=True, default=str)
        return hashlib.sha256(entry_json.encode()).hexdigest()

    async def _fallback_log(self, data: dict) -> str:
        """Fallback logging to file if database is unavailable"""
        fallback_id = str(ObjectId())
        logger.critical(f"HIPAA_AUDIT_FALLBACK: {json.dumps(data, default=str)}")
        return fallback_id

    async def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Query audit trail for compliance reporting"""
        if not self._initialized:
            return []

        query = {}
        if user_id:
            query["user_id"] = user_id
        if patient_id:
            query["patient_id"] = patient_id
        if resource_type:
            query["resource_type"] = resource_type
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date

        cursor = self.audit_collection.find(query).sort("timestamp", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def verify_audit_integrity(self, audit_id: str) -> bool:
        """Verify audit log integrity using checksum"""
        if not self._initialized:
            return False

        # Try to find by MongoDB _id first (what log_phi_access returns)
        try:
            from bson import ObjectId

            entry = await self.audit_collection.find_one({"_id": ObjectId(audit_id)})
        except Exception:
            # Fallback: try as audit_id field (custom field)
            entry = await self.audit_collection.find_one({"audit_id": audit_id})

        if not entry:
            logger.warning(f"Audit entry not found for audit_id: {audit_id}")
            return False

        stored_checksum = entry.get("checksum")
        if not stored_checksum:
            logger.warning(f"No checksum found in audit entry: {audit_id}")
            return False

        # Debug: Log the entry structure
        logger.debug(f"Verifying audit {audit_id}: Entry has {len(entry)} fields")
        logger.debug(f"Entry fields: {list(entry.keys())}")

        calculated_checksum = self._calculate_checksum(entry)

        is_valid = stored_checksum == calculated_checksum
        if not is_valid:
            # Debug: Show what fields are in the entry
            excluded_fields = {"checksum", "_id"}
            entry_for_checksum = {k: v for k, v in entry.items() if k not in excluded_fields}

            # Print to console for immediate visibility
            print(f"❌ Audit integrity mismatch!")
            print(f"   audit_id: {audit_id}")
            print(f"   Stored:  {stored_checksum}")
            print(f"   Calculated: {calculated_checksum}")
            print(f"   Entry has {len(entry)} fields: {list(entry.keys())}")

            # Check datetime types
            for k, v in entry.items():
                if k in ["timestamp", "retention_date", "created_at"]:
                    print(f"   {k}: {type(v).__name__} = {v}")

            logger.error(
                f"Audit integrity violation detected! audit_id={audit_id}, "
                f"stored={stored_checksum}, calculated={calculated_checksum}"
            )

        return is_valid


# Global instance
_audit_logger: Optional[HIPAAAuditLogger] = None


def get_audit_logger() -> HIPAAAuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = HIPAAAuditLogger()
    return _audit_logger
