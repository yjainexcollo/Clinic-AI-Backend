"""
Automatic prompt version detection and management.

This service:
1. Extracts prompt templates from code
2. Calculates hash to detect changes
3. Automatically creates new versions when prompts change
4. Stores versions in MongoDB (4 documents, one per scenario, each with versions array)
5. Updates PROMPT_VERSIONS registry
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from clinicai.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS, MAJOR_VERSIONS
from clinicai.adapters.external.prompt_extractors import extract_template
from clinicai.adapters.db.mongo.models.prompt_version_m import PromptVersionMongo, PromptVersionEntry

logger = logging.getLogger(__name__)


def _scenario_to_field_value(scenario: PromptScenario) -> str:
    """Convert PromptScenario enum to database field value."""
    mapping = {
        PromptScenario.INTAKE: "intake",
        PromptScenario.PREVISIT_SUMMARY: "previsit",
        PromptScenario.SOAP: "soap",
        PromptScenario.POSTVISIT_SUMMARY: "postvisit",
    }
    field_value = mapping.get(scenario)
    if not field_value:
        raise ValueError(f"No database mapping for scenario: {scenario.value}")
    return field_value


def _scenario_to_version_prefix(scenario: PromptScenario) -> str:
    """Convert PromptScenario enum to version string prefix (e.g., 'INTAKE', 'PREVISIT')."""
    mapping = {
        PromptScenario.INTAKE: "INTAKE",
        PromptScenario.PREVISIT_SUMMARY: "PREVISIT",
        PromptScenario.SOAP: "SOAP",
        PromptScenario.POSTVISIT_SUMMARY: "POSTVISIT",
    }
    prefix = mapping.get(scenario)
    if not prefix:
        raise ValueError(f"No version prefix mapping for scenario: {scenario.value}")
    return prefix


class PromptVersionManager:
    """Manages automatic prompt version detection and storage."""

    def __init__(self):
        self._versions_cache: Optional[Dict[PromptScenario, str]] = None

    async def _ensure_unique_index(self):
        """Ensure unique index exists on scenario field to prevent duplicates."""
        try:
            from clinicai.core.config import get_settings
            from motor.motor_asyncio import AsyncIOMotorClient
            
            settings = get_settings()
            client = AsyncIOMotorClient(settings.database.uri)
            db = client[settings.database.db_name]
            collection = db["prompt_versions"]
            
            # Check if unique index exists
            indexes = await collection.list_indexes().to_list(length=None)
            has_unique_index = any(
                idx.get("name") == "scenario_1" and idx.get("unique") is True
                for idx in indexes
            )
            
            if not has_unique_index:
                try:
                    # Create unique index on scenario field
                    await collection.create_index("scenario", unique=True, name="scenario_1")
                    logger.info("✅ Created unique index on 'scenario' field")
                except Exception as index_error:
                    # If index creation fails due to duplicates, log warning but continue
                    if "duplicate key" in str(index_error).lower() or "E11000" in str(index_error):
                        logger.warning(
                            "⚠️  Cannot create unique index due to duplicate scenarios. "
                            "Run cleanup manually or delete duplicates first."
                        )
                    else:
                        raise
            else:
                logger.debug("✅ Unique index on 'scenario' field already exists")
        except Exception as e:
            logger.warning(f"⚠️  Failed to ensure unique index (non-critical): {e}")
            # Don't fail initialization if index creation fails

    async def _cleanup_invalid_documents(self):
        """
        Clean up invalid documents to ensure only 4 documents exist (one per scenario).
        
        Removes:
        - Documents with invalid scenario values
        - Duplicate documents (keeps the most recent one per scenario)
        - Old documents from previous structure (documents without 'versions' array)
        """
        try:
            # Expected scenario values
            expected_scenarios = {"intake", "previsit", "soap", "postvisit"}
            
            # Find all documents
            all_docs = await PromptVersionMongo.find({}).to_list()
            
            # Track documents by scenario
            docs_by_scenario: Dict[str, List[PromptVersionMongo]] = {}
            invalid_docs = []
            
            for doc in all_docs:
                scenario = getattr(doc, "scenario", None)
                
                # Check if document has 'versions' field (new structure)
                if not hasattr(doc, "versions") or not isinstance(getattr(doc, "versions", None), list):
                    invalid_docs.append(doc)
                    logger.info(f"🗑️  Marking document for deletion: old structure (no versions array), ID: {doc.id}")
                    continue
                
                # Check if scenario is valid
                if scenario not in expected_scenarios:
                    invalid_docs.append(doc)
                    logger.info(f"🗑️  Marking document for deletion: invalid scenario '{scenario}', ID: {doc.id}")
                    continue
                
                # Group by scenario
                if scenario not in docs_by_scenario:
                    docs_by_scenario[scenario] = []
                docs_by_scenario[scenario].append(doc)
            
            # Handle duplicates: keep the one with most versions (or most recent if equal)
            for scenario, docs in docs_by_scenario.items():
                if len(docs) > 1:
                    # Sort by number of versions (descending), then by _id (most recent)
                    docs.sort(key=lambda d: (len(getattr(d, "versions", [])), str(d.id)), reverse=True)
                    # Keep the first one, mark others for deletion
                    for duplicate_doc in docs[1:]:
                        invalid_docs.append(duplicate_doc)
                        logger.info(f"🗑️  Marking duplicate document for deletion: scenario '{scenario}', ID: {duplicate_doc.id}")
            
            # Delete invalid/duplicate documents
            for doc in invalid_docs:
                try:
                    await doc.delete()
                    logger.info(f"✅ Deleted invalid/duplicate document ID: {doc.id}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete document {doc.id}: {e}")
            
            if invalid_docs:
                logger.info(f"🧹 Cleanup completed: removed {len(invalid_docs)} invalid/duplicate documents")
            else:
                logger.debug("✅ No invalid documents found, collection is clean")
                
        except Exception as e:
            logger.warning(f"⚠️  Cleanup failed (non-critical): {e}", exc_info=True)
            # Don't fail initialization if cleanup fails

    async def initialize_versions(self) -> Dict[PromptScenario, str]:
        """
        Initialize prompt versions on startup.
        
        For each scenario:
        1. Clean up invalid/duplicate documents
        2. Extract current template from code
        3. Calculate hash
        4. Check if version exists in DB
        5. If hash changed, create new version
        6. Update PROMPT_VERSIONS dict
        
        Returns:
            Dict mapping scenarios to version strings
        """
        import asyncio
        
        logger.info("🔍 Starting automatic prompt version detection...")
        
        # Clean up invalid/duplicate documents first
        await self._cleanup_invalid_documents()
        
        # Then ensure unique index exists to prevent future duplicates
        await self._ensure_unique_index()
        
        versions: Dict[PromptScenario, str] = {}
        
        # Process scenarios in parallel for better performance
        async def process_scenario(scenario: PromptScenario) -> Tuple[PromptScenario, str]:
            try:
                version = await self._check_and_update_scenario(scenario)
                logger.info(f"✅ {scenario.value}: {version}")
                return (scenario, version)
            except Exception as e:
                logger.error(f"❌ Failed to process {scenario.value}: {e}", exc_info=True)
                # Fallback to hardcoded version if available
                fallback_version = PROMPT_VERSIONS.get(scenario)
                if fallback_version:
                    logger.warning(f"⚠️  Using fallback version for {scenario.value}: {fallback_version}")
                    return (scenario, fallback_version)
                else:
                    raise
        
        # Process all tracked scenarios concurrently (defined in MAJOR_VERSIONS)
        # This automatically excludes scenarios not in MAJOR_VERSIONS (like RED_FLAG)
        tasks = [process_scenario(scenario) for scenario in MAJOR_VERSIONS.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results and handle exceptions
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to process scenario: {result}", exc_info=True)
                continue
            scenario, version = result
            versions[scenario] = version
        
        # Update the global PROMPT_VERSIONS dict
        PROMPT_VERSIONS.update(versions)
        self._versions_cache = versions
        
        logger.info(f"✅ Prompt version detection completed: {len(versions)} scenarios")
        return versions

    async def _get_or_create_scenario_document(self, scenario: PromptScenario) -> PromptVersionMongo:
        """
        Get or create the document for a specific scenario.
        
        Each scenario has its own document with a versions array.
        Unique constraint on scenario field ensures only one document per scenario.
        """
        scenario_value = _scenario_to_field_value(scenario)
        
        # Find document for this scenario using dictionary-style query (more reliable)
        doc = await PromptVersionMongo.find_one({"scenario": scenario_value})
        if not doc:
            # Create new document for this scenario
            doc = PromptVersionMongo(scenario=scenario_value, versions=[])
            try:
                await doc.insert()
                logger.info(f"📝 Created new prompt versions document for scenario: {scenario_value}")
            except Exception as e:
                # If insert fails (e.g., duplicate key), try to find again (race condition)
                logger.warning(f"Insert failed for {scenario_value}, trying to find existing document: {e}")
                doc = await PromptVersionMongo.find_one({"scenario": scenario_value})
                if not doc:
                    raise
        return doc

    async def _check_and_update_scenario(self, scenario: PromptScenario) -> str:
        """
        Check if prompt changed and create new version if needed.
        
        Handles:
        1. Major version changes (manual via MAJOR_VERSIONS)
        2. Minor version changes (auto-detected via hash)
        """
        # Extract current template from code
        try:
            current_template = extract_template(scenario)
        except Exception as e:
            logger.error(f"Failed to extract template for {scenario.value}: {e}")
            raise
        
        # Calculate hash
        template_hash = self._calculate_hash(current_template)
        
        # Get desired major version from code configuration
        target_major = MAJOR_VERSIONS.get(scenario, 1)
        
        # Get the document for this scenario
        version_doc = await self._get_or_create_scenario_document(scenario)
        
        # Find current version in versions array
        current_version_entry = next((v for v in version_doc.versions if v.is_current), None)
        
        if current_version_entry:
            current_major = current_version_entry.major_version
            current_minor = current_version_entry.minor_version
            
            # Case 1: Major version upgrade (Manual trigger)
            if target_major > current_major:
                logger.info(f"{scenario.value}: Major version change detected ({current_major} -> {target_major})")
                return await self._create_new_version(
                    version_doc=version_doc,
                    scenario=scenario,
                    template=current_template,
                    template_hash=template_hash,
                    major_version=target_major,
                    minor_version=0
                )
            
            # Case 2: Template changed (Auto minor increment)
            if current_version_entry.template_hash != template_hash:
                logger.info(f"{scenario.value}: Template changed! Auto-incrementing minor version...")
                return await self._create_new_version(
                    version_doc=version_doc,
                    scenario=scenario,
                    template=current_template,
                    template_hash=template_hash,
                    major_version=current_major,
                    minor_version=current_minor + 1
                )
                
            # Case 3: No change
            logger.debug(f"{scenario.value}: No change detected (v{current_version_entry.version})")
            return current_version_entry.version
            
        else:
            # No version exists - create first version
            logger.info(f"{scenario.value}: No version found in DB, creating initial version {target_major}.0...")
            return await self._create_new_version(
                version_doc=version_doc,
                scenario=scenario,
                template=current_template,
                template_hash=template_hash,
                major_version=target_major,
                minor_version=0
            )

    async def _create_new_version(
        self,
        version_doc: PromptVersionMongo,
        scenario: PromptScenario,
        template: str,
        template_hash: str,
        major_version: int,
        minor_version: int
    ) -> str:
        """Create a new prompt version and add to the versions array."""
        # Generate version string: SCENARIO_V_X.Y (e.g., INTAKE_V_1.3)
        scenario_prefix = _scenario_to_version_prefix(scenario)
        version_string = f"{scenario_prefix}_V_{major_version}.{minor_version}"
        version_number = (major_version * 1000) + minor_version
        
        now = datetime.utcnow()
        
        # Mark all existing versions in the array as not current
        for version_entry in version_doc.versions:
            version_entry.is_current = False
        
        # Create new version entry
        new_version_entry = PromptVersionEntry(
            version=version_string,
            template_hash=template_hash,
            template_content=template,
            is_current=True,
            created_at=now,
            major_version=major_version,
            minor_version=minor_version,
            version_number=version_number
        )
        
        # Add new version to array
        version_doc.versions.append(new_version_entry)
        
        # Save the document
        await version_doc.save()
        
        logger.info(
            f"📝 Created new version: {scenario.value} → {version_string} "
            f"(hash: {template_hash[:8]}...)"
        )
        
        return version_string

    @staticmethod
    def _calculate_hash(template: str) -> str:
        """Calculate SHA256 hash of normalized template."""
        return hashlib.sha256(template.encode('utf-8')).hexdigest()

    def get_current_version(self, scenario: PromptScenario) -> Optional[str]:
        """Get current version from cache (for runtime use)."""
        if self._versions_cache:
            return self._versions_cache.get(scenario)
        return PROMPT_VERSIONS.get(scenario)


# Global instance
_version_manager: Optional[PromptVersionManager] = None


def get_prompt_version_manager() -> PromptVersionManager:
    """Get or create the global prompt version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = PromptVersionManager()
    return _version_manager
