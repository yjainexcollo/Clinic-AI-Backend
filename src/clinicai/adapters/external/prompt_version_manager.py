"""
Automatic prompt version detection and management.

This service:
1. Extracts prompt templates from code
2. Calculates hash to detect changes
3. Automatically creates new versions when prompts change
4. Stores versions in MongoDB
5. Updates PROMPT_VERSIONS registry
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

from clinicai.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS, MAJOR_VERSIONS
from clinicai.adapters.external.prompt_extractors import extract_template
from clinicai.adapters.db.mongo.models.prompt_version_m import PromptVersionMongo

logger = logging.getLogger(__name__)


class PromptVersionManager:
    """Manages automatic prompt version detection and storage."""

    def __init__(self):
        self._versions_cache: Optional[Dict[PromptScenario, str]] = None

    async def initialize_versions(self) -> Dict[PromptScenario, str]:
        """
        Initialize prompt versions on startup.
        
        For each scenario:
        1. Extract current template from code
        2. Calculate hash
        3. Check if version exists in DB
        4. If hash changed, create new version
        5. Update PROMPT_VERSIONS dict
        
        Returns:
            Dict mapping scenarios to version strings
        """
        import asyncio
        
        logger.info("🔍 Starting automatic prompt version detection...")
        
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
        
        # Find current version in DB
        current_version_doc = await PromptVersionMongo.find_one(
            PromptVersionMongo.scenario == scenario.value,
            PromptVersionMongo.is_current == True
        )
        
        if current_version_doc:
            current_major = getattr(current_version_doc, 'major_version', 1) # Backward compat default
            current_minor = getattr(current_version_doc, 'minor_version', 0) # Backward compat default
            
            # Case 1: Major version upgrade (Manual trigger)
            if target_major > current_major:
                logger.info(f"{scenario.value}: Major version change detected ({current_major} -> {target_major})")
                return await self._create_new_version(
                    scenario=scenario,
                    template=current_template,
                    template_hash=template_hash,
                    major_version=target_major,
                    minor_version=0
                )
            
            # Case 2: Template changed (Auto minor increment)
            if current_version_doc.template_hash != template_hash:
                logger.info(f"{scenario.value}: Template changed! Auto-incrementing minor version...")
                return await self._create_new_version(
                    scenario=scenario,
                    template=current_template,
                    template_hash=template_hash,
                    major_version=current_major,
                    minor_version=current_minor + 1
                )
                
            # Case 3: No change
            logger.debug(f"{scenario.value}: No change detected (v{current_version_doc.version})")
            return current_version_doc.version
            
        else:
            # No version exists - create first version
            logger.info(f"{scenario.value}: No version found in DB, creating initial version {target_major}.0...")
            return await self._create_new_version(
                scenario=scenario,
                template=current_template,
                template_hash=template_hash,
                major_version=target_major,
                minor_version=0
            )

    async def _create_new_version(
        self,
        scenario: PromptScenario,
        template: str,
        template_hash: str,
        major_version: int,
        minor_version: int
    ) -> str:
        """Create a new prompt version in MongoDB."""
        # Mark old version as not current
        await PromptVersionMongo.find(
            PromptVersionMongo.scenario == scenario.value,
            PromptVersionMongo.is_current == True
        ).update_many({"$set": {"is_current": False}})
        
        # Generate version string: X.Y
        version_string = f"{major_version}.{minor_version}"
        version_number = (major_version * 1000) + minor_version
        
        now = datetime.utcnow()
        
        # Create new version document
        version_doc = PromptVersionMongo(
            scenario=scenario.value,
            version=version_string,
            template_hash=template_hash,
            template_content=template,
            is_current=True,
            created_at=now,
            # Semantic fields
            major_version=major_version,
            minor_version=minor_version,
            version_number=version_number
        )
        
        await version_doc.insert()
        
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

