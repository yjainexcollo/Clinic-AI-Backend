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

from clinicai.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS
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
        
        # Process all scenarios concurrently
        tasks = [process_scenario(scenario) for scenario in PromptScenario]
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
        
        Returns:
            Current version string (existing or newly created)
        """
        # Extract current template from code
        try:
            current_template = extract_template(scenario)
        except Exception as e:
            logger.error(f"Failed to extract template for {scenario.value}: {e}")
            raise
        
        # Calculate hash
        template_hash = self._calculate_hash(current_template)
        
        # Find current version in DB
        current_version_doc = await PromptVersionMongo.find_one(
            PromptVersionMongo.scenario == scenario.value,
            PromptVersionMongo.is_current == True
        )
        
        if current_version_doc:
            # Check if hash changed
            if current_version_doc.template_hash == template_hash:
                # No change - return existing version
                logger.debug(f"{scenario.value}: No change detected (hash: {template_hash[:8]}...)")
                return current_version_doc.version
            
            # Hash changed - create new version
            logger.info(f"{scenario.value}: Template changed! Creating new version...")
            return await self._create_new_version(
                scenario=scenario,
                template=current_template,
                template_hash=template_hash,
                previous_version_number=current_version_doc.version_number
            )
        else:
            # No version exists - create first version
            logger.info(f"{scenario.value}: No version found in DB, creating initial version...")
            return await self._create_new_version(
                scenario=scenario,
                template=current_template,
                template_hash=template_hash,
                previous_version_number=0
            )

    async def _create_new_version(
        self,
        scenario: PromptScenario,
        template: str,
        template_hash: str,
        previous_version_number: int
    ) -> str:
        """Create a new prompt version in MongoDB."""
        # Mark old version as not current (use transaction-like approach)
        await PromptVersionMongo.find(
            PromptVersionMongo.scenario == scenario.value,
            PromptVersionMongo.is_current == True
        ).update_many({"$set": {"is_current": False}})
        
        # Generate new version string with timestamp for uniqueness
        new_version_number = previous_version_number + 1
        now = datetime.utcnow()
        today = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H%M%S")  # Add time for uniqueness
        
        # Format: SCENARIO_V{NUMBER}_{DATE}_{TIMESTAMP}
        scenario_name = scenario.value.upper().replace("_", "")
        version_string = f"{scenario_name}_V{new_version_number}_{today}_{timestamp}"
        
        # Create new version document
        version_doc = PromptVersionMongo(
            scenario=scenario.value,
            version=version_string,
            template_hash=template_hash,
            template_content=template,
            is_current=True,
            version_number=new_version_number,
            created_at=now
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

