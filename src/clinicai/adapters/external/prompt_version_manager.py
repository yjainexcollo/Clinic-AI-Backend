"""
Automatic prompt version detection and management.

This service:
1. Extracts prompt templates from code
2. Calculates hash to detect changes
3. Only creates new versions when code is deployed (git commit hash changes)
4. Skips version creation during development/testing (same git commit)
5. Stores versions in MongoDB (4 documents, one per scenario, each with versions array)
6. Updates PROMPT_VERSIONS registry
"""

import hashlib
import logging
import os
import subprocess
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


def _get_git_commit_hash() -> Optional[str]:
    """
    Get current git commit hash if available, but ONLY if on main/master branch.
    
    This prevents false version creations from local/feature branch commits.
    Only tracks commits on main/master branch (deployed code).
    
    Tries multiple methods:
    1. Environment variable GITHUB_SHA (set by GitHub Actions) - always trusted
    2. Environment variable GIT_COMMIT (set manually) - always trusted
    3. Git command: git rev-parse HEAD (only if on main/master branch)
    
    Returns:
        Git commit hash string or None if unavailable or not on main/master
    """
    # Try environment variables first (CI/CD deployments) - always trusted
    github_sha = os.getenv("GITHUB_SHA")
    if github_sha:
        logger.debug(f"Using git commit from GITHUB_SHA: {github_sha[:8]}...")
        return github_sha
    
    git_commit = os.getenv("GIT_COMMIT")
    if git_commit:
        logger.debug(f"Using git commit from GIT_COMMIT: {git_commit[:8]}...")
        return git_commit
    
    # Try git command (local development or deployments with git available)
    # IMPORTANT: Only return commit hash if on main/master branch
    try:
        import pathlib
        
        # Find project root by looking for .git directory
        # Start from this file and walk up until we find .git
        current_file = pathlib.Path(__file__).resolve()
        current_dir = current_file.parent
        
        # Walk up the directory tree to find .git folder
        project_root = None
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / ".git").exists() or (parent / ".git").is_dir():
                project_root = parent
                break
        
        # If .git not found, try going up 5 levels (typical structure)
        if not project_root:
            project_root = current_file.parent.parent.parent.parent.parent
        
        if project_root and project_root.exists():
            # First, check current branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(project_root),
                timeout=5,
                check=False
            )
            
            if branch_result.returncode == 0:
                current_branch = branch_result.stdout.strip()
                
                # Only track commits on main/master branch
                if current_branch not in ["main", "master", "production"]:
                    logger.debug(
                        f"Skipping version tracking: on branch '{current_branch}' "
                        f"(only tracking main/master/production). "
                        f"Version will be created when code is merged to main/master."
                    )
                    return None
                
                # On main/master branch, get commit hash
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=str(project_root),
                    timeout=5,
                    check=False
                )
                if result.returncode == 0:
                    commit_hash = result.stdout.strip()
                    if commit_hash:
                        logger.debug(
                            f"Using git commit from git command (branch: {current_branch}): "
                            f"{commit_hash[:8]}..."
                        )
                        return commit_hash
            else:
                # Could not determine branch, skip tracking
                logger.debug(f"Could not determine git branch: {branch_result.stderr}")
                return None
                
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Could not get git commit hash: {e}")
    
    # No git commit available or not on main/master branch
    logger.info(
        "ℹ️  No git commit hash available or not on main/master branch. "
        "Version tracking will only create versions on deployment "
        "(when code is on main/master branch and git commit changes). "
        "Set GITHUB_SHA or GIT_COMMIT environment variable for deployment-based tracking."
    )
    return None


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
        - Documents with invalid scenario values (including enum values like "previsit_summary")
        - Duplicate documents (keeps the most recent one per scenario)
        - Old documents from previous structure (documents without 'versions' array)
        - Documents with enum values that should be mapped to correct field values
        """
        try:
            # Expected scenario values (database field values)
            expected_scenarios = {"intake", "previsit", "soap", "postvisit"}
            
            # Mapping from enum values to field values (for migration/detection)
            enum_to_field_mapping = {
                "previsit_summary": "previsit",
                "soap_summary": "soap",
                "postvisit_summary": "postvisit",
                "intake": "intake",  # Already correct
            }
            
            # Find all documents
            all_docs = await PromptVersionMongo.find({}).to_list()
            
            # Track documents by scenario (using field values, not enum values)
            docs_by_scenario: Dict[str, List[PromptVersionMongo]] = {}
            invalid_docs = []
            migration_candidates: Dict[str, PromptVersionMongo] = {}  # field_value -> doc to migrate
            
            for doc in all_docs:
                scenario = getattr(doc, "scenario", None)
                
                # Check if document has 'versions' field (new structure)
                if not hasattr(doc, "versions") or not isinstance(getattr(doc, "versions", None), list):
                    invalid_docs.append(doc)
                    logger.info(f"🗑️  Marking document for deletion: old structure (no versions array), ID: {doc.id}")
                    continue
                
                # Check if scenario is an enum value that needs migration
                if scenario in enum_to_field_mapping:
                    correct_field_value = enum_to_field_mapping[scenario]
                    logger.info(
                        f"🔄 Found document with enum value '{scenario}' (should be '{correct_field_value}'), "
                        f"ID: {doc.id}. Will migrate or delete if duplicate exists."
                    )
                    # Check if correct document already exists
                    existing_correct = await PromptVersionMongo.find_one({"scenario": correct_field_value})
                    if existing_correct:
                        # Correct document exists, mark enum-value doc for deletion
                        invalid_docs.append(doc)
                        logger.info(
                            f"🗑️  Marking enum-value document for deletion: '{scenario}' "
                            f"(correct document '{correct_field_value}' already exists), ID: {doc.id}"
                        )
                    else:
                        # No correct document exists, mark for migration
                        if correct_field_value not in migration_candidates:
                            migration_candidates[correct_field_value] = doc
                        else:
                            # Multiple enum-value docs for same scenario, keep the one with more versions
                            existing_candidate = migration_candidates[correct_field_value]
                            if len(getattr(doc, "versions", [])) > len(getattr(existing_candidate, "versions", [])):
                                invalid_docs.append(existing_candidate)
                                migration_candidates[correct_field_value] = doc
                            else:
                                invalid_docs.append(doc)
                    continue
                
                # Check if scenario is valid (must be in expected_scenarios)
                if scenario not in expected_scenarios:
                    invalid_docs.append(doc)
                    logger.info(f"🗑️  Marking document for deletion: invalid scenario '{scenario}', ID: {doc.id}")
                    continue
                
                # Group by scenario (valid field value)
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
            
            # Migrate enum-value documents to correct field values
            for correct_field_value, enum_doc in migration_candidates.items():
                try:
                    original_scenario = getattr(enum_doc, "scenario", None)
                    # Check again if correct document exists (race condition check)
                    existing_correct = await PromptVersionMongo.find_one({"scenario": correct_field_value})
                    if existing_correct:
                        # Correct document was created in parallel, delete enum-value doc
                        invalid_docs.append(enum_doc)
                        logger.info(
                            f"🗑️  Marking enum-value document for deletion: '{original_scenario}' "
                            f"(correct document '{correct_field_value}' exists), ID: {enum_doc.id}"
                        )
                    else:
                        # Migrate: update scenario field to correct value
                        enum_doc.scenario = correct_field_value
                        await enum_doc.save()
                        logger.info(
                            f"✅ Migrated document from enum value '{original_scenario}' to field value '{correct_field_value}', "
                            f"ID: {enum_doc.id}"
                        )
                        # Add to docs_by_scenario for duplicate checking
                        if correct_field_value not in docs_by_scenario:
                            docs_by_scenario[correct_field_value] = []
                        docs_by_scenario[correct_field_value].append(enum_doc)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to migrate document {enum_doc.id}: {e}")
                    # If migration fails, mark for deletion
                    invalid_docs.append(enum_doc)
            
            # Delete invalid/duplicate documents
            for doc in invalid_docs:
                try:
                    await doc.delete()
                    logger.info(f"✅ Deleted invalid/duplicate document ID: {doc.id}, scenario: {getattr(doc, 'scenario', 'unknown')}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete document {doc.id}: {e}")
            
            total_removed = len(invalid_docs)
            # Count successfully migrated documents (not in invalid_docs)
            total_migrated = sum(
                1 for doc in migration_candidates.values()
                if doc not in invalid_docs
            )
            
            if total_removed > 0 or total_migrated > 0:
                logger.info(
                    f"🧹 Cleanup completed: removed {total_removed} invalid/duplicate documents, "
                    f"migrated {total_migrated} enum-value documents"
                )
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
        4. Get current git commit hash
        5. Check if version exists in DB
        6. Only create new version if:
           - Major version manually incremented, OR
           - Template hash changed AND git commit hash changed (deployment detected)
        7. Update PROMPT_VERSIONS dict
        
        Note: Versions are only created on deployment (when git commit changes),
        not during development/testing (same git commit).
        
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
        
        # Defensive check: ensure we never use enum values directly
        enum_values = {"previsit_summary", "soap_summary", "postvisit_summary"}
        if scenario_value in enum_values:
            raise ValueError(
                f"Invalid scenario value '{scenario_value}' (enum value detected). "
                f"Use _scenario_to_field_value() to convert enum to field value."
            )
        
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
        
        Only creates versions when code is deployed (git commit changes),
        not during development/testing (same git commit).
        
        Handles:
        1. Major version changes (manual via MAJOR_VERSIONS) - always creates version
        2. Minor version changes (auto-detected) - only if git commit changed (deployment)
        """
        # Extract current template from code
        try:
            current_template = extract_template(scenario)
        except Exception as e:
            logger.error(f"Failed to extract template for {scenario.value}: {e}")
            raise
        
        # Calculate hash
        template_hash = self._calculate_hash(current_template)
        
        # Get current git commit hash
        current_git_commit = _get_git_commit_hash()
        
        # Get desired major version from code configuration
        target_major = MAJOR_VERSIONS.get(scenario, 1)
        
        # Get the document for this scenario
        version_doc = await self._get_or_create_scenario_document(scenario)
        
        # Find current version in versions array
        current_version_entry = next((v for v in version_doc.versions if v.is_current), None)
        
        if current_version_entry:
            current_major = current_version_entry.major_version
            current_minor = current_version_entry.minor_version
            last_git_commit = current_version_entry.git_commit
            
            # Case 1: Major version upgrade (Manual trigger - always create version)
            if target_major > current_major:
                logger.info(
                    f"{scenario.value}: Major version change detected ({current_major} -> {target_major}). "
                    f"Creating new version regardless of git commit."
                )
                return await self._create_new_version(
                    version_doc=version_doc,
                    scenario=scenario,
                    template=current_template,
                    template_hash=template_hash,
                    major_version=target_major,
                    minor_version=0,
                    git_commit=current_git_commit
                )
            
            # Case 2: Template changed - check if deployment detected (git commit changed)
            if current_version_entry.template_hash != template_hash:
                # Only create new version if git commit has changed (deployment detected)
                if current_git_commit and current_git_commit != last_git_commit:
                    logger.info(
                        f"{scenario.value}: Template changed and new deployment detected "
                        f"(commit: {last_git_commit[:8] if last_git_commit else 'none'} -> {current_git_commit[:8]}). "
                        f"Auto-incrementing minor version..."
                    )
                    return await self._create_new_version(
                        version_doc=version_doc,
                        scenario=scenario,
                        template=current_template,
                        template_hash=template_hash,
                        major_version=current_major,
                        minor_version=current_minor + 1,
                        git_commit=current_git_commit
                    )
                elif not current_git_commit:
                    # No git commit available - skip version creation (development/testing mode)
                    # This prevents versions from being created on every startup during development
                    logger.info(
                        f"{scenario.value}: Template changed but no git commit available. "
                        f"Skipping version creation (development mode). "
                        f"Version will be created on next deployment when git commit is available. "
                        f"Set GITHUB_SHA or GIT_COMMIT environment variable for deployment-based tracking."
                    )
                    return current_version_entry.version
                else:
                    # Template changed but same git commit (development/testing - don't track)
                    logger.info(
                        f"{scenario.value}: Template changed but same git commit "
                        f"({current_git_commit[:8]}). Skipping version creation (development mode). "
                        f"Version will be created on next deployment."
                    )
                    return current_version_entry.version
                
            # Case 3: No change
            logger.debug(f"{scenario.value}: No change detected (v{current_version_entry.version})")
            return current_version_entry.version
            
        else:
            # No version exists - create first version
            logger.info(
                f"{scenario.value}: No version found in DB, creating initial version {target_major}.0... "
                f"(git commit: {current_git_commit[:8] if current_git_commit else 'none'})"
            )
            return await self._create_new_version(
                version_doc=version_doc,
                scenario=scenario,
                template=current_template,
                template_hash=template_hash,
                major_version=target_major,
                minor_version=0,
                git_commit=current_git_commit
            )

    async def _create_new_version(
        self,
        version_doc: PromptVersionMongo,
        scenario: PromptScenario,
        template: str,
        template_hash: str,
        major_version: int,
        minor_version: int,
        git_commit: Optional[str] = None
    ) -> str:
        """
        Create a new prompt version and add to the versions array.
        
        Args:
            version_doc: The document containing versions array for this scenario
            scenario: The prompt scenario
            template: The full prompt template content
            template_hash: SHA256 hash of the template
            major_version: Major version number
            minor_version: Minor version number
            git_commit: Git commit hash (optional, for deployment tracking)
        """
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
            version_number=version_number,
            git_commit=git_commit
        )
        
        # Add new version to array
        version_doc.versions.append(new_version_entry)
        
        # Save the document
        await version_doc.save()
        
        git_info = f" (git: {git_commit[:8]}...)" if git_commit else " (no git commit)"
        logger.info(
            f"📝 Created new version: {scenario.value} → {version_string} "
            f"(hash: {template_hash[:8]}...){git_info}"
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
