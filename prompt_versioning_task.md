# Complete Prompt Versioning Implementation Guide

This document provides a complete guide to implementing automatic prompt version tracking in any FastAPI + MongoDB + Azure Application Insights codebase. Use this guide to recreate the entire system in a new repository.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Code Files](#code-files)
6. [Integration Points](#integration-points)
7. [Testing & Verification](#testing--verification)
8. [Monitoring & Queries](#monitoring--queries)

---

## Overview

### What This System Does

- **Automatic Version Detection**: Detects when prompt templates change in code
- **MongoDB Storage**: Stores version history in MongoDB
- **Telemetry Integration**: Includes version strings in Application Insights telemetry
- **Zero Manual Work**: No manual version updates needed - fully automatic

### Key Features

- ✅ Automatic version creation when prompts change
- ✅ SHA256 hash-based change detection
- ✅ Version history preserved in MongoDB
- ✅ Telemetry tracking in Application Insights
- ✅ Fallback to default versions if DB fails
- ✅ Non-breaking (startup won't fail if version detection fails)

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│              Prompt Version System                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. prompt_registry.py                                  │
│     - PromptScenario enum                               │
│     - PROMPT_VERSIONS dict (runtime registry)           │
│                                                         │
│  2. prompt_version_m.py (MongoDB Model)                 │
│     - Stores version history                            │
│     - Fields: scenario, version, template_hash, etc.    │
│                                                         │
│  3. prompt_extractors.py                               │
│     - Extracts templates from source code               │
│     - Normalizes templates (removes dynamic vars)       │
│                                                         │
│  4. prompt_version_manager.py                           │
│     - Detects changes (hash comparison)                 │
│     - Creates new versions                              │
│     - Updates PROMPT_VERSIONS registry                  │
│                                                         │
│  5. app.py (Integration)                                │
│     - Initializes on startup                            │
│     - Registers MongoDB model                           │
│                                                         │
│  6. llm_gateway.py (Telemetry)                          │
│     - Uses PROMPT_VERSIONS for telemetry               │
│     - Includes llm.prompt_version in spans              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Flow

1. **On Startup**:
   - Extract templates from code
   - Calculate hashes
   - Compare with MongoDB
   - Create new version if changed
   - Update PROMPT_VERSIONS dict

2. **During Runtime**:
   - LLM calls use PROMPT_VERSIONS[scenario]
   - Version included in telemetry automatically

---

## File Structure

```
backend/
├── src/
│   └── your_package/
│       ├── adapters/
│       │   ├── db/
│       │   │   └── mongo/
│       │   │       └── models/
│       │   │           └── prompt_version_m.py          # NEW
│       │   └── external/
│       │       ├── prompt_registry.py                    # NEW
│       │       ├── prompt_extractors.py                  # NEW
│       │       ├── prompt_version_manager.py             # NEW
│       │       ├── llm_gateway.py                        # MODIFY
│       │       ├── soap_service_openai.py               # MODIFY (if exists)
│       │       └── question_service_openai.py            # MODIFY (if exists)
│       └── app.py                                        # MODIFY
```

---

## Step-by-Step Implementation

### Step 1: Create Prompt Registry

**File**: `src/your_package/adapters/external/prompt_registry.py`

```python
from __future__ import annotations

from enum import Enum


class PromptScenario(str, Enum):
    """LLM scenarios for telemetry and prompt versioning."""

    INTAKE = "intake"
    PREVISIT_SUMMARY = "previsit_summary"
    RED_FLAG = "red_flag"
    SOAP = "soap_summary"
    POSTVISIT_SUMMARY = "postvisit_summary"

    # Add your scenarios here:
    # YOUR_SCENARIO = "your_scenario"


# Default/fallback versions (used before DB initialization)
_DEFAULT_VERSIONS: dict[PromptScenario, str] = {
    PromptScenario.INTAKE: "INTAKE_V1_2025-12-02",
    PromptScenario.PREVISIT_SUMMARY: "PREVISIT_V1_2025-12-02",
    PromptScenario.RED_FLAG: "RED_FLAG_V1_2025-12-02",
    PromptScenario.SOAP: "SOAP_V1_2025-12-02",
    PromptScenario.POSTVISIT_SUMMARY: "POSTVISIT_V1_2025-12-02",
    # Add defaults for your scenarios
}

# Runtime versions (updated by PromptVersionManager on startup)
PROMPT_VERSIONS: dict[PromptScenario, str] = _DEFAULT_VERSIONS.copy()


__all__ = ["PromptScenario", "PROMPT_VERSIONS"]
```

**Key Points**:
- Define all your LLM scenarios as enum values
- Set default version strings (format: `SCENARIO_V1_YYYY-MM-DD`)
- PROMPT_VERSIONS dict will be updated automatically on startup

---

### Step 2: Create MongoDB Model

**File**: `src/your_package/adapters/db/mongo/models/prompt_version_m.py`

```python
"""
MongoDB Beanie model for storing prompt version history.
"""

from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field

from your_package.adapters.external.prompt_registry import PromptScenario


class PromptVersionMongo(Document):
    """MongoDB model for prompt version tracking."""

    scenario: str = Field(..., description="Prompt scenario (e.g., 'soap_summary')")
    version: str = Field(..., description="Version string (e.g., 'SOAP_V1_2025-12-02')")
    template_hash: str = Field(..., description="SHA256 hash of the prompt template")
    template_content: str = Field(..., description="Full prompt template content")
    
    # Metadata
    is_current: bool = Field(default=True, description="Whether this is the current active version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this version was created")
    git_commit: Optional[str] = Field(None, description="Git commit hash (if available)")
    
    # Version number extraction (for sorting)
    version_number: int = Field(..., description="Numeric version (1, 2, 3, etc.)")

    class Settings:
        name = "prompt_versions"
        indexes = [
            "scenario",
            "version",
            "template_hash",
            "is_current",
            [("scenario", 1), ("is_current", 1)],  # Compound index for finding current version by scenario
            [("scenario", 1), ("version_number", -1)],  # Compound index for version history queries
        ]
```

**Key Points**:
- Uses Beanie ODM for MongoDB
- Stores template hash for change detection
- Stores full template content for reference
- Indexes optimized for common queries

---

### Step 3: Create Template Extractors

**File**: `src/your_package/adapters/external/prompt_extractors.py`

```python
"""
Prompt template extractors for automatic version detection.

Each extractor function extracts the static template part of a prompt,
normalizing dynamic variables (like {transcript}, {context}) to placeholders.
"""

import re
from typing import Callable, Dict
from your_package.adapters.external.prompt_registry import PromptScenario


def normalize_template(template: str) -> str:
    """
    Normalize a prompt template by:
    1. Replacing dynamic variables with placeholders
    2. Normalizing whitespace
    3. Removing version strings (they change)
    """
    # Replace common dynamic variables with placeholders
    normalized = re.sub(
        r'\{[^}]+\.get\([^)]+\)\}',  # {dict.get('key', 'default')}
        '{DYNAMIC_DICT}',
        template
    )
    normalized = re.sub(
        r'\{[^}]+\[[^\]]+\]\}',  # {dict['key']}
        '{DYNAMIC_DICT}',
        normalized
    )
    normalized = re.sub(
        r'\{[a-zA-Z_][a-zA-Z0-9_]*\}',  # {variable}
        '{DYNAMIC_VAR}',
        normalized
    )
    
    # Remove version strings (they change with each version)
    normalized = re.sub(
        r'Prompt version:\s*\{?[A-Z0-9_]+\}?',
        'Prompt version: {VERSION}',
        normalized,
        flags=re.IGNORECASE
    )
    
    # Normalize whitespace
    normalized = re.sub(r'[ \t]+', ' ', normalized)
    normalized = re.sub(r'\n\s*\n\s*\n+', '\n\n', normalized)
    normalized = normalized.strip()
    
    return normalized


def extract_soap_prompt() -> str:
    """Extract SOAP prompt template from soap_service_openai.py"""
    import inspect
    from your_package.adapters.external import soap_service_openai
    
    method = getattr(soap_service_openai.SOAPServiceOpenAI, 'generate_soap_note', None)
    if not method:
        raise ValueError("Could not find generate_soap_note method")
    
    source = inspect.getsource(method)
    
    # Extract English prompt (else block)
    match = re.search(
        r'else:\s+prompt\s*=\s*f?"""(.*?)"""',
        source,
        re.DOTALL
    )
    
    if not match:
        # Try Spanish version as fallback
        match = re.search(
            r'if lang == "sp":\s+prompt\s*=\s*f?"""(.*?)"""',
            source,
            re.DOTALL
        )
    
    if not match:
        raise ValueError("Could not extract SOAP prompt template")
    
    template = match.group(1)
    return normalize_template(template)


def extract_postvisit_prompt() -> str:
    """Extract post-visit summary prompt template"""
    import inspect
    from your_package.adapters.external import soap_service_openai
    
    method = getattr(soap_service_openai.SOAPServiceOpenAI, 'generate_post_visit_summary', None)
    if not method:
        raise ValueError("Could not find generate_post_visit_summary method")
    
    source = inspect.getsource(method)
    
    match = re.search(
        r'else:\s+prompt\s*=\s*f?"""(.*?)"""',
        source,
        re.DOTALL
    )
    
    if not match:
        match = re.search(
            r'prompt\s*=\s*f?"""(.*?)"""',
            source,
            re.DOTALL
        )
    
    if not match:
        raise ValueError("Could not extract post-visit prompt template")
    
    template = match.group(1)
    return normalize_template(template)


def extract_intake_prompt() -> str:
    """Extract intake question prompt template (uses system_prompt)"""
    import inspect
    from your_package.adapters.external import question_service_openai
    
    method = getattr(question_service_openai.QuestionServiceOpenAI, 'generate_next_question', None)
    if not method:
        raise ValueError("Could not find generate_next_question method")
    
    source = inspect.getsource(method)
    
    # Extract English system_prompt (else block)
    match = re.search(
        r'else:\s+system_prompt\s*=\s*"""(.*?)"""',
        source,
        re.DOTALL
    )
    
    if not match:
        match = re.search(
            r'system_prompt\s*=\s*"""(.*?)"""',
            source,
            re.DOTALL
        )
    
    if not match:
        raise ValueError("Could not extract intake prompt template")
    
    template = match.group(1)
    return normalize_template(template)


def extract_previsit_prompt() -> str:
    """Extract pre-visit summary prompt template"""
    import inspect
    from your_package.adapters.external import question_service_openai
    
    method = getattr(question_service_openai.QuestionServiceOpenAI, 'generate_previsit_summary', None)
    if not method:
        raise ValueError("Could not find generate_previsit_summary method")
    
    source = inspect.getsource(method)
    
    match = re.search(
        r'else:\s+prompt\s*=\s*f?"""(.*?)"""',
        source,
        re.DOTALL
    )
    
    if not match:
        match = re.search(
            r'prompt\s*=\s*f?"""(.*?)"""',
            source,
            re.DOTALL
        )
    
    if not match:
        raise ValueError("Could not extract pre-visit prompt template")
    
    template = match.group(1)
    return normalize_template(template)


def extract_redflag_prompt() -> str:
    """Extract red-flag checker prompt template"""
    import inspect
    from your_package.adapters.external import question_service_openai
    
    method = getattr(question_service_openai.QuestionServiceOpenAI, 'check_red_flags', None)
    if not method:
        raise ValueError("Could not find check_red_flags method")
    
    source = inspect.getsource(method)
    
    match = re.search(
        r'else:\s+prompt\s*=\s*f?"""(.*?)"""',
        source,
        re.DOTALL
    )
    
    if not match:
        match = re.search(
            r'prompt\s*=\s*f?"""(.*?)"""',
            source,
            re.DOTALL
        )
    
    if not match:
        raise ValueError("Could not extract red-flag prompt template")
    
    template = match.group(1)
    return normalize_template(template)


# Registry of extractors
EXTRACTORS: Dict[PromptScenario, Callable[[], str]] = {
    PromptScenario.SOAP: extract_soap_prompt,
    PromptScenario.POSTVISIT_SUMMARY: extract_postvisit_prompt,
    PromptScenario.INTAKE: extract_intake_prompt,
    PromptScenario.PREVISIT_SUMMARY: extract_previsit_prompt,
    PromptScenario.RED_FLAG: extract_redflag_prompt,
    # Add your extractors here:
    # PromptScenario.YOUR_SCENARIO: extract_your_scenario_prompt,
}


def extract_template(scenario: PromptScenario) -> str:
    """Extract normalized template for a given scenario."""
    extractor = EXTRACTORS.get(scenario)
    if not extractor:
        raise ValueError(f"No extractor found for scenario: {scenario}")
    
    return extractor()
```

**Key Points**:
- Each extractor uses `inspect.getsource()` to get method source code
- Uses regex to extract prompt strings from source
- Normalizes templates (removes dynamic variables)
- Add extractors for each of your scenarios

**Customization**:
- Adjust regex patterns to match your prompt format
- Handle multi-language prompts if needed
- Extract from different method names/locations

---

### Step 4: Create Version Manager

**File**: `src/your_package/adapters/external/prompt_version_manager.py`

```python
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
from typing import Dict, Optional

from your_package.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS
from your_package.adapters.external.prompt_extractors import extract_template
from your_package.adapters.db.mongo.models.prompt_version_m import PromptVersionMongo

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
        logger.info("🔍 Starting automatic prompt version detection...")
        
        versions: Dict[PromptScenario, str] = {}
        
        for scenario in PromptScenario:
            try:
                version = await self._check_and_update_scenario(scenario)
                versions[scenario] = version
                logger.info(f"✅ {scenario.value}: {version}")
            except Exception as e:
                logger.error(f"❌ Failed to process {scenario.value}: {e}", exc_info=True)
                # Fallback to hardcoded version if available
                fallback_version = PROMPT_VERSIONS.get(scenario)
                if fallback_version:
                    versions[scenario] = fallback_version
                    logger.warning(f"⚠️  Using fallback version for {scenario.value}: {fallback_version}")
                else:
                    raise
        
        # Update the global PROMPT_VERSIONS dict
        PROMPT_VERSIONS.update(versions)
        self._versions_cache = versions
        
        logger.info("✅ Prompt version detection completed")
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
        # Mark old version as not current
        await PromptVersionMongo.find(
            PromptVersionMongo.scenario == scenario.value,
            PromptVersionMongo.is_current == True
        ).update_many({"$set": {"is_current": False}})
        
        # Generate new version string
        new_version_number = previous_version_number + 1
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        # Format: SCENARIO_V{NUMBER}_{DATE}
        scenario_name = scenario.value.upper().replace("_", "")
        version_string = f"{scenario_name}_V{new_version_number}_{today}"
        
        # Create new version document
        version_doc = PromptVersionMongo(
            scenario=scenario.value,
            version=version_string,
            template_hash=template_hash,
            template_content=template,
            is_current=True,
            version_number=new_version_number,
            created_at=datetime.utcnow()
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
```

**Key Points**:
- Singleton pattern for version manager
- Hash-based change detection
- Automatic version creation
- Updates global PROMPT_VERSIONS dict
- Error handling with fallback

---

### Step 5: Integrate in app.py

**File**: `src/your_package/app.py`

**Add to imports section** (around line 72-74):
```python
from your_package.adapters.db.mongo.models.prompt_version_m import (
    PromptVersionMongo,
)
```

**Add to Beanie initialization** (in `lifespan` function, around line 98-108):
```python
await init_beanie(
    database=db,
    document_models=[
        PatientMongo,  # Your existing models
        VisitMongo,
        # ... other models ...
        PromptVersionMongo,  # ADD THIS
    ],
)
```

**Add version initialization** (after database connection, around line 110-135):
```python
# Initialize automatic prompt version detection
try:
    msg = "Initializing automatic prompt version detection..."
    print(msg, flush=True)
    logger.info(msg)
    from your_package.adapters.external.prompt_version_manager import (
        get_prompt_version_manager,
    )
    version_manager = get_prompt_version_manager()
    versions = await version_manager.initialize_versions()
    msg = f"✅ Prompt versions initialized: {len(versions)} scenarios"
    print(msg, flush=True)
    logger.info(msg)
    for scenario, version in versions.items():
        logger.info(f"   {scenario.value}: {version}")
except Exception as e:
    # Don't fail startup if prompt version detection fails
    warning_msg = f"⚠️  Prompt version detection failed: {e}"
    print(warning_msg, flush=True)
    logger.warning(warning_msg)
    logger.warning("Using fallback versions from prompt_registry.py")
    logging.warning(f"Prompt version detection error: {e}", exc_info=True)
```

**Key Points**:
- Register PromptVersionMongo in Beanie
- Initialize version detection after DB connection
- Error handling prevents startup failure
- Falls back to default versions if detection fails

---

### Step 6: Integrate in LLM Gateway

**File**: `src/your_package/adapters/external/llm_gateway.py`

**Add import** (at top):
```python
from your_package.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS
```

**Modify `call_llm_with_telemetry` function** (around line 34):
```python
async def call_llm_with_telemetry(
    ai_client: AzureAIClient,
    scenario: PromptScenario,  # Use PromptScenario enum
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: Optional[int] = None,
) -> Any:
    """
    Central gateway for LLM calls with telemetry.
    """
    prompt_version = PROMPT_VERSIONS.get(scenario, "UNKNOWN")  # Get version
    start_time = time.perf_counter()

    with trace_operation(
        "llm_call",
        {
            "llm.scenario": scenario.value,
            "llm.prompt_version": prompt_version,  # Include version
            "llm.model": model,
        },
    ) as span:
        # ... rest of function ...
        # Add version to span attributes
        add_span_attribute(span, "llm.prompt_version", prompt_version)
        # ... rest of function ...
```

**Key Points**:
- Use PromptScenario enum for type safety
- Get version from PROMPT_VERSIONS dict
- Include version in telemetry spans
- Version automatically included in Application Insights

---

### Step 7: Update Service Files

**For each service file that makes LLM calls:**

**Add imports**:
```python
from your_package.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS
from your_package.adapters.external.llm_gateway import call_llm_with_telemetry
```

**Example: soap_service_openai.py**
```python
# Get version
prompt_version = PROMPT_VERSIONS[PromptScenario.SOAP]

# Include in prompt (optional but recommended)
prompt = f"""
Prompt version: {prompt_version}

You are a clinical scribe...
"""

# Use call_llm_with_telemetry (not direct client calls)
response = await call_llm_with_telemetry(
    ai_client=self._client,
    scenario=PromptScenario.SOAP,  # Use enum
    messages=[...],
    model=...,
    temperature=...,
)
```

**Key Points**:
- Import PromptScenario and PROMPT_VERSIONS
- Use PromptScenario enum (not strings)
- Use call_llm_with_telemetry gateway
- Optionally include version in prompt text

---

## Code Files

### Complete File List

1. **prompt_registry.py** - Scenario enum and version registry
2. **prompt_version_m.py** - MongoDB model
3. **prompt_extractors.py** - Template extraction logic
4. **prompt_version_manager.py** - Version detection and creation
5. **app.py** - Integration (modify existing)
6. **llm_gateway.py** - Telemetry integration (modify existing)
7. **service files** - Update to use PromptScenario (modify existing)

---

## Integration Points

### Required Modifications

1. **app.py**:
   - Import PromptVersionMongo
   - Register in Beanie models
   - Initialize version detection on startup

2. **llm_gateway.py**:
   - Import PromptScenario, PROMPT_VERSIONS
   - Use PromptScenario enum
   - Include version in telemetry

3. **Service files** (soap_service, question_service, etc.):
   - Import PromptScenario, PROMPT_VERSIONS
   - Import call_llm_with_telemetry
   - Use PromptScenario enum
   - Use call_llm_with_telemetry instead of direct client calls

---

## Testing & Verification

### 1. Verify Files Created

```bash
# Check all files exist
ls src/your_package/adapters/external/prompt_*.py
ls src/your_package/adapters/db/mongo/models/prompt_version_m.py
```

### 2. Check Imports

```python
# Test imports work
from your_package.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS
from your_package.adapters.external.prompt_version_manager import get_prompt_version_manager
from your_package.adapters.db.mongo.models.prompt_version_m import PromptVersionMongo
```

### 3. Startup Verification

When you start the app, you should see:
```
✅ Database connection established
Initializing automatic prompt version detection...
✅ soap_summary: SOAP_V1_2025-12-02
✅ intake: INTAKE_V1_2025-12-02
...
✅ Prompt versions initialized: 5 scenarios
```

### 4. MongoDB Verification

```javascript
// Check versions were created
db.prompt_versions.find({ is_current: true })

// Check version history
db.prompt_versions.find({ scenario: "soap_summary" }).sort({ version_number: -1 })
```

### 5. Telemetry Verification

In Application Insights, run:
```kql
traces
| where timestamp > ago(1h)
| where name == "llm_call"
| distinct customDimensions.llm_prompt_version
```

Should show version strings like `SOAP_V1_2025-12-02`.

---

## Monitoring & Queries

### Application Insights Queries

See `PROMPT_VERSION_TRACKING_QUERIES.md` for 21 ready-to-use KQL queries.

**Quick Query**:
```kql
traces
| where timestamp > ago(24h)
| where name == "llm_call"
| summarize count() by PromptVersion = customDimensions.llm_prompt_version
```

### MongoDB Queries

**Current versions**:
```javascript
db.prompt_versions.find({ is_current: true })
```

**Version history**:
```javascript
db.prompt_versions.find({ scenario: "soap_summary" })
  .sort({ version_number: -1 })
```

---

## Customization Guide

### Adding New Scenarios

1. **Add to PromptScenario enum** (prompt_registry.py):
```python
class PromptScenario(str, Enum):
    # ... existing ...
    YOUR_NEW_SCENARIO = "your_new_scenario"
```

2. **Add default version** (prompt_registry.py):
```python
_DEFAULT_VERSIONS: dict[PromptScenario, str] = {
    # ... existing ...
    PromptScenario.YOUR_NEW_SCENARIO: "YOURNEWSCENARIO_V1_2025-12-02",
}
```

3. **Create extractor** (prompt_extractors.py):
```python
def extract_your_scenario_prompt() -> str:
    """Extract your scenario prompt template"""
    import inspect
    from your_package.adapters.external import your_service
    
    method = getattr(your_service.YourService, 'your_method', None)
    if not method:
        raise ValueError("Could not find your_method")
    
    source = inspect.getsource(method)
    match = re.search(r'prompt\s*=\s*f?"""(.*?)"""', source, re.DOTALL)
    
    if not match:
        raise ValueError("Could not extract prompt template")
    
    template = match.group(1)
    return normalize_template(template)
```

4. **Register extractor** (prompt_extractors.py):
```python
EXTRACTORS: Dict[PromptScenario, Callable[[], str]] = {
    # ... existing ...
    PromptScenario.YOUR_NEW_SCENARIO: extract_your_scenario_prompt,
}
```

5. **Use in service**:
```python
from your_package.adapters.external.prompt_registry import PromptScenario

response = await call_llm_with_telemetry(
    ai_client=self._client,
    scenario=PromptScenario.YOUR_NEW_SCENARIO,
    messages=[...],
    model=...,
    temperature=...,
)
```

---

## Troubleshooting

### Issue: Version detection fails on startup

**Symptoms**: Warning message in logs, fallback versions used

**Check**:
1. MongoDB connection working?
2. Template extraction working? (check extractor regex)
3. Method names correct in extractors?

**Solution**: Check startup logs for specific error

### Issue: Wrong version in telemetry

**Check**:
1. Did app restart after prompt change?
2. Is PROMPT_VERSIONS dict updated? (check logs)

**Solution**: Restart app to trigger version detection

### Issue: Template extraction fails

**Check**:
1. Method name correct?
2. Prompt format matches regex?
3. Prompt uses triple quotes `"""`?

**Solution**: Adjust extractor regex to match your prompt format

---

## Summary Checklist

- [ ] Create `prompt_registry.py` with all scenarios
- [ ] Create `prompt_version_m.py` MongoDB model
- [ ] Create `prompt_extractors.py` with all extractors
- [ ] Create `prompt_version_manager.py` version manager
- [ ] Modify `app.py` to register model and initialize
- [ ] Modify `llm_gateway.py` to use PROMPT_VERSIONS
- [ ] Update all service files to use PromptScenario
- [ ] Test startup - verify versions initialized
- [ ] Test MongoDB - verify versions created
- [ ] Test telemetry - verify versions in Application Insights
- [ ] Change a prompt - verify new version created automatically

---

## Key Principles

1. **Automatic**: No manual version updates needed
2. **Non-Breaking**: Startup won't fail if detection fails
3. **Backward Compatible**: Falls back to defaults
4. **Telemetry Ready**: Versions automatically tracked
5. **History Preserved**: All versions stored in MongoDB

---

**Last Updated**: 2025-01-15
**Version**: 1.0
**Maintained By**: Your Team

---

## Next Steps After Implementation

1. Deploy to staging environment
2. Verify versions are created on first startup
3. Make a test prompt change
4. Verify new version created automatically
5. Check Application Insights for version tracking
6. Set up monitoring dashboards using provided queries

---

**This guide contains everything needed to recreate the prompt versioning system in any new repository.**

