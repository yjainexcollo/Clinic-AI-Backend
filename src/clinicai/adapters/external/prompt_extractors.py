"""
Prompt template extractors for automatic version detection.

Each extractor function extracts the static template part of a prompt,
normalizing dynamic variables (like {transcript}, {context}) to placeholders.
"""

import re
from typing import Callable, Dict
from clinicai.adapters.external.prompt_registry import PromptScenario


def normalize_template(template: str) -> str:
    """
    Normalize a prompt template by:
    1. Replacing dynamic variables with placeholders
    2. Normalizing whitespace
    3. Removing version strings (they change)
    
    This ensures that only structural changes to prompts create new versions,
    not changes to dynamic data.
    """
    if not template:
        return ""
    
    normalized = template
    
    # Replace common dynamic variables with placeholders (order matters!)
    # First handle complex expressions
    normalized = re.sub(
        r'\{[^}]+\.get\([^)]+\)\}',  # {dict.get('key', 'default')}
        '{DYNAMIC_DICT}',
        normalized
    )
    normalized = re.sub(
        r'\{[^}]+\[[^\]]+\]\}',  # {dict['key']} or {dict["key"]}
        '{DYNAMIC_DICT}',
        normalized
    )
    # Then handle simple variables (but not the placeholders we just created)
    normalized = re.sub(
        r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}',  # {variable} - but not {DYNAMIC_DICT} etc.
        lambda m: '{DYNAMIC_VAR}' if m.group(1) not in ['DYNAMIC_DICT', 'DYNAMIC_VAR', 'VERSION'] else m.group(0),
        normalized
    )
    
    # Remove version strings (they change with each version)
    normalized = re.sub(
        r'Prompt version:\s*\{?[A-Z0-9_\-]+\}?',
        'Prompt version: {VERSION}',
        normalized,
        flags=re.IGNORECASE
    )
    
    # Normalize whitespace (preserve intentional line breaks)
    normalized = re.sub(r'[ \t]+', ' ', normalized)  # Multiple spaces/tabs -> single space
    normalized = re.sub(r' *\n *', '\n', normalized)  # Clean up line breaks
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)  # Max 2 consecutive newlines
    normalized = normalized.strip()
    
    return normalized


def extract_soap_prompt() -> str:
    """Extract SOAP prompt template from soap_service_openai.py"""
    import inspect
    from clinicai.adapters.external import soap_service_openai
    
    method = getattr(soap_service_openai.OpenAISoapService, 'generate_soap_note', None)
    if not method:
        raise ValueError("Could not find generate_soap_note method")
    
    source = inspect.getsource(method)
    
    # Extract English prompt (else block) - handle f-strings
    # Pattern matches: else:\n            prompt = f"""..."""
    match = re.search(
        r'else:\s+prompt\s*=\s*f?"""(.*?)"""',
        source,
        re.DOTALL
    )
    
    if not match:
        # Try without 'else:' prefix
        match = re.search(
            r'prompt\s*=\s*f?"""You are a clinical scribe(.*?)"""',
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
    from clinicai.adapters.external import soap_service_openai
    
    method = getattr(soap_service_openai.OpenAISoapService, 'generate_post_visit_summary', None)
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
    """Extract intake question prompt template (uses system_prompt from Agent-03)"""
    import inspect
    from clinicai.adapters.external import question_service_openai
    
    # Try to get Agent-03's system prompt from QuestionGenerator class
    try:
        question_gen_class = getattr(question_service_openai, 'QuestionGenerator', None)
        if question_gen_class:
            method = getattr(question_gen_class, '_generate_question', None)
            if method:
                source = inspect.getsource(method)
                # Extract English system_prompt (else block for Agent-03)
                match = re.search(
                    r'else:\s+system_prompt\s*=\s*"""(.*?)"""',
                    source,
                    re.DOTALL
                )
                if match:
                    template = match.group(1)
                    return normalize_template(template)
    except Exception:
        pass
    
    # Fallback: try to extract from generate_next_question method
    method = getattr(question_service_openai.QuestionServiceOpenAI, 'generate_next_question', None)
    if not method:
        raise ValueError("Could not find generate_next_question method")
    
    source = inspect.getsource(method)
    
    # Extract English system_prompt (else block for Agent-03)
    match = re.search(
        r'else:\s+system_prompt\s*=\s*"""(.*?)"""',
        source,
        re.DOTALL
    )
    
    if not match:
        # Try to find Agent-03 prompt specifically
        match = re.search(
            r'system_prompt\s*=\s*"""You are AGENT-03(.*?)"""',
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
    from clinicai.adapters.external import question_service_openai
    
    method = getattr(question_service_openai.QuestionServiceOpenAI, 'generate_previsit_summary', None)
    if not method:
        raise ValueError("Could not find generate_previsit_summary method")
    
    source = inspect.getsource(method)
    
    # Try to find prompt in messages construction
    match = re.search(
        r'prompt\s*=\s*f?"""(.*?)"""',
        source,
        re.DOTALL
    )
    
    if not match:
        # Try to find in messages list construction
        match = re.search(
            r'"content":\s*f?"""(.*?)"""',
            source,
            re.DOTALL
        )
    
    if not match:
        raise ValueError("Could not extract pre-visit prompt template")
    
    template = match.group(1)
    return normalize_template(template)


def extract_redflag_prompt() -> str:
    """Extract red-flag checker prompt template (from Agent-01 analyze_condition)"""
    import inspect
    from clinicai.adapters.external import question_service_openai
    
    # Agent-01 is MedicalContextAnalyzer
    try:
        analyzer_class = getattr(question_service_openai, 'MedicalContextAnalyzer', None)
        if analyzer_class:
            method = getattr(analyzer_class, 'analyze_condition', None)
            if method:
                source = inspect.getsource(method)
                # Extract English system_prompt (else block)
                match = re.search(
                    r'else:\s+system_prompt\s*=\s*"""(.*?)"""',
                    source,
                    re.DOTALL
                )
                if match:
                    template = match.group(1)
                    return normalize_template(template)
    except Exception:
        pass
    
    # Fallback: try to find in QuestionServiceOpenAI
    method = getattr(question_service_openai.QuestionServiceOpenAI, 'analyze_condition', None)
    if not method:
        raise ValueError("Could not find analyze_condition method")
    
    source = inspect.getsource(method)
    
    # Extract English system_prompt (else block)
    match = re.search(
        r'else:\s+system_prompt\s*=\s*"""(.*?)"""',
        source,
        re.DOTALL
    )
    
    if not match:
        # Try to find Agent-01 prompt specifically
        match = re.search(
            r'system_prompt\s*=\s*"""You are AGENT-01(.*?)"""',
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

