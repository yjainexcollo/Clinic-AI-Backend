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
    """
    Extract intake prompt template combining all 3 agents.
    
    CHANGE NOTE (2025-12): Updated to extract prompts from all 3 intake agents:
    - Agent-01 (Medical Context Analyzer)
    - Agent-02 (Coverage & Fact Extractor) 
    - Agent-03 (Question Generator)
    
    This ensures that a change to ANY agent will trigger a new INTAKE version.
    """
    import inspect
    import logging
    from clinicai.adapters.external import question_service_openai
    
    combined_prompts = []
    
    # ============================================================================
    # Extract Agent-01 (Medical Context Analyzer) prompt
    # ============================================================================
    try:
        analyzer_class = getattr(question_service_openai, 'MedicalContextAnalyzer', None)
        if analyzer_class:
            method = getattr(analyzer_class, 'analyze_condition', None)
            if method:
                source = inspect.getsource(method)
                # Extract English system_prompt (else block)
                # Pattern: else:  # English\n            system_prompt = """..."""
                match = re.search(
                    r'else:\s*#\s*English\s+system_prompt\s*=\s*"""(.*?)"""',
                    source,
                    re.DOTALL
                )
                if not match:
                    # Try without comment
                    match = re.search(
                        r'else:\s+system_prompt\s*=\s*"""You are AGENT-01(.*?)"""',
                        source,
                        re.DOTALL
                    )
                if match:
                    agent1_prompt = normalize_template(match.group(1) if match.group(1).startswith("You are") else "You are AGENT-01" + match.group(1))
                    combined_prompts.append(f"=== AGENT-01: MEDICAL CONTEXT ANALYZER ===\n{agent1_prompt}")
                else:
                    logging.warning("Could not extract Agent-01 prompt with regex")
    except Exception as e:
        logging.warning(f"Could not extract Agent-01 prompt: {e}")
    
    # ============================================================================
    # Extract Agent-02 (Coverage & Fact Extractor) prompt
    # ============================================================================
    try:
        extractor_class = getattr(question_service_openai, 'AnswerExtractor', None)
        if extractor_class:
            method = getattr(extractor_class, 'extract_covered_information', None)
            if method:
                source = inspect.getsource(method)
                # Extract English system_prompt (else block)
                # Pattern: else:\n            # CHANGE NOTE (2025-12): Aligned English Agent-2 schema...\n            system_prompt = """..."""
                match = re.search(
                    r'else:\s+#.*?system_prompt\s*=\s*"""(.*?)"""',
                    source,
                    re.DOTALL
                )
                if not match:
                    # Try simpler pattern
                    match = re.search(
                        r'else:\s+system_prompt\s*=\s*"""You are AGENT-02(.*?)"""',
                        source,
                        re.DOTALL
                    )
                if match:
                    agent2_prompt = normalize_template(match.group(1) if match.group(1).startswith("You are") else "You are AGENT-02" + match.group(1))
                    combined_prompts.append(f"=== AGENT-02: COVERAGE & FACT EXTRACTOR ===\n{agent2_prompt}")
                else:
                    logging.warning("Could not extract Agent-02 prompt with regex")
    except Exception as e:
        logging.warning(f"Could not extract Agent-02 prompt: {e}")
    
    # ============================================================================
    # Extract Agent-03 (Question Generator) prompt
    # ============================================================================
    try:
        generator_class = getattr(question_service_openai, 'QuestionGenerator', None)
        if generator_class:
            method = getattr(generator_class, 'generate_question', None)
            if method:
                source = inspect.getsource(method)
                # Extract English system_prompt (else block, non-deep-diagnostic)
                # Look for the main system_prompt assignment (not deep diagnostic)
                # Pattern: else:\n            system_prompt = """..."""
                # We need to find the FIRST occurrence (not the deep diagnostic ones)
                matches = list(re.finditer(
                    r'else:\s+system_prompt\s*=\s*"""(.*?)"""',
                    source,
                    re.DOTALL
                ))
                if matches:
                    # Take the first match (main prompt, not deep diagnostic)
                    match = matches[0]
                    agent3_prompt = normalize_template(match.group(1))
                    combined_prompts.append(f"=== AGENT-03: QUESTION GENERATOR ===\n{agent3_prompt}")
                else:
                    logging.warning("Could not extract Agent-03 prompt with regex")
    except Exception as e:
        logging.warning(f"Could not extract Agent-03 prompt: {e}")
    
    # ============================================================================
    # Combine all prompts
    # ============================================================================
    if not combined_prompts:
        raise ValueError("Could not extract any intake agent prompts")
    
    # Join all prompts with separator
    combined_template = "\n\n".join(combined_prompts)
    
    # Return the combined, normalized template
    return combined_template


def extract_previsit_prompt() -> str:
    """Extract pre-visit summary prompt template"""
    import inspect
    from clinicai.adapters.external import question_service_openai
    
    # FIXED: Changed QuestionServiceOpenAI -> OpenAIQuestionService
    # FIXED: Changed generate_previsit_summary -> generate_pre_visit_summary
    method = getattr(question_service_openai.OpenAIQuestionService, 'generate_pre_visit_summary', None)
    if not method:
        raise ValueError("Could not find generate_pre_visit_summary method")
    
    source = inspect.getsource(method)
    
    # Try to find prompt in messages construction
    # The prompt is constructed with string concatenation, not triple quotes
    match = re.search(
        r'prompt\s*=\s*\(',
        source
    )
    
    if match:
        # Extract the multi-line string concatenation
        # Pattern: prompt = ("line1" "line2" ...)
        prompt_match = re.search(
            r'prompt\s*=\s*\((.*?)\)\s*$',
            source,
            re.DOTALL | re.MULTILINE
        )
        if prompt_match:
            # Extract just the string content, removing quotes and concatenation
            template = prompt_match.group(1)
            # Clean up the extracted content
            template = re.sub(r'"\s*"', '', template)  # Remove quote concatenation
            template = re.sub(r'^"', '', template)  # Remove leading quote
            template = re.sub(r'"$', '', template)  # Remove trailing quote
            return normalize_template(template)
    
    # Fallback: try triple-quoted strings
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
    import logging
    from clinicai.adapters.external import question_service_openai
    
    # Agent-01 is MedicalContextAnalyzer
    try:
        analyzer_class = getattr(question_service_openai, 'MedicalContextAnalyzer', None)
        if analyzer_class:
            method = getattr(analyzer_class, 'analyze_condition', None)
            if method:
                source = inspect.getsource(method)
                # Extract English system_prompt (else block)
                # Pattern: else:  # English\n            system_prompt = """..."""
                match = re.search(
                    r'else:\s*#\s*English\s+system_prompt\s*=\s*"""(.*?)"""',
                    source,
                    re.DOTALL
                )
                if not match:
                    # Try without comment
                    match = re.search(
                        r'else:\s+system_prompt\s*=\s*"""You are AGENT-01(.*?)"""',
                        source,
                        re.DOTALL
                    )
                if match:
                    template = match.group(1) if match.group(1).startswith("You are") else "You are AGENT-01" + match.group(1)
                    return normalize_template(template)
                else:
                    logging.warning("Could not extract Agent-01 prompt with regex in extract_redflag_prompt")
    except Exception as e:
        logging.warning(f"Could not extract Agent-01 prompt in extract_redflag_prompt: {e}")
    
    # FIXED: Changed QuestionServiceOpenAI -> OpenAIQuestionService
    # Fallback: try to find in OpenAIQuestionService (though analyze_condition is in MedicalContextAnalyzer)
    try:
        service_class = getattr(question_service_openai, 'OpenAIQuestionService', None)
        if service_class:
            # The service class doesn't have analyze_condition, it's in MedicalContextAnalyzer
            # So this fallback will fail, but we keep it for completeness
            method = getattr(service_class, 'analyze_condition', None)
            if method:
                source = inspect.getsource(method)
                match = re.search(
                    r'else:\s+system_prompt\s*=\s*"""(.*?)"""',
                    source,
                    re.DOTALL
                )
                if match:
                    template = match.group(1)
                    return normalize_template(template)
    except Exception as e:
        logging.warning(f"Fallback extraction failed in extract_redflag_prompt: {e}")
    
    raise ValueError("Could not extract red-flag prompt template from any source")


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
