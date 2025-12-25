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
    
    IMPORTANT: Only tracks hardcoded system prompts, not dynamic runtime values.
    """
    if not template:
        return ""
    
    normalized = template
    
    # STEP 1: Replace complex f-string expressions with nested conditionals
    # Pattern: f"{f'...' if condition else ''}" -> {DYNAMIC_VAR}
    normalized = re.sub(
        r'f"\{f\'[^\']*?\'[^}]*?\}"',
        '{DYNAMIC_VAR}',
        normalized
    )
    
    # STEP 2: Replace method calls and complex expressions
    # Pattern: {self.method(...)} or {obj.method(...)} -> {DYNAMIC_VAR}
    normalized = re.sub(
        r'\{[^}]+\._[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\}',  # {self._method(...)}
        '{DYNAMIC_VAR}',
        normalized
    )
    normalized = re.sub(
        r'\{[^}]+\.format\([^)]*\)\}',  # {obj.format(...)}
        '{DYNAMIC_VAR}',
        normalized
    )
    
    # STEP 3: Replace dict.get() calls
    # Pattern: {dict.get('key', 'default')} -> {DYNAMIC_DICT}
    normalized = re.sub(
        r'\{[^}]+\.get\([^)]+\)\}',  # {dict.get('key', 'default')}
        '{DYNAMIC_DICT}',
        normalized
    )
    
    # STEP 4: Replace dict indexing
    # Pattern: {dict['key']} or {dict["key"]} -> {DYNAMIC_DICT}
    normalized = re.sub(
        r'\{[^}]+\[[^\]]+\]\}',  # {dict['key']} or {dict["key"]}
        '{DYNAMIC_DICT}',
        normalized
    )
    
    # STEP 5: Replace simple f-string variables (but preserve our placeholders)
    # Pattern: f"{variable}" -> {DYNAMIC_VAR} (but only if not already a placeholder)
    normalized = re.sub(
        r'f"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"',
        lambda m: '{DYNAMIC_VAR}' if m.group(1) not in ['DYNAMIC_DICT', 'DYNAMIC_VAR', 'VERSION'] else m.group(0),
        normalized
    )
    
    # STEP 6: Replace simple variables (not in f-strings)
    # Pattern: {variable} -> {DYNAMIC_VAR} (but preserve our placeholders)
    normalized = re.sub(
        r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}',  # {variable} - but not {DYNAMIC_DICT} etc.
        lambda m: '{DYNAMIC_VAR}' if m.group(1) not in ['DYNAMIC_DICT', 'DYNAMIC_VAR', 'VERSION'] else m.group(0),
        normalized
    )
    
    # STEP 7: Remove version strings (they change with each version)
    normalized = re.sub(
        r'Prompt version:\s*\{?[A-Z0-9_\-]+\}?',
        'Prompt version: {VERSION}',
        normalized,
        flags=re.IGNORECASE
    )
    
    # STEP 8: Normalize whitespace (preserve intentional line breaks)
    normalized = re.sub(r'[ \t]+', ' ', normalized)  # Multiple spaces/tabs -> single space
    normalized = re.sub(r' *\n *', '\n', normalized)  # Clean up line breaks
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)  # Max 2 consecutive newlines
    normalized = normalized.strip()
    
    return normalized


def extract_soap_prompt() -> str:
    """Extract SOAP prompt template from soap_service_openai.py (ENGLISH VERSION ONLY)."""
    import inspect
    from clinicai.adapters.external import soap_service_openai
    
    method = getattr(soap_service_openai.OpenAISoapService, 'generate_soap_note', None)
    if not method:
        raise ValueError("Could not find generate_soap_note method")
    
    source = inspect.getsource(method)
    
    # Find the else block (English version)
    else_start = source.find('else:')
    if else_start == -1:
        raise ValueError("Could not find else block (English prompt) in SOAP method")
    
    # Extract English prompt (else block) - handle f-strings
    # Pattern matches: else:\n            prompt = f"""..."""
    match = re.search(
        r'else:\s+prompt\s*=\s*f?"""(.*?)"""',
        source[else_start:],
        re.DOTALL
    )
    
    if not match:
        # Try without 'else:' prefix (fallback)
        match = re.search(
            r'prompt\s*=\s*f?"""You are a clinical scribe(.*?)"""',
            source,
            re.DOTALL
        )
    
    if not match:
        raise ValueError("Could not extract SOAP prompt template")
    
    template = match.group(1)
    
    # Validate it's English (not Spanish)
    if 'Eres un escribano' in template or 'Genera una nota SOAP' in template:
        raise ValueError("Extracted Spanish prompt instead of English! Check extraction logic.")
    
    if 'You are a clinical scribe' not in template:
        raise ValueError("Extracted prompt does not contain English marker! Check extraction logic.")
    
    # CRITICAL: Replace {ordered_en_sections} with {DYNAMIC_VAR} BEFORE normalization
    # This variable contains the JSON schema which is dynamically built
    template = template.replace('{ordered_en_sections}', '{DYNAMIC_VAR}')
    
    return normalize_template(template)


def extract_postvisit_prompt() -> str:
    """Extract post-visit summary prompt template (ENGLISH VERSION ONLY)."""
    import inspect
    from clinicai.adapters.external import soap_service_openai
    
    method = getattr(soap_service_openai.OpenAISoapService, 'generate_post_visit_summary', None)
    if not method:
        raise ValueError("Could not find generate_post_visit_summary method")
    
    source = inspect.getsource(method)
    
    # Find the else block (English version)
    else_start = source.find('else:')
    if else_start == -1:
        raise ValueError("Could not find else block (English prompt) in post-visit method")
    
    # Extract English prompt from else block
    match = re.search(
        r'else:\s+prompt\s*=\s*f?"""(.*?)"""',
        source[else_start:],
        re.DOTALL
    )
    
    if not match:
        # Fallback: try without 'else:' prefix
        match = re.search(
            r'prompt\s*=\s*f?"""You are generating a post-visit summary(.*?)"""',
            source,
            re.DOTALL
        )
    
    if not match:
        raise ValueError("Could not extract post-visit prompt template")
    
    template = match.group(1)
    
    # Validate it's English (not Spanish)
    if 'Estás generando un resumen post-consulta' in template or 'Genera un resumen post-consulta' in template:
        raise ValueError("Extracted Spanish prompt instead of English! Check extraction logic.")
    
    if 'You are generating a post-visit summary' not in template:
        raise ValueError("Extracted prompt does not contain English marker! Check extraction logic.")
    
    return normalize_template(template)


def extract_intake_prompt() -> str:
    """
    Extract intake prompt template combining all 3 agents.
    
    CHANGE NOTE (2025-12): Updated to extract prompts from all 3 intake agents:
    - Agent-01 (Medical Context Analyzer)
    - Agent-02 (Coverage & Fact Extractor) 
    - Agent-03 (Question Generator)
    
    This ensures that a change to ANY agent will trigger a new INTAKE version.
    
    REQUIRES: All 3 agents must be successfully extracted, or an error is raised.
    This prevents partial templates from being versioned.
    """
    import inspect
    import logging
    from clinicai.adapters.external import question_service_openai
    
    combined_prompts = []
    errors = []
    
    # ============================================================================
    # Extract Agent-01 (Medical Context Analyzer) prompt
    # ============================================================================
    try:
        analyzer_class = getattr(question_service_openai, 'MedicalContextAnalyzer', None)
        if not analyzer_class:
            raise ValueError("MedicalContextAnalyzer class not found")
        
        method = getattr(analyzer_class, 'analyze_condition', None)
        if not method:
            raise ValueError("analyze_condition method not found in MedicalContextAnalyzer")
        
        source = inspect.getsource(method)
        # Extract English system_prompt (directly defined, not in else block)
        # Pattern: system_prompt = """You are AGENT-01...
        match = re.search(
            r'system_prompt\s*=\s*"""You are AGENT-01(.*?)"""',
            source,
            re.DOTALL
        )
        if not match:
            raise ValueError("Could not extract Agent-01 prompt with regex")
        
        agent1_prompt = normalize_template("You are AGENT-01" + match.group(1))
        combined_prompts.append(f"=== AGENT-01: MEDICAL CONTEXT ANALYZER ===\n{agent1_prompt}")
        
    except Exception as e:
        error_msg = f"Failed to extract Agent-01 prompt: {e}"
        errors.append(error_msg)
        logging.error(error_msg)
    
    # ============================================================================
    # Extract Agent-02 (Coverage & Fact Extractor) prompt
    # ============================================================================
    try:
        extractor_class = getattr(question_service_openai, 'AnswerExtractor', None)
        if not extractor_class:
            raise ValueError("AnswerExtractor class not found")
        
        method = getattr(extractor_class, 'extract_covered_information', None)
        if not method:
            raise ValueError("extract_covered_information method not found in AnswerExtractor")
        
        source = inspect.getsource(method)
        # Extract English system_prompt (directly defined, not in else block)
        # Pattern: system_prompt = """You are AGENT-02...
        match = re.search(
            r'system_prompt\s*=\s*"""You are AGENT-02(.*?)"""',
            source,
            re.DOTALL
        )
        if not match:
            raise ValueError("Could not extract Agent-02 prompt with regex")
        
        agent2_prompt = normalize_template("You are AGENT-02" + match.group(1))
        combined_prompts.append(f"=== AGENT-02: COVERAGE & FACT EXTRACTOR ===\n{agent2_prompt}")
        
    except Exception as e:
        error_msg = f"Failed to extract Agent-02 prompt: {e}"
        errors.append(error_msg)
        logging.error(error_msg)
    
    # ============================================================================
    # Extract Agent-03 (Question Generator) prompt
    # ============================================================================
    try:
        generator_class = getattr(question_service_openai, 'QuestionGenerator', None)
        if not generator_class:
            raise ValueError("QuestionGenerator class not found")
        
        # Use generate_question_for_topic method (not generate_question)
        method = getattr(generator_class, 'generate_question_for_topic', None)
        if not method:
            raise ValueError("generate_question_for_topic method not found in QuestionGenerator")
        
        source = inspect.getsource(method)
        # Extract English system_prompt (the main system_prompt assignment)
        # Pattern: system_prompt = f"""You are AGENT-03...
        match = re.search(
            r'system_prompt\s*=\s*f?"""You are AGENT-03(.*?)"""',
            source,
            re.DOTALL
        )
        if not match:
            raise ValueError("Could not extract Agent-03 prompt with regex")
        
        agent3_prompt = normalize_template("You are AGENT-03" + match.group(1))
        combined_prompts.append(f"=== AGENT-03: QUESTION GENERATOR ===\n{agent3_prompt}")
        
    except Exception as e:
        error_msg = f"Failed to extract Agent-03 prompt: {e}"
        errors.append(error_msg)
        logging.error(error_msg)
    
    # ============================================================================
    # Validate that all 3 agents were extracted
    # ============================================================================
    if errors:
        raise ValueError(
            f"Failed to extract all required intake agent prompts. Errors: {'; '.join(errors)}"
        )
    
    if len(combined_prompts) != 3:
        extracted_agents = [p.split('===')[1].strip() for p in combined_prompts]
        raise ValueError(
            f"Expected 3 agent prompts but got {len(combined_prompts)}. "
            f"Extracted agents: {extracted_agents}"
        )
    
    # Join all prompts with separator
    combined_template = "\n\n".join(combined_prompts)
    
    # Return the combined, normalized template
    return combined_template


def extract_previsit_prompt() -> str:
    """
    Extract pre-visit summary prompt template (ENGLISH VERSION ONLY).
    
    The method has both Spanish (if lang == "sp") and English (else) versions.
    We extract only the English version for version tracking.
    """
    import inspect
    from clinicai.adapters.external import question_service_openai
    
    method = getattr(question_service_openai.OpenAIQuestionService, 'generate_pre_visit_summary', None)
    if not method:
        raise ValueError("Could not find generate_pre_visit_summary method")
    
    source = inspect.getsource(method)
    
    # Find the else block that contains the English prompt
    # The English prompt starts with "Role & Task" (not "Rol y Tarea" which is Spanish)
    # Use regex to extract content between else: prompt = ( and the closing ) before try:
    
    # Pattern: else: followed by prompt = ( then everything until matching closing paren before try:
    # We'll match from "Role & Task" (English marker) to just before the closing paren
    else_start = source.find('else:')
    if else_start == -1:
        raise ValueError("Could not find else block (English prompt) in pre-visit summary method")
    
    # Find "Role & Task" marker (English version starts here, Spanish starts with "Rol y Tarea")
    role_marker_start = source.find('"Role & Task', else_start)
    if role_marker_start == -1:
        raise ValueError("Could not find 'Role & Task' marker in English prompt")
    
    # Find the matching closing parenthesis for prompt = (
    # Start counting from the opening paren in "prompt = (" 
    prompt_assign_start = source.rfind('prompt = (', else_start, role_marker_start)
    if prompt_assign_start == -1:
        raise ValueError("Could not find prompt assignment in else block")
    
    # Count parentheses to find matching closing paren
    paren_start = prompt_assign_start + source[prompt_assign_start:].find('(')
    paren_count = 0
    pos = paren_start
    
    while pos < len(source):
        if source[pos] == '(':
            paren_count += 1
        elif source[pos] == ')':
            paren_count -= 1
            if paren_count == 0:
                # Found matching closing paren
                break
        pos += 1
    
    if paren_count != 0:
        raise ValueError("Could not properly match parentheses in pre-visit prompt")
    
    # Extract the template (from "Role & Task" to just before the closing paren)
    template = source[role_marker_start:pos]
    
    # CRITICAL: Replace ALL f-string dynamic variables with placeholders BEFORE normalizing
    # These are runtime values that shouldn't affect version tracking
    
    # Replace f-string variables: f"{variable}" -> {DYNAMIC_VAR}
    template = re.sub(r'f"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"', '{DYNAMIC_VAR}', template)
    
    # Replace complex f-string expressions with method calls
    template = re.sub(r'f"\{self\.[^}]+\}"', '{DYNAMIC_VAR}', template)
    
    # Replace conditional f-string expressions: f"{f'...' if ... else ''}"
    template = re.sub(r'f"\{f\'[^\']*?\'[^}]*?\}"', '{DYNAMIC_VAR}', template)
    
    # Replace specific known variables (explicit list for safety)
    known_dynamic_vars = [
        'prefs_snippet',
        'section_definitions_text',
        'exclusion_rules_text',
        'headings_text',
        'guidelines_text',
        'example_block',
        'medication_images_info',
    ]
    for var in known_dynamic_vars:
        # Match f"{var}" or just {var} in the template
        template = re.sub(rf'f?"\{{\s*{re.escape(var)}\s*\}}"', '{DYNAMIC_VAR}', template)
        template = re.sub(rf'\{{\s*{re.escape(var)}\s*\}}', '{DYNAMIC_VAR}', template)
    
    # Remove quote concatenation (Python automatically concatenates adjacent string literals)
    # Pattern: "text1" "text2" -> "text1text2" (remove quotes between)
    template = re.sub(r'"\s*\n\s*"', '', template)  # Newline between quotes
    template = re.sub(r'"\s+"', ' ', template)  # Adjacent quotes with space -> space
    template = re.sub(r'"\s*"', '', template)  # Adjacent quotes -> concatenate
    
    # Remove leading/trailing quotes and whitespace
    template = template.strip().strip('"')
    
    # Validate it's English (not Spanish)
    if 'Rol y Tarea' in template or 'Motivo de Consulta:' in template:
        raise ValueError("Extracted Spanish prompt instead of English! Check extraction logic.")
    
    if 'Role & Task' not in template:
        raise ValueError("Extracted prompt does not contain English marker 'Role & Task'! Check extraction logic.")
    
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
