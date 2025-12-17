"""
Shared constants for Clinic-AI application.
"""

# Topics allowed across agents (kept as list for ordering; treat as set where needed)
ALLOWED_TOPICS = [
    "duration",
    "associated_symptoms",
    
    "current_medications",
    "pain_assessment",
    "pain_characterization",
    "location_radiation",
    "travel_history",
    "chronic_monitoring",
    "lab_tests",
    "screening",
    "allergies",
    "past_medical_history",
    "lifestyle_functional_impact",
    "menstrual_cycle",
    "functional_impact",
    "daily_impact",
    "lifestyle_factors",
    "triggers",
    "aggravating_factors",
    "temporal",
    "progression",
    "frequency",
    "past_evaluation",
    "other",
    "exploratory",
    "family_history",
]

# Travel-related keywords used to detect travel questions / safety
TRAVEL_KEYWORDS = [
    "travel",
    "traveled",
    "traveled",
    "trip",
    "foreign",
    "abroad",
    "viaje",
    "viajado",
    "extranjero",
]

# Menstrual-related keywords used across agents and safety validator
MENSTRUAL_KEYWORDS = [
    "menstrual",
    "period",
    "menses",
    "cycle",
    "menstruation",
    "pms",
    "menstruo",
    "periodo",
    "menstruación",
    "regla",
    "ciclo menstrual",
]

# High-risk complaint keywords (for PMH/allergy gating & safety)
HIGH_RISK_COMPLAINT_KEYWORDS = [
    "chest pain",
    "pressure in chest",
    "tightness in chest",
    "shortness of breath",
    "difficulty breathing",
    "trouble breathing",
    "anaphylaxis",
    "allergic reaction",
    "severe allergy",
    "swelling of face",
    "swelling of tongue",
    "passing out",
    "fainting",
    "loss of consciousness",
    "dolor en el pecho",
    "falta de aire",
    "dificultad para respirar",
    "reacción alérgica",
]

# Stopwords for similarity checks (EN + ES, small & pragmatic)
SIMILARITY_STOPWORDS = {
    "how", "what", "when", "why", "where", "which",
    "is", "are", "do", "does", "did",
    "the", "a", "an", "in", "on", "of", "to", "for",
    "you", "your",
    "y", "que", "qué", "el", "la", "los", "las", "un", "una", "en", "de",
    "por", "para", "cómo", "cuándo",
}
