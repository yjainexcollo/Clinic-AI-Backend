"""
Speaker mapping utility for Azure Speech Service.
Maps speaker IDs from diarization to Doctor/Patient labels.
"""

from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def map_speakers_to_doctor_patient(
    structured_dialogue: List[Dict[str, str]],
    speaker_info: Optional[Dict[str, Any]] = None,
    language: str = "en"
) -> List[Dict[str, str]]:
    """
    Map speaker IDs (Speaker 1, Speaker 2) to Doctor/Patient labels.
    
    Uses heuristics to identify which speaker is likely the Doctor vs Patient:
    - Doctor: More questions, medical terminology, instructions, assessments
    - Patient: First-person experiences, symptom descriptions, responses to questions
    
    Args:
        structured_dialogue: List of dialogue turns with speaker labels like {"Speaker 1": "text"}
        speaker_info: Optional speaker metadata from Azure Speech Service
        language: Language code (en/sp)
    
    Returns:
        List of dialogue turns with Doctor/Patient labels
    """
    if not structured_dialogue:
        return []
    
    # Determine labels based on language
    doctor_label = "Doctor"
    patient_label = "Paciente" if language.lower() in ["sp", "es", "es-es", "es-mx", "spanish"] else "Patient"
    
    # Extract speaker IDs and their content
    speaker_content: Dict[str, List[str]] = {}
    speaker_order: List[str] = []
    
    for turn in structured_dialogue:
        if not isinstance(turn, dict) or len(turn) != 1:
            continue
        
        speaker_id = list(turn.keys())[0]
        text = list(turn.values())[0]
        
        if speaker_id not in speaker_content:
            speaker_content[speaker_id] = []
            speaker_order.append(speaker_id)
        
        speaker_content[speaker_id].append(text)
    
    if len(speaker_content) == 0:
        return structured_dialogue
    
    # If only one speaker, assume it's mixed dialogue (shouldn't happen with diarization)
    if len(speaker_content) == 1:
        logger.warning("Only one speaker detected in diarization, cannot map to Doctor/Patient")
        return structured_dialogue
    
    # Score each speaker to determine Doctor vs Patient
    speaker_scores: Dict[str, Dict[str, float]] = {}
    
    for speaker_id, texts in speaker_content.items():
        combined_text = " ".join(texts).lower()
        
        # Doctor indicators
        doctor_score = 0.0
        
        # Questions (strong indicator)
        question_count = sum(1 for text in texts if "?" in text)
        doctor_score += question_count * 2.0
        
        # Medical terminology
        medical_terms = [
            "diagnosis", "prescribe", "examine", "recommend", "treatment",
            "medication", "symptoms", "condition", "test", "results",
            "diagnóstico", "prescribir", "examinar", "recomendar", "tratamiento",
            "medicamento", "síntomas", "condición", "prueba", "resultados"
        ]
        for term in medical_terms:
            if term in combined_text:
                doctor_score += 1.0
        
        # Instructions/commands
        instruction_patterns = [
            "let me", "i'll", "we'll", "can you", "please",
            "déjame", "voy a", "vamos a", "puedes", "por favor"
        ]
        for pattern in instruction_patterns:
            if pattern in combined_text:
                doctor_score += 1.5
        
        # Clinical assessments
        assessment_patterns = [
            "i see", "it appears", "that's a good sign", "i suspect",
            "veo", "parece", "es una buena señal", "sospecho"
        ]
        for pattern in assessment_patterns:
            if pattern in combined_text:
                doctor_score += 1.0
        
        # Patient indicators
        patient_score = 0.0
        
        # First-person experiences
        first_person_patterns = [
            "i have", "i feel", "i've been", "i took", "i went",
            "tengo", "siento", "he estado", "tomé", "fui"
        ]
        for pattern in first_person_patterns:
            if pattern in combined_text:
                patient_score += 1.5
        
        # Symptom descriptions
        symptom_patterns = [
            "it hurts", "it's painful", "it started", "it gets worse",
            "me duele", "es doloroso", "comenzó", "empeora"
        ]
        for pattern in symptom_patterns:
            if pattern in combined_text:
                patient_score += 1.0
        
        # Short responses (likely patient responding to doctor)
        short_responses = sum(1 for text in texts if len(text.split()) <= 5 and any(
            word in text.lower() for word in ["yes", "no", "okay", "sí", "no", "bien"]
        ))
        patient_score += short_responses * 0.5
        
        # Store scores
        speaker_scores[speaker_id] = {
            "doctor": doctor_score,
            "patient": patient_score
        }
    
    # Determine mapping: speaker with higher doctor score = Doctor
    speaker_mapping: Dict[str, str] = {}
    
    if len(speaker_scores) == 2:
        speakers = list(speaker_scores.keys())
        speaker1_id, speaker2_id = speakers[0], speakers[1]
        
        speaker1_doctor_score = speaker_scores[speaker1_id]["doctor"]
        speaker2_doctor_score = speaker_scores[speaker2_id]["doctor"]
        
        if speaker1_doctor_score > speaker2_doctor_score:
            speaker_mapping[speaker1_id] = doctor_label
            speaker_mapping[speaker2_id] = patient_label
        elif speaker2_doctor_score > speaker1_doctor_score:
            speaker_mapping[speaker1_id] = patient_label
            speaker_mapping[speaker2_id] = doctor_label
        else:
            # Tie - use order: first speaker is usually Doctor (greeting)
            if speaker_order.index(speaker1_id) < speaker_order.index(speaker2_id):
                speaker_mapping[speaker1_id] = doctor_label
                speaker_mapping[speaker2_id] = patient_label
            else:
                speaker_mapping[speaker1_id] = patient_label
                speaker_mapping[speaker2_id] = doctor_label
        
        logger.info(f"Speaker mapping: {speaker1_id} -> {speaker_mapping[speaker1_id]} (doctor_score: {speaker1_doctor_score:.1f}), "
                   f"{speaker2_id} -> {speaker_mapping[speaker2_id]} (doctor_score: {speaker2_doctor_score:.1f})")
    else:
        # More than 2 speakers - map first to Doctor, rest to Patient
        logger.warning(f"More than 2 speakers detected ({len(speaker_scores)}), using simple mapping")
        for i, speaker_id in enumerate(speaker_order):
            if i == 0:
                speaker_mapping[speaker_id] = doctor_label
            else:
                speaker_mapping[speaker_id] = patient_label
    
    # Apply mapping to dialogue
    mapped_dialogue: List[Dict[str, str]] = []
    
    for turn in structured_dialogue:
        if not isinstance(turn, dict) or len(turn) != 1:
            mapped_dialogue.append(turn)
            continue
        
        speaker_id = list(turn.keys())[0]
        text = list(turn.values())[0]
        
        # Map speaker ID to Doctor/Patient
        if speaker_id in speaker_mapping:
            mapped_label = speaker_mapping[speaker_id]
            mapped_dialogue.append({mapped_label: text})
        elif speaker_id in ["Doctor", "Patient", "Paciente"]:
            # Already mapped, keep as is
            mapped_dialogue.append(turn)
        else:
            # Unknown speaker, default to Patient
            logger.warning(f"Unknown speaker ID: {speaker_id}, defaulting to {patient_label}")
            mapped_dialogue.append({patient_label: text})
    
    return mapped_dialogue

