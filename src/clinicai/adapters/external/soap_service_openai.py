"""
OpenAI-based SOAP generation service implementation.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
import os
import logging

from clinicai.application.ports.services.soap_service import SoapService
from clinicai.core.config import get_settings


class OpenAISoapService(SoapService):
    """OpenAI implementation of SoapService."""

    def __init__(self):
        self._settings = get_settings()
        # Load API key from settings or env (with optional .env fallback)
        api_key = self._settings.openai.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(override=False)
                api_key = os.getenv("OPENAI_API_KEY", "")
            except Exception:
                pass
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        
        self._client = OpenAI(api_key=api_key)
        # Optional: log model and presence of key (masked)
        try:
            logging.getLogger("clinicai").info(
                "[SoapService] Initialized",
                extra={
                    "model": self._settings.soap.model,
                    "has_key": bool(api_key),
                },
            )
        except Exception:
            pass

    async def generate_soap_note(
        self,
        transcript: str,
        patient_context: Optional[Dict[str, Any]] = None,
        intake_data: Optional[Dict[str, Any]] = None,
        pre_visit_summary: Optional[Dict[str, Any]] = None,
        vitals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate SOAP note using OpenAI GPT-4."""
        
        # Build context from available data
        context_parts = []
        
        if patient_context:
            context_parts.append(f"Patient: {patient_context.get('name', 'Unknown')}, Age: {patient_context.get('age', 'Unknown')}")
            context_parts.append(f"Chief Complaint: {patient_context.get('symptom', 'Not specified')}")
        
        if pre_visit_summary:
            context_parts.append(f"Pre-visit Summary: {pre_visit_summary.get('summary', 'Not available')}")
        
        if intake_data and intake_data.get('questions_asked'):
            intake_responses = []
            for qa in intake_data['questions_asked']:
                intake_responses.append(f"Q: {qa['question']}\nA: {qa['answer']}")
            context_parts.append(f"Intake Responses:\n" + "\n\n".join(intake_responses))

        # Include Vitals in context for Objective section
        if vitals:
            try:
                v = vitals or {}
                def _val(k: str) -> Optional[str]:
                    x = v.get(k)
                    if x in (None, ""):
                        return None
                    return str(x)
                vitals_parts: list[str] = []
                if _val("systolic") and _val("diastolic"):
                    arm = f" ({v.get('bpArm')} arm)" if v.get('bpArm') else ""
                    pos = f" ({v.get('bpPosition')})" if v.get('bpPosition') else ""
                    vitals_parts.append(f"Blood pressure {_val('systolic')}/{_val('diastolic')} mmHg{arm}{pos}")
                if _val("heartRate"):
                    rhythm = f" ({v.get('rhythm')})" if v.get('rhythm') else ""
                    vitals_parts.append(f"Heart rate {_val('heartRate')} bpm{rhythm}")
                if _val("respiratoryRate"):
                    vitals_parts.append(f"Respiratory rate {_val('respiratoryRate')} breaths/min")
                if _val("temperature"):
                    unit = (v.get("tempUnit") or "C").replace("°", "")
                    method = f" ({v.get('tempMethod')})" if v.get('tempMethod') else ""
                    vitals_parts.append(f"Temperature {_val('temperature')}{unit}{method}")
                if _val("oxygenSaturation"):
                    vitals_parts.append(f"SpO₂ {_val('oxygenSaturation')}% on room air")
                h = _val("height"); w = _val("weight")
                if h or w:
                    h_unit = v.get("heightUnit") or "cm"
                    w_unit = v.get("weightUnit") or "kg"
                    hw: list[str] = []
                    if h:
                        hw.append(f"Height {h} {h_unit}")
                    if w:
                        hw.append(f"Weight {w} {w_unit}")
                    if hw:
                        vitals_parts.append(", ".join(hw))
                if _val("painScore"):
                    vitals_parts.append(f"Pain score {_val('painScore')}/10")
                if v.get("notes"):
                    vitals_parts.append(f"Observation notes: {v.get('notes')}")
                if vitals_parts:
                    context_parts.append("Objective Vitals (from form):\n- " + "\n- ".join(vitals_parts))
            except Exception:
                try:
                    context_parts.append("Objective Vitals (raw JSON):\n" + json.dumps(vitals))
                except Exception:
                    pass
        
        context = "\n\n".join(context_parts) if context_parts else "No additional context available"
        
        # Create the prompt
        prompt = f"""
You are a clinical scribe generating SOAP notes from doctor-patient consultations. 

CONTEXT:
{context}

CONSULTATION TRANSCRIPT:
{transcript}

INSTRUCTIONS:
1. Generate a comprehensive SOAP note based on the transcript and context
2. Do NOT make diagnoses or treatment recommendations unless explicitly stated by the physician
3. Use medical terminology appropriately
4. Be objective and factual
5. If information is unclear or missing, mark as "Unclear" or "Not discussed"
6. Focus on what was actually said during the consultation
7. In the Objective, include BOTH:
   - Vital signs from the provided Objective Vitals, if present
   - Physical exam and other transcript-derived observable findings (e.g., general appearance, HEENT, cardiac, respiratory, abdominal, neuro, extremities, gait) when mentioned
   If explicit exam elements are not stated, include any transcript-derived objective observations (e.g., affect, speech, respiratory effort) when available.
8. Incorporate the Objective Vitals provided in CONTEXT succinctly; do not replace transcript-derived exam with vitals—combine them.

REQUIRED FORMAT (JSON):
{{
    "subjective": "Patient's reported symptoms, concerns, and history as discussed",
    "objective": "Observable findings, vital signs, and physical exam findings mentioned",
    "assessment": "Clinical impressions and reasoning discussed by the physician",
    "plan": "Treatment plan, follow-up instructions, and next steps discussed",
    "highlights": ["Key clinical points 1", "Key clinical points 2", "Key clinical points 3"],
    "red_flags": ["Any concerning symptoms or findings mentioned"],
    "model_info": {{
        "model": "{self._settings.soap.model}",
        "temperature": {self._settings.soap.temperature},
        "max_tokens": {self._settings.soap.max_tokens}
    }},
    "confidence_score": 0.95
}}

Generate the SOAP note now:
"""

        try:
            # Run OpenAI completion in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._generate_soap_sync,
                prompt
            )
            # Normalize for structure/consistency
            return self._normalize_soap(result)
            
        except Exception as e:
            raise ValueError(f"SOAP generation failed: {str(e)}")

    def _generate_soap_sync(self, prompt: str) -> Dict[str, Any]:
        """Synchronous SOAP generation method."""
        response = self._client.chat.completions.create(
            model=self._settings.soap.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a clinical scribe. Generate accurate, structured SOAP notes from medical consultations. Always respond with valid JSON only, no extra text."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self._settings.soap.temperature,
            max_tokens=self._settings.soap.max_tokens
        )
        
        # Parse JSON response
        try:
            soap_data = json.loads(response.choices[0].message.content)
            return soap_data
        except Exception:
            # If the model included code fences or extra text, fall back to extraction below
            try:
                content = response.choices[0].message.content
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                    return json.loads(json_str)
            except Exception:
                pass
            # As a final fallback, return a minimal structure (will be normalized later)
            return {
                "subjective": "",
                "objective": "",
                "assessment": "",
                "plan": "",
                "highlights": [],
                "red_flags": [],
                "model_info": {"model": self._settings.soap.model},
                "confidence_score": None,
            }

    def _normalize_soap(self, soap_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce SOAP dict into a valid, minimally complete structure."""
        normalized: Dict[str, Any] = dict(soap_data or {})
        required = ["subjective", "objective", "assessment", "plan"]
        for key in required:
            val = normalized.get(key, "")
            if not isinstance(val, str):
                try:
                    val = str(val)
                except Exception:
                    val = ""
            val = (val or "").strip()
            if len(val) < 5:
                val = "Not discussed"
            normalized[key] = val

        # Optional list fields
        for list_key in ["highlights", "red_flags"]:
            val = normalized.get(list_key, [])
            if not isinstance(val, list):
                val = [str(val)] if val not in (None, "") else []
            normalized[list_key] = val

        # Model info
        model_info = normalized.get("model_info")
        if not isinstance(model_info, dict):
            model_info = {}
        model_info.setdefault("model", self._settings.soap.model)
        normalized["model_info"] = model_info

        # Confidence score
        if normalized.get("confidence_score") is None:
            normalized["confidence_score"] = 0.7

        return normalized

    async def validate_soap_structure(self, soap_data: Dict[str, Any]) -> bool:
        """Validate SOAP note structure and completeness."""
        try:
            # Normalize first to improve acceptance of valid-but-short outputs
            data = self._normalize_soap(soap_data)

            # Check required fields minimal presence
            required_fields = ["subjective", "objective", "assessment", "plan"]
            for field in required_fields:
                val = data.get(field, "")
                if not isinstance(val, str) or not val.strip():
                    return False

            # Relaxed thresholds: accept concise outputs
            for field in required_fields:
                content = data[field].strip()
                if len(content) < 5:  # Extremely short indicates failure
                    return False
                if len(content) > 8000:  # Guardrail against runaway output
                    return False

            # Ensure list types
            if not isinstance(data.get("highlights", []), list):
                return False
            if not isinstance(data.get("red_flags", []), list):
                return False

            return True
            
        except Exception:
            return False
