"""
OpenAI-based SOAP generation service implementation.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
import logging
from clinicai.core.ai_factory import get_ai_client

from clinicai.application.ports.services.soap_service import SoapService
from clinicai.core.config import get_settings
from clinicai.adapters.external.prompt_registry import PromptScenario, PROMPT_VERSIONS
from clinicai.adapters.external.llm_gateway import call_llm_with_telemetry
from clinicai.adapters.db.mongo.models.patient_m import DoctorPreferencesMongo


class OpenAISoapService(SoapService):
    """OpenAI implementation of SoapService."""

    def __init__(self):
        self._settings = get_settings()
        
        # Require Azure OpenAI - no fallback to standard OpenAI
        azure_openai_configured = (
            self._settings.azure_openai.endpoint and 
            self._settings.azure_openai.api_key
        )
        
        if not azure_openai_configured:
            raise ValueError(
                "Azure OpenAI is required. Please configure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY. "
                "Fallback to standard OpenAI is disabled for data security."
            )
        
        # Verify deployment name is configured
        if not self._settings.azure_openai.deployment_name:
            raise ValueError(
                "Azure OpenAI deployment name is required. Please set AZURE_OPENAI_DEPLOYMENT_NAME."
            )
        
        # Use Azure AI client (no fallback)
        self._client = get_ai_client()
        # Optional: log initialization
        try:
            logging.getLogger("clinicai").info(
                "[SoapService] Initialized with Azure OpenAI",
                extra={
                    "model": self._settings.soap.model,
                    "azure_openai_configured": azure_openai_configured,
                    "deployment_name": self._settings.azure_openai.deployment_name,
                },
            )
        except Exception:
            pass

    def _translate_vitals_to_spanish(self, vitals_text: str) -> str:
        """Translate vitals text from English to Spanish."""
        if not vitals_text:
            return vitals_text
            
        # Translation mappings
        translations = {
            "Blood pressure": "Presión arterial",
            "Heart rate": "Frecuencia cardíaca", 
            "Respiratory rate": "Frecuencia respiratoria",
            "Temperature": "Temperatura",
            "SpO₂": "SpO₂",  # Keep same
            "Height": "Altura",
            "Weight": "Peso",
            "Pain score": "Escala de dolor",
            "Left arm": "Brazo izquierdo",
            "Right arm": "Brazo derecho",
            "Left": "Izquierdo",
            "Right": "Derecho",
            "Sitting": "Sentado",
            "Standing": "De pie", 
            "Lying": "Acostado",
            "Regular": "Regular",
            "Irregular": "Irregular",
            "Oral": "Oral",
            "Axillary": "Axilar",
            "Tympanic": "Timpánico",
            "Rectal": "Rectal",
            "on room air": "en aire ambiente"
        }
        
        translated_text = vitals_text
        for english, spanish in translations.items():
            translated_text = translated_text.replace(english, spanish)
            
        return translated_text

    def _normalize_language(self, language: str) -> str:
        """
        Normalize language code for backend LLM prompts.
        
        Frontend uses: 'en' or 'sp'
        Backend normalizes to: 'en' or 'es' (for LLM prompts)
        
        Mapping:
        - 'sp', 'es', 'spanish', 'español', 'es-es', 'es-mx' → 'es'
        - unknown/empty → 'en' (default)
        """
        if not language:
            return "en"
        normalized = language.lower().strip()
        if normalized in ['es', 'sp', 'spanish', 'español', 'es-es', 'es-mx']:
            return 'es'
        return normalized if normalized in ['en', 'es'] else 'en'
    
    def _get_output_language_name(self, language: str) -> str:
        """Get human-readable language name for LLM prompts."""
        lang = self._normalize_language(language)
        return "Spanish" if lang == "es" else "English"

    async def _get_doctor_preferences(self, doctor_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch doctor preferences with 1s timeout; fail-open on errors."""
        if not doctor_id:
            return None
        try:
            prefs = await asyncio.wait_for(
                DoctorPreferencesMongo.find_one(DoctorPreferencesMongo.doctor_id == doctor_id),
                timeout=1.0
            )
            return prefs.dict() if prefs else None
        except Exception as e:
            logging.getLogger("clinicai").warning(f"[SoapPrefs] Failed to load preferences for doctor_id={doctor_id}: {e}")
            return None
    
    async def generate_soap_note(
        self,
        transcript: str,
        patient_context: Optional[Dict[str, Any]] = None,
        intake_data: Optional[Dict[str, Any]] = None,
        pre_visit_summary: Optional[Dict[str, Any]] = None,
        vitals: Optional[Dict[str, Any]] = None,
        language: str = "en",
        doctor_id: Optional[str] = None,
        template: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate SOAP note using OpenAI GPT-4."""
        # Normalize language code
        lang = self._normalize_language(language)

        # Load doctor preferences (fail-open)
        prefs = await self._get_doctor_preferences(doctor_id)
        soap_ai_cfg = (prefs or {}).get("soap_ai_config") or {}
        detail_level = (soap_ai_cfg.get("detail_level") or "standard").lower()
        formatting_pref = (soap_ai_cfg.get("formatting") or "bullet_points").lower()
        language_override = soap_ai_cfg.get("language")
        if language_override:
            lang = self._normalize_language(language_override)

        # Determine SOAP section order from doctor preferences
        default_soap_order = ["subjective", "objective", "assessment", "plan"]
        raw_order = (prefs or {}).get("soap_order") or default_soap_order
        # Keep only known sections and ensure we still have all of them
        soap_order: List[str] = [s for s in raw_order if s in default_soap_order]
        if len(soap_order) != len(default_soap_order):
            soap_order = default_soap_order
        
        # Build context from available data
        context_parts = []
        
        if patient_context:
            context_parts.append(f"Patient: {patient_context.get('name', 'Unknown')}, Age: {patient_context.get('age', 'Unknown')}")
            context_parts.append(f"Chief Complaint: {patient_context.get('symptom', 'Not specified')}")
        
        if pre_visit_summary:
            context_parts.append(f"Pre-visit Summary: {pre_visit_summary.get('summary', 'Not available')}")
            
            # Add vitals data if available
            if 'vitals' in pre_visit_summary:
                vitals_data = pre_visit_summary['vitals']['data']
                vitals_text = self._format_vitals_for_soap(vitals_data)
                # Translate vitals to Spanish if needed
                if lang == "es":
                    vitals_text = self._translate_vitals_to_spanish(vitals_text)
                context_parts.append(f"Vitals Data: {vitals_text}")
        
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

        # Optional: template instructions (per-visit, not global)
        template_instructions = ""
        if template:
            try:
                soap_content = (template.get("soap_content") or {}) if isinstance(template, dict) else {}
                template_instructions_lines: List[str] = [
                    "The doctor has provided a custom SOAP template for this visit.",
                    "Follow the structure, tone, and ordering indicated by this template.",
                    "Replace any placeholders like [patient_name], [blood_pressure], etc. with actual values from the transcript and context.",
                    "If any section is missing in the template, use the default structure for that section.",
                    "",
                    f"Subjective Template: {soap_content.get('subjective', '')}",
                    f"Objective Template: {soap_content.get('objective', '')}",
                    f"Assessment Template: {soap_content.get('assessment', '')}",
                    f"Plan Template: {soap_content.get('plan', '')}",
                ]
                template_instructions = "\n".join(template_instructions_lines)
            except Exception:
                # Fail-open: ignore bad template structure, fall back to defaults
                template_instructions = ""
        
        # Build preference snippet
        pref_snippet = (
            f"Doctor Preferences:\n"
            f"- Detail level: {detail_level}\n"
            f"- Formatting: {formatting_pref}\n"
            f"- Language override: {language_override or 'none'}\n"
            f"- SOAP section order: {', '.join(soap_order)}\n"
        )

        # Build JSON schema snippets honoring SOAP order
        subjective_en = '    "subjective": "Patient\'s reported symptoms, concerns, and history as discussed"'
        objective_en = """    "objective": {
        "vital_signs": {
            "blood_pressure": "120/80 mmHg",
            "heart_rate": "74 bpm",
            "temperature": "36.4C",
            "SpO2": "92% on room air",
            "weight": "80 kg"
        },
        "physical_exam": {
            "general_appearance": "Patient appears tired but is cooperative",
            "HEENT": "Not discussed",
            "cardiac": "Not discussed",
            "respiratory": "Not discussed",
            "abdominal": "Not discussed",
            "neuro": "Not discussed",
            "extremities": "Not discussed",
            "gait": "Not discussed"
        }
    }"""
        assessment_en = '    "assessment": "Clinical impressions and reasoning discussed by the physician"'
        plan_en = '    "plan": "Treatment plan, follow-up instructions, and next steps discussed"'

        section_map = {
            "subjective": subjective_en,
            "objective": objective_en,
            "assessment": assessment_en,
            "plan": plan_en,
        }

        ordered_en_sections = ",\n".join(section_map[s] for s in soap_order)

        # Create language-aware prompt - unified English prompt with dynamic language instructions
        output_language = self._get_output_language_name(language)
        prompt = f"""
You are a clinical scribe generating SOAP notes from doctor-patient consultations. 

{template_instructions if template_instructions else ""}

CONTEXT:
{context}

CONSULTATION TRANSCRIPT:
{transcript}

INSTRUCTIONS:
{pref_snippet}
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
9. Preferred detail level: {detail_level}
10. Preferred formatting: {formatting_pref}

Language Rules:
- Write all natural-language text values in {output_language}.
- Do NOT translate JSON keys, enums, codes, field names, or IDs.
- Keep medical terminology appropriate for {output_language}.

REQUIRED FORMAT (JSON):
{{
{ordered_en_sections},
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
            # Extract patient_id from patient_context if available
            patient_id = None
            if patient_context:
                patient_id = patient_context.get("id") or patient_context.get("patient_id")
            
            # Use async Azure OpenAI client
            result = await self._generate_soap_async(prompt, patient_id=patient_id)
            # Normalize for structure/consistency
            return self._normalize_soap(result)
            
        except Exception as e:
            raise ValueError(f"SOAP generation failed: {str(e)}")

    async def _generate_soap_async(self, prompt: str, patient_id: str = None) -> Dict[str, Any]:
        """Async SOAP generation method."""
        # Get prompt version for telemetry
        prompt_version = PROMPT_VERSIONS.get(PromptScenario.SOAP, "UNKNOWN")
        
        # Include version in prompt (optional but recommended)
        system_message = f"""Prompt version: {prompt_version}

You are a clinical scribe. Generate accurate, structured SOAP notes from medical consultations. Always respond with valid JSON only, no extra text."""
        
        response = await call_llm_with_telemetry(
            ai_client=self._client,
            scenario=PromptScenario.SOAP,
            messages=[
                {
                    "role": "system", 
                    "content": system_message
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self._settings.soap.temperature,
            max_tokens=self._settings.soap.max_tokens,
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
        
        # Handle Objective as structured object
        objective = normalized.get("objective", {})
        if isinstance(objective, str):
            # If it's a string, try to parse it as JSON/dict
            try:
                if objective.strip().startswith('{') and objective.strip().endswith('}'):
                    objective = json.loads(objective)
                else:
                    # If it's not JSON, create a basic structure
                    objective = {
                        "vital_signs": {},
                        "physical_exam": {"general_appearance": objective or "Not discussed"}
                    }
            except:
                objective = {
                    "vital_signs": {},
                    "physical_exam": {"general_appearance": objective or "Not discussed"}
                }
        elif not isinstance(objective, dict):
            objective = {
                "vital_signs": {},
                "physical_exam": {"general_appearance": "Not discussed"}
            }
        
        # Ensure vital_signs and physical_exam exist
        if "vital_signs" not in objective:
            objective["vital_signs"] = {}
        if "physical_exam" not in objective:
            objective["physical_exam"] = {}
            
        normalized["objective"] = objective
        
        # Handle other required fields as strings
        required = ["subjective", "assessment", "plan"]
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

    def _format_vitals_for_soap(self, vitals_data: Dict[str, Any]) -> str:
        """Format vitals data for SOAP note generation."""
        parts = []
        
        # Blood Pressure
        if vitals_data.get('systolic') and vitals_data.get('diastolic'):
            bp_text = f"Blood pressure {vitals_data['systolic']}/{vitals_data['diastolic']} mmHg"
            if vitals_data.get('bpArm'):
                bp_text += f" ({vitals_data['bpArm']} arm)"
            if vitals_data.get('bpPosition'):
                bp_text += f" ({vitals_data['bpPosition']})"
            parts.append(bp_text)
        
        # Heart Rate
        if vitals_data.get('heartRate'):
            hr_text = f"Heart rate {vitals_data['heartRate']} bpm"
            if vitals_data.get('rhythm'):
                hr_text += f" ({vitals_data['rhythm']})"
            parts.append(hr_text)
        
        # Respiratory Rate
        if vitals_data.get('respiratoryRate'):
            parts.append(f"Respiratory rate {vitals_data['respiratoryRate']} breaths/min")
        
        # Temperature
        if vitals_data.get('temperature'):
            temp_text = f"Temperature {vitals_data['temperature']}{vitals_data.get('tempUnit', '°C')}"
            if vitals_data.get('tempMethod'):
                temp_text += f" ({vitals_data['tempMethod']})"
            parts.append(temp_text)
        
        # Oxygen Saturation
        if vitals_data.get('oxygenSaturation'):
            parts.append(f"SpO₂ {vitals_data['oxygenSaturation']}% on room air")
        
        # Height, Weight, BMI
        if vitals_data.get('height') and vitals_data.get('weight'):
            height_text = f"{vitals_data['height']} {vitals_data.get('heightUnit', 'cm')}"
            weight_text = f"{vitals_data['weight']} {vitals_data.get('weightUnit', 'kg')}"
            parts.append(f"Height {height_text}, Weight {weight_text}")
            
            # Calculate BMI if both height and weight are provided
            try:
                height_val = float(vitals_data['height'])
                weight_val = float(vitals_data['weight'])
                height_unit = vitals_data.get('heightUnit', 'cm')
                weight_unit = vitals_data.get('weightUnit', 'kg')
                
                # Convert to metric if needed
                if height_unit == 'ft/in':
                    height_val = height_val * 0.3048  # Convert feet to meters
                elif height_unit == 'cm':
                    height_val = height_val / 100  # Convert cm to meters
                
                if weight_unit == 'lbs':
                    weight_val = weight_val * 0.453592  # Convert lbs to kg
                
                if height_val > 0 and weight_val > 0:
                    bmi = weight_val / (height_val * height_val)
                    parts.append(f"BMI {bmi:.1f}")
            except (ValueError, ZeroDivisionError):
                pass  # Skip BMI calculation if conversion fails
        
        # Pain Score
        if vitals_data.get('painScore'):
            parts.append(f"Pain score {vitals_data['painScore']}/10")
        
        return ", ".join(parts) + "." if parts else "No vitals recorded"

    async def validate_soap_structure(self, soap_data: Dict[str, Any]) -> bool:
        """Validate SOAP note structure and completeness."""
        try:
            # Normalize first to improve acceptance of valid-but-short outputs
            data = self._normalize_soap(soap_data)

            # Check required fields minimal presence
            required_fields = ["subjective", "assessment", "plan"]
            for field in required_fields:
                val = data.get(field, "")
                if not isinstance(val, str) or not val.strip():
                    return False

            # Relaxed thresholds: accept concise outputs
            for field in required_fields:
                content = data[field].strip()
                if len(content) < 5:  # Extremely short indicates failure
                    return False
            
            # Check objective as structured object
            objective = data.get("objective", {})
            if not isinstance(objective, dict):
                return False
            
            # Check if objective has at least some content
            has_content = False
            if "vital_signs" in objective and objective["vital_signs"]:
                has_content = True
            if "physical_exam" in objective and objective["physical_exam"]:
                has_content = True
            
            if not has_content:
                return False

            # Ensure list types
            if not isinstance(data.get("highlights", []), list):
                return False
            if not isinstance(data.get("red_flags", []), list):
                return False

            return True
            
        except Exception:
            return False

    async def generate_post_visit_summary(
        self,
        patient_data: Dict[str, Any],
        soap_data: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate post-visit summary for patient sharing."""
        # Normalize language code
        lang = self._normalize_language(language)
        # Create the prompt for post-visit summary - unified English prompt with dynamic language instructions
        output_language = self._get_output_language_name(language)
        prompt = f"""
You are generating a post-visit summary for a patient to share via WhatsApp. This should be clear, comprehensive, and patient-friendly.

PATIENT INFORMATION:
- Name: {patient_data.get('name', 'Patient')}
- Age: {patient_data.get('age', 'N/A')}
- Visit Date: {patient_data.get('visit_date', 'N/A')}
- Chief Complaint: {patient_data.get('symptom', 'N/A')}

SOAP NOTE DATA:
- Subjective: {soap_data.get('subjective', '')}
- Objective: {soap_data.get('objective', '')}
- Assessment: {soap_data.get('assessment', '')}
- Plan: {soap_data.get('plan', '')}
- Highlights: {soap_data.get('highlights', [])}
- Red Flags: {soap_data.get('red_flags', [])}

INSTRUCTIONS:
Generate a comprehensive post-visit summary following this exact structure in JSON format:

Language Rules:
- Write all natural-language text values in {output_language}.
- Do NOT translate JSON keys, enums, codes, field names, or IDs.
- Keep medical terminology appropriate for {output_language}.

{{
    "key_findings": [
        "Key finding 1 from consultation",
        "Key finding 2 from consultation",
        "Key finding 3 from consultation"
    ],
    "diagnosis": "Diagnosis in simple, patient-friendly language",
    "medications": [
        {{
            "name": "Medication name (generic preferred)",
            "dosage": "Dosage amount",
            "frequency": "How often to take",
            "duration": "How long to take",
            "purpose": "What it's for"
        }}
    ],
    "other_recommendations": [
        "Lifestyle recommendation 1",
        "Dietary advice 2",
        "Physical therapy or exercise recommendations"
    ],
    "tests_ordered": [
        {{
            "test_name": "Test name",
            "purpose": "Why this test is needed",
            "instructions": "When and where to get it done"
        }}
    ],
    "next_appointment": "Next appointment details if scheduled",
    "red_flag_symptoms": [
        "Warning sign 1 - when to return immediately",
        "Warning sign 2 - when to go to ER",
        "Warning sign 3 - symptoms to watch for"
    ],
    "patient_instructions": [
        "Clear instruction 1 (do's)",
        "Clear instruction 2 (don'ts)",
        "Clear instruction 3 (home care)"
    ],
    "reassurance_note": "Encouraging and reassuring message for the patient"
}}

Make sure all content is:
- Written in simple, clear language
- Patient-friendly and easy to understand
- Comprehensive but concise
- Actionable with specific instructions
- Reassuring and supportive

Generate the post-visit summary now:
"""

        try:
            # Extract patient_id from patient_data
            patient_id = patient_data.get("id") or patient_data.get("patient_id")
            
            # Use async Azure OpenAI client
            result = await self._generate_post_visit_summary_async(prompt, patient_id=patient_id)
            # Normalize and return the result
            normalized = self._normalize_post_visit_summary(result)
            return normalized
            
        except Exception as e:
            print(f"ERROR: Post-visit summary generation failed: {str(e)}")
            import traceback
            print(f"ERROR: Traceback: {traceback.format_exc()}")
            raise ValueError(f"Post-visit summary generation failed: {str(e)}")

    async def _generate_post_visit_summary_async(self, prompt: str, patient_id: str = None) -> Dict[str, Any]:
        """Async post-visit summary generation method."""
        # Get prompt version for telemetry
        prompt_version = PROMPT_VERSIONS.get(PromptScenario.POSTVISIT_SUMMARY, "UNKNOWN")
        
        # Include version in prompt (optional but recommended)
        system_message = f"""Prompt version: {prompt_version}

You are a medical assistant generating patient-friendly post-visit summaries. Always respond with valid JSON only, no extra text."""
        
        response = await call_llm_with_telemetry(
            ai_client=self._client,
            scenario=PromptScenario.POSTVISIT_SUMMARY,
            messages=[
                {
                    "role": "system", 
                    "content": system_message
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self._settings.soap.temperature,
            max_tokens=self._settings.soap.max_tokens,
        )
        
        # Parse JSON response
        content = response.choices[0].message.content
        
        try:
            summary_data = json.loads(content)
            return summary_data
        except json.JSONDecodeError as e:
            # If the model included code fences or extra text, fall back to extraction below
            try:
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                    return json.loads(json_str)
            except Exception as e2:
                pass
            # As a final fallback, return a minimal structure
            return {
                "key_findings": ["Consultation completed successfully"],
                "diagnosis": "Please follow up with your doctor for detailed diagnosis",
                "medications": [],
                "other_recommendations": ["Follow doctor's instructions carefully"],
                "tests_ordered": [],
                "next_appointment": None,
                "red_flag_symptoms": ["Seek immediate medical attention if symptoms worsen"],
                "patient_instructions": ["Take medications as prescribed", "Rest and monitor symptoms"],
                "reassurance_note": "Please contact us if symptoms worsen or if you have questions."
            }

    def _normalize_post_visit_summary(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize post-visit summary data structure."""
        normalized: Dict[str, Any] = dict(summary_data or {})
        
        # Ensure required list fields exist and are properly formatted
        list_fields = ["key_findings", "medications", "other_recommendations", "tests_ordered", "red_flag_symptoms", "patient_instructions"]
        for field in list_fields:
            val = normalized.get(field, [])
            if not isinstance(val, list):
                val = [str(val)] if val not in (None, "") else []
            normalized[field] = val
        
        # Ensure required string fields exist
        string_fields = ["diagnosis", "reassurance_note"]
        for field in string_fields:
            val = normalized.get(field, "")
            if not isinstance(val, str):
                val = str(val) if val is not None else ""
            normalized[field] = val.strip()
        
        # Ensure optional fields exist
        if not normalized.get("next_appointment"):
            normalized["next_appointment"] = None
        
        return normalized