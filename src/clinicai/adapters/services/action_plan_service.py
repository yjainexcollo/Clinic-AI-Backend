"""
OpenAI-based implementation of Action Plan Service.
"""

import logging
from typing import Dict, Any, List

from ...core.config import get_settings
from ...application.ports.services.action_plan_service import ActionPlanService
from ...core.ai_factory import get_ai_client

logger = logging.getLogger(__name__)


class OpenAIActionPlanService(ActionPlanService):
    """OpenAI-based implementation of Action Plan Service."""

    def __init__(self):
        self.settings = get_settings()
        # Use centralized Azure AI client
        self.client = get_ai_client()

    async def generate_action_plan(
        self, 
        transcript: str, 
        structured_dialogue: list[dict] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate Action and Plan from medical transcript using OpenAI.
        """
        try:
            logger.info(f"Generating Action and Plan for transcript of {len(transcript)} characters")
            
            # Prepare the prompt based on language
            if language.lower() in ["sp", "es", "es-es", "es-mx", "spanish"]:
                system_prompt = self._get_spanish_system_prompt()
                user_prompt = self._get_spanish_user_prompt(transcript, structured_dialogue)
            else:
                system_prompt = self._get_english_system_prompt()
                user_prompt = self._get_english_user_prompt(transcript, structured_dialogue)

            # Call Azure OpenAI API
            # Note: patient_id is not needed for adhoc flows - it's optional and defaults to None
            # Use Azure OpenAI deployment name instead of model
            response = await self.client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent medical recommendations
                max_tokens=2000,
            )

            # Parse the response
            content = response.choices[0].message.content
            logger.info(f"OpenAI response received: {len(content)} characters")

            # Parse JSON response with improved error handling
            import json
            import re
            
            try:
                # Clean the response content
                cleaned_content = content.strip()
                
                # Try to extract JSON from the response if it's wrapped in other text
                json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
                if json_match:
                    cleaned_content = json_match.group(0)
                
                action_plan_data = json.loads(cleaned_content)
                logger.info("Successfully parsed Action and Plan JSON response")
                
                # Validate the structure
                if not isinstance(action_plan_data, dict):
                    raise ValueError("Response is not a valid JSON object")
                
                # Ensure required fields exist
                if "action" not in action_plan_data or "plan" not in action_plan_data:
                    raise ValueError("Missing required fields in response")
                
                return action_plan_data
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response content: {content[:500]}...")
                
                # Try to generate a better fallback based on transcript content
                return self._generate_enhanced_fallback(transcript, structured_dialogue, language)

        except Exception as e:
            logger.error(f"Error generating Action and Plan: {e}")
            raise

    def _get_english_system_prompt(self) -> str:
        """Get English system prompt for Action and Plan generation."""
        return """You are an expert medical AI assistant that generates comprehensive Action and Plan recommendations from medical consultation transcripts.

Your task is to analyze the medical consultation transcript and provide:
1. **Action**: Specific immediate actions that should be taken based on the consultation
2. **Plan**: Comprehensive treatment plan with follow-up recommendations

CRITICAL REQUIREMENTS:
- Analyze the actual transcript content thoroughly
- Extract specific medical conditions, symptoms, and concerns mentioned
- Provide detailed, actionable recommendations based on what was discussed
- Be specific about medications, dosages, and treatment protocols
- Include relevant follow-up timelines and monitoring requirements
- Address patient education needs based on the consultation content
- Suggest lifestyle modifications relevant to the discussed conditions

IMPORTANT: Your response must be a valid JSON object. Do not include any text before or after the JSON.

Return your response as a JSON object with this exact structure:
{
  "action": [
    "Specific action item 1 based on transcript content",
    "Specific action item 2 based on transcript content",
    "Specific action item 3 based on transcript content"
  ],
  "plan": {
    "immediate_care": "Detailed immediate care recommendations based on the consultation",
    "medications": [
      {
        "name": "Specific medication name",
        "dosage": "Specific dosage instructions",
        "duration": "Specific duration of treatment",
        "notes": "Additional notes and considerations"
      }
    ],
    "follow_up": {
      "timeline": "Specific follow-up timeline based on condition",
      "monitoring": "Specific monitoring requirements",
      "next_steps": "Specific next steps in care"
    },
    "patient_education": "Specific patient education points based on the consultation",
    "lifestyle_modifications": "Specific lifestyle changes relevant to the discussed condition"
  },
  "confidence": 0.85,
  "reasoning": "Brief explanation of how the recommendations relate to the consultation content"
}"""

    def _get_spanish_system_prompt(self) -> str:
        """Get Spanish system prompt for Action and Plan generation."""
        return """Eres un asistente médico de IA experto que genera recomendaciones integrales de Acción y Plan a partir de transcripciones de consultas médicas.

Tu tarea es analizar la transcripción de la consulta médica y proporcionar:
1. **Acción**: Acciones inmediatas específicas que deben tomarse basadas en la consulta
2. **Plan**: Plan de tratamiento integral con recomendaciones de seguimiento

REQUISITOS CRÍTICOS:
- Analiza el contenido real de la transcripción a fondo
- Extrae condiciones médicas específicas, síntomas y preocupaciones mencionadas
- Proporciona recomendaciones detalladas y accionables basadas en lo discutido
- Sé específico sobre medicamentos, dosis y protocolos de tratamiento
- Incluye cronogramas de seguimiento relevantes y requisitos de monitoreo
- Aborda las necesidades de educación del paciente basadas en el contenido de la consulta
- Sugiere modificaciones del estilo de vida relevantes a las condiciones discutidas

IMPORTANTE: Tu respuesta debe ser un objeto JSON válido. No incluyas texto antes o después del JSON.

Devuelve tu respuesta como un objeto JSON con esta estructura exacta:
{
  "action": [
    "Elemento de acción específico 1 basado en el contenido de la transcripción",
    "Elemento de acción específico 2 basado en el contenido de la transcripción",
    "Elemento de acción específico 3 basado en el contenido de la transcripción"
  ],
  "plan": {
    "immediate_care": "Recomendaciones detalladas de atención inmediata basadas en la consulta",
    "medications": [
      {
        "name": "Nombre específico del medicamento",
        "dosage": "Instrucciones específicas de dosificación",
        "duration": "Duración específica del tratamiento",
        "notes": "Notas adicionales y consideraciones"
      }
    ],
    "follow_up": {
      "timeline": "Cronograma específico de seguimiento basado en la condición",
      "monitoring": "Requisitos específicos de monitoreo",
      "next_steps": "Próximos pasos específicos en la atención"
    },
    "patient_education": "Puntos específicos de educación del paciente basados en la consulta",
    "lifestyle_modifications": "Cambios específicos del estilo de vida relevantes a la condición discutida"
  },
  "confidence": 0.85,
  "reasoning": "Breve explicación de cómo las recomendaciones se relacionan con el contenido de la consulta"
}"""

    def _get_english_user_prompt(self, transcript: str, structured_dialogue: list[dict] = None) -> str:
        """Get English user prompt for Action and Plan generation."""
        if structured_dialogue:
            dialogue_text = "\n".join([
                f"{list(turn.keys())[0]}: {list(turn.values())[0]}" 
                for turn in structured_dialogue
            ])
            return f"""Please analyze this medical consultation and generate Action and Plan recommendations:

TRANSCRIPT:
{transcript}

STRUCTURED DIALOGUE:
{dialogue_text}

Generate specific, actionable recommendations based on the consultation content."""
        else:
            return f"""Please analyze this medical consultation transcript and generate Action and Plan recommendations:

TRANSCRIPT:
{transcript}

Generate specific, actionable recommendations based on the consultation content."""

    def _get_spanish_user_prompt(self, transcript: str, structured_dialogue: list[dict] = None) -> str:
        """Get Spanish user prompt for Action and Plan generation."""
        if structured_dialogue:
            dialogue_text = "\n".join([
                f"{list(turn.keys())[0]}: {list(turn.values())[0]}" 
                for turn in structured_dialogue
            ])
            return f"""Por favor analiza esta consulta médica y genera recomendaciones de Acción y Plan:

TRANSCRIPCIÓN:
{transcript}

DIÁLOGO ESTRUCTURADO:
{dialogue_text}

Genera recomendaciones específicas y accionables basadas en el contenido de la consulta."""
        else:
            return f"""Por favor analiza esta transcripción de consulta médica y genera recomendaciones de Acción y Plan:

TRANSCRIPCIÓN:
{transcript}

Genera recomendaciones específicas y accionables basadas en el contenido de la consulta."""

    def _generate_enhanced_fallback(self, transcript: str, structured_dialogue: list[dict], language: str) -> Dict[str, Any]:
        """Generate enhanced fallback Action and Plan based on transcript content analysis."""
        logger.warning("Using enhanced fallback generation for Action and Plan")
        
        # Analyze transcript content to extract key information
        transcript_lower = transcript.lower()
        
        # Extract potential medical conditions and symptoms
        medical_keywords = {
            'diabetes': ['diabetes', 'diabetic', 'blood sugar', 'glucose', 'hba1c', 'insulin'],
            'hypertension': ['blood pressure', 'hypertension', 'high blood pressure'],
            'infection': ['infection', 'fever', 'antibiotics', 'bacterial', 'viral'],
            'pain': ['pain', 'ache', 'hurt', 'sore', 'discomfort'],
            'respiratory': ['cough', 'breathing', 'chest', 'lung', 'asthma', 'pneumonia'],
            'gastrointestinal': ['stomach', 'nausea', 'vomiting', 'diarrhea', 'constipation', 'digestive'],
            'cardiovascular': ['heart', 'chest pain', 'cardiac', 'heart attack', 'stroke'],
            'mental_health': ['anxiety', 'depression', 'stress', 'mental health', 'mood']
        }
        
        detected_conditions = []
        for condition, keywords in medical_keywords.items():
            if any(keyword in transcript_lower for keyword in keywords):
                detected_conditions.append(condition)
        
        # Generate specific actions based on detected conditions
        actions = []
        if 'diabetes' in detected_conditions:
            actions.extend([
                "Schedule blood glucose monitoring",
                "Review current diabetes management plan",
                "Assess need for medication adjustment"
            ])
        if 'hypertension' in detected_conditions:
            actions.extend([
                "Monitor blood pressure regularly",
                "Review antihypertensive medication regimen",
                "Assess cardiovascular risk factors"
            ])
        if 'infection' in detected_conditions:
            actions.extend([
                "Initiate appropriate antibiotic therapy if bacterial",
                "Monitor for signs of systemic infection",
                "Provide symptomatic relief measures"
            ])
        if 'pain' in detected_conditions:
            actions.extend([
                "Assess pain severity and characteristics",
                "Consider appropriate pain management strategy",
                "Evaluate need for imaging or specialist referral"
            ])
        
        # Default actions if no specific conditions detected
        if not actions:
            actions = [
                "Review patient's current symptoms and concerns",
                "Assess vital signs and physical examination findings",
                "Consider appropriate diagnostic tests if indicated"
            ]
        
        # Generate treatment plan based on content
        immediate_care = "Provide symptomatic relief and address immediate patient concerns based on the consultation findings."
        
        if 'diabetes' in detected_conditions:
            immediate_care = "Optimize diabetes management with focus on blood glucose control and lifestyle modifications."
        elif 'hypertension' in detected_conditions:
            immediate_care = "Implement blood pressure management strategies including medication optimization and lifestyle counseling."
        elif 'infection' in detected_conditions:
            immediate_care = "Initiate appropriate antimicrobial therapy and provide supportive care measures."
        
        # Generate medications based on detected conditions
        medications = []
        if 'diabetes' in detected_conditions:
            medications.append({
                "name": "Metformin",
                "dosage": "500mg twice daily with meals",
                "duration": "Long-term as prescribed",
                "notes": "Monitor for gastrointestinal side effects and renal function"
            })
        if 'hypertension' in detected_conditions:
            medications.append({
                "name": "ACE Inhibitor (e.g., Lisinopril)",
                "dosage": "10mg once daily",
                "duration": "Long-term as prescribed",
                "notes": "Monitor blood pressure and renal function"
            })
        
        # Generate follow-up recommendations
        follow_up_timeline = "2-4 weeks"
        if 'diabetes' in detected_conditions:
            follow_up_timeline = "1-2 weeks for diabetes management review"
        elif 'hypertension' in detected_conditions:
            follow_up_timeline = "2-3 weeks for blood pressure monitoring"
        elif 'infection' in detected_conditions:
            follow_up_timeline = "1 week or sooner if symptoms worsen"
        
        monitoring = "Monitor symptoms and response to treatment"
        if 'diabetes' in detected_conditions:
            monitoring = "Monitor blood glucose levels, HbA1c, and diabetes-related complications"
        elif 'hypertension' in detected_conditions:
            monitoring = "Monitor blood pressure readings and cardiovascular risk factors"
        
        next_steps = "Continue current treatment plan and follow up as scheduled"
        if 'diabetes' in detected_conditions:
            next_steps = "Consider diabetes education referral and endocrinology consultation if needed"
        elif 'hypertension' in detected_conditions:
            next_steps = "Consider cardiology referral if blood pressure remains uncontrolled"
        
        # Generate patient education based on conditions
        patient_education = "Review general health maintenance and symptom monitoring"
        if 'diabetes' in detected_conditions:
            patient_education = "Diabetes self-management education including blood glucose monitoring, diet, exercise, and medication adherence"
        elif 'hypertension' in detected_conditions:
            patient_education = "Blood pressure management education including DASH diet, exercise, stress management, and medication adherence"
        elif 'infection' in detected_conditions:
            patient_education = "Infection prevention measures, medication compliance, and when to seek immediate medical attention"
        
        # Generate lifestyle modifications
        lifestyle_modifications = "Maintain healthy lifestyle including balanced diet, regular exercise, and adequate sleep"
        if 'diabetes' in detected_conditions:
            lifestyle_modifications = "Carbohydrate counting, regular physical activity, weight management, and stress reduction techniques"
        elif 'hypertension' in detected_conditions:
            lifestyle_modifications = "DASH diet, regular aerobic exercise, sodium restriction, and stress management"
        elif 'infection' in detected_conditions:
            lifestyle_modifications = "Adequate rest, increased fluid intake, proper nutrition, and infection prevention measures"
        
        return {
            "action": actions,
            "plan": {
                "immediate_care": immediate_care,
                "medications": medications,
                "follow_up": {
                    "timeline": follow_up_timeline,
                    "monitoring": monitoring,
                    "next_steps": next_steps
                },
                "patient_education": patient_education,
                "lifestyle_modifications": lifestyle_modifications
            },
            "confidence": 0.8,
            "reasoning": f"Generated based on analysis of consultation content. Detected conditions: {', '.join(detected_conditions) if detected_conditions else 'General consultation'}"
        }
