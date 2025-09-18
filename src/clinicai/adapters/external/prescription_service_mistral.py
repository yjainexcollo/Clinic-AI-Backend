"""
Mistral AI implementation of PrescriptionService for prescription image analysis.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import UploadFile

from clinicai.application.ports.services.prescription_service import PrescriptionService
from clinicai.core.config import get_settings
from clinicai.core.utils.file_utils import create_directory


class MistralPrescriptionService(PrescriptionService):
    """Mistral AI implementation of PrescriptionService."""

    def __init__(self):
        self._settings = get_settings()
        self._logger = logging.getLogger("clinicai")
        
        # Initialize Mistral client
        api_key = self._settings.mistral.api_key or os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            self._logger.warning("[PrescriptionService] MISTRAL_API_KEY not set - prescription analysis will be disabled")
            self._client = None
            return
        
        try:
            # Try to import and initialize Mistral client with different API versions
            try:
                from mistralai import Mistral
                self._client = Mistral(api_key=api_key)
                self._api_version = "new"
                self._logger.info("[PrescriptionService] Initialized with Mistral AI (new API)")
            except Exception as e1:
                self._logger.warning(f"Failed to initialize new Mistral client: {e1}")
                try:
                    from mistralai import MistralClient
                    self._client = MistralClient(api_key=api_key)
                    self._api_version = "legacy"
                    self._logger.info("[PrescriptionService] Initialized with Mistral AI (legacy API)")
                except Exception as e2:
                    self._logger.error(f"Failed to initialize both Mistral clients: {e1}, {e2}")
                    self._client = None
                    self._api_version = None
                    
        except Exception as e:
            self._logger.error(f"[PrescriptionService] Failed to initialize Mistral client: {e}")
            self._client = None
            self._api_version = None

    async def process_prescriptions(
        self, 
        patient_id: str, 
        visit_id: str, 
        files: List[UploadFile]
    ) -> Dict[str, Any]:
        """Process uploaded prescription images and extract structured data."""
        
        if not files:
            raise ValueError("No files provided")
        
        # Check if Mistral client is available
        if self._client is None:
            return {
                "patient_id": patient_id,
                "visit_id": visit_id,
                "medicines": [],
                "tests": [],
                "instructions": [],
                "raw_text": "",
                "processing_status": "disabled",
                "message": "Prescription analysis is disabled - MISTRAL_API_KEY not configured or client initialization failed"
            }
        
        # Create temporary directory for processing
        temp_dir = os.path.join(tempfile.gettempdir(), f"prescription_{patient_id}_{visit_id}")
        create_directory(temp_dir)
        
        image_paths = []
        extracted_texts = []
        processing_errors = []
        
        try:
            # Save uploaded files and extract text
            for i, file in enumerate(files):
                if not file.filename:
                    self._logger.warning(f"File {i+1} has no filename, skipping")
                    continue
                    
                # Validate file type
                if not self._is_valid_image_file(file.filename):
                    error_msg = f"Invalid image file: {file.filename}"
                    self._logger.warning(error_msg)
                    processing_errors.append(error_msg)
                    continue
                
                # Save file temporarily
                image_path = os.path.join(temp_dir, file.filename)
                try:
                    with open(image_path, "wb") as f:
                        content = await file.read()
                        f.write(content)
                    
                    # Check if file was written successfully
                    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                        error_msg = f"Failed to save file: {file.filename}"
                        self._logger.error(error_msg)
                        processing_errors.append(error_msg)
                        continue
                        
                    image_paths.append(image_path)
                    self._logger.info(f"Successfully saved file: {file.filename} ({os.path.getsize(image_path)} bytes)")
                    
                except Exception as e:
                    error_msg = f"Error saving file {file.filename}: {str(e)}"
                    self._logger.error(error_msg)
                    processing_errors.append(error_msg)
                    continue
                
                # Extract text from image
                try:
                    self._logger.info(f"Attempting to extract text from: {file.filename}")
                    text = await self.extract_text_from_image(image_path)
                    if text and text.strip():
                        extracted_texts.append(text)
                        self._logger.info(f"Successfully extracted {len(text)} characters from {file.filename}")
                    else:
                        error_msg = f"No text extracted from {file.filename}"
                        self._logger.warning(error_msg)
                        processing_errors.append(error_msg)
                except Exception as e:
                    error_msg = f"Failed to extract text from {file.filename}: {str(e)}"
                    self._logger.error(error_msg)
                    processing_errors.append(error_msg)
                    continue
            
            if not extracted_texts:
                error_details = {
                    "total_files": len(files),
                    "valid_files": len(image_paths),
                    "successful_extractions": len(extracted_texts),
                    "errors": processing_errors
                }
                return {
                    "patient_id": patient_id,
                    "visit_id": visit_id,
                    "medicines": [],
                    "tests": [],
                    "instructions": [],
                    "raw_text": "",
                    "processing_status": "failed",
                    "message": "No text could be extracted from the uploaded images",
                    "debug_info": error_details
                }
            
            # Combine all extracted text
            combined_text = "\n\n".join(extracted_texts)
            self._logger.info(f"Combined text length: {len(combined_text)} characters")
            
            # Parse prescription data
            parsed_data = await self.parse_prescription_data(combined_text)
            
            return {
                "patient_id": patient_id,
                "visit_id": visit_id,
                "medicines": parsed_data.get("medicines", []),
                "tests": parsed_data.get("tests", []),
                "instructions": parsed_data.get("instructions", []),
                "raw_text": combined_text,
                "processing_status": "success",
                "message": f"Successfully processed {len(extracted_texts)} out of {len(files)} images",
                "debug_info": {
                    "total_files": len(files),
                    "successful_extractions": len(extracted_texts),
                    "errors": processing_errors
                }
            }
            
        except Exception as e:
            self._logger.error(f"Failed to process prescriptions: {e}", exc_info=True)
            return {
                "patient_id": patient_id,
                "visit_id": visit_id,
                "medicines": [],
                "tests": [],
                "instructions": [],
                "raw_text": "",
                "processing_status": "failed",
                "message": f"Failed to process prescriptions: {str(e)}",
                "debug_info": {
                    "error": str(e),
                    "errors": processing_errors
                }
            }
        
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(temp_dir)

    async def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from a single prescription image using Mistral vision."""
        
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "image/jpeg"
            
            self._logger.info(f"Image details - Path: {image_path}, MIME: {mime_type}, Size: {len(image_data)} chars")
            
            # Create the prompt for prescription analysis
            prompt = """
            Analyze this prescription image and extract all text content. 
            Focus on:
            - Medicine names and dosages
            - Test recommendations
            - Instructions for the patient
            - Any other medical information
            
            Return the extracted text exactly as it appears in the image.
            """
            
            # Call Mistral vision API
            response = await asyncio.to_thread(
                self._call_mistral_vision,
                prompt,
                image_data,
                mime_type
            )
            
            self._logger.info(f"Mistral vision response length: {len(response)} characters")
            return response.strip()
            
        except Exception as e:
            self._logger.error(f"Failed to extract text from image {image_path}: {e}", exc_info=True)
            raise

    async def parse_prescription_data(self, raw_text: str) -> Dict[str, Any]:
        """Parse raw prescription text into structured data using Mistral."""
        
        try:
            prompt = f"""
            Analyze the following prescription text and extract structured information:
            
            TEXT:
            {raw_text}
            
            Extract and return a JSON object with the following structure:
            {{
                "medicines": [
                    {{
                        "name": "Medicine name",
                        "dose": "Dosage information",
                        "frequency": "How often to take",
                        "duration": "Duration of treatment"
                    }}
                ],
                "tests": ["List of recommended tests"],
                "instructions": ["List of patient instructions"]
            }}
            
            Guidelines:
            - Extract ALL medicines mentioned with their details
            - Include any lab tests or investigations recommended
            - Include all patient instructions and advice
            - If information is not available, use null or empty strings
            - Return ONLY valid JSON, no additional text
            """
            
            response = await asyncio.to_thread(
                self._call_mistral_chat,
                prompt
            )
            
            # Parse JSON response
            try:
                # Try to extract JSON from response
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    return json.loads(json_str)
                else:
                    # Try to parse the entire response as JSON
                    return json.loads(response)
            except json.JSONDecodeError:
                self._logger.warning("Failed to parse JSON response, using fallback")
                return self._create_fallback_parsing(raw_text)
                
        except Exception as e:
            self._logger.error(f"Failed to parse prescription data: {e}", exc_info=True)
            return self._create_fallback_parsing(raw_text)

    def _call_mistral_vision(self, prompt: str, image_data: str, mime_type: str) -> str:
        """Synchronous call to Mistral vision API."""
        try:
            self._logger.info(f"Calling Mistral vision API with model: {self._settings.mistral.vision_model}")
            
            if self._api_version == "new":
                # New SDK
                response = self._client.chat.complete(
                    model=self._settings.mistral.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=self._settings.mistral.max_tokens,
                    temperature=self._settings.mistral.temperature
                )
                result = response.choices[0].message.content
            else:
                # Legacy SDK
                response = self._client.chat(
                    model=self._settings.mistral.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=self._settings.mistral.max_tokens,
                    temperature=self._settings.mistral.temperature
                )
                result = response.choices[0].message.content
            
            self._logger.info(f"Mistral vision API response: {result[:100]}..." if len(result) > 100 else result)
            return result
            
        except Exception as e:
            self._logger.error(f"Mistral vision API call failed: {e}", exc_info=True)
            raise

    def _call_mistral_chat(self, prompt: str) -> str:
        """Synchronous call to Mistral chat API."""
        try:
            self._logger.info(f"Calling Mistral chat API with model: {self._settings.mistral.chat_model}")
            
            if self._api_version == "new":
                # New SDK
                response = self._client.chat.complete(
                    model=self._settings.mistral.chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical assistant specialized in parsing prescription information. Always respond with valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self._settings.mistral.max_tokens,
                    temperature=self._settings.mistral.temperature
                )
                result = response.choices[0].message.content
            else:
                # Legacy SDK
                response = self._client.chat(
                    model=self._settings.mistral.chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical assistant specialized in parsing prescription information. Always respond with valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self._settings.mistral.max_tokens,
                    temperature=self._settings.mistral.temperature
                )
                result = response.choices[0].message.content
            
            self._logger.info(f"Mistral chat API response: {result[:100]}..." if len(result) > 100 else result)
            return result
            
        except Exception as e:
            self._logger.error(f"Mistral chat API call failed: {e}", exc_info=True)
            raise

    def _is_valid_image_file(self, filename: str) -> bool:
        """Check if the file is a valid image format."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        return Path(filename).suffix.lower() in valid_extensions

    def _cleanup_temp_files(self, temp_dir: str):
        """Clean up temporary files."""
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            self._logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

    def _create_fallback_parsing(self, raw_text: str) -> Dict[str, Any]:
        """Create fallback parsing when AI parsing fails."""
        return {
            "medicines": [],
            "tests": [],
            "instructions": [],
            "raw_text": raw_text,
            "note": "AI parsing failed, raw text provided"
        }