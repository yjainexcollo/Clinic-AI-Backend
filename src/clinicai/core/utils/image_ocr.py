"""Enhanced OCR helper with quality assessment and confidence scoring.

Attempts to use pytesseract if available. If not installed or tesseract
binary is missing on the host, returns an empty string without raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class OCRResult:
    """OCR processing result with quality assessment."""

    text: str
    confidence: float  # 0.0 to 1.0
    quality: str  # 'excellent', 'good', 'poor', 'failed'
    extracted_medications: List[str]
    suggestions: List[str]
    word_count: int
    has_medication_keywords: bool


def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image file, or return empty string on failure."""
    result = extract_text_with_quality(image_path)
    return result.text


def extract_text_with_quality(image_path: str) -> OCRResult:
    """Extract text from an image file with quality assessment."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        img = Image.open(image_path)

        # Get text and confidence data
        text: str = pytesseract.image_to_string(img) or ""
        confidence_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Calculate average confidence
        confidences = [int(conf) for conf in confidence_data["conf"] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

        # Normalize whitespace
        normalized_text = " ".join(text.split()).strip()
        word_count = len(normalized_text.split()) if normalized_text else 0

        # Analyze for medication keywords
        medication_keywords = [
            "mg",
            "ml",
            "tablet",
            "capsule",
            "syrup",
            "drops",
            "injection",
            "twice",
            "daily",
            "morning",
            "evening",
            "night",
            "before",
            "after",
            "food",
            "empty stomach",
            "with water",
            "prescription",
            "dosage",
            "medicine",
            "drug",
            "pharmaceutical",
            "rx",
            "take",
            "apply",
        ]

        text_lower = normalized_text.lower()
        has_medication_keywords = any(keyword in text_lower for keyword in medication_keywords)

        # Extract potential medication names (simple pattern matching)
        extracted_medications = _extract_medication_names(normalized_text)

        # Determine quality based on multiple factors
        quality, suggestions = _assess_ocr_quality(
            normalized_text,
            avg_confidence,
            word_count,
            has_medication_keywords,
            len(extracted_medications),
        )

        return OCRResult(
            text=normalized_text,
            confidence=avg_confidence,
            quality=quality,
            extracted_medications=extracted_medications,
            suggestions=suggestions,
            word_count=word_count,
            has_medication_keywords=has_medication_keywords,
        )

    except Exception as e:
        return OCRResult(
            text="",
            confidence=0.0,
            quality="failed",
            extracted_medications=[],
            suggestions=[
                "Image could not be processed. Please try:",
                "• Ensure the image is clear and well-lit",
                "• Check that text is readable and not blurry",
                "• Try taking the photo from a different angle",
                "• Make sure the image file is not corrupted",
            ],
            word_count=0,
            has_medication_keywords=False,
        )


def _extract_medication_names(text: str) -> List[str]:
    """Extract potential medication names from text."""
    # Simple pattern matching for common medication name patterns
    patterns = [
        r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b",  # CamelCase words
        r"\b[A-Z]{2,}\b",  # All caps words (like brand names)
        r"\b\w+cin\b",  # Common medication suffixes
        r"\b\w+mycin\b",
        r"\b\w+pril\b",
        r"\b\w+sartan\b",
        r"\b\w+statin\b",
    ]

    medications = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        medications.extend(matches)

    # Remove common non-medication words
    exclude_words = {
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "among",
        "under",
        "over",
        "take",
        "with",
        "food",
        "water",
        "morning",
        "evening",
        "night",
        "daily",
        "twice",
        "once",
        "before",
        "after",
        "empty",
        "stomach",
    }

    # Filter out short words and common words
    filtered_medications = [med for med in medications if len(med) > 3 and med.lower() not in exclude_words]

    return list(set(filtered_medications))  # Remove duplicates


def _assess_ocr_quality(
    text: str,
    confidence: float,
    word_count: int,
    has_medication_keywords: bool,
    medication_count: int,
) -> tuple[str, List[str]]:
    """Assess OCR quality and provide suggestions."""

    suggestions = []

    # Check if text is empty or too short
    if not text or word_count < 3:
        return "failed", [
            "No readable text found in the image. Please try:",
            "• Ensure the image is clear and well-lit",
            "• Check that text is readable and not blurry",
            "• Make sure the medication label is visible",
            "• Try taking the photo from a different angle",
        ]

    # Check confidence level
    if confidence < 0.3:
        quality = "poor"
        suggestions.extend(
            [
                "Text quality is poor. Please try:",
                "• Ensure better lighting when taking the photo",
                "• Hold the camera steady to avoid blur",
                "• Make sure the text is in focus",
            ]
        )
    elif confidence < 0.6:
        quality = "good"
        suggestions.extend(
            [
                "Text quality is acceptable but could be better:",
                "• Consider retaking with better lighting",
                "• Ensure the text is fully visible",
            ]
        )
    else:
        quality = "excellent"
        suggestions.append("Text quality is excellent!")

    # Check for medication-specific content
    if not has_medication_keywords:
        if quality == "excellent":
            quality = "good"
        elif quality == "good":
            quality = "poor"

        suggestions.extend(
            [
                "No medication-related keywords detected. Please ensure:",
                "• The image shows a medication label or prescription",
                "• The text includes dosage information (mg, ml, etc.)",
                "• The image contains medication names or instructions",
            ]
        )

    # Check medication count
    if medication_count == 0 and has_medication_keywords:
        suggestions.append("Consider retaking the photo to capture medication names more clearly")
    elif medication_count > 0:
        suggestions.append(f"Found {medication_count} potential medication(s)")

    # Add general tips
    suggestions.extend(
        [
            "",
            "Tips for better OCR results:",
            "• Use good lighting (natural light works best)",
            "• Keep the camera steady",
            "• Ensure text is horizontal and not rotated",
            "• Avoid shadows and reflections",
            "• Make sure the entire label is visible",
        ]
    )

    return quality, suggestions
