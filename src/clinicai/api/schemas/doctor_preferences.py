from typing import List, Optional

from pydantic import BaseModel, Field


class PreVisitSectionConfig(BaseModel):
    """
    Configuration for one pre-visit section.

    Recognized section_key values in backend:
    - chief_complaint
    - hpi
    - history
    - review_of_systems
    - current_medication

    selected_fields is optional; if omitted or empty, the backend will
    still honor the enabled flag for the section. Field-level filtering
    is reserved for future use.
    """

    section_key: str
    enabled: bool = Field(default=True)
    selected_fields: Optional[List[str]] = Field(default_factory=list)


class PreVisitAIConfig(BaseModel):
    style: Optional[str] = Field(default=None, description='concise | standard | comprehensive')
    focus_areas: Optional[List[str]] = Field(default=None, description='Preferred focus areas')
    include_red_flags: Optional[bool] = Field(default=None, description='Whether to highlight red flags')


class SoapAIConfig(BaseModel):
    detail_level: Optional[str] = Field(default=None, description='brief | standard | detailed')
    formatting: Optional[str] = Field(default=None, description='bullet_points | paragraphs')
    language: Optional[str] = Field(default=None, description='Language override (e.g., en, hi)')


class DoctorPreferencesResponse(BaseModel):
    doctor_id: str
    # Intake preferences (legacy, keep backward compatibility)
    global_categories: List[str] = Field(default_factory=list)
    selected_categories: List[str] = Field(default_factory=list)
    max_questions: int = 12
    # New doctor preferences
    soap_order: List[str] = Field(default_factory=list)
    pre_visit_config: List[PreVisitSectionConfig] = Field(default_factory=list)
    pre_visit_ai_config: Optional[PreVisitAIConfig] = None
    soap_ai_config: Optional[SoapAIConfig] = None


class UpsertDoctorPreferencesRequest(BaseModel):
    doctor_id: str
    soap_order: List[str] = Field(default_factory=list)
    pre_visit_config: List[PreVisitSectionConfig] = Field(default_factory=list)
    # Optional legacy intake prefs (do not break old payloads)
    categories: Optional[List[str]] = None
    max_questions: Optional[int] = None
    global_categories: Optional[List[str]] = None
    # Optional new AI configs
    pre_visit_ai_config: Optional[PreVisitAIConfig] = None
    soap_ai_config: Optional[SoapAIConfig] = None

