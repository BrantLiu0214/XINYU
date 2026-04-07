from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    emotion_label: str
    emotion_scores: dict[str, float] = Field(default_factory=dict)
    intent_label: str
    intent_scores: dict[str, float] = Field(default_factory=dict)
    intensity_score: float
    risk_aux_score: float
    keyword_hits: list[str] = Field(default_factory=list)
    low_confidence: bool = False


class RiskAssessment(BaseModel):
    risk_score: float
    risk_level: str
    reasons: list[str] = Field(default_factory=list)
    suggested_resource_level: str
