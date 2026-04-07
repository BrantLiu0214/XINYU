from pydantic import BaseModel, Field

# AnalysisResult and RiskAssessment are defined in schemas.analysis (their natural home,
# as they are produced by NLPService and RiskService respectively). They are imported
# here so PromptBundle can reference them, and so existing imports of these types from
# schemas.prompt continue to work without change.
from backend.app.schemas.analysis import AnalysisResult, RiskAssessment

__all__ = ["AnalysisResult", "RiskAssessment", "ContextWindow", "PromptBundle"]


class ContextWindow(BaseModel):
    conversation_summary: str | None = None
    summary_version: int | None = None
    covered_until_message_id: str | None = None
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    latest_risk_level: str | None = None
    emotion_history: list[str] = Field(default_factory=list)
    emotion_trend: str = "stable"  # "escalating" | "stable" | "de-escalating"


class PromptBundle(BaseModel):
    system_prompt: str
    conversation_summary: str | None = None
    summary_version: int | None = None
    covered_until_message_id: str | None = None
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    user_message: str
    analysis: AnalysisResult
    risk: RiskAssessment
