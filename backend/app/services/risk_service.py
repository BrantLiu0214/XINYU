from __future__ import annotations

from backend.app.core.crisis_keywords import L2_KEYWORDS as _L2_KEYWORDS, L3_KEYWORDS as _L3_KEYWORDS
from backend.app.schemas.analysis import AnalysisResult, RiskAssessment

_HIGH_RISK_LEVELS: frozenset[str] = frozenset({"L2", "L3"})


class RiskService:
    """Pure, stateless risk evaluator.  No database access."""

    def evaluate(self, analysis: AnalysisResult, recent_levels: list[str]) -> RiskAssessment:
        reasons: list[str] = []

        l3_hits = [k for k in analysis.keyword_hits if k in _L3_KEYWORDS]
        l2_hits = [k for k in analysis.keyword_hits if k in _L2_KEYWORDS]

        if l3_hits:
            reasons.append(f"高危关键词命中: {', '.join(l3_hits)}")
        if l2_hits:
            reasons.append(f"中危关键词命中: {', '.join(l2_hits)}")
        if analysis.risk_aux_score >= 0.50:
            reasons.append(f"风险辅助分数 {analysis.risk_aux_score:.2f}")
        if analysis.intensity_score >= 0.75:
            reasons.append(f"情绪强度 {analysis.intensity_score:.2f}")

        escalating = self._is_escalating(recent_levels)
        if escalating:
            reasons.append("近期风险等级持续上升")

        # L3: any direct crisis signal
        if (
            l3_hits
            or (analysis.intent_label == "crisis" and analysis.risk_aux_score >= 0.65)
            or (analysis.emotion_label == "hopelessness" and analysis.intensity_score >= 0.90)
        ):
            risk_level = "L3"
            risk_score = max(0.85, analysis.risk_aux_score)
            reasons.append("触发危机响应路径")
            suggested = "urgent"

        # L2: moderate risk
        elif (
            l2_hits
            or analysis.risk_aux_score >= 0.50
            or (analysis.emotion_label == "hopelessness" and analysis.intensity_score >= 0.65)
            or escalating
        ):
            risk_level = "L2"
            risk_score = max(0.55, analysis.risk_aux_score)
            suggested = "standard"

        # L1: elevated but not warning
        elif analysis.risk_aux_score >= 0.25 or analysis.intensity_score >= 0.60:
            risk_level = "L1"
            risk_score = max(0.30, analysis.risk_aux_score)
            suggested = "none"

        # L0: normal
        else:
            risk_level = "L0"
            risk_score = analysis.risk_aux_score
            suggested = "none"

        if not reasons:
            reasons.append("无显著风险信号")

        return RiskAssessment(
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            reasons=reasons,
            suggested_resource_level=suggested,
        )

    def _is_escalating(self, recent_levels: list[str]) -> bool:
        """Return True if at least two of the most recent levels are high-risk (L2/L3)."""
        high = [lv for lv in recent_levels if lv in _HIGH_RISK_LEVELS]
        return len(high) >= 2
