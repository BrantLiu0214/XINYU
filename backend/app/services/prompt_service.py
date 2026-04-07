from __future__ import annotations

from backend.app.schemas.analysis import AnalysisResult, RiskAssessment
from backend.app.schemas.prompt import PromptBundle
from backend.app.services.context_service import ContextService

HIGH_RISK_LEVELS = {"L2", "L3"}


class PromptService:
    def __init__(self, context_service: ContextService) -> None:
        self._context_service = context_service

    async def build(
        self,
        session_id: str,
        user_message: str,
        analysis: AnalysisResult,
        risk: RiskAssessment,
    ) -> PromptBundle:
        await self._context_service.refresh_summary_if_needed(session_id)
        context_window = await self._context_service.build_window(session_id, user_message)

        return PromptBundle(
            system_prompt=self._build_system_prompt(
                risk.risk_level,
                analysis.low_confidence,
                context_window.emotion_trend,
                context_window.emotion_history,
            ),
            conversation_summary=context_window.conversation_summary,
            summary_version=context_window.summary_version,
            covered_until_message_id=context_window.covered_until_message_id,
            recent_messages=context_window.recent_messages,
            user_message=user_message,
            analysis=analysis,
            risk=risk,
        )

    def _build_system_prompt(
        self,
        risk_level: str,
        low_confidence: bool = False,
        emotion_trend: str = "stable",
        emotion_history: list[str] | None = None,
    ) -> str:
        base_rules = [
            "You are the XinYu emotional support assistant.",
            "Provide empathetic, concrete, reality-based support.",
            "Do not claim to diagnose or replace a licensed clinician.",
            "Do not provide self-harm, suicide, or violence instructions.",
            "Do not make promises you cannot verify in the real world.",
            "Use the conversation summary, recent messages, and current analysis to stay consistent.",
        ]

        if risk_level in HIGH_RISK_LEVELS:
            base_rules.extend(
                [
                    "This is the high-risk support path.",
                    "Lead with concern and encourage immediate help from trusted people or professional crisis resources.",
                    "Prioritize immediate offline safety and support over open-ended discussion.",
                ]
            )
        else:
            base_rules.extend(
                [
                    "This is the standard support path.",
                    "Lead with empathy, clarify the main stressor, and offer one or two practical next steps.",
                ]
            )

        if emotion_trend == "de-escalating" and emotion_history:
            arc = " → ".join(emotion_history)
            base_rules.append(
                f"对话记录显示用户情绪已从「{arc}」逐步好转。"
                "请肯定用户的积极变化，避免过度强调负面状态，以支持其继续向好的方向发展。"
            )
        elif emotion_trend == "escalating" and emotion_history:
            arc = " → ".join(emotion_history)
            base_rules.append(
                f"对话记录显示用户情绪持续加重（{arc}）。"
                "请以更多耐心和关注回应，适时评估是否需要升级支持力度。"
            )

        if low_confidence:
            base_rules.append(
                "The NLP classifier has low confidence about the user's intent this turn. "
                "Use the conversation context to infer what kind of support the user is seeking "
                "(venting, empathy, advice, or crisis) and respond accordingly."
            )

        return "\n".join(base_rules)
