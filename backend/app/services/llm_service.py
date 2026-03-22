from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from backend.app.schemas.prompt import PromptBundle


class LLMProvider(Protocol):
    async def stream_reply(self, prompt: PromptBundle) -> AsyncIterator[str]: ...


class FakeLLMProvider:
    """Deterministic fake provider for tests and local development without API credentials.

    Replies with a short fixed sequence of tokens.  The crisis path is detected
    by checking the system prompt for the "high-risk" marker inserted by PromptService.
    """

    _NORMAL_TOKENS = ["听起来你最近压力不小，", "先给自己一些空间。"]
    _CRISIS_TOKENS = ["我听到你了，", "这一刻一定很难熬。", "你现在安全吗？"]

    async def stream_reply(self, prompt: PromptBundle) -> AsyncIterator[str]:
        tokens = (
            self._CRISIS_TOKENS if "high-risk" in prompt.system_prompt else self._NORMAL_TOKENS
        )
        for token in tokens:
            yield token


class DoubaoLLMProvider:
    """Real provider adapter for Doubao (Volcengine Ark, OpenAI-compatible API).

    Requires the ``openai`` package and valid credentials:
        XINYU_DOUBAO_API_KEY  — Ark API key
        XINYU_DOUBAO_MODEL    — endpoint ID (e.g. "ep-20240101xxxxxx-xxxxx")
        XINYU_DOUBAO_BASE_URL — defaults to https://ark.volcengine.com/api/v3
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://ark.volcengine.com/api/v3",
    ) -> None:
        import openai  # imported lazily so missing openai doesn't break FakeLLMProvider

        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def stream_reply(self, prompt: PromptBundle) -> AsyncIterator[str]:
        messages = self._build_messages(prompt)
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_messages(self, prompt: PromptBundle) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": prompt.system_prompt}]

        if prompt.conversation_summary:
            messages.append({
                "role": "system",
                "content": f"[会话摘要] {prompt.conversation_summary}",
            })

        for msg in prompt.recent_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Inject NLP analysis so the LLM can tailor its response to the
        # specific emotion, intent, and intensity detected by the NLP model.
        a = prompt.analysis
        r = prompt.risk
        reasons_str = "、".join(r.reasons) if r.reasons else "无"
        messages.append({
            "role": "system",
            "content": (
                f"[当前情绪分析]\n"
                f"情绪类型：{a.emotion_label}，意图类型：{a.intent_label}，"
                f"情绪强度：{a.intensity_score:.0%}，风险评分：{a.risk_aux_score:.2f}\n"
                f"风险等级：{r.risk_level}，触发原因：{reasons_str}\n"
                f"请根据以上分析调整回复策略：针对不同意图（如情绪宣泄vs寻求建议）采用不同的支持方式，"
                f"并根据情绪强度和风险等级决定表达的紧迫程度。"
            ),
        })

        messages.append({"role": "user", "content": prompt.user_message})
        return messages
