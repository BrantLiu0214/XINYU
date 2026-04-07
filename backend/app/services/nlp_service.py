from __future__ import annotations

import asyncio
import re
from typing import Protocol

from backend.app.core.crisis_keywords import L2_KEYWORDS as _L2_KW, L3_KEYWORDS as _L3_KW
from backend.app.schemas.analysis import AnalysisResult

# ── Post-processing rule patterns ─────────────────────────────────────────────
# Rule 1: anxiety → fear  — explicit fear word + no diffuse-anxiety markers
_FEAR_WORDS = re.compile(r"(害怕|恐惧|很怕|极度害怕|特别害怕|非常害怕)")
_DIFFUSE_ANXIETY = re.compile(r"(莫名|说不清|无缘无故|不知道为什么|说不出来|没有原因|不知道是什么)")

# Rule 2: fear → anxiety  — diffuse worry framing + no specific nameable object
_SPECIFIC_OBJ = re.compile(
    r"(手术|失败|黑暗|独处|拒绝|飞机|蛇|死亡|考试|冲突|"
    r"医院|血|针|注射|人群|高处|电梯|水灾|火灾|雷|狗|猫|"
    r"老板|面试|公开|演讲|发言|被抛弃|失业|分手)"
)

# Rule 3: shame → anger  — external attribution markers, no self-blame
_EXTERNAL_ATTR = re.compile(r"(凭什么|太不公平|他根本|他们根本|对方根本|不公平|让我很生气|让我愤怒|气死我了|太可气|非常愤怒|极其愤怒|愤怒极了)")
_SELF_BLAME = re.compile(r"(都是我的错|是我的错|我的错|我不好|我太差|我让大家失望|我不够好|是我不对|我做错了)")

# Rule 4: hopelessness → sadness  — no futility language + moderate intensity
_FUTILITY = re.compile(
    r"(没有意义|撑不下去|没有盼头|提不起劲|提不起来劲|麻木了|麻木|活着没劲|活着没什么劲|"
    r"徒劳|看不到希望|没有希望|没有出路|看不见出路|快撑不住|撑不住了|"
    r"不知道还能撑|失去了方向|失去方向|活着没什么意思|活下去的意义|"
    r"什么都没有意义|做什么都没意义|感觉没有盼头)"
)

# Rule 5: implicit crisis safety net  — action-framing phrases escalate intent to crisis
# Patterns require crisis-specific context; general phrases like "告别" / "最后一次" are excluded.
_IMPLICIT_CRISIS = re.compile(
    r"(消失了.*会更好|不在了.*没人在乎|不在了.*轻松|"
    r"不想再继续(了，已经|这一切|活|了.*认真)|已经决定了.*结束|"
    r"想好了.*怎么结束|做好了(最后的)?准备|这是最后一次.*说|"
    r"写了.*遗书|写了.*告别信|不打算继续(活|这一切)|"
    r"怎么让这一切停|让这一切彻底停)"
)

# Rule 6: neutral → sadness  — explicit sadness-charge word present, sadness score elevated
_SAD_CHARGE = re.compile(
    r"(难受|低落|伤心|难过|想哭|心碎|不是滋味|心里很沉|心里难受|心里一直很难受|难过不已)"
)

# Rule 7: self_disclosure → venting  — present-time marker + emotional charge = current distress
_PRESENT_TIME = re.compile(
    r"(今天|最近|这几天|这段时间|今晚|今年|现在|此刻|刚才|这周|这个月|最近这段时间)"
)
_EMOTION_CHARGE = re.compile(
    r"(难受|低落|伤心|难过|焦虑|担心|害怕|恐惧|愤怒|委屈|崩溃|很烦|很累|压力大|憋屈|难熬|心里很)"
)


class NLPService(Protocol):
    async def analyze(self, text: str) -> AnalysisResult: ...


class RealNLPService:
    """Multi-task NLP service backed by the trained MentalHealthMultiTaskModel.

    Heavy imports (torch, transformers) are deferred to ``__init__`` so the
    backend starts normally when ``StubNLPService`` is used instead (e.g. in
    test environments without GPU packages installed).

    Inference runs in a thread-pool executor to avoid blocking the event loop.
    After model inference, the keyword lists from crisis_keywords.py are
    scanned to produce ``keyword_hits`` — this ensures the risk service
    receives keyword context even when the model's risk_aux score is low.
    """

    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoTokenizer

        from backend.app.core.crisis_keywords import L2_KEYWORDS, L3_KEYWORDS
        from nlp.train.model import (
            EMOTION_LABELS,
            INTENT_LABELS,
            MentalHealthMultiTaskModel,
        )

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._emotion_labels = EMOTION_LABELS
        self._intent_labels = INTENT_LABELS
        self._l2_keywords = L2_KEYWORDS
        self._l3_keywords = L3_KEYWORDS

        heads = torch.load(
            f"{model_path}/task_heads.pt", map_location="cpu", weights_only=True
        )
        base_model_name: str = heads["base_model_name"]
        # Load fine-tuned encoder from model_path, not original HF weights.
        model = MentalHealthMultiTaskModel(model_path)
        model.emotion_head.load_state_dict(heads["emotion_head"])
        model.intent_head.load_state_dict(heads["intent_head"])
        model.intensity_head.load_state_dict(heads["intensity_head"])
        model.risk_head.load_state_dict(heads["risk_head"])
        model.eval()
        model.to(self._device)
        self._model = model

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

    @staticmethod
    def _apply_rules(
        text: str,
        emotion_label: str,
        intent_label: str,
        emotion_scores: dict[str, float],
        intensity: float,
        intent_scores: dict[str, float] | None = None,
    ) -> tuple[str, str]:
        """Deterministic post-processing rules to correct systematic model confusions.

        Applied after model inference; only fires when lexical evidence is strong
        enough to override the model's softmax decision.
        """
        # Rule 1: anxiety → fear
        # Model mistakes specific-object fear for anxiety.
        # Fires when: predicted anxiety, fear score ≥ 0.18, fear word present,
        # no diffuse-anxiety language ("莫名/说不清/无缘无故").
        if (
            emotion_label == "anxiety"
            and emotion_scores.get("fear", 0) >= 0.18
            and _FEAR_WORDS.search(text)
            and not _DIFFUSE_ANXIETY.search(text)
        ):
            emotion_label = "fear"

        # Rule 2: fear → anxiety
        # Model mistakes diffuse worry for specific-object fear.
        # Fires when: predicted fear, anxiety score ≥ 0.18, diffuse markers present,
        # no specific nameable object in text.
        elif (
            emotion_label == "fear"
            and emotion_scores.get("anxiety", 0) >= 0.18
            and _DIFFUSE_ANXIETY.search(text)
            and not _SPECIFIC_OBJ.search(text)
        ):
            emotion_label = "anxiety"

        # Rule 3: shame → anger
        # Model mistakes externally-directed anger for shame.
        # Fires when: predicted shame, anger score ≥ 0.15, external attribution
        # markers present, no self-blame language.
        if (
            emotion_label == "shame"
            and emotion_scores.get("anger", 0) >= 0.15
            and _EXTERNAL_ATTR.search(text)
            and not _SELF_BLAME.search(text)
        ):
            emotion_label = "anger"

        # Rule 4: hopelessness → sadness
        # Model over-predicts hopelessness for sad-but-not-futile sentences.
        # Fires when: predicted hopelessness at low intensity (< 0.62),
        # no futility language present, AND an explicit sadness-charge word is present.
        # The positive sadness-charge requirement prevents over-firing on subtle hopelessness.
        if (
            emotion_label == "hopelessness"
            and intensity < 0.62
            and not _FUTILITY.search(text)
            and _SAD_CHARGE.search(text)
        ):
            emotion_label = "sadness"

        # Rule 5: implicit crisis safety net
        # Catches action-framed crisis that the model may classify as venting.
        # Pattern matches preparation / finality language without explicit L3 keywords.
        if (
            intent_label != "crisis"
            and _IMPLICIT_CRISIS.search(text)
        ):
            intent_label = "crisis"

        # Rule 6: neutral → sadness
        # Model under-predicts sadness when intensity is low but explicit sadness-charge
        # words are present. Fires when: predicted neutral, sadness score ≥ 0.25,
        # at least one explicit sadness-charge word present.
        if (
            emotion_label == "neutral"
            and emotion_scores.get("sadness", 0) >= 0.10
            and _SAD_CHARGE.search(text)
        ):
            emotion_label = "sadness"

        # Rule 7: self_disclosure → venting
        # Model classifies present-tense emotional distress as background self_disclosure.
        # Fires when: predicted self_disclosure, emotion is not neutral, both a
        # present-time marker and an emotional-charge word are present,
        # and the model's own venting score is ≥ 0.15.
        _i_scores = intent_scores or {}
        if (
            intent_label == "self_disclosure"
            and emotion_label != "neutral"
            and _PRESENT_TIME.search(text)
            and _EMOTION_CHARGE.search(text)
            and _i_scores.get("venting", 0) >= 0.12
        ):
            intent_label = "venting"

        return emotion_label, intent_label

    def _sync_analyze(self, text: str) -> AnalysisResult:
        enc = self._tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            out = self._model(
                enc["input_ids"].to(self._device),
                enc["attention_mask"].to(self._device),
            )

        emotion_probs = self._torch.softmax(out.emotion_logits[0], dim=-1).cpu().tolist()
        intent_probs = self._torch.softmax(out.intent_logits[0], dim=-1).cpu().tolist()
        intensity = float(out.intensity[0].item())
        risk_aux = float(out.risk_aux[0].item())

        emotion_scores = {k: round(v, 4) for k, v in zip(self._emotion_labels, emotion_probs)}
        intent_scores = {k: round(v, 4) for k, v in zip(self._intent_labels, intent_probs)}

        emotion_label = self._emotion_labels[int(max(range(len(emotion_probs)), key=lambda i: emotion_probs[i]))]
        intent_label = self._intent_labels[int(max(range(len(intent_probs)), key=lambda i: intent_probs[i]))]

        # Apply deterministic boundary rules over the raw model output
        emotion_label, intent_label = self._apply_rules(
            text, emotion_label, intent_label, emotion_scores, intensity, intent_scores
        )

        keyword_hits: list[str] = [kw for kw in self._l3_keywords | self._l2_keywords if kw in text]
        low_confidence: bool = max(intent_probs) < 0.50

        return AnalysisResult(
            emotion_label=emotion_label,
            emotion_scores=emotion_scores,
            intent_label=intent_label,
            intent_scores=intent_scores,
            intensity_score=round(intensity, 4),
            risk_aux_score=round(risk_aux, 4),
            keyword_hits=keyword_hits,
            low_confidence=low_confidence,
        )

    async def analyze(self, text: str) -> AnalysisResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_analyze, text)


class StubNLPService:
    """Deterministic stub used until Module 05 implements the real model pipeline.

    Pass a custom ``fixed_result`` to control the analysis output in tests that
    need specific emotion/intent/risk combinations.
    """

    _DEFAULT = AnalysisResult(
        emotion_label="anxiety",
        emotion_scores={"anxiety": 0.65, "neutral": 0.35},
        intent_label="venting",
        intent_scores={"venting": 0.80, "seeking_empathy": 0.20},
        intensity_score=0.55,
        risk_aux_score=0.20,
        keyword_hits=[],
    )

    def __init__(self, fixed_result: AnalysisResult | None = None) -> None:
        self._result = fixed_result or self._DEFAULT

    async def analyze(self, text: str) -> AnalysisResult:
        # Model inference is stubbed, but keyword scanning must still run so
        # the risk service can escalate L2/L3 for crisis-keyword messages.
        keyword_hits = [kw for kw in _L3_KW | _L2_KW if kw in text]
        return self._result.model_copy(update={"keyword_hits": keyword_hits})
