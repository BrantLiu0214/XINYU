from __future__ import annotations

import asyncio
from typing import Protocol

from backend.app.schemas.analysis import AnalysisResult


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

        keyword_hits: list[str] = [kw for kw in self._l3_keywords | self._l2_keywords if kw in text]

        return AnalysisResult(
            emotion_label=emotion_label,
            emotion_scores=emotion_scores,
            intent_label=intent_label,
            intent_scores=intent_scores,
            intensity_score=round(intensity, 4),
            risk_aux_score=round(risk_aux, 4),
            keyword_hits=keyword_hits,
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
        # text is intentionally unused; the stub always returns the fixed result.
        del text
        return self._result
