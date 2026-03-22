"""Multi-task mental-health text classifier.

Architecture: shared Chinese encoder (RoBERTa-wwm-ext or bert-base-chinese)
with four independent prediction heads:
  - emotion classification  (7 classes)
  - intent classification   (6 classes)
  - intensity regression    (scalar 0–1)
  - risk auxiliary score    (scalar 0–1, trained on risk_flag)

Usage:
    from nlp.train.model import MentalHealthMultiTaskModel, EMOTION_LABELS, INTENT_LABELS

    model = MentalHealthMultiTaskModel()               # RoBERTa (default)
    model = MentalHealthMultiTaskModel("bert-base-chinese")  # BERT baseline
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from transformers import AutoModel

# ── Label taxonomies ──────────────────────────────────────────────────────────
# Order is significant: index i in the head output corresponds to label[i].

EMOTION_LABELS: list[str] = [
    "anxiety",      # 0
    "sadness",      # 1
    "anger",        # 2
    "fear",         # 3
    "shame",        # 4
    "hopelessness", # 5
    "neutral",      # 6
]

INTENT_LABELS: list[str] = [
    "venting",             # 0
    "seeking_advice",      # 1
    "seeking_empathy",     # 2
    "crisis",              # 3
    "self_disclosure",     # 4
    "information_seeking", # 5
]

NUM_EMOTIONS = len(EMOTION_LABELS)
NUM_INTENTS = len(INTENT_LABELS)
HIDDEN_SIZE = 768  # CLS vector dimension for both RoBERTa-wwm-ext and bert-base-chinese


class ModelOutput(NamedTuple):
    emotion_logits: torch.Tensor   # (batch, 7)  — raw logits before softmax
    intent_logits: torch.Tensor    # (batch, 6)  — raw logits before softmax
    intensity: torch.Tensor        # (batch, 1)  — sigmoid output in (0, 1)
    risk_aux: torch.Tensor         # (batch, 1)  — sigmoid output in (0, 1)


class MentalHealthMultiTaskModel(nn.Module):
    """Shared-encoder multi-task classifier for mental-health dialogue.

    Args:
        base_model_name: HuggingFace model identifier.  Use
            ``"hfl/chinese-roberta-wwm-ext"`` (default) for the primary model
            or ``"bert-base-chinese"`` for the BERT baseline comparison.
    """

    def __init__(self, base_model_name: str = "hfl/chinese-roberta-wwm-ext") -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(p=0.1)

        self.emotion_head = nn.Linear(HIDDEN_SIZE, NUM_EMOTIONS)
        self.intent_head = nn.Linear(HIDDEN_SIZE, NUM_INTENTS)
        self.intensity_head = nn.Linear(HIDDEN_SIZE, 1)
        self.risk_head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> ModelOutput:
        encoder_kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        # CLS token representation — shape (batch, hidden_size)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])

        emotion_logits = self.emotion_head(cls)                          # (batch, 7)
        intent_logits = self.intent_head(cls)                            # (batch, 6)
        intensity = torch.sigmoid(self.intensity_head(cls))              # (batch, 1)
        risk_aux = torch.sigmoid(self.risk_head(cls))                    # (batch, 1)

        return ModelOutput(
            emotion_logits=emotion_logits,
            intent_logits=intent_logits,
            intensity=intensity,
            risk_aux=risk_aux,
        )
